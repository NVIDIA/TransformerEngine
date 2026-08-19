# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from accelerate import Accelerator


class HyperParameters:
    def __init__(self):
        # "bf16" (improvements 1-2) or "mxfp8" (improvement 3).
        self.mixed_precision = "bf16"

        self.model_name = "mistralai/Mixtral-8x7B-v0.1"
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 4
        self.max_seq_length = 2048
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 3

        self.weights_cache_dir = ""
        self.hf_access_token = ""
        self.expert_parallel_size = 8

        # "loop" (improvement 1) or "grouped_op" (improvements 2-3).
        self.expert_ffn_mode = "grouped_op"

        # "te_mixtral" (improvements 1-2, BF16) or "te_mixtral_mxfp8" (improvement 3).
        self.model_impl = "te_mixtral"


def get_dataloaders(accelerator: Accelerator, hyperparams: HyperParameters):
    from collator import DataCollatorWithFlattening

    dataset = load_dataset(hyperparams.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=hyperparams.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    pad_multiple = 32 if hyperparams.mixed_precision == "mxfp8" else 16
    bshd_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_multiple,
    )
    data_collator = DataCollatorWithFlattening(
        collator=bshd_collator,
        pad_to_multiple_of=pad_multiple,
        separator_id=-100,
    )

    sampler = None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if hyperparams.expert_parallel_size > 1 and world_size > 1:
        ep_size = hyperparams.expert_parallel_size
        dp_size = world_size // ep_size
        global_rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=global_rank // ep_size,
            shuffle=True,
            drop_last=True,
        )

    train_dataloader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        drop_last=True,
    )
    return train_dataloader


def ensure_model_is_downloaded(hyperparams: HyperParameters):
    assert hyperparams.model_name in [
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1",
    ], "Only Mixtral-8x7B-v0.1 and Mixtral-8x22B-v0.1 are supported."

    from huggingface_hub import login, snapshot_download

    try:
        login(hyperparams.hf_access_token)
    except Exception as e:
        if "Invalid token passed!" in str(e):
            print(
                "Please provide a valid HF Access Token. "
                "See: https://huggingface.co/docs/hub/en/security-tokens"
            )
        else:
            print(f"Login exception: {e}")

    hyperparams.weights_cache_dir = snapshot_download(
        repo_id=hyperparams.model_name,
        cache_dir=hyperparams.weights_cache_dir or None,
    )
    print(f"Model cache directory: {hyperparams.weights_cache_dir}")


def init_baseline_model(hyperparams: HyperParameters):
    """Load the vanilla HuggingFace Mixtral model in BF16."""
    ensure_model_is_downloaded(hyperparams)

    config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    config._attn_implementation = "flash_attention_2"
    load_kwargs = {"config": config, "torch_dtype": torch.bfloat16}
    if int(os.environ.get("WORLD_SIZE", "1")) == 1 and torch.cuda.device_count() > 1:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(hyperparams.weights_cache_dir, **load_kwargs)
    if not hasattr(model, "hf_device_map"):
        model = model.cuda()
    model.config.use_cache = False
    return model


def _enable_fused_mxfp8_grouped_mlp() -> None:
    """Improvement 3: enable the fused ``ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8`` and
    backward kernel in the installed TE without recompiling.

    ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` must be set *before*
    ``transformer_engine.pytorch.ops`` is imported — the fusion is registered
    at TE module-import-time. ``run_finetune_ep.py`` sniffs ``--improvement 3``
    and sets the env var before importing ``utils``.

    We also (a) relax the SM-version check from ``!= 10`` to ``>= 10`` so
    SM>=11 successors of B300 fire the kernel, and (b) wrap the cudnn-frontend
    grouped-GEMM wrappers so the installed TE's ``c_dtype`` kwarg (dropped by
    cudnn-frontend 1.23.0) is silently filtered out.
    """
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    import inspect
    import cudnn  # type: ignore
    from transformer_engine.pytorch.ops.fused import forward_grouped_mlp as _fwd_mod
    from transformer_engine.pytorch.ops.fused import backward_grouped_mlp as _bwd_mod
    from transformer_engine.pytorch.utils import get_device_compute_capability

    def _make_is_supported(kernel_method_names):
        def _is_supported(cls) -> bool:
            if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
                return False
            if get_device_compute_capability()[0] < 10:
                return False
            try:
                for method_name in kernel_method_names:
                    getattr(cls, method_name)()
            except ImportError:
                return False
            return True

        return _is_supported

    def _make_compat_kernel(real_callable):
        accepted = set(inspect.signature(real_callable).parameters)

        def _compat(**kwargs):
            for k in list(kwargs):
                if k not in accepted:
                    kwargs.pop(k)
            return real_callable(**kwargs)

        return _compat

    def _patch_kernel_method(cls, method_name, wrapper_name):
        compat = _make_compat_kernel(getattr(cudnn, wrapper_name))

        def _kernel_classmethod(_cls):
            return compat

        setattr(cls, method_name, classmethod(_kernel_classmethod))

    fwd_cls = _fwd_mod.ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8
    bwd_cls = _bwd_mod.BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8
    fwd_cls.is_supported = classmethod(
        _make_is_supported(("grouped_gemm_glu_kernel", "grouped_gemm_quant_kernel"))
    )
    bwd_cls.is_supported = classmethod(
        _make_is_supported(("grouped_gemm_dglu_kernel", "grouped_gemm_quant_kernel"))
    )
    _patch_kernel_method(fwd_cls, "grouped_gemm_glu_kernel", "grouped_gemm_glu_wrapper_sm100")
    _patch_kernel_method(fwd_cls, "grouped_gemm_quant_kernel", "grouped_gemm_quant_wrapper_sm100")
    _patch_kernel_method(bwd_cls, "grouped_gemm_dglu_kernel", "grouped_gemm_dglu_wrapper_sm100")
    _patch_kernel_method(bwd_cls, "grouped_gemm_quant_kernel", "grouped_gemm_quant_wrapper_sm100")


def init_te_mixtral_model(hyperparams: HyperParameters):
    """Load Mixtral with TE-optimised MoE blocks."""
    ensure_model_is_downloaded(hyperparams)

    import transformer_engine.common.recipe as te_recipe

    if hyperparams.model_impl == "te_mixtral_mxfp8":
        if hyperparams.mixed_precision != "mxfp8":
            raise ValueError("model_impl='te_mixtral_mxfp8' requires mixed_precision='mxfp8'.")
        _enable_fused_mxfp8_grouped_mlp()
        from te_mixtral_mxfp8 import TEMixtralMXFP8ForCausalLM as ForCausalLM
        from te_mixtral_mxfp8 import replace_params
    else:
        from te_mixtral import TEMixtralForCausalLM as ForCausalLM
        from te_mixtral import replace_params

    base_config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    base_config._attn_implementation = "flash_attention_2"
    te_config = ForCausalLM.config_class(**base_config.to_dict())
    te_config.expert_parallel_size = hyperparams.expert_parallel_size
    if hasattr(te_config, "expert_ffn_mode"):
        te_config.expert_ffn_mode = hyperparams.expert_ffn_mode

    fp8_recipe = None
    if hyperparams.mixed_precision == "mxfp8":
        fp8_recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
        te_config.layer_precision = ["fp8"] * te_config.num_hidden_layers
    elif hyperparams.mixed_precision != "bf16":
        raise ValueError(
            f"Unsupported mixed_precision={hyperparams.mixed_precision!r}; use 'bf16' or 'mxfp8'."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if world_size % hyperparams.expert_parallel_size != 0:
            raise ValueError(
                f"WORLD_SIZE ({world_size}) must be a multiple of "
                f"expert_parallel_size ({hyperparams.expert_parallel_size})."
            )
    elif hyperparams.expert_parallel_size != 1:
        raise ValueError("expert_parallel_size > 1 requires torchrun distributed launch.")

    hf_model = AutoModelForCausalLM.from_pretrained(
        hyperparams.weights_cache_dir,
        config=base_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = ForCausalLM(te_config, fp8_recipe=fp8_recipe).to(
        device=f"cuda:{local_rank}",
        dtype=torch.bfloat16,
    )
    te_state_dict = model.state_dict()
    replace_params(hf_model.state_dict(), te_state_dict, model.config)
    missing, unexpected = model.load_state_dict(te_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading TE state dict: {unexpected}")
    non_extra_missing = [key for key in missing if not key.endswith("_extra_state")]
    if non_extra_missing:
        raise RuntimeError(f"Missing non-extra-state keys in TE model: {non_extra_missing}")
    del hf_model

    model._te_mixtral_dp_group = None
    model._te_mixtral_dp_size = 1

    if hyperparams.expert_parallel_size > 1:
        ep_size = hyperparams.expert_parallel_size
        dp_size = world_size // ep_size
        global_rank = dist.get_rank()
        dp_group = None
        if dp_size == 1:
            ep_group = dist.group.WORLD
        else:
            # Rank layout is [DP, EP]. EP groups are contiguous ranks; DP groups
            # contain the same local expert shard across DP replicas.
            ep_group = None
            for dp_rank in range(dp_size):
                ranks = list(range(dp_rank * ep_size, (dp_rank + 1) * ep_size))
                group = dist.new_group(ranks=ranks)
                if global_rank in ranks:
                    ep_group = group
            for ep_rank in range(ep_size):
                ranks = [dp_rank * ep_size + ep_rank for dp_rank in range(dp_size)]
                group = dist.new_group(ranks=ranks)
                if global_rank in ranks:
                    dp_group = group
            if ep_group is None:
                raise RuntimeError(f"Rank {global_rank} was not assigned to an EP group.")
        model.model.set_ep_groups(ep_group=ep_group)
        model._te_mixtral_dp_group = dp_group
        model._te_mixtral_dp_size = dp_size

    model.config.use_cache = False
    return model


def build_adamw(model, hyperparams: HyperParameters):
    params = [param for param in model.parameters() if param.requires_grad]
    use_fused = hyperparams.expert_parallel_size == 1
    return AdamW(
        params=params,
        lr=hyperparams.learning_rate,
        fused=use_fused,
        foreach=False,
    )


def sync_data_parallel_gradients(model) -> None:
    dp_group = getattr(model, "_te_mixtral_dp_group", None)
    if dp_group is None:
        return

    dp_size = getattr(model, "_te_mixtral_dp_size", dist.get_world_size(dp_group))
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
        param.grad.div_(dp_size)


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def wrap_with_accelerator(model, hyperparams: HyperParameters):
    # The TE-native MXFP8 model handles its own FP8 autocast; keep
    # Accelerate's mixed_precision on bf16 to avoid double-wrapping the recipe.
    use_te_mxfp8 = hyperparams.mixed_precision == "mxfp8"
    accelerator_mixed_precision = "bf16" if use_te_mxfp8 else hyperparams.mixed_precision

    accelerator = Accelerator(
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=accelerator_mixed_precision,
    )

    train_dataloader = get_dataloaders(accelerator, hyperparams)
    optimizer = build_adamw(model, hyperparams)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyperparams.num_warmup_steps,
        num_training_steps=hyperparams.num_warmup_steps + hyperparams.num_training_steps,
    )

    if hyperparams.expert_parallel_size > 1:
        # EP path: keep the DP-aware sampler intact and manually sync DP gradients.
        # The dataloader is intentionally not prepared by Accelerate, so keep
        # the scheduler as a plain PyTorch scheduler to avoid stepping it once
        # per process.
        optimizer = accelerator.prepare(optimizer)
        return accelerator, model, optimizer, train_dataloader, lr_scheduler

    if hasattr(model, "hf_device_map"):
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, lr_scheduler
        )
        return accelerator, model, optimizer, train_dataloader, lr_scheduler

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    return accelerator, model, optimizer, train_dataloader, lr_scheduler


def finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler):
    """Run a short fine-tuning loop and report median step time."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    # Cycle the dataloader so long sweeps don't hit StopIteration when
    # batch * world_size * num_steps exceeds the dataset size.
    def _cycle(loader):
        while True:
            for x in loader:
                yield x

    train_dataloader = enumerate(_cycle(train_dataloader))

    for _ in range(hyperparams.num_warmup_steps):
        _, batch = next(train_dataloader)
        if hyperparams.expert_parallel_size > 1:
            batch = move_batch_to_device(batch, accelerator.device)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            sync_data_parallel_gradients(model)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    step_times_ms: list[float] = []
    is_printer = int(os.environ.get("LOCAL_RANK", "0")) == 0
    torch.cuda.synchronize()

    for step_idx in range(hyperparams.num_training_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        _, batch = next(train_dataloader)
        if hyperparams.expert_parallel_size > 1:
            batch = move_batch_to_device(batch, accelerator.device)

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()

            accelerator.backward(loss)
            sync_data_parallel_gradients(model)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        end.record()
        end.synchronize()
        step_ms = start.elapsed_time(end)
        step_times_ms.append(step_ms)
        if is_printer:
            print(
                f"[step {step_idx + 1}/{hyperparams.num_training_steps}] {step_ms:.1f} ms",
                flush=True,
            )

    accelerator.end_training()

    n = len(step_times_ms)
    median_ms = sorted(step_times_ms)[n // 2]
    last_ms = step_times_ms[-1]
    print(
        f"{n} fine-tuning steps complete!\n"
        f"Median time per step:  {median_ms:.0f} ms\n"
        f"Last step time:        {last_ms:.0f} ms"
    )


def run_te_mixtral_finetune(hyperparams: HyperParameters):
    model = init_te_mixtral_model(hyperparams)
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(
        model, hyperparams
    )
    finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)


def run_hf_baseline_finetune(hyperparams: HyperParameters):
    model = init_baseline_model(hyperparams)
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(
        model, hyperparams
    )
    finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)
