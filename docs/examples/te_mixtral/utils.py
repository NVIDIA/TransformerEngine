# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import sys
import time
import os
import IPython

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.device_mesh import DeviceMesh

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs


class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"

        # Default to Mixtral 8x7B.
        self.model_name = "mistralai/Mixtral-8x7B-v0.1"

        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 4
        self.max_seq_length = 256
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 3

        # Set by the user or populated automatically on download.
        self.weights_cache_dir = ""
        self.hf_access_token = ""
        self.expert_parallel_size = 8


hyperparams = HyperParameters()


def get_dataloaders(accelerator: Accelerator, hyperparams: HyperParameters):
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

    # Pad to multiple of 16 for both FP8 and BF16.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=16,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
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

    supplied_cache_dir = hyperparams.weights_cache_dir or None
    hyperparams.weights_cache_dir = snapshot_download(
        repo_id=hyperparams.model_name, cache_dir=supplied_cache_dir
    )
    print(f"Model cache directory: {hyperparams.weights_cache_dir}")


def init_baseline_model(hyperparams: HyperParameters):
    """Load the vanilla HuggingFace Mixtral model in BF16."""
    ensure_model_is_downloaded(hyperparams)

    config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams.weights_cache_dir,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    model.config.use_cache = False
    return model


def init_te_mixtral_model(hyperparams: HyperParameters):
    """Load Mixtral with TE-optimised MoE blocks."""
    ensure_model_is_downloaded(hyperparams)

    from te_mixtral import NVMixtralForCausalLM, replace_params

    base_config = AutoConfig.from_pretrained(hyperparams.weights_cache_dir)
    base_config._attn_implementation = "flash_attention_2"
    te_config = NVMixtralForCausalLM.config_class(**base_config.to_dict())
    te_config.expert_parallel_size = hyperparams.expert_parallel_size

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if world_size != hyperparams.expert_parallel_size:
            raise ValueError(
                "For this minimal EP setup, WORLD_SIZE must match expert_parallel_size. "
                f"Got WORLD_SIZE={world_size}, expert_parallel_size={hyperparams.expert_parallel_size}."
            )
    elif hyperparams.expert_parallel_size != 1:
        raise ValueError("expert_parallel_size > 1 requires torchrun distributed launch.")

    # Load the HF model on CPU and map weights into TE structure.
    hf_model = AutoModelForCausalLM.from_pretrained(
        hyperparams.weights_cache_dir,
        config=base_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model = NVMixtralForCausalLM(te_config).to(
        device=f"cuda:{local_rank}",
        dtype=torch.bfloat16,
    )
    te_state_dict = model.state_dict()
    replace_params(hf_model.state_dict(), te_state_dict, model.config)
    model.load_state_dict(te_state_dict, strict=False)
    del hf_model

    if hyperparams.expert_parallel_size > 1:
        ep_mesh = DeviceMesh("cuda", torch.arange(world_size))
        model.model.set_ep_groups(ep_group=dist.group.WORLD, ep_mesh=ep_mesh)

    model.config.use_cache = False
    return model


def build_adamw(model, hyperparams: HyperParameters):
    """Build AdamW optimizer compatible with mixed Tensor/DTensor parameters."""
    dtensor_params = []
    tensor_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, DTensor) or isinstance(param.data, DTensor):
            dtensor_params.append(param)
        else:
            tensor_params.append(param)

    param_groups = []
    if tensor_params:
        param_groups.append({"params": tensor_params})
    if dtensor_params:
        param_groups.append({"params": dtensor_params})

    use_fused = hyperparams.expert_parallel_size == 1 and not dtensor_params
    return AdamW(
        params=param_groups,
        lr=hyperparams.learning_rate,
        fused=use_fused,
        foreach=False,
    )


def wrap_with_accelerator(model, hyperparams: HyperParameters):
    """Wrap the model in HuggingFace Accelerate (with optional FP8 support)."""
    fp8_kwarg_handler = (
        [FP8RecipeKwargs(backend="te")] if hyperparams.mixed_precision == "fp8" else None
    )

    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=fp8_kwarg_handler,
    )

    train_dataloader = get_dataloaders(accelerator, hyperparams)
    optimizer = build_adamw(model, hyperparams)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=hyperparams.num_warmup_steps,
        num_training_steps=hyperparams.num_training_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    return accelerator, model, optimizer, train_dataloader, lr_scheduler


def finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler):
    """Run a short fine-tuning loop and report average step time."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    train_dataloader = enumerate(train_dataloader)

    # Warmup iterations (not timed).
    for _ in range(hyperparams.num_warmup_steps):
        _, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    for _ in range(hyperparams.num_training_steps):
        _, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    torch.cuda.synchronize()
    end.record()
    accelerator.end_training()

    ms_per_step = start.elapsed_time(end) / hyperparams.num_training_steps
    print(
        f"{hyperparams.num_training_steps} fine-tuning steps complete!\n"
        f"Average time per step: {ms_per_step:.0f} ms"
    )


def run_te_mixtral_finetune(hyperparams: HyperParameters):
    """Convenience entrypoint: init TE Mixtral, wrap, and run fine-tuning."""
    model = init_te_mixtral_model(hyperparams)
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)
    finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)


def restart_jupyter_notebook():
    """Flush GPU memory by restarting the Jupyter kernel."""
    IPython.Application.instance().kernel.do_shutdown(True)

    if torch.cuda.memory_allocated() != 0:
        import warnings

        warnings.warn("GPU memory not fully flushed — trying secondary method.")
        from IPython.core.display import HTML

        HTML("<script>Jupyter.notebook.kernel.restart()</script>")

        if torch.cuda.memory_allocated() != 0:
            print("Please restart the Jupyter kernel manually.")

    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
        torch.set_warn_always(False)
