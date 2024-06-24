# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import time
import sys
import IPython

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs


class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"
        # self.model_name = "" # <== Add model weight location here
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 8
        self.max_seq_length = 256
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 10


hyperparams = HyperParameters()


def get_dataloaders(accelerator: Accelerator, hyperparams):
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

    # Simply pad to the multiple of 16 for both FP8 and BF16 precision
    pad_to_multiple_of = 16
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": hyperparams.batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader


def init_baseline_model(hyperparams):
    # Init the model
    config = AutoConfig.from_pretrained(hyperparams.model_name)
    # make sure to use flash_attention to do iso comparison with TELlamaModel
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM. So adding here for 1:1 comparison
    model.config.use_cache = False

    return model


def init_te_llama_model(hyperparams):
    # Init the model
    from te_llama import TELlamaForCausalLM

    config = AutoConfig.from_pretrained(hyperparams.model_name)
    config._attn_implementation = "flash_attention_2"
    model = TELlamaForCausalLM.from_pretrained_local(
        hyperparams.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    # Needed for the cases when using TELlamaForCausalLM
    model.config.use_cache = False

    return model


def wrap_with_accelerator(model, hyperparams):
    # Create FP8 kwarg handler if required
    fp8_kwarg_handler = (
        [FP8RecipeKwargs(backend="te")] if hyperparams.mixed_precision == "fp8" else None
    )

    # Init HF accelerator that's used for training
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=fp8_kwarg_handler,
    )
    # accelerator.print(f'State: {accelerator.state}')
    train_dataloader = get_dataloaders(accelerator, hyperparams)

    # Wrap model, optimizer/scheduler, dataloaders in accelerate
    optimizer = AdamW(params=model.parameters(), lr=hyperparams.learning_rate, fused=True)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=hyperparams.num_training_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    return accelerator, model, optimizer, train_dataloader, lr_scheduler


def finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    train_dataloader = enumerate(train_dataloader)

    # Warmup iters
    for _ in range(hyperparams.num_warmup_steps):
        step, batch = next(train_dataloader)
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # Get the timers ready
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start.record()
    # Training iters
    for _ in range(hyperparams.num_training_steps):
        step, batch = next(train_dataloader)
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

    print(
        f"{hyperparams.num_training_steps} finetuning steps complete!\nAverage time taken per step:"
        f" {(start.elapsed_time(end)/hyperparams.num_training_steps):.0f} milliseconds"
    )


def restart_jupyter_notebook():
    # Try restarting the Jupyter kernel
    IPython.Application.instance().kernel.do_shutdown(True)

    # Check whether the device memory has been flushed
    if torch.cuda.memory_allocated() != 0:
        import warnings

        warnings.warn("The device memory hasn't been flushed, trying with a second method!")

        # Try restarting the Jupyter kernel another way
        # Restart the kernel
        from IPython.core.display import HTML

        HTML("<script>Jupyter.notebook.kernel.restart()</script>")

        if torch.cuda.memory_allocated() != 0:
            print(
                "The device memory hasn't been flushed, try manually restarting the Jupyter kernel!"
            )

    # Suppress the warnings
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
        torch.set_warn_always(False)
