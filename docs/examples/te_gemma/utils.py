# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import time
import sys
import IPython

from te_gemma_loading_weights import load_te_model

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils.dataclasses import FP8RecipeKwargs


from te_gemma import TEGemmaForCausalLM, TEGemmaForCausalLMCudaGraphs

class HyperParameters:
    def __init__(self):
        self.mixed_precision = "bf16"
        self.model_name = None 

        self.fp8 = False

        # Weights in fp8
        self.fp8_model_weights_filename = None
        self.fp8_model_init = False

        # Cuda graphs
        self.generation_cuda_graphs = False
        self.cuda_graphs_static_batch_size = 16
        self.cuda_graphs_static_max_seq_len = 256
        self.cuda_graphs_static_max_context_len = 16

        # Finetuning settings.
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 8
        self.max_seq_length = 256
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps=5
        self.num_training_steps=10

        # QKV format.
        self.fuse_qkv_params=False
        self.qkv_format = "bshd"
        
hyperparams = HyperParameters()

def get_dataloaders(accelerator:Accelerator, hyperparams):
    dataset = load_dataset(hyperparams.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=hyperparams.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )

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
    # make sure to use flash_attention to do iso comparison with TEGemmaModel
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        hyperparams.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    return model

def init_te_gemma_model(hyperparams):
    cls = TEGemmaForCausalLMCudaGraphs if hyperparams.generation_cuda_graphs else TEGemmaForCausalLM
    config = AutoConfig.from_pretrained(hyperparams.model_name)
    config._attn_implementation = "flash_attention_2"
    # Adding all params from the hyperparams to the config to make the code simpler.
    for key, value in hyperparams.__dict__.items():
                setattr(config, key, value)
    model = load_te_model(cls, config)
    if hyperparams.generation_cuda_graphs:
        model.record()
    return model


def wrap_with_accelerator(model, hyperparams):
    # Create FP8 kwarg handler if required
    fp8_kwarg_handler = [FP8RecipeKwargs(backend="te")] if hyperparams.mixed_precision == "fp8" else None

    # Init HF accelerator that's used for training
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=fp8_kwarg_handler
    )
    #accelerator.print(f'State: {accelerator.state}')
    train_dataloader = get_dataloaders(accelerator, hyperparams)

    # Wrap model, optimizer/scheduler, dataloaders in accelerate
    optimizer = AdamW(params = model.parameters(), lr=hyperparams.learning_rate, fused=True)
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
    optimizer.zero_grad()
    train_dataloader = enumerate(train_dataloader)

    def run_iters(num_iters):
        for _ in range(num_iters):
            _, batch = next(train_dataloader)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    run_iters(hyperparams.num_warmup_steps) # Warmup iters

    # Get the timers ready
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    start.record()
    run_iters(hyperparams.num_training_steps) # Training iters
    torch.cuda.synchronize()
    end.record()
    accelerator.end_training()

    print(f"""{hyperparams.num_training_steps} finetuning steps complete!\n
          Average time taken per step: 
          {(start.elapsed_time(end)/hyperparams.num_training_steps):.0f} 
          milliseconds""")

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
            print("The device memory hasn't been flushed, try manually restarting the Jupyter kernel!")

    # Suppress the warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        torch.set_warn_always(False)

@torch.no_grad()
def run_forward_pass(model, hyperparams, num_iters):
    """
        It runs num_iters forward passes with sample data.
    """
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision="no"
    )
    train_dataloader = get_dataloaders(accelerator, hyperparams)

    model.train()
    train_dataloader = enumerate(train_dataloader)

    for _ in range(num_iters):
        _, batch = next(train_dataloader)
        batch["input_ids"] = batch["input_ids"].cuda()
        model(batch["input_ids"])

"""
    Benchmarking and example generation functions.
"""

def print_sample_of_generated_texts(model):
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
    inputs = tokenizer(["Another string ... ", "I "] * 32, return_tensors="pt", padding=True)


    max_length = inputs['input_ids'].size(1)
    new_length = ((max_length + 63) // 64) * 128
    inputs['input_ids'] = torch.nn.functional.pad(inputs['input_ids'], (new_length - max_length, 0), value=tokenizer.pad_token_id)
    inputs['attention_mask'] = torch.nn.functional.pad(inputs['attention_mask'], (new_length - max_length, 0), value=0)


    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['attention_mask'] = inputs['attention_mask'].cuda()

    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for text in generated_texts[:2]:
        print(text)
        print("=" * 100)



def benchmark_generation(model):
    batch_size = 64
    context_length = 128
    max_new_tokens = 1024 - 128
    print(f"Benchmarking for batch_size={batch_size} and total tokens = {context_length + max_new_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
    inputs = tokenizer(["a" * context_length] * batch_size, return_tensors="pt", padding=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    
    model.generate(
        inputs['input_ids'].cuda(),
        max_new_tokens=max_new_tokens
    )
    torch.cuda.synchronize()
    end.record()
    
    print(f"Benchmark with context_length={context_length} and max_new_tokens={max_new_tokens} took {start.elapsed_time(end)} ms.")
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
