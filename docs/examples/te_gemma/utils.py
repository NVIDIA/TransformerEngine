# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import sys
import IPython
import random
import string

from te_gemma_loading_weights import load_te_model
import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset


from te_gemma import TEGemmaForCausalLM, TEGemmaForCausalLMCudaGraphs

random.seed(42)
torch.manual_seed(42)


class RunConfiguration:
    def __init__(self):
        self.mixed_precision = "bf16"
        self.model_name = None

        # FP8 precision settings
        self.fp8 = False
        self.fp8_model_weights_filename = None
        self.fp8_model_init = False

        # Cuda graphs
        self.generation_cuda_graphs = False
        self.cuda_graphs_static_batch_size = 64
        self.cuda_graphs_static_max_seq_len = 512
        self.cuda_graphs_static_max_context_len = 512

        # Finetuning/calibration/generation settings
        self.dataset_name = "timdettmers/openassistant-guanaco"
        self.dataset_text_field = "text"
        self.learning_rate = 1.41e-5
        self.batch_size = 64
        self.max_seq_length = 512
        self.gradient_accumulation_steps = 1
        self.num_warmup_steps = 5
        self.num_training_steps = 10

        # Coalesced QKV params or not
        self.fuse_qkv_params = False

        # Attention
        self.is_paged = False

        # This is either provided by the user or it will be set when the
        # model weights are downloaded.
        self.weights_cache_dir = ""


# Global variable for the run configuration so that it can be easily accessed
# throughout the jupyter notebook with an `import * from utils` statement
run_config = RunConfiguration()


def get_dataloaders(run_config):
    """
    Returns a basic dataloader for the dataset which contains tokenized batches
    of text.
    """
    dataset = load_dataset(run_config.dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(run_config.model_name)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=run_config.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    # Tokenize the dataset
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # Simply pad to the multiple of 16 for both FP8 and BF16 precision
    pad_to_multiple_of = 16
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": run_config.batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader


def ensure_model_is_downloaded(run_config):
    """
    Downloads and caches the model weights if not already downloaded. A valid
    Huggingface Access Token is required to download the model weights.
    """
    assert run_config.model_name in [
        "google/gemma-7b",
    ], "Only Gemma 7B model is supported!"

    # Login using Huggingface Hub API
    from huggingface_hub import login

    try:
        login(run_config.hf_access_token)
    except Exception as e:
        if "Invalid token passed!" in str(e):
            print(
                "Please pass a valid HF Access Token! More info at"
                " https://huggingface.co/docs/hub/en/security-tokens."
            )
        else:
            print(f"Exception is {e}")

    # Download the model if it doesn't exist
    from huggingface_hub import snapshot_download

    supplied_cache_dir = (
        run_config.weights_cache_dir if run_config.weights_cache_dir != "" else None
    )
    run_config.weights_cache_dir = snapshot_download(
        repo_id=run_config.model_name, cache_dir=supplied_cache_dir
    )


def init_baseline_model(run_config):
    """
    Initializes a baseline HF Gemma model with the model name provided in
    the run_config.
    """

    # Download and cache the weights if not already downloaded
    ensure_model_is_downloaded(run_config)

    # Init the model
    config = AutoConfig.from_pretrained(run_config.model_name)

    # Make sure to use flash_attention to do iso comparison with TEGemmaModel
    config._attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        run_config.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    ).cuda()

    return model


def init_te_gemma_model(run_config):
    """
    Initializes a Gemma model with `GemmaDecoderLayer`s swapped with
    `TransformerLayer`s from TransformerEngine. In case CUDA Graphs are enabled,
    the model is initialized from `TEGemmaForCausalLMCudaGraphs` class.
    """

    # Download and cache the weights if not already downloaded
    ensure_model_is_downloaded(run_config)

    cls = TEGemmaForCausalLMCudaGraphs if run_config.generation_cuda_graphs else TEGemmaForCausalLM
    config = AutoConfig.from_pretrained(run_config.model_name)

    # Inject all fields from the `run_config` to the model `config` to make the
    # code simpler.
    for key, value in run_config.__dict__.items():
        setattr(config, key, value)

    # Initialize the model and move it to the GPU.
    model = load_te_model(cls, config).cuda()

    # Record the model if CUDA Graphs are enabled.
    if run_config.generation_cuda_graphs:
        model.record()

    return model


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


@torch.no_grad()
def run_forward_pass(model, run_config, num_iters):
    """
    Runs the forward pass of the model with sample data. Intended to use for
    warmup and/or calibration.
    """
    train_dataloader = get_dataloaders(run_config)

    model.train()
    train_dataloader = enumerate(train_dataloader)

    for _ in range(num_iters):
        _, batch = next(train_dataloader)
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


###############################################################################
# Benchmarking and example generation functions.
###############################################################################


def print_sample_of_generated_texts(model, run_config):
    """
    Prints a sample of generated texts from the input model.
    """

    tokenizer = AutoTokenizer.from_pretrained(run_config.model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        "Here are the two facts about GPUs:",
        "Some facts about NVIDIA:",
        "The fundamental theorem of calculus for the layman:",
        "A fact about AI:",
    ]

    # Repeat prompts to match batch size
    prompts *= run_config.batch_size // len(prompts)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    max_total_tokens = (
        run_config.max_seq_length
        if not run_config.generation_cuda_graphs
        else run_config.cuda_graphs_static_max_seq_len
    )

    max_length = inputs["input_ids"].size(1)
    new_length = ((max_length + 63) // 64) * max_total_tokens

    # Add padding to the left
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"], (new_length - max_length, 0), value=tokenizer.pad_token_id
    )

    # Add padding to the left (only intended for baseline generation with HF
    # which expects padding to the left)
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (new_length - max_length, 0), value=0
    )

    inputs["input_ids"] = inputs["input_ids"].cuda()
    inputs["attention_mask"] = inputs["attention_mask"].cuda()

    outputs = model.generate(**inputs, max_new_tokens=50)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def print_output(prompts, generated_texts, idx):
        print("=" * 30 + f" Generation example {idx+1} " + "=" * 30)
        print(f'Prompt: "{generated_texts[idx][: len(prompts[idx])]}"')
        print(f'Generated text: "{generated_texts[idx][len(prompts[idx]) :]}"')

    # Print the output from first two prompts
    for i in range(2):
        print_output(prompts, generated_texts, i)


def _generate_random_words(num_words, max_word_length):
    """
    Generates random words for the benchmark.
    """

    words = []
    for _ in range(num_words):
        word_length = random.randint(1, max_word_length)
        word = "".join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)
    return words


def benchmark_generation(model, run_config, context_length=20):
    """
    Benchmarks the generation time for a random input to the model.
    """

    batch_size = run_config.batch_size

    max_total_tokens = (
        run_config.max_seq_length
        if not run_config.generation_cuda_graphs
        else run_config.cuda_graphs_static_max_seq_len
    )
    max_new_tokens = max_total_tokens - context_length

    print("\n" + "=" * 80)
    print(
        f"Benchmarking for batch_size = {batch_size}, prefill tokens ="
        f" {context_length} and max new tokens = {max_new_tokens}"
    )

    input_str = _generate_random_words(batch_size, context_length)

    tokenizer = AutoTokenizer.from_pretrained(run_config.model_name)
    inputs = tokenizer(input_str, return_tensors="pt", padding=True)

    max_context_tokens = inputs["input_ids"].size(1)

    # Add padding to the left
    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (max_total_tokens - max_context_tokens, 0),
        value=tokenizer.pad_token_id,
    )

    # Add padding to the left (only intended for baseline generation with HF
    # which expects padding to the left)
    inputs["attention_mask"] = torch.nn.functional.pad(
        inputs["attention_mask"], (max_total_tokens - max_context_tokens, 0), value=0
    )

    inputs["input_ids"] = inputs["input_ids"].cuda()
    inputs["attention_mask"] = inputs["attention_mask"].cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()

    model.generate(inputs["input_ids"].cuda(), max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    end.record()

    print(f"Time: {start.elapsed_time(end)/1000:.2f} s.")
