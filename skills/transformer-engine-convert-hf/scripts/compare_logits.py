#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare a Transformer Engine recipe with the Hugging Face model it was converted from.

The weight mapping check cannot tell two same-shaped weights apart, and it says nothing about
the forward pass. This script settles both: it loads the Hugging Face (HF) model and the
recipe's Transformer Engine (TE) model from the same checkpoint, runs the same input through
them and compares the outputs.

How close is close enough depends on the model, so the tolerance is measured instead of
guessed. The HF model is also run in float32; the distance between HF in BF16 and HF in
float32 is the rounding noise of the model itself, and the TE model passes when it is no
further from the float32 reference than about that noise. A misplaced weight, a wrong norm or
a wrong constructor argument puts it tens of times further away.

A model that does not fit on the GPU is compared a few layers at a time (`--layers-per-slice`):
each slice of the checkpoint is cut out with make_subcheckpoint.py and loaded by both models,
and the hidden states of one slice are the input of the next. The TE model is fed the HF
hidden states (the error of that slice alone) and its own (the error accumulated from layer
0), so the last slice gives the logits of the whole TE model.

`--train-step` also runs one forward and backward pass under te.autocast with the
low-precision recipe the model is meant to train in, and `--check-weights` the weight mapping
check; both run on every slice.

Usage, in the environment where the recipe runs:
    python compare_logits.py te_llama.py /path/to/checkpoint --train-step auto
    python compare_logits.py te_big.py /path/to/checkpoint --layers-per-slice 4 --check-weights
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import torch
import transformers

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill and recipe directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_weight_mapping  # pylint: disable=wrong-import-position
import make_subcheckpoint  # pylint: disable=wrong-import-position

TEXT = (
    "The Transformer architecture replaced recurrence with attention. In a decoder-only "
    "language model every layer mixes information across positions and then transforms each "
    "position with a feed-forward network. Mixture-of-experts models route each token to a "
    "few of many feed-forward experts, so only a fraction of the parameters is used per "
    "token.\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
    "    return a\n\nLa capitale de la France est Paris, et la capitale de l'Allemagne est "
    "Berlin. In 1969 Apollo 11 landed on the Moon; Neil Armstrong and Buzz Aldrin walked on "
    "its surface while Michael Collins stayed in orbit. The speed of light in vacuum is "
    "299792458 metres per second, and water boils at 100 degrees Celsius at sea level. "
)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
# The TE model may be this many times further from the float32 reference than HF is.
NOISE_FACTOR = 2.0
NOISE_FLOOR = 2e-3
# Without a reference: the largest relative error accepted as BF16 rounding.
MAX_ERROR_WITHOUT_REFERENCE = 0.05


def load_hf(model_dir, dtype):
    """Load the HF model class named in config.json."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        architecture = json.load(handle)["architectures"][0]
    model_class = getattr(transformers, architecture)
    try:
        model = model_class.from_pretrained(model_dir, dtype=dtype)
    except TypeError:  # older transformers
        model = model_class.from_pretrained(model_dir, torch_dtype=dtype)
    return model.cuda().eval()


def load_te(recipe_path, model_dir, dtype, model_class=None):
    """Load the recipe's TE model with its own loader."""
    model = check_weight_mapping.load_recipe_model(recipe_path, model_dir, model_class, dtype)
    return model.cuda().eval()


def make_input_ids(model_dir, vocab_size, batch, seq):
    """Token ids of real text when the checkpoint has a tokenizer, random ids otherwise."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        ids = tokenizer(TEXT * (1 + batch * seq // 100), return_tensors="pt").input_ids[0]
        if ids.numel() >= batch * seq:
            return ids[: batch * seq].reshape(batch, seq), "real text"
    except Exception:  # pylint: disable=broad-except
        pass
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab_size, (batch, seq), generator=generator), "random token ids"


def run(model, layers_path, inputs, dtype):
    """Return (output of the last decoder layer, logits), both float32 on the CPU."""
    captured = {}

    def hook(_module, _args, output):
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.get_submodule(layers_path)[-1].register_forward_hook(hook)
    inputs = {
        key: value.cuda() if value.dtype == torch.long else value.to("cuda", dtype)
        for key, value in inputs.items()
    }
    try:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
    finally:
        handle.remove()
    return captured["hidden"].float().cpu(), logits.float().cpu()


def relative_error(test, reference):
    """||test - reference|| / ||reference||."""
    return ((test.double() - reference.double()).norm() / reference.double().norm()).item()


def top1_agreement(test, reference):
    """Fraction of positions where both predict the same token."""
    return (test.argmax(-1) == reference.argmax(-1)).float().mean().item()


def release_gpu_memory():
    """Return the memory of models the caller has deleted to the GPU."""
    gc.collect()
    torch.cuda.empty_cache()


def train_step(model, inputs, labels, dtype, choice):
    """One forward and backward pass under te.autocast; return (ok, message)."""
    import transformer_engine.pytorch as te  # pylint: disable=import-outside-toplevel
    from transformer_engine.common import recipe  # pylint: disable=import-outside-toplevel

    if choice == "auto":  # the choice step 6 of the skill makes
        choice = "mxfp8" if te.is_mxfp8_available() else "fp8" if te.is_fp8_available() else None
    if choice is None:
        return True, "skipped: neither MXFP8 nor FP8 is available on this GPU"
    low_precision = {
        "mxfp8": recipe.MXFP8BlockScaling,
        "fp8": recipe.DelayedScaling,
        "fp8-current": recipe.Float8CurrentScaling,
    }[choice]()
    inputs = {
        key: value.cuda() if value.dtype == torch.long else value.to("cuda", dtype)
        for key, value in inputs.items()
    }
    labels = labels.cuda()

    def loss_of(logits):
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )

    model.train()
    with te.autocast(enabled=False), torch.no_grad():
        high = loss_of(model(**inputs, use_cache=False).logits).item()
    model.zero_grad(set_to_none=True)
    for _ in range(2):  # delayed scaling needs statistics before its numbers mean anything
        with te.autocast(enabled=True, recipe=low_precision), torch.no_grad():
            model(**inputs, use_cache=False)
    with te.autocast(enabled=True, recipe=low_precision):
        logits = model(**inputs, use_cache=False).logits
    loss = loss_of(logits)
    loss.backward()  # outside autocast
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    finite = all(torch.isfinite(grad.float()).all().item() for grad in grads)
    message = (
        f"{choice}: loss {loss.item():.4f} (BF16 {high:.4f}), "
        f"{len(grads)} parameters received gradients, all finite: {finite}"
    )
    ok = bool(torch.isfinite(loss)) and finite and len(grads) > 0
    # A loss far from the BF16 loss means the low-precision path computes something else.
    ok = ok and abs(loss.item() - high) <= 0.25 * max(abs(high), 1.0)
    return ok, message


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="the recipe file, for example te_llama.py")
    parser.add_argument("model_dir", help="local checkpoint directory with .safetensors files")
    parser.add_argument("--model-class", help="recipe class that has from_pretrained_local")
    parser.add_argument(
        "--layers-per-slice",
        type=int,
        default=0,
        help="compare this many decoder layers at a time; 0 (default) loads the whole model",
    )
    parser.add_argument("--slices", help="first:last slice to run, to resume a sliced run")
    parser.add_argument("--work-dir", help="where slices and hidden states are written")
    parser.add_argument("--precision", default="bf16", choices=list(DTYPES))
    parser.add_argument(
        "--no-reference", action="store_true", help="skip the float32 HF run that sets the noise"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=96)
    parser.add_argument(
        "--train-step",
        default="none",
        choices=["none", "auto", "mxfp8", "fp8", "fp8-current"],
        help="also run a forward and backward pass under te.autocast with this recipe",
    )
    parser.add_argument(
        "--check-weights", action="store_true", help="also run the weight mapping check"
    )
    parser.add_argument("--ignore", help="regular expression for checkpoint entries to skip")
    args = parser.parse_args()
    dtype = DTYPES[args.precision]
    reference = not args.no_reference and args.precision != "fp32"

    import transformer_engine.pytorch as te  # pylint: disable=import-outside-toplevel

    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    names = make_subcheckpoint.checkpoint_entries(args.model_dir)
    prefix, depth, _ = make_subcheckpoint.find_layer_prefix(names, config)
    layers_path = prefix.rstrip(".")
    ignored = make_subcheckpoint.ignored_by_hf(args.model_dir)  # entries HF does not load
    if args.ignore:
        ignored = f"{ignored}|({args.ignore})"

    per_slice = args.layers_per_slice or depth
    slices = [(first, min(first + per_slice, depth)) for first in range(0, depth, per_slice)]
    run_first, run_last = 0, len(slices)
    if args.slices:
        run_first, run_last = (int(part) for part in args.slices.split(":"))
    # Slices and hidden states go to a directory of this run's own unless one is named: two
    # runs must not share it, and resuming with --slices needs the earlier run's directory.
    work_dir = args.work_dir
    if work_dir is None:
        if args.slices:
            parser.error("--slices resumes an earlier run: pass that run's --work-dir")
        base = "/dev/shm" if os.path.isdir("/dev/shm") else None
        work_dir = tempfile.mkdtemp(prefix="compare_logits_", dir=base)
    os.makedirs(work_dir, exist_ok=True)

    def stream(kind, index):
        """File with the hidden states one model produced for one slice."""
        return os.path.join(work_dir, f"hidden_{kind}_{index}.pt")

    print(f"{depth} decoder layers at {layers_path}; {len(slices)} slice(s) of {per_slice}")
    failed = []
    for index, (first, last) in enumerate(slices):
        if not run_first <= index < run_last:
            continue
        whole = (first, last) == (0, depth)
        model_dir = args.model_dir
        if not whole:
            model_dir = os.path.join(work_dir, "slice")
            print(make_subcheckpoint.make_subcheckpoint(args.model_dir, model_dir, first, last))

        # Inputs: token ids for the slice with layer 0, hidden states afterwards.
        hf_model = load_hf(model_dir, dtype)
        if index == 0:
            vocab_size = hf_model.get_input_embeddings().num_embeddings
            ids, kind = make_input_ids(args.model_dir, vocab_size, args.batch, args.seq)
            print(f"input: {args.batch} x {args.seq} tokens of {kind}")
            hf_inputs = te_inputs = {"input_ids": ids}
        else:
            hf_inputs = {"inputs_embeds": torch.load(stream("hf", index - 1))}
            te_inputs = {"inputs_embeds": torch.load(stream("te", index - 1))}
        hf_hidden, hf_logits = run(hf_model, layers_path, hf_inputs, dtype)
        torch.save(hf_hidden, stream("hf", index))
        del hf_model
        release_gpu_memory()

        # Float32 HF: on the same input as the others (this slice's rounding noise) and on its
        # own hidden states (the rounding noise accumulated from layer 0).
        ref_hidden = ref_logits = None
        if reference:
            ref_model = load_hf(model_dir, torch.float32)
            ref_hidden, ref_logits = run(ref_model, layers_path, hf_inputs, torch.float32)
            if index > 0:
                ref_inputs = {"inputs_embeds": torch.load(stream("ref", index - 1))}
                ref_own_hidden, ref_logits = run(ref_model, layers_path, ref_inputs, torch.float32)
                torch.save(ref_own_hidden, stream("ref", index))
            else:
                torch.save(ref_hidden, stream("ref", index))
            del ref_model
            release_gpu_memory()

        te_model = load_te(args.recipe, model_dir, dtype, args.model_class)
        with te.autocast(enabled=False):
            te_hidden, te_logits = run(te_model, layers_path, hf_inputs, dtype)
            if index == 0:
                own_hidden, own_logits = te_hidden, te_logits
            else:
                own_hidden, own_logits = run(te_model, layers_path, te_inputs, dtype)
        torch.save(own_hidden, stream("te", index))

        # This slice alone: both models got the same input.
        error = relative_error(te_hidden, hf_hidden)
        line = f"layers {first}:{last}  TE vs HF {error:.2e}"
        if ref_hidden is not None:
            te_error = relative_error(te_hidden, ref_hidden)
            hf_error = relative_error(hf_hidden, ref_hidden)
            ok = te_error <= max(NOISE_FACTOR * hf_error, NOISE_FLOOR)
            line += f"  | vs float32 HF: TE {te_error:.2e}, HF itself {hf_error:.2e}"
        else:
            ok = error <= MAX_ERROR_WITHOUT_REFERENCE
        line += f"  | top-1 agreement {top1_agreement(te_logits, hf_logits):.4f}"
        if not whole:
            line += f"  | from layer 0: {relative_error(own_hidden, hf_hidden):.2e}"
        print(line + ("  ok" if ok else "  FAILED"))
        if not ok:
            failed.append(f"layers {first}:{last} differ from HF by more than rounding")

        if args.check_weights:
            problems, table = check_weight_mapping.check_weight_mapping(
                check_weight_mapping.Checkpoint(model_dir), te_model.state_dict(), ignored
            )
            if index == run_first:
                print("Where the converted weights went:")
                for row in table:
                    print("  " + row)
            print(f"layers {first}:{last}  weight mapping: {len(problems)} problem(s)")
            for problem in problems[: check_weight_mapping.MAX_PROBLEMS_SHOWN]:
                print("  " + problem)
            if problems:
                failed.append(f"layers {first}:{last}: weight mapping")

        if index == len(slices) - 1:
            line = (
                f"whole model, logits: TE vs HF {relative_error(own_logits, hf_logits):.2e}, "
                f"top-1 agreement {top1_agreement(own_logits, hf_logits):.4f}"
            )
            ok = True
            if ref_logits is not None:
                te_error = relative_error(own_logits, ref_logits)
                hf_error = relative_error(hf_logits, ref_logits)
                ok = te_error <= max(NOISE_FACTOR * hf_error, NOISE_FLOOR)
                line += (
                    f"  | vs float32 HF: TE {te_error:.2e} (top-1 "
                    f"{top1_agreement(own_logits, ref_logits):.4f}), HF itself {hf_error:.2e} "
                    f"(top-1 {top1_agreement(hf_logits, ref_logits):.4f})"
                )
            print(line + ("  ok" if ok else "  FAILED"))
            if not ok:
                failed.append("the whole model's logits differ from HF by more than rounding")
        if args.train_step != "none":  # on every slice, so that every layer is trained once
            try:
                ok, message = train_step(
                    te_model, hf_inputs, hf_logits.argmax(-1), dtype, args.train_step
                )
            except Exception as error:  # pylint: disable=broad-except
                ok, message = False, f"{type(error).__name__}: {str(error)[:600]}"
            print(f"layers {first}:{last}  train step " + ("ok: " if ok else "FAILED: ") + message)
            if not ok:
                failed.append(f"layers {first}:{last}: the low-precision train step")
        del te_model
        release_gpu_memory()
        if not whole:
            shutil.rmtree(model_dir)
        if failed and failed[-1].endswith("train step"):
            break  # a failed GPU kernel can leave the process unusable

    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    if args.work_dir is None:
        shutil.rmtree(work_dir)
    if failed:
        print(f"\nFAILED: {len(failed)} problem(s)")
        for problem in failed:
            print("  " + problem)
        return 1
    print("\nOK: the TE model matches the HF model to within rounding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
