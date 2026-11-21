#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check that a Transformer Engine recipe carried over every Hugging Face weight exactly.

The script loads the recipe's Transformer Engine (TE) model from a local checkpoint and then
looks for every Hugging Face (HF) weight inside the TE weights. It does not use the recipe's
own mapping, so it works for any recipe and cannot repeat a mistake made there:

* a weight whose name is unchanged (embeddings, final norm, LM head) must be identical;
* every other weight is compared row by row along its last dimension: each row must appear,
  byte for byte, in a TE weight of the same layer;
* every row of every TE weight must come from the checkpoint.

This finds weights that were never copied, copied in part, overwritten, copied into another
layer or changed in value, and TE parameters left at their initial values. It prints where
each weight went. It does not know what a row means, so two same-shaped weights stored in
each other's place (gate and up, key and value) still pass: read the printed table, and
compare logits with the HF model to settle the order.

Weights must be copied without changing their values or transposing them, and a 1-D weight
must be copied whole.

Usage, in the environment where the recipe runs:
    python check_weight_mapping.py te_llama.py /path/to/checkpoint
"""

import argparse
import collections
import collections.abc
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

import safetensors
import torch
import transformers

DIGEST_BYTES = 16
MAX_PROBLEMS_SHOWN = 40


class Checkpoint(collections.abc.Mapping):
    """The weights in a directory of .safetensors files, read one tensor at a time."""

    def __init__(self, model_dir):
        self.files = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safetensors.safe_open(path, framework="pt") as shard:
                self.files.update({name: path for name in shard.keys()})

    def __getitem__(self, name):
        with safetensors.safe_open(self.files[name], framework="pt") as shard:
            return shard.get_tensor(name)

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)


def layer_of(name):
    """Return the name's prefix through its first number, e.g. "model.layers.3."."""
    match = re.match(r"\D*\d+\.", name)
    return match.group() if match else ""


def row_digests(tensor, dtype):
    """Return one digest of the exact bytes of each row (along the last dimension)."""
    tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    rows = tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() > 1 else tensor.reshape(1, -1)
    raw = rows.view(torch.uint8).numpy()
    return [hashlib.blake2b(row.tobytes(), digest_size=DIGEST_BYTES).digest() for row in raw]


def describe(indices, total):
    """Summarize which rows of a TE weight an HF weight occupies."""
    low, high = min(indices), max(indices) + 1
    if len(indices) == total:
        return "all rows"
    if high - low == len(indices):
        return f"rows {low}:{high} of {total}"
    return f"{len(indices)} rows between {low} and {high} of {total}, interleaved"


def check_weight_mapping(hf_weights, te_state_dict, ignore=None):
    """Compare HF weights with the converted TE weights.

    Returns (problems, table): a list of messages, empty when every weight is accounted for,
    and one line per kind of converted weight saying where it went.
    """
    te_weights = {
        name: value
        for name, value in te_state_dict.items()
        if value.is_floating_point() and not name.endswith("_extra_state")
    }
    dtype = collections.Counter(value.dtype for value in te_weights.values()).most_common(1)[0][0]
    kept = set(te_weights) & set(hf_weights)
    # Tied weights (LM head and embedding) share storage; one copy accounts for both.
    accounted = {te_weights[name].data_ptr() for name in kept}

    # Rows of the remaining TE weights that no HF weight has claimed yet.
    unclaimed = collections.defaultdict(list)
    totals = {}
    for name in sorted(set(te_weights) - kept):
        if te_weights[name].data_ptr() in accounted:
            continue
        accounted.add(te_weights[name].data_ptr())
        digests = row_digests(te_weights[name], dtype)
        totals[name] = len(digests)
        for index, digest in enumerate(digests):
            unclaimed[layer_of(name), digest].append((name, index))

    problems, table = [], []
    for name in hf_weights:
        if ignore and re.search(ignore, name):
            continue
        value = hf_weights[name]
        if name in kept:
            target = te_weights[name]
            if not torch.equal(value.to(target.dtype), target.cpu()):
                problems.append(f"{name}: differs from the TE weight of the same name")
            continue
        layer, landed, missing = layer_of(name), collections.defaultdict(list), 0
        for digest in row_digests(value, dtype):
            candidates = unclaimed.get((layer, digest))
            if candidates:
                te_name, index = candidates.pop()
                landed[te_name].append(index)
            else:
                missing += 1
        if missing:
            where = f"the TE weights under {layer}" if layer else "the TE weights"
            problems.append(f"{name}: {missing} of its rows are not in {where}")
        places = sorted(
            f"{te_name} ({describe(rows, totals[te_name])})" for te_name, rows in landed.items()
        )
        line = f"{name} -> {', '.join(places) or 'not found'}"
        line = line.replace(layer, re.sub(r"\d+", "N", layer)) if layer else line
        if line not in table:
            table.append(line)

    leftover = collections.Counter(name for rows in unclaimed.values() for name, _ in rows)
    for name, count in sorted(leftover.items()):
        problems.append(f"{name}: {count} of its rows do not come from the checkpoint")
    return problems, table


def load_recipe_model(recipe_path, model_dir, model_class=None):
    """Import the recipe and build its TE model from the checkpoint."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(recipe_path)))
    spec = importlib.util.spec_from_file_location("te_recipe", recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)

    loaders = {
        name: obj
        for name, obj in vars(recipe).items()
        if isinstance(obj, type) and "from_pretrained_local" in vars(obj)
    }
    if model_class:
        loaders = {name: obj for name, obj in loaders.items() if name == model_class}
    if len(loaders) != 1:
        raise SystemExit(
            f"Expected one class with from_pretrained_local in {recipe_path}, found "
            f"{sorted(loaders)}. Pass --model-class."
        )
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        config = transformers.AutoConfig.for_model(**json.load(handle))
    (loader,) = loaders.values()
    return loader.from_pretrained_local(model_dir, config=config, torch_dtype=torch.bfloat16)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="the recipe file, for example te_llama.py")
    parser.add_argument("model_dir", help="local checkpoint directory with .safetensors files")
    parser.add_argument("--model-class", help="recipe class that has from_pretrained_local")
    parser.add_argument("--ignore", help="regular expression for checkpoint entries to skip")
    args = parser.parse_args()

    checkpoint = Checkpoint(args.model_dir)
    if not checkpoint:
        raise SystemExit(f"No .safetensors files in {args.model_dir}")
    model = load_recipe_model(args.recipe, args.model_dir, args.model_class)
    problems, table = check_weight_mapping(checkpoint, model.state_dict(), args.ignore)

    print("Where the converted weights went:")
    for line in table:
        print("  " + line)
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for problem in problems[:MAX_PROBLEMS_SHOWN]:
            print("  " + problem)
        return 1
    print(f"\nOK: all {len(checkpoint)} checkpoint weights are in the TE model, byte for byte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
