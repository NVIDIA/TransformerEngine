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
  byte for byte, in a TE weight of the same layer, and the rows must keep their order (rows
  of one weight may be spread over several TE weights or interleaved with other rows, but
  not shuffled; expert 5 must not end up behind expert 7);
* every row of every TE weight must come from the checkpoint.

This finds weights that were never copied, copied in part, overwritten, copied into another
layer, reordered or changed in value, and TE parameters left at their initial values. It
prints where each weight went.

It does not know what a row means, so two weights of the same shape stored in each other's
place (gate and up, key and value, two norms) still pass. Only the model's output can tell
those apart: run compare_logits.py as well, and treat this check as necessary, not sufficient.

Weights must be copied without changing their values or transposing them, and a 1-D weight
must be copied whole. Checkpoint entries that HF itself does not load are skipped.

The whole TE model is loaded. For a model that does not fit, check a few layers at a time:
`compare_logits.py --layers-per-slice N --check-weights` runs this check on every slice.

Usage, in the environment where the recipe runs:
    python check_weight_mapping.py te_llama.py /path/to/checkpoint
"""

import argparse
import collections
import collections.abc
import glob
import importlib.util
import json
import os
import re
import sys

import safetensors
import torch
import transformers

sys.dont_write_bytecode = True  # keep __pycache__ out of the skill and recipe directories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_subcheckpoint import ignored_by_hf  # pylint: disable=wrong-import-position

# Rows are fingerprinted this many elements at a time.
FINGERPRINT_CHUNK = 1 << 27
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


def as_rows(tensor):
    """View a weight as rows along its last dimension; a 1-D weight is one row."""
    tensor = tensor.detach()
    return tensor.reshape(-1, tensor.shape[-1]) if tensor.dim() > 1 else tensor.reshape(1, -1)


_PROJECTIONS = {}


def row_digests(tensor, dtype):
    """Return one fingerprint of the exact bits of each row (along the last dimension).

    A fingerprint is the row width and two 64-bit projections of the row's raw bits onto fixed
    random vectors, computed on the GPU when there is one. Matches found this way are
    confirmed with an exact comparison afterwards.
    """
    rows = as_rows(tensor).to(dtype).contiguous()
    width = rows.shape[1]
    device = rows.device if rows.is_cuda or not torch.cuda.is_available() else torch.device("cuda")
    if (width, device) not in _PROJECTIONS:
        generator = torch.Generator().manual_seed(width)
        _PROJECTIONS[width, device] = torch.randint(
            -(2**62), 2**62, (2, width), generator=generator, dtype=torch.int64
        ).to(device)
    weights = _PROJECTIONS[width, device]
    bits = rows.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[rows.element_size()])
    digests = []
    for chunk in bits.split(max(1, FINGERPRINT_CHUNK // width)):
        raw = chunk.to(device).to(torch.int64)  # sums wrap around, which is fine for a hash
        pairs = torch.stack([(raw * weights[0]).sum(-1), (raw * weights[1]).sum(-1)], dim=-1)
        digests += [(width, first, second) for first, second in pairs.tolist()]
    return digests


def describe(indices, total):
    """Summarize which rows of a TE weight an HF weight occupies."""
    low, high = min(indices), max(indices) + 1
    if len(indices) == total:
        return "all rows"
    if high - low == len(indices):
        return f"rows {low}:{high} of {total}"
    return f"{len(indices)} rows between {low} and {high} of {total}, interleaved"


def natural_key(name):
    """Sort key that orders "weight2" before "weight10"."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def layer_generic(name, layer):
    """Replace the layer number with N."""
    if not layer or not name.startswith(layer):
        return name
    return re.sub(r"\d+", "N", layer) + name[len(layer) :]


def without_indices(name, layer):
    """Replace the layer number with N and every later index (an expert index) with {i}.

    Only whole name components and the number after "weight" or "bias" count as an index, so
    "experts.3.w1.weight" becomes "experts.{i}.w1.weight" and w1, w2, w3 stay apart.
    """
    if not layer or not name.startswith(layer):
        return name
    rest = re.sub(r"(?<![^.])\d+(?![^.])", "{i}", name[len(layer) :])
    rest = re.sub(r"(weight|bias)\d+$", r"\1{i}", rest)
    return re.sub(r"\d+", "N", layer) + rest


def check_weight_mapping(hf_weights, te_state_dict, ignore=None, allow_reorder=None):
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

    # Rows of the remaining TE weights that no HF weight has claimed yet, in natural order so
    # that identical rows are claimed first come, first served.
    unclaimed = collections.defaultdict(collections.deque)
    totals = {}
    for name in sorted(set(te_weights) - kept, key=natural_key):
        if te_weights[name].data_ptr() in accounted:
            continue
        accounted.add(te_weights[name].data_ptr())
        digests = row_digests(te_weights[name], dtype)
        totals[name] = len(digests)
        for index, digest in enumerate(digests):
            unclaimed[layer_of(name), digest].append((name, index))

    problems, table = [], collections.Counter()
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
        sources = collections.defaultdict(list)  # TE name -> rows of this weight, same order
        previous, reordered = None, None
        for row, digest in enumerate(row_digests(value, dtype)):
            candidates = unclaimed.get((layer, digest))
            if candidates:
                te_name, index = candidates.popleft()
                landed[te_name].append(index)
                sources[te_name].append(row)
                place = (natural_key(te_name), index)
                if previous is not None and place < previous and reordered is None:
                    reordered = f"{te_name} row {index}"
                previous = place
            else:
                missing += 1
        if missing:
            where = f"the TE weights under {layer}" if layer else "the TE weights"
            problems.append(f"{name}: {missing} of its rows are not in {where}")
        for te_name, indices in landed.items():  # a fingerprint is not proof: compare the bits
            copied = as_rows(te_weights[te_name])[torch.tensor(indices)].cpu()
            if not torch.equal(as_rows(value)[torch.tensor(sources[te_name])].to(dtype), copied):
                problems.append(f"{name}: differs from its rows in {te_name}")
        if reordered and not (allow_reorder and re.search(allow_reorder, name)):
            problems.append(
                f"{name}: its rows are in a different order in the TE weights (first at "
                f"{reordered}); if the layout is intended, pass --allow-reorder"
            )
        # Destinations that differ only in an index (one per expert) share one entry.
        per_index = without_indices(name, layer) != without_indices(name, layer).replace("{i}", "")
        groups = collections.defaultdict(list)
        for te_name, rows in landed.items():
            where = describe(rows, totals[te_name])
            groups[f"{without_indices(te_name, layer)} ({where})"].append(te_name)
        places = []
        for collapsed, members in groups.items():
            if len(members) > 1:
                places.append(f"{collapsed} x{len(members)}")
            elif per_index:
                places.append(collapsed)
            else:  # keep the exact name: the index says which weight it is
                places.append(
                    collapsed.replace(
                        without_indices(members[0], layer), layer_generic(members[0], layer), 1
                    )
                )
        table[f"{without_indices(name, layer)} -> {', '.join(sorted(places)) or 'not found'}"] += 1

    leftover = collections.Counter(name for rows in unclaimed.values() for name, _ in rows)
    for name, count in sorted(leftover.items()):
        problems.append(f"{name}: {count} of its rows do not come from the checkpoint")
    return problems, list(table)


_RECIPES = {}


def load_recipe_model(recipe_path, model_dir, model_class=None, dtype=torch.bfloat16):
    """Import the recipe and build its TE model from the checkpoint."""
    recipe_path = os.path.abspath(recipe_path)
    if recipe_path not in _RECIPES:  # imported once, so a sliced run uses one version of it
        sys.path.insert(0, os.path.dirname(recipe_path))
        spec = importlib.util.spec_from_file_location("te_recipe", recipe_path)
        _RECIPES[recipe_path] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RECIPES[recipe_path])
    recipe = _RECIPES[recipe_path]

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
    return loader.from_pretrained_local(model_dir, config=config, torch_dtype=dtype)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="the recipe file, for example te_llama.py")
    parser.add_argument("model_dir", help="local checkpoint directory with .safetensors files")
    parser.add_argument("--model-class", help="recipe class that has from_pretrained_local")
    parser.add_argument("--ignore", help="regular expression for checkpoint entries to skip")
    parser.add_argument(
        "--allow-reorder",
        help="regular expression for checkpoint weights whose rows are reordered on purpose",
    )
    args = parser.parse_args()

    checkpoint = Checkpoint(args.model_dir)
    if not checkpoint:
        raise SystemExit(f"No .safetensors files in {args.model_dir}")
    model = load_recipe_model(args.recipe, args.model_dir, args.model_class)
    ignore = "|".join(filter(None, [args.ignore, ignored_by_hf(args.model_dir)])) or None
    problems, table = check_weight_mapping(
        checkpoint, model.state_dict(), ignore, args.allow_reorder
    )
    skipped = sum(1 for name in checkpoint if ignore and re.search(ignore, name))

    print("Where the converted weights went:")
    for line in table:
        print("  " + line)
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for problem in problems[:MAX_PROBLEMS_SHOWN]:
            print("  " + problem)
        return 1
    note = f" ({skipped} entries that Hugging Face does not load were skipped)" if skipped else ""
    print(
        f"\nOK: all {len(checkpoint) - skipped} checkpoint weights are in the TE model, "
        f"byte for byte{note}."
    )
    print("Same-shaped weights can still be swapped: run compare_logits.py to settle the order.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
