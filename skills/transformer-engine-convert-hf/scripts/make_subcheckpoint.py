#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cut a range of decoder layers out of a Hugging Face checkpoint.

A model that does not fit on the GPU can still be checked a few layers at a time. This writes
a checkpoint directory that holds decoder layers [first, last) renumbered from 0, every weight
outside the decoder layers (embeddings, final norm, LM head, other towers) and a config.json
with the matching depth. Hugging Face's `from_pretrained` and a recipe's loader both read it
like any other checkpoint, so nothing in the model under test is special-cased.

The decoder layers are found from the checkpoint itself: the weight-name prefix that is
numbered 0 to `num_hidden_layers - 1`, for example "model.layers." or
"model.language_model.layers.". Per-layer config lists such as `layer_types` are cut to the
same range.

Usage:
    python make_subcheckpoint.py /path/to/checkpoint /dev/shm/layers_0_4 --layers 0:4
"""

import argparse
import collections
import glob
import json
import os
import re
import shutil

import safetensors
import safetensors.torch

MAX_COPIED_FILE_BYTES = 64 * 2**20


def checkpoint_entries(model_dir):
    """Return {weight name: shard path} for every .safetensors file in the directory."""
    entries = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safetensors.safe_open(path, framework="pt") as shard:
            entries.update({name: path for name in shard.keys()})
    if not entries:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    return entries


# Buffers of old checkpoints that Hugging Face drops for every model
# (PreTrainedModel._adjust_missing_and_unexpected_keys), on top of each class's own list.
LEGACY_BUFFERS = [r"rotary_emb\.inv_freq", r"(^|\.)position_ids$"]


def ignored_by_hf(model_dir):
    """Return a regular expression for the checkpoint entries HF does not load."""
    import transformers  # pylint: disable=import-outside-toplevel

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        architectures = json.load(handle).get("architectures") or []
    patterns = list(LEGACY_BUFFERS)
    for architecture in architectures:
        hf_class = getattr(transformers, architecture, None)
        patterns += getattr(hf_class, "_keys_to_ignore_on_load_unexpected", None) or []
    return "|".join(f"({pattern})" for pattern in patterns)


def find_depth_configs(config):
    """Return the (possibly nested) config dicts that have `num_hidden_layers`."""
    found = []
    if isinstance(config, dict):
        if isinstance(config.get("num_hidden_layers"), int):
            found.append(config)
        for value in config.values():
            found.extend(find_depth_configs(value))
    return found


def find_layer_prefix(names, config, prefix=None):
    """Return (prefix, depth, config dict that owns the depth) of the decoder layers.

    A prefix qualifies when it is numbered 0..depth-1 for a `num_hidden_layers` in the config.
    When several qualify (a vision tower can be as deep as the decoder) the one with the most
    weights wins; pass `prefix` to choose.
    """
    counts, indices = collections.Counter(), collections.defaultdict(set)
    for name in names:
        match = re.match(r"(\D*?)(\d+)\.", name)
        if match:
            counts[match.group(1)] += 1
            indices[match.group(1)].add(int(match.group(2)))
    candidates = []
    for owner in find_depth_configs(config):
        depth = owner["num_hidden_layers"]
        for candidate, found in indices.items():
            if found == set(range(depth)) and prefix in (None, candidate):
                candidates.append((counts[candidate], candidate, depth, owner))
    if not candidates:
        raise SystemExit(
            "No weight-name prefix is numbered 0..num_hidden_layers-1"
            + (f" and equals {prefix!r}." if prefix else ". Pass --prefix.")
            + f" Numbered prefixes: { {p: len(i) for p, i in indices.items()} }"
        )
    _, chosen, depth, owner = max(candidates, key=lambda item: item[0])
    return chosen, depth, owner


def make_subcheckpoint(model_dir, out_dir, first, last, prefix=None, drop="auto"):
    """Write layers [first, last) of the checkpoint to `out_dir`; return a summary string.

    `drop` is a regular expression for entries to leave out; "auto" leaves out the entries HF
    itself does not load (an auxiliary head can be larger than a decoder layer), None keeps all.
    """
    if drop == "auto":
        drop = ignored_by_hf(model_dir)
    entries = checkpoint_entries(model_dir)
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    prefix, depth, owner = find_layer_prefix(entries, config, prefix)
    if not 0 <= first < last <= depth:
        raise SystemExit(f"--layers {first}:{last} is outside 0:{depth}")
    layer_name = re.compile(re.escape(prefix) + r"(\d+)(\..+)")

    # output file -> [(shard, name in the shard, new name)]
    plan = collections.defaultdict(list)
    for name, shard in entries.items():
        if drop and re.search(drop, name):
            continue
        match = layer_name.fullmatch(name)
        if not match:
            plan["model-rest.safetensors"].append((shard, name, name))
        elif first <= int(match.group(1)) < last:
            index = int(match.group(1)) - first
            new_name = f"{prefix}{index}{match.group(2)}"
            plan[f"model-layer-{index:04d}.safetensors"].append((shard, name, new_name))

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    weight_map, total_bytes = {}, 0
    for filename in sorted(plan):
        by_shard = collections.defaultdict(list)
        for shard, name, new_name in plan[filename]:
            by_shard[shard].append((name, new_name))
        tensors = {}
        for shard, pairs in by_shard.items():
            with safetensors.safe_open(shard, framework="pt") as source:
                for name, new_name in pairs:
                    tensors[new_name] = source.get_tensor(name)
        safetensors.torch.save_file(tensors, os.path.join(out_dir, filename), {"format": "pt"})
        weight_map.update({name: filename for name in tensors})
        total_bytes += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        del tensors

    # Depth and every per-layer list (layer_types, ...) follow the range.
    for key, value in owner.items():
        if isinstance(value, list) and len(value) == depth:
            owner[key] = value[first:last]
    owner["num_hidden_layers"] = last - first
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(
        os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, handle)
    # Tokenizer and other small files, so the directory is usable on its own.
    for path in glob.glob(os.path.join(model_dir, "*")):
        name = os.path.basename(path)
        if (
            os.path.isfile(path)
            and not name.endswith(".safetensors")
            and name not in ("config.json", "model.safetensors.index.json")
            and os.path.getsize(path) <= MAX_COPIED_FILE_BYTES
        ):
            shutil.copy(path, out_dir)
    return (
        f"layers {first}:{last} of {depth} ({prefix}N.) -> {out_dir}: "
        f"{len(weight_map)} weights, {total_bytes / 2**30:.1f} GiB"
    )


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", help="local checkpoint directory with .safetensors files")
    parser.add_argument("out_dir", help="directory to write; an existing one is replaced")
    parser.add_argument("--layers", required=True, help="first:last, last not included")
    parser.add_argument("--prefix", help='weight-name prefix of the layers, e.g. "model.layers."')
    parser.add_argument(
        "--drop",
        default="auto",
        help=(
            "regular expression for checkpoint entries to leave out (default: the entries "
            "Hugging Face itself does not load)"
        ),
    )
    parser.add_argument("--keep-all", action="store_true", help="leave no entry out")
    args = parser.parse_args()
    first, last = (int(part) for part in args.layers.split(":"))
    drop = None if args.keep_all else args.drop
    print(make_subcheckpoint(args.model_dir, args.out_dir, first, last, args.prefix, drop))


if __name__ == "__main__":
    main()
