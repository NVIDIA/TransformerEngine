#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Static check of a Transformer Engine recipe generated from a Hugging Face model.

The recipe files are parsed, not imported, so neither Transformer Engine nor a GPU is
needed. The check answers three questions:

* Does the recipe use a fused TE block, rather than only swapping single layers?
* Is there a weight mapping from the Hugging Face checkpoint?
* Does that mapping write the fused parameters the chosen blocks own?

It also warns about common mistakes, says whether a ``te.autocast`` precision hook is
present, and prints the TE constructor arguments it found so they can be compared with
the output of ``inspect_hf_config.py``.

This is a structural check. It cannot prove that the mapping is numerically right;
that needs the parity check described in ``references/validation.md``.

Usage:
    python check_recipe.py te_llama.py
    python check_recipe.py te_mixtral.py hf_to_te_weights.py

Exit code 0 means no failures (warnings do not change it), 1 means at least one failure.
"""

import argparse
import ast
import json
import re
import sys
import warnings

FUSED_BLOCKS = (
    "TransformerLayer",
    "MultiheadAttention",
    "LayerNormLinear",
    "LayerNormMLP",
    "GroupedLinear",
)
TRACKED_CALLS = FUSED_BLOCKS + ("RotaryPositionEmbedding",)
PRECISION_SCOPES = ("quantized_model_init", "autocast", "fp8_model_init", "fp8_autocast")
DEPRECATED = {"fp8_autocast": "autocast", "fp8_model_init": "quantized_model_init"}
RECIPE_CLASSES = (
    "DelayedScaling",
    "Float8CurrentScaling",
    "Float8BlockScaling",
    "MXFP8BlockScaling",
    "NVFP4BlockScaling",
)
MAPPING_NAME = re.compile(r"replace_params|hf_to_te|(load|copy|map)_\w*weights")


def dotted_name(node):
    """Return "a.b.c" for a Name/Attribute chain and "" for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


class RecipeVisitor(ast.NodeVisitor):
    """Collect the facts the checks need from one or more parsed recipe files."""

    def __init__(self):
        self.imports_te = False
        self.aliases = {}  # local name -> original name, for transformer_engine imports
        self.imported_from = {}  # local name -> module it was imported from
        self.blocks = set()
        self.constructor_arguments = {}
        self.functions = set()
        self.calls = set()
        self.strings = []
        self.prose = []
        self.recipe_classes = set()
        self.strict_false = False
        self.coverage_evidence = False
        self.high_precision_scope = False
        self.rope_base_evidence = False
        self.deprecated = set()
        self._class_stack = []

    def resolve(self, node):
        """Return the final component of a call or base name, following TE import aliases."""
        name = dotted_name(node)
        if not name:
            return ""
        if "." in name:
            return name.rsplit(".", 1)[-1]
        return self.aliases.get(name, name)

    def visit_Import(self, node):  # pylint: disable=invalid-name
        """Record `import transformer_engine...`."""
        for alias in node.names:
            if alias.name.split(".")[0] == "transformer_engine":
                self.imports_te = True

    def visit_ImportFrom(self, node):  # pylint: disable=invalid-name
        """Record `from ... import ...`, keeping aliases of TE names."""
        module = node.module or ""
        from_te = module.split(".")[0] == "transformer_engine"
        self.imports_te = self.imports_te or from_te
        for alias in node.names:
            local = alias.asname or alias.name
            self.imported_from[local] = module
            if from_te:
                self.aliases[local] = alias.name

    def visit_ClassDef(self, node):  # pylint: disable=invalid-name
        """Record classes that subclass a fused TE block."""
        te_bases = [base for base in map(self.resolve, node.bases) if base in FUSED_BLOCKS]
        self.blocks.update(te_bases)
        self._class_stack.append(te_bases[0] if te_bases else None)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):  # pylint: disable=invalid-name
        """Record function names."""
        self.functions.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Expr(self, node):  # pylint: disable=invalid-name
        """Keep docstrings and bare string statements apart; they are prose, not key names."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self.prose.append(node.value.value)
            return
        self.generic_visit(node)

    def visit_Constant(self, node):  # pylint: disable=invalid-name
        """Record string literals (parameter key fragments live here)."""
        if isinstance(node.value, str):
            self.strings.append(node.value)

    def visit_Name(self, node):  # pylint: disable=invalid-name
        """Notice identifiers that show a coverage check or a deprecated API."""
        self._note_identifier(node.id)

    def visit_Attribute(self, node):  # pylint: disable=invalid-name
        """Notice attribute names that show a coverage check or a deprecated API."""
        self._note_identifier(node.attr)
        self.generic_visit(node)

    def visit_Assign(self, node):  # pylint: disable=invalid-name
        """Notice RoPE frequencies being overwritten with the HF values."""
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "inv_freq":
                self.rope_base_evidence = True
        self.generic_visit(node)

    def visit_Call(self, node):  # pylint: disable=invalid-name
        """Record TE constructor calls, their keyword arguments, and telltale calls."""
        name = self.resolve(node.func)
        keywords = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
        self.calls.add(name)

        target = None
        if name in TRACKED_CALLS:
            target = name
        elif self._is_super_init(node) and self._class_stack and self._class_stack[-1]:
            target = self._class_stack[-1]
        if target is not None:
            if target in FUSED_BLOCKS:
                self.blocks.add(target)
            self.constructor_arguments.setdefault(target, {}).update(keywords)

        if name == "load_state_dict" and "strict" in keywords:
            if keywords["strict"] == "False":
                self.strict_false = True
            else:
                self.coverage_evidence = True
        if name in PRECISION_SCOPES and keywords.get("enabled") == "False":
            self.high_precision_scope = True
        if name == "RotaryPositionEmbedding" and "rotary_base" in keywords:
            self.rope_base_evidence = True
        self.generic_visit(node)

    def _note_identifier(self, identifier):
        if identifier in DEPRECATED:
            self.deprecated.add(identifier)
        if identifier in RECIPE_CLASSES:
            self.recipe_classes.add(identifier)
        if identifier in ("isnan", "missing_keys", "unexpected_keys"):
            self.coverage_evidence = True

    @staticmethod
    def _is_super_init(node):
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and dotted_name(func.value.func) == "super"
        )


def run_checks(visitor):
    """Turn the collected facts into a list of {id, status, message} results."""
    results = []

    def add(check_id, status, message):
        results.append({"id": check_id, "status": status, "message": message})

    strings = " ".join(visitor.strings).lower()
    blocks = visitor.blocks

    if visitor.imports_te:
        add("imports-te", "pass", "transformer_engine is imported.")
    else:
        add("imports-te", "fail", "No import of transformer_engine found.")

    if blocks:
        add("fused-block", "pass", f"Fused TE blocks used: {', '.join(sorted(blocks))}.")
    else:
        add(
            "fused-block",
            "fail",
            "No fused TE block (TransformerLayer, MultiheadAttention, LayerNormLinear, "
            "LayerNormMLP, GroupedLinear). Replacing single layers with te.Linear fuses nothing.",
        )

    mapping_defs = sorted(name for name in visitor.functions if MAPPING_NAME.search(name))
    mapping_imports = sorted(
        f"{name} (from {module})"
        for name, module in visitor.imported_from.items()
        if MAPPING_NAME.search(name) and name not in visitor.functions
    )
    if mapping_defs:
        add("weight-mapping", "pass", f"Weight mapping defined: {', '.join(mapping_defs)}.")
    elif mapping_imports:
        add(
            "weight-mapping",
            "fail",
            f"The weight mapping is imported, not defined here: {', '.join(mapping_imports)}. "
            "Pass the file that defines it as well.",
        )
    else:
        add(
            "weight-mapping",
            "fail",
            "No weight-mapping function found (expected a function such as replace_params).",
        )

    missing = []
    if blocks & {"TransformerLayer", "MultiheadAttention"} and "layernorm_qkv" not in strings:
        missing.append("layernorm_qkv (fused input norm + QKV)")
    if blocks & {"TransformerLayer", "LayerNormMLP"}:
        if "layernorm_mlp" not in strings and "fc1_weight" not in strings:
            missing.append("layernorm_mlp / fc1_weight (fused norm + MLP)")
    if "LayerNormLinear" in blocks and "layer_norm_weight" not in strings:
        missing.append("layer_norm_weight (fused norm + projection)")
    if "GroupedLinear" in blocks and "expert" not in strings:
        missing.append("expert weights (GroupedLinear)")
    if not blocks:
        add("fused-parameters-mapped", "fail", "No fused block, so no fused parameter is mapped.")
    elif missing:
        add(
            "fused-parameters-mapped",
            "fail",
            "No parameter key in the recipe refers to: " + "; ".join(missing) + ".",
        )
    else:
        add("fused-parameters-mapped", "pass", "Keys for every fused block appear in the recipe.")

    if "RotaryPositionEmbedding" in visitor.calls and not visitor.rope_base_evidence:
        add(
            "rope-base",
            "warn",
            "RotaryPositionEmbedding is built without rotary_base and inv_freq is not overwritten. "
            "TE defaults to base 10000; pass rotary_base=config.rope_theta if the config has one.",
        )
    if visitor.strict_false and not visitor.coverage_evidence:
        add(
            "coverage",
            "warn",
            "load_state_dict(strict=False) is used and no coverage check was found. A wrong key "
            "name would leave a layer at random initialization without an error.",
        )
    is_moe = "GroupedLinear" in blocks or any(name.startswith("moe_") for name in visitor.calls)
    if is_moe and not visitor.high_precision_scope:
        add(
            "router-precision",
            "warn",
            "MoE recipe with no quantized_model_init(enabled=False) / autocast(enabled=False) "
            "scope. Keep the router gate out of quantization.",
        )
    for old in sorted(visitor.deprecated):
        add("deprecated-api", "warn", f"{old} is deprecated; use {DEPRECATED[old]}.")

    hook = []
    if visitor.calls & {"autocast", "fp8_autocast"}:
        hook.append("autocast call")
    if visitor.recipe_classes:
        hook.append("recipe " + ", ".join(sorted(visitor.recipe_classes)))
    if any("autocast(" in text for text in visitor.prose):
        hook.append("autocast example in a docstring")
    if hook:
        add("precision-hook", "info", "Precision hook found: " + "; ".join(hook) + ".")
    else:
        add(
            "precision-hook",
            "info",
            "No precision hook (no autocast call, TE recipe class or docstring example). "
            "Expected for a plain conversion; a fine-tuning recipe needs one.",
        )

    fuse_values = {
        args.get("fuse_qkv_params")
        for block, args in visitor.constructor_arguments.items()
        if block in ("TransformerLayer", "MultiheadAttention")
    }
    if "False" in fuse_values:
        add(
            "fuse-qkv",
            "info",
            "fuse_qkv_params=False: norm and QKV are still fused, with three separate weights.",
        )
    return results


def check_files(paths):
    """Parse the given files and return the full report as a dict."""
    visitor = RecipeVisitor()
    parse_errors = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle, warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                visitor.visit(ast.parse(handle.read(), filename=path))
        except (OSError, SyntaxError) as err:
            parse_errors.append({"id": "parse", "status": "fail", "message": f"{path}: {err}"})

    results = parse_errors + run_checks(visitor)
    failed = any(result["status"] == "fail" for result in results)
    return {
        "status": "fail" if failed else "pass",
        "files": list(paths),
        "fused_blocks": sorted(visitor.blocks),
        "te_constructor_arguments": visitor.constructor_arguments,
        "checks": results,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="recipe file(s), including the weight mapping")
    args = parser.parse_args()
    report = check_files(args.files)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
