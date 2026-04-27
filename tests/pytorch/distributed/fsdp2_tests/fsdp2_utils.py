# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared utility functions for FSDP2 distributed tests."""

import transformer_engine.common.recipe
from transformer_engine.pytorch import QuantizedTensor


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs


def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)
