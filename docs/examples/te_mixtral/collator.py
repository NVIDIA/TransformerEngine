# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Data collator for THD sequence packing (variable-length flash attention).

Adapted from bionemo-recipes. Only the subset needed by this tutorial is included.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


def _pt_flatten_collate(features: list[dict[str, list[int]]], return_position_ids: bool = False):
    """Flatten a list of tokenized samples into a single packed batch with cumulative sequence lengths."""
    is_labels_provided = "labels" in features[0]
    sample_lengths = [len(sample["input_ids"]) for sample in features]

    batch = {}
    batch["max_length_q"] = batch["max_length_k"] = max(sample_lengths)
    batch["input_ids"] = torch.tensor(
        [[token for sample in features for token in sample["input_ids"]]], dtype=torch.int64
    )
    if is_labels_provided:
        batch["labels"] = torch.tensor(
            [[label for sample in features for label in sample["labels"]]], dtype=torch.int64
        )
    cu_seq_lens = torch.zeros(len(features) + 1, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(torch.tensor(sample_lengths), dim=0, dtype=torch.int32)
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.tensor(
            [[v for sample in features for v in sample["attention_mask"]]], dtype=torch.int64
        )
    if return_position_ids:
        batch["position_ids"] = torch.hstack(
            [torch.arange(sample_len, dtype=torch.int64) for sample_len in sample_lengths]
        ).unsqueeze(0)

    return batch


def _pt_pad_to_multiple_of(batch: dict[str, Any], pad_to_multiple_of: int, token_pad: int, label_pad: int):
    """Pad a batch to a multiple of ``pad_to_multiple_of`` by appending a mock sequence."""
    remainder = -batch["input_ids"].numel() % pad_to_multiple_of
    if remainder == 0:
        return batch

    batch["input_ids"] = torch.cat(
        [batch["input_ids"], torch.full((1, remainder), token_pad, dtype=batch["input_ids"].dtype)], dim=1
    )
    if "labels" in batch:
        batch["labels"] = torch.cat(
            [batch["labels"], torch.full((1, remainder), label_pad, dtype=batch["labels"].dtype)], dim=1
        )
    if "cu_seq_lens_q" in batch:
        batch["cu_seq_lens_q"] = torch.cat(
            [
                batch["cu_seq_lens_q"],
                torch.tensor([batch["cu_seq_lens_q"][-1] + remainder], dtype=batch["cu_seq_lens_q"].dtype),
            ],
            dim=0,
        )
        batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]
    if "max_length_q" in batch:
        batch["max_length_q"] = max(batch["max_length_q"], remainder)
        batch["max_length_k"] = batch["max_length_q"]
    if "attention_mask" in batch:
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.zeros((1, remainder), dtype=batch["attention_mask"].dtype)], dim=1
        )
    if "position_ids" in batch:
        batch["position_ids"] = torch.cat(
            [batch["position_ids"], torch.arange(remainder, dtype=batch["position_ids"].dtype).unsqueeze(0)], dim=1
        )

    return batch


@dataclass
class DataCollatorWithFlattening:
    """Data collator that flattens variable-length sequences into a single packed tensor for flash attention.

    Wraps a ``DataCollatorForLanguageModeling`` and produces THD-format batches with
    ``cu_seq_lens_q`` / ``cu_seq_lens_k`` metadata for TE's fused attention kernels.

    Args:
        collator: The base collator for MLM/CLM masking.
        pad_to_multiple_of: If set, pads the total token count to be divisible by this number.
        separator_id: Label value inserted at sequence boundaries (typically -100 for causal LM).
    """

    collator: DataCollatorForLanguageModeling
    pad_to_multiple_of: int | None = None
    separator_id: int | None = None

    def __call__(self, features, return_tensors=None):
        """Pack features into a single THD batch with flash-attention metadata."""
        if return_tensors is not None and return_tensors != "pt":
            raise NotImplementedError(f"Only return_tensors='pt' is supported, got '{return_tensors}'")

        bshd_batch = self.collator(features, return_tensors=return_tensors)
        packed_batch = _pt_flatten_collate(features)

        masked_input_ids = bshd_batch["input_ids"][bshd_batch["attention_mask"].bool()].unsqueeze(0)
        masked_labels = bshd_batch["labels"][bshd_batch["attention_mask"].bool()].unsqueeze(0)

        if self.separator_id is not None:
            masked_labels[:, packed_batch["cu_seq_lens_q"][1:-1]] = self.separator_id

        packed_batch["input_ids"] = masked_input_ids
        packed_batch["labels"] = masked_labels

        if self.pad_to_multiple_of is not None:
            pad_token_id = self.collator.tokenizer.pad_token_id
            if not isinstance(pad_token_id, int):
                logger.warning(f"tokenizer.pad_token_id is not an integer, using 1 instead: {pad_token_id}")
                pad_token_id = 1
            packed_batch = _pt_pad_to_multiple_of(
                packed_batch,
                self.pad_to_multiple_of,
                token_pad=pad_token_id,
                label_pad=-100,
            )

        return packed_batch
