# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DumpTensors Feature support for nvidia-dlframework-inspect."""

import os
from typing import Dict, Optional

import torch
import torch.distributed as dist

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.logging import get_logger
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.debug.features.utils import next_enabled_iter
from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer


class TensorLogger:
    """Logger for saving tensors to files. Each rank saves to its own directory."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TensorLogger._initialized:
            return
        self.root_dir = None
        TensorLogger._initialized = True

    def initialize(self, root_log_dir: str):
        """Initialize the TensorLogger with the root directory for tensor dumps."""
        self.root_dir = self._expected_root_dir(root_log_dir)
        os.makedirs(self.root_dir, exist_ok=True)

        debug_api.log_message(
            f"TensorLogger initialized. Saving tensors to: {self.root_dir}",
        )

    def _expected_root_dir(self, root_log_dir: str) -> str:
        """Return the rank-specific dump directory for the provided root log path."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        return os.path.join(root_log_dir, "tensor_dumps", f"rank_{rank}")

    def ensure_initialized(self, root_log_dir: str) -> None:
        """Reinitialize logger if debug session log directory changed."""
        expected_root_dir = self._expected_root_dir(root_log_dir)
        if self.root_dir != expected_root_dir or not os.path.isdir(expected_root_dir):
            self.initialize(root_log_dir)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize layer/tensor names for use in file paths."""
        for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " ", "."]:
            name = name.replace(char, "_")
        return name

    def save_tensor(
        self,
        tensor,
        layer_name: str,
        tensor_name: str,
        iteration: int,
    ):
        """Save a tensor (or dict of tensors) to a file."""
        if self.root_dir is None:
            raise RuntimeError(
                "[TE DumpTensors] TensorLogger not initialized. Call initialize() first."
            )

        safe_layer_name = self._sanitize_name(layer_name)
        safe_tensor_name = self._sanitize_name(tensor_name)
        iter_dir = os.path.join(self.root_dir, f"iter_{iteration:06d}")
        os.makedirs(iter_dir, exist_ok=True)
        filepath = os.path.join(iter_dir, f"{safe_layer_name}_{safe_tensor_name}.pt")

        if os.path.exists(filepath):
            debug_api.log_message(f"[TE DumpTensors] Overwriting existing dump file: {filepath}")
        torch.save(tensor, filepath)


def _get_tensor_logger() -> TensorLogger:
    """Get the singleton TensorLogger instance."""
    return TensorLogger()


@Registry.register_feature(namespace="transformer_engine")
class DumpTensors(TEConfigAPIMapper):
    """
    Dump tensors to files for debugging purposes.

    This feature saves tensors to disk using torch.save(). It supports dumping
    both high-precision tensors (before quantization) and quantized tensors.

    Each tensor is saved to a separate file with the iteration number, layer name,
    and tensor name in the filename. Files are organized per-rank in distributed settings.

    Parameters
    ----------
    high_precision_tensor : bool
        If True, dump the high-precision tensor (before quantization).
    quantized_tensor : bool
        If True, dump the quantized tensor (after quantization).
    tensors/tensors_struct : List[str]
        list of tensors to dump:
            - activation
            - gradient
            - weight
            - output
            - wgrad
            - dgrad
    freq : Optional[int], default = 1
        frequency of dumping tensors, tensors will be dumped every `freq` steps
    start_step : Optional[int], default = 0
        start step of dumping tensors
    end_step : Optional[int], default = -1
        end step of dumping tensors (-1 means no end)
    start_end_list : Optional[list([int, int])], default = None
        non-overlapping list of (start, end) pairs in incremental order.
        If not None, will ignore start_step and end_step

    Example
    -------
    .. code-block:: yaml

        dump_tensors_example:
            enabled: True
            layers:
                layer_name_regex_pattern: .*(fc1|self_attention).*
            transformer_engine:
                DumpTensors:
                    enabled: True
                    tensors_struct:
                        - tensor: activation
                          high_precision_tensor: True
                          quantized_tensor: True
                          freq: 100
                        - tensor: weight
                          high_precision_tensor: True
                          quantized_tensor: False
                          freq: 500

    Output Structure
    ----------------
    Files are saved to: ``{nvdlfw_inspect_log_dir}/tensor_dumps/rank_{rank}/iter_{iter:06d}/``

    Each tensor is saved as a dictionary in a single file:
        ``{layer}_{tensor}.pt``

    Dictionary keys:
        - ``high_precision``: pre-quantization tensor (if high_precision_tensor=True)
        - ``quantized``: quantized tensor object (if quantized_tensor=True)

    .. note::
        The ``quantized`` value is a pickled ``QuantizedTensor`` object. Loading it
        (with ``weights_only=False``) requires the same version of TransformerEngine
        to be installed.

    Loading and Analyzing Dumped Tensors
    ------------------------------------
    .. code-block:: python

        import torch

        # Load dumped tensor (requires the same TE version that produced the dump)
        data = torch.load("tensor_dumps/rank_0/iter_000100/fc1_activation.pt",
                          weights_only=False)

        hp = data["high_precision"]                 # original high-precision tensor
        qt = data["quantized"]                      # QuantizedTensor object
        dequant = qt.dequantize(dtype=hp.dtype)     # dequantize back to high precision

        mse = torch.mean((hp - dequant) ** 2).item()
        print(f"MSE between original and dequantized: {mse}")
    """

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run inspect_tensor() in the forward."""
        run_current, next_iter = next_enabled_iter(
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
            config.get("freq", 1),
            iteration,
        )
        return run_current, next_iter

    @api_method
    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
        tensor: Optional[torch.Tensor],
        rowwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        columnwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        quantizer: Optional[Quantizer] = None,
        tp_size: int = 1,
    ):  # pylint: disable=unused-argument
        """
        API call used to dump tensors to files.

        Supports dumping both high-precision tensors and quantized tensors based on config.
        """
        # We support one-sided availability (only rowwise or only columnwise tensor).
        # If both are present, require them to be the same object to avoid ambiguity.
        if (
            rowwise_quantized_tensor is not None
            and columnwise_quantized_tensor is not None
            and rowwise_quantized_tensor is not columnwise_quantized_tensor
        ):
            raise ValueError(
                "[NVTORCH INSPECT ERROR] DumpTensors expects rowwise_quantized_tensor and "
                "columnwise_quantized_tensor to be the same object when both are provided."
            )

        quantized_tensor = (
            rowwise_quantized_tensor
            if rowwise_quantized_tensor is not None
            else columnwise_quantized_tensor
        )

        dump_hp = config.get("high_precision_tensor", False)
        dump_quant = config.get("quantized_tensor", False)

        if not dump_hp and not dump_quant:
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: Neither high_precision_tensor nor "
                "quantized_tensor is enabled. Nothing to dump.",
                layer_name,
            )
            return

        tensor_logger = _get_tensor_logger()
        tensor_logger.ensure_initialized(get_logger().root_log_dir)

        # Build dictionary with all tensors to dump
        dump_dict: Dict[str, torch.Tensor] = {}

        if dump_hp and tensor is not None:
            dump_dict["high_precision"] = tensor.detach().clone()
        elif dump_hp and tensor is None:
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: high_precision_tensor is True but "
                f"no high-precision tensor available for {tensor_name}. Skipping.",
                layer_name,
            )

        if dump_quant and quantized_tensor is not None:
            dump_dict["quantized"] = quantized_tensor.detach().clone()
        elif dump_quant and quantized_tensor is None:
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: quantized_tensor is True but "
                f"no quantized tensor available for {tensor_name}. Skipping.",
                layer_name,
            )

        if dump_dict:
            tensor_logger.save_tensor(
                tensor=dump_dict,
                layer_name=layer_name,
                tensor_name=tensor_name,
                iteration=iteration,
            )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}, API=inspect_tensor: "
                f"Dumped {tensor_name} at iteration {iteration} (keys: {list(dump_dict.keys())})",
                layer_name,
            )
        else:
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: No tensors available to dump for "
                f"{tensor_name} at iteration {iteration}. No file written.",
                layer_name,
            )
