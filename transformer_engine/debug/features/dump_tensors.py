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
from transformer_engine.pytorch.constants import TE_DType_To_Torch
from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor


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
        self.rank = 0
        TensorLogger._initialized = True

    def initialize(self, root_log_dir: str):
        """Initialize the TensorLogger with the root directory for tensor dumps."""
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        self.root_dir = os.path.join(
            root_log_dir, "tensor_dumps", f"rank_{self.rank}"
        )
        os.makedirs(self.root_dir, exist_ok=True)

        debug_api.log_message(
            f"TensorLogger initialized. Saving tensors to: {self.root_dir}",
            log_level="info",
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize layer/tensor names for use in file paths."""
        for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
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
                "[TE DumpTensors] TensorLogger not initialized. "
                "Call initialize() first."
            )

        safe_layer_name = self._sanitize_name(layer_name)
        safe_tensor_name = self._sanitize_name(tensor_name)

        filename = f"{safe_layer_name}_{safe_tensor_name}_iter_{iteration:06d}.pt"
        filepath = os.path.join(self.root_dir, filename)

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
    dump_quantized_internals : bool, default = False
        If True, include extracted internal data from quantized tensors
        (raw data, scales, etc.) in the output dictionary.
        Useful for offline analysis. Output format may change between versions.
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
                          dump_quantized_internals: True
                          freq: 100
                        - tensor: weight
                          high_precision_tensor: True
                          quantized_tensor: False
                          freq: 500

    Output Structure
    ----------------
    Files are saved to: ``{nvdlfw_inspect_log_dir}/tensor_dumps/rank_{rank}/``

    Each tensor is saved as a dictionary in a single file:
        ``{layer}_{tensor}_iter_{iter:06d}.pt``

    Dictionary keys:
        - ``high_precision``: pre-quantization tensor (if high_precision_tensor=True)
        - ``quantized``: quantized tensor object (if quantized_tensor=True)
        - Additional internal components when dump_quantized_internals=True
          (raw data, scales, etc. - format may change between versions)
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
        tensor: torch.Tensor,
        rowwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        columnwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        quantizer: Optional[Quantizer] = None,
    ):  # pylint: disable=unused-argument
        """
        API call used to dump tensors to files.

        Supports dumping both high-precision tensors and quantized tensors based on config.
        """
        # Assert that rowwise and columnwise are the same (or one is None)
        assert rowwise_quantized_tensor is columnwise_quantized_tensor, (
            "[NVTORCH INSPECT ERROR] DumpTensors expects rowwise_quantized_tensor and "
            "columnwise_quantized_tensor to be the same object or both None."
        )

        quantized_tensor = rowwise_quantized_tensor

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
        if tensor_logger.root_dir is None:
            tensor_logger.initialize(get_logger().root_log_dir)

        # Build dictionary with all tensors to dump
        dump_dict: Dict[str, torch.Tensor] = {}

        if dump_hp and tensor is not None:
            dump_dict["high_precision"] = tensor

        if dump_quant and quantized_tensor is not None:
            dump_dict["quantized"] = quantized_tensor

            # Add internals for quantized tensors
            if config.get("dump_quantized_internals", False):
                internals = self._get_quantized_internals(quantized_tensor)
                dump_dict.update(internals)

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

    def _get_quantized_internals(
        self,
        quantized_tensor: QuantizedTensor,
    ) -> Dict[str, torch.Tensor]:
        """Get internal components of quantized tensors (raw data, scales, etc.)."""
        if isinstance(quantized_tensor, Float8Tensor):
            tensors = _get_extended_tensors_fp8(quantized_tensor)
        elif isinstance(quantized_tensor, Float8BlockwiseQTensor):
            tensors = _get_extended_tensors_fp8_blockwise(quantized_tensor)
        elif isinstance(quantized_tensor, MXFP8Tensor):
            tensors = _get_extended_tensors_mxfp8(quantized_tensor)
        elif isinstance(quantized_tensor, NVFP4Tensor):
            tensors = _get_extended_tensors_nvfp4(quantized_tensor)
        else:
            return {}

        # Filter out None values
        return {k: v for k, v in tensors.items() if v is not None}


def _get_extended_tensors_fp8(tensor: Float8Tensor) -> Dict[str, torch.Tensor]:
    """Get extended tensors for Float8Tensor: raw FP8 data, transpose, and scale."""
    torch_fp8_dtype = TE_DType_To_Torch[tensor._fp8_dtype]
    result = {
        "data": tensor._data.view(torch_fp8_dtype),
        "scale_inv": tensor._scale_inv,
    }
    if tensor._transpose is not None and not tensor._transpose_invalid:
        result["transpose"] = tensor._transpose.view(torch_fp8_dtype)
    return result


def _get_extended_tensors_fp8_blockwise(
    tensor: Float8BlockwiseQTensor,
) -> Dict[str, Optional[torch.Tensor]]:
    """Get extended tensors for Float8BlockwiseQTensor: raw FP8 data and block scales."""
    torch_fp8_dtype = TE_DType_To_Torch[tensor._fp8_dtype]
    result: Dict[str, Optional[torch.Tensor]] = {}

    if tensor._rowwise_data is not None:
        result["rowwise_data"] = tensor._rowwise_data.view(torch_fp8_dtype)
    if tensor._columnwise_data is not None:
        result["columnwise_data"] = tensor._columnwise_data.view(torch_fp8_dtype)

    # Block scaling factors (FP32)
    if tensor._rowwise_scale_inv is not None:
        result["rowwise_block_scale_inv"] = tensor._rowwise_scale_inv
    if tensor._columnwise_scale_inv is not None:
        result["columnwise_block_scale_inv"] = tensor._columnwise_scale_inv

    return result


def _get_extended_tensors_mxfp8(tensor: MXFP8Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """Get extended tensors for MXFP8Tensor: raw FP8 data and block scales (E8M0)."""
    torch_fp8_dtype = TE_DType_To_Torch[tensor._fp8_dtype]
    result: Dict[str, Optional[torch.Tensor]] = {}

    if tensor._rowwise_data is not None:
        result["rowwise_data"] = tensor._rowwise_data.view(torch_fp8_dtype)
    if tensor._columnwise_data is not None:
        result["columnwise_data"] = tensor._columnwise_data.view(torch_fp8_dtype)

    # Block scaling factors (E8M0 format)
    if tensor._rowwise_scale_inv is not None:
        result["rowwise_block_scale_inv"] = tensor._rowwise_scale_inv.view(torch.float8_e8m0fnu)
    if tensor._columnwise_scale_inv is not None:
        result["columnwise_block_scale_inv"] = tensor._columnwise_scale_inv.view(torch.float8_e8m0fnu)

    return result


def _get_extended_tensors_nvfp4(tensor: NVFP4Tensor) -> Dict[str, Optional[torch.Tensor]]:
    """Get extended tensors for NVFP4Tensor: raw packed FP4 data, block scales, and amax."""
    result: Dict[str, Optional[torch.Tensor]] = {}

    # Raw data (packed FP4, 2 values per byte)
    if tensor._rowwise_data is not None:
        result["rowwise_data"] = tensor._rowwise_data
    if tensor._columnwise_data is not None:
        result["columnwise_data"] = tensor._columnwise_data

    # Block scaling factors (E4M3 format)
    if tensor._rowwise_scale_inv is not None:
        result["rowwise_block_scale_inv"] = tensor._rowwise_scale_inv.view(torch.float8_e4m3fn)
    if tensor._columnwise_scale_inv is not None:
        result["columnwise_block_scale_inv"] = tensor._columnwise_scale_inv.view(torch.float8_e4m3fn)

    # Input absolute maximum value (used to compute tensor scale)
    if tensor._amax_rowwise is not None:
        result["amax_rowwise"] = tensor._amax_rowwise
    if tensor._amax_columnwise is not None:
        result["amax_columnwise"] = tensor._amax_columnwise

    return result
