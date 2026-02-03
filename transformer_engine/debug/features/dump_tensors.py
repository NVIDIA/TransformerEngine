# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DumpTensors Feature support for nvidia-dlframework-inspect."""

from typing import Dict, Optional

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.logging import get_tensor_logger
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.debug.features.utils import next_enabled_iter
from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor


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
    extended_quantized_tensor_log : bool, default = False
        If True, dump additional files with raw data and scales for quantized tensors:
        - For Float8Tensor: raw_data (uint8), scale_inv (FP32)
        - For MXFP8Tensor: rowwise_raw_data, columnwise_raw_data (uint8),
          rowwise_scale_inv, columnwise_scale_inv (decoded to FP32)
        - For NVFP4Tensor: rowwise_raw_data, columnwise_raw_data (uint8),
          rowwise_scale_inv, columnwise_scale_inv (decoded to FP32),
          rowwise_amax, columnwise_amax (FP32)
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
                          extended_quantized_tensor_log: True
                          freq: 100
                        - tensor: weight
                          high_precision_tensor: True
                          quantized_tensor: False
                          freq: 500

    Output Structure
    ----------------
    Files are saved to: ``nvdlfw_inspect_tensor_dumps/rank_{rank}/``

    Basic files:
        - ``{layer}_{tensor}_iter_{iter}_high_precision.pt``
        - ``{layer}_{tensor}_iter_{iter}_quantized.pt``

    Extended files (when extended_quantized_tensor_log=True):
        - ``{layer}_{tensor}_iter_{iter}_raw_data.pt``
        - ``{layer}_{tensor}_iter_{iter}_scale_inv.pt``
        - (MXFP8/NVFP4) ``{layer}_{tensor}_iter_{iter}_rowwise_scale_inv.pt``
        - (NVFP4) ``{layer}_{tensor}_iter_{iter}_rowwise_amax.pt``
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

        tensor_logger = get_tensor_logger()

        # Dump high-precision tensor
        if dump_hp and tensor is not None:
            tensor_logger.save_tensor(
                tensor=tensor,
                layer_name=layer_name,
                tensor_name=tensor_name,
                iteration=iteration,
                suffix="_high_precision",
            )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}, API=inspect_tensor: "
                f"Dumped high-precision {tensor_name} at iteration {iteration}",
                layer_name,
            )

        # Dump quantized tensor
        if dump_quant and quantized_tensor is not None:
            tensor_logger.save_tensor(
                tensor=quantized_tensor,
                layer_name=layer_name,
                tensor_name=tensor_name,
                iteration=iteration,
                suffix="_quantized",
            )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}, API=inspect_tensor: "
                f"Dumped quantized {tensor_name} at iteration {iteration}",
                layer_name,
            )

            # Extended logging for quantized tensors
            if config.get("extended_quantized_tensor_log", False):
                self._dump_extended_quantized_info(
                    tensor_logger, quantized_tensor, layer_name, tensor_name, iteration
                )

        elif dump_quant and quantized_tensor is None:
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: quantized_tensor is True but "
                f"no quantized tensor available for {tensor_name}. Skipping.",
                layer_name,
            )

    def _dump_extended_quantized_info(
        self,
        tensor_logger,
        quantized_tensor: QuantizedTensor,
        layer_name: str,
        tensor_name: str,
        iteration: int,
    ):
        """Dump extended debug info for quantized tensors (raw data and scales)."""

        if isinstance(quantized_tensor, Float8Tensor):
            # Float8Tensor: raw_data (uint8), scale_inv (FP32)
            tensor_logger.save_tensor(
                tensor=quantized_tensor._data,
                layer_name=layer_name,
                tensor_name=tensor_name,
                iteration=iteration,
                suffix="_raw_data",
            )
            tensor_logger.save_tensor(
                tensor=quantized_tensor._scale_inv,
                layer_name=layer_name,
                tensor_name=tensor_name,
                iteration=iteration,
                suffix="_scale_inv",
            )

        elif isinstance(quantized_tensor, MXFP8Tensor):
            # MXFP8Tensor: raw data and scales (decoded from E8M0)
            if quantized_tensor._rowwise_data is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._rowwise_data,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_rowwise_raw_data",
                )
            if quantized_tensor._columnwise_data is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._columnwise_data,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_columnwise_raw_data",
                )
            # Decode E8M0 scales to FP32
            if quantized_tensor._rowwise_scale_inv is not None:
                decoded = torch.pow(
                    torch.tensor(2.0, device=quantized_tensor._rowwise_scale_inv.device),
                    quantized_tensor._rowwise_scale_inv.to(torch.float32) - 127.0,
                )
                tensor_logger.save_tensor(
                    tensor=decoded,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_rowwise_scale_inv",
                )
            if quantized_tensor._columnwise_scale_inv is not None:
                decoded = torch.pow(
                    torch.tensor(2.0, device=quantized_tensor._columnwise_scale_inv.device),
                    quantized_tensor._columnwise_scale_inv.to(torch.float32) - 127.0,
                )
                tensor_logger.save_tensor(
                    tensor=decoded,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_columnwise_scale_inv",
                )

        elif isinstance(quantized_tensor, NVFP4Tensor):
            # NVFP4Tensor: raw data, scales (decoded from E4M3), and amax
            if quantized_tensor._rowwise_data is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._rowwise_data,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_rowwise_raw_data",
                )
            if quantized_tensor._columnwise_data is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._columnwise_data,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_columnwise_raw_data",
                )
            # Decode E4M3 scales to FP32
            if quantized_tensor._rowwise_scale_inv is not None:
                decoded = quantized_tensor._rowwise_scale_inv.view(torch.float8_e4m3fn).to(
                    torch.float32
                )
                tensor_logger.save_tensor(
                    tensor=decoded,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_rowwise_scale_inv",
                )
            if quantized_tensor._columnwise_scale_inv is not None:
                decoded = quantized_tensor._columnwise_scale_inv.view(torch.float8_e4m3fn).to(
                    torch.float32
                )
                tensor_logger.save_tensor(
                    tensor=decoded,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_columnwise_scale_inv",
                )
            # Amax values (already FP32)
            if quantized_tensor._amax_rowwise is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._amax_rowwise,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_rowwise_amax",
                )
            if quantized_tensor._amax_columnwise is not None:
                tensor_logger.save_tensor(
                    tensor=quantized_tensor._amax_columnwise,
                    layer_name=layer_name,
                    tensor_name=tensor_name,
                    iteration=iteration,
                    suffix="_columnwise_amax",
                )
