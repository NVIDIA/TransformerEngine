# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FakeQuant Feature support for nvidia-dlframework-inspect"""

import math
from typing import Optional, Tuple

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.utils import append_parent_docstring


import transformer_engine_torch as tex
from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.constants import MXFP8_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.tensor import Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.quantization import _default_sf_compute


# Block length used by Float8BlockQuantizer (hard-coded to 128 in TE).
_FP8_BLOCKWISE_BLOCK_LEN = 128


def _build_per_tensor_fp8_quantizer(tensor: torch.Tensor, fp8_dtype: tex.DType) -> Quantizer:
    """Per-tensor current scaling FP8 quantizer (E4M3 / E5M2)."""
    fp8_max = (
        Format.E4M3.value.max_fwd
        if fp8_dtype == tex.DType.kFloat8E4M3
        else Format.E5M2.value.max_fwd
    )
    amax = tensor.abs().max().float()
    scale = _default_sf_compute(amax, torch.ones(1, device=tensor.device), fp8_max, 0)
    return Float8Quantizer(scale, amax, fp8_dtype)


def _build_mxfp8_quantizer(_tensor: torch.Tensor, fp8_dtype: tex.DType) -> Quantizer:
    """MXFP8 (1x32 block scaling) quantizer."""
    return MXFP8Quantizer(fp8_dtype=fp8_dtype)


def _build_fp8_blockwise_quantizer(
    _tensor: torch.Tensor, fp8_dtype: tex.DType, *, block_scaling_dim: int
) -> Quantizer:
    """Float8 blockwise quantizer (128x128 2D tiles or 1x128 1D rows)."""
    return Float8BlockQuantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
        block_scaling_dim=block_scaling_dim,
    )


def _check_blockwise_shape(tensor: torch.Tensor, block_size: int, fp8_format: str) -> None:
    """Validate that tensor shape is compatible with a blockwise quantizer.

    For blockwise formats, the last dim must be a multiple of block_size (true
    hard requirement of the quantizer kernel). The leading dim is NOT required
    to be a multiple of block_size: when it is not, ``_pad_for_blockwise()``
    pads it transparently and ``fake_quantize`` slices the padded tail off
    after dequantize. This matches the behavior needed for MoE GroupedLinear
    where the per-expert M-dim is routing-dependent and rarely 128-aligned.
    """
    if tensor.ndim < 2:
        raise ValueError(
            f"[NVTORCH INSPECT ERROR] FakeQuant quant_format={fp8_format} requires a tensor with "
            f"ndim >= 2, got shape {tuple(tensor.shape)}."
        )
    last = tensor.shape[-1]
    if last % block_size != 0:
        raise ValueError(
            f"[NVTORCH INSPECT ERROR] FakeQuant quant_format={fp8_format} requires "
            f"tensor.shape[-1] ({last}) to be divisible by block_size={block_size}. "
            f"Got shape {tuple(tensor.shape)}."
        )


def _pad_for_blockwise(tensor: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Optional[int]]:
    """Pad leading dim up to a multiple of ``block_size``.

    Returns ``(padded_tensor, original_leading)``. ``original_leading`` is
    ``None`` when no padding was needed, otherwise it is the original size of
    the flattened leading dim, used to slice the dequantized output back to
    the caller's shape.

    Padding is done with zeros along a flattened 2D view; rows containing pad
    zeros end up forming the partial last block, which the blockwise quantizer
    handles cleanly (a zero block has scale=1 and contributes no error after
    we discard the pad).
    """
    if tensor.ndim < 2:
        return tensor, None
    last = tensor.shape[-1]
    leading = math.prod(tensor.shape[:-1])
    if leading % block_size == 0:
        return tensor, None

    pad_rows = block_size - (leading % block_size)
    flat = tensor.reshape(leading, last)
    pad = flat.new_zeros((pad_rows, last))
    padded = torch.cat([flat, pad], dim=0)
    return padded, leading


# Format string -> (factory(tensor, fp8_dtype, **factory_kwargs) -> Quantizer,
#                   fp8_dtype: tex.DType,
#                   factory_kwargs: dict,
#                   block_size: Optional[int] for shape validation, None for per-tensor formats)
_FORMAT_DISPATCH = {
    # Per-tensor current scaling FP8
    "FP8E4M3": (_build_per_tensor_fp8_quantizer, tex.DType.kFloat8E4M3, {}, None),
    "FP8E5M2": (_build_per_tensor_fp8_quantizer, tex.DType.kFloat8E5M2, {}, None),
    # MXFP8 (1x32 block scaling)
    "MXFP8E4M3": (_build_mxfp8_quantizer, tex.DType.kFloat8E4M3, {}, MXFP8_BLOCK_SCALING_SIZE),
    "MXFP8E5M2": (_build_mxfp8_quantizer, tex.DType.kFloat8E5M2, {}, MXFP8_BLOCK_SCALING_SIZE),
    # Float8 blockwise: 2D 128x128 tiles
    "FP8_BLOCKWISE_E4M3": (
        _build_fp8_blockwise_quantizer,
        tex.DType.kFloat8E4M3,
        {"block_scaling_dim": 2},
        _FP8_BLOCKWISE_BLOCK_LEN,
    ),
    "FP8_BLOCKWISE_E5M2": (
        _build_fp8_blockwise_quantizer,
        tex.DType.kFloat8E5M2,
        {"block_scaling_dim": 2},
        _FP8_BLOCKWISE_BLOCK_LEN,
    ),
    # Float8 blockwise: 1D 1x128 rows
    "FP8_BLOCKWISE_1D_E4M3": (
        _build_fp8_blockwise_quantizer,
        tex.DType.kFloat8E4M3,
        {"block_scaling_dim": 1},
        _FP8_BLOCKWISE_BLOCK_LEN,
    ),
    "FP8_BLOCKWISE_1D_E5M2": (
        _build_fp8_blockwise_quantizer,
        tex.DType.kFloat8E5M2,
        {"block_scaling_dim": 1},
        _FP8_BLOCKWISE_BLOCK_LEN,
    ),
}


def fake_quantize(tensor: torch.Tensor, fp8_format: str, out=None):
    """Quantize ``tensor`` to the requested FP8 format and immediately dequantize it.

    Supports per-tensor FP8 (FP8E4M3 / FP8E5M2), MXFP8 (MXFP8E4M3 / MXFP8E5M2) and
    Float8 blockwise scaling (FP8_BLOCKWISE_{,1D_}E4M3 / FP8_BLOCKWISE_{,1D_}E5M2).

    For block-scaled formats, if ``prod(shape[:-1])`` is not a multiple of the
    block size the leading dim is zero-padded internally and the dequantized
    output is sliced back to the original shape. This makes the feature usable
    with MoE GroupedLinear where the per-expert M-dim is dynamic.
    """

    assert tensor.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), "[NVTORCH INSPECT ERROR] Unsupported tensor type."
    assert tensor.is_cuda, "[NVTORCH INSPECT ERROR] Must be a GPU tensor."

    if fp8_format not in _FORMAT_DISPATCH:
        raise ValueError(
            "[NVTORCH INSPECT ERROR] Unsupported FakeQuant quant_format "
            f"{fp8_format!r}. Supported formats: {sorted(_FORMAT_DISPATCH)}."
        )

    factory, fp8_dtype, factory_kwargs, block_size = _FORMAT_DISPATCH[fp8_format]

    original_shape = tensor.shape
    qinput = tensor
    original_leading: Optional[int] = None
    if block_size is not None:
        _check_blockwise_shape(tensor, block_size, fp8_format)
        qinput, original_leading = _pad_for_blockwise(tensor, block_size)

    quantizer = factory(qinput, fp8_dtype, **factory_kwargs)
    dequantized = quantizer(qinput).dequantize()

    if original_leading is not None:
        # Slice off the padded rows and restore the caller's logical shape.
        dequantized = dequantized[:original_leading].reshape(original_shape)

    if out is not None:
        # Called from DebugQuantizer.update_quantized() (weight workspace
        # cache write-back). `out` may be a QuantizedTensor (e.g.
        # Float8BlockwiseQTensor allocated by parent_quantizer.make_empty)
        # or a plain torch.Tensor. Use the QuantizedTensor's own quantize_()
        # path when available so the fake-quanted bf16 result is re-encoded
        # into the cache's native format (this is the correct semantics for
        # same-recipe fake-quant: the second cast is near-identity, and for
        # cross-recipe fake-quant it captures the additional cast error).
        if hasattr(out, "quantize_"):
            out.quantize_(dequantized, noop_flag=None)
        else:
            out.copy_(dequantized)
        return None
    return dequantized


@Registry.register_feature(namespace="transformer_engine")
@append_parent_docstring(parent=TEConfigAPIMapper)
class FakeQuant(TEConfigAPIMapper):
    """

    Disables FP8 GEMM. Fake quantizes chosen tensors to FP8 - using per-tensor scaling factor, not delayed scaling - and runs high-precision GEMM.

    .. figure:: ./img/fake_quant.svg
        :align: center

        Fig 1: Comparison of FP8 FPROP GEMM with the same GEMM in BF16 with fake quantization of activation tensor. Green tensors have the same values, but different dtypes.



    Parameters
    ----------

    gemms/gemms_struct: List[str]
        list of gemms to fake quantize

            - fprop
            - dgrad
            - wgrad
    tensors/tensors_struct: List[str]
        list of tensors to fake quantize

            - activation
            - gradient
            - weight
            - output
            - wgrad
            - dgrad

    quant_format: str
        specifies the FP8 format / scaling strategy to emulate:

            Per-tensor current scaling FP8:

                - FP8E4M3
                - FP8E5M2

            MXFP8 (1x32 block scaling):

                - MXFP8E4M3
                - MXFP8E5M2

            Float8 blockwise scaling - 128x128 2D tiles (default `Float8BlockScaling`):

                - FP8_BLOCKWISE_E4M3
                - FP8_BLOCKWISE_E5M2

            Float8 blockwise scaling - 1x128 1D rows:

                - FP8_BLOCKWISE_1D_E4M3
                - FP8_BLOCKWISE_1D_E5M2

        Shape requirements:

        - MXFP8*:           ``shape[-1]`` and ``prod(shape[:-1])`` must both
                            be divisible by 32.
        - FP8_BLOCKWISE_*:  ``shape[-1]`` must be divisible by 128.
                            ``prod(shape[:-1])`` does NOT need to be 128-aligned;
                            FakeQuant pads it internally and slices the
                            dequantized output back to the caller's shape.
                            This makes the feature work with MoE GroupedLinear
                            where per-expert token counts are routing-dependent.

    Example
    -------
    .. code-block:: yaml

        example_fake_quant_fp8:
            enabled: True
            layers:
                layer_types: [transformer_layer.layernorm_mlp.fc1]
            transformer_engine:
                FakeQuant:
                    enabled: True
                    quant_format: FP8_BLOCKWISE_E4M3
                    gemms_struct:
                    - gemm: fprop
                        tensors: [activation, weight]
                    - gemm: dgrad
                        tensors: [gradient]
    """

    def _supported_formats(self):
        """Returns formats that one can fake quantize tensor to."""
        return list(_FORMAT_DISPATCH)

    @api_method
    def fp8_gemm_enabled(
        self, config, layer_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call responsible for selecting between high-precision and FP8 GEMM execution."""
        return False, None

    @api_method
    def modify_tensor_enabled(
        self, config, layer_name: str, tensor_name: str, gemm: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run process_tensor() in the forward."""
        return True, iteration + 1

    @api_method
    def modify_tensor(
        self,
        config,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        default_quantizer: Quantizer,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):  # pylint: disable=unused-argument
        """API call used to process the tensor."""

        for key in config.keys():
            if key not in ["gemm", "tensor", "quant_format"]:
                raise ValueError(f'[NVTORCH INSPECT ERROR] Unexpected key in config: "{key}".')

        if "quant_format" not in config:
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format missing for Tensor: {tensor_name} in the config yaml for"
                " FakeQuant feature which is a required field"
            )
        if config["quant_format"] not in self._supported_formats():
            raise ValueError(
                f"[NVTORCH INSPECT ERROR] Feature={self.__class__.__name__}, API=process_tensor:"
                f" quant_format: {config['quant_format']} for Tensor: {tensor_name} in the config"
                " yaml for FakeQuant feature is not supported"
            )
        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=process_tensor: {gemm}, {tensor_name}",
            layer_name,
            extra_cachable_args=(gemm, tensor_name),
        )

        quant_format = config["quant_format"]
        q_tensor = fake_quantize(tensor, quant_format, out=out)
        if dtype is not None:
            q_tensor = q_tensor.to(dtype)
        return q_tensor
