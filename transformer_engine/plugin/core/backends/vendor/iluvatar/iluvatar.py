# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch

from ....ops import TEFLBackendBase, FP8TensorMeta


def _load_iluvatar_libs():
    import ctypes
    import os
    import subprocess
    from pathlib import Path
    import importlib.util
    import sysconfig
    import platform
    import glob as glob_module

    def get_ext():
        system = platform.system()
        return ".so" if system == "Linux" else ".dylib" if system == "Darwin" else ".dll"

    ext = get_ext()

    def try_load_lib(name, search_patterns):
        for env_var in [f"{name.upper()}_HOME", f"{name.upper()}_PATH"]:
            path = os.environ.get(env_var)
            if path:
                libs = glob_module.glob(f"{path}/**/lib{name}{ext}*", recursive=True)
                if libs:
                    libs.sort(reverse=True, key=os.path.basename)
                    try:
                        return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)
                    except:
                        pass

        cuda_home = os.environ.get("IX_HOME") or os.environ.get("IX_PATH") or "/usr/local/corex"
        for pattern in search_patterns:
            libs = glob_module.glob(f"{cuda_home}/**/{pattern}", recursive=True)
            if libs:
                libs.sort(reverse=True, key=os.path.basename)
                try:
                    return ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)
                except:
                    pass

        try:
            result = subprocess.check_output(f"ldconfig -p | grep 'lib{name}{ext}'", shell=True)
            for line in result.decode().split('\n'):
                if f"lib{name}" in line and "=>" in line:
                    so_path = line.split(">")[1].strip()
                    if so_path:
                        return ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        except:
            pass

        try:
            return ctypes.CDLL(f"lib{name}{ext}", mode=ctypes.RTLD_GLOBAL)
        except:
            return None

    try:
        try_load_lib("cudnn", [f"libcudnn{ext}*"])
        try_load_lib("nvrtc", [f"libnvrtc{ext}*"])
        try_load_lib("curand", [f"libcurand{ext}*"])

        te_path = Path(importlib.util.find_spec("transformer_engine_iluvatar").origin).parent.parent
        for search_dir in [te_path, te_path / "transformer_engine_iluvatar/libs"]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"libixte_common{ext}*"))
                if matches:
                    ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
                    return True
        return False
    except Exception as e:
        print(f"[ILUVATAR] Failed to load ILUVATAR libs: {e}")
        return False

_iluvatar_libs_loaded = False

def _ensure_iluvatar_libs():
    global _iluvatar_libs_loaded
    if not _iluvatar_libs_loaded:
        _iluvatar_libs_loaded = _load_iluvatar_libs()
    return _iluvatar_libs_loaded

def _check_iluvatar_available() -> bool:
    if not torch.cuda.is_available():
        return False
    import os
    try:
        if not _ensure_iluvatar_libs():
           return False
        import transformer_engine_iluvatar
        return True
    except (ImportError, OSError) as e:
        print(f"[ILUVATAR] Import failed: {e}")
        return False

def _get_tex():
    import transformer_engine_iluvatar.pytorch.ixte_torch
    return transformer_engine_iluvatar.pytorch.ixte_torch

def _torch_dtype_to_te_dtype(torch_dtype, tex_module):
    if torch_dtype is None:
        return None

    NativeDType = tex_module.DType
    if type(torch_dtype).__name__ == 'DType' and type(torch_dtype).__module__ == 'transformer_engine_iluvatar.pytorch.ixte_torch':
        return torch_dtype

    if hasattr(torch_dtype, 'name') and hasattr(torch_dtype, 'value'):
        from transformer_engine.plugin.core.ops import DType as PyDType
        if isinstance(torch_dtype, PyDType):
            dtype_name = torch_dtype.name
            if hasattr(NativeDType, dtype_name):
                return getattr(NativeDType, dtype_name)

    dtype_map = {
        torch.uint8: NativeDType.kByte,
        torch.float8_e4m3fn: NativeDType.kFloat8E4M3,
        torch.float8_e5m2: NativeDType.kFloat8E5M2,
        torch.int32: NativeDType.kInt32,
        torch.float32: NativeDType.kFloat32,
        torch.half: NativeDType.kFloat16,
        torch.bfloat16: NativeDType.kBFloat16,
    }

    return dtype_map.get(torch_dtype, torch_dtype)

def _convert_dtype_params(func):
    import functools
    import inspect
    import os

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        dtype_params = ['otype', 'output_dtype', 'bias_type']

        from transformer_engine.plugin.core.ops import DType as PyDType

        def needs_conversion(val):
            return isinstance(val, torch.dtype) or isinstance(val, PyDType)

        for param_name in dtype_params:
            if param_name in kwargs:
                value = kwargs[param_name]
                if needs_conversion(value):
                    converted = self._to_te_dtype(value)
                    kwargs[param_name] = converted

        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())[1:]

        args_list = list(args)
        for i, (param_name, arg_value) in enumerate(zip(param_names, args_list)):
            if param_name in dtype_params and needs_conversion(arg_value):
                converted = self._to_te_dtype(arg_value)
                args_list[i] = converted

        return func(self, *args_list, **kwargs)

    return wrapper

class IluvatarBackend(TEFLBackendBase):
    @staticmethod
    def check_available() -> bool:
        return _check_iluvatar_available()

    def __init__(self):
        self._tex = None

    def _get_tex(self):
        if self._tex is None:
            self._tex = _get_tex()
        return self._tex

    def _to_te_dtype(self, torch_dtype):
        return _torch_dtype_to_te_dtype(torch_dtype, self._get_tex())

    def is_available(self) -> bool:
        return _check_iluvatar_available()
    
    def get_flash_attention_class(self):
        raise NotImplementedError("get_flash_attention_class - not implemented in iluvatar backend")

    def get_attention_backend(self, attention_params=None):
        raise NotImplementedError("get_attention_backend - not implemented in iluvatar backend")
    
    def quantize(
        self,
        tensor: torch.Tensor,
        quantizer: Any,
        output: Optional[torch.Tensor] = None,
        noop: Optional[torch.Tensor] = None,
    ) -> Any:
        tex = self._get_tex()
        return tex.quantize(tensor, quantizer, output, noop)

    @_convert_dtype_params
    def dequantize(
        self,
        input: torch.Tensor,
        otype: torch.dtype,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.dequantize(input, otype)

    def bgrad_quantize(
        self,
        input: torch.Tensor,
        quantizer: Any,
    ) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.bgrad_quantize(input, quantizer)

    @_convert_dtype_params
    def generic_gemm(
        self,
        A: torch.Tensor,
        transA: bool,
        B: torch.Tensor,
        transB: bool,
        D: torch.Tensor,
        quantizer: Any,
        output_dtype: torch.dtype,
        bias: Optional[torch.Tensor],
        bias_type: Any,
        gelu: bool,
        gelu_in: Optional[torch.Tensor],
        grad: bool,
        workspace: torch.Tensor,
        workspace_size: int,
        accumulate: bool,
        use_split_accumulator: bool,
        comm_overlap: Optional[Any] = None,
        comm_type: Optional[Any] = None,
        extra_output: Optional[torch.Tensor] = None,
        bulk_overlap: bool = False,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ) -> Any:
        # Check shape
        tex = self._get_tex()

        if bias_type is None:
            bias_type = self._to_te_dtype(torch.bfloat16)

        return tex.generic_gemm(
            A, transA, B, transB, D, quantizer, output_dtype,
            bias, bias_type, gelu, gelu_in, grad, workspace, workspace_size,
            accumulate, use_split_accumulator, comm_overlap, comm_type,
            extra_output, bulk_overlap, alpha, beta
        )

    def te_general_grouped_gemm(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.te_general_grouped_gemm(*args, **kwargs)

    def gelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.gelu(input, quantizer)

    def geglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.geglu(input, quantizer)

    def qgelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgelu(input, quantizer)

    def qgeglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.qgeglu(input, quantizer)

    def relu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.relu(input, quantizer)

    def reglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.reglu(input, quantizer)

    def srelu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.srelu(input, quantizer)

    def sreglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.sreglu(input, quantizer)

    def silu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.silu(input, quantizer)

    def swiglu(self, input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.swiglu(input, quantizer)

    def clamped_swiglu(
            self,
            input: torch.Tensor,
            quantizer: Any,
            limit: float = 7.0,
            alpha: float = 1.702,
        ) -> Any:
        tex = self._get_tex()
        return tex.clamped_swiglu(input, quantizer, limit, alpha)

    def dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgelu(grad, fwd_input, quantizer)

    def dgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dgeglu(grad, fwd_input, quantizer)

    def dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgelu(grad, fwd_input, quantizer)

    def dqgeglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dqgeglu(grad, fwd_input, quantizer)

    def drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.drelu(grad, fwd_input, quantizer)

    def dreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dreglu(grad, fwd_input, quantizer)

    def dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsrelu(grad, fwd_input, quantizer)

    def dsreglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsreglu(grad, fwd_input, quantizer)

    def dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dsilu(grad, fwd_input, quantizer)

    def dswiglu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Any:
        tex = self._get_tex()
        return tex.dswiglu(grad, fwd_input, quantizer)

    def clamped_dswiglu(
        self,
        grad: torch.Tensor,
        fwd_input: torch.Tensor,
        quantizer: Any,
        limit: float = 7.0,
        alpha: float = 1.702,
    ) -> Any:
        tex = self._get_tex()
        return tex.clamped_dswiglu(grad, fwd_input, quantizer, limit, alpha)

    def dbias_dgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.dbias_dgelu(grad, fwd_input, quantizer)

    def dbias_dsilu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.dbias_dsilu(grad, fwd_input, quantizer)

    def dbias_drelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.dbias_drelu(grad, fwd_input, quantizer)

    def dbias_dqgelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.dbias_dqgelu(grad, fwd_input, quantizer)

    def dbias_dsrelu(self, grad: torch.Tensor, fwd_input: torch.Tensor, quantizer: Any) -> Tuple[torch.Tensor, Any]:
        tex = self._get_tex()
        return tex.dbias_dsrelu(grad, fwd_input, quantizer)

    @_convert_dtype_params
    def layernorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()

        orig_shape = input.shape
        if input.ndim > 2:
            input = input.view(-1, input.shape[-1])

        y, mu, rsigma = tex.layernorm_fwd(
            input, weight, bias, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma
        )

        if len(orig_shape) > 2:
            y = y.view(*orig_shape)
        return y, mu, rsigma

    def layernorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tex = self._get_tex()

        orig_shape = dy.shape
        if dy.ndim > 2:
            dy = dy.view(-1, dy.shape[-1])
            x = x.view(-1, x.shape[-1])

        dx, dgamma, dbeta = tex.layernorm_bwd(dy, x, mu, rsigma, gamma, sm_margin, zero_centered_gamma)

        if len(orig_shape) > 2:
            dx = dx.view(*orig_shape)
        return dx, dgamma, dbeta

    @_convert_dtype_params
    def rmsnorm_fwd(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        ln_out: Optional[torch.Tensor],
        quantizer: Any,
        otype: torch.dtype,
        sm_margin: int,
        zero_centered_gamma: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        tex = self._get_tex()

        orig_shape = input.shape
        if input.ndim > 2:
            input = input.view(-1, input.shape[-1])

        y, y_quant, rsigma = tex.rmsnorm_fwd(
            input, weight, eps, ln_out, quantizer, otype, sm_margin, zero_centered_gamma
        )

        if len(orig_shape) > 2:
            y = y.view(*orig_shape)
            if y_quant is not None:
                y_quant = y_quant.view(*orig_shape)
        return y, y_quant, rsigma

    def rmsnorm_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        rsigma: torch.Tensor,
        gamma: torch.Tensor,
        sm_margin: int = 0,
        zero_centered_gamma: bool = False,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()

        orig_shape = dy.shape
        if dy.ndim > 2:
            dy = dy.view(-1, dy.shape[-1])
            x = x.view(-1, x.shape[-1])

        dx, dw = tex.rmsnorm_bwd(dy, x, rsigma, gamma, sm_margin, zero_centered_gamma)

        if len(orig_shape) > 2:
            dx = dx.view(*orig_shape)
        return dx, dw

    def rmsnorm_bwd_add(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.rmsnorm_bwd_add(*args, **kwargs)

    def multi_tensor_quantize(
        self,
        tensor_list: List[torch.Tensor],
        quantizer_list: List[Any],
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.multi_tensor_quantize(tensor_list, quantizer_list)

    def split_quantize(
        self,
        tensor: torch.Tensor,
        split_sections: List[int],
        quantizer_list: List[Any],
    ) -> List[Any]:
        tex = self._get_tex()
        return tex.split_quantize(tensor, split_sections, quantizer_list)

    def moe_permute_fwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex._moe_permute_fwd(*args, **kwargs)
    
    def moe_permute_bwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex._moe_permute_bwd(*args, **kwargs)

    def moe_unpermute_fwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex._moe_unpermute_fwd(*args, **kwargs)

    def moe_unpermute_bwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex._moe_unpermute_bwd(*args, **kwargs)

    def scaled_softmax_forward(self, input: torch.Tensor, scale: float) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_forward(input, scale)

    def scaled_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_softmax_backward(output_grad, softmax_output, scale)

    def scaled_masked_softmax_forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_forward(input, mask, scale)

    def scaled_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_masked_softmax_backward(output_grad, softmax_output, scale)

    def scaled_upper_triang_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_forward(input, scale)

    def scaled_upper_triang_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_upper_triang_masked_softmax_backward(output_grad, softmax_output, scale)  

    def scaled_aligned_causal_masked_softmax_forward(
        self,
        input: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_forward(input, scale)

    def scaled_aligned_causal_masked_softmax_backward(
        self,
        output_grad: torch.Tensor,
        softmax_output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.scaled_aligned_causal_masked_softmax_backward(output_grad, softmax_output, scale)

    def get_fused_attn_backend(self, *args, **kwargs) -> int:
        tex = self._get_tex()

        args_list = list(args)

        def convert_enum(py_enum, native_enum_class):
            if py_enum is None:
                return None

            if type(py_enum).__module__ == 'transformer_engine_torch_nv':
                return py_enum

            if hasattr(py_enum, 'name'):
                enum_name = py_enum.name
                if hasattr(native_enum_class, enum_name):
                    return getattr(native_enum_class, enum_name)

            if hasattr(py_enum, 'value'):
                enum_value = int(py_enum.value)
                for member_name in dir(native_enum_class):
                    if not member_name.startswith('_'):
                        try:
                            member = getattr(native_enum_class, member_name)
                            if hasattr(member, 'value') and int(member.value) == enum_value:
                                return member
                        except:
                            pass

            if hasattr(py_enum, 'value'):
                return int(py_enum.value)

            return py_enum

        if len(args) > 1:
            args_list[1] = self._to_te_dtype(args[1])
        if len(args) > 2:
            args_list[2] = self._to_te_dtype(args[2])
        if len(args) > 3:
            args_list[3] = convert_enum(args[3], tex.NVTE_QKV_Layout)
        if len(args) > 4:
            args_list[4] = convert_enum(args[4], tex.NVTE_Bias_Type)
        if len(args) > 5:
            args_list[5] = convert_enum(args[5], tex.NVTE_Mask_Type)
        if len(args) > 6:
            args_list[6] = convert_enum(args[6], tex.NVTE_Softmax_Type)

        return tex.get_fused_attn_backend(*args_list, **kwargs)

    def fused_attn_fwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()

        def convert_enum(py_enum, native_enum_class):
            if py_enum is None:
                return None
            if type(py_enum).__module__ == 'transformer_engine_torch_nv':
                return py_enum
            if hasattr(py_enum, 'name'):
                enum_name = py_enum.name
                if hasattr(native_enum_class, enum_name):
                    return getattr(native_enum_class, enum_name)
            return py_enum

        args_list = list(args)
        if len(args) > 6:
            args_list[6] = convert_enum(args[6], tex.NVTE_QKV_Layout)
        if len(args) > 7:
            args_list[7] = convert_enum(args[7], tex.NVTE_Bias_Type)
        if len(args) > 8:
            args_list[8] = convert_enum(args[8], tex.NVTE_Mask_Type)
        if len(args) > 9:
            args_list[9] = convert_enum(args[9], tex.NVTE_Softmax_Type)

        return tex.fused_attn_fwd(*args_list, **kwargs)

    def fused_attn_bwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()

        def convert_enum(py_enum, native_enum_class):
            if py_enum is None:
                return None
            if type(py_enum).__module__ == 'transformer_engine_torch_nv':
                return py_enum
            if hasattr(py_enum, 'name'):
                enum_name = py_enum.name
                if hasattr(native_enum_class, enum_name):
                    return getattr(native_enum_class, enum_name)
            return py_enum

        args_list = list(args)
        if len(args) > 5:
            args_list[5] = convert_enum(args[5], tex.NVTE_QKV_Layout)
        if len(args) > 6:
            args_list[6] = convert_enum(args[6], tex.NVTE_Bias_Type)
        if len(args) > 7:
            args_list[7] = convert_enum(args[7], tex.NVTE_Mask_Type)
        if len(args) > 8:
            args_list[8] = convert_enum(args[8], tex.NVTE_Softmax_Type)
        if len(args) > 19:
            args_list[19] = self._to_te_dtype(args[19])

        if 'dqkv_dtype' in kwargs:
            kwargs['dqkv_dtype'] = self._to_te_dtype(kwargs['dqkv_dtype'])

        return tex.fused_attn_bwd(*args_list, **kwargs)

    def fa_prepare_fwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fa_prepare_fwd(*args, **kwargs)

    def fa_prepare_bwd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fa_prepare_bwd(*args, **kwargs)

    def copy_to_kv_cache(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.copy_to_kv_cache(*args, **kwargs)

    def convert_thd_to_bshd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.convert_thd_to_bshd(*args, **kwargs)

    def convert_bshd_to_thd(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.convert_bshd_to_thd(*args, **kwargs)

    def fused_rope_forward(self, *args, **kwargs) -> Any:
        assert args[2] is None, "[Iluvatar] fused_rope_forward does not support start_position now."
        assert args[3].name == "NVTE_SBHD", f"[Iluvatar] fused_rope_forward expect NVTE_SBHD, but got {args[3].name}."
        tex = self._get_tex()
        return tex.fused_rope_forward(args[0], args[1], False, False, 1.0)

    def fused_rope_backward(self, *args, **kwargs) -> Any:
        assert args[2].name == "NVTE_SBHD", f"[Iluvatar] fused_rope_backward expect NVTE_SBHD, but got {args[2].name}."
        tex = self._get_tex()
        return tex.fused_rope_backward(args[0], args[1], False, False, 1.0)

    def fused_qkv_rope_forward(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fused_qkv_rope_forward(*args, **kwargs)

    def fused_qkv_rope_backward(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fused_qkv_rope_backward(*args, **kwargs)

    def fused_topk_with_score_function_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        num_groups: int,
        group_topk: int,
        scaling_factor: float,
        score_function: Any,
        expert_bias: Optional[torch.Tensor],
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_fwd(
            logits, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias
        )

    def fused_topk_with_score_function_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        routing_map: torch.Tensor,
        intermediate_output: torch.Tensor,
        grad_probs: torch.Tensor,
        topk: int,
        use_pre_softmax: bool,
        scaling_factor: float,
        score_function: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_topk_with_score_function_bwd(
            num_tokens, num_experts, routing_map, intermediate_output,
            grad_probs, topk, use_pre_softmax, scaling_factor, score_function
        )

    def fused_score_for_moe_aux_loss_fwd(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_fwd(logits, topk, score_function)

    def fused_score_for_moe_aux_loss_bwd(
        self,
        num_tokens: int,
        num_experts: int,
        intermediate_output: torch.Tensor,
        grad_scores: torch.Tensor,
        topk: int,
        score_function: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_score_for_moe_aux_loss_bwd(
            num_tokens, num_experts, intermediate_output, grad_scores, topk, score_function
        )

    def fused_moe_aux_loss_fwd(
        self,
        probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        total_num_tokens: int,
        num_experts: int,
        num_rows: int,
        num_cols: int,
        topk: int,
        coeff: float,
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_fwd(
            probs, tokens_per_expert, total_num_tokens, num_experts,
            num_rows, num_cols, topk, coeff
        )

    def fused_moe_aux_loss_bwd(
        self,
        Const_buf: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_rows: int,
        num_cols: int,
        grad_aux_loss: torch.Tensor,
    ) -> Any:
        tex = self._get_tex()
        return tex.fused_moe_aux_loss_bwd(
            Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss
        )

    def dropout_fwd(
        self,
        input: torch.Tensor,
        dropout_probability: float,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tex = self._get_tex()
        return tex.dropout_fwd(input, dropout_probability, out)

    def dropout_bwd(
        self,
        grad_output: torch.Tensor,
        mask: torch.Tensor,
        dropout_probability: float,
        grad_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tex = self._get_tex()
        return tex.dropout_bwd(grad_output, mask, dropout_probability, grad_input)

    def fp8_transpose(
        self,
        input: torch.Tensor,
        dtype: Any,
        *,
        out: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        tex.fp8_transpose(input, dtype, out=out)

    def swap_first_dims(
        self,
        tensor: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        tex.swap_first_dims(tensor, out=out)

    def compute_amax(
        self,
        input: torch.Tensor,
        amax: torch.Tensor,
    ) -> None:
        tex = self._get_tex()
        tex.compute_amax(input, amax)

    def fused_amax_and_scale_update_after_reduction(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.fused_amax_and_scale_update_after_reduction(*args, **kwargs)

    def fp8_block_scaling_compute_partial_amax(
        self,
        tensor: torch.Tensor,
        amax: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
    ) -> None:
        tex = self._get_tex()
        tex.fp8_block_scaling_compute_partial_amax(tensor, amax, h, w, start_offset, block_len)

    def fp8_block_scaling_partial_cast(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        scale: torch.Tensor,
        h: int,
        w: int,
        start_offset: int,
        block_len: int,
        out_dtype: Any,
    ) -> None:
        tex = self._get_tex()
        tex.fp8_block_scaling_partial_cast(inp, out, scale, h, w, start_offset, block_len, out_dtype)

    def fused_multi_row_padding(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fused_multi_row_padding(*args, **kwargs)

    def fused_multi_row_unpadding(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.fused_multi_row_unpadding(*args, **kwargs)

    def get_cublasLt_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cublasLt_version()

    def get_cudnn_version(self) -> int:
        tex = self._get_tex()
        return tex.get_cudnn_version()

    def get_num_cublas_streams(self) -> int:
        tex = self._get_tex()
        return tex.get_num_cublas_streams()

    def thd_read_half_tensor(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_read_half_tensor(*args, **kwargs)

    def thd_second_half_lse_correction(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_second_half_lse_correction(*args, **kwargs)

    def thd_read_second_half_lse(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_read_second_half_lse(*args, **kwargs)

    def thd_out_correction(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_out_correction(*args, **kwargs)

    def thd_grad_correction(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_grad_correction(*args, **kwargs)

    def thd_get_partitioned_indices(self, *args, **kwargs) -> Any:
        tex = self._get_tex()
        return tex.thd_get_partitioned_indices(*args, **kwargs)

    def init_nvshmem_backend(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.init_nvshmem_backend(*args, **kwargs)

    def create_nvshmem_tensor(self, *args, **kwargs) -> torch.Tensor:
        tex = self._get_tex()
        return tex.create_nvshmem_tensor(*args, **kwargs)

    def nvshmem_send_on_current_stream(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.nvshmem_send_on_current_stream(*args, **kwargs)

    def nvshmem_wait_on_current_stream(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.nvshmem_wait_on_current_stream(*args, **kwargs)

    def nvshmem_finalize(self) -> None:
        tex = self._get_tex()
        tex.nvshmem_finalize()

    def multi_tensor_scale(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: float,
    ) -> None:
        tex = self._get_tex()
        tex.multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale)

    def multi_tensor_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        return tex.multi_tensor_l2norm(chunk_size, noop_flag, tensor_lists, per_tensor)

    def multi_tensor_unscale_l2norm(
        self,
        chunk_size: int,
        noop_flag: torch.Tensor,
        tensor_lists: List[List[torch.Tensor]],
        scale: torch.Tensor,
        per_tensor: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        tex = self._get_tex()
        return tex.multi_tensor_unscale_l2norm(chunk_size, noop_flag, tensor_lists, scale, per_tensor)

    def multi_tensor_adam(
        self,
        chunk_size: int = None,
        noop_flag: torch.Tensor = None,
        tensor_lists: List[List[torch.Tensor]] = None,
        lr: float = None,
        beta1: float = None,
        beta2: float = None,
        eps: float = None,
        step: int = None,
        mode: int = None,
        bias_correction: int = None,
        weight_decay: float = None,
    ):
        tex = self._get_tex()
        if chunk_size is None:
            return tex.multi_tensor_adam
        tex.multi_tensor_adam(
            chunk_size, noop_flag, tensor_lists, lr, beta1, beta2,
            eps, step, mode, bias_correction, weight_decay
        )

    def multi_tensor_adam_param_remainder(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_adam_param_remainder(*args, **kwargs)

    def multi_tensor_adam_fp8(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_adam_fp8(*args, **kwargs)

    def multi_tensor_adam_capturable(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_adam_capturable(*args, **kwargs)

    def multi_tensor_adam_capturable_master(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_adam_capturable_master(*args, **kwargs)

    def multi_tensor_sgd(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_sgd(*args, **kwargs)

    def multi_tensor_compute_scale_and_scale_inv(self, *args, **kwargs) -> None:
        tex = self._get_tex()
        tex.multi_tensor_compute_scale_and_scale_inv(*args, **kwargs)

    def bulk_overlap_ag_with_external_gemm(
        self,
        allgather_communicator: Any,
        send_stream: Any,
        recv_stream: Any,
    ) -> Any:
        tex = self._get_tex()
        return tex.bulk_overlap_ag_with_external_gemm(allgather_communicator, send_stream, recv_stream)

    def create_fp8_tensor_meta(self) -> FP8TensorMeta:
        tex = self._get_tex()
        return tex.FP8TensorMeta()

    def create_comm_overlap_helper(
        self,
        world_group: Optional[Any] = None,
        intra_node_group: Optional[Any] = None,
    ) -> Any:
        tex = self._get_tex()
        if world_group is None:
            return tex.CommOverlapHelper()
        return tex.CommOverlapHelper(world_group, intra_node_group)

    def create_comm_overlap(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        num_splits: int = 3,
        num_max_streams: int = 3,
        comm_cga_size: int = 2,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 16,
        set_sm_margin: bool = True,
        atomic_gemm: bool = False,
        rs_overlap_first_gemm: bool = False,
    ) -> Any:
        tex = self._get_tex()
        return tex.CommOverlap(
            buffer_shape, buffer_dtype, helper, tp_size,
            num_splits, num_max_streams, comm_cga_size,
            gemm_priority, comm_priority, num_comm_sm,
            set_sm_margin, atomic_gemm, rs_overlap_first_gemm
        )

    def create_comm_overlap_p2p(
        self,
        buffer_shape: List[int],
        buffer_dtype: torch.dtype,
        helper: Any,
        tp_size: int,
        comm_type: Any,
        num_max_streams: int = 3,
        comm_cga_size: int = 1,
        gemm_priority: int = 0,
        comm_priority: int = 0,
        num_comm_sm: int = 1,
        set_sm_margin: bool = False,
        atomic_gemm: bool = False,
        use_ce: bool = True,
        aggregate: bool = False,
    ) -> Any:
        tex = self._get_tex()
        return tex.CommOverlapP2P(
            buffer_shape, buffer_dtype, helper, tp_size, comm_type,
            num_max_streams, comm_cga_size, gemm_priority, comm_priority,
            num_comm_sm, set_sm_margin, atomic_gemm, use_ce, aggregate
        )

    
    
