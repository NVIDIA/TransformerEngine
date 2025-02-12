# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
This file contains DebugQuantizer and DebugQuantizedTensor objects,
which are wrappers over Quantizer and QuantizedTensor.
These wrapper add logic related to the debugging, using the nvdlfw_inspect package.
"""

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Union
import torch

import transformer_engine_torch as tex


from ...pytorch.tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)

aten = torch.ops.aten

_tensor_to_gemm_names_map = {
    "weight": ["fprop", "dgrad"],
    "activation": ["fprop", "wgrad"],
    "output": ["fprop", None],
    "gradient": ["dgrad", "wgrad"],
    "wgrad": ["wgrad", None],
    "dgrad": ["dgrad", None],
}

API_CALL_MODIFY = "modify_tensor()"
STANDARD_FP8_QUANTIZE = "FP8 Quantize"
HIGH_PRECISION = "High Precision"


class DebugQuantizer(Quantizer):
    """
    DebugQuantizer is Quantizer object used for debugging with nvidia-dlframework-inspect.
    It allows to add custom calls inside quantization process - which enable to modify tensors
    or gather tensors stats.
    """

    def __init__(
        self,
        layer_name: str,
        tensor_name: str,
        parent_quantizer: Optional[Quantizer],
        tp_group: torch.distributed.ProcessGroup,
    ):
        import nvdlfw_inspect.api as debug_api

        super().__init__(rowwise=True, columnwise=True)
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.parent_quantizer = parent_quantizer
        self.tp_group = tp_group  # used in inspect_tensor calls
        self.iteration = debug_api.DEBUG_MANAGER._trainer_iteration_count

        self.rowwise_gemm_name, self.columnwise_gemm_name = _tensor_to_gemm_names_map[tensor_name]

        # The values of the inspect_tensor_enabled, inspect_tensor_postquantize_enabled,
        # rowwise_tensor_plan and  columnwise_tensor_plan are computed.
        # These fields indicate the path which API calls will be inserted.
        #
        # inspect_tensor*_enabled are bool fields,
        # inicating whether some feature will need to run inspect_tensor_* calls.
        #
        # *_tensor_plan are one of [API_TENSOR_MODIFY, STANDARD_FP8_QUANTIZE, HIGH_PRECISION]
        # determining what will happen when quantizer is used for that tensor.
        self.output_tensor = tensor_name in ["output", "wgrad", "dgrad"]
        if self.output_tensor:
            self.inspect_tensor_enabled, self.rowwise_tensor_plan = (
                self.get_plans_for_output_tensors()
            )
        else:
            self.inspect_tensor_enabled, self.inspect_tensor_postquantize_enabled = (
                self.get_enabled_look_at_tensors()
            )
            self.rowwise_tensor_plan, self.columnwise_tensor_plan = self.get_tensors_plan()

            self.log_messages_about_plans()

    def get_plans_for_output_tensors(self) -> Tuple[bool, str]:
        """
        Returns tuple (inspect_tensor_enabled: bool, plan: str). Plan is one of the
        CALL_PROCESS_TENSOR or HIGH_PRECISION, because debug quantizer does not support
        gemm output in FP8.
        """
        import nvdlfw_inspect.api as debug_api

        inspect_tensor_enabled = debug_api.transformer_engine.inspect_tensor_enabled(
            layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=self.iteration
        )
        modify_enabled = debug_api.transformer_engine.modify_tensor_enabled(
            layer_name=self.layer_name,
            gemm=self.rowwise_gemm_name,
            tensor_name=self.tensor_name,
            iteration=self.iteration,
        )
        plan = API_CALL_MODIFY if modify_enabled else HIGH_PRECISION

        return inspect_tensor_enabled, plan

    def get_enabled_look_at_tensors(self):
        """
        Returns tuple of booleans determining which functions look_at_tensor_*(...) should be called.
        """
        import nvdlfw_inspect.api as debug_api

        inspect_tensor_enabled = debug_api.transformer_engine.inspect_tensor_enabled(
            layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=self.iteration
        )
        inspect_tensor_postquantize_enabled = (
            debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
                layer_name=self.layer_name, tensor_name=self.tensor_name, iteration=self.iteration
            )
        )

        return inspect_tensor_enabled, inspect_tensor_postquantize_enabled

    def get_tensors_plan(self):
        """
        Retruns (rowwise_plan, columnwise_plan), that each element of the tuple is one of the
        API_CALL_MODIFY, STANDARD_FP8_QUANTIZE, HIGH_PRECISION, indicating what quanitzer
        does with these tensors.
        """
        import nvdlfw_inspect.api as debug_api

        rowwise_plan = None
        columnwise_plan = None

        modify_rowwise = debug_api.transformer_engine.modify_tensor_enabled(
            layer_name=self.layer_name,
            gemm=self.rowwise_gemm_name,
            tensor_name=self.tensor_name,
            iteration=self.iteration,
        )
        if modify_rowwise:
            rowwise_plan = API_CALL_MODIFY
        else:
            if self.parent_quantizer is not None:
                fp8_quantize = debug_api.transformer_engine.fp8_gemm_enabled(
                    layer_name=self.layer_name,
                    gemm=self.rowwise_gemm_name,
                    iteration=self.iteration,
                )
                if fp8_quantize:
                    rowwise_plan = STANDARD_FP8_QUANTIZE
        if rowwise_plan is None:
            rowwise_plan = HIGH_PRECISION

        if self.columnwise_gemm_name is not None:
            modify_columnwise = debug_api.transformer_engine.modify_tensor_enabled(
                layer_name=self.layer_name,
                gemm=self.columnwise_gemm_name,
                tensor_name=self.tensor_name,
                iteration=self.iteration,
            )
            if modify_columnwise:
                columnwise_plan = API_CALL_MODIFY
            else:
                if self.parent_quantizer is not None:
                    fp8_quantize = debug_api.transformer_engine.fp8_gemm_enabled(
                        layer_name=self.layer_name,
                        gemm=self.columnwise_gemm_name,
                        iteration=self.iteration,
                    )
                    if fp8_quantize:
                        columnwise_plan = STANDARD_FP8_QUANTIZE
        if columnwise_plan is None:
            columnwise_plan = HIGH_PRECISION

        return rowwise_plan, columnwise_plan

    def log_messages_about_plans(self):
        """
        Logs the messages about the plans for each of the tensors.
        """
        import nvdlfw_inspect.api as debug_api

        debug_api.log_message(
            f"Tensor: {self.tensor_name}, gemm {self.rowwise_gemm_name} -"
            f" {self.rowwise_tensor_plan}",
            layer_name=self.layer_name,
            extra_cachable_args=(self.rowwise_gemm_name, self.tensor_name),
        )
        debug_api.log_message(
            f"Tensor: {self.tensor_name}, gemm {self.columnwise_gemm_name} -"
            f" {self.columnwise_tensor_plan}",
            layer_name=self.layer_name,
            extra_cachable_args=(self.columnwise_gemm_name, self.tensor_name),
        )

    def _call_inspect_tensor_api(
        self, tensor, rowwise_gemm_tensor=None, columnwise_gemm_tensor=None
    ):
        import nvdlfw_inspect.api as debug_api

        args = {
            "layer_name": self.layer_name,
            "tensor": tensor,
            "tensor_name": self.tensor_name,
            "iteration": debug_api.DEBUG_MANAGER._trainer_iteration_count,
            "tp_group": self.tp_group,
        }
        if tensor is not None:
            debug_api.transformer_engine.inspect_tensor(**args)

        if self.output_tensor:
            return

        if self.rowwise_tensor_plan in [API_CALL_MODIFY, STANDARD_FP8_QUANTIZE]:
            args["tensor"] = rowwise_gemm_tensor
            args["rowwise"] = True
            debug_api.transformer_engine.inspect_tensor_postquantize(**args)
        if self.columnwise_tensor_plan in [API_CALL_MODIFY, STANDARD_FP8_QUANTIZE]:
            args["tensor"] = columnwise_gemm_tensor
            args["rowwise"] = False
            debug_api.transformer_engine.inspect_tensor_postquantize(**args)

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        out: Optional[Union[torch.Tensor, DebugQuantizedTensor]] = None,
        dtype: torch.dtype = None,
    ):
        """Returns DebugQuantizedTensor object."""
        import nvdlfw_inspect.api as debug_api

        assert not self.output_tensor
        if out is not None:
            return self.update_quantized(tensor, self)

        # 1. If there is fp8 quantization in at least one of the gemms,
        #    the quantization using the self.parent_quantizer is performed.

        # rowwise gemm corresponds to the rowwise_usage in fp8, similarly with columnwise
        rowwise_gemm_quantize = (
            self.rowwise_usage and self.rowwise_tensor_plan == STANDARD_FP8_QUANTIZE
        )
        columnwise_gemm_quantize = (
            self.columnwise_usage and self.columnwise_tensor_plan == STANDARD_FP8_QUANTIZE
        )
        if columnwise_gemm_quantize and not rowwise_gemm_quantize:
            rowwise_gemm_quantize = True  # only columnwise quantization not implemented

        rowwise_gemm_tensor, columnwise_gemm_tensor = None, None
        if STANDARD_FP8_QUANTIZE in [self.rowwise_tensor_plan, self.columnwise_tensor_plan]:
            self.parent_quantizer.set_usage(
                rowwise=rowwise_gemm_quantize, columnwise=columnwise_gemm_quantize
            )
            quantized_tensor = self.parent_quantizer(tensor)
            # if both rowwise_tensor_plan and columnwise_tensor_plan need to be in fp8,
            # one tensor with columnwise=True and rowwise=True is computed
            # and both rowwise_tensor_plan and columnwise_tensor_plan point to it.
            if self.rowwise_tensor_plan == STANDARD_FP8_QUANTIZE:
                rowwise_gemm_tensor = quantized_tensor
            if self.columnwise_tensor_plan == STANDARD_FP8_QUANTIZE:
                columnwise_gemm_tensor = quantized_tensor

        # 2. modify_tensor() is called, if it is used.
        if self.columnwise_tensor_plan == API_CALL_MODIFY:
            columnwise_gemm_tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.columnwise_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=self.iteration,
                dtype=dtype,
            )
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            rowwise_gemm_tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.rowwise_gemm_name,
                tensor=tensor,
                default_quantizer=self.parent_quantizer,
                iteration=self.iteration,
                dtype=dtype,
            )

        # 3. If some tensors still are not defined we use high precision tensor.
        if self.rowwise_tensor_plan == HIGH_PRECISION:
            rowwise_gemm_tensor = tensor.to(dtype)
        if self.columnwise_tensor_plan == HIGH_PRECISION:
            columnwise_gemm_tensor = tensor.to(dtype)

        self._call_inspect_tensor_api(tensor, rowwise_gemm_tensor, columnwise_gemm_tensor)

        # sometimes we may want to return simple tensor with only rowwise_gemm
        if self.tensor_name in ["wgrad", "dgrad", "output"]:
            return rowwise_gemm_tensor

        return DebugQuantizedTensor(
            shape=tensor.shape,
            dtype=tensor.dtype,
            rowwise_gemm_tensor=rowwise_gemm_tensor,
            columnwise_gemm_tensor=columnwise_gemm_tensor,
            quantizer=self,
            layer_name=self.layer_name,
            tensor_name=self.tensor_name,
        )

    def process_gemm_output(self, tensor: torch.Tensor):
        """This call is invoked after the gemm to inspect and modify the output tensor."""
        import nvdlfw_inspect.api as debug_api

        assert self.parent_quantizer is None, "FP8 output is not supported for debug=True."
        assert self.output_tensor
        tensor_to_gemm = {"output": "fprop", "wgrad": "wgrad", "dgrad": "dgrad"}
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            tensor = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                gemm=tensor_to_gemm[self.tensor_name],
                tensor_name=self.tensor_name,
                tensor=tensor,
                iteration=self.iteration,
                default_quantizer=self.parent_quantizer,
            )
        self._call_inspect_tensor_api(tensor)
        return tensor

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensor:
        """Override make_empty() from Quantizer class."""
        if self.parent_quantizer is not None:
            return self.parent_quantizer(shape, dtype=dtype, device=device)
        return torch.empty(shape, dtype=dtype, device=device)

    def calibrate(self, tensor: torch.Tensor):
        """Calibration override, should not be invoked."""
        raise RuntimeError("[NVTORCH-INSPECT ERROR] Calibration with debug is not supported")

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Update quantized tensor - used in weight caching."""
        import nvdlfw_inspect.api as debug_api

        assert noop_flag is None, "CUDA Graphs are not supported with debug=True!"

        updated_rowwise_gemm = False
        if self.parent_quantizer is not None:
            if (
                dst.rowwise_gemm_tensor is not None
                and self.rowwise_tensor_plan == STANDARD_FP8_QUANTIZE
            ):
                if hasattr(dst.rowwise_gemm_tensor, "quantize_"):
                    dst.rowwise_gemm_tensor.quantize_(src, noop_flag=None)
                else:
                    tex.quantize(src, self.parent_quantizer, dst.rowwise_gemm_tensor, None)
                updated_rowwise_gemm = True
            if (
                dst.columnwise_gemm_tensor is not None
                and self.columnwise_tensor_plan == STANDARD_FP8_QUANTIZE
                and not updated_rowwise_gemm
            ):
                if hasattr(dst.columnwise_gemm_tensor, "quantize_"):
                    dst.columnwise_gemm_tensor.quantize_(src, noop_flag=None)
                else:
                    tex.quantize(src, self.parent_quantizer, dst.columnwise_gemm_tensor, None)

        if self.columnwise_tensor_plan == API_CALL_MODIFY:
            out = debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.columnwise_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.columnwise_gemm_tensor,
                iteration=self.iteration,
            )
            assert out is None, (
                "API call debug_api.transformer_engine.modify_tensor with out != None should"
                " return None"
            )
        if self.rowwise_tensor_plan == API_CALL_MODIFY:
            debug_api.transformer_engine.modify_tensor(
                layer_name=self.layer_name,
                tensor_name=self.tensor_name,
                gemm=self.rowwise_gemm_name,
                tensor=src,
                default_quantizer=self.parent_quantizer,
                out=dst.rowwise_gemm_tensor,
                iteration=self.iteration,
            )

        if self.rowwise_tensor_plan == HIGH_PRECISION:
            dst.rowwise_gemm_tensor.copy_(src)
        if self.columnwise_tensor_plan == HIGH_PRECISION:
            # if they are the same tensor object, it is sufficient to update one
            if dst.columnwise_gemm_tensor is not dst.columnwise_gemm_tensor:
                dst.columnwise_gemm_tensor.copy_(src)

        self._call_inspect_tensor_api(src, dst.rowwise_gemm_tensor, dst.columnwise_gemm_tensor)

    def any_feature_enabled(self) -> bool:
        """Returns bool if there is at least one API call enabled."""
        if self.output_tensor:
            return self.inspect_tensor_enabled or self.rowwise_gemm_name == API_CALL_MODIFY
        if (
            self.inspect_tensor_enabled
            or self.inspect_tensor_postquantize_enabled
            or self.rowwise_gemm_name == API_CALL_MODIFY
            or self.columnwise_gemm_name == API_CALL_MODIFY
        ):
            return True
        if self.parent_quantizer is not None:
            if self.rowwise_gemm_name != STANDARD_FP8_QUANTIZE:
                return True
            if self.columnwise_tensor_plan != STANDARD_FP8_QUANTIZE:
                return True
        return False


class DebugQuantizedTensor(QuantizedTensor):
    """
    Class containing quantized tensors after debug. Depending on configuration
    it can contain one or two different objects. These objects can be accessed by the method
    get_tensor().
    """

    def __new__(
        cls,
        shape,
        dtype,
        rowwise_gemm_tensor,
        columnwise_gemm_tensor,
        quantizer,
        requires_grad=False,
        layer_name=None,
        tensor_name=None,
    ):
        instance = super().__new__(cls, shape, dtype, requires_grad=requires_grad)

        instance.rowwise_gemm_tensor = rowwise_gemm_tensor
        instance.columnwise_gemm_tensor = columnwise_gemm_tensor
        instance.quantizer = quantizer
        instance._layer_name = layer_name
        instance._tensor_name = tensor_name

        return instance

    def prepare_for_saving(self):
        """ " Prepare for saving method override"""
        tensor_list, tensor_objects_list = prepare_for_saving(
            self.rowwise_gemm_tensor, self.columnwise_gemm_tensor
        )
        assert len(tensor_objects_list) == 2
        # pylint: disable=unbalanced-tuple-unpacking
        self.rowwise_gemm_tensor, self.columnwise_gemm_tensor = tensor_objects_list
        return tensor_list, self

    def restore_from_saved(self, tensors):
        """ " Restore for saved method override"""
        (self.rowwise_gemm_tensor, self.columnwise_gemm_tensor), saved_tensors = restore_from_saved(
            [self.rowwise_gemm_tensor, self.columnwise_gemm_tensor],
            tensors,
            return_saved_tensors=True,
        )
        return saved_tensors

    def quantize_(self, tensor, *, noop_flag=None):
        """ " quantize_ method override"""
        assert noop_flag is None, "CUDA Graphs are not supported with debug=True!"
        self.quantizer.update_quantized(tensor, self)

    def dequantize(self, *, dtype=None):
        """ " dequantize method override"""
        if dtype is None:
            dtype = self.rowwise_gemm_tensor.dtype
        return self.rowwise_gemm_tensor.dequantize().to(dtype)

    def get_tensor(self, transpose: bool):
        """Is used in the python gemm() to get tensor or transpose of the tensor."""
        return self.rowwise_gemm_tensor if not transpose else self.columnwise_gemm_tensor

    def update_usage(self, rowwise_usage=True, columnwise_usage=True):
        pass

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """ " Method use to define .slice() on DebugQuantizedTensor."""
        if func in [aten.slice.Tensor]:
            tensor = args[0]
            rowwise_gemm_tensor = tensor.rowwise_gemm_tensor.__torch_dispatch__(
                func,
                types,
                [rowwise_gemm_tensor] + list(args[1:]),
                kwargs,
            )
            return DebugQuantizedTensor(
                shape=rowwise_gemm_tensor.shape,
                dtype=tensor.dtype,
                rowwise_gemm_tensor=rowwise_gemm_tensor,
                columnwise_gemm_tensor=None,
                quantizer=tensor.quantizer,
                requires_grad=tensor.requires_grad,
                layer_name=tensor._layer_name,
                tensor_name=tensor._tensor_name,
            )

        return super().__torch_dispatch__(func, types, args, kwargs)
