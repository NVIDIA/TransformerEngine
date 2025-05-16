# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""API definition for nvidia-dlframework-inspect."""

import copy
from typing import Dict, Union
from nvdlfw_inspect.base import BaseNamespaceAPI, BaseConfigAPIMapper
from nvdlfw_inspect.registry import Registry

import torch

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import get_all_tensor_types
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.tensor import Quantizer, QuantizedTensor


class TEConfigAPIMapper(BaseConfigAPIMapper):
    """Class responsible for determining which NV DLFW Inspect API should be run for each tensor and gemm."""

    def parse_config_and_api(self, config, **kwargs):
        """Process the config and returns True if the config and api args match, along with processed config."""
        processed_config = None
        config_copy = copy.deepcopy(config)
        gemm_parsing = kwargs.get("gemm_parsing", False)
        tensor_parsing = kwargs.get("tensor_parsing", False)

        if gemm_parsing:
            # parse with GEMM and/or tensor
            processed_config = self._process_transformer_engine_config(config_copy, **kwargs)
        elif tensor_parsing:
            # parse with only tensor
            processed_config = self._process_tensor_config(config_copy, kwargs["tensor_name"])

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")
        return True, processed_config

    def _validate_gemm(self, gemm):
        assert gemm in ["fprop", "wgrad", "dgrad"], (
            f"[NVTORCH INSPECT ERROR] Invalid gemm: {gemm}. It must be one of the ['fprop',"
            " 'wgrad', 'dgrad']."
        )

    def _process_transformer_engine_config(self, config, **kwargs):
        """
        Return config specific to a particular tensor name and gemm that matches the api args.
        """
        if "gemms_struct" in config:
            for cfg in config["gemms_struct"]:
                self._validate_gemm(cfg["gemm"])
                if cfg["gemm"] == kwargs["gemm"]:
                    if kwargs["tensor_parsing"]:
                        cfg = self._process_tensor_config(cfg, kwargs["tensor_name"])
                        if not cfg:
                            return None
                    cfg_copy = copy.deepcopy(cfg)
                    config.pop("gemms_struct")
                    assert (
                        "enabled" not in cfg_copy
                    ), "[NVTORCH INSPECT ERROR] Enabled field should not be part of gemms_struct"
                    config.update(cfg_copy)
                    return config
            return None
        if "gemms" in config:
            for gemm in config["gemms"]:
                self._validate_gemm(gemm)
            if kwargs["gemm"] in config["gemms"]:
                if kwargs["tensor_parsing"]:
                    cfg = self._process_tensor_config(config, kwargs["tensor_name"])
                    if not cfg:
                        return None
                config["gemm"] = kwargs["gemm"]
                config.pop("gemms")
                return config
            return None
        raise ValueError(
            "[NVTORCH INSPECT ERROR] Provide 'gemms_struct: List[Dict]' or 'gemms: List[str]'"
            " in the config yaml"
        )


required_kwargs = {
    "fp8_gemm_enabled": ["gemm"],
    "modify_tensor_enabled": ["tensor_name", "gemm"],
    "modify_tensor": ["tensor_name", "gemm"],
    "inspect_tensor": ["tensor_name"],
    "inspect_tensor_postquantize": ["tensor_name"],
    "inspect_tensor_enabled": ["tensor_name"],
    "inspect_tensor_postquantize_enabled": ["tensor_name"],
    "default": ["tensor_name", "gemm"],
}


# pylint: disable=unused-argument
class TEDefaultFeatures:
    """Transformer Engine API calls default behavior."""

    def fp8_gemm_enabled(self, config: Dict, layer_name: str, gemm: str, iteration: int) -> bool:
        """
        If the tensor is not processed using *modify_tensor* and the fp8 recipe is enabled,
        then the decision whether to cast it to fp8 is based on the value returned by the call *fp8_gemm_enabled*.
        If the tensor is processed using *modify_tensor* or fp8 autocast is not enabled,
        the result of this call does not matter.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        gemm: str
            one of [`fprop`, `dgrad`, `wgrad`],
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.

        Returns
        -------

        bool - default is True
        """
        return True  # if it is false, fp8_gemm will be turned off. Otherwise nothing happens.

    def modify_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        iteration: int,
    ) -> bool:
        """
        It is used to determine whether *modify_tensor* will be run for a given GEMM and tensor name. It has **higher priority** than fp8_gemm, if *modify_tensor_enabled* returns True, then modify_tensor call is invoked for the respective tensor no matter what.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        gemm: str
            one of [`fprop`, `dgrad`, `wgrad`],
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.

        Returns
        -------

        bool - default is False
        """
        return False

    def modify_tensor(
        self,
        config: Dict,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        tensor: torch.Tensor,
        default_quantizer: Quantizer,
        iteration: int,
        out: Union[torch.Tensor, QuantizedTensor],
    ) -> Union[torch.Tensor, QuantizedTensor, None]:
        """
        It allows tensor modification.
        For example, feature `FakeQuant` uses it to emulate casting to FP8.
        It can be invoked at most once for each tensor within a given GEMM operation.

        This call is invoked if `modify_tensor_enabled` returns `True` and the feature is enabled for the *tensor_name* and *gemm*.
        Then it is called **instead of** the default quantization.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        tensor: torch.Tensor
            tensor in high precision,
        gemm: str
            one of [`fprop`, `dgrad`, `wgrad`],
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
        default_quantizer : Quantizer
            quantizer which is used to cast the tensor to lower precision
            if *modify_tensor* is not invoked. For example,
            feature per tensor scale uses it to obtain FP8 dtype of the tensor.
            If the recipe indicates that the tensor is not cast - for example,
            if running without FP8 autocast, then `default_quantizer=None`,
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.
        out: Union[torch.Tensor, QuantizedTensor]
            output tensor, used in the weight caching mechanism.


        Returns
        -------

        Union[torch.Tensor, transformer_engine.pytorch.QuantizerTensor, None]
            can be `torch.Tensor` or one of the Transformer Engine's `QuantizedTensor` -
            the rule is that both tensors returned for each GEMM should have the same type.
            If both are `Float8Tensor`, then GEMM is run in FP8.
            If both are `torch.Tensor`, GEMM is run in high precision.
            Please take that into account especially if only one tensor of the GEMM
            is processed by the `modify_tensor()`. For example, `FakeQuant`
            disabled FP8 GEMM to ensure that the second tensor is also in high precision.
            If the tensor is not the input for any GEMM - namely  `output`,
            `wgrad` and `dgrad` - the return type would match the input type.
        Should return `None` if `out` is not `None`.

        """
        raise NotImplementedError(
            "modify_tensor_enabled() returned True, modify_tensor() was invoked, but it is not"
            " handled by any API."
        )

    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
    ) -> None:
        """
        The feature is invoked if *inspect_tensor_enabled* returns `True`. It can be used to obtain information on the high precision tensor. For example, it is run by the `LogTensorStats` feature.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
        tensor: torch.Tensor
            tensor in high precision,
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.
        tp_group: torch.distributed.ProcessGroup
            process group for the tensor parallel group. This is used for weight statistics reduction.
            This is not reduction group from debug_api.

        Returns
        -------

        Should return nothing.
        """

    def inspect_tensor_postquantize(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        gemm: str,
        tensor: torch.Tensor,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
    ) -> None:
        """
        Similar to *inspect_tensor*, but is run after one of the: fp8 cast, modify_tensor if they are run. If none of the fp8 cast or modify_tensor is invoked, then *inspect_tensor_postquantize* is also not invoked. The feature LogFp8Stats uses this call to collect FP8 statistics after the quantization.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
        tensor: torch.Tensor
            tensor in fp8 or processed tensor after the modify_tensor call,
        gemm: str
            one of [`fprop`, `dgrad`, `wgrad`],
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.
        tp_group: torch.distributed.ProcessGroup
            process group for the tensor parallel group. This is used for weight statistics reduction.
            This is not reduction group from debug_api.

        Returns
        -------

        Should return nothing.
        """

    def inspect_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
    ) -> bool:
        """
        It is a routing call, which is run at the initialization of the layer. If it returns true, then *inspect_tensor* for a given GEMM and tensor will be invoked.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`].
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.

        Returns
        -------

        bool - default is False
        """
        return False

    def inspect_tensor_postquantize_enabled(
        self,
        config: Dict,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        iteration: int,
    ) -> bool:
        """
        It is a routing call, which is run at the initialization of the layer.
        If it returns true, then *inspect_tensor_postquantize* for
        a given GEMM and tensor will be invoked.

        Parameters
        ----------

        config: Dict
            dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
        layer_name: str
        gemm: str
            one of [`fprop`, `dgrad`, `wgrad`],
        tensor_name: str
            one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
        iteration: int
            iteration number - equal to the number of times `debug_api.step()` was called.

        Returns
        -------

        bool - default is False
        """
        return False


@Registry.register_namespace_api(namespace="transformer_engine")
class TransformerEngineAPI(BaseNamespaceAPI):
    """
    Transformer Engine API class that contains default APIs that are invoked when a config is not provided
    or a layer is not selected in the config.
    TransformerEngine specific features must override these APIs wherever required.
    The overridden APIs will be invoked whenever the corresponding feature is enabled in the config.
    """

    def __init__(self):
        BaseNamespaceAPI.__init__(self)
        self._default_api_impl = TEDefaultFeatures()
        self._cacheable_api_kwargs_map = {
            "fp8_gemm": ["gemm"],
            "modify_tensor": ["tensor_name", "gemm"],
            "inspect_tensor": ["tensor_name"],
            "inspect_tensor_postquantize": ["tensor_name"],
            "inspect_tensor_enabled": ["tensor_name"],
            "inspect_tensor_postquantize_enabled": ["tensor_name"],
            "modify_tensor_enabled": ["tensor_name"],
        }

    def is_multiple_feature_invocation_allowed(self, api_name):
        """
        Check if API allows executing multiple features for a single call
        """
        return api_name in {
            "fp8_gemm_enabled",
            "inspect_tensor",
            "inspect_tensor_postquantize",
            "inspect_tensor_enabled",
            "inspect_tensor_postquantize_enabled",
        }

    def input_assertions_hook(self, api_name, **kwargs):
        """
        These args must be passed as kwargs in the API call for all TransformerEngine specific APIs.
        """

        if api_name in required_kwargs:
            for kwarg in required_kwargs[api_name]:
                assert kwarg in kwargs, (
                    f"[NVTORCH INSPECT ERROR] Cannot route API, too ambiguous. Provide {kwarg} in"
                    f" {api_name}."
                )
        else:
            for kwarg in required_kwargs["default"]:
                assert kwarg in kwargs, (
                    f"[NVTORCH INSPECT ERROR] Cannot route API, too ambiguous. Provide {kwarg} in"
                    f" {api_name}."
                )

    def routing_condition(self, api_name, config, _, feature_obj, **kwargs):
        """
        Overridden APIs are selected based on the GEMM name in the config and kwargs.
        """
        tensor_parsing = "tensor_name" in required_kwargs[api_name]
        gemm_parsing = "gemm" in required_kwargs[api_name]
        status, modified_config = feature_obj.parse_config_and_api(
            config, gemm_parsing=gemm_parsing, tensor_parsing=tensor_parsing, **kwargs
        )
        return status, modified_config

    def output_assertions_hook(self, api_name, ret, **kwargs):
        """Output hooks used to check correctness of the outputs of the API calls."""
        if "enabled" in api_name or api_name == "fp8_gemm":
            assert isinstance(ret, bool)
        if api_name in ["inspect_tensor", "inspect_tensor_postquantize"]:
            assert ret is None
        if api_name == "modify_tensor":
            assert type(ret) in get_all_tensor_types()
            if (
                type(ret) == torch.Tensor  # pylint: disable=unidiomatic-typecheck
                and "dtype" in kwargs
            ):
                if kwargs["dtype"] is not None:
                    assert ret.dtype == kwargs["dtype"]

    def step(self):
        """This function is called by the nvidia-dlframework-inspect after every debug_api.step()"""
        STATS_BUFFERS.log_stats()

    def end_debug(self):
        """This function is called by the nvidia-dlframework-inspect after every debug_api.end_debug()"""
        TEDebugState._reset()
