# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""API definition for nvidia-dlframework-inspect."""

import copy

from nvdlfw_inspect.base import BaseNamespaceAPI, BaseConfigAPIMapper
from nvdlfw_inspect.registry import Registry

import torch

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import all_tensor_types
from transformer_engine.debug.pytorch.debug_state import TEDebugState


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


class TEDefaultFeatures:
    """Transformer Engine API calls default behaviour."""

    def fp8_gemm_enabled(self, *_args, **_kwargs):
        """API call responsible for choice between high-precision and FP8 GEMM execution."""
        return True  # if it is false, fp8_gemm will be turned off. Otherwise nothing happens.

    def modify_tensor_enabled(self, *_args, **_kwargs):
        """API call used to determine whether to run modify_tensor() in the forward."""
        return False

    def modify_tensor(self, *_args, **_kwargs):
        """API call used to process the tensor."""
        raise NotImplementedError(
            "modify_tensor_enabled() returned True, modify_tensor() was invoked, but it is not"
            " handled by any API."
        )

    def inspect_tensor(self, *_args, **_kwargs):
        """API call used to collect the data about the tensor before modify_tensor()/quantization."""

    def inspect_tensor_postquantize(self, *_args, **_kwargs):
        """API call used to collect the data about the tensor after modify_tensor()/quantization."""

    def inspect_tensor_enabled(self, *_args, **_kwargs):
        """API call used to determine whether to run look_at_tensor_before_process() in the forward."""
        return False

    def inspect_tensor_postquantize_enabled(self, *_args, **_kwargs):
        """API call used to determine whether to run look_at_tensor_after_process() in the forward."""
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
        Check if API allowes executing multiple features for a single call
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
        """Output hooks using to check correctness of the outputs of the API calls."""
        if "enabled" in api_name or api_name == "fp8_gemm":
            assert isinstance(ret, bool)
        if api_name in ["inspect_tensor", "inspect_tensor_postquantize"]:
            assert ret is None
        if api_name == "modity_tensor":
            assert type(ret) in all_tensor_types
            if type(ret) == torch.Tensor:  # pylint: disable=unidiomatic-typecheck
                assert ret.dtype == kwargs["dtype"]

    def step(self):
        """This function is called by the nvidia-dlframework-inspect after every debug_api.step()"""
        STATS_BUFFERS.log_stats()

    def end_debug(self):
        """This function is called by the nvidia-dlframework-inspect after every debug_api.end_debug()"""
        TEDebugState.reset()
