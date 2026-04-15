# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""AutoswitchGemm Feature support for nvidia-dlframework-inspect."""

import copy
import logging
import os
from typing import Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.logging import get_logger
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.debug.features.utils import next_enabled_iter


class _AutoswitchGemmMetricLogger:
    """Writes per-rank autoswitch metrics to a dedicated log file."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if _AutoswitchGemmMetricLogger._initialized:
            return
        self.root_dir = None
        self.log_file = None
        self.logger = None
        _AutoswitchGemmMetricLogger._initialized = True

    @staticmethod
    def _get_rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _expected_paths(self, root_log_dir: str) -> Tuple[str, str]:
        rank = self._get_rank()
        root_dir = os.path.join(root_log_dir, "nvdlfw_inspect_autoswitchgemm_logs")
        log_file = os.path.join(root_dir, f"nvdlfw_inspect_globalrank-{rank}.log")
        return root_dir, log_file

    def initialize(self, root_log_dir: str) -> None:
        """Initialize rank-local logger under autoswitch log directory."""
        root_dir, log_file = self._expected_paths(root_log_dir)
        os.makedirs(root_dir, exist_ok=True)

        rank = self._get_rank()
        logger_name = f"nvdlfw_inspect.autoswitchgemm.rank{rank}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        self.root_dir = root_dir
        self.log_file = log_file
        self.logger = logger

    def ensure_initialized(self, root_log_dir: Optional[str]) -> bool:
        """Ensure logger tracks current debug session's root log dir."""
        if not root_log_dir:
            return False
        expected_root_dir, expected_log_file = self._expected_paths(root_log_dir)
        if (
            self.logger is None
            or self.root_dir != expected_root_dir
            or self.log_file != expected_log_file
            or not os.path.isdir(expected_root_dir)
        ):
            self.initialize(root_log_dir)
        return self.logger is not None

    def log_scalar(
        self,
        layer_name: str,
        gemm: str,
        metric_name: str,
        iteration: int,
        value: float,
    ) -> None:
        """Log metric in LogTensorStats-like `iteration/value` format."""
        if self.logger is None:
            return
        metric_key = f"{layer_name}_{gemm}_{metric_name}"
        self.logger.info(
            f"{metric_key} \t\t\t\t iteration={iteration:06d} \t\t\t\t value={value:.8f}"
        )


def _get_autoswitch_metric_logger() -> _AutoswitchGemmMetricLogger:
    """Get singleton autoswitch metric logger."""
    return _AutoswitchGemmMetricLogger()


class _GemmSwitchState:
    """Autoswitch state tracked independently for each (layer, gemm)."""

    def __init__(self):
        self.disable_until_iter = -1
        self.last_applied_metric_snapshot = None
        self.last_reason = ""


@Registry.register_feature(namespace="transformer_engine")
class AutoswitchGemm(TEConfigAPIMapper):
    """
    Dynamically switches selected GEMMs between quantized and high-precision execution.

    The feature continuously monitors quantization quality for selected tensors and,
    when quality degrades beyond configured thresholds, temporarily disables quantized
    GEMM for the affected operation.

    The decision is made per `(layer_name, gemm)`:

    - `fp8_gemm_enabled(..., gemm="fprop")` controls FPROP GEMM
    - `fp8_gemm_enabled(..., gemm="dgrad")` controls DGRAD GEMM
    - `fp8_gemm_enabled(..., gemm="wgrad")` controls WGRAD GEMM

    The API name `fp8_gemm_enabled` is kept for backward compatibility with the
    debug API; the switch applies to all quantized formats supported by TE.
    When multiple tensors are monitored for a GEMM, their metrics are aggregated
    with OR semantics: if any monitored tensor breaches thresholds, the GEMM
    switches to high precision.

    Parameters
    ----------

    gemms / gemms_struct: List[str]
        GEMMs to control:

            - fprop
            - dgrad
            - wgrad

    tensors / tensors_struct: Optional[List[str]]
        Tensors to monitor:

            - activation
            - weight
            - gradient

        If omitted, tensors are inferred from selected GEMMs:

            - fprop -> activation, weight
            - dgrad -> gradient, weight
            - wgrad -> activation, gradient

    underflow_threshold_pct: float, default = 5.0
        Trigger switch to high precision if underflow percentage exceeds this value.

    mse_threshold: float, default = 1e-4
        Trigger switch to high precision if quantization MSE exceeds this value.

    The switch decision is same-iteration:
    metrics computed at iteration `n` are consumed in iteration `n`
    after all GEMM input tensors are prepared.
    The switch is applied for one iteration.

    allow_fp8_model_params_dequantized_weight: bool, default = False
        If True, allows `fprop`/`dgrad` to switch to high precision even when
        fp8 model parameters are enabled by using a temporary dequantized weight
        tensor for GEMM execution.
        If False, `fprop`/`dgrad` stay quantized for such layers.

    freq/start_step/end_step/start_end_list: Optional
        Sampling controls for tensor inspection calls.

    Example
    -------
    .. code-block:: yaml

        example_autoswitch_gemm:
            enabled: True
            layers:
                layer_types: [qkv]
            transformer_engine:
                AutoswitchGemm:
                    enabled: True
                    gemms: [fprop, dgrad, wgrad]
                    underflow_threshold_pct: 3.0
                    mse_threshold: 1e-4
                    # decision is computed and consumed in the same iteration
    """

    _GEMM_TO_TENSORS = {
        "fprop": {"activation", "weight"},
        "dgrad": {"gradient", "weight"},
        "wgrad": {"activation", "gradient"},
    }

    # Mirrors DebugQuantizer's internal mapping.
    _TENSOR_TO_GEMMS = {
        "weight": ("fprop", "dgrad"),
        "activation": ("fprop", "wgrad"),
        "gradient": ("dgrad", "wgrad"),
        "output": ("fprop", None),
        "wgrad": ("wgrad", None),
        "dgrad": ("dgrad", None),
    }

    _DEFAULT_UNDERFLOW_THRESHOLD_PCT = 5.0
    _DEFAULT_MSE_THRESHOLD = 1e-4
    _DEFAULT_ALLOW_FP8_MODEL_PARAMS_DEQUANTIZED_WEIGHT = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gemm_state: Dict[Tuple[str, str], _GemmSwitchState] = {}
        self._latest_metrics: Dict[Tuple[str, str], Dict[str, float | int | str]] = {}
        self._layer_has_fp8_model_params: Dict[str, bool] = {}

    def parse_config_and_api(self, config, **kwargs):
        """
        Parse config for GEMM-routing and tensor-inspection APIs.

        Unlike the default TEConfigAPIMapper behavior, this implementation supports
        tensor inspection even when `tensors` is omitted by inferring monitored
        tensors from selected GEMMs.
        """
        processed_config = None
        config_copy = copy.deepcopy(config)

        gemm = kwargs.get("gemm", None)
        tensor_name = kwargs.get("tensor_name", None)

        if gemm is not None and tensor_name is None:
            processed_config = self._process_transformer_engine_config(config_copy, **kwargs)
        elif tensor_name is not None:
            if "tensors" in config_copy or "tensors_struct" in config_copy:
                processed_config = self._process_tensor_config(config_copy, tensor_name)
            else:
                monitored_tensors = self._infer_monitored_tensors(config_copy)
                if tensor_name not in monitored_tensors:
                    return False, None
                processed_config = config_copy
                processed_config["tensor"] = tensor_name

        if not processed_config:
            return False, None

        if "enabled" in processed_config:
            processed_config.pop("enabled")

        return True, processed_config

    def _infer_monitored_tensors(self, config: Dict) -> Set[str]:
        """Infer tensors to inspect from configured GEMMs."""
        configured_gemms = self._extract_configured_gemms(config)
        if not configured_gemms:
            configured_gemms = set(self._GEMM_TO_TENSORS.keys())

        tensors = set()
        for gemm in configured_gemms:
            self._validate_gemm(gemm)
            tensors.update(self._GEMM_TO_TENSORS[gemm])
        return tensors

    @staticmethod
    def _extract_configured_gemms(config: Dict) -> Set[str]:
        """Extract GEMM names from config keys `gemm`, `gemms`, and `gemms_struct`."""
        gemms = set()
        if "gemm" in config:
            gemms.add(config["gemm"])
        if "gemms" in config:
            gemms.update(config["gemms"])
        if "gemms_struct" in config:
            for cfg in config["gemms_struct"]:
                if "gemm" in cfg:
                    gemms.add(cfg["gemm"])
        return gemms

    @staticmethod
    def _config_float(config: Dict, key: str, default: Optional[float]) -> Optional[float]:
        """Read optional float value from config."""
        value = config.get(key, default)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _config_bool(config: Dict, key: str, default: bool) -> bool:
        """Read bool value from config."""
        value = config.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _get_root_log_dir() -> Optional[str]:
        """Best-effort retrieval of nvdlfw_inspect root log directory."""
        try:
            root_log_dir = getattr(get_logger(), "root_log_dir", None)
        except Exception:  # pylint: disable=broad-except
            return None
        return root_log_dir

    def _get_metrics_logger(self) -> Optional[_AutoswitchGemmMetricLogger]:
        """Return initialized autoswitch metric logger if log dir is available."""
        metric_logger = _get_autoswitch_metric_logger()
        if metric_logger.ensure_initialized(self._get_root_log_dir()):
            return metric_logger
        return None

    def _get_or_create_state(self, layer_name: str, gemm: str) -> _GemmSwitchState:
        key = (layer_name, gemm)
        if key not in self._gemm_state:
            self._gemm_state[key] = _GemmSwitchState()
        return self._gemm_state[key]

    def _update_metric(
        self,
        layer_name: str,
        gemm: str,
        iteration: int,
        tensor_name: str,
        underflow_pct: float,
        mse: float,
    ) -> None:
        """Store the latest quality metric for a `(layer, gemm)` pair."""
        metric_logger = self._get_metrics_logger()
        if metric_logger is not None:
            metric_logger.log_scalar(
                layer_name, gemm, f"{tensor_name}_underflow_pct", iteration, underflow_pct
            )
            metric_logger.log_scalar(layer_name, gemm, f"{tensor_name}_mse", iteration, mse)

        key = (layer_name, gemm)
        entry = self._latest_metrics.get(key)

        if entry is None or int(entry["iteration"]) < iteration:
            self._latest_metrics[key] = {
                "iteration": iteration,
                "underflow_pct": underflow_pct,
                "mse": mse,
                "tensor_name": tensor_name,
            }
            return

        if int(entry["iteration"]) == iteration:
            if underflow_pct >= float(entry["underflow_pct"]):
                entry["underflow_pct"] = underflow_pct
                entry["tensor_name"] = tensor_name
            entry["mse"] = max(float(entry["mse"]), mse)

    @staticmethod
    def _dequantize_like(
        quantized_tensor,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[torch.Tensor]:
        """Best-effort dequantization helper used for quality metrics."""
        if quantized_tensor is None or not hasattr(quantized_tensor, "dequantize"):
            return None

        try:
            dequantized = quantized_tensor.dequantize(dtype=dtype)
        except TypeError:
            dequantized = quantized_tensor.dequantize()
            if dequantized.dtype != dtype:
                dequantized = dequantized.to(dtype)

        if dequantized.shape != shape:
            expected_numel = 1
            for dim in shape:
                expected_numel *= int(dim)
            if dequantized.numel() != expected_numel:
                return None
            dequantized = dequantized.reshape(shape)
        return dequantized

    @staticmethod
    def _compute_metrics(
        tensor: Optional[torch.Tensor],
        quantized_tensor,
    ) -> Optional[Tuple[float, float]]:
        """Compute underflow percentage and MSE for one tensor."""
        if tensor is None or tensor.numel() == 0:
            return None

        if not tensor.is_floating_point():
            return None

        dequantized = AutoswitchGemm._dequantize_like(quantized_tensor, tensor.dtype, tensor.shape)
        if dequantized is None:
            return None

        tensor_fp32 = tensor.float()
        dequantized_fp32 = dequantized.float()

        underflow_count = torch.count_nonzero((tensor_fp32 != 0) & (dequantized_fp32 == 0))
        underflow_pct = (underflow_count.float() * 100.0 / tensor_fp32.numel()).item()

        mse = torch.mean((tensor_fp32 - dequantized_fp32) ** 2).item()
        return underflow_pct, mse

    def _consume_new_metric_and_maybe_arm_switch(
        self,
        layer_name: str,
        gemm: str,
        iteration: int,
        config: Dict,
        state: _GemmSwitchState,
    ) -> None:
        """Consume current-iteration metrics and arm switch for one iteration."""
        metric = self._latest_metrics.get((layer_name, gemm))
        if metric is None:
            return

        metric_iter = int(metric["iteration"])
        if metric_iter != iteration:
            # Autoswitch consumes metrics only in the iteration they were produced.
            return

        metric_snapshot = (
            metric_iter,
            float(metric["underflow_pct"]),
            float(metric["mse"]),
            str(metric["tensor_name"]),
        )
        if metric_snapshot == state.last_applied_metric_snapshot:
            return
        state.last_applied_metric_snapshot = metric_snapshot

        underflow_threshold = self._config_float(
            config, "underflow_threshold_pct", self._DEFAULT_UNDERFLOW_THRESHOLD_PCT
        )
        mse_threshold = self._config_float(config, "mse_threshold", self._DEFAULT_MSE_THRESHOLD)

        reasons = []
        metric_underflow = float(metric["underflow_pct"])
        metric_mse = float(metric["mse"])

        if underflow_threshold is not None and metric_underflow > underflow_threshold:
            reasons.append(
                f"underflow={metric_underflow:.4f}% > threshold={underflow_threshold:.4f}%"
            )
        if mse_threshold is not None and metric_mse > mse_threshold:
            reasons.append(f"mse={metric_mse:.6e} > threshold={mse_threshold:.6e}")

        if not reasons:
            return

        state.disable_until_iter = iteration
        state.last_reason = "; ".join(reasons)

        debug_api.log_message(
            f"Feature={self.__class__.__name__}: switch {gemm} to high precision in"
            f" iter={iteration}. Triggered by {metric['tensor_name']} sampled at iter={metric_iter}:"
            f" {state.last_reason}",
            layer_name,
            extra_cachable_args=(gemm, "switch"),
        )

    @api_method
    def fp8_gemm_enabled(
        self,
        config,
        layer_name: str,
        gemm: str,
        iteration: int,
        final_decision: bool = False,
    ):
        """Decide whether selected GEMM should run quantized (True) or high precision (False)."""
        state = self._get_or_create_state(layer_name, gemm)
        metric_logger = self._get_metrics_logger()

        fp8_model_params_layer = self._layer_has_fp8_model_params.get(layer_name, False)
        allow_fp8_model_params_fallback = self._config_bool(
            config,
            "allow_fp8_model_params_dequantized_weight",
            self._DEFAULT_ALLOW_FP8_MODEL_PARAMS_DEQUANTIZED_WEIGHT,
        )

        # With fp8 model parameters enabled, fprop/dgrad can switch to high precision
        # only when dequantized fallback is explicitly enabled in config.
        if (
            gemm in {"fprop", "dgrad"}
            and fp8_model_params_layer
            and not allow_fp8_model_params_fallback
        ):
            state.disable_until_iter = -1
            if final_decision and metric_logger is not None:
                metric_logger.log_scalar(layer_name, gemm, "quantized_enabled", iteration, 1.0)
                metric_logger.log_scalar(
                    layer_name, gemm, "switch_blocked_fp8_model_params", iteration, 1.0
                )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: skip switch for {gemm} at"
                f" iter={iteration} because fp8 model parameters are enabled.",
                layer_name,
                extra_cachable_args=(gemm, "skip_fp8_model_params"),
            )
            return True, iteration + 1

        if gemm in {"fprop", "dgrad"} and fp8_model_params_layer and allow_fp8_model_params_fallback:
            if final_decision and metric_logger is not None:
                metric_logger.log_scalar(
                    layer_name, gemm, "fp8_model_params_dequantized_fallback", iteration, 1.0
                )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: {gemm} allows fp8-model-params"
                " dequantized-weight fallback.",
                layer_name,
                extra_cachable_args=(gemm, "fp8_model_params_dequantized_fallback"),
            )

        self._consume_new_metric_and_maybe_arm_switch(layer_name, gemm, iteration, config, state)

        if iteration <= state.disable_until_iter:
            if final_decision and metric_logger is not None:
                metric_logger.log_scalar(layer_name, gemm, "quantized_enabled", iteration, 0.0)
                metric_logger.log_scalar(
                    layer_name,
                    gemm,
                    "disable_until_iter",
                    iteration,
                    float(state.disable_until_iter),
                )
            debug_api.log_message(
                f"Feature={self.__class__.__name__}: {gemm} forced high precision at"
                f" iter={iteration} (disable_until={state.disable_until_iter}).",
                layer_name,
                extra_cachable_args=(gemm, "high_precision"),
            )
            return False, iteration + 1

        if final_decision and metric_logger is not None:
            metric_logger.log_scalar(layer_name, gemm, "quantized_enabled", iteration, 1.0)
        return True, iteration + 1

    @api_method
    def inspect_tensor_enabled(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
    ):  # pylint: disable=unused-argument
        """Enable metric collection according to the standard freq/start/end controls."""
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
        tp_group: torch.distributed.ProcessGroup,  # pylint: disable=unused-argument
        tensor: Optional[torch.Tensor],
        rowwise_quantized_tensor: Optional[torch.Tensor] = None,
        columnwise_quantized_tensor: Optional[torch.Tensor] = None,
        quantizer=None,  # pylint: disable=unused-argument
        tp_size: int = 1,  # pylint: disable=unused-argument
    ):
        """Collect quantization quality metrics for autoswitch decisions."""
        if tensor_name == "weight" and tensor is None:
            # Weight tensor unavailable in high precision indicates fp8 model params.
            self._layer_has_fp8_model_params[layer_name] = True

        _ = config
        gemms = self._TENSOR_TO_GEMMS.get(tensor_name, (None, None))

        rowwise_gemm, columnwise_gemm = gemms
        if rowwise_gemm is not None:
            metrics = self._compute_metrics(tensor, rowwise_quantized_tensor)
            if metrics is not None:
                self._update_metric(
                    layer_name, rowwise_gemm, iteration, tensor_name, metrics[0], metrics[1]
                )

        if columnwise_gemm is not None:
            metrics = self._compute_metrics(tensor, columnwise_quantized_tensor)
            if metrics is not None:
                self._update_metric(
                    layer_name, columnwise_gemm, iteration, tensor_name, metrics[0], metrics[1]
                )
