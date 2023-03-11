"""Top level Transformer Engine PyTorch modules"""
from keras import backend, layers, initializers
from keras.mixed_precision import autocast_variable
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from typing import Union, Callable

import tensorflow as tf
import transformer_engine_tensorflow as tex

from .constants import TE_DType
from .fp8 import (
    is_fp8_enabled,
    get_fp8_recipe,
    get_default_fp8_recipe,
    get_fp8_te_dtype,
    is_first_fp8_module,
    new_fp8_context_id,
    get_fp8_context_id,
    set_fp8_context_id,
    amax_and_scale_update,
    set_amax_buffer_key_deletion,
    get_meta_tensor_key,
)
from .jit import (
    bgrad_dgelu_fused,
)

stream_lib = load_library.load_op_library(
    resource_loader.get_path_to_datafile(
        tf.sysconfig.get_lib() + "/../lib_get_stream.so"
    )
)

def get_stream_id():
    """Get stream index for GPU tasks."""
    return stream_lib.get_stream().numpy()[0]

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_cublas_workspace = None


def get_workspace():
    """Returns workspace for cublas."""
    global _cublas_workspace
    if _cublas_workspace is None:
        _cublas_workspace = tf.zeros([33_554_432], dtype=tf.int8)
    return _cublas_workspace


def get_autocast_bias(dtype, bias_var, use_bias):
    """Get casted bias for fp8 gemm."""
    if not use_bias:
        return None
    with autocast_variable.enable_auto_cast_variables(dtype):
        bias = bias_var.value()
    if bias.dtype == tf.float32:
        bias = tf.cast(bias, dtype=tf.bfloat16)
    return bias


def get_init_method(user_input, default_init_method):
    """Get initializer method for variables."""
    if user_input is None:
        return default_init_method

    if callable(user_input):
        return user_input

    assert isinstance(user_input, str)
    return initializers.get(user_input)


def cast_to_fp8_wrapper(x, fp8_meta, amax_index, fwd, output_dtype, stream_id):
    """Wrapper to call the tex.cast_to_fp8."""
    scaling_key = get_meta_tensor_key(fwd)
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    x_fp8 = tex.cast_to_fp8(
        x, scale, output_dtype, amax, scale_inv, amax_index, stream_id
    )
    return x_fp8


def cast_from_fp8_wrapper(x, fp8_meta, amax_index, fwd, idtype, odtype, sid):
    """Wrapper to call the tex.cast_from_fp8."""
    scaling_key = "scaling_fwd" if fwd else "scaling_bwd"
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    x_fp8 = tex.cast_from_fp8(x, scale_inv, idtype, odtype, amax_index, sid)
    return x_fp8


def fp8_cast_transpose_fused_wrapper(x, fp8_meta, amax_index, fwd, output_dtype,
                                     sid):
    """Wrapper to call the tex.fp8_cast_transpose_fused."""
    scaling_key = get_meta_tensor_key(fwd)
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    x_fp8, x_t_fp8 = tex.fp8_cast_transpose_fused(
        x, scale, output_dtype, amax, scale_inv, amax_index, sid
    )
    return x_fp8, x_t_fp8


def fp8_cast_transpose_bgrad_fused_wrapper(
    x, fp8_meta, amax_index, fwd, output_dtype, sid
):
    """Wrapper to call the tex.fp8_cast_transpose_bgrad_fused."""
    scaling_key = get_meta_tensor_key(fwd)
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    grad_bias, grad_fp8, grad_t_fp8 = tex.fp8_cast_transpose_bgrad_fused(
        x, scale, output_dtype, amax, scale_inv, amax_index, sid
    )
    return grad_bias, grad_fp8, grad_t_fp8


def fp8_cast_transpose_bgrad_dgelu_fused_wrapper(
    dy, x, fp8_meta, amax_index, fwd, output_dtype, sid
):
    """Wrapper to call the tex.fp8_fused_cast_transpose_bgrad_dgelu."""
    scaling_key = get_meta_tensor_key(fwd)
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    dbias, dgelu_c, dgelu_t = tex.fp8_fused_cast_transpose_bgrad_dgelu(
        dy, x, scale, output_dtype, amax, scale_inv, amax_index, sid
    )
    return dbias, dgelu_c, dgelu_t


def fp8_gelu_wrapper(x, fp8_meta, amax_index, fwd, output_dtype, sid):
    """Wrapper to call the tex.te_gelu."""
    scaling_key = get_meta_tensor_key(fwd)
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    y_fp8 = tex.te_gelu(x, scale, output_dtype, amax, scale_inv, amax_index,
                        sid)
    return y_fp8


def matmul_wrapper(
    inp,
    weight,
    mode,
    output_dtype,
    sid,
    use_bias=False,
    bias=None,
    grad=False,
    gelu=False,
    gelu_input=None,
):
    """Wrapper to call the tex.te_gemm for the non-fp8 gemm."""
    A = inp
    B = weight
    A_dtype, B_dtype = TE_DType[A.dtype], TE_DType[B.dtype]
    A_offset, B_offset = -1, -1
    if mode in ("fwd", "fc1_fwd", "fc2_fwd"):
        transA, transB = False, False
    elif mode in ("bwd_input", "fc1_bwd_input", "fc2_bwd_input"):
        transA, transB = False, True
    elif mode in ("bwd_weight", "fc1_bwd_weight", "fc2_bwd_weight"):
        transA, transB = True, False

    return tex.te_gemm(
        B,
        None,
        B_dtype,
        B_offset,
        A,
        None,
        A_dtype,
        A_offset,
        get_workspace(),
        use_bias,
        bias,
        gelu,
        gelu_input,
        transB,
        transA,
        grad,
        False,  # accumulate
        False,  # accumulate
        TE_DType[output_dtype],
        sid,
    )


def fp8_matmul_wrapper(
    inp,
    weight,
    fp8_meta,
    mode,
    A_dtype,
    B_dtype,
    output_dtype,
    use_split_accumulate,
    sid,
    use_bias=False,
    bias=None,
):
    """Wrapper to call the tex.te_gemm for the fp8 gemm."""
    A = inp
    B = weight
    if mode in ("fwd", "fc1_fwd"):
        A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        A_offset = tex.FP8FwdTensors.GEMM1_INPUT
        B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        B_offset = tex.FP8FwdTensors.GEMM1_WEIGHT
    elif mode == "fc2_fwd":
        A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        A_offset = tex.FP8FwdTensors.GEMM2_INPUT
        B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        B_offset = tex.FP8FwdTensors.GEMM2_WEIGHT
    elif mode == "bwd_input":
        A_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        A_offset = tex.FP8BwdTensors.GRAD_OUTPUT1
        B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        B_offset = tex.FP8FwdTensors.GEMM1_WEIGHT
    elif mode == "fc1_bwd_input":
        A_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        A_offset = tex.FP8BwdTensors.GRAD_OUTPUT2
        B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        B_offset = tex.FP8FwdTensors.GEMM1_WEIGHT
    elif mode == "fc2_bwd_input":
        A_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        A_offset = tex.FP8BwdTensors.GRAD_OUTPUT1
        B_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        B_offset = tex.FP8FwdTensors.GEMM2_WEIGHT
    elif mode == "bwd_weight":
        A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        A_offset = tex.FP8FwdTensors.GEMM1_INPUT
        B_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        B_offset = tex.FP8BwdTensors.GRAD_OUTPUT1
    elif mode == "fc2_bwd_weight":
        A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        A_offset = tex.FP8FwdTensors.GEMM2_INPUT
        B_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        B_offset = tex.FP8BwdTensors.GRAD_OUTPUT1
    elif mode == "fc1_bwd_weight":
        A_scale_inv = fp8_meta["scaling_fwd"]["scale_inv"].value()
        A_offset = tex.FP8FwdTensors.GEMM1_INPUT
        B_scale_inv = fp8_meta["scaling_bwd"]["scale_inv"].value()
        B_offset = tex.FP8BwdTensors.GRAD_OUTPUT2

    return tex.te_gemm(
        B,
        B_scale_inv,
        B_dtype,
        B_offset,
        A,
        A_scale_inv,
        A_dtype,
        A_offset,
        get_workspace(),
        use_bias,
        bias,
        False,  # use_gelu
        None,  # gelu_input
        True,  # transa
        False,  # transb
        False,  # grad
        False,  # accumulate
        use_split_accumulate,
        TE_DType[output_dtype],
        sid,
    )


def layernorm_fwd_fp8_wrapper(
    x, ln_gamma, ln_beta, epsilon, fp8_meta, amax_index, output_dtype, sid
):
    """Wrapper to call the tex.layernorm_fwd_fp8."""
    scaling_key = "scaling_fwd"
    scale = fp8_meta[scaling_key]["scale"].value()
    amax = fp8_meta[scaling_key]["amax_history"].value()
    scale_inv = fp8_meta[scaling_key]["scale_inv"].value()
    ln_out, mu, rsigma = tex.layernorm_fwd_fp8(
        x,
        ln_gamma,
        ln_beta,
        epsilon,
        scale,
        output_dtype,
        amax,
        scale_inv,
        amax_index,
        sid,
    )
    return ln_out, mu, rsigma


# The DelayedScaling object is not supported in TF autograd. So, to avoid
# passing this object to the custom gradient function, we only extract the
# useful information.
def get_recipe_attrs(recipe):
    """Get attributes from the recipe."""
    fp8_dtype_fwd = get_fp8_te_dtype(recipe, fprop_tensor=True)
    fp8_dtype_bwd = get_fp8_te_dtype(recipe, fprop_tensor=False)
    override_linear_precision = recipe.override_linear_precision
    return (fp8_dtype_fwd, fp8_dtype_bwd, override_linear_precision)


# TransformerEngineBaseModule is a mixin class and its init function will pass
# through all the positional and keyword arguments to other subclasses. Make
# sure this class is inherited first.
class TransformerEngineBaseModule:
    """Base TE module."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # fp8 related
        self.fp8 = False
        self.fp8_meta = {}
        self.fp8_meta["recipe"] = get_default_fp8_recipe()
        self.fp8_meta_tensors_initialized = False
        self.fp8_weight_shapes = []
        self.stream_id = get_stream_id()

    def set_meta_tensor(self, fwd):
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key = "scaling_fwd" if fwd else "scaling_bwd"
        num_fp8_tensors = (
            self.fp8_meta["num_gemms"] * 2 if fwd else
            self.fp8_meta["num_gemms"]
        )

        self.fp8_meta[fp8_meta_tensor_key] = {}
        self.fp8_meta[fp8_meta_tensor_key]["scale"] = tf.Variable(
            tf.ones((num_fp8_tensors), dtype=tf.float32), trainable=False
        )
        self.fp8_meta[fp8_meta_tensor_key]["scale_inv"] = tf.Variable(
            tf.ones((num_fp8_tensors), dtype=tf.float32), trainable=False
        )
        self.fp8_meta[fp8_meta_tensor_key]["amax_history"] = tf.Variable(
            tf.zeros(
                (self.fp8_meta["recipe"].amax_history_len, num_fp8_tensors),
                dtype=tf.float32,
            ),
            trainable=False,
        )

    def init_fp8_meta_tensors(self):
        """Init scales and amaxes."""
        # Checkpoint loaded
        if self.fp8_meta_tensors_initialized:
            return

        self.set_meta_tensor(True)
        self.set_meta_tensor(False)

    def fp8_init(self, num_gemms=1):
        """Initialize fp8 related metadata and tensors during fprop."""
        if not is_fp8_enabled():
            self.fp8 = False
            return

        # FP8 is already enabled and recipe is the same, don't do anything.
        if self.fp8 and get_fp8_recipe() == self.fp8_meta["recipe"]:
            return

        # Set FP8, recipe, and other FP8 metadata
        self.fp8 = True
        self.fp8_meta["recipe"] = get_fp8_recipe()
        self.fp8_meta["num_gemms"] = num_gemms

        # Set FP8_MAX per tensor according to recipe
        fp8_format_val = self.fp8_meta["recipe"].fp8_format.value
        self.fp8_meta["fp8_max_fwd"] = fp8_format_val.max_fwd
        self.fp8_meta["fp8_max_bwd"] = fp8_format_val.max_bwd

        # Allocate scales and amaxes
        self.init_fp8_meta_tensors()

    def pre_forward(self, training, num_gemms=1):
        """Checks and prep for FWD."""
        self.fp8_init(num_gemms=num_gemms)

        if self.fp8:
            if self.fp8_meta.get("update_amax_and_scale_fwd", False):
                # Previous iteration was grad_enabled
                amax_and_scale_update(self.fp8_meta, True)
                set_amax_buffer_key_deletion(self.fp8_meta, forward=True)

            if training:
                self.fp8_meta["first_module"] = is_first_fp8_module()

                if self.fp8_meta["first_module"]:
                    self.fp8_meta["autocast_id_fwd"] = new_fp8_context_id()
                    set_fp8_context_id(self.fp8_meta["autocast_id_fwd"])
                else:
                    self.fp8_meta["autocast_id_fwd"] = get_fp8_context_id()

                self.fp8_meta["update_amax_and_scale_fwd"] = True

                # Create an empty tensor as a placeholder for the backprop to
                # correctly know how many tensors to autograd.
                self.fp8_meta["autocast_id_bwd"] = -1
            else:
                self.fp8_meta["update_amax_and_scale_fwd"] = False

    def pre_backward(self):
        """Checks and prep for BWD."""
        # From previous iteration
        amax_and_scale_update(self.fp8_meta, False)
        set_amax_buffer_key_deletion(self.fp8_meta, forward=False)


class Dense(TransformerEngineBaseModule, layers.Layer):
    """
    Applies a linear transformation to the incoming data :math:`y = xW + b`

    On NVIDIA GPUs it is a drop-in replacement for `tf.keras.layers.Dense`.

    Parameters
    ----------
    units : int
      size of each output sample.
    use_bias : bool, default = `True`
      if set to `False`, the layer will not learn an additive bias.
    kernel_initializer: Callable, default = `None`
      used for initializing weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    bias_initializer: Callable, default = `None`
      used for initializing biases in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.

    Parallelism parameters
    ----------------------
    skip_weight_param_allocation: bool, default = `False`
      if set to `True`, weight parameter is not allocated and must be passed as
      a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units: int,
        use_bias: bool = True,
        return_bias: bool = False,
        kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        skip_weight_param_allocation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.units = units
        self.use_bias = use_bias
        self.return_bias = return_bias
        self.kernel_initializer = get_init_method(
            kernel_initializer, initializers.RandomNormal(mean=0.0,
                                                          stddev=0.023)
        )
        self.bias_initializer = get_init_method(
            bias_initializer, initializers.get("zeros")
        )
        self.skip_weight_param_allocation = skip_weight_param_allocation

    def build(self, input_shape):
        """One-time allocation of the variables."""
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer should be "
                f"defined. Found None. Full input shape received: {input_shape}"
            )

        self.kernel = None
        self.bias = None
        if not self.skip_weight_param_allocation:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(last_dim, self.units),
                initializer=self.kernel_initializer,
                trainable=True,
            )

            if self.use_bias or self.return_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )

        # fp8 related
        self.fp8_weight_shapes.append((last_dim, self.units))

        self.built = True

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed
            # from model.
            training = False
        return training

    def non_fp8_matmul(
        self,
        inp: tf.Tensor,
        kernel_var: tf.Variable,
        bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd non-fp8 matmul."""
        @tf.custom_gradient
        def non_fp8_matmul_func(x):
            # Use value() to convert from Variable to EagerTensor
            kernel_val = kernel_var.value()
            bias = get_autocast_bias(
                self._compute_dtype_object, bias_var, self.use_bias
            )

            output_dtype = self._compute_dtype_object
            outputs = matmul_wrapper(
                x, kernel_val, "fwd", output_dtype, self.stream_id,
                self.use_bias, bias,
            )

            def grad_fn(upstream, variables=None):
                grad_x = matmul_wrapper(
                    upstream, kernel_val, "bwd_input", output_dtype,
                    self.stream_id,
                )
                grad_weight = matmul_wrapper(
                    x, upstream, "bwd_weight", output_dtype, self.stream_id
                )
                if self.use_bias:
                    grad_bias = tf.math.reduce_sum(upstream, axis=0)

                grad_inputs = [grad_x]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("bias:0") and self.use_bias:
                        grad_vars.append(grad_bias)
                    elif v.name.endswith("kernel:0"):
                        grad_vars.append(grad_weight)
                return grad_inputs, grad_vars

            return outputs, grad_fn

        return non_fp8_matmul_func(inp)

    def fp8_matmul(
        self,
        inp: tf.Tensor,
        kernel_var: tf.Variable,
        bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd fp8 matmul."""
        fp8_meta = self.fp8_meta
        fp8_dtype_fwd, fp8_dtype_bwd, override_linear_precision = \
            get_recipe_attrs(fp8_meta["recipe"])

        @tf.custom_gradient
        def fp8_matmul_func(x):
            # Use value() to convert from Variable to EagerTensor
            kernel_val = kernel_var.value()
            bias = get_autocast_bias(
                self._compute_dtype_object, bias_var, self.use_bias
            )

            if not override_linear_precision.wgrad:
                x_fp8, x_t_fp8 = fp8_cast_transpose_fused_wrapper(
                    x,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    True,
                    fp8_dtype_fwd,
                    self.stream_id,
                )
            else:
                x_fp8 = cast_to_fp8_wrapper(
                    x,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    True,
                    fp8_dtype_fwd,
                    self.stream_id,
                )

            weight_fp8, weight_t_fp8 = fp8_cast_transpose_fused_wrapper(
                kernel_val,
                fp8_meta,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                True,
                fp8_dtype_fwd,
                self.stream_id,
            )

            output_dtype = self._compute_dtype_object
            outputs = fp8_matmul_wrapper(
                x_fp8,
                weight_t_fp8,
                fp8_meta,
                "fwd",
                fp8_dtype_fwd,
                fp8_dtype_fwd,
                output_dtype,
                _2X_ACC_FPROP,
                self.stream_id,
                self.use_bias,
                bias,
            )

            def grad_fn(upstream, variables=None):
                self.pre_backward()
                if self.use_bias:
                    (
                        grad_bias,
                        grad_fp8,
                        grad_t_fp8,
                    ) = fp8_cast_transpose_bgrad_fused_wrapper(
                        upstream,
                        fp8_meta,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        False,
                        fp8_dtype_bwd,
                        self.stream_id,
                    )
                else:
                    if not override_linear_precision.wgrad:
                        grad_fp8, grad_t_fp8 = fp8_cast_transpose_fused_wrapper(
                            upstream,
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )
                    else:
                        grad_fp8 = cast_to_fp8_wrapper(
                            upstream,
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )

                grad_x = fp8_matmul_wrapper(
                    grad_fp8,
                    weight_fp8,
                    fp8_meta,
                    "bwd_input",
                    fp8_dtype_bwd,
                    fp8_dtype_fwd,
                    output_dtype,
                    _2X_ACC_DGRAD,
                    self.stream_id,
                )

                if not override_linear_precision.wgrad:
                    grad_weight = fp8_matmul_wrapper(
                        x_t_fp8,
                        grad_t_fp8,
                        fp8_meta,
                        "bwd_weight",
                        fp8_dtype_fwd,
                        fp8_dtype_bwd,
                        output_dtype,
                        _2X_ACC_WGRAD,
                        self.stream_id,
                    )
                else:
                    grad_weight = matmul_wrapper(
                        x, upstream, "bwd_weight", output_dtype, self.stream_id
                    )

                grad_inputs = [grad_x]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("bias:0") and self.use_bias:
                        grad_vars.append(grad_bias)
                    elif v.name.endswith("kernel:0"):
                        grad_vars.append(grad_weight)

                return grad_inputs, grad_vars

            return outputs, grad_fn

        return fp8_matmul_func(inp)

    def call(
        self,
        inputs,
        kernel=None,
        bias=None,
        training=None,
    ):
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : tf.Tensor
          Input tensor.
        weight : tf.Variable, default = None
          An optional weight tensor for the module. This argument is compulsory
          if module is initialized with `skip_weight_param_allocation=True`
        bias : tf.Variable, default = None
          An optional bias tensor for the module. This argument is compulsory if
          module is initialized with `skip_weight_param_allocation=True` and one
          of `use_bias` or `return_bias`
        training : {True, False, None}, default = None
          Whether this is in the training context.
        """
        # self.pre_forward needs to be called outside the following branch,
        # since it will set the self.fp8 if the autocast is detected.
        training = self._get_training_value(training)
        self.pre_forward(training)

        kernel_var = (kernel if self.skip_weight_param_allocation else
                      self.kernel)
        bias_var = bias if self.skip_weight_param_allocation else self.bias
        if kernel_var is None:
            raise ValueError("No valid kernel is provided")

        inputmat = tf.reshape(inputs, shape=(-1, inputs.shape[-1]))
        if self.fp8:
            outputmat = self.fp8_matmul(inputmat, kernel_var, bias_var)
        else:
            outputmat = self.non_fp8_matmul(inputmat, kernel_var, bias_var)

        outputs = tf.reshape(
            outputmat, shape=(-1, *inputs.shape[1:-1], outputmat.shape[-1])
        )

        if self.return_bias:
            return outputs, bias_var
        return outputs

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer),
                "skip_weight_param_allocation":
                    self.skip_weight_param_allocation,
            }
        )


class LayerNorm(layers.Layer):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    """

    def __init__(
        self, epsilon=1e-3, gamma_initializer="ones", beta_initializer="zeros",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.stream = get_stream_id()

    def build(self, input_shape):
        """One-time allocation of the variables."""
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer should be "
                f"defined. Found None. Full input shape received: {input_shape}"
            )

        self.gamma = self.add_weight(
            name="gamma",
            shape=(last_dim,),
            initializer=self.gamma_initializer,
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(last_dim,),
            initializer=self.beta_initializer,
            trainable=True,
        )

        self.built = True

    @tf.custom_gradient
    def layernorm(self, inp: tf.Tensor):
        """Prep fwd+bwd non-fp8 layernorm."""
        gamma = self.gamma.value()
        ln_out, mu, rsigma = tex.layernorm_fwd(
            inp, gamma, self.beta.value(), self.epsilon, self.stream
        )

        def grad_fn(upstream, variables=None):
            # pylint: disable=unused-argument
            dxmat, dgamma, dbeta = tex.layernorm_bwd(
                upstream, inp, mu, rsigma, gamma, self.stream
            )

            grad_inputs = [tf.reshape(dxmat, inp.shape)]
            grad_vars = [dgamma, dbeta]
            return grad_inputs, grad_vars

        return ln_out, grad_fn

    def call(self, inputs):
        """LayerNorm FWD"""
        inputmat = tf.reshape(inputs, shape=(-1, inputs.shape[-1]))
        outputmat = self.layernorm(inputmat)
        outputs = tf.reshape(outputmat, shape=inputs.shape)
        return outputs

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer),
                "beta_initializer": initializers.serialize(
                    self.beta_initializer),
            }
        )


class LayerNormDense(TransformerEngineBaseModule, layers.Layer):
    """
    Applies layer normalization followed by linear transformation to the
    incoming data.

    Parameters
    ----------
    units : int
      size of each output sample.
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    use_bias : bool, default = `True`
      if set to `False`, the layer will not learn an additive bias.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    kernel_initializer : Callable, default = `None`
      used for initializing GEMM weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    bias_initializer : Callable, default = `None`
      used for initializing GEMM bias in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.
    return_layernorm_output : bool, default = `False`
      if set to `True`, output of layernorm is returned from the forward
      together with the output of the linear transformation.
      Example use case: residual connection for transformer module is taken post
      layernorm.

    Parallelism parameters
    ----------------------
    skip_weight_param_allocation: bool, default = `False`
      if set to `True`, weight parameter is not allocated and must be passed as
      a keyword argument `weight` during the forward pass.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units,
        epsilon=1e-3,
        gamma_initializer: Union[Callable, str, None] = None,
        beta_initializer: Union[Callable, str, None] = None,
        return_layernorm_output=False,
        use_bias=True,
        return_bias=False,
        kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        skip_weight_param_allocation=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.units = units
        self.epsilon = epsilon
        self.gamma_initializer = get_init_method(
            gamma_initializer, initializers.get("ones")
        )
        self.beta_initializer = get_init_method(
            beta_initializer, initializers.get("zeros")
        )
        self.return_layernorm_output = return_layernorm_output
        self.use_bias = use_bias
        self.return_bias = return_bias
        self.kernel_initializer = get_init_method(
            kernel_initializer, initializers.RandomNormal(mean=0.0,
                                                          stddev=0.023)
        )
        self.bias_initializer = get_init_method(
            bias_initializer, initializers.get("zeros")
        )
        self.skip_weight_param_allocation = skip_weight_param_allocation

    def build(self, input_shape):
        """One-time allocation of the variables."""
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer should be "
                f"defined. Found None. Full input shape received: {input_shape}"
            )

        self.gamma = self.add_weight(
            name="gamma",
            shape=(last_dim,),
            initializer=self.gamma_initializer,
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(last_dim,),
            initializer=self.beta_initializer,
            trainable=True,
        )

        self.kernel = None
        self.bias = None
        if not self.skip_weight_param_allocation:
            self.kernel = self.add_weight(
                name="kernel",
                shape=(last_dim, self.units),
                initializer=self.kernel_initializer,
                trainable=True,
            )

            if self.use_bias or self.return_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )

        # fp8 related
        self.fp8_weight_shapes.append((last_dim, self.units))

        self.built = True

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed
            # from model.
            training = False
        return training

    def non_fp8_layernorm_matmul(
        self,
        inp: tf.Tensor,
        gamma_var: tf.Variable,
        beta_var: tf.Variable,
        kernel_var: tf.Variable,
        bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd non-fp8 layernorm followed by matmul."""
        @tf.custom_gradient
        def non_fp8_layernorm_matmul_func(x):
            # Use value() to convert from Variable to EagerTensor
            kernel_val = kernel_var.value()
            gamma_val = gamma_var.value()
            beta_val = beta_var.value()

            ln_out, mu, rsigma = tex.layernorm_fwd(
                x, gamma_val, beta_val, self.epsilon, self.stream_id
            )

            bias = get_autocast_bias(
                self._compute_dtype_object, bias_var, self.use_bias
            )

            output_dtype = self._compute_dtype_object
            outputs = matmul_wrapper(
                ln_out,
                kernel_val,
                "fwd",
                output_dtype,
                self.stream_id,
                self.use_bias,
                bias,
            )

            def grad_fn(*upstream, variables=None):
                grad_x = matmul_wrapper(
                    upstream[0], kernel_val, "bwd_input", output_dtype,
                    self.stream_id,
                )
                grad_weight = matmul_wrapper(
                    ln_out, upstream[0], "bwd_weight", output_dtype,
                    self.stream_id,
                )
                if self.use_bias:
                    grad_bias = tf.math.reduce_sum(upstream[0], axis=0)

                if self.return_layernorm_output:
                    assert len(upstream) == 2
                    grad_x = grad_x + upstream[1]

                dxmat, dgamma, dbeta = tex.layernorm_bwd(
                    grad_x, x, mu, rsigma, gamma_val, self.stream_id
                )

                grad_inputs = [dxmat]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("gamma:0"):
                        grad_vars.append(dgamma)
                    elif v.name.endswith("bias:0") and self.use_bias:
                        grad_vars.append(grad_bias)
                    elif v.name.endswith("kernel:0"):
                        grad_vars.append(grad_weight)
                    elif v.name.endswith("beta:0"):
                        grad_vars.append(dbeta)

                return grad_inputs, grad_vars

            if self.return_layernorm_output:
                return (outputs, ln_out), grad_fn
            return outputs, grad_fn

        return non_fp8_layernorm_matmul_func(inp)

    def fp8_layernorm_matmul(
        self,
        inp: tf.Tensor,
        gamma_var: tf.Variable,
        beta_var: tf.Variable,
        kernel_var: tf.Variable,
        bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd fp8 layernorm followed by matmul."""
        fp8_meta = self.fp8_meta
        fp8_dtype_fwd, fp8_dtype_bwd, override_linear_precision = \
            get_recipe_attrs(fp8_meta["recipe"])

        @tf.custom_gradient
        def fp8_layernorm_matmul_func(x):
            # Use value() to convert from Variable to EagerTensor
            kernel_val = kernel_var.value()
            gamma_val = gamma_var.value()
            beta_val = beta_var.value()

            if not self.return_layernorm_output:
                ln_out, mu, rsigma = layernorm_fwd_fp8_wrapper(
                    x,
                    gamma_val,
                    beta_val,
                    self.epsilon,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_fwd,
                    self.stream_id,
                )
            else:
                ln_out_return, mu, rsigma = tex.layernorm_fwd(
                    x, gamma_val, beta_val, self.epsilon, self.stream_id
                )
                ln_out = cast_to_fp8_wrapper(
                    ln_out_return,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    True,
                    fp8_dtype_fwd,
                    self.stream_id,
                )

            bias = get_autocast_bias(
                self._compute_dtype_object, bias_var, self.use_bias
            )

            weight_fp8, weight_t_fp8 = fp8_cast_transpose_fused_wrapper(
                kernel_val,
                fp8_meta,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                True,
                fp8_dtype_fwd,
                self.stream_id,
            )

            output_dtype = self._compute_dtype_object
            outputs = fp8_matmul_wrapper(
                ln_out,
                weight_t_fp8,
                fp8_meta,
                "fwd",
                fp8_dtype_fwd,
                fp8_dtype_fwd,
                output_dtype,
                _2X_ACC_FPROP,
                self.stream_id,
                self.use_bias,
                bias,
            )

            def grad_fn(*upstream, variables=None):
                self.pre_backward()
                if self.use_bias:
                    (
                        grad_bias,
                        grad_fp8,
                        grad_t_fp8,
                    ) = fp8_cast_transpose_bgrad_fused_wrapper(
                        upstream[0],
                        fp8_meta,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        False,
                        fp8_dtype_bwd,
                        self.stream_id,
                    )
                else:
                    if not override_linear_precision.wgrad:
                        grad_fp8, grad_t_fp8 = fp8_cast_transpose_fused_wrapper(
                            upstream[0],
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )
                    else:
                        grad_fp8 = cast_to_fp8_wrapper(
                            upstream[0],
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )

                grad_x = fp8_matmul_wrapper(
                    grad_fp8,
                    weight_fp8,
                    fp8_meta,
                    "bwd_input",
                    fp8_dtype_bwd,
                    fp8_dtype_fwd,
                    output_dtype,
                    _2X_ACC_DGRAD,
                    self.stream_id,
                )

                if not override_linear_precision.wgrad:
                    ln_out_t = tex.fp8_transpose(ln_out, fp8_dtype_fwd,
                                                 self.stream_id)
                    grad_weight = fp8_matmul_wrapper(
                        ln_out_t,
                        grad_t_fp8,
                        fp8_meta,
                        "bwd_weight",
                        fp8_dtype_fwd,
                        fp8_dtype_bwd,
                        output_dtype,
                        _2X_ACC_WGRAD,
                        self.stream_id,
                    )
                else:
                    ln_out_c = cast_from_fp8_wrapper(
                        ln_out,
                        fp8_meta,
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        True,
                        fp8_dtype_fwd,
                        TE_DType[x.dtype],
                        self.stream_id,
                    )
                    grad_weight = matmul_wrapper(
                        ln_out_c,
                        upstream[0],
                        "bwd_weight",
                        output_dtype,
                        self.stream_id,
                    )

                if self.return_layernorm_output:
                    assert len(upstream) == 2
                    grad_x = grad_x + upstream[1]

                dxmat, dgamma, dbeta = tex.layernorm_bwd(
                    grad_x, x, mu, rsigma, gamma_val, self.stream_id
                )

                grad_inputs = [dxmat]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("gamma:0"):
                        grad_vars.append(dgamma)
                    elif v.name.endswith("bias:0") and self.use_bias:
                        grad_vars.append(grad_bias)
                    elif v.name.endswith("kernel:0"):
                        grad_vars.append(grad_weight)
                    elif v.name.endswith("beta:0"):
                        grad_vars.append(dbeta)

                return grad_inputs, grad_vars

            if self.return_layernorm_output:
                return (outputs, ln_out_return), grad_fn
            return outputs, grad_fn

        return fp8_layernorm_matmul_func(inp)

    def call(
        self,
        inputs,
        kernel=None,
        bias=None,
        training=None,
    ):
        """
        Apply layer normalization to the input followed by a linear
        transformation.

        Parameters
        ----------
        inputs : tf.Tensor
          Input tensor.
        kernel : tf.Variable, default = None
          An optional weight tensor for the module. This argument is compulsory
          if module is initialized with `skip_weight_param_allocation=True`
        bias : tf.Variable, default = None
          An optional bias tensor for the module. This argument is compulsory if
          module is initialized with `skip_weight_param_allocation=True` and one
          of `use_bias` or `return_bias`
        training : {True, False, None}, default = None
          Whether this is in the training context.
        """
        # self.pre_forward needs to be called outside the following branch,
        # since it has side effects to set the self.fp8 if the autocast is
        # detected.
        training = self._get_training_value(training)
        self.pre_forward(training)

        kernel_var = (kernel if self.skip_weight_param_allocation else
                      self.kernel)
        bias_var = bias if self.skip_weight_param_allocation else self.bias
        if kernel_var is None:
            raise ValueError("No valid kernel is provided")

        inputmat = tf.reshape(inputs, shape=(-1, inputs.shape[-1]))
        if self.fp8:
            outputs = self.fp8_layernorm_matmul(
                inputmat, self.gamma, self.beta, kernel_var, bias_var
            )
        else:
            outputs = self.non_fp8_layernorm_matmul(
                inputmat, self.gamma, self.beta, kernel_var, bias_var
            )
        if self.return_layernorm_output:
            outputmat, ln_outputmat = outputs
        else:
            outputmat = outputs

        outputs = tf.reshape(
            outputmat, shape=(-1, *inputs.shape[1:-1], outputmat.shape[-1])
        )
        if self.return_bias:
            if self.return_layernorm_output:
                ln_outputs = tf.reshape(ln_outputmat, shape=inputs.shape)
                return (outputs, bias_var, ln_outputs)
            return outputs, bias_var
        if self.return_layernorm_output:
            ln_outputs = tf.reshape(ln_outputmat, shape=inputs.shape)
            return (outputs, ln_outputs)
        return outputs

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "epsilon": self.epsilon,
                "gamma_initializer": initializers.serialize(
                    self.gamma_initializer),
                "beta_initializer": initializers.serialize(
                    self.beta_initializer),
                "return_layernorm_output": self.return_layernorm_output,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer),
                "skip_weight_param_allocation":
                    self.skip_weight_param_allocation,
            }
        )


class LayerNormMLP(TransformerEngineBaseModule, layers.Layer):
    """
    Applies layer normalization on the input followed by the MLP module,
    consisting of 2 successive linear transformations, separated by the GeLU
    activation.

    Parameters
    ----------
    units : int
      size of each input sample.
    ffn_units : int
      intermediate size to which input samples are projected.
    epsilon : float, default = 1e-3
      a value added to the denominator of layer normalization for numerical
      stability.
    gamma_initializer: Callable, default = `None`
      used for initializing LayerNorm gamma in the following way:
      `gamma_initializer(weight)`. When set to `None`, defaults to `ones`.
    beta_initializer: Callable, default = `None`
      used for initializing LayerNorm beta in the following way:
      `beta_initializer(weight)`. When set to `None`, defaults to `zeros`.
    use_bias : bool, default = `True`
      if set to `False`, the FC2 layer will not learn an additive bias.
    kernel_initializer: Callable, default = `None`
      used for initializing FC1 weights in the following way:
      `kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    ffn_kernel_initializer: Callable, default = `None`
      used for initializing FC2 weights in the following way:
      `ffn_kernel_initializer(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
      if set to `True`, output of layernorm is returned from the forward
      together with the output of the linear transformation.
      Example use case: residual connection for transformer module is taken post
      layernorm.
    bias_initializer: Callable, default = `None`
      used for initializing FC1 and FC2 bias in the following way:
      `bias_initializer(weight)`. When set to `None`, defaults to `zeros`.

    Optimization parameters
    -----------------------
    return_bias : bool, default = `False`
      when set to `True`, this module will not apply the additive bias itself,
      but instead return the bias value during the forward pass together with
      the output of the linear transformation :math:`y = xW`. This is useful
      when the bias addition can be fused to subsequent operations.
    """

    def __init__(
        self,
        units: int,
        ffn_units: int,
        epsilon: float = 1e-3,
        gamma_initializer: Union[Callable, str, None] = None,
        beta_initializer: Union[Callable, str, None] = None,
        return_layernorm_output: bool = False,
        use_bias: bool = True,
        return_bias: bool = False,
        kernel_initializer: Union[Callable, str, None] = None,
        ffn_kernel_initializer: Union[Callable, str, None] = None,
        bias_initializer: Union[Callable, str, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fc1_units = units
        self.fc2_units = ffn_units
        self.epsilon = epsilon

        self.gamma_initializer = get_init_method(
            gamma_initializer, initializers.get("ones")
        )
        self.beta_initializer = get_init_method(
            beta_initializer, initializers.get("zeros")
        )
        self.return_layernorm_output = return_layernorm_output
        self.use_bias = use_bias
        self.return_bias = return_bias
        self.kernel1_initializer = get_init_method(
            kernel_initializer, initializers.RandomNormal(mean=0.0,
                                                          stddev=0.023)
        )
        self.kernel2_initializer = get_init_method(
            ffn_kernel_initializer, initializers.RandomNormal(mean=0.0,
                                                              stddev=0.023)
        )
        self.bias_initializer = get_init_method(
            bias_initializer, initializers.get("zeros")
        )

    def build(self, input_shape):
        """One-time allocation of the variables."""
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer should be "
                f"defined. Found None. Full input shape received: {input_shape}"
            )

        self.gamma = self.add_weight(
            name="gamma",
            shape=(last_dim,),
            initializer=self.gamma_initializer,
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(last_dim,),
            initializer=self.beta_initializer,
            trainable=True,
        )

        self.fc1_kernel = self.add_weight(
            name="fc1_kernel",
            shape=(last_dim, self.fc1_units),
            initializer=self.kernel1_initializer,
            trainable=True,
        )
        self.fc1_bias = self.add_weight(
            name="fc1_bias",
            shape=(self.fc1_units,),
            initializer=self.bias_initializer,
            trainable=True,
        )

        # fp8 related
        self.fp8_weight_shapes.append((last_dim, self.fc1_units))

        self.fc2_kernel = self.add_weight(
            name="fc2_kernel",
            shape=(self.fc1_units, self.fc2_units),
            initializer=self.kernel2_initializer,
            trainable=True,
        )

        self.fc2_bias = None
        if self.use_bias or self.return_bias:
            self.fc2_bias = self.add_weight(
                name="fc2_bias",
                shape=(self.fc2_units,),
                initializer=self.bias_initializer,
                trainable=True,
            )

        # fp8 related
        self.fp8_weight_shapes.append((self.fc1_units, self.fc2_units))

        self.built = True

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passe from
            # model.
            training = False
        return training

    def non_fp8_layernorm_mlp(
        self,
        inp: tf.Tensor,
        gamma_var: tf.Variable,
        beta_var: tf.Variable,
        fc1_kernel_var: tf.Variable,
        fc1_bias_var: tf.Variable,
        fc2_kernel_var: tf.Variable,
        fc2_bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd non-fp8 layernorm followed by mlp."""
        @tf.custom_gradient
        def non_fp8_layernorm_mlp_func(x):
            # Use value() to convert from Variable to EagerTensor
            fc1_kernel_val = fc1_kernel_var.value()
            fc2_kernel_val = fc2_kernel_var.value()
            gamma_val = gamma_var.value()
            beta_val = beta_var.value()

            ln_out, mu, rsigma = tex.layernorm_fwd(
                x, gamma_val, beta_val, self.epsilon, self.stream_id
            )

            fc1_bias = get_autocast_bias(
                self._compute_dtype_object, fc1_bias_var, use_bias=True
            )
            fc2_bias = get_autocast_bias(
                self._compute_dtype_object, fc2_bias_var, self.use_bias
            )

            output_dtype = self._compute_dtype_object
            # TODO(kaixih): Ideally, we should set gelu=True to fuse the gelu in
            # cuBlasLt calls. However, it seems it is slower than the unfused
            # version. Fix this when cuBlasLt improves the issue.
            fc1_out = matmul_wrapper(
                ln_out,
                fc1_kernel_val,
                "fc1_fwd",
                output_dtype,
                self.stream_id,
                use_bias=True,
                bias=fc1_bias,
            )
            gelu_out = tex.te_gelu(
                fc1_out, None, TE_DType[output_dtype], None, None, 0,
                self.stream_id,
            )

            fc2_out = matmul_wrapper(
                gelu_out,
                fc2_kernel_val,
                "fc2_fwd",
                output_dtype,
                self.stream_id,
                use_bias=self.use_bias,
                bias=fc2_bias,
            )

            def grad_fn(*upstream, variables=None):
                fc2_dgrad = matmul_wrapper(
                    upstream[0],
                    fc2_kernel_val,
                    "fc2_bwd_input",
                    output_dtype,
                    self.stream_id,
                    grad=True,
                    gelu=True,
                    gelu_input=fc1_out,
                )

                fc2_wgrad = matmul_wrapper(
                    gelu_out, upstream[0], "bwd_weight", output_dtype,
                    self.stream_id,
                )
                if self.use_bias:
                    fc2_bias_grad = tf.math.reduce_sum(upstream[0], axis=0)

                dgelu = fc2_dgrad

                fc1_dgrad = matmul_wrapper(
                    dgelu, fc1_kernel_val, "fc1_bwd_input", output_dtype,
                    self.stream_id,
                )
                fc1_wgrad = matmul_wrapper(
                    ln_out, dgelu, "bwd_weight", output_dtype, self.stream_id
                )
                fc1_bias_grad = tf.math.reduce_sum(dgelu, axis=0)

                d_ln_out = fc1_dgrad

                if self.return_layernorm_output:
                    assert len(upstream) == 2
                    d_ln_out = d_ln_out + upstream[1]

                dxmat, dgamma, dbeta = tex.layernorm_bwd(
                    d_ln_out, x, mu, rsigma, gamma_val, self.stream_id
                )

                grad_inputs = [dxmat]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("gamma:0"):
                        grad_vars.append(dgamma)
                    elif v.name.endswith("fc1_kernel:0"):
                        grad_vars.append(fc1_wgrad)
                    elif v.name.endswith("fc1_bias:0"):
                        grad_vars.append(fc1_bias_grad)
                    elif v.name.endswith("fc2_kernel:0"):
                        grad_vars.append(fc2_wgrad)
                    elif v.name.endswith("fc2_bias:0") and self.use_bias:
                        grad_vars.append(fc2_bias_grad)
                    elif v.name.endswith("beta:0"):
                        grad_vars.append(dbeta)

                return grad_inputs, grad_vars

            if self.return_layernorm_output:
                return (fc2_out, ln_out), grad_fn
            return fc2_out, grad_fn

        return non_fp8_layernorm_mlp_func(inp)

    def fp8_layernorm_mlp(
        self,
        inp: tf.Tensor,
        gamma_var: tf.Variable,
        beta_var: tf.Variable,
        fc1_kernel_var: tf.Variable,
        fc1_bias_var: tf.Variable,
        fc2_kernel_var: tf.Variable,
        fc2_bias_var: Union[tf.Variable, None] = None,
    ):
        """Prep fwd+bwd fp8 layernorm followed by mlp."""
        fp8_meta = self.fp8_meta
        fp8_dtype_fwd, fp8_dtype_bwd, override_linear_precision = \
            get_recipe_attrs(fp8_meta["recipe"])

        @tf.custom_gradient
        def fp8_layernorm_mlp_func(x):
            # Use value() to convert from Variable to EagerTensor
            fc1_kernel_val = fc1_kernel_var.value()
            fc2_kernel_val = fc2_kernel_var.value()
            gamma_val = gamma_var.value()
            beta_val = beta_var.value()

            if not self.return_layernorm_output:
                ln_out, mu, rsigma = layernorm_fwd_fp8_wrapper(
                    x,
                    gamma_val,
                    beta_val,
                    self.epsilon,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_fwd,
                    self.stream_id,
                )
            else:
                ln_out_return, mu, rsigma = tex.layernorm_fwd(
                    x, gamma_val, beta_val, self.epsilon, self.stream_id
                )
                ln_out = cast_to_fp8_wrapper(
                    ln_out_return,
                    fp8_meta,
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    True,
                    fp8_dtype_fwd,
                    self.stream_id,
                )

            fc1_bias = get_autocast_bias(
                self._compute_dtype_object, fc1_bias_var, use_bias=True
            )
            fc2_bias = get_autocast_bias(
                self._compute_dtype_object, fc2_bias_var, self.use_bias
            )

            fc1_weight_fp8, fc1_weight_t_fp8 = fp8_cast_transpose_fused_wrapper(
                fc1_kernel_val,
                fp8_meta,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                True,
                fp8_dtype_fwd,
                self.stream_id,
            )

            fc2_weight_fp8, fc2_weight_t_fp8 = fp8_cast_transpose_fused_wrapper(
                fc2_kernel_val,
                fp8_meta,
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                True,
                fp8_dtype_fwd,
                self.stream_id,
            )

            output_dtype = self._compute_dtype_object
            fc1_out = fp8_matmul_wrapper(
                ln_out,
                fc1_weight_t_fp8,
                fp8_meta,
                "fc1_fwd",
                fp8_dtype_fwd,
                fp8_dtype_fwd,
                output_dtype,
                _2X_ACC_FPROP,
                self.stream_id,
                use_bias=True,
                bias=fc1_bias,
            )

            gelu_out = fp8_gelu_wrapper(
                fc1_out,
                fp8_meta,
                tex.FP8FwdTensors.GEMM2_INPUT,
                True,
                fp8_dtype_fwd,
                self.stream_id,
            )

            fc2_out = fp8_matmul_wrapper(
                gelu_out,
                fc2_weight_t_fp8,
                fp8_meta,
                "fc2_fwd",
                fp8_dtype_fwd,
                fp8_dtype_fwd,
                output_dtype,
                _2X_ACC_FPROP,
                self.stream_id,
                use_bias=self.use_bias,
                bias=fc2_bias,
            )

            def grad_fn(*upstream, variables=None):
                self.pre_backward()
                if self.use_bias:
                    (
                        fc2_bias_grad,
                        grad_fp8,
                        grad_t_fp8,
                    ) = fp8_cast_transpose_bgrad_fused_wrapper(
                        upstream[0],
                        fp8_meta,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        False,
                        fp8_dtype_bwd,
                        self.stream_id,
                    )
                else:
                    if not override_linear_precision.wgrad:
                        grad_fp8, grad_t_fp8 = fp8_cast_transpose_fused_wrapper(
                            upstream[0],
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )
                    else:
                        grad_fp8 = cast_to_fp8_wrapper(
                            upstream[0],
                            fp8_meta,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            False,
                            fp8_dtype_bwd,
                            self.stream_id,
                        )

                fc2_dgrad = fp8_matmul_wrapper(
                    grad_fp8,
                    fc2_weight_fp8,
                    fp8_meta,
                    "fc2_bwd_input",
                    fp8_dtype_bwd,
                    fp8_dtype_fwd,
                    output_dtype,
                    _2X_ACC_DGRAD,
                    self.stream_id,
                )

                if not override_linear_precision.wgrad:
                    gelu_out_t = tex.fp8_transpose(
                        gelu_out, fp8_dtype_fwd, self.stream_id
                    )
                    fc2_wgrad = fp8_matmul_wrapper(
                        gelu_out_t,
                        grad_t_fp8,
                        fp8_meta,
                        "fc2_bwd_weight",
                        fp8_dtype_fwd,
                        fp8_dtype_bwd,
                        output_dtype,
                        _2X_ACC_WGRAD,
                        self.stream_id,
                    )

                    (
                        fc1_bias_grad,
                        dgelu,
                        dgelu_t,
                    ) = fp8_cast_transpose_bgrad_dgelu_fused_wrapper(
                        fc2_dgrad,
                        fc1_out,
                        fp8_meta,
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        False,
                        fp8_dtype_bwd,
                        self.stream_id,
                    )
                else:
                    gelu_out_c = cast_from_fp8_wrapper(
                        gelu_out,
                        fp8_meta,
                        tex.FP8FwdTensors.GEMM2_INPUT,
                        True,
                        fp8_dtype_fwd,
                        TE_DType[x.dtype],
                        self.stream_id,
                    )
                    fc2_wgrad = matmul_wrapper(
                        gelu_out_c,
                        upstream[0],
                        "bwd_weight",
                        output_dtype,
                        self.stream_id,
                    )

                    # Different from PyTorch implementation, the fc1_out has
                    # already added bias. So we don't need to pass fc1_bias
                    # here.
                    fc1_bias_grad, dgelu_no_fp8 = bgrad_dgelu_fused(fc2_dgrad,
                                                                    fc1_out)
                    dgelu = cast_to_fp8_wrapper(
                        dgelu_no_fp8,
                        fp8_meta,
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        False,
                        fp8_dtype_bwd,
                        self.stream_id,
                    )
                    dgelu_t = None

                fc1_dgrad = fp8_matmul_wrapper(
                    dgelu,
                    fc1_weight_fp8,
                    fp8_meta,
                    "fc1_bwd_input",
                    fp8_dtype_bwd,
                    fp8_dtype_fwd,
                    output_dtype,
                    _2X_ACC_DGRAD,
                    self.stream_id,
                )

                if not override_linear_precision.wgrad:
                    ln_out_t = tex.fp8_transpose(ln_out, fp8_dtype_fwd,
                                                 self.stream_id)
                    fc1_wgrad = fp8_matmul_wrapper(
                        ln_out_t,
                        dgelu_t,
                        fp8_meta,
                        "fc1_bwd_weight",
                        fp8_dtype_fwd,
                        fp8_dtype_bwd,
                        output_dtype,
                        _2X_ACC_WGRAD,
                        self.stream_id,
                    )
                else:
                    ln_out_c = cast_from_fp8_wrapper(
                        ln_out,
                        fp8_meta,
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        True,
                        fp8_dtype_fwd,
                        TE_DType[x.dtype],
                        self.stream_id,
                    )
                    fc1_wgrad = matmul_wrapper(
                        ln_out_c,
                        dgelu_no_fp8,
                        "bwd_weight",
                        output_dtype,
                        self.stream_id,
                    )

                d_ln_out = fc1_dgrad

                if self.return_layernorm_output:
                    assert len(upstream) == 2
                    d_ln_out = d_ln_out + upstream[1]

                dxmat, dgamma, dbeta = tex.layernorm_bwd(
                    d_ln_out, x, mu, rsigma, gamma_val, self.stream_id
                )

                grad_inputs = [dxmat]
                grad_vars = []
                for v in variables:
                    if v.name.endswith("gamma:0"):
                        grad_vars.append(dgamma)
                    elif v.name.endswith("fc1_kernel:0"):
                        grad_vars.append(fc1_wgrad)
                    elif v.name.endswith("fc1_bias:0"):
                        grad_vars.append(fc1_bias_grad)
                    elif v.name.endswith("fc2_kernel:0"):
                        grad_vars.append(fc2_wgrad)
                    elif v.name.endswith("fc2_bias:0") and self.use_bias:
                        grad_vars.append(fc2_bias_grad)
                    elif v.name.endswith("beta:0"):
                        grad_vars.append(dbeta)

                return grad_inputs, grad_vars

            if self.return_layernorm_output:
                return (fc2_out, ln_out_return), grad_fn
            return fc2_out, grad_fn

        return fp8_layernorm_mlp_func(inp)

    def call(
        self,
        inputs,
        training=None,
    ):
        """
        Apply layer normalization to the input followed by a feedforward network
        (MLP Block).

        Parameters
        ----------
        inputs : tf.Tensor
          Input tensor.
        training : {True, False, None}, default = None
          Whether this is in the training context.
        """
        # self.pre_forward needs to be called outside the following branch,
        # since it has side effects to set the self.fp8 if the autocast is
        # detected.
        training = self._get_training_value(training)
        self.pre_forward(training, num_gemms=2)

        inputmat = tf.reshape(inputs, shape=(-1, inputs.shape[-1]))
        if self.fp8:
            outputs = self.fp8_layernorm_mlp(
                inputmat,
                self.gamma,
                self.beta,
                self.fc1_kernel,
                self.fc1_bias,
                self.fc2_kernel,
                self.fc2_bias,
            )
        else:
            outputs = self.non_fp8_layernorm_mlp(
                inputmat,
                self.gamma,
                self.beta,
                self.fc1_kernel,
                self.fc1_bias,
                self.fc2_kernel,
                self.fc2_bias,
            )
        if self.return_layernorm_output:
            outputmat, ln_outputmat = outputs
        else:
            outputmat = outputs

        outputs = tf.reshape(
            outputmat, shape=(-1, *inputs.shape[1:-1], outputmat.shape[-1])
        )
        if self.return_bias:
            if self.return_layernorm_output:
                ln_outputs = tf.reshape(ln_outputmat, shape=inputs.shape)
                return (outputs, self.fc2_bias, ln_outputs)
            return outputs, self.fc2_bias
        if self.return_layernorm_output:
            ln_outputs = tf.reshape(ln_outputmat, shape=inputs.shape)
            return (outputs, ln_outputs)
        return outputs

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.fc1_units,
                "ffn_hidden_size": self.fc2_units,
                "epsilon": self.epsilon,
                "gamma_init_method": initializers.serialize(
                    self.gamma_initializer),
                "beta_init_method": initializers.serialize(
                    self.beta_initializer),
                "return_layernorm_output": self.return_layernorm_output,
                "use_bias": self.use_bias,
                "init_method": initializers.serialize(self.kernel1_initializer),
                "output_layer_init_method": initializers.serialize(
                    self.kernel2_initializer
                ),
                "bias_init_method": initializers.serialize(
                    self.bias_initializer),
            }
        )
