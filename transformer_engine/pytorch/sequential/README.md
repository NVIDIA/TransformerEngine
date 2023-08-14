# Extending `te.Sequential`
## Recipe: Adding a new `module`

Let's say you're adding `awesomeLU`:
1. In `modules` create `awesomelu.py`.
2. In `modules`/`awesomelu.py` create `class AwesomeLU(BaseModule)`.
3. In `modules`/`awesomelu.py` implement `AwesomeLU`, analogically to existing modules.
    1. `AwesomeLU.__init__` must follow this schema:
        ```
        def __init__(self, ...):
        ```
        Initialize the (indirect) `nn.Module` superclass explicitly, to be able to assign `nn.Parameter`s to `self`:
        ```
            nn.Module.__init__(self)  # type: ignore
        ```
        Assign `nn.Parameter`s to `self`, save configurable state, perform other necessary initialization:
        ```
            ...
        ```
        Initialize the (direct) `BaseModule` superclass, with a list of operations that this module is to be converted to:
        ```
            super().__init__(
                ops.ABC(...),
                ops.XYZ(...),
                ...
            )
        ```
    2. If your module contains trainable parameters, and (at least some of) these parameters are randomly initialied (like `weight` and `bias` in `Linear`, but not `gamma` or `beta` in `LayerNorm`), allow the user to specify a custom initializer for these parameters, but provide a default one, if possible:
        ```
        def __init__(
            self,
            weight_init_method: ParameterInitMethod = _default_weight_init_method,
            ...
        ):
            ...
            self.weight = nn.Parameter(
                weight_init_method(torch.empty(...))
            )
            ...
        ```
    3. If (at least some of) the operations are to be executed conditionally (like adding bias in a `Linear`), you can pass `None` to `BaseModule.__init__` instead:
        ```
        def __init__(self, do_xyz: bool, ...):
            nn.Module.__init__(self)  # type: ignore

            ...

            super().__init__(
                ops.ABC(...),
                ops.XYZ(...) if do_xyz else None,
                ...
            )
        ```
    4. If (at least some of) the operations are not unary and use trainable parameters, pass them to their initializer (the parameters must be owned by the module object), converted to `nvte.Tensor` objects:
        ```
            super().__init__(
                ops.ABC(make_nvte_tensor(self.weight)),
                ...
            )
        ```
    5. If your module is stateful, expose all configurable state through `extra_repl`:
        ```
        def extra_repr(self):
            return f"do_xyz={self.do_xyz}"
        ```
4. In `modules`/`__init__.py` add `from awesomelu import AwesomeLU`.
5. In `modules`/`__init__.py` insert `AwesomeLU` to the module's `__all__` list, while maintaining lexicographical order (for consistency).


## Recipe: Adding a new `Op`

Let's say you're adding `awesomeLU`:
1. In `ops` create `awesomelu.py`.
2. In `ops`/`awesomelu.py` create `class AwesomeLU(Op)`.
3. In `ops`/`awesomelu.py` implement `AwesomeLU`, analogically to existing operation implementations
    1. In `AwesomeLU.__init__`:
        1. Take any secondary inputs to the forward pass as arguments:
            ```
            def __init__(
                weight: nvte.Tensor,
            ```
        2. Allow for configuring the type of:
            * The primary input to the operation in the forward pass `x` (input activation).
            * The input to the operation in the backward pass `dy` (partial derivative of the loss over the operation's activation `∂L/∂y`).
            * The output of the operation in the forward pass `y` (activation).
            * The primary output of the operation in the backward pass `dx` (partial derivative of the loss over the operation's input activation `∂L/∂x`).
            * The parametrized inputs to the operation in the forward pass (ex. `weight`, `bias`)
            * The secondary outputs of the operation in the backward pass (partial derivative of the loss over the operation's parametrized inputs, ex. `dweight`, `dbias`)
                ```
                    x_dtype: nvte.DType | None = ...,
                    weight_dtype: nvte.DType | None = ...,
                    dy_dtype: nvte.DType | None = ...,
                    y_dtype: nvte.DType = ...,
                    dx_dtype: nvte.DType = ...,
                    dweight_dtype: nvte.DType = ...,
                ):
                ```
        3. Note that if `x`, `dy` or (at least some of) the parameters can be processed by the operation's computations, without changing their type, this is to be signalled by using `None`:
            > ```
            >     x_dtype: nvte.DType | None = ...,
            >     weight_dtype: nvte.DType | None = ...,
            >     dy_dtype: nvte.DType | None = ...,
            > ```
        4. Provide defaults for these types to allow for constructing the operation object `AwesomeLu` without having to explicitly specify the types. Choose such default types that will result in optimal performance in the FP8 computational regime.
    2. In `AwesomeLU.args` return the list of all tensor attributes of `AwesomeLU` that require gradients.
    3. In `AwesomeLU.forward` provide the implementation of the forward pass of the operation:
        1. The input activation is to be taken as an argument to the `forward` function. _Note: Contrary to Pytorch, any parameters or configuration, can be conveniently accessed using the `self` object._
            ```
            def forward(self, x: nvte.Tensor):
            ```
        2. Remember to cast all `Tensor`-typed inputs to their requested types before performing computations on them, ex.:
            ```
                x = nvte.cast_checked(x, self.x_dtype)
                weight = nvte.cast_checked(self.weight, self.weight_dtype)
                bias = nvte.cast_checked(self.bias, self.bias_dtype)
            ```
        3. Return all auxilary tensors needed for the backward pass in a `Context` (`dict[Tensor]`) object. **Do not** store auxilary tensors in the `self` object. **Do not** return non-`Tensor` objects. These **may** be stored in the `self` object, and will remain accessible in the backward pass. **Do not** rely on the context being the same object. The dictionary keys **must** be valid Python identifier names. Example:
            ```
                return y, {"x": x, "weight": weight, "mu": mu, "rsigma": rsigma}
            ```
        4. If no auxilary tensors are needed for the backward pass, return an empty context.
    4. In `AwesomeLU.inference` provide the implementation of the forward pass of the operation, optimized for inference-time use.
    5. In `AwesomeLU.backward` provide the implementation of the backward pass of the operation:
        1. Retrieve the tensors stored in the forward pass inside the context, by using their keys. **Do not** attempt to access other keys of the dictionary. Example:
            ```
            def backward(self, ctx: Context, dy: nvte.Tensor):
                x, weight, mu, rsigma = ctx["x"], ctx["weight"], ctx["mu"], ctx["rsigma"]
            ```
        2. Remember to cast `dy` to its request type, before performing computations on it:
            ```
                dy = nvte.cast_checked(dy, self.dy_dtype)
            ```
        3. Return `dy` and a list of the gradients of all tensors returned by `AwesomeLU.args` in **the same order** (if `args` returns `[weight, bias]`, `backward` **must** return `dy, [dweight, dbias]`).
        4. If `AwesomeLU.args` returns `[]`, return `dy, []`.
    6. Remember to use fused implementations, when possible. For example, in some cases, using a sequence of `nvte.cast_checked` calls may be suboptimal, when, for example, `nvte.multi_cast_transpose` could be used instead, if the tensors are to be later transposed.
4. In `ops`/`__init__.py` add `from awesomelu import AwesomeLU`.
5. In `ops`/`__init__.py` insert `AwesomeLU` to the module's `__all__` list, while maintaining lexicographical order (for consistency).
6. Remember to implement fusions concerning `AwesomeLU`.

## Recipe: Adding a new `nvte.` function

Let's say you're adding support for `nvte_awesomelu`.
1. If `awesome_lu` is not present in `nvte`/`_nvte.pyi`:
    * If all parameters of `nvte_awesomelu` have one of these types...
        * `NVTEDType`
        * `NVTE_Fused_Attn_Backed`
        * `NVTE_QKV_Layout`
        * `NVTE_BiasType`
        * `NVTE_Mask_Type`
        * `NVTETensorPack`
        * `NVTETensor`
        * [the types automatically converted by Pybind11](https://pybind11.readthedocs.io/en/stable/advanced/cast/overview.html#conversion-table)
    * ...then:
        * In `cpp_extensions`/`pybind.cpp` register `nvte_awesomelu`:
            ```
            m.def("nvte_awesomelu", wrap(nvte_awesomelu));
            ```
    * ...else if the mapping of C++ arguments to Python arguments is a bijection, and the semantic meaning of the arguments is preserved, and the order of the arguments is preserved, and the mapping of C++ arguments' types to their their Python-side equivalents' types is a bijection, then, assuming an argument to `nvte_awesomelu` has a C type `c_type` that is to be exposed to the Python side as `PyType` that is to be converted by Pybind to `conv_type` then:
        1. If necessary, implement a C++ wrapper `conv_type` type over `c_type` to expose to the Python side as `PyType` and register it in Pybind using `py::class_<conv_type>(m, "PyType", py::module_local())` or similar.
        2. Specialize the `wrapped_arg` template:
            ```
            template <> struct wrapped_arg<c_type> : trait<conv_type> {};
            ```
        3. Register `nvte_awesomelu`:
            ```
            m.def("nvte_awesomelu", wrap(nvte_awesomelu));
            ```
    * ...else:
        * Manually implement a C++ wrapper over `nvte_awesomelu`
        * Register the wrapper to pybind using `m.def`.
    * In `nvte`/`_nvte.pyi` describe the Python-side interface to `nvte_awesomelu`, by replacing the C++ types with their Python-side equivalents - either types defined in `nvte`/`_nvte.pyi` or according to [builtin Pybind11 conversions](https://pybind11.readthedocs.io/en/stable/advanced/cast/overview.html#conversion-table) or your custom `PyType`s. Change `NVTETensorPack` into `typing.Sequence[Tensor]`.
2. In `nvte` create `awesomelu.py` importing `_nvte` using `from . import _nvte`.
3. In `nvte`/`awesomelu.py` implement function `awesomelu`.
    * Note: usually, if `nvte_awesomelu` requires temporary tensors, such as `workspace` or `barrier`, construct them inside of `awesomelu`, rather than take them as parameters.
    * Note: allow the user to specify the type of the output, if `nvte_awesome` supports that.
    * Note: the current computational pass (`forward`, `backward`, or `inference`) can be accessed through `_common.pass_`.
4. In `nvte`/`__init__.py` add `from awesomelu import awesomelu`.
5. In `nvte`/`__init__.py` insert `awesomelu` to the module's `__all__` list, while maintaining lexicographical order (for consistency).

## Recipe: Adding a new fusion

A fusions is an optimized implementation of a sequence of operations.

There are three types of fusions:
* fusions of inference passes
* fusions of the forward passes
* fusions of the backward passes

Specifically, there may be a fusion of forward passes that does not have a backward counterpart, and vice-versa.

To implement a fusion of the inference passes of operations `A`, `B`, and `C`:
1. In an appropriate existing or new file in `fusions` declare a function:
    ```
    @register_fusion_inference
    def a_b_c_inf_fused(a: A, b: B, c: C, x: nvte.Tensor):
    ```
2. The fusion must be equivalent to the sequence of inference passes it replaces.

To implement a fusion of the forward passes of operations `A`, `B`, and `C`:
1. In an appropriate existing or new file in `fusions` declare a function:
    ```
    @register_fusion_forward
    def a_b_c_fwd_fused(a: A, b: B, c: C, x: nvte.Tensor):
    ```
2. From `a_b_c_fwd_fused`, return:
    ```
    y, (a_ctx, b_ctx, c_ctx)
    ```
    Where `a_ctx`, `b_ctx`, and `c_ctx` are valid contexts of the corresponding `Op`s. Specifically:
    ```
    y, (a_ctx, b_ctx, c_ctx) = a_b_c_fwd_fused(a, b, c, x)
    dy = ... # ∂L/∂y
    dx2, a_grads = a.backward(a, a_ctx, dy)
    dx1, b_grads = b.backward(b, b_ctx, dx2)
    dx, c_grads = c.backward(c, c_ctx, dx1)
    ```
    **Must** be equivalent to:
    ```
    x1, a_ctx = a.forward(x)
    x2, b_ctx = b.forward(x1)
    y, c_ctx = c.forward(x2)
    dy = ... # `∂L/∂y`
    dx2, a_grads = a.backward(a, a_ctx, dy)
    dx1, b_grads = b.backward(b, b_ctx, dx2)
    dx, c_grads = c.backward(c, c_ctx, dy1)
    ```

To implement a fusion of the backward passes of operations `A`, `B`, and `C`:
1. In an appropriate existing or new file in `fusions` declare a function:
    ```
    @register_fusion_backward
    def a_b_c_bwd_fused(a: A, b: B, c: C, a_ctx: Context, b_ctx: Context, c_ctx: Context, dy: nvte.Tensor):
    ```
    Where `a_ctx`, `b_ctx`, and `c_ctx` are valid contexts of the corresponding `Op`s.
2. From `a_b_c_bwd_fused`, return:
    ```
    y, (a_grads, b_grads, c_cgrads)
    ```
    Where `a_grads`, `b_grads`, and `c_grads` are valid gradients of the corresponding `Op`s. Specifically:
    ```
    x1, a_ctx = a.forward(x)
    x2, b_ctx = b.forward(x1)
    y, c_ctx = c.forward(x2)
    dy = ... # `∂L/∂y`
    dx, (a_grads, b_grads, c_grads) = a_b_c_bwd_fused(a, b, c, a_ctx, b_ctx, c_ctx, dy)
    ```
    **Must** be equivalent to:
    ```
    x1, a_ctx = a.forward(x)
    x2, b_ctx = b.forward(x1)
    y, c_ctx = c.forward(x2)
    dy = ... # `∂L/∂y`
    dx2, a_grads = a.backward(a, a_ctx, dy)
    dx1, b_grads = b.backward(b, b_ctx, dx2)
    dx, c_grads = c.backward(c, c_ctx, dy1)
    ```
