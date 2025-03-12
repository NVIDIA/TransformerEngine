# Calls to Nvidia-DL-Framework-Inspect

Let's look deeper into how Nvidia-DL-Framework-Inspect with Transformer Engine work together. TransformerEngine layers have some hook calls inside each of the GEMMs. Users can define feature classes or use feature classes provided with TE. File `config.yaml` describes which hooks need to be used for which layers. Nvidia-DL-Framework-Inspect combines 3 things: TE training, feature classes and `config.yaml` and takes care of inserting hooks in the correct places. This process is illustrated in the image below.

<figure align="center">
<img src="./img/api_calls1.svg">
    <figcaption> Fig 1: Example of Nvidia-DL-Framework-Inspect affecting training script with 1 Linear Layer. For tensors mentioned in `config.yaml`, behavior of `modify_tensor_enabled()` and `modify_tensor()` calls are substituted with definitions from the feature class. Other calls return default values - in fact they do nothing. </figcaption>
</figure>

In this page, all calls from TransformerEngine to the Nvidia-DL-Framework-Inspect for each GEMM are listed. The order of these calls is illustrated in the image below.


<figure align="center">
<img src="./img/api_calls2.svg">
    <figcaption> Fig 2: The calls to Nvidia-DL-Framework-Inspect done for Transformer Engine. There are 2 types of calls: GEMM calls and routing calls.</figcaption>
</figure>



### Categories of the API calls

There are 3 categories of API calls, each is used for different purposes:

- GEMM calls - invoked during every GEMM, is used to process or quantize tensors and collect information about them,
- routing calls - invoked at the beginning of the forward pass - they indicate whether a feature is going to use `modify_tensor()`, ... etc.

If all routing calls for the layer return `False`, then the layer is invoked in an optimized version with Transformer Engine fusions.
If any of the routing calls return `True`, layers are run without the fusion. It is necessary because some tensors cannot be accessed
if fusion happens. An important remark is that if no feature is used for the layer, then it should perform as fast as the layer without initializing `debug_api`.


### modify_tensor

It allows inserting tensor processing. For example, feature `FakeQuant` uses it to emulate casting to FP8, but the tensor is returned in higher precision. It can be invoked at most once for each tensor within a given GEMM operation.

This call is invoked if `modify_tensor_enabled` returns `True` and the feature is enabled for the *tensor_name* and *gemm*.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `tensor: torch.Tensor` - tensor in high precision,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
- `default_quantizer : Quantizer` - quantizer which is used to cast the tensor to lower precision if *modify_tensor* is not invoked. For example, feature per tensor scale uses it to obtain FP8 dtype of the tensor. If the recipe indicates that the tensor is not cast - for example, if running without FP8 autocast, then `default_quantizer=None`,
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.
- `out: Union[torch.Tensor, transformer_engine.pytorch.QuantizerTensor]` - output tensor, used in the weight caching mechanism.

Should return:

- `Union[torch.Tensor, transformer_engine.pytorch.QuantizerTensor, None]` - can be `torch.Tensor` or one of the Transformer Engine's `QuantizedTensor` - the rule is that both tensors returned for each GEMM should have the same type. If both are `Float8Tensor`, then GEMM is run in FP8. If both are `torch.Tensor`, GEMM is run in high precision. Please take that into account especially if only one tensor of the GEMM is processed by the `modify_tensor()`. For example, `FakeQuant` disabled FP8 GEMM to ensure that the second tensor is also in high precision. If the tensor is not the input for any GEMM - namely  `output`, `wgrad` and `dgrad` - the return type would match the input type. 
Should return `None` if `out` is not `None`.

Default behavior:

- It returns an unchanged tensor.

### inspect_tensor

The feature is invoked if *inspect_tensor_enabled* returns `True`. It can be used to obtain information on the high precision tensor. For example, it is run by the `LogTensorStats` feature.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
- `tensor` - tensor in high precision,
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called,

Should return nothing.

Default behavior:

- It does nothing.

### inspect_tensor_postquantize

Similar to *inspect_tensor*, but is run after one of the: fp8 cast, modify_tensor if they are run. If none of the fp8 cast or modify_tensor is invoked, then *inspect_tensor_postquantize* is also not invoked. The feature LogFp8Stats uses this call to collect FP8 statistics after the quantization.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
- `tensor` - tensor in fp8 or processed tensor after the modify_tensor call,
- `rowwise: bool` - whether this is the tensor or its transpose,
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.

Should return nothing.


### modify_tensor_enabled

It is used to determine whether *modify_tensor* will be run for a given GEMM and tensor name. It has **higher priority** than fp8_gemm, if *modify_tensor_enabled* returns True, then modify_tensor call is invoked for the respective tensor no matter what.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`],
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.

Should return:

- `output: bool`

Default behavior:

- It returns `False`.

### fp8_gemm_enabled

If the tensor is not processed using *modify_tensor* and the fp8 recipe is enabled, then the decision whether to cast it to fp8 is based on the value returned by the call *fp8_gemm_enabled*. If the tensor is processed using *modify_tensor* and or fp8 autocast is not enabled, the result of this call does not matter.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.

Should return:

- `fp_gemm: bool` – tensor after processing.


Default behavior:

- It returns `True`.


### inspect_tensor_enabled

It is a routing call, which is run at the initialization of the layer. If it returns true, then *inspect_tensor* for a given GEMM and tensor will be invoked for every forward.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`].
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.

Should return:

- `output: bool`

Default behavior:

- It returns `False`.

### inspect_tensor_postquantize_enabled


It is a routing call, which is run at the initialization of the layer. If it returns true, then *inspect_tensor_postquantize* for a given GEMM and tensor will be invoked for every forward.

Args:

- `config: Dict` - dictionary containing information from `config.yaml` corresponding to the feature, tensor_name and gemm.
- `layer_name: str`,
- `gemm: str` – one of [`fprop`, `dgrad`, `wgrad`],
- `tensor_name: str` – one of [`activation`, `weight`, `gradient`, `output`, `wgrad`, `dgrad`].
- `rowwise: bool` - whether this is the tensor or its transpose,
- `iteration: int` - iteration number - equal to the number of times `debug_api.step()` was called.

Should return:

- `output: bool`

Default behavior:

- It returns `False`.