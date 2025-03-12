..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

# Debug features

## LogTensorStats

This feature handles the logging of basic tensor statistics.  

For a distributed setting, the auxiliary stats are computed for each node and gathered after the `debug_api.step()` call. Do not forget to invoke `debug_api.step()` at every step to log stats!  

`LogTensorStats` supports micro-batching. If multiple forward/backward passes are invoked per `debug_api.step()`, then stats for all tensors except weights will be accumulated.  

`LogTensorStats` can induce significant overhead. To mitigate this issue, logging stats with `freq > 1` is recommended. If `LogTensorStats` is not used in a given step, the overhead is smaller. Moreover, if no other feature is used for the layer, the TE layer will run as fast as it would without `debug_api` initialized.  

Keys:

- `stats`: list of statistics to log:
  - `min`
  - `max`
  - `mean`
  - `std`
  - `l1_norm`
  - `l2_norm`
  - `cur_amax` – maximal absolute value of a tensor,
  - `dynamic_range` – equal to `torch.log2(amax) - torch.log2(amin)`,
- `tensors/tensors_struct`: 
  - activation
  - gradient
  - weight
- `freq`: Optional[int]
- `start_step`: Optional[int]
- `end_step`: Optional[int]
- `start_end_list`: Optional[list([int, int])], non-overlapping list of (start, end) pairs in incremental order. Default = None. If not None, will ignore start_step and end_step

**Example**
```yaml
example_tensor_stat_collection:
  enabled: True
  layers:
    layer_name_regex_pattern: .*(fc1|self_attention).*
  transformer_engine:
    LogTensorStats:
      enabled: True
      tensors_struct:
        - tensor: activation
          stats: [mean]
          freq: 10
          start_step: 5
          end_step: 100
        - tensor: gradient
          stats: [mean, max, min]
          freq: 2
          start_end_list: [[0, 20], [80, 100]]
        - tensor: weight
          stats: [dynamic_range]
```



## LogFp8TensorStats

This feature handles logging of FP8 tensor stats. 


For a distributed setting, the auxiliary stats are computed for each node and gathered after the `debug_api.step()` call. Do not forget to invoke `debug_api.step()` at every step to log stats!  

`LogFp8TensorStats` supports micro-batching. If multiple forward/backward passes are invoked per `debug_api.step()`, then stats for all tensors except weights will be accumulated.  

`LogFp8TensorStats` can induce significant overhead. To mitigate this issue, logging stats with `freq > 1` is recommended. If `LogFp8TensorStats` is not used in a given step, the overhead is smaller. Moreover, if no other feature is used for the layer, the TE layer will run as fast as it would without `debug_api` initialized.  


Keys:

- `stats`:
  - underflows%
  - overflows%
- `tensors/tensors_struct`: 
  - activation
  - gradient
  - weight
- `freq`: Optional[int]
- `start_step`: Optional[int]
- `end_step`: Optional[int]
- `start_end_list`: Optional[list([int, int])], non-overlapping list of (start, end) pairs in incremental order. Default = None. If not None, will ignore start_step and end_step

**Example**
```yaml
example_fp8_tensor_stat_collection:
  enabled: True
  layers:
    layer_types: [layernorm_linear]
  transformer_engine:
    LogFp8TensorStats:
        enabled: True
        tensors_struct: 
        - tensor: activation
          stats: [underflows%, overflows%]
          freq: 1
        - tensor: gradient
          stats: [overflows%]
          freq: 5
        start_step: 0
        end_step: 80
```

## DisableFp8Gemm

GEMM operations are executed in higher precision, even when FP8 autocast is enabled.

Keys:

- `gemms`: 
  - fprop
  - dgrad
  - wgrad

**Example**
```yaml
example_disable_fp8_gemm:
  enabled: True
  layers:
    layer_types: [fc1]
  transformer_engine:
    DisableFp8Gemm:
      enabled: True
      gemms: [dgrad, wgrad]
```



## DisableFp8Layer

Disables all FP8 GEMMs in the layer.


**Example**
```yaml
example_disable_fp8_layer:
  enabled: True
  layers:
    layer_types: [fc1]
  transformer_engine:
    DisableFp8Layer:
      enabled: True
```

## PerTensorScaling

Transformer Engine uses delayed scaling strategy on Hopper by default - you can read about it in [fp8 tutorial](../examples/fp8_primer.ipynb).
You can switch this strategy to current scaling by using this option. Then amax and dynamic range will be computed using the current tensor, not the historical ones. It can improve stability and accuracy of the training, but it's slower than delayed scaling. 

Note that tensors in this feature are Hopper `Float8Tensor` containing one scaling factor per tensor.


Keys:

- `gemms/gemms_struct`:
  - fprop
  - dgrad
  - wgrad
- `tensors/tensors_struct`:
  - activation
  - gradient
  - weight
- `margin`: int - impacts the computation of scaling factors, default is 0, `amax = amax * (2^margin)`.

**Example**
```yaml
example_per_tensor_scaling:
  enabled: True
  layers:
    layer_types: [transformer_layer.self_attn.layernorm_q]
  transformer_engine:
      PerTensorScaling:
        enabled: True
        margin: 1
        gemms: [dgrad]
        tensors: [weight, activation]
```



## FakeQuant

Disables FP8 GEMM. Fake quantizes chosen tensors to FP8 - using per-tensor scaling factor, not delayed scaling - and runs high-precision GEMM.

<figure align="center">
<img src="./img/fake_quant.svg">
    <figcaption> Fig 1: Comparison of FP8 FPROP GEMM with the same GEMM in BF16 with fake quantization of activation tensor. Green tensors have the same values, but different dtypes. </figcaption>
</figure>

- `gemms/gemms_struct`: 
  - fprop
  - dgrad
  - wgrad
- `tensors/tensors_struct`:
  - activation
  - gradient
  - weight
- `quant_format` - specifies the FP8 format to use: 
  - FP8E5M2
  - FP8E4M3
- `margin`: int - impacts the computation of scaling factors, default is 0, `amax = amax * (2^margin)`.

**Example**
```yaml
example_fake_quant_fp8:
  enabled: True
  layers:
    layer_types: [transformer_layer.layernorm_mlp.fc1]
  transformer_engine:
      FakeQuant:
        enabled: True
        quant_format: FP8E5M2
        gemms_struct:
        - gemm: fprop
          tensors: [activation, weight]
        - gemm: dgrad
          tensors: [gradient]
```