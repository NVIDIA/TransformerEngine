# Basic Transformer Encoder Example with Optional FP8 #

This example uses Transformer Encoder to demonstrate the Transformer Engine usage. And more focus on scaling up training on multiple GPUs. Highly recommend studying the [MNIST example of the Transformer Engine](/examples/jax/mnist) before reading this example. The Transformer Engine is built on top of [Flax](https://github.com/google/flax). Thus, examples use `pjit` to set up multiple GPU training. The basic pjit usage can be referred to [Scale up Flax Modules on multiple devices with pjit](https://flax.readthedocs.io/en/latest/guides/flax_on_pjit.html).

## Single GPU ##

1. Setup dataset: This is done by using the `tfds` library to download the GLUE/CoLA dataset and using `nltk` to tokenize the sentences. This example focuses on Transformer Engine usage. Thus, a simple algorithm is used to convert tokens to INT32 tensors as input to the embedding layer. The `get_datasets` and `data_preprocess` routines are used for this purpose.

2. Define model: The `Net` class is a small Transformer Encoder model for sentence classification. The Transformer Engine provides `te.TransformerLayer` as encoder block and `te.DenseGeneral`. The structure of encoder block can be referred to [Scaling Up Models and Data with t5x and seqio](https://arxiv.org/abs/2203.17189)

3. Build training loop: The `train_and_evaluate` is the main routine to initialize the model and start training and evaluating. Use `fp8_autocast` context manager to enable FP8 training and check `var_collect` if the variable collection contains `Float8`.

4. Training process: In `train_step`, combine the FP8 metadata and latest model parameters into var_collect as a frozen dictionary and fill it to the gradient function.

5. Evaluating process: Same as the training process, the FP8 metadata needs to be in var_collect and fill it into a loss function, if enabling FP8 computing.

### Run ###

```bash
python test_single_gpu_encoder.py
python test_single_gpu_encoder.py --use-fp8
```

## Multiple GPU with Data Parallelism ##

1. The data parallelism (DP) divides a mini-batch for multiple devices, and each device has complete model parameters. In this example, the first dimension of input tensor is `batch_size` which is 64 by default, and uses 8 GPUs to train the model, so each device takes 8 sentences at once. The "dividing" is called "sharding" in the JAX documents.

2. In order to let JAX know how to do sharding, the `device_mesh` needs to be defined and each axis need to be named. A common way to annotate axis names is `data` which means the mesh dimension used for data-parallel sharding of the batch dimension of inputs and activations. And the first argument of `te.ShardingResource` is the name of the device axis which is used for data parallelism.

3. On the model side, the logical axis of each weight tensor of the model can be named. The `te.TransformerLayer` has the default names, which are stored in `abs_var_collect`, a collection of variables returned by `jax.eval_shape(encoder.init, ...)`. The key index is `params_axes`. The `te.DenseGeneral` doesn't have the default named axis because it is generic. Also, data-parallel sharding doesn't need to divide weight tensor, so named axis is not required for this case.

4. The next is to create sharding rules, mapping the device axis to the logical axis. The `te.extend_logical_axis_rules` under fp8_autocast will return a list of pairs of the mapping, such as `(('batch', 'data'), ...)`. The first is the logical axis and second is the device axis.

5. Refer structure of `abs_var_collect['params']` and `abs_var_collect['params_axes']` to set up `PartitionSpec` for pjit. All logical axes should be replaced by device axes. If the value of PartitionSpec is None, that means no sharding, broadcasting the data to every device. Note that the `params_axes` attribute is provided by Transformer Engine. The Flax's module doesn't have it, such as `nn.Embed`. For nn.Embed, assigning an empty PartitionSpec is fine because each device has its own embedding layer in DP mode. The `get_params_pspec` routine is used for this purpose. Because each device has a complete model in DP mode, all values of PartitionSpec in params_pspec should be None. This will be different in the model parallelism example.

6. Fill in `params_pspec` and `encoder.init` to pjit to get a compiled function, `pjit_encoder_init `, and use it to initialize the model, so JAX now can know how to do the sharding.

7. The `train_step` and `eval_step` also need to be compiled by pjit. Thus, every input and output argument has to be set up `PartitionSpec` if the argument contains a tensor. For instance, the `input_pspec` is `PartitionSpec('data', None)` because the input shape is (batch size, sequence length). Then, the rest of the workflow is similar to the previous example.

8. Use `CUDA_VISIBLE_DEVICES` to control the number of GPUs used. For example, if the system has 8 GPUs but only 4 GPUs need to be used, then:
   ```sh
   export CUDA_VISIBLE_DEVICES=0,1,2,3
   python test_multigpu_encoder.py
   ```
   Please refer to [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables) for more details.

### Run ###

```bash
python test_multigpu_encoder.py
python test_multigpu_encoder.py --use-fp8
```

## Multiple GPU with Model Parallelism ##

1. The model parallelism as known as tensor parallelism (TP) divides a model for multiple devices, and each device has part of model parameters. This example inherits previous DP example, but divides a model to two devices.

2. To set up device mesh for TP, adding a new named axis called `model`, which is used for sharding parameters of the model across devices. This example divides the model to two parts (`num_gpu_tp = 2`). One device only has half of the model.

3. On the model side, The `te.TransformerLayer` doesn't need additional settings because it has the default axis name already. It will be divided by `DEVICE_TP_AXIS` when model initialization. The first `te.DenseGeneral` is divided by columns and second one is divided by rows for TP. Because `te.DenseGeneral` doesn't have the default named axis, the names must be set manually by passing `kernel_axes` and `bias_axes` arguments. Then, the rest of the workflow is similar to the previous example.

4. The tips for debugging TP:
    * Use [inspect_array_sharding](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.inspect_array_sharding.html) or [visualize_array_sharding](https://jax.readthedocs.io/en/latest/_autosummary/jax.debug.visualize_array_sharding.html) to check the shape of activations and weights.
    * Check the shape of device buffer of weight tensor. For instance, `var_collect['params']['DenseGeneral_0']['kernel'].device_buffers[device_id].shape`. The `device_id` is an integer. If a weight tensor's shape is (256, 256) and you intend to divide it for two devices by second dimension, then the shape returned by device_buffers should be (256, 128).
    * Dump XLA HLO by setting `XLA_FLAGS` and see whether it contains unexpected `all-gather` operations or not.
    ```python
    import os
    os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_proto --xla_dump_hlo_as_text --xla_dump_hlo_as_html --xla_dump_to=<path to store XLA HLO>"
    ```
5. If the model parallelism example is run in the container, it is recommended to add `--ipc=host` in launch arguments. Otherwise, it might trigger UCX errors.
   ```sh
   docker run --gpus=all --ipc=host ...
   ```

### Run ###

```bash
python test_model_parallel_encoder.py
python test_model_parallel_encoder.py --use-fp8
```


## Multiple Processes with Model Parallelism ##

1. This example inherits previous model parallelism example, but uses multiprocessing instead of single-program multiple-data (SPMD). It uses 1 GPU per process.

2. The benefit of multiprocessing is to setup hardware affinity for GPUs, such as NUMA binding. It may help improve performance and stability. Please refer to [Best Practices When Benchmarking CUDA Applications](https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2019-s9956/) for more details.

3. The quick way to check system topology is to use `nvidia-smi`, for example:
   ```sh
   $ nvidia-smi topo -mp
           CPU Affinity    NUMA Affinity
   GPU0    48-63,176-191   3
   GPU1    48-63,176-191   3
   GPU2    16-31,144-159   1
   GPU3    16-31,144-159   1
   GPU4    112-127,240-255 7
   GPU5    112-127,240-255 7
   GPU6    80-95,208-223   5
   GPU7    80-95,208-223   5
   ```

4. It is recommended to set the environment variable `CUDA_DEVICE_ORDER` to `PCI_BUS_ID` before running the example with the affinity setting. To ensure that the device order is aligned between CUDA and `nvidia-smi`. Please refer to [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables) for more details.

5. `jax.distributed.initialize` must be called before any other JAX or Flax API, otherwise `jax.local_devices` will be incorrect. `jax.distributed.shutdown` should be the last API call.

6. Unlike SPMD, the input tensor must be sharded manually and be wrapped by `jax.make_array_from_single_device_arrays`. Otherwise, the sharding will be incorrect. Using DP=4, TP=2 as an example, the device mesh looks like:
   ```python
   mesh.device_ids = [[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7]]
   ```
   Assume that the process ID is mapped to GPU ID. The process 0 and process 1 are grouped for model parallelism, the process 2 and process 3 are grouped together too, and so on. Thus, process 0 and process 1 need to share the same micro-batch in the training step, process 0 and process 2, 4, and 6 have different micro-batch.

### Run ###

If the system has 8 GPUs, the basic commands are:
```bash
python test_multiprocessing_encoder.py --num-process 8 --process-id 0 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 1 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 2 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 3 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 4 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 5 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 6 &
python test_multiprocessing_encoder.py --num-process 8 --process-id 7 &
```
The correct setting for hardware affinity is system dependent. Taking the above system topology as an example, the command can be:
```bash
numactl --cpunodebind=48  --membind=3 python test_multiprocessing_encoder.py --num-process 8 --process-id 0 &
numactl --cpunodebind=49  --membind=3 python test_multiprocessing_encoder.py --num-process 8 --process-id 1 &
numactl --cpunodebind=16  --membind=1 python test_multiprocessing_encoder.py --num-process 8 --process-id 2 &
numactl --cpunodebind=17  --membind=1 python test_multiprocessing_encoder.py --num-process 8 --process-id 3 &
numactl --cpunodebind=112 --membind=7 python test_multiprocessing_encoder.py --num-process 8 --process-id 4 &
numactl --cpunodebind=113 --membind=7 python test_multiprocessing_encoder.py --num-process 8 --process-id 5 &
numactl --cpunodebind=80  --membind=5 python test_multiprocessing_encoder.py --num-process 8 --process-id 6 &
numactl --cpunodebind=81  --membind=5 python test_multiprocessing_encoder.py --num-process 8 --process-id 7 &
```