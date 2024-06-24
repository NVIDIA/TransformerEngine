# Basic MNIST Example with Optional FP8 #

This example uses MNIST training to demonstrate the Transformer Engine usage. The Transformer Engine is built on top of [Flax](https://github.com/google/flax), a neural network library and ecosystem for JAX. Thus, the Transformer Engine is free to interoperate with other Flax modules. The basic Flax usage can be referred to [Flax Basics](https://flax.readthedocs.io/en/latest/guides/flax_basics.html).

1. Setup dataset: The first step is to prepare the dataset. This is done by using the `tfds` library to download the MNIST dataset and perform image preprocessing. The `get_datasets` routine is used for this purpose.

2. Define model: The `Net` class is a small CNN model for image classification. It has an option to switch between using `nn.Dense` provided by Flax and `te.DenseGeneral` provided by the Transformer Engine. This allows for easy comparison between the two libraries.

3. Build training loop: The `train_and_evaluate` is the main routine to initialize the model and start training and evaluating. For FP8 training, the key is `te.fp8_autocast` context manager. If fp8_autocast is enabled, it will cast all `te.DenseGeneral` to FP8 precision. The `var_collect` is a collection including needed information for model training, such as parameters and FP8 metadata, which is necessary for correct casting of BF16 tensors into FP8 tensors at runtime. If fp8_autocast is turned on and print var_collect, you will see FP8 metadata inside, such as `fp8_meta_collection` section. The training and evaluating with FP8 have to be done under  fp8_autocast. If not, then fp8_autocast will deconstruct the FP8 metadata, and the model will fall back to higher floating point precision, such as BF16 in this example. To check if FP8 is enabled, use the `check_fp8` routine. If model initialization with FP8 works fine, the string returned by jax.make_jaxpr should include the `Float8` keyword.

4. Training process: In `apply_model`, the main difference between normal Flax usage and this example is, with FP8 training, the FP8 metadata has to be filled into the gradient function `grad_fn`. Otherwise, the Transformer Engine doesn't know how to cast the BF16 tensor into FP8 tensor at runtime correctly. The FP8 metadata doesn't belong in model parameters (`state.params`), so we need to manually combine the metadata and latest model parameters into var_collect as a frozen dictionary and fill it to the gradient function.

5. Evaluating process: The evaluating process is the same as the training process. Need to ensure FP8 metadata is inside var_collect and fill it into loss function.

6. Additional options: The `te.fp8_autocast` context manager has additional options
   * FP8 Recipe: control FP8 training behavior. See the [FP8 tutorial](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) for a detailed explanation of FP8 recipes and the supported options.

## Run ##

1. Use Flax to train MNIST with BF16 as usual
```bash
python test_single_gpu_mnist.py
```

2. Use `te.DenseGeneral` provided by Transformer Engine to train MNIST with BF16
```bash
python test_single_gpu_mnist.py --use-te
```

3. Use `te.DenseGeneral` provided by Transformer Engine to train MNIST and enable FP8 training and evaluation.
```bash
python test_single_gpu_mnist.py --use-fp8
```
