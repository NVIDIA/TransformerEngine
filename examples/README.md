# Examples

We provide a variety of examples for deep learning frameworks including [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch), [JAX](https://jax.readthedocs.io/en/latest/), and [PaddlePaddle](https://github.com/PaddlePaddle/Paddle). 
Additionally, we offer [Jupyter notebook tutorials](https://github.com/NVIDIA/TransformerEngine/tree/main/docs/examples) and a selection of [third-party examples](). Please be aware that these third-party examples might need specific, older versions of dependencies to function properly.

Please discuss in an [issue](https://github.com/NVIDIA/TransformerEngine/issues) a feature you would like to implement in an example before submitting a PR.

# PyTorch

- [Accelerate Hugging Face Llama models with TE](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/tutorial_accelerate_hf_llama_with_te.ipynb)
  - Provides code examples and explanations for integrating TE with the LLaMA2 and LLaMA2 models.
- [PyTorch FSDP with FP8](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/fsdp)
  - **Distributed Training**: How to set up and run distributed training using PyTorch’s FullyShardedDataParallel (FSDP) strategy.
  - **TE Integration**: Instructions on integrating TE/FP8 with PyTorch for optimized performance.
  - **Checkpointing**: Methods for applying activation checkpointing to manage memory usage during training.
- [Attention backends in TE](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/attention/attention.ipynb)
  - **Attention Backends**: Describes various attention backends supported by Transformer Engine, including framework-native, fused, and flash-attention backends, and their performance benefits.
  - **Flash vs. Non-Flash**: Compares the flash algorithm with the standard non-flash algorithm, highlighting memory and computational efficiency improvements.
  - **Backend Selection**: Details the logic for selecting the most appropriate backend based on availability and performance, and provides user control options for backend selection.
- [Overlapping Communication with GEMM](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/comm_gemm_overlap)
  - Training a TE module with GEMM and communication overlap, including various configurations and command-line arguments for customization.
- [Performance Optimizations](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/advanced_optimizations.ipynb)
  - **Multi-GPU Training**: How to use TE with data, tensor, and sequence parallelism.
  - **Gradient Accumulation Fusion**: Utilizing Tensor Cores to accumulate outputs directly into FP32 for better numerical accuracy.
  - **FP8 Weight Caching**: Avoiding redundant FP8 casting during multiple gradient accumulation steps to improve efficiency.
- [Introduction to FP8](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb)
  - Overview of FP8 datatypes (E4M3, E5M2), mixed precision training, delayed scaling strategies, and code examples for FP8 configuration and usage.
- [TE Quickstart](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb)
  - Introduction to TE, building a Transformer Layer using PyTorch, and instructions on integrating TE modules like Linear and LayerNorm.
- [Basic MNIST Example](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/mnist)

# JAX

# PaddlePaddle