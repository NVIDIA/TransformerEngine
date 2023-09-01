# `te.Sequential`
While it originally started as just an implementation of an `nn.Sequential`-like module, `te.Sequential` is essentially becoming a reimplementation of the current PyTorch-side Transformer Engine API. The main goals of this refactoring are:
- **Increased expressivity**. Instead of using configuration flags, you can declare different Transformer architectures, by declaring their structure directly, within a `te.Sequential` module:
    - _Old API:_
        ```python
        gpt = te.TransformerLayer(
            HIDDEN_SIZE,
            4 * HIDDEN_SIZE,
            NUM_HEADS,
            apply_residual_connection_post_layernorm=False,
            output_layernorm=False,
            layer_type="encoder"
        )
        ```
    - _**New API:**_
        ```python
        gpt = te.Sequential(
            te.Residual(
                te.LayerNorm(HIDDEN_SIZE),
                te.Linear(HIDDEN_SIZE, 3 * HIDDEN_SIZE),
                te.MultiHeadedSelfAttention(
                    HIDDEN_SIZE,
                    NUM_HEADS,
                    te.DotProductAttention
                ),
                te.Linear(3 * HIDDEN_SIZE, HIDDEN_SIZE),
            ),
            te.Residual(
                te.LayerNorm(HIDDEN_SIZE),
                te.Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
                te.GELU(),
                te.Linear(4 * HIDDEN_SIZE, HIDDEN_SIZE),
            )
        )
        ```
- **Added flexibility**. Instead of using preavailable fused modules, you can use a `te.Sequential` that will perform inter-module fusions automatically:
    - _Old API:_
        ```python
        mlp = te.LayerNormMLP(
            HIDDEN_SIZE,
            4 * HIDDEN_SIZE,
            activation="swiglu",
            normalization="RMSNorm",
        )
        ```
    - _**New API:**_
        ```python
        mpl = te.Sequential(
            te.RMSNorm(HIDDEN_SIZE),
            te.Linear(HIDDEN_SIZE, 4 * HIDDEN_SIZE),
            te.SwiGLU(),
            te.Linear(4 * HIDDEN_SIZE, HIDDEN_SIZE),
        )
        ```
- **Improved performance**. Now, using `torch.compile(te.Sequential(...), fullgraph=True)`, you can fuse your model to a single FX graph for accelerated execution by PyTorch. **##NOT WORKING YET due to various issues in Torch Dynamo; see `compute_pipeline_function.py`##**

## Modules
`Sequential` is meant to be used with Transformer-like models that operate on tokens. As such, provided are modules typically most used when implement such architectures:
- `te.Linear` - a PyTorch-like linear layer supporting FP8 operations for accelerated performance on Hopper and Ada architectures.
- `te.LayerNorm` - a PyTorch-like LayerNorm with custom FP8 kernels manually fine-tuned for best performance on Hopper and Ada architectures.
- `te.RMSNorm` - an alternative normalization layer [[Zhang and Sennrich, 2019]](https://arxiv.org/abs/1910.07467) beating LayerNorm in computational and training performance, with custom FP8 kernels manually fine-tuned for best performance on Hopper and Ada architectures.
- `te.***LU` - a collection of activation functions most suitable for Transformer-based architectures with custom kernels supporting FP8 tensors for reduce memory bandwith consumption. Supported activation functions include `te.ReLU` (Transformer, GPT-1, T5), `te.GELU` (GPT-2, GPT-3, BERT), `te.SwiGLU` (PaLM, LLaMA), `te.GeGLU` (LaMDA), and `te.ReGLU`.
- `te.GroupedQueryAttention` - a generalized form of the attention mechanism, of which `te.MultiQuerySelfAttention` and `te.MultiHeadedSelfAttention` are special cases. These attention layers support for different attention mechanism implementations including `te.DotProductAttention`, `te.BlockSparseAttention`, `te.HungryHungryHippoes`... **##NOT YET IMPLEMENTED##**
- `te.Residual` - models a residual connection with a model. Its function is analogous to `te.Sequential`, except it adds the incoming activation to its final output. **##NOT YET IMPLEMENTED##**

## Input format
Usually, the input during the process of training of a Transformer model is composed of multiple sequences, forming a batch. The `te.Sequential` module accepts such a batch as input in one of a few formats.

Usually, batches are processed as rank-3 tensors of the form `(batch_size, seq_len, hidden_dim)`.
The problem with this is that this requires adding padding to make all sequences have the same length. To solve this issue, the input to the `te.Sequential` module is composed of two tensors: _`tokens`_`(total_tokens, hidden_dim)` + _`seq_lens`_`(batch_size)`, where the _`tokens`_ tensor is a concatenation of all sequences in the batch, and _`seq_lens`_ is a tensor containing the length of each sequence in the batch. Specifying _`seq_lens`_ is necessary for self-attention.

Given any `m: te.Sequential`, it can be invoked in one of three ways:
1. `m(x, seq_lens)` where `x` and `seq_lens` are respectively a 2D and a 1D tensor, as defined above.
2. `m(x)` where `x` is a 2D tensor - this is equivalent to `m(x, torch.Tensor([x.shape[0]]))`, ie. _`seq_lens`_ is `torch.Tensor([x.shape[0]])` or, simply, `x` is treated as a single sequence.
3. `m(x)` where `x` is a 3D tensor - this is equivalent to `m(x.view(-1, x.shape[-1]), torch.Tensor([x.shape[0]] * x.shape[1]))`, which means that `x` is "flattened" from being a 3D tensor to a 2D tensor, and each of its previous slices is assumed to have been a single sequence.

## Notes
* The GELU activation function is implemented as an approximation. For numerical results equivalent to PyTorch, use `nn.GELU(approximate="tanh")`.
* Due to limitations of TorchDynamo, some standard modules cannot be used. Some compatible replacements are provided in `utils.py`. Examples include `contextmanager` (replacement for `contextlib.contextmanager`) and `cache` (replacement for `functools.cache`).

## Idea
The main idea behind `te.Sequential` is that it doesn't have to execute eagerly, contrary to how PyTorch usually works. This is thanks to the fact that usually, its constitutent modules are provided during initialization and do not change since. This allows for performing optimizations such as fusions.

The main limitation of PyTorch that Transformer Engine is dealing with is that PyTorch does not have support for FP8 `dtype`s. Meanwhile, by taking advantage of these optimized formats, performance on the Hopper and Ada architectures can be significantly increased.

`te.Sequential` allows for sidestepping this issue by encapsulating the communications between subsequent modules. A bare `Linear` layer cannot return an FP8 tensor, even if the next operation supports that as an input, as there is no way to express this is PyTorch user code. However, by encapsulating both layers inside the `Sequential`, the communication between them happens in a way oblivious to the user. Only the input and output of the whole `Sequential` need to be representible as PyTorch tensors.
