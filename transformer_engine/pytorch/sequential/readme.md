`Sequential` is meant to be used with Transformer-like models that operate on tokens.

Usually, tensors in Pytorch are 3D: `(batch_size, seq_len, hidden_dim)`.
The problem with this is that this requires adding padding to make all sequences have the same length.

So, here, it is different. The input is two tensors: _`tokens`_`(total_tokens, hidden_dim)` + _`seq_lens`_`(batch_size)`.
For the most part, _`seq_lens`_ is unused. Only self-attention takes it into account.

Given any `m: BaseModule`, it can be invoked in one of three ways:
1. `m(x, seq_lens)` where `x` and `seq_lens` are respectively a 2D and a 1D tensor, as defined above.
2. `m(x)` where `x` is a 2D tensor - this is equivalent to `m(x, torch.Tensor([x.shape[0]]))`, ie. _`seq_lens`_ is `torch.Tensor([x.shape[0]])` or, simply, `x` is treated as a single token sequence.
3. `m(x)` where `x` is a 3D tensor - this is equivalent to `m(x.view(-1, x.shape[-1]), torch.Tensor([x.shape[0]] * x.shape[1]))`, which means that `x` is "flattened" from being a 3D tensor to a 2D tensor, and each of its previous slices is assumed to have been a single sequence.
