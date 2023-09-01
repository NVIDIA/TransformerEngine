The provided modules are a PyTorch interface to a framework-oblivious implementation present in `ops`. All modules are decomposed into `Op`s. An `Op` models a practically atomic operation. For example, a `Linear` layer is split into either an `MMT` (MatMulTranspose) and `Add` `Op` or into just an `MMT` `Op`. Such an `Op` can be thought of as a combination of an `nn.Module` and an `autograd.Function`, in the sense that it:
1. Stores its trainable parameters (exposed through `require_grad`), like an `nn.Module`.
2. Provides a `forward`, `backward` (and `inference`) method, like an `autograd.Function`.
This is done to reduce the amount of needless boilerplate code. This allows for `Op` implementations to remain short, clean, and simple.
The `Sequential` module itself is just a wrapper around a `ComputePipeline` object that is actually responsible for executing its constituent `Op`s, as well as managing the interaction between them, such as type inference or model parallelism.
