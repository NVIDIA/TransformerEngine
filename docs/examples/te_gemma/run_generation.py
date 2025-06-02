from utils import *

hyperparams.model_name = "/perfhome/repos/ckpt/models/gemma-7b-hf/" # "/tmp/gemma-7b-hf/" # <== Add model weight location here e.g. "/path/to/downloaded/gemma/weights"
hyperparams.qkv_format = "thd"

# hyperparams.generation_cuda_graphs = True # 709.8s
hyperparams.generation_cuda_graphs = True

if hyperparams.generation_cuda_graphs:
    # It is necessary to preallocate a static buffer.
    # CUDA graphs require static input tensors for every kernel.
    # This approach may result in a slight increase in memory consumption;
    # however, the substantial speedup achieved makes it worthwhile.
    hyperparams.cuda_graphs_static_batch_size = 64
    hyperparams.cuda_graphs_static_max_seq_len = 1024
    hyperparams.cuda_graphs_static_max_context_len = 128

hyperparams.is_paged = False
model = init_te_gemma_model(hyperparams)

print_sample_of_generated_texts(model)
benchmark_generation(model)
