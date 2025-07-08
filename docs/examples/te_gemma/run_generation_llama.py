from utils import *

hyperparams.model_name = (  # "/tmp/gemma-7b-hf/" # <== Add model weight location here e.g. "/path/to/downloaded/gemma/weights"
    "/perfhome/repos/ckpt/models/llama2-7b-hf/"
)
hyperparams.qkv_format = "thd"

# model = init_te_llama_model(hyperparams)
model = init_baseline_model(hyperparams)

print_sample_of_generated_texts(model)
# benchmark_generation(model)
