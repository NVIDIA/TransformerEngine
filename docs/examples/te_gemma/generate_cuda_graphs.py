import os

os.environ['CUDNN_LOGLEVEL_DBG'] = '3'
os.environ['CUDNN_LOGDEST_DBG'] = 'backlog.txt'
#Restart the notebook (to flush the GPU memory)
from utils import restart_jupyter_notebook
#restart_jupyter_notebook()
import transformer_engine.pytorch as te

from torch.cuda.amp import autocast


# Import necessary packages and methods
from utils import *

from transformer_engine.pytorch import fp8_model_init
from transformer_engine.common.recipe import Format, DelayedScaling


hyperparams.model_name = "../../../../gemma-weights"
hyperparams.fuse_qkv_params = True
model = init_te_gemma_model(hyperparams, fp8_model_init=True).cuda()

print("Loading model")
model_state_dict = torch.load('model_fp8_state_dict.pth')
model.load_state_dict(model_state_dict)
print("Model loaded")

tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
inputs = tokenizer(["I love when"] * 32, return_tensors="pt", padding=True)

inputs['input_ids'] = inputs['input_ids'].cuda()
inputs['attention_mask'] = inputs['attention_mask'].cuda()

import time



start_time = time.time()

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")
torch.manual_seed(1234)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    with autocast(dtype=torch.bfloat16, cache_enabled=False):
        with torch.no_grad():
            model.eval()
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                use_cuda_graphs=True
            )


end_time = time.time()
duration = end_time - start_time

generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for text in generated_texts[:12]:
    print("-" * 50)
    print(text)

print(f"Duration = {duration}")
