# Restart the notebook (to flush the GPU memory)
from utils import restart_jupyter_notebook
#restart_jupyter_notebook()


# Import necessary packages and methods
from utils import *
import accelerate

# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
## Weights can be downloaded from: https://llama.meta.com/llama-downloads/
hyperparams.model_name = "../../../../gemma-weights"  # <== Add model weight location here e.g. "/path/to/downloaded/llama/weights"
hyperparams.mixed_precision = "bf16"
hyperparams.fuse_qkv_params = False

# Init the model and accelerator wrapper
model = init_te_gemma_model(hyperparams).cuda()
#accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)

model = model.to(torch.bfloat16).cuda()

tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)
inputs = tokenizer(["I love when "] * 64, return_tensors="pt", padding=True)

inputs['input_ids'] = inputs['input_ids'].cuda()
inputs['attention_mask'] = inputs['attention_mask'].cuda()

import time

# Początek pomiaru czasu
start_time = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=40
)

# Koniec pomiaru czasu
end_time = time.time()

# Obliczamy czas trwania operacji
duration = end_time - start_time
print(f"Generation time: {duration} seconds")


# Decode the output tensor to text
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Display the generated text
for text in generated_texts:
    print(text)