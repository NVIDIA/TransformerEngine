from utils import *
import torch
from tqdm import tqdm  # For progress bar
import transformer_engine.pytorch as te


# Import necessary packages and methods
from utils import *
import accelerate

from transformer_engine.pytorch import fp8_model_init
from transformer_engine.common.recipe import Format, DelayedScaling

# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
## Weights can be downloaded from: https://llama.meta.com/llama-downloads/

hyperparams.model_name = "../../../../gemma-weights"
hyperparams.fuse_qkv_params = True
model = init_te_gemma_model(hyperparams, fp8_model_init=True).cuda()


print("Loading model")
model_state_dict = torch.load('model_fp8_state_dict.pth')
model.load_state_dict(model_state_dict)
print("Model loaded")


dataset = load_dataset(hyperparams.dataset_name, split="train")
tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)

accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=[FP8RecipeKwargs(backend="te")]
    )
train_dataloader = enumerate(get_dataloaders(accelerator, hyperparams))


model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
with torch.no_grad():  # No need to compute gradients during evaluation
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        for _, batch in tqdm(train_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].cuda()
            
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)

            predictions = torch.argmax(outputs.logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.numel()

accuracy = total_correct / total_samples
print(f"Accuraccy = {accuracy}")


