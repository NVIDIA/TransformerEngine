from utils import *
import torch
from tqdm import tqdm  # For progress bar

# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
## Weights can be downloaded from: https://llama.meta.com/llama-downloads/
hyperparams.model_name = "../../../../gemma-weights"  # <== Add model weight location here e.g. "/path/to/downloaded/llama/weights"
hyperparams.fuse_qkv_params = True

# Init the model and accelerator wrapper
model = init_te_gemma_model(hyperparams, fp8_model_init=False).cuda()
 
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

with torch.no_grad():  # No need to compute gradients during evaluation
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