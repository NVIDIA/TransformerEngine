# Import necessary packages and methods
import transformer_engine.pytorch as te
from utils import *
import accelerate
from transformer_engine.pytorch import fp8_model_init
from transformer_engine.common.recipe import Format, DelayedScaling
import torch


hyperparams.model_name = "../../../../gemma-weights"
hyperparams.fuse_qkv_params = True
model = init_te_gemma_model(hyperparams, fp8_model_init=False).cuda()
model = model.to(torch.bfloat16)


accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
        mixed_precision=hyperparams.mixed_precision,
        kwargs_handlers=[FP8RecipeKwargs(backend="te")]
    )
train_dataloader = get_dataloaders(accelerator, hyperparams)

tokenizer = AutoTokenizer.from_pretrained(hyperparams.model_name)

print("Calibration started")
with te.fp8_autocast(enabled=False, calibrating=True):
    model.train()
    train_dataloader = enumerate(train_dataloader)

    for i in range(100):
        step, batch = next(train_dataloader)
        batch["input_ids"] = batch["input_ids"].cuda()
        outputs = model.generate(
            **batch,
            max_new_tokens=1
        )
print("calibration_finished")

print("scale_fwd computation started")
with te.fp8_autocast(enabled=True):
    for i in range(10):
        step, batch = next(train_dataloader)
        batch["input_ids"] = batch["input_ids"].cuda()
        outputs = model.generate(
            **batch,
            max_new_tokens=1
        )
print("scale_fwd_computation ended")

print("Casting weights...")
model_fp8 = init_te_gemma_model(hyperparams, fp8_model_init=True).cuda()
model_fp8.load_state_dict(model.state_dict())
print("Weights casted")


print("Saving model...")
torch.save(model_fp8.state_dict(), 'model_fp8_state_dict.pth')
print("Model saved!")