# Basic Example for Using PyTorch Fully Sharded Data Parallel mode with Transformer Engine

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py
torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py --no-fp8
torchrun --standalone --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) fsdp.py --defer-init
```
