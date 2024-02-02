"""Cuda graphs tests."""
import argparse

import torch
import transformer_engine.pytorch as te
import apex


def str_to_optimizer(optim):
    """Get optimizer."""
    if optim == "sgd":
        return torch.optim.SGD
    if optim == "adamw":
        return torch.optim.AdamW
    if optim == "fused_sgd":
        return apex.optimizers.FusedSGD
    return apex.optimizers.FusedAdam


def str_to_torch_dtype(dtype):
    """Get pytorch dtype."""
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def manual_seed(seed):
    """Set seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_data(args, warmup=False, gen_labels=False):
    """Generate synthetic data."""
    dtype = str_to_torch_dtype(args.dtype)
    gen_func = torch.ones if warmup else torch.randn
    if args.module == "dpa":
        inputs = [gen_func(
            args.seq_length, args.bs, args.nheads,
            args.embed, device="cuda", requires_grad=True, dtype=dtype
        ) for _ in range(3)]
    else:
        inputs = [gen_func(args.seq_length, args.bs,
                              args.hdim, device="cuda", requires_grad=True, dtype=dtype)]

    if not gen_labels:
        return inputs

    target = torch.randn(args.seq_length, args.bs, args.hdim, device="cuda", dtype=dtype)
    return inputs, target


def print_values(model, output):
    """Debug."""
    values = []
    for param in model.parameters():
        values.append(param.sum().item())
        if param.grad is not None:
            values.append(param.grad.sum().item())
    values.append(output.sum().item())
    print(values)


def parse_args():
    """Arguments."""
    parser = argparse.ArgumentParser(description="Args for testing CUDA graphs with TE layers.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dtype', type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument('--optimizer', type=str, default="fused_adamw",
                        choices=["fused_adamw", "fused_sgd", "sgd", "adamw"])
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--module', default="linear",
                        choices=['linear', 'layernorm_linear', 'layernorm_mlp',
                                 'transformer', 'dpa', 'mha'])
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--graph-mode', default="full", choices=['full', 'individual'])
    parser.add_argument('--num-warmup-iters', type=int, default=3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--hdim', type=int, default=768)
    parser.add_argument('--seq-length', type=int, default=2048)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--nheads', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    return parser.parse_args()


def train(args):
    """Train."""

    dtype = str_to_torch_dtype(args.dtype)

    # Create modules.
    if args.module == "transformer":
        modules = [te.TransformerLayer(
                        args.hdim, args.hdim, args.nheads,
                        hidden_dropout=args.dropout,
                        attention_dropout=args.dropout,
                        params_dtype=dtype,
                    ) for _ in range(args.num_layers)]
    elif args.module == "layernorm_mlp":
        modules = [te.LayerNormMLP(
            args.hdim, args.hdim, params_dtype=dtype
        ) for _ in range(args.num_layers)]
    elif args.module == "layernorm_linear":
        modules = [te.LayerNormLinear(
            args.hdim, args.hdim, params_dtype=dtype
        ) for _ in range(args.num_layers)]
    elif args.module == "mha":
        modules = [te.MultiheadAttention(
            args.hdim, args.nheads, attention_dropout=args.dropout, params_dtype=dtype
        ) for _ in range(args.num_layers)]
    elif args.module == "dpa":
        assert args.hdim % args.nheads == 0, "Err."
        assert args.num_layers == 1, "Err."
        args.embed = args.hdim // args.nheads
        modules = [te.DotProductAttention(
                    args.nheads, args.embed, attention_dropout=args.dropout
                    ) for _ in range(args.num_layers)]
    else:
        modules = [te.Linear(
            args.hdim, args.hdim, device="cuda", params_dtype=dtype
        ) for _ in range(args.num_layers)]

    # Generate model and wrap API to return graphed version.
    if args.graph:
        # Graph entire module at once.
        if args.graph_mode == "full":
            model = modules[0] if args.module == "dpa" else torch.nn.Sequential(*modules)
            model = te.make_graphed_callables(
                    model,
                    generate_data(args, warmup=True),
                    num_warmup_iters=args.num_warmup_iters,
                    enabled=args.fp8)
        else:
            modules = [te.make_graphed_callables(
                module,
                generate_data(args, warmup=True),
                num_warmup_iters=args.num_warmup_iters,
                enabled=args.fp8) for module in modules]
            model = modules[0] if args.module == "dpa" else torch.nn.Sequential(*modules)
    else:
        model = modules[0] if args.module == "dpa" else torch.nn.Sequential(*modules)

    # Loss function and optimizer.
    loss_fn = torch.nn.MSELoss()
    optimizer = str_to_optimizer(args.optimizer)(model.parameters(), lr=0.001)

    # Launch.
    for _ in range(args.steps):
        inputs, target = generate_data(args, gen_labels=True)
        with te.fp8_autocast(enabled=args.fp8):
            output = model(*inputs)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Debug.
    print_values(model, output)


if __name__ == "__main__":
    arguments = parse_args()
    manual_seed(arguments.seed)
    train(arguments)
