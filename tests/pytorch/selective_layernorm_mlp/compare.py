
import time

import torch
from transformer_engine.pytorch import SelectiveLayerNormMLP, LayerNormMLP

torch.manual_seed(1234)
device = torch.device("cuda")

class _Sequential(torch.nn.Sequential):
    """Sequential model that forwards keyword arguments to modules"""

    def forward(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
        x = input_
        for module in self:
            x = module(x, **kwargs)
        return x

class ModelConfig:
    def __init__(
        self, 
        hidden_size: int = 128, 
        ffn_hidden_size: int = 512,
        layers: int = 1,
    ):
        self._hidden_size = hidden_size
        self._ffn_hidden_size = ffn_hidden_size
        self._layers = layers

    def build(self):

        ln_list, sln_list = [], []
        for _ in range(self._layers):
            ln = LayerNormMLP(self._hidden_size, self._ffn_hidden_size).to(device)
            sln = SelectiveLayerNormMLP(self._hidden_size, self._ffn_hidden_size).to(device)
            with torch.no_grad():
                ln.layer_norm_weight = torch.nn.Parameter(sln.layer_norm_weight.clone())
                ln.layer_norm_bias = torch.nn.Parameter(sln.layer_norm_bias.clone())
                ln.fc1_weight = torch.nn.Parameter(sln.fc1_weight.clone())
                ln.fc2_weight = torch.nn.Parameter(sln.fc2_weight.clone())
                ln.fc1_bias = torch.nn.Parameter(sln.fc1_bias.clone())
                ln.fc2_bias = torch.nn.Parameter(sln.fc2_bias.clone())
            ln_list.append(ln)
            sln_list.append(sln)

        ln_model = _Sequential(*ln_list)
        sln_model = _Sequential(*sln_list)

        return ln_model, sln_model

config = {
    # "small": ModelConfig(128, 512, 12),
    # "medium": ModelConfig(512, 2048, 12),
    # "large": ModelConfig(1024, 4096, 12),
    "huge": ModelConfig(2048, 8192, 12),
}

data_sizes = [2**7, 2**10, 2**14, 2**16]#2**18]

class Profiler:
    def __init__(self):
        self.stats = {
            "ln_stats": {
                "fwd_stats": {
                    "mem": [],
                    "time": [],
                },
                "bwd_stats": {
                    "mem": [],
                    "time": [],
                }
            },
            "sln_stats": {
                "fwd_stats": {
                    "mem": [],
                    "time": [],
                },
                "bwd_stats": {
                    "mem": [],
                    "time": [],
                }
            },
            "diff": {
                "out": [],
                "layer_norm_weight": [],
                "layer_norm_bias": [],
                "fc1_weight": [],
                "fc1_bias": [],
                "fc2_weight": [],
                "fc2_bias": [],
            }

        }

    def compare(self, ln_model, sln_model, data):

        use_cuda = device.type == "cuda" and torch.cuda.is_available()

        def _warmup(model, tensor):
            for i in range(3):
                model(tensor).sum().backward()

        def _run_fwd(model, tensor):

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            out = model(tensor)
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            mem = 0.0
            if use_cuda:
                peak_mem = torch.cuda.max_memory_allocated(device)
                mem = max(0.0, float(peak_mem - start_mem))
            return out, elapsed, mem

        def _run_bwd(model, out):

            model.zero_grad(set_to_none=False)

            loss = out.sum()

            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            mem = 0.0
            if use_cuda:
                peak_mem = torch.cuda.max_memory_allocated(device)
                mem = max(0.0, float(peak_mem - start_mem))

            param_grads = self._collect_param_grads(model)
            return param_grads, elapsed, mem

        _warmup(ln_model, data.clone())
        ln_fwd_out, ln_fwd_time, ln_fwd_mem = _run_fwd(ln_model, data.clone())
        ln_grads, ln_bwd_time, ln_bwd_mem = _run_bwd(ln_model, ln_fwd_out)

        _warmup(sln_model, data.clone())
        sln_fwd_out, sln_fwd_time, sln_fwd_mem = _run_fwd(sln_model, data.clone())
        sln_grads, sln_bwd_time, sln_bwd_mem = _run_bwd(sln_model, sln_fwd_out)

        self.stats["ln_stats"]["fwd_stats"]["time"].append(ln_fwd_time)
        self.stats["ln_stats"]["fwd_stats"]["mem"].append(ln_fwd_mem)
        self.stats["sln_stats"]["fwd_stats"]["time"].append(sln_fwd_time)
        self.stats["sln_stats"]["fwd_stats"]["mem"].append(sln_fwd_mem)

        # Track maximum absolute difference between outputs as a convergence metric.
        self.stats["diff"]["out"].append(self._max_diff(ln_fwd_out, sln_fwd_out))

        self.stats["ln_stats"]["bwd_stats"]["time"].append(ln_bwd_time)
        self.stats["ln_stats"]["bwd_stats"]["mem"].append(ln_bwd_mem)
        self.stats["sln_stats"]["bwd_stats"]["time"].append(sln_bwd_time)
        self.stats["sln_stats"]["bwd_stats"]["mem"].append(sln_bwd_mem)

        for key in ["layer_norm_weight", "layer_norm_bias", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"]:
            self.stats["diff"][key].append(self._max_diff(ln_grads[key], sln_grads[key]))

    def summarize(self):
        """Print a concise summary of collected statistics."""
        def _summarize(values):
            if not values:
                return {"avg": 0.0, "min": 0.0, "max": 0.0}
            return {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

        for name in ["ln_stats", "sln_stats"]:
            fwd_stats = self.stats[name]["fwd_stats"]
            bwd_stats = self.stats[name]["bwd_stats"]
            print(f"{name.upper()}")
            fwd_time = _summarize(fwd_stats["time"])
            fwd_mem = _summarize(fwd_stats["mem"])
            print(
                f"  Forward - time(ms): {fwd_time['avg']*1000:.3f} "
                f"[{fwd_time['min']*1000:.3f}, {fwd_time['max']*1000:.3f}] "
                f"mem(MB): {fwd_mem['avg']/1e6:.3f} "
            )

            bwd_time = _summarize(bwd_stats["time"])
            bwd_mem = _summarize(bwd_stats["mem"])
            print(
                f"  Backward - time(ms): {bwd_time['avg']*1000:.3f} "
                f"[{bwd_time['min']*1000:.3f}, {bwd_time['max']*1000:.3f}] "
                f"mem(MB): {bwd_mem['avg']/1e6:.3f}"
            )
            print()

        diff_stats = self.stats["diff"]
        fwd_diffs = diff_stats["out"]
        summary = sum(fwd_diffs) / len(fwd_diffs)
        print(f"Forward output max diff avg: {summary:.3e}")

        print("Gradient max diff averages:")
        for key in ["layer_norm_weight", "layer_norm_bias", "fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"]:
            summary = sum(diff_stats[key]) / len(diff_stats[key])
            print(f"  {key}: {summary:.3e}")
        print()

    def _max_diff(self, ref, other):
        """Return max absolute difference between two tensors or collections."""
        if ref is None or other is None:
            return 0.0
        if isinstance(ref, (list, tuple)):
            diffs = [self._max_diff(r, o) for r, o in zip(ref, other)]
            return max(diffs) if diffs else 0.0
        return torch.max(torch.abs(ref.detach() - other.detach())).item()

    def _collect_param_grads(self, model):
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            key = self._param_key(name)
            if key is not None:
                grads[key] = param.grad.detach().clone()
        return grads

    def _param_key(self, name):
        return name.split(".")[-1]

def main():

    for size in config:

        ln_model, sln_model = config[size].build()

        for seq_len in data_sizes:

            profiler = Profiler()

            dummy_data = torch.randn((seq_len, config[size]._hidden_size), device=device)

            profiler.compare(ln_model, sln_model, dummy_data)

            print(f"summarizing comparison for seq={seq_len}, hidden={config[size]._hidden_size}, ffn_fidden={config[size]._ffn_hidden_size}, layers={config[size]._layers}\n")
            profiler.summarize()

if __name__ == "__main__":
    main()
