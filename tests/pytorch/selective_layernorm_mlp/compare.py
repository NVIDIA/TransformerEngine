import time
import torch
from transformer_engine.pytorch import SelectiveLayerNormMLP, LayerNormMLP
from collections import defaultdict

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
                sln.layer_norm_weight = torch.nn.Parameter(ln.layer_norm_weight.clone())
                sln.layer_norm_bias = torch.nn.Parameter(ln.layer_norm_bias.clone())
                sln.fc1_weight = torch.nn.Parameter(ln.fc1_weight.clone())
                sln.fc2_weight = torch.nn.Parameter(ln.fc2_weight.clone())
                sln.fc1_bias = torch.nn.Parameter(ln.fc1_bias.clone())
                sln.fc2_bias = torch.nn.Parameter(ln.fc2_bias.clone())
            ln_list.append(ln)
            sln_list.append(sln)

        ln_model = _Sequential(*ln_list)
        sln_model = _Sequential(*sln_list)

        return ln_model, sln_model


config = {
    "small": ModelConfig(128, 512, 12),
    "medium": ModelConfig(512, 2048, 12),
    "large": ModelConfig(1024, 4096, 12),
    "huge": ModelConfig(2048, 8192, 12),
}

seq_sizes = [2**7, 2**10, 2**14, 2**16]

class Profiler:
    def __init__(self):
        self.stats = defaultdict(
            lambda: {
                "ln_stats": {
                    "fwd_stats": {
                        "mem": 0,
                        "time": 0,
                    },
                    "bwd_stats": {
                        "mem": 0,
                        "time": 0,
                    },
                },
                "sln_stats": {
                    "fwd_stats": {
                        "mem": 0,
                        "time": 0,
                    },
                    "bwd_stats": {
                        "mem": 0,
                        "time": 0,
                    },
                },
                "diff": {
                    "out": 0,
                    "layer_norm_weight": 0,
                    "layer_norm_bias": 0,
                    "fc1_weight": 0,
                    "fc1_bias": 0,
                    "fc2_weight": 0,
                    "fc2_bias": 0,
                },
            }
        )

    def compare(self, desc, ln_model, sln_model, data):

        def _warmup(model, tensor):
            for _ in range(10):
                model(tensor).sum().backward()

        def _run_fwd(model, tensor):

            torch.cuda.reset_peak_memory_stats(device)
            start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated(device)
            start_time.record()
            out = model(tensor)
            end_time.record()
            end_time.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            peak_mem = torch.cuda.max_memory_allocated(device)
            mem = float(peak_mem - start_mem)

            return out, elapsed, mem

        def _run_bwd(model, out):

            model.zero_grad(set_to_none=False)
            loss = out.sum()

            torch.cuda.reset_peak_memory_stats(device)
            start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated(device)
            start_time.record()
            loss.backward()
            end_time.record()
            end_time.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            peak_mem = torch.cuda.max_memory_allocated(device)
            mem = float(peak_mem - start_mem)

            param_grads = self._collect_param_grads(model)
            return param_grads, elapsed, mem

        _warmup(ln_model, data.clone())
        ln_fwd_out, ln_fwd_time, ln_fwd_mem = _run_fwd(ln_model, data.clone())
        ln_grads, ln_bwd_time, ln_bwd_mem = _run_bwd(ln_model, ln_fwd_out)

        _warmup(sln_model, data.clone())
        sln_fwd_out, sln_fwd_time, sln_fwd_mem = _run_fwd(sln_model, data.clone())
        sln_grads, sln_bwd_time, sln_bwd_mem = _run_bwd(sln_model, sln_fwd_out)

        self.stats[desc]["ln_stats"]["fwd_stats"]["time"] = ln_fwd_time
        self.stats[desc]["ln_stats"]["fwd_stats"]["mem"] = ln_fwd_mem
        self.stats[desc]["sln_stats"]["fwd_stats"]["time"] = sln_fwd_time
        self.stats[desc]["sln_stats"]["fwd_stats"]["mem"] = sln_fwd_mem

        # Track maximum absolute difference between outputs as a convergence metric.
        self.stats[desc]["diff"]["out"] = self._max_diff(ln_fwd_out, sln_fwd_out)

        self.stats[desc]["ln_stats"]["bwd_stats"]["time"] = ln_bwd_time
        self.stats[desc]["ln_stats"]["bwd_stats"]["mem"] = ln_bwd_mem
        self.stats[desc]["sln_stats"]["bwd_stats"]["time"] = sln_bwd_time
        self.stats[desc]["sln_stats"]["bwd_stats"]["mem"] = sln_bwd_mem

        for key in [
            "layer_norm_weight",
            "layer_norm_bias",
            "fc1_weight",
            "fc1_bias",
            "fc2_weight",
            "fc2_bias",
        ]:
            self.stats[desc]["diff"][key] = self._max_diff(ln_grads[key], sln_grads[key])

    def summarize(self):
        _modules = [("ln_stats", "LayerNormMLP"), ("sln_stats", "SelectiveLayerNormMLP")]
        _metric_map = {"time": (1, "ms"), "mem": (1e-6, "MB")}

        left_w  = 18  # "fwd time" / "bwd mem" label
        col1_w  = max(len(name) for _, name in _modules) + 2
        col2_w  = col1_w
        val_w   = 16  # number width

        def header(metric, unit):
            title = f"{metric.upper()} ({unit})"
            print(title)
            print(f"{'':<{left_w}}{_modules[0][1]:>{col1_w}}{_modules[1][1]:>{col2_w}}")
            print(f"{'-'*left_w}{'-'*col1_w}{'-'*col2_w}")

        for desc in self.stats:
            print("#" * 80 + "\n")
            print(desc + "\n")

            for metric in ["time", "mem"]:
                scale, unit = _metric_map[metric]
                header(metric, unit)
                for stage in ["fwd", "bwd"]:
                    v1 = self.stats[desc][_modules[0][0]][f"{stage}_stats"][metric] * scale
                    v2 = self.stats[desc][_modules[1][0]][f"{stage}_stats"][metric] * scale
                    # format with thousands separators and 3 decimals, aligned
                    s1 = f"{v1:>{val_w},.3f}"
                    s2 = f"{v2:>{val_w},.3f}"
                    print(f"{(stage+' ' + metric + ':'):<{left_w}}{s1:>{col1_w}}{s2:>{col2_w}}")
                print()  # blank line after each metric table

            # Errors block
            print("MAX ABSOLUTE ERRORS")
            print(f"{'output:':<30}{self.stats[desc]['diff']['out']:>14.3e}")
            for key in ["layer_norm_weight","layer_norm_bias","fc1_weight","fc1_bias","fc2_weight","fc2_bias"]:
                label = f"{key}.grad:"
                print(f"{label:<30}{self.stats[desc]['diff'][key]:>14.3e}")
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

    profiler = Profiler()

    for size in config:

        ln_model, sln_model = config[size].build()

        for seq_len in seq_sizes:

            dummy_data = torch.randn((seq_len, config[size]._hidden_size), device=device)

            desc = f"seq={seq_len}, hidden={config[size]._hidden_size}, ffn_fidden={config[size]._ffn_hidden_size}, layers={config[size]._layers}\n"
            profiler.compare(desc, ln_model, sln_model, dummy_data)


    profiler.summarize()


if __name__ == "__main__":
    main()
