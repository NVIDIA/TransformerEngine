# TE-FL Custom Backend Examples

This directory contains examples demonstrating two ways to add custom backends.

## Two Approaches

| Approach | Use Case | Example File |
|----------|----------|--------------|
| **In-tree** | Open source contribution, direct integration | `example_intree.py` |
| **Out-of-tree** | Closed-source / third-party plugin, standalone package | `example_outtree.py` |

## Quick Start

```bash
cd transformer_engine/plugin/examples

# In-tree approach
python example_intree.py

# Out-of-tree approach
python example_outtree.py
```

## In-tree Approach (3 Steps)

```python
from transformer_engine.plugin.core import (
    OpRegistry, OpManager, OpImpl, BackendImplKind
)

# 1. Define your operator implementation
def my_rmsnorm(input, weight, eps=1e-5, **kwargs):
    variance = input.pow(2).mean(-1, keepdim=True)
    return input * torch.rsqrt(variance + eps) * weight, torch.rsqrt(variance + eps)

# 2. Register to Registry
registry = OpRegistry()
registry.register_impl(OpImpl(
    op_name="rmsnorm_fwd",
    impl_id="vendor.mybackend",
    kind=BackendImplKind.VENDOR,
    vendor="mybackend",
    fn=my_rmsnorm,
    priority=200,
))

# 3. Call via Manager
manager = OpManager(registry)
output, rsigma = manager.call("rmsnorm_fwd", input, weight)
```

## Out-of-tree Approach (Plugin Package)

### Plugin Package Structure

```
my_vendor_plugin/
├── __init__.py      # Contains register(registry) function
└── setup.py         # or pyproject.toml
```

### \_\_init\_\_.py

```python
from transformer_engine.plugin.core import OpImpl, BackendImplKind

def my_rmsnorm(input, weight, eps=1e-5, **kwargs):
    # Your implementation
    ...

def register(registry):
    """Called automatically by TE-FL"""
    registry.register_impl(OpImpl(
        op_name="rmsnorm_fwd",
        impl_id="vendor.myvendor",
        kind=BackendImplKind.VENDOR,
        vendor="myvendor",
        fn=my_rmsnorm,
        priority=200,
    ))
```

### Loading Methods

```bash
# Method 1: Environment variable
export TE_FL_PLUGIN_MODULES=my_vendor_plugin
python your_script.py

# Method 2: pip install (requires entry_points configuration)
pip install my-vendor-plugin
python your_script.py
```

## Environment Variables

### Backend Selection

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `TE_FL_PREFER` | Preferred backend type (highest priority) | `flagos` / `vendor` / `reference` | `flagos` |
| `TE_FL_PREFER_VENDOR` | Prefer vendor backend (legacy, lower priority than `TE_FL_PREFER`) | `1` = prefer vendor, `0` = prefer flagos | `0` |
| `TE_FL_STRICT` | Strict mode - raise error if preferred implementation fails instead of fallback | `1` = strict, `0` = allow fallback | `0` |

### Vendor Filtering

| Variable | Description | Example |
|----------|-------------|---------|
| `TE_FL_ALLOW_VENDORS` | Whitelist of allowed vendors (comma-separated) | `nvidia,amd` |
| `TE_FL_DENY_VENDORS` | Blacklist of denied vendors (comma-separated) | `vendor_a,vendor_b` |

### Per-Operator Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `TE_FL_PER_OP` | Per-operator backend ordering | `rmsnorm_fwd=vendor:acme\|flagos;rope_fwd=flagos\|reference` |

Format: `op_name=backend1|backend2;op_name2=backend3|backend4`

### Plugin Discovery

| Variable | Description | Example |
|----------|-------------|---------|
| `TE_FL_PLUGIN_MODULES` | Plugin modules to load (comma-separated) | `my_plugin,another_plugin` |

### Build Configuration

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `TE_FL_SKIP_CUDA` | Skip CUDA backend (both build-time and runtime) | `1` = skip, `0` = enable | `0` |
| `CUDA_HOME` | CUDA installation path | `/usr/local/cuda` | Auto-detected |
| `CUDA_PATH` | Alternative CUDA path variable | `/usr/local/cuda` | Auto-detected |

### Logging

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `TEFL_LOG_LEVEL` | Log level for TE-FL | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

## Examples

### Prefer vendor backend
```bash
export TE_FL_PREFER=vendor
python your_script.py
```

### Only allow specific vendors
```bash
export TE_FL_ALLOW_VENDORS=nvidia,acme
python your_script.py
```

### Custom per-operator ordering
```bash
# Use acme vendor for rmsnorm, flagos for others
export TE_FL_PER_OP="rmsnorm_fwd=vendor:acme|flagos"
python your_script.py
```

### Skip CUDA and use FlagOS only
```bash
export TE_FL_SKIP_CUDA=1
export TE_FL_PREFER=flagos
python your_script.py
```

### Enable debug logging
```bash
export TEFL_LOG_LEVEL=DEBUG
python your_script.py
```

## Expected Output

When running, you should see logs like:

```
[TE-FL manager.py:133 INFO] Registered impl_ids: ['default.flagos', 'reference.torch', 'vendor.mybackend']
[TE-FL manager.py:390 INFO] Op 'rmsnorm_fwd' using 'vendor.mybackend' (kind=vendor, vendor=mybackend)
```
