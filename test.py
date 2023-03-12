import torch
import transformer_engine as te
import transformer_engine_extensions as tex

from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.pytorch.cpp_extensions import cudnn_fmha_fwd
from transformer_engine.common import recipe

b = 6
s_q = 128
h = 4
d = 16
#qkv = torch.randn([b, s_q, 3, h, d], dtype=torch.bfloat16, device="cuda")
qkv = torch.randn([b, s_q, 3, h, d], device="cuda").to(dtype=torch.int8) 
O = torch.empty( b, s_q, h, d, dtype=qkv.dtype, device="cuda")
actualSeqlen = torch.ones(b, dtype=torch.int32, device="cuda")
fp8_recipe = recipe.DelayedScaling(0, 1, recipe.Format.E4M3)
with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    cudnn_fmha_fwd(qkv, actualSeqlen, O)
print(O)
