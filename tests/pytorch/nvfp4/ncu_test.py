import transformer_engine.pytorch as te                                                                                                                                                                    
import transformer_engine_torch as tex                                                                                                                                                                     
from transformer_engine.pytorch import NVFP4Quantizer                                                                                                                                                      
import torch    
                                                                                                                                                                                                           
M, N, num_experts = 16384, 7168, 64                                                                                                                                                                        
x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")                                                                                                                                                 
splits = [M // num_experts] * num_experts                                                                                                                                                                  
split_tensor = torch.tensor(splits, dtype=torch.int64, device="cuda")
                                                                                                                                                                                                           
# warmup        
q = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)                                                                                                                  
for _ in range(3):                                                                                                                                                                                         
    tex.group_quantize(x, q, num_experts, split_tensor)
torch.cuda.synchronize()                                                                                                                                                                                   
                
# single measured launch                                                                                                                                                                                   
tex.group_quantize(x, q, num_experts, split_tensor)
torch.cuda.synchronize()              
