import torch
import transformer_engine as te

from transformer_engine.pytorch.cpu_offload import get_cpu_offload_context

#Initialize a CPU offload context to enable activation offloading and set number of layers
# to be offloaded to 1
context, sync_func = get_cpu_offload_context(True, 1, True, False)


#Define a 2 Linear layer model
layer = []
for i in range(2):
    layer.append(te.pytorch.Linear(1024,1024,bias=False,device="cuda"))

#Create dummy inputs on GPU
input_state = torch.rand(1024,1024).cuda()

#Wrap the forward prop under the context
with context:
    hidden = layer[0](input_state)

#Use synchronize function to sync across layers
hidden = sync_func(hidden)

with context:
    output = layer[1](hidden)

output = sync_func(output)

#Trigger backward
output.sum().backward()
