"""
Run this example with command:

torchrun --nproc_per_node=2 fsdp_example.py
"""
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def main(device):
    model = SimpleModel()

    mesh = init_device_mesh(device, (2,))
    print(mesh)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    print(model)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inp = torch.rand((20, 10))
    output = model(inp)
    print(output)
    loss = output.sum()
    loss.backward()
    optim.step()

if __name__ == "__main__":
    import os
    from datetime import timedelta
    print(f"Process ID: {os.getpid()}")
    backend = "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=5))
    main("cpu")
    dist.destroy_process_group()