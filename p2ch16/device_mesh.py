"""
Run this example with command:

torchrun --nproc_per_node=8 device_mesh.py
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank=} in a world with {world_size=}")
dist.init_process_group("gloo")

from torch.distributed.device_mesh import init_device_mesh
mesh_2d = init_device_mesh(
    "cpu", (2, 4), mesh_dim_names=("replicate", "model_parallel")
)

class SimpleModelPart(nn.Module):
    def __init__(self):
        super(SimpleModelPart, self).__init__()
        self.rank = dist.get_rank()
        self.model_part = f"{self.rank % mesh_2d['model_parallel'].size()}"
        self.fc = nn.Linear(5, 5)
        print(f"Rank {self.rank} {self.fc.weight}")

    def forward(self, x):
        print(f"Rank {self.rank} calling forward for model part {self.model_part}")
        x = torch.relu(self.fc(x))
        return x

model_part = SimpleModelPart()
# Synchronize model parameters
replica_mesh = mesh_2d["replicate"]
replica_group = replica_mesh.get_group()
def sync_model(replica_mesh):
    print(f"Mesh {replica_mesh} averaging model parameters")
    for p in model_part.parameters():
        dist.all_reduce(p, group=replica_group)
        p.data /= replica_mesh.size()

sync_model(replica_mesh)

# Execute model parallelism
def forward_pass(model_part, model_parallel_mesh):
    model_parallel_group = model_parallel_mesh.get_group()
    local_rank = model_parallel_mesh.get_local_rank()
    global_rank = model_parallel_mesh.get_rank()
    if local_rank == 0:
        x = torch.randn(5, 5)
        output = model_part(x)
        dist.send(output, dst=global_rank + 1, group=model_parallel_group)
    elif local_rank == model_parallel_mesh.size() - 1:
        x = torch.zeros(5, 5)
        dist.recv(x, src=global_rank - 1, group=model_parallel_group)
        output = model_part(x)
        return output
    else:
        x = torch.zeros(5, 5)
        dist.recv(x, src=global_rank - 1, group=model_parallel_group)
        output = model_part(x)
        dist.send(output, dst=global_rank + 1, group=model_parallel_group)

model_parallel_mesh = mesh_2d["model_parallel"]
forward_pass(model_part, model_parallel_mesh)
