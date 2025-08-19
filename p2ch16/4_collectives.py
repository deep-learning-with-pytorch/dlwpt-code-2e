import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def perform_broadcast(rank):
    print(f"Process {rank} performing broadcast")
    if rank == 0:
        tensor = torch.tensor([123.0])
    else:
        tensor = torch.zeros(1)
    print(f"Process {rank} has tensor {tensor}")
    dist.broadcast(tensor, src=0)
    print(f"Process {rank} received tensor: {tensor}")

def perform_all_reduce(rank):
    print(f"Process {rank} performing all_reduce")
    tensor = torch.tensor([float(rank)])
    print(f"Process {rank} has tensor {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Process {rank} received tensor: {tensor}")

def init_process(rank, world_size, backend="gloo"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Finished initializing process {rank}")
    dist.barrier() # synchronize the processes so prints look nicer
    perform_broadcast(rank)
    dist.barrier() # synchronize the processes so prints look nicer
    perform_all_reduce(rank)

if __name__ == "__main__":
    num_processes = 3
    mp.spawn(init_process, args=(num_processes,), nprocs=num_processes)