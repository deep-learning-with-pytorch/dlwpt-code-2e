import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def init_process_with_store(rank, world_size, backend="gloo"):
    store = dist.TCPStore(
        host_name=os.environ["MASTER_ADDR"], 
        port=int(os.environ["MASTER_PORT"]),
        world_size=world_size,
        is_master=rank == 0
    )
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)
    print(f"Finished initializing process {rank}")

def init_process(rank, world_size, backend="gloo"):
    print(f"Initializing process {rank} using {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Finished initializing process {rank}")

if __name__ == "__main__":
    print("Using torch.multiprocessing")
    num_processes = 4
    print(f"Spawning {num_processes} processes")
    mp.spawn(init_process_with_store, args=(num_processes,), nprocs=num_processes)