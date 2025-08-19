import torch
import torch.distributed as dist
import torch.nn as nn
import os
import torch.multiprocessing as mp


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def init_process(rank, world_size, backend="gloo"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Finished initializing process {rank}")
    if rank == 0:
        model_part1 = nn.Linear(1, 2)
        input_batch = torch.tensor([[1.0]])
        part1_activations = model_part1(input_batch)
        print(f"Process {rank} activations from part 1: {part1_activations}")
        dist.send(part1_activations, dst=1)
    elif rank == 1:
        model_part2 = nn.Linear(2, 3)
        part1_activations = torch.zeros(2)
        dist.recv(part1_activations, src=0)
        part2_activations = model_part2(part1_activations)
        dist.send(part2_activations, dst=2)
        print(f"Process {rank} activations from part 2: {part2_activations}")
    elif rank == 2:
        model_part3 = nn.Linear(3, 1)
        part2_activations = torch.zeros(3)
        dist.recv(part2_activations, src=1)
        part3_output = model_part3(part2_activations)
        print(f"Process {rank} output from part 3: {part3_output}")
    dist.barrier()

if __name__ == "__main__":
    num_processes = 3
    mp.spawn(init_process, args=(num_processes,), nprocs=num_processes)