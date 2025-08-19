# torchrun --nproc-per-node=4 2_torchrun.py
import torch.distributed as dist
import os

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"{rank=} initializing")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

if __name__ == "__main__":
    main()