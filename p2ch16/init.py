import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def create_store(host_name, port, rank, world_size) -> dist.Store:
    store = dist.TCPStore(host_name, port, world_size, is_master=rank == 0)
    return store

def initialize_distributed(rank, world_size, host_name="localhost", port=12345):
    store = create_store(host_name, port, rank, world_size)
    dist.init_process_group(
        backend='gloo',
        store=store,
        rank=rank,
        world_size=world_size,
    )

def main(rank, world_size):
    store = create_store('localhost', 24300, rank, world_size)
    store.set('key', 'value')
    print(f"rank {rank}: {store.get('key')}")

    initialize_distributed(rank, world_size)
    dist.barrier()
    print(f"rank {rank}: initialized")

if __name__ == '__main__':
    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size)