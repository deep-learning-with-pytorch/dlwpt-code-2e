from init import *

def main(rank, world_size):
    initialize_distributed(rank, world_size)

if __name__ == '__main__':
    world_size = 4
    mp.spawn(main, args=(world_size,), nprocs=world_size)