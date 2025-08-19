import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup(rank, world_size, backend):
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_data_loader(batch_size, rank, world_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='../data/p2ch16', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def train(rank, world_size, backend, epochs=2, batch_size=64):
    setup(rank, world_size, backend)
    
    # Set the device
    device = torch.device(f'cuda:{rank}' if backend == "nccl" else 'cpu')
    
    # Create model and move it to the appropriate device
    model = SimpleModel().to(device)
    
    # Wrap the model with DDP
    model = DDP(model)
    
    # Prepare data loader
    train_loader = get_data_loader(batch_size, rank, world_size)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
    
    cleanup()

def main():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        backend = "nccl"
    else:
        world_size = 2
        backend = "gloo"

    mp.spawn(train, args=(world_size, backend), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()