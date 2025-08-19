import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

def validate_models_same(model, rank, file_path='model.pth'):
    try:
        model_state_dict = model.state_dict()
        if rank == 0:
            torch.save(model_state_dict, file_path)

        dist.barrier()  # Ensure all processes reach this point before proceeding
        # Load the saved state_dict to ensure consistency
        saved_state_dict = torch.load(file_path)
        for param_name in model_state_dict.keys():
            param1 = model_state_dict[param_name]
            param2 = saved_state_dict[param_name]
            if rank == 1:
                print(f"Comparing '{param_name}': {param1} vs. {param2}")
            torch.allclose(param1, param2)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(3, 2)
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
def main(rank, world_size, backend):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Setting the seed ensures that the model is initialized the same way across all processes
    torch.manual_seed(0)

    model = SimpleModel()
    opt = torch.optim.SGD(SimpleModel().parameters(), lr=0.01)

    # 1. Forward
    input_data = torch.randn(3) + rank
    output = model(input_data)
    loss = output.mean()

    # 2. Backward
    opt.zero_grad()
    loss.backward()

    # 3. Allreduce gradients
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

    # 4. Update parameters
    opt.step()
    
    # Validation logic
    validate_models_same(model, rank)

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        backend = "nccl"
    else:
        world_size = 2
        backend = "gloo"

    mp.spawn(main, args=(world_size, backend), nprocs=world_size, join=True)
