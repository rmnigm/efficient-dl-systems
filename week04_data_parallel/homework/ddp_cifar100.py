import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.profiler import profile, ProfilerActivity
from torchvision.datasets import CIFAR100

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self, sync_bn="own"):
        super().__init__()
        assert sync_bn in ["own", "torch", "none"]
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        if sync_bn == "none":
            self.bn1 = nn.BatchNorm1d(128, affine=False)
        elif sync_bn == "own":
            self.bn1 = SyncBatchNorm(128)
        elif sync_bn == "torch":
            self.bn1 = nn.SyncBatchNorm(128, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def setup_dataloaders():
    train_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=True,
    )
    val_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=False,
    )
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank), batch_size=64)
    val_loader = DataLoader(val_dataset, sampler=DistributedSampler(val_dataset, size, rank), batch_size=64)
    return train_loader, val_loader


def run_training(rank, size):
    torch.manual_seed(0)
    train_loader, val_loader = setup_dataloaders()
    model = Net()
    device = torch.device("cpu")  # replace with "cuda" afterwards
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(train_loader)
    num_accum_steps = 2
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        for _ in range(2):
            epoch_loss = torch.zeros((1,), device=device)
            model.train()
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss += loss.detach()
                loss.backward()
                if (i + 1) % num_accum_steps == 0:
                    average_gradients(model)
                    optimizer.step()
                    optimizer.zero_grad()

                acc = (output.argmax(dim=1) == target).float().mean()

                print(f"Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}, acc: {acc}")
                epoch_loss = 0
    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table())
        # model.eval()
        # with torch.no_grad():
        #     for data, target in val_loader:
        #         data = data.to(device)
        #         target = target.to(device)
        #         output = model(data)
        #         loss = torch.nn.functional.cross_entropy(output, target)


def run_training_torch_tooling(rank, size):
    torch.manual_seed(0)
    train_loader, val_loader = setup_dataloaders()
    model = Net()
    device = torch.device("cpu")  # replace with "cuda" afterwards
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(train_loader)

    for _ in range(10):
        epoch_loss = torch.zeros((1,), device=device)
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            acc = (output.argmax(dim=1) == target).float().mean()

            print(f"Rank {dist.get_rank()}, loss: {epoch_loss / num_batches}, acc: {acc}")
            epoch_loss = 0
        # model.eval()
        # with torch.no_grad():
        #     for data, target in val_loader:
        #         data = data.to(device)
        #         target = target.to(device)
        #         output = model(data)
        #         loss = torch.nn.functional.cross_entropy(output, target)
    

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend="gloo")  # replace with "nccl" when testing on several GPUs
