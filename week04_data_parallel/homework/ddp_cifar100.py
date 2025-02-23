import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100
import torch.distributed as dist

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="gloo"):
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


def reduce_metric(metric_value, op=dist.ReduceOp.SUM):
    dist.all_reduce(metric_value, op=op)
    return metric_value


def average_ratio_metric(numerator, denominator):
    dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return numerator / denominator


def scatter_dataset(rank, size, dataset, device):
    if rank == 0:
        leftover = len(dataset) - len(dataset) % size
        scatter_data_list = [[] for _ in range(size)]
        target_data_list = [[] for _ in range(size)]
        leftover_data_list, leftover_target_list = [], []
        for i, (data, target) in enumerate(dataset):
            if i < leftover:
                scatter_data_list[i % size].append(data)
                target_data_list[i % size].append(target)
            else:
                leftover_data_list.append(data)
                leftover_target_list.append(target)
        scatter_data_list = [torch.stack(subs).to(device) for subs in scatter_data_list]
        scatter_target_list = [torch.tensor(subs).to(device) for subs in target_data_list]
    else:
        scatter_data_list = None
        scatter_target_list = None
        leftover_data_list = None
        leftover_target_list = None
    scattered_data = torch.empty((len(dataset) // size, 3, 32, 32), dtype=torch.float32, device=device)
    scattered_target = torch.empty((len(dataset) // size, ), dtype=torch.int64, device=device)
    dist.scatter(scattered_data, scatter_data_list, src=0)
    dist.scatter(scattered_target, scatter_target_list, src=0)
    if rank == 0 and leftover_data_list:
        leftover_data = torch.stack(leftover_data_list)
        leftover_target = torch.tensor(leftover_target_list)
        scattered_data = torch.cat((scattered_data, leftover_data))
        scattered_target = torch.cat((scattered_target, leftover_target))
    val_dataset = torch.utils.data.TensorDataset(scattered_data, scattered_target)
    return val_dataset


def setup_dataloaders(rank, size, device):
    train_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        # download=True,
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
        # download=True,
        train=False,
    )
    val_dataset = scatter_dataset(rank, size, val_dataset, device)
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler(train_dataset, size, rank), batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    return train_loader, val_loader