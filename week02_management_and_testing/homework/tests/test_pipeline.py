import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel

from utils import set_seed


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./cifar10",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


def _train_run(dataset, learning_rate, hidden_size, device, num_epochs: int = 3):
    set_seed(42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=hidden_size),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    ).to(device)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=learning_rate)
    
    losses = []
    for _ in range(0, num_epochs):
        loss, _ = train_epoch(ddpm, dataloader, optimizer, device)
        losses.append(loss.cpu().item())
    return losses


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    subset_train_dataset = Subset(train_dataset, list(range(0, 12)))
    losses_1 = _train_run(subset_train_dataset, 5e-4, 32, device)
    losses_2 = _train_run(subset_train_dataset, 5e-4, 32, device)
    losses_3 = _train_run(subset_train_dataset, 5e-3, 16, device)

    assert losses_1 == pytest.approx(losses_2, abs=3e-3)
    assert losses_1 != pytest.approx(losses_3, abs=3e-3)
