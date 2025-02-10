import pathlib

import wandb
import hydra
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch, make_grid
from modeling.unet import UnetModel
from utils import set_seed


# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main():
    config = OmegaConf.load("params.yaml")
    # OmegaConf.save(config, "config.yaml")
    wandb.login()
    run = wandb.init(
        project=config.train.wandb_project,
        name=config.train.wandb_run,
        config=config,
    )
    set_seed(config.train.seed)
    pathlib.Path("samples").mkdir(exist_ok=True)
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file("params.yaml")
    run.log_artifact(artifact)
    
    device = config.train.device
    eps_model = hydra.utils.instantiate(
        config.eps_model
        ).to(device)
    ddpm = hydra.utils.instantiate(
        config.diffusion,
        eps_model=eps_model,
        ).to(device)

    transforms_options = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    if config.train.data.random_flip:
        transforms_options.append(transforms.RandomHorizontalFlip())

    dataset = CIFAR10(
        "cifar10/train",
        transform=transforms.Compose(transforms_options),
    )
    dataloader = hydra.utils.instantiate(config.dataloader, dataset=dataset)
    optim = hydra.utils.instantiate(config.optim, ddpm.parameters())

    num_epochs = config.train.num_epochs
    for epoch in range(num_epochs):
        train_loss, loss_ema, input_batch = train_epoch(ddpm, dataloader, optim, device)

        input_image = make_grid(input_batch).detach().cpu() * 0.5 + 0.5
        input_pil = transforms.functional.to_pil_image(input_image)
        
        with torch.no_grad():
            output_image = generate_samples(ddpm, device, f"samples/{epoch:02d}.png")
        output_image = output_image.detach().cpu() * 0.5 + 0.5
        output_pil = transforms.functional.to_pil_image(output_image)

        logs = {
            "output_batch": wandb.Image(output_pil),
            "input_batch": wandb.Image(input_pil),
            "lr": config.optim.lr,
            "train_loss": train_loss,
            "train_loss_ema": loss_ema,
        }
        run.log(logs, step=epoch)

    torch.save(ddpm.state_dict(), "model_weights.pth")
    run.finish()


if __name__ == "__main__":
    main()
