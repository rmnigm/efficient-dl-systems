import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import task3.dataset as dataset
import pandas as pd

from torch.utils.data import DataLoader
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from task3.utils import Settings, Clothes, seed_everything
from task3.vit import ViT
from task3.profiler import Profile, Schedule



def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv(f"{Clothes.directory}/{Clothes.csv_name}")
    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        f"{Clothes.directory}/{Clothes.train_val_img_dir}", val_frame, transform=val_transforms
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer):
    epoch_loss, epoch_accuracy = 0, 0
    model.train()
    for data, label in tqdm(train_loader, desc="Train"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc.item() / len(train_loader)
        epoch_loss += loss.item() / len(train_loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss, epoch_accuracy


def train_epoch_with_profiler(model, train_loader, criterion, optimizer):
    epoch_loss, epoch_accuracy = 0, 0
    model.train()
    with Profile(
        model, schedule=Schedule(0, 1, 3, 1)
        ) as prof:
        for data, label in tqdm(train_loader, desc="Train", total=prof.schedule.total_steps()):
            data = data.to(Settings.device)
            label = label.to(Settings.device)
            output = model(data)
            loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prof.step()
            if prof.schedule.get_phase() == "done":
                break
    prof.to_perfetto("own-trace.json")
    return prof.summary()


def train_epoch_with_torch_profiler(model, train_loader, criterion, optimizer):
    epoch_loss, epoch_accuracy = 0, 0
    model.train()
    with profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
        cnt = 0 # i was too lazy to wait full epoch
        for data, label in tqdm(train_loader, desc="Train"):
            data = data.to(Settings.device)
            label = label.to(Settings.device)
            output = model(data)
            loss = criterion(output, label)
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item() / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prof.step()
            cnt += 1
            if cnt >= 4:
                break
    prof.export_chrome_trace("torch-trace.json")
    return prof.key_averages()

    
def eval_epoch(model, val_loader, criterion):
    val_loss, val_accuracy = 0, 0
    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(val_loader)
        val_loss += loss.item() / len(val_loader)
    return val_loss, val_accuracy


def profile_vit(mode="own"):
    seed_everything()
    model = get_vit_model()
    train_loader, _ = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)
    if mode == "own":
        summary = train_epoch_with_profiler(model, train_loader, criterion, optimizer)
    elif mode == "torch":
        summary = train_epoch_with_torch_profiler(model, train_loader, criterion, optimizer)
    return summary


if __name__ == "__main__":
    profile_vit()
