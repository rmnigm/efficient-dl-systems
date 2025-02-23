import os
import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ddp_cifar100 import Net, init_process, average_gradients, setup_dataloaders, average_ratio_metric, reduce_metric


torch.set_num_threads(1)


def run_custom_pipeline(rank, size):
    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")  # replace with "cuda" afterwards
    train_loader, val_loader = setup_dataloaders(rank, size, device)
    model = Net("own")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_epochs = 30
    num_accum_steps = 3    
    train_num_batches = len(train_loader)
    val_num_batches = len(val_loader)

    start_time = time.perf_counter()
    for ep in range(num_epochs):
        epoch_loss = torch.zeros((1,), device=device)
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.detach()
            loss.backward()
            if (i + 1) % (num_accum_steps + 1) == 0:
                average_gradients(model)
                optimizer.step()
                optimizer.zero_grad()
        train_loss = reduce_metric(epoch_loss).item() / (train_num_batches * size)

        epoch_loss = torch.zeros((1,), device=device)
        model.eval()
        with torch.no_grad():
            hits = torch.zeros((1,), device=device)
            total = torch.zeros((1,), device=device)
            for i, (data, target) in enumerate(val_loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss += loss.detach()
                hits += (output.argmax(dim=1) == target).float().sum()
                total += target.numel()
        val_acc = average_ratio_metric(hits, total).item()
        val_loss = reduce_metric(epoch_loss).item() / (val_num_batches * size)
        if rank == 0:
            print(f"Epoch {ep} | train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, accuracy: {val_acc:.5f}")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    if rank == 0:
        print(f"Final validation accuracy: {val_acc:.5f}", flush=True)
    print(f"Memory allocated in process {rank}: {torch.cuda.max_memory_allocated(device=device)}", flush=True)
    print(f"Memory reserved in process {rank}: {torch.cuda.max_memory_reserved(device=device)}", flush=True)
    print(f"Training time in process {rank}: {total_time:.2f} seconds", flush=True)
    print(f"Training time per epoch in process {rank}: {total_time / num_epochs:.2f} seconds", flush=True)


def run_torch_pipeline(rank, size):
    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")  # replace with "cuda" afterwards
    train_loader, val_loader = setup_dataloaders(rank, size, device)
    model = Net("torch")
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_epochs = 30
    num_accum_steps = 3
    train_num_batches = len(train_loader)
    val_num_batches = len(val_loader)

    start_time = time.perf_counter()
    for ep in range(num_epochs):
        epoch_loss = torch.zeros((1,), device=device)
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            if (i + 1) % (num_accum_steps + 1) != 0:
                with model.no_sync():
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
            else:
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        train_loss = reduce_metric(epoch_loss).item() / (train_num_batches * size)

        epoch_loss = torch.zeros((1,), device=device)
        model.eval()
        with torch.no_grad():
            hits = torch.zeros((1,), device=device)
            total = torch.zeros((1,), device=device)
            for i, (data, target) in enumerate(val_loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss += loss.detach()
                hits += (output.argmax(dim=1) == target).float().sum()
                total += target.numel()
        val_acc = average_ratio_metric(hits, total).item()
        val_loss = reduce_metric(epoch_loss).item() / (val_num_batches * size)
        if rank == 0:
            print(f"Epoch {ep} | train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, accuracy: {val_acc:.5f}")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    if rank == 0:
        print(f"Final validation accuracy: {val_acc:.5f}", flush=True)
    print(f"Memory allocated in process {rank}: {torch.cuda.max_memory_allocated(device=device)}", flush=True)
    print(f"Memory reserved in process {rank}: {torch.cuda.max_memory_reserved(device=device)}", flush=True)
    print(f"Training time in process {rank}: {total_time:.2f} seconds", flush=True)
    print(f"Training time per epoch in process {rank}: {total_time / num_epochs:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", action="store_true", default=False)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.custom:
        init_process(local_rank, fn=run_custom_pipeline, backend="nccl")  # replace with "nccl" when testing on several GPUs
    else:
        init_process(local_rank, fn=run_torch_pipeline, backend="nccl")  # replace with "nccl" when testing on several GPUs
