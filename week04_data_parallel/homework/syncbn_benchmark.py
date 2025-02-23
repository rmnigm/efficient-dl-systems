import os
import time
import torch
from syncbn import SyncBatchNorm
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    fn(local_rank)


def benchmark_custom_syncbn(local_rank, hid_dim, batch_size):
    torch.manual_seed(0)
    device = torch.device(f"cuda:{local_rank}")
    bn = SyncBatchNorm(hid_dim).to(device)
    start = time.perf_counter()
    for _ in range(100):
        x = torch.randn(batch_size, hid_dim, requires_grad=True).to(device)
        bn(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    total_time = end - start
    reserved_memory = torch.cuda.max_memory_reserved(device=device)
    allocated_memory = torch.cuda.max_memory_allocated(device=device)
    custom_results = (total_time, reserved_memory, allocated_memory)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dist.barrier()
    bn = nn.SyncBatchNorm(hid_dim).to(device)
    start = time.perf_counter()
    for _ in range(100):
        x = torch.randn(batch_size, hid_dim, requires_grad=True).to(device)
        bn(x)
        torch.cuda.synchronize()
    end = time.perf_counter()
    total_time = end - start
    reserved_memory = torch.cuda.max_memory_reserved(device=device)
    allocated_memory = torch.cuda.max_memory_allocated(device=device)
    torch_results = (total_time, reserved_memory, allocated_memory)
    return custom_results, torch_results


def print_results(results):
    for (hd, bs), (custom_results, torch_results) in results.items():
        print(f"hidden_dim: {hd}, batch_size: {bs}")
        print(f"Custom SyncBN: {custom_results[0] * 10:.4f} (ms / ep), {custom_results[1] * 1e-6:.4f} MB reserved, {custom_results[2] * 1e-6:.4f} MB allocated")
        print(f"Torch SyncBN: {torch_results[0] * 10:.4f} (ms / ep), {torch_results[1] * 1e-6:.4f} MB reserved, {torch_results[2] * 1e-6:.4f} MB allocated")
        print()


def run_benchmarks(local_rank):
    hid_dim = [128, 256, 512, 1024]
    batch_size = [32, 64]
    results = {}
    # warmup
    _ = benchmark_custom_syncbn(local_rank, 128, 32)
    for hd in hid_dim:
        for bs in batch_size:
            custom_results, torch_results = benchmark_custom_syncbn(local_rank, hd, bs)
            results[(hd, bs)] = (custom_results, torch_results)
    if local_rank == 0:
        print(f"Rank {local_rank}")
        print_results(results)
        dist.barrier()
    else:
        dist.barrier()
        print(f"Rank {local_rank}")
        print_results(results)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, run_benchmarks, backend="nccl")