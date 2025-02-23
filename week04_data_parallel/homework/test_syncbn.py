import os
import time
import pytest
import torch
from syncbn import SyncBatchNorm
import torch.multiprocessing as mp
import torch.distributed as dist


def init_process(rank, size, fn, args, master_port=12355, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, *args)


def _compare_impl(rank, size, hid_dim, sbn_x, bn_x):
    sbn = SyncBatchNorm(hid_dim)
    bn = torch.nn.BatchNorm1d(hid_dim)
    bn_x.requires_grad = True
    sbn_x.requires_grad = True
    local_batch_size = sbn_x.shape[0]
    global_batch_size = local_batch_size * size

    sbn_output = sbn(sbn_x)
    bn_output = bn(bn_x)
    
    rank_slice = slice(rank * local_batch_size, (rank + 1) * local_batch_size)
    bn_loss = bn_output[:global_batch_size//2, ...].sum()
    global_idx_left = local_batch_size * rank
    global_idx_right = local_batch_size * (rank + 1)
    loss_breakpoint = min(global_idx_right, global_batch_size // 2)
    shifted_loss_breakpoint = loss_breakpoint - global_idx_left
    sbn_loss = sbn_output[:shifted_loss_breakpoint].sum()
    bn_loss.backward()
    sbn_loss.backward()
    assert torch.allclose(sbn_output, bn_output[rank_slice], atol=1e-3, rtol=0)
    print(sbn_x.grad, bn_x.grad)
    assert torch.allclose(sbn_x.grad, bn_x.grad[rank_slice], atol=1e-3, rtol=0)


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [8, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = mp.get_context("spawn")
    bn_x = torch.randn(batch_size, hid_dim)
    sbn_x = torch.chunk(bn_x.detach().clone(), num_workers, dim=0)

    processes = []
    for rank in range(num_workers):
        args = (hid_dim, sbn_x[rank].detach(), bn_x.detach())
        process = ctx.Process(target=init_process, args=(rank, num_workers, _compare_impl, args))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
