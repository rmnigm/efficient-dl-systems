import os
import pytest
import torch
from syncbn import SyncBatchNorm
import torch.multiprocessing as mp
import torch.distributed as dist


def init_process(inputs, rank, size, sync_bn, queue, master_port=12355, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    output = sync_bn(inputs)
    queue.put(output.detach())
    


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = mp.get_context("spawn")
    bn_x = torch.randn(batch_size, hid_dim, 64, requires_grad=True)
    sbn_x = torch.chunk(bn_x.detach().clone(), num_workers, dim=0)
    
    sync_bn = SyncBatchNorm(hid_dim)
    processes = []
    for rank in range(num_workers):
        chunk = sbn_x[rank].detach()
        chunk.requires_grad = True
        queue = ctx.Queue()
        process = ctx.Process(target=init_process, args=(chunk, rank, num_workers, sync_bn, queue))
        process.start()
        processes.append((process, queue))
    outputs = []
    for process, queue in processes:
        process.join()
        output = queue.get()
        outputs.append(output)
    sync_bn_outputs = torch.cat(outputs, dim=0)
    sync_bn_outputs.requires_grad = True
    
    bn = torch.nn.BatchNorm1d(hid_dim)
    outp_bn = bn(bn_x)
    
    sync_bn_outputs[:batch_size // 2, ...].sum().backward()
    outp_bn[:batch_size // 2, ...].sum().backward()
    sbn_x = torch.cat(sbn_x, dim=0)
    assert torch.allclose(sync_bn_outputs, outp_bn, atol=1e-3, rtol=0)
    assert torch.allclose(sbn_x, bn_x, atol=1e-3, rtol=0)
