import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        b, c = input.shape
        input_sum = torch.sum(input, dim=0)
        input_sum_sq = torch.sum(input ** 2, dim=0)
        buffer = torch.empty((c, 2), device=input.device)
        buffer[:, 0] = input_sum
        buffer[:, 1] = input_sum_sq
        size = b * dist.get_world_size()
        dist.all_reduce(buffer.detach(), op=dist.ReduceOp.SUM)
        mean = buffer[:, 0] / size
        mean_sq = buffer[:, 1] / size
        std = torch.sqrt(mean_sq - mean ** 2)
        output = (input - mean[None, :]) / (std + eps)[None, :]
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * mean)
            running_std.mul_(1 - momentum).add_(momentum * std * size / (size - 1))
        ctx.save_for_backward(input - mean[None, :], output, 1 / (std + eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shifted_input, output, inv_std = ctx.saved_tensors
        b, c = output.shape
        size = b * dist.get_world_size()

        grad_output *= inv_std
        grad_std = (-0.5 * (grad_output * output * inv_std)).sum(0)
        grad_mean = grad_output.sum(0) + 2 * shifted_input.sum(0) / size
        buffer = torch.empty((2, c), device=output.device)
        buffer[0], buffer[1] = grad_mean, grad_std
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        grad_mean, grad_std = buffer[0], buffer[1]
        grad_input = grad_output
        grad_input += (2 * grad_std * shifted_input - grad_mean) / size
        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        self.running_mean = self.running_mean.to(input.device)
        self.running_std = self.running_std.to(input.device)
        if not self.training:
            output = (input - self.running_mean[None, :]) / (self.running_std[None, :] + self.eps)
        else:
            output = sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        return output
