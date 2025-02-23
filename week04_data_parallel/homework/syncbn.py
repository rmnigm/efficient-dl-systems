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
        b, c, l = input.shape
        input_sum = torch.sum(input, dim=(0, 2))
        input_sum_sq = torch.sum(input ** 2, dim=(0, 2))
        buffer = torch.empty((c, 2), device=input.device)
        buffer[:, 0] = input_sum
        buffer[:, 1] = input_sum_sq
        size = b * l * dist.get_world_size()
        dist.all_reduce(buffer.detach(), op=dist.ReduceOp.SUM)
        mean = buffer[:, 0] / size
        mean_sq = buffer[:, 1] / size
        std = torch.sqrt(mean_sq - mean ** 2)
        output = (input - mean[None, :, None]) / (std + eps)[None, :, None]
        with torch.no_grad():
            running_mean = running_mean.to(input.device)
            running_std = running_std.to(input.device)  
            running_mean = running_mean * (1 - momentum) + momentum * mean
            running_std = running_std * (1 - momentum) + momentum * std * size / (size - 1)
        ctx.save_for_backward(output, mean, 1 / (std + eps))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        output, mean, inv_std = ctx.saved_tensors
        inv_std, normalized, bias_x = inv_std, output, mean
        b, c, _ = normalized.shape

        grad_output_norm = grad_output * inv_std
        grad_var = (-0.5 * (grad_output_norm * normalized * inv_std)).sum(0)
        grad_mean = grad_output_norm.sum(0) + 2 * bias_x.sum(0) / b

        dist.all_reduce((info_grad := torch.cat([grad_mean, grad_var])), op=dist.ReduceOp.SUM)
        grad_mean, grad_var = info_grad.split(c)

        return grad_output_norm + (2 * grad_var * bias_x - grad_mean) / b, None, None, None, None


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
        if len(input.shape) == 2:
            input = input[:, :, None]
        if not self.training:
            output = (input - self.running_mean[None, :, None].to(input.device)) / (self.running_std[None, :, None].to(input.device) + self.eps)
        else:
            output = sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        if len(output.shape) == 3:
            output = output[:, :, 0]
        return output
