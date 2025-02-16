import torch


class GradScaler:
    def __init__(self, mode: str = "dynamic", scale: float = 2.0 ** 16):
        assert mode in ["dynamic", "static"]
        self._mode = mode
        self._scale = scale
        self._inv_scale = 1 / self._scale
        self._increase_factor = 2.0
        self._decrease_factor = 0.5
        self._no_overflow_interval = 2000
        self._num_iters_no_overflow = 0
        self._overflow = False

    def scale(self, loss):
        return loss * self._scale
    
    def _static_step(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(self._inv_scale)
        optimizer.step()
    
    def _dynamic_step(self, optimizer):
        self._overflow = False
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if p.grad.data.isnan().any() or p.grad.data.isinf().any():
                        self._overflow = True
                        break
                    else:
                        p.grad.data.mul_(self._inv_scale)
            if self._overflow:
                break
        if self._overflow:
            optimizer.zero_grad()
        else:
            optimizer.step()

    def step(self, optimizer):
        if self._mode == "static":
            self._static_step(optimizer)
        elif self._mode == "dynamic":
            self._dynamic_step(optimizer)

    def update(self):
        if self._mode == "static":
            return
        if self._overflow:
            self._num_iters_no_overflow = 0
            self._scale = max(self._scale * self._decrease_factor, 1)
            self._inv_scale = 1 / self._scale
        else:
            self._num_iters_no_overflow += 1
            if self._num_iters_no_overflow >= self._no_overflow_interval:
                self._scale = min(self._scale * self._increase_factor, 1)
                self._inv_scale = 1 / self._scale
                self._num_iters_no_overflow = 0
