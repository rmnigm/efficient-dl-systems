# Week 2 Home Assignment

## Bugs

1. Incosistent device in `forward`, `sample` methods from `diffusion.py`
  - everything should be on the same device.
  - found by running all tests bot on `cpu` and `cuda`.
  - `timestep` in `forward`, `x_i` and `z` in `sample`.

2. Errors in formula for `x_t`.
  - `sqrt_one_minus_alpha_prod` needed for correct diffusion process.
  - found by comparing with article and test runs with expected loss.

3. `eps` should be sampled from normal distribution instead of uniform.
  - `torch.rand_like()` changed to `torch.randn_like()`.
  - found by comparing with article and test runs with expected loss.

4. Broadcasting in `forward` of `unet.py`
  - `temb` needs expanded with two more dimensions to enable broadcasting.  
  - found by running all tests (failed due to incosistent dims).

5. Deterministic testing to ensure no flapping in tests with random initialization
  - added a fixed seed fixture with session scope via `conftest.py`.
  - used in all tests, reused in `test_training` to ensure same results with equal hyperparameters.



## Testing Coverage

```sh
pytest --cov=modeling tests/
```

| Name                      | Stmts | Miss | Cover |
|---------------------------|-------|------|-------|
| modeling/__init__.py      | 0     | 0    | 100%  |
| modeling/diffusion.py     | 34    | 6    | 82%  |
| modeling/training.py      | 30    | 6    | 80%  |
| modeling/unet.py          | 68    | 0    | 100%  |
|---------------------------|-------|------|-------|
| **TOTAL**                 | 132   | 12   | 91%  |


## Code changes
- Added `test_training` for integration test of whole pipeline, ensured change of results from different hyperparameters and no change with same.
- Added `conftest.py` for fixed seed fixture.
- Added parametrization with Hydra configs, script fully controlled with `yaml` files.
- Added DVC support for experiments, added random seed fixing in training script for reproducibility.
- Changed training functions to return losses or images for logging.
- Added `wandb` logging to training script `main.py` with images, all parameters and config.
- Updated `requirements.txt` with new dependencies (`uv` wasn't available on YSDA GPU server, did not have time).

---

## How to run
1. Install dependencies:
```sh
pip install -r requirements.txt
```
2. Configure DVC with enabling Hydra configs
```sh
dvc config hydra.enabled True
```
3. Set hyperparameters (or leave unchanged for default versions) in `conf` folder. Resulting parameter set will be available in `params.yaml` after experiment run or in `wandb` as an artifact.
4. Run experiment and wait for finish:
```sh
dvc exp run
```

## Runs
- `wandb` workspace - https://wandb.ai/rmnigm/effdl-homework-01.
- Latest generations are not always the best, ~90 epoch are better!
