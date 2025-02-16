from enum import Enum
from time import time
import functools

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm, trange

from task2.dataset import (
    BrainDataset,
    BigBrainDataset,
    UltraBigBrainDataset,
    UltraBigBrainBatchSampler,
    UltraDuperBigBrainDataset,
    collate_batch,
    collate_packed_batch,
    MAX_LENGTH,
)
from task2.transformer import TransformerModel, generate_square_subsequent_mask


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model() -> torch.nn.Module:
    vocab_size = AutoTokenizer.from_pretrained("bert-base-uncased").vocab_size
    model = TransformerModel(vocab_size, 768, 8, 1024, 12)
    return model


def run_epoch_packed_batch(model, device, dataset, collate_packed_batch):
    train_loader = DataLoader(dataset, collate_fn=collate_packed_batch, batch_size=None)
    pad_token_ratio, timers = [], []
    data_iter = iter(train_loader)
    while True:
        start_time = time()
        try:
            batch, src_mask = next(data_iter)
        except StopIteration:
            break
        with torch.no_grad():
            model(batch.to(device), src_mask.to(device))
        torch.cuda.synchronize()
        end_time = time()
        pad_token_ratio.append(((batch == 0).sum() / torch.numel(batch)).item())
        timers.append(end_time - start_time)
    # skip warmup 5 iterations
    return timers[5:], pad_token_ratio[5:]


def run_epoch_inner(model, device, dataset, collate_batch_fn, sampler = None):
    if sampler is not None:
        train_loader = DataLoader(dataset, collate_fn=collate_batch_fn, batch_sampler=sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch_fn)
    pad_token_ratio, timers = [], []
    data_iter = iter(train_loader)
    for _ in trange(len(train_loader)):
        start_time = time()
        batch = next(data_iter)
        src_mask = generate_square_subsequent_mask(batch.shape[1]).to(device)
        with torch.no_grad():
            model(batch.to(device), src_mask)
        torch.cuda.synchronize()
        end_time = time()
        pad_token_ratio.append(((batch == 0).sum() / torch.numel(batch)).item())
        timers.append(end_time - start_time)
    # skip warmup 5 iterations
    return timers[5:], pad_token_ratio[5:]


def run_epoch(data_mode: DataMode, k: int = 640) -> None:
    device = torch.device("cuda:0")
    data_path = "task2/wikitext-103-raw-v1/train.txt"
    model = get_gpt2_model().to(device)
    if data_mode == DataMode.BRAIN:
        train_dataset = BrainDataset(data_path)
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch)
    elif data_mode == DataMode.BIG_BRAIN:
        train_dataset = BigBrainDataset(data_path)
        collate_batch_fn = functools.partial(collate_batch, max_length=None)
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        train_dataset = UltraBigBrainDataset(data_path, k=k)
        sampler = UltraBigBrainBatchSampler(batch_size=16, bins=train_dataset.bins)
        collate_batch_fn = functools.partial(collate_batch, max_length=None)
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn, sampler)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        train_dataset = UltraDuperBigBrainDataset(data_path)
        timers, pad_token_ratio = run_epoch_packed_batch(model, device, train_dataset, collate_packed_batch)
    return pd.DataFrame({"pad_token_ratio": pad_token_ratio, "time": timers}).describe()
