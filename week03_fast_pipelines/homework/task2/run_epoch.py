from enum import Enum
from time import time
import functools

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from task2.dataset import (
    BrainDataset,
    BigBrainDataset,
    UltraBigBrainDataset,
    UltraBigBrainBatchSampler,
    UltraDuperBigBrainDataset,
    collate_batch,
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


def run_epoch_inner(model, device, dataset, collate_batch_fn, sampler = None, custom_attn_mask = False):
    if sampler is not None:
        train_loader = DataLoader(dataset, collate_fn=collate_batch_fn, batch_sampler=sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch_fn)
    pad_token_ratio, timers = [], []
    for batch in tqdm(train_loader):
        src_mask = generate_square_subsequent_mask(batch.shape[1]).to(device)
        pad_token_ratio.append(((batch == 0).sum() / torch.numel(batch)).item())
        start_time = time()
        with torch.no_grad():
            model(batch.to(device), src_mask)
        torch.cuda.synchronize()
        end_time = time()
        timers.append(end_time - start_time)
    # skip warmup 5 iterations
    return timers[5:], pad_token_ratio[5:]


def run_epoch(data_mode: DataMode, k: int = 640) -> None:
    device = torch.device("cuda:0")
    data_path = "task2/wikitext-103-raw-v1/train.txt"
    model = get_gpt2_model().to(device)
    if data_mode == DataMode.BRAIN:
        train_dataset = BrainDataset(data_path)
        collate_batch_fn = functools.partial(collate_batch, max_length=MAX_LENGTH)
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn)
    elif data_mode == DataMode.BIG_BRAIN:
        train_dataset = BigBrainDataset(data_path)
        collate_batch_fn = collate_batch
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        train_dataset = UltraBigBrainDataset(data_path, k=k)
        sampler = UltraBigBrainBatchSampler(batch_size=16, bins=train_dataset.bins)
        collate_batch_fn = collate_batch
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn, sampler)
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        train_dataset = UltraBigBrainDataset(data_path)
        collate_batch_fn = collate_batch
        timers, pad_token_ratio = run_epoch_inner(model, device, train_dataset, collate_batch_fn)
    return pd.DataFrame({"pad_token_ratio": pad_token_ratio, "time": timers}).describe()
