import math
from collections import defaultdict
from typing import Optional

import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer


MAX_LENGTH = 640


def yield_texts(data_path: str, max_lines: int = 5000):
    num_lines = 0
    with open(data_path, "r") as f:
        for line in f:
            text = line.strip()
            if text:
                yield text
            num_lines += 1
            if num_lines > max_lines:
                break


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        for text in yield_texts(data_path):
            self.texts.append(self.tokenizer.encode(text, truncation=True, padding=True, max_length=max_length))
            
    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        for text in yield_texts(data_path):
            self.texts.append(self.tokenizer.encode(text, truncation=True, padding=False, max_length=max_length))

    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, k: int = 640):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        self.bins = defaultdict(list)
        idx = 0
        for text in yield_texts(data_path):
            tokens = self.tokenizer.encode(text, truncation=True, padding=False, max_length=max_length)
            self.texts.append(tokens)
            self.bins[math.ceil(len(tokens) / k)].append(idx)
            idx += 1

    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __iter__(self):
        pass


def collate_batch(
    batch: list[list[int]], max_length: Optional[int] = None
) -> tuple[torch.Tensor]:
    """Pad each sequence of the incoming sequences list"""
    if max_length is None:
        max_length = max(len(text) for text in batch)
    texts = torch.full((len(batch), max_length), 0)
    for i, text in enumerate(batch):
        texts[i, :len(text)] = torch.tensor(text, dtype=torch.int64)
    return texts


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, bins: dict, seed: int = 42, max_length: Optional[int] = MAX_LENGTH):
        self.bins = bins
        self.bin_lengths = list(bins.keys())
        self.batch_size = batch_size
        self.num_batches = 0
        self.rng = random.Random(seed)
        for bin in self.bins.values():
            self.num_batches += math.ceil(len(bin) / batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        iterable = []
        self.rng.shuffle(self.bin_lengths)
        for k in self.bin_lengths:
            bin = self.bins[k]
            self.rng.shuffle(bin)
            for i in range(0, len(bin), self.batch_size):
                indices = bin[i:i+self.batch_size]
                iterable.append(indices)
        return iter(iterable)
