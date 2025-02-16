import math
from collections import defaultdict
from typing import Optional

import torch
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
from tqdm import tqdm

MAX_LENGTH = 640


def get_texts(data_path: str, max_lines: int = 300000):
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
    def __init__(self, texts: list[str], max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        for text in tqdm(texts):
            self.texts.append(self.tokenizer.encode(text, truncation=True, padding=True, max_length=max_length))
            
    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class BigBrainDataset(Dataset):
    def __init__(self, texts: list[str], max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        for text in tqdm(texts):
            self.texts.append(self.tokenizer.encode(text, truncation=True, padding=False, max_length=max_length))

    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class UltraBigBrainDataset(Dataset):
    def __init__(self, texts: list[str], max_length: int = MAX_LENGTH, k: int = 640):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        self.bins = defaultdict(list)
        idx = 0
        for text in tqdm(texts):
            tokens = self.tokenizer.encode(text, truncation=True, padding=False, max_length=max_length)
            self.texts.append(tokens)
            self.bins[math.ceil(len(tokens) / k)].append(idx)
            idx += 1

    def __getitem__(self, idx: int):
        return self.texts[idx]
    
    def __len__(self):
        return len(self.texts)


class UltraDuperBigBrainDataset(IterableDataset):
    def __init__(self, texts: list[str], max_length: int = MAX_LENGTH, seed: int = 42):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = []
        for text in tqdm(texts):
            self.texts.append(self.tokenizer.encode(text, truncation=True, padding=False, max_length=max_length))
        random.Random(seed).shuffle(self.texts)
        self.max_length = max_length
        
    def __iter__(self):
        current_text = []
        current_lengths = []
        for text in self.texts:
            tl = len(text)
            current_text += text
            excess = len(current_text) - self.max_length
            if excess < 0:
                current_lengths.append(tl)
            else:
                if excess > 0:
                    current_text = current_text[:-excess]
                    current_lengths.append(tl - excess)
                yield current_text, current_lengths
                current_text = []
                current_lengths = []
        yield current_text, current_lengths


def collate_batch(
    batch: list[list[int]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor]:
    """Pad each sequence of the incoming sequences list"""
    if max_length is None:
        max_length = max(len(text) for text in batch)
    texts = torch.full((len(batch), max_length), 0)
    for i, text in enumerate(batch):
        texts[i, :len(text)] = torch.tensor(text, dtype=torch.int64)
    return texts


def build_attn_mask(size: int, lengths: list[int]) -> torch.Tensor:
    mask = lambda size: torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    attn_mask = torch.full((size, size), float("-inf"))
    start = 0
    for length in lengths:
        end = start + length
        attn_mask[start:end, start:end] = mask(length)
        start = end
    return attn_mask


def collate_packed_batch(
    batch: tuple[list[int], list[int]], max_length: Optional[int] = None
) -> tuple[torch.Tensor]:
    """Pad each sequence of the incoming sequences list"""
    text, lengths = batch
    text_tensor = torch.tensor(text, dtype=torch.int64)[None, :]
    attn_mask = build_attn_mask(len(text), lengths)
    return text_tensor, attn_mask


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, bins: dict, seed: int = 42, max_length: Optional[int] = MAX_LENGTH):
        self.bins = bins
        self.batch_size = batch_size
        self.num_batches = 0
        self.rng = random.Random(seed)
        self.bin_lengths = []
        for k, bin in self.bins.items():
            self.bin_lengths.append(k)
            self.num_batches += math.ceil(len(bin) / batch_size)
            self.rng.shuffle(bin)
        self.rng.shuffle(self.bin_lengths)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for k in self.bin_lengths:
            bin = self.bins[k]
            for i in range(0, len(bin), self.batch_size):
                indices = bin[i:i+self.batch_size]
                yield indices
