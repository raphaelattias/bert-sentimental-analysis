from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.datamodules.rotten_datamodule import RottenDataModule


class TextDataSet(Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = {'text': self.texts[idx], 'label': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)


class TextDataModule(RottenDataModule):
    def __init__(self, texts_list=["test"], labels_list=[0], batch_size: int = 64, num_workers: int = 0, pin_memory: bool = False
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.data_test = TextDataSet(texts_list, labels_list)
