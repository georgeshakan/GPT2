import torch
from .config import SEQUENCE_LENGTH
import os
import tiktoken
import numpy as np


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.encoder = tiktoken.encoding_for_model("gpt2")

    def _get_batch(self, encoding, batch_size):
        data = torch.tensor(encoding).long()
        ind = torch.randint(len(data) - SEQUENCE_LENGTH - 1, (batch_size,))
        X = torch.stack([data[i : i + SEQUENCE_LENGTH] for i in ind])
        y = torch.stack([data[i + 1 : i + SEQUENCE_LENGTH + 1] for i in ind])
        return X, y

    def get_batch(self, batch_size, train=True):
        sub_dir = "train" if train else "val"
        data_dir = os.path.join(self.data_dir, sub_dir)

        while True:
            file = np.random.choice(os.listdir(data_dir))

            file_path = os.path.join(data_dir, file)
            with open(file_path, "r") as f:
                data = f.read()
            encoding = self.encoder.encode(data)
            if len(encoding) > SEQUENCE_LENGTH:
                break

        X, y = self._get_batch(encoding, batch_size)

        return X, y
