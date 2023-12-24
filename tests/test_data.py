from gpt.model import GPT
from gpt.config import SEQUENCE_LENGTH
import torch
from gpt.config import DATA_DIR, SEQUENCE_LENGTH


from gpt.data import DataLoader

import torch


def test_data_loader():
    data = DataLoader(DATA_DIR)

    X, y = data.get_batch(6)
    assert X.shape == (6, SEQUENCE_LENGTH)
    assert y.shape == (6, SEQUENCE_LENGTH)
    X, y = data.get_batch(1)
    assert X.shape == (1, SEQUENCE_LENGTH)
    assert y.shape == (1, SEQUENCE_LENGTH)
