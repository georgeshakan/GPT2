from gpt.model import GPT
from gpt.config import SEQUENCE_LENGTH
import torch
from gpt.config import SEQUENCE_LENGTH, VOCAB_SIZE, N_EMBED
import torch


def test_model():
    model = GPT(n_embed=N_EMBED, vocab_size=VOCAB_SIZE)
    x = torch.randint(0, VOCAB_SIZE, (4, SEQUENCE_LENGTH))
    logits = model(x)
    assert logits.shape == (4, SEQUENCE_LENGTH, VOCAB_SIZE)
    assert model.pos_embed.size() == (1, SEQUENCE_LENGTH, N_EMBED)
