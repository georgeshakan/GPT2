import os
import torch
import tiktoken

encoder = tiktoken.encoding_for_model("gpt2")


VOCAB_SIZE = encoder.n_vocab
SEQUENCE_LENGTH = 8
BATCH_SIZE = 4
STEPS = 10
N_EMBED = 256
N_HEADS = 8
DROPOUT_RATE = 0.1
N_BLOCKS = 2

DATA_DIR = "data"

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
