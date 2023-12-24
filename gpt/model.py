from torch import nn
from .config import SEQUENCE_LENGTH, DROPOUT_RATE, N_HEADS, N_BLOCKS
import torch


class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class CasualSelfAttention(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.embed = nn.Linear(n_embed, 3 * n_embed)

    def forward(self, x):
        B, T, C = x.size()
        # x = self.embed(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.n_embed = n_embed
        self.n_head = num_heads
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.c_attn = nn.Linear(n_embed, 3 * n_embed)
        self.attn = torch.nn.MultiheadAttention(
            n_embed, num_heads, dropout=DROPOUT_RATE, batch_first=True
        )
        self.layer_norm_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = self.layer_norm_1(x)

        # get query, key, value for
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = x + self.attn(q, k, v)[0]
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class GPT(nn.Module):
    """
    GPT model:
    """

    def __init__(self, n_embed, vocab_size):
        super().__init__()
        self.n_embed = n_embed
        self.embedding_layer = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(SEQUENCE_LENGTH, n_embed)

        self.blocks = [Block(n_embed, N_HEADS) for _ in range(N_BLOCKS)]

        self.layer_norm = nn.LayerNorm(n_embed)
        self.final_linear_layer = nn.Linear(n_embed, vocab_size)

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        print(x.device)
        pos = torch.arange(
            0, SEQUENCE_LENGTH, dtype=torch.long, device=x.device
        ).unsqueeze(0)
        self.pos_embed = self.positional_embedding(pos)

        x = embeddings + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        logits = self.final_linear_layer(x)
        return logits
