from gpt.config import N_EMBED, N_HEADS


def test_embed_and_n_heads():
    assert N_EMBED % N_HEADS == 0, "N_EMBED must be divisible by N_HEADS"
