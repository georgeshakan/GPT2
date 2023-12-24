import torch
import time
from torch import nn
from .model import GPT
from .data import DataLoader
from .config import (
    SEQUENCE_LENGTH,
    STEPS,
    BATCH_SIZE,
    DATA_DIR,
    N_EMBED,
    DEVICE,
    VOCAB_SIZE,
)


def trainer(checkpoint=None):
    start_time = time.time()

    data = DataLoader(DATA_DIR)

    model = GPT(n_embed=N_EMBED, vocab_size=VOCAB_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(DEVICE)

    initial_step = 0
    if checkpoint is not None:
        checkpoint_torch = torch.load(f"checkpoints/{checkpoint}.pt")
        model.load_state_dict(checkpoint_torch["model_state_dict"])
        optimizer.load_state_dict(checkpoint_torch["optimizer_state_dict"])
        initial_step = int(checkpoint.split(".")[0])

    loss_fn = nn.CrossEntropyLoss()

    for step in range(initial_step, initial_step + STEPS):
        model.train()

        X, y = data.get_batch(BATCH_SIZE)
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        model_output = model(X).view(BATCH_SIZE * SEQUENCE_LENGTH, -1)

        loss = loss_fn(model_output, y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if step % 1000 == 0 and step != initial_step:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoints/{step}.pt",
            )

            with torch.inference_mode():
                val_batch_size = 200
                val_X, val_y = data.get_batch(val_batch_size, train=False)
                val_X = val_X.to(DEVICE)
                val_y = val_y.to(DEVICE)
                model_output = model(val_X).view(val_batch_size * SEQUENCE_LENGTH, -1)
                val_loss = loss_fn(model_output, val_y.view(-1))
                print("time:", time.time() - start_time)
                print("loss:", val_loss.item())
