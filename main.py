from gpt import train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    train.trainer(args.checkpoint)
