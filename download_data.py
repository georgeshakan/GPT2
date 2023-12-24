if __name__ == "__main__":
    from datasets import load_dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", type=str, required=True)
    args = parser.parse_args()

    ds = load_dataset("open-web-math/open-web-math", streaming=True)
    for j, d in enumerate(ds["train"]):
        if j % 1000 == 20:
            print(j)
        extension = "val"
        if j < 0.8 * args.num_files:
            extension = "train"

        with open(f"data/{extension}/{j+1}.txt", "w") as f:
            f.write(d["text"])
