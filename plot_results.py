import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="runs/latest", help="Folder containing losses.csv")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save PNG (e.g., results.png)")
    args = ap.parse_args()

    csv_path = os.path.join(args.run_dir, "losses.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Train once with the updated training script to create it."
        )

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7,4.5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Patch Diffusion training curve")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"Saved plot to {args.save}")
    plt.show()

if __name__ == "__main__":
    main()
