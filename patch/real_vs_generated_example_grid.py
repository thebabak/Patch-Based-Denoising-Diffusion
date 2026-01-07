import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_real_example(path, idx=0):
    arr = np.load(path).astype(np.float32)  # (N,H,W)
    if arr.ndim != 3:
        raise ValueError(f"Expected (N,H,W) at {path}, got {arr.shape}")
    img = arr[idx]
    # assume raw 0-255; normalise to [0,1]
    vmin, vmax = 0.0, 255.0
    img = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return img


def load_generated_example(gen_dir, mean, std, fname="sample_0001.npy"):
    p = os.path.join(gen_dir, fname)
    if not os.path.exists(p):
        # fall back to any .npy in the directory
        npys = [f for f in os.listdir(gen_dir) if f.endswith(".npy")]
        if not npys:
            raise FileNotFoundError(f"No .npy files found in {gen_dir}")
        p = os.path.join(gen_dir, sorted(npys)[0])
    arr = np.load(p).astype(np.float32)
    # expected shape (H,W) or (1,H,W) in train scale [-1,1]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W) at {p}, got {arr.shape}")
    # inverse normalisation: approximate 0-255, then map to [0,1]
    I = 5.0 * arr * std + mean  # see Eq. (1) in paper
    I = np.clip(I, 0.0, 255.0)
    img = np.clip(I / 255.0, 0.0, 1.0)
    return img


def main():
    real_paths = {
        "adhd": r"synthetic_npy\adhd_40.npy",
        "bd": r"synthetic_npy\bd_49_clean.npy",
        "health": r"synthetic_npy\health_121.npy",
        "schz": r"synthetic_npy\schz_27.npy",
    }
    gen_root = "synthetic_generated_patch260107"

    # load global mean/std from trained checkpoint (same as used in training)
    ckpt_path = os.path.join("runs_patch260107", "final.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mean = float(ckpt["mean"])
    std = float(ckpt["std"])

    groups = ["adhd", "bd", "health", "schz"]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)

    for col, cls in enumerate(groups):
        # real (top row)
        real_img = load_real_example(real_paths[cls], idx=0)
        ax_r = axes[0, col]
        ax_r.imshow(real_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_r.set_title(f"{cls} real", fontsize=9)
        ax_r.axis("off")

        # generated (bottom row), mapped back to ~0-255 then [0,1]
        gen_dir = os.path.join(gen_root, cls)
        gen_img = load_generated_example(gen_dir, mean, std)
        ax_g = axes[1, col]
        ax_g.imshow(gen_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_g.set_title(f"{cls} generated", fontsize=9)
        ax_g.axis("off")

    outdir = gen_root
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "real_vs_generated_example_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
