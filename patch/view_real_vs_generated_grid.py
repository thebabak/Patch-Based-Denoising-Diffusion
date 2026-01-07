import os
import numpy as np
import matplotlib.pyplot as plt


def load_first_slice(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3:
        img = arr[0]
    elif arr.ndim == 2:
        img = arr
    else:
        raise ValueError(f"Unexpected shape {arr.shape} for {path}")
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def to_01(img: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    x = (img - vmin) / (vmax - vmin)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def main():
    synth_dir = "synthetic_npy"
    run_dir = "runs_test_patchddpm2"

    real_paths = {
        "adhd": os.path.join(synth_dir, "adhd_40.npy"),
        "bd": os.path.join(synth_dir, "bd_49_clean.npy"),
        "health": os.path.join(synth_dir, "health_121.npy"),
        "schz": os.path.join(synth_dir, "schz_27.npy"),
    }

    gen_files = [
        "sample_epoch_5.npy",
        "sample_epoch_10.npy",
        "sample_epoch_20.npy",
        "sample_final.npy",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    # Row 0: real examples (one per class)
    for i, (cls, p) in enumerate(real_paths.items()):
        ax = axes[0, i]
        if os.path.exists(p):
            img = load_first_slice(p)
            ax.imshow(to_01(img), cmap="gray")
            ax.set_title(f"real: {cls}")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    # Row 1: generated samples from run_dir
    for i, fname in enumerate(gen_files):
        ax = axes[1, i]
        p = os.path.join(run_dir, fname)
        if os.path.exists(p):
            img = load_first_slice(p)
            ax.imshow(to_01(img), cmap="gray")
            ax.set_title(f"gen: {fname}")
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "real_vs_generated_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
