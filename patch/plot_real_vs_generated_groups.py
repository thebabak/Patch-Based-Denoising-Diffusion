import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def to_train_scale(x, mean, std):
    z = (x - mean) / (std + 1e-8)
    z = np.clip(z, -5.0, 5.0) / 5.0
    return z


def global_mean_std(real_paths):
    arrs = []
    for pth in real_paths.values():
        a = np.load(pth).astype(np.float32)
        a = a[np.isfinite(a)]
        if a.size:
            arrs.append(a.reshape(-1))
    if not arrs:
        return np.nan, np.nan
    big = np.concatenate(arrs, axis=0)
    return float(big.mean()), float(big.std() + 1e-8)


def main():
    real_paths = {
        "adhd": r"synthetic_npy\adhd_40.npy",
        "bd": r"synthetic_npy\bd_49_clean.npy",
        "health": r"synthetic_npy\health_121.npy",
        "schz": r"synthetic_npy\schz_27.npy",
    }
    gen_root = "synthetic_generated_patch260107"

    mean, std = global_mean_std(real_paths)
    print(f"Global mean/std (finite only): mean={mean:.6f}, std={std:.6f}")

    groups = ["adhd", "bd", "health", "schz"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, cls in zip(axes, groups):
        real = np.load(real_paths[cls]).astype(np.float32)
        real_ts = to_train_scale(real, mean, std).reshape(-1)

        cls_dir = os.path.join(gen_root, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        if not files:
            print(f"No generated files for {cls} in {cls_dir}")
            continue
        gen_imgs = [np.load(f).astype(np.float32) for f in files]
        gen = np.stack(gen_imgs, axis=0).reshape(-1)

        ax.hist(real_ts, bins=50, range=(-1, 1), density=True, alpha=0.5, label="real (train)")
        ax.hist(gen, bins=50, range=(-1, 1), density=True, alpha=0.5, label="generated")
        ax.set_title(cls)
        ax.set_xlim(-1, 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    outdir = gen_root
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "real_vs_generated_group_histograms.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
