# make_fig2_epoch_grid.py
import os, glob
import numpy as np
import matplotlib.pyplot as plt

def load_img(path):
    a = np.load(path)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return a

def main(outdir="paper_assets", samples_dir="runs_proposal_patchddpm", vmin=-1.0, vmax=1.0):
    os.makedirs(outdir, exist_ok=True)

    epoch_paths = []
    for e in range(1, 11):
        p = os.path.join(samples_dir, f"sample_epoch_{e}.npy")
        if os.path.exists(p):
            epoch_paths.append(p)

    final_path = os.path.join(samples_dir, "sample_final.npy")
    if os.path.exists(final_path):
        epoch_paths.append(final_path)

    if len(epoch_paths) == 0:
        raise FileNotFoundError(f"No sample_epoch_*.npy found in: {samples_dir}")

    imgs = [load_img(p) for p in epoch_paths]
    titles = [os.path.splitext(os.path.basename(p))[0] for p in epoch_paths]

    n = len(imgs)
    fig_w = max(12, 2.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    save_path = os.path.join(outdir, "figure2_epoch_samples.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", save_path)

if __name__ == "__main__":
    main()

