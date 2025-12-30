# make_fig1_pipeline.py
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, xy, text, w=2.8, h=0.9):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5
    )
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=11)

def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.5))

def main(outdir="paper_assets"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_axis_off()

    # Boxes
    box(ax, (0.5, 1.1), "Patch Extraction\n(patch, stride)")
    box(ax, (3.9, 1.1), "DDPM Training on Patches\n(noise-pred loss)")
    box(ax, (7.3, 1.1), "Overlap Consistency\n(L1 on shared region)")
    box(ax, (10.7, 1.1), "Blending Reconstruction\n(Hann-weighted overlap)")

    # Arrows
    arrow(ax, (3.3, 1.55), (3.9, 1.55))
    arrow(ax, (6.7, 1.55), (7.3, 1.55))
    arrow(ax, (10.1, 1.55), (10.7, 1.55))

    # Optional small notes (below)
    ax.text(0.5, 0.35, "Input: (N,H,W) per group\nNormalize → ~[-1,1]", fontsize=10)
    ax.text(3.9, 0.35, "Optional: coord-channels\nOptional: class conditioning", fontsize=10)
    ax.text(7.3, 0.35, "Reduces seam artifacts\nacross patch borders", fontsize=10)
    ax.text(10.7, 0.35, "Output: full image sample\nsaved as .npy + .png", fontsize=10)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)

    path = os.path.join(outdir, "figure1_pipeline.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)

if __name__ == "__main__":
    main()
