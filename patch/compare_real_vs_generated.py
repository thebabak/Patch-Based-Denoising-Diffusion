import os
import glob
import numpy as np
import pandas as pd

def stats(arr: np.ndarray):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    total = arr.size
    fin = int(finite.sum())
    bad = total - fin
    if fin == 0:
        return dict(total=total, bad=bad, bad_pct=100.0, mean=np.nan, std=np.nan,
                    min=np.nan, max=np.nan, p01=np.nan, p99=np.nan)
    v = arr[finite].astype(np.float64)
    return dict(
        total=total,
        bad=bad,
        bad_pct=100.0 * bad / total,
        mean=float(v.mean()),
        std=float(v.std() + 1e-8),
        min=float(v.min()),
        max=float(v.max()),
        p01=float(np.percentile(v, 1)),
        p99=float(np.percentile(v, 99)),
    )

def load_loss_history(outdir):
    path = os.path.join(outdir, "loss_history.npy")
    if not os.path.exists(path):
        return None
    hist = np.load(path, allow_pickle=True).item()
    # last epoch values
    last = {
        "final_loss": float(hist["loss"][-1]),
        "final_loss_main": float(hist["loss_main"][-1]),
        "final_loss_cons": float(hist["loss_cons"][-1]),
        "epochs_logged": int(len(hist["epoch"]))
    }
    return last

def main():
    # ---- EDIT THESE IF NEEDED ----
    outdir = "runs_proposal_patchddpm"
    real_paths = {
        "adhd": r"synthetic_npy\adhd_40.npy",
        "bd": r"synthetic_npy\bd_49.npy",
        "health": r"synthetic_npy\health_121.npy",
        "schz": r"synthetic_npy\schz_27.npy",
    }
    # -----------------------------

    rows = []

    # Real/original stats (per class)
    for cls, p in real_paths.items():
        arr = np.load(p)  # (N,H,W)
        if arr.ndim != 3:
            raise ValueError(f"{p} must be (N,H,W) but got {arr.shape}")
        s = stats(arr)
        rows.append({
            "group": "REAL_ORIGINAL",
            "name": cls,
            "path": p,
            "shape": str(arr.shape),
            "N_images": int(arr.shape[0]),
            **s
        })

    # Generated sample stats (each file)
    gen_files = sorted(glob.glob(os.path.join(outdir, "sample_epoch_*.npy"))) + \
                sorted(glob.glob(os.path.join(outdir, "sample_final.npy")))
    for f in gen_files:
        arr = np.load(f)  # usually (H,W)
        s = stats(arr)
        rows.append({
            "group": "GENERATED_SAMPLE",
            "name": os.path.basename(f),
            "path": f,
            "shape": str(arr.shape),
            "N_images": 1,
            **s
        })

    df = pd.DataFrame(rows)

    # Attach training loss summary (same values repeated for convenience)
    loss = load_loss_history(outdir)
    if loss is not None:
        for k, v in loss.items():
            df[k] = v

    # Save CSV
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "real_vs_generated_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Print a clean view
    view_cols = [
        "group", "name", "N_images", "shape",
        "bad_pct", "mean", "std", "min", "max", "p01", "p99"
    ]
    if loss is not None:
        view_cols += ["final_loss", "final_loss_main", "final_loss_cons", "epochs_logged"]

    print("\n=== REAL vs GENERATED (summary) ===")
    print(df[view_cols].to_string(index=False))
    print(f"\n✅ Saved: {csv_path}")

if __name__ == "__main__":
    main()
