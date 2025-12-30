import os, glob
import numpy as np
import pandas as pd

def stats(arr):
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

def to_train_scale(x, mean, std):
    # same rule as your dataset __getitem__
    z = (x - mean) / (std + 1e-8)
    z = np.clip(z, -5.0, 5.0) / 5.0
    return z

def global_mean_std(real_paths):
    arrs = []
    for p in real_paths.values():
        a = np.load(p).astype(np.float32)
        a = a[np.isfinite(a)]
        if a.size:
            arrs.append(a.reshape(-1))
    if not arrs:
        return np.nan, np.nan
    big = np.concatenate(arrs, axis=0)
    return float(big.mean()), float(big.std() + 1e-8)

def load_loss(outdir):
    path = os.path.join(outdir, "loss_history.npy")
    if not os.path.exists(path):
        return None
    hist = np.load(path, allow_pickle=True).item()
    return {
        "final_loss": float(hist["loss"][-1]),
        "final_loss_main": float(hist["loss_main"][-1]),
        "final_loss_cons": float(hist["loss_cons"][-1]),
        "epochs_logged": int(len(hist["epoch"]))
    }

def main():
    outdir = "runs_proposal_patchddpm"
    real_paths = {
        "adhd": r"synthetic_npy\adhd_40.npy",
        "bd": r"synthetic_npy\bd_49.npy",  # change to bd_49_clean.npy after cleaning
        "health": r"synthetic_npy\health_121.npy",
        "schz": r"synthetic_npy\schz_27.npy",
    }

    mean, std = global_mean_std(real_paths)
    print(f"Global mean/std (finite only): mean={mean:.6f}, std={std:.6f}")

    rows = []

    # Real: raw + train-scale
    for cls, p in real_paths.items():
        arr = np.load(p).astype(np.float32)  # (N,H,W)
        rows.append({
            "group": "REAL_RAW_0_255", "name": cls, "path": p,
            "shape": str(arr.shape), "N_images": int(arr.shape[0]),
            **stats(arr)
        })

        arr_ts = to_train_scale(arr, mean, std)
        rows.append({
            "group": "REAL_TRAIN_SCALE_-1_1", "name": cls, "path": p,
            "shape": str(arr_ts.shape), "N_images": int(arr_ts.shape[0]),
            **stats(arr_ts)
        })

    # Generated: already train-scale
    gen_files = sorted(glob.glob(os.path.join(outdir, "sample_epoch_*.npy")))
    gen_files += sorted(glob.glob(os.path.join(outdir, "sample_final.npy")))
    for f in gen_files:
        g = np.load(f).astype(np.float32)
        rows.append({
            "group": "GENERATED_-1_1", "name": os.path.basename(f), "path": f,
            "shape": str(g.shape), "N_images": 1,
            **stats(g)
        })

    df = pd.DataFrame(rows)

    loss = load_loss(outdir)
    if loss:
        for k, v in loss.items():
            df[k] = v

    csv_path = os.path.join(outdir, "real_vs_generated_fair.csv")
    df.to_csv(csv_path, index=False)

    show_cols = ["group","name","N_images","shape","bad_pct","mean","std","min","max","p01","p99"]
    if loss:
        show_cols += ["final_loss","final_loss_main","final_loss_cons","epochs_logged"]

    print("\n=== FAIR REAL vs GENERATED (same scale) ===")
    print(df[show_cols].to_string(index=False))
    print("\n✅ Saved:", csv_path)

if __name__ == "__main__":
    main()
