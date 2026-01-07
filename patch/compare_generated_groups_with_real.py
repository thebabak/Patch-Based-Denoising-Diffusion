import os
import glob
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
    # Real data
    real_paths = {
        "adhd": r"synthetic_npy\adhd_40.npy",
        "bd": r"synthetic_npy\bd_49_clean.npy",
        "health": r"synthetic_npy\health_121.npy",
        "schz": r"synthetic_npy\schz_27.npy",
    }

    # Generated per-group output root (from generate_group_samples.py)
    gen_root = "synthetic_generated_patch260107"

    mean, std = global_mean_std(real_paths)
    print(f"Global mean/std (finite only): mean={mean:.6f}, std={std:.6f}")

    rows = []

    # Real: raw + train-scale per group
    for cls, pth in real_paths.items():
        arr = np.load(pth).astype(np.float32)  # (N,H,W)
        rows.append({
            "group": "REAL_RAW_0_255", "name": cls, "path": pth,
            "shape": str(arr.shape), "N_images": int(arr.shape[0]),
            **stats(arr)
        })

        arr_ts = to_train_scale(arr, mean, std)
        rows.append({
            "group": "REAL_TRAIN_SCALE_-1_1", "name": cls, "path": pth,
            "shape": str(arr_ts.shape), "N_images": int(arr_ts.shape[0]),
            **stats(arr_ts)
        })

    # Generated: train-scale, many samples per group
    for cls in ["adhd", "bd", "health", "schz"]:
        cls_dir = os.path.join(gen_root, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        if not files:
            print(f"No generated files found for group {cls} in {cls_dir}")
            continue
        imgs = [np.load(f).astype(np.float32) for f in files]
        arr = np.stack(imgs, axis=0)  # (N,H,W)
        rows.append({
            "group": "GENERATED_GROUP_-1_1", "name": cls, "path": cls_dir,
            "shape": str(arr.shape), "N_images": int(arr.shape[0]),
            **stats(arr)
        })

    df = pd.DataFrame(rows)

    out_csv = os.path.join(gen_root, "real_vs_generated_groups.csv")
    df.to_csv(out_csv, index=False)

    show_cols = ["group", "name", "N_images", "shape", "bad_pct", "mean", "std", "min", "max", "p01", "p99"]
    print("\n=== REAL vs GENERATED GROUPS (fair, per group) ===")
    print(df[show_cols].to_string(index=False))
    print("\n✅ Saved:", out_csv)


if __name__ == "__main__":
    main()
