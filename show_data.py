# show_data.py
# Loads your .npy files, prints summary stats, and plots a few samples per class.

import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

x = np.load("adhd_40.npy")[0]   # (116,152)

row_std = x.std(axis=1)         # variation across columns per row
col_std = x.std(axis=0)         # variation across rows per column

print("Row-std (mean):", row_std.mean(), "min:", row_std.min(), "max:", row_std.max())
print("Col-std (mean):", col_std.mean(), "min:", col_std.min(), "max:", col_std.max())

# show top 10 most-varying rows
top = np.argsort(-row_std)[:10]
print("Most varying rows:", top)

FILES = {
    "health": "health_121.npy",
    "schz":   "schz_27.npy",
    "adhd":   "adhd_40.npy",
    "bd":     "bd_49.npy",
}

def summarize(name, arr):
    return {
        "group": name,
        "shape": arr.shape,
        "dtype": arr.dtype,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }

def main(data_dir="."):
    all_arrays = {}

    # Load + summary
    print("=== DATASET SUMMARY ===")
    for name, fn in FILES.items():
        path = os.path.join(data_dir, fn)
        arr = np.load(path)
        all_arrays[name] = arr

        s = summarize(name, arr)
        print(
            f"{s['group']:>6} | shape={s['shape']} dtype={s['dtype']} "
            f"min={s['min']:.3f} max={s['max']:.3f} mean={s['mean']:.3f} std={s['std']:.3f} "
            f"NaN={s['nan_count']} Inf={s['inf_count']}"
        )

    # Plot a few samples from each group
    print("\n=== PLOTTING SAMPLES (first/middle/last) ===")
    for name, arr in all_arrays.items():
        n = arr.shape[0]
        idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
        for idx in idxs:
            x = arr[idx]  # (116, 152)
            plt.figure(figsize=(6, 4))
            plt.title(f"{name} sample {idx}  (shape={x.shape})")
            plt.imshow(x, aspect="auto")  # no custom colors as requested
            plt.colorbar()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main(".")
