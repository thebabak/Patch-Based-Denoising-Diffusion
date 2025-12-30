import numpy as np

src = r"synthetic_npy\bd_49.npy"
dst = r"synthetic_npy\bd_49_clean.npy"

arr = np.load(src)

print("Before:", arr.shape, "finite% =", np.isfinite(arr).mean()*100)

# Option 1 (safe): replace non-finite with 0
arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

print("After:", arr.shape, "finite% =", np.isfinite(arr).mean()*100)

np.save(dst, arr)
print("Saved:", dst)
