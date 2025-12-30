import os
import numpy as np

def gaussian2d(H, W, cy, cx, sy, sx):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    return np.exp(-(((y - cy) ** 2) / (2 * sy ** 2) + ((x - cx) ** 2) / (2 * sx ** 2)))

def normalize01(a):
    a = a - a.min()
    a = a / (a.max() + 1e-8)
    return a

def make_brain_like(H=128, W=128, seed=None, style="health"):
    rng = np.random.default_rng(seed)

    # base smooth field (low-freq)
    base = np.zeros((H, W), dtype=np.float32)
    for _ in range(rng.integers(3, 7)):
        cy = rng.uniform(0.25*H, 0.75*H)
        cx = rng.uniform(0.25*W, 0.75*W)
        sy = rng.uniform(10, 25)
        sx = rng.uniform(10, 25)
        amp = rng.uniform(0.4, 1.0)
        base += amp * gaussian2d(H, W, cy, cx, sy, sx).astype(np.float32)

    base = normalize01(base)

    # add subtle texture (like MR noise)
    tex = rng.normal(0, 1, (H, W)).astype(np.float32)
    tex = normalize01(tex)
    img = 0.8 * base + 0.2 * tex

    # style differences (simulate 4 groups)
    if style == "health":
        # clean + smoother
        img = 0.9 * base + 0.1 * tex
    elif style == "adhd":
        # slightly darker + extra blobs
        for _ in range(rng.integers(1, 3)):
            cy = rng.uniform(0.2*H, 0.8*H)
            cx = rng.uniform(0.2*W, 0.8*W)
            sy = rng.uniform(6, 14)
            sx = rng.uniform(6, 14)
            img += 0.25 * gaussian2d(H, W, cy, cx, sy, sx).astype(np.float32)
        img = img * 0.85
    elif style == "bd":
        # mild stripe artifact + contrast
        stripes = (np.sin(np.linspace(0, 6*np.pi, W))[None, :] * 0.08).astype(np.float32)
        img = img + stripes
        img = img ** 0.8
    elif style == "schz":
        # add a “lesion-like” bright spot
        cy = rng.uniform(0.35*H, 0.65*H)
        cx = rng.uniform(0.35*W, 0.65*W)
        lesion = gaussian2d(H, W, cy, cx, rng.uniform(4, 9), rng.uniform(4, 9)).astype(np.float32)
        img = img + 0.35 * lesion

    img = normalize01(img)

    # map to a more MR-like range (still float)
    img = (img * 255.0).astype(np.float32)
    return img

def make_group(style, N=50, H=128, W=128, seed=0):
    arr = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        arr[i] = make_brain_like(H, W, seed=seed+i, style=style)
    return arr

if __name__ == "__main__":
    outdir = "synthetic_npy"
    os.makedirs(outdir, exist_ok=True)

    H, W = 128, 128

    # choose counts like your original dataset
    data_adhd   = make_group("adhd",   N=40,  H=H, W=W, seed=100)
    data_bd     = make_group("bd",     N=49,  H=H, W=W, seed=200)
    data_health = make_group("health", N=121, H=H, W=W, seed=300)
    data_schz   = make_group("schz",   N=27,  H=H, W=W, seed=400)

    np.save(os.path.join(outdir, "adhd_40.npy"), data_adhd)
    np.save(os.path.join(outdir, "bd_49.npy"), data_bd)
    np.save(os.path.join(outdir, "health_121.npy"), data_health)
    np.save(os.path.join(outdir, "schz_27.npy"), data_schz)

    print("✅ Saved synthetic dataset:")
    print(" ", os.path.join(outdir, "adhd_40.npy"), data_adhd.shape)
    print(" ", os.path.join(outdir, "bd_49.npy"), data_bd.shape)
    print(" ", os.path.join(outdir, "health_121.npy"), data_health.shape)
    print(" ", os.path.join(outdir, "schz_27.npy"), data_schz.shape)
