# patch251229.py
# Patch-DDPM + overlap consistency + overlap-aware reconstruction (CUDA/GPU)
# Robust version: detects + fixes NaN/Inf in .npy datasets (the main reason you got nan loss)

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------------
# CUDA
# ----------------------------
def get_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Install CUDA-enabled PyTorch + NVIDIA driver.")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
    return device


# ----------------------------
# Utils: save/show npy + plots
# ----------------------------
def save_npy_as_png(arr: np.ndarray, png_path: str, vmin: float = -1.0, vmax: float = 1.0):
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) or (1,H,W). Got {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    img01 = (arr - vmin) / (vmax - vmin + 1e-8)
    img01 = np.clip(img01, 0.0, 1.0)

    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    plt.imsave(png_path, img01, cmap="gray")


def plot_epoch_curves(outdir: str, history: dict, show: bool = False):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "loss_history.npy"), history)

    epochs = history["epoch"]
    plt.figure()
    plt.plot(epochs, history["loss"], label="total loss")
    plt.plot(epochs, history["loss_main"], label="mse (noise pred)")
    plt.plot(epochs, history["loss_cons"], label="consistency")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    png_path = os.path.join(outdir, "loss_curve.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"📈 Saved plot: {png_path}")


def save_epoch_sample(full_tensor: torch.Tensor, outdir: str, name: str):
    os.makedirs(outdir, exist_ok=True)
    sample_np = full_tensor.squeeze(0).detach().cpu().numpy()  # (H,W)
    sample_np = np.nan_to_num(sample_np, nan=0.0, posinf=0.0, neginf=0.0)

    npy_path = os.path.join(outdir, f"{name}.npy")
    png_path = os.path.join(outdir, f"{name}.png")

    np.save(npy_path, sample_np)
    save_npy_as_png(sample_np, png_path, vmin=-1.0, vmax=1.0)
    print(f"🖼️ Saved sample: {npy_path} and {png_path}")
    return npy_path, png_path


# ----------------------------
# Data (robust)
# ----------------------------
def _load_npy_float(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path} must be (N,H,W), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _finite_stats(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    n_total = arr.size
    n_finite = int(finite.sum())
    n_bad = n_total - n_finite
    if n_finite == 0:
        return dict(total=n_total, finite=n_finite, bad=n_bad, fmin=None, fmax=None, fmean=None, fstd=None)
    vals = arr[finite]
    return dict(
        total=n_total,
        finite=n_finite,
        bad=n_bad,
        fmin=float(vals.min()),
        fmax=float(vals.max()),
        fmean=float(vals.mean()),
        fstd=float(vals.std() + 1e-8),
    )


def _fix_nonfinite(arr: np.ndarray, mode: str = "mean") -> np.ndarray:
    # Replace NaN/Inf with finite mean (or 0 if no finite values)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any() and mode == "mean":
        fill = float(arr[finite].mean())
    else:
        fill = 0.0
    arr = arr.copy()
    arr[~finite] = fill
    return arr


class NPYGroupDataset(Dataset):
    """
    Loads group npy arrays shaped (N,H,W). Returns normalized float tensor (1,H,W) and class id.
    Fixes NaN/Inf if requested.
    """
    def __init__(self, npy_paths: Dict[str, str], fix_nonfinite: bool = True):
        self.class_names = list(npy_paths.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.per_class_counts = {}
        arrays = []
        items = []

        print("📊 Dataset summary:")
        for cls, p in npy_paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")
            arr = _load_npy_float(p)  # (N,H,W)

            st = _finite_stats(arr)
            print(f"   - {cls}: {arr.shape[0]} images | bad={st['bad']} / total={st['total']}")

            if fix_nonfinite and st["bad"] > 0:
                arr = _fix_nonfinite(arr, mode="mean")
                st2 = _finite_stats(arr)
                print(f"     ✅ fixed non-finite → bad={st2['bad']}")

            self.per_class_counts[cls] = int(arr.shape[0])
            arrays.append(arr)

            y = self.class_to_idx[cls]
            for i in range(arr.shape[0]):
                items.append((arr[i], y))

        self.items = items

        # Compute global mean/std on finite values only
        big = np.concatenate([a.reshape(-1) for a in arrays], axis=0)
        finite = np.isfinite(big)
        if finite.sum() == 0:
            self.mean, self.std = 0.0, 1.0
            print("⚠️ WARNING: all values are non-finite. Using mean=0 std=1")
        else:
            vals = big[finite]
            self.mean = float(vals.mean())
            self.std = float(vals.std() + 1e-8)
            if not np.isfinite(self.std) or self.std < 1e-8:
                self.std = 1.0

        H, W = self.items[0][0].shape
        self.H, self.W = int(H), int(W)

        print(f"   - Image size: H={self.H}, W={self.W}")
        print(f"   - Global mean={self.mean:.6f}, std={self.std:.6f}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, y = self.items[idx]  # (H,W)
        img = np.nan_to_num(img, nan=self.mean, posinf=self.mean, neginf=self.mean).astype(np.float32, copy=False)

        x = (img - self.mean) / self.std
        x = np.clip(x, -5.0, 5.0) / 5.0  # ~[-1,1]

        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x = torch.from_numpy(x).unsqueeze(0)  # (1,H,W)
        return x, torch.tensor(y, dtype=torch.long)


def patch_aug(x: torch.Tensor, p_noise: float = 0.25) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[1])
    k = int(torch.randint(0, 4, (1,)).item())
    x = torch.rot90(x, k, dims=[1, 2])

    scale = 1.0 + (torch.rand(()) - 0.5) * 0.10
    shift = (torch.rand(()) - 0.5) * 0.05
    x = torch.clamp(x * scale + shift, -1.0, 1.0)

    if torch.rand(()) < p_noise:
        x = torch.clamp(x + 0.03 * torch.randn_like(x), -1.0, 1.0)

    # guard
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def make_coord_channels(patch: int, top: int, left: int, H: int, W: int, device=None):
    yy = torch.linspace(-1, 1, patch, device=device).view(1, patch, 1).expand(1, patch, patch)
    xx = torch.linspace(-1, 1, patch, device=device).view(1, 1, patch).expand(1, patch, patch)
    abs_y = (2.0 * (top / max(1, (H - patch))) - 1.0) * torch.ones_like(yy)
    abs_x = (2.0 * (left / max(1, (W - patch))) - 1.0) * torch.ones_like(xx)
    return torch.cat([yy, xx, abs_y, abs_x], dim=0)  # (4,P,P)


class PatchPairDataset(Dataset):
    def __init__(
        self,
        base: NPYGroupDataset,
        patch: int = 64,
        stride: int = 32,
        samples_per_epoch: int = 40000,
        coord_channels: bool = True,
        use_pairs: bool = True,
    ):
        self.base = base
        self.patch = patch
        self.stride = stride
        self.samples_per_epoch = samples_per_epoch
        self.coord_channels = coord_channels
        self.use_pairs = use_pairs
        self.H, self.W = base.H, base.W

        if patch > self.H or patch > self.W:
            raise ValueError(f"patch={patch} is bigger than image {(self.H, self.W)}")

    def __len__(self):
        return self.samples_per_epoch

    def _random_top_left(self):
        top = int(torch.randint(0, self.H - self.patch + 1, (1,)).item())
        left = int(torch.randint(0, self.W - self.patch + 1, (1,)).item())
        return top, left

    def _random_overlap_pair(self):
        t1, l1 = self._random_top_left()
        dt = int(torch.randint(-self.stride, self.stride + 1, (1,)).item())
        dl = int(torch.randint(-self.stride, self.stride + 1, (1,)).item())
        t2 = max(0, min(self.H - self.patch, t1 + dt))
        l2 = max(0, min(self.W - self.patch, l1 + dl))

        if (min(t1 + self.patch, t2 + self.patch) - max(t1, t2) <= 0) or \
           (min(l1 + self.patch, l2 + self.patch) - max(l1, l2) <= 0):
            t2, l2 = t1, l1
        return (t1, l1), (t2, l2)

    def __getitem__(self, _):
        idx = int(torch.randint(0, len(self.base), (1,)).item())
        img, y = self.base[idx]  # (1,H,W)

        if self.use_pairs:
            (t1, l1), (t2, l2) = self._random_overlap_pair()
        else:
            t1, l1 = self._random_top_left()
            t2, l2 = t1, l1

        p1 = img[:, t1:t1+self.patch, l1:l1+self.patch]
        p2 = img[:, t2:t2+self.patch, l2:l2+self.patch]

        p1 = patch_aug(p1)
        p2 = patch_aug(p2)

        if self.coord_channels:
            c1 = make_coord_channels(self.patch, t1, l1, self.H, self.W)
            c2 = make_coord_channels(self.patch, t2, l2, self.H, self.W)
            p1 = torch.cat([p1, c1], dim=0)
            p2 = torch.cat([p2, c2], dim=0)

        return {
            "p1": p1,
            "p2": p2,
            "y": y,
            "t1": torch.tensor([t1, l1], dtype=torch.long),
            "t2": torch.tensor([t2, l2], dtype=torch.long),
        }


# ----------------------------
# Diffusion
# ----------------------------
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


@dataclass
class DiffusionConfig:
    T: int = 1000


class Diffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, device):
        super().__init__()
        self.cfg = cfg
        betas = cosine_beta_schedule(cfg.T).to(device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = torch.cat([torch.tensor([1.0], device=device), ac[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", ac)
        self.register_buffer("alphas_cumprod_prev", ac_prev)
        self.register_buffer("sqrt_ac", torch.sqrt(ac))
        self.register_buffer("sqrt_omac", torch.sqrt(1.0 - ac))

        post_var = betas * (1.0 - ac_prev) / (1.0 - ac)
        self.register_buffer("posterior_variance", post_var)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_ac[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_omac[t].view(-1, 1, 1, 1)
        xt = s1 * x0 + s2 * noise
        xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
        noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
        return xt, noise

    def predict_x0_from_eps(self, xt, t, eps):
        s1 = self.sqrt_ac[t].view(-1, 1, 1, 1)
        s2 = self.sqrt_omac[t].view(-1, 1, 1, 1)
        x0 = (xt - s2 * eps) / (s1 + 1e-8)
        return torch.nan_to_num(x0, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# UNet
# ----------------------------
class SinTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, class_dim=0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.tproj = nn.Linear(time_dim, out_ch)
        self.yproj = nn.Linear(class_dim, out_ch) if class_dim > 0 else None
        self.short = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, y_emb=None):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.tproj(t_emb).view(-1, h.shape[1], 1, 1)
        if self.yproj is not None and y_emb is not None:
            h = h + self.yproj(y_emb).view(-1, h.shape[1], 1, 1)
        h = self.c2(F.silu(self.n2(h)))
        h = h + self.short(x)
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)


class SmallUNet(nn.Module):
    def __init__(self, in_ch, base=64, time_dim=256, num_classes=0, class_dim=128):
        super().__init__()
        self.t = nn.Sequential(
            SinTimeEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.use_class = num_classes > 0
        self.y = nn.Embedding(num_classes, class_dim) if self.use_class else None
        if not self.use_class:
            class_dim = 0

        self.inconv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.d1 = ResBlock(base, base, time_dim, class_dim)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = ResBlock(base, base * 2, time_dim, class_dim)
        self.p2 = nn.MaxPool2d(2)

        self.mid = ResBlock(base * 2, base * 2, time_dim, class_dim)

        self.u2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        # cat(u2(base), a2(2*base)) => 3*base
        self.ub2 = ResBlock(base * 3, base, time_dim, class_dim)

        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        # cat(u1(base), a1(base)) => 2*base
        self.ub1 = ResBlock(base * 2, base, time_dim, class_dim)

        self.outn = nn.GroupNorm(8, base)
        self.outc = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t, y=None):
        te = self.t(t)
        ye = self.y(y) if self.use_class else None

        x0 = self.inconv(x)
        a1 = self.d1(x0, te, ye)
        b1 = self.p1(a1)
        a2 = self.d2(b1, te, ye)
        b2 = self.p2(a2)

        m = self.mid(b2, te, ye)

        u2 = self.u2(m)
        u2 = torch.cat([u2, a2], dim=1)
        u2 = self.ub2(u2, te, ye)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, a1], dim=1)
        u1 = self.ub1(u1, te, ye)

        out = self.outc(F.silu(self.outn(u1)))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Consistency loss on overlap
# ----------------------------
def overlap_coords(t1l1, t2l2, patch: int):
    t1, l1 = int(t1l1[0]), int(t1l1[1])
    t2, l2 = int(t2l2[0]), int(t2l2[1])

    top = max(t1, t2)
    left = max(l1, l2)
    bottom = min(t1 + patch, t2 + patch)
    right = min(l1 + patch, l2 + patch)

    oh = bottom - top
    ow = right - left
    if oh <= 0 or ow <= 0:
        return None

    p1_top = top - t1
    p1_left = left - l1
    p2_top = top - t2
    p2_left = left - l2
    return (p1_top, p1_left, p2_top, p2_left, oh, ow)


def consistency_loss_x0(x0_1, x0_2, t1l1, t2l2, patch: int):
    total = 0.0
    count = 0
    B = x0_1.shape[0]
    for i in range(B):
        oc = overlap_coords(t1l1[i], t2l2[i], patch)
        if oc is None:
            continue
        p1t, p1l, p2t, p2l, oh, ow = oc
        a = x0_1[i, :, p1t:p1t+oh, p1l:p1l+ow]
        b = x0_2[i, :, p2t:p2t+oh, p2l:p2l+ow]
        total = total + F.l1_loss(a, b)
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=x0_1.device)
    return total / count


# ----------------------------
# Overlap-aware blending reconstruction
# ----------------------------
def hann2d(p: int, device):
    wy = torch.hann_window(p, periodic=False, device=device).view(p, 1)
    wx = torch.hann_window(p, periodic=False, device=device).view(1, p)
    return wy * wx


@torch.no_grad()
def reconstruct_blend(patches, coords, out_hw: Tuple[int, int], patch: int, device):
    H, W = out_hw
    acc = torch.zeros((1, H, W), device=device)
    wacc = torch.zeros((1, H, W), device=device)
    win = hann2d(patch, device).unsqueeze(0)

    for k, (top, left) in enumerate(coords):
        acc[:, top:top+patch, left:left+patch] += patches[k] * win
        wacc[:, top:top+patch, left:left+patch] += win

    out = acc / (wacc + 1e-8)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Sampling (patch -> full)
# ----------------------------
@torch.no_grad()
def sample_patch_ddpm(model, diff, patch: int, in_ch: int, steps: int, device, y=None, coord=None):
    x_img = torch.randn((1, 1, patch, patch), device=device)
    if coord is not None:
        x = torch.cat([x_img, coord.unsqueeze(0)], dim=1)
    else:
        x = torch.randn((1, in_ch, patch, patch), device=device)

    for t in reversed(range(steps)):
        tt = torch.tensor([t], device=device, dtype=torch.long)
        eps = model(x, tt, y=y)

        beta = diff.betas[tt].view(1, 1, 1, 1)
        alpha = diff.alphas[tt].view(1, 1, 1, 1)
        ac = diff.alphas_cumprod[tt].view(1, 1, 1, 1)

        mean = (1 / torch.sqrt(alpha)) * (x[:, :1] - (beta / torch.sqrt(1 - ac)) * eps)
        if t > 0:
            var = diff.posterior_variance[tt].view(1, 1, 1, 1)
            mean = mean + torch.sqrt(var) * torch.randn_like(mean)

        if coord is not None:
            x = torch.cat([mean, x[:, 1:]], dim=1)
        else:
            x[:, :1] = mean

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.tanh(x[:, :1]).squeeze(0)


@torch.no_grad()
def sample_full_image(model, diff, H, W, patch, stride, in_ch, steps, device, y=None, use_coords=True):
    coords = []
    plist = []
    for top in range(0, H - patch + 1, stride):
        for left in range(0, W - patch + 1, stride):
            coords.append((top, left))
            coord = make_coord_channels(patch, top, left, H, W, device=device) if use_coords else None
            p = sample_patch_ddpm(model, diff, patch, in_ch, steps, device, y=y, coord=coord)
            plist.append(p.unsqueeze(0))
    patches = torch.cat(plist, dim=0)
    return reconstruct_blend(patches, coords, (H, W), patch, device)


# ----------------------------
# Train
# ----------------------------
def train(args):
    device = get_cuda_device()

    npy_paths = {"adhd": args.adhd, "bd": args.bd, "health": args.health, "schz": args.schz}
    base = NPYGroupDataset(npy_paths, fix_nonfinite=args.fix_nonfinite)

    ds = PatchPairDataset(
        base,
        patch=args.patch,
        stride=args.stride,
        samples_per_epoch=args.samples_per_epoch,
        coord_channels=args.coord_channels,
        use_pairs=(args.lambda_consistency > 0),
    )

    loader = DataLoader(
        ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=True,
    )

    in_ch = 1 + (4 if args.coord_channels else 0)
    diff = Diffusion(DiffusionConfig(T=args.T), device=device)

    num_classes = 4 if args.class_cond else 0
    model = SmallUNet(
        in_ch=in_ch,
        base=args.base_ch,
        time_dim=args.time_dim,
        num_classes=num_classes,
        class_dim=args.class_dim,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)

    os.makedirs(args.outdir, exist_ok=True)

    history = {"epoch": [], "loss": [], "loss_main": [], "loss_cons": []}
    step = 0
    bad_steps_in_a_row = 0

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}")

        epoch_loss = 0.0
        epoch_main = 0.0
        epoch_cons = 0.0
        epoch_batches = 0

        for batch in pbar:
            p1 = batch["p1"].to(device, non_blocking=True)
            p2 = batch["p2"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            t1l1 = batch["t1"].to(device, non_blocking=True)
            t2l2 = batch["t2"].to(device, non_blocking=True)

            # guard inputs
            p1 = torch.nan_to_num(p1, nan=0.0, posinf=0.0, neginf=0.0)
            p2 = torch.nan_to_num(p2, nan=0.0, posinf=0.0, neginf=0.0)

            B = p1.size(0)
            t = torch.randint(0, diff.cfg.T, (B,), device=device, dtype=torch.long)

            x01 = p1[:, :1]
            x02 = p2[:, :1]
            xt1, n1 = diff.q_sample(x01, t)
            xt2, n2 = diff.q_sample(x02, t)

            xt1 = torch.cat([xt1, p1[:, 1:]], dim=1)
            xt2 = torch.cat([xt2, p2[:, 1:]], dim=1)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=not args.no_amp):
                eps1 = model(xt1, t, y if args.class_cond else None)
                loss_main = F.mse_loss(eps1, n1)

                if args.train_on_p2:
                    eps2 = model(xt2, t, y if args.class_cond else None)
                    loss_main = 0.5 * (loss_main + F.mse_loss(eps2, n2))
                else:
                    eps2 = model(xt2, t, y if args.class_cond else None)

                if args.lambda_consistency > 0:
                    x0hat1 = diff.predict_x0_from_eps(xt1[:, :1], t, eps1)
                    x0hat2 = diff.predict_x0_from_eps(xt2[:, :1], t, eps2)
                    loss_cons = consistency_loss_x0(x0hat1, x0hat2, t1l1, t2l2, args.patch)
                else:
                    loss_cons = torch.tensor(0.0, device=device)

                loss = loss_main + args.lambda_consistency * loss_cons

            # non-finite guard
            if not torch.isfinite(loss):
                bad_steps_in_a_row += 1
                print("⚠️ Non-finite loss detected. Skipping this step.")
                if bad_steps_in_a_row >= args.max_bad_steps:
                    raise RuntimeError(
                        f"Too many non-finite steps in a row ({bad_steps_in_a_row}). "
                        f"Dataset likely still has bad values or settings too aggressive."
                    )
                continue
            bad_steps_in_a_row = 0

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1
            epoch_loss += float(loss.item())
            epoch_main += float(loss_main.item())
            epoch_cons += float(loss_cons.item())
            epoch_batches += 1

            pbar.set_postfix(loss=float(loss.item()), main=float(loss_main.item()), cons=float(loss_cons.item()), step=step)

            if step % args.save_every == 0:
                torch.save(
                    {"model": model.state_dict(), "mean": base.mean, "std": base.std, "args": vars(args)},
                    os.path.join(args.outdir, f"ckpt_step_{step}.pt"),
                )

        # epoch averages
        if epoch_batches == 0:
            raise RuntimeError("No successful training steps in this epoch (all were non-finite).")

        avg_loss = epoch_loss / epoch_batches
        avg_main = epoch_main / epoch_batches
        avg_cons = epoch_cons / epoch_batches

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["loss_main"].append(avg_main)
        history["loss_cons"].append(avg_cons)

        plot_epoch_curves(args.outdir, history, show=args.show_plot)

        if args.sample_every_epoch:
            model.eval()
            y_cond = torch.tensor([0], device=device) if args.class_cond else None
            full = sample_full_image(
                model, diff, base.H, base.W,
                patch=args.patch, stride=args.stride,
                in_ch=in_ch, steps=min(args.sample_steps, args.T),
                device=device, y=y_cond, use_coords=args.coord_channels
            )
            save_epoch_sample(full, args.outdir, f"sample_epoch_{epoch+1}")
            model.train()

    torch.save(
        {"model": model.state_dict(), "mean": base.mean, "std": base.std, "args": vars(args)},
        os.path.join(args.outdir, "final.pt"),
    )

    # final sample
    model.eval()
    y_cond = torch.tensor([0], device=device) if args.class_cond else None
    full = sample_full_image(
        model, diff, base.H, base.W,
        patch=args.patch, stride=args.stride,
        in_ch=in_ch, steps=min(args.sample_steps, args.T),
        device=device, y=y_cond, use_coords=args.coord_channels
    )
    save_epoch_sample(full, args.outdir, "sample_final")

    print("\n✅ Training done.")
    print(f"📂 Outputs folder: {args.outdir}")


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--adhd", required=True)
    ap.add_argument("--bd", required=True)
    ap.add_argument("--health", required=True)
    ap.add_argument("--schz", required=True)

    ap.add_argument("--outdir", default="runs_proposal_patchddpm")
    ap.add_argument("--workers", type=int, default=2)

    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)

    ap.add_argument("--coord_channels", action="store_true", default=False)
    ap.add_argument("--class_cond", action="store_true", default=False)
    ap.add_argument("--train_on_p2", action="store_true", default=False)
    ap.add_argument("--sample_every_epoch", action="store_true", default=False)
    ap.add_argument("--lambda_consistency", type=float, default=0.0)

    ap.add_argument("--T", type=int, default=1000)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples_per_epoch", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--time_dim", type=int, default=256)
    ap.add_argument("--class_dim", type=int, default=128)

    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--sample_steps", type=int, default=200)

    ap.add_argument("--no_amp", action="store_true", default=False)
    ap.add_argument("--show_plot", action="store_true", default=False)

    # NEW: fix dataset NaN/Inf + stop if too many bad steps
    ap.add_argument("--fix_nonfinite", action="store_true", default=True)
    ap.add_argument("--no_fix_nonfinite", dest="fix_nonfinite", action="store_false")
    ap.add_argument("--max_bad_steps", type=int, default=200)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
