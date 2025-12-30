# benchmark_speed.py
# Compare CPU vs CUDA speed for ONE diffusion training step loop (patch-based).
# Uses: data.py, diffusion.py, model_unet.py

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import NpyGroupDataset, compute_global_norm_stats
from diffusion import DDPM
from model_unet import UNetSmall


def make_coord_channels(H, W, device):
    yy = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).repeat(1, 1, 1, W)
    xx = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).repeat(1, 1, H, 1)
    return torch.cat([xx, yy], dim=1)  # (1,2,H,W)


def center_crop(x, p):
    # x: (B,C,H,W) -> (B,C,p,p)
    B, C, H, W = x.shape
    if p > H or p > W:
        raise ValueError(f"patch {p} bigger than {(H,W)}")
    top = (H - p) // 2
    left = (W - p) // 2
    return x[:, :, top : top + p, left : left + p]


@torch.no_grad()
def get_one_batch(ds, batch_size):
    # deterministic batch for fair benchmark
    idx = np.arange(len(ds))
    np.random.seed(0)
    np.random.shuffle(idx)
    take = idx[: max(batch_size, 8)]
    sub = Subset(ds, take.tolist())
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, drop_last=True)
    return next(iter(loader))


def benchmark(device, model, ddpm, x0, y, p, use_coords, steps, warmup, lr, amp):
    """
    Times 'steps' optimizer updates after 'warmup' warmup steps.
    """
    use_cuda = (device.type == "cuda")
    use_amp = bool(amp) and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    B, _, H, W = x0.shape
    x0 = x0.to(device)
    y = y.to(device)

    coord_full = make_coord_channels(H, W, device=device) if use_coords else None

    def step_once():
        # fixed patch crop for fairness
        x_patch = center_crop(x0, p)  # (B,1,p,p)

        if use_coords:
            coord_patch = center_crop(coord_full.repeat(B, 1, 1, 1), p)  # (B,2,p,p)
            x_in = torch.cat([x_patch, coord_patch], dim=1)              # (B,3,p,p)
        else:
            x_in = x_patch                                               # (B,1,p,p)

        t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_patch)
        x_t_data = ddpm.q_sample(x_patch, t, noise=noise)
        x_t = torch.cat([x_t_data, x_in[:, 1:]], dim=1) if use_coords else x_t_data

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            noise_pred = model(x_t, t, cond=y)
            loss = F.mse_loss(noise_pred, noise)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return float(loss.item())

    # Warmup
    if use_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        step_once()
    if use_cuda:
        torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    if use_cuda:
        torch.cuda.synchronize()
    for _ in range(steps):
        step_once()
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    sec_per_step = total / steps
    steps_per_sec = steps / total
    return total, sec_per_step, steps_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--use_coords", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--amp", action="store_true", help="AMP only affects CUDA run")
    args = ap.parse_args()

    # Load + normalize (train split stats)
    raw_ds = NpyGroupDataset(data_dir=args.data_dir, normalize=None, return_label=True)
    n = len(raw_ds)
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    n_train = int(0.9 * n)
    train_idx = idx[:n_train]
    stats = compute_global_norm_stats(raw_ds, indices=train_idx)
    ds = NpyGroupDataset(data_dir=args.data_dir, normalize=stats, return_label=True)

    x0, y = get_one_batch(ds, args.batch_size)  # x0: (B,1,116,152)

    H, W = x0.shape[-2], x0.shape[-1]
    in_ch = 1 + (2 if args.use_coords else 0)

    # Create identical model for both runs (same init seed)
    torch.manual_seed(0)
    model_cpu = UNetSmall(in_ch=in_ch, base_ch=64, time_ch=256, num_classes=4, class_drop_prob=0.1)
    ddpm_cpu = DDPM(T=args.T, device=torch.device("cpu"))

    print("\n=== Benchmark settings ===")
    print(f"batch_size={args.batch_size} | patch={args.patch} | use_coords={args.use_coords} | T={args.T}")
    print(f"steps={args.steps} warmup={args.warmup} | amp={args.amp}")

    # CPU benchmark
    cpu = torch.device("cpu")
    total, sps, fps = benchmark(
        cpu, model_cpu, ddpm_cpu, x0, y,
        p=args.patch,
        use_coords=args.use_coords,
        steps=args.steps, warmup=args.warmup,
        lr=args.lr,
        amp=False,
    )
    print("\n[CPU]")
    print(f"total_time: {total:.3f}s | sec/step: {sps:.6f} | steps/sec: {fps:.2f}")

    # CUDA benchmark (if available)
    if torch.cuda.is_available():
        gpu = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True

        # fresh model with same init for fair compare
        torch.manual_seed(0)
        model_gpu = UNetSmall(in_ch=in_ch, base_ch=64, time_ch=256, num_classes=4, class_drop_prob=0.1)
        ddpm_gpu = DDPM(T=args.T, device=gpu)

        total, sps, fps = benchmark(
            gpu, model_gpu, ddpm_gpu, x0, y,
            p=args.patch,
            use_coords=args.use_coords,
            steps=args.steps, warmup=args.warmup,
            lr=args.lr,
            amp=args.amp,
        )
        print("\n[CUDA]")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"total_time: {total:.3f}s | sec/step: {sps:.6f} | steps/sec: {fps:.2f}")
    else:
        print("\nCUDA not available (torch.cuda.is_available() == False).")

if __name__ == "__main__":
    main()
