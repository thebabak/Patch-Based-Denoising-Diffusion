import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import NpyGroupDataset, compute_global_norm_stats
from diffusion import DDPM
from model_unet import UNetSmall

def make_coord_channels(H, W, device):
    yy = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).repeat(1,1,1,W)
    xx = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).repeat(1,1,H,1)
    return torch.cat([xx, yy], dim=1)  # (1,2,H,W)

def random_crop(x, p):
    B, C, H, W = x.shape
    if p > H or p > W:
        raise ValueError(f"patch {p} bigger than {(H,W)}")
    tops = torch.randint(0, H - p + 1, (B,), device=x.device)
    lefts = torch.randint(0, W - p + 1, (B,), device=x.device)
    out = []
    for b in range(B):
        out.append(x[b:b+1, :, tops[b]:tops[b]+p, lefts[b]:lefts[b]+p])
    return torch.cat(out, dim=0)

def pick_device(device_str: str, gpu_id: int):
    """
    Returns torch.device with optional GPU selection.
    """
    device_str = (device_str or "").lower()
    if device_str.startswith("cuda") and torch.cuda.is_available():
        gpu_id = int(gpu_id)
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"gpu_id={gpu_id} invalid. cuda device count={torch.cuda.device_count()}")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--patch_sizes", type=str, default="32,64")
    ap.add_argument("--use_coords", action="store_true")

    # GPU / speed options
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use if device=cuda")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    ap.add_argument("--multi_gpu", action="store_true", help="Use DataParallel if multiple GPUs exist")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true", help="Pin memory in DataLoader (recommended for CUDA)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default="latest")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device, args.gpu_id)
    use_cuda = (device.type == "cuda")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] Using {device} : {torch.cuda.get_device_name(device)} | CUDA devices={torch.cuda.device_count()}")
    else:
        print("[CPU] Using CPU")

    # Load dataset (no normalization yet)
    raw_ds = NpyGroupDataset(data_dir=args.data_dir, normalize=None, return_label=True)

    # Train/val split
    n = len(raw_ds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.9 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # Normalization from train split only
    stats = compute_global_norm_stats(raw_ds, indices=train_idx)
    ds = NpyGroupDataset(data_dir=args.data_dir, normalize=stats, return_label=True)

    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist())

    pin = bool(args.pin_memory) if use_cuda else False
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=pin
    )

    H, W = ds.H, ds.W
    patch_sizes = [int(s) for s in args.patch_sizes.split(",") if s.strip()]
    in_ch = 1 + (2 if args.use_coords else 0)

    model = UNetSmall(in_ch=in_ch, base_ch=64, time_ch=256, num_classes=4, class_drop_prob=0.1).to(device)

    # Multi-GPU (simple)
    if args.multi_gpu and use_cuda and torch.cuda.device_count() > 1:
        print(f"[Multi-GPU] DataParallel enabled across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    ddpm = DDPM(T=args.T, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP scaler (CUDA only)
    use_amp = bool(args.amp) and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    losses_csv = os.path.join(run_dir, "losses.csv")
    if not os.path.exists(losses_csv):
        with open(losses_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,best_val\n")
    ckpt_path = os.path.join(run_dir, "model.pt")
    meta_path = os.path.join(run_dir, "meta.json")

    coord_full = make_coord_channels(H, W, device=device) if args.use_coords else None

    def forward_batch(x0, y):
        B = x0.shape[0]
        p = random.choice(patch_sizes)

        x_patch = random_crop(x0, p)  # (B,1,p,p)
        if args.use_coords:
            coord_patch = random_crop(coord_full.repeat(B,1,1,1), p)  # (B,2,p,p)
            x_in = torch.cat([x_patch, coord_patch], dim=1)
        else:
            x_in = x_patch

        t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_patch)
        x_t_data = ddpm.q_sample(x_patch, t, noise=noise)
        x_t = torch.cat([x_t_data, x_in[:, 1:]], dim=1) if args.use_coords else x_t_data

        # Predict noise for data channel
        noise_pred = model(x_t, t, cond=y)
        return F.mse_loss(noise_pred, noise)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x0, y in pbar:
            x0 = x0.to(device, non_blocking=pin)
            y = y.to(device, non_blocking=pin)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = forward_batch(x0, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(train_losses[-50:])))

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x0, y in val_loader:
                x0 = x0.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    vloss = forward_batch(x0, y)
                val_losses.append(float(vloss.item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            # Save underlying module if DataParallel
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({"model": state}, ckpt_path)

            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "H": H, "W": W,
                    "T": ddpm.T,
                    "patch_sizes": patch_sizes,
                    "use_coords": bool(args.use_coords),
                    "norm": stats,
                    "best_val": best_val,
                    "device": str(device),
                    "amp": use_amp,
                }, f, indent=2)

        # Log epoch losses
        train_loss_epoch = float(np.mean(train_losses)) if train_losses else float("inf")
        with open(losses_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss_epoch},{val_loss},{best_val}\n")
        print(f"Epoch {epoch}: val_loss={val_loss:.6f} best={best_val:.6f} (saved {ckpt_path})")

if __name__ == "__main__":
    main()
