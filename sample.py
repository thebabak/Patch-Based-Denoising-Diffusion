import os
import argparse
import json
import numpy as np
import torch

from diffusion import DDPM
from model_unet import UNetSmall

LABEL_MAP = {"health":0, "schz":1, "adhd":2, "bd":3}

def pick_device(device_str: str, gpu_id: int):
    device_str = (device_str or "").lower()
    if device_str.startswith("cuda") and torch.cuda.is_available():
        gpu_id = int(gpu_id)
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"gpu_id={gpu_id} invalid. cuda device count={torch.cuda.device_count()}")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def make_coord_channels(H, W, device):
    yy = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).repeat(1,1,1,W)
    xx = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).repeat(1,1,H,1)
    return torch.cat([xx, yy], dim=1)  # (1,2,H,W)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--label", type=str, default="health")
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="samples.npy")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="Use AMP during sampling (CUDA only)")
    args = ap.parse_args()

    device = pick_device(args.device, args.gpu_id)
    use_cuda = (device.type == "cuda")
    use_amp = bool(args.amp) and use_cuda

    if use_cuda:
        print(f"[GPU] Using {device} : {torch.cuda.get_device_name(device)}")
    else:
        print("[CPU] Using CPU")

    meta_path = os.path.join(os.path.dirname(args.ckpt), "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    H, W = meta["H"], meta["W"]
    T = meta["T"]
    use_coords = meta["use_coords"]
    norm = meta["norm"]

    y = torch.full((args.n,), LABEL_MAP[args.label], device=device, dtype=torch.long)

    in_ch = 1 + (2 if use_coords else 0)
    model = UNetSmall(in_ch=in_ch, base_ch=64, time_ch=256, num_classes=4, class_drop_prob=0.0).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(T=T, device=device)
    extra = make_coord_channels(H, W, device=device).repeat(args.n, 1, 1, 1) if use_coords else None

    x = torch.randn((args.n, 1, H, W), device=device)
    for t in reversed(range(ddpm.T)):
        t_in = torch.full((args.n,), t, device=device, dtype=torch.long)
        x_in = torch.cat([x, extra], dim=1) if extra is not None else x

        with torch.cuda.amp.autocast(enabled=use_amp):
            eps_cond = model(x_in, t_in, cond=y)
            if args.cfg_scale > 0:
                eps_uncond = model(x_in, t_in, cond=None)
                eps = eps_uncond + args.cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

        beta_t = ddpm.betas[t]
        alpha_t = ddpm.alphas[t]
        alpha_bar_t = ddpm.alphas_cumprod[t]

        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        coef1 = beta_t * torch.sqrt(ddpm.alphas_cumprod_prev[t]) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - ddpm.alphas_cumprod_prev[t]) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x

        if t == 0:
            x = mean
        else:
            x = mean + torch.sqrt(ddpm.posterior_variance[t]) * torch.randn_like(x)

    arr = x.squeeze(1).cpu().numpy()
    arr = arr * (norm["std"] + 1e-8) + norm["mean"]
    np.save(args.out, arr)
    print(f"Saved: {args.out} shape={arr.shape}")

if __name__ == "__main__":
    main()
