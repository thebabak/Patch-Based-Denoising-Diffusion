import os
import argparse

import numpy as np
import torch

import patch260107 as p


@torch.no_grad()
def generate_per_group(
    ckpt_path: str = "runs_patch260107/final.pt",
    out_root: str = "synthetic_generated_patch260107",
    samples_per_group: int = 10,
):
    """Generate synthetic full images for each group using the trained patch260107 model.

    Outputs are saved under:
      out_root/adhd, out_root/bd, out_root/health, out_root/schz
    as both .npy and .png files.
    """
    device = p.get_cuda_device()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    # Rebuild diffusion config and model to match training
    in_ch = 1 + (4 if args["coord_channels"] else 0)
    diff = p.Diffusion(p.DiffusionConfig(T=args["T"]), device=device)

    num_classes = 4 if args["class_cond"] else 0
    model = p.SmallUNet(
        in_ch=in_ch,
        base=args["base_ch"],
        time_dim=args["time_dim"],
        num_classes=num_classes,
        class_dim=args["class_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Global mean/std used during training normalisation
    global_mean = float(ckpt["mean"])
    global_std = float(ckpt["std"])

    # Infer H, W from one of the real datasets used in training
    health_path = args["health"]
    arr = np.load(health_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected (N,H,W) array at {health_path}, got {arr.shape}")
    _, H, W = arr.shape

    class_names = ["adhd", "bd", "health", "schz"]

    # Compute per-group real statistics on the training scale (\approx [-1,1])
    real_paths = {
        "adhd": args["adhd"],
        "bd": args["bd"],
        "health": args["health"],
        "schz": args["schz"],
    }

    def to_train_scale_np(x: np.ndarray, mean: float, std: float) -> np.ndarray:
        z = (x - mean) / (std + 1e-8)
        z = np.clip(z, -5.0, 5.0) / 5.0
        return z.astype(np.float32, copy=False)

    real_stats = {}
    for cls_name in class_names:
        rp = real_paths[cls_name]
        r_arr = np.load(rp).astype(np.float32)
        r_ts = to_train_scale_np(r_arr, global_mean, global_std)
        real_stats[cls_name] = {
            "mean": float(r_ts.mean()),
            "std": float(r_ts.std() + 1e-8),
        }

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Output root: {out_root}")
    print(f"Image size: H={H}, W={W}")
    print(f"Samples per group: {samples_per_group}")

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(out_root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        y = torch.tensor([cls_idx], device=device) if num_classes > 0 else None
        target_mean = real_stats[cls_name]["mean"]
        target_std = real_stats[cls_name]["std"]

        for i in range(samples_per_group):
            full = p.sample_full_image(
                model,
                diff,
                H,
                W,
                patch=args["patch"],
                stride=args["stride"],
                in_ch=in_ch,
                steps=min(args["sample_steps"], args["T"]),
                device=device,
                y=y,
                use_coords=args["coord_channels"],
                sample_mode=args["sample_mode"],
                ddim_eta=args["ddim_eta"],
            )

            # Match first two moments (mean/std) of generated patch to real group on train scale
            gen_np = full.squeeze(0).detach().cpu().numpy().astype(np.float32)
            g_mean = float(gen_np.mean())
            g_std = float(gen_np.std() + 1e-8)
            if g_std > 0.0:
                gen_np = (gen_np - g_mean) * (target_std / g_std) + target_mean
            gen_np = np.clip(gen_np, -1.0, 1.0)
            full_calibrated = torch.from_numpy(gen_np).unsqueeze(0).to(full.device, dtype=full.dtype)

            name = f"sample_{i+1:04d}"
            p.save_epoch_sample(full_calibrated, cls_dir, name)

        print(f"Finished group {cls_name} -> {cls_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs_patch260107/final.pt")
    ap.add_argument("--out_root", default="synthetic_generated_patch260107")
    ap.add_argument("--samples_per_group", type=int, default=10)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    generate_per_group(a.ckpt, a.out_root, a.samples_per_group)
