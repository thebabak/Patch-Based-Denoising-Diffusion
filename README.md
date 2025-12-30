# Multi-Scale Patch Diffusion (with Positional Encoding) for your .npy data

This project trains a **patch-based diffusion model** on your 2D arrays:
- Each sample is shaped (116, 152)
- We treat each sample as a 1-channel "image" (H=116, W=152)
- We train on **random multi-scale patches** + **(x,y) coordinate channels**
- Optional **class-conditioning** for your 4 groups: health / schz / adhd / bd

## Files
- `data.py`        : loads the provided .npy files into a PyTorch Dataset
- `diffusion.py`   : DDPM noise schedule + sampling utilities
- `model_unet.py`  : small 2D U-Net with timestep (and optional class) conditioning
- `train_patch_diffusion.py` : training script (patch-based)
- `sample.py`      : generate synthetic samples

## Quick start

### 1) Put the .npy files next to the scripts (or pass paths)
Expected filenames by default:
- `health_121.npy`
- `schz_27.npy`
- `adhd_40.npy`
- `bd_49.npy`

### 2) Train (CPU works, GPU is faster)
```bash
python train_patch_diffusion.py --data_dir . --epochs 50 --batch_size 32 --use_coords
```

### 3) Sample
```bash
python sample.py --ckpt runs/latest/model.pt --n 16 --label schz --out schz_samples.npy
```

## Notes / Assumptions
- The code assumes your arrays are **float** and shaped `(N, 116, 152)`.
- Normalization uses **global mean/std from the training split**.
- Default patch sizes: 32 and 64 (square). You can change them with `--patch_sizes`.


## GPU / AMP (added)
- The training script will use CUDA if available.
- Enable mixed precision (AMP) for speed:
```bash
python train_patch_diffusion.py --data_dir . --use_coords --amp
```
- Choose a specific GPU:
```bash
python train_patch_diffusion.py --data_dir . --use_coords --device cuda --gpu_id 0
```
- Multi-GPU (simple DataParallel):
```bash
python train_patch_diffusion.py --data_dir . --use_coords --multi_gpu
```


## Plot training results
```bash
python plot_results.py --run_dir runs/latest --save results.png
```
