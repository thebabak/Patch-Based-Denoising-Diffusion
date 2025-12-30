import os
import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = ["health", "schz", "adhd", "bd"]
DEFAULT_FILES = {
    "health": "health_121.npy",
    "schz":   "schz_27.npy",
    "adhd":   "adhd_40.npy",
    "bd":     "bd_49.npy",
}

class NpyGroupDataset(Dataset):
    '''
    Loads multiple .npy arrays, each shaped (N, H, W), assigns an integer label per group.

    Returns:
      x : torch.FloatTensor, shape (1, H, W)
      y : torch.LongTensor, scalar in {0..num_classes-1}
    '''
    def __init__(self, data_dir=".", files=None, normalize=None, return_label=True):
        self.data_dir = data_dir
        self.files = files or DEFAULT_FILES
        self.return_label = return_label

        xs, ys = [], []
        for label_name in LABELS:
            fname = self.files.get(label_name)
            if fname is None:
                continue
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            arr = np.load(path)  # (N, H, W)
            if arr.ndim != 3:
                raise ValueError(f"{path} must be 3D (N,H,W). Got shape={arr.shape}")
            arr = arr.astype(np.float32, copy=False)
            xs.append(arr)
            ys.append(np.full((arr.shape[0],), LABELS.index(label_name), dtype=np.int64))

        self.x = np.concatenate(xs, axis=0)  # (N_total, H, W)
        self.y = np.concatenate(ys, axis=0)  # (N_total,)
        self.normalize = normalize  # dict with mean/std OR None

        self.H = self.x.shape[1]
        self.W = self.x.shape[2]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]  # (H,W)
        if self.normalize is not None:
            mean = self.normalize["mean"]
            std = self.normalize["std"]
            x = (x - mean) / (std + 1e-8)
        x = torch.from_numpy(x).unsqueeze(0)  # (1,H,W)
        if self.return_label:
            y = torch.tensor(int(self.y[idx]), dtype=torch.long)
            return x, y
        return x

def compute_global_norm_stats(dataset, indices=None):
    '''
    Compute global mean/std over selected indices in the dataset.
    Uses dataset.x (numpy array) and returns {"mean": float, "std": float}
    '''
    if indices is None:
        arr = dataset.x
    else:
        arr = dataset.x[indices]
    mean = float(arr.mean())
    std = float(arr.std())
    return {"mean": mean, "std": std}
