import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    '''Sinusoidal timestep embedding. t: (B,) -> (B, dim)'''
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNetSmall(nn.Module):
    '''
    Small 2D U-Net with timestep conditioning.
    - Input can include extra channels (e.g., coord channels).
    - Output predicts noise for the data channel (1 channel).

    Optional class conditioning:
    - If cond is provided, we add a class embedding to the timestep embedding.
    '''
    def __init__(self, in_ch=3, base_ch=64, time_ch=256, num_classes=4, class_drop_prob=0.1):
        super().__init__()
        self.time_ch = time_ch
        self.num_classes = num_classes
        self.class_drop_prob = class_drop_prob

        self.time_mlp = nn.Sequential(
            nn.Linear(time_ch, time_ch),
            nn.SiLU(),
            nn.Linear(time_ch, time_ch),
        )
        self.class_emb = nn.Embedding(num_classes, time_ch)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_ch)
        self.pool = nn.AvgPool2d(2)
        self.down2 = ResBlock(base_ch, base_ch * 2, time_ch)

        self.mid = ResBlock(base_ch * 2, base_ch * 2, time_ch)

        self.up2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, time_ch)
        self.up1 = ResBlock(base_ch + base_ch, base_ch, time_ch)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, t, cond=None):
        temb = self.time_mlp(timestep_embedding(t, self.time_ch))

        if cond is not None:
            if self.training and self.class_drop_prob > 0:
                drop = (torch.rand(cond.shape[0], device=cond.device) < self.class_drop_prob)
                cond = cond.clone()
                cond[drop] = 0
            emb = temb + self.class_emb(cond)
        else:
            emb = temb

        h0 = self.in_conv(x)
        h1 = self.down1(h0, emb)
        h1p = self.pool(h1)

        h2 = self.down2(h1p, emb)
        h2p = self.pool(h2)

        hm = self.mid(h2p, emb)

        hu2 = F.interpolate(hm, scale_factor=2, mode="nearest")
        hu2 = torch.cat([hu2, h2], dim=1)
        hu2 = self.up2(hu2, emb)

        hu1 = F.interpolate(hu2, scale_factor=2, mode="nearest")
        hu1 = torch.cat([hu1, h1], dim=1)
        hu1 = self.up1(hu1, emb)

        return self.out_conv(F.silu(self.out_norm(hu1)))
