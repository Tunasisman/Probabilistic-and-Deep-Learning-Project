"""
EECS face‑compression wrapper.

encode(images)  :  B × 1 × 192 × 160  ->  B × 32
decode(latents) :  B × 32             ->  B × 1 × 192 × 160

Weights are in float‑16 (faces_reconst_ae_fp16.pt, ~13 MiB).
"""

import os, cv2, torch, torch.nn as nn
import numpy as np
from pathlib import Path

# architecture (same with training script)
def conv(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = conv( 1, 32, s=2)
        self.c2 = conv(32, 64, s=2)
        self.c3 = conv(64,128, s=2)
        self.c4 = conv(128,256, s=2)
        self.c5 = conv(256,512, s=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(512, 32)
    def forward(self, x):
        x = self.c1(x); x = self.c2(x); x = self.c3(x)
        x = self.c4(x); x = self.c5(x)
        return self.fc(self.gap(x).flatten(1))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch)
        )
    def forward(self, x): return x + self.body(x)

def up_block(in_c, out_c, res=True):
    layers=[nn.ConvTranspose2d(in_c,out_c,4,2,1,bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
    if res: layers += [ResBlock(out_c), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 512*6*5)
        self.up1 = up_block(512,256)
        self.up2 = up_block(256,128)
        self.up3 = up_block(128,64)
        self.up4 = up_block(64,32, res=False)
        self.up5 = up_block(32,16, res=False)
        self.out = nn.Sequential(nn.Conv2d(16,1,1), nn.Sigmoid())
    def forward(self, z):
        x = self.fc(z).view(-1,512,6,5)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        x = self.up4(x); x = self.up5(x)
        return self.out(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

# Load FP16 weights
_ckpt = torch.load("faces_reconst_ae_fp16.pt", map_location="cpu")["model_fp16"]
_model = AutoEncoder().half() # network in FP16
_model.load_state_dict(_ckpt, strict=True)
_model.eval()

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)

# CLAHE pre‑processor (ClipLimit 2.0, 8×8 tiles)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def _apply_clahe(batch: torch.Tensor) -> torch.Tensor:
    b, _, h, w = batch.shape
    inp = (batch.clamp(0,1).mul(255).round().byte().numpy())  # uint8
    out = np.empty_like(inp)
    for i in range(b):
        out[i,0] = _clahe.apply(inp[i,0])
    out = torch.from_numpy(out.astype(np.float32) / 255.0)
    return out

@torch.no_grad()
def encode(images: torch.Tensor) -> torch.Tensor:
    imgs_eq = _apply_clahe(images.cpu())             # preprocessing on CPU
    z = _model.enc(imgs_eq.to(_device).half())       # forward in FP16
    return z.float().cpu()                           # back to float32 / CPU

@torch.no_grad()
def decode(latents: torch.Tensor) -> torch.Tensor:
    recon = _model.dec(latents.to(_device).half())
    return recon.float().clamp(0,1).cpu()

# quick self‑test (executed only when run directly, not on import)
if __name__ == "__main__":
    x = torch.rand(4,1,192,160)
    z = encode(x)
    y = decode(z)
    print("latents:", z.shape, z.dtype, z.min().item(), z.max().item())
    print("recon:",  y.shape, y.dtype, y.min().item(), y.max().item())
