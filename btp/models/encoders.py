"""Dedicated per-modality encoders, each producing a diagonal Gaussian (mu, log sigma^2)."""
import timm
import torch
import torch.nn.functional as F
from torch import nn

LOGVAR_RANGE = (-8.0, 8.0)


class GaussianHead(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        self.mu = nn.Linear(d_in, dim)
        self.logvar = nn.Linear(d_in, dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, h):
        return self.mu(h), self.logvar(h).clamp(*LOGVAR_RANGE)


class VisionEncoder(nn.Module):
    """ViT backbone (DINOv2 ViT-S/14) -> CLS token -> (mu_v, logvar_v)."""

    def __init__(self, name, dim, pretrained=True, img_size=224):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)
        self.head = GaussianHead(self.vit.num_features, dim)

    def forward(self, img):
        return self.head(self.vit(img))


class ProfileEncoder(nn.Module):
    """1D-CNN over an along-track sensor profile (B, C, L) -> (mu, logvar).

    First differences are appended so the network sees gradients along track, which
    are invariant to the slowly varying offsets real sensors drift by.
    """

    def __init__(self, in_ch, dim, width=(64, 128, 128, 256)):
        super().__init__()
        layers, c = [], in_ch * 2
        for i, w in enumerate(width):
            layers += [nn.Conv1d(c, w, 5, stride=1 if i == 0 else 2, padding=2),
                       nn.BatchNorm1d(w), nn.GELU(),
                       nn.Conv1d(w, w, 3, padding=1), nn.BatchNorm1d(w), nn.GELU()]
            c = w
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(2 * c, 512), nn.GELU(), nn.Dropout(0.1))
        self.head = GaussianHead(512, dim)

    def forward(self, x):
        dx = F.pad(x[..., 1:] - x[..., :-1], (1, 0))   # = torch.diff(prepend=x[..., :1]), ONNX-exportable
        h = self.conv(torch.cat([x, dx], 1))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(self.mlp(h))
