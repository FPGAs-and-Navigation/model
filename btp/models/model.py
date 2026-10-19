"""Full localizer: encoders -> (mu, Sigma) per modality -> UDML fusion -> regression head."""
import torch
from torch import nn

from ..config import MODALITIES
from .encoders import ProfileEncoder, VisionEncoder
from .udml import UDML


class RegressionHead(nn.Module):
    """FC network -> normalised (E, N) and an isotropic log-variance of the position."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(),
                                 nn.Linear(256, 3))

    def forward(self, z):
        o = self.net(z)
        return o[:, :2], o[:, 2].clamp(-16, 4)


class Localizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, self.mods = cfg["model"], list(cfg["modalities"])
        dim = m["dim"]
        enc = {}
        if "vision" in self.mods:
            enc["vision"] = VisionEncoder(m["vit"], dim, m["pretrained"], cfg["img"]["size"])
        if "mag" in self.mods:
            enc["mag"] = ProfileEncoder(4, dim)
        if "grav" in self.mods:
            enc["grav"] = ProfileEncoder(5, dim)
        self.enc = nn.ModuleDict(enc)
        self.udml = UDML(dim, len(MODALITIES), m["attn_heads"])
        # One head serves the fused latent and each modality alone (auxiliary outputs),
        # which keeps all latents in the same space and gives per-modality accuracy.
        self.head = RegressionHead(dim)
        self.dim = dim

    def forward(self, batch):
        img = batch.get("img")
        B = batch["avail"].shape[0]
        dev = batch["avail"].device
        mu = torch.zeros(B, len(MODALITIES), self.dim, device=dev)
        logvar = torch.zeros_like(mu)
        inputs = {"vision": img, "mag": batch.get("mag"), "grav": batch.get("grav")}
        for k, name in enumerate(MODALITIES):
            if name in self.enc:
                m, lv = self.enc[name](inputs[name])
                mu[:, k], logvar[:, k] = m.float(), lv.float()
        avail = batch["avail"]
        z, diag = self.udml(mu, logvar, avail, sample=self.training)
        pos, pos_logvar = self.head(z)
        aux_pos, aux_lv = self.head(mu.reshape(B * len(MODALITIES), -1))
        return dict(pos=pos, pos_logvar=pos_logvar, mu=mu, logvar=logvar, avail=avail,
                    aux_pos=aux_pos.view(B, -1, 2), aux_logvar=aux_lv.view(B, -1), **diag)


def gaussian_nll(pred, logvar, target):
    """Isotropic 2D Gaussian NLL (up to a constant) in normalised coordinates."""
    return 0.5 * ((pred - target) ** 2).sum(-1) * torch.exp(-logvar) + logvar


def loss_fn(out, target, cfg):
    t = cfg["train"]
    fused = gaussian_nll(out["pos"], out["pos_logvar"], target).mean()
    avail = out["avail"].float()
    aux = gaussian_nll(out["aux_pos"], out["aux_logvar"], target.unsqueeze(1))
    aux = (aux * avail).sum() / avail.sum().clamp(min=1)
    mu, lv = out["mu"], out["logvar"]
    kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean(-1)
    kl = (kl * avail).sum() / avail.sum().clamp(min=1)
    total = fused + t["aux_weight"] * aux + t["kl_beta"] * kl
    return total, dict(loss=total.item(), nll=fused.item(), aux_nll=aux.item(), kl=kl.item())
