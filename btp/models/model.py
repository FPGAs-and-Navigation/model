"""Full localizer: encoders -> (mu, Sigma) per modality -> UDML fusion -> head.

The head either regresses lat/lon directly (`model.head: regress`) or classifies the
map cell (`classify`), which gives a multi-modal distribution for the coarse-to-fine
pipeline in btp/c2f.py.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..config import MODALITIES
from ..geo import cell_centers
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


class ClassHead(nn.Module):
    """FC network -> logits over map cells; `pos` is the most likely cell centre."""

    def __init__(self, dim, centers):
        super().__init__()
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, len(centers)))

    def forward(self, z):
        logits = self.net(z).float()
        centers = self.centers.float()                  # the buffer is half in an FP16 export
        p = logits.softmax(-1)
        mean = p @ centers
        var = (p * ((centers[None] - mean[:, None]) ** 2).sum(-1)).sum(-1) / 2
        return centers[logits.argmax(-1)], torch.log(var.clamp(min=1e-8)), logits


class Localizer(nn.Module):
    def __init__(self, cfg, ext):
        super().__init__()
        m, self.mods = cfg["model"], list(cfg["modalities"])
        self.kind = m.get("head", "regress")
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
        if self.kind == "classify":
            centers, _ = cell_centers(ext, cfg["coarse"]["cell"])
            self.head = ClassHead(dim, torch.as_tensor(ext.normalize(*centers.T)).T)
            self.label_sigma = cfg["coarse"]["label_sigma"] * cfg["coarse"]["cell"] / ext.half
        else:
            self.head = RegressionHead(dim)
        self.dim = dim

    def forward(self, batch):
        img = batch.get("img")
        B = batch["avail"].shape[0]
        dev = batch["avail"].device
        dt = next(self.udml.parameters()).dtype          # float32 in training, float16 in an FP16 export
        mu = torch.zeros(B, len(MODALITIES), self.dim, device=dev, dtype=dt)
        logvar = torch.zeros_like(mu)
        inputs = {"vision": img, "mag": batch.get("mag"), "grav": batch.get("grav")}
        for k, name in enumerate(MODALITIES):
            if name in self.enc:
                m, lv = self.enc[name](inputs[name])
                mu[:, k], logvar[:, k] = m.to(dt), lv.to(dt)
        avail = batch["avail"]
        z, diag = self.udml(mu, logvar, avail, sample=self.training)
        K = len(MODALITIES)
        out = dict(mu=mu, logvar=logvar, avail=avail, **diag)
        if self.kind == "classify":
            out["pos"], out["pos_logvar"], out["logits"] = self.head(z)
            aux_pos, aux_lv, aux_logits = self.head(mu.reshape(B * K, -1))
            out["aux_logits"] = aux_logits.view(B, K, -1)
        else:
            out["pos"], out["pos_logvar"] = self.head(z)
            aux_pos, aux_lv = self.head(mu.reshape(B * K, -1))
        out["aux_pos"], out["aux_logvar"] = aux_pos.view(B, K, 2), aux_lv.view(B, K)
        return out

    def soft_labels(self, target):
        """Gaussian credit over cells around the true position (normalised coordinates)."""
        d2 = ((self.head.centers[None] - target[:, None]) ** 2).sum(-1)
        return F.softmax(-d2 / (2 * self.label_sigma ** 2), -1)


def gaussian_nll(pred, logvar, target):
    """Isotropic 2D Gaussian NLL (up to a constant) in normalised coordinates."""
    return 0.5 * ((pred - target) ** 2).sum(-1) * torch.exp(-logvar) + logvar


def soft_ce(logits, q):
    return -(q * F.log_softmax(logits.float(), -1)).sum(-1)


def loss_fn(out, target, cfg, model=None):
    t = cfg["train"]
    avail = out["avail"].float()
    if "logits" in out:
        q = model.soft_labels(target)
        fused = soft_ce(out["logits"], q).mean()
        aux = soft_ce(out["aux_logits"], q.unsqueeze(1))
    else:
        fused = gaussian_nll(out["pos"], out["pos_logvar"], target).mean()
        aux = gaussian_nll(out["aux_pos"], out["aux_logvar"], target.unsqueeze(1))
    aux = (aux * avail).sum() / avail.sum().clamp(min=1)
    mu, lv = out["mu"], out["logvar"]
    kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv).mean(-1)
    kl = (kl * avail).sum() / avail.sum().clamp(min=1)
    total = fused + t["aux_weight"] * aux + t["kl_beta"] * kl
    return total, dict(loss=total.item(), nll=fused.item(), aux_nll=aux.item(), kl=kl.item())
