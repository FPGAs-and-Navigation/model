"""UDML core: noise-aware uncertainty (rho), modality dependency (alpha), unbiased weights (w)."""
import torch
import torch.nn.functional as F
from torch import nn


class UDML(nn.Module):
    def __init__(self, dim, n_mod=3, heads=4):
        super().__init__()
        self.mod_emb = nn.Parameter(torch.zeros(n_mod, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, mu, logvar, avail, sample=False):
        """mu, logvar: (B, K, D); avail: (B, K) bool. Returns fused z and diagnostics.

        rho_k   = mean sigma_k^2                 (noise-aware uncertainty)
        alpha_k = attention that modality k receives from the other available
                  modalities, i.e. how well the others agree with it (dependency)
        w_k     = softmax_k(log alpha_k - log rho_k) over available modalities
        """
        B, K, D = mu.shape
        var = logvar.exp()
        rho = var.mean(-1)

        tok = self.norm(mu + self.mod_emb[:K])
        _, A = self.attn(tok, tok, tok, key_padding_mask=~avail, need_weights=True,
                         average_attn_weights=True)            # A[b, query, key]
        eye = torch.eye(K, dtype=torch.bool, device=mu.device)
        A = A.masked_fill(eye, 0) * avail.unsqueeze(-1)        # only other, present queries count
        n_other = (avail.sum(-1, keepdim=True) - 1).clamp(min=1)
        alpha = A.sum(1) / n_other                              # mass received by each key
        alpha = torch.where(avail.sum(-1, keepdim=True) > 1, alpha, torch.ones_like(alpha))

        logits = torch.log(alpha.clamp(min=1e-6)) - torch.log(rho.clamp(min=1e-6))
        w = F.softmax(logits.masked_fill(~avail, float("-inf")), dim=-1)

        z_k = mu + var.sqrt() * torch.randn_like(mu) if sample else mu
        z = (w.unsqueeze(-1) * z_k).sum(1)
        return z, dict(rho=rho, alpha=alpha, w=w)
