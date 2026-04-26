# smartmap_mdv/losses.py
# - NT-Xent(InfoNCE) contrastive loss
# - VarCLR regularization

from __future__ import annotations
import torch
import torch.nn.functional as F

# --------------------------
# Contrastive Loss (NT-Xent / InfoNCE)
# --------------------------
def nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
    normalize: bool = True,
    clamp: float | None = 20.0,
) -> torch.Tensor:
    """
    NT-Xent(=InfoNCE) Loss.
    Args:
        z1, z2: [B, D] embedding 
        temperature: softmax temperature
        normalize: If True, using L2 norm 
    Returns:
        scalar loss (torch.Tensor)
    """
    assert z1.shape == z2.shape, "z1, z2 must have the same shape [B, D]"
    N = z1.size(0)
    if N < 2:
        return z1.sum() * 0.0

    if normalize:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

    z = torch.cat([z1, z2], dim=0)           # [2N, D]
    sim = z @ z.t()                           # [2N, 2N] 
    sim = sim / temperature

    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    pos = torch.arange(N, 2 * N, device=z.device)
    pos = torch.cat([pos, torch.arange(0, N, device=z.device)], dim=0).long()

    if clamp is not None:
        sim = torch.clamp(sim, min=-clamp, max=clamp)

    loss = F.cross_entropy(sim, pos)
    return loss


# --------------------------
# VarCLR regularization
# --------------------------
def varclr_regularizer(
    z1: torch.Tensor,
    z2: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
    cov_weight: float = 1.0,
    var_scale: float = 1.0,
) -> torch.Tensor:

    v = variance_loss(z1, gamma=gamma, eps=eps) + variance_loss(z2, gamma=gamma, eps=eps)
    c = covariance_loss(z1) + covariance_loss(z2)
    return var_scale * v + cov_weight * c

def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:

    if z.size(0) < 2:
        return z.sum() * 0.0
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)   # [D]
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:

    if z.size(0) < 2:
        return z.sum() * 0.0
    zc = z - z.mean(dim=0, keepdim=True)                   # [B, D]
    N = z.size(0)
    cov = (zc.T @ zc) / max(N - 1, 1)                      # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()
