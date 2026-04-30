from __future__ import annotations

import torch
import torch.nn.functional as F

# smartmap_mdv/losses.py
# - NT-Xent / InfoNCE contrastive objective
# - VarCLR-style variance and covariance regularization


# --------------------------
# Contrastive objective (NT-Xent / InfoNCE)
# --------------------------
def nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
    normalize: bool = True,
    clamp: float | None = 20.0,
) -> torch.Tensor:
    """
    NT-Xent (= InfoNCE) loss.

    Args:
        z1, z2: [B, D] embeddings from two correlated views.
        temperature: Softmax temperature.
        normalize: If True, apply L2 normalization before similarity.
        clamp: Optional logit clamp for numerical stability.

    Returns:
        Scalar loss.
    """
    assert z1.shape == z2.shape, "z1, z2 must have the same shape [B, D]"
    N = z1.size(0)
    if N < 2:
        return z1.sum() * 0.0

    if normalize:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

    # Stack both views so each sample has one positive counterpart.
    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = z @ z.t()  # [2N, 2N]
    sim = sim / temperature

    # Remove trivial self-comparisons from the softmax denominator.
    mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    # The positive pair for each item is its counterpart in the other half.
    pos = torch.arange(N, 2 * N, device=z.device)
    pos = torch.cat([pos, torch.arange(0, N, device=z.device)], dim=0).long()

    if clamp is not None:
        # Clamp extreme logits to avoid unstable exponentials.
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
    cov_min_samples: int = 32,
) -> torch.Tensor:
    # Keep the same VarCLR objective, but estimate its moments on the combined
    # two-view batch to reduce noise under small training batches.
    z = torch.cat([z1, z2], dim=0)
    v = variance_loss(z, gamma=gamma, eps=eps)
    c = covariance_loss(z)
    cov_scale = min(1.0, float(z.size(0)) / float(max(cov_min_samples, 1)))
    return var_scale * v + (cov_weight * cov_scale) * c


# --------------------------
# Individual regularization terms
# --------------------------
def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # Penalize embedding dimensions whose batch standard deviation collapses.
    if z.size(0) < 2:
        return z.sum() * 0.0
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # [D]
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    # Penalize off-diagonal covariance so embedding dimensions stay decorrelated.
    if z.size(0) < 2:
        return z.sum() * 0.0
    zc = z - z.mean(dim=0, keepdim=True)  # [B, D]
    n, d = zc.shape
    cov = (zc.T @ zc) / max(n - 1, 1)  # [D, D]
    off_diag_mask = ~torch.eye(d, device=z.device, dtype=torch.bool)
    off_diag_vals = cov[off_diag_mask]
    if off_diag_vals.numel() == 0:
        return z.sum() * 0.0
    return (off_diag_vals ** 2).mean()
