"""Action-conditioned latent geometry diagnostics.

Implements the setup, Jacobian-based action separability (G_z = J_z^T J_z) and
sample-based action separability (centered Gram K_z of predicted futures) for a
one-step action-conditioned predictor F(z, a).
"""

import torch
from torch import nn


class OneStepPredictor(nn.Module):
    """Wraps a JEPA as F(z, a) -> z_hat_{t+1}.

    The state z is the context the predictor conditions on: the embedding history
    emb (B, H, D) and the H-1 preceding action blocks past_act (B, H-1, k).
    The action a (B, k) is the last action block, whose effect we study.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def encode_state(self, pixels, actions):
        """
        pixels: (B, H, C, H_img, W_img)
        actions: (B, H, k) -- the last block is returned as the dataset reference a0
        """
        emb = self.model.encode({"pixels": pixels})["emb"]  # (B, H, D)
        return emb, actions[:, :-1], actions[:, -1]

    def forward(self, emb, past_act, a):
        """
        emb: (B, H, D), past_act: (B, H-1, k), a: (B, k) -> (B, D)
        """
        act = torch.cat([past_act, a.unsqueeze(1)], dim=1)  # (B, H, k)
        act_emb = self.model.action_encoder(act)
        return self.model.predict(emb, act_emb)[:, -1]

    def predict_many(self, emb, past_act, actions):
        """Evaluate F(z_b, a_bn) for N actions per state.
        actions: (B, N, k) -> (B, N, D)
        """
        B, N = actions.shape[:2]
        rep = lambda x: x.repeat_interleave(N, dim=0)
        out = self(rep(emb), rep(past_act), actions.flatten(0, 1))
        return out.view(B, N, -1)


#########################
##  Setup: contrasts   ##
#########################

def latent_contrast(F, emb, past_act, a, b):
    """D_z(a, b) = F(z, a) - F(z, b). a, b: (B, k) -> (B, D)"""
    return F(emb, past_act, a) - F(emb, past_act, b)


def action_jacobian(F, emb, past_act, a0, method="autograd", eps=1e-2):
    """J_z = dF(z, a)/da at a = a0.  Returns (B, D, k).

    method:
      - "autograd": exact reverse-mode Jacobian (one backward per latent dim, batched).
      - "forward":  forward finite differences, k+1 predictor evaluations per state.
      - "central":  central finite differences, 2k predictor evaluations per state.
    """
    B, k = a0.shape

    if method == "autograd":
        # samples are independent (eval mode), so d(sum_b F_b)/da_b = dF_b/da_b
        def f(a):
            return F(emb, past_act, a).sum(0)  # (D,)

        with torch.enable_grad():
            J = torch.func.jacrev(f)(a0)  # (D, B, k)
        return J.permute(1, 0, 2).contiguous()

    E = torch.eye(k, device=a0.device, dtype=a0.dtype) * eps  # (k, k)
    if method == "forward":
        probes = torch.cat([a0[:, None], a0[:, None] + E], dim=1)  # (B, k+1, k)
        Y = F.predict_many(emb, past_act, probes)  # (B, k+1, D)
        J = (Y[:, 1:] - Y[:, :1]) / eps  # (B, k, D)
    elif method == "central":
        probes = torch.cat([a0[:, None] + E, a0[:, None] - E], dim=1)  # (B, 2k, k)
        Y = F.predict_many(emb, past_act, probes)
        J = (Y[:, :k] - Y[:, k:]) / (2 * eps)
    else:
        raise ValueError(f"unknown jacobian method {method}")
    return J.transpose(1, 2).contiguous()  # (B, D, k)


########################################
##  Property I: Jacobian separability  ##
########################################

def action_gram(J):
    """G_z = J_z^T J_z. J: (B, D, k) -> (B, k, k)"""
    return J.transpose(-1, -2) @ J


def pullback_sq_dist(G, delta_a):
    """delta_a^T G delta_a ~= ||D_z(a0 + delta_a, a0)||^2. G: (B, k, k), delta_a: (B, k) -> (B,)"""
    return torch.einsum("bi,bij,bj->b", delta_a, G, delta_a)


def spectrum(M):
    """Eigenvalues of a batch of symmetric PSD matrices, descending, clamped at 0. (B, n, n) -> (B, n)"""
    M = 0.5 * (M + M.transpose(-1, -2))
    return torch.linalg.eigvalsh(M.double()).flip(-1).clamp_min(0)


def spectral_stats(eig, rtol=1e-6):
    """Rank / conditioning / sensitivity summary of a descending PSD spectrum (B, n).

    rank: number of eigenvalues above rtol * lambda_1
    cond: lambda_1 / lambda_min over the full spectrum (inf when collapsed)
    cond_nonzero: lambda_1 / lambda_rank, conditioning within the non-collapsed subspace
    trace: sum of eigenvalues (= ||J||_F^2 for G_z, total variation for K_z)
    """
    lam_max = eig[:, 0]
    lam_min = eig[:, -1]
    rank = (eig > rtol * lam_max[:, None]).sum(-1)
    cond = torch.where(lam_min > rtol * lam_max, lam_max / lam_min, torch.full_like(lam_max, float("inf")))
    lam_r = eig.gather(-1, (rank - 1).clamp_min(0)[:, None])[:, 0]
    return {
        "eigvals": eig,
        "trace": eig.sum(-1),
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "rank": rank,
        "cond": cond,
        "cond_nonzero": lam_max / lam_r,
    }


#############################################
##  Sample-based action separability (K_z)  ##
#############################################

def centered_gram(Y):
    """K_z = Y~^T Y~ over N sampled futures. Y: (B, N, D) -> (B, N, N)"""
    Yc = Y - Y.mean(dim=1, keepdim=True)
    return Yc @ Yc.transpose(-1, -2)


def sq_dist_from_gram(K):
    """||F(z,a_j) - F(z,a_l)||^2 = K_jj + K_ll - 2 K_jl. (B, N, N) -> (B, N, N)"""
    d = torch.diagonal(K, dim1=-2, dim2=-1)
    return (d[:, :, None] + d[:, None, :] - 2 * K).clamp_min(0)


def pairwise_stats(sq_dist):
    """Min / mean off-diagonal latent distance between sampled futures. (B, N, N) -> dict of (B,)"""
    N = sq_dist.size(-1)
    off = ~torch.eye(N, dtype=torch.bool, device=sq_dist.device)
    dist = sq_dist.sqrt()[:, off]  # (B, N*(N-1))
    return {"min_dist": dist.min(-1).values, "mean_dist": dist.mean(-1)}
