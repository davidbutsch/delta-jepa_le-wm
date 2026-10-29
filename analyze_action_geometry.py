"""Measure action-conditioned latent geometry of pretrained LeWM checkpoints.

For each checkpoint and each sampled dataset state z_i:
  - Property I (Jacobian): J_i = dF(z_i, a)/da at a0, G_i = J_i^T J_i, and its spectrum.
  - Sample-based: K_i = centered Gram of F(z_i, a_n) over N shared sampled actions.

Example:
    python analyze_action_geometry.py --policy 30ogb-cube_lewm/weights_epoch_30.pt
    python analyze_action_geometry.py --policy 30ogb-cube_lewm/weights_epoch_{1..30}.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from action_geometry import (
    OneStepPredictor,
    action_gram,
    action_jacobian,
    centered_gram,
    pairwise_stats,
    pullback_sq_dist,
    spectral_stats,
    spectrum,
    sq_dist_from_gram,
)
from utils import get_column_normalizer, get_img_preprocessor


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", required=True, help="checkpoint(s) relative to $STABLEWM_HOME/checkpoints")
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-states", type=int, default=256, help="M: number of initial states")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ref", choices=["zero", "dataset"], default="zero",
                   help="a0: zero (= mean action, shared across states) or the dataset action at each state")
    p.add_argument("--jacobian", choices=["autograd", "forward", "central"], default="autograd")
    p.add_argument("--fd-eps", type=float, default=1e-2, help="finite-difference step (normalized action units)")
    p.add_argument("--num-actions", type=int, default=64, help="N: sampled actions for K_z")
    p.add_argument("--action-scale", type=float, default=1.0, help="sampled actions: a0 + scale * N(0, I)")
    p.add_argument("--rank-rtol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="action_geometry_results")
    return p.parse_args()


def load_states(args):
    """Sample M dataset windows of history_size frames with z-scored actions."""
    dataset = swm.data.load_dataset(
        args.dataset,
        transform=None,
        num_steps=args.history_size,
        frameskip=args.frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size),
        get_column_normalizer(dataset, "action", "action"),
    )
    g = np.random.default_rng(args.seed)
    idx = np.sort(g.choice(len(dataset), size=args.num_states, replace=False))
    rows = [dataset[int(i)] for i in idx]
    pixels = torch.stack([r["pixels"] for r in rows])  # (M, H, C, H_img, W_img)
    actions = torch.nan_to_num(torch.stack([r["action"] for r in rows]), 0.0)  # (M, H, k)
    return idx, pixels, actions


@torch.no_grad()
def analyze(F, pixels, actions, sampled_eps, args, device):
    out = {k: [] for k in ["a0", "J", "G", "fd_check", "Y_min_dist", "Y_mean_dist", "K_eig", "K_dist_err"]}
    for s in range(0, len(pixels), args.batch_size):
        emb, past_act, a_data = F.encode_state(
            pixels[s : s + args.batch_size].to(device), actions[s : s + args.batch_size].to(device)
        )
        a0 = a_data if args.ref == "dataset" else torch.zeros_like(a_data)

        # -- Property I: local Jacobian and action-response Gram matrix
        J = action_jacobian(F, emb, past_act, a0, method=args.jacobian, eps=args.fd_eps)  # (B, D, k)
        G = action_gram(J)  # (B, k, k)

        # check the pullback approximation ||D_z(a0 + da, a0)||^2 ~= da^T G da on a random small da
        da = torch.randn_like(a0) * args.fd_eps
        exact = (F(emb, past_act, a0 + da) - F(emb, past_act, a0)).pow(2).sum(-1)
        approx = pullback_sq_dist(G, da)
        fd_check = (approx - exact).abs() / exact.clamp_min(1e-12)

        # -- Sample-based separability: N shared actions around a0
        acts = a0[:, None] + args.action_scale * sampled_eps.to(device)[None]  # (B, N, k)
        Y = F.predict_many(emb, past_act, acts)  # (B, N, D)
        K = centered_gram(Y.double())
        sq = sq_dist_from_gram(K)
        direct = torch.cdist(Y.double(), Y.double()).pow(2)
        pw = pairwise_stats(sq)

        out["a0"].append(a0.cpu())
        out["J"].append(J.cpu())
        out["G"].append(G.cpu())
        out["fd_check"].append(fd_check.cpu())
        out["Y_min_dist"].append(pw["min_dist"].cpu())
        out["Y_mean_dist"].append(pw["mean_dist"].cpu())
        out["K_eig"].append(spectrum(K).cpu())
        out["K_dist_err"].append((sq - direct).abs().amax(dim=(-1, -2)).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(res, args):
    G_stats = spectral_stats(spectrum(res["G"]), args.rank_rtol)
    K_stats = spectral_stats(res["K_eig"], args.rank_rtol)
    k = res["G"].size(-1)
    N = res["K_eig"].size(-1)

    def med(x):
        x = x.double()
        return float(x[torch.isfinite(x)].median()) if torch.isfinite(x).any() else float("inf")

    summary = {
        "k": k,
        "d": res["J"].size(1),
        "M": res["G"].size(0),
        "N": N,
        # Property I (Jacobian)
        "G_trace_median": med(G_stats["trace"]),
        "G_lambda_max_median": med(G_stats["lambda_max"]),
        "G_lambda_min_median": med(G_stats["lambda_min"]),
        "G_rank_mean": float(G_stats["rank"].double().mean()),
        "G_full_rank_frac": float((G_stats["rank"] == k).double().mean()),
        "G_log10_cond_median": med(G_stats["cond"].log10()),
        "G_log10_cond_nonzero_median": med(G_stats["cond_nonzero"].log10()),
        "pullback_rel_err_median": med(res["fd_check"]),
        # sample-based
        "K_trace_median": med(K_stats["trace"]),
        "K_rank_mean": float(K_stats["rank"].double().mean()),
        "K_rank_bound": min(res["J"].size(1), N - 1),
        "Y_min_dist_median": med(res["Y_min_dist"]),
        "Y_mean_dist_median": med(res["Y_mean_dist"]),
        "K_dist_identity_max_err": float(res["K_dist_err"].max()),
    }
    return summary, G_stats, K_stats


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx, pixels, actions = load_states(args)
    k = actions.size(-1)
    # shared action perturbations so K_{z_i} are comparable across states and checkpoints
    sampled_eps = torch.randn(args.num_actions, k, generator=torch.Generator().manual_seed(args.seed))

    rows = []
    for policy in args.policy:
        model = swm.wm.utils.load_pretrained(policy).to(device).eval()
        model.requires_grad_(False)
        F = OneStepPredictor(model).eval()

        res = analyze(F, pixels, actions, sampled_eps, args, device)
        summary, G_stats, K_stats = summarize(res, args)
        summary = {"policy": policy, **summary}
        rows.append(summary)
        print(json.dumps(summary, indent=2))

        name = policy.replace("/", "__").removesuffix(".pt")
        np.savez_compressed(
            out_dir / f"{name}.npz",
            state_idx=idx,
            a0=res["a0"].numpy(),
            G=res["G"].numpy(),
            G_eigvals=G_stats["eigvals"].numpy(),
            G_cond=G_stats["cond"].numpy(),
            G_cond_nonzero=G_stats["cond_nonzero"].numpy(),
            G_rank=G_stats["rank"].numpy(),
            K_eigvals=K_stats["eigvals"].numpy(),
            K_rank=K_stats["rank"].numpy(),
            Y_min_dist=res["Y_min_dist"].numpy(),
            Y_mean_dist=res["Y_mean_dist"].numpy(),
            sampled_eps=sampled_eps.numpy(),
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "results": rows}, f, indent=2)
    cols = ["G_trace_median", "G_rank_mean", "G_log10_cond_nonzero_median", "K_trace_median", "K_rank_mean", "Y_min_dist_median"]
    with open(out_dir / "summary.tsv", "w") as f:
        f.write("\t".join(["policy", *cols]) + "\n")
        for r in rows:
            f.write("\t".join([r["policy"], *(f"{r[c]:.6g}" for c in cols)]) + "\n")
    print((out_dir / "summary.tsv").read_text())


if __name__ == "__main__":
    main()
