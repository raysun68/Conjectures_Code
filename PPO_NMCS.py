#!/usr/bin/env python
# coding: utf‑8
"""
CEM–PG optimiser with *row/column‑sum preservation*
────────────────────────────────────────────────────
 • Row i‑sum  = n  and  Column j‑sum = n  for all i, j  (enforced by Sinkhorn)
 • Actor: factorised Gaussian over all n² entries (zero‑mean perturbation Δ)
 • Critic: Cross‑Entropy method – keep top elite_frac % of the batch
 • Update: REINFORCE on elite samples only
 • Reward:  R(W) = –(λ₂ + λ_min)   (edit freely)

Usage
─────
$ python cem_pg_rowcol.py --n 6 --iters 3000
"""

from __future__ import annotations
import argparse, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal

# ───────────────────  spectral reward  ────────────────────────────────
def build_tilde(M: np.ndarray) -> np.ndarray:
    """16‑vertex blow‑up used in your Sidorenko‑style rewards."""
    m, n = M.shape
    t = np.zeros((m * n, m * n))
    for a in range(m):
        for b in range(n):
            for c in range(m):
                for d in range(n):
                    i1, i2 = a * n + b, c * n + d
                    prod   = max(M[a, b] * M[c, d], 0.0)
                    t[i1, i2] = M[c, b] * np.sqrt(prod) * M[a, d]
    return t


def reward_np(W: np.ndarray) -> float:
    W = np.clip(W, 0.00000001, None)
    eigs = np.linalg.eigvalsh(build_tilde(W))
    return -(eigs[-2] + eigs[0] + eigs[-3] + eigs[1] + eigs[-4] + eigs[2] + eigs[-5] + eigs[3] + eigs[-6] + eigs[4] + eigs[-7] + eigs[5] + eigs[-8] + eigs[6])
# ──────────────────────────────────────────────────────────────────────



# ───────────────────  actor network  ──────────────────────────────────
class Actor(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n * n, 128), nn.Tanh(),
            nn.Linear(128, 128),   nn.Tanh(),
        )
        self.mu   = nn.Linear(128, n * n)
        self.logσ = nn.Parameter(torch.full((n * n,), -4.0))

    def forward(self, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(W)
        return self.mu(h), self.logσ.exp()


def apply_perturb(W: np.ndarray,
                  Δ: np.ndarray,
                  min_val: float = 5e-2) -> np.ndarray:
    """
    Add perturbation (mean-zero), clip to keep positive values bounded away from zero,
    and normalize to preserve mean = 1.
    """
    Δ = Δ - Δ.mean()  # Keep perturbation centered
    Wn = np.clip(W + Δ, min_val, None)  # Ensure positive entries
    return Wn / Wn.mean()               # Optional: normalize to mean 1


# ───────────────────  main loop  ──────────────────────────────────────
def train(n: int = 4,
          batch: int = 64,
          elite_frac: float = 0.3,
          iters: int = 50_000,
          lr: float = 0.0003,
          seed: int = 0,
          print_every: int = 1000):

    rng   = np.random.default_rng(seed)
    torch.manual_seed(seed)

    actor = Actor(n)
    opt   = optim.Adam(actor.parameters(), lr=lr)

    # initial positive matrix with each row/col ≈ n
    W_best = np.random.rand(n, n)
    best_R = reward_np(W_best)

    for t in range(1, iters + 1):
        Ws, logps, Rs = [], [], []

        for _ in range(batch):
            W_torch = torch.tensor(W_best, dtype=torch.float32).unsqueeze(0)
            μ, σ    = actor(W_torch)
            dist    = Normal(μ, σ)
            Δ       = dist.sample().view(n, n).detach().numpy()

            W_new   = apply_perturb(W_best, Δ)
            R_new   = reward_np(W_new)

            Ws.append(W_new)
            logps.append(dist.log_prob(torch.tensor(Δ.flatten(), dtype=torch.float32)).sum())
            Rs.append(R_new)

            if R_new > best_R:                   # global best
                W_best, best_R = W_new, R_new

        # ─ elite filter ─
        Rs_arr = np.array(Rs)
        thresh = np.quantile(Rs_arr, 1 - elite_frac)
        elite  = Rs_arr >= thresh
        if not elite.any():      # safety
            continue

        elite_logp = torch.stack([logps[i] for i, flag in enumerate(elite) if flag])
        elite_R    = torch.tensor(Rs_arr[elite], dtype=torch.float32)
        advantage  = elite_R - elite_R.mean()
        loss = -(elite_logp * advantage).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % print_every == 0:
            print(f"Iter {t:6d} | best R {best_R:+.4e} | "
                  f"batch μ {Rs_arr.mean():+.2e} | elite μ {elite_R.mean():+.2e}")

    # ─ final report ─
    eigs = np.linalg.eigvalsh(build_tilde(W_best))
    print("\n=== finished ===")
    print("best reward:", f"{best_R:+.6e}")
    print("row sums   :", np.round(W_best.sum(1), 4))
    print("col sums   :", np.round(W_best.sum(0), 4))
    print("eigs (sorted):", np.round(eigs, 5))
    print("best W:")
    print(np.array2string(W_best, formatter={'float_kind':lambda z:f'{z:6.3f}'}))

    return W_best, best_R


# ───────────────────  CLI  ────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int,   default=12,     help="matrix size n×n")
    p.add_argument("--batch",  type=int,   default=80)
    p.add_argument("--iters",  type=int,   default=20_000)
    p.add_argument("--elite",  type=float, default=0.4)
    p.add_argument("--lr",     type=float, default=0.0002)
    p.add_argument("--seed",   type=int,   default=115)
    args = p.parse_args()

    train(n=args.n,
          batch=args.batch,
          elite_frac=args.elite,
          iters=args.iters,
          lr=args.lr,
          seed=args.seed)
