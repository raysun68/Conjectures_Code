"""
amcs_C5_over_C3.py
==================

AMCS‑style search that tries to **beat the 2‑barrier** for
    log t(C₅) / log t(C₃)

Key ideas
---------
* seed = 5‑block blow‑up of C₅ (clique blocks + cycle links)
* NMCS + AMCS (depth / level branching)
* adaptive ε (expands on improvement, shrinks on failure)
* keep off‑diagonal weights ≥ w_min to avoid degenerate cliques
* diagonal weights forced to ≥ 0.9

Run: ``python amcs_C5_over_C3.py``

Tweak `PARAMS` at the bottom for longer / deeper runs.
"""

import numpy as np
import math


# ------------------------------------------------------------------
# 1. Eigen‑trace utilities
# ------------------------------------------------------------------
def graphon_operator_matrix(alphas, W):
    sqrt_alpha = np.sqrt(alphas)
    return sqrt_alpha[:, None] * W * sqrt_alpha[None, :]


def t_cycle(alphas, W, k: int) -> float:
    M = graphon_operator_matrix(alphas, W)
    eig = np.linalg.eigvalsh(M)
    return float(np.sum(eig ** k))


def ratio_C5_C3(alphas, W):
    t3 = t_cycle(alphas, W, 3)
    t5 = t_cycle(alphas, W, 5)
    if not (1e-6 < t3 < 0.99) or t5 <= 0:
        return -float("inf")
    return math.log(t5) / math.log(t3)


# ------------------------------------------------------------------
# 2.  Seed graphon  (5‑block blow‑up of C₅)
# ------------------------------------------------------------------
def seed_graphon(eps_link: float = 0.15):
    m = 5
    alphas = np.ones(m) / m
    W = np.zeros((m, m))
    np.fill_diagonal(W, 1.0)
    for i in range(m):
        j = (i + 1) % m
        W[i, j] = W[j, i] = eps_link
    # sprinkle a little noise to break symmetry
    noise = np.random.uniform(0, 0.02, size=(m, m))
    W = np.clip((W + noise + noise.T) / 2, 0.0, 1.0)
    return alphas, W


# ------------------------------------------------------------------
# 3.  Perturbation operator
# ------------------------------------------------------------------
def perturb_W(W, eps, w_min=0.01):
    m = W.shape[0]
    # add symmetric noise
    n = np.random.uniform(-eps, eps, size=(m, m))
    W2 = (W + n + n.T) / 2
    # occasionally (10%) resample a random off‑diag entry
    if np.random.rand() < 0.1:
        i, j = np.random.randint(0, m, 2)
        if i != j:
            W2[i, j] = W2[j, i] = np.random.rand()
    # enforce constraints
    np.fill_diagonal(W2, np.clip(np.diag(W2), 0.9, 1.0))
    off_mask = ~np.eye(m, dtype=bool)
    W2[off_mask] = np.clip(W2[off_mask], w_min, 1.0)
    return W2


# ------------------------------------------------------------------
# 4.  Nested Monte‑Carlo Search
# ------------------------------------------------------------------
def nmcs(alphas, W, depth, level, branching, eps, w_min):
    if depth == 0:
        return W, ratio_C5_C3(alphas, W)

    best_W, best_score = W, ratio_C5_C3(alphas, W)

    for _ in range(branching):
        child_W = perturb_W(W, eps, w_min)
        if level == 0:
            cand_W, cand_s = nmcs(alphas, child_W, depth - 1, 0, branching, eps, w_min)
        else:
            cand_W, cand_s = nmcs(alphas, child_W, depth - 1, level - 1, branching, eps, w_min)

        if cand_s > best_score:
            best_W, best_score = cand_W, cand_s

    return best_W, best_score


# ------------------------------------------------------------------
# 5.  Adaptive Monte‑Carlo Search driver
# ------------------------------------------------------------------
def amcs(alphas, W, *,
         init_eps=0.2, eps_min=1e-4, eps_max=0.6,
         improve_factor=1.25, decay_factor=0.9,
         init_depth=2, init_level=1,
         max_depth=6, max_level=4,
         branching=4, patience=40, w_min=0.01,
         iterations=50000, log_every=1000):
    best_W = W
    best_s = ratio_C5_C3(alphas, W)
    eps = init_eps
    depth, level = init_depth, init_level
    stale = 0

    for it in range(1, iterations + 1):
        cand_W, cand_s = nmcs(alphas, best_W, depth, level, branching, eps, w_min)

        if cand_s > best_s:
            best_W, best_s = cand_W, cand_s
            eps = min(eps * improve_factor, eps_max)
            stale = 0
        else:
            eps = max(eps * decay_factor, eps_min)
            stale += 1

        # escalate search complexity if stuck
        if stale >= patience:
            stale = 0
            if level < max_level:
                level += 1
            elif depth < max_depth:
                depth += 1

        if it % log_every == 0:
            print(f"iter {it:7d}: ratio={best_s:.6f}, eps={eps:.4f}, depth={depth}, level={level}")

    return best_W, best_s


# ------------------------------------------------------------------
# 6.  Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    alphas, W0 = seed_graphon()

    print("Initial ratio =", ratio_C5_C3(alphas, W0))

    PARAMS = dict(
        init_eps=0.25,
        iterations=10000,
        branching=5,
        patience=6,
        log_every=1000,
    )

    best_W, best_ratio = amcs(alphas, W0, **PARAMS)

    print("\n=========== FINAL ===========")
    print("Best log‑ratio  =", best_ratio)
    print("Block masses α =", alphas)
    print("Best W matrix:\n", np.round(best_W, 3))
