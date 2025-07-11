"""
m×n matrix search with minimal 4‑cell patterns
(successful patterns reinforced; magnitude x ∈ [‑eps, eps]).
"""

import numpy as np

# ----------------------------------------------------------------------
#  Sidorenko helpers
def build_tilde_M(M):
    m, n = M.shape
    tilde = np.zeros((m*n, m*n))
    for a in range(m):
        for b in range(n):
            for c in range(m):
                for d in range(n):
                    idx1 = a*n + b
                    idx2 = c*n + d
                    prod = max(M[a, b] * M[c, d], 0.0)
                    tilde[idx1, idx2] = M[c, b] * np.sqrt(prod) * M[a, d]
    return tilde

def sidorenko_eigenvalue_check(M):
    M = np.maximum(M, 0)
    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
    tilde = build_tilde_M(M)
    if not np.all(np.isfinite(tilde)):
        return -np.inf
    eig = np.linalg.eigvalsh(tilde)
    lhs = np.sum(eig**5)
    rhs = (np.sum(M))**15 / (M.size)**10
    smallest = eig[0]
    second_largest = eig[-2]
    return -(second_largest + smallest)
# ----------------------------------------------------------------------

def generate_patterns(m: int, n: int):
    """
    Return the (m‑1)(n‑1) independent 4‑cell patterns:

        rows (i, m‑1)  for  i = 0 … m‑2
        cols (j, n‑1)  for  j = 0 … n‑2
    """
    patterns = []
    for i in range(m - 1):
        for j in range(n - 1):
            patterns.append((i, m - 1, j, n - 1))   # (i1,i2,j1,j2)
    return patterns                                   # length (m-1)(n-1)

def apply_4cell(W, pattern, x):
    """
    Apply a 4‑cell update; return None if x cannot be applied.
    """
    i1, i2, j1, j2 = pattern
    xmin = -min(W[i1, j2], W[i2, j1])
    xmax =  min(W[i1, j1], W[i2, j2])
    if xmin > xmax:          # numerical edge case
        return None
    x = np.clip(x, xmin, xmax)
    if x == 0.0:
        return None
    Wn = W.copy()
    Wn[i1, j1] += x
    Wn[i1, j2] -= x
    Wn[i2, j1] -= x
    Wn[i2, j2] += x
    return Wn

# ----------------------------------------------------------------------
def search(
        m=4,
        n=4,
        W0=None,
        max_iter=50_000,
        eps_init=1e-3,
        eps_min=1e-6,
        eps_max=1e-1,
        seed=0,
        print_every=1000,
):
    """
    Bandit‑reinforced hill‑climb using the (m‑1)(n‑1) minimal 4‑cell patterns.

    * eps starts at eps_init
    * eps *= 3  whenever a move is accepted
    * eps /= 2  after 100 consecutive rejected moves
    """

    rng = np.random.default_rng(seed)
    patterns = generate_patterns(m, n)
    K = len(patterns)

    # initial matrix
    if W0 is None:
        W = np.ones((m, n), dtype=float)
    else:
        W = np.array(W0, dtype=float).reshape(m, n)

    best_score = sidorenko_eigenvalue_check(W)

    counts = np.ones(K, dtype=float)          # bandit counts (Laplace)
    eps = eps_init
    no_improve_streak = 0                     # count consecutive rejections

    print(f"Start score: {best_score:.6e}   eps = {eps:g}")

    for t in range(1, max_iter + 1):

        # --- choose pattern via soft counts ----------------------------
        probs = counts / counts.sum()
        k      = rng.choice(K, p=probs)
        pat    = patterns[k]
        x      = rng.uniform(-eps, eps)

        W_new = apply_4cell(W, pat, x)
        if W_new is None:          # illegal magnitude
            continue

        new_score = sidorenko_eigenvalue_check(W_new)

        # --- accept / reject -------------------------------------------
        if new_score > best_score:          # SUCCESS
            W           = W_new
            best_score  = new_score
            counts[k]  += 1.0

            # adapt eps: enlarge
            eps = min(eps * 3.0, eps_max)
            no_improve_streak = 0

            print(f"Iter {t:>7}: score = {best_score:.6e} | pattern {k:<4d} | eps ↑ {eps:.3g}")

        else:                               # FAIL
            no_improve_streak += 1
            if no_improve_streak >= 100:
                eps = max(eps / 2.0, eps_min)
                no_improve_streak = 0
                print(f"Iter {t:>7}: no improv 100× → eps ↓ {eps:.3g}")

        # --- periodic status print -------------------------------------
        if t % print_every == 0:
            print(f"\n--- Iter {t} (eps={eps:.3g}) ---")
            print("Current best score:", f"{best_score:.6e}")
            print("Row sums:", np.round(W.sum(axis=1), 6))
            print("Col sums:", np.round(W.sum(axis=0), 6))
            print("Current W:")
            print(np.array2string(
                W, formatter={'float_kind': lambda z: f'{z:8.5f}'}
            ))
            print()

    # ------------------------------------------------------------------
    print("\n=== Finished ===")
    print("Best score:", f"{best_score:.6e}")
    print("Row sums:", np.round(W.sum(axis=1), 6))
    print("Col sums:", np.round(W.sum(axis=0), 6))
    print("Pattern success counts:", counts.astype(int))
    print("Final W:")
    print(np.array2string(
        W, formatter={'float_kind': lambda z: f'{z:8.5f}'}
    ))
    return W, best_score, counts


# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # Generate and print
    
    W0 = np.array([
    [1.20, 1.10, 1.00, 0.90, 0.80, 1.30, 1.20, 0.50, 1.10, 0.90],
    [0.90, 1.20, 1.10, 1.00, 0.90, 0.80, 1.30, 1.20, 0.50, 1.10],
    [1.10, 0.90, 1.20, 1.10, 1.00, 0.90, 0.80, 1.30, 1.20, 0.50],
    [0.50, 1.10, 0.90, 1.20, 1.10, 1.00, 0.90, 0.80, 1.30, 1.20],
    [1.20, 0.50, 1.10, 0.90, 1.20, 1.10, 1.00, 0.90, 0.80, 1.30],
    [1.30, 1.20, 0.50, 1.10, 0.90, 1.20, 1.10, 1.00, 0.90, 0.80],
    [0.80, 1.30, 1.20, 0.50, 1.10, 0.90, 1.20, 1.10, 1.00, 0.90],
    [0.90, 0.80, 1.30, 1.20, 0.50, 1.10, 0.90, 1.20, 1.10, 1.00],
    [1.00, 0.90, 0.80, 1.30, 1.20, 0.50, 1.10, 0.90, 1.20, 1.10],
    [1.10, 1.00, 0.90, 0.80, 1.30, 1.20, 0.50, 1.10, 0.90, 1.20]
])
    # Example: 5×6 matrix
    search(
    m=10,
    n=10,
    W0=W0,
    max_iter=50_000,
    eps_init=5e-1,
    eps_min=1e-6,
    eps_max=1,
    seed=28,
    print_every=1000
)

