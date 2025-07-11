import numpy as np

# ----------------------------------------------------------------------
#  Sidorenko / Spectral helpers  
def build_tilde_M(M):
    m, n = M.shape
    tilde = np.zeros((m*n, m*n))
    for a in range(m):
        for b in range(n):
            for c in range(m):
                for d in range(n):
                    idx1 = a*n + b
                    idx2 = c*n + d
                    prod = max(M[a,b]*M[c,d], 0.0)
                    tilde[idx1,idx2] = M[c,b]*np.sqrt(prod)*M[a,d]
    return tilde

def largest_eigenvalue_reward(M):
    """Return the negative of the largest eigenvalue of tilde_M."""
    M = np.maximum(M, 0)
    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
    tilde = build_tilde_M(M)
    if not np.all(np.isfinite(tilde)):
        return -np.inf
    eigvals = np.linalg.eigvalsh(tilde)
    lhs = np.sum(eigvals**5)
    rhs = (np.sum(M))**15 / (M.size)**10
    smallest = eigvals[0]
    second_largest = eigvals[-2]
    return -(second_largest + smallest)
#    return (rhs / lhs - 1)
# ----------------------------------------------------------------------

def generate_patterns_4x4():
    """Return the 9 (n-1)^2 independent 4‑cell patterns."""
    pat = []
    for i in range(3):
        for j in range(3):
            pat.append((i, 3, j, 3))  # (i1,i2,j1,j2)
    return pat  # length 9

def apply_4cell(W, pattern, x):
    """Safely apply 4‑cell update and return new matrix."""
    i1, i2, j1, j2 = pattern
    xmin = -min(W[i1,j2], W[i2,j1])
    xmax =  min(W[i1,j1], W[i2,j2])
    if xmin > xmax:
        return None
    x_clipped = np.clip(x, xmin, xmax)
    if x_clipped == 0.0:
        return None
    Wn = W.copy()
    Wn[i1,j1] += x_clipped
    Wn[i1,j2] -= x_clipped
    Wn[i2,j1] -= x_clipped
    Wn[i2,j2] += x_clipped
    return Wn

def normalize_mean(W):
    """Scale so the mean is exactly 1 (keeps row/col sums at 4)."""
    return W / W.mean()

# ----------------------------------------------------------------------
def search(max_iter=50_000, eps=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    patterns = generate_patterns_4x4()
    K = len(patterns)

    # initial 4×4 matrix with row/col sums 4
    W = np.array([
        [0.91, 1.21, 0.55, 1.33],
        [0.61, 1.27, 0.97, 1.15],
        [1.45, 0.67, 1.09, 0.79],
        [1.03, 0.85, 1.39, 0.73]
    ])
    W = normalize_mean(W)

    best_score = largest_eigenvalue_reward(W)

    # bandit counts (Laplace smoothing)
    counts = np.ones(K, dtype=float)
    print(f"Start score (−max eigenvalue): {best_score:.6e}")

    for t in range(1, max_iter+1):
        probs = counts / counts.sum()
        k = rng.choice(K, p=probs)
        pat = patterns[k]

        x = rng.uniform(-eps, eps)
        W_new = apply_4cell(W, pat, x)
        if W_new is None:
            continue

        W_new = normalize_mean(W_new)
        new_score = largest_eigenvalue_reward(W_new)

        if new_score > best_score:
            W = W_new
            best_score = new_score
            counts[k] += 1.0

        if t % 1000 == 0 or t <= 20:
            print(f"Iter {t:>6}: 👍  −λ_max = {best_score:.6e}  (pattern {k})")
            print("Current W:")
            print(np.array2string(W, formatter={'float_kind':lambda x: f"{x:8.5f}"}))
            print()

       # final report
    print("\n=== Finished ===")
    print("Best −λ_max:", f"{best_score:.6e}")
    print("Row sums:", np.round(W.sum(axis=1), 6))
    print("Col sums:", np.round(W.sum(axis=0), 6))
    print("Pattern success counts:", counts.astype(int))
    print("Final W:")
    print(np.array2string(W, formatter={'float_kind':lambda x: f"{x:8.5f}"}))

    # Print eigenvalues of final tilde_M
    tilde_final = build_tilde_M(W)
    eigvals_final = np.linalg.eigvalsh(tilde_final)
    print("Final tilde_M eigenvalues:")
    print(np.round(eigvals_final, 6))

    return W, best_score, counts


# ----------------------------------------------------------------------
if __name__ == "__main__":
    search(max_iter=100000, eps=1e-1, seed=10)
