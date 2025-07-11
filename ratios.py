import numpy as np

def build_tilde_M(M):
    m, n = M.shape
    mn = m * n
    tilde_M = np.zeros((mn, mn))
    for a in range(m):
        for b in range(n):
            for c in range(m):
                for d in range(n):
                    idx1 = a * n + b
                    idx2 = c * n + d
                    M_cb = M[c, b]
                    M_ab = M[a, b]
                    M_cd = M[c, d]
                    M_ad = M[a, d]
                    product = max(M_ab * M_cd, 0)
                    tilde_M[idx1, idx2] = M_cb * np.sqrt(product) * M_ad
    return tilde_M

def sidorenko_eigenvalue_check(M):
    M = np.maximum(M, 0)
    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
    tilde_M = build_tilde_M(M)
    if not np.all(np.isfinite(tilde_M)):
        return float("-inf")
    eigenvalues = np.linalg.eigvalsh(tilde_M)
    lhs = np.sum(eigenvalues**5)
    rhs = (np.sum(M))**15 / (M.size)**10
    return (rhs / lhs - 1)

# --- Main evaluation loop ---
n_trials = 10_000
p = 0.99
ratios = []
raw_scores = []

for _ in range(n_trials):
    base_W = np.ones((4, 4))
    perturbation = np.random.uniform(-1, 1, size=(4, 4))
    perturbation /= (np.max(np.abs(perturbation)) * 40)
    W = base_W + perturbation
    W = np.clip(W, 0.01, None)  # Ensure non-negative weights

    try:
        score = sidorenko_eigenvalue_check(W)
        if not np.isfinite(score) or score == 0:
            continue

        W_stretched = W + (p - 1) * (W - np.mean(W))
        W_stretched = np.clip(W_stretched, 0.01, None)
        score_stretched = sidorenko_eigenvalue_check(W_stretched)

        if np.isfinite(score_stretched):
            ratio = score_stretched / score
            ratios.append(ratio)
            raw_scores.append(score)

    except Exception:
        continue

ratios = np.array(ratios)
raw_scores = np.array(raw_scores)

print(f"\nValid trials: {len(ratios)} / {n_trials}")
print(f"Mean ratio (p={p}): {np.mean(ratios):.6f}")
print(f"Min ratio: {np.min(ratios):.6f}")
print(f"Max ratio: {np.max(ratios):.6f}")
print(f"Fraction with ratio < 1: {np.mean(ratios < 1):.4f}")

print("\n--- Raw Sidorenko score stats ---")
print(f"Mean: {np.mean(raw_scores):.6f}")
print(f"Median: {np.median(raw_scores):.6f}")
print(f"Min: {np.min(raw_scores):.6f}")
print(f"Max: {np.max(raw_scores):.6f}")
print(f"Std Dev: {np.std(raw_scores):.6f}")

