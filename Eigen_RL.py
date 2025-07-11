import numpy as np

def sidorenko_eigenvalue_check(W):
    W = np.maximum(W, 0)
    W = np.nan_to_num(W, nan=0.0, posinf=1.0, neginf=0.0)
    n = W.shape[0]
    tilde_W = np.zeros((n*n, n*n))

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    idx1 = a * n + b
                    idx2 = c * n + d
                    product = max(W[a, b] * W[c, d], 0)
                    tilde_W[idx1, idx2] = W[c, b] * np.sqrt(product) * W[a, d]

    eigvals = np.linalg.eigvalsh(tilde_W)
    lhs = np.sum(eigvals ** 5)
    rhs = (np.sum(W)) ** 15 / (n * n) ** 10
    return rhs / lhs - 1

def CEM_optimize(score_fn, n_dim=4, pop_size=1000, elite_frac=0.3, n_iter=2000, init_mean=1.0, init_std=0.5):
    mean = np.full((n_dim, n_dim), init_mean)
    std = np.full((n_dim, n_dim), init_std)

    best_W = None
    best_score = -np.inf

    for it in range(n_iter):
        samples = np.random.normal(loc=mean, scale=std, size=(pop_size, n_dim, n_dim))
        samples = np.clip(samples, 1e-3, None)

        scores = np.array([score_fn(W) for W in samples])
        elite_idxs = scores.argsort()[-int(pop_size * elite_frac):]
        elites = samples[elite_idxs]

        mean = np.mean(elites, axis=0)
        std = np.std(elites, axis=0)

        top_score = scores[elite_idxs[-1]]
        if top_score > best_score:
            best_score = top_score
            best_W = samples[elite_idxs[-1]]

        print(f"[Iter {it+1}] Best Score: {best_score:.6f}, Mean score: {np.mean(scores):.6f}")

    return best_W, best_score

# Run the optimization
best_matrix, best_val = CEM_optimize(sidorenko_eigenvalue_check)
print("\nBest matrix:\n", best_matrix)
print("Best Sidorenko ratio error:", best_val)
