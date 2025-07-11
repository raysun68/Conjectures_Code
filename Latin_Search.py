import numpy as np
import torch

torch.set_default_dtype(torch.float64)  # Global default

##########################
# Latin Square Sampler (Theorem 7)
##########################

def init_cyclic_latin(n):
    return np.array([[ (i + j) % n for j in range(n)] for i in range(n)], dtype=int)

def verify_latin(L):
    n = L.shape[0]
    for r in range(n):
        if len(set(L[r])) != n:
            return False
    for c in range(n):
        if len(set(L[:, c])) != n:
            return False
    return True

def find_cycle_in_row_pair_graph(L, r1, r2, start_col):
    n = L.shape[1]
    cycle = [start_col]
    current_col = start_col
    visited = set([start_col])
    while True:
        symbol = L[r1, current_col]
        next_cols = np.where(L[r2] == symbol)[0]
        if len(next_cols) != 1:
            return None
        next_col = next_cols[0]
        if next_col in visited:
            break
        cycle.append(next_col)
        visited.add(next_col)
        current_col = next_col
    return cycle

def swap_two_row_cycle(L, r1, r2, cycle):
    L2 = L.copy()
    for c in cycle:
        L2[r1, c], L2[r2, c] = L[r2, c], L[r1, c]
    return L2

def find_pivot_third_row(L, r1, r2, cycle, l):
    col_l = cycle[l-1]
    s = L[r1, col_l]
    n = L.shape[0]
    for t in range(n):
        if t != r1 and t != r2 and L[t, col_l] == s:
            return t, s, col_l
    return None, None, None

def second_stage_toggle(L, r1, t, s):
    cols_r1 = np.where(L[r1] == s)[0]
    cols_t = np.where(L[t] == s)[0]
    for col_start in np.random.permutation(np.concatenate((cols_r1, cols_t))):
        cycle = find_cycle_in_row_pair_graph(L, r1, t, col_start)
        if cycle is None:
            continue
        L2 = swap_two_row_cycle(L, r1, t, cycle)
        if verify_latin(L2):
            return L2
    return L

def sample_latin_theorem7(n, steps=10000):
    L = init_cyclic_latin(n)
    successful_moves = 0

    for step in range(steps):
        r1, r2 = np.random.choice(n, 2, replace=False)
        l = np.random.randint(2, n+1)
        s = np.random.choice(L[r1])
        start_cols = np.where(L[r1] == s)[0]
        if len(start_cols) != 1:
            continue
        start_col = start_cols[0]
        cycle = find_cycle_in_row_pair_graph(L, r1, r2, start_col)
        if cycle is None:
            continue

        if len(cycle) <= l:
            L2 = swap_two_row_cycle(L, r1, r2, cycle)
            if verify_latin(L2):
                L = L2
                successful_moves += 1
        else:
            partial_cycle = cycle[:l]
            L2 = swap_two_row_cycle(L, r1, r2, partial_cycle)
            t, pivot_symbol, pivot_col = find_pivot_third_row(L2, r1, r2, cycle, l)
            if t is not None:
                L3 = second_stage_toggle(L2, r1, t, pivot_symbol)
                if verify_latin(L3):
                    L = L3
                    successful_moves += 1
        if step % 1000 == 0 or step == steps - 1:
            print(f"Latin Chain Step {step} | Successful moves: {successful_moves}")
    return L

##########################
# Sidorenko Optimization
##########################

def build_tilde_M(M):
    m, n = M.shape
    mn = m * n
    tilde_M = np.zeros((mn, mn), dtype=np.float64)
    for a in range(m):
        for b in range(n):
            for c in range(m):
                for d in range(n):
                    idx1 = a * n + b
                    idx2 = c * n + d
                    prod = max(M[a, b] * M[c, d], 0.0)
                    tilde_M[idx1, idx2] = M[c, b] * np.sqrt(prod) * M[a, d]
    return tilde_M

def sidorenko_reward(M):
    M = M.astype(np.float64)
    M = np.maximum(M, 0)
    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)
    tilde = build_tilde_M(M)
    if not np.all(np.isfinite(tilde)):
        return -float("inf"), np.array([])
    eigs = np.linalg.eigvalsh(tilde)
    lhs = np.sum(eigs ** 5)
    rhs = (np.sum(M)) ** 15 / (M.size) ** 10
    return -(eigs[-2] + eigs[0]), eigs

def optimize_weights(n=9, steps=3000, epsilon=0.05, seed=34):
    np.random.seed(seed)
    torch.manual_seed(seed)

    L = sample_latin_theorem7(n, steps=150000)
    print("Sampled Latin Square:\n", L)

    L_tensor = torch.tensor(L, dtype=torch.long)

    weights = torch.tensor(np.random.rand(n), dtype=torch.float64)
    weights = torch.clamp(weights, min=1e-6)
    weights /= weights.mean()

    best_score, best_eigs = sidorenko_reward(weights[L_tensor].numpy())
    best_weights = weights.clone()

    for step in range(steps):
        noise = torch.randn_like(best_weights) * epsilon
        proposal = best_weights + noise
        proposal = torch.clamp(proposal, min=1e-6)
        proposal /= proposal.mean()

        score, eigs = sidorenko_reward(proposal[L_tensor].numpy())
        if score > best_score:
            best_score = score
            best_weights = proposal.clone()
            best_eigs = eigs
            epsilon *= 1.2

        if step % 1000 == 0 or step == steps - 1:
            epsilon *= 0.6
            print(f"Step {step:4d} | Score: {score:+.12e} | Best: {best_score:+.12e} | Weights: {best_weights.numpy()}")
            if step == steps - 1:
                print("Best eigenvalues of tilde matrix:\n", best_eigs)

    return best_weights.numpy(), L

# Entry point
if __name__ == "__main__":
    optimize_weights(n=15, steps=20000, epsilon=0.2, seed=100)
