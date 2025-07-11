import numpy as np
import itertools
import random
from itertools import product
from numba import njit  # library to optimize performance

def count_homomorphisms_backtrack(H_adj, G_adj):
    """
    Counts the number of homomorphisms from graph H to graph G using backtracking.
    """
    h = H_adj.shape[0]
    g = G_adj.shape[0]
    count = 0
    mapping = [-1] * h

    def backtrack(pos):
        nonlocal count
        if pos == h:
            count += 1
            return
        for candidate in range(g):
            valid = True
            for prev in range(pos):
                if (H_adj[pos, prev] == 1 and G_adj[candidate, mapping[prev]] != 1) or \
                   (H_adj[prev, pos] == 1 and G_adj[mapping[prev], candidate] != 1):
                    valid = False
                    break
            if valid:
                mapping[pos] = candidate
                backtrack(pos + 1)
                mapping[pos] = -1

    backtrack(0)
    return count

def compute_t_H_G_backtrack(H_adj, G_adj):
    """
    Computes the homomorphism density t(H,G) via explicit enumeration.
    """
    g = G_adj.shape[0]
    total_maps = g ** H_adj.shape[0]
    hom_count = count_homomorphisms_backtrack(H_adj, G_adj)
    return hom_count / total_maps

def average_degree(G_adj):
    """
    Returns the average edge density of the adjacency matrix G_adj.
    """
    n = G_adj.shape[0]
    total_degree = np.sum(G_adj)
    return total_degree / (n ** 2)

@njit
def _compute_t_recursive(n, edges, W_block, assignment, pos):
    """
    Recursively computes the weighted contribution of all vertex mappings
    to t(H, W), assuming H has edges `edges`, and W is a stochastic block matrix.
    """
    if pos == n:
        prob = 1.0
        for i in range(len(edges)):
            u, v = edges[i]
            prob *= W_block[assignment[u], assignment[v]]
        return prob

    total_prob = 0.0
    num_blocks = W_block.shape[0]
    for i in range(num_blocks):
        assignment[pos] = i
        total_prob += _compute_t_recursive(n, edges, W_block, assignment, pos + 1)
    
    return total_prob

@njit
def compute_t_G_W(H, W_block):
    """
    Computes t(H, W_block), the homomorphism density of H into a graphon W.
    """
    n = H.shape[0]
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if H[i, j] == 1:
                edge_list.append((i, j))
    edges = np.array(edge_list, dtype=np.int64)

    num_blocks = W_block.shape[0]
    block_volume = 1.0 / num_blocks
    assignment = np.zeros(n, dtype=np.int64)
    total_prob_sum = _compute_t_recursive(n, edges, W_block, assignment, 0)
    
    t = total_prob_sum * (block_volume ** n)
    return t

def sidorenko_gap(H, W_block):
    """
    Computes the Sidorenko gap: p^e(H) - t(H, W) and returns (gap, t, p^e).
    """
    t = compute_t_G_W(H, W_block)
    p = np.mean(W_block)
    num_edges = int(np.sum(H) // 2)
    return p ** num_edges - t, t, p ** num_edges

def sidorenko_ratio(H, W_block):
    """
    Computes log(p^e(H)/t(H, W)), used to quantify Sidorenko ratio violation.
    """
    t = compute_t_G_W(H, W_block)
    p = np.mean(W_block)
    num_edges = int(np.sum(H) // 2)
    return np.log(p ** num_edges / t), t, p ** num_edges

def symmetrize(A):
    """
    Returns a symmetrized version of matrix A.
    """
    return (A + A.T) / 2

def perturb(W, epsilon):
    """
    Applies symmetric zero-mean perturbation to W, normalized to preserve mean.
    """
    perturbation = np.random.uniform(-epsilon, epsilon, size=W.shape)
    perturbation = symmetrize(perturbation)
    perturbation -= np.mean(perturbation)

    W_new = W + perturbation
    W_new = W_new / np.mean(W_new)
    return W_new, perturbation

import numpy as np

def perturb_Eigen(W, epsilon):
    """
    Applies a safe asymmetric perturbation to W such that:
    - All entries remain nonnegative.
    - The mean is preserved (normalization).
    """
    W = np.asarray(W)
    original_mean = np.mean(W)
    
    # Entrywise: max allowable negative perturbation without going below 0
    lower_bounds = -np.minimum(W, epsilon)
    upper_bounds = np.full_like(W, epsilon)
    
    # Sample entrywise perturbations in valid asymmetric ranges
    perturbation = np.random.uniform(lower_bounds, upper_bounds)
    
    # Center the perturbation so total sum is zero (preserves row/col sums roughly)
    perturbation -= np.mean(perturbation)

    # Apply perturbation
    W_new = W + perturbation
    
    # Ensure nonnegativity due to possible rounding errors
    W_new = np.maximum(W_new, 0)

    # Normalize mean back to original
    new_mean = np.mean(W_new)
    if new_mean > 0:
        W_new *= (original_mean / new_mean)
    
    return W_new, perturbation

import numpy as np

def multiplicative_perturbation_subgrid(W, epsilon, grid_size=4, rng=None):
    """
    Multiplies each weight in a random subgrid by a factor in (1 - ε, 1 + ε),
    then rescales the matrix to keep global mean = 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = W.shape
    i0 = rng.integers(0, m - grid_size + 1)
    j0 = rng.integers(0, n - grid_size + 1)

    # Generate multiplicative factors
    mult = rng.uniform(epsilon, 1 / epsilon, size=(grid_size, grid_size))

    # Apply multiplicative perturbation to a subgrid
    W_new = W.copy()
    W_new[i0:i0 + grid_size, j0:j0 + grid_size] *= mult

    # Rescale to keep mean exactly 1
    W_new /= np.mean(W_new)

    return W_new, mult


def perturb_Eigen_fix(W, epsilon, fixed_count=20, seed=43, max_attempts=5000):
    """
    Applies asymmetric perturbation to W, while keeping `fixed_count` randomly chosen
    entries near 1.0 (in range 0.999 to 1.001). These entries remain unperturbed.

    Parameters:
        W: np.ndarray - base weight matrix
        epsilon: float - perturbation magnitude
        fixed_count: int - number of fixed near-1 entries
        seed: int - RNG seed for fixed position selection
        max_attempts: int - number of tries to get non-negative perturbed matrix

    Returns:
        W_new: np.ndarray - perturbed matrix
        perturbation: np.ndarray - actual perturbation (0 at fixed entries)
    """
    original_mean = np.mean(W)
    shape = W.shape
    size = W.size

    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(size, size=fixed_count, replace=False)
    fixed_positions = np.unravel_index(flat_indices, shape)
    fixed_mask = np.zeros_like(W, dtype=bool)
    fixed_mask[fixed_positions] = True

    for _ in range(max_attempts):
        perturbation = np.random.uniform(-epsilon, epsilon, size=W.shape)
        perturbation -= np.mean(perturbation)

        # Zero perturbation for fixed entries
        perturbation[fixed_mask] = 0.0

        W_new = W + perturbation

        # Set fixed entries to values very close to 1
        W_new[fixed_mask] = rng.uniform(0.999, 1.001, size=fixed_count)

        if np.all(W_new >= 0):
            non_fixed_mask = ~fixed_mask
            new_mean = np.mean(W_new[non_fixed_mask])
            if new_mean > 0:
                scale = original_mean / new_mean
                W_new[non_fixed_mask] *= scale
            return W_new, perturbation


    raise RuntimeError("Failed to generate a valid perturbation without negative entries.")

def build_tilde_M(M):
    """
    Constructs the tilde_M matrix from M for the spectral version of Sidorenko checking.
    """
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
    """
    Spectral test for Sidorenko-type inequalities using tilde_M eigenvalue moments.
    Reward = -(2nd largest eigenvalue + smallest eigenvalue).
    """
    M = np.maximum(M, 0)
    M = np.nan_to_num(M, nan=0.0, posinf=1.0, neginf=0.0)

    tilde_M = build_tilde_M(M)

    if not np.all(np.isfinite(tilde_M)):
        raise ValueError("tilde_M contains non-finite values!")

    if not np.allclose(tilde_M, tilde_M.T, atol=1e-10):
        print("Warning: tilde_M is not symmetric")

    eigenvalues = np.linalg.eigvalsh(tilde_M)

    # Sort in increasing order
    lhs = np.sum(eigenvalues**5)
    rhs = (np.sum(M))**15 / (M.size)**10
  # return (rhs / lhs - 1)
    return -(eigenvalues[0] + eigenvalues[-2])

def normalize_to_fixed_Linf_and_zero_mean(P):
    P = P - np.mean(P)
    max_abs = np.max(np.abs(P))
    if max_abs == 0:
        P = np.random.uniform(-1, 1, size=P.shape)
        P -= np.mean(P)
        max_abs = np.max(np.abs(P))
    return P / max_abs

def update_W_strict_Linf(W_prev, P_prev, epsilon=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    delta = rng.uniform(-epsilon, epsilon, size=P_prev.shape)
    P_new = P_prev + delta
    P_new = normalize_to_fixed_Linf_and_zero_mean(P_new)
    W_new = 1 + 0.1 * P_new
    return W_new, P_new
