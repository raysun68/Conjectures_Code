import numpy as np
import itertools
import random
from itertools import product
from numba import njit  # library to optimize performance

def count_homomorphisms_backtrack(H_adj, G_adj):
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
    g = G_adj.shape[0]
    total_maps = g ** H_adj.shape[0]
    hom_count = count_homomorphisms_backtrack(H_adj, G_adj)
    return hom_count / total_maps

def average_degree(G_adj):
    n = G_adj.shape[0]
    total_degree = np.sum(G_adj)
    return total_degree / (n ** 2)


@njit # numba decorator
def _compute_t_recursive(n, edges, W_block, assignment, pos):
    """
    A Numba-friendly recursive helper to replace itertools.product.
    This function calculates the sum of probabilities for all possible assignments.
    """
    # Base Case: If the assignment is fully built (pos == n)
    if pos == n:
        # Calculate the probability for this one complete assignment
        prob = 1.0
        for i in range(len(edges)):
            u, v = edges[i]
            prob *= W_block[assignment[u], assignment[v]]
        return prob

    # Recursive Step: Iterate through possibilities for the current position
    total_prob = 0.0
    num_blocks = W_block.shape[0]
    for i in range(num_blocks):
        assignment[pos] = i
        # Recurse to fill the next position and add the result
        total_prob += _compute_t_recursive(n, edges, W_block, assignment, pos + 1)
    
    return total_prob

@njit # numba decorator
def compute_t_G_W(H, W_block):
    """
    This function uses the recursive helper to perform its calculation
    in a Numba-compatible way.
    """
    n = H.shape[0]
    
    # Create a list of edges.
    # Creating a numpy array for numba
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if H[i, j] == 1:
                edge_list.append((i, j))
    edges = np.array(edge_list, dtype=np.int64)

    num_blocks = W_block.shape[0]
    block_volume = 1.0 / num_blocks
    
    # call to the recursive function
    assignment = np.zeros(n, dtype=np.int64)
    total_prob_sum = _compute_t_recursive(n, edges, W_block, assignment, 0)
    
    t = total_prob_sum * (block_volume ** n)
    return t

def sidorenko_gap(H, W_block):
    t = compute_t_G_W(H, W_block)
    p = np.mean(W_block)
    num_edges = int(np.sum(H) // 2)
    return p ** num_edges - t, t, p ** num_edges

def sidorenko_ratio(H, W_block):
    t = compute_t_G_W(H, W_block)
    p = np.mean(W_block)
    num_edges = int(np.sum(H) // 2)
    return np.log(p ** num_edges / t), t, p ** num_edges

def symmetrize(A):
    return (A + A.T) / 2

def perturb(W, epsilon):
    # Add small symmetric zero-mean noise
    perturbation = np.random.uniform(-epsilon, epsilon, size=W.shape)
    perturbation = symmetrize(perturbation)
    perturbation -= np.mean(perturbation)

    W_new = W + perturbation
    W_new = W_new / np.mean(W_new)  # Ensure normalization
    return W_new, perturbation

def perturb_Eigen(W, epsilon, max_attempts=5000):
    """
    Add asymmetric, zero-mean perturbation to W,
    rejecting any that lead to negative entries.
    """
    original_mean = np.mean(W)

    for _ in range(max_attempts):
        perturbation = np.random.uniform(-epsilon, epsilon, size=W.shape)
        perturbation -= np.mean(perturbation)

        W_new = W + perturbation

        if np.all(W_new >= 0):
            # Renormalize to preserve original mean
            new_mean = np.mean(W_new)
            if new_mean > 0:
                W_new *= (original_mean / new_mean)
            return W_new, perturbation
    raise RuntimeError("Failed to generate a valid perturbation without negative entries.")

def optimize_graphon(H, W, steps=100):
    best_gap, best_t, best_p_e = sidorenko_ratio(H, W)
    W_best = W.copy()

    for step in range(steps):
        W_new = perturb(W)
        new_gap, new_t, new_p_e = sidorenko_ratio(H, W_new)
        delta = abs(new_gap) - abs(best_gap)

        if delta < 0:
            W = W_new
            if abs(new_gap) < abs(best_gap):
                W_best = W.copy()
                best_gap, best_t, best_p_e = new_gap, new_t, new_p_e

    return W_best, best_gap, best_t, best_p_e
  
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
        raise ValueError("tilde_M contains non-finite values!")

    if not np.allclose(tilde_M, tilde_M.T, atol=1e-10):
        print("Warning: tilde_M is not symmetric")

    eigenvalues = np.linalg.eigvalsh(tilde_M)
    lhs = np.sum(eigenvalues**5)
    rhs = (np.sum(M))**15 / (M.size)**10
    return np.log(rhs / lhs)
