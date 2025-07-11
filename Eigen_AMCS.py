import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *

def NMCS_graphon(H, current_W, steps, score_function, epsilon):
    """
    Monte Carlo search with multiplicative 3x3 subgrid perturbations.
    """
    best_W = current_W.copy()
    best_score = score_function(best_W)
    rng = np.random.default_rng()

    for step in range(steps):
        candidate_W, _ = perturb_Eigen(current_W, epsilon)
        candidate_score = score_function(candidate_W)

        if candidate_score > best_score:
            best_W = candidate_W
            best_score = candidate_score

    return best_W


def AMCS_graphon(H, initial_W, max_depth=8, max_level=6, max_steps = 10, epsilon = 0.1):
    """
    The Adaptive Monte Carlo Search algorithm adapted for graphon optimization.
    """
    score_function = lambda W: sidorenko_eigenvalue_check(W)

    print("--- Starting AMCS for Graphons ---")
    current_W = initial_W.copy()
    current_score = score_function(current_W)
    print(f"Initial Score (neg abs gap): {current_score:.4e}")
    print(f"Initial Sidorenko Gap: {sidorenko_eigenvalue_check(current_W):.4e}")

    depth = 0
    level = 1

    while level <= max_level:
        nmcs_steps = max_steps * level
        next_W = NMCS_graphon(H, current_W, steps=nmcs_steps, score_function=score_function, epsilon = epsilon)
        next_score = score_function(next_W)

        if depth == 0:
          print(f"\n--- Trying level {level} ---")
          print(f"Best score (lvl {level}, dpt {depth}, search steps {nmcs_steps}): {max(next_score, current_score):.4e}")
          print(f"Perturbation length: {epsilon}")
          print("New best W:")
          print(np.round(current_W, 3))

              # --- Evaluate score under a slightly contracted W (p = 0.99) ---
          p = 0.99
          W_stretched = current_W + (p - 1) * (current_W - np.mean(current_W))
          try:
              score_stretched = sidorenko_eigenvalue_check(W_stretched)
              if current_score != 0:
                  ratio = score_stretched / current_score
              else:
                  ratio = float('inf') if score_stretched != 0 else 1.0
              print(f"Score ratio of W_stretched (p={p}) vs current_W: {ratio:.4e}")
          except Exception as e:
              print(f"Failed to evaluate stretched W: {e}")

        if next_score > current_score:
            current_W = next_W.copy()
            current_score = next_score
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            depth = 0
            level += 1

    return current_W, abs(sidorenko_eigenvalue_check(current_W))
