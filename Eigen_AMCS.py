import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *

def NMCS_graphon(H, current_W, steps, score_function, epsilon):
    """
    Growth phase for graphons.
    Applies gradually decaying reuse of last successful perturbation.
    """
    best_W_local = current_W.copy()
    best_score_local = score_function(best_W_local)

    last_successful_perturbation = None
    reuse_decay_counter = 0

    for step in range(steps):
        _, random_perturb = perturb_Eigen(best_W_local, epsilon)

        if last_successful_perturbation is not None:
            alpha = min(1.0, reuse_decay_counter / 10)
            perturbation = alpha * random_perturb + (1 - alpha) * last_successful_perturbation
        else:
            perturbation = random_perturb

        candidate_W = best_W_local + perturbation
        candidate_W /= np.mean(candidate_W)

        candidate_score = score_function(candidate_W)

        if candidate_score > best_score_local:
            best_W_local = candidate_W
            best_score_local = candidate_score
            last_successful_perturbation = perturbation
            reuse_decay_counter = 0
        else:
            reuse_decay_counter += 1

    return best_W_local

def AMCS_graphon(H, initial_W, max_depth=8, max_level=6, max_steps = 10, epsilon = 0.1):
    """
    The Adaptive Monte Carlo Search algorithm adapted for graphon optimization.
    """
    score_function = lambda W: -abs(sidorenko_eigenvalue_check(W))

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
