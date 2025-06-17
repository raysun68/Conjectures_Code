import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *

def NMCS_graphon(H, current_W, steps, score_function, epsilon):
    """
    This is the "growth" or exploration phase for graphons.
    It performs a local search for a fixed number of steps to find a better W.
    """
    best_W_local = current_W.copy()
    best_score_local = score_function(best_W_local)

    for _ in range(steps):
        # Always perturb from the best matrix found so far in this local search
        candidate_W = perturb(best_W_local, epsilon)
        candidate_score = score_function(candidate_W)
        
        if candidate_score > best_score_local:
            best_W_local = candidate_W
            best_score_local = candidate_score
            
    return best_W_local

def AMCS_graphon(H, initial_W, max_depth=8, max_level=6, max_steps = 10, epsilon = 0.1):
    """
    The Adaptive Monte Carlo Search algorithm adapted for graphon optimization.
    """
    score_function = lambda W: -abs(sidorenko_ratio(H, W)[0])

    print("--- Starting AMCS for Graphons ---")
    current_W = initial_W.copy()
    current_score = score_function(current_W)
    print(f"Initial Score (neg abs gap): {current_score:.4e}")
    print(f"Initial Sidorenko Gap: {sidorenko_ratio(H, current_W)[0]:.4e}")

    depth = 0
    level = 1

    while level <= max_level:
        nmcs_steps = max_steps * level
        next_W = NMCS_graphon(H, current_W, steps=nmcs_steps, score_function=score_function, epsilon = epsilon)
        next_score = score_function(next_W)

        if depth == 0:
            print(f"\n--- Trying level {level} ---")
            print(f"Best score (lvl {level}, dpt {depth}, search steps {nmcs_steps}): {max(next_score, current_score):.4e}")
            print("Perturbation length: {epsilon}")
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

    return current_W, abs(sidorenko_ratio(H, current_W)[0])
