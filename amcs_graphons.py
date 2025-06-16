import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *

def NMCS_graphon(H, current_W, steps, score_function):
    """
    This is the "growth" or exploration phase for graphons.
    It performs a local search for a fixed number of steps to find a better W.
    """
    best_W_local = current_W.copy()
    best_score_local = score_function(best_W_local)

    for _ in range(steps):
        # Always perturb from the best matrix found so far in this local search
        candidate_W = perturb(best_W_local)
        candidate_score = score_function(candidate_W)
        
        if candidate_score > best_score_local:
            best_W_local = candidate_W
            best_score_local = candidate_score
            
    return best_W_local

def AMCS_graphon(H, initial_W, max_depth=8, max_level=6):
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

    # Main AMCS loop
    while level <= max_level:
        if depth == 0:
            print(f"\n--- Trying level {level} ---")

        nmcs_steps = 10 * level

        # Run the exploration phase
        next_W = NMCS_graphon(H, current_W, steps=nmcs_steps, score_function=score_function)
        next_score = score_function(next_W)

        # --- Adaptive Logic ---
        if next_score > current_score:
            current_W = next_W.copy()
            current_score = next_score
            depth = 0  # reset
        elif depth < max_depth:
            depth += 1
        else:
            depth = 0
            level += 1

    return current_W, abs(sidorenko_ratio(H, current_W)[0])
            
    # --- Final Results ---
    final_gap, _, _ = sidorenko_ratio(H, current_W)
    print("\n--- AMCS Finished ---")
    print("Final optimized W:")
    print(np.round(current_W, 5))
    print(f"Final Sidorenko gap: {final_gap:.4e}")
    
    return current_W, final_gap
