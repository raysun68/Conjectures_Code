import numpy as np
import random
import itertools
from time import time

# --- Parameters ---
def all_partitions_fixed(items, n):
    # Generate all partitions of `items` into `n` non-empty subsets
    for indices in itertools.product(range(n), repeat=len(items)):
        allocation = [[] for _ in range(n)]
        for item_idx, agent in enumerate(indices):
            allocation[agent].append(items[item_idx])
        if all(allocation):  # no empty bundle
            yield allocation

# --- Violation Score ---
def compute_R_RL(valuations, items):
    n = len(valuations)
    best_score = float('inf')
    best_alloc = None
    num = 0
    cnt = 0
    for allocation in all_partitions_fixed(items, n):
        max_violation = float('-inf')
        num = num + 1
        for i in range(n):
            for j in range(n):
                if i == j or not allocation[j]:
                    continue
                for g in allocation[j]:
                    value_other = sum(valuations[i][item] for item in allocation[j] if item != g)
                    value_self = sum(valuations[i][item] for item in allocation[i])
                    violation = value_other - value_self
                    max_violation = max(max_violation, violation)
        if max_violation < best_score:
            best_score = max_violation
            best_alloc = allocation
        if max_violation < 0:
            cnt = cnt + 1
    return best_score, best_alloc, cnt, num

# --- Perturbation ---
def perturb_utilities(U, epsilon):
    # Add noise
    U_new = U + np.random.uniform(-epsilon, epsilon, size=U.shape)
    # Ensure positivity
    U_new = np.maximum(U_new, 1e-8)  # or a smaller threshold if desired
    # Normalize so the average is 1
    U_new /= np.mean(U_new)
    return U_new

# --- NMCS Subroutine ---
def NMCS_util(U_init, steps, score_function, epsilon):
    current_U = U_init.copy()
    current_score = score_function(current_U)

    best_U = current_U.copy()
    best_score = current_score

    for step in range(steps):
        candidate_U = perturb_utilities(current_U, epsilon)
        candidate_score = score_function(candidate_U)

        if candidate_score > current_score:  # maximize score (more violation)
            current_U = candidate_U
            current_score = candidate_score
            if current_score > best_score:
                best_U = current_U.copy()
                best_score = current_score
        else:
            current_U = candidate_U

    print(f"Best score after NMCS loop: {best_score:.4e}")
    print("Current utility matrix after NMCS:")
    print(np.round(best_U, 2))
    return best_U

def AMCS_util(U_init, max_depth=8, max_level=6, max_steps=10, epsilon=0.1):
    score_function = lambda U: compute_R_RL(U, list(range(U.shape[1])))[0]  # negative if no violation, 0+ if violation

    current_U = U_init.copy()
    best_U = current_U.copy()
    best_score = score_function(current_U)

    depth = 0
    level = 1

    print("--- Starting AMCS for EFX Violation Search ---")
    print(f"Initial violation score (negative= no violation, >=0 violation): {best_score:.4e}\n")

    while level <= max_level:
        nmcs_steps = max_steps * level
        print(f"--- Level {level}, Depth {depth} ---")
        next_U = NMCS_util(current_U, steps=nmcs_steps, score_function=score_function, epsilon=epsilon)
        next_score = score_function(next_U)

        if next_score > best_score:
            current_U = next_U.copy()
            best_score = next_score
            best_U = next_U.copy()
            depth = 0
        elif depth < max_depth:
            depth += 1
        else:
            level += 1
            depth = 0

    return best_U, best_score

# --- Main ---
if __name__ == "__main__":
    num_agents = 4
    num_items = 7
    epsilon = 0.2
    min_epsilon = 0.005
    patience_limit = 5

    print("=== EFX Violation Search with AMCS ===")
    print(f"Agents: {num_agents}, Items: {num_items}, Initial ε: {epsilon:.2f}")

    U_init = np.random.rand(num_agents, num_items)

    best_U = U_init.copy()
    best_score = compute_R_RL(best_U, list(range(num_items)))[0]
    no_improvement = 0

    start_time = time()
    round_num = 0

    while no_improvement < patience_limit:
        round_num += 1
        print(f"\n[Round {round_num}] ε = {epsilon:.4f}, No improvement: {no_improvement}")

        U_candidate, candidate_score = AMCS_util(
            best_U,
            max_depth=10,
            max_level=5,
            max_steps=10,
            epsilon=epsilon
        )

        if candidate_score > best_score:
            print(f"Improved: {candidate_score:.4e} > {best_score:.4e}")
            best_score = candidate_score
            best_U = U_candidate.copy()
            no_improvement = 0
            if epsilon > min_epsilon:
                epsilon *= 0.9  # gentle decay
        else:
            no_improvement += 1
            print(f"No improvement: {candidate_score:.4e} ≤ {best_score:.4e}")
            if epsilon > min_epsilon:
                epsilon *= 0.8  # more aggressive decay

    elapsed = time() - start_time

    print("\n=== Search Complete ===")
    print(f"Maximum EFX violation found: {best_score:.4e}")
    print("Final utility matrix:")
    print(np.round(best_U, 2))
    print(f"Elapsed time: {elapsed:.2f} seconds")

    # Optional: save best utility matrix
    np.savetxt("U_efx_violation_best.csv", best_U, fmt="%.4f", delimiter=",")
