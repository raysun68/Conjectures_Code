import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *
from Eigen_AMCS import *

if __name__ == "__main__":
    ''' ---- Define the graph H ---- '''
    H = np.array([
        [0,0,0,0,0,0,0,1,1,1], [0,0,0,0,0,1,0,0,1,1], [0,0,0,0,0,1,1,0,0,1],
        [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,1,1,1,0], [0,1,1,1,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0], [1,0,0,1,1,0,0,0,0,0], [1,1,0,0,1,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0]
    ])
    #H = np.zeros((11, 11), dtype=int)
    #edges = [
    #(0, 6), (0, 7), (0, 8),
    #(1, 6), (1, 8), (1, 9),
    #(2, 7), (2, 8), (2, 10),
    #(3, 6), (3, 9),
    #(4, 7), (4, 9),
    #(5, 6), (5, 8), (5, 9), (5, 10)
    #]


    #for u, v in edges:
    #    H[u, v] = 1
    #    H[v, u] = 1

    ''' ---- Load initial W ---- '''
    
    W_initial = np.random.rand(9, 9)
    W_initial = W_initial / np.mean(W_initial)
   # W_initial = np.random.uniform(0, 2, size=W.shape)
   # W_initial = symmetrize(W_initial)
   # W_initial -= (np.mean(W_initial) - 1)
   # W_initial = np.loadtxt("W_optimized.csv", delimiter=",")

    ''' ---- Iterative AMCS Loop ---- '''
    patience_limit = 10
    no_improvement_runs = 0
    best_W = W_initial.copy()
    best_score = sidorenko_eigenvalue_check(best_W)
    cur_ep = 0.5
    min_length = 0.0001
    m = np.sum(H) // 2 # Test local condition
    start_time = time()

    with open("W_ratio.csv", "w") as f:
      f.write("# Sidorenko ratios and candidate matrices (every 3rd non-improvement)\n\n")
    
    while no_improvement_runs < patience_limit:
        print(f"\n[AMCS Run] No improvement runs: {no_improvement_runs}")
        
        W_candidate, _ = AMCS_graphon(H, best_W, max_depth=10, max_level=10, max_steps=10, epsilon=cur_ep)
    
        candidate_score = sidorenko_eigenvalue_check(W_candidate)
        candidate_ratio = candidate_score / (np.max(np.abs(W_candidate - 1)) ** 4)
    
        if candidate_score > best_score:
            print(f"Improved: {candidate_score:.4e} > {best_score:.4e}")
            print(f"Corresponding Sidorenko ratio (candidate_ratio): {candidate_ratio:.4e}")
            best_W = W_candidate
            best_score = candidate_score
            no_improvement_runs = 0
            np.savetxt("W_optimized.csv", best_W, delimiter=",")
            cur_ep = min(cur_ep * 3, 0.8)
        else:
            no_improvement_runs += 1
            print(f"No improvement: {candidate_score:.4e} ≤ {best_score:.4e}")
            
            if no_improvement_runs % 3 == 0:
              with open("W_ratio_3.csv", "a") as f:
                f.write(f"# No improvement run {no_improvement_runs}, Ratio: {candidate_ratio:.6e}\n")
                np.savetxt(f, W_candidate, delimiter=",", fmt="%.6f")
                f.write("\n")

            if cur_ep > min_length:
                cur_ep *= 0.5

    
        if np.max(np.abs(best_W - 1)) < 1 / (4 * m):
            print(f"Early stopping: max|W - 1| < {1/(4 * m):.4f}")
            break
    
    # After loop finishes
    print(f"\nSearch complete. Best penalized score: {best_score:.4e}")
    final_ratio = sidorenko_eigenvalue_check(best_W)
    print(f"Final Sidorenko ratio (best_W): {final_ratio:.4e}")
    print(f"Total elapsed time: {time() - start_time:.2f} seconds")

#[[ 0.004  1.762  1.505  0.004  3.029  0.004  4.209  0.004  0.004]
# [ 0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.261]
# [ 0.015  3.018  2.575  0.004  5.187  0.004  7.210  0.004  0.004]
# [ 0.004  0.004  0.004  0.004  0.004  0.004  0.004  0.097  0.004]
# [ 0.004  0.004  0.004  0.046  0.004  0.004  0.004  0.166  0.004]
# [ 0.121  0.004  0.004  0.133  0.004  0.004  0.004  0.227  0.004]
# [ 0.004  3.085  2.635  0.004  5.303  0.004  7.373  0.004  0.004]
# [ 0.004  3.332  2.845  0.004  5.730  0.004  7.962  0.004  0.004]
# [ 0.004  2.171  1.855  0.004  3.733  0.004  5.190  0.004  0.004]]
