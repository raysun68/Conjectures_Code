import numpy as np
import itertools
import random
from itertools import product
from time import time
from helpers import *
from amcs_graphons import *

if __name__ == "__main__":
    ''' ---- Define the graph H ---- '''
    H = np.array([
        [0,0,0,0,0,0,0,1,1,1], [0,0,0,0,0,1,0,0,1,1], [0,0,0,0,0,1,1,0,0,1],
        [0,0,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,1,1,1,0], [0,1,1,1,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0], [1,0,0,1,1,0,0,0,0,0], [1,1,0,0,1,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0]
    ])

    ''' ---- Load initial W ---- '''
    W_initial = np.loadtxt("W_optimized.csv", delimiter=",")

    ''' ---- Iterative AMCS Loop ---- '''
    patience_limit = 3
    no_improvement_runs = 0
    best_W = W_initial.copy()
    best_score = sidorenko_ratio(H, best_W)[0]

    start_time = time()

    while no_improvement_runs < patience_limit:
        print(f"\n[AMCS Run] No improvement runs: {no_improvement_runs}")

        W_candidate, _ = AMCS_graphon(H, best_W, max_depth=10, max_level=6)
        candidate_score = sidorenko_ratio(H, W_candidate)[0]

        if candidate_score > best_score:
            print(f"Improved: {candidate_score:.6f} > {best_score:.6f}")
            best_W = W_candidate
            best_score = candidate_score
            no_improvement_runs = 0
            np.savetxt("W_optimized.csv", best_W, delimiter=",")
        else:
            print(f"No improvement: {candidate_score:.6f} ≤ {best_score:.6f}")
            no_improvement_runs += 1

    print(f"\nSearch complete. Best Sidorenko ratio: {best_score:.6f}")
    print(f"Total elapsed time: {time() - start_time:.2f} seconds")


#np.array([[0.47778, 0.50628, 0.51637, 0.50856, 0.46856],
 #[0.50628, 0.50467, 0.45889, 0.50019, 0.50719],
 #[0.51645, 0.45889, 0.51398, 0.48558, 0.50236],
 #[0.50856, 0.50019, 0.48558, 0.50931, 0.47397],
 #[0.46856, 0.50719, 0.50236, 0.47397, 0.52521]])
# np.loadtxt("W_optimized.csv", delimiter=",")

 #[0.1, 1.2, 1.3, 1.2],
  #[1.2, 0.2, 1.3, 1.1],
  #[1.3, 1.3, 0.1, 1.2],
  #[1.2, 1.1, 1.2, 0.2]
