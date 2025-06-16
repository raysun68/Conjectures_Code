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

    ''' ---- Define initial W ---- '''
    '''IMPORTANT: W MUST BE SYMMETRIC'''
    # W_initial = np.array([
    #     [0.89, 0.49, 0.35, 0.5], [0.31, 0.78, 0.2, 0.59],
    #     [0.39, 0.71, 0.51, 0.2], [0.11, 0.87, 0.39, 0.68],
    # ])
    W_initial = np.loadtxt("W_optimized.csv", delimiter=",")
    W_initial = W_initial + 0.02 * (W_initial - np.mean(W_initial))

    # --- Run the AMCS Optimization ---
    start_time = time()
    W_final, final_gap = AMCS_graphon(H, W_initial, max_depth = 10, max_level = 6)
    np.savetxt("W_optimized.csv", W_final, delimiter=",")
    print(f"\nTotal search time: {time() - start_time:.2f} seconds")

#[[0.47778 0.50628 0.51637 0.50856 0.46856]
# [0.50628 0.50467 0.45889 0.50019 0.50719]
# [0.51645 0.45889 0.51398 0.48558 0.50236]
# [0.50856 0.50019 0.48558 0.50931 0.47397]
# [0.46856 0.50719 0.50236 0.47397 0.52521]]
# np.loadtxt("W_optimized.csv", delimiter=",")
