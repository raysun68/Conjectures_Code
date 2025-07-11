import numpy as np

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
        return float("-inf"), None
    eigenvalues = np.linalg.eigvalsh(tilde_M)
    lhs = np.sum(eigenvalues**5)
    rhs = (np.sum(M))**15 / (M.size)**10
    return rhs / lhs - 1, eigenvalues

# Parse into numpy array
#W = np.array([
#[0.950079,1.200000,0.950051,0.950104,0.950112],
#[0.993628,0.950084,1.012875,1.020456,1.022866],
#[1.015341,0.950120,1.013134,1.024315,0.997005],
#[1.019360,0.950042,0.997448,1.013928,1.019137],
#[1.021502,0.950101,1.026407,0.991108,1.010796]
#])
W = np.array([
    [1.20, 0.90, 0.80, 1.10],
    [0.80, 1.20, 1.10, 0.90],
    [1.10, 0.80, 0.90, 1.20],
    [0.90, 1.10, 1.20, 0.80]
])
W = np.array([
    [1.0206547 , 1.039586  , 0.9773796 , 0.9891711 , 0.9901293 , 0.98908746, 1.0403948 , 0.970496  , 0.98310065],
    [1.039586  , 0.9773796 , 0.9891711 , 0.9901293 , 0.98908746, 1.0403948 , 0.970496  , 0.98310065, 1.0206547 ],
    [0.9773796 , 0.9891711 , 0.9901293 , 0.98908746, 1.0403948 , 0.970496  , 0.98310065, 1.0206547 , 1.039586  ],
    [0.9891711 , 0.9901293 , 0.98908746, 1.0403948 , 0.970496  , 0.98310065, 1.0206547 , 1.039586  , 0.9773796 ],
    [0.9901293 , 0.98908746, 1.0403948 , 0.970496  , 0.98310065, 1.0206547 , 1.039586  , 0.9773796 , 0.9891711 ],
    [0.98908746, 1.0403948 , 0.970496  , 0.98310065, 1.0206547 , 1.039586  , 0.9773796 , 0.9891711 , 0.9901293 ],
    [1.0403948 , 0.970496  , 0.98310065, 1.0206547 , 1.039586  , 0.9773796 , 0.9891711 , 0.9901293 , 0.98908746],
    [0.970496  , 0.98310065, 1.0206547 , 1.039586  , 0.9773796 , 0.9891711 , 0.9901293 , 0.98908746, 1.0403948 ],
    [0.98310065, 1.0206547 , 1.039586  , 0.9773796 , 0.9891711 , 0.9901293 , 0.98908746, 1.0403948 , 0.970496  ]
])
#W = np.array([
#[1.022544834903217570e+00,9.882808449694164832e-01,9.980743087708238148e-01,9.911096412234395858e-01],
#[9.897054112579033447e-01,1.000636987894977903e+00,9.851475614048055274e-01,1.024501242676079782e+00],
#[9.903961333865767269e-01,9.877938668725897431e-01,1.022352820449602184e+00,9.994461262333801388e-01],
#[9.973566180616084687e-01,1.023287326145273290e+00,9.944129254417017894e-01,9.849533503086042030e-01]
#])

# Run Sidorenko eigenvalue check
score, eigenvalues = sidorenko_eigenvalue_check(W)

# Output
print("W =")
print(np.array2string(W, formatter={'float_kind':lambda x: f'{x:0.8f}'}))
print("\nRow sums:", np.round(W.sum(axis=1), 8))
print("Col sums:", np.round(W.sum(axis=0), 8))
print("\nSidorenko eigenvalue ratio (rhs / lhs - 1):", f"{score:.6e}")
print("\nEigenvalues of tilde_M:")
print(np.round(eigenvalues, 8))
print(eigenvalues[2] + eigenvalues[-4])
print(np.linalg.matrix_rank(W))
