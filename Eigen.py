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
                    tilde_M[idx1, idx2] = M_cb * np.sqrt(M_ab * M_cd) * M_ad
                    
    return tilde_M

def sidorenko_eigenvalue_check(M):
    m, n = M.shape
    tilde_M = build_tilde_M(M)
    eigenvalues = np.linalg.eigvalsh(tilde_M)  # symmetric -> faster & stable
    lhs = np.sum(eigenvalues**5)
    rhs = (np.sum(M))**15 / (m * n)**10
    print(f"Spectral sum (LHS): {lhs:.10e}")
    print(f"Sidorenko lower bound (RHS): {rhs:.10e}")
    print("Satisfied?" , lhs >= rhs)
    return np.log(rhs / lhs)

# Example usage
M = np.array([
[9.840803879765919193e-01,1.003460339213723129e+00,1.004954416657813265e+00,1.007491862712353248e+00],
[1.003460339213723129e+00,1.014248164971214949e+00,9.980314304936561687e-01,9.842829033637422453e-01],
[1.004954416657813265e+00,9.980314304936561687e-01,9.883362729262412660e-01,1.008678735192904963e+00],
[1.007491862712353248e+00,9.842829033637422453e-01,1.008678735192904963e+00,9.995357988575671593e-01]
])
print(sidorenko_eigenvalue_check(M))
