import numpy as np
import matplotlib.pyplot as plt

v_x = 0.1
v_y = 0.8
v_z = 0.2

m_x = 0.2
m_y = 0.8
m_z = 0.1

k_min, k_max, k_points = 0, 10, 200
s_min, s_max, s_points = 0, 10, 200

lambda0_v = 2*(v_x**2 + v_y**2)/(1 + v_x**2 + v_y**2 + v_z**2)
lambda0_m = 2*(m_x**2 + m_y**2)/(1 + m_x**2 + m_y**2 + m_z**2)

lambda0 = min(lambda0_v, lambda0_m)
print(lambda0)
if lambda0 < 10e-8:
    print("lambda0 is 0")

if v_x**2 + v_y**2 + v_z**2 > 1:
    print("check v bloch vector again")

if m_x**2 + m_y**2 + m_z**2 > 1:
    print("check m bloch vector again")


# =========================
# 1) Pauli basis and helpers
# =========================
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sy = np.array([[0, -1j],
               [1j,  0]], dtype=complex)
sz = np.array([[1,  0],
               [0, -1]], dtype=complex)

SIGMAS = [I2, sx, sy, sz]  # sigma_0, sigma_1, sigma_2, sigma_3

def kron(a, b):
    return np.kron(a, b)

# Precompute sigma_i ⊗ sigma_j for speed
SIGMA_TENSORS = [[kron(SIGMAS[i], SIGMAS[j]) for j in range(4)] for i in range(4)]

# =========================
# 2) Fixed coefficient vectors v and m (EDIT THESE)
#    v[0] and m[0] must be 1
# =========================
# If you need complex entries, add dtype=complex and use complex numbers below.
v = np.array([1.0, v_x, v_y, v_z], dtype=float)
m = np.array([1.0, m_x, m_y, m_z], dtype=float)


# =========================
# 3) Your M_H(k,s) and M_Q(k,s) definitions (EDIT THESE)
#    Must return 4x4 (real or complex) arrays.
# =========================
def M_H(k, s):
    # Example placeholder; replace with your true M_H(k,s)
    A = np.array([
        [2*k-s, k, -k, -s],
        [k , s, s, -k],
        [k, -s, -s, k],
        [-s, -k, -k, -2*k-s]
    ], dtype=float)
    return A

def M_Q(k, s):
    # Example placeholder; replace with your true M_Q(k,s)
    B = np.array([
        [3*k-s, k, -k, k-s],
        [k, -k+s, k+s, -k],
        [k, -k-s, k-s, k],
        [k-s, -k, -k, -k-s]
    ], dtype=float)
    return B

# =========================
# 4) Build operator and compute lowest eigenvalue
# =========================
def build_operator(M_ij, v, m):
    """

    """
    H = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            H += M_ij[i, j] * v[i] * m[j] * SIGMA_TENSORS[i][j]

    return H

def lowest_eigenvalue(M_func, k, s):
    M = M_func(k, s)
    H = build_operator(M, v, m)
    vals = np.linalg.eigvalsh(H)
    return float(np.min(vals).real)

# =========================
# 5) Sweep grids and plot
# =========================
def compute_sign_grid(M_func, k_grid, s_grid):
    Z = np.empty((len(k_grid), len(s_grid)), dtype=int)
    for si, s in enumerate(s_grid):
        for ki, k in enumerate(k_grid):
            val = lowest_eigenvalue(M_func, k, s)
            Z[ki, si] = 1 if val >= 0 else 0
    return Z

# =========================
# 6) Main plotting
# =========================
if __name__ == "__main__":


    k_grid = np.linspace(k_min, k_max, k_points)
    s_grid = np.linspace(s_min, s_max, s_points)

    Z_H = compute_sign_grid(M_H, k_grid, s_grid)
    Z_Q = compute_sign_grid(M_Q, k_grid, s_grid)

    # =========================
    # 7) Plot binary maps
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    cmap = plt.get_cmap('gray', 2)  # blue for negative, red for nonnegative

    im1 = axes[0].imshow(Z_H, extent=[s_min, s_max, k_min, k_max],
                         origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[0].plot([s_min, s_max], [s_min, s_max], 'r', linewidth=1, label = "k=s")

    axes[0].set_xlim(s_min, s_max)
    axes[0].set_ylim(k_min, k_max)
    axes[0].set_title("Not Full Kraus Rank: negative vs non-negative regions")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("k")
    axes[0].legend()

    im2 = axes[1].imshow(Z_Q, extent=[s_min, s_max, k_min, k_max],
                         origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].plot([s_min, s_max], [s_min, s_max], 'r', linewidth=1,label = "k=s")
    if lambda0 > 10e-8:
        axes[1].plot([s_min, s_max], [(1/lambda0)*s_min, (1/lambda0)*s_max], 'b', linewidth=1, label = "lambda condition")
    axes[1].set_xlim(s_min, s_max)
    axes[1].set_ylim(k_min, k_max)
    axes[1].set_title("Full Kraus Rank: negative vs non-negative regions")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("k")
    axes[1].legend()

    cbar = fig.colorbar(im1, ax=axes, ticks=[0, 1], fraction=0.05, pad=0.04)
    cbar.ax.set_yticklabels(['Negative', '≥ 0'])
    cbar.set_label("Lowest eigenvalue sign")

    plt.show()