import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.01

v_x = 0.3
v_y = 0.1
v_z = 0.3

m_x = 0.3
m_y = 0.3
m_z = 0.1

k_min, k_max, k_points = 0, 50, 100
s_min, s_max, s_points = 0, 50, 100

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

def build_operator(M_ij):
    """

    """
    H = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            H += M_ij[i, j]  * SIGMA_TENSORS[j][i]

    return H

def notFullKraus_lind_noise(k):

    A = np.array( [
        [0, v[1]*m[0], v[2]*m[0], 2 * v[3]*m[0]],
        [v[0]*m[1], 2*v[1]*m[1], 2*v[2]*m[1], 3*v[3]*m[1]],
        [v[0]*m[2], 2*v[1]*m[2], 2*v[2]*m[2], 3*v[3]*m[2]],
        [2*v[0]*m[3], 3*v[1]*m[3], 3*v[2]*m[3], 4*v[3]*m[3]]
    ])

    return -0.5 * k * build_operator(A)

def notFullKraus_lind_noise_PT(k):

    A = np.array([
        [0, v[1] * m[0], -v[2] * m[0], v[3] * m[0]],
        [v[0] * m[1], 2 * v[1] * m[1], -2 * v[2] * m[1], 3 * v[3] * m[1]],
        [v[0] * m[2], 2 * v[1] * m[2], -2 * v[2] * m[2], 3 * v[3] * m[2]],
        [2 * v[0] * m[3], 3 * v[1] * m[3], -3 * v[2] * m[3], 4 * v[3] * m[3]]

    ])

    return -0.5 * k * build_operator(A)


def lind_hamiltonian_zz(s):

    H = np.array([
        [0 , 0, 0, 0],
        [0, 0, 0, -m[2]],
        [0,0,0, m[1]],
        [0, -v[2], v[1], 0]

    ])

    return 0.5 * s * build_operator(H)

def lind_hamiltonian_zz_PT(s):

    H = np.array([
        [0 , -v[2]*m[3] , -v[1]*m[3], 0],
        [-m[2]*v[3] , 0, 0, -m[2]],
        [m[1]*v[3], 0, 0, m[1]],
        [0 , -v[2], -v[1], 0]

    ])
    return 0.5 * s * build_operator(H)

def notFullKraus_dm(k,s):
    A = np.array([
        [1,v[1],v[2],v[3]],
        [m[1],v[1]*m[1],v[2]*m[1],v[3]*m[1]],
        [m[2],v[1]*m[2],v[2]*m[2],v[3]*m[2]],
        [m[3], v[1]*m[3], v[2]*m[3],v[3]*m[3]]
    ])

    return 0.25 * build_operator(A) + epsilon * (notFullKraus_lind_noise(k) + lind_hamiltonian_zz(s))


def notFullKraus_dm_PT(k,s):
    A = np.array([
        [1, v[1], -v[2], v[3]],
        [m[1], v[1] * m[1], -v[2] * m[1], v[3] * m[1]],
        [m[2], v[1] * m[2], -v[2] * m[2], v[3] * m[2]],
        [m[3], v[1] * m[3], -v[2] * m[3], v[3] * m[3]]
    ])

    return 0.25 * build_operator(A) + epsilon * (notFullKraus_lind_noise_PT(k) + lind_hamiltonian_zz_PT(s))
# =========================
# 4) Build operator and compute lowest eigenvalue
# =========================


def lowest_eigenvalue(M_func, k, s):
    M = M_func(k, s)
    #H = build_operator(M)
    vals = np.linalg.eigvalsh(M)
    #print(np.min(vals).real)
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

    #print(build_operator(np.array([[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])))

    k_grid = np.linspace(k_min, k_max, k_points)
    s_grid = np.linspace(s_min, s_max, s_points)

    Z_notFull = compute_sign_grid(notFullKraus_dm, k_grid, s_grid)
    Z_notFull_PT = compute_sign_grid(notFullKraus_dm_PT, k_grid, s_grid)


    # =========================
    # 7) Plot binary maps
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    cmap = plt.get_cmap('gray', 2)  # blue for negative, red for nonnegative

    im1 = axes[0].imshow(Z_notFull, extent=[s_min, s_max, k_min, k_max],
                         origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[0].plot([s_min, s_max], [s_min, s_max], 'r', linewidth=1, label = "k=s")

    axes[0].set_xlim(s_min, s_max)
    axes[0].set_ylim(k_min, k_max)
    axes[0].set_title("Density Matrix : Not Full Kraus Rank")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("k")
    axes[0].legend()

    im2 = axes[1].imshow(Z_notFull_PT, extent=[s_min, s_max, k_min, k_max],
                         origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].plot([s_min, s_max], [s_min, s_max], 'r', linewidth=1,label = "k=s")
    if lambda0 > 10e-8:
        axes[1].plot([s_min, s_max], [(1/lambda0)*s_min, (1/lambda0)*s_max], 'b', linewidth=1, label = "lambda condition")
    axes[1].set_xlim(s_min, s_max)
    axes[1].set_ylim(k_min, k_max)
    axes[1].set_title("Partial Transpose : Not Full Kraus Rank")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("k")
    axes[1].legend()

    cbar = fig.colorbar(im1, ax=axes, ticks=[0, 1], fraction=0.05, pad=0.04)
    cbar.ax.set_yticklabels(['Negative', '≥ 0'])
    cbar.set_label("Lowest eigenvalue sign")

    plt.show()