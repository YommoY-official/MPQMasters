import numpy as np
from matplotlib.sphinxext.mathmpl import latex_math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


I2 = np.eye(2,dtype = complex)
X = np.array([[0,1],[1,0]], dtype = complex)
Y = np.array([[0,-1j],[1j,0]], dtype = complex)
Z = np.array([[1,0],[0,-1]], dtype = complex)


def trace_norm(A,ord = 1):

    if ord == 1:
        return float(np.sum(np.linalg.svd(A, compute_uv = False)))
    elif ord == 2:
        return float(np.linalg.norm(A))
    else:
        return None


def haar_unitary(d,rng):
    Z = (rng.normal(size = (d,d)) + 1j*rng.normal(size = (d,d))) / np.sqrt(2)
    Q,R = np.linalg.qr(Z)
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * np.conj(phases)
    return Q

def channel_single_qubit(rho,px,py):
    return (1-px-py) * rho + px * (X @ rho @ X) + py * (Y @ rho @ Y)

def channel_two_qubits_local(rho, px, py):
    ops = [I2,X,Y]
    ws = [1-px-py, px, py]

    out = np.zeros((4,4), dtype = complex)
    for a, wa in zip(ops,ws):
        for b, wb in zip(ops,ws):
            K = np.kron(a,b)
            out += (wa * wb) *(K @ rho @ K.conj().T)
    return out

def ruskai_value_formula(px,py):
    lam_x = 1-2*px
    lam_y = 1-2*py
    lam_z = 1-2*(px+py)
    return max(abs(lam_x),abs(lam_y),abs(lam_z))

def monte_carlo_sup_two_qubit(px,py,n_sample = 20000, seed = 0):
    rng = np.random.default_rng(seed)
    d = 4

    e0 = np.zeros((d,1),dtype = complex); e0[0,0] = 1
    e1 = np.zeros((d,1),dtype = complex); e1[1,0] = 1

    best = -1.0
    for _ in range(n_sample):
        U = haar_unitary(d,rng)
        psi = U @ e0
        phi = U @ e1

        delta = psi @ psi.conj().T - phi @ phi.conj().T
        out = channel_two_qubits_local(delta, px, py)
        val = 0.5 * trace_norm(out)

        if val > best:
            best = val
    return best
def ruskai_mc_diff(px,py,n_sample = 20000, seed = 0):
    return ruskai_value_formula(px,py) - monte_carlo_sup_two_qubit(px,py,n_sample, seed)

if __name__ == '__main__':

    size = 20
    n_sample = 20000

    formula_list = np.zeros((size,size))
    mc_list = np.zeros((size,size))
    px_list = np.linspace(0,1,size)
    py_list = np.linspace(0,1,size)




    for i in range(size):
        for j in range(size):
            if px_list[i] + py_list[j] >= 1:
                formula_list[i,j] = 0
                mc_list[i,j] = 0
            else:
                #r = ruskai_value_formula(px_list[i],py_list[j])
                m = monte_carlo_sup_two_qubit(px_list[i],py_list[j],n_sample = n_sample, seed = 1)
                #print(r-m)
                #formula_list[i,j] = r
                mc_list[i,j] = m


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(px_list, py_list, mc_list)


    ax.set_xlabel("px")
    ax.set_ylabel("py")

    plt.show()

    print("Strictly less than 1:" ,np.all(mc_list) < 1)



