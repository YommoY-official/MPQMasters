import numpy as np
import matplotlib.pyplot as plt


def kron(a, b):
    return np.kron(a, b)

def commute(A,B):
    return A @ B - B @ A

def anticommute(A,B):
    return A @ B + B @ A

def hamiltonian(slist,hamlist):

    A = np.zeros((4,4), dtype = complex)

    for i in range(len(hamlist)):
        A += slist[i] * hamlist[i]

    if not np.array_equal(A, np.conjugate(A.T)) :
        print(A-np.conjugate(A.T))
        print("Hamiltonian is not Hermitian")
    return A


def lindbladian_noise(rho, k , jump_op_list):

    A = np.zeros((4,4), dtype = complex)

    for jump_op in jump_op_list:
        A += jump_op @ rho @ np.conjugate(jump_op.T) - 0.5 * anticommute(np.conjugate(jump_op.T) @ jump_op , rho)

    return k * A

def lindbladian_evolution(rho, epsilon, H, k, jump_op_list):
    return rho + epsilon * ( (-1j)*commute(H,rho) + lindbladian_noise(rho, k, jump_op_list) )


def partial_transpose(rho, dims = [2,2], subsystem = 0):
    """
    Compute the partial transpose of a bipartite density matrix.

    Parameters:
        rho : np.ndarray
            The density matrix (square, shape (dA*dB, dA*dB))
        dims : tuple
            Dimensions of subsystems (dA, dB)
        subsystem : int
            0 → transpose A, 1 → transpose B
    """
    dA, dB = dims
    rho = rho.reshape(dA, dB, dA, dB)

    if subsystem == 0:
        # transpose subsystem A
        rho_pt = rho.transpose(2, 1, 0, 3)
    elif subsystem == 1:
        # transpose subsystem B
        rho_pt = rho.transpose(0, 3, 2, 1)
    else:
        raise ValueError("subsystem must be 0 (A) or 1 (B)")

    return rho_pt.reshape(dA * dB, dA * dB)

def lowest_eigenvalue(rho):
    eigvals = np.linalg.eigvalsh(rho)
    l = np.sort(eigvals)
    x = float(l[0])
    y = float(l[1])
    #x = float(np.min(eigvals).real)

    if np.abs(x) < 5 * 10 ** (-4):
        x = 0
    return x,y

def plot_k( rho, epsilon, H, jump_op_list):
    k_range = np.linspace(0, 30, 100)
    k_max = 0
    flag = 0
    min_eigval_list = []
    second_eigval_list = []

    #find k_max
    for k in k_range:
        output_rho = lindbladian_evolution(rho, epsilon, H, k, jump_op_list)
        lowest_eigval, second_lowest_eigval = lowest_eigenvalue(output_rho)
        if flag == 0 and lowest_eigval < 0:
            print(lowest_eigenvalue(output_rho))
            k_max = k
            flag = 1
        lowest_eigval_pt , second_lowest_eigval_pt = lowest_eigenvalue(partial_transpose(output_rho))
        min_eigval_list.append(lowest_eigval_pt)
        second_eigval_list.append(second_lowest_eigval_pt)

    plt.plot(k_range, min_eigval_list,'*')
    #plt.plot(k_range, second_eigval_list,'o')
    yrange = np.linspace(min(min_eigval_list),max(min_eigval_list),30)
    plt.plot( [k_max for __ in range(len(yrange))], yrange  , color = 'red', label = "First order DM Negative")
    plt.plot(k_range, [0 for __ in range(len(k_range))], color = 'blue')
    plt.title("Kappa vs Lowest Eigenvalue of PT of first order Lindblad")
    plt.xlabel("Kappa")
    plt.ylabel("Lowest Eigenvalue")
    plt.legend()
    plt.show()

def check_dm(rho):
    if not np.array_equal(rho, np.conjugate(rho.T)):
        return False

    vals = np.linalg.eigvalsh(rho)
    if np.min(vals) < 0:
        return False

    if np.trace(rho) != 1:
        return False

    return True


I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sy = np.array([[0, -1j],
               [1j,  0]], dtype=complex)
sz = np.array([[1,  0],
               [0, -1]], dtype=complex)
SIGMAS = [I2, sx, sy, sz]  # sigma_0, sigma_1, sigma_2, sigma_3



#================Define ================
epsilon = 0.01
k = 0.1
s = 0.1

# initial density matrix
initial_rho = np.array([
    [ 1 , 0 , 0 , 1 ],
    [ 0 , 0 , 0 , 0 ],
    [ 0 , 0 , 0 , 0 ],
    [ 1 , 0 , 0 , 1  ]

])

initial_rho = initial_rho / np.trace(initial_rho)

if not check_dm(initial_rho):
    print("Initial density matrix is invalid")

# jump operators
jump_op_list = [kron(sx,sz), kron(sy, sz), kron(sz,sx), kron(sz,sy)]

# Hamiltonian

s_list = [1 , 1 , 1 , 1]

hamil_list = [kron(sx,sz), kron(sy, sz), kron(sz, sx), kron(sz, sy)]

H = hamiltonian(s_list,hamil_list)


#====================================================================================


plot_k(initial_rho, epsilon, H, jump_op_list)

