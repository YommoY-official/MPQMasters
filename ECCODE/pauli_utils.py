import numpy as np
from itertools import product

# ---------------------------------------------------------------------------
# Pauli algebra — binary symplectic representation
# ---------------------------------------------------------------------------
# Single-qubit Pauli (ignoring global phase): (x, z) in F_2^2
#   (0,0)=I  (1,0)=X  (1,1)=Y  (0,1)=Z
#
# n-qubit Pauli: (xvec, zvec) in F_2^n x F_2^n
# Multiplication (mod phase):  (x1,z1)*(x2,z2) = (x1^x2, z1^z2)
# Anticommutation:  {P,Q} != 0  iff  xP.zQ + zP.xQ = 1  (mod 2)

PAULI_LABEL   = {(0, 0): 'I', (1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}
PAULI_FROM_CH = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
LOGICAL_NAMES = ['I', 'X', 'Y', 'Z']


def pauli_str(xv, zv):
    """Convert binary symplectic vectors to Pauli string, e.g. 'XZZXI'."""
    return ''.join(PAULI_LABEL[(int(xv[i]), int(zv[i]))] for i in range(len(xv)))


def str_to_pauli(s):
    """Convert Pauli string to (xvec, zvec)."""
    s = s.upper()
    xv = np.array([PAULI_FROM_CH[c][0] for c in s], dtype=int)
    zv = np.array([PAULI_FROM_CH[c][1] for c in s], dtype=int)
    return xv, zv


def pauli_weight(xv, zv):
    """Hamming weight of an n-qubit Pauli (number of non-identity sites)."""
    return int(np.sum(xv | zv))


def all_n_qubit_paulis(n):
    """Enumerate all 4^n Paulis as list of (xvec, zvec)."""
    out = []
    for bits in product([0, 1], repeat=2 * n):
        xv = np.array(bits[0::2], dtype=int)
        zv = np.array(bits[1::2], dtype=int)
        out.append((xv, zv))
    return out

def gate_identity(xv, zv):
    """Trivial gate (I, X, Z, wait): errors unchanged in symplectic frame."""
    return xv.copy(), zv.copy()


def gate_hadamard(xv, zv):
    """Transversal Hadamard H^n: X<->Z on each qubit."""
    return zv.copy(), xv.copy()


def gate_phase_S(xv, zv):
    """Transversal S^n: X->Y, Z->Z on each qubit.  z_i -> x_i ^ z_i."""
    return xv.copy(), (xv ^ zv)


def gate_phase_Sdg(xv, zv):
    """Transversal Sdg^n: X->-Y ~ Y, Z->Z.  Same symplectic action as S."""
    return xv.copy(), (xv ^ zv)


def make_custom_gate(F_x, F_z):
    """
    Build a gate from an explicit symplectic matrix acting blockwise:
        [x']   [F_x]   [x]
        [z'] = [F_z] . [z]   (all arithmetic mod 2)

    F_x, F_z : (n, n) integer arrays over F_2
    """
    F_x = np.asarray(F_x, dtype=int)
    F_z = np.asarray(F_z, dtype=int)

    def _gate(xv, zv):
        return (F_x @ xv) % 2, (F_z @ zv) % 2

    return _gate
