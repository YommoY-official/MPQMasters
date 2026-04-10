import numpy as np
from itertools import product

from pauli_utils import (
    PAULI_LABEL, PAULI_FROM_CH, LOGICAL_NAMES,
    pauli_str, str_to_pauli, pauli_weight, all_n_qubit_paulis,
)


# ---------------------------------------------------------------------------
# Transversal gate actions  (xvec, zvec) -> (xvec', zvec')
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# StabilizerCode class
# ---------------------------------------------------------------------------

class StabilizerCode:
    """
    Code-agnostic [[n,1,d]] stabilizer code (single logical qubit).

    Parameters
    ----------
    n : int
        Number of physical qubits.
    generators : list of (xvec, zvec)
        Stabilizer generators, each a pair of np.ndarray of shape (n,) over F_2.
    logical_x : (xvec, zvec)
        Logical X operator representative.
    logical_z : (xvec, zvec)
        Logical Z operator representative.
    name : str, optional
        Human-readable label.

    Attributes
    ----------
    corrections : dict {syndrome_int: (xvec, zvec)}
        Minimum-weight Pauli correction for each syndrome value.
    n_syndromes : int
        Total number of syndromes = 2^r where r = len(generators).
    """

    def __init__(self, n, generators, logical_x, logical_z, name=''):
        self.n = n
        self.generators = generators
        self.logical_x = logical_x
        self.logical_z = logical_z
        self.name = name
        self.r = len(generators)
        self.n_syndromes = 2 ** self.r

        # Pre-compute generator matrices for fast syndrome computation
        self._gx = np.array([gx for gx, _ in generators], dtype=int)  # (r, n)
        self._gz = np.array([gz for _, gz in generators], dtype=int)  # (r, n)

        self.corrections = self._build_corrections()

    # ------------------------------------------------------------------
    # Syndrome
    # ------------------------------------------------------------------

    def compute_syndrome(self, xv, zv):
        """
        Compute r-bit syndrome of Pauli (xv, zv).
        Syndrome bit i = 1 iff Pauli anticommutes with generator i.
        Returns int in [0, 2^r).
        """
        # bits[i] = (xv . gz_i + zv . gx_i) mod 2
        bits = (self._gz @ xv + self._gx @ zv) % 2
        s = 0
        for i in range(self.r):
            s |= int(bits[i]) << (self.r - 1 - i)
        return s

    def syndrome_bits(self, s):
        """Syndrome integer -> bit array of length r (MSB first)."""
        return np.array([(s >> (self.r - 1 - i)) & 1
                         for i in range(self.r)], dtype=int)

    # ------------------------------------------------------------------
    # Logical class
    # ------------------------------------------------------------------

    def logical_class(self, xv, zv):
        """
        Determine the logical Pauli class of a zero-syndrome Pauli.

        Returns
        -------
        idx : int   0=I, 1=X, 2=Y, 3=Z
        bits : tuple (x_L, z_L)
        """
        xl_x, xl_z = self.logical_x
        zl_x, zl_z = self.logical_z
        x_log = int((np.dot(xv, zl_z) + np.dot(zv, zl_x)) % 2)
        z_log = int((np.dot(xv, xl_z) + np.dot(zv, xl_x)) % 2)
        label = (x_log, z_log)
        idx = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}[label]
        return idx, label

    # ------------------------------------------------------------------
    # Correction table
    # ------------------------------------------------------------------

    def _build_corrections(self):
        """
        Build dict {syndrome: (xvec, zvec)} mapping each syndrome to its
        minimum-weight canonical correction Pauli.
        """
        best = {}
        for bits in product([0, 1], repeat=2 * self.n):
            xv = np.array(bits[0::2], dtype=int)
            zv = np.array(bits[1::2], dtype=int)
            s = self.compute_syndrome(xv, zv)
            w = int(np.sum(xv | zv))
            if s not in best or w < best[s][0]:
                best[s] = (w, xv.copy(), zv.copy())
        return {s: (xv, zv) for s, (w, xv, zv) in best.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def correction_str(self, s):
        """Return the correction Pauli for syndrome s as a string."""
        xv, zv = self.corrections[s]
        return pauli_str(xv, zv)

    def __repr__(self):
        return (f"StabilizerCode('{self.name}', n={self.n}, "
                f"r={self.r}, n_syndromes={self.n_syndromes})")


# ---------------------------------------------------------------------------
# ExRec simulation functions
# ---------------------------------------------------------------------------

