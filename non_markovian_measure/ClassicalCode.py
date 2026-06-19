import numpy as np
import itertools
from typing import Callable, Sequence
class ClassicalCode:
    """
    Classical linear code [[n, k]] defined by a binary parity-check matrix.

    Parameters
    ----------
    H : np.ndarray, shape (m, n)
        Parity-check matrix over F_2.  Stored mod 2.

    Attributes
    ----------
    H       : np.ndarray[int], shape (m, n)
    n       : int  -- number of physical bits
    m       : int  -- number of parity checks  (= n - k)
    k       : int  -- number of logical bits
    d       : int  -- code distance (min weight over nonzero codewords;
                      0 if there are no nonzero codewords)
    dim     : int  -- 2**n, state-space dimension
    S       : list[tuple[int, ...]]  -- all 2**m syndrome labels
    decoder : dict[tuple[int, ...], np.ndarray[int]]
              minimum-weight lookup table: syndrome -> length-n correction bitstring
    """

    _I2: np.ndarray = np.eye(2)
    _X:  np.ndarray = np.array([[0., 1.], [1., 0.]])

    def __init__(self, H: np.ndarray) -> None:
        self.H:       np.ndarray             = np.array(H, dtype=int) % 2
        self.n:       int                    = self.H.shape[1]
        self.m:       int                    = self.H.shape[0]
        self.k:       int                    = self.n - self.m
        self.dim:     int                    = 2 ** self.n
        self.S:       list[tuple[int, ...]]  = list(itertools.product([0, 1], repeat=self.m))
        self.decoder: dict[tuple[int, ...], np.ndarray] = self._build_decoder()
        self.d:       int                    = self.distance()

    # ------------------------------------------------------------------
    # Code operations
    # ------------------------------------------------------------------

    def syndrome(self, e: Sequence[int]) -> tuple[int, ...]:
        """
        Parameters
        ----------
        e : Sequence[int], length n -- binary error pattern (0/1 entries)

        Returns
        -------
        tuple[int, ...], length m -- syndrome H e mod 2
        """
        return tuple(int(b) for b in (self.H @ np.array(e, dtype=int)) % 2)

    def _build_decoder(self) -> dict[tuple[int, ...], np.ndarray]:
        """
        Returns
        -------
        dict mapping each syndrome tuple to its minimum-weight correction
        bitstring (np.ndarray[int], length n)
        """
        table: dict[tuple[int, ...], np.ndarray] = {}
        for w in range(self.n + 1):
            for combo in itertools.combinations(range(self.n), w):
                e = np.zeros(self.n, dtype=int)
                for i in combo:
                    e[i] = 1
                s = self.syndrome(e)
                if s not in table:
                    table[s] = e.copy()
            if len(table) == 2 ** self.m:
                break
        return table

    def distance(self) -> int:
        """
        Code distance: the minimum Hamming weight over all nonzero codewords.

        A codeword is a binary vector c (length n) with zero syndrome
        (H c = 0 mod 2).  All 2**n binary vectors are enumerated and the
        minimum weight among the nonzero codewords is returned.

        Returns
        -------
        int -- minimum nonzero-codeword weight, or 0 if the all-zeros
               vector is the only codeword (k == 0).
        """
        zero_syn = tuple(0 for _ in range(self.m))
        best = 0
        for x in range(1, self.dim):
            bits = self.to_bits(x)
            if self.syndrome(bits) == zero_syn:
                w = sum(bits)
                if best == 0 or w < best:
                    best = w
        return best

    def x_string(self, bits: Sequence[int]) -> np.ndarray:
        """
        Tensor product of single-qubit X gates controlled by a binary string.

        Parameters
        ----------
        bits : Sequence[int], length n -- apply X on position j iff bits[j] == 1

        Returns
        -------
        np.ndarray[complex], shape (dim, dim)
        """
        M = np.array([[1.0]])
        for b in bits:
            M = np.kron(M, self._X if b else self._I2)
        return M

    def R_op(self, s: tuple[int, ...]) -> np.ndarray:
        """
        Recovery (correction) operator for syndrome s: X^{decoder[s]}.

        Parameters
        ----------
        s : tuple[int, ...], length m -- syndrome label (entries in {0, 1})

        Returns
        -------
        np.ndarray[complex], shape (dim, dim)
        """
        return self.x_string(self.decoder[tuple(int(b) % 2 for b in s)])

    def basis_state(self, i: int) -> np.ndarray:
        """
        Pure computational-basis density matrix |i><i|.

        Parameters
        ----------
        i : int -- index in [0, dim)

        Returns
        -------
        np.ndarray[complex], shape (dim, dim)
        """
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[i, i] = 1.0
        return rho

    def to_bits(self, x: int) -> list[int]:
        """
        Integer to n-bit list, MSB first.

        Parameters
        ----------
        x : int -- value in [0, 2**n)

        Returns
        -------
        list[int], length n
        """
        return [(x >> (self.n - 1 - i)) & 1 for i in range(self.n)]

    def P_syndrome(self, sigma: tuple[int, ...]) -> np.ndarray:
        """
        Projector onto computational-basis states whose syndrome equals sigma.

        Parameters
        ----------
        sigma : tuple[int, ...], length m

        Returns
        -------
        np.ndarray[float], shape (dim, dim) -- diagonal projector
        """
        P = np.zeros((self.dim, self.dim))
        for x in range(self.dim):
            if self.syndrome(self.to_bits(x)) == tuple(sigma):
                P[x, x] = 1.0
        return P

    def build_logical_unitary(self) -> np.ndarray:
        """
        Unitary U mapping the physical space C^{2^n} to C^{2^k} tensor C^{2^m}.

        U|x>_phys = |l(x)>_L tensor |s_idx(x)>_S

        where:
          s     = syndrome(x)                       -- syndrome of physical state x
          y     = R(s)|x>                           -- corrected codeword index
          l     = position of y in codespace sorted by integer value
          s_idx = code.S.index(s)                  -- syndrome label index

        U is a real permutation matrix, so U^dagger = U^T.

        Returns
        -------
        np.ndarray[float], shape (dim, dim)
        """
        zero_syn  = tuple(0 for _ in range(self.m))
        dim_S     = 2 ** self.m
        codespace = sorted(y for y in range(self.dim)
                           if self.syndrome(self.to_bits(y)) == zero_syn)
        log_idx   = {y: l for l, y in enumerate(codespace)}

        U   = np.zeros((self.dim, self.dim))
        e_x = np.zeros(self.dim)
        for x in range(self.dim):
            s_x    = self.syndrome(self.to_bits(x))
            e_x[:] = 0.0
            e_x[x] = 1.0
            y      = int(np.argmax(self.R_op(s_x) @ e_x))
            l      = log_idx[y]
            s_idx  = self.S.index(s_x)
            U[l * dim_S + s_idx, x] = 1.0
        return U

    def decoding_table(self) -> dict[tuple[int, ...], list[int]]:
        """Return the decoder as a human-readable dict {syndrome: correction_list}."""
        return {s: list(c) for s, c in self.decoder.items()}

    def __repr__(self) -> str:
        return f"ClassicalCode(n={self.n}, k={self.k}, m={self.m})"

