"""
Logical-space channel for the QEC non-Markovianity study.

The logical channel is obtained by:
  1. embedding a logical input rho_L into the zero-syndrome codespace sector,
  2. evolving the full physical state under the syndrome-resolved transfer map
     Lambda_t = sum_{e1..eT} prod_i p(ei) * Q(eT,eT-1)...Q(e2,e1) * Q(e1, e_in),
  3. rotating into the (logical tensor syndrome) basis and tracing out syndrome.

  Phi_L(t)[rho_L] = Tr_S[ U * Lambda_t(rho_phys) * U^dagger ]

The snapshots Lambda_t (physical superoperators) and the transfer
superoperators SQ are built once in __init__ and reused for every t.
"""

import itertools
import numpy as np

from ClassicalCode import ClassicalCode
from error_models import (
    ErrorModel,
    SyndromeProcess,
    iid_bitflip_error,
    iid_syndrome,
)


# ------------------------------------------------------------------
# Vectorisation helpers
# Column-major vectorization: rho -> K rho K-dagger becomes (K* kron K) vec(rho).
# ------------------------------------------------------------------

def to_super(K: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    K : np.ndarray[complex], shape (d, d)

    Returns
    -------
    np.ndarray[complex], shape (d^2, d^2) -- superoperator K* kron K
    """
    return np.kron(K.conj(), K)


def vec(rho: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    rho : np.ndarray[complex], shape (d, d)

    Returns
    -------
    np.ndarray[complex], shape (d^2,) -- column-major vectorisation
    """
    return rho.reshape(-1, order='F')


def unvec(v: np.ndarray, dim: int) -> np.ndarray:
    """
    Parameters
    ----------
    v   : np.ndarray[complex], shape (d^2,)
    dim : int -- d

    Returns
    -------
    np.ndarray[complex], shape (d, d) -- column-major un-vectorisation
    """
    return v.reshape(dim, dim, order='F')


def partial_trace_syndrome(sigma: np.ndarray, code: ClassicalCode) -> np.ndarray:
    """
    Partial trace over the syndrome (second) tensor factor of sigma.

    sigma is interpreted as acting on C^{2^k} tensor C^{2^m} with row/col index
    l * 2^m + s_idx.  The syndrome index is contracted:
        (rho_L)[l, l'] = sum_s  sigma[l * 2^m + s,  l' * 2^m + s]

    Parameters
    ----------
    sigma : np.ndarray[complex], shape (dim, dim)
            density matrix in the (logical tensor syndrome) product basis
    code  : ClassicalCode

    Returns
    -------
    np.ndarray[complex], shape (2^k, 2^k) -- reduced logical density matrix
    """
    dim_L = 2 ** code.k
    dim_S = 2 ** code.m
    return np.einsum('iaja->ij', sigma.reshape(dim_L, dim_S, dim_L, dim_S))


class LogicalChannel:
    """
    Snapshots of the logical channel Phi_L(t) for t = 0..T at fixed noise rate q.

    The channel object is state-independent: build it once, then apply it to any
    logical density matrix via  channel(t, rho_L)  (i.e. __call__).

    Parameters
    ----------
    code          : ClassicalCode
    T             : int                   -- maximum time step (snapshots 0..T)
    q             : float                 -- physical noise parameter (passed to p_error)
    p_error       : ErrorModel            -- physical (data-qubit) error model;
                                             default iid_bitflip_error
    syndrome      : SyndromeProcess | None -- syndrome-readout error process; defaults
                                             to iid_syndrome(code, q).  Its `memory`
                                             attribute sets how many previous syndrome
                                             labels each step conditions on (0 = i.i.d.
                                             Markovian, >= 1 = temporally correlated /
                                             non-Markovian)

    Attributes
    ----------
    code  : ClassicalCode
    T     : int
    q     : float
    U     : np.ndarray[float], shape (dim, dim)  -- code.build_logical_unitary()
    chans : dict[int, np.ndarray]                -- {t: physical superoperator (dim^2, dim^2)}
    """

    def __init__(
        self,
        code: ClassicalCode,
        T: int,
        q: float,
        p_error: ErrorModel = iid_bitflip_error,
        syndrome: SyndromeProcess | None = None,
        classical: bool = True,
    ) -> None:
        if syndrome is None:
            syndrome = iid_syndrome(code, q)
        self.code:      ClassicalCode   = code
        self.T:         int             = T
        self.q:         float           = q
        self.p_error:   ErrorModel      = p_error
        self.syndrome:  SyndromeProcess = syndrome
        self.classical: bool            = classical

        self.U:     np.ndarray = code.build_logical_unitary()
        self._prepare_logical_labels()   # self._codespace, self._log_label

        # cached pieces for the (superoperator) logical-space wrapper
        dim_S          = 2 ** code.m
        e_in_idx       = code.S.index(tuple(0 for _ in range(code.m)))
        self._zero_syn = np.zeros((dim_S, dim_S))
        self._zero_syn[e_in_idx, e_in_idx] = 1.0

        # classical (diagonal) fast path builds cheap 2^n x 2^n stochastic snapshots;
        # the full complex 2^(2n) x 2^(2n) superoperators are built only when needed
        # (classical=False), e.g. for coherent inputs / genuine quantum codes.
        if classical:
            self.chans:  dict[int, np.ndarray] | None = None
            self.cchans: dict[int, np.ndarray] | None = self._build_classical_channels()
        else:
            self.chans  = self._build_logical_channels()
            self.cchans = None

    def _prepare_logical_labels(self) -> None:
        """
        Precompute, for the classical fast path:
          self._codespace : np.ndarray[int], shape (2^k,)  -- physical index of each
                            logical basis state (the sorted zero-syndrome codewords)
          self._log_label : np.ndarray[int], shape (2^n,)  -- logical label l(x) of
                            every physical basis index x (decode syndrome, correct,
                            look up codeword position).  Mirrors build_logical_unitary
                            without building the (dim, dim) permutation matrix.
        """
        code = self.code
        n = code.n
        zero_syn  = tuple(0 for _ in range(code.m))
        codespace = sorted(y for y in range(code.dim)
                           if code.syndrome(code.to_bits(y)) == zero_syn)
        log_idx   = {y: l for l, y in enumerate(codespace)}

        def frombits(bits) -> int:                    # inverse of code.to_bits (MSB first)
            v = 0
            for i, b in enumerate(bits):
                v |= int(b) << (n - 1 - i)
            return v

        self._frombits = frombits
        self._codespace = np.array(codespace, dtype=int)
        log_label = np.empty(code.dim, dtype=int)
        for x in range(code.dim):
            s = code.syndrome(code.to_bits(x))
            y = x ^ frombits(code.decoder[s])         # apply recovery R(s) = X^{decoder[s]}
            log_label[x] = log_idx[y]
        self._log_label = log_label

    # ------------------------------------------------------------------
    # Channel construction
    # ------------------------------------------------------------------

    def _build_SQ(self) -> dict[tuple[int, ...], dict[tuple[int, ...], np.ndarray]]:
        """
        Build the transition superoperator SQ[e_i][e_{i-1}] for one QEC step.

        SQ[ei][eim1] = sum_{e~} p_error(e~, q) * super(R_s X^{e~})
        with  s = (H e~ + e_i + e_{i-1}) mod 2.

        Returns
        -------
        dict[tuple[int,...], dict[tuple[int,...], np.ndarray[complex]]]
            SQ[e_i][e_{i-1}], each value shape (dim^2, dim^2)
        """
        code = self.code
        SQ: dict[tuple[int, ...], dict[tuple[int, ...], np.ndarray]] = {ei: {} for ei in code.S}
        for ei in code.S:
            for eim1 in code.S:
                acc = np.zeros((code.dim ** 2, code.dim ** 2), dtype=complex)
                for et in itertools.product([0, 1], repeat=code.n):
                    s   = tuple((np.array(code.syndrome(et)) + np.array(ei) + np.array(eim1)) % 2)
                    V   = code.R_op(s) @ code.x_string(et)
                    acc += self.p_error(et, self.q) * to_super(V)
                SQ[ei][eim1] = acc
        return SQ

    def _build_logical_channels(self) -> dict[int, np.ndarray]:
        """
        Build physical-space superoperator snapshots Lambda_t, t = 0..T.

        The syndrome-error labels form a process (self.syndrome) that conditions
        each new label on the previous ``memory`` labels; the per-step transfer
        superoperator SQ[e_t][e_{t-1}] is weighted by prob(e_t | history).  A
        single loop handles any memory: the running dict C is keyed by the last
        ``L = max(1, memory)`` labels (most-recent first) -- one label is always
        needed because SQ couples e_t to e_{t-1}, the rest carry the process
        history.  Labels before t = 1 are the zero label (e_0 = e_{-1} = ... = 0).

        memory = 0 reproduces the i.i.d. (Markovian) channel; memory >= 1 gives a
        temporally correlated (non-Markovian) one.  No trailing R(eT) decode is
        applied at any snapshot.

        Returns
        -------
        dict[int, np.ndarray[complex]] -- {t: physical superoperator, shape (dim^2, dim^2)}
        """
        code = self.code
        ISUP = np.eye(code.dim ** 2, dtype=complex)
        zero = tuple(0 for _ in range(code.m))
        S    = code.S
        SQ   = self._build_SQ()

        syn    = self.syndrome
        memory = syn.memory
        L      = max(1, memory)                 # labels carried in the running state
        pad    = (zero,) * (L - 1)              # older labels, all zero at t = 1

        chans: dict[int, np.ndarray] = {0: ISUP.copy()}

        # t = 1: previous labels e_0 = ... = e_{1-memory} = zero.
        hist0 = (zero,) * memory
        C: dict[tuple, np.ndarray] = {}
        for e1 in S:
            w = syn.prob(e1, hist0)
            if w == 0.0:
                continue
            C[(e1,) + pad] = w * SQ[e1][zero]   # state = (e1, 0, ..., 0)
        chans[1] = sum(C.values()) if C else np.zeros_like(ISUP)

        for t in range(2, self.T + 1):
            C_next: dict[tuple, np.ndarray] = {}
            for state, mat in C.items():
                e_prev  = state[0]              # e_{t-1}, the previous SQ label
                history = state[:memory]        # last `memory` labels, most-recent first
                for e_t in S:
                    w = syn.prob(e_t, history)
                    if w == 0.0:
                        continue
                    new_state = (e_t,) + state[:-1]        # drop oldest label
                    contrib   = w * (SQ[e_t][e_prev] @ mat)
                    if new_state in C_next:
                        C_next[new_state] += contrib
                    else:
                        C_next[new_state] = contrib
            C = C_next
            chans[t] = sum(C.values()) if C else np.zeros_like(ISUP)
        return chans

    # ------------------------------------------------------------------
    # Classical (diagonal-only) fast path
    #
    # Every error/recovery operator is a bit-flip permutation, so on the
    # computational basis it just maps population at index x to (x XOR shift).
    # The whole superoperator machinery collapses to real 2^n x 2^n stochastic
    # matrices -- exact for classical codes, cheap enough to reach large n.
    # ------------------------------------------------------------------

    def _build_SQ_classical(self) -> dict[tuple[int, ...], np.ndarray]:
        """
        Classical one-step transfer matrices, keyed by the syndrome-label
        *difference* d = e_i XOR e_{i-1}.

        The dense SQ[e_i][e_{i-1}] depends on the pair only through d (because
        s = syndrome(e~) XOR e_i XOR e_{i-1} = syndrome(e~) XOR d), so there are
        only 2^m distinct matrices, each

            SQ_cl[d][y, x] = sum_{e~ : x XOR m(e~) XOR m(decoder[s]) == y} p_error(e~, q),
            s = syndrome(e~) XOR d.

        Returns
        -------
        dict[tuple[int,...], np.ndarray[float]] -- {d: (dim, dim) column-stochastic}
        """
        code      = self.code
        dim       = code.dim
        frombits  = self._frombits
        xall      = np.arange(dim)

        # per physical error e~: (shift m(e~), syndrome, probability) -- skip zero-prob
        errs = []
        for e in itertools.product([0, 1], repeat=code.n):
            p_e = self.p_error(e, self.q)
            if p_e == 0.0:
                continue
            errs.append((frombits(e), code.syndrome(e), p_e))

        SQ_cl: dict[tuple[int, ...], np.ndarray] = {}
        for d in code.S:
            M = np.zeros((dim, dim))
            for me, syn_e, p_e in errs:
                s   = tuple(a ^ b for a, b in zip(syn_e, d))
                sh  = me ^ frombits(code.decoder[s])
                # population at column x moves to row (x XOR sh)
                np.add.at(M, (xall ^ sh, xall), p_e)
            SQ_cl[d] = M
        return SQ_cl

    def _build_classical_channels(self) -> dict[int, np.ndarray]:
        """
        Classical analogue of _build_logical_channels: real (dim, dim) stochastic
        population-transfer snapshots Lambda_t on the physical computational basis.
        Identical memory-keyed loop, with SQ_cl[e_t XOR e_prev] replacing the dense
        SQ[e_t][e_prev] and a real identity as the t=0 snapshot.
        """
        code  = self.code
        dim   = code.dim
        I     = np.eye(dim)
        zero  = tuple(0 for _ in range(code.m))
        S     = code.S
        SQ_cl = self._build_SQ_classical()

        syn    = self.syndrome
        memory = syn.memory
        L      = max(1, memory)
        pad    = (zero,) * (L - 1)

        def xor(a: tuple, b: tuple) -> tuple:
            return tuple(x ^ y for x, y in zip(a, b))

        chans: dict[int, np.ndarray] = {0: I.copy()}

        hist0 = (zero,) * memory
        C: dict[tuple, np.ndarray] = {}
        for e1 in S:
            w = syn.prob(e1, hist0)
            if w == 0.0:
                continue
            C[(e1,) + pad] = w * SQ_cl[xor(e1, zero)]      # xor(e1, 0) = e1
        chans[1] = sum(C.values()) if C else np.zeros((dim, dim))

        for t in range(2, self.T + 1):
            C_next: dict[tuple, np.ndarray] = {}
            for state, mat in C.items():
                e_prev  = state[0]
                history = state[:memory]
                for e_t in S:
                    w = syn.prob(e_t, history)
                    if w == 0.0:
                        continue
                    new_state = (e_t,) + state[:-1]
                    contrib   = w * (SQ_cl[xor(e_t, e_prev)] @ mat)
                    if new_state in C_next:
                        C_next[new_state] += contrib
                    else:
                        C_next[new_state] = contrib
            C = C_next
            chans[t] = sum(C.values()) if C else np.zeros((dim, dim))
        return chans

    def _logical_stochastic(self, C: np.ndarray) -> np.ndarray:
        """
        Reduce a physical population-transfer matrix C (dim, dim) to the logical
        column-stochastic matrix A (2^k, 2^k):
            A[i, j] = sum_{x : l(x) == i}  C[x, codespace[j]].
        Columns are aggregated by logical label; the codespace columns pick the
        zero-syndrome logical inputs.
        """
        dim_L = 2 ** self.code.k
        Cj    = C[:, self._codespace]              # (dim, 2^k): logical inputs
        A     = np.zeros((dim_L, dim_L))
        np.add.at(A, self._log_label, Cj)          # sum rows grouped by logical label
        return A

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def snapshot(self, t: int) -> np.ndarray:
        """
        Raw physical snapshot Lambda_t.  Superoperator (dim^2, dim^2) when
        classical=False, else the classical population-transfer matrix (dim, dim).
        """
        return self.cchans[t] if self.classical else self.chans[t]

    def __call__(self, t: int, rho_L: np.ndarray) -> np.ndarray:
        """
        Apply the logical channel at time t.

        Parameters
        ----------
        t     : int -- time step in [0, T]
        rho_L : np.ndarray[complex], shape (2^k, 2^k) -- logical input density matrix

        Returns
        -------
        np.ndarray[complex], shape (2^k, 2^k) -- logical output density matrix

        Notes
        -----
        In the classical (default) mode only *diagonal* logical inputs are
        supported -- the dynamics preserve diagonals, so the output is
        diag(A_t @ diag(rho_L)).  Pass ``classical=False`` at construction to
        apply the channel to coherent (non-diagonal) states.
        """
        if self.classical:
            diag = np.diag(rho_L)
            if not np.allclose(rho_L, np.diag(diag), atol=1e-12):
                raise ValueError(
                    "classical=True LogicalChannel supports only diagonal logical "
                    "inputs; construct with classical=False for coherent states.")
            out = self.stochastic_matrix(t) @ np.real(diag)
            return np.diag(out).astype(complex)

        code       = self.code
        rho_phys   = self.U.T.conj() @ np.kron(rho_L, self._zero_syn) @ self.U
        sigma_phys = unvec(self.chans[t] @ vec(rho_phys), code.dim)
        sigma_rot  = self.U @ sigma_phys @ self.U.T.conj()
        return partial_trace_syndrome(sigma_rot, code)

    # ------------------------------------------------------------------
    # Convenience measure methods (delegate to measures.py)
    # ------------------------------------------------------------------

    def stochastic_matrix(self, t: int) -> np.ndarray:
        """
        Classical column-stochastic matrix of the logical channel at time t.

        Since the logical channel preserves diagonal (classical) states, it is
        equivalent to a stochastic matrix with entries
            A[i, j] = <i| Phi_t(|j><j|) |i> = p(i | j).
        Each column is the diagonal of Phi_t applied to the basis state |j><j|,
        so columns sum to 1 and entries are non-negative by construction.

        Parameters
        ----------
        t : int -- time step in [0, T]

        Returns
        -------
        np.ndarray[float], shape (2^k, 2^k) -- column-stochastic matrix A_t
        """
        if self.classical:                       # cheap: aggregate the classical snapshot
            return self._logical_stochastic(self.cchans[t])

        dim_L = 2 ** self.code.k                  # superoperator route (classical=False)
        A = np.zeros((dim_L, dim_L))
        for j in range(dim_L):
            rho_j = np.zeros((dim_L, dim_L), dtype=complex)
            rho_j[j, j] = 1.0
            A[:, j] = np.real(np.diag(self(t, rho_j)))
        return A

    def is_divisible(self, T: int | None = None,
                     tol: float = 1e-9) -> tuple[bool, dict[int, bool]]:
        """
        Check stochastic (P-)divisibility of the logical channel over all steps.

        For each step the intermediate propagator M(t) = A_t @ inv(A_{t-1}) is
        formed from the stochastic snapshots and tested with
        measures.is_stochastic.  A_0 = Phi_0 = I, so the check runs t = 1..T.
        A singular A_{t-1} (non-invertible step) is reported as not divisible.

        Parameters
        ----------
        T   : int | None -- last step to check (defaults to self.T)
        tol : float      -- tolerance passed to is_stochastic

        Returns
        -------
        tuple[bool, dict[int, bool]]
            (divisible_at_all_steps, {t: M(t) is stochastic})
        """
        from measures import is_stochastic
        T = self.T if T is None else T
        per_step: dict[int, bool] = {}
        A_prev = self.stochastic_matrix(0)   # identity at t=0
        for t in range(1, T + 1):
            A_t = self.stochastic_matrix(t)
            try:
                M = A_t @ np.linalg.inv(A_prev)
            except np.linalg.LinAlgError:
                per_step[t] = False           # singular step: not divisible
            else:
                per_step[t] = is_stochastic(M, tol)
            A_prev = A_t
        return all(per_step.values()), per_step

    def blp_measure(self, rho1: np.ndarray, rho2: np.ndarray, T: int | None = None) -> float:
        """BLP non-Markovian measure for the input pair (rho1, rho2).  See measures.blp_measure."""
        from measures import blp_measure
        return blp_measure(self, self.T if T is None else T, rho1, rho2)

    def max_blp_measure(self, T: int | None = None) -> tuple[float, tuple[int, int]]:
        """Max BLP measure over logical basis-state pairs.  See measures.max_blp_measure."""
        from measures import max_blp_measure
        return max_blp_measure(self.code, self, self.T if T is None else T)

    def __repr__(self) -> str:
        return (f"LogicalChannel(code={self.code!r}, T={self.T}, q={self.q}, "
                f"p_error={getattr(self.p_error, '__name__', self.p_error)})")
