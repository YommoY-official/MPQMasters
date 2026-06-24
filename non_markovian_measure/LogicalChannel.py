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
    SyndromeDistFn,
    SyndromeKernelFn,
    SyndromeKernel2Fn,
    iid_bitflip_error,
    make_iid_syndrome_dist,
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
    q             : float                 -- noise parameter
    p_error         : ErrorModel              -- physical error model; default iid_bitflip_error
    syndrome_dist   : SyndromeDistFn | None   -- syndrome label distribution;
                                                 defaults to make_iid_syndrome_dist(code)
    syndrome_kernel : SyndromeKernelFn | None -- syndrome-label transition kernel
                                                 K(e_t, e_{t-1}); when given it overrides
                                                 syndrome_dist and builds a temporally
                                                 correlated (non-Markovian) channel
    syndrome_kernel2: SyndromeKernel2Fn | None -- two-step-history syndrome kernel
                                                 K(e_t, e_{t-1}, e_{t-2}); when given it
                                                 takes precedence over syndrome_kernel and
                                                 syndrome_dist (e_1 seeded i.i.d. from the
                                                 fallback once two-step history is unavailable)

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
        syndrome_dist: SyndromeDistFn | None = None,
        syndrome_kernel: SyndromeKernelFn | None = None,
        syndrome_kernel2: SyndromeKernel2Fn | None = None,
    ) -> None:
        if syndrome_dist is None:
            syndrome_dist = make_iid_syndrome_dist(code)
        self.code:             ClassicalCode              = code
        self.T:                int                        = T
        self.q:                float                      = q
        self.p_error:          ErrorModel                 = p_error
        self.syndrome_dist:    SyndromeDistFn             = syndrome_dist
        self.syndrome_kernel:  SyndromeKernelFn | None    = syndrome_kernel
        self.syndrome_kernel2: SyndromeKernel2Fn | None   = syndrome_kernel2

        self.U:     np.ndarray            = code.build_logical_unitary()
        self.chans: dict[int, np.ndarray] = self._build_logical_channels()

        # cached pieces for the logical-space wrapper
        dim_S          = 2 ** code.m
        e_in_idx       = code.S.index(tuple(0 for _ in range(code.m)))
        self._zero_syn = np.zeros((dim_S, dim_S))
        self._zero_syn[e_in_idx, e_in_idx] = 1.0

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

        First step uses Q(e1, e_in) with e_in = (0,...,0); the syndrome then
        evolves freely.  No trailing R(eT) decode is applied at any snapshot.

        If a syndrome_kernel K(e_t, e_{t-1}) is supplied, the syndrome-error
        labels form a Markov chain (initialised from e_in = 0) and the per-step
        marginal weight psyn[e_t] is replaced by the transition probability
        K(e_t, e_{t-1}); this is what produces a non-Markovian logical channel.
        The i.i.d. case is recovered by K(e_t, e_{t-1}) = psyn[e_t].

        Returns
        -------
        dict[int, np.ndarray[complex]] -- {t: physical superoperator, shape (dim^2, dim^2)}
        """
        code = self.code
        ISUP = np.eye(code.dim ** 2, dtype=complex)
        e_in = tuple(0 for _ in range(code.m))
        S    = code.S
        SQ   = self._build_SQ()

        chans: dict[int, np.ndarray] = {0: ISUP.copy()}
        if self.syndrome_kernel2 is not None:
            K2 = self.syndrome_kernel2
            # State carried as the pair (last label, previous label).  e_0 = 0 and a
            # virtual e_{-1} = 0 seed e_1 via the kernel's i.i.d. fallback branch
            # (0 XOR 0 != all-ones), so two-step history only acts from t = 2 on.
            C2 = {(e1, e_in): K2(e1, e_in, e_in) * SQ[e1][e_in] for e1 in S}
            chans[1] = sum(C2.values())
            for t in range(2, self.T + 1):
                C2 = {
                    (e_t, e_tm1): sum(
                        K2(e_t, e_tm1, e_tm2) * SQ[e_t][e_tm1] @ C2[(e_tm1, e_tm2)]
                        for e_tm2 in S if (e_tm1, e_tm2) in C2
                    )
                    for e_t in S for e_tm1 in S
                }
                chans[t] = sum(C2.values())
            return chans

        if self.syndrome_kernel is not None:
            K = self.syndrome_kernel
            # first step: e_1 ~ K(.|e_in=0), conditioning the chain on e_in
            C        = {e1: K(e1, e_in) * SQ[e1][e_in] for e1 in S}
            chans[1] = sum(C[e1] for e1 in S)                  # no trailing R(e1)
            for t in range(2, self.T + 1):
                C        = {ei: sum(K(ei, eim1) * SQ[ei][eim1] @ C[eim1] for eim1 in S)
                            for ei in S}
                chans[t] = sum(C[et] for et in S)              # no trailing R(eT)
            return chans

        psyn     = self.syndrome_dist(self.q)
        C        = {e1: psyn[e1] * SQ[e1][e_in] for e1 in S}   # first step: Q(e1, e_in=0)
        chans[1] = sum(C[e1] for e1 in S)                      # no trailing R(e1)
        for t in range(2, self.T + 1):
            C        = {ei: psyn[ei] * sum(SQ[ei][eim1] @ C[eim1] for eim1 in S) for ei in S}
            chans[t] = sum(C[et] for et in S)                  # no trailing R(eT)
        return chans

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def snapshot(self, t: int) -> np.ndarray:
        """Raw physical superoperator Lambda_t, shape (dim^2, dim^2)."""
        return self.chans[t]

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
        """
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
        dim_L = 2 ** self.code.k
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
