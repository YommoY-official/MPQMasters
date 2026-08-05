"""
Time-dependent (per-bit, per-time-step) independent-noise logical channels.

The stock LogicalChannel is *time-homogeneous*: it folds the physical error model
into a single per-step transfer operator and reuses it at every step, and neither
`p_error(e, q)` nor `SyndromeProcess.prob(e_t, history)` ever sees the time index.

This module lifts that restriction for the classical (diagonal) fast path.  At
each step t the physical- and syndrome-error distributions may differ, so a fresh
one-step population-transfer matrix is built per step.  Both noise sources are
MEMORYLESS but time-inhomogeneous:

    phys_dist(t) -> {d (len-n tuple): prob}   physical (data-qubit) error dist at step t
    syn_dist(t)  -> {e (len-m tuple): prob}   syndrome-readout error dist at step t

The dynamics match LogicalChannel exactly (same recovery / labelling), so with a
constant, equal-across-bits rate the output reproduces
    LogicalChannel(code, T, q, ...).stochastic_matrix(t).

The three physically interesting schedules -- physical only, syndrome only, and
both combined -- are thin wrappers over the same engine (`physical_only`,
`syndrome_only`, `combined`).  `ScheduleChannel` wraps the resulting A_t list in
the read API (`stochastic_matrix`, `is_divisible`) that the non-Markovianity
analysis (is_divisible, L1/Frobenius measures, A_heatmaps) already consumes.
"""

import itertools

import numpy as np

from ClassicalCode import ClassicalCode


def _logical_reducer(code: ClassicalCode):
    """Mirror LogicalChannel._prepare_logical_labels.

    Returns (frombits, reduce_to_logical) where reduce_to_logical maps a physical
    (dim, dim) population-transfer matrix to the logical (2^k, 2^k) column-stochastic
    matrix  A[i, j] = sum_{x : l(x) == i} C[x, codespace[j]].
    """
    n, k = code.n, code.k
    zero_syn = tuple(0 for _ in range(code.m))
    codespace = sorted(y for y in range(code.dim)
                       if code.syndrome(code.to_bits(y)) == zero_syn)
    log_idx = {y: l for l, y in enumerate(codespace)}

    def frombits(bits) -> int:                       # inverse of code.to_bits (MSB first)
        v = 0
        for i, b in enumerate(bits):
            v |= int(b) << (n - 1 - i)
        return v

    log_label = np.empty(code.dim, dtype=int)
    for x in range(code.dim):
        s = code.syndrome(code.to_bits(x))
        log_label[x] = log_idx[x ^ frombits(code.decoder[s])]   # apply recovery R(s)
    codespace = np.array(codespace, dtype=int)

    def reduce_to_logical(C: np.ndarray) -> np.ndarray:
        A = np.zeros((2 ** k, 2 ** k))
        np.add.at(A, log_label, C[:, codespace])
        return A

    return frombits, reduce_to_logical


def time_dependent_snapshots(code: ClassicalCode, Tmax: int,
                             phys_dist=None, syn_dist=None) -> list[np.ndarray]:
    """Logical stochastic snapshots A_0..A_Tmax for time-dependent, memoryless noise.

    Parameters
    ----------
    code      : ClassicalCode
    Tmax      : int
    phys_dist : callable(t) -> {d: prob} | None
                Physical error distribution over length-n patterns at step t (1..Tmax).
                None => perfect physical channel (all weight on the zero pattern).
    syn_dist  : callable(t) -> {e: prob} | None
                Syndrome-label distribution over length-m patterns at step t.
                None => perfect syndrome readout (all weight on the zero label).

    Returns
    -------
    list[np.ndarray] -- [A_0, ..., A_Tmax], each (2^k, 2^k) column-stochastic.
    """
    frombits, reduce_to_logical = _logical_reducer(code)
    dim, xall = code.dim, np.arange(code.dim)
    zero_syn = tuple(0 for _ in range(code.m))
    perfect_phys = {tuple(0 for _ in range(code.n)): 1.0}
    phys_dist = phys_dist or (lambda t: perfect_phys)
    syn_dist  = syn_dist  or (lambda t: {zero_syn: 1.0})

    def step_matrices(t):
        """One-step classical transfer M[delta] for every syndrome-label difference.

        M[delta][x⊕sh, x] += p(d),  with full syndrome  s = H d ⊕ delta,
        recovery shift  sh = m(d) ⊕ m(decoder[s]),  delta = e_t ⊕ e_{t-1}.
        """
        items = [(frombits(d), code.syndrome(d), p)
                 for d, p in phys_dist(t).items() if p > 0]
        Ms = {}
        for delta in code.S:
            M = np.zeros((dim, dim))
            for me, syn_e, p_e in items:
                s  = tuple(a ^ b for a, b in zip(syn_e, delta))
                sh = me ^ frombits(code.decoder[s])
                np.add.at(M, (xall ^ sh, xall), p_e)
            Ms[delta] = M
        return Ms

    A_list = [reduce_to_logical(np.eye(dim))]            # t = 0 identity
    C = {zero_syn: np.eye(dim)}                          # running transfer, keyed by e_{t-1}
    for t in range(1, Tmax + 1):
        Ms, C_next = step_matrices(t), {}
        for e_prev, mat in C.items():
            for e_t, w in syn_dist(t).items():
                if w == 0.0:
                    continue
                delta = tuple(a ^ b for a, b in zip(e_t, e_prev))
                C_next[e_t] = C_next.get(e_t, 0) + w * (Ms[delta] @ mat)
        C = C_next
        A_list.append(reduce_to_logical(sum(C.values())))
    return A_list


def bernoulli_dist(pbits) -> dict:
    """Product-Bernoulli distribution over bit patterns.

    pbits[b] is the flip probability of bit b; returns {pattern: prob} over all
    2^len(pbits) patterns, prob = prod_b pbits[b]^bit_b (1 - pbits[b])^(1-bit_b).
    """
    dist = {}
    for pat in itertools.product([0, 1], repeat=len(pbits)):
        pr = 1.0
        for b, bit in enumerate(pat):
            pr *= pbits[b] if bit else (1 - pbits[b])
        dist[pat] = pr
    return dist


def physical_only(code: ClassicalCode, Tmax: int, p_phys: np.ndarray) -> list[np.ndarray]:
    """Physical errors only (perfect syndrome).  p_phys shape (n, Tmax):
    p_phys[b, t-1] = flip probability of data bit b at step t."""
    p_phys = np.asarray(p_phys, float)
    return time_dependent_snapshots(
        code, Tmax, phys_dist=lambda t: bernoulli_dist(p_phys[:, t - 1]))


def syndrome_only(code: ClassicalCode, Tmax: int, p_syn: np.ndarray) -> list[np.ndarray]:
    """Syndrome errors only (perfect physical).  p_syn shape (m, Tmax):
    p_syn[b, t-1] = flip probability of syndrome bit b at step t."""
    p_syn = np.asarray(p_syn, float)
    return time_dependent_snapshots(
        code, Tmax, syn_dist=lambda t: bernoulli_dist(p_syn[:, t - 1]))


def combined(code: ClassicalCode, Tmax: int,
             p_phys: np.ndarray, p_syn: np.ndarray) -> list[np.ndarray]:
    """Both physical and syndrome errors, each with its own (n|m, Tmax) schedule."""
    p_phys = np.asarray(p_phys, float)
    p_syn  = np.asarray(p_syn, float)
    return time_dependent_snapshots(
        code, Tmax,
        phys_dist=lambda t: bernoulli_dist(p_phys[:, t - 1]),
        syn_dist =lambda t: bernoulli_dist(p_syn[:, t - 1]))


class ScheduleChannel:
    """Adapter exposing the LogicalChannel read API over a precomputed A_t list.

    Lets the existing non-Markovianity analysis (is_divisible, L1/Frobenius
    measures, A_heatmaps) consume a time-dependent channel unchanged.
    """

    def __init__(self, code: ClassicalCode, A_list: list[np.ndarray]) -> None:
        self.code = code
        self._A = A_list
        self.T = len(A_list) - 1

    def stochastic_matrix(self, t: int) -> np.ndarray:
        return self._A[t]

    def is_divisible(self, T: int | None = None,
                     tol: float = 1e-9) -> tuple[bool, dict[int, bool]]:
        """Stochastic (P-)divisibility over all steps; same logic as LogicalChannel."""
        from measures import is_stochastic
        T = self.T if T is None else T
        per_step: dict[int, bool] = {}
        A_prev = self._A[0]
        for t in range(1, T + 1):
            A_t = self._A[t]
            try:
                M = A_t @ np.linalg.inv(A_prev)
            except np.linalg.LinAlgError:
                per_step[t] = False
            else:
                per_step[t] = is_stochastic(M, tol)
            A_prev = A_t
        return all(per_step.values()), per_step

    def __repr__(self) -> str:
        return f"ScheduleChannel(code={self.code!r}, T={self.T})"


# ----------------------------------------------------------------------
# Fast Monte-Carlo engine
#
# time_dependent_snapshots rebuilds the code's decode/shift structure on every
# call, which is wasteful when sweeping thousands of random schedules.  The
# structure (recovery shifts, logical labels, ...) depends only on `code`, so we
# precompute it once and reuse it across all samples.  The per-step transfer is a
# group convolution: M[r, x] = w[r XOR x] with w[shift] = sum_{d: shift(d)==shift}
# p(d), built with np.bincount and read off via a precomputed XOR table.  Only the
# two codespace columns are propagated (dim x 2), since the logical reduction only
# ever reads those.  Outputs are identical to the reference builders (validated).
# ----------------------------------------------------------------------

def _mc_context(code: ClassicalCode) -> dict:
    """Precompute all code-dependent structure needed to build A_t snapshots."""
    frombits, _ = _logical_reducer(code)
    n, m, dim = code.n, code.m, code.dim
    S = [tuple(s) for s in code.S]

    phys_patterns = list(itertools.product([0, 1], repeat=n))
    phys_bits = np.array(phys_patterns, dtype=float)                 # (dim, n)
    me_arr    = np.array([frombits(p) for p in phys_patterns])       # (dim,)
    syn_of    = [tuple(code.syndrome(p)) for p in phys_patterns]
    dec_shift = {s: frombits(code.decoder[s]) for s in S}
    syn_bits  = np.array(S, dtype=float)                             # (2^m, m)
    zero_idx  = S.index(tuple(0 for _ in range(m)))

    rows = np.arange(dim)
    xor_table = np.bitwise_xor.outer(rows, rows)                     # (dim, dim)

    # per syndrome-label difference delta: recovery shift for each physical pattern
    sh_by_delta = {}
    for delta in S:
        sh = np.empty(dim, dtype=int)
        for i in range(dim):
            s = tuple(a ^ b for a, b in zip(syn_of[i], delta))
            sh[i] = me_arr[i] ^ dec_shift[s]
        sh_by_delta[delta] = sh

    zero_syn = tuple(0 for _ in range(m))
    codespace = np.array(sorted(y for y in range(dim)
                                if code.syndrome(code.to_bits(y)) == zero_syn))
    log_idx = {y: l for l, y in enumerate(codespace)}
    log_label = np.empty(dim, dtype=int)
    for x in range(dim):
        s = code.syndrome(code.to_bits(x))
        log_label[x] = log_idx[x ^ frombits(code.decoder[s])]

    # precomputed delta between every pair of labels (i=e_t, j=e_prev)
    delta_of = {(i, j): tuple(a ^ b for a, b in zip(S[i], S[j]))
                for i in range(len(S)) for j in range(len(S))}

    return dict(n=n, m=m, dim=dim, k=code.k, S=S, zero_idx=zero_idx,
                phys_bits=phys_bits, syn_bits=syn_bits, xor_table=xor_table,
                sh_by_delta=sh_by_delta, codespace=codespace, log_label=log_label,
                delta_of=delta_of)


def _mc_snapshots(ctx: dict, Tmax: int, p_phys, p_syn) -> list[np.ndarray]:
    """A_0..A_Tmax using the precomputed context.  p_phys/p_syn shaped (n|m, Tmax)
    or None (perfect).  Matches time_dependent_snapshots exactly."""
    dim, S = ctx['dim'], ctx['S']
    xor_table, sh_by_delta = ctx['xor_table'], ctx['sh_by_delta']
    phys_bits, syn_bits = ctx['phys_bits'], ctx['syn_bits']
    codespace, log_label = ctx['codespace'], ctx['log_label']
    zero_idx, delta_of = ctx['zero_idx'], ctx['delta_of']
    nlab = len(S)

    perfect_pe = np.zeros(dim); perfect_pe[0] = 1.0                 # all-zeros physical pattern
    perfect_ws = np.zeros(nlab); perfect_ws[zero_idx] = 1.0

    V0 = np.zeros((dim, 2)); V0[codespace, [0, 1]] = 1.0
    A_list = [np.eye(2)]                                            # A_0 = reduce(V0) = I_2
    C = {zero_idx: V0}
    for t in range(1, Tmax + 1):
        pe = perfect_pe if p_phys is None else \
            np.where(phys_bits == 1, p_phys[:, t - 1], 1 - p_phys[:, t - 1]).prod(1)
        ws = perfect_ws if p_syn is None else \
            np.where(syn_bits == 1, p_syn[:, t - 1], 1 - p_syn[:, t - 1]).prod(1)
        support = np.nonzero(ws > 0)[0]

        Mcache: dict = {}
        def getM(delta):
            M = Mcache.get(delta)
            if M is None:
                w = np.bincount(sh_by_delta[delta], weights=pe, minlength=dim)
                M = w[xor_table]
                Mcache[delta] = M
            return M

        C_next: dict = {}
        for e_prev, V in C.items():
            for e_t in support:
                contrib = ws[e_t] * (getM(delta_of[(e_t, e_prev)]) @ V)
                if e_t in C_next:
                    C_next[e_t] += contrib
                else:
                    C_next[e_t] = contrib.copy()
        C = C_next
        sumV = sum(C.values())
        A = np.zeros((2, 2)); np.add.at(A, log_label, sumV)
        A_list.append(A)
    return A_list


def _det2(A: np.ndarray) -> float:
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def first_nonmarkov_step(A_list: list[np.ndarray], tol: float = 1e-9,
                         det_thresh: float = 1e-6, absorb_window: int = 3):
    """First step at which the channel is GENUINELY non-Markovian, or None.

    The maximally mixed logical state [[.5,.5],[.5,.5]] (det = 0) is the channel's
    absorbing fixed point: |det(A_t)| = prod|1 - 2 a_i| shrinks monotonically to 0,
    after which every A_s is 0.5 everywhere and the propagator M(s) = A_s inv(A_{s-1})
    is a 0/0 artifact.  So:

      * Absorption: once |det(A_t)| and the next ``absorb_window`` determinants are
        all below ``det_thresh``, conclude the fixed point is reached and STOP.
      * Singular predecessor: a step whose A_{t-1} is below ``det_thresh`` has an
        ill-conditioned inverse -> skip it (divisibility undefined there).

    Returns
    -------
    int | None -- the earliest well-conditioned step whose propagator is genuinely
    non-stochastic (before absorption), or None if the channel stays divisible.
    """
    from measures import is_stochastic
    T = len(A_list) - 1
    dets = [_det2(A) for A in A_list]
    for t in range(1, T + 1):
        # reached the maximally mixed fixed point -> stop tracking
        window = range(t, min(t + absorb_window + 1, T + 1))
        if all(abs(dets[s]) < det_thresh for s in window):
            break
        if abs(dets[t - 1]) < det_thresh:        # ill-conditioned predecessor -> skip
            continue
        try:
            M = A_list[t] @ np.linalg.inv(A_list[t - 1])
        except np.linalg.LinAlgError:
            continue
        if not is_stochastic(M, tol):
            return t                             # first genuine non-Markovian step
    return None


def _is_nonmarkov(A_list: list[np.ndarray], tol: float = 1e-9,
                  det_thresh: float = 1e-6, absorb_window: int = 3) -> bool:
    """True iff the channel is genuinely non-Markovian (see first_nonmarkov_step)."""
    return first_nonmarkov_step(A_list, tol, det_thresh, absorb_window) is not None


def l1_measure_vs_time(A_list: list[np.ndarray], det_thresh: float = 1e-6,
                       absorb_window: int = 3):
    """Per-step L1 non-Markovianity measure D(M(t)) with absorption-aware masking.

    D(M(t)) = l1_stochastic_distance(A_t inv(A_{t-1})) penalises negativity and
    column-sum deviation of the one-step propagator; it is 0 at divisible
    (Markovian) steps and > 0 where information flows back.  Steps with an
    ill-conditioned predecessor, and every step at/after maximally mixed
    absorption, are returned as NaN (the measure is undefined there).

    Returns
    -------
    (ts, measure, A00, absorb_t):
        ts       : np.ndarray[int]   -- 1..T
        measure  : np.ndarray[float] -- D(M(t)), NaN where undefined
        A00      : np.ndarray[float] -- logical survival A_t[0,0] for t = 1..T
        absorb_t : int | None        -- first step judged maximally mixed, else None
    """
    from measures import l1_stochastic_distance
    T = len(A_list) - 1
    dets = [_det2(A) for A in A_list]
    ts = np.arange(1, T + 1)
    measure = np.full(T, np.nan)
    A00 = np.array([A_list[t][0, 0] for t in range(1, T + 1)])
    absorb_t = None
    for i, t in enumerate(ts):
        window = range(t, min(t + absorb_window + 1, T + 1))
        if absorb_t is None and all(abs(dets[s]) < det_thresh for s in window):
            absorb_t = t
        if absorb_t is not None:                 # absorbed -> undefined thereafter
            continue
        if abs(dets[t - 1]) < det_thresh:        # ill-conditioned predecessor -> undefined
            continue
        try:
            M = A_list[t] @ np.linalg.inv(A_list[t - 1])
        except np.linalg.LinAlgError:
            continue
        measure[i] = l1_stochastic_distance(M)
    return ts, measure, A00, absorb_t


def sample_genuine_schedule(code: ClassicalCode, Tmax: int, case: str, rng=None,
                            tol: float = 1e-9, det_thresh: float = 1e-6,
                            absorb_window: int = 3, max_tries: int = 100000) -> dict | None:
    """Draw random schedules until one is genuinely non-Markovian; record and return it.

    Returns a dict {'p_phys', 'p_syn', 'first_step', 'A'} (p_phys/p_syn is None for
    the unused channel), where 'A' is the recorded schedule's A_t snapshot list --
    ready to feed to l1_measure_vs_time.  Returns None if none found in max_tries.
    """
    if case not in ('phys', 'syn', 'comb'):
        raise ValueError("case must be 'phys', 'syn' or 'comb'")
    if rng is None:
        rng = np.random.default_rng()
    ctx = _mc_context(code)
    n, m = code.n, code.m
    for _ in range(max_tries):
        p_phys = rng.random((n, Tmax)) if case in ('phys', 'comb') else None
        p_syn  = rng.random((m, Tmax)) if case in ('syn', 'comb') else None
        A = _mc_snapshots(ctx, Tmax, p_phys, p_syn)
        s = first_nonmarkov_step(A, tol, det_thresh, absorb_window)
        if s is not None:
            return {'p_phys': p_phys, 'p_syn': p_syn, 'first_step': s, 'A': A}
    return None


def monte_carlo_nonmarkov(code: ClassicalCode, Tmax: int, n_samples: int,
                          case: str, rng=None, tol: float = 1e-9,
                          det_thresh: float = 1e-6, absorb_window: int = 3,
                          checkpoints=None) -> dict[int, float]:
    """Fraction of random independent-noise schedules that are non-Markovian.

    Parameters
    ----------
    code       : ClassicalCode
    Tmax       : int   -- circuit depth.
    n_samples  : int   -- number of random schedules to draw.
    case       : {'phys', 'syn', 'comb'} -- which bits get independent per-bit,
                 per-step flip probabilities (physical only / syndrome only / both).
    rng        : np.random.Generator | None
    tol        : float -- tolerance for the stochasticity test.
    checkpoints : iterable[int] | None -- if given, also report the running
                 fraction after this many samples (cumulative), so a single pass
                 yields several sample-count columns.  Defaults to [n_samples].

    Returns
    -------
    dict[int, float] -- {n: (# non-Markovian in first n samples) / n}.
    """
    if case not in ('phys', 'syn', 'comb'):
        raise ValueError("case must be 'phys', 'syn' or 'comb'")
    if rng is None:
        rng = np.random.default_rng()
    ctx = _mc_context(code)
    n, m = code.n, code.m
    checkpoints = sorted(checkpoints) if checkpoints else [n_samples]
    counts = {c: 0 for c in checkpoints}

    nm = 0
    for i in range(1, n_samples + 1):
        p_phys = rng.random((n, Tmax)) if case in ('phys', 'comb') else None
        p_syn  = rng.random((m, Tmax)) if case in ('syn', 'comb') else None
        if _is_nonmarkov(_mc_snapshots(ctx, Tmax, p_phys, p_syn),
                         tol, det_thresh, absorb_window):
            nm += 1
        if i in counts:
            counts[i] = nm
    return {c: counts[c] / c for c in checkpoints}


def random_invertible_gf2(m: int, rng=None) -> np.ndarray:
    """Random invertible m x m matrix over GF(2) (rejection sampling)."""
    from random_codes import _gf2_rank
    if rng is None:
        rng = np.random.default_rng()
    while True:
        P = rng.integers(0, 2, size=(m, m))
        if _gf2_rank(P) == m:
            return P


def relabel_syndrome(code: ClassicalCode, P: np.ndarray) -> ClassicalCode:
    """Same code, different syndrome map: H' = P H (mod 2), P invertible m x m.

    Left-multiplying the parity-check matrix by an invertible m x m matrix
    preserves the code (null space / codewords, hence distance and k) but relabels
    the syndromes and therefore the minimum-weight decoder.  (An n x n *right*
    multiply would instead change the code.)
    """
    return ClassicalCode((np.asarray(P) @ code.H) % 2)


def monte_carlo_syndrome_relabel(code: ClassicalCode, Tmax: int, n_instances: int,
                                 rng=None, tol: float = 1e-9, det_thresh: float = 1e-6,
                                 absorb_window: int = 3) -> np.ndarray:
    """Monte Carlo over syndrome RELABELINGS of a fixed code (syndrome errors only).

    Each instance draws a random invertible m x m P (-> H' = P H: same code, new
    syndrome map) plus a random per-bit, per-step syndrome-error schedule, then
    tests genuine non-Markovianity.  (For m parity checks there are only
    |GL(m, F_2)| distinct relabelings, e.g. 6 for m = 2.)

    Returns
    -------
    np.ndarray[int] -- first genuine non-Markovian step for each non-Markovian
    instance (Markovian instances omitted); its length is the non-Markovian count.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = code.m
    steps: list[int] = []
    for _ in range(n_instances):
        code2 = relabel_syndrome(code, random_invertible_gf2(m, rng))
        ctx = _mc_context(code2)
        p_syn = rng.random((m, Tmax))
        s = first_nonmarkov_step(_mc_snapshots(ctx, Tmax, None, p_syn),
                                 tol, det_thresh, absorb_window)
        if s is not None:
            steps.append(s)
    return np.array(steps, dtype=int)


def monte_carlo_relabel(code: ClassicalCode, Tmax: int, p_syn: np.ndarray,
                        n_samples: int, rng=None, tol: float = 1e-9,
                        det_thresh: float = 1e-6, absorb_window: int = 3):
    """Vary the syndrome RELABELING with a FIXED syndrome-error schedule.

    The syndrome-error probabilities p_syn (shape (m, Tmax)) are held fixed; only
    the syndrome map changes: each sample draws a random invertible m x m P
    (-> H' = P H, the SAME code with relabeled syndromes / decoder).  This isolates
    the effect of the syndrome labeling on non-Markovianity.

    Returns
    -------
    tavg        : np.ndarray[n_samples] -- time-averaged L1 measure per relabeling,
                  sum_t D(M(t)) / Tmax (undefined / absorbed steps contribute 0).
    first_steps : np.ndarray[int]       -- first genuine non-Markov step for the
                  relabelings that are non-Markovian (length = non-Markov count).
    """
    if rng is None:
        rng = np.random.default_rng()
    m = code.m
    tavg = np.empty(n_samples)
    first_steps: list[int] = []
    for i in range(n_samples):
        code2 = relabel_syndrome(code, random_invertible_gf2(m, rng))
        A = _mc_snapshots(_mc_context(code2), Tmax, None, p_syn)
        _, measure, _, _ = l1_measure_vs_time(A, det_thresh, absorb_window)
        tavg[i] = np.nansum(measure) / Tmax
        s = first_nonmarkov_step(A, tol, det_thresh, absorb_window)
        if s is not None:
            first_steps.append(s)
    return tavg, np.array(first_steps, dtype=int)


def monte_carlo_first_steps(code: ClassicalCode, Tmax: int, n_samples: int,
                            case: str, rng=None, tol: float = 1e-9,
                            det_thresh: float = 1e-6,
                            absorb_window: int = 3) -> np.ndarray:
    """First genuine non-Markovian step for every non-Markovian random schedule.

    Same sampling as monte_carlo_nonmarkov, but records *when* (which step)
    genuine non-Markovianity first appears.  Markovian samples contribute nothing.

    Returns
    -------
    np.ndarray[int] -- one entry per non-Markovian sample: its first genuine
    non-Markovian step (length = # non-Markovian samples <= n_samples).
    """
    if case not in ('phys', 'syn', 'comb'):
        raise ValueError("case must be 'phys', 'syn' or 'comb'")
    if rng is None:
        rng = np.random.default_rng()
    ctx = _mc_context(code)
    n, m = code.n, code.m
    steps: list[int] = []
    for _ in range(n_samples):
        p_phys = rng.random((n, Tmax)) if case in ('phys', 'comb') else None
        p_syn  = rng.random((m, Tmax)) if case in ('syn', 'comb') else None
        s = first_nonmarkov_step(_mc_snapshots(ctx, Tmax, p_phys, p_syn),
                                 tol, det_thresh, absorb_window)
        if s is not None:
            steps.append(s)
    return np.array(steps, dtype=int)
