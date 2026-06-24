"""
Error models for the QEC non-Markovianity study.

An ErrorModel is any callable  f(e, p) -> float  where:
  e : Sequence[int]  -- binary pattern (length n for physical, length m for syndrome)
  p : float          -- noise parameter in [0, 1]
  return : float     -- probability of that pattern

A SyndromeDistFn is a callable  f(p) -> dict[tuple[int,...], float]  returning
a full distribution over code.S for a given noise parameter p.

A SyndromeKernelFn is a callable  K(e_t, e_{t-1}) -> float  giving the transition
probability of syndrome-error label e_t given the previous label e_{t-1}.  Unlike
a SyndromeDistFn (a per-step marginal applied independently), a kernel carries
temporal correlation, which is what makes the resulting logical channel
non-Markovian.
"""

from typing import Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:                       # avoid a runtime import cycle
    from ClassicalCode import ClassicalCode

ErrorModel        = Callable[[Sequence[int], float], float]
SyndromeDistFn    = Callable[[float], dict[tuple[int, ...], float]]
SyndromeKernelFn  = Callable[[tuple[int, ...], tuple[int, ...]], float]
SyndromeKernel2Fn = Callable[[tuple[int, ...], tuple[int, ...], tuple[int, ...]], float]




# ------------------------------------------------------------------
# Physical-error models  (applied to the n-bit error pattern e~)
# ------------------------------------------------------------------

def iid_bitflip_error(e: Sequence[int], p: float) -> float:
    """
    i.i.d. bit-flip: each bit independently flipped with probability p.

    Parameters
    ----------
    e : Sequence[int] -- binary error pattern of any length
    p : float         -- bit-flip probability per site

    Returns
    -------
    float -- p^w * (1-p)^(len(e)-w),  where w = Hamming weight of e
    """
    w = int(sum(e))
    n = len(e)
    return p ** w * (1 - p) ** (n - w)


def perfect_physical_error(e: Sequence[int], p: float) -> float:
    """
    Noiseless physical channel: probability 1 on the all-zeros pattern, 0 else.

    Independent of p, so it can be slotted in wherever an ErrorModel is expected.

    Parameters
    ----------
    e : Sequence[int] -- binary error pattern
    p : float         -- ignored

    Returns
    -------
    float -- 1.0 if e is all zeros, else 0.0
    """
    return 1.0 if int(sum(e)) == 0 else 0.0


def single_error_model(e: Sequence[int], p: float) -> float:
    """
    At-most-one-error model: prob (1-p) for no error, p/n for each single-bit
    error, 0 for weight >= 2.  Models a channel where at most one bit is faulty.

    Parameters
    ----------
    e : Sequence[int] -- binary error pattern
    p : float         -- total single-error probability in [0, 1]

    Returns
    -------
    float
    """
    w = int(sum(e))
    n = len(e)
    if w == 0:
        return 1.0 - p
    if w == 1:
        return p / n
    return 0.0


def burst_error_model(e: Sequence[int], p: float) -> float:
    """
    Burst-error model: errors form a single contiguous block.
    P(no error) = 1 - p; each distinct contiguous burst is equally probable.

    Parameters
    ----------
    e : Sequence[int] -- binary error pattern
    p : float         -- probability of any burst occurring

    Returns
    -------
    float  (0 for non-contiguous error patterns)
    """
    bits  = list(e)
    n     = len(bits)
    ones  = [i for i, b in enumerate(bits) if b]
    if not ones:
        return 1.0 - p
    L = len(ones)
    if ones != list(range(ones[0], ones[0] + L)):   # not contiguous
        return 0.0
    n_bursts = n * (n + 1) // 2                      # total distinct contiguous blocks
    return p / n_bursts


# ------------------------------------------------------------------
# Syndrome-label distribution factories  (applied to the m-bit syndrome label)
# ------------------------------------------------------------------

def make_iid_syndrome_dist(code: 'ClassicalCode') -> SyndromeDistFn:
    """
    Factory: i.i.d. bit-flip syndrome distribution bound to *code*.
    Each of the m syndrome bits is independently flipped with probability p.

    Parameters
    ----------
    code : ClassicalCode

    Returns
    -------
    SyndromeDistFn -- callable  f(p: float) -> dict[tuple[int,...], float]
    """
    def dist(p: float) -> dict[tuple[int, ...], float]:
        return {e: iid_bitflip_error(e, p) for e in code.S}
    dist.__name__ = 'iid_bitflip_syndrome'
    return dist


def make_perfect_syndrome_dist(code: 'ClassicalCode') -> SyndromeDistFn:
    """
    Factory: noiseless syndrome distribution bound to *code*.
    All weight is on the zero syndrome -- syndrome is always measured correctly.

    Parameters
    ----------
    code : ClassicalCode

    Returns
    -------
    SyndromeDistFn -- callable  f(p: float) -> dict[tuple[int,...], float]
    """
    zero = tuple(0 for _ in range(code.m))
    def dist(p: float) -> dict[tuple[int, ...], float]:
        return {e: (1.0 if e == zero else 0.0) for e in code.S}
    dist.__name__ = 'perfect_syndrome'
    return dist


def make_single_flip_syndrome_dist(code: 'ClassicalCode') -> SyndromeDistFn:
    """
    Factory: at-most-one-flip syndrome distribution bound to *code*.
    Prob (1-p) for no flip, p/m per single-bit syndrome error, 0 for weight >= 2.

    Parameters
    ----------
    code : ClassicalCode

    Returns
    -------
    SyndromeDistFn -- callable  f(p: float) -> dict[tuple[int,...], float]
    """
    def dist(p: float) -> dict[tuple[int, ...], float]:
        result: dict[tuple[int, ...], float] = {}
        for e in code.S:
            w = int(sum(e))
            if w == 0:
                result[e] = 1.0 - p
            elif w == 1:
                result[e] = p / code.m
            else:
                result[e] = 0.0
        return result
    dist.__name__ = 'single_flip_syndrome'
    return dist


# ------------------------------------------------------------------
# Syndrome-label transition kernels  (temporally correlated => non-Markovian)
# ------------------------------------------------------------------

def make_non_markovian_syndrome_kernel(code: 'ClassicalCode',
                                       q: float, p: float) -> SyndromeKernelFn:
    """
    Factory: non-Markovian (temporally correlated) syndrome-error kernel.

    At each step the syndrome error e_t repeats the previous one e_{t-1} with
    probability q; otherwise (prob 1 - q) it is freshly drawn from the i.i.d.
    bit-flip distribution with parameter p:

        K(e_t | e_{t-1}) = q * [e_t == e_{t-1}] + (1 - q) * iid_bitflip(e_t, p)

    For each fixed e_{t-1} this is a valid distribution over code.S (the
    iid_bitflip masses sum to 1 and the delta term contributes q), so columns
    are normalised.  The repeat term q correlates consecutive labels in time,
    which is what renders the induced logical channel non-Markovian.

    Parameters
    ----------
    code : ClassicalCode
    q    : float -- probability of repeating the previous syndrome error
    p    : float -- bit-flip parameter of the fresh-draw distribution

    Returns
    -------
    SyndromeKernelFn -- callable  K(e_t, e_{t-1}) -> float
    """
    iid = {e: iid_bitflip_error(e, p) for e in code.S}

    def kernel(e_t: tuple[int, ...], e_tm1: tuple[int, ...]) -> float:
        same = 1.0 if e_t == e_tm1 else 0.0
        return q * same + (1.0 - q) * iid[e_t]
    kernel.__name__ = 'non_markovian_syndrome'
    return kernel


def make_exp_error_model_1(code: 'ClassicalCode',
                           q: float, p: float) -> SyndromeKernel2Fn:
    """
    Factory: experimental two-step-history syndrome kernel (arbitrary, for probing
    non-Markovian behaviour).

    For t >= 2 the transition depends on the two previous syndrome errors.  If
    their mod-2 sum (XOR) is the all-ones string, the next error is forced toward
    all-ones; otherwise it is i.i.d. bit-flip(p):

        if (e_{t-1} XOR e_{t-2}) == 11...1:
            K(e_t | e_{t-1}, e_{t-2}) = q * [e_t == 11...1] + (1 - q) * iid_bitflip(e_t, p)
        else:
            K(e_t | e_{t-1}, e_{t-2}) = iid_bitflip(e_t, p)

    Each branch is normalised over e_t (the iid masses sum to 1 and the delta term
    contributes q).  With q = 1 the all-ones history pins e_t to all-ones.

    Parameters
    ----------
    code : ClassicalCode
    q    : float -- probability of forcing all-ones when the history condition fires
    p    : float -- bit-flip parameter of the i.i.d. fallback distribution

    Returns
    -------
    SyndromeKernel2Fn -- callable  K(e_t, e_{t-1}, e_{t-2}) -> float
    """
    allones = tuple(1 for _ in range(code.m))
    iid     = {e: iid_bitflip_error(e, p) for e in code.S}

    def kernel2(e_t: tuple[int, ...],
                e_tm1: tuple[int, ...],
                e_tm2: tuple[int, ...]) -> float:
        xor = tuple(a ^ b for a, b in zip(e_tm1, e_tm2))
        if xor == allones:
            same = 1.0 if e_t == allones else 0.0
            return q * same + (1.0 - q) * iid[e_t]
        return iid[e_t]
    kernel2.__name__ = 'exp_error_model_1'
    return kernel2
