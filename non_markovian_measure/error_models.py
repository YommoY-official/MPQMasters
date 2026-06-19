"""
Error models for the QEC non-Markovianity study.

An ErrorModel is any callable  f(e, p) -> float  where:
  e : Sequence[int]  -- binary pattern (length n for physical, length m for syndrome)
  p : float          -- noise parameter in [0, 1]
  return : float     -- probability of that pattern

A SyndromeDistFn is a callable  f(p) -> dict[tuple[int,...], float]  returning
a full distribution over code.S for a given noise parameter p.
"""

from typing import Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:                       # avoid a runtime import cycle
    from ClassicalCode import ClassicalCode

ErrorModel     = Callable[[Sequence[int], float], float]
SyndromeDistFn = Callable[[float], dict[tuple[int, ...], float]]


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
