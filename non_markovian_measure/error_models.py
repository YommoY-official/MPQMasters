"""
Error models for the QEC non-Markovianity study.

There are two, physically distinct, noise sources -- both described by the *same*
simple interface "return the probability of a bitstring":

1. Physical (data-qubit) errors.  An ``ErrorModel`` is any callable
       f(e, p) -> float
   giving the probability of the length-n error pattern ``e`` at noise rate ``p``.
   This corrupts the state; the true syndrome is ``H @ e``.

2. Syndrome-readout errors.  A ``SyndromeProcess`` gives the probability of the
   length-m syndrome-error label ``e_t`` at time ``t``, possibly conditioned on
   the previous ``memory`` labels:
       prob(e_t, history) -> float,
       history = (e_{t-1}, e_{t-2}, ..., e_{t-memory})   # most-recent first
   ``memory == 0`` is memoryless (i.i.d.) syndrome noise -> Markovian channel;
   ``memory >= 1`` carries temporal correlation -> non-Markovian channel.  All
   parameters are baked in at construction, so ``prob`` takes no noise argument.

The single ``SyndromeProcess`` abstraction replaces the former trio of
``syndrome_dist`` / ``syndrome_kernel`` / ``syndrome_kernel2`` slots.
"""

from typing import Callable, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:                       # avoid a runtime import cycle
    from ClassicalCode import ClassicalCode

ErrorModel  = Callable[[Sequence[int], float], float]
Label       = Tuple[int, ...]
# prob(e_t, history) -> float,  history = tuple of previous labels (most-recent first)
SyndromeProbFn = Callable[[Label, Tuple[Label, ...]], float]




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


def perfect_physical_error(e: Sequence[int], p : float) -> float:
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
# Syndrome-readout error process  (applied to the m-bit syndrome label)
#
# One abstraction for every syndrome-noise model, memoryless or correlated.
# Build one with a factory below and pass it to LogicalChannel(syndrome=...).
# ------------------------------------------------------------------

class SyndromeProcess:
    """
    Probability model for the syndrome-readout error label at each time step.

    A syndrome-error label ``e`` (a length-``code.m`` tuple in {0, 1}^m) is drawn
    each step from a distribution that may condition on the previous ``memory``
    labels:

        prob(e_t, history) -> float,
        history = (e_{t-1}, e_{t-2}, ..., e_{t-memory})   # most-recent first;
                                                          # zero label padded for t <= memory

    ``memory == 0`` is memoryless i.i.d. syndrome noise (Markovian logical
    channel).  ``memory >= 1`` conditions on the recent past and generally makes
    the logical channel non-Markovian.  All parameters (rates, etc.) are baked in
    at construction, so ``prob`` needs no noise argument.

    Attributes
    ----------
    code   : ClassicalCode
    memory : int              -- number of previous labels ``prob`` conditions on
    """

    def __init__(self, code: 'ClassicalCode', prob_fn: SyndromeProbFn,
                 memory: int = 0, name: str = 'syndrome_process') -> None:
        self.code:    'ClassicalCode' = code
        self.memory:  int             = int(memory)
        self._prob:   SyndromeProbFn  = prob_fn
        self.__name__:  str           = name

    def prob(self, e_t: Label, history: Tuple[Label, ...]) -> float:
        """Probability of label ``e_t`` given ``history`` (the last ``memory`` labels)."""
        return self._prob(e_t, history)

    def __repr__(self) -> str:
        return f"SyndromeProcess(name={self.__name__!r}, memory={self.memory})"


# -- Memoryless syndrome processes (memory = 0) --------------------------------

def iid_syndrome(code: 'ClassicalCode', p: float) -> SyndromeProcess:
    """i.i.d. bit-flip syndrome noise: each of the m syndrome bits flips with prob p."""
    table = {e: iid_bitflip_error(e, p) for e in code.S}
    return SyndromeProcess(code, lambda e_t, history: table[e_t],
                           memory=0, name='iid_syndrome')


def perfect_syndrome(code: 'ClassicalCode') -> SyndromeProcess:
    """Noiseless syndrome readout: all weight on the zero label (always correct)."""
    zero = tuple(0 for _ in range(code.m))
    return SyndromeProcess(code, lambda e_t, history: 1.0 if e_t == zero else 0.0,
                           memory=0, name='perfect_syndrome')


def single_flip_syndrome(code: 'ClassicalCode', p: float) -> SyndromeProcess:
    """At-most-one syndrome flip: prob (1-p) no flip, p/m per single flip, 0 for weight >= 2."""
    m = code.m
    table = {e: ((1.0 - p) if sum(e) == 0 else (p / m if sum(e) == 1 else 0.0))
             for e in code.S}
    return SyndromeProcess(code, lambda e_t, history: table[e_t],
                           memory=0, name='single_flip_syndrome')


# -- Temporally correlated syndrome processes (memory >= 1) --------------------

def sticky_syndrome(code: 'ClassicalCode', q: float, p: float) -> SyndromeProcess:
    """
    Sticky (memory-1) syndrome noise: e_t repeats the previous label e_{t-1} with
    probability q, else is freshly drawn i.i.d. bit-flip(p):

        prob(e_t | e_{t-1}) = q * [e_t == e_{t-1}] + (1 - q) * iid_bitflip(e_t, p)

    The repeat term correlates consecutive labels, making the logical channel
    non-Markovian.  Each column (fixed e_{t-1}) is a normalised distribution.
    Was ``make_non_markovian_syndrome_kernel``.
    """
    table = {e: iid_bitflip_error(e, p) for e in code.S}

    def prob(e_t: Label, history: Tuple[Label, ...]) -> float:
        e_tm1 = history[0]
        same  = 1.0 if e_t == e_tm1 else 0.0
        return q * same + (1.0 - q) * table[e_t]

    return SyndromeProcess(code, prob, memory=1, name='sticky_syndrome')


def exp_syndrome_1(code: 'ClassicalCode', q: float, p: float) -> SyndromeProcess:
    """
    Experimental two-step-history (memory-2) syndrome process, for probing
    non-Markovian behaviour.  If the XOR of the two previous labels is all-ones,
    the next label is forced toward all-ones; otherwise it is i.i.d. bit-flip(p):

        if (e_{t-1} XOR e_{t-2}) == 11...1:
            prob(e_t | ...) = q * [e_t == 11...1] + (1 - q) * iid_bitflip(e_t, p)
        else:
            prob(e_t | ...) = iid_bitflip(e_t, p)

    With q = 1 the all-ones history pins e_t to all-ones.  Was
    ``make_exp_error_model_1``.
    """
    allones = tuple(1 for _ in range(code.m))
    table   = {e: iid_bitflip_error(e, p) for e in code.S}

    def prob(e_t: Label, history: Tuple[Label, ...]) -> float:
        e_tm1, e_tm2 = history[0], history[1]
        xor = tuple(a ^ b for a, b in zip(e_tm1, e_tm2))
        if xor == allones:
            same = 1.0 if e_t == allones else 0.0
            return q * same + (1.0 - q) * table[e_t]
        return table[e_t]

    return SyndromeProcess(code, prob, memory=2, name='exp_syndrome_1')
