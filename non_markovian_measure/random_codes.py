"""
Random classical-code sampling and distance-distribution utilities.

Two full-rank binary parity-check-matrix samplers (uniform-brute vs. standard
form [I | A]) plus a Monte-Carlo tally of code distances.  All matrices follow
the ClassicalCode convention  H.shape == (m, n)  with  m = n - k.
"""

from typing import Callable

import numpy as np

from ClassicalCode import ClassicalCode


def _gf2_rank(M: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2)."""
    A = (np.array(M, dtype=int) % 2).copy()
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        pivot = None
        for r in range(rank, rows):
            if A[r, c]:
                pivot = r
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(rows):
            if r != rank and A[r, c]:
                A[r] = (A[r] + A[rank]) % 2
        rank += 1
    return rank


def sample_H_brute(n: int, k: int) -> np.ndarray:
    """Sample a full-rank binary parity-check matrix H of shape (n - k, n).

    Matches the ClassicalCode convention H.shape == (m, n) with m = n - k.
    Each entry is drawn uniformly from {0, 1}; matrices are re-sampled until
    the rows are linearly independent over GF(2) (rank == n - k).
    """
    m = n - k
    while True:
        H = np.random.randint(0, 2, size=(m, n))
        if _gf2_rank(H) == m:
            return H


def sample_H(n: int, k: int) -> np.ndarray:
    """Sample a full-rank binary parity-check matrix H of shape (n - k, n).

    Standard form [I_{n-k} | A] with a random (n - k, k) block A; the identity
    block guarantees rank n - k (full row rank). Matches ClassicalCode's
    H.shape == (m, n) convention with m = n - k.
    """
    m = n - k
    A = np.random.randint(0, 2, size=(m, k))
    return np.hstack([np.eye(m, dtype=int), A])


def distance_distribution(sampler: Callable[[], np.ndarray],
                          n_samples: int = 100) -> dict[int, int]:
    """Tally code distances over randomly sampled parity-check matrices.

    Parameters
    ----------
    sampler   : Callable[[], np.ndarray] -- zero-arg function returning a
                parity-check matrix H (shape (m, n)) on each call.
    n_samples : int -- number of matrices to draw (default 100).

    Returns
    -------
    dict[int, int] -- {code distance: number of occurrences}, sorted by distance.
    """
    counts: dict[int, int] = {}
    for _ in range(n_samples):
        d = ClassicalCode(sampler()).distance()
        counts[d] = counts.get(d, 0) + 1
    return dict(sorted(counts.items()))
