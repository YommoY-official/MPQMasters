"""
Non-Markovianity (BLP) measures for logical channels.

These are channel-agnostic: they operate on any callable
  channel(t, rho) -> rho_out
together with logical density matrices.  trace_norm(diag(p1-p2)) = ||p1-p2||_1
= 2 * total-variation distance in the classical (diagonal) case.
"""

from typing import Callable
import numpy as np

from ClassicalCode import ClassicalCode

Channel = Callable[[int, np.ndarray], np.ndarray]


def trace_norm(A: np.ndarray) -> float:
    """Trace norm ||A||_1 = sum of singular values (= sum |eigvals| for Hermitian A)."""
    return float(np.sum(np.abs(np.linalg.eigvalsh(A))))


def blp_measure(channel: Channel, T: int, rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    BLP non-Markovian measure: sum of positive increments of the trace distance
    D(t) = 1/2 ||Phi_t(rho1 - rho2)||_1 over t = 1..T-1.

    Parameters
    ----------
    channel : Callable[[int, np.ndarray], np.ndarray]
    T       : int
    rho1,
    rho2    : np.ndarray[complex] -- logical input density matrices

    Returns
    -------
    float
    """
    rho = rho1 - rho2
    measure = 0.0
    for t in range(1, T):
        measure += max(0.0, 0.5 * (trace_norm(channel(t, rho)) - trace_norm(channel(t - 1, rho))))
    return measure


def logical_basis_state(code: ClassicalCode, l: int) -> np.ndarray:
    """Pure logical-basis density matrix |l><l|, shape (2^k, 2^k)."""
    dim_L = 2 ** code.k
    rho = np.zeros((dim_L, dim_L), dtype=complex)
    rho[l, l] = 1.0
    return rho


def max_measure(code: ClassicalCode, channel: Channel, T: int) -> tuple[float, tuple[int, int]]:
    """
    Maximum BLP measure over all pairs of logical basis states, using a prebuilt channel.

    Parameters
    ----------
    code    : ClassicalCode
    channel : Callable[[int, np.ndarray], np.ndarray]
    T       : int

    Returns
    -------
    tuple[float, tuple[int, int]] -- (max_measure_value, (i, j))
    """
    dim_L = 2 ** code.k
    best_val, best_pair = -1.0, (0, 0)
    for i in range(dim_L):
        for j in range(i + 1, dim_L):
            v = blp_measure(channel, T,
                            logical_basis_state(code, i),
                            logical_basis_state(code, j))
            if v > best_val:
                best_val, best_pair = v, (i, j)
    return best_val, best_pair
