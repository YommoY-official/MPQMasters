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
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(A))))


def is_stochastic(A: np.ndarray, tol: float = 1e-9) -> bool:
    """True if all column of A sum up to 1 and all element is larger or equal to 0 ."""
    A = np.asarray(A)
    nonneg = bool(np.all(A >= -tol))
    col_sums_ok = bool(np.allclose(A.sum(axis=0), 1.0, atol=tol))
    return nonneg and col_sums_ok


def _project_rows_to_simplex(M: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of every row of M onto the probability simplex.

    For each row vector ``a`` (length n) this returns the unique point ``p`` with
    ``sum(p) == 1`` and ``p >= 0`` minimising ``||a - p||_2`` (Duchi et al., 2008).
    The whole matrix is handled at once (sort + cumsum), with no per-row loop.

    Algorithm per row:
      1. u    = a sorted in descending order.
      2. find rho = largest 1-indexed k with  u[k] + (1/k)(1 - sum(u[1..k])) > 0.
      3. theta = (1/rho)(1 - sum(u[1..rho]))           (the row's Lagrange offset)
      4. p[j]  = max(a[j] + theta, 0).

    Parameters
    ----------
    M : np.ndarray, shape (m, n) -- rows to project.

    Returns
    -------
    np.ndarray, shape (m, n) -- each row projected onto the simplex.
    """
    m, n = M.shape
    # 1. sort each row in descending order
    u = np.sort(M, axis=1)[:, ::-1]
    # cumulative sums shifted by the simplex total (1): cssv[:, j] = sum(u[1..j]) - 1
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, n + 1)                      # 1-indexed positions 1..n
    # 2. condition  u[j] - (sum(u[1..j]) - 1)/j > 0  (equivalent to the spec form)
    cond = u - cssv / ind > 0
    # largest 0-indexed position where the condition still holds (rho-1)
    rho = n - 1 - np.argmax(cond[:, ::-1], axis=1)
    # 3. theta as defined in the spec, per row:  theta = (1 - sum(u[1..rho])) / rho
    theta = cssv[np.arange(m), rho] / (rho + 1.0)   # note: this is -theta_spec
    # 4. p[j] = max(a[j] + theta_spec, 0) = max(a[j] - theta, 0)
    return np.maximum(M - theta[:, None], 0.0)


def stochastic_distance(A: np.ndarray, axis: int = 0, tol: float = 1e-12) -> float:
    """
    Frobenius distance from a square matrix A to the set of stochastic matrices.

    Measures how far A is from being stochastic (a proxy for "how non-Markovian"
    a transfer/propagator matrix is): ``d(A, S) = ||A - P||_F`` where P is the
    row/column-wise Euclidean projection of A onto the probability simplex,
    enforcing both the sum-to-1 and the non-negativity constraints.  The problem
    decouples across rows/columns, so P is built by projecting each one
    independently onto the simplex.

    By default ``axis=0`` (column-stochastic: every column sums to 1), matching
    this module's ``is_stochastic`` / ``LogicalChannel.stochastic_matrix``
    convention.  Use ``axis=1`` for row-stochastic (right-stochastic).

    An already-stochastic matrix projects to itself, so the distance is 0.

    Parameters
    ----------
    A    : np.ndarray, shape (n, n) -- square matrix.
    axis : int   -- 0 => columns sum to 1 (default), 1 => rows sum to 1.
    tol  : float -- distances below this are snapped to exactly 0.0.

    Returns
    -------
    float -- ``||A - P||_F`` (>= 0), exactly 0.0 if A is already stochastic.

    Raises
    ------
    ValueError -- if A is not a square 2-D matrix, is empty, or contains
                  NaN/Inf entries.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be a 2-D matrix, got ndim={A.ndim}")
    if A.size == 0:
        raise ValueError("A must be non-empty")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must not contain NaN or Inf values")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    # Project the constrained axis: rows for axis=1, columns (=> rows of A.T) for axis=0.
    if axis == 0:
        P = _project_rows_to_simplex(A.T).T
    else:
        P = _project_rows_to_simplex(A)

    dist = float(np.linalg.norm(A - P, ord="fro"))
    return 0.0 if dist < tol else dist

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


def max_blp_measure(code: ClassicalCode, channel: Channel, T: int) -> tuple[float, tuple[int, int]]:
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


# ----------------------------------------------------------------------
# Unit tests for stochastic_distance  (run with:  python measures.py)
# ----------------------------------------------------------------------

def _test_stochastic_distance() -> None:
    rng = np.random.default_rng(0)

    # Test 1: identity matrix is doubly stochastic -> distance 0.
    assert stochastic_distance(np.eye(4)) == 0.0

    # Test 2: a valid column-stochastic matrix -> distance 0.
    S = np.array([[0.2, 0.5, 0.1],
                  [0.3, 0.5, 0.6],
                  [0.5, 0.0, 0.3]])
    assert np.allclose(S.sum(axis=0), 1.0)        # sanity: columns sum to 1
    assert stochastic_distance(S, axis=0) == 0.0

    # Test 2b: a valid row-stochastic matrix with axis=1 -> distance 0.
    assert stochastic_distance(S.T, axis=1) == 0.0

    # Test 3: negatives and wrong column sums -> a finite distance > 0.
    A = np.array([[1.0, -0.5, 2.0],
                  [0.0,  0.5, 0.0],
                  [3.0,  1.0, 0.0]])
    d = stochastic_distance(A, axis=0)
    assert isinstance(d, float) and d > 0.0

    # Test 4: the projection P really is stochastic and is the closest point.
    P = _project_rows_to_simplex(A.T).T
    assert is_stochastic(P)
    assert np.isclose(d, np.linalg.norm(A - P, ord="fro"))
    # any other stochastic matrix is no closer than the projection
    for _ in range(20):
        cols = rng.random(A.shape)
        Q = cols / cols.sum(axis=0, keepdims=True)   # random column-stochastic
        assert np.linalg.norm(A - Q, ord="fro") >= d - 1e-9

    # Test 5: error handling.
    for bad, kind in [(np.ones((2, 3)), "non-square"),
                      (np.empty((0, 0)), "empty"),
                      (np.array([[np.nan, 0.0], [0.0, 1.0]]), "NaN"),
                      (np.array([[np.inf, 0.0], [0.0, 1.0]]), "Inf")]:
        try:
            stochastic_distance(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kind} input")

    print("stochastic_distance: all tests passed")


if __name__ == "__main__":
    _test_stochastic_distance()
