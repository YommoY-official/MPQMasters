"""
Information-theoretic diagnostics on integer-valued samples.

Conventions
-----------
All inputs are 1-D numpy arrays of identical length N. Each entry is a
non-negative integer encoding the value in that shot. Joint distributions
are formed by stacking columns and using `np.unique` along axis 0.

All entropies / mutual informations are reported in BITS, with the
Miller-Madow bias correction added:

    H_unbiased  ≈  H_plugin  +  (K - 1) / (2 N ln 2)

where K is the number of categories actually observed and N is the sample
size.

Bootstrap CIs use the percentile method on `n_resamples` resamples.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint counts and entropy
# ---------------------------------------------------------------------------

def _joint_counts(*arrays: np.ndarray) -> np.ndarray:
    """Counts per unique row of stacked column arrays."""
    cols = [np.asarray(a).reshape(-1) for a in arrays]
    n = cols[0].size
    for c in cols[1:]:
        if c.size != n:
            raise ValueError("all arrays must have the same length")
    stacked = np.stack(cols, axis=1)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    return counts


def entropy_mm(*arrays: np.ndarray) -> float:
    """Miller-Madow-corrected Shannon entropy in bits of the joint distribution."""
    counts = _joint_counts(*arrays)
    N = counts.sum()
    if N == 0:
        return 0.0
    p = counts.astype(float) / N
    p_pos = p[p > 0]
    H_plug = float(-np.sum(p_pos * np.log2(p_pos)))
    K = int(counts.size)
    return H_plug + (K - 1) / (2.0 * N * np.log(2.0))


# ---------------------------------------------------------------------------
# Mutual information and conditional MI
# ---------------------------------------------------------------------------

def mutual_info(X: np.ndarray, Y: np.ndarray) -> float:
    """I(X ; Y) in bits with Miller-Madow correction on each entropy term."""
    return entropy_mm(X) + entropy_mm(Y) - entropy_mm(X, Y)


def conditional_mutual_info(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """
    I(X ; Y | Z) in bits, computed via
        I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z).
    """
    return entropy_mm(X, Z) + entropy_mm(Y, Z) - entropy_mm(X, Y, Z) - entropy_mm(Z)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    estimator: Callable[..., float],
    samples: Dict[str, np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = None,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI for `estimator(**samples)`.

    samples : dict of identical-length 1-D arrays.
    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    keys = list(samples.keys())
    arrs = [np.asarray(samples[k]).reshape(-1) for k in keys]
    n = arrs[0].size
    point = float(estimator(**{k: a for k, a in zip(keys, arrs)}))
    boots = np.zeros(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled = {k: a[idx] for k, a in zip(keys, arrs)}
        boots[b] = float(estimator(**resampled))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> dict:
    rng = np.random.default_rng(0)
    checks: dict = {}

    # T1: independent uniform variables -- I should be ~0.
    n = 50_000
    X = rng.integers(0, 4, size=n)
    Y = rng.integers(0, 4, size=n)
    I_indep = mutual_info(X, Y)
    assert abs(I_indep) < 0.01, f"I(X;Y) for indep X,Y should be ~0, got {I_indep}"
    checks['T1_independent_MI_near_zero'] = round(I_indep, 5)

    # T2: perfectly correlated Y = X -- I should equal H(X).
    Y2 = X.copy()
    I_corr = mutual_info(X, Y2)
    H_X = entropy_mm(X)
    assert abs(I_corr - H_X) < 0.005, f"I(X;X) should = H(X) = {H_X}, got {I_corr}"
    checks['T2_perfect_corr_MI_eq_H_X'] = (round(H_X, 5), round(I_corr, 5))

    # T3: chain X -> Y -> Z (Markov) -- I(X;Z|Y) should be ~0.
    # Build: X random, Y = noisy(X), Z = noisy(Y).
    Y3 = X ^ rng.integers(0, 2, size=n)
    Z3 = Y3 ^ rng.integers(0, 2, size=n)
    cmi = conditional_mutual_info(X, Z3, Y3)
    assert abs(cmi) < 0.02, f"I(X;Z|Y) for Markov chain should be ~0, got {cmi}"
    checks['T3_Markov_chain_CMI_near_zero'] = round(cmi, 5)

    # T4: bootstrap CI is finite and brackets the point estimate.
    point, lo, hi = bootstrap_ci(
        lambda X, Y: mutual_info(X, Y),
        samples={'X': X, 'Y': Y2},
        n_resamples=200, seed=1,
    )
    assert lo <= point <= hi
    checks['T4_bootstrap_CI_brackets_point'] = (round(lo, 4), round(point, 4), round(hi, 4))

    if verbose:
        print("diagnostics._selftest passed:")
        for k, v in checks.items():
            print(f"  {k:46s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
