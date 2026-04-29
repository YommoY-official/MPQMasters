"""
High-level Monte Carlo wrappers around `rec.simulate_recs`.

These thin functions return the structured output expected by experiments.py,
matching the spirit of the original spec:

    run_single_rec(s0, p, n_shots)  -> structured array with
        s_in, s_out, L
    run_two_recs(s0, p, n_shots)    -> structured array with
        s_in, s_1, s_2, L1, L2

`p` is interpreted as the symmetric mutex Wait noise: p_x = p_z = p,
correlated_xz = False. Use `simulate_recs` directly for richer noise
configurations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .rec import simulate_recs
from .noise_model import WaitNoise, ECNoise


_SINGLE_DTYPE = np.dtype([
    ('s_in', np.int32),
    ('s_out', np.int32),
    ('L', np.int8),
])

_TWO_DTYPE = np.dtype([
    ('s_in', np.int32),
    ('s_1', np.int32),     # syndrome at end of Rec 1
    ('s_2', np.int32),     # syndrome at end of Rec 2
    ('L1', np.int8),
    ('L2', np.int8),
])


def run_single_rec(s0: int, p: float, n_shots: int,
                   ec_noise: Optional[ECNoise] = None,
                   seed: Optional[int] = None) -> np.ndarray:
    """One Rec = Wait + EC. Returns structured array of length n_shots."""
    rng = np.random.default_rng(seed)
    if ec_noise is None:
        ec_noise = ECNoise()
    res = simulate_recs(
        n_recs=1, n_shots=n_shots, s_in=s0,
        wait_noise=WaitNoise(p_x=p, p_z=p, correlated_xz=False),
        ec_noise=ec_noise,
        rng=rng,
    )
    out = np.empty(n_shots, dtype=_SINGLE_DTYPE)
    out['s_in'] = s0
    out['s_out'] = res.syndromes[:, 0]
    out['L'] = res.logical_classes[:, 0]
    return out


def run_two_recs(s0: int, p: float, n_shots: int,
                 ec_noise: Optional[ECNoise] = None,
                 seed: Optional[int] = None) -> np.ndarray:
    """Two Recs = (Wait+EC) twice. Returns structured array of length n_shots."""
    rng = np.random.default_rng(seed)
    if ec_noise is None:
        ec_noise = ECNoise()
    res = simulate_recs(
        n_recs=2, n_shots=n_shots, s_in=s0,
        wait_noise=WaitNoise(p_x=p, p_z=p, correlated_xz=False),
        ec_noise=ec_noise,
        rng=rng,
    )
    out = np.empty(n_shots, dtype=_TWO_DTYPE)
    out['s_in'] = s0
    out['s_1'] = res.syndromes[:, 0]
    out['s_2'] = res.syndromes[:, 1]
    out['L1'] = res.logical_classes[:, 0]
    out['L2'] = res.logical_classes[:, 1]
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> dict:
    checks: dict = {}

    # p = 0 sanity: single Rec
    a = run_single_rec(s0=0, p=0.0, n_shots=100, seed=1)
    assert np.all(a['L'] == 0)
    assert np.all(a['s_out'] == 0)
    checks['p0_single_rec_clean'] = True

    # p = 0 with non-zero s_in
    a = run_single_rec(s0=5, p=0.0, n_shots=100, seed=1)
    assert np.all(a['s_in'] == 5)
    assert np.all(a['s_out'] == 5)   # first EC measures s_in then corrects
    assert np.all(a['L'] == 0)
    checks['p0_single_rec_s_in_match'] = True

    # p = 0 sanity: two Recs
    b = run_two_recs(s0=0, p=0.0, n_shots=100, seed=2)
    assert np.all(b['s_1'] == 0) and np.all(b['s_2'] == 0)
    assert np.all(b['L1'] == 0) and np.all(b['L2'] == 0)
    checks['p0_two_rec_clean'] = True

    # noise produces variation
    c = run_two_recs(s0=0, p=0.05, n_shots=10_000, seed=3)
    assert (c['L2'] != 0).sum() > 0
    checks['p_pos_two_rec_logical_errors_seen'] = int((c['L2'] != 0).sum())

    if verbose:
        print("simulation._selftest passed:")
        for k, v in checks.items():
            print(f"  {k:38s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
