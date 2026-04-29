"""
Top-level experiment drivers.

Experiment A: Single-Rec transition kernel.
    For each s_in in {0, ..., 63}, sample n_shots of run_single_rec(s_in, p),
    estimate
        T[s_in, L, s_out]   -- joint joint distribution of (L, s_out)
                                given the injected s_in.
    From T, derive:
        P(L | s_in)          (averaged over s_out)
        P(s_out | s_in)      (averaged over L)        -- the "syndrome
                                                          transition matrix"
        lambda_2             -- second-largest eigenvalue of P(s_out|s_in)
                                (in absolute value).

Experiment B: Two-Rec Markov property.
    Fix s_in = 0. Sample n_shots of run_two_recs.
    Compute four mutual / conditional mutual informations with bootstrap CIs:

        (i)   I(L_2 ; L_1 | s_in)
              -- predicted > 0  (marginal logical process is non-Markov)
        (ii)  I(L_2 ; L_1 | s_1, s_in)
              -- predicted ≈ 0  (augmenting with the boundary syndrome
                                 makes the augmented process Markov)
        (iii) I(s_2 ; s_1 | s_in)
              -- predicted > 0  (syndromes carry correlated memory)
        (iv)  I(L_2 ; s_in | s_1)
              -- predicted ≈ 0  (data-processing on the augmented chain --
                                 conditioned on s_1, L_2 is independent of s_in)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .simulation import run_single_rec, run_two_recs
from .noise_model import ECNoise
from .diagnostics import (
    mutual_info, conditional_mutual_info, bootstrap_ci, entropy_mm
)


# ---------------------------------------------------------------------------
# Experiment A
# ---------------------------------------------------------------------------

@dataclass
class ExperimentAResult:
    p: float
    n_shots_per_s_in: int
    T: np.ndarray                # (64, 4, 64) joint counts of (L, s_out) per s_in
    P_L_given_sin: np.ndarray    # (64, 4)
    P_sout_given_sin: np.ndarray # (64, 64)  rows = s_in
    lambda_2: float              # |second largest eigenvalue| of P_sout_given_sin


def run_experiment_a(p: float, n_shots: int = 1_000_000,
                     seed: int = 1234,
                     ec_noise: 'ECNoise' = None) -> ExperimentAResult:
    """
    Experiment A driver. Returns transition tensor + spectral diagnostic.
    """
    n_syn = 64
    T = np.zeros((n_syn, 4, n_syn), dtype=np.int64)
    for s_in in range(n_syn):
        out = run_single_rec(s_in, p, n_shots, ec_noise=ec_noise,
                             seed=seed + s_in)
        # accumulate joint counts of (L, s_out) for this s_in
        L = out['L']
        s_out = out['s_out']
        # use np.add.at for vectorized scatter
        flat = (L.astype(np.int64) * n_syn + s_out.astype(np.int64))
        bincount = np.bincount(flat, minlength=4 * n_syn).reshape(4, n_syn)
        T[s_in] = bincount

    # Marginalise to P(L | s_in)
    P_L_given_sin = T.sum(axis=2).astype(float)
    P_L_given_sin /= P_L_given_sin.sum(axis=1, keepdims=True).clip(min=1)

    # Marginalise to P(s_out | s_in)
    P_sout_given_sin = T.sum(axis=1).astype(float)
    P_sout_given_sin /= P_sout_given_sin.sum(axis=1, keepdims=True).clip(min=1)

    # Spectral diagnostic
    eigs = np.linalg.eigvals(P_sout_given_sin)
    abs_eigs = np.sort(np.abs(eigs))[::-1]
    # Largest is 1 (or near 1) for any stochastic matrix; report 2nd.
    lambda_2 = float(abs_eigs[1]) if len(abs_eigs) > 1 else 0.0

    return ExperimentAResult(
        p=p, n_shots_per_s_in=n_shots,
        T=T,
        P_L_given_sin=P_L_given_sin,
        P_sout_given_sin=P_sout_given_sin,
        lambda_2=lambda_2,
    )


# ---------------------------------------------------------------------------
# Experiment B
# ---------------------------------------------------------------------------

@dataclass
class ExperimentBResult:
    p: float
    n_shots: int
    rows: List[Dict]   # one dict per row of the table


def _verdict(point: float, lo: float, hi: float, tol: float = 5e-3,
             expect: str = '') -> str:
    """
    Simple verdict logic for the predicted-zero or predicted-positive rows.
    `tol` is a slack for "approximately zero" judged in bits.
    """
    if expect == '> 0':
        if lo > tol:
            return 'PASS'
        if hi <= tol:
            return 'FAIL'
        return 'INCONCLUSIVE'
    if expect == '~ 0':
        if hi <= tol:
            return 'PASS'
        if lo > tol:
            return 'FAIL'
        return 'INCONCLUSIVE'
    return ''


def run_experiment_b(p: float, n_shots: int = 1_000_000,
                     seed: int = 5678,
                     n_bootstrap: int = 500,
                     ec_noise: 'ECNoise' = None) -> ExperimentBResult:
    """
    Experiment B driver: correlation and sufficient-statistic tests on two
    consecutive Recs (Wait — EC — Wait — EC) with s_in = 0 fixed.

    In our (Wait — EC) Rec structure, with perfect EC, the residual data
    Pauli at the start of Rec 2 is exactly the logical Pauli L_1 (modulo
    stabilizers, which don't affect future evolution). Hence L_1 alone
    is the "state" of the data going into Rec 2, and the correct
    sufficient-statistic test is

         I(L_2 ; s_1 | L_1)  ≈ 0     (s_1 adds no info beyond L_1)

    which says L_t is the sufficient statistic, NOT (L_t, s_t).

    By contrast, s_1 alone is NOT a sufficient statistic for L_2: knowing
    s_1 tells us the noise's syndrome but not which logical-error coset
    the residual is in. So

         I(L_2 ; L_1 | s_1)  > 0     (L_1 carries info beyond s_1)

    These complementary asymmetric tests pin down where the memory lives.
    """
    out = run_two_recs(s0=0, p=p, n_shots=n_shots, ec_noise=ec_noise, seed=seed)
    L1 = out['L1']; L2 = out['L2']
    s1 = out['s_1']; s2 = out['s_2']

    rows: List[Dict] = []

    # (i) marginal logical-error correlation
    pt, lo, hi = bootstrap_ci(
        lambda X, Y: mutual_info(X, Y),
        samples={'X': L2, 'Y': L1},
        n_resamples=n_bootstrap, seed=seed + 1)
    rows.append(dict(
        name='I(L_2 ; L_1)',
        expected='> 0',
        point=pt, lo=lo, hi=hi,
        verdict=_verdict(pt, lo, hi, expect='> 0'),
        note='consecutive logical errors are correlated',
    ))

    # (ii) marginal syndrome correlation
    pt, lo, hi = bootstrap_ci(
        lambda X, Y: mutual_info(X, Y),
        samples={'X': s2, 'Y': s1},
        n_resamples=n_bootstrap, seed=seed + 2)
    rows.append(dict(
        name='I(s_2 ; s_1)',
        expected='> 0',
        point=pt, lo=lo, hi=hi,
        verdict=_verdict(pt, lo, hi, expect='> 0'),
        note='syndromes carry correlated memory',
    ))

    # (iii) THE Markov-property test in our (Wait,EC) structure:
    #       given L_1, does s_1 add ANY info about L_2?
    pt, lo, hi = bootstrap_ci(
        lambda X, Y, Z: conditional_mutual_info(X, Y, Z),
        samples={'X': L2, 'Y': s1, 'Z': L1},
        n_resamples=n_bootstrap, seed=seed + 3)
    rows.append(dict(
        name='I(L_2 ; s_1 | L_1)',
        expected='~ 0',
        point=pt, lo=lo, hi=hi,
        verdict=_verdict(pt, lo, hi, expect='~ 0'),
        note='L_1 alone is the sufficient statistic: s_1 adds nothing',
    ))

    # (iv) complementary asymmetric test: given s_1, is L_1 still informative?
    pt, lo, hi = bootstrap_ci(
        lambda X, Y, Z: conditional_mutual_info(X, Y, Z),
        samples={'X': L2, 'Y': L1, 'Z': s1},
        n_resamples=n_bootstrap, seed=seed + 4)
    rows.append(dict(
        name='I(L_2 ; L_1 | s_1)',
        expected='> 0',
        point=pt, lo=lo, hi=hi,
        verdict=_verdict(pt, lo, hi, expect='> 0'),
        note='s_1 alone is NOT sufficient -- L_1 partitions same-syndrome '
             'noise patterns by logical class',
    ))

    return ExperimentBResult(p=p, n_shots=n_shots, rows=rows)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_experiment_b_table(res: ExperimentBResult) -> None:
    print(f"\nExperiment B  --  p={res.p}, n_shots={res.n_shots}")
    print(f"{'quantity':24s}{'expect':10s}{'point':>10s}"
          f"{'95% lo':>10s}{'95% hi':>10s}  {'verdict':14s}note")
    print('-' * 110)
    for row in res.rows:
        print(f"{row['name']:24s}{row['expected']:10s}"
              f"{row['point']:10.5f}{row['lo']:10.5f}{row['hi']:10.5f}"
              f"  {row['verdict']:14s}{row.get('note','')}")
