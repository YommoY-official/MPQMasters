"""
Plotting / experiment drivers for logical-channel non-Markovianity.

- markovianity_vs_distance : Monte-Carlo count of non-Markovian random codes,
                             grouped by code distance.
- A_heatmaps               : (time, param) heatmaps of det(A_t) and A00(t) for
                             ANY error model, via a channel-builder callback.
"""

import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from ClassicalCode import ClassicalCode
from LogicalChannel import LogicalChannel


def markovianity_vs_distance(sampler: Callable[[int, int], np.ndarray],
                             n: int, k: int, q: float,
                             n_samples: int = 100, T: int = 50,
                             plot: bool = True) -> dict[int, dict[str, int]]:
    """Sample random [n, k] codes, build their logical channels at noise q, and
    count how many are non-Markovian (stochastically non-divisible), by distance.

    NOTE: LogicalChannel builds 2^(2n) superoperators, so cost explodes with n
    (n=3 ~0.2s/sample, n=4 ~12s/sample). Keep n small or n_samples low.

    Parameters
    ----------
    sampler   : Callable[[int, int], np.ndarray] -- returns a parity-check matrix
                H of shape (n-k, n) given (n, k).
    n, k      : int   -- code parameters.
    q         : float -- bit-flip noise rate.
    n_samples : int   -- number of random codes to draw.
    T         : int   -- time steps used by LogicalChannel.is_divisible.
    plot      : bool  -- show the histogram.

    Returns
    -------
    dict[int, dict[str, int]] -- {distance: {'total': , 'non_markovian': }}.
    """
    stats: dict[int, dict[str, int]] = {}
    for _ in range(n_samples):
        code      = ClassicalCode(sampler(n, k))
        channel   = LogicalChannel(code, T, q)
        divisible = channel.is_divisible()[0]
        s = stats.setdefault(code.d, {'total': 0, 'non_markovian': 0})
        s['total'] += 1
        if not divisible:
            s['non_markovian'] += 1
    stats = dict(sorted(stats.items()))

    if plot:
        ds     = list(stats)
        totals = [stats[d]['total']          for d in ds]
        nonm   = [stats[d]['non_markovian']  for d in ds]
        x      = np.arange(len(ds))
        plt.figure(figsize=(7, 4))
        plt.bar(x, totals, color='lightsteelblue', label='total codes')
        plt.bar(x, nonm,   color='crimson',        label='non-Markovian (False)')
        plt.xticks(x, ds)
        plt.xlabel('code distance  d')
        plt.ylabel('number of codes')
        plt.title(f'Non-Markovianity vs distance  '
                  f'(n={n}, k={k}, q={q}, {n_samples} samples, T={T})')
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    return stats


def A_heatmaps(make_channel, param_values, Tmax,
               param_label='error rate  q', title='', save=None):
    """Two (time, param) heatmaps of the logical stochastic matrix A_t.

    Parameters
    ----------
    make_channel : callable(param, Tmax) -> LogicalChannel
        Builds the channel at one y-axis value.  This is where you plug in the
        error model, e.g.
            lambda q, T: LogicalChannel(code, T, q)                                  # iid both-noisy
            lambda q, T: LogicalChannel(code, T, q, p_error=perfect_physical_error,
                                        syndrome=sticky_syndrome(code, q, p))         # sticky
            lambda q, T: LogicalChannel(code, T, q, p_error=perfect_physical_error,
                                        syndrome=exp_syndrome_1(code, q, p))          # 2-step history
    param_values : 1D array-like -- y-axis sweep values.
    Tmax         : int           -- x-axis is t = 0..Tmax.
    param_label  : str           -- y-axis label (name of the swept parameter).
    title, save  : figure suptitle and optional PNG path.

    Returns
    -------
    (fig, det, A00) with det, A00 shaped (len(param_values), Tmax+1).

    Notes
    -----
    Left panel  = det(A_t): invertibility.  det=0 => singular (== the condition
                  `is_divisible` flags as not divisible).  For k=1, det = 1 - 2 p_L.
    Right panel = A00(t): logical survival.  Monotone decay is consistent with a
                  Markovian channel; a revival (A00 increases) is information
                  backflow => NON-Markovian (marked red).
    """
    pv = np.asarray(param_values, float)
    ts = np.arange(Tmax + 1)
    det = np.empty((len(pv), Tmax + 1)); A00 = np.empty_like(det)
    for i, v in enumerate(pv):
        ch = make_channel(v, Tmax)
        for t in ts:
            A = ch.stochastic_matrix(t)
            det[i, t] = np.linalg.det(A); A00[i, t] = A[0, 0]
    det = np.where(np.abs(det) < 1e-12, 0.0, det)          # squash float noise around 0

    # pixel edges so integer t and each param row are centred
    tedge = np.arange(-0.5, Tmax + 1.5, 1.0)
    if len(pv) > 1:
        dp = np.diff(pv)
        pedge = np.concatenate(([pv[0]-dp[0]/2], (pv[:-1]+pv[1:])/2, [pv[-1]+dp[-1]/2]))
    else:
        pedge = np.array([pv[0]-0.5, pv[0]+0.5])
    Tg, Pg = np.meshgrid(tedge, pedge)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # --- det / invertibility ---
    pm1 = ax1.pcolormesh(Tg, Pg, det, cmap='RdBu_r',
                         norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading='flat')
    ax1.set(xlabel='time step  t', ylabel=param_label, title=r'$\det(A_t)$  — invertibility')
    fig.colorbar(pm1, ax=ax1, label=r'$\det(A_t)$')

    # --- A00 / monotonicity & backflow ---
    pm2 = ax2.pcolormesh(Tg, Pg, A00, cmap='viridis', vmin=float(A00.min()), vmax=1.0, shading='flat')
    revival = np.diff(A00, axis=1) > 1e-9                  # A00 back up => non-Markovian
    ri, rt = np.where(revival)
    if revival.sum():
        ax2.scatter(ts[rt+1], pv[ri], s=6, c='red', label=f'revival: {int(revival.sum())} cells')
        ax2.legend(loc='upper right', fontsize=8)
    else:
        ax2.text(0.5, 0.05, 'no revivals (monotone) => Markovian', transform=ax2.transAxes,
                 ha='center', fontsize=9, color='white')
    ax2.set(xlabel='time step  t', ylabel=param_label, title=r'$A_{00}(t)$  — monotonicity / backflow')
    fig.colorbar(pm2, ax=ax2, label=r'$A_{00}(t)$')

    if title: fig.suptitle(title, fontsize=12)
    if save:
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches='tight')
    return fig, det, A00
