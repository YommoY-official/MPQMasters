"""
Noise model for the Pauli-tracker simulation of Steane EC.

V1 -- noise on data qubits during Wait gadgets:
    Each data qubit independently gets one of:
        I  with prob 1 - p_x - p_z   (mutually-exclusive default)
        X  with prob p_x
        Z  with prob p_z

    With `correlated_xz=True`, X and Z are sampled INDEPENDENTLY
    (so Y = X·Z occurs with prob p_x*p_z). This is the hook for the
    extension you mentioned -- flip one flag, no other code changes.

V2 -- noise on ancilla / measurement / recovery during EC:
    Captured by ECNoise. For v1 leave all fields = 0 (perfect EC).
    The Pauli-tracker step in `rec.py` already reads these fields so
    case 2 only requires turning them on (and supplying the dynamics
    inside the EC step that consumes them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Wait-location noise on data qubits
# ---------------------------------------------------------------------------

@dataclass
class WaitNoise:
    """
    Pauli noise applied at each Wait gadget on each data qubit.

    Attributes
    ----------
    p_x, p_z : float
        Per-qubit probabilities for X and Z.
    correlated_xz : bool
        False (default): X and Z are MUTUALLY EXCLUSIVE per qubit.
                        I/X/Z occur with probs (1-p_x-p_z, p_x, p_z).
                        Requires p_x + p_z <= 1.
        True: X and Z are INDEPENDENT (Y can occur).
              I/X/Z/Y occur with probs
                  (1-p_x)(1-p_z), p_x(1-p_z), (1-p_x)p_z, p_x*p_z.
    """
    p_x: float = 0.0
    p_z: float = 0.0
    correlated_xz: bool = False

    def __post_init__(self):
        if not (0.0 <= self.p_x <= 1.0 and 0.0 <= self.p_z <= 1.0):
            raise ValueError(f"p_x and p_z must lie in [0, 1] "
                             f"(got p_x={self.p_x}, p_z={self.p_z})")
        if (not self.correlated_xz) and (self.p_x + self.p_z > 1.0 + 1e-12):
            raise ValueError(
                f"With correlated_xz=False, X and Z are mutually exclusive, "
                f"so p_x + p_z must be <= 1 (got {self.p_x + self.p_z})")

    def sample(self, n_qubits: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample one Pauli error per qubit. Returns (xv, zv), each of
        shape (n_qubits,) with int8 entries in {0, 1}.
        """
        if self.correlated_xz:
            xv = (rng.random(n_qubits) < self.p_x).astype(np.int8)
            zv = (rng.random(n_qubits) < self.p_z).astype(np.int8)
        else:
            xv = np.zeros(n_qubits, dtype=np.int8)
            zv = np.zeros(n_qubits, dtype=np.int8)
            r = rng.random(n_qubits)
            x_mask = r < self.p_x
            z_mask = (r >= self.p_x) & (r < self.p_x + self.p_z)
            xv[x_mask] = 1
            zv[z_mask] = 1
        return xv, zv

    def sample_batch(self, n_qubits: int, n_shots: int,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized version: returns (xv, zv) of shape (n_shots, n_qubits)."""
        if self.correlated_xz:
            xv = (rng.random((n_shots, n_qubits)) < self.p_x).astype(np.int8)
            zv = (rng.random((n_shots, n_qubits)) < self.p_z).astype(np.int8)
        else:
            r = rng.random((n_shots, n_qubits))
            xv = (r < self.p_x).astype(np.int8)
            zv = ((r >= self.p_x) & (r < self.p_x + self.p_z)).astype(np.int8)
        return xv, zv


# ---------------------------------------------------------------------------
# EC-location noise (placeholder for case 2; defaults to perfect EC)
# ---------------------------------------------------------------------------

from dataclasses import field


@dataclass
class ECNoise:
    """
    Per-location noise inside the explicit CNOT-level Steane EC.

    Each field is a `WaitNoise` describing the Pauli error injected at that
    location. Default = WaitNoise() i.e. no noise (perfect EC, recovers
    the case-1 abstraction exactly).

    Locations modelled by `_apply_ec_step` in rec.py:

        ancilla_prep : applied to the 7 ancilla qubits right after they
                        are prepared in |0̄⟩ or |+̄⟩, before the
                        transversal CNOTs. Captures prep / idle errors.
        cnot_data   : applied to each of the 7 data qubits AT each CNOT
                       location (so once per Z-extraction CNOT block, once
                       per X-extraction CNOT block — two independent draws).
        cnot_anc    : same on the 7 ancilla qubits per CNOT block.
        meas        : applied to ancilla qubits immediately before
                       measurement. For a Z-basis measurement, only X-flips
                       (p_x) flip the readout; for X-basis only Z-flips
                       (p_z) do. Both are kept in the model; they have
                       different physical meanings on the ancilla state.
        recovery    : applied to the 7 data qubits AFTER the decoder's
                       Pauli correction is XOR'd in. Captures imperfect
                       recovery application.

    For the user's case 2 ("ancilla qubits also prone to the same X/Z
    mutex noise as data"), set ancilla_prep = WaitNoise(p_x=p, p_z=p,
    correlated_xz=False); leave the rest at default for a minimal model,
    or turn cnot_anc / meas on for a richer one.
    """
    ancilla_prep: 'WaitNoise' = field(default_factory=WaitNoise)
    cnot_data:    'WaitNoise' = field(default_factory=WaitNoise)
    cnot_anc:     'WaitNoise' = field(default_factory=WaitNoise)
    meas:         'WaitNoise' = field(default_factory=WaitNoise)
    recovery:     'WaitNoise' = field(default_factory=WaitNoise)

    @property
    def is_perfect(self) -> bool:
        zero = WaitNoise(0.0, 0.0)
        return (self.ancilla_prep == zero and self.cnot_data == zero
                and self.cnot_anc == zero and self.meas == zero
                and self.recovery == zero)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> dict:
    rng = np.random.default_rng(0)
    checks: dict = {}
    n_qubits = 7
    n_shots = 200_000

    # Mutually-exclusive case 1
    p = 0.05
    nm = WaitNoise(p_x=p, p_z=p, correlated_xz=False)
    xv, zv = nm.sample_batch(n_qubits, n_shots, rng)
    px_emp = float(xv.mean())
    pz_emp = float(zv.mean())
    pxz_emp = float((xv & zv).mean())
    assert abs(px_emp - p) < 0.005, f"X rate {px_emp} vs {p}"
    assert abs(pz_emp - p) < 0.005, f"Z rate {pz_emp} vs {p}"
    assert pxz_emp == 0.0, f"with mutex, X and Z must never coincide; got {pxz_emp}"
    checks['mutex_p_x_rate'] = round(px_emp, 4)
    checks['mutex_p_z_rate'] = round(pz_emp, 4)
    checks['mutex_no_simultaneous_XZ'] = pxz_emp == 0.0

    # Independent (correlated_xz=True) case
    nm2 = WaitNoise(p_x=p, p_z=p, correlated_xz=True)
    xv, zv = nm2.sample_batch(n_qubits, n_shots, rng)
    pxz_emp = float((xv & zv).mean())
    assert abs(pxz_emp - p * p) < 0.001, (
        f"Y rate (X & Z) should be p*p = {p*p}, got {pxz_emp}")
    checks['indep_p_y_rate'] = round(pxz_emp, 6)

    # Constructor validation
    try:
        WaitNoise(p_x=0.6, p_z=0.6, correlated_xz=False)
    except ValueError:
        checks['mutex_validates_p_x_plus_p_z_le_1'] = True

    # ECNoise
    en = ECNoise()
    assert en.is_perfect
    en2 = ECNoise(ancilla_prep=WaitNoise(0.01, 0.01))
    assert not en2.is_perfect
    checks['ECNoise_perfect_default'] = True

    if verbose:
        print("noise_model._selftest passed:")
        for k, v in checks.items():
            print(f"  {k:40s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
