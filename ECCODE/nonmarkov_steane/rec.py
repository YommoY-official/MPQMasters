"""
Pauli-tracker simulation of n consecutive Recs.

  1 Rec   = Wait — EC
  n Recs  = (Wait — EC) repeated n times.

State carried across the simulation:
    `(xv, zv)`  -- the cumulative Pauli on the 7 data qubits, as int8
                  arrays of shape (..., 7). Batched: leading axis = shot.

Per Rec, the simulation:
    1. Wait gadget: sample one Pauli per data qubit per shot, XOR into (xv, zv).
    2. EC:
         a. Compute the true syndrome of (xv, zv).
         b. Optionally flip syndrome bits (case-2 hook, off by default).
         c. Apply the canonical correction `corrections[s_meas]` to (xv, zv).
         d. Optionally inject a recovery error (case-2 hook, off by default).
         e. Record the syndrome the decoder saw and the post-EC logical class.

For v1 (perfect EC, data-only noise):
    s_meas = s_true and the post-EC syndrome is 0 in every shot.
For v2 (case 2):
    inject syndrome-bit-flip / recovery / ancilla-back-propagated noise via
    `ECNoise`. The Pauli tracker is structured so adding these is a tiny
    edit inside `_apply_ec_step`.

Everything is vectorized over `n_shots` for speed -- 1e6 shots/Rec is ~1 s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .steane_code import build_steane_code
from .noise_model import WaitNoise, ECNoise


_CODE = build_steane_code()
N_DATA = _CODE.n          # 7
R_BITS = _CODE.r          # 6
N_SYN = _CODE.n_syndromes # 64

# Pre-compute correction lookup as ndarrays for vectorized indexing.
# `_CORR_X[s]` (resp. `_CORR_Z[s]`) is the X-part (resp. Z-part) of
# corrections[s], shape (N_DATA,), int8.
_CORR_X = np.zeros((N_SYN, N_DATA), dtype=np.int8)
_CORR_Z = np.zeros((N_SYN, N_DATA), dtype=np.int8)
for s in range(N_SYN):
    cx, cz = _CODE.corrections[s]
    _CORR_X[s] = cx.astype(np.int8)
    _CORR_Z[s] = cz.astype(np.int8)

# Pre-compute symplectic generator matrices for vectorized syndrome computation.
# syndrome[i] = (gz[i] @ xv + gx[i] @ zv) mod 2
# Bit i (MSB = bit 0 from generator order) packs into syndrome integer with
# weight 2^(R_BITS-1-i), matching StabilizerCode.compute_syndrome.
_GX = np.array([gx for gx, _ in _CODE.generators], dtype=np.int8)   # (R, N_DATA)
_GZ = np.array([gz for _, gz in _CODE.generators], dtype=np.int8)   # (R, N_DATA)
_BIT_WEIGHTS = np.array(
    [1 << (R_BITS - 1 - i) for i in range(R_BITS)], dtype=np.int64
)  # (R,)

# Pre-compute parity-check sub-matrices for the explicit CNOT-level EC.
# Z-stabilizers (rows 3..5 = generators g4, g5, g6) detect X errors:
#   Z-syndrome bit = H_Z @ xv_anc (mod 2)  in the Z-extraction.
# X-stabilizers (rows 0..2 = generators g1, g2, g3) detect Z errors:
#   X-syndrome bit = H_X @ zv_anc (mod 2)  in the X-extraction.
# Generator order (MSB first in the syndrome integer):
#   bit 5 = g1 (X-type),  bit 4 = g2 (X-type),  bit 3 = g3 (X-type)
#   bit 2 = g4 (Z-type),  bit 1 = g5 (Z-type),  bit 0 = g6 (Z-type)
_H_X = _GX[:3]    # (3, N_DATA), X-content of X-stabilizers
_H_Z = _GZ[3:]    # (3, N_DATA), Z-content of Z-stabilizers
_X_STAB_BIT_WEIGHTS = np.array([1 << 5, 1 << 4, 1 << 3], dtype=np.int64)  # (3,)
_Z_STAB_BIT_WEIGHTS = np.array([1 << 2, 1 << 1, 1 << 0], dtype=np.int64)  # (3,)


def syndrome_batch(xv: np.ndarray, zv: np.ndarray) -> np.ndarray:
    """
    Vectorized syndrome computation matching StabilizerCode.compute_syndrome.

    xv, zv : shape (..., N_DATA), int8 in {0, 1}
    returns: shape (...), int64 in [0, N_SYN)
    """
    # bits[..., i] = (gz[i] . xv + gx[i] . zv) % 2
    bits_x = np.einsum('rn,...n->...r', _GZ, xv) % 2     # (..., R)
    bits_z = np.einsum('rn,...n->...r', _GX, zv) % 2     # (..., R)
    bits = (bits_x + bits_z) % 2                         # (..., R)
    return (bits.astype(np.int64) * _BIT_WEIGHTS).sum(axis=-1)


def logical_class_batch(xv: np.ndarray, zv: np.ndarray) -> np.ndarray:
    """
    Vectorized logical-class lookup matching StabilizerCode.logical_class
    for ZERO-SYNDROME Paulis on the [[7,1,3]] Steane code with X̄ = X⊗7,
    Z̄ = Z⊗7.

      x_log = parity of xv
      z_log = parity of zv
      idx mapping {(0,0): 0=I, (1,0): 1=X, (1,1): 2=Y, (0,1): 3=Z}

    xv, zv : shape (..., N_DATA), int8
    returns: shape (...), int8 in {0, 1, 2, 3}
    """
    x_log = xv.sum(axis=-1) % 2
    z_log = zv.sum(axis=-1) % 2
    # (x_log, z_log) -> idx
    idx = np.where(
        x_log == 0,
        np.where(z_log == 0, 0, 3),   # (0,0)->I=0, (0,1)->Z=3
        np.where(z_log == 0, 1, 2),   # (1,0)->X=1, (1,1)->Y=2
    )
    return idx.astype(np.int8)


# ---------------------------------------------------------------------------
# Simulation steps (batched over shots)
# ---------------------------------------------------------------------------

def _apply_wait_step(xv: np.ndarray, zv: np.ndarray,
                     wait_noise: WaitNoise,
                     rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply one Wait gadget to a batch of data Paulis.

    xv, zv : (n_shots, N_DATA)
    """
    n_shots = xv.shape[0]
    dx, dz = wait_noise.sample_batch(N_DATA, n_shots, rng)
    return (xv ^ dx, zv ^ dz)


def _apply_ec_step(xv: np.ndarray, zv: np.ndarray,
                   ec_noise: ECNoise,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply one Steane EC at the explicit CNOT level for a batch of shots.

    Steane EC = Z-extraction (detects X errors) + X-extraction (detects Z),
    each consisting of:
        prep ancilla (|0̄⟩ for Z-ext, |+̄⟩ for X-ext) + ancilla-prep noise
        transversal CNOTs + per-CNOT noise on data and ancilla
        pre-measurement noise on ancilla
        Z-basis (resp. X-basis) measurement

    The Pauli tracker carries an ancilla Pauli (xv_a, zv_a) per extraction.
    CNOT propagation rules (control c, target t):
        X_c -> X_c X_t,   Z_c -> Z_c
        X_t -> X_t,       Z_t -> Z_t Z_c
    With anc=target (Z-ext, CX(D, anc)):
        xv_a += xv_d   (data X spreads to ancilla)
        zv_d += zv_a   (ancilla Z spreads back to data)
    With anc=control (X-ext, CX(anc, D)):
        xv_d += xv_a   (ancilla X spreads to data)
        zv_a += zv_d   (data Z spreads to ancilla)

    For Z-basis measurement of a noisy |0̄⟩ codeword, the per-qubit bit
    flips iff there's an X-error on that ancilla qubit (and X stabilizer
    bits are H_Z @ xv_anc, since the random codeword's contribution to
    that parity is zero by construction). Symmetrically for MX of |+̄⟩
    (Z-errors flip).

    With ec_noise == ECNoise() (all sub-noises = WaitNoise(0,0)) this
    function returns syndromes and post-correction data Paulis IDENTICAL
    to the abstract `compute_syndrome + corrections` lookup -- verified
    in the selftest.
    """
    n_shots = xv.shape[0]

    # =====================================================================
    # Z-syndrome extraction (detects X errors via |0̄⟩ ancilla, MZ)
    # =====================================================================
    # Prep ancilla in |0̄⟩ : Pauli error from prep noise.
    xv_a, zv_a = ec_noise.ancilla_prep.sample_batch(N_DATA, n_shots, rng)

    # Transversal CX(D, anc) propagation:
    #   xv_a += xv_d, zv_d += zv_a, xv_d unchanged, zv_a unchanged.
    xv_a = xv_a ^ xv
    zv = zv ^ zv_a

    # Per-CNOT noise on each qubit (data + ancilla), one fresh sample each.
    if ec_noise.cnot_data.p_x > 0 or ec_noise.cnot_data.p_z > 0:
        dx, dz = ec_noise.cnot_data.sample_batch(N_DATA, n_shots, rng)
        xv = xv ^ dx
        zv = zv ^ dz
    if ec_noise.cnot_anc.p_x > 0 or ec_noise.cnot_anc.p_z > 0:
        ax, az = ec_noise.cnot_anc.sample_batch(N_DATA, n_shots, rng)
        xv_a = xv_a ^ ax
        zv_a = zv_a ^ az

    # Pre-measurement noise on ancilla.  For MZ the X-component flips bits.
    if ec_noise.meas.p_x > 0 or ec_noise.meas.p_z > 0:
        mx, mz = ec_noise.meas.sample_batch(N_DATA, n_shots, rng)
        xv_a = xv_a ^ mx
        zv_a = zv_a ^ mz

    # Z-basis bits = H_Z @ xv_a  (mod 2).  3 syndrome bits.
    z_syn_bits = np.einsum('rn,sn->sr', _H_Z, xv_a) % 2     # (n_shots, 3)

    # =====================================================================
    # X-syndrome extraction (detects Z errors via |+̄⟩ ancilla, MX)
    # =====================================================================
    # Fresh ancilla in |+̄⟩ : Pauli error from prep noise.
    xv_a2, zv_a2 = ec_noise.ancilla_prep.sample_batch(N_DATA, n_shots, rng)

    # Transversal CX(anc, D) propagation:
    #   xv_d += xv_a2, zv_a2 += zv_d, xv_a2 unchanged, zv_d unchanged.
    xv = xv ^ xv_a2
    zv_a2 = zv_a2 ^ zv

    if ec_noise.cnot_data.p_x > 0 or ec_noise.cnot_data.p_z > 0:
        dx, dz = ec_noise.cnot_data.sample_batch(N_DATA, n_shots, rng)
        xv = xv ^ dx
        zv = zv ^ dz
    if ec_noise.cnot_anc.p_x > 0 or ec_noise.cnot_anc.p_z > 0:
        ax, az = ec_noise.cnot_anc.sample_batch(N_DATA, n_shots, rng)
        xv_a2 = xv_a2 ^ ax
        zv_a2 = zv_a2 ^ az

    # Pre-measurement noise on ancilla.  For MX the Z-component flips bits.
    if ec_noise.meas.p_x > 0 or ec_noise.meas.p_z > 0:
        mx, mz = ec_noise.meas.sample_batch(N_DATA, n_shots, rng)
        xv_a2 = xv_a2 ^ mx
        zv_a2 = zv_a2 ^ mz

    # X-basis bits = H_X @ zv_a2  (mod 2).
    x_syn_bits = np.einsum('rn,sn->sr', _H_X, zv_a2) % 2    # (n_shots, 3)

    # =====================================================================
    # Combine into 6-bit syndrome integer  (X-stabs in MSB, Z-stabs in LSB)
    # =====================================================================
    s_meas = (
        (x_syn_bits.astype(np.int64) * _X_STAB_BIT_WEIGHTS).sum(axis=-1)
        + (z_syn_bits.astype(np.int64) * _Z_STAB_BIT_WEIGHTS).sum(axis=-1)
    )

    # Apply correction = corrections[s_meas].
    xv = xv ^ _CORR_X[s_meas]
    zv = zv ^ _CORR_Z[s_meas]

    # Recovery-application error.
    if ec_noise.recovery.p_x > 0 or ec_noise.recovery.p_z > 0:
        rx, rz = ec_noise.recovery.sample_batch(N_DATA, n_shots, rng)
        xv = xv ^ rx
        zv = zv ^ rz

    return xv, zv, s_meas


# ---------------------------------------------------------------------------
# Public run function
# ---------------------------------------------------------------------------

@dataclass
class RecResult:
    """
    Outputs of a batched n_shots-by-n_recs Pauli-tracker run.

    Shapes (n_shots given as N for brevity):
        s_in           : (N,) int64        -- injected input syndrome (constant per call)
        syndromes      : (N, n_recs) int64 -- syndrome the decoder saw at each EC
        logical_classes: (N, n_recs) int8  -- L in {0=I,1=X,2=Y,3=Z} after each EC
        final_pauli    : (N, 2, N_DATA) int8 -- (xv, zv) at end of last Rec
                          (in zero-syndrome regime when EC is perfect)
    """
    s_in: int
    syndromes: np.ndarray
    logical_classes: np.ndarray
    final_pauli: Optional[np.ndarray] = None


def simulate_recs(
    n_recs: int,
    n_shots: int,
    s_in: int = 0,
    wait_noise: WaitNoise = WaitNoise(),
    ec_noise: ECNoise = ECNoise(),
    rng: Optional[np.random.Generator] = None,
    keep_final_pauli: bool = False,
) -> RecResult:
    """
    Run `n_recs` consecutive Recs (each = Wait — EC) on `n_shots` independent
    shots in parallel, with the data initialized to corrections[s_in].

    Parameters
    ----------
    n_recs : int
        Number of Recs (= number of Wait+EC pairs).
    n_shots : int
        Number of independent shots (Monte-Carlo trials).
    s_in : int in [0, 64)
        Injected incoming syndrome. Data starts at corrections[s_in].
    wait_noise : WaitNoise
        Noise applied at each Wait. Defaults to no noise.
    ec_noise : ECNoise
        Noise applied at each EC. Defaults to perfect EC.
    rng : np.random.Generator
        Random source. If None, uses np.random.default_rng().
    keep_final_pauli : bool
        If True, also return the (xv, zv) Pauli at the end of the last Rec.

    Returns
    -------
    RecResult
    """
    if rng is None:
        rng = np.random.default_rng()
    s_in = int(s_in)
    assert 0 <= s_in < N_SYN

    # Initialize data Pauli to corrections[s_in], broadcast across shots.
    cx0 = _CORR_X[s_in]
    cz0 = _CORR_Z[s_in]
    xv = np.broadcast_to(cx0, (n_shots, N_DATA)).copy()
    zv = np.broadcast_to(cz0, (n_shots, N_DATA)).copy()

    syndromes = np.zeros((n_shots, n_recs), dtype=np.int64)
    logicals = np.zeros((n_shots, n_recs), dtype=np.int8)

    for t in range(n_recs):
        xv, zv = _apply_wait_step(xv, zv, wait_noise, rng)
        xv, zv, s_meas = _apply_ec_step(xv, zv, ec_noise, rng)
        syndromes[:, t] = s_meas
        logicals[:, t] = logical_class_batch(xv, zv)

    final = np.stack([xv, zv], axis=1) if keep_final_pauli else None
    return RecResult(
        s_in=s_in,
        syndromes=syndromes,
        logical_classes=logicals,
        final_pauli=final,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> dict:
    rng = np.random.default_rng(0)
    code = _CODE
    checks: dict = {}

    # T1: zero noise. Per shot, all syndromes must be 0 (perfect EC absorbs
    # the s_in injection at the first EC; subsequent ECs see no error)
    # and L must be 0=I.
    for s_in in [0, 1, 7, 31, 63]:
        res = simulate_recs(n_recs=2, n_shots=64, s_in=s_in,
                            wait_noise=WaitNoise(0.0, 0.0),
                            ec_noise=ECNoise(),
                            rng=np.random.default_rng(1234 + s_in))
        # First EC sees exactly s_in (since wait1 is noiseless), corrects to 0.
        assert np.all(res.syndromes[:, 0] == s_in), (
            f"T1 s_in={s_in}: first syndrome should equal s_in; "
            f"got {np.unique(res.syndromes[:,0]).tolist()}")
        assert np.all(res.syndromes[:, 1] == 0), (
            f"T1 s_in={s_in}: second syndrome should be 0; "
            f"got {np.unique(res.syndromes[:,1]).tolist()}")
        assert np.all(res.logical_classes == 0), (
            f"T1 s_in={s_in}: all logical classes should be I (0); "
            f"got {np.unique(res.logical_classes).tolist()}")
    checks['T1_zero_noise_clean_recovery'] = True

    # T2: Inject single-qubit X error WITHOUT correction lookup, and check
    # that simulate_recs at p=0 with s_in matching the Pauli's syndrome
    # corrects it to L=I.
    for q in range(7):
        xv = np.zeros((1, 7), dtype=np.int8); xv[0, q] = 1
        zv = np.zeros((1, 7), dtype=np.int8)
        s = int(syndrome_batch(xv, zv)[0])
        # Use this s as the s_in: simulate_recs starts data at corrections[s];
        # at p=0, EC1 sees s and corrects to identity.
        res = simulate_recs(n_recs=1, n_shots=4, s_in=s,
                            wait_noise=WaitNoise(0.0, 0.0),
                            rng=np.random.default_rng(99 + q),
                            keep_final_pauli=True)
        assert np.all(res.logical_classes == 0)
        assert np.all(res.final_pauli == 0)
    checks['T2_single_X_corrected_to_identity'] = 7

    # T3a: vectorized syndrome matches StabilizerCode.compute_syndrome on
    #      a random batch of Paulis.
    n = 200
    xv = rng.integers(0, 2, size=(n, 7), dtype=np.int8)
    zv = rng.integers(0, 2, size=(n, 7), dtype=np.int8)
    s_batch = syndrome_batch(xv, zv)
    for i in range(n):
        ref = code.compute_syndrome(xv[i].astype(int), zv[i].astype(int))
        assert int(s_batch[i]) == ref, f"syndrome mismatch at {i}: {s_batch[i]} vs {ref}"
    checks['T3a_vectorized_syndrome_matches'] = n

    # T3b: vectorized logical_class matches StabilizerCode.logical_class
    #      on a constructed batch of zero-syndrome Paulis (one per L class).
    # Build I, X̄, Ȳ, Z̄ representatives.
    Is = (np.zeros(7, np.int8), np.zeros(7, np.int8))
    Xs = (np.ones(7, np.int8), np.zeros(7, np.int8))            # X̄ = X⊗7
    Zs = (np.zeros(7, np.int8), np.ones(7, np.int8))            # Z̄ = Z⊗7
    Ys = (np.ones(7, np.int8), np.ones(7, np.int8))             # Ȳ = X̄·Z̄
    expected_L = [0, 1, 2, 3]                                   # I, X, Y, Z
    xv_test = np.stack([Is[0], Xs[0], Ys[0], Zs[0]])
    zv_test = np.stack([Is[1], Xs[1], Ys[1], Zs[1]])
    L_batch = logical_class_batch(xv_test, zv_test)
    assert L_batch.tolist() == expected_L, (
        f"logical_class batch: got {L_batch.tolist()}, expected {expected_L}")
    for i in range(4):
        ref_L, _ = code.logical_class(xv_test[i].astype(int), zv_test[i].astype(int))
        assert int(L_batch[i]) == ref_L
    checks['T3b_vectorized_logical_class_matches'] = 4

    # T4: at p > 0, syndromes should sometimes be non-zero.
    res = simulate_recs(n_recs=2, n_shots=2000, s_in=0,
                        wait_noise=WaitNoise(p_x=0.05, p_z=0.05),
                        rng=np.random.default_rng(42))
    n_nontrivial = int((res.syndromes[:, 0] != 0).sum())
    assert n_nontrivial > 0, "noise should produce some non-zero syndromes"
    checks['T4_p_pos_produces_nonzero_syndromes'] = n_nontrivial

    # T5: case-2 hook -- ancilla noise produces non-zero second syndrome
    # even at p_wait = 0 (the ancilla X errors during the Z-extraction
    # spread back as Z onto the data via CNOT propagation, so the second
    # EC reads a residual error).
    res = simulate_recs(n_recs=2, n_shots=5000, s_in=0,
                        wait_noise=WaitNoise(0.0, 0.0),
                        ec_noise=ECNoise(ancilla_prep=WaitNoise(p_x=0.05, p_z=0.05)),
                        rng=np.random.default_rng(7))
    n_resid = int((res.syndromes[:, 1] != 0).sum())
    assert n_resid > 0, "with ancilla noise, residual syndromes expected"
    checks['T5_ancilla_noise_produces_residuals'] = n_resid

    # T6: explicit-CNOT EC matches the abstract syndrome computation when
    # ec_noise = ECNoise() (all zero). For a randomly chosen pre-EC data
    # Pauli, the syndrome reported by simulate_recs(n_recs=1) should
    # equal compute_syndrome of (corrections[s_in] XOR Wait noise) sampled
    # from the same RNG, because the explicit CNOT model is mathematically
    # equivalent to the abstraction in case 1.
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    n = 500
    res_a = simulate_recs(n_recs=1, n_shots=n, s_in=0,
                          wait_noise=WaitNoise(p_x=0.05, p_z=0.05),
                          rng=rng_a, keep_final_pauli=True)
    # Reproduce the same sampling path manually.
    cx0 = _CORR_X[0]; cz0 = _CORR_Z[0]
    xv_b = np.broadcast_to(cx0, (n, 7)).copy()
    zv_b = np.broadcast_to(cz0, (n, 7)).copy()
    dx, dz = WaitNoise(p_x=0.05, p_z=0.05).sample_batch(7, n, rng_b)
    xv_b ^= dx; zv_b ^= dz
    s_ref = syndrome_batch(xv_b, zv_b)
    cx = _CORR_X[s_ref]; cz = _CORR_Z[s_ref]
    xv_b ^= cx; zv_b ^= cz
    assert np.array_equal(res_a.syndromes[:, 0], s_ref), (
        "explicit CNOT EC syndrome mismatch with abstract model in case 1")
    assert np.array_equal(res_a.final_pauli[:, 0], xv_b), (
        "explicit CNOT EC post-correction xv mismatch")
    assert np.array_equal(res_a.final_pauli[:, 1], zv_b), (
        "explicit CNOT EC post-correction zv mismatch")
    checks['T6_explicit_matches_abstract_at_zero_ec_noise'] = n

    if verbose:
        print("rec._selftest passed:")
        for k, v in checks.items():
            print(f"  {k:46s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
