"""
Steane EC gadget — Stim circuit fragments.

Two-block Steane-style EC for the [[7,1,3]] code, non-FT ancilla preparation.

ONE EXTRACTION ROUND
--------------------
Z-syndrome extraction (detects X errors):
    1. Prepare 7-qubit ancilla in |0̄⟩ via non-FT encoder.
    2. CX(data_i, anc_i) transversally   (control = data, target = ancilla).
    3. Measure ancilla in Z basis        (M).
X-syndrome extraction (detects Z errors):
    1. Prepare 7-qubit ancilla in |+̄⟩ = H̄|0̄⟩.
    2. CX(anc_i, data_i) transversally   (control = ancilla, target = data).
    3. Measure ancilla in X basis        (MX).

The 14 measurement outcomes (7 Z-anc, 7 X-anc) are post-processed into a
6-bit syndrome integer via
    StabilizerCode.compute_syndrome(xv=m_z, zv=m_x).
The X-type syndrome bits (positions 5..3 in the integer) detect Z errors and
are fed by `m_x`; the Z-type bits (positions 2..0) detect X errors and are
fed by `m_z` — consistent with `nonmarkov_steane.steane_code`.

PAULI CORRECTION
----------------
Stim cannot apply non-linear classically-conditioned Paulis (the Hamming
lookup is non-linear in the syndrome bits). We therefore do NOT apply
corrections inside the Stim circuit. Instead, post-processing tracks a
running cumulative software correction; the corrected logical class is
extracted at the end of each Rec via the Bell-pair logical-observable trick
(see rec.py / simulation.py). This is mathematically equivalent to applying
the correction physically (verified algebraically: `s_t^phys = m_t XOR m_{t-1}`
for consecutive ECs in a no-correction circuit, identical to what a real FT
decoder would observe at EC t).

CONSENSUS
---------
For fault tolerance, the EC gadget repeats one extraction round 3 times and
takes a per-bit majority vote on the 6-bit syndromes. This catches single
faults in the syndrome extraction itself (per QECCbook §12.2.2).

V1 LIMITATION (non-FT ancilla prep)
-----------------------------------
A single fault in the ancilla encoder can become two correlated errors on
the ancilla and propagate to the data. The qualitative non-Markovian
correlations between consecutive Recs do not depend on this; full Steane
verified ancilla prep is a TODO for v2.

NOISE CONVENTIONS (Stim)
------------------------
DEPOLARIZE1(p): applies X, Y, or Z each with probability p/3
                (verified empirically; total non-identity probability = p).
DEPOLARIZE2(p): applies one of 15 non-identity 2-qubit Paulis each with
                probability p/15 (total non-identity = p).
Measurement bit-flip: X_ERROR(p) on the qubit immediately before M,
                      or Z_ERROR(p) before MX.
State preparation:    X_ERROR(p) immediately after R (= RZ),
                      or Z_ERROR(p) after RX.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import stim

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from .steane_code import build_steane_code

# ---------------------------------------------------------------------------
# Encoder for |0̄⟩ on 7 fresh qubits  (non-FT, Steane stabilizer projectors).
# Each X-stabilizer has one "anchor" qubit appearing in only that stabilizer:
#     g1 = XXXXIII   anchor = q3 (only g1 contains q3)
#     g2 = XXIIXXI   anchor = q5 (only g2 contains q5)
#     g3 = XIXIXIX   anchor = q6 (only g3 contains q6)
# Apply H on each anchor, then CNOT anchor -> other support qubits.
# ---------------------------------------------------------------------------

_ZBAR_ANCHORS = [3, 5, 6]
_ZBAR_CNOTS: List[Tuple[int, int]] = [
    (3, 0), (3, 1), (3, 2),   # spread g1 from q3
    (5, 0), (5, 1), (5, 4),   # spread g2 from q5
    (6, 0), (6, 2), (6, 4),   # spread g3 from q6
]


def _append_prep_zero_bar(circuit: stim.Circuit, qubits: Sequence[int], p: float) -> None:
    """Append non-FT |0̄⟩ preparation on the 7 qubits in `qubits`."""
    q = list(qubits)
    assert len(q) == 7
    circuit.append('R', q)
    if p > 0:
        circuit.append('X_ERROR', q, p)
    h_targets = [q[i] for i in _ZBAR_ANCHORS]
    circuit.append('H', h_targets)
    if p > 0:
        circuit.append('DEPOLARIZE1', h_targets, p)
    for (ctrl, tgt) in _ZBAR_CNOTS:
        pair = [q[ctrl], q[tgt]]
        circuit.append('CX', pair)
        if p > 0:
            circuit.append('DEPOLARIZE2', pair, p)


def _append_prep_plus_bar(circuit: stim.Circuit, qubits: Sequence[int], p: float) -> None:
    """Append non-FT |+̄⟩ preparation: |0̄⟩ then transversal H̄."""
    _append_prep_zero_bar(circuit, qubits, p)
    qs = list(qubits)
    circuit.append('H', qs)
    if p > 0:
        circuit.append('DEPOLARIZE1', qs, p)


def _append_prep_data_zero_bar(circuit: stim.Circuit, data: Sequence[int], p: float) -> None:
    """
    Encode |0̄⟩ into the 7 data qubits at the start of the experiment.
    Reuses the same non-FT encoder used for ancilla prep.
    """
    _append_prep_zero_bar(circuit, data, p)


# ---------------------------------------------------------------------------
# One EC extraction round (Z-syndrome block + X-syndrome block).
# ---------------------------------------------------------------------------

@dataclass
class ECExtractionRecord:
    """Bookkeeping for one EC extraction round.

    Stores the indices into the measurement record (negative-indexed from
    the END of the circuit at the moment the round was appended) for each
    of the 14 ancilla measurements. After all measurements are appended,
    callers should rebase these to absolute indices via
    `rebase(total_measurement_count)`.
    """
    z_anc_qubits: List[int] = field(default_factory=list)   # 7 qubits, M outcomes
    x_anc_qubits: List[int] = field(default_factory=list)   # 7 qubits, MX outcomes
    z_meas_record_offset: int = -1   # absolute index of the FIRST z-anc M in the record
    x_meas_record_offset: int = -1   # absolute index of the FIRST x-anc MX in the record


def append_ec_round(
    circuit: stim.Circuit,
    data: Sequence[int],
    z_anc: Sequence[int],
    x_anc: Sequence[int],
    p: float,
    measurement_count_so_far: int,
) -> ECExtractionRecord:
    """
    Append one Steane EC extraction round (one Z-syndrome block + one X-syndrome
    block). Returns an ECExtractionRecord pinning down where the 14 ancilla
    measurements live in the global measurement record.
    """
    assert len(data) == 7 and len(z_anc) == 7 and len(x_anc) == 7
    rec = ECExtractionRecord(
        z_anc_qubits=list(z_anc),
        x_anc_qubits=list(x_anc),
    )

    # ----- Z-syndrome block (detects X errors) -----
    _append_prep_zero_bar(circuit, z_anc, p)
    # Transversal CX(data -> z_anc)
    cx_pairs_z: List[int] = []
    for i in range(7):
        cx_pairs_z.extend([data[i], z_anc[i]])
    circuit.append('CX', cx_pairs_z)
    if p > 0:
        circuit.append('DEPOLARIZE2', cx_pairs_z, p)
    # Measurement noise just before M
    if p > 0:
        circuit.append('X_ERROR', list(z_anc), p)
    rec.z_meas_record_offset = measurement_count_so_far
    circuit.append('M', list(z_anc))
    measurement_count_so_far += 7

    # ----- X-syndrome block (detects Z errors) -----
    _append_prep_plus_bar(circuit, x_anc, p)
    # Transversal CX(x_anc -> data)
    cx_pairs_x: List[int] = []
    for i in range(7):
        cx_pairs_x.extend([x_anc[i], data[i]])
    circuit.append('CX', cx_pairs_x)
    if p > 0:
        circuit.append('DEPOLARIZE2', cx_pairs_x, p)
    # Measurement noise: MX measures in X basis; pre-measurement X_ERROR is
    # equivalent to a Z_ERROR on the qubit (since MX ~ H then M; X commuted
    # past H is Z). Stim's MX is a single instruction; the standard
    # convention for noisy MX is Z_ERROR(p) before MX.
    if p > 0:
        circuit.append('Z_ERROR', list(x_anc), p)
    rec.x_meas_record_offset = measurement_count_so_far
    circuit.append('MX', list(x_anc))
    measurement_count_so_far += 7

    return rec


# ---------------------------------------------------------------------------
# Multi-round (consensus) EC: 3 extraction rounds, per-bit majority vote
# ---------------------------------------------------------------------------

@dataclass
class ECRecord:
    """Bookkeeping for one full EC gadget (3 consensus rounds)."""
    rounds: List[ECExtractionRecord] = field(default_factory=list)


def append_ec_gadget(
    circuit: stim.Circuit,
    data: Sequence[int],
    ancilla_qubit_pool: Sequence[int],
    p: float,
    measurement_count_so_far: int,
    n_rounds: int = 3,
) -> Tuple[ECRecord, int]:
    """
    Append one full Steane EC gadget (n_rounds extraction rounds).
    `ancilla_qubit_pool` must provide n_rounds * 14 distinct qubits (used 7
    for z_anc and 7 for x_anc per round, no reset/reuse in v1).

    Returns (ECRecord, updated measurement_count_so_far).
    """
    pool = list(ancilla_qubit_pool)
    needed = n_rounds * 14
    assert len(pool) >= needed, (
        f"need {needed} ancilla qubits, got {len(pool)}")
    record = ECRecord()
    for r in range(n_rounds):
        z_anc = pool[14 * r: 14 * r + 7]
        x_anc = pool[14 * r + 7: 14 * (r + 1)]
        round_rec = append_ec_round(
            circuit=circuit,
            data=data,
            z_anc=z_anc,
            x_anc=x_anc,
            p=p,
            measurement_count_so_far=measurement_count_so_far,
        )
        measurement_count_so_far += 14
        record.rounds.append(round_rec)
    return record, measurement_count_so_far


# ---------------------------------------------------------------------------
# Postprocessing: extract syndrome from raw measurement bits.
# ---------------------------------------------------------------------------

_CODE = build_steane_code()


def syndrome_from_round(round_rec: ECExtractionRecord, all_bits: np.ndarray) -> int:
    """
    Compute the 6-bit syndrome integer (per code conventions) for ONE
    extraction round, given the full per-shot measurement-record bit array.
    """
    z_off = round_rec.z_meas_record_offset
    x_off = round_rec.x_meas_record_offset
    m_z = all_bits[z_off: z_off + 7].astype(int)   # X-error parities -> Z-stabs
    m_x = all_bits[x_off: x_off + 7].astype(int)   # Z-error parities -> X-stabs
    return int(_CODE.compute_syndrome(m_z, m_x))


def syndrome_from_gadget(ec_rec: ECRecord, all_bits: np.ndarray) -> int:
    """Per-bit majority-vote consensus 6-bit syndrome over EC's rounds."""
    per_round = np.array([syndrome_from_round(r, all_bits) for r in ec_rec.rounds],
                         dtype=int)
    bits_per_round = np.zeros((len(per_round), _CODE.r), dtype=int)
    for k, s in enumerate(per_round):
        bits_per_round[k] = _CODE.syndrome_bits(s)
    consensus_bits = (bits_per_round.sum(axis=0) > (len(per_round) // 2)).astype(int)
    s_consensus = 0
    for i in range(_CODE.r):
        s_consensus |= int(consensus_bits[i]) << (_CODE.r - 1 - i)
    return s_consensus


def syndrome_from_gadget_batch(ec_rec: ECRecord, all_bits_batch: np.ndarray) -> np.ndarray:
    """
    Vectorized version: all_bits_batch shape (n_shots, n_measurements).
    Returns shape (n_shots,) of int syndromes.
    """
    n_shots = all_bits_batch.shape[0]
    out = np.zeros(n_shots, dtype=np.int64)
    for shot in range(n_shots):
        out[shot] = syndrome_from_gadget(ec_rec, all_bits_batch[shot])
    return out


# ---------------------------------------------------------------------------
# Self-tests for Step 2.
# ---------------------------------------------------------------------------

def _build_test_circuit(
    inject_pauli: Sequence[Tuple[str, int]] = (),
    p: float = 0.0,
    n_rounds: int = 3,
) -> Tuple[stim.Circuit, ECRecord]:
    """
    Build a noiseless test circuit:
       - prepare 7 data qubits in |0̄⟩
       - inject the listed Paulis on the data
       - run one EC gadget (n_rounds consensus rounds)
    Returns (circuit, EC bookkeeping).
    """
    data = list(range(7))
    pool = list(range(7, 7 + 14 * n_rounds))
    c = stim.Circuit()
    _append_prep_data_zero_bar(c, data, p)
    for op, q in inject_pauli:
        assert op in ('X', 'Y', 'Z'), f"unknown Pauli {op}"
        c.append(op, [data[q]])
    ec_rec, _ = append_ec_gadget(c, data, pool, p, measurement_count_so_far=0,
                                 n_rounds=n_rounds)
    return c, ec_rec


def _selftest(verbose: bool = True) -> dict:
    """
    Step-2 self-tests:
      (T1) Zero noise + zero injected error: syndrome must be 0 every shot.
      (T2) Zero noise + single-qubit X on each data qubit q: syndrome must
           equal the Steane code's syndrome for X_q on every shot AND the
           lookup correction must be exactly X_q (weight-1 correction).
      (T3) Same as T2 but for Z_q.
      (T4) Same as T2 but for Y_q (= X_q · Z_q): syndrome should equal
           syndrome(X_q) XOR syndrome(Z_q) and correction lookup gives
           Y_q (weight 1).
    """
    rng = np.random.default_rng(0)
    code = _CODE
    checks = {}
    n_shots = 200

    # --- T1: identity ---
    c, ec_rec = _build_test_circuit(inject_pauli=(), p=0.0)
    sampler = c.compile_sampler(seed=int(rng.integers(0, 2**31 - 1)))
    bits = sampler.sample(shots=n_shots).astype(int)
    syndromes = syndrome_from_gadget_batch(ec_rec, bits)
    assert np.all(syndromes == 0), (
        f"T1 failed: noiseless |0̄⟩ produced non-zero syndromes "
        f"(unique: {np.unique(syndromes)})")
    checks['T1_identity_syndrome_always_zero'] = int(n_shots)

    # --- T2: single-X errors ---
    for q in range(7):
        xv = np.zeros(7, dtype=int); xv[q] = 1
        zv = np.zeros(7, dtype=int)
        expected_syndrome = int(code.compute_syndrome(xv, zv))
        c, ec_rec = _build_test_circuit(inject_pauli=[('X', q)], p=0.0)
        sampler = c.compile_sampler(seed=int(rng.integers(0, 2**31 - 1)))
        bits = sampler.sample(shots=n_shots).astype(int)
        syndromes = syndrome_from_gadget_batch(ec_rec, bits)
        assert np.all(syndromes == expected_syndrome), (
            f"T2 failed for X_{q}: expected syndrome {expected_syndrome}, "
            f"got unique {np.unique(syndromes).tolist()}")
        cx, cz = code.corrections[expected_syndrome]
        # The correction MUST be exactly X on qubit q (weight 1).
        expected_cx = np.zeros(7, dtype=int); expected_cx[q] = 1
        assert np.array_equal(cx, expected_cx) and not np.any(cz), (
            f"T2 lookup failed for X_{q}: got correction "
            f"x={cx.tolist()}, z={cz.tolist()}")
    checks['T2_single_X_correctly_decoded_for_each_qubit'] = 7

    # --- T3: single-Z errors ---
    for q in range(7):
        xv = np.zeros(7, dtype=int)
        zv = np.zeros(7, dtype=int); zv[q] = 1
        expected_syndrome = int(code.compute_syndrome(xv, zv))
        c, ec_rec = _build_test_circuit(inject_pauli=[('Z', q)], p=0.0)
        sampler = c.compile_sampler(seed=int(rng.integers(0, 2**31 - 1)))
        bits = sampler.sample(shots=n_shots).astype(int)
        syndromes = syndrome_from_gadget_batch(ec_rec, bits)
        assert np.all(syndromes == expected_syndrome), (
            f"T3 failed for Z_{q}: expected {expected_syndrome}, "
            f"got unique {np.unique(syndromes).tolist()}")
        cx, cz = code.corrections[expected_syndrome]
        expected_cz = np.zeros(7, dtype=int); expected_cz[q] = 1
        assert (not np.any(cx)) and np.array_equal(cz, expected_cz), (
            f"T3 lookup failed for Z_{q}: got x={cx.tolist()}, z={cz.tolist()}")
    checks['T3_single_Z_correctly_decoded_for_each_qubit'] = 7

    # --- T4: single-Y errors ---
    for q in range(7):
        xv = np.zeros(7, dtype=int); xv[q] = 1
        zv = np.zeros(7, dtype=int); zv[q] = 1
        expected_syndrome = int(code.compute_syndrome(xv, zv))
        c, ec_rec = _build_test_circuit(inject_pauli=[('Y', q)], p=0.0)
        sampler = c.compile_sampler(seed=int(rng.integers(0, 2**31 - 1)))
        bits = sampler.sample(shots=n_shots).astype(int)
        syndromes = syndrome_from_gadget_batch(ec_rec, bits)
        assert np.all(syndromes == expected_syndrome), (
            f"T4 failed for Y_{q}: expected {expected_syndrome}, "
            f"got unique {np.unique(syndromes).tolist()}")
        cx, cz = code.corrections[expected_syndrome]
        expected_cx = np.zeros(7, dtype=int); expected_cx[q] = 1
        expected_cz = np.zeros(7, dtype=int); expected_cz[q] = 1
        assert np.array_equal(cx, expected_cx) and np.array_equal(cz, expected_cz), (
            f"T4 lookup failed for Y_{q}: got x={cx.tolist()}, z={cz.tolist()}")
    checks['T4_single_Y_correctly_decoded_for_each_qubit'] = 7

    if verbose:
        print("steane_ec_circuit._selftest passed:")
        for k, v in checks.items():
            print(f"  {k:55s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
