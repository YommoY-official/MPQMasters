"""
[[7,1,3]] Steane code as a StabilizerCode instance.

Generators (standard Hamming form):
    g1 = X X X X I I I
    g2 = X X I I X X I
    g3 = X I X I X I X
    g4 = Z Z Z Z I I I
    g5 = Z Z I I Z Z I
    g6 = Z I Z I Z I Z

Logicals (transversal representatives):
    X̄ = X⊗7
    Z̄ = Z⊗7

Syndrome bit ordering follows the generator order above (MSB = g1, LSB = g6),
matching `StabilizerCode.compute_syndrome`'s native bit packing.

Bits 5..3 = X-type generator outcomes (detect Z errors)
Bits 2..0 = Z-type generator outcomes (detect X errors)
"""

import os, sys
import numpy as np

# Make sibling ECCODE modules importable when this file is imported as
# `nonmarkov_steane.steane_code`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from StabilizerCode import StabilizerCode  # noqa: E402


def build_steane_code() -> StabilizerCode:
    """Return a StabilizerCode instance for the [[7,1,3]] Steane code."""
    generators = [
        # X-type
        (np.array([1, 1, 1, 1, 0, 0, 0]), np.zeros(7, dtype=int)),  # g1: XXXXIII
        (np.array([1, 1, 0, 0, 1, 1, 0]), np.zeros(7, dtype=int)),  # g2: XXIIXXI
        (np.array([1, 0, 1, 0, 1, 0, 1]), np.zeros(7, dtype=int)),  # g3: XIXIXIX
        # Z-type
        (np.zeros(7, dtype=int), np.array([1, 1, 1, 1, 0, 0, 0])),  # g4: ZZZZIII
        (np.zeros(7, dtype=int), np.array([1, 1, 0, 0, 1, 1, 0])),  # g5: ZZIIZZI
        (np.zeros(7, dtype=int), np.array([1, 0, 1, 0, 1, 0, 1])),  # g6: ZIZIZIZ
    ]
    logical_x = (np.ones(7, dtype=int), np.zeros(7, dtype=int))   # XXXXXXX
    logical_z = (np.zeros(7, dtype=int), np.ones(7, dtype=int))   # ZZZZZZZ
    return StabilizerCode(
        n=7,
        generators=generators,
        logical_x=logical_x,
        logical_z=logical_z,
        name='[[7,1,3]]',
    )


def _symplectic_inner(a, b):
    """Symplectic inner product of two Paulis (xa,za) and (xb,zb), mod 2."""
    xa, za = a
    xb, zb = b
    return int((np.dot(xa, zb) + np.dot(za, xb)) % 2)


def _selftest(verbose: bool = True) -> dict:
    """
    Sanity checks for the Steane code instance.
    Returns a dict of checks-passed counters; raises AssertionError on failure.
    """
    code = build_steane_code()
    checks = {}

    # 1. All generator pairs commute.
    n_pairs = 0
    for i in range(code.r):
        for j in range(i + 1, code.r):
            assert _symplectic_inner(code.generators[i], code.generators[j]) == 0, (
                f"generators {i} and {j} anticommute")
            n_pairs += 1
    checks['generator_pairs_commute'] = n_pairs

    # 2. Logicals commute with all generators.
    for i, g in enumerate(code.generators):
        assert _symplectic_inner(g, code.logical_x) == 0, f"X̄ anticommutes with g{i}"
        assert _symplectic_inner(g, code.logical_z) == 0, f"Z̄ anticommutes with g{i}"
    checks['logicals_commute_with_generators'] = code.r

    # 3. X̄ and Z̄ anticommute.
    assert _symplectic_inner(code.logical_x, code.logical_z) == 1, "X̄ commutes with Z̄"
    checks['logicals_anticommute'] = 1

    # 4. Corrections cover all 64 syndromes.
    assert len(code.corrections) == 64, (
        f"correction table has {len(code.corrections)} entries, expected 64")
    checks['n_syndromes'] = len(code.corrections)

    # 5. Every weight-1 single-qubit Pauli (21 of them) has a weight-≤1 correction.
    n_weight1 = 0
    for q in range(code.n):
        for (xb, zb) in [(1, 0), (1, 1), (0, 1)]:  # X, Y, Z on qubit q
            xv = np.zeros(code.n, dtype=int); xv[q] = xb
            zv = np.zeros(code.n, dtype=int); zv[q] = zb
            s = code.compute_syndrome(xv, zv)
            cx, cz = code.corrections[s]
            w = int(np.sum(cx | cz))
            assert w <= 1, (
                f"weight-1 Pauli on qubit {q} (x={xb},z={zb}) has correction "
                f"of weight {w}, syndrome={s}")
            n_weight1 += 1
    checks['weight1_paulis_have_weight_le_1_correction'] = n_weight1
    assert n_weight1 == 21

    # 6. corrections[0] = identity (zero syndrome -> no correction).
    cx0, cz0 = code.corrections[0]
    assert int(np.sum(cx0 | cz0)) == 0, "corrections[0] is not the identity"
    checks['identity_for_zero_syndrome'] = 1

    if verbose:
        print(f"steane_code._selftest passed: {code}")
        for k, v in checks.items():
            print(f"  {k:46s} = {v}")
    return checks


if __name__ == "__main__":
    _selftest()
