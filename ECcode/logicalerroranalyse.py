"""
[[5,1,3]] Logical Error Calculator via U†EU (star-decoder formalism)
=====================================================================
Implements Gottesman QECCbook-2024 §14.4.2–14.4.3.

For every physical Pauli E on 5 qubits and every incoming syndrome s_in,
computes:
  - the logical Pauli L ∈ {I, X, Y, Z}  (the decoded logical error)
  - the outgoing syndrome s_out           (passed to the next exRec)

This is the core quantity needed for the syndrome-conditioned logical error
table: the non-Markovian structure arises because s_out of one exRec becomes
s_in of the next.

Usage
-----
  python five_qubit_logical_errors.py

All functions are individually importable for use in larger simulations.
"""

import numpy as np
from itertools import product

# =============================================================================
# SECTION 1: Pauli algebra over F_2 (binary symplectic representation)
# =============================================================================
#
# A single-qubit Pauli (ignoring phase) is represented as (x, z) ∈ F_2².
#   x=0,z=0  →  I
#   x=1,z=0  →  X
#   x=1,z=1  →  Y  (= iXZ, but phase doesn't affect syndromes or cosets)
#   x=0,z=1  →  Z
#
# An n-qubit Pauli is represented as two binary vectors:
#   xvec ∈ F_2^n  (X part)
#   zvec ∈ F_2^n  (Z part)
#
# Pauli multiplication (mod phase):
#   (x1,z1)·(x2,z2) = (x1⊕x2, z1⊕z2)   (XOR)
#
# Commutation: P and Q anticommute iff  xP·zQ + xQ·zP = 1  (mod 2)
#   (dot products over F_2)

PAULIS_1Q = {
    'I': (np.array([0]), np.array([0])),
    'X': (np.array([1]), np.array([0])),
    'Y': (np.array([1]), np.array([1])),
    'Z': (np.array([0]), np.array([1])),
}

PAULI_LABEL = {(0,0):'I', (1,0):'X', (1,1):'Y', (0,1):'Z'}

def pauli_multiply(xv1, zv1, xv2, zv2):
    """Multiply two n-qubit Paulis (ignoring phase). Returns (xv, zv)."""
    return (xv1 ^ xv2), (zv1 ^ zv2)

def pauli_anticommutes(xv1, zv1, xv2, zv2):
    """Returns True iff the two Paulis anticommute."""
    return bool((np.dot(xv1, zv2) + np.dot(zv1, xv2)) % 2)

def pauli_weight(xv, zv):
    """Number of qubits where Pauli acts non-trivially."""
    return int(np.sum((xv | zv)))

def all_n_qubit_paulis(n):
    """
    Enumerate all 4^n n-qubit Paulis as list of (xvec, zvec) arrays.
    Order: iterate over all (x0,z0,x1,z1,...) in lex order.
    """
    paulis = []
    for bits in product([0,1], repeat=2*n):
        xv = np.array(bits[0::2], dtype=int)
        zv = np.array(bits[1::2], dtype=int)
        paulis.append((xv, zv))
    return paulis

# =============================================================================
# SECTION 2: [[5,1,3]] stabilizer code definition
# =============================================================================
#
# Stabilizer generators (from Gottesman Table 3.2 / Eq. 3.11-3.12):
#   g1 = X Z Z X I
#   g2 = I X Z Z X
#   g3 = X I X Z Z
#   g4 = Z X I X Z
#
# Logical operators:
#   X_L = X X X X X
#   Z_L = Z Z Z Z Z
#
# These are the standard cyclic generators. Note g5 = g1·g2·g3·g4 = Z Z X I X
# is also in the stabilizer but redundant.

N = 5  # number of physical qubits

# Stabilizer generators as (xvec, zvec)
STAB_GENS = [
    (np.array([1,0,0,1,0]), np.array([0,1,1,0,0])),  # g1: XZZXI
    (np.array([0,1,0,0,1]), np.array([0,0,1,1,0])),  # g2: IXZZX
    (np.array([1,0,1,0,0]), np.array([0,0,0,1,1])),  # g3: XIXZZ
    (np.array([0,1,0,1,0]), np.array([1,0,0,0,1])),  # g4: ZXIXZ
]

# Logical operators (minimum weight representatives; both have weight 5)
XLOG = (np.array([1,1,1,1,1]), np.array([0,0,0,0,0]))  # XXXXX
ZLOG = (np.array([0,0,0,0,0]), np.array([1,1,1,1,1]))  # ZZZZZ

# =============================================================================
# SECTION 3: Syndrome computation
# =============================================================================
#
# The syndrome of a Pauli E is the 4-bit string:
#   s_i = 1  iff  E anticommutes with generator g_i
#         0  iff  E commutes with generator g_i
#
# Equivalently, s_i = (xE · zg_i + zE · xg_i) mod 2.
# The syndrome is an integer 0..15 (4-bit, MSB = s1).

def compute_syndrome(xv, zv):
    """
    Compute the 4-bit syndrome of Pauli (xv, zv) w.r.t. [[5,1,3]] generators.
    Returns integer in 0..15.
    """
    s = 0
    for i, (gx, gz) in enumerate(STAB_GENS):
        bit = (np.dot(xv, gz) + np.dot(zv, gx)) % 2
        s |= (int(bit) << (3 - i))  # MSB = generator 0
    return s

def syndrome_to_bits(s):
    """Convert syndrome integer 0..15 to 4-bit array [s0,s1,s2,s3]."""
    return np.array([(s >> (3-i)) & 1 for i in range(4)], dtype=int)

# Verify: all stabilizer generators should have syndrome 0
for i, (gx, gz) in enumerate(STAB_GENS):
    assert compute_syndrome(gx, gz) == 0, f"Generator {i} has nonzero syndrome!"

# =============================================================================
# SECTION 4: Build the canonical correction table
# =============================================================================
#
# For each syndrome s ∈ {0..15}, we want the minimum-weight Pauli Q_s such
# that syndrome(Q_s) = s. This is the correction the decoder applies.
#
# Algorithm:
#   1. Enumerate all 4^5 = 1024 Paulis (ignoring phase).
#   2. Group them by syndrome.
#   3. For each syndrome, pick the minimum-weight representative.
#
# Syndrome 0 has 64 elements (the stabilizer group S ∪ logical coset partners).
# Each of the 15 non-zero syndromes also has 64 elements.
# The minimum-weight element of each syndrome-0 coset of N(S) is the correction.
#
# For [[5,1,3]] with independent depolarizing noise, the minimum-weight
# correction is unique for syndromes arising from weight-1 errors (15 syndromes)
# and the weight-0 error (syndrome 0 → correction I).

def build_correction_table():
    """
    Returns: dict {syndrome_int: (xvec, zvec)} mapping each syndrome to its
    minimum-weight canonical correction Pauli.
    """
    # For each syndrome, track the best (lowest weight) Pauli seen so far
    best = {}  # syndrome -> (weight, xvec, zvec)

    for xv, zv in all_n_qubit_paulis(N):
        s = compute_syndrome(xv, zv)
        w = pauli_weight(xv, zv)
        if s not in best or w < best[s][0]:
            best[s] = (w, xv.copy(), zv.copy())

    return {s: (xv, zv) for s, (w, xv, zv) in best.items()}

CORRECTIONS = build_correction_table()

# Sanity check: correction for syndrome 0 should be I (weight 0)
cx, cz = CORRECTIONS[0]
assert pauli_weight(cx, cz) == 0, "Correction for syndrome 0 is not I!"

# =============================================================================
# SECTION 5: Logical class of a Pauli
# =============================================================================
#
# Given a Pauli E ∈ N(S) (i.e., commuting with all stabilizers, equivalently
# syndrome(E) = 0 after correction), determine which coset of S it belongs to:
#
#   E ∈ S        → logical I    (acts trivially on encoded state)
#   E ∈ X_L · S  → logical X    (flips logical qubit)
#   E ∈ Z_L · S  → logical Z    (phases logical qubit)
#   E ∈ Y_L · S  → logical Y    (= X_L · Z_L up to phase)
#
# To determine this, check whether E commutes or anticommutes with Z_L and X_L:
#   anticommutes with Z_L → has logical X component
#   anticommutes with X_L → has logical Z component
#
# This gives the 2-bit logical Pauli (x_L, z_L).

def logical_class(xv, zv):
    """
    Determine the logical Pauli class of E = (xv, zv), assuming E ∈ N(S)
    (i.e., syndrome is 0, or we have already applied the correction).

    Returns: integer 0=I, 1=X, 2=Y, 3=Z
    and the 2-bit string (x_logical, z_logical).
    """
    # x_logical = anticommutes with Z_L?
    xl_x, xl_z = XLOG
    zl_x, zl_z = ZLOG
    x_log = int(pauli_anticommutes(xv, zv, *ZLOG))
    z_log = int(pauli_anticommutes(xv, zv, *XLOG))
    label = (x_log, z_log)
    idx = {(0,0):0, (1,0):1, (1,1):2, (0,1):3}[label]
    return idx, label

# Verify logical operators themselves:
assert logical_class(*XLOG) == (1, (1,0)), "X_L should be logical X"
assert logical_class(*ZLOG) == (3, (0,1)), "Z_L should be logical Z"
# I should be logical I:
assert logical_class(np.zeros(5,int), np.zeros(5,int)) == (0, (0,0)), "I should be logical I"

# =============================================================================
# SECTION 6: The core U†EU computation
# =============================================================================
#
# For a physical Pauli E and incoming syndrome s_in, compute:
#
#   Step 1: The incoming state has error Q_{s_in} on it (the canonical error
#           for syndrome s_in). The ∗-encoder prepared a state with exactly
#           that error pattern.
#
#   Step 2: The exRec applies E on top. The combined physical error seen by
#           the trailing EC is:
#               E_combined = E · Q_{s_in}    (Pauli multiplication mod phase)
#
#   Step 3: The trailing EC measures the syndrome of E_combined:
#               s_out = syndrome(E_combined)
#
#   Step 4: The trailing EC applies correction Q_{s_out}, leaving:
#               E_residual = Q_{s_out}† · E_combined = Q_{s_out} · E · Q_{s_in}
#           (Paulis are self-inverse, so Q† = Q)
#
#   Step 5: E_residual ∈ N(S) (it commutes with all stabilizers because
#           its syndrome is 0). Its logical class is the logical error L.
#
# The output is (L, s_out) for this (E, s_in) pair.
#
# Why does s_in matter? Because E_combined = E · Q_{s_in}, and Q_{s_in} can
# change the syndrome: syndrome(E · Q_{s_in}) ≠ syndrome(E) in general.
# This is exactly the syndrome-dependence that produces non-Markovian behavior.

def u_dagger_E_u(xv_E, zv_E, s_in):
    """
    Compute the logical error and output syndrome for physical Pauli E
    acting on a state with incoming syndrome s_in.

    Parameters
    ----------
    xv_E, zv_E : np.ndarray of shape (5,), dtype int
        Binary symplectic representation of E.
    s_in : int  (0..15)
        Incoming syndrome (syndrome of the pre-existing error on the data).

    Returns
    -------
    logical_idx : int  (0=I, 1=X, 2=Y, 3=Z)
    s_out : int  (0..15)
        Syndrome of E_combined; becomes s_in for the next exRec.
    """
    # Step 1: get canonical correction for incoming syndrome
    Qx, Qz = CORRECTIONS[s_in]

    # Step 2: E_combined = E · Q_{s_in}  (mod phase; XOR in F_2)
    xc, zc = pauli_multiply(xv_E, zv_E, Qx, Qz)

    # Step 3: s_out = syndrome of E_combined
    s_out = compute_syndrome(xc, zc)

    # Step 4: apply correction Q_{s_out} to get residual
    Rx, Rz = CORRECTIONS[s_out]
    xr, zr = pauli_multiply(Rx, Rz, xc, zc)

    # Step 5: logical class of residual (which is in N(S) by construction)
    logical_idx, _ = logical_class(xr, zr)

    return logical_idx, s_out

# =============================================================================
# SECTION 7: Build the full logical error table
# =============================================================================
#
# For every physical Pauli E (all 4^5 = 1024 of them) and every incoming
# syndrome s_in ∈ {0..15}, compute (L, s_out).
#
# Table shape: [1024 Paulis] × [16 syndromes] → (logical_error, s_out)
#
# This is the complete syndrome-conditioned logical channel.

def build_logical_error_table():
    """
    Build the full (E, s_in) → (L, s_out) table for all Paulis and syndromes.

    Returns
    -------
    all_paulis : list of (xvec, zvec), length 1024
    logical_table : np.ndarray of shape (1024, 16), dtype int
        logical_table[e_idx, s_in] = logical error index (0=I,1=X,2=Y,3=Z)
    syndrome_table : np.ndarray of shape (1024, 16), dtype int
        syndrome_table[e_idx, s_in] = s_out (0..15)
    """
    all_paulis = all_n_qubit_paulis(N)  # length 4^5 = 1024
    n_paulis = len(all_paulis)
    logical_table  = np.zeros((n_paulis, 16), dtype=int)
    syndrome_table = np.zeros((n_paulis, 16), dtype=int)

    for e_idx, (xv, zv) in enumerate(all_paulis):
        for s_in in range(16):
            L, s_out = u_dagger_E_u(xv, zv, s_in)
            logical_table[e_idx, s_in]  = L
            syndrome_table[e_idx, s_in] = s_out

    return all_paulis, logical_table, syndrome_table

# =============================================================================
# SECTION 8: Pauli channel → logical channel (summing over all Paulis)
# =============================================================================
#
# Under depolarizing noise with single-qubit error probability p, the
# probability of a specific weight-w Pauli P is:
#
#   Prob(P) = (p/3)^w · (1-p)^(5-w)   for each of 3^w non-trivial Paulis of weight w
#
# This assigns a probability to every element of all_n_qubit_paulis(N).
#
# The syndrome-conditioned logical channel is then:
#
#   Λ(L, s_out | s_in) = Σ_E  Prob(E) · δ(L = L(E,s_in)) · δ(s_out = s_out(E,s_in))
#
# This is a 4×16×16 tensor (L, s_out, s_in) or equivalently a 64×64 transfer
# matrix if you index (L, s_in) jointly.

def depolarizing_prob(xv, zv, p):
    """Probability of Pauli (xv,zv) under independent depolarizing channel."""
    w = pauli_weight(xv, zv)
    return (p/3)**w * (1-p)**(N-w)

def build_channel_tensor(p, logical_table, syndrome_table, all_paulis):
    """
    Build the syndrome-conditioned logical channel tensor.

    channel[L, s_out, s_in] = Prob( logical error L, output syndrome s_out
                                    | incoming syndrome s_in )

    Parameters
    ----------
    p : float  — single-qubit depolarizing error rate
    logical_table, syndrome_table, all_paulis : from build_logical_error_table()

    Returns
    -------
    channel : np.ndarray of shape (4, 16, 16)
    """
    channel = np.zeros((4, 16, 16))

    for e_idx, (xv, zv) in enumerate(all_paulis):
        prob = depolarizing_prob(xv, zv, p)
        for s_in in range(16):
            L     = logical_table[e_idx, s_in]
            s_out = syndrome_table[e_idx, s_in]
            channel[L, s_out, s_in] += prob

    return channel

# =============================================================================
# SECTION 9: Analysis utilities
# =============================================================================

LOGICAL_NAMES = ['I', 'X', 'Y', 'Z']

def print_syndrome_conditioned_errors(logical_table, syndrome_table,
                                      all_paulis, max_weight=2):
    """
    Print the logical error table for all Paulis up to a given weight,
    grouped by (E, s_in) → (L, s_out).
    """
    print(f"\n{'='*70}")
    print(f"  Logical errors for weight ≤ {max_weight} Paulis, all incoming syndromes")
    print(f"{'='*70}")
    print(f"  {'Pauli E':15s}  {'wt':>3}  {'s_in':>5}  {'L':>4}  {'s_out':>6}")
    print(f"  {'-'*50}")

    for e_idx, (xv, zv) in enumerate(all_paulis):
        w = pauli_weight(xv, zv)
        if w > max_weight:
            continue

        # Build Pauli label string
        label = ''.join(PAULI_LABEL[(int(xv[i]), int(zv[i]))] for i in range(N))

        for s_in in range(16):
            L     = logical_table[e_idx, s_in]
            s_out = syndrome_table[e_idx, s_in]
            # Only show non-trivial cases (L != I or s_out != s_in) to reduce noise
            if L != 0 or s_out != s_in:
                print(f"  {label:15s}  {w:3d}  {s_in:5d}  "
                      f"{LOGICAL_NAMES[L]:>4}  {s_out:6d}")

def channel_summary(channel, p):
    """
    Print key statistics of the logical channel:
    - Total logical error probability (marginalized over syndromes)
    - Per-Pauli logical error rates
    - How much the logical error depends on s_in (non-Markovianity indicator)
    """
    print(f"\n{'='*60}")
    print(f"  Logical channel summary  (p = {p:.4f})")
    print(f"{'='*60}")

    # Marginal over syndromes: assume uniform incoming syndrome
    # (approximation; true distribution depends on the noise model)
    for s_in in range(16):
        total_by_L = channel[:, :, s_in].sum(axis=1)  # shape (4,)
        # Normalization check
        assert abs(total_by_L.sum() - 1.0) < 1e-10, \
            f"Channel not normalized for s_in={s_in}: sum={total_by_L.sum()}"

    # Syndrome-averaged logical error rates
    avg = channel.mean(axis=2)  # average over s_in uniformly → shape (4,16)
    avg_L = avg.sum(axis=1)     # marginal over s_out → shape (4,)
    print(f"\n  Syndrome-averaged logical error rates (uniform s_in prior):")
    for i, name in enumerate(LOGICAL_NAMES):
        print(f"    P(logical {name}) = {avg_L[i]:.6f}")

    # Non-Markovianity indicator: variation of logical error rate with s_in
    # If the channel is Markovian (no syndrome memory), this would be zero.
    per_sin = channel.sum(axis=1)  # shape (4, 16) → per_sin[L, s_in]
    p_err_per_sin = 1.0 - per_sin[0, :]  # P(non-trivial logical error | s_in)
    print(f"\n  P(logical error | s_in) — syndrome dependence:")
    print(f"    min = {p_err_per_sin.min():.6f},  "
          f"max = {p_err_per_sin.max():.6f},  "
          f"range = {p_err_per_sin.max()-p_err_per_sin.min():.6f}")
    print(f"    (range > 0 confirms syndrome-dependent logical errors)")
    for s_in in range(16):
        print(f"    s_in={s_in:2d}: P(err) = {p_err_per_sin[s_in]:.6f}")

def transfer_matrix(channel):
    """
    Build the 64×64 transfer matrix T[(L,s_out), (I,s_in)] = channel[L,s_out,s_in].
    This is the object whose spectral decomposition gives the non-Markovian
    noise parameters (secular equation, dominant eigenvalue λ_1, etc.).

    The state space is (logical Pauli L, syndrome s) with 4×16 = 64 states.
    For the Markovian sub-process (L_t, s_t), T is a stochastic matrix.

    Returns T as a (64, 64) array indexed as T[L_out*16 + s_out, L_in*16 + s_in].
    (For the physical process, L_in is always I=0 since the ∗-decoder resets
    the logical error; the non-trivial structure is in the s dependence.)
    """
    T = np.zeros((64, 64))
    for L_out in range(4):
        for s_out in range(16):
            for s_in in range(16):
                # Input state is always (I, s_in) from the ∗-decoder perspective
                T[L_out*16 + s_out, s_in] = channel[L_out, s_out, s_in]
    return T

# =============================================================================
# SECTION 10: Worked example — reproduce Gottesman §14.4.3 logic manually
# =============================================================================
#
# Gottesman's example (adapted to [[5,1,3]] instead of [[7,1,3]]):
#
# Two specific incoming errors E1 = X2, E2 = Z5 give different logical errors
# when the exRec has a fixed internal fault path. This section demonstrates
# the syndrome-dependence explicitly for a simple case.

def worked_example():
    print(f"\n{'='*60}")
    print("  Worked example: syndrome-dependent logical error")
    print(f"{'='*60}")
    print("  (Analogous to Gottesman §14.4.3, Fig. 14.6)")
    print()

    # Fix an internal fault producing physical Pauli E_fault = Y1 X3
    # (fault on qubit 1 gives Y error; fault in wait step gives X on qubit 3)
    # This is a weight-2 internal fault making the exRec bad.
    xf = np.array([1,0,1,0,0])   # Y on q1, X on q3
    zf = np.array([1,0,0,0,0])   # (Y = X·Z, so z-part only on q1)
    label_fault = ''.join(PAULI_LABEL[(int(xf[i]),int(zf[i]))] for i in range(N))
    print(f"  Internal fault produces physical error E_fault = {label_fault}")
    print()

    for s_in in [0, 1, 2, 4, 8]:
        # s_in = 0: no pre-existing error
        # s_in > 0: various pre-existing errors
        L, s_out = u_dagger_E_u(xf, zf, s_in)
        Qx, Qz = CORRECTIONS[s_in]
        q_label = ''.join(PAULI_LABEL[(int(Qx[i]),int(Qz[i]))] for i in range(N))
        print(f"  s_in = {s_in:2d}  (canonical error = {q_label})")
        print(f"    → logical error = {LOGICAL_NAMES[L]},  s_out = {s_out}")
    print()
    print("  Note: same internal fault → different logical errors depending on s_in.")
    print("  This is the precise mechanism of non-Markovian logical noise.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("Building [[5,1,3]] logical error table...")
    all_paulis, log_table, syn_table = build_logical_error_table()
    print(f"  Done. {len(all_paulis)} Paulis × 16 syndromes = {len(all_paulis)*16} entries.")

    # Verify: identity Pauli should give logical I and preserve syndrome
    I_idx = 0  # first entry in all_n_qubit_paulis is (0000,0000) = I⊗5
    for s in range(16):
        L, s_out = log_table[I_idx, s], syn_table[I_idx, s]
        assert L == 0,   f"I should give logical I, got {LOGICAL_NAMES[L]} for s_in={s}"
        assert s_out == s, f"I should preserve syndrome, got {s_out} for s_in={s}"
    print("  Verification passed: I⊗5 gives logical I and preserves syndrome.")

    # Worked example
    worked_example()

    # Print table for weight ≤ 1 Paulis (single-qubit errors)
    # For these, syndrome uniquely identifies the error → logical I always
    print_syndrome_conditioned_errors(log_table, syn_table, all_paulis, max_weight=1)

    # Build channel for a specific error rate
    p = 0.01
    print(f"\nBuilding logical channel tensor for p = {p}...")
    channel = build_channel_tensor(p, log_table, syn_table, all_paulis)
    channel_summary(channel, p)

    # Transfer matrix
    T = transfer_matrix(channel)
    eigenvalues = np.linalg.eigvals(T)
    # Sort by magnitude descending
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    print(f"\n  Transfer matrix: top 5 eigenvalues by magnitude:")
    for i in range(5):
        lam = eigenvalues[i]
        print(f"    λ_{i+1} = {lam.real:+.6f} {lam.imag:+.6f}i   |λ| = {abs(lam):.6f}")

    print("\nDone.")