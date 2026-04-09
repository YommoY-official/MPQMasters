import numpy as np
from itertools import product

# ============================================================
# 1. Pauli matrices and n-qubit tensor products
# ============================================================
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)

def kron_n(*ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# ============================================================
# 2. [[5,1,3]] stabilizer generators and logical operators
# ============================================================
gens = [
    kron_n(X, Z, Z, X, I2),  # M1 = XZZXI
    kron_n(I2, X, Z, Z, X),  # M2 = IXZZX
    kron_n(X, I2, X, Z, Z),  # M3 = XIXZZ
    kron_n(Z, X, I2, X, Z),  # M4 = ZXIXZ
]

Zbar = kron_n(Z, Z, Z, Z, Z)   # logical Z
Xbar = kron_n(X, X, X, X, X)   # logical X

# ============================================================
# 3. Syndrome projectors Π_s (32×32 each)
# ============================================================
def make_projector(s_bits):
    """Π_s = (1/16) Σ_{g in S} (-1)^{s·λ(g)} g"""
    proj = np.zeros((32, 32), dtype=complex)
    for bits in product([0,1], repeat=4):
        g = np.eye(32, dtype=complex)
        phase = 0
        for i, b in enumerate(bits):
            if b:
                g = g @ gens[i]
                phase += s_bits[i]
        proj += ((-1) ** phase) * g
    return proj / 16.0

syndromes = list(product([0,1], repeat=4))   # 16 syndromes
projectors = {s: make_projector(s) for s in syndromes}

# Sanity check
assert np.allclose(sum(projectors.values()), np.eye(32), atol=1e-10)

# ============================================================
# 4. Canonical correction operators Q_s (5-qubit Paulis)
#    For [[5,1,3]] (perfect code): each syndrome s has a
#    unique weight-0 or weight-1 Pauli
# ============================================================
def pauli_syndrome(P):
    """4-bit syndrome of 5-qubit Pauli P w.r.t. generators."""
    return tuple(
        0 if np.allclose(P @ g - g @ P, 0, atol=1e-10) else 1
        for g in gens
    )

canonical_error = {(0,0,0,0): np.eye(32, dtype=complex)}

single_qubit_paulis = [X, Y, Z]
for qubit in range(5):
    for pauli in single_qubit_paulis:
        factors = [I2]*5
        factors[qubit] = pauli
        P = kron_n(*factors)
        s = pauli_syndrome(P)
        if s not in canonical_error:
            canonical_error[s] = P

assert len(canonical_error) == 16, "Perfect code: all 16 syndromes covered"
print("All 16 syndromes covered by weight ≤ 1 Paulis ✓")

# ============================================================
# 5. Codewords |0̄⟩ and |1̄⟩
#    Built from the codespace projector Π_{0000}
# ============================================================
Pi_code = projectors[(0,0,0,0)]

eigvals, eigvecs = np.linalg.eigh(Pi_code)
code_basis = eigvecs[:, np.abs(eigvals - 1.0) < 1e-8]
assert code_basis.shape[1] == 2

# Diagonalize Zbar within codespace to get |0̄⟩, |1̄⟩
Zbar_in_code = code_basis.conj().T @ Zbar @ code_basis
zvals, zvecs = np.linalg.eigh(Zbar_in_code)
# Sort: +1 eigenvalue → |0̄⟩, -1 eigenvalue → |1̄⟩
order = np.argsort(zvals)[::-1]
codewords = code_basis @ zvecs[:, order]
ket0L, ket1L = codewords[:, 0], codewords[:, 1]

# Verify
assert np.allclose(Zbar @ ket0L,  ket0L, atol=1e-10), "|0̄⟩ must be +1 eigenstate of Zbar"
assert np.allclose(Zbar @ ket1L, -ket1L, atol=1e-10), "|1̄⟩ must be -1 eigenstate of Zbar"
print("Codewords |0̄⟩, |1̄⟩ constructed ✓")

# ============================================================
# 6. Build the 32×32 star-decoder unitary U*
#
#    Input space:  32-dim physical space
#    Output space: C^2 (logical) ⊗ C^16 (syndrome)
#                  ordered as |b⟩ ⊗ |s_idx⟩
#                  → row index = 16*b + s_idx
#
#    Action: U* (Q_s |b̄⟩) = |b⟩ ⊗ |s_idx⟩
# ============================================================
U_star = np.zeros((32, 32), dtype=complex)

for s_idx, s in enumerate(syndromes):
    Qs = canonical_error[s]
    for b, ket in enumerate([ket0L, ket1L]):
        input_vec  = Qs @ ket              # 5-qubit physical state
        output_row = 16 * b + s_idx        # row in (logical ⊗ syndrome) space
        # U* maps input_vec → e_{output_row}
        # so the output_row-th row of U* equals input_vec†
        U_star[output_row, :] = input_vec.conj()

# Verify unitarity
assert np.allclose(U_star @ U_star.conj().T, np.eye(32), atol=1e-8), "U* not unitary!"
assert np.allclose(U_star.conj().T @ U_star, np.eye(32), atol=1e-8), "U* not unitary!"
print("U* (star-decoder) is 32×32 unitary ✓")

# Verify action
e32 = np.eye(32)
for s_idx, s in enumerate(syndromes):
    Qs = canonical_error[s]
    for b, ket in enumerate([ket0L, ket1L]):
        out = U_star @ (Qs @ ket)
        expected_row = 16 * b + s_idx
        assert np.allclose(out, e32[expected_row], atol=1e-8), f"Failed s={s}, b={b}"
print("Action U*(Q_s|b̄⟩) = |b⟩⊗|s⟩ verified for all 32 basis states ✓")

# ============================================================
# 7. Abstract FTEC as a superoperator
#
#    FTEC(ρ) = Σ_s Q_s† Π_s ρ Π_s Q_s
#
#    This measures syndrome s, then applies correction Q_s†
#    Output is always in the codespace (syndrome reset to 0)
# ============================================================
def apply_FTEC(rho):
    """Apply abstract FTEC to 32×32 density matrix rho."""
    out = np.zeros((32, 32), dtype=complex)
    for s, Qs in canonical_error.items():
        Pi_s = projectors[s]
        branch = Pi_s @ rho @ Pi_s          # project onto syndrome s
        corrected = Qs.conj().T @ branch @ Qs  # apply correction Q_s†
        out += corrected
    return out

# ============================================================
# 8. Helper: apply U* as a superoperator
#    V(ρ) = U* ρ U*†   (32×32 → 32×32 in logical⊗syndrome space)
# ============================================================
def apply_Ustar(rho):
    return U_star @ rho @ U_star.conj().T

def apply_Ustar_dag(rho):
    return U_star.conj().T @ rho @ U_star

# ============================================================
# 9. Define Zbar (Z⊗Z⊗Z⊗Z⊗Z) as a channel
# ============================================================
def apply_Zbar(rho):
    return Zbar @ rho @ Zbar

# ============================================================
# 10. The full circuit as a superoperator:
#
#     U* ∘ FTEC ∘ Zbar ∘ FTEC ∘ U*†
#
#     Input:  32×32 density matrix in (logical ⊗ syndrome) space
#     Output: 32×32 density matrix in (logical ⊗ syndrome) space
# ============================================================
def full_circuit(rho_logical_syndrome):
    rho = apply_Ustar_dag(rho_logical_syndrome)   # encode with syndrome info
    rho = apply_FTEC(rho)                          # EC round 1
    rho = apply_Zbar(rho)                          # transversal Z gate
    rho = apply_FTEC(rho)                          # EC round 2
    rho = apply_Ustar(rho)                         # decode
    return rho

# ============================================================
# 11. Build the full superoperator as a 1024×1024 matrix
#     by applying it to all 32×32 basis density matrices
#
#     This lets us inspect the Choi matrix and compare to
#     the expected answer: Z(·)Z ⊗ |0⟩⟨0|
# ============================================================
dim = 32
super_matrix = np.zeros((dim**2, dim**2), dtype=complex)

for i in range(dim):
    for j in range(dim):
        basis_rho = np.zeros((dim, dim), dtype=complex)
        basis_rho[i, j] = 1.0
        out = full_circuit(basis_rho)
        # out[k,l] = super_matrix[k*dim+l, i*dim+j]
        super_matrix[:, i*dim + j] = out.reshape(-1)

print("\nSuperoperator built (1024×1024) ✓")

# ============================================================
# 12. Build the EXPECTED superoperator:
#
#     Expected: logical Z gate ⊗ syndrome reset to |0⟩⟨0|
#
#     In our output space (C^2 logical ⊗ C^16 syndrome):
#     row/col index = 16*b + s_idx
#
#     Expected action on input (b_in, s_in) → (b_out, s_out):
#       logical:  Z|b_in⟩ = (-1)^b_in |b_in⟩
#       syndrome: any s_in → s_out = 0 (index 0)
#
#     So: ρ_out[16*b + 0, 16*b' + 0]
#           = (-1)^b (-1)^b' * (sum over s_in,s_in' of ρ_in[16*b+s, 16*b'+s'])
#               ... but FTEC also decoheres syndromes, so off-diag in syndrome
#               are killed. Let's compute it properly from the logical channel.
# ============================================================

# Build expected superoperator from scratch
Z_logical = np.array([[1,0],[0,-1]], dtype=complex)  # logical Z in C^2
proj_s0 = np.zeros((16,16), dtype=complex)
proj_s0[0,0] = 1.0   # |0⟩⟨0| in syndrome space (s_idx=0 is syndrome (0,0,0,0))

# Expected output space: C^2 ⊗ C^16
# with index ordering 16*b + s_idx
# Apply Z on logical, project syndrome to |0⟩⟨0|
expected_super = np.zeros((dim**2, dim**2), dtype=complex)

for i in range(dim):
    for j in range(dim):
        # Decompose input indices into (b_in, s_in) and (b_in', s_in')
        b_in  = i // 16;  s_in  = i % 16
        b_in2 = j // 16;  s_in2 = j % 16

        # Z acts on logical: Z|b⟩ = (-1)^b |b⟩
        z_phase = (-1)**b_in * (-1)**b_in2

        # Output: logical unchanged (b_in → b_in), syndrome → 0
        out_i = 16 * b_in  + 0   # s_idx = 0
        out_j = 16 * b_in2 + 0   # s_idx = 0

        expected_super[out_i * dim + out_j,  # wrong indexing — fix below
                       i * dim + j] = z_phase

# Rewrite expected_super with correct (row = out_i*dim+out_j) indexing
expected_super2 = np.zeros((dim**2, dim**2), dtype=complex)
for i in range(dim):
    for j in range(dim):
        b_in  = i // 16;  s_in  = i % 16
        b_in2 = j // 16;  s_in2 = j % 16
        z_phase = (-1)**b_in * (-1)**b_in2
        out_i = 16 * b_in  + 0
        out_j = 16 * b_in2 + 0
        expected_super2[out_i * dim + out_j, i * dim + j] += z_phase

# ============================================================
# 13. Compare actual vs expected
# ============================================================
diff = np.max(np.abs(super_matrix - expected_super2))
print(f"\nMax deviation from expected (Z ⊗ |0⟩⟨0|) channel: {diff:.6e}")

if diff < 1e-6:
    print("✓ CIRCUIT CORRECT: U* FTEC Zbar FTEC U*† = Z_logical ⊗ reset_syndrome")
else:
    print("✗ Mismatch — investigating...")
    # Check a few specific input/output pairs manually
    for b in [0, 1]:
        ket = ket0L if b == 0 else ket1L
        rho_in_logical = np.outer(np.eye(32)[16*b], np.eye(32)[16*b])
        rho_out = full_circuit(rho_in_logical)
        # Check: syndrome should be in s_idx=0 sector only
        print(f"  b={b}: output diagonal = {np.real(np.diag(rho_out))[:4]}...")

# ============================================================
# 14. Bonus: test on a superposition input
#     Input: |+⟩_L ⊗ |s=0⟩  →  expected output: |+⟩_L with Z phase ⊗ |s=0⟩
#     Z|+⟩ = |−⟩, so input |+⟩_L should become |−⟩_L
# ============================================================
print("\n--- Superposition test ---")
# |+⟩_L in logical⊗syndrome space: row 16*0+0 and 16*1+0 with equal amplitude
ket_plus_in = np.zeros(32, dtype=complex)
ket_plus_in[16*0 + 0] = 1/np.sqrt(2)   # |0̄⟩ ⊗ |s=0⟩
ket_plus_in[16*1 + 0] = 1/np.sqrt(2)   # |1̄⟩ ⊗ |s=0⟩
rho_plus = np.outer(ket_plus_in, ket_plus_in.conj())

rho_out_plus = full_circuit(rho_plus)

# Expected: Z|+⟩ = |−⟩, so off-diagonal should flip sign
ket_minus_expected = np.zeros(32, dtype=complex)
ket_minus_expected[16*0 + 0] =  1/np.sqrt(2)
ket_minus_expected[16*1 + 0] = -1/np.sqrt(2)
rho_minus_expected = np.outer(ket_minus_expected, ket_minus_expected.conj())

diff_super = np.max(np.abs(rho_out_plus - rho_minus_expected))
print(f"Z|+⟩_L → |−⟩_L test, max deviation: {diff_super:.6e}")
if diff_super < 1e-6:
    print("✓ Correct: logical Z maps |+⟩_L to |−⟩_L")