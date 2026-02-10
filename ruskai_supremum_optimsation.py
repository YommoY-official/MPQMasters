import torch
import math
import numpy as np
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Pauli matrices as TORCH tensors (complex)
# ----------------------------
I2 = torch.eye(2, dtype=torch.complex128, device=device)
X  = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128, device=device)
Y  = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=device)
Z  = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128, device=device)


# ----------------------------
# Channel: local Pauli noise (I, X, Y) on each qubit
# Φ(ρ) = Σ_{i,j} (K_i ⊗ K_j) ρ (K_i ⊗ K_j)†
# ----------------------------
def channel(rho, px, py):
    # px, py are python floats or torch scalars
    px_t = torch.as_tensor(px, dtype=torch.float64, device=device)
    py_t = torch.as_tensor(py, dtype=torch.float64, device=device)

    # ensure valid probs (for safety)
    p0 = torch.clamp(1.0 - px_t - py_t, min=0.0)

    kraus = [
        torch.sqrt(p0).to(torch.complex128) * I2,
        torch.sqrt(px_t).to(torch.complex128) * X,
        torch.sqrt(py_t).to(torch.complex128) * Y,
    ]

    out = torch.zeros_like(rho)
    for K1 in kraus:
        for K2 in kraus:
            op = torch.kron(K1, K2)  # 4x4
            out = out + op @ rho @ op.conj().T
    return out


def trace_norm_hermitian(H):
    # For Hermitian H: ||H||_1 = sum |eigs|
    evals = torch.linalg.eigvalsh(H)
    return torch.sum(torch.abs(evals))


def density_matrix(psi):
    return psi[:, None] @ psi.conj()[None, :]

def softplus_unit_sphere(u, eps=1e-12):
    """
    u: real tensor of shape (4,)
    returns: tensor (4,) with sum of squares = 1 and all entries > 0
    """
    v = F.softplus(u)                  # v_i > 0
    norm = torch.sqrt(torch.sum(v*v) + eps)
    return v / norm

# ----------------------------
# Your objective, but made safe & differentiable
# We optimize over an unconstrained raw vector in R^11
# and map it to your param format:
# [a1,a2,theta2,a3,theta3,b1,phi1,b2,phi2,b3,phi3]
# ----------------------------



def objective_from_raw(raw, px=0.3, py=0.2, eps=1e-12):
    """
    raw: torch.float64 tensor shape (13,) unconstrained
    returns: scalar real tensor to maximize
    """

    # amplitudes in (0,1)
    # indices: a0=11,a1=0,a2=1,a3=3,b0=12,b1=5,b2=7,b3=9
    sig = torch.sigmoid


    # angles in [0, 2π)
    twopi = 2.0 * math.pi
    theta2 = twopi * sig(raw[2])
    theta3 = twopi * sig(raw[4])

    phi1   = twopi * sig(raw[6])
    phi2   = twopi * sig(raw[8])
    phi3   = twopi * sig(raw[10])

    # a0, b0 with clamping to avoid sqrt of negative
    # ---- amplitudes for psi ----
    u_psi = raw[[11,0,1,3]]  # pick 4 real params
    a0, a1, a2, a3 = softplus_unit_sphere(u_psi)

    # ---- amplitudes for phi ----
    u_phi = raw[[12,5,7,9]]  # pick 4 real params
    b0, b1, b2, b3 = softplus_unit_sphere(u_phi)
    # compute theta1 from your constraint (with eps to avoid division by 0)
    denom = a1 * b1
    denom = torch.clamp(denom, min=eps)

    # complex expression inside log
    # NOTE: keep everything as complex here
    num = (
        a0 * b0
        + a2 * b2 * torch.exp(1j * (theta2 - phi2))
        + a3 * b3 * torch.exp(1j * (theta3 - phi3))
    )
    z = -(num / denom)  # complex

    theta1 = phi1 + torch.atan( torch.imag(z)  / torch.real(z) ) # complex

    # build psi, phi in C^4
    psi = torch.zeros(4, dtype=torch.complex128, device=device)
    psi[0] = a0.to(torch.complex128)
    psi[1] = a1.to(torch.complex128) * torch.exp(1j * theta1)
    psi[2] = a2.to(torch.complex128) * torch.exp(1j * theta2)
    psi[3] = a3.to(torch.complex128) * torch.exp(1j * theta3)

    phi = torch.zeros(4, dtype=torch.complex128, device=device)
    phi[0] = b0.to(torch.complex128)
    phi[1] = b1.to(torch.complex128) * torch.exp(1j * phi1)
    phi[2] = b2.to(torch.complex128) * torch.exp(1j * phi2)
    phi[3] = b3.to(torch.complex128) * torch.exp(1j * phi3)

    delta = density_matrix(psi) - density_matrix(phi)  # 4x4 Hermitian
    H = channel(delta, px=px, py=py)




    return 0.5 * trace_norm_hermitian(H).real


# ----------------------------
# Optimizer runner (maximization via minimizing -objective)
# ----------------------------
def maximize(px=0.3, py=0.2, steps=4000, lr=2e-2, restarts=10, seed=0, print_every=400):
    torch.manual_seed(seed)

    best_val = -float("inf")
    best_raw = None

    for r in range(restarts):
        raw = (0.1 * torch.randn(13, device=device)).requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)

        for t in range(steps):
            opt.zero_grad()
            f = objective_from_raw(raw, px=px, py=py)
            loss = -f
            loss.backward()
            opt.step()

            if (t + 1) % print_every == 0:
                print(f"[restart {r+1}/{restarts}] step {t+1:5d} | f = {f.item():.12f}")

        final_f = objective_from_raw(raw, px=px, py=py).detach().item()
        if final_f > best_val:
            best_val = final_f
            best_raw = raw.detach().clone()

    return best_raw, best_val


if __name__ == "__main__":

    #add for differnte grids of px and py, and then compare with 1-2px 1-2py and 1-2px-2py
    best_raw, best_val = maximize(px=0.1, py=0.5, steps=2000, lr=2e-2, restarts=5, seed=5)
    print("\nBest value found:", best_val)
    print("Best raw parameters:", best_raw.cpu().numpy())
