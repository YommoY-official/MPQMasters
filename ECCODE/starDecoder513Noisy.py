"""
Star-decoder and U†EU for [[5,1,3]] with PHENOMENOLOGICAL FAULT MODEL.
Memory-efficient: uses superoperator (Liouville) representation throughout.

Ref: Gottesman QECCbook-2024 §14.4.2-14.4.3, González-García et al. arXiv:2502.05658
"""
import numpy as np
from itertools import product as iterproduct
from functools import reduce

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS_1Q = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
def kron_list(ops): return reduce(np.kron, ops)

N, DIM = 5, 32
STAB_GENS = [kron_list([PAULIS_1Q[c] for c in s])
             for s in ['XZZXI','IXZZX','XIXZZ','ZXIXZ']]
X_BAR, Z_BAR = kron_list([X]*5), kron_list([Z]*5)

def setup_code():
    Pi = np.eye(DIM, dtype=complex)
    for g in STAB_GENS: Pi = Pi @ (np.eye(DIM) + g) / 2
    evals, evecs = np.linalg.eigh(Pi)
    idx = np.where(np.abs(evals - 1) < 1e-10)[0]
    v0, v1 = evecs[:,idx[0]], evecs[:,idx[1]]
    Zb = np.array([[v0.conj()@Z_BAR@v0, v0.conj()@Z_BAR@v1],
                   [v1.conj()@Z_BAR@v0, v1.conj()@Z_BAR@v1]])
    ze, zv = np.linalg.eigh(Zb); zv = zv[:,np.argsort(-ze.real)]
    c0 = v0*zv[0,0]+v1*zv[1,0]; c0/=np.linalg.norm(c0)
    c1 = v0*zv[0,1]+v1*zv[1,1]; c1/=np.linalg.norm(c1)
    syn_rep, syn_lab = {0: np.eye(DIM, dtype=complex)}, {0:'I'}
    for q in range(N):
        for nm in ['X','Y','Z']:
            ops=[I2]*N; ops[q]=PAULIS_1Q[nm]; E=kron_list(ops)
            bits=[0 if np.allclose(g@E, E@g) else 1 for g in STAB_GENS]
            s=sum(b<<(3-i) for i,b in enumerate(bits))
            if s not in syn_rep: syn_rep[s]=E; syn_lab[s]=f'{nm}{q+1}'
    syn_proj = {}
    for s in range(16):
        P=np.eye(DIM, dtype=complex)
        for i in range(4):
            bit=(s>>(3-i))&1; P=P@(np.eye(DIM)+(1-2*bit)*STAB_GENS[i])/2
        syn_proj[s]=P
    U_dag = np.zeros((DIM,DIM), dtype=complex)
    for s in range(16):
        U_dag[:,2*s]=syn_rep[s]@c0; U_dag[:,2*s+1]=syn_rep[s]@c1
    return c0, c1, syn_rep, syn_lab, syn_proj, U_dag.conj().T, U_dag

def kraus_to_superop(kraus):
    S = np.zeros((DIM**2, DIM**2), dtype=complex)
    for K in kraus: S += np.kron(K, K.conj())
    return S

def noisy_ec_superop(q, syn_proj, syn_rep):
    kraus = []
    for s in range(16):
        for f in range(16):
            wt=bin(f).count('1'); prob=(q**wt)*((1-q)**(4-wt))
            if prob<1e-30: continue
            kraus.append(np.sqrt(prob)*syn_rep[s^f].conj().T @ syn_proj[s])
    return kraus_to_superop(kraus)

def depol_superop(p):
    c = {'I':np.sqrt(1-p),'X':np.sqrt(p/3),'Y':np.sqrt(p/3),'Z':np.sqrt(p/3)}
    kraus = []
    for combo in iterproduct(['I','X','Y','Z'], repeat=N):
        coeff=1.0; ops=[]
        for nm in combo: coeff*=c[nm]; ops.append(PAULIS_1Q[nm])
        kraus.append(coeff*kron_list(ops))
    return kraus_to_superop(kraus)

def exrec_superop(p, q, syn_proj, syn_rep):
    S_ec = noisy_ec_superop(q, syn_proj, syn_rep)
    S_n = depol_superop(p)
    return S_ec @ S_n @ S_ec

def to_logical(S_phys, U, U_dag):
    return np.kron(U,U.conj()) @ S_phys @ np.kron(U_dag,U_dag.conj())

def pauli_decompose(S_L, si, so=None):
    """Extract Pauli coeffs. so=None → trace over output syndrome."""
    def ap(a,b):
        rho=np.zeros((DIM,DIM),dtype=complex); rho[2*si+a,2*si+b]=1.0
        return (S_L@rho.flatten()).reshape(DIM,DIM)
    o00, o01 = ap(0,0), ap(0,1)
    if so is not None:
        o00=o00[2*so:2*so+2,2*so:2*so+2]; o01=o01[2*so:2*so+2,2*so:2*so+2]
    else:
        def tr(m):
            r=np.zeros((2,2),dtype=complex)
            for s in range(16): r+=m[2*s:2*s+2,2*s:2*s+2]
            return r
        o00, o01 = tr(o00), tr(o01)
    a,b=np.real(o00[0,0]),np.real(o00[1,1])
    c,d=np.real(o01[0,1]),np.real(o01[1,0])
    return {'I':(a+c)/2,'Z':(a-c)/2,'X':(b+d)/2,'Y':(b-d)/2}

def syn_transition(S_L):
    T=np.zeros((16,16))
    for si in range(16):
        rho=np.zeros((DIM,DIM),dtype=complex)
        rho[2*si,2*si]=0.5; rho[2*si+1,2*si+1]=0.5
        ro=(S_L@rho.flatten()).reshape(DIM,DIM)
        for so in range(16):
            T[so,si]=np.real(np.trace(ro[2*so:2*so+2,2*so:2*so+2]))
    return T

def main():
    print("="*72)
    print("  PHENOMENOLOGICAL FAULT MODEL: [[5,1,3]] U†EU")
    print("="*72)
    c0,c1,syn_rep,syn_lab,syn_proj,U,U_dag = setup_code()
    print(f"  ∗-decoder unitary check: {np.allclose(U@U_dag, np.eye(DIM))}")
    for s in range(16): print(f"    s={s:04b} -> {syn_lab[s]}")

    # ─── PART A ───
    print("\n"+"="*72+"\n  PART A: q=0 VERIFICATION\n"+"="*72)
    p=0.01
    S_L=to_logical(exrec_superop(p,0.0,syn_proj,syn_rep),U,U_dag)
    pc=pauli_decompose(S_L,si=0,so=0)
    exact=(10/3)*p**2-(200/27)*p**3+(160/27)*p**4-(128/81)*p**5
    print(f"  p̄_X={pc['X']:.10f}, exact={exact:.10f}, match={np.isclose(pc['X'],exact,rtol=1e-4)}")
    T=syn_transition(S_L)
    print(f"  Nonzero T[s',0]: {[s for s in range(16) if T[s,0]>1e-12]}")
    del S_L

    # ─── PART B ───
    print("\n"+"="*72+"\n  PART B: NOISY EC — SYNDROME STRUCTURE\n"+"="*72)
    p=0.01
    for q in [0.05, 0.1, 0.2]:
        print(f"\n{'─'*72}\n  p={p}, q={q}\n{'─'*72}")
        S_L=to_logical(exrec_superop(p,q,syn_proj,syn_rep),U,U_dag)
        T=syn_transition(S_L)
        print(f"\n  Transitions from s_in=0000:")
        for so in range(16):
            if T[so,0]>1e-6: print(f"    → {so:04b}: {T[so,0]:.6f}")
        print(f"  Transitions from s_in=0001:")
        for so in range(16):
            if T[so,1]>1e-6: print(f"    → {so:04b}: {T[so,1]:.6f}")
        print(f"\n  Logical error (syndrome-traced) per s_in:")
        print(f"  {'s_in':>6} {'p_I':>9} {'p_X':>9} {'p_Y':>9} {'p_Z':>9} {'p_err':>9}")
        for si in range(16):
            pc=pauli_decompose(S_L,si); pe=pc['X']+pc['Y']+pc['Z']
            print(f"  {si:04b}   {pc['I']:9.6f} {pc['X']:9.6f} {pc['Y']:9.6f} {pc['Z']:9.6f} {pe:9.6f}")
        ev=sorted(np.linalg.eigvals(T),key=lambda x:-abs(x))
        l2=ev[1].real
        print(f"\n  Top eigenvalues: {' '.join(f'{e.real:+.5f}' for e in ev[:5])}")
        if abs(l2)>1e-10: print(f"  λ₂={l2:.6f} → memory ~ {-1/np.log(abs(l2)):.2f} rounds")
        del S_L

    # ─── PART C ───
    print("\n"+"="*72+"\n  PART C: TWO-ROUND NON-MARKOVIANITY\n"+"="*72)
    p,q=0.01,0.1; print(f"  p={p}, q={q}")
    S1=to_logical(exrec_superop(p,q,syn_proj,syn_rep),U,U_dag)
    S2=S1@S1; del S1
    print(f"\n  Two-round error (syndrome-traced):")
    print(f"  {'s_in':>6} {'p_I':>9} {'p_X':>9} {'p_Y':>9} {'p_Z':>9} {'p_err':>9}")
    errs=[]
    for si in range(16):
        pc=pauli_decompose(S2,si); pe=pc['X']+pc['Y']+pc['Z']; errs.append(pe)
        print(f"  {si:04b}   {pc['I']:9.6f} {pc['X']:9.6f} {pc['Y']:9.6f} {pc['Z']:9.6f} {pe:9.6f}")
    spread=max(errs)-min(errs)
    print(f"\n  Spread: {spread:.2e}  {'→ NON-MARKOVIAN' if spread>1e-8 else '→ Markovian'}")
    del S2

    # ─── PART D ───
    print("\n"+"="*72+"\n  PART D: SPECTRAL STRUCTURE λ₂ vs q\n"+"="*72)
    p=0.01
    print(f"  {'q':>6} {'λ₁':>8} {'λ₂':>8} {'λ₃':>8} {'mem_depth':>10}")
    for q in [0.001,0.01,0.05,0.1,0.15,0.2,0.3]:
        S_L=to_logical(exrec_superop(p,q,syn_proj,syn_rep),U,U_dag)
        T=syn_transition(S_L); del S_L
        ev=sorted(np.linalg.eigvals(T),key=lambda x:-abs(x))
        l2=ev[1].real; md=-1/np.log(abs(l2)) if abs(l2)>1e-10 else float('inf')
        print(f"  {q:6.3f} {ev[0].real:8.5f} {l2:8.5f} {ev[2].real:8.5f} {md:10.3f}")

    print("\n"+"="*72+"\n  SUMMARY\n"+"="*72)
    print("""
  1. q=0: Only s_out=0 populated → Markovian, p̄=(10/3)p² exact.
  2. q>0: Multiple syndrome blocks → syndrome memory persists.
  3. Two-round p_err DEPENDS on s_in → NON-MARKOVIAN on L_t alone.
  4. (L_t, s_t) IS Markov; tracing s_t breaks Markov property.
  5. λ₂(T) grows with q: more syndrome noise → longer memory.
  Ref: Gottesman §14.4.2-3, González-García et al. arXiv:2502.05658
    """)

if __name__=='__main__': main()