"""
Brute-force fault-path enumeration -> logical stochastic matrix A[i, j].

Independent reimplementation of LogicalChannel's model (rebuilds decode/label
from `code` primitives, does NOT reuse channel internals) so it can verify the
channel.  Per time step there are  n data-error + m=(n-k) syndrome-error
locations => (2n-k)*T fault locations in the whole circuit.  We sum over every
fault path with total weight <= max_faults, weighting by its probability.

Dynamics matching LogicalChannel: at step t, s_t = H d_t (+) e_t (+) e_{t-1},
the net X applied is  d_t (+) decoder[s_t], and the whole circuit applies X^F
with F = XOR_t (d_t (+) decoder[s_t]).  Logical input j (codeword c_j) reads
out as logical label l(c_j (+) F).

NOTE on normalization.  Each path carries its full i.i.d. probability
q^w (1-q)^(L-w).  Summing ALL weights gives columns = 1 (a proper channel,
equals LogicalChannel).  Truncating at max_faults keeps only a subset, so the
raw columns sum to P(w <= max_faults) < 1 -- the dropped >=(max_faults+1)-fault
mass.  normalize=True divides that out, giving the channel CONDITIONED on
w <= max_faults (columns = 1).
"""

import itertools

import numpy as np

from error_models import iid_bitflip_error, iid_syndrome


def brute_logical_channel(code, T, q, max_faults=2, normalize=True,
                          syndrome=None, p_error=iid_bitflip_error):
    """Logical column-stochastic matrix from fault paths with total weight <= max_faults.

    max_faults=None sums ALL weights (exact; matches LogicalChannel).
    normalize=True renormalizes a truncation to columns = 1 (condition on <= max_faults).
    """
    n, m, k = code.n, code.m, code.k
    if syndrome is None:
        syndrome = iid_syndrome(code, q)
    memory   = syndrome.memory
    zero_syn = tuple([0] * m)

    # independent decode/label tables
    codespace = sorted(y for y in range(code.dim)
                       if code.syndrome(code.to_bits(y)) == zero_syn)
    log_idx   = {y: l for l, y in enumerate(codespace)}
    def frombits(bits):                      # inverse of code.to_bits (MSB first)
        v = 0
        for i, b in enumerate(bits):
            v |= int(b) << (n - 1 - i)
        return v
    labels = [log_idx[x ^ frombits(code.decoder[code.syndrome(code.to_bits(x))])]
              for x in range(code.dim)]

    dimL = 2 ** k
    A    = np.zeros((dimL, dimL))
    # fault locations: n data bits + m syndrome bits, per step
    locs = ([('D', t, b) for t in range(T) for b in range(n)]
            + [('E', t, b) for t in range(T) for b in range(m)])
    L    = len(locs)                         # = (2n-k)*T

    def contribution(active):
        D = [[0] * n for _ in range(T)]
        E = [[0] * m for _ in range(T)]
        for kind, t, b in active:
            (D if kind == 'D' else E)[t][b] = 1
        D = [tuple(x) for x in D]
        E = [tuple(x) for x in E]
        prob, e_prev, F, hist = 1.0, zero_syn, 0, [zero_syn] * memory
        for t in range(T):
            d, e = D[t], E[t]
            prob *= p_error(d, q)                                  # physical error
            prob *= syndrome.prob(e, tuple(hist[:memory]))        # syndrome error
            if prob == 0.0:
                return None
            s = tuple(code.syndrome(d)[i] ^ e[i] ^ e_prev[i] for i in range(m))
            F ^= frombits(d) ^ frombits(code.decoder[s])
            e_prev, hist = e, [e] + hist
        return prob, F

    weights = range(L + 1) if max_faults is None else range(min(max_faults, L) + 1)
    for w in weights:
        for active in itertools.combinations(locs, w):
            res = contribution(active)
            if res is None:
                continue
            prob, F = res
            for j in range(dimL):
                A[labels[codespace[j] ^ F], j] += prob

    if normalize:                            # condition on w <= max_faults => columns = 1
        col = A.sum(axis=0, keepdims=True)
        A = np.divide(A, col, out=np.zeros_like(A), where=col > 0)
    return A
