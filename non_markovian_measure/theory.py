"""
Closed-form (Mathematica-derived) logical-channel references for cross-checking.

Each function returns the logical stochastic matrix for the 3-bit repetition
code at time step T = 3, as an analytic function of the noise rate q and the
input logical-|1> weight a.  Derived symbolically in Mathematica; used to
validate LogicalChannel.  For other time steps the formulas (and exact
constants) would have to be re-derived.
"""

import numpy as np


def theory_result_perfect_syndrome_t3(q, a):
    """T = 3, perfect syndrome, iid physical bit-flip -- Mathematica result."""
    element = (a * (1 - 6 * q**2 + 4 * q**3)**3
               + q**2 * (9 - 6 * q - 54 * q**2 + 72 * q**3 + 84 * q**4
                         - 216 * q**5 + 144 * q**6 - 32 * q**7))
    return np.array([[element, 0], [0, 1 - element]])


def theory_result_perfect_physical_t3(q, a):
    """T = 3, perfect physical, iid syndrome bit-flip -- Mathematica result."""
    element = (a + 4 * q**2 - 8 * a * q**2 - 4 * q**3 + 8 * a * q**3
               - 8 * q**4 + 16 * a * q**4 + 16 * q**5 - 32 * a * q**5
               - 8 * q**6 + 16 * a * q**6)
    return np.array([[element, 0], [0, 1 - element]])
