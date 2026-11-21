"""Off-target scoring (MIT/Hsu-2013 style position-weighted score)."""
import numpy as np

# Mismatch penalty weights for positions 1..20 (5' -> 3', PAM-proximal = 20).
_M = np.array([0, 0, 0.014, 0, 0, 0.395, 0.317, 0, 0.389, 0.079,
               0.445, 0.508, 0.613, 0.851, 0.732, 0.828, 0.615, 0.804,
               0.685, 0.583])


def off_target_score(mismatch_positions, n_mismatch: int) -> float:
    """Return 0-100; higher = more likely to be cut (worse off-target)."""
    if n_mismatch == 0:
        return 100.0
    s1 = 1.0
    for p in mismatch_positions:
        s1 *= 1.0 - _M[p]
    if n_mismatch > 1:
        pos = sorted(mismatch_positions)
        d = (pos[-1] - pos[0]) / (n_mismatch - 1)
        s2 = 1.0 / (((19 - d) / 19) * 4 + 1)
    else:
        s2 = 1.0
    s3 = 1.0 / (n_mismatch ** 2)
    return round(100.0 * s1 * s2 * s3, 3)
