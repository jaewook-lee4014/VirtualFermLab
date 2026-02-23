"""pH-dependent growth modulation.

Cardinal pH model (Rosso *et al.*, 1995) and lag-phase switch from the
MPR pH experiment notebook.
"""

from __future__ import annotations

import numpy as np


def cardinal_pH_factor(
    pH: float,
    pH_min: float,
    pH_opt: float,
    pH_max: float,
) -> float:
    """Rosso (1995) Cardinal pH factor.

    .. math::

        f(\\text{pH}) = \\frac{(\\text{pH}-\\text{pH}_{\\min})(\\text{pH}-\\text{pH}_{\\max})}
        {(\\text{pH}-\\text{pH}_{\\min})(\\text{pH}-\\text{pH}_{\\max})
         - (\\text{pH}-\\text{pH}_{\\text{opt}})^2}

    Returns 0 outside (pH_min, pH_max) and 1.0 at pH_opt.

    Parameters
    ----------
    pH : float
        Current pH value.
    pH_min, pH_opt, pH_max : float
        Cardinal pH values defining the growth response.

    Returns
    -------
    float
        Growth modulation factor in [0, 1].
    """
    if pH <= pH_min or pH >= pH_max:
        return 0.0

    num = (pH - pH_min) * (pH - pH_max)
    denom = num - (pH - pH_opt) ** 2

    if denom == 0.0:
        return 0.0

    f = num / denom
    return max(0.0, min(f, 1.0))


def lag_switch(t: float, lag: float) -> float:
    """Binary lag-phase switch.

    Returns 0.0 when ``t < lag`` and 1.0 otherwise.

    Parameters
    ----------
    t : float
        Current time (h).
    lag : float
        Duration of lag phase (h).

    Returns
    -------
    float
        0.0 or 1.0.
    """
    return 0.0 if t < lag else 1.0
