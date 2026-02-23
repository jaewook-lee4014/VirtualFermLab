"""Enzyme regulation functions for dual-substrate fermentation.

Extracted from ``plantODE()`` branches in the ESCAPE25 notebook.
Three enzyme induction strategies are supported:

* **Direct inhibition** (``enzyme_induction=False``) — competitive inhibition
  of substrate 2 by substrate 1.
* **Enzyme induction** (``enzyme_induction=True``) — an explicit enzyme pool
  *Z* mediates substrate 2 consumption.
* **Kompala cybernetic** (``enzyme_induction='Kompala'``) — cybernetic matching
  / proportional allocation laws with two enzyme pools *Z1*, *Z2*.
"""

from __future__ import annotations

import numpy as np


# ---- Direct inhibition ----

def direct_inhibition(S_inhibitor: float, K_I: float) -> float:
    """Competitive inhibition factor: 1 / (1 + S_inhibitor / K_I).

    Parameters
    ----------
    S_inhibitor : float
        Concentration of the inhibiting substrate (g/L).
    K_I : float
        Inhibition constant (g/L).

    Returns
    -------
    float
        Inhibition factor in (0, 1].
    """
    return 1.0 / (1.0 + S_inhibitor / K_I)


# ---- Enzyme induction ----

def enzyme_induction_factor(Z: float, K_Z_S: float) -> float:
    """Enzyme-mediated factor: Z / (K_Z_S + Z).

    Parameters
    ----------
    Z : float
        Enzyme concentration (g/L or units).
    K_Z_S : float
        Half-saturation constant for enzyme action.

    Returns
    -------
    float
        Factor in [0, 1).
    """
    return Z / (K_Z_S + Z)


def enzyme_production_rate(
    total_growth_rate: float,
    X: float,
    S2: float,
    Ks2: float,
    S1: float,
    K_Z_S: float,
    K_Z_c: float,
) -> float:
    """Rate of enzyme production for the enzyme-induction model.

    dZ/dt (production term) = K_Z_c * total_gr * X * S2/(Ks2+S2) * 1/(1+S1*K_Z_S)

    Parameters
    ----------
    total_growth_rate : float
        Sum of growth rates on each substrate.
    X : float
        Biomass concentration (g/L).
    S2 : float
        Substrate 2 concentration (g/L).
    Ks2 : float
        Half-saturation for substrate 2 (g/L).
    S1 : float
        Substrate 1 concentration (g/L).
    K_Z_S : float
        Enzyme inhibition constant related to S1.
    K_Z_c : float
        Enzyme production rate constant.

    Returns
    -------
    float
        dZ/dt production term.
    """
    return K_Z_c * total_growth_rate * X * S2 / (Ks2 + S2) * 1.0 / (1.0 + S1 * K_Z_S)


# ---- Kompala cybernetic model ----

def kompala_matching_law(gr1: float, gr2: float) -> tuple[float, float]:
    """Cybernetic matching law: v_i = gr_i / max(gr_i).

    Parameters
    ----------
    gr1, gr2 : float
        Individual growth rates.

    Returns
    -------
    tuple[float, float]
        (v1, v2) matching variables.
    """
    gr_max = max(gr1, gr2)
    if gr_max == 0:
        return 0.0, 0.0
    return gr1 / gr_max, gr2 / gr_max


def kompala_proportional_law(gr1: float, gr2: float) -> tuple[float, float]:
    """Cybernetic proportional law: u_i = gr_i / (gr1 + gr2).

    Parameters
    ----------
    gr1, gr2 : float
        Individual growth rates.

    Returns
    -------
    tuple[float, float]
        (u1, u2) proportional allocation variables.
    """
    total = gr1 + gr2
    if total == 0:
        return 0.0, 0.0
    return gr1 / total, gr2 / total
