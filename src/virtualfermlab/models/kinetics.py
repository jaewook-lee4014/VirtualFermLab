"""Substrate kinetic factor functions.

Extracted from ``plantODE()`` in the ESCAPE25 notebook.
"""

from __future__ import annotations

import numpy as np


def monod_factor(S: float, Ks: float) -> float:
    """Monod substrate limitation: S / (S + Ks).

    Parameters
    ----------
    S : float
        Substrate concentration (g/L).
    Ks : float
        Half-saturation constant (g/L).

    Returns
    -------
    float
        Value in [0, 1].
    """
    return S / (S + Ks)


def contois_factor(S: float, Ks: float, X: float) -> float:
    """Contois substrate limitation: S / (S + Ks * X).

    Parameters
    ----------
    S : float
        Substrate concentration (g/L).
    Ks : float
        Half-saturation constant (g/L).
    X : float
        Biomass concentration (g/L).

    Returns
    -------
    float
        Value in [0, 1].
    """
    return S / (S + Ks * X)


def substrate_factor(
    S: float,
    Ks: float,
    growth_model: str,
    X: float = 1.0,
) -> float:
    """Dispatch to Monod or Contois substrate factor.

    Parameters
    ----------
    S : float
        Substrate concentration (g/L).
    Ks : float
        Half-saturation constant (g/L).
    growth_model : str
        ``"Monod"`` or ``"Contois"``.
    X : float
        Biomass concentration (g/L). Only used for Contois.

    Returns
    -------
    float
        Substrate limitation factor.
    """
    if growth_model == "Monod":
        return monod_factor(S, Ks)
    elif growth_model == "Contois":
        return contois_factor(S, Ks, X)
    else:
        raise ValueError(f"Unknown growth model: {growth_model!r}. Use 'Monod' or 'Contois'.")
