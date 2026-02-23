"""Parameter bounds and counts for calibration/validation.

Ported from ESCAPE25 ``get_bounds()`` and ``get_N_params()``.
"""

from __future__ import annotations


def get_n_params(enzyme_induction: bool | str, fit_type: str) -> int:
    """Return the number of free parameters.

    Parameters
    ----------
    enzyme_induction : bool or str
        ``False``, ``True``, or ``"Kompala"``.
    fit_type : str
        ``"calibration"`` or ``"validation"``.

    Returns
    -------
    int
    """
    if fit_type == "validation":
        if enzyme_induction == "Kompala":
            return 3
        elif enzyme_induction is True:
            return 2
        else:
            return 1
    elif fit_type == "calibration":
        if enzyme_induction == "Kompala":
            return 12
        elif enzyme_induction is True:
            return 12
        else:
            return 8
    raise ValueError(f"Invalid fit_type: {fit_type!r}")


def get_bounds(
    params: dict,
    fit_type: str,
    LB: float = 0.001,
    UB: float = 1.0,
) -> list[tuple[float, float]]:
    """Parameter bounds for ``differential_evolution``.

    Parameters
    ----------
    params : dict
        Must contain ``enzyme_induction``.
    fit_type : str
        ``"calibration"`` or ``"validation"``.
    LB, UB : float
        Lower/upper bounds for all parameters.

    Returns
    -------
    list of (float, float)
    """
    ei = params.get("enzyme_induction", False)
    n = get_n_params(ei, fit_type)
    return [(LB, UB)] * n


def get_param_name_order(params: dict, fit_type: str) -> list[str]:
    """Ordered list of parameter names corresponding to the x-vector.

    Parameters
    ----------
    params : dict
        Must contain ``enzyme_induction``.
    fit_type : str
        ``"calibration"`` or ``"validation"``.

    Returns
    -------
    list of str
    """
    ei = params.get("enzyme_induction", False)

    if fit_type == "calibration":
        if ei is True:
            return ["mu_max1", "mu_max2", "Yx1", "Yx2", "K_s1", "K_s2",
                    "K_Z_c", "K_Z_S", "K_Z_d", "K_I", "X0", "Z0"]
        elif ei == "Kompala":
            return ["mu_max1", "mu_max2", "Yx1", "Yx2", "K_s1", "K_s2",
                    "K_Z_c", "K_Z_S", "K_Z_d", "X0", "Z1", "Z2"]
        else:
            return ["mu_max1", "mu_max2", "Yx1", "Yx2", "K_s1", "K_s2",
                    "K_I", "X0"]
    elif fit_type == "validation":
        if ei is True:
            return ["X0", "Z0"]
        elif ei == "Kompala":
            return ["X0", "Z1", "Z2"]
        else:
            return ["X0"]
    raise ValueError(f"Invalid fit_type: {fit_type!r}")
