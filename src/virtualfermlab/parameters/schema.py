"""Pydantic models for strain parameters with uncertainty.

Provides typed, validated schemas for parameter distributions, substrates,
inhibition, pH, and complete strain profiles.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class DistributionSpec(BaseModel):
    """Specification for a single parameter value with uncertainty.

    Attributes
    ----------
    type : str
        Distribution type.
    value : float
        Central / nominal value.
    std : float or None
        Standard deviation (for normal / lognormal).
    low, high : float or None
        Bounds (for uniform / triangular).
    confidence : str
        Quality tag — ``"A"`` (same-strain measured), ``"B"`` (literature),
        ``"C"`` (estimated).
    source : str or None
        Citation or data provenance.
    """

    type: Literal["fixed", "normal", "uniform", "lognormal", "triangular"] = "fixed"
    value: float
    std: float | None = None
    low: float | None = None
    high: float | None = None
    confidence: Literal["A", "B", "C"] = "C"
    source: str | None = None


class SubstrateParams(BaseModel):
    """Kinetic parameters for one substrate."""

    name: str
    mu_max: DistributionSpec
    Ks: DistributionSpec
    Yxs: DistributionSpec


class InhibitionParams(BaseModel):
    """Cross-inhibition between two substrates."""

    inhibitor: str
    inhibited: str
    K_I: DistributionSpec
    mechanism: Literal["direct", "enzyme", "kompala"] = "direct"


class CardinalPHParams(BaseModel):
    """Cardinal pH model parameters (Rosso 1995)."""

    pH_min: DistributionSpec
    pH_opt: DistributionSpec
    pH_max: DistributionSpec


class EnzymeParams(BaseModel):
    """Enzyme regulation parameters."""

    K_Z_c: DistributionSpec
    K_Z_S: DistributionSpec
    K_Z_d: DistributionSpec


class StrainProfile(BaseModel):
    """Complete parameter profile for a microbial strain.

    Attributes
    ----------
    name : str
        Strain identifier (e.g. ``"F_venenatum_A35"``).
    cardinal_pH : CardinalPHParams or None
        pH growth model parameters.
    substrates : dict
        Substrate name → :class:`SubstrateParams`.
    inhibitions : list
        Cross-inhibition entries.
    enzyme_params : EnzymeParams or None
        Enzyme regulation parameters.
    """

    name: str
    cardinal_pH: CardinalPHParams | None = None
    substrates: dict[str, SubstrateParams] = {}
    inhibitions: list[InhibitionParams] = []
    enzyme_params: EnzymeParams | None = None
