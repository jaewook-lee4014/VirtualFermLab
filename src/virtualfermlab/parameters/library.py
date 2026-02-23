"""YAML-based parameter library.

Load, save and list :class:`StrainProfile` objects backed by YAML files in
``parameters/defaults/``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from virtualfermlab.parameters.schema import StrainProfile

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def load_strain_profile(name_or_path: str | Path) -> StrainProfile:
    """Load a strain profile from a YAML file or SQLite cache.

    Parameters
    ----------
    name_or_path : str or Path
        Either a strain name (looked up in ``defaults/``) or a full path.

    Returns
    -------
    StrainProfile

    Raises
    ------
    FileNotFoundError
        If the strain is not found in YAML files or the SQLite cache.
    """
    p = Path(name_or_path)
    if not p.suffix:
        p = _DEFAULTS_DIR / f"{name_or_path}.yaml"
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f)
        return StrainProfile(**data)

    # Try SQLite cache
    try:
        from virtualfermlab.discovery.db import load_strain_profile_cache

        cached = load_strain_profile_cache(str(name_or_path))
        if cached is not None:
            return cached
    except Exception:
        pass

    raise FileNotFoundError(f"Strain profile not found: {name_or_path}")


def save_strain_profile(profile: StrainProfile, path: str | Path) -> None:
    """Save a strain profile to a YAML file.

    Parameters
    ----------
    profile : StrainProfile
    path : str or Path
    """
    with open(path, "w") as f:
        yaml.dump(profile.model_dump(), f, default_flow_style=False, sort_keys=False)


def list_available_strains() -> list[str]:
    """Return names of strain profiles in the defaults directory and SQLite cache."""
    yaml_strains = [p.stem for p in _DEFAULTS_DIR.glob("*.yaml")] if _DEFAULTS_DIR.exists() else []
    try:
        from virtualfermlab.discovery.db import list_cached_strains

        db_strains = list_cached_strains()
    except Exception:
        db_strains = []
    return sorted(set(yaml_strains + db_strains))
