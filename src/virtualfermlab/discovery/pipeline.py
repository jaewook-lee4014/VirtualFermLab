"""Discovery pipeline orchestrator.

Runs the five-stage automatic workflow:
1. Paper search  2. LLM extraction  3. DB storage  4. Taxonomy match
5. Build StrainProfile
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import queue
import threading

from virtualfermlab.discovery import db
from virtualfermlab.discovery.llm_extraction import LLMClient, extract_from_queue
from virtualfermlab.discovery.paper_search import _SENTINEL, search_papers_into_queue
from virtualfermlab.discovery.taxonomy import find_most_similar
from virtualfermlab.parameters.library import list_available_strains, load_strain_profile
from virtualfermlab.parameters.schema import (
    CardinalPHParams,
    DistributionSpec,
    InhibitionParams,
    StrainProfile,
    SubstrateParams,
)

logger = logging.getLogger(__name__)

# Biological defaults used when no better information is available.
_DEFAULT_MU_MAX = 0.1
_DEFAULT_KS = 0.2
_DEFAULT_YXS = 0.3


@dataclass
class DiscoveryResult:
    """Outcome of a discovery pipeline run."""

    strain_name: str
    profile: StrainProfile | None = None
    source: str = "default_fallback"
    papers_found: int = 0
    params_extracted: int = 0
    similar_strain: str | None = None
    similarity_score: float | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)


def _record_stage(
    result: DiscoveryResult,
    stage: str,
    status: str,
    message: str,
    duration_ms: float,
) -> None:
    result.stages.append({
        "stage": stage,
        "status": status,
        "message": message,
        "duration_ms": round(duration_ms, 1),
    })


def _notify(
    progress_cb: Callable | None,
    stage: str,
    status: str,
    message: str = "",
) -> None:
    if progress_cb is not None:
        progress_cb({"stage": stage, "status": status, "message": message})


# ------------------------------------------------------------------
# Profile builder
# ------------------------------------------------------------------


def _make_dist(
    value: float,
    confidence: str = "C",
    source: str = "",
) -> DistributionSpec:
    """Create a fixed-value distribution spec."""
    return DistributionSpec(
        type="fixed", value=value, confidence=confidence, source=source
    )


def _build_profile(
    strain_name: str,
    extracted_params: list[dict],
    similar_strain_name: str | None,
    similar_profile: StrainProfile | None,
) -> StrainProfile:
    """Assemble a :class:`StrainProfile` using a priority cascade.

    Priority: extracted literature values > similar strain values >
    biological defaults.
    """
    # Index extracted params by (parameter_name, substrate)
    ext_index: dict[tuple[str, str | None], dict] = {}
    for p in extracted_params:
        key = (p.get("name", p.get("parameter_name", "")), p.get("substrate"))
        if key not in ext_index:
            ext_index[key] = p

    def _get_value(
        param_name: str,
        substrate: str | None,
        default: float,
        sim_value: float | None = None,
    ) -> DistributionSpec:
        """Resolve a single parameter value through the cascade."""
        # 1. Extracted from literature
        ext = ext_index.get((param_name, substrate))
        if ext is not None:
            return _make_dist(
                float(ext["value"]), confidence="B", source="literature"
            )
        # 2. Similar strain
        if sim_value is not None:
            return _make_dist(sim_value, confidence="C", source=f"taxonomy match: {similar_strain_name}")
        # 3. Biological default
        return _make_dist(default, confidence="C", source="biological default")

    # Determine substrates: from extracted params, similar profile, or default glucose/xylose
    substrate_names: list[str] = []
    for key in ext_index:
        if key[1] and key[1] not in substrate_names:
            substrate_names.append(key[1])

    if not substrate_names and similar_profile is not None:
        substrate_names = list(similar_profile.substrates.keys())

    if not substrate_names:
        substrate_names = ["glucose", "xylose"]

    # Build SubstrateParams
    substrates: dict[str, SubstrateParams] = {}
    for sub_name in substrate_names:
        sim_sub = None
        if similar_profile is not None:
            sim_sub = similar_profile.substrates.get(sub_name)

        substrates[sub_name] = SubstrateParams(
            name=sub_name,
            mu_max=_get_value(
                "mu_max", sub_name, _DEFAULT_MU_MAX,
                sim_sub.mu_max.value if sim_sub else None,
            ),
            Ks=_get_value(
                "Ks", sub_name, _DEFAULT_KS,
                sim_sub.Ks.value if sim_sub else None,
            ),
            Yxs=_get_value(
                "Yxs", sub_name, _DEFAULT_YXS,
                sim_sub.Yxs.value if sim_sub else None,
            ),
        )

    # Cardinal pH
    cardinal_pH = None
    ph_min_ext = ext_index.get(("pH_min", None))
    ph_opt_ext = ext_index.get(("pH_opt", None))
    ph_max_ext = ext_index.get(("pH_max", None))

    if ph_min_ext or ph_opt_ext or ph_max_ext:
        sim_ph = similar_profile.cardinal_pH if similar_profile else None
        cardinal_pH = CardinalPHParams(
            pH_min=_make_dist(
                float(ph_min_ext["value"]) if ph_min_ext else
                (sim_ph.pH_min.value if sim_ph else 3.5),
                confidence="B" if ph_min_ext else "C",
            ),
            pH_opt=_make_dist(
                float(ph_opt_ext["value"]) if ph_opt_ext else
                (sim_ph.pH_opt.value if sim_ph else 6.0),
                confidence="B" if ph_opt_ext else "C",
            ),
            pH_max=_make_dist(
                float(ph_max_ext["value"]) if ph_max_ext else
                (sim_ph.pH_max.value if sim_ph else 7.5),
                confidence="B" if ph_max_ext else "C",
            ),
        )
    elif similar_profile is not None and similar_profile.cardinal_pH is not None:
        cardinal_pH = similar_profile.cardinal_pH

    # Inhibitions — copy from similar strain
    inhibitions: list[InhibitionParams] = []
    if similar_profile is not None:
        inhibitions = similar_profile.inhibitions

    # Enzyme params — copy from similar strain if available
    enzyme_params = None
    if similar_profile is not None:
        enzyme_params = similar_profile.enzyme_params

    return StrainProfile(
        name=strain_name,
        cardinal_pH=cardinal_pH,
        substrates=substrates,
        inhibitions=inhibitions,
        enzyme_params=enzyme_params,
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------


def run_discovery(
    strain_name: str,
    progress_cb: Callable | None = None,
) -> DiscoveryResult:
    """Execute the full five-stage discovery pipeline.

    Each stage can fail independently; the pipeline degrades gracefully
    and always returns *some* usable profile.
    """
    result = DiscoveryResult(strain_name=strain_name)

    # --- Stage 1+2: Producer-consumer (parallel search + LLM extraction) ---
    _notify(progress_cb, "search", "running")
    _notify(progress_cb, "extract", "running")
    t0 = time.monotonic()
    paper_queue: queue.Queue = queue.Queue()
    client = LLMClient()
    extracted_params: list[dict] = []
    consumer_result: list[list[dict]] = [[]]  # mutable container for thread result

    def _consumer() -> None:
        consumer_result[0] = extract_from_queue(paper_queue, _SENTINEL, client)

    consumer_thread = threading.Thread(target=_consumer, daemon=True)
    consumer_thread.start()

    try:
        papers_pushed = search_papers_into_queue(strain_name, paper_queue)
        result.papers_found = papers_pushed
        _record_stage(
            result, "search", "completed",
            f"Found {papers_pushed} papers",
            (time.monotonic() - t0) * 1000,
        )
    except Exception as exc:
        logger.exception("Paper search failed")
        _record_stage(result, "search", "failed", str(exc), (time.monotonic() - t0) * 1000)
    finally:
        # Ensure sentinel is sent even if producer fails, so consumer exits
        try:
            paper_queue.put(_SENTINEL)
        except Exception:
            pass

    _notify(progress_cb, "search", "completed", f"{result.papers_found} papers")

    consumer_thread.join()
    extracted_params = consumer_result[0]
    result.params_extracted = len(extracted_params)

    t_extract_end = time.monotonic()
    if extracted_params:
        _record_stage(
            result, "extract", "completed",
            f"Extracted {len(extracted_params)} params",
            (t_extract_end - t0) * 1000,
        )
    else:
        _record_stage(
            result, "extract", "completed",
            "No params extracted",
            (t_extract_end - t0) * 1000,
        )
    _notify(progress_cb, "extract", "completed", f"{len(extracted_params)} params")

    # --- Stage 3: Storage (already done as side-effects) ---------------
    _notify(progress_cb, "store", "running")
    t0 = time.monotonic()
    _record_stage(result, "store", "completed", "Data persisted in stages 1-2", (time.monotonic() - t0) * 1000)
    _notify(progress_cb, "store", "completed")

    # --- Stage 4: Taxonomy match --------------------------------------
    _notify(progress_cb, "taxonomy", "running")
    t0 = time.monotonic()
    similar_strain_name: str | None = None
    similar_profile: StrainProfile | None = None
    try:
        known = list_available_strains()
        similar_strain_name, score = find_most_similar(strain_name, known)
        result.similar_strain = similar_strain_name
        result.similarity_score = score
        if similar_strain_name:
            similar_profile = load_strain_profile(similar_strain_name)
        msg = f"Best match: {similar_strain_name} (sim={score:.2f})" if similar_strain_name else "No match"
        _record_stage(result, "taxonomy", "completed", msg, (time.monotonic() - t0) * 1000)
    except Exception as exc:
        logger.exception("Taxonomy match failed")
        _record_stage(result, "taxonomy", "failed", str(exc), (time.monotonic() - t0) * 1000)
    _notify(progress_cb, "taxonomy", "completed")

    # --- Stage 5: Build profile ----------------------------------------
    _notify(progress_cb, "build_profile", "running")
    t0 = time.monotonic()
    try:
        profile = _build_profile(strain_name, extracted_params, similar_strain_name, similar_profile)
        result.profile = profile

        # Determine source label
        if extracted_params:
            result.source = "literature"
        elif similar_strain_name:
            result.source = "taxonomy_match"
        else:
            result.source = "default_fallback"

        # Cache the built profile
        db.save_strain_profile_cache(
            strain_name, profile, result.source,
            similar_strain_name, result.similarity_score,
        )
        _record_stage(result, "build_profile", "completed", f"Profile built ({result.source})", (time.monotonic() - t0) * 1000)
    except Exception as exc:
        logger.exception("Profile build failed")
        _record_stage(result, "build_profile", "failed", str(exc), (time.monotonic() - t0) * 1000)
        # Last-resort fallback: minimal default profile
        result.profile = StrainProfile(
            name=strain_name,
            substrates={
                "glucose": SubstrateParams(
                    name="glucose",
                    mu_max=_make_dist(_DEFAULT_MU_MAX),
                    Ks=_make_dist(_DEFAULT_KS),
                    Yxs=_make_dist(_DEFAULT_YXS),
                ),
                "xylose": SubstrateParams(
                    name="xylose",
                    mu_max=_make_dist(_DEFAULT_MU_MAX * 0.5),
                    Ks=_make_dist(_DEFAULT_KS),
                    Yxs=_make_dist(_DEFAULT_YXS),
                ),
            },
        )
        result.source = "default_fallback"
    _notify(progress_cb, "build_profile", "completed")

    return result
