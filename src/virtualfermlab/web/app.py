"""Flask web application for VirtualFermLab.

Provides an interactive UI for configuring and running fermentation
simulations, Monte Carlo analysis, and virtual experiments.
"""

from __future__ import annotations

import io
import json
import threading
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
)

from virtualfermlab.experiments.analysis import (
    heatmap_data,
    pareto_front,
    rank_conditions,
)
from virtualfermlab.experiments.doe import (
    ExperimentCondition,
    generate_conditions,
)
from virtualfermlab.experiments.monte_carlo import (
    MonteCarloResult,
    _run_single,
)
from virtualfermlab.models.ode_systems import ModelConfig
from virtualfermlab.parameters.distributions import sample_params
from virtualfermlab.parameters.library import list_available_strains, load_strain_profile
from virtualfermlab.parameters.schema import StrainProfile
from virtualfermlab.simulator.integrator import SimulationResult, simulate
from virtualfermlab.web.plotly_charts import (
    plot_heatmap_plotly,
    plot_mc_distributions_plotly,
    plot_mc_envelope_plotly,
    plot_pareto_plotly,
    plot_trajectory_plotly,
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = "virtualfermlab-dev-key"

# ---------------------------------------------------------------------------
# Background task management
# ---------------------------------------------------------------------------


@dataclass
class BackgroundTask:
    task_id: str
    status: str = "pending"
    progress: int = 0
    total: int = 0
    result: Any = None
    error: str = ""


_tasks: dict[str, BackgroundTask] = {}
_tasks_lock = threading.Lock()
_results_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helper: build params dict from form data + strain profile
# ---------------------------------------------------------------------------


def _nominal_params(profile: StrainProfile) -> dict:
    """Extract nominal (central) values from a strain profile.

    Unlike ``sample_params`` which draws from distributions, this returns
    only the ``.value`` field of each DistributionSpec — suitable for
    single deterministic simulations.
    """
    params: dict = {}
    substrates = list(profile.substrates.values())
    for i, sub in enumerate(substrates, start=1):
        suffix = str(i) if len(substrates) > 1 else ""
        params[f"mu_max{suffix}"] = sub.mu_max.value
        params[f"K_s{suffix}"] = sub.Ks.value
        params[f"Yx{suffix}"] = sub.Yxs.value

    for inh in profile.inhibitions:
        params["K_I"] = inh.K_I.value

    if profile.enzyme_params is not None:
        ep = profile.enzyme_params
        params["K_Z_c"] = ep.K_Z_c.value
        params["K_Z_S"] = ep.K_Z_S.value
        params["K_Z_d"] = ep.K_Z_d.value

    if profile.cardinal_pH is not None:
        cp = profile.cardinal_pH
        params["pH_min"] = cp.pH_min.value
        params["pH_opt"] = cp.pH_opt.value
        params["pH_max"] = cp.pH_max.value

    return params


def _build_params(data: dict, profile: StrainProfile, enzyme_mode: str = "direct") -> dict:
    """Assemble simulation params from user form data and strain profile.

    Uses nominal (deterministic) values from the profile.
    """
    params = _nominal_params(profile)

    S_total = float(data.get("total_concentration", 30.0))
    ratio = float(data.get("ratio", 0.5))

    params["S1"] = S_total * ratio
    params["S2"] = S_total * (1.0 - ratio)
    params["S_in1"] = params["S1"]
    params["S_in2"] = params["S2"]
    params["X0"] = float(data.get("X0", 0.03))
    params["y0"] = 0.0
    params["dilutionRate"] = float(data.get("dilution_rate", 0.0))

    # Enzyme initial conditions (non-zero to bootstrap growth)
    if enzyme_mode == "enzyme":
        params.setdefault("Z0", 0.01)
    elif enzyme_mode == "kompala":
        params.setdefault("Z1", 0.01)
        params.setdefault("Z2", 0.01)

    return params


def _build_config(data: dict) -> ModelConfig:
    """Construct a ModelConfig from form data."""
    use_cardinal_pH = bool(data.get("use_cardinal_pH", False))
    use_lag = bool(data.get("use_lag", False))

    return ModelConfig(
        n_substrates=int(data.get("n_substrates", 2)),
        n_feeds=int(data.get("n_feeds", 1)),
        growth_model=data.get("growth_model", "Monod"),
        enzyme_mode=data.get("enzyme_mode", "direct"),
        use_cardinal_pH=use_cardinal_pH,
        use_lag=use_lag,
        pH=float(data["pH"]) if use_cardinal_pH and data.get("pH") else None,
        pH_min=float(data["pH_min"]) if use_cardinal_pH and data.get("pH_min") else None,
        pH_opt=float(data["pH_opt"]) if use_cardinal_pH and data.get("pH_opt") else None,
        pH_max=float(data["pH_max"]) if use_cardinal_pH and data.get("pH_max") else None,
        lag=float(data["lag"]) if use_lag and data.get("lag") else None,
    )


def _serialize_profile(profile: StrainProfile) -> dict:
    """Convert a StrainProfile to a JSON-safe dictionary."""
    d = profile.model_dump()
    return d


def _result_summary(result: SimulationResult) -> dict:
    """Extract key metrics from a SimulationResult."""
    summary = {
        "yield_biomass": round(result.yield_biomass, 4),
        "final_biomass": round(result.final_biomass, 4),
        "mu_max_effective": round(result.mu_max_effective, 4),
        "final_substrates": {},
    }
    for name, series in result.substrates.items():
        summary["final_substrates"][name] = round(float(series[-1]), 4)
    return summary


# ---------------------------------------------------------------------------
# Routes: Pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes: API — Strain
# ---------------------------------------------------------------------------


@app.route("/api/strains")
def api_strains():
    strains = list_available_strains()
    return jsonify({"strains": strains})


@app.route("/api/strain/<name>")
def api_strain(name):
    try:
        profile = load_strain_profile(name)
        return jsonify(_serialize_profile(profile))
    except Exception as e:
        return jsonify({"error": str(e)}), 404


# ---------------------------------------------------------------------------
# Routes: API — Single Simulation
# ---------------------------------------------------------------------------


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    try:
        strain_name = data.get("strain", "F_venenatum_A35")
        profile = load_strain_profile(strain_name)
        config = _build_config(data)
        params = _build_params(data, profile, enzyme_mode=config.enzyme_mode)
        times = np.linspace(
            float(data.get("time_start", 0)),
            float(data.get("time_end", 100)),
            int(data.get("n_points", 500)),
        )
        method = data.get("method", "odeint")
        result = simulate(config, params, times, method=method)

        # Cache for export
        task_id = str(uuid.uuid4())[:8]
        _results_cache[task_id] = result

        plot_json = plot_trajectory_plotly(result)
        summary = _result_summary(result)

        return jsonify({
            "plot_json": plot_json,
            "summary": summary,
            "task_id": task_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes: API — Monte Carlo
# ---------------------------------------------------------------------------


def _run_single_with_trajectory(
    condition: ExperimentCondition,
    profile: StrainProfile,
    config: ModelConfig,
    times: np.ndarray,
    seed: int,
):
    """Run one MC replicate and return both metrics and the full trajectory."""
    rng = np.random.default_rng(seed)
    sampled = sample_params(profile, rng)

    S_total = condition.total_concentration
    S1 = S_total * condition.ratio
    S2 = S_total * (1.0 - condition.ratio)

    params = {
        **sampled,
        "S1": S1,
        "S2": S2,
        "S_in1": S1,
        "S_in2": S2,
        "X0": 0.03,
        "y0": 0.0,
        "dilutionRate": condition.dilution_rate,
    }

    if config.enzyme_mode == "enzyme":
        params.setdefault("Z0", 0.01)
    elif config.enzyme_mode == "kompala":
        params.setdefault("Z1", 0.01)
        params.setdefault("Z2", 0.01)

    # Apply cardinal pH from sampled params
    if config.use_cardinal_pH and condition.pH is not None:
        config_run = ModelConfig(
            n_substrates=config.n_substrates,
            n_feeds=config.n_feeds,
            growth_model=config.growth_model,
            enzyme_mode=config.enzyme_mode,
            use_cardinal_pH=True,
            pH=condition.pH,
            pH_min=sampled.get("pH_min", config.pH_min),
            pH_opt=sampled.get("pH_opt", config.pH_opt),
            pH_max=sampled.get("pH_max", config.pH_max),
            use_lag=config.use_lag,
            lag=config.lag,
        )
    else:
        config_run = config

    max_plausible = params["X0"] + S_total * 2.0

    try:
        result = simulate(config_run, params, times)
        yield_val = result.yield_biomass
        mu_val = result.mu_max_effective

        # Hard reject: non-finite state or biomass exceeding physical bound
        if (
            np.any(~np.isfinite(result.X))
            or not np.isfinite(yield_val)
            or yield_val > max_plausible
        ):
            return {"yield": np.nan, "mu_max": np.nan, "conversion": np.nan, "trajectory": None}

        # Soft filter: mu_max_effective can be noisy from stiff solver
        # artifacts — keep yield/trajectory but cap the mu value.
        if not np.isfinite(mu_val) or mu_val > 2.0:
            mu_val = np.nan

        S1_final = result.substrates.get("S1", result.substrates.get("S", np.array([S1])))[-1]
        conversion = 1.0 - S1_final / S1 if S1 > 0 else 0.0
        return {"yield": yield_val, "mu_max": mu_val, "conversion": conversion, "trajectory": result}
    except Exception:
        return {"yield": np.nan, "mu_max": np.nan, "conversion": np.nan, "trajectory": None}


def _mc_worker(
    task_id: str,
    condition: ExperimentCondition,
    profile: StrainProfile,
    config: ModelConfig,
    times: np.ndarray,
    n_samples: int,
    seed: int,
):
    """Background worker for Monte Carlo simulation."""
    task = _tasks[task_id]
    task.status = "running"
    task.total = n_samples

    yields = []
    mu_maxs = []
    conversions = []
    trajectories = []

    for i in range(n_samples):
        try:
            r = _run_single_with_trajectory(condition, profile, config, times, seed + i)
            yields.append(r["yield"])
            mu_maxs.append(r["mu_max"])
            conversions.append(r["conversion"])
            if r["trajectory"] is not None:
                trajectories.append(r["trajectory"])
        except Exception:
            yields.append(np.nan)
            mu_maxs.append(np.nan)
            conversions.append(np.nan)

        with _tasks_lock:
            task.progress = i + 1

    mc_result = MonteCarloResult(
        condition=condition,
        yields=np.array(yields),
        mu_max_values=np.array(mu_maxs),
        substrate_conversions=np.array(conversions),
    )
    task.result = {"mc": mc_result, "trajectories": trajectories}
    task.status = "completed"


@app.route("/api/monte-carlo/start", methods=["POST"])
def api_mc_start():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    try:
        strain_name = data.get("strain", "F_venenatum_A35")
        profile = load_strain_profile(strain_name)
        config = _build_config(data)

        condition = ExperimentCondition(
            strain=strain_name,
            substrate_A=data.get("substrate_A", "glucose"),
            substrate_B=data.get("substrate_B", "xylose"),
            ratio=float(data.get("ratio", 0.5)),
            pH=float(data.get("pH", 6.0)),
            total_concentration=float(data.get("total_concentration", 30.0)),
            dilution_rate=float(data.get("dilution_rate", 0.0)),
        )

        times = np.linspace(
            float(data.get("time_start", 0)),
            float(data.get("time_end", 100)),
            int(data.get("n_points", 500)),
        )

        n_samples = int(data.get("n_samples", 200))
        seed = int(data.get("seed", 42))

        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(task_id=task_id, total=n_samples)
        _tasks[task_id] = task

        t = threading.Thread(
            target=_mc_worker,
            args=(task_id, condition, profile, config, times, n_samples, seed),
            daemon=True,
        )
        t.start()

        return jsonify({"task_id": task_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/monte-carlo/status/<task_id>")
def api_mc_status(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "status": task.status,
        "progress": task.progress,
        "total": task.total,
        "error": task.error,
    })


@app.route("/api/monte-carlo/result/<task_id>")
def api_mc_result(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400

    task_data = task.result
    mc = task_data["mc"]
    trajectories = task_data["trajectories"]
    _results_cache[task_id] = mc

    dist_json = plot_mc_distributions_plotly(mc)
    envelope_json = plot_mc_envelope_plotly(trajectories)
    ci_lo, ci_hi = mc.ci95_yield

    n_valid = int(np.sum(np.isfinite(mc.yields)))

    return jsonify({
        "dist_json": dist_json,
        "envelope_json": envelope_json,
        "summary": {
            "mean_yield": round(mc.mean_yield, 4),
            "std_yield": round(mc.std_yield, 4),
            "mean_mu_max": round(mc.mean_mu_max, 4),
            "std_mu_max": round(mc.std_mu_max, 4),
            "ci95_yield": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_samples": len(mc.yields),
            "n_valid": n_valid,
        },
        "task_id": task_id,
    })


# ---------------------------------------------------------------------------
# Routes: API — Virtual Experiment (DOE + MC)
# ---------------------------------------------------------------------------


def _ve_worker(
    task_id: str,
    conditions: list,
    profile: StrainProfile,
    config: ModelConfig,
    times: np.ndarray,
    n_samples: int,
    seed: int,
):
    """Background worker for virtual experiment."""
    task = _tasks[task_id]
    task.status = "running"
    task.total = len(conditions)

    mc_results = []

    for i, cond in enumerate(conditions):
        yields = []
        mu_maxs = []
        conversions = []

        for j in range(n_samples):
            try:
                r = _run_single(cond, profile, config, times, seed + i * n_samples + j)
                yields.append(r["yield"])
                mu_maxs.append(r["mu_max"])
                conversions.append(r["conversion"])
            except Exception:
                yields.append(np.nan)
                mu_maxs.append(np.nan)
                conversions.append(np.nan)

        mc_result = MonteCarloResult(
            condition=cond,
            yields=np.array(yields),
            mu_max_values=np.array(mu_maxs),
            substrate_conversions=np.array(conversions),
        )
        mc_results.append(mc_result)

        with _tasks_lock:
            task.progress = i + 1

    task.result = mc_results
    task.status = "completed"


@app.route("/api/virtual-experiment/start", methods=["POST"])
def api_ve_start():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON data"}), 400

    try:
        strain_name = data.get("strain", "F_venenatum_A35")
        profile = load_strain_profile(strain_name)
        config = _build_config(data)

        # Generate DOE conditions
        sub_a = data.get("substrate_A", "glucose")
        sub_b = data.get("substrate_B", "xylose")
        pH_min = float(data.get("pH_range_min", 4.0))
        pH_max = float(data.get("pH_range_max", 7.0))
        ratio_min = float(data.get("ratio_range_min", 0.0))
        ratio_max = float(data.get("ratio_range_max", 1.0))
        n_continuous = int(data.get("n_continuous", 20))
        seed = int(data.get("seed", 42))

        conditions = generate_conditions(
            strains=[strain_name],
            substrates=[(sub_a, sub_b)],
            pH_range=(pH_min, pH_max),
            ratio_range=(ratio_min, ratio_max),
            total_concentration=float(data.get("total_concentration", 30.0)),
            n_continuous=n_continuous,
            seed=seed,
        )

        times = np.linspace(
            float(data.get("time_start", 0)),
            float(data.get("time_end", 100)),
            int(data.get("n_points", 500)),
        )

        n_samples = int(data.get("n_samples", 50))

        # Objective function settings
        obj_config = {
            "ranking_expr": data.get("obj_ranking", "mean_yield - 0.5 * std_yield"),
            "heatmap_metric": data.get("obj_heatmap", "mean_yield"),
            "pareto_obj1": data.get("obj_pareto_1", "mean_yield"),
            "pareto_obj2": data.get("obj_pareto_2", "mean_mu_max"),
        }

        task_id = str(uuid.uuid4())[:8]
        task = BackgroundTask(task_id=task_id, total=len(conditions))
        _tasks[task_id] = task

        # Store objective config alongside the task
        _results_cache[task_id + "_obj"] = obj_config

        t = threading.Thread(
            target=_ve_worker,
            args=(task_id, conditions, profile, config, times, n_samples, seed),
            daemon=True,
        )
        t.start()

        return jsonify({"task_id": task_id, "n_conditions": len(conditions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/virtual-experiment/status/<task_id>")
def api_ve_status(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "status": task.status,
        "progress": task.progress,
        "total": task.total,
        "error": task.error,
    })


def _build_ranking_fn(expr: str):
    """Build a ranking score function from a user expression.

    Allowed variables: mean_yield, std_yield, mean_mu_max, std_mu_max.
    Returns a callable ``f(MonteCarloResult) -> float``.
    """
    allowed_names = {"mean_yield", "std_yield", "mean_mu_max", "std_mu_max"}
    code = compile(expr, "<objective>", "eval")
    # Validate: only allowed names
    for name in code.co_names:
        if name not in allowed_names:
            raise ValueError(
                f"Unknown variable '{name}' in objective. "
                f"Allowed: {', '.join(sorted(allowed_names))}"
            )

    def score_fn(r: MonteCarloResult) -> float:
        ns = {
            "mean_yield": r.mean_yield,
            "std_yield": r.std_yield,
            "mean_mu_max": r.mean_mu_max,
            "std_mu_max": r.std_mu_max,
        }
        return float(eval(code, {"__builtins__": {}}, ns))

    return score_fn


_VALID_OBJ_ATTRS = {"mean_yield", "std_yield", "mean_mu_max", "std_mu_max"}


@app.route("/api/virtual-experiment/result/<task_id>")
def api_ve_result(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400

    mc_results = task.result
    _results_cache[task_id] = mc_results

    # Retrieve objective config
    obj_config = _results_cache.pop(task_id + "_obj", {})
    ranking_expr = obj_config.get("ranking_expr", "mean_yield - 0.5 * std_yield")
    heatmap_metric = obj_config.get("heatmap_metric", "mean_yield")
    pareto_obj1 = obj_config.get("pareto_obj1", "mean_yield")
    pareto_obj2 = obj_config.get("pareto_obj2", "mean_mu_max")

    # Validate heatmap/pareto attrs
    if heatmap_metric not in _VALID_OBJ_ATTRS:
        heatmap_metric = "mean_yield"
    if pareto_obj1 not in _VALID_OBJ_ATTRS:
        pareto_obj1 = "mean_yield"
    if pareto_obj2 not in _VALID_OBJ_ATTRS:
        pareto_obj2 = "mean_mu_max"

    # Analysis — ranking with user-defined score
    try:
        score_fn = _build_ranking_fn(ranking_expr)
    except Exception:
        score_fn = None  # fall back to default
    rankings_df = rank_conditions(mc_results, score_fn=score_fn)
    rankings_html = rankings_df.to_html(
        classes="table table-sm table-striped", index=False, float_format="%.4f"
    )

    # Heatmap
    heatmap_json = None
    try:
        hm_df = heatmap_data(mc_results, z=heatmap_metric)
        heatmap_json = plot_heatmap_plotly(
            hm_df, title=f"pH x Ratio Heatmap ({heatmap_metric})"
        )
    except Exception:
        pass

    # Pareto
    pareto_json = None
    try:
        pareto_df = pareto_front(mc_results, obj1=pareto_obj1, obj2=pareto_obj2)
        pareto_json = plot_pareto_plotly(pareto_df, obj1=pareto_obj1, obj2=pareto_obj2)
    except Exception:
        pass

    return jsonify({
        "rankings_html": rankings_html,
        "heatmap_json": heatmap_json,
        "pareto_json": pareto_json,
        "n_conditions": len(mc_results),
        "task_id": task_id,
    })


# ---------------------------------------------------------------------------
# Routes: API — Strain Discovery
# ---------------------------------------------------------------------------


def _discovery_worker(task_id: str, strain_name: str):
    """Background worker for the strain discovery pipeline."""
    task = _tasks[task_id]
    task.status = "running"
    task.total = 5  # 5 stages

    stage_map = {"search": 1, "extract": 2, "store": 3, "taxonomy": 4, "build_profile": 5}

    def progress_cb(info):
        with _tasks_lock:
            task.result = {**(task.result or {}), "current_stage": info}
            task.progress = stage_map.get(info.get("stage"), 0)

    try:
        from virtualfermlab.discovery.pipeline import run_discovery

        result = run_discovery(strain_name, progress_cb=progress_cb)

        # Serialise the DiscoveryResult for JSON transport
        result_dict = {
            "strain_name": result.strain_name,
            "source": result.source,
            "papers_found": result.papers_found,
            "params_extracted": result.params_extracted,
            "similar_strain": result.similar_strain,
            "similarity_score": result.similarity_score,
            "stages": result.stages,
            "profile": _serialize_profile(result.profile) if result.profile else None,
        }
        task.result = {"discovery": result_dict}
        task.status = "completed"
    except Exception as e:
        task.error = str(e)
        task.status = "failed"


@app.route("/api/discovery/start", methods=["POST"])
def api_discovery_start():
    data = request.get_json(force=True)
    strain_name = data.get("strain_name", "").strip()
    if not strain_name:
        return jsonify({"error": "strain_name is required"}), 400

    task_id = str(uuid.uuid4())[:8]
    task = BackgroundTask(task_id=task_id, total=5)
    _tasks[task_id] = task

    t = threading.Thread(
        target=_discovery_worker,
        args=(task_id, strain_name),
        daemon=True,
    )
    t.start()

    return jsonify({"task_id": task_id})


@app.route("/api/discovery/status/<task_id>")
def api_discovery_status(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404

    current_stage = None
    if isinstance(task.result, dict):
        current_stage = task.result.get("current_stage")

    return jsonify({
        "status": task.status,
        "progress": task.progress,
        "total": task.total,
        "error": task.error,
        "current_stage": current_stage,
    })


@app.route("/api/discovery/result/<task_id>")
def api_discovery_result(task_id):
    task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    if task.status != "completed":
        return jsonify({"error": "Task not completed yet"}), 400

    return jsonify(task.result.get("discovery", {}))


# ---------------------------------------------------------------------------
# Routes: API — Export
# ---------------------------------------------------------------------------


@app.route("/api/export/csv/<task_id>")
def api_export_csv(task_id):
    cached = _results_cache.get(task_id)
    if cached is None:
        return jsonify({"error": "No result found"}), 404

    buf = io.StringIO()

    if isinstance(cached, SimulationResult):
        data = {"time": cached.times.tolist(), "X": cached.X.tolist()}
        for name, series in cached.substrates.items():
            data[name] = series.tolist()
        for name, series in cached.enzymes.items():
            data[name] = series.tolist()
        data["totalOutput"] = cached.total_output.tolist()
        pd.DataFrame(data).to_csv(buf, index=False)

    elif isinstance(cached, MonteCarloResult):
        data = {
            "yield": cached.yields.tolist(),
            "mu_max": cached.mu_max_values.tolist(),
            "substrate_conversion": cached.substrate_conversions.tolist(),
        }
        pd.DataFrame(data).to_csv(buf, index=False)

    elif isinstance(cached, list):
        # Virtual experiment results
        rows = []
        for mc in cached:
            rows.append({
                "strain": mc.condition.strain,
                "substrate_A": mc.condition.substrate_A,
                "substrate_B": mc.condition.substrate_B,
                "ratio": mc.condition.ratio,
                "pH": mc.condition.pH,
                "mean_yield": mc.mean_yield,
                "std_yield": mc.std_yield,
                "mean_mu_max": mc.mean_mu_max,
                "std_mu_max": mc.std_mu_max,
            })
        pd.DataFrame(rows).to_csv(buf, index=False)

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=result_%s.csv" % task_id},
    )


@app.route("/api/export/json/<task_id>")
def api_export_json(task_id):
    cached = _results_cache.get(task_id)
    if cached is None:
        return jsonify({"error": "No result found"}), 404

    result_data: dict = {}

    if isinstance(cached, SimulationResult):
        result_data = {
            "type": "simulation",
            "yield_biomass": cached.yield_biomass,
            "mu_max_effective": cached.mu_max_effective,
            "times": cached.times.tolist(),
            "X": cached.X.tolist(),
            "substrates": {k: v.tolist() for k, v in cached.substrates.items()},
            "enzymes": {k: v.tolist() for k, v in cached.enzymes.items()},
        }

    elif isinstance(cached, MonteCarloResult):
        ci_lo, ci_hi = cached.ci95_yield
        result_data = {
            "type": "monte_carlo",
            "mean_yield": cached.mean_yield,
            "std_yield": cached.std_yield,
            "mean_mu_max": cached.mean_mu_max,
            "ci95_yield": [ci_lo, ci_hi],
            "n_samples": len(cached.yields),
        }

    elif isinstance(cached, list):
        rows = []
        for mc in cached:
            rows.append({
                "strain": mc.condition.strain,
                "substrate_A": mc.condition.substrate_A,
                "substrate_B": mc.condition.substrate_B,
                "ratio": mc.condition.ratio,
                "pH": mc.condition.pH,
                "mean_yield": mc.mean_yield,
                "std_yield": mc.std_yield,
                "mean_mu_max": mc.mean_mu_max,
            })
        result_data = {"type": "virtual_experiment", "conditions": rows}

    json_str = json.dumps(result_data, indent=2)
    return Response(
        json_str,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=result_%s.json" % task_id},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VirtualFermLab Web UI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
