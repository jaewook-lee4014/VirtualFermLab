# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VirtualFermLab is a research platform for modeling and simulating waste-to-protein fermentation processes. It provides ODE-based kinetic models, parameter estimation, Monte Carlo uncertainty analysis, virtual experiment design (DOE), and an automated literature discovery pipeline with LLM-based parameter extraction.

## Common Commands

```bash
# Install (editable, with all optional deps)
pip install -e ".[web,dev,discovery,extraction_api]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_models/test_kinetics.py -v

# Run a single test by name
python -m pytest tests/test_models/test_kinetics.py -k "test_monod" -v

# Start the Flask web UI (default port 8080)
python -m virtualfermlab.web.app --host 0.0.0.0 --port 8080

# Start the extraction REST API (default port 5001)
bash scripts/start_extraction_api.sh
# or directly:
python -m virtualfermlab.extraction_api.app --port 5001

# Start vLLM server for discovery pipeline (GPU required)
bash scripts/start_vllm.sh

# Switch web UI theme
./scripts/switch_ui.sh v1   # Bootstrap dark
./scripts/switch_ui.sh v2   # Preprint gradient
```

## Architecture

### Package layout (`src/virtualfermlab/`)

- **models/**: ODE-based growth models. `kinetics.py` has Monod/Contois rate functions. `enzyme_regulation.py` has direct inhibition, enzyme induction, and Kompala cybernetic modes. `ph_model.py` has cardinal pH and lag-phase factors. `ode_systems.py` composes these via `ModelConfig` dataclass + `FermentationODE` class — this is the central ODE right-hand side that replaces the notebook's monolithic `plantODE()`.
- **simulator/**: `integrator.py` wraps `scipy.integrate.odeint`/`solve_ivp` into a `simulate()` function returning `SimulationResult`. `steady_state.py` handles continuous fermentation equilibria.
- **fitting/**: Parameter estimation pipeline. `calibrate.py`/`validate.py` run calibration/validation loops. `fitters.py` provides `least_squares` and `differential_evolution` solvers. `objectives.py` defines residual functions. `metrics.py` has MAE, MSE, BIC, R², Log-Likelihood. `bounds.py` manages parameter ranges.
- **parameters/**: `schema.py` defines Pydantic models (`StrainProfile`, `SubstrateParams`, `DistributionSpec` with fixed/normal/uniform/lognormal/triangular types and A/B/C confidence levels). `library.py` loads YAML strain profiles from `defaults/`. `distributions.py` samples from `DistributionSpec` for Monte Carlo.
- **experiments/**: `doe.py` generates Latin Hypercube experiment conditions. `monte_carlo.py` runs MC simulations. `analysis.py` provides ranking, heatmap data, and Pareto front extraction.
- **discovery/**: 5-stage automated pipeline (paper search → LLM extraction → DB storage → taxonomy matching → StrainProfile generation). `paper_search.py` crawls PubMed + Semantic Scholar in parallel with a producer-consumer pattern. `llm_extraction.py` calls a vLLM server (OpenAI-compatible API, default Qwen2.5-32B-Instruct). `db.py` is thread-safe SQLite. `taxonomy.py` uses NCBI Taxonomy lineage for similar-strain matching.
- **extraction_api/**: Standalone stateless REST API wrapping LLM extraction from the discovery pipeline. Endpoints: `GET /health`, `GET /model-info`, `POST /extract/abstract`, `POST /extract/fulltext`. Uses Flask; no database or numpy dependency. Tests inject a mock `LLMClient` via the `create_app(client=...)` factory.
- **data/**: `loaders.py` (CSV loading), `transforms.py` (OD600→biomass via cubic polynomial), `resampling.py` (bootstrap/jackknife).
- **viz/**: Matplotlib plotting — `trajectories.py`, `heatmaps.py`, `ph_analysis.py`.
- **io/**: `export.py` for CSV/JSON result export.
- **web/**: Flask app serving the UI. `app.py` defines all API endpoints. `plotly_charts.py` generates interactive Plotly charts. Async tasks (Monte Carlo, virtual experiments, discovery) run in background threads with polling status endpoints.

### Core data flow

`ModelConfig` + params dict → `FermentationODE` (ODE RHS) → `simulate()` → `SimulationResult`. The `SimulationResult` carries computed properties (`yield_biomass`, `final_biomass`, `mu_max_effective`) used by Monte Carlo analysis and the web UI. The discovery pipeline produces `StrainProfile` objects that feed into parameter sampling via `distributions.py`.

### ODE model configuration

Models are configured via `ModelConfig(growth_model=, enzyme_mode=, ...)` with three enzyme modes:
- **direct**: competitive inhibition, state vector `[X, S1, S2, totalOutput]`
- **enzyme**: single enzyme pool Z, state vector `[X, S1, S2, Z, totalOutput]`
- **kompala**: cybernetic with Z1/Z2, state vector `[X, S1, S2, Z1, Z2, totalOutput]`

Growth models: `"Monod"` or `"Contois"`. Optional: cardinal pH correction (`use_cardinal_pH`) and lag phase (`use_lag`).

### Web API structure

Async endpoints (Monte Carlo, virtual experiments, discovery) follow a consistent pattern: `POST .../start` returns a job ID, `GET .../status/<id>` polls progress, `GET .../result/<id>` returns the final output. Export via `/api/export/csv/<id>` and `/api/export/json/<id>`.

### Baseline notebooks (`0.TomsCode/1.Code/`)

The original analysis notebooks remain as reference. `ESCAPE25PEwRandomSampling.ipynb` does dual-substrate (glucose/xylose) model fitting with random subsampling. `MPRpHExp18112025Analysis.ipynb` analyzes pH-dependent growth across pH 4.0–6.5. CSV data files live in `0.TomsCode/0.Data/`.

## Key Environment Variables

- `UI_PORT` (default: 8080) — Flask web server port
- `API_PORT` (default: 5001) — Extraction API server port
- `VLLM_BASE_URL` (default: `http://localhost:8000/v1`) — vLLM server endpoint for discovery
- `VLLM_PORT` (default: 8000) — vLLM server port
- `VLLM_MODEL` (default: `Qwen/Qwen2.5-32B-Instruct`) — model for LLM extraction
- `VLLM_MAX_TOKENS` (default: 4096) — max token limit for LLM extraction

## Testing

Tests use pytest with shared fixtures in `tests/conftest.py` (provides `rng`, `batch_params_direct`, `batch_params_enzyme`, `batch_params_kompala`, `config_direct`, `config_enzyme`, `config_kompala`). Test directories mirror the package structure under `tests/test_<module>/`.

## Key Domain Terms

- OD600: optical density at 600 nm (proxy for cell concentration)
- μ_max: maximum specific growth rate (1/h)
- K_s: substrate saturation constant (g/L)
- Y_xs: biomass yield on substrate (g/g)
- K_I: substrate inhibition constant
- Cardinal pH: pH_min/pH_opt/pH_max model for growth rate correction
- Dilution rate (D): flow rate / volume in continuous fermentation
