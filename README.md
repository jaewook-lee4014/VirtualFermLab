# VirtualFermLab

A research platform for modeling, simulating, and optimizing waste-to-protein fermentation processes.

Provides a unified pipeline spanning experimental data-driven parameter estimation, automated literature discovery (LLM-based parameter extraction), Monte Carlo uncertainty analysis, and virtual experiment design (DOE).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Service Architecture](#service-architecture)
- [Web UI API Endpoints](#web-ui-api-endpoints)
- [Extraction API Endpoints](#extraction-api-endpoints)
- [Discovery Pipeline](#discovery-pipeline)
- [ODE Growth Models](#ode-growth-models)
- [Parameter System](#parameter-system)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Baseline Notebooks](#baseline-notebooks)
- [Dependencies](#dependencies)
- [Glossary](#glossary)

---

## Project Structure

```
VirtualFermLab/
├── src/virtualfermlab/          # Python package (pip install -e .)
│   ├── models/                  # ODE-based growth models
│   │   ├── kinetics.py          #   Monod, Contois rate equations
│   │   ├── enzyme_regulation.py #   Enzyme induction / Kompala cybernetic
│   │   ├── ph_model.py          #   Cardinal pH model
│   │   ├── ode_systems.py       #   Unified ODE system (ModelConfig)
│   │   └── analytical.py        #   Analytical models (exponential cap, etc.)
│   │
│   ├── simulator/               # ODE integration engine
│   │   ├── integrator.py        #   simulate() — odeint/solve_ivp wrapper
│   │   └── steady_state.py      #   Steady-state analysis
│   │
│   ├── fitting/                 # Parameter estimation
│   │   ├── calibrate.py         #   Calibration loop
│   │   ├── validate.py          #   Validation loop
│   │   ├── objectives.py        #   Objective function definitions
│   │   ├── metrics.py           #   MAE, MSE, BIC, R², Log-Likelihood
│   │   ├── bounds.py            #   Parameter bounds
│   │   ├── fitters.py           #   least_squares / differential_evolution
│   │   └── growth_rate.py       #   Model-free growth rate estimation
│   │
│   ├── parameters/              # Strain parameter management
│   │   ├── schema.py            #   StrainProfile, SubstrateParams (Pydantic)
│   │   ├── library.py           #   YAML-based strain library loader
│   │   ├── distributions.py     #   Distribution-based parameter sampling
│   │   └── defaults/            #   Built-in strain profiles (YAML)
│   │       └── F_venenatum_A35.yaml
│   │
│   ├── discovery/               # Automated paper search + LLM extraction
│   │   ├── pipeline.py          #   5-stage discovery orchestrator
│   │   ├── paper_search.py      #   PubMed / Semantic Scholar crawler (parallel)
│   │   ├── llm_extraction.py    #   vLLM-based parameter extraction
│   │   ├── db.py                #   SQLite storage layer (thread-safe)
│   │   ├── name_resolver.py     #   Strain name normalization
│   │   ├── taxonomy.py          #   NCBI Taxonomy-based similar strain matching
│   │   └── prompts.py           #   LLM prompt templates
│   │
│   ├── extraction_api/          # LLM parameter extraction REST API (standalone)
│   │   └── app.py               #   Flask app + extraction endpoints
│   │
│   ├── experiments/             # Virtual experiment design
│   │   ├── doe.py               #   Latin Hypercube DOE condition generation
│   │   ├── monte_carlo.py       #   Monte Carlo simulation
│   │   └── analysis.py          #   Ranking, Heatmap, Pareto front
│   │
│   ├── data/                    # Data preprocessing
│   │   ├── loaders.py           #   CSV loader
│   │   ├── transforms.py        #   OD600 → biomass conversion
│   │   └── resampling.py        #   Bootstrap / Jackknife resampling
│   │
│   ├── viz/                     # Visualization (matplotlib)
│   │   ├── trajectories.py      #   Time-series trajectory plots
│   │   ├── heatmaps.py          #   DOE heatmaps
│   │   └── ph_analysis.py       #   pH analysis plots
│   │
│   ├── io/                      # Input / Output
│   │   └── export.py            #   CSV / JSON result export
│   │
│   └── web/                     # Flask web UI
│       ├── app.py               #   Flask app + all API endpoints
│       ├── plotly_charts.py     #   Plotly chart generation
│       ├── templates/           #   HTML templates (v1 / v2 themes)
│       │   ├── base.html / base_v1.html / base_v2.html
│       │   └── index.html / index_v1.html / index_v2.html
│       └── static/              #   CSS, JavaScript
│
├── tests/                       # pytest test suite
│   ├── conftest.py              #   Shared fixtures (rng, params, configs)
│   ├── test_models/
│   ├── test_simulator/
│   ├── test_fitting/
│   ├── test_parameters/
│   ├── test_experiments/
│   ├── test_discovery/
│   └── test_extraction_api/
│
├── scripts/                     # Server launch scripts
│   ├── start_vllm.sh            #   Start vLLM model server
│   ├── start_ui.sh              #   Start Flask UI server
│   ├── start_extraction_api.sh  #   Start Extraction API server
│   └── switch_ui.sh             #   Switch UI theme (v1 / v2)
│
├── 0.TomsCode/                  # Baseline notebooks (original)
│   ├── 0.Data/                  #   Experimental CSV data
│   └── 1.Code/                  #   Jupyter notebooks
│
└── pyproject.toml               # Package configuration
```

---

## Installation

### Prerequisites

- Python 3.9+
- (Optional) NVIDIA GPU + CUDA — required for the vLLM model server
- (Optional) `libc6-dev`, `python3-dev` — required for Triton compilation in vLLM

### Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Full install (web UI + extraction API + discovery + dev tools)
pip install -e ".[web,dev,discovery,extraction_api]"

# Additional install for GPU environments
pip install vllm
```

---

## Quick Start

```bash
# 1) Start the vLLM model server (GPU node — required for Discovery / Extraction)
bash scripts/start_vllm.sh

# 2) Start the web UI
bash scripts/start_ui.sh

# 3) Start the Extraction API (optional — standalone REST service)
bash scripts/start_extraction_api.sh
```

Open `http://localhost:8080` in a browser to access the web UI.

---

## Service Architecture

The project consists of **three independent processes**:

```
┌──────────────────────────────────────────────────────────────────┐
│                       Flask Web UI (:8080)                        │
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌────────────┐  ┌─────────────┐  │
│  │ Simulation│  │ Monte     │  │ Virtual    │  │ Discovery   │  │
│  │ API       │  │ Carlo API │  │ Experiment │  │ Pipeline    │  │
│  └───────────┘  └───────────┘  └────────────┘  └──────┬──────┘  │
│                                                        │         │
│       Local processing (CPU)                           │         │
└────────────────────────────────────────────────────────┼─────────┘
                                                         │ HTTP
┌──────────────────────────────────────────┐             │
│     Extraction API (:5001)               │             │
│     Standalone REST service              │             │
│     POST /extract/abstract               │             │
│     POST /extract/fulltext          ─────┼─────┐       │
└──────────────────────────────────────────┘     │       │
                                                 │       │
                  ┌──────────────────────────────┼───────┼───────┐
                  │        External API calls     │       │       │
                  │                              ▼       ▼       │
                  │  ┌───────────┐   ┌──────────────────────┐    │
                  │  │ PubMed    │   │ vLLM Server (:8000)  │    │
                  │  │ Semantic  │   │ Qwen2.5-32B-Instruct │    │
                  │  │ Scholar   │   │ (GPU, OpenAI API)    │    │
                  │  └───────────┘   └──────────────────────┘    │
                  └──────────────────────────────────────────────┘
```

- **Web UI** — Main interface covering simulation, Monte Carlo, DOE, and Discovery. Calls vLLM directly during discovery.
- **Extraction API** — Standalone stateless REST service that extracts fermentation parameters from paper text. Can be used independently of the Web UI.
- **vLLM Server** — Serves the Qwen2.5-32B-Instruct model via an OpenAI-compatible API. Both the Discovery Pipeline and Extraction API depend on this server.

> The vLLM server is **not required** for simulation, Monte Carlo, or DOE features — those work without it.

### Starting Each Service

#### 1. vLLM Model Server (GPU required)

```bash
bash scripts/start_vllm.sh
```

- Default model: `Qwen/Qwen2.5-32B-Instruct`
- Default port: `8000`
- Tensor parallelism: 2 GPUs (verified on L40S x2)
- To use a local model path: `VLLM_MODEL=/path/to/model bash scripts/start_vllm.sh`

#### 2. Flask Web UI

```bash
bash scripts/start_ui.sh
```

- Default port: `8080`
- vLLM connection: set `VLLM_BASE_URL` (default `http://localhost:8000/v1`)

#### 3. Extraction API

```bash
bash scripts/start_extraction_api.sh
```

- Default port: `5001`
- vLLM connection: set `VLLM_BASE_URL` and `VLLM_MODEL`

#### Switching UI Themes

```bash
./scripts/switch_ui.sh v1   # Bootstrap dark theme
./scripts/switch_ui.sh v2   # Preprint-v1 gradient theme
```

---

## Web UI API Endpoints

Async endpoints (Monte Carlo, Virtual Experiment, Discovery) follow a consistent pattern:
`POST .../start` returns a task ID → `GET .../status/<id>` polls progress → `GET .../result/<id>` retrieves the result.

### Strain Information

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/strains` | List available strains |
| GET | `/api/strain/<name>` | Get strain profile details |

### Simulation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/simulate` | Run a single ODE simulation |

### Monte Carlo

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/monte-carlo/start` | Start MC simulation (async) |
| GET | `/api/monte-carlo/status/<id>` | Poll progress |
| GET | `/api/monte-carlo/result/<id>` | Result + Plotly charts |

### Virtual Experiment (DOE + MC)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/virtual-experiment/start` | Generate DOE conditions + run MC (async) |
| GET | `/api/virtual-experiment/status/<id>` | Poll progress |
| GET | `/api/virtual-experiment/result/<id>` | Ranking, Heatmap, Pareto |

### Strain Discovery

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/discovery/start` | Start the automated discovery pipeline (async) |
| GET | `/api/discovery/status/<id>` | 5-stage progress status |
| GET | `/api/discovery/result/<id>` | Discovery result + StrainProfile |

### Result Export

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/export/csv/<id>` | Download result as CSV |
| GET | `/api/export/json/<id>` | Download result as JSON |

---

## Extraction API Endpoints

A standalone REST service that extracts fermentation parameters from paper text using an LLM. Operates independently of the Web UI and can be called directly from external systems.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status + vLLM connectivity check |
| GET | `/model-info` | Model name, token limit, list of extractable parameters |
| POST | `/extract/abstract` | Extract parameters from a paper abstract |
| POST | `/extract/fulltext` | Extract parameters from full paper text (auto-chunked) |

### Request / Response Examples

**`POST /extract/abstract`**

```json
// Request
{
  "title": "Growth kinetics of Fusarium venenatum in continuous culture",
  "abstract": "The maximum specific growth rate (μmax) was 0.28 h−1 ..."
}

// Response
{
  "parameters": [
    {
      "name": "mu_max",
      "value": 0.28,
      "unit": "1/h",
      "substrate": "glucose",
      "strain_name": "Fusarium venenatum A3/5",
      "confidence": "A",
      "evidence": "The maximum specific growth rate (μmax) was 0.28 h−1",
      "conditions": {"mode": "continuous", "pH": null, "temperature": null}
    }
  ],
  "n_parameters": 1,
  "source": "abstract"
}
```

**`POST /extract/fulltext`** — Automatically chunks long text and deduplicates extracted parameters across chunks.

```json
// Request
{
  "title": "Paper title",
  "full_text": "Full paper text (can be very long)..."
}

// Response
{
  "parameters": [...],
  "n_parameters": 8,
  "n_chunks": 3,
  "source": "fulltext"
}
```

### Extractable Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `mu_max` | Maximum specific growth rate | 1/h |
| `Ks` | Substrate saturation constant | g/L |
| `Yxs` | Biomass yield on substrate | g/g |
| `K_I` | Substrate inhibition constant | g/L |
| `pH_opt` | Optimal growth pH | — |
| `pH_min` | Minimum growth pH | — |
| `pH_max` | Maximum growth pH | — |
| `lag_time` | Lag phase duration | h |

### Error Codes

| Code | Meaning |
|------|---------|
| 400 | Missing required fields (`title`, `abstract` / `full_text`) |
| 503 | vLLM server unreachable |

---

## Discovery Pipeline

A 5-stage automated pipeline that performs **literature search → LLM parameter extraction → similar strain matching → automatic StrainProfile generation** for unknown strains:

```
Stage 1: Paper Search
  Parallel crawling of PubMed + Semantic Scholar APIs (ThreadPoolExecutor)
  DOI-based deduplication, results pushed to queue (producer)
      │
      ▼  (producer-consumer pattern — papers processed as they arrive)
Stage 2: LLM Extraction
  Papers pulled from queue and sent to vLLM for parameter extraction (consumer)
  Extracts: μ_max, Ks, Yxs, K_I, pH_opt/min/max, lag_time
  Value range validation + SQLite storage
      │
Stage 3: DB Storage
  Persisted to papers + extracted_params tables
      │
Stage 4: Taxonomy Match
  Similarity scoring against existing library strains using NCBI Taxonomy lineage
      │
Stage 5: Build StrainProfile
  Priority: literature-extracted values > similar strain values > biological defaults
  Pydantic StrainProfile object creation + caching
```

### Usage Example (Web UI)

```bash
# Start discovery
curl -X POST http://localhost:8080/api/discovery/start \
  -H "Content-Type: application/json" \
  -d '{"strain_name": "Fusarium venenatum", "max_papers": 10}'

# Response: {"task_id": "abc12345"}

# Check status
curl http://localhost:8080/api/discovery/status/abc12345

# Get result
curl http://localhost:8080/api/discovery/result/abc12345
```

---

## ODE Growth Models

### Core Data Flow

```
ModelConfig + params dict → FermentationODE (ODE RHS) → simulate() → SimulationResult
```

`SimulationResult` provides computed properties (`yield_biomass`, `final_biomass`, `mu_max_effective`) used by Monte Carlo analysis and the Web UI.

### Supported Models

| Growth Model | Enzyme Mode | Key Equation |
|-------------|-------------|--------------|
| **Monod** | Direct inhibition | `μ = μ_max · S/(S+K_s)`, xylose uses `1/(1+S1/K_I)` |
| **Contois** | Direct inhibition | `μ = μ_max · S/(S+K_s·X)` |
| **Monod** | Enzyme induction | Enzyme Z as separate ODE, `enzyme_factor = Z/(K_Z_S+Z)` |
| **Contois** | Enzyme induction | Above + Contois growth equation |
| **Monod** | Kompala cybernetic | Two enzymes Z1, Z2 + matching/proportional law |

### Model Configuration

```python
from virtualfermlab.models.ode_systems import ModelConfig

config = ModelConfig(
    growth_model="Monod",       # "Monod" or "Contois"
    enzyme_mode="direct",       # "direct", "enzyme", "kompala"
    use_cardinal_pH=True,       # Cardinal pH correction
    use_lag=True,               # Lag phase modeling
)
```

### State Variables

| Mode | State Vector |
|------|-------------|
| Direct | `[X, S1, S2, totalOutput]` |
| Enzyme | `[X, S1, S2, Z, totalOutput]` |
| Kompala | `[X, S1, S2, Z1, Z2, totalOutput]` |

### Additional Features

- **Cardinal pH**: Growth rate correction based on `pH_min`, `pH_opt`, `pH_max`
- **Lag phase**: Lag time modeling
- **Continuous fermentation**: Supports dilution rate (`D`) and feed substrate concentration (`S_in`)

---

## Parameter System

All parameters are structured and managed through `StrainProfile` (Pydantic model).

### YAML Profile Example

```yaml
# parameters/defaults/F_venenatum_A35.yaml
name: F_venenatum_A35
substrates:
  glucose:
    mu_max: {type: normal, value: 0.12, std: 0.02, confidence: A}
    Ks:     {type: fixed, value: 0.18, confidence: A}
    Yxs:    {type: uniform, low: 0.28, high: 0.36, confidence: B}
  xylose:
    mu_max: {type: normal, value: 0.06, std: 0.01, confidence: B}
cardinal_pH:
  pH_min: {type: fixed, value: 3.5}
  pH_opt: {type: fixed, value: 6.0}
  pH_max: {type: fixed, value: 7.5}
```

### DistributionSpec Types

Each parameter is represented as a `DistributionSpec` and used directly for Monte Carlo sampling:

| Type | Fields | Description |
|------|--------|-------------|
| `fixed` | `value` | Fixed value |
| `normal` | `value`, `std` | Normal distribution |
| `uniform` | `low`, `high` | Uniform distribution |
| `lognormal` | `value`, `std` | Log-normal distribution |
| `triangular` | `low`, `value`, `high` | Triangular distribution |

### Confidence Levels

| Level | Meaning |
|-------|---------|
| A | Directly measured experimentally |
| B | Literature-cited value |
| C | Estimated or default value |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_models/test_kinetics.py -v

# Run a specific test by name
python -m pytest tests/test_models/test_kinetics.py -k "test_monod" -v

# Run Extraction API tests
python -m pytest tests/test_extraction_api/ -v
```

Tests use shared fixtures from `tests/conftest.py`:
- `rng` — Reproducible random number generator
- `batch_params_direct`, `batch_params_enzyme`, `batch_params_kompala` — Parameter sets per mode
- `config_direct`, `config_enzyme`, `config_kompala` — ModelConfig per mode

### Test Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_models/             # ODE model tests
├── test_simulator/          # Integration engine tests
├── test_fitting/            # Parameter estimation tests
├── test_parameters/         # Strain parameter tests
├── test_experiments/        # MC / DOE tests
├── test_discovery/          # Discovery pipeline tests
└── test_extraction_api/     # Extraction API tests (uses mock LLMClient)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UI_PORT` | `8080` | Flask web UI server port |
| `API_PORT` | `5001` | Extraction API server port |
| `VLLM_PORT` | `8000` | vLLM server port |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_MODEL` | `Qwen/Qwen2.5-32B-Instruct` | Model used for LLM extraction |
| `VLLM_MAX_TOKENS` | `4096` | Max token limit for LLM extraction |
| `VLLM_TP_SIZE` | `2` | Tensor parallel GPU count |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face model cache path |

---

## Baseline Notebooks

Original experimental analysis notebooks are located in `0.TomsCode/`.

| Notebook | Description |
|----------|-------------|
| `ESCAPE25PEwRandomSampling.ipynb` | Dual-substrate (glucose + xylose) model selection, parameter estimation, uncertainty analysis |
| `MPRpHExp18112025Analysis.ipynb` | pH 4.0–6.5 growth kinetics analysis, Monod vs exponential cap comparison |

### CSV Data

| File | Description |
|------|-------------|
| `1-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 1:1 calibration data |
| `2-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 2:1 validation data |
| `MPR18112025CSV - Sheet1.csv` | Growth data across 6 pH conditions (4.0–6.5) |

---

## Dependencies

### Core

| Package | Purpose |
|---------|---------|
| numpy, scipy | ODE integration, optimization |
| pandas | Data processing |
| pydantic | Parameter schema validation |
| pyyaml | Strain profile loading |
| joblib | Parallel processing |
| matplotlib | Visualization (notebooks / export) |
| requests | External API calls (PubMed, Semantic Scholar, vLLM) |

### Optional (by feature group)

| Group | Packages | Purpose |
|-------|----------|---------|
| `web` | flask, plotly | Web UI server, interactive charts |
| `extraction_api` | flask, requests | LLM extraction REST API |
| `discovery` | requests | Paper discovery pipeline |
| `dev` | pytest, pytest-cov | Testing |

### GPU Only

| Package | Purpose |
|---------|---------|
| vllm | LLM model serving (OpenAI-compatible API) |
| torch (CUDA) | GPU computation |

---

## Glossary

| Term | Description |
|------|-------------|
| OD600 | Optical density at 600 nm (proxy for cell concentration) |
| μ_max | Maximum specific growth rate (1/h) |
| K_s | Substrate saturation constant (g/L) |
| Y_xs | Biomass yield on substrate (g/g) |
| K_I | Substrate inhibition constant (g/L) |
| Cardinal pH | pH_min / pH_opt / pH_max model for growth rate correction |
| Dilution rate (D) | Flow rate / volume in continuous fermentation (1/h) |
| DOE | Design of Experiments |
| Monte Carlo | Stochastic simulation reflecting parameter uncertainty |
| Kompala cybernetic | Enzyme-level optimization model for multi-substrate utilization |
