# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VirtualFermLab is a research project for modeling and simulating waste-to-protein fermentation processes. The codebase provides computational tools for fitting experimental microbial growth data to ODE-based kinetic models, optimizing fermentation parameters, and analyzing pH-dependent growth characteristics.

## Repository Structure

All executable code lives in Jupyter notebooks under `0.TomsCode/`. There is no formal Python package, no build system, no test suite, and no CI/CD pipeline. CSV data files sit alongside the notebooks.

## Running the Notebooks

```bash
# Required packages (no requirements.txt exists)
pip install numpy scipy pandas matplotlib

# Launch notebooks
jupyter notebook 0.TomsCode/ESCAPE25PEwRandomSampling.ipynb
jupyter notebook 0.TomsCode/MPRpHExp18112025Analysis.ipynb
```

Notebooks are self-contained and must be run with cells executed sequentially. CSV data files must remain in the same directory as the notebooks.

## Architecture

### ESCAPE25PEwRandomSampling.ipynb — Fermentation Process Optimization
- Fits dual-substrate (glucose/xylose) fermentation data to ODE growth models
- Supports Monod and Contois kinetics with three enzyme induction strategies (Direct Inhibition, Enzyme Inhibition, Optimal Enzyme Production)
- Key functions: `plantODE()` (growth kinetics ODEs), `calibration_function()`/`validation_function()` (parameter fitting), `tidy_data()` (OD600→biomass conversion), `get_traj()` (ODE solver), `resample_data()` (bootstrap uncertainty)
- Uses `scipy.optimize.least_squares` with bounds for parameter estimation and `scipy.integrate.solve_ivp`/`odeint` for ODE integration
- Evaluates fit quality via MAE, MSE, BIC, Log-Likelihood across 5 random subsampling repeats

### MPRpHExp18112025Analysis.ipynb — pH-Dependent Growth Analysis
- Characterizes microbial growth rates across pH 4.0–6.5
- Compares Monod-with-lag vs exponential-with-cap models
- Key functions: `fit_monod_lag_fixedYxs()`, `fit_lagged_exponential_cap_fixedYxs()`, `monod_lag_odes()`, `simulate_monod_lag()`

### Data Pipeline
OD600 measurements are converted to biomass concentration via an empirical cubic polynomial, then fed to parameter fitting routines. Fitted parameters (μ_max, K_s, Y_xs, lag time) are exported to CSV.

## Key Domain Conventions

- OD600: optical density at 600 nm (proxy for cell concentration)
- μ_max: maximum specific growth rate (1/h)
- K_s: substrate saturation constant (g/L)
- Y_xs: biomass yield on substrate (g/g)
- Dilution rate: flow rate / volume in continuous fermentation
