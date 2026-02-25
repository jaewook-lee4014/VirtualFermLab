# 0.TomsCode — Baseline Notebooks

This directory contains the original experimental analysis notebooks and data that served as the foundation for the VirtualFermLab package.

> For full project documentation, see the [root README](../README.md).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `ESCAPE25PEwRandomSampling.ipynb` | Dual-substrate (glucose + xylose) model selection, parameter estimation, uncertainty analysis |
| `MPRpHExp18112025Analysis.ipynb` | pH 4.0–6.5 growth kinetics analysis, Monod vs exponential cap comparison |

## CSV Data (`0.Data/`)

| File | Description |
|------|-------------|
| `1-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 1:1 calibration data |
| `2-to-1-GlucoseXyloseMicroplateGrowth.csv` | Glucose:Xylose 2:1 validation data |
| `MPR18112025CSV - Sheet1.csv` | Growth data across 6 pH conditions (4.0–6.5) |

## Directory Structure

```
0.TomsCode/
├── 0.Data/          # Experimental CSV data
└── 1.Code/          # Jupyter analysis notebooks
```
