#!/usr/bin/env python3
"""Retrain HPLC prediction models using synthetic + real data.

Evaluates all 6 scenarios:
  A) Baseline: real data only (cross-ratio evaluation)
  B) Augmented: synthetic (train) + real (test) — cross-ratio evaluation

Usage:
    python scripts/train_with_synthetic.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from virtualfermlab.data.loaders import load_growth_csv
from virtualfermlab.data.transforms import tidy_data
from virtualfermlab.prediction.features import build_features
from virtualfermlab.prediction.evaluation import run_scenario, compare_scenarios


SYNTH_DIR = PROJECT_ROOT / "synthetic_data"
DATA_DIR = PROJECT_ROOT / "0.TomsCode" / "0.Data"


def load_real_data():
    """Load and tidy real experimental data."""
    df_11 = tidy_data(load_growth_csv(DATA_DIR / "1-to-1-GlucoseXyloseMicroplateGrowth.csv"))
    df_21 = tidy_data(load_growth_csv(DATA_DIR / "2-to-1-GlucoseXyloseMicroplateGrowth.csv"))
    return df_11, df_21


def load_synthetic_data():
    """Load synthetic data and convert to the same format as real data."""
    from virtualfermlab.data.transforms import od600_to_biomass

    dfs = {}
    for ratio_tag in ["1-to-1", "2-to-1"]:
        path = SYNTH_DIR / f"synthetic_{ratio_tag}_GlucoseXylose.csv"
        if not path.exists():
            logger.warning("Synthetic data not found: %s", path)
            continue
        df = pd.read_csv(path)
        # Rename to match real data format
        df = df.rename(columns={"[Xylose] (g/L)": "[Xylose] (g/L)"})
        # Add biomass column
        df["[Biomass] (g/L)"] = od600_to_biomass(df["OD600 (-)"])
        # Drop trajectory_id for feature extraction
        if "trajectory_id" in df.columns:
            df = df.drop(columns=["trajectory_id"])
        dfs[ratio_tag] = df

    return dfs.get("1-to-1"), dfs.get("2-to-1")


def combine_real_and_synthetic(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine real + synthetic data into one DataFrame."""
    if df_synth is None:
        return df_real
    # Ensure same columns
    cols = [c for c in df_real.columns if c in df_synth.columns]
    return pd.concat([df_real[cols], df_synth[cols]], ignore_index=True)


def main():
    logger.info("=" * 70)
    logger.info("HPLC Prediction Model — Synthetic Data Augmentation Evaluation")
    logger.info("=" * 70)

    # 1. Load data
    logger.info("Loading real data...")
    real_11, real_21 = load_real_data()
    logger.info("  Real 1:1: %d rows, Real 2:1: %d rows", len(real_11), len(real_21))

    logger.info("Loading synthetic data...")
    synth_11, synth_21 = load_synthetic_data()
    if synth_11 is not None:
        logger.info("  Synthetic 1:1: %d rows", len(synth_11))
    if synth_21 is not None:
        logger.info("  Synthetic 2:1: %d rows", len(synth_21))

    # 2. Evaluate all 6 scenarios
    all_results_baseline = {}
    all_results_augmented = {}

    for scenario in range(1, 7):
        logger.info("-" * 50)
        logger.info("Scenario %d", scenario)

        # ── A) Baseline: train on real 1:1, test on real 2:1 ──
        try:
            X_train_b, y_train_b = build_features(real_11, scenario)
            X_test_b, y_test_b = build_features(real_21, scenario)

            if len(X_train_b) == 0 or len(X_test_b) == 0:
                logger.warning("  Baseline: empty features, skipping")
                continue

            result_b = run_scenario(scenario, X_train_b, y_train_b, X_test_b, y_test_b)
            all_results_baseline[scenario] = result_b
            m = result_b["metrics"]
            for _, row in m[m["model"] == "XGBoost"].iterrows():
                logger.info("  BASELINE XGB %s: MAE=%.3f, R²=%.3f",
                            row["target"], row["MAE"], row["R2"])
        except Exception as e:
            logger.error("  Baseline scenario %d failed: %s", scenario, e)

        # ── B) Augmented: train on real_1:1 + synth, test on real 2:1 ──
        try:
            # Combine real 1:1 + synthetic 1:1 + synthetic 2:1 for training
            train_dfs = [real_11]
            if synth_11 is not None:
                train_dfs.append(synth_11)
            if synth_21 is not None:
                train_dfs.append(synth_21)

            cols = list(real_11.columns)
            combined_train = pd.concat(
                [d[[c for c in cols if c in d.columns]] for d in train_dfs],
                ignore_index=True,
            )

            X_train_a, y_train_a = build_features(combined_train, scenario)
            X_test_a, y_test_a = build_features(real_21, scenario)

            if len(X_train_a) == 0 or len(X_test_a) == 0:
                logger.warning("  Augmented: empty features, skipping")
                continue

            # Align columns (synthetic may have slightly different feature set)
            common_cols = X_train_a.columns.intersection(X_test_a.columns)
            X_train_a = X_train_a[common_cols]
            X_test_a = X_test_a[common_cols]

            result_a = run_scenario(scenario, X_train_a, y_train_a, X_test_a, y_test_a)
            all_results_augmented[scenario] = result_a
            m = result_a["metrics"]
            for _, row in m[m["model"] == "XGBoost"].iterrows():
                logger.info("  AUGMENTED XGB %s: MAE=%.3f, R²=%.3f",
                            row["target"], row["MAE"], row["R2"])
        except Exception as e:
            logger.error("  Augmented scenario %d failed: %s", scenario, e)

    # 3. Comparison summary
    logger.info("=" * 70)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 70)

    if all_results_baseline:
        baseline_df = compare_scenarios(all_results_baseline)
        baseline_xgb = baseline_df[baseline_df["model"] == "XGBoost"]
        logger.info("\n=== BASELINE (real data only) ===")
        logger.info("\n%s", baseline_xgb.to_string(index=False))

    if all_results_augmented:
        augmented_df = compare_scenarios(all_results_augmented)
        augmented_xgb = augmented_df[augmented_df["model"] == "XGBoost"]
        logger.info("\n=== AUGMENTED (real + synthetic) ===")
        logger.info("\n%s", augmented_xgb.to_string(index=False))

    # 4. Side-by-side delta
    if all_results_baseline and all_results_augmented:
        logger.info("\n=== IMPROVEMENT (Augmented - Baseline) ===")
        logger.info("%-10s %-10s %10s %10s %10s", "Scenario", "Target", "dMAE", "dRMSE", "dR²")
        for sc in sorted(set(all_results_baseline) & set(all_results_augmented)):
            bm = all_results_baseline[sc]["metrics"]
            am = all_results_augmented[sc]["metrics"]
            for target in ("glucose", "xylose"):
                b_row = bm[(bm["model"] == "XGBoost") & (bm["target"] == target)].iloc[0]
                a_row = am[(am["model"] == "XGBoost") & (am["target"] == target)].iloc[0]
                d_mae = a_row["MAE"] - b_row["MAE"]
                d_rmse = a_row["RMSE"] - b_row["RMSE"]
                d_r2 = a_row["R2"] - b_row["R2"]
                logger.info("%-10d %-10s %+10.3f %+10.3f %+10.3f", sc, target, d_mae, d_rmse, d_r2)

    # Save results
    output_dir = PROJECT_ROOT / "synthetic_data"
    if all_results_baseline:
        baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    if all_results_augmented:
        augmented_df.to_csv(output_dir / "augmented_metrics.csv", index=False)
    logger.info("\nMetrics saved to %s/", output_dir)


if __name__ == "__main__":
    main()
