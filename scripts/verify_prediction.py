#!/usr/bin/env python3
"""Verify HPLC prediction pipeline: features, models, and scenario comparison."""

from virtualfermlab.prediction.features import build_features, SCENARIO_DESCRIPTIONS
from virtualfermlab.prediction.scenarios import load_datasets, cross_ratio_split
from virtualfermlab.prediction.evaluation import run_scenario, compare_scenarios

# ── 1. Load data ──────────────────────────────────────────────────────
df_11, df_21 = load_datasets()
print(f"Loaded 1:1 data: {df_11.shape}, 2:1 data: {df_21.shape}")

# ── 2. Feature building for all scenarios ─────────────────────────────
print("\n=== Feature Building ===")
for sc in range(1, 7):
    X, y = build_features(df_11, sc)
    nan_count = X.isna().sum().sum()
    print(
        f"  Scenario {sc} ({SCENARIO_DESCRIPTIONS[sc][:40]:40s}): "
        f"{X.shape[0]:3d} samples, {X.shape[1]:2d} features, NaN={nan_count}"
    )

# ── 3. Cross-ratio evaluation ────────────────────────────────────────
print("\n=== Cross-Ratio Evaluation (Train 1:1 -> Test 2:1) ===")
all_results = {}
for sc in range(1, 7):
    X_tr, y_tr, X_te, y_te = cross_ratio_split(df_11, df_21, sc)
    res = run_scenario(sc, X_tr, y_tr, X_te, y_te)
    all_results[sc] = res
    xgb_g = res["metrics"].query("model == 'XGBoost' and target == 'glucose'")
    xgb_x = res["metrics"].query("model == 'XGBoost' and target == 'xylose'")
    print(
        f"  Scenario {sc}: "
        f"Glucose MAE={xgb_g['MAE'].values[0]:.3f} R2={xgb_g['R2'].values[0]:.3f} | "
        f"Xylose MAE={xgb_x['MAE'].values[0]:.3f} R2={xgb_x['R2'].values[0]:.3f}"
    )

# ── 4. Reverse direction ─────────────────────────────────────────────
print("\n=== Reverse Evaluation (Train 2:1 -> Test 1:1) ===")
for sc in range(1, 7):
    X_tr, y_tr, X_te, y_te = cross_ratio_split(df_21, df_11, sc)
    res = run_scenario(sc, X_tr, y_tr, X_te, y_te)
    xgb_g = res["metrics"].query("model == 'XGBoost' and target == 'glucose'")
    xgb_x = res["metrics"].query("model == 'XGBoost' and target == 'xylose'")
    print(
        f"  Scenario {sc}: "
        f"Glucose MAE={xgb_g['MAE'].values[0]:.3f} R2={xgb_g['R2'].values[0]:.3f} | "
        f"Xylose MAE={xgb_x['MAE'].values[0]:.3f} R2={xgb_x['R2'].values[0]:.3f}"
    )

# ── 5. Summary table ─────────────────────────────────────────────────
print("\n=== Full Summary (XGBoost, Train 1:1 -> Test 2:1) ===")
summary = compare_scenarios(all_results)
xgb = summary.query("model == 'XGBoost'").sort_values(["target", "scenario"])
print(xgb.to_string(index=False))

# ── 6. Trend check ───────────────────────────────────────────────────
print("\n=== Trend Check ===")
glc_maes = [
    all_results[sc]["metrics"]
    .query("model == 'XGBoost' and target == 'glucose'")["MAE"]
    .values[0]
    for sc in range(1, 7)
]
xyl_maes = [
    all_results[sc]["metrics"]
    .query("model == 'XGBoost' and target == 'xylose'")["MAE"]
    .values[0]
    for sc in range(1, 7)
]
print(f"  Glucose MAE trend: {[f'{m:.2f}' for m in glc_maes]}")
print(f"  Xylose  MAE trend: {[f'{m:.2f}' for m in xyl_maes]}")
if glc_maes[0] > glc_maes[-1]:
    print("  -> Glucose MAE decreases from Scenario 1 to 6 (expected)")
else:
    print("  -> WARNING: Glucose MAE does NOT decrease monotonically")

print("\nVerification complete.")
