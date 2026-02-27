"""Feature engineering for HPLC prediction scenarios.

Six scenarios of increasing information availability, from single-point
prediction (Scenario 1) to full growth curve + partial HPLC history
(Scenario 6).  Each ``scenario_N_features`` function accepts a tidy
DataFrame and returns ``(X, y)`` where *X* is a feature DataFrame and *y*
holds the glucose/xylose targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtualfermlab.data.transforms import od600_to_biomass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure biomass column exists and normalise xylose column name."""
    df = df.copy()
    if "[Xylose] (g/L) " in df.columns:
        df.rename(
            columns={"[Xylose] (g/L) ": "[Xylose] (g/L)"},
            inplace=True,
        )
    if "[Biomass] (g/L)" not in df.columns:
        df["[Biomass] (g/L)"] = od600_to_biomass(df["OD600 (-)"])
    return df


def _compute_time_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Median OD600, biomass, glucose, xylose per unique time point."""
    return (
        df.groupby("Time (h)")
        .agg(
            {
                "OD600 (-)": "median",
                "[Biomass] (g/L)": "median",
                "[Glucose] (g/L)": "median",
                "[Xylose] (g/L)": "median",
            }
        )
        .reset_index()
        .sort_values("Time (h)")
        .reset_index(drop=True)
    )


def _safe_log(x):
    """Natural log with floor at 1e-6."""
    return np.log(np.maximum(np.asarray(x, dtype=float), 1e-6))


# ---------------------------------------------------------------------------
# Scenario 1 — Point Prediction
# ---------------------------------------------------------------------------

def scenario_1_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-point features: *time*, *od600*, *biomass*.

    Every row in *df* becomes one sample.
    """
    df = _prepare_df(df)
    features = pd.DataFrame(
        {
            "time": df["Time (h)"].values,
            "od600": df["OD600 (-)"].values,
            "biomass": df["[Biomass] (g/L)"].values,
        }
    )
    targets = pd.DataFrame(
        {
            "glucose": df["[Glucose] (g/L)"].values,
            "xylose": df["[Xylose] (g/L)"].values,
        }
    )
    return features, targets


# ---------------------------------------------------------------------------
# Scenario 2 — Short OD600 Window (3 consecutive time points)
# ---------------------------------------------------------------------------

def scenario_2_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3-point OD600 window.  Growth rate and acceleration features.

    Requires >= 3 unique time points; first two time points are dropped.
    Window medians are used for biomass; each replicate at *t* is a sample.
    """
    df = _prepare_df(df)
    medians = _compute_time_medians(df)
    times = medians["Time (h)"].values
    med_bio = medians["[Biomass] (g/L)"].values

    feat_rows: list[dict] = []
    tgt_rows: list[dict] = []

    for i in range(2, len(times)):
        t0, t1, t2 = times[i - 2], times[i - 1], times[i]
        b0, b1, b2 = med_bio[i - 2], med_bio[i - 1], med_bio[i]

        dt1 = t2 - t1
        dt0 = t1 - t0
        dt_total = t2 - t0
        gr_recent = (b2 - b1) / dt1 if dt1 > 0 else 0.0
        gr_early = (b1 - b0) / dt0 if dt0 > 0 else 0.0

        base = {
            "time": t2,
            "biomass_t": b2,
            "biomass_t_minus_1": b1,
            "biomass_t_minus_2": b0,
            "growth_rate_recent": gr_recent,
            "growth_rate_early": gr_early,
            "growth_acceleration": gr_recent - gr_early,
            "specific_growth_rate": gr_recent / b2 if b2 > 0 else 0.0,
            "fold_change": b2 / b0 if b0 > 0 else 0.0,
            "ln_biomass_t": float(_safe_log(b2)),
            "ln_biomass_t_minus_1": float(_safe_log(b1)),
            "ln_biomass_t_minus_2": float(_safe_log(b0)),
            "delta_t_recent": dt1,
            "delta_t_total": dt_total,
        }

        for _, row in df.loc[df["Time (h)"] == t2].iterrows():
            feat_rows.append({**base, "od600": row["OD600 (-)"]})
            tgt_rows.append(
                {
                    "glucose": row["[Glucose] (g/L)"],
                    "xylose": row["[Xylose] (g/L)"],
                }
            )

    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------------------------------------------------------------
# Scenario 3 — Short OD600 + HPLC History (3 OD + 2 HPLC)
# ---------------------------------------------------------------------------

def scenario_3_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3-point OD600 window **plus** HPLC at *t-2* and *t-1*."""
    df = _prepare_df(df)
    medians = _compute_time_medians(df)
    times = medians["Time (h)"].values
    med_bio = medians["[Biomass] (g/L)"].values
    med_glc = medians["[Glucose] (g/L)"].values
    med_xyl = medians["[Xylose] (g/L)"].values

    feat_rows: list[dict] = []
    tgt_rows: list[dict] = []

    for i in range(2, len(times)):
        t0, t1, t2 = times[i - 2], times[i - 1], times[i]
        b0, b1, b2 = med_bio[i - 2], med_bio[i - 1], med_bio[i]
        g0, g1 = med_glc[i - 2], med_glc[i - 1]
        x0, x1 = med_xyl[i - 2], med_xyl[i - 1]

        dt1 = t2 - t1
        dt0 = t1 - t0
        dt_total = t2 - t0
        gr_recent = (b2 - b1) / dt1 if dt1 > 0 else 0.0
        gr_early = (b1 - b0) / dt0 if dt0 > 0 else 0.0

        glc_rate = (g0 - g1) / dt0 if dt0 > 0 else 0.0
        xyl_rate = (x0 - x1) / dt0 if dt0 > 0 else 0.0
        delta_b = b1 - b0
        yield_glc = delta_b / (g0 - g1 + 1e-6) if g0 > g1 else 0.0
        yield_xyl = delta_b / (x0 - x1 + 1e-6) if x0 > x1 else 0.0

        base = {
            # -- OD600 window (same as Scenario 2) --
            "time": t2,
            "biomass_t": b2,
            "biomass_t_minus_1": b1,
            "biomass_t_minus_2": b0,
            "growth_rate_recent": gr_recent,
            "growth_rate_early": gr_early,
            "growth_acceleration": gr_recent - gr_early,
            "specific_growth_rate": gr_recent / b2 if b2 > 0 else 0.0,
            "fold_change": b2 / b0 if b0 > 0 else 0.0,
            "ln_biomass_t": float(_safe_log(b2)),
            "ln_biomass_t_minus_1": float(_safe_log(b1)),
            "ln_biomass_t_minus_2": float(_safe_log(b0)),
            "delta_t_recent": dt1,
            "delta_t_total": dt_total,
            # -- HPLC history --
            "glucose_t_minus_1": g1,
            "glucose_t_minus_2": g0,
            "xylose_t_minus_1": x1,
            "xylose_t_minus_2": x0,
            "glucose_consumption_rate": glc_rate,
            "xylose_consumption_rate": xyl_rate,
            "substrate_ratio": g1 / (x1 + 1e-6),
            "total_substrate": g1 + x1,
            "glucose_depleted": 1.0 if g1 < 0.5 else 0.0,
            "yield_est_glucose": yield_glc,
            "yield_est_xylose": yield_xyl,
        }

        for _, row in df.loc[df["Time (h)"] == t2].iterrows():
            feat_rows.append({**base, "od600": row["OD600 (-)"]})
            tgt_rows.append(
                {
                    "glucose": row["[Glucose] (g/L)"],
                    "xylose": row["[Xylose] (g/L)"],
                }
            )

    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------------------------------------------------------------
# Scenario 4 — Longer Window (5 OD + 4 HPLC)
# ---------------------------------------------------------------------------

def scenario_4_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5-point OD600 window plus HPLC at *t-4* … *t-1*."""
    df = _prepare_df(df)
    medians = _compute_time_medians(df)
    times = medians["Time (h)"].values
    med_bio = medians["[Biomass] (g/L)"].values
    med_glc = medians["[Glucose] (g/L)"].values
    med_xyl = medians["[Xylose] (g/L)"].values

    feat_rows: list[dict] = []
    tgt_rows: list[dict] = []

    for i in range(4, len(times)):
        # 5 time points: oldest → newest
        ts = [times[i - 4 + j] for j in range(5)]
        bs = [med_bio[i - 4 + j] for j in range(5)]
        gs = [med_glc[i - 4 + j] for j in range(5)]
        xs = [med_xyl[i - 4 + j] for j in range(5)]

        t_curr = ts[-1]
        b_curr = bs[-1]

        # Inter-point intervals and growth rates
        dts = [ts[j + 1] - ts[j] for j in range(4)]
        grs = [
            (bs[j + 1] - bs[j]) / dts[j] if dts[j] > 0 else 0.0
            for j in range(4)
        ]

        # ln(biomass) linear-regression slope ≈ μ
        ln_bs = np.array([float(_safe_log(b)) for b in bs])
        ts_arr = np.array(ts)
        ln_slope = float(np.polyfit(ts_arr, ln_bs, 1)[0]) if np.std(ts_arr) > 0 else 0.0

        # Substrate consumption rates (between first 4 HPLC points)
        glc_rates = [
            (gs[j] - gs[j + 1]) / dts[j] if dts[j] > 0 else 0.0
            for j in range(3)
        ]
        xyl_rates = [
            (xs[j] - xs[j + 1]) / dts[j] if dts[j] > 0 else 0.0
            for j in range(3)
        ]

        base: dict = {
            "time": t_curr,
        }
        # Biomass at 5 points
        for j in range(5):
            base[f"biomass_{j}"] = bs[j]
        # Growth rates at 4 intervals
        for j in range(4):
            base[f"growth_rate_{j}"] = grs[j]
        base["ln_biomass_slope"] = ln_slope
        base["growth_acceleration"] = grs[-1] - grs[0]
        base["specific_growth_rate"] = grs[-1] / b_curr if b_curr > 0 else 0.0
        base["fold_change"] = b_curr / bs[0] if bs[0] > 0 else 0.0
        base["ln_biomass_current"] = float(_safe_log(b_curr))
        base["delta_t_total"] = ts[-1] - ts[0]
        # HPLC at t-4 … t-1
        for j in range(4):
            base[f"glucose_{j}"] = gs[j]
            base[f"xylose_{j}"] = xs[j]
        for j in range(3):
            base[f"glucose_consumption_rate_{j}"] = glc_rates[j]
            base[f"xylose_consumption_rate_{j}"] = xyl_rates[j]
        base["glucose_consumption_accel"] = glc_rates[-1] - glc_rates[0] if len(glc_rates) >= 2 else 0.0
        base["xylose_consumption_accel"] = xyl_rates[-1] - xyl_rates[0] if len(xyl_rates) >= 2 else 0.0
        base["substrate_ratio"] = gs[3] / (xs[3] + 1e-6)
        base["total_substrate"] = gs[3] + xs[3]
        base["glucose_depleted"] = 1.0 if gs[3] < 0.5 else 0.0

        for _, row in df.loc[df["Time (h)"] == t_curr].iterrows():
            feat_rows.append({**base, "od600": row["OD600 (-)"]})
            tgt_rows.append(
                {
                    "glucose": row["[Glucose] (g/L)"],
                    "xylose": row["[Xylose] (g/L)"],
                }
            )

    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------------------------------------------------------------
# Scenario 5 — Full OD600 + Initial HPLC
# ---------------------------------------------------------------------------

def scenario_5_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full growth-curve summary statistics plus initial HPLC values."""
    df = _prepare_df(df)
    medians = _compute_time_medians(df)
    times = medians["Time (h)"].values
    med_bio = medians["[Biomass] (g/L)"].values

    initial_glucose = medians["[Glucose] (g/L)"].iloc[0]
    initial_xylose = medians["[Xylose] (g/L)"].iloc[0]
    t0 = times[0]
    time_range = times[-1] - t0

    feat_rows: list[dict] = []
    tgt_rows: list[dict] = []

    for i in range(len(times)):
        t = times[i]
        up_to = med_bio[: i + 1]
        times_up = times[: i + 1]

        b_curr = up_to[-1]
        b_max = float(np.max(up_to))
        b_mean = float(np.mean(up_to))
        b_std = float(np.std(up_to)) if len(up_to) > 1 else 0.0
        b_integral = float(np.trapezoid(up_to, times_up)) if len(up_to) > 1 else 0.0

        # Growth rates
        if len(up_to) > 1:
            grs = []
            for j in range(len(up_to) - 1):
                dt = times_up[j + 1] - times_up[j]
                if dt > 0:
                    grs.append((up_to[j + 1] - up_to[j]) / dt)
            max_gr = max(grs) if grs else 0.0
            curr_gr = grs[-1] if grs else 0.0
        else:
            max_gr = curr_gr = 0.0

        phase_frac = b_curr / b_max if b_max > 0 else 0.0
        elapsed = t - t0
        elapsed_frac = elapsed / time_range if time_range > 0 else 0.0

        base = {
            "time": t,
            "biomass_current": b_curr,
            "biomass_max": b_max,
            "biomass_mean": b_mean,
            "biomass_std": b_std,
            "biomass_integral": b_integral,
            "max_growth_rate": max_gr,
            "current_growth_rate": curr_gr,
            "specific_growth_rate": curr_gr / b_curr if b_curr > 0 else 0.0,
            "growth_phase_fraction": phase_frac,
            "ln_biomass_current": float(_safe_log(b_curr)),
            "initial_glucose": initial_glucose,
            "initial_xylose": initial_xylose,
            "initial_total_substrate": initial_glucose + initial_xylose,
            "elapsed_time": elapsed,
            "elapsed_fraction": elapsed_frac,
            "n_timepoints": i + 1,
        }

        for _, row in df.loc[df["Time (h)"] == t].iterrows():
            feat_rows.append({**base, "od600": row["OD600 (-)"]})
            tgt_rows.append(
                {
                    "glucose": row["[Glucose] (g/L)"],
                    "xylose": row["[Xylose] (g/L)"],
                }
            )

    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------------------------------------------------------------
# Scenario 6 — Full OD600 + Partial HPLC
# ---------------------------------------------------------------------------

def scenario_6_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full growth curve plus all HPLC measurements up to *t-1*."""
    df = _prepare_df(df)
    medians = _compute_time_medians(df)
    times = medians["Time (h)"].values
    med_bio = medians["[Biomass] (g/L)"].values
    med_glc = medians["[Glucose] (g/L)"].values
    med_xyl = medians["[Xylose] (g/L)"].values

    initial_glucose = med_glc[0]
    initial_xylose = med_xyl[0]
    t0 = times[0]
    time_range = times[-1] - t0

    feat_rows: list[dict] = []
    tgt_rows: list[dict] = []

    for i in range(1, len(times)):  # need >= 1 preceding HPLC
        t = times[i]
        up_to = med_bio[: i + 1]
        times_up = times[: i + 1]

        b_curr = up_to[-1]
        b_max = float(np.max(up_to))
        b_mean = float(np.mean(up_to))
        b_std = float(np.std(up_to)) if len(up_to) > 1 else 0.0
        b_integral = float(np.trapezoid(up_to, times_up)) if len(up_to) > 1 else 0.0

        if len(up_to) > 1:
            grs = []
            for j in range(len(up_to) - 1):
                dt = times_up[j + 1] - times_up[j]
                if dt > 0:
                    grs.append((up_to[j + 1] - up_to[j]) / dt)
            max_gr = max(grs) if grs else 0.0
            curr_gr = grs[-1] if grs else 0.0
        else:
            max_gr = curr_gr = 0.0

        phase_frac = b_curr / b_max if b_max > 0 else 0.0
        elapsed = t - t0
        elapsed_frac = elapsed / time_range if time_range > 0 else 0.0

        # Partial HPLC history (indices 0 … i-1)
        known_g = med_glc[:i]
        known_x = med_xyl[:i]
        known_t = times[:i]
        last_g = float(known_g[-1])
        last_x = float(known_x[-1])
        t_since = t - known_t[-1]

        if len(known_g) >= 2:
            glc_trend = float(np.polyfit(known_t, known_g, 1)[0])
            xyl_trend = float(np.polyfit(known_t, known_x, 1)[0])
            dt_last = known_t[-1] - known_t[-2]
            if dt_last > 0:
                rec_glc_cons = (known_g[-2] - known_g[-1]) / dt_last
                rec_xyl_cons = (known_x[-2] - known_x[-1]) / dt_last
            else:
                rec_glc_cons = rec_xyl_cons = 0.0
        else:
            glc_trend = xyl_trend = 0.0
            rec_glc_cons = rec_xyl_cons = 0.0

        base = {
            # -- Growth-curve summary (Scenario 5 core) --
            "time": t,
            "biomass_current": b_curr,
            "biomass_max": b_max,
            "biomass_mean": b_mean,
            "biomass_std": b_std,
            "biomass_integral": b_integral,
            "max_growth_rate": max_gr,
            "current_growth_rate": curr_gr,
            "specific_growth_rate": curr_gr / b_curr if b_curr > 0 else 0.0,
            "growth_phase_fraction": phase_frac,
            "ln_biomass_current": float(_safe_log(b_curr)),
            "initial_glucose": initial_glucose,
            "initial_xylose": initial_xylose,
            "initial_total_substrate": initial_glucose + initial_xylose,
            "elapsed_time": elapsed,
            "elapsed_fraction": elapsed_frac,
            "n_timepoints": i + 1,
            # -- Partial HPLC --
            "last_known_glucose": last_g,
            "last_known_xylose": last_x,
            "time_since_last_hplc": t_since,
            "glucose_trend": glc_trend,
            "xylose_trend": xyl_trend,
            "recent_glucose_consumption": rec_glc_cons,
            "recent_xylose_consumption": rec_xyl_cons,
            "glucose_depleted": 1.0 if last_g < 0.5 else 0.0,
            "substrate_ratio": last_g / (last_x + 1e-6),
        }

        for _, row in df.loc[df["Time (h)"] == t].iterrows():
            feat_rows.append({**base, "od600": row["OD600 (-)"]})
            tgt_rows.append(
                {
                    "glucose": row["[Glucose] (g/L)"],
                    "xylose": row["[Xylose] (g/L)"],
                }
            )

    return pd.DataFrame(feat_rows), pd.DataFrame(tgt_rows)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

SCENARIO_FUNCTIONS = {
    1: scenario_1_features,
    2: scenario_2_features,
    3: scenario_3_features,
    4: scenario_4_features,
    5: scenario_5_features,
    6: scenario_6_features,
}

SCENARIO_DESCRIPTIONS = {
    1: "Point Prediction (time + OD600 only)",
    2: "Short OD600 Window (3 time points)",
    3: "Short OD600 + HPLC History (3 OD + 2 HPLC)",
    4: "Longer Window + HPLC History (5 OD + 4 HPLC)",
    5: "Full OD600 + Initial HPLC",
    6: "Full OD600 + Partial HPLC (up to t-1)",
}


def build_features(
    df: pd.DataFrame,
    scenario: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature matrix and targets for the given *scenario* number."""
    if scenario not in SCENARIO_FUNCTIONS:
        raise ValueError(
            f"Unknown scenario {scenario}. "
            f"Choose from {sorted(SCENARIO_FUNCTIONS)}."
        )
    return SCENARIO_FUNCTIONS[scenario](df)
