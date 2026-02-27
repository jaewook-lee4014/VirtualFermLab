#!/usr/bin/env python3
"""Generate synthetic HPLC training data from literature-crawled kinetic params.

Loads screened kinetic parameters from the discovery DB, builds plausible
parameter sets, runs batch ODE simulations (dual-substrate Monod + direct
inhibition), and outputs CSVs matching the real experimental data format.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-samples 500 --output-dir synthetic_data
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "src" / "virtualfermlab" / "discovery" / "strains.db"

# ── OD600 ↔ Biomass polynomial coefficients ───────────────────────
_OD_COEFF = (0.2927, 0.2631, 1.2808, -0.1596)  # a3, a2, a1, a0


def biomass_to_od600(X: float) -> float:
    """Invert the cubic polynomial: OD600 → biomass = a3*OD³ + a2*OD² + a1*OD + a0."""
    a3, a2, a1, a0 = _OD_COEFF

    def f(od):
        return a3 * od**3 + a2 * od**2 + a1 * od + a0 - X

    # OD600 is physically between 0 and ~4
    try:
        return brentq(f, 0.0, 5.0)
    except ValueError:
        # Out of range — clamp
        return max(0.0, min((X - a0) / a1 if a1 != 0 else 0.0, 5.0))


def biomass_to_od600_vec(X_arr: np.ndarray) -> np.ndarray:
    """Vectorised biomass → OD600 conversion."""
    return np.array([biomass_to_od600(float(x)) for x in X_arr])


# ── Substrate screening (same logic as screen_params.py) ──────────

TARGET_SUBSTRATES = {
    "glucose", "xylose", "fructose", "sucrose", "maltose", "galactose",
    "cellobiose", "cellulose", "starch", "lactose", "arabinose", "l-arabinose",
}
KEYWORD_MATCHES = [
    "straw", "biomass", "corn", "starch", "cellulose", "hemicellulose",
    "sugar", "wood", "bran", "hull",
]
EXCLUDED_SUBSTRATES = {
    "methanol", "glycerol", "sorbitol", "caffeine", "cu(ii)",
    "pyruvate", "casein", "sargassum spp.", "dhs",
    "methyl 3,4-dihydroxycinnamate",
}


def _is_relevant_substrate(substrate) -> bool:
    if substrate is None or str(substrate).lower() in ("none", "null", ""):
        return True
    s = str(substrate).strip().lower()
    if s in EXCLUDED_SUBSTRATES:
        return False
    if s in TARGET_SUBSTRATES:
        return True
    return any(kw in s for kw in KEYWORD_MATCHES)


# ── Load & process DB params ──────────────────────────────────────

def load_kinetic_params(db_path: Path) -> dict:
    """Load screened kinetic params from DB, grouped by parameter name.

    Returns dict with keys: mu_max, Ks, Yxs, K_I, lag_time, pH_opt
    Each value is a list of floats.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT strain_name, parameter_name, value, unit, substrate "
        "FROM extracted_params"
    ).fetchall()
    conn.close()

    params = defaultdict(list)
    for r in rows:
        if not _is_relevant_substrate(r["substrate"]):
            continue
        pname = r["parameter_name"]
        try:
            val = float(r["value"])
        except (ValueError, TypeError):
            continue

        # Unit normalisation
        unit = str(r["unit"] or "").strip().lower()

        if pname == "mu_max":
            # Convert 1/day → 1/h if needed
            if "day" in unit or "/d" in unit:
                val /= 24.0
            if 0.001 < val < 2.0:  # sanity check
                params["mu_max"].append(val)

        elif pname == "Ks":
            # Convert mM → g/L approximately (MW glucose ≈ 180)
            if "mm" in unit:
                val = val * 0.180  # mM → g/L for glucose
            if 0.001 < val < 50.0:  # sanity: Ks rarely > 50 g/L
                params["Ks"].append(val)

        elif pname == "Yxs":
            # Must be in g/g and between 0 and 1
            if "g/g" in unit or unit == "":
                if 0.01 < val < 1.0:
                    params["Yxs"].append(val)

        elif pname == "K_I":
            if "g/l" in unit or unit == "":
                if 0.01 < val < 200.0:
                    params["K_I"].append(val)

        elif pname == "lag_time":
            if 0 < val < 200:
                params["lag_time"].append(val)

        elif pname == "pH_opt":
            if 2.0 < val < 9.0:
                params["pH_opt"].append(val)

    return dict(params)


# ── ODE system (batch, dual-substrate, direct inhibition) ─────────

def batch_ode(y, t, mu1, mu2, Ks1, Ks2, Yx1, Yx2, K_I, lag):
    """Batch fermentation ODE: Monod + direct glucose→xylose inhibition."""
    X, S1, S2 = y[0], y[1], y[2]

    # Lag phase
    if t < lag:
        return [0.0, 0.0, 0.0]

    # Prevent negative concentrations
    X = max(X, 1e-8)
    S1 = max(S1, 0.0)
    S2 = max(S2, 0.0)

    # Growth rates (Monod)
    gr1 = mu1 * S1 / (Ks1 + S1)
    gr2 = mu2 * S2 / (Ks2 + S2) / (1.0 + S1 / K_I)

    dX = (gr1 + gr2) * X
    dS1 = -gr1 / Yx1 * X
    dS2 = -gr2 / Yx2 * X

    return [dX, dS1, dS2]


# ── Synthetic data generation ─────────────────────────────────────

def generate_one_trajectory(
    params: dict,
    time_points: np.ndarray,
    S1_init: float,
    S2_init: float,
    X0: float,
    noise_std: float = 0.02,
    n_replicates: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate one fermentation trajectory and return a DataFrame.

    Mimics the real CSV format with multiple replicates per time point.
    """
    if rng is None:
        rng = np.random.default_rng()

    y0 = [X0, S1_init, S2_init]
    sol = odeint(
        batch_ode, y0, time_points,
        args=(
            params["mu_max1"], params["mu_max2"],
            params["Ks1"], params["Ks2"],
            params["Yx1"], params["Yx2"],
            params["K_I"], params["lag"],
        ),
        mxstep=5000,
    )

    X_sim = sol[:, 0]
    S1_sim = np.clip(sol[:, 1], 0.0, None)
    S2_sim = np.clip(sol[:, 2], 0.0, None)

    # Convert biomass → OD600
    od600_sim = biomass_to_od600_vec(np.clip(X_sim, 0.0, None))

    # Build DataFrame with replicates (add noise)
    rows = []
    for i, t in enumerate(time_points):
        for _ in range(n_replicates):
            od_noisy = max(0.0, od600_sim[i] + rng.normal(0, noise_std))
            glc_noisy = max(0.0, S1_sim[i] + rng.normal(0, noise_std * 5))
            xyl_noisy = max(0.0, S2_sim[i] + rng.normal(0, noise_std * 5))
            rows.append({
                "Time (h)": round(t, 3),
                "OD600 (-)": round(od_noisy, 4),
                "[Glucose] (g/L)": round(glc_noisy, 4),
                "[Xylose] (g/L)": round(xyl_noisy, 4),
            })

    return pd.DataFrame(rows)


def sample_param_set(
    kinetic_params: dict,
    rng: np.random.Generator,
    ratio: str = "1:1",
) -> dict:
    """Sample a plausible parameter set from the literature distributions.

    Uses the literature values as anchors and adds controlled variation.
    """
    def _sample_from_list(values, scale=0.2):
        """Pick a value from the list and perturb it."""
        base = rng.choice(values)
        return max(1e-4, base * (1.0 + rng.normal(0, scale)))

    mu_vals = kinetic_params.get("mu_max", [0.15])
    ks_vals = kinetic_params.get("Ks", [0.2])
    yxs_vals = kinetic_params.get("Yxs", [0.4])
    ki_vals = kinetic_params.get("K_I", [0.5])
    lag_vals = kinetic_params.get("lag_time", [5.0])

    mu1 = _sample_from_list(mu_vals, 0.3)
    mu2 = mu1 * rng.uniform(0.15, 0.6)  # xylose growth typically slower
    Ks1 = _sample_from_list(ks_vals, 0.3)
    Ks2 = Ks1 * rng.uniform(0.5, 2.0)
    Yx1 = _sample_from_list(yxs_vals, 0.2)
    Yx2 = Yx1 * rng.uniform(0.5, 1.2)
    K_I = _sample_from_list(ki_vals, 0.3)
    lag = _sample_from_list(lag_vals, 0.3)
    lag = max(0.0, min(lag, 50.0))  # clamp

    # Initial substrate based on ratio
    if ratio == "1:1":
        S1_init = rng.uniform(13.0, 17.0)
        S2_init = rng.uniform(13.0, 17.0)
    else:  # 2:1
        S1_init = rng.uniform(18.0, 22.0)
        S2_init = rng.uniform(8.0, 12.0)

    X0 = rng.uniform(0.05, 0.3)

    return {
        "mu_max1": mu1, "mu_max2": mu2,
        "Ks1": Ks1, "Ks2": Ks2,
        "Yx1": Yx1, "Yx2": Yx2,
        "K_I": K_I, "lag": lag,
        "S1_init": S1_init, "S2_init": S2_init,
        "X0": X0,
    }


def validate_trajectory(df: pd.DataFrame) -> bool:
    """Check that a simulated trajectory is biologically plausible."""
    medians = df.groupby("Time (h)").median()

    # Biomass should increase over time
    od_first = medians["OD600 (-)"].iloc[0]
    od_last = medians["OD600 (-)"].iloc[-1]
    if od_last < od_first * 1.2:
        return False  # negligible growth

    # At least some glucose should be consumed
    glc_first = medians["[Glucose] (g/L)"].iloc[0]
    glc_last = medians["[Glucose] (g/L)"].iloc[-1]
    if glc_last > glc_first * 0.9:
        return False  # barely any consumption

    # OD shouldn't be absurdly high
    if medians["OD600 (-)"].max() > 4.0:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic fermentation data from literature kinetics"
    )
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of synthetic trajectories per ratio (default: 200)")
    parser.add_argument("--output-dir", type=str, default="synthetic_data",
                        help="Output directory for synthetic CSVs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load kinetic parameters from DB
    logger.info("Loading kinetic parameters from %s", args.db_path)
    kinetic_params = load_kinetic_params(Path(args.db_path))
    for k, v in kinetic_params.items():
        logger.info("  %s: %d values, range [%.4f, %.4f], median=%.4f",
                     k, len(v), min(v), max(v), np.median(v))

    # 2. Time points matching real experiments
    time_points = np.array([24, 32.333, 48, 55.5, 72, 80, 96, 104, 120, 128, 144])

    # 3. Generate synthetic data for both ratios
    all_dfs = {"1:1": [], "2:1": []}
    param_records = []

    for ratio in ["1:1", "2:1"]:
        logger.info("Generating %d trajectories for ratio %s", args.n_samples, ratio)
        generated = 0
        attempts = 0

        while generated < args.n_samples and attempts < args.n_samples * 5:
            attempts += 1
            pset = sample_param_set(kinetic_params, rng, ratio)

            df = generate_one_trajectory(
                pset, time_points,
                S1_init=pset["S1_init"],
                S2_init=pset["S2_init"],
                X0=pset["X0"],
                noise_std=0.02,
                n_replicates=3,
                rng=rng,
            )

            if validate_trajectory(df):
                # Tag with trajectory ID
                df["trajectory_id"] = f"{ratio}_{generated:04d}"
                all_dfs[ratio].append(df)
                pset["ratio"] = ratio
                pset["trajectory_id"] = f"{ratio}_{generated:04d}"
                param_records.append(pset)
                generated += 1

        logger.info("  Generated %d/%d (attempts=%d)", generated, args.n_samples, attempts)

    # 4. Save results
    for ratio, dfs in all_dfs.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        ratio_tag = ratio.replace(":", "-to-")
        out_path = output_dir / f"synthetic_{ratio_tag}_GlucoseXylose.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Saved %s (%d rows, %d trajectories)",
                     out_path.name, len(combined), len(dfs))

    # 5. Save parameter log
    param_df = pd.DataFrame(param_records)
    param_path = output_dir / "parameter_sets.csv"
    param_df.to_csv(param_path, index=False)
    logger.info("Parameter sets saved to %s (%d sets)", param_path.name, len(param_df))

    # 6. Summary
    total = sum(len(dfs) for dfs in all_dfs.values())
    logger.info("DONE: %d total synthetic trajectories generated", total)


if __name__ == "__main__":
    main()
