"""Feature drift detection module using Population Stability Index (PSI) and two-sample KS tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_prepared_data
from src.train import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) between reference and current series.

    Works for both numeric features (quantile binning on reference) and categorical features.
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    if pd.api.types.is_numeric_dtype(ref_clean):
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(ref_clean.to_numpy(dtype=float), quantiles))
        if len(edges) < 2:
            return 0.0
        ref_counts, _ = np.histogram(ref_clean.to_numpy(dtype=float), bins=edges)
        cur_counts, _ = np.histogram(cur_clean.to_numpy(dtype=float), bins=edges)
    else:
        ref_str = ref_clean.astype(str)
        cur_str = cur_clean.astype(str)
        categories = sorted(set(ref_str.unique()) | set(cur_str.unique()))
        ref_counts = np.array([np.sum(ref_str == cat) for cat in categories], dtype=float)
        cur_counts = np.array([np.sum(cur_str == cat) for cat in categories], dtype=float)

    ref_props = np.clip(ref_counts / max(np.sum(ref_counts), 1.0), eps, None)
    cur_props = np.clip(cur_counts / max(np.sum(cur_counts), 1.0), eps, None)

    # Re-normalize after clipping so distributions remain valid probability vectors
    ref_props = ref_props / np.sum(ref_props)
    cur_props = cur_props / np.sum(cur_props)

    psi_value = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(0.0, round(psi_value, 6))


def simulate_covariate_shift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthesize a drifted production batch exhibiting strong covariate shift."""
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    if "tenure" in drifted.columns:
        # Shift customers toward shorter tenure (higher early churn risk)
        drifted["tenure"] = np.clip(np.round(drifted["tenure"] * 0.45 + rng.normal(-4, 3, len(drifted))), 0, 72)

    if "MonthlyCharges" in drifted.columns:
        # Price increase shift (+$18/month on average)
        drifted["MonthlyCharges"] = np.clip(
            np.round(drifted["MonthlyCharges"] + 18.0 + rng.normal(0, 6.0, len(drifted)), 2),
            18.0,
            140.0,
        )

    if "Contract" in drifted.columns:
        # Surge in Month-to-month contracts (78%)
        drifted["Contract"] = rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=len(drifted),
            p=[0.78, 0.12, 0.10],
        )

    if "InternetService" in drifted.columns:
        drifted["InternetService"] = rng.choice(
            ["Fiber optic", "DSL", "No"],
            size=len(drifted),
            p=[0.65, 0.22, 0.13],
        )

    return drifted


def evaluate_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    scenario_name: str,
) -> Dict[str, Any]:
    """Run PSI across all features and two-sample KS tests across numeric features."""
    feature_reports: List[Dict[str, Any]] = []
    alerts: List[str] = []

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        psi = compute_psi(reference_df[col], current_df[col])
        ks_stat: float | None = None
        ks_pvalue: float | None = None

        if col in NUMERIC_FEATURES:
            ref_num = pd.to_numeric(reference_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            cur_num = pd.to_numeric(current_df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(ref_num) > 1 and len(cur_num) > 1:
                stat, pval = ks_2samp(ref_num, cur_num)
                ks_stat = round(float(stat), 6)
                ks_pvalue = float(f"{pval:.6e}")

        if psi >= 0.25 or (ks_stat is not None and ks_pvalue is not None and ks_stat > 0.15 and ks_pvalue < 0.01):
            status = "ALERT_DRIFT"
            alerts.append(f"{col} ({status}: PSI={psi:.4f}, KS={ks_stat})")
        elif psi >= 0.10:
            status = "WARN_MODERATE_SHIFT"
            alerts.append(f"{col} ({status}: PSI={psi:.4f})")
        else:
            status = "OK"

        feature_reports.append(
            {
                "feature": col,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "status": status,
            }
        )

    max_psi = max((item["psi"] for item in feature_reports), default=0.0)
    return {
        "scenario": scenario_name,
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "max_psi": round(float(max_psi), 6),
        "alert_count": len(alerts),
        "retraining_recommended": any(item["status"] == "ALERT_DRIFT" for item in feature_reports),
        "alerts": alerts,
        "features": feature_reports,
    }


def run_drift_simulation(output_dir: Path = ARTIFACTS_DIR) -> Dict[str, Any]:
    """Compare reference training/test slice against (1) unshifted test slice and (2) shifted slice."""
    splits = load_prepared_data()
    reference_df = splits.X_train
    unshifted_current = splits.X_test
    shifted_current = simulate_covariate_shift(splits.X_test, seed=42)

    no_drift_report = evaluate_drift(reference_df, unshifted_current, scenario_name="unshifted_test_batch")
    shifted_report = evaluate_drift(reference_df, shifted_current, scenario_name="synthetic_covariate_shift")

    summary = {
        "unshifted_test_batch": no_drift_report,
        "synthetic_covariate_shift": shifted_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved drift monitoring report to %s", report_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    drift_summary = run_drift_simulation()
    print(
        json.dumps(
            {
                "unshifted_alerts": drift_summary["unshifted_test_batch"]["alerts"],
                "shifted_alerts": drift_summary["synthetic_covariate_shift"]["alerts"],
                "retraining_recommended": drift_summary["synthetic_covariate_shift"]["retraining_recommended"],
            },
            indent=2,
        )
    )