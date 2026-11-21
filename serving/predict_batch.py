"""Production CLI batch scoring script for vectorized customer churn inference."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from serving.app import classify_risk_tier
from src.data_pipeline import FEATURE_COLUMNS, clean_raw_dataframe
from src.train import ARTIFACTS_DIR, MODEL_VERSION, run_training_pipeline

logger = logging.getLogger(__name__)


def score_batch_dataframe(
    raw_df: pd.DataFrame,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate input DataFrame, execute vectorized inference, and return scored DataFrame + stats."""
    start_perf = time.perf_counter()

    if "customerID" not in raw_df.columns:
        raise ValueError("Input CSV must contain a 'customerID' column.")

    missing_features = [col for col in FEATURE_COLUMNS if col not in raw_df.columns]
    if missing_features:
        raise ValueError(f"Input CSV missing required feature columns: {sorted(missing_features)}")

    primary_path = artifacts_dir / "churn_pipeline.joblib"
    fallback_path = artifacts_dir / "fallback_pipeline.joblib"
    config_path = artifacts_dir / "threshold_config.json"

    if not primary_path.exists() or not fallback_path.exists():
        run_training_pipeline(artifacts_dir)

    threshold = 0.50
    model_version = MODEL_VERSION
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        threshold = float(cfg.get("optimal_threshold", 0.50))
        model_version = str(cfg.get("model_version", MODEL_VERSION))

    # Prepare features cleanly while preserving customer IDs
    if "Churn" not in raw_df.columns:
        temp_df = raw_df.copy()
        temp_df["Churn"] = "No"
        X_clean, _, customer_ids = clean_raw_dataframe(temp_df)
    else:
        X_clean, _, customer_ids = clean_raw_dataframe(raw_df)

    degraded_mode = False
    model_source = "primary_lightgbm"

    try:
        pipeline = joblib.load(primary_path)
        probs = np.clip(pipeline.predict_proba(X_clean)[:, 1], 0.0, 1.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Primary batch scoring failed (%s); falling back to baseline model.", exc)
        degraded_mode = True
        model_source = "fallback_dummy_prior_model"
        fallback_pipe = joblib.load(fallback_path)
        probs = np.clip(fallback_pipe.predict_proba(X_clean)[:, 1], 0.0, 1.0)

    preds = (probs >= threshold).astype(int)
    tiers = [classify_risk_tier(float(p), threshold) for p in probs]

    output_df = pd.DataFrame(
        {
            "customerID": customer_ids.to_numpy(),
            "churn_probability": np.round(probs, 6),
            "churn_prediction": preds,
            "risk_tier": tiers,
            "threshold_used": threshold,
            "model_source": model_source,
            "model_version": model_version,
            "degraded_mode": degraded_mode,
        }
    )

    elapsed_sec = max(time.perf_counter() - start_perf, 1e-6)
    throughput_rps = len(output_df) / elapsed_sec

    stats: Dict[str, Any] = {
        "rows_scored": int(len(output_df)),
        "predicted_churn_count": int(np.sum(preds == 1)),
        "predicted_churn_rate": round(float(np.mean(preds == 1)), 4),
        "elapsed_seconds": round(elapsed_sec, 4),
        "throughput_records_per_sec": round(throughput_rps, 2),
        "degraded_mode": degraded_mode,
        "model_source": model_source,
    }
    return output_df, stats


def run_batch_cli(input_csv: Path, output_csv: Path, artifacts_dir: Path = ARTIFACTS_DIR) -> Dict[str, Any]:
    """Read input CSV, run batch inference, write output CSV, and log execution stats."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw_df = pd.read_csv(input_csv)
    scored_df, stats = score_batch_dataframe(raw_df, artifacts_dir=artifacts_dir)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_csv, index=False)
    stats["output_csv"] = str(output_csv)

    logger.info(
        "Batch scoring complete | rows=%d | churn_pred=%d | elapsed=%.3fs | throughput=%.1f rec/s | output=%s",
        stats["rows_scored"],
        stats["predicted_churn_count"],
        stats["elapsed_seconds"],
        stats["throughput_records_per_sec"],
        output_csv,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized CLI batch scorer for customer churn.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to input customer CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to output predictions CSV.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR, help="Directory with model artifacts.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stats = run_batch_cli(args.input_csv, args.output_csv, artifacts_dir=args.artifacts_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()