"""Model training, class-imbalance weighting, business-cost threshold tuning, and serialization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.data_pipeline import SplitDatasets, build_preprocessor, load_prepared_data

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_VERSION = "churn-lgbm-v1.0.0"

# Explicit retention economics (USD per subscriber):
# False Negative (missed churner): lose expected net contribution / retention value ($500)
# False Positive (unnecessary retention incentive offered to non-churner): discount/outreach cost ($50)
DEFAULT_COST_FN: float = 250.0
DEFAULT_COST_FP: float = 60.0


def compute_expected_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> Dict[str, float]:
    """Compute total and per-customer expected churn intervention cost."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_cost = float(fn * cost_fn + fp * cost_fp)
    n_customers = max(int(len(y_true)), 1)
    # Baseline cost if no retention intervention is sent to anyone (predict all 0 -> all actual positives are FN)
    no_action_cost = float(np.sum(y_true == 1) * cost_fn)
    net_savings = float(no_action_cost - total_cost)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": round(total_cost, 2),
        "cost_per_customer": round(total_cost / n_customers, 4),
        "no_action_baseline_cost": round(no_action_cost, 2),
        "net_cost_savings": round(net_savings, 2),
    }


def optimize_threshold_for_business_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> Tuple[float, Dict[str, Any]]:
    """Grid-search decision threshold on validation probabilities to minimize expected business cost."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.50
    best_cost = float("inf")
    best_stats: Dict[str, Any] = {}

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        cost_info = compute_expected_business_cost(y_true, preds, cost_fn=cost_fn, cost_fp=cost_fp)
        if cost_info["total_cost"] < best_cost:
            best_cost = cost_info["total_cost"]
            best_threshold = float(round(float(thresh), 4))
            best_stats = {
                **cost_info,
                "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
            }

    return best_threshold, best_stats


def train_models(
    splits: SplitDatasets,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train baselines and tuned LightGBM classifier; tune threshold on validation fold."""
    y_train = splits.y_train.to_numpy()
    y_val = splits.y_val.to_numpy()

    neg_count = int(np.sum(y_train == 0))
    pos_count = max(int(np.sum(y_train == 1)), 1)
    imbalance_ratio = float(neg_count / pos_count)

    # 1. Baseline A: DummyClassifier (prior distribution)
    dummy_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="prior", random_state=random_state)),
        ]
    )
    dummy_pipe.fit(splits.X_train, y_train)

    # 2. Baseline B: Class-weighted Logistic Regression
    logreg_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    logreg_pipe.fit(splits.X_train, y_train)

    # 3. Tuned LightGBM across candidate hyperparameters using validation PR-AUC & business cost
    candidate_configs = [
        {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30, "subsample": 0.8},
        {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 23, "min_child_samples": 25, "subsample": 0.85},
        {"n_estimators": 200, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.9},
    ]

    best_lgbm_pipe: Pipeline | None = None
    best_val_score = -float("inf")
    best_params: Dict[str, Any] = {}

    for cfg in candidate_configs:
        candidate_pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LGBMClassifier(
                        objective="binary",
                        scale_pos_weight=imbalance_ratio,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                        **cfg,
                    ),
                ),
            ]
        )
        candidate_pipe.fit(splits.X_train, y_train)
        val_probs = candidate_pipe.predict_proba(splits.X_val)[:, 1]
        val_pr_auc = float(average_precision_score(y_val, val_probs))
        _, val_cost_stats = optimize_threshold_for_business_cost(
            y_val, val_probs, cost_fn=cost_fn, cost_fp=cost_fp
        )
        # Composite selection score: PR-AUC + normalized net cost savings on validation set
        no_action_val = max(val_cost_stats["no_action_baseline_cost"], 1.0)
        normalized_savings = val_cost_stats["net_cost_savings"] / no_action_val
        selection_score = val_pr_auc + 0.25 * normalized_savings

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_lgbm_pipe = candidate_pipe
            best_params = {
                **cfg,
                "scale_pos_weight": round(imbalance_ratio, 4),
                "val_pr_auc": round(val_pr_auc, 4),
                "val_selection_score": round(selection_score, 4),
            }

    assert best_lgbm_pipe is not None
    val_lgbm_probs = best_lgbm_pipe.predict_proba(splits.X_val)[:, 1]
    optimal_threshold, val_opt_stats = optimize_threshold_for_business_cost(
        y_val, val_lgbm_probs, cost_fn=cost_fn, cost_fp=cost_fp
    )

    return {
        "dummy_pipeline": dummy_pipe,
        "logreg_pipeline": logreg_pipe,
        "lgbm_pipeline": best_lgbm_pipe,
        "best_params": best_params,
        "optimal_threshold": optimal_threshold,
        "val_threshold_stats": val_opt_stats,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
    }


def save_artifacts(trained: Dict[str, Any], splits: SplitDatasets, output_dir: Path = ARTIFACTS_DIR) -> Dict[str, Path]:
    """Serialize primary model, fallback baseline model, and threshold/version metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_path = output_dir / "churn_pipeline.joblib"
    fallback_path = output_dir / "fallback_pipeline.joblib"
    threshold_path = output_dir / "threshold_config.json"
    audit_path = output_dir / "leakage_audit.json"

    joblib.dump(trained["lgbm_pipeline"], primary_path)
    joblib.dump(trained["dummy_pipeline"], fallback_path)

    threshold_meta = {
        "model_version": MODEL_VERSION,
        "model_family": "LightGBM",
        "optimal_threshold": trained["optimal_threshold"],
        "cost_fn": trained["cost_fn"],
        "cost_fp": trained["cost_fp"],
        "best_hyperparameters": trained["best_params"],
        "validation_threshold_summary": trained["val_threshold_stats"],
        "risk_tiers": {
            "low_max_exclusive": round(trained["optimal_threshold"] * 0.65, 4),
            "medium_max_exclusive": trained["optimal_threshold"],
            "high_min_inclusive": trained["optimal_threshold"],
        },
    }
    threshold_path.write_text(json.dumps(threshold_meta, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(splits.leakage_audit, indent=2), encoding="utf-8")

    logger.info("Saved primary model to %s", primary_path)
    logger.info("Saved fallback model to %s", fallback_path)
    logger.info("Saved threshold config (threshold=%.4f) to %s", trained["optimal_threshold"], threshold_path)

    return {
        "primary_model": primary_path,
        "fallback_model": fallback_path,
        "threshold_config": threshold_path,
        "leakage_audit": audit_path,
    }


def run_training_pipeline(output_dir: Path = ARTIFACTS_DIR) -> Dict[str, Any]:
    """End-to-end training entrypoint."""
    splits = load_prepared_data()
    trained = train_models(splits)
    saved_paths = save_artifacts(trained, splits, output_dir=output_dir)
    return {
        "optimal_threshold": trained["optimal_threshold"],
        "best_params": trained["best_params"],
        "artifacts": {k: str(v) for k, v in saved_paths.items()},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    summary = run_training_pipeline()
    print(json.dumps(summary, indent=2))