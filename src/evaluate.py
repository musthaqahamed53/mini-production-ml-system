"""Comprehensive model evaluation, business-cost analysis, and publication-quality plotting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_pipeline import SplitDatasets, load_prepared_data
from src.train import (
    ARTIFACTS_DIR,
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    compute_expected_business_cost,
    train_models,
)

logger = logging.getLogger(__name__)


def compute_recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.70,
) -> Dict[str, float]:
    """Compute maximum achievable recall at precision >= min_precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Note: precision_recall_curve returns arrays of length n_thresholds + 1
    eligible_indices = np.where(precisions[:-1] >= min_precision)[0]
    if len(eligible_indices) == 0:
        return {
            "target_precision": float(min_precision),
            "recall_at_target_precision": 0.0,
            "actual_precision": float(np.max(precisions[:-1])) if len(precisions) > 1 else 0.0,
            "threshold_for_target_precision": 1.0,
        }
    best_idx = eligible_indices[int(np.argmax(recalls[eligible_indices]))]
    return {
        "target_precision": float(min_precision),
        "recall_at_target_precision": round(float(recalls[best_idx]), 4),
        "actual_precision": round(float(precisions[best_idx]), 4),
        "threshold_for_target_precision": round(float(thresholds[best_idx]), 4),
    }


def evaluate_model_on_split(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    min_precision: float = 0.70,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> Dict[str, Any]:
    """Compute PR-AUC, Recall@Precision, F1-macro, ROC-AUC, and business cost metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision_pos = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall_pos = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall_at_p = compute_recall_at_precision(y_true, y_prob, min_precision=min_precision)
    cost_metrics = compute_expected_business_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)

    return {
        "model": model_name,
        "threshold": round(float(threshold), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_churn": round(precision_pos, 4),
        "recall_churn": round(recall_pos, 4),
        "recall_at_precision": recall_at_p,
        "business_cost": cost_metrics,
    }


def generate_evaluation_plots(
    y_test: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
    primary_probs: np.ndarray,
    optimal_threshold: float,
    output_dir: Path,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> Dict[str, str]:
    """Generate and save Precision-Recall curve, Confusion Matrix, and Threshold Cost curve."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Precision-Recall Curve comparison
    pr_path = output_dir / "pr_curve.png"
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    for label, probs in probs_by_model.items():
        ap = average_precision_score(y_test, probs)
        PrecisionRecallDisplay.from_predictions(y_test, probs, name=f"{label} (AP={ap:.3f})", ax=ax)
    ax.axhline(y=0.70, color="crimson", linestyle="--", linewidth=1.1, label="Precision >= 0.70 Target")
    ax.set_title("Precision-Recall Curves on Held-Out Test Set")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(pr_path)
    plt.close(fig)

    # 2. Confusion Matrix at tuned threshold for primary model
    cm_path = output_dir / "confusion_matrix.png"
    y_pred_opt = (primary_probs >= optimal_threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=150)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_opt,
        display_labels=["Retained (0)", "Churned (1)"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Test Confusion Matrix (LightGBM @ threshold={optimal_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    # 3. Threshold vs. Expected Business Cost Curve
    cost_curve_path = output_dir / "threshold_cost_curve.png"
    thresholds = np.linspace(0.05, 0.95, 181)
    costs = [
        compute_expected_business_cost(y_test, (primary_probs >= t).astype(int), cost_fn=cost_fn, cost_fp=cost_fp)["total_cost"]
        for t in thresholds
    ]
    no_action_cost = compute_expected_business_cost(
        y_test, np.zeros_like(y_test), cost_fn=cost_fn, cost_fp=cost_fp
    )["total_cost"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    ax.plot(thresholds, costs, color="#1f77b4", linewidth=2.0, label="Expected Retention Cost ($)")
    ax.axhline(no_action_cost, color="#7f7f7f", linestyle=":", linewidth=1.4, label=f"No-Action Cost (${no_action_cost:,.0f})")
    ax.axvline(optimal_threshold, color="#d62728", linestyle="--", linewidth=1.4, label=f"Tuned Threshold ({optimal_threshold:.2f})")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Total Expected Cost on Test Set ($)")
    ax.set_title("Business Cost vs. Decision Threshold (Test Set)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(cost_curve_path)
    plt.close(fig)

    return {
        "pr_curve": str(pr_path),
        "confusion_matrix": str(cm_path),
        "threshold_cost_curve": str(cost_curve_path),
    }


def run_evaluation(
    splits: SplitDatasets | None = None,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Evaluate all trained models on validation and test sets and export JSON + plots."""
    if splits is None:
        splits = load_prepared_data()

    primary_path = artifacts_dir / "churn_pipeline.joblib"
    fallback_path = artifacts_dir / "fallback_pipeline.joblib"
    threshold_path = artifacts_dir / "threshold_config.json"

    if not primary_path.exists() or not fallback_path.exists() or not threshold_path.exists():
        trained = train_models(splits)
        dummy_pipe = trained["dummy_pipeline"]
        logreg_pipe = trained["logreg_pipeline"]
        lgbm_pipe = trained["lgbm_pipeline"]
        optimal_threshold = float(trained["optimal_threshold"])
        cost_fn = float(trained["cost_fn"])
        cost_fp = float(trained["cost_fp"])
    else:
        lgbm_pipe = joblib.load(primary_path)
        dummy_pipe = joblib.load(fallback_path)
        cfg = json.loads(threshold_path.read_text(encoding="utf-8"))
        optimal_threshold = float(cfg["optimal_threshold"])
        cost_fn = float(cfg.get("cost_fn", DEFAULT_COST_FN))
        cost_fp = float(cfg.get("cost_fp", DEFAULT_COST_FP))

        # Fit class-weighted LogisticRegression on training fold for side-by-side comparison
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from src.data_pipeline import build_preprocessor

        logreg_pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
            ]
        )
        logreg_pipe.fit(splits.X_train, splits.y_train.to_numpy())

    y_val = splits.y_val.to_numpy()
    y_test = splits.y_test.to_numpy()

    val_probs_lgbm = lgbm_pipe.predict_proba(splits.X_val)[:, 1]
    test_probs_dummy = dummy_pipe.predict_proba(splits.X_test)[:, 1]
    test_probs_logreg = logreg_pipe.predict_proba(splits.X_test)[:, 1]
    test_probs_lgbm = lgbm_pipe.predict_proba(splits.X_test)[:, 1]

    report: Dict[str, Any] = {
        "cost_model": {"cost_fn_usd": cost_fn, "cost_fp_usd": cost_fp},
        "validation_primary_model": evaluate_model_on_split(
            "LightGBM (Tuned)", y_val, val_probs_lgbm, threshold=optimal_threshold, cost_fn=cost_fn, cost_fp=cost_fp
        ),
        "test_models": {
            "dummy_prior": evaluate_model_on_split(
                "DummyClassifier (Prior)", y_test, test_probs_dummy, threshold=optimal_threshold, cost_fn=cost_fn, cost_fp=cost_fp
            ),
            "logistic_regression_balanced": evaluate_model_on_split(
                "LogisticRegression (Balanced)", y_test, test_probs_logreg, threshold=0.50, cost_fn=cost_fn, cost_fp=cost_fp
            ),
            "lightgbm_tuned": evaluate_model_on_split(
                "LightGBM (Tuned + Cost Threshold)", y_test, test_probs_lgbm, threshold=optimal_threshold, cost_fn=cost_fn, cost_fp=cost_fp
            ),
        },
    }

    report_path = artifacts_dir / "metrics_report.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_paths = generate_evaluation_plots(
        y_test=y_test,
        probs_by_model={
            "Dummy (Prior)": test_probs_dummy,
            "LogReg (Balanced)": test_probs_logreg,
            "LightGBM (Tuned)": test_probs_lgbm,
        },
        primary_probs=test_probs_lgbm,
        optimal_threshold=optimal_threshold,
        output_dir=artifacts_dir,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )

    logger.info("Saved evaluation metrics to %s", report_path)
    return report, plot_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    eval_report, plots = run_evaluation()
    print(json.dumps({"metrics_report": eval_report, "plots": plots}, indent=2))