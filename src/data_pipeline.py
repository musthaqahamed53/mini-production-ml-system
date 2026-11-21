"""Data ingestion, cleaning, explicit leakage audit, and leak-free stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.fetch_data import DEFAULT_CSV_PATH, fetch_and_cache_dataset

TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]

NUMERIC_FEATURES: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class SplitDatasets:
    """Container for train, validation, and test splits plus customer IDs and audit report."""

    X_train: pd.DataFrame
    y_train: pd.Series
    ids_train: pd.Series

    X_val: pd.DataFrame
    y_val: pd.Series
    ids_val: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series
    ids_test: pd.Series

    leakage_audit: Dict[str, object]


def clean_raw_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Clean raw Telco churn records without fitting any data-dependent statistics.

    Returns:
        (features_df, binary_target_series, customer_id_series)
    """
    working = df.copy()
    working.columns = [col.strip() for col in working.columns]

    if TARGET_COLUMN not in working.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    customer_ids = (
        working["customerID"].astype(str).str.strip()
        if "customerID" in working.columns
        else pd.Series([f"ROW-{i}" for i in range(len(working))], index=working.index)
    )

    # Convert target 'Yes'/'No' (or 1/0) to binary integer {1, 0}
    raw_target = working[TARGET_COLUMN]
    if raw_target.dtype == object or str(raw_target.dtype) == "string":
        target_clean = raw_target.astype(str).str.strip().str.lower()
        invalid_targets = set(target_clean.unique()) - {"yes", "no"}
        if invalid_targets:
            raise ValueError(f"Unexpected target labels: {sorted(invalid_targets)}")
        y = (target_clean == "yes").astype(int)
    else:
        y = raw_target.astype(int)

    # Coerce TotalCharges to numeric (raw Telco contains whitespace strings for new customers)
    if "TotalCharges" in working.columns:
        working["TotalCharges"] = pd.to_numeric(
            working["TotalCharges"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}),
            errors="coerce",
        )

    for num_col in NUMERIC_FEATURES:
        if num_col in working.columns:
            working[num_col] = pd.to_numeric(working[num_col], errors="coerce")

    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in working.columns:
            # Preserve missing values as np.nan so SimpleImputer handles them cleanly
            series_str = working[cat_col].where(working[cat_col].notna(), other=np.nan)
            working[cat_col] = series_str.astype("object").where(
                series_str.isna(),
                series_str.astype(str).str.strip(),
            )
            working.loc[working[cat_col] == "", cat_col] = np.nan

    # Drop ID and target columns before constructing feature matrix
    X = working[FEATURE_COLUMNS].copy()
    return X, y, customer_ids


def run_leakage_audit(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    ids_train: pd.Series,
    ids_val: pd.Series,
    ids_test: pd.Series,
) -> Dict[str, object]:
    """Perform programmatic checks for split isolation, ID leakage, and target leakage."""
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    index_disjoint = (
        len(train_idx & val_idx) == 0
        and len(train_idx & test_idx) == 0
        and len(val_idx & test_idx) == 0
    )

    set_train_ids = set(ids_train)
    set_val_ids = set(ids_val)
    set_test_ids = set(ids_test)
    id_overlap_exists = (
        len(set_train_ids & set_val_ids) > 0
        or len(set_train_ids & set_test_ids) > 0
        or len(set_val_ids & set_test_ids) > 0
    )

    forbidden_cols_present = [c for c in [*ID_COLUMNS, TARGET_COLUMN] if c in X_train.columns]

    # Check if any feature in X_train is a 1:1 exact match with y_train
    exact_target_leaks: List[str] = []
    for col in X_train.columns:
        non_null = X_train[col].notna()
        if non_null.sum() > 0:
            nunique_per_class = X_train.loc[non_null].groupby(y_train.loc[non_null])[col].nunique()
            if (nunique_per_class == 1).all() and X_train[col].nunique(dropna=True) == 2:
                exact_target_leaks.append(col)

    audit_report: Dict[str, object] = {
        "split_ratios_target": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_sizes_actual": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "positive_rate_by_split": {
            "train": round(float(y_train.mean()), 4),
            "val": round(float(y_val.mean()), 4),
            "test": round(float(y_test.mean()), 4),
        },
        "index_partitions_disjoint": index_disjoint,
        "customer_id_overlap_across_splits": id_overlap_exists,
        "forbidden_id_or_target_columns_in_features": forbidden_cols_present,
        "exact_target_leak_columns": exact_target_leaks,
        "preprocessing_fit_policy": "fit_on_train_only",
        "audit_passed": (
            index_disjoint
            and not id_overlap_exists
            and len(forbidden_cols_present) == 0
            and len(exact_target_leaks) == 0
        ),
    }
    if not audit_report["audit_passed"]:
        raise RuntimeError(f"Data leakage audit failed: {audit_report}")
    return audit_report


def create_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> SplitDatasets:
    """Create a strict 70/15/15 stratified train/validation/test split.

    Mathematical justification for stratified partitioning on cross-sectional churn snapshots:
    When subscriber snapshots are cross-sectional (no time-series ordering column) with minority
    positive rate p ~ 0.265, simple random sampling produces binomial variance Var(p_hat) = p(1-p)/n
    independently in each split, risking severe class-count skew in smaller validation/test slices.
    Stratified sampling partitions each stratum (y=0, y=1) proportionally across splits, reducing
    sampling variance of class proportions and stabilizing precision/recall/threshold estimation.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0.")

    # Step 1: Split off test_ratio (15%) stratified by y
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: Split remainder (85%) into train (70% overall) and val (15% overall)
    # Relative validation fraction within train_val = 0.15 / (0.70 + 0.15) = 0.15 / 0.85
    relative_val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    audit = run_leakage_audit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ids_train=ids_train,
        ids_val=ids_val,
        ids_test=ids_test,
    )

    return SplitDatasets(
        X_train=X_train,
        y_train=y_train,
        ids_train=ids_train,
        X_val=X_val,
        y_val=y_val,
        ids_val=ids_val,
        X_test=X_test,
        y_test=y_test,
        ids_test=ids_test,
        leakage_audit=audit,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build leak-free preprocessing transformer for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def load_prepared_data(csv_path: Path = DEFAULT_CSV_PATH, random_state: int = 42) -> SplitDatasets:
    """Fetch (if needed), clean, split 70/15/15 stratified, and audit churn dataset."""
    resolved_path = fetch_and_cache_dataset(csv_path)
    raw_df = pd.read_csv(resolved_path)
    X, y, customer_ids = clean_raw_dataframe(raw_df)
    return create_stratified_splits(X, y, customer_ids, random_state=random_state)


if __name__ == "__main__":
    splits = load_prepared_data()
    print("Leakage Audit:", splits.leakage_audit)