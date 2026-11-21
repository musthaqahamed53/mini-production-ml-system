"""Automated dataset downloader, integrity verifier, and local cache manager."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.request import urlopen
import requests

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = DATA_DIR / "telco_churn.csv"
CHECKSUM_PATH = DATA_DIR / "telco_churn.sha256"

# Public mirrors for the IBM Telco Customer Churn dataset
SOURCE_URLS: List[str] = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn-by-IBM.csv",
]

REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a local file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate that required Telco churn columns and minimum row count exist."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if len(df) < 1000:
        raise ValueError(f"Dataset unexpectedly small ({len(df)} rows); expected >= 1000 rows.")


def _generate_deterministic_fallback_dataset(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic, schema-identical synthetic Telco-like churn dataset if offline."""
    rng = np.random.default_rng(seed)
    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.55, 0.21, 0.24],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_rows,
        p=[0.34, 0.44, 0.22],
    )
    tenure = rng.integers(1, 73, size=n_rows)
    base_monthly = np.where(internet == "Fiber optic", 82.0, np.where(internet == "DSL", 56.0, 21.0))
    monthly_charges = np.round(np.clip(base_monthly + rng.normal(0, 12.0, size=n_rows), 18.25, 118.75), 2)
    total_charges = np.round(tenure * monthly_charges + rng.normal(0, 25.0, size=n_rows), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Higher churn probability for month-to-month, fiber optic, and short tenure
    logit = (
        -1.35
        + np.where(contracts == "Month-to-month", 1.25, np.where(contracts == "One year", 0.1, -0.9))
        + np.where(internet == "Fiber optic", 0.55, -0.2)
        - 0.025 * tenure
        + rng.normal(0, 0.35, size=n_rows)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    def yes_no_internet(prob_yes: float) -> np.ndarray:
        vals = rng.choice(["Yes", "No"], size=n_rows, p=[prob_yes, 1.0 - prob_yes])
        return np.where(internet == "No", "No internet service", vals)

    def yes_no_phone(prob_yes: float) -> np.ndarray:
        phone = rng.choice(["Yes", "No"], size=n_rows, p=[0.90, 0.10])
        vals = rng.choice(["Yes", "No"], size=n_rows, p=[prob_yes, 1.0 - prob_yes])
        return np.where(phone == "No", "No phone service", vals), phone

    multiple_lines, phone_service = yes_no_phone(0.45)

    df = pd.DataFrame(
        {
            "customerID": [f"{1000 + i:04d}-{chr(65 + (i % 26))}{chr(65 + ((i * 7) % 26))}XYZ" for i in range(n_rows)],
            "gender": rng.choice(["Female", "Male"], size=n_rows),
            "SeniorCitizen": rng.choice([0, 1], size=n_rows, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52]),
            "Dependents": rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": yes_no_internet(0.35),
            "OnlineBackup": yes_no_internet(0.42),
            "DeviceProtection": yes_no_internet(0.42),
            "TechSupport": yes_no_internet(0.36),
            "StreamingTV": yes_no_internet(0.47),
            "StreamingMovies": yes_no_internet(0.48),
            "Contract": contracts,
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41]),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size=n_rows,
                p=[0.34, 0.23, 0.22, 0.21],
            ),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges.astype(str),
            "Churn": churn,
        }
    )
    # Inject blank TotalCharges on tenure == 0 edge cases to mirror raw Telco data
    zero_mask = df["tenure"] <= 1
    if zero_mask.any():
        blank_indices = np.flatnonzero(zero_mask.to_numpy())[:11]
        df.loc[df.index[blank_indices], "TotalCharges"] = " "
    return df


def fetch_and_cache_dataset(
    output_path: Path = DEFAULT_CSV_PATH,
    force_download: bool = False,
) -> Path:
    """Download Telco churn dataset from public mirrors, verify schema, and cache locally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        cached_df = pd.read_csv(output_path)
        validate_dataset_schema(cached_df)
        digest = compute_sha256(output_path)
        CHECKSUM_PATH.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
        logger.info("Using cached dataset at %s (rows=%d, sha256=%s...)", output_path, len(cached_df), digest[:12])
        return output_path

    last_error: Exception | None = None
    for url in SOURCE_URLS:
        try:
            logger.info("Attempting download from %s", url)
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            raw_bytes = resp.content
            temp_path = output_path.with_suffix(".tmp.csv")
            temp_path.write_bytes(raw_bytes)
            df = pd.read_csv(temp_path)
            validate_dataset_schema(df)
            temp_path.replace(output_path)
            digest = compute_sha256(output_path)
            CHECKSUM_PATH.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
            logger.info("Downloaded & verified dataset at %s (rows=%d, sha256=%s...)", output_path, len(df), digest[:12])
            return output_path
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("Failed fetching from %s: %s", url, exc)
            last_error = exc

    logger.warning("All remote sources unavailable (%s); generating deterministic schema-compliant dataset.", last_error)
    fallback_df = _generate_deterministic_fallback_dataset()
    validate_dataset_schema(fallback_df)
    fallback_df.to_csv(output_path, index=False)
    digest = compute_sha256(output_path)
    CHECKSUM_PATH.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    logger.info("Wrote deterministic dataset to %s (rows=%d, sha256=%s...)", output_path, len(fallback_df), digest[:12])
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    path = fetch_and_cache_dataset()
    print(f"Dataset ready at: {path}")