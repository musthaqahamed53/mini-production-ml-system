"""Empirical latency (p50, p95, p99) and batch throughput (records/sec) profiler."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from data.fetch_data import DEFAULT_CSV_PATH, fetch_and_cache_dataset
from serving.predict_batch import score_batch_dataframe

SAMPLE_PAYLOAD: Dict[str, Any] = {
    "customer_id": "BENCH-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.50,
    "TotalCharges": 1014.00,
}


def benchmark_api_latency(
    base_url: str = "http://127.0.0.1:8000",
    num_requests: int = 250,
    warmup_requests: int = 20,
    timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """Send warmup + measured requests to POST /predict and compute p50, p95, p99 latencies."""
    if num_requests < 200:
        raise ValueError("num_requests must be >= 200 to satisfy empirical benchmark requirements.")

    predict_url = f"{base_url.rstrip('/')}/predict"
    session = requests.Session()

    for i in range(warmup_requests):
        payload = {**SAMPLE_PAYLOAD, "customer_id": f"WARMUP-{i:04d}"}
        resp = session.post(predict_url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()

    latencies_ms: List[float] = []
    status_codes: Dict[int, int] = {}

    wall_start = time.perf_counter()
    for i in range(num_requests):
        payload = {**SAMPLE_PAYLOAD, "customer_id": f"BENCH-{i:04d}"}
        t0 = time.perf_counter()
        resp = session.post(predict_url, json=payload, timeout=timeout_sec)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        code = resp.status_code
        status_codes[code] = status_codes.get(code, 0) + 1
        resp.raise_for_status()

    wall_elapsed = max(time.perf_counter() - wall_start, 1e-6)
    arr = np.asarray(latencies_ms, dtype=float)

    return {
        "endpoint": predict_url,
        "measured_requests": int(num_requests),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "api_throughput_rps": round(float(num_requests / wall_elapsed), 2),
        "http_status_counts": status_codes,
    }


def benchmark_batch_throughput(csv_path: Path = DEFAULT_CSV_PATH) -> Dict[str, Any]:
    """Measure vectorized batch inference throughput in records/sec."""
    resolved_csv = fetch_and_cache_dataset(csv_path)
    df = pd.read_csv(resolved_csv)
    _, stats = score_batch_dataframe(df)
    return stats


def print_summary_table(api_stats: Dict[str, Any], batch_stats: Dict[str, Any]) -> None:
    """Print a clean tabular summary of API latency percentiles and batch throughput."""
    print("\n=== Empirical Performance Benchmark Summary ===")
    print(f"{'Workload':<34} | {'Count':>7} | {'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | {'Throughput':>14}")
    print("-" * 96)
    print(
        f"{'Online POST /predict (HTTP)':<34} | "
        f"{api_stats['measured_requests']:>7} | "
        f"{api_stats['p50_ms']:>10.2f} | "
        f"{api_stats['p95_ms']:>10.2f} | "
        f"{api_stats['p99_ms']:>10.2f} | "
        f"{str(api_stats['api_throughput_rps']) + ' req/s':>14}"
    )
    batch_latency_ms = batch_stats["elapsed_seconds"] * 1000.0
    print(
        f"{'Batch CLI Vectorized Scoring':<34} | "
        f"{batch_stats['rows_scored']:>7} | "
        f"{batch_latency_ms:>10.2f} | "
        f"{'N/A':>10} | "
        f"{'N/A':>10} | "
        f"{str(batch_stats['throughput_records_per_sec']) + ' rec/s':>14}"
    )
    print("-" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FastAPI churn endpoint and batch throughput.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL of running FastAPI service.")
    parser.add_argument("--requests", type=int, default=250, help="Number of measured API requests (>= 200).")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV_PATH, help="CSV used for batch throughput.")
    args = parser.parse_args()

    batch_stats = benchmark_batch_throughput(args.input_csv)
    api_stats = benchmark_api_latency(base_url=args.url, num_requests=args.requests)
    print_summary_table(api_stats, batch_stats)
    print(json.dumps({"api": api_stats, "batch": batch_stats}, indent=2))


if __name__ == "__main__":
    main()