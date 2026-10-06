"""Fixed result schema and Parquet persistence for reproducible figures."""
from __future__ import annotations
import runtime as _runtime

from datetime import datetime, timezone
from pathlib import Path


RESULT_COLUMNS = (
    "config_hash", "experiment", "method", "seed", "profile", "v_kmh", "fd_hz", "ds_ns",
    "snr_db", "eta_p", "eta_sig", "eta_eff", "regime", "kappa", "d_frames", "aging",
    "Pi_H_mean", "Pi_H_std", "Pi_hat_mean", "Pi_hat_bias", "Pi_spearman", "nmse_db",
    "mse_linear", "ber_uncoded", "bler_coded", "blocks", "errors", "capped",
    "sel_latency_ms", "gphi_params", "gphi_flops", "ftheta_flops", "ce", "ce_ci_lo",
    "ce_ci_hi", "git_sha", "timestamp",
)


def normalize_result(row: dict) -> dict:
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown result columns: {sorted(unknown)}")
    result = {key: row.get(key) for key in RESULT_COLUMNS}
    result["timestamp"] = result["timestamp"] or datetime.now(timezone.utc).isoformat()
    return result


def write_parquet(path, rows):
    """Write one row per operating point. Figure code must only read this output."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # keep lightweight software tests usable
        raise RuntimeError("Parquet output requires pyarrow") from exc
    normalized = [normalize_result(row) for row in rows]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(normalized), destination)
