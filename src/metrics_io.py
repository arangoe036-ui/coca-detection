"""P6-A Phase 1 — single append-only metrics writer for the baseline ladder.

Every method (ndvi_threshold | random_forest | unet | null_*) writes ONE JSON
object per fold to ``outputs/metrics/baseline_ladder.jsonl`` through ``write_run``.
No metric may live only in stdout or prose (prereg §8): this is the one sink.

Each record carries provenance (git commit, timestamp, fit years, calibration
scalar, tile counts, track) plus a ``metrics`` block whose contents come from the
existing ``src/evaluate.py`` helpers — this module writes no metric math itself.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

JSONL_NAME = "baseline_ladder.jsonl"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def metrics_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["outputs_dir"]) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run(cfg: dict, method: str, fold_year: int, metrics: dict,
              *, track: str, n_train_tiles: int, n_test_tiles: int,
              calibration_scalar: float | None, fit_years: list[int],
              extra: dict | None = None) -> Path:
    """Append one run record to outputs/metrics/baseline_ladder.jsonl.

    track: "A" (spatial, test blocks) | "B" (temporal, LOYO aoi_ratio).
    """
    out = metrics_dir(cfg)
    rec = {
        "method": method,
        "track": track,
        "fold_year": int(fold_year),
        "n_train_tiles": int(n_train_tiles),
        "n_test_tiles": int(n_test_tiles),
        "calibration_scalar": (None if calibration_scalar is None
                               else float(calibration_scalar)),
        "fit_years": [int(y) for y in fit_years],
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: (None if v is None else float(v)) for k, v in metrics.items()},
    }
    if extra:
        rec["extra"] = extra
    path = out / JSONL_NAME
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return path


def read_runs(cfg: dict) -> list[dict]:
    path = metrics_dir(cfg) / JSONL_NAME
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
