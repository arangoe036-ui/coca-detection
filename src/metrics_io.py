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
from datetime import UTC, datetime
from pathlib import Path

JSONL_NAME = "baseline_ladder.jsonl"


def git_commit() -> str:
    """Short HEAD sha for the run's provenance, or ``"unknown"``.

    Degrading to ``"unknown"`` is correct rather than fatal: metrics must still be
    writable from a source tarball, a container without git, or a fresh clone
    before the first commit. But the fallback is narrowed to the failures that can
    actually occur, so a genuine bug in this function surfaces instead of being
    silently relabelled as "no git here":

    * ``OSError`` (incl. ``FileNotFoundError``) — git not installed or not on PATH;
    * ``subprocess.SubprocessError`` (incl. ``CalledProcessError``) — not a repo,
      or no commits yet, so ``rev-parse`` exits non-zero.

    Note this guards **provenance only**. The append in ``write_run`` is
    deliberately unguarded: prereg §8 makes this the single sink every reported
    number must trace to, so a failed write must propagate, never be swallowed.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
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
        # Which generation of the DATA this was computed on. Three now exist and they are
        # not comparable: gen1 carried a reflectance offset bug, 25% blank coverage in
        # 2019/2020 AND a train/test pixel overlap; gen2 fixed only the offset. Without
        # this field a gen1 and a gen3 number look identical in the file.
        "data_generation": str(cfg["project"]["data_generation"]),
        "method": method,
        "track": track,
        "fold_year": int(fold_year),
        "n_train_tiles": int(n_train_tiles),
        "n_test_tiles": int(n_test_tiles),
        "calibration_scalar": (None if calibration_scalar is None
                               else float(calibration_scalar)),
        "fit_years": [int(y) for y in fit_years],
        "git_commit": git_commit(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
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
