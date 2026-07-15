"""Shared utilities: config loading, seeding, path helpers.

This module is intentionally dependency-light (only PyYAML) so that
``config loads`` — the P0 acceptance check — passes before the heavy ML/geo
stack is installed.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load a YAML config, compute derived fields, and resolve paths.

    Derived: ``model.in_channels`` = #S2 bands + #indices + #S1 bands.
    Paths under the ``paths`` block are resolved to absolute paths rooted at
    the repo root (so scripts can be run from anywhere).
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Derived: input channel count for the model.
    img = cfg.get("imagery", {})
    n_channels = (
        len(img.get("s2_bands", []))
        + len(img.get("indices", []))
        + len(img.get("s1_bands", []))
    )
    cfg.setdefault("model", {})
    if cfg["model"].get("in_channels") in (None, "null"):
        cfg["model"]["in_channels"] = n_channels

    # Resolve declared paths to absolute (rooted at repo root).
    resolved: dict[str, str] = {}
    for key, rel in cfg.get("paths", {}).items():
        resolved[key] = str((REPO_ROOT / rel).resolve())
    cfg["paths"] = {**cfg.get("paths", {}), **resolved}

    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/torch RNGs when those libs are importable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # numpy/torch are optional at P0
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create the output/data directories declared in the config."""
    for key in ("data_dir", "outputs_dir", "tiles_dir", "imagery_dir",
                "labels_dir", "checkpoints_dir", "ui_data_dir"):
        p = cfg.get("paths", {}).get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # `python src/utils.py` prints a loaded-config summary — the P0 smoke test.
    c = load_config()
    print(f"[ok] config loaded: {c['project']['name']}")
    print(f"     region={c['aoi']['region']}  year={c['year']}")
    print(f"     model.in_channels={c['model']['in_channels']} (S2+indices+S1)")
    print(f"     repo_root={REPO_ROOT}")
