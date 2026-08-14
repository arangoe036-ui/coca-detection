"""Guard against silent config drift.

Three real defects came from config keys nothing validated: ``scl_mask_classes``
omitting SCL 0/1 (nodata treated as valid, blanking 25% of the 2019/2020 AOI),
``cloud_cover_max: 40`` starving cloudy regions of scenes, and
``encoder_weights: ssl4eo`` silently routing to a no-op.

APPROACH: an explicit DECLARED registry, not a general dataflow analysis. Config
is read through aliases (``tc = cfg["tiling"]``), loops
(``for key in (...): cfg["paths"].get(key)``) and computed field names, so an AST
tracer would be both fragile and invasive. Instead every leaf key in
``config/default.yaml`` must be listed in exactly one of the two dicts below, and
each entry is cross-checked against the source:

* ``CONFIG_KEYS_READ`` — the key's name must appear as a string literal somewhere
  under ``src/`` or ``scripts/`` (i.e. some code does name it);
* ``CONFIG_KEYS_UNUSED`` — the key's name must NOT appear anywhere, so the moment
  someone wires it up this test fails and the registry has to be updated.

THIS REGISTRY MUST BE UPDATED WHENEVER config/default.yaml CHANGES.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from src.utils import REPO_ROOT

# key -> a module that reads it (one representative reader; several may read it).
CONFIG_KEYS_READ: dict[str, str] = {
    "project.name": "src/infer.py",
    "project.seed": "src/train.py",
    "project.data_generation": "src/metrics_io.py",
    "year": "src/data/labels.py",
    "aoi.region": "src/data/stac_export.py",
    "aoi.bbox": "src/data/multiyear.py",
    "aoi.utm_epsg": "src/data/stac_export.py",
    "aoi.quick_bbox": "src/data/stac_export.py",
    "imagery.backend": "src/data/stac_export.py",
    "imagery.stac_url": "src/data/stac_export.py",
    "imagery.s2_collection": "src/data/stac_export.py",
    "imagery.s1_collection": "src/data/stac_export.py",
    "imagery.s2_bands": "src/data/stac_export.py",
    "imagery.s1_bands": "src/data/stac_export.py",
    "imagery.indices": "src/data/stac_export.py",
    "imagery.cloud_cover_max": "src/data/stac_export.py",
    "imagery.scl_mask_classes": "src/data/stac_export.py",
    "imagery.n_seasonal_composites": "src/data/stac_export.py",
    "imagery.resolution_m": "src/data/tiling.py",
    "imagery.full_subtile_deg": "src/data/stac_export.py",
    "imagery.cropland_mask": "src/data/masks.py",
    "labels.socrata_domain": "src/data/labels.py",
    "labels.coca_grid_resource_id": "src/data/labels.py",
    "labels.coca_grid_year_field": "src/data/labels.py",
    "labels.coca_grid_geom_field": "src/data/labels.py",
    "labels.coca_grid_id_field": "src/data/labels.py",
    "labels.validation_resource_id": "src/data/labels.py",
    "labels.validation_year_field": "src/data/labels.py",
    "labels.boundaries_url": "src/infer.py",
    "labels.mask_type": "src/data/labels.py",
    "labels.presence_threshold_ha": "src/data/labels.py",
    "tiling.tile_px": "src/data/tiling.py",
    "tiling.stride_px": "src/data/tiling.py",
    "tiling.split_block_km": "src/data/tiling.py",
    "tiling.split_ratios.train": "src/data/tiling.py",
    "tiling.split_ratios.val": "src/data/tiling.py",
    "tiling.split_ratios.test": "src/data/tiling.py",
    "tiling.oversample_positive": "src/data/dataset.py",
    "model.task": "src/train.py",
    "model.encoder": "src/models/unet.py",
    "model.encoder_weights": "src/models/unet.py",
    "model.in_channels": "src/models/unet.py",
    "model.classes": "src/models/unet.py",
    "loss.type": "src/models/losses.py",
    "loss.dice_weight": "src/models/losses.py",
    "loss.focal_weight": "src/models/losses.py",
    "loss.focal_gamma": "src/models/losses.py",
    "loss.bce_pos_weight": "src/models/losses.py",
    "train.epochs": "src/train.py",
    "train.batch_size": "src/data/dataset.py",
    "train.lr": "src/train.py",
    "train.early_stop_patience": "src/train.py",
    "train.overfit_sanity_tiles": "src/train.py",
    "eval.threshold": "src/infer.py",
    "eval.metrics": "src/baselines/compare.py",
    "infer.window_px": "src/infer.py",
    "infer.window_overlap_px": "src/infer.py",
    "infer.morph_kernel_px": "src/infer.py",
    "infer.min_polygon_ha": "src/infer.py",
    "infer.presence_fraction": "src/infer.py",
    "infer.gate_threshold": "src/infer.py",
    "infer.calibration_path": "src/infer.py",
    "paths.data_dir": "src/utils.py",
    "paths.outputs_dir": "src/evaluate.py",
    "paths.tiles_dir": "src/data/tiling.py",
    "paths.imagery_dir": "src/data/stac_export.py",
    "paths.labels_dir": "src/data/labels.py",
    "paths.checkpoints_dir": "src/train.py",
    "paths.ui_data_dir": "src/infer.py",
}

# Keys that NO Python code reads. Declared deliberately, with why, so that an
# unused key is a documented decision instead of an assumption.
CONFIG_KEYS_UNUSED: dict[str, str] = {
    "imagery.composite_reducer": "stac_export always takes the median; 'geomedian' unimplemented",
    "imagery.export_tile_px": "stac_export sizes sub-tiles from full_subtile_deg instead",
    "imagery.export_overlap_px": "stac_export sub-tiles do not overlap",
    "labels.min_mapping_unit_px": "speck removal not implemented in rasterize_mask",
    "model.arch": "only the U-Net exists; models/unet.py hardcodes the architecture",
    "train.optimizer": "train.py hardcodes AdamW (matches the value, does not read it)",
    "train.scheduler": "train.py hardcodes CosineAnnealingLR",
    "train.amp": "mixed precision is not implemented (no autocast/GradScaler anywhere)",
    "train.early_stop_metric": "train.py hardcodes early stopping on val IoU",
    "train.num_workers": "DataLoaders hardcode num_workers=0 (Windows); config says 4",
    "infer.admin_boundaries": "infer.py always uses the GADM boundaries_url",
    "ui.engine": "the map page is hand-written Leaflet",
    "ui.basemap": "basemap is hardcoded in the UI page",
    "ui.mapbox_token": "unused while the basemap is Esri World Imagery",
}


def _leaf_keys(node, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_keys(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1]


def _source_string_literals() -> set[str]:
    """Every string literal in src/ and scripts/ — 'the code at least names this'."""
    literals: set[str] = set()
    for py in [*(REPO_ROOT / "src").rglob("*.py"), *(REPO_ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                literals.add(node.value)
    return literals


def _config_leaves() -> set[str]:
    raw = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    return set(_leaf_keys(raw))


def test_config_has_no_unknown_keys():
    """Every config key is declared, and every declared key exists in the config."""
    declared = set(CONFIG_KEYS_READ) | set(CONFIG_KEYS_UNUSED)
    present = _config_leaves()

    undeclared = sorted(present - declared)
    assert not undeclared, ("config/default.yaml has keys nothing declares (add them to "
                            f"CONFIG_KEYS_READ or CONFIG_KEYS_UNUSED): {undeclared}")
    missing = sorted(declared - present)
    assert not missing, f"declared config keys absent from config/default.yaml: {missing}"
    assert not (set(CONFIG_KEYS_READ) & set(CONFIG_KEYS_UNUSED))


def test_declared_read_keys_are_named_in_source():
    """Each key declared as read is at least mentioned by src/ or scripts/ code."""
    literals = _source_string_literals()
    orphans = sorted(k for k in CONFIG_KEYS_READ if k.split(".")[-1] not in literals)
    assert not orphans, f"declared as read but never named in the source: {orphans}"


def test_declared_unused_keys_are_still_unused():
    """If an 'unused' key gets wired up, the registry must be updated (this fails)."""
    literals = _source_string_literals()
    now_used = sorted(k for k in CONFIG_KEYS_UNUSED if k.split(".")[-1] in literals)
    assert not now_used, ("these keys are declared unused but now appear in the source — "
                          f"move them to CONFIG_KEYS_READ: {now_used}")


def test_config_load_still_derives_in_channels():
    """load_config computes model.in_channels from the band lists (a silent-drift spot)."""
    from src.utils import load_config

    cfg = load_config()
    img = cfg["imagery"]
    assert cfg["model"]["in_channels"] == (len(img["s2_bands"]) + len(img["indices"])
                                          + len(img["s1_bands"]))


def test_scl_mask_includes_nodata_and_saturated():
    """Regression guard on the A13 fix: SCL 0/1 must stay masked."""
    raw = yaml.safe_load((REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert {0, 1} <= set(raw["imagery"]["scl_mask_classes"])


def test_path_keys_match_ensure_dirs():
    """Every paths.* key is one ensure_dirs() creates (a missing dir fails mid-export)."""
    src = (REPO_ROOT / "src" / "utils.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    ensure = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "ensure_dirs")
    created = {c.value for n in ast.walk(ensure) if isinstance(n, ast.Tuple)
               for c in n.elts if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    declared = {k.split(".")[-1] for k in CONFIG_KEYS_READ if k.startswith("paths.")}
    assert declared == created, f"paths keys {sorted(declared)} vs ensure_dirs {sorted(created)}"
