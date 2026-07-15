"""Optional cropland mask to restrict inference to agricultural pixels (plan §6.1).

Sources: ESA WorldCover v200 or Google Dynamic World. Enabled via
``imagery.cropland_mask`` in the config (null = disabled).
"""

from __future__ import annotations


def build_cropland_mask(cfg: dict):
    """Return a boolean cropland mask on the imagery grid, or None if disabled."""
    if not cfg.get("imagery", {}).get("cropland_mask"):
        return None
    raise NotImplementedError("Optional: implement ESA WorldCover / Dynamic World mask.")
