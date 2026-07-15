"""DEPRECATED — replaced by stac_export.py.

The project has no Google Earth Engine access, so imagery is sourced from the
Microsoft Planetary Computer STAC API instead (free, no GEE). See
`src/data/stac_export.py`. This shim remains only so old references fail loudly.
"""

raise ImportError(
    "gee_export is deprecated — no GEE access. Use src.data.stac_export "
    "(Microsoft Planetary Computer backend)."
)
