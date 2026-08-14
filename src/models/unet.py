"""Baseline U-Net. NOTE: the encoder is currently RANDOMLY INITIALISED (plan §7.1).

`config`'s `encoder_weights: ssl4eo` lands in `geo_keys` below, so `None` is passed to
`smp.Unet`, and `_load_geo_encoder_weights()` is still a no-op that only warns. The net
effect is that naming a geo source *disables* the ImageNet weights smp would otherwise
load. Setting `encoder_weights: imagenet` is the cheap lever; wiring real geo weights is
the expensive one. (Docstring corrected 2026-08-10 — it previously asserted the encoder
was satellite-pretrained.)

Wraps segmentation_models_pytorch. The KEY accuracy lever (plan §4) is using a
backbone pretrained on satellite imagery (torchgeo SSL4EO weights, or a geo
foundation model: Prithvi / Clay / SatMAE) — NOT plain ImageNet. The first conv
is adapted to accept C = optical bands + indices + SAR bands.

Import requires torch + segmentation-models-pytorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_unet(cfg: dict) -> nn.Module:
    """Build the baseline U-Net from config (`model` block)."""
    import segmentation_models_pytorch as smp

    mc = cfg["model"]
    in_ch = mc["in_channels"]
    weights = mc.get("encoder_weights")

    # smp accepts "imagenet"/None natively. Geo weights (ssl4eo/prithvi/clay/satmae)
    # are loaded separately below, so pass None to smp and inject afterward.
    geo_keys = {"ssl4eo", "prithvi", "clay", "satmae"}
    smp_weights = None if (weights in geo_keys) else weights

    model = smp.Unet(
        encoder_name=mc["encoder"],
        encoder_weights=smp_weights,
        in_channels=in_ch,
        classes=mc["classes"],
    )

    if weights in geo_keys:
        _load_geo_encoder_weights(model, encoder=mc["encoder"], source=weights, in_ch=in_ch)

    return model


def _load_geo_encoder_weights(model: nn.Module, encoder: str, source: str, in_ch: int) -> None:
    """Load satellite-pretrained encoder weights and adapt the first conv to in_ch.

    TODO (P2): wire torchgeo's SSL4EO weight enums (or Prithvi/Clay/SatMAE state
    dicts). Until then this is a no-op with a warning so the loop still runs on
    randomly-initialized-encoder weights.
    """
    import warnings

    warnings.warn(
        f"Geo-pretrained weights '{source}' for encoder '{encoder}' not wired yet "
        f"(P2 TODO). Proceeding with default init for {in_ch}-channel input.",
        stacklevel=2,
    )


if __name__ == "__main__":
    from src.utils import load_config

    cfg = load_config()
    print(f"U-Net config: encoder={cfg['model']['encoder']} "
          f"weights={cfg['model']['encoder_weights']} in_ch={cfg['model']['in_channels']}")
