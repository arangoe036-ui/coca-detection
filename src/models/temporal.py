"""P6 stretch — temporal model over a time series (T × C × H × W) (plan §7.4).

Options: U-TAE (PASTIS-style temporal U-Net), ConvLSTM, or per-pixel
1D-CNN/LSTM/Transformer. Rationale: coca is evergreen and multi-harvest, so its
temporal signature is distinctive. Build only after the single-composite
baseline (P2–P4) is validated.
"""

from __future__ import annotations


def build_temporal_model(cfg: dict):
    raise NotImplementedError("P6 stretch: implement U-TAE / ConvLSTM temporal model.")
