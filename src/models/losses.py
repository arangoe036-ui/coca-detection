"""Dice + Focal loss for imbalanced binary segmentation (plan §7.2).

Real implementation (pure torch) — used from P2 onward. Import requires torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss on sigmoid probabilities. Expects logits [B,1,H,W]."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        prob = prob.flatten(1)
        target = target.flatten(1).float()
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017). Expects logits [B,1,H,W]."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Weighted sum of Dice + Focal (plan §7.2 default)."""

    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma)
        self.dw = dice_weight
        self.fw = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, target) + self.fw * self.focal(logits, target)


class SigmoidMSELoss(nn.Module):
    """MSE on sigmoid-bounded output vs a [0,1] fraction target (density regression)."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(torch.sigmoid(logits), target.float())


def build_loss(cfg: dict) -> nn.Module:
    """Construct the loss from config. Regression task -> MSE on sigmoid fraction."""
    if cfg.get("model", {}).get("task") == "regression":
        return SigmoidMSELoss()
    lc = cfg["loss"]
    if lc["type"] == "dice_focal":
        return DiceFocalLoss(lc["dice_weight"], lc["focal_weight"], lc["focal_gamma"])
    if lc["type"] == "bce_posweight":
        pw = torch.tensor([lc["bce_pos_weight"]])
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    raise ValueError(f"Unknown loss type: {lc['type']}")
