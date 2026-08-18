"""P3 — Evaluation (plan §8).

Loads a trained checkpoint, tunes the decision threshold on the validation split
(maximizing F1), and reports coca-class IoU / F1 / precision / recall + average
precision (PR-AUC) on the test split. Optionally saves a PR curve if matplotlib
is present.

The **area sanity check** (predicted hectares vs the official figure) needs a
stitched full-AOI prediction, so it lives in ``infer.py`` (P4), which has the
full raster.

    python -m src.evaluate --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.dataset import build_dataloader
from src.models.unet import build_unet
from src.train import pick_device
from src.utils import load_config


@torch.no_grad()
def _collect(model, loader, device):
    """Return flattened (probs, targets) over a loader."""
    ps, ts = [], []
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        prob = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        ps.append(prob)
        ts.append(yb.numpy().ravel())
    return np.concatenate(ps), np.concatenate(ts)


class DegenerateComparison(AssertionError):
    """A metric was requested on inputs that cannot produce a measurement.

    Subclasses ``AssertionError`` so existing ``except AssertionError`` handlers
    and ``pytest.raises(AssertionError)`` still catch it, while callers that want
    to distinguish "the comparison was impossible" from "the comparison was bad"
    can catch this specifically.
    """


def _metrics_at(probs, targets, thr, t_thr=0.5):
    """Presence IoU/F1/precision/recall at a decision cut. FROZEN by prereg A19.

    **Degeneracy guard (prereg A21, added 2026-08-14).** With no positive target
    every one of ``tp``, ``fp``, ``fn`` is 0, so the ``+ 1e-6`` terms below turn
    ``0/0`` into a printed **0.000** with no warning — indistinguishable from a
    method that genuinely scored zero. That is exactly how gen3 shipped: its test
    split held no coca, and every ``presence_iou`` on it (U-Net, RF, NDVI and the
    A16 persistence nulls alike) was a non-measurement dressed as a measurement.
    A16 exists to stop an unearned claim, so this raises rather than returning.

    The guard fires on a degenerate **comparison**, never on a degenerate
    **result**:

    * empty input, or a target with no positive under ``t_thr`` -> raise. There is
      nothing to be right or wrong about; no value is defensible.
    * an all-zero *prediction* against a target that does contain positives ->
      **returned normally**. ``tp = 0`` and ``fn > 0`` give ``iou = 0/fn = 0``
      exactly, with the epsilon immaterial. "This method detects nothing here" is
      an earned result, and A16's decision rule explicitly provides for a model
      that loses to its null. Raising on it would also crash a legitimately
      all-zero persistence variant out of A19's max-of-three, which would *lower*
      the U-Net's bar — the one direction A19 forbids.

    Values on valid input are unchanged; the arithmetic below is untouched.
    """
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    if probs.size == 0 or targets.size == 0:
        raise DegenerateComparison(
            f"_metrics_at got an empty array (probs={probs.size}, targets={targets.size}) "
            "— an empty comparison is not a score of 0.0; check the split has rows")

    pred = probs > thr
    t = targets > t_thr
    n_pos = int(t.sum())
    if n_pos == 0:
        raise DegenerateComparison(
            f"_metrics_at: target has NO positive pixel above t_thr={t_thr} "
            f"(n={targets.size}) — iou/precision/recall/f1 would all be 0/0 returned "
            "as 0.000 with no warning. This is the gen3 blocker B1; fix the split "
            "(prereg A20) rather than reporting the number")

    tp = np.logical_and(pred, t).sum()
    fp = np.logical_and(pred, ~t).sum()
    fn = np.logical_and(~pred, t).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return {"iou": float(iou), "f1": float(f1), "precision": float(prec), "recall": float(rec)}


def evaluate(cfg: dict, checkpoint: str) -> dict:
    device = pick_device()
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    mean, std = ck["mean"], ck["std"]
    model = build_unet(cfg).to(device)
    model.load_state_dict(ck["model"])
    test_loader, _, _ = build_dataloader(cfg, "test", mean=mean, std=std, augment=False)
    tp_, tt_ = _collect(model, test_loader, device)

    if cfg.get("model", {}).get("task") == "regression":
        return _evaluate_regression(cfg, tp_, tt_)
    return _evaluate_segmentation(cfg, model, mean, std, device, tp_, tt_)


def _evaluate_regression(cfg: dict, probs, targets) -> dict:
    """Density regression: MAE/RMSE/bias on the fraction + summed-hectares ratio +
    presence IoU/F1 (spatial quality at a small presence threshold)."""
    err = probs - targets
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())
    ha_ratio = float(probs.sum() / (targets.sum() + 1e-6))  # ~ predicted_ha / true_ha on test
    thr = 0.02  # >2% cover = "present"; ground truth present if the cell has any coca
    pm = _metrics_at(probs, targets, thr, t_thr=0.0)
    m = {"mae": mae, "rmse": rmse, "bias": bias, "ha_ratio_test": ha_ratio,
         "presence_iou": pm["iou"], "presence_f1": pm["f1"]}
    print(f"[eval] TEST (regression): MAE={mae:.4f} RMSE={rmse:.4f} bias={bias:+.4f}")
    print(f"[eval]   summed-hectares ratio (pred/true) on test = {ha_ratio:.2f}  (1.0 = calibrated)")
    print(f"[eval]   presence@{thr}: IoU={pm['iou']:.3f} F1={pm['f1']:.3f} "
          f"P={pm['precision']:.3f} R={pm['recall']:.3f}")
    return m


def _evaluate_segmentation(cfg, model, mean, std, device, tp_, tt_) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    val_loader, _, _ = build_dataloader(cfg, "val", mean=mean, std=std, augment=False)
    vp, vt = _collect(model, val_loader, device)
    prec, rec, thr = precision_recall_curve(vt > 0.5, vp)
    f1s = 2 * prec * rec / (prec + rec + 1e-6)
    best_thr = float(thr[max(int(np.argmax(f1s)) - 1, 0)]) if len(thr) else 0.5
    print(f"[eval] tuned threshold on val: {best_thr:.3f}  (val F1={f1s.max():.3f})")
    m = _metrics_at(tp_, tt_, best_thr)
    ap = float(average_precision_score(tt_ > 0.5, tp_)) if (tt_ > 0.5).any() else float("nan")
    m.update({"average_precision": ap, "threshold": best_thr})
    print(f"[eval] TEST @thr={best_thr:.3f}: IoU={m['iou']:.3f} F1={m['f1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} AP={ap:.3f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate coca model (P3).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--emit-metrics", action="store_true",
                    help="P6-A: also persist the metrics dict to "
                         "outputs/metrics/metrics.json (retire prose-only numbers).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    m = evaluate(cfg, args.checkpoint)
    if args.emit_metrics:
        import json
        from pathlib import Path
        out = Path(cfg["paths"]["outputs_dir"]) / "metrics"
        out.mkdir(parents=True, exist_ok=True)
        rec = {"checkpoint": args.checkpoint, "task": cfg.get("model", {}).get("task"),
               "metrics": {k: float(v) for k, v in m.items()}}
        (out / "metrics.json").write_text(json.dumps(rec, indent=2))
        print(f"[eval] wrote outputs/metrics/metrics.json")
