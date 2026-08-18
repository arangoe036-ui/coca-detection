"""Phase 4 — assemble the comparison tables from baseline_ladder.jsonl.

Emits outputs/metrics/ladder_table.md (+ .csv) with Track A (spatial) and Track B
(temporal aoi_ratio incl. nulls), per-fold rows AND aggregates, plus the
pre-registered paired per-fold deltas vs the U-Net (prereg §5/A3). No p-values or
CIs at n=6 (prereg §5). Every cell traces to a JSONL row.

    python -m src.baselines.compare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines.common import YEARS
from src.metrics_io import read_runs
from src.utils import load_config

# Frozen U-Net Track B aoi_ratio (docs/v2.1_loyo_results.md) — the only per-fold U-Net
# magnitude numbers on disk (prereg §4). RETRACTED: these are gen1, computed under the
# reflectance offset (F1) and the blank-coverage (F2) defects, whose effects cancelled
# across complementary year-groups. Kept because the project annotates rather than
# overwrites, and because deleting them would leave `track_b_table` unable to run at all
# — but `main` only emits that table when the current generation has Track B rows.
UNET_B = {2019: 0.95, 2020: 0.59, 2021: 0.90, 2022: 1.48, 2023: 1.01, 2024: 0.79}

B_METHODS = ["unet", "random_forest", "ndvi_threshold",
             "null_historical_mean", "null_constant_density"]
#: Operand set of the FROZEN prereg §5 paired verdict ("is a segmentation model warranted
#: over the imagery baselines?"). The persistence nulls are deliberately NOT in here: A16 is
#: a separate registered rule with its own decision bands, and folding a no-skill floor into
#: §5's "best baseline" would silently redefine a frozen comparison after seeing data.
A_METHODS = ["unet", "random_forest", "ndvi_threshold"]
#: The A16/A19 no-skill floor. `PERSISTENCE_SCORE` is the per-fold max that A16 is applied
#: to; the variants are shown alongside it so the max is auditable.
PERSISTENCE_VARIANTS = ["persistence_last_year", "persistence_freq_ever",
                        "persistence_freq_majority"]
PERSISTENCE_SCORE = "persistence_max"
#: Everything displayed in the Track A tables (superset of the §5 operands).
A_DISPLAY = [*A_METHODS, *PERSISTENCE_VARIANTS, PERSISTENCE_SCORE]
LABEL = {"unet": "U-Net", "random_forest": "Random forest",
         "ndvi_threshold": "NDVI threshold",
         "null_historical_mean": "N2 historical mean",
         "null_constant_density": "N1 constant density",
         "persistence_last_year": "P1 last year",
         "persistence_freq_ever": "P2 ever",
         "persistence_freq_majority": "P2 majority",
         PERSISTENCE_SCORE: "**Persistence score (max)**"}


def _by(runs, track):
    d = {}
    for r in runs:
        if r["track"] == track:
            d.setdefault(r["method"], {})[r["fold_year"]] = r
    return d


def _fmt(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def track_b_table(runs) -> list[str]:
    d = _by(runs, "B")
    ratios = {"unet": UNET_B}
    for m in B_METHODS:
        if m == "unet":
            continue
        ratios[m] = {y: d.get(m, {}).get(y, {}).get("metrics", {}).get("aoi_ratio")
                     for y in YEARS}
    L = ["## Track B — out-of-year magnitude (aoi_ratio = predicted / official ha)",
         "",
         ("> Track B sums over the AOI, so it is **structurally incapable of judging "
          "spatial skill** — that is Track A's job (prereg A2). U-Net row is the frozen "
          "v2.1 LOYO result."), "",
         "| held-out year | " + " | ".join(LABEL[m] for m in B_METHODS) + " |",
         "|---|" + "---|" * len(B_METHODS)]
    for y in YEARS:
        L.append(f"| {y} | " + " | ".join(_fmt(ratios[m].get(y), 2) for m in B_METHODS) + " |")
    # aggregates
    def agg(m):
        r = np.array([ratios[m][y] for y in YEARS if ratios[m].get(y) is not None], float)
        return r.mean(), r.std(), abs(r.mean() - 1), int(((r >= .85) & (r <= 1.15)).sum()), r.min(), r.max()
    L += ["| **mean** | " + " | ".join(_fmt(agg(m)[0], 2) for m in B_METHODS) + " |",
          "| **std (S)** | " + " | ".join(_fmt(agg(m)[1], 3) for m in B_METHODS) + " |",
          "| **\\|mean−1\\| (C)** | " + " | ".join(_fmt(agg(m)[2], 3) for m in B_METHODS) + " |",
          "| **K in [0.85,1.15]** | " + " | ".join(f"{agg(m)[3]}/6" for m in B_METHODS) + " |",
          "| **band** | " + " | ".join(f"{agg(m)[4]:.2f}–{agg(m)[5]:.2f}" for m in B_METHODS) + " |"]
    # paired deltas vs U-Net
    L += ["", "### Paired per-fold deltas vs U-Net  (δ = |ratio−1|(X) − |ratio−1|(U-Net); δ>0 ⇒ U-Net closer)",
          "", "| year | " + " | ".join(LABEL[m] for m in B_METHODS if m != "unet") + " |",
          "|---|" + "---|" * (len(B_METHODS) - 1)]
    du = {y: abs(UNET_B[y] - 1) for y in YEARS}
    for y in YEARS:
        cells = []
        for m in B_METHODS:
            if m == "unet":
                continue
            v = ratios[m].get(y)
            cells.append("—" if v is None else f"{abs(v-1)-du[y]:+.3f}")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    verdict = []
    TIE = 1e-3   # 5.4: |δ| below this is a numerical tie, not a win (e.g. 2020 RF)
    for m in B_METHODS:
        if m == "unet":
            continue
        deltas = [abs(ratios[m][y]-1)-du[y] for y in YEARS if ratios[m].get(y) is not None]
        if not deltas:
            continue
        w = sum(x > TIE for x in deltas)      # U-Net wins
        t = sum(abs(x) <= TIE for x in deltas)  # ties
        loss = sum(x < -TIE for x in deltas)  # U-Net losses
        call = ("U-Net clearly better (≥5/6 wins)" if w >= 5 else
                f"{LABEL[m]} clearly better (≥5/6 wins)" if loss >= 5 else
                "indistinguishable at n=6")
        verdict.append(f"- **U-Net vs {LABEL[m]}:** {w}W / {t}T / {loss}L, "
                       f"mean δ = {np.mean(deltas):+.3f} → **{call}**")
    L += ["", "**Paired verdicts (prereg §5, ≥5/6 rule):**", *verdict]
    return L


def track_a_table(runs) -> list[str]:
    d = _by(runs, "A")
    metrics = ["mae", "rmse", "presence_iou", "presence_f1"]
    L = ["## Track A — spatial skill on the held-out `test` blocks (per year)",
         "", ("> Fit on all-years train blocks, evaluated on all-years spatially-held-out "
              "test blocks. This is the track that judges whether a segmentation model is "
              "warranted."), ""]
    for mt in metrics:
        L += [f"### {mt}", "",
              "| year | " + " | ".join(LABEL[m] for m in A_DISPLAY) + " |",
              "|---|" + "---|" * len(A_DISPLAY)]
        for y in YEARS:
            row = [d.get(m, {}).get(y, {}).get("metrics", {}).get(mt) for m in A_DISPLAY]
            L.append(f"| {y} | " + " | ".join(_fmt(v, 4 if mt in ("mae","rmse") else 3) for v in row) + " |")
        means = []
        for m in A_DISPLAY:
            vals = [d.get(m, {}).get(y, {}).get("metrics", {}).get(mt) for y in YEARS]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else None)
        L.append("| **mean** | " + " | ".join(_fmt(v, 4 if mt in ("mae","rmse") else 3) for v in means) + " |")
        L.append("")
    # paired IoU verdict: U-Net vs best baseline per year
    iou = {m: {y: d.get(m, {}).get(y, {}).get("metrics", {}).get("presence_iou") for y in YEARS}
           for m in A_METHODS}
    wins = 0; n = 0
    for y in YEARS:
        u = iou["unet"].get(y)
        base = [iou[m].get(y) for m in A_METHODS if m != "unet"]
        base = [b for b in base if b is not None]
        if u is None or not base:
            continue
        n += 1; wins += (u > max(base))
    if n:
        call = "U-Net wins spatially (≥5/6)" if wins >= 5 else "spatial tie / not a clear win"
        L += [(f"**Paired presence-IoU verdict (prereg §5):** U-Net > best *imagery* "
               f"baseline in {wins}/{n} years → **{call}**."),
              "", ("> §5 compares against RF and NDVI only. Beating an imagery baseline is "
                   "not the same as beating a no-skill floor — that is A16 below, and it is "
                   "the rule that decides whether the spatial claim exists as a *model* "
                   "result.")]
    return [*L, "", *a16_verdict(runs)]


def a16_verdict(runs) -> list[str]:
    """Apply A16's frozen decision rule: U-Net presence-IoU vs the A19 persistence score.

    The rule and its three bands were registered before any of this was computed and may
    not be moved after seeing the numbers (prereg A16):

    * **>=5/6 folds** won by the U-Net -> the spatial claim is earned and publishable as a
      model result;
    * **3-4/6** -> report as indistinguishable from persistence; the claim shrinks to
      "reproduces the official spatial pattern", with persistence named as an equally good
      method;
    * **<=2/6** -> the spatial claim is **not publishable as a model result** and is
      reported as a negative result with the same prominence as the counting one.

    Ties count as losses for the U-Net: the null is the incumbent, so "no better than
    where it was last year" is not a win. Folds where either side is missing are excluded
    and the honest denominator is stated rather than assumed to be 6.
    """
    d = _by(runs, "A")
    unet, pers = d.get("unet", {}), d.get(PERSISTENCE_SCORE, {})
    L = ["### A16 — does the U-Net beat the persistence floor? (frozen rule, ≥5/6)", ""]
    if not pers:
        return [*L,
                ("**NOT COMPUTED — no `persistence_max` rows in the sink.** Run "
                 "`python -m src.baselines.persistence`. Until this row exists the spatial "
                 "claim has no no-skill floor and must not be published (KNOWN_DEFECTS: "
                 "\"Not yet measured, and it could sink the headline\").")]
    L += ["| held-out year | U-Net IoU | persistence score | argmax variant | δ (U−P) | U-Net wins? |",
          "|---|--:|--:|---|--:|---|"]
    wins, n = 0, 0
    for y in YEARS:
        u = unet.get(y, {}).get("metrics", {}).get("presence_iou")
        p = pers.get(y, {}).get("metrics", {}).get("presence_iou")
        which = pers.get(y, {}).get("extra", {}).get("argmax_variant", "—")
        if u is None or p is None:
            L.append(f"| {y} | {_fmt(u)} | {_fmt(p)} | {which} | — | excluded (missing arm) |")
            continue
        n += 1
        won = u > p
        wins += won
        L.append(f"| {y} | {u:.3f} | {p:.3f} | {which} | {u - p:+.3f} | "
                 f"{'yes' if won else 'no'} |")
    if not n:
        return [*L, "", "**NOT COMPUTED — no fold has both arms.**"]
    if wins >= 5:
        call = ("**spatial claim EARNED** — publishable as a model result (≥5/6)")
    elif wins >= 3:
        call = ("**INDISTINGUISHABLE from persistence** — the claim shrinks to "
                "\"reproduces the official spatial pattern\"; persistence must be named as "
                "an equally good method (3–4/6)")
    else:
        call = ("**NOT PUBLISHABLE as a model result** — report as a negative result with "
                "the same prominence as the counting one (≤2/6). Do NOT retune the U-Net "
                "in response (A16)")
    return [*L, "",
            f"**U-Net beats the persistence score in {wins}/{n} folds → {call}.**",
            ("" if n == 6 else
             f"\n> Denominator is **{n}, not 6** — {6 - n} fold(s) lack one arm. The ≥5/6 "
             "bar is not reachable at all below n=5; do not rescale it.")]


def _dedup_last(runs):
    """Keep the LAST record per (data_generation, method, track, fold_year) so a re-run
    supersedes a stale row without rewriting the append-only JSONL.

    ``data_generation`` is part of the key deliberately. Without it, a gen4 row silently
    overwrote a gen3 row of the same method/fold — which sounds harmless but is the exact
    mechanism for manufacturing a **false verdict**: gen3's test split held no coca, so a
    gen3 persistence row is `0.000`, and pairing that against a gen4 U-Net number would
    read as a clean sweep for the model. Generations are not comparable (gen1 had three
    defects, gen2 fixed one, gen3 was clean but had an empty test fold, gen4 rebuilt the
    split under A20 and trains on 22% fewer rows), so they must not collide in one key.
    Cross-generation *mixing* is prevented separately by `_one_generation`.
    """
    keep = {}
    for r in runs:
        keep[(r.get("data_generation"), r["method"], r["track"], r["fold_year"])] = r
    return list(keep.values())


def _one_generation(runs, cfg):
    """Restrict to rows computed on the CURRENT data generation, loudly.

    No table in this file may mix generations: every comparison here is paired per fold,
    and a pair drawn from two generations is not a comparison at all. Rows with no
    `data_generation` field predate the stamp and are untrustworthy by default, so they
    are dropped too. Dropping is reported, never silent — the count is the audit trail
    for anyone who wonders why a row they remember writing is absent.
    """
    want = str(cfg["project"]["data_generation"])
    kept = [r for r in runs if str(r.get("data_generation")) == want]
    dropped = {}
    for r in runs:
        if str(r.get("data_generation")) != want:
            dropped[str(r.get("data_generation"))] = dropped.get(
                str(r.get("data_generation")), 0) + 1
    if dropped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        print(f"[compare] data_generation={want}: kept {len(kept)} rows, "
              f"EXCLUDED {sum(dropped.values())} from other generations ({detail}). "
              "Generations are not comparable and are never mixed in one table.")
    if not kept:
        raise SystemExit(
            f"no rows for data_generation={want} in baseline_ladder.jsonl "
            f"(file holds: {detail if dropped else 'nothing'}). Re-run the arms on the "
            "current data rather than reading the older generation's numbers.")
    return kept, want, dropped


def main(cfg):
    runs, gen, dropped = _one_generation(read_runs(cfg), cfg)
    runs = _dedup_last(runs)
    if not runs:
        raise SystemExit("no runs in baseline_ladder.jsonl")
    lines = ["# Baseline ladder — comparison tables",
             "", (f"Generated from `outputs/metrics/baseline_ladder.jsonl` "
                  f"({len(runs)} runs, **data_generation = `{gen}`**). Reproduce: "
                  f"`python -m src.baselines.compare`."), ""]
    if dropped:
        lines += [(f"> {sum(dropped.values())} row(s) from other data generations "
                   f"({', '.join(f'`{k}`={v}' for k, v in sorted(dropped.items()))}) are "
                   "**excluded**: generations are not comparable and every table here is "
                   "paired per fold."), ""]
    lines += track_a_table(runs)
    # Track B (hectare counting) is a retired deliverable, and its U-Net row is a frozen
    # gen1 dict that has since been RETRACTED — so the section is emitted only when the
    # current generation actually has Track B rows to put beside it. Printing the paired
    # deltas off retracted constants would be exactly the cross-generation pairing that
    # `_one_generation` exists to prevent, just hardcoded instead of read from the sink.
    if _by(runs, "B"):
        lines += ["", "---", "",
                  ("> ⚠ The U-Net row below is the **frozen v2.1 (gen1) LOYO result, which "
                   "is RETRACTED** (KNOWN_DEFECTS: the F1/F2 cancellation artifact). It is "
                   f"not a `{gen}` number and the paired deltas against it are not evidence. "
                   "Track B (total hectares) is a retired deliverable."), ""]
        lines += track_b_table(runs)
    else:
        lines += ["", "---", "",
                  "## Track B — out-of-year magnitude", "",
                  (f"No Track B rows on `{gen}`. Section omitted rather than printed "
                   "against the retracted gen1 U-Net constants. Counting is a retired "
                   "deliverable (A12: the level signal does not exist).")]
    out = Path(cfg["paths"]["outputs_dir"]) / "metrics"
    # encoding is explicit: the tables carry →, ≥, δ, — and the default on Windows is
    # cp1252, which raises UnicodeEncodeError. It had never fired because these tables
    # had never been generated on this platform — the same class of latent portability
    # defect as the three that surfaced in stac_export.py.
    (out / "ladder_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # CSV: one row per (track, method, year)
    import csv
    with open(out / "ladder_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["track", "method", "fold_year", "aoi_ratio", "mae", "rmse",
                    "presence_iou", "presence_f1", "calibration_scalar"])
        for r in runs:
            m = r["metrics"]
            w.writerow([r["track"], r["method"], r["fold_year"], m.get("aoi_ratio"),
                        m.get("mae"), m.get("rmse"), m.get("presence_iou"),
                        m.get("presence_f1"), r.get("calibration_scalar")])
    print(f"[compare] wrote {out/'ladder_table.md'} and .csv from {len(runs)} runs")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Assemble the baseline-ladder tables.")
    ap.add_argument("--config", default=None)
    main(load_config(ap.parse_args().config))
