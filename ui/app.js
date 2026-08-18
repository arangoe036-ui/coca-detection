/* Coca monitor — Catatumbo. Vanilla Leaflet, no build step.
 *
 * The map's job is to make ONE comparison inspectable: the model's density against the
 * official census, on the same 75 m grid, over the same out-of-fold footprint, with the
 * same colour ramp. Two pre-registered nulls (A16 spatial, A17 ranking) beat this model,
 * so "reproduces the official pattern from free imagery" is the claim — and a viewer
 * should be able to check it by eye instead of trusting a caveat.
 *
 * Every number the panel shows is read from data/metrics.json, which is copied out of
 * outputs/metrics/baseline_ladder.jsonl by src/rank_municipal.py. Nothing here is
 * hardcoded: prereg §8 forbids a reported number that lives only in prose, and a figure
 * typed into a caveat string is prose.
 *
 * Artifacts, all regenerated 2026-08-18 on gen4 (the previous set was 2026-08-09 gen1,
 * from a checkpoint that no longer exists):
 *   data/density_<year>_cog.tif    model, out-of-fold, 75 m
 *   data/official_<year>_cog.tif   census ~1 km cells, same footprint, 75 m
 *   data/municipal_coca_<year>.geojson|csv
 *   data/metrics.json
 */

/* Fail visibly, never blankly. If a vendor script is missing the old code threw
 * "L is not defined" on its first statement and left a black rectangle with an empty year
 * dropdown — indistinguishable, to a viewer, from "the model found nothing here". */
(function requireDeps() {
  const missing = [];
  if (typeof L === "undefined") missing.push("vendor/leaflet.js");
  if (typeof parseGeoraster === "undefined") missing.push("vendor/georaster.min.js");
  if (typeof GeoRasterLayer === "undefined") missing.push("vendor/georaster-layer-for-leaflet.min.js");
  if (!missing.length) return;
  const el = document.getElementById("map");
  if (el) {
    el.innerHTML = `<div class="fatal"><b>The map could not start.</b>
      <span>These files failed to load:</span><code>${missing.join("<br>")}</code>
      <span>They are served from this site, so this is a serving or file-permission
      problem, not a network one. Everything else on the page is unaffected.</span></div>`;
  }
  throw new Error("missing vendor assets: " + missing.join(", "));
})();

const YEARS = [2019, 2020, 2021, 2022, 2023, 2024];
const INSPECT_ZOOM = 13;

/* Magma-derived ramp. Chosen over the old YlOrBr for one concrete reason: the choropleth
 * and the raster used to share YlOrBr, so "where the coca is" was indistinguishable from
 * the polygon fill underneath it and the map read as an orange wash. The density layer now
 * owns colour outright; municipal geometry is line work only. Dark→bright also survives
 * the dark basemap and the satellite imagery underneath. */
const RAMP = ["#180f3d", "#440f76", "#7e2482", "#b63679", "#e75263", "#fb8861", "#fec287", "#fcfdbf"];
const ACCENT = "#f0a02a";
const MAX_FRAC = 0.29;   /* Ramp top. Must be >= the max cover fraction in any published
                          * COG or the ramp clips silently; measured across the twelve gen4
                          * rasters: 0.261–0.282. Was 0.26, which three of them exceeded. */
/* Published rasters are uint16 holding round(cover_fraction * DENSITY_SCALE) — 2.5 MB
 * instead of 7.6 MB each, which is what a viewer on a tunnelled connection waits on. The
 * authoritative value is `density_scale` in metrics.json; this constant only matches the
 * writer so a missing metrics.json degrades to correct rather than to a saturated map. */
let densityScale = 10000;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const fmt = (v) => Math.round(v || 0).toLocaleString("en-US");
const pct1 = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");
const num3 = (v) => (v == null ? "—" : Number(v).toFixed(3));

function rampRGB(t) {
  const hexes = RAMP.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  t = Math.max(0, Math.min(1, t));
  const x = t * (hexes.length - 1), i = Math.floor(x), f = x - i;
  const a = hexes[i], b = hexes[Math.min(i + 1, hexes.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return [c[0], c[1], c[2]];
}

// ---- map + base layers -------------------------------------------------------
const map = L.map("map", { zoomControl: true, minZoom: 6, maxZoom: 18, zoomSnap: 0.25 })
  .setView([8.8, -72.85], 9);
L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);

const baseLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "© OpenStreetMap · © CARTO", maxZoom: 20, className: "basemap-dark",
}).addTo(map);
/* Basemap tiles are the one remaining network dependency, and they are decoration: the
 * data layers are local files. Tell the viewer rather than leaving them to wonder whether
 * the emptiness is the map or the model. */
let tileErrors = 0;
baseLayer.on("tileerror", () => {
  if (++tileErrors === 6) document.getElementById("basemap-warn").hidden = false;
});
baseLayer.on("tileload", () => { document.getElementById("basemap-warn").hidden = true; });

map.createPane("imagery");
const imgPane = map.getPane("imagery");
imgPane.style.zIndex = 250;
imgPane.style.opacity = 0;
imgPane.style.transition = reduceMotion ? "none" : "opacity .45s ease";
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Imagery © Esri, Maxar, Earthstar Geographics", maxZoom: 19, pane: "imagery",
}).addTo(map);

/* Two raster panes so Compare can clip one against the other. Rasters sit above the
 * basemap and below the municipal outlines, and pass clicks through to the polygons. */
for (const [name, z] of [["rasterA", 420], ["rasterB", 421]]) {
  map.createPane(name);
  const p = map.getPane(name);
  p.style.zIndex = z;
  p.style.pointerEvents = "none";
}
map.createPane("outline");
map.getPane("outline").style.zIndex = 450;

let imageryOn = false;
function setImagery(on) {
  imageryOn = on;
  imgPane.style.opacity = on ? 1 : 0;
  const chk = document.getElementById("imagery-chk");
  if (chk) chk.checked = on;
}
document.getElementById("imagery-chk").onchange = (e) => setImagery(e.target.checked);
// Zooming in past a municipality is an inspection gesture: bring in real imagery so a
// human can look at the ground the model flagged ("algorithm flags, analyst verifies").
map.on("zoomend", () => { if (map.getZoom() >= INSPECT_ZOOM && !imageryOn) setImagery(true); });

// ---- state -------------------------------------------------------------------
let mode = "model";            // model | official | swipe
let threshold = 0.01;
let overlayOpacity = 0.85;
let rasters = {};              // { model, official } parsed georasters for the current year
let layers = {};               // live GeoRasterLayers by pane
let outlineLayer;
let allMetrics = null;
let currentYear = null;
let aoiBounds = null;
const layersByName = new Map();

// ---- density rasters ---------------------------------------------------------
function colorFn(vals) {
  const raw = vals[0];
  if (raw == null || Number.isNaN(raw) || raw <= 0) return null;
  const f = raw / densityScale;
  if (f < threshold) return null;
  const [r, g, b] = rampRGB(f / MAX_FRAC);
  /* Alpha rises with cover instead of a hard on/off at the threshold. The old hard cut
   * turned a continuous field into binary speckle, which is what made the layer read as
   * noise rather than as intensity. */
  const a = 0.35 + 0.65 * Math.min(1, f / (MAX_FRAC * 0.6));
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function makeRasterLayer(which, pane) {
  const georaster = rasters[which];
  if (!georaster) return null;
  const lyr = new GeoRasterLayer({
    georaster, pane, opacity: overlayOpacity, resolution: 512,
    pixelValuesToColorFn: colorFn,
  });
  lyr.addTo(map);
  return lyr;
}

function renderLayers() {
  for (const k of Object.keys(layers)) {
    if (layers[k]) map.removeLayer(layers[k]);
  }
  layers = {};
  if (mode === "model") layers.a = makeRasterLayer("model", "rasterA");
  else if (mode === "official") layers.a = makeRasterLayer("official", "rasterA");
  else {
    layers.a = makeRasterLayer("official", "rasterA");   // right side of the swipe
    layers.b = makeRasterLayer("model", "rasterB");      // left side, clipped
  }
  applySwipe();
}

async function loadRasters(year) {
  const get = async (kind) => {
    try {
      /* ?v is not decoration: the rasters changed dtype in this revision, and a browser
       * serving a cached float32 raster against the new uint16 scale would divide by
       * 10000 and render an empty map. */
      const res = await fetch(`data/${kind}_${year}_cog.tif?v=3`);
      if (!res.ok) throw new Error(`${kind} ${res.status}`);
      return await parseGeoraster(await res.arrayBuffer());
    } catch (e) {
      console.warn(`${kind} raster unavailable for ${year}`, e);
      return null;
    }
  };
  const [model, official] = await Promise.all([get("density"), get("official")]);
  rasters = { model, official };
}

// ---- Compare (swipe) ---------------------------------------------------------
let swipeX = 0.5;
function applySwipe() {
  const handle = document.getElementById("swipe-handle");
  const labels = document.getElementById("swipe-labels");
  const on = mode === "swipe";
  handle.hidden = !on;
  labels.hidden = !on;
  const cA = layers.a && layers.a.getContainer();   // census, right side
  const cB = layers.b && layers.b.getContainer();   // model, left side
  if (!on) {
    if (cA) cA.style.clip = "";
    if (cB) cB.style.clip = "";
    return;
  }
  /* Clip each layer's own container with `clip: rect()` in LAYER-POINT coordinates — the
   * technique leaflet-side-by-side uses, and for a specific reason. Two earlier attempts
   * failed here:
   *   - clipping only the top layer left the other drawn underneath it across the whole
   *     map, so both halves rendered as the same layer;
   *   - `clip-path: inset(...)` on the *pane* erased both layers, because Leaflet panes are
   *     zero-size boxes whose children merely overflow, so a pixel inset on a 0x0 box clips
   *     away everything.
   * Layer points are the container's own coordinate system, and they shift on every pan and
   * zoom, which is why this re-runs on `move`/`zoom` below. */
  const size = map.getSize();
  const nw = map.containerPointToLayerPoint([0, 0]);
  const se = map.containerPointToLayerPoint([size.x, size.y]);
  const splitX = nw.x + (se.x - nw.x) * swipeX;
  if (cB) cB.style.clip = `rect(${nw.y}px, ${splitX}px, ${se.y}px, ${nw.x}px)`;
  if (cA) cA.style.clip = `rect(${nw.y}px, ${se.x}px, ${se.y}px, ${splitX}px)`;
  handle.style.left = `${Math.round(size.x * swipeX)}px`;
}
(function setupSwipeDrag() {
  const wrap = document.getElementById("map-wrap");
  const handle = document.getElementById("swipe-handle");
  let dragging = false;
  const move = (clientX) => {
    const r = wrap.getBoundingClientRect();
    swipeX = Math.min(0.97, Math.max(0.03, (clientX - r.left) / r.width));
    applySwipe();
  };
  handle.addEventListener("pointerdown", (e) => { dragging = true; handle.setPointerCapture(e.pointerId); });
  handle.addEventListener("pointermove", (e) => { if (dragging) move(e.clientX); });
  handle.addEventListener("pointerup", () => { dragging = false; });
  window.addEventListener("resize", applySwipe);
})();

// ---- municipal outlines + popups --------------------------------------------
function flyToFeature(bounds) {
  setImagery(true);
  if (reduceMotion) map.fitBounds(bounds, { maxZoom: INSPECT_ZOOM });
  else map.flyToBounds(bounds, { maxZoom: INSPECT_ZOOM, duration: 1.1 });
}

function popupHtml(p) {
  /* Unranked municipalities render as unranked, never as a prediction of zero. A22
   * membership depends only on the census and the split geometry, never on the model. */
  if (!p.in_ranking) {
    return `<div class="popup-title">${p.label || p.name}</div>
      <div class="popup-row"><span>Not ranked</span><b>—</b></div>
      <div class="popup-note">No official census value for this year, or below the
        out-of-fold coverage floor. Excluded from every metric on this page.</div>`;
  }
  const agree = p.model_rank === p.official_rank;
  return `<div class="popup-title">${p.label || p.name}</div>
    <div class="popup-rank ${agree ? "ok" : "off"}">
      rank ${Math.round(p.model_rank)} model · ${Math.round(p.official_rank)} census
      ${agree ? "· agrees" : "· disagrees"}</div>
    <div class="popup-row"><span>Mean cover, model</span><b>${pct1(p.pred_density)}</b></div>
    <div class="popup-row"><span>Mean cover, census</span><b>${pct1(p.official_density)}</b></div>
    <div class="popup-row"><span>Land analysed</span><b>${fmt(p.oof_px * 0.04)} ha</b></div>
    <div class="popup-row"><span>Census total (whole municipality)</span>
      <b>${p.official_ha ? fmt(p.official_ha) + " ha" : "—"}</b></div>
    <div class="popup-note">The two cover figures are the like-for-like comparison: both are
      means over the <em>same</em> ${fmt(p.oof_px)} pixels. The census total is not — it counts
      the whole municipality, of which ${pct1(p.oof_share_of_canvas)} is scored here, so
      hectare ratios are not a model error measure. Predictions are uncalibrated.</div>`;
}

async function loadChoropleth(year) {
  const res = await fetch(`data/municipal_coca_${year}.geojson`);
  if (!res.ok) throw new Error(`municipal geojson ${res.status}`);
  const gj = await res.json();
  if (outlineLayer) map.removeLayer(outlineLayer);
  layersByName.clear();
  outlineLayer = L.geoJSON(gj, {
    pane: "outline",
    /* Line work only. The fill used to carry a second sequential ramp, which fought the
     * raster for the same visual channel; a municipality's numbers are one click away
     * instead. Unranked ones are dashed so exclusion is visible, not implied. */
    style: (f) => ({
      color: f.properties.in_ranking ? "#9fb2c4" : "#6b7885",
      weight: f.properties.in_ranking ? 1.1 : 1,
      dashArray: f.properties.in_ranking ? null : "3 4",
      fillColor: "#ffffff", fillOpacity: 0.02,
    }),
    onEachFeature: (f, lyr) => {
      const p = f.properties;
      layersByName.set(p.name, lyr);
      lyr.bindPopup(popupHtml(p), { maxWidth: 330 });
      lyr.on("mouseover", () => lyr.setStyle({ weight: 2.4, color: ACCENT, fillOpacity: 0.06 }));
      lyr.on("mouseout", () => outlineLayer.resetStyle(lyr));
      lyr.on("click", () => flyToFeature(lyr.getBounds()));
    },
  }).addTo(map);
  aoiBounds = outlineLayer.getBounds();
  return gj;
}

// ---- panel -------------------------------------------------------------------
function buildScoreboard(year) {
  const m = (allMetrics && allMetrics.years && allMetrics.years[String(year)]) || {};
  const beat = (a, b) => (a == null || b == null ? "" : a > b ? "win" : "loss");
  document.getElementById("scoreboard").innerHTML = `
    <h2>Measured on this year <span class="h2-note">held out</span></h2>
    <div class="score">
      <div class="score-row head"><span></span><span>model</span><span>null (t−1)</span></div>
      <div class="score-row">
        <span title="Agreement about which ~1 km cells contain coca (prereg A16).">cells, IoU</span>
        <b class="${beat(m.model_iou, m.null_iou)}">${num3(m.model_iou)}</b>
        <b>${num3(m.null_iou)}</b>
      </div>
      <div class="score-row">
        <span title="Spearman rank correlation of municipalities against the census (prereg A17).">ranking, ρ</span>
        <b class="${beat(m.model_rho, m.null_rho)}">${num3(m.model_rho)}</b>
        <b>${num3(m.null_rho)}</b>
      </div>
      <div class="score-row">
        <span title="Adjacent pairs of the census ordering whose order this arm reverses. ρ alone is never quoted without it (A17).">rank inversions</span>
        <b>${m.model_inv == null ? "—" : m.model_inv}</b>
        <b>${m.null_inv == null ? "—" : m.null_inv}</b>
      </div>
    </div>
    <p class="hint">The right column is the null: <b>last year's census, reused unchanged</b>.
      It wins both, every year — so this map reproduces the official pattern rather than
      improving on it. Numbers come from <code>metrics.json</code>, straight out of the
      metrics sink.</p>`;
}

function buildHotspots(gj) {
  const feats = gj.features.map((f) => f.properties).filter((p) => p.in_ranking);
  feats.sort((a, b) => b.pred_density - a.pred_density);
  const max = Math.max(...feats.map((p) => p.pred_density), 1e-9);
  document.getElementById("rank-n").textContent = `n = ${feats.length}`;
  document.getElementById("hotspots").innerHTML = feats.map((p, i) => {
    const agree = p.model_rank === p.official_rank;
    const [r, g, b] = rampRGB(p.pred_density / MAX_FRAC);
    return `<li tabindex="0" data-name="${p.name}" class="${agree ? "" : "mismatch"}">
      <span class="rank">${i + 1}</span>
      <span class="nm">${p.label || p.name}</span>
      <span class="bar"><i style="width:${(100 * p.pred_density / max).toFixed(1)}%;
        background:rgb(${r},${g},${b})"></i></span>
      <span class="val">${pct1(p.pred_density)}</span>
      <span class="ranks ${agree ? "ok" : "off"}"
        title="model rank ${Math.round(p.model_rank)}, census rank ${Math.round(p.official_rank)}">
        M${Math.round(p.model_rank)}·C${Math.round(p.official_rank)}</span>
    </li>`;
  }).join("");
  const select = (name) => {
    const l = layersByName.get(name);
    if (l) { flyToFeature(l.getBounds()); l.openPopup(); }
  };
  document.querySelectorAll("#hotspots li").forEach((li) => {
    li.onclick = () => select(li.dataset.name);
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(li.dataset.name); } };
  });
}

function buildHowTo() {
  const yrs = (allMetrics && allMetrics.years) || {};
  const rows = YEARS.map((y) => {
    const m = yrs[String(y)] || {};
    return `<tr><td>${y}</td><td>${num3(m.model_iou)}</td><td>${num3(m.null_iou)}</td>
      <td>${num3(m.model_rho)}</td><td>${num3(m.null_rho)}</td></tr>`;
  }).join("");
  document.getElementById("howto-body").innerHTML = `
    <p><strong>What this shows.</strong> A U-Net predicts the fraction of each pixel under coca
      from free Sentinel-1 + Sentinel-2 imagery. Every pixel displayed was predicted by a model
      that never trained on it — six models, one per spatial fold, each scoring only its own
      held-out fold (pre-registration A20/A22).</p>
    <p><strong>Both pre-registered nulls beat the model.</strong> The comparison is against
      <em>reusing the previous census unchanged</em>. It locates coca better and ranks
      municipalities better, in all six years:</p>
    <table class="mini">
      <thead><tr><th>year</th><th>IoU model</th><th>IoU null</th><th>ρ model</th><th>ρ null</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p>The decision rules and the ≥5/6 bar were fixed in writing before any of these numbers
      existed, and the model was not retuned afterwards. The defensible claim is that free
      imagery <em>reproduces</em> Colombia's official pattern — not that it improves on it. That
      is a real negative result, and it is the deliverable.</p>
    <p><strong>Resolution: ~1 km cells, not fields.</strong> The census labels are a ~1 km grid
      with one value spread over every pixel of a cell, so agreement can only ever be about
      <em>which ~1 km cells</em> contain coca. Published rasters are blurred to 75 m on purpose:
      plot-level geometry would be usable for targeting, which is not what this is for.</p>
    <p><strong>Gaps in the raster are honest.</strong> Blank patches are tile positions dropped to
      keep the folds pixel-disjoint; nobody scored them, so nothing is drawn there. The census
      layer is masked to the identical footprint, so the two layers always cover the same ground.</p>
    <p><strong>Hectares are out of scope.</strong> A separate pre-registered test found no
      imagery-based estimator beats a historical average of past censuses at totalling hectares:
      coca is ~3.6% of this region and the whole 2020→2024 swing is 0.8% of its area, 20–30×
      below ordinary year-to-year variation in the same imagery. Nothing here is calibrated to
      match a census total, because fitting a scalar to the census makes it match by
      construction.</p>`;
}

// ---- legend ------------------------------------------------------------------
function buildLegend() {
  const stops = RAMP.map((c, i) => `${c} ${(100 * i / (RAMP.length - 1)).toFixed(0)}%`).join(", ");
  const label = mode === "official" ? "Census cover" : mode === "swipe" ? "Cover (both layers)" : "Model cover";
  document.getElementById("legend").innerHTML = `
    <div class="legend-title">${label}</div>
    <div class="ramp" style="background:linear-gradient(90deg, ${stops})"></div>
    <div class="ramp-ticks"><span>${(100 * threshold).toFixed(0)}%</span><span>${(100 * MAX_FRAC).toFixed(0)}%+</span></div>
    <div class="legend-foot">share of each 75 m pixel under coca · blank = not scored (fold gap)</div>`;
}

// ---- mode switching ----------------------------------------------------------
const HINTS = {
  model: "Model prediction, out-of-fold: every pixel comes from a model that never trained on it.",
  official: "Official census, rasterised from the ~1 km grid and masked to the same footprint.",
  swipe: "Drag the divider: model on the left, census on the right. Same ramp, same ground.",
};
function setMode(next) {
  mode = next;
  for (const [id, key] of [["lyr-model", "model"], ["lyr-official", "official"], ["lyr-swipe", "swipe"]]) {
    const btn = document.getElementById(id);
    btn.classList.toggle("on", key === mode);
    btn.setAttribute("aria-pressed", String(key === mode));
  }
  document.getElementById("layer-hint").textContent = HINTS[mode];
  renderLayers();
  buildLegend();
}
document.getElementById("lyr-model").onclick = () => setMode("model");
document.getElementById("lyr-official").onclick = () => setMode("official");
document.getElementById("lyr-swipe").onclick = () => setMode("swipe");

document.getElementById("opacity").addEventListener("input", (e) => {
  overlayOpacity = e.target.value / 100;
  document.getElementById("op-val").textContent = `${e.target.value}%`;
  for (const l of Object.values(layers)) if (l) l.setOpacity(overlayOpacity);
});
document.getElementById("threshold").addEventListener("input", (e) => {
  threshold = e.target.value / 100;
  document.getElementById("th-val").textContent = `${e.target.value}%`;
  renderLayers();
  buildLegend();
});

// ---- year loading ------------------------------------------------------------
function busy(on) { document.getElementById("loading").hidden = !on; }

async function loadYear(year) {
  busy(true);
  currentYear = year;
  try {
    const gj = await loadChoropleth(year);
    buildScoreboard(year);
    buildHotspots(gj);
    await loadRasters(year);
    renderLayers();
    buildLegend();
    fitAoi();
  } catch (e) {
    console.error("failed to load year", year, e);
  } finally {
    busy(false);
  }
}

/* Fit after layout has settled. The old code called fitBounds once, inline, before the web
 * font had loaded — the panel then reflowed, the map got wider, and Leaflet kept the zoom
 * it had computed for the narrower box, which is why the map opened showing half of
 * Colombia with the study area as a speck. */
function fitAoi() {
  map.invalidateSize();
  /* Prefer the published study-area bbox over the municipal envelope: Curumani and San
   * Jose de Cucuta reach well past the AOI, so their bounds framed the map two zoom
   * levels too wide with the data as a speck. Falls back to the polygons if metrics.json
   * is missing. */
  const bb = allMetrics && allMetrics.aoi_bbox;
  const b = bb ? L.latLngBounds([[bb[1], bb[0]], [bb[3], bb[2]]]) : aoiBounds;
  if (!b || !b.isValid()) return;
  map.fitBounds(b, { padding: [24, 24] });
}

const yearSel = document.getElementById("year");
yearSel.innerHTML = YEARS.map((y) => `<option value="${y}">${y}</option>`).join("");
yearSel.onchange = (e) => loadYear(Number(e.target.value));

(async function init() {
  try {
    const res = await fetch("data/metrics.json");
    if (res.ok) allMetrics = await res.json();
  } catch (e) {
    console.warn("metrics.json unavailable — the scoreboard will show em dashes", e);
  }
  if (allMetrics && allMetrics.density_scale) densityScale = allMetrics.density_scale;
  if (allMetrics && allMetrics.data_generation) {
    document.getElementById("gen-chip").textContent = `${allMetrics.data_generation} · out-of-fold`;
  }
  buildHowTo();
  const q = Number(new URLSearchParams(location.search).get("year"));
  const initial = YEARS.includes(q) ? q : YEARS[YEARS.length - 1];
  yearSel.value = String(initial);
  await loadYear(initial);
  setMode("model");
})();

if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitAoi);
window.addEventListener("resize", () => { map.invalidateSize(); applySwipe(); });
map.on("move zoom zoomend viewreset moveend", applySwipe);
