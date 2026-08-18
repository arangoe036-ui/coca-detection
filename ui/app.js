/* "Where is the coca?" — Catatumbo. Vanilla Leaflet, no build step, no CDN.
 *
 * Design rules this file follows, each one the result of getting it wrong first:
 *
 * 1. NOTHING is fetched from the internet to make the page work. Leaflet, the GeoTIFF
 *    reader, the satellite photos, the coca layers and the numbers are all local files. A
 *    viewer on a restricted or tunnelled connection previously got a black rectangle, which
 *    is indistinguishable from "no coca here".
 * 2. Plain words on screen. The underlying work is pre-registered and full of terms like
 *    out-of-fold, IoU and Spearman rho; none of that belongs in the interface. It lives in
 *    the "Technical detail" panel, once, for people who want it.
 * 3. The satellite photo is on by default and the coca sits on top of it, so the map always
 *    shows real ground instead of a floating heat map.
 * 4. Squares are drawn crisply, not smoothed: 75 m is the honest unit of this map, and
 *    blurring it would imply precision that does not exist.
 * 5. Every number on screen comes from data/metrics.json, copied out of the project's
 *    metrics file. Nothing is typed in by hand.
 */

/* Fail visibly, never blankly. */
(function requireDeps() {
  const missing = [];
  if (typeof L === "undefined") missing.push("vendor/leaflet.js");
  if (typeof parseGeoraster === "undefined") missing.push("vendor/georaster.min.js");
  if (typeof GeoRasterLayer === "undefined") missing.push("vendor/georaster-layer-for-leaflet.min.js");
  if (!missing.length) return;
  const el = document.getElementById("map");
  if (el) {
    el.innerHTML = `<div class="fatal"><b>The map could not start.</b>
      <span>These files did not load:</span><code>${missing.join("<br>")}</code>
      <span>They are stored next to this page, so this is a serving problem, not an internet
      problem.</span></div>`;
  }
  throw new Error("missing vendor assets: " + missing.join(", "));
})();

const YEARS = [2019, 2020, 2021, 2022, 2023, 2024];
const ESRI_ZOOM = 13;

/* Yellow → orange → deep red. Reads as "coca here" over green vegetation and grey-brown
 * ground, and stays legible on the plain dark background too. ONE ramp for both the model
 * layer and the official survey: comparing two maps drawn in different colours is not a
 * comparison. */
const COCA_RAMP = ["#ffe98a", "#ffcc4d", "#ff9f43", "#f4692c", "#d92b1f", "#8c1007"];
const ACCENT = "#f0a02a";
let displayMax = 0.35;      // top of the colour scale; overridden from metrics.json
let densityScale = 10000;   // published rasters are uint16 holding cover × this
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const fmt = (v) => Math.round(v || 0).toLocaleString("en-US");
const pct1 = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");
const num2 = (v) => (v == null ? "—" : Number(v).toFixed(2));

function rampRGB(t) {
  const hexes = COCA_RAMP.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  t = Math.max(0, Math.min(1, t));
  const x = t * (hexes.length - 1), i = Math.floor(x), f = x - i;
  const a = hexes[i], b = hexes[Math.min(i + 1, hexes.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return [c[0], c[1], c[2]];
}

// ---- map ---------------------------------------------------------------------
const map = L.map("map", { zoomControl: true, minZoom: 6, maxZoom: 18, zoomSnap: 0.25 })
  .setView([8.8, -72.85], 9);
L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);

/* A plain street map underneath, if the internet is there. It is the only optional network
 * call on the page; when it fails the satellite photo shipped with the page still covers the
 * whole study area. */
const streetLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "© OpenStreetMap · © CARTO", maxZoom: 20,
}).addTo(map);
let tileErrors = 0;
streetLayer.on("tileerror", () => {
  if (++tileErrors === 6) document.getElementById("basemap-warn").hidden = false;
});
streetLayer.on("tileload", () => { document.getElementById("basemap-warn").hidden = true; });

for (const [name, z] of [["sat", 300], ["esri", 320], ["rasterA", 420], ["rasterB", 421], ["outline", 450]]) {
  map.createPane(name);
  map.getPane(name).style.zIndex = z;
  if (name !== "outline") map.getPane(name).style.pointerEvents = "none";
}

const esriLayer = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { attribution: "Imagery © Esri, Maxar", maxZoom: 19, pane: "esri", opacity: 0 }).addTo(map);
let esriOn = false;
function setEsri(on) {
  esriOn = on;
  esriLayer.setOpacity(on ? 1 : 0);
  document.getElementById("esri-chk").checked = on;
}
document.getElementById("esri-chk").onchange = (e) => setEsri(e.target.checked);
map.on("zoomend", () => { if (map.getZoom() >= ESRI_ZOOM && !esriOn && navigator.onLine) setEsri(true); });

// ---- state -------------------------------------------------------------------
let mode = "model";
let threshold = 0.05;
let overlayOpacity = 0.7;
let rasters = {};
let layers = {};
let satLayer = null;
let outlineLayer;
let allMetrics = null;
let aoiBounds = null;
const layersByName = new Map();

// ---- coca layers -------------------------------------------------------------
function colorFn(vals) {
  const raw = vals[0];
  if (raw == null || Number.isNaN(raw) || raw <= 0) return null;
  const f = raw / densityScale;
  if (f < threshold) return null;
  const [r, g, b] = rampRGB(f / displayMax);
  /* Alpha climbs with coverage so faint areas stay faint rather than switching on hard at
   * the threshold — but never below 0.55, because a square that passes the threshold has to
   * be visible against a busy satellite photo. */
  const a = 0.55 + 0.45 * Math.min(1, f / (displayMax * 0.5));
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function makeRasterLayer(which, pane) {
  const georaster = rasters[which];
  if (!georaster) return null;
  const lyr = new GeoRasterLayer({
    georaster, pane, opacity: overlayOpacity,
    resolution: 1024,            // enough that a 75 m square covers >= 1 screen pixel
    pixelValuesToColorFn: colorFn,
  });
  lyr.addTo(map);
  return lyr;
}

function renderLayers() {
  for (const k of Object.keys(layers)) if (layers[k]) map.removeLayer(layers[k]);
  layers = {};
  if (mode === "model") layers.a = makeRasterLayer("model", "rasterA");
  else if (mode === "official") layers.a = makeRasterLayer("official", "rasterA");
  else {
    layers.a = makeRasterLayer("official", "rasterA");   // right of the divider
    layers.b = makeRasterLayer("model", "rasterB");      // left of it
  }
  applySwipe();
}

async function fetchRaster(kind, year) {
  try {
    /* ?v matters: these files changed type and scale between revisions, and a cached older
     * one would be divided by the new scale and render as an empty map. */
    const res = await fetch(`data/${kind}_${year}_cog.tif?v=5`);
    if (!res.ok) throw new Error(`${kind} ${res.status}`);
    return await parseGeoraster(await res.arrayBuffer());
  } catch (e) {
    console.warn(`${kind} raster unavailable for ${year}`, e);
    return null;
  }
}

async function loadRasters(year) {
  const [model, official, rgb] = await Promise.all([
    fetchRaster("density", year), fetchRaster("official", year), fetchRaster("rgb", year),
  ]);
  rasters = { model, official, rgb };
  if (satLayer) { map.removeLayer(satLayer); satLayer = null; }
  if (rgb) {
    /* The satellite photo is the same composite the model looked at, for the same year, so
     * the ground under a prediction is the ground that prediction was made from. A global
     * undated mosaic could be showing a different decade. */
    satLayer = new GeoRasterLayer({ georaster: rgb, pane: "sat", resolution: 1024 });
    satLayer.addTo(map);
    applySatVisibility();
  }
}

function applySatVisibility() {
  if (!satLayer) return;
  const on = document.getElementById("sat-chk").checked;
  satLayer.getContainer().style.display = on ? "" : "none";
}
document.getElementById("sat-chk").onchange = applySatVisibility;

// ---- side by side ------------------------------------------------------------
let swipeX = 0.5;
function applySwipe() {
  const handle = document.getElementById("swipe-handle");
  const labels = document.getElementById("swipe-labels");
  const on = mode === "swipe";
  handle.hidden = !on;
  labels.hidden = !on;
  const cA = layers.a && layers.a.getContainer();
  const cB = layers.b && layers.b.getContainer();
  if (!on) {
    if (cA) cA.style.clip = "";
    if (cB) cB.style.clip = "";
    return;
  }
  /* `clip: rect()` on each layer's own container, in layer-point coordinates — the
   * leaflet-side-by-side technique. Two other approaches failed: clipping only the top
   * layer left the other drawn underneath it everywhere, and `clip-path: inset()` on the
   * pane erased both, because Leaflet panes are zero-size boxes whose children merely
   * overflow. Layer points move with the map, hence the move/zoom handlers at the bottom. */
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
})();

// ---- town outlines + popups --------------------------------------------------
function flyToFeature(bounds) {
  if (navigator.onLine) setEsri(true);
  if (reduceMotion) map.fitBounds(bounds, { maxZoom: ESRI_ZOOM });
  else map.flyToBounds(bounds, { maxZoom: ESRI_ZOOM, duration: 1.1 });
}

function popupHtml(p) {
  const name = p.label || p.name;
  if (!p.in_ranking) {
    return `<div class="popup-title">${name}</div>
      <div class="popup-row"><span>Not included</span><b>—</b></div>
      <div class="popup-note">The official survey has no figure for this town this year, or too
        little of it was measured here — so it is left out of every number on this page.</div>`;
  }
  const agree = p.model_rank === p.official_rank;
  return `<div class="popup-title">${name}</div>
    <div class="popup-rank ${agree ? "ok" : "off"}">
      ${agree ? `Both rank it #${Math.round(p.model_rank)}`
              : `This map #${Math.round(p.model_rank)} · official #${Math.round(p.official_rank)}`}</div>
    <div class="popup-row"><span>Ground under coca — this map</span><b>${pct1(p.pred_density)}</b></div>
    <div class="popup-row"><span>Ground under coca — official</span><b>${pct1(p.official_density)}</b></div>
    <div class="popup-row"><span>Official total, whole town</span>
      <b>${p.official_ha ? fmt(p.official_ha) + " ha" : "—"}</b></div>
    <div class="popup-note">The two percentages are the fair comparison — both measured over the
      same ground. The official total covers the whole town, of which ${pct1(p.oof_share_of_canvas)}
      was measured here, so comparing hectares directly would mislead.</div>`;
}

async function loadTowns(year) {
  const res = await fetch(`data/municipal_coca_${year}.geojson?v=5`);
  if (!res.ok) throw new Error(`towns ${res.status}`);
  const gj = await res.json();
  if (outlineLayer) map.removeLayer(outlineLayer);
  layersByName.clear();
  outlineLayer = L.geoJSON(gj, {
    pane: "outline",
    style: (f) => ({
      color: f.properties.in_ranking ? "#eaf1f8" : "#93a1b1",
      weight: f.properties.in_ranking ? 1.2 : 1,
      dashArray: f.properties.in_ranking ? null : "3 4",
      fillColor: "#ffffff", fillOpacity: 0.01,
    }),
    onEachFeature: (f, lyr) => {
      const p = f.properties;
      layersByName.set(p.name, lyr);
      lyr.bindPopup(popupHtml(p), { maxWidth: 320 });
      lyr.on("mouseover", () => lyr.setStyle({ weight: 2.6, color: ACCENT }));
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
  const cls = (a, b, higherBetter = true) => {
    if (a == null || b == null) return "";
    return (higherBetter ? a > b : a < b) ? "win" : "loss";
  };
  document.getElementById("scoreboard").innerHTML = `
    <h2>Does it match the official survey?</h2>
    <div class="score">
      <div class="score-row head"><span></span><span>this map</span><span>last year's survey</span></div>
      <div class="score-row">
        <span title="Of the 1 km squares containing coca, the share both maps agree on. 1.00 would be a perfect match.">same squares found</span>
        <b class="${cls(m.model_iou, m.null_iou)}">${num2(m.model_iou)}</b>
        <b>${num2(m.null_iou)}</b>
      </div>
      <div class="score-row">
        <span title="How closely the order of towns matches the official order. 1.00 is identical.">same order of towns</span>
        <b class="${cls(m.model_rho, m.null_rho)}">${num2(m.model_rho)}</b>
        <b>${num2(m.null_rho)}</b>
      </div>
      <div class="score-row">
        <span title="Neighbouring pairs of towns that are swapped compared with the official order. Fewer is better.">towns out of order</span>
        <b class="${cls(m.model_inv, m.null_inv, false)}">${m.model_inv == null ? "—" : m.model_inv}</b>
        <b>${m.null_inv == null ? "—" : m.null_inv}</b>
      </div>
    </div>
    <p class="hint">Higher is better on the first two rows, lower on the last. The right-hand
      column is the simplest possible alternative: <b>just reuse last year's official survey</b>.
      It scores better in every year — so this map mostly re-finds coca officials had already
      mapped, rather than finding coca they missed. That is the honest headline, and the test
      rules were written down before any of it was measured.</p>`;
}

function buildTownList(gj) {
  const feats = gj.features.map((f) => f.properties).filter((p) => p.in_ranking);
  feats.sort((a, b) => b.pred_density - a.pred_density);
  const max = Math.max(...feats.map((p) => p.pred_density), 1e-9);
  document.getElementById("rank-n").textContent = `${feats.length} towns`;
  document.getElementById("hotspots").innerHTML = feats.map((p, i) => {
    const agree = p.model_rank === p.official_rank;
    const [r, g, b] = rampRGB(p.pred_density / displayMax);
    return `<li tabindex="0" data-name="${p.name}">
      <span class="rank">${i + 1}</span>
      <span class="nm">${p.label || p.name}</span>
      <span class="bar"><i style="width:${(100 * p.pred_density / max).toFixed(1)}%;
        background:rgb(${r},${g},${b})"></i></span>
      <span class="val">${pct1(p.pred_density)}</span>
      <span class="ranks ${agree ? "ok" : "off"}"
        title="${agree ? "same position as the official survey"
                       : `this map #${Math.round(p.model_rank)}, official survey #${Math.round(p.official_rank)}`}">
        ${agree ? "same" : `${Math.round(p.model_rank)}/${Math.round(p.official_rank)}`}</span>
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

function buildExplainers() {
  document.getElementById("howto-body").innerHTML = `
    <p><strong>The colours are coca.</strong> Each coloured square is 75 m across and its colour
      says how much of that square is planted with coca — pale yellow a little, deep red a lot.
      Everything uncoloured is ordinary ground: forest, pasture, towns, rivers.</p>
    <p><strong>Two maps of the same thing.</strong> "This map" is made by a computer model that
      only ever sees satellite photos. "Official survey" is Colombia's government coca census for
      the same year. Use <em>Side by side</em> and drag the divider to compare them: same colours,
      same ground, so anything that looks different is a real difference.</p>
    <p><strong>The official survey looks blocky for a reason.</strong> It is published as 1 km
      squares with one figure each. That is also the finest question either map can answer —
      "which 1 km square", never "which field".</p>
    <p><strong>The blank gaps are honest.</strong> Empty rectangles are ground nobody measured.
      Those strips were held back so the model could be tested on land it had never learned from;
      testing it on land it had already studied would flatter it. Both maps are blanked in exactly
      the same places, so the comparison stays fair.</p>
    <p><strong>What it is for.</strong> Seeing where coca is concentrated and how that shifts
      between years — not counting hectares, and not locating farms. Everything here is
      deliberately blurred to 75 m and summarised by town.</p>`;

  const yrs = (allMetrics && allMetrics.years) || {};
  const rows = YEARS.map((y) => {
    const m = yrs[String(y)] || {};
    return `<tr><td>${y}</td><td>${num2(m.model_iou)}</td><td>${num2(m.null_iou)}</td>
      <td>${num2(m.model_rho)}</td><td>${num2(m.null_rho)}</td></tr>`;
  }).join("");
  document.getElementById("technical-body").innerHTML = `
    <p>A U-Net (ResNet-34 encoder, 18 channels: ten Sentinel-2 L2A bands, six indices,
      Sentinel-1 RTC VV/VH) predicts coca cover fraction per 20 m pixel. Labels are Colombia's
      ~1 km census density grid burned uniformly into each cell, so every figure here is
      cell-level agreement, never field-level.</p>
    <p>Displayed predictions are <strong>out-of-fold</strong>: six models, one per spatial fold of
      a stratified macro-block split, each scoring only its own held-out fold. Folds are
      pixel-disjoint; tile positions straddling a boundary are dropped, which is what the blank
      strips are.</p>
    <p>Both comparisons are against a persistence null — the previous census carried forward.
      Decision rules, thresholds and the ≥5/6 bar were pre-registered before anything was
      computed, and the model was not retuned afterwards.</p>
    <table class="mini">
      <thead><tr><th>year</th><th>IoU model</th><th>IoU null</th><th>ρ model</th><th>ρ null</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p>Presence IoU at a 0.02 cover cut; Spearman ρ over ${(yrs["2024"] || {}).n_muni || 9}
      municipalities, always reported with adjacent-inversion counts because at this n one swap
      moves ρ materially. The model loses 0/6 folds on IoU and 0/6 years on ρ, so both the spatial
      and ranking claims are reported as negative results. Totals are uncalibrated: fitting a
      scalar to the census would make them match by construction.</p>
    <p>The colour scale tops out at ${(100 * displayMax).toFixed(0)}% cover and is shared by both
      layers so they stay comparable; the official layer's brightest ~1% of squares clip there.
      Rasters are uint16 holding cover × ${densityScale} at 75 m, and the satellite layer is the
      same year's Sentinel-2 composite the model was given.</p>`;
}

// ---- legend ------------------------------------------------------------------
function buildLegend() {
  const stops = COCA_RAMP.map((c, i) => `${c} ${(100 * i / (COCA_RAMP.length - 1)).toFixed(0)}%`).join(", ");
  const label = mode === "official" ? "Official survey" : mode === "swipe" ? "Both maps" : "This map";
  document.getElementById("legend").innerHTML = `
    <div class="legend-title">${label} — coca cover</div>
    <div class="ramp" style="background:linear-gradient(90deg, ${stops})"></div>
    <div class="ramp-ticks"><span>${(100 * threshold).toFixed(0)}%</span>
      <span>${(100 * displayMax).toFixed(0)}%+</span></div>
    <div class="legend-foot">how much of each 75 m square is coca · blank = not measured</div>`;
}

// ---- mode switching ----------------------------------------------------------
const HINTS = {
  model: "Coca found by this project, from free satellite photos of the year you picked.",
  official: "Colombia's official government coca survey for the same year, drawn the same way.",
  swipe: "Drag the divider: this map on the left, the official survey on the right.",
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
  document.getElementById("th-val").textContent = `${e.target.value}% coca`;
  renderLayers();
  buildLegend();
});

// ---- loading -----------------------------------------------------------------
function busy(on) { document.getElementById("loading").hidden = !on; }

async function loadYear(year) {
  busy(true);
  try {
    const gj = await loadTowns(year);
    buildScoreboard(year);
    buildTownList(gj);
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

/* Fit after layout settles. Fitting once, inline, framed the map for a narrower panel and
 * opened it showing half of Colombia. The study-area box is used rather than the town
 * outlines because two towns reach far outside it. */
function fitAoi() {
  map.invalidateSize();
  const bb = allMetrics && allMetrics.aoi_bbox;
  const b = bb ? L.latLngBounds([[bb[1], bb[0]], [bb[3], bb[2]]]) : aoiBounds;
  if (b && b.isValid()) map.fitBounds(b, { padding: [24, 24] });
}

const yearSel = document.getElementById("year");
yearSel.innerHTML = YEARS.map((y) => `<option value="${y}">${y}</option>`).join("");
yearSel.onchange = (e) => loadYear(Number(e.target.value));

(async function init() {
  try {
    const res = await fetch("data/metrics.json?v=5");
    if (res.ok) allMetrics = await res.json();
  } catch (e) {
    console.warn("metrics.json unavailable — the scoreboard will show dashes", e);
  }
  if (allMetrics) {
    if (allMetrics.density_scale) densityScale = allMetrics.density_scale;
    if (allMetrics.display_max) displayMax = allMetrics.display_max;
  }
  buildExplainers();
  const q = Number(new URLSearchParams(location.search).get("year"));
  const initial = YEARS.includes(q) ? q : YEARS[YEARS.length - 1];
  yearSel.value = String(initial);
  await loadYear(initial);
  setMode("model");
})();

if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitAoi);
window.addEventListener("resize", () => { map.invalidateSize(); applySwipe(); });
map.on("move zoom zoomend viewreset moveend", applySwipe);
