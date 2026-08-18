/* Coca hotspot monitor v2 (P5 + v2 Part B).
   Municipal choropleth + pixel-level density overlay (downsampled COG) over a
   light basemap, with Esri imagery crossfade for zoom-in, opacity/threshold
   controls, draw-a-box hectare sums, and a data-driven year selector with a
   nowcast badge for years lacking an official census. */

// Years shown. 2026-08-10: ONLY years with a real official census are listed now.
//
// The 2025/2026 "nowcast" years were removed along with their artifacts. src/nowcast.py
// never set cfg["year"], so those files carried the 2023 census in `official_ha`, a `ratio`
// computed against 2023, and a `year` column that literally read 2023 — and popupHtml()
// below does not branch on `nowcast`, so clicking a municipality on the 2025 map rendered
// "Official 23,030 ha / Ratio 1.50x" against a census that does not exist. The panel-level
// caveat was defeated by the popup one click away. That is fabricated validation, and it
// could not be caveated into correctness because the wrong values were inside the data.
//
// Every year below has a genuine census, so `official_ha` and `ratio` are now always real.
//
// 2026-08-18: regenerated on gen4 from OUT-OF-FOLD predictions (prereg A22) — six models,
// one per spatial fold, each predicting only ground it never trained on. All six years now
// have both a choropleth and a 75 m density COG. The previous artifacts were built
// 2026-08-09 on gen1 data (reflectance-offset bug, 25% blank coverage, a leaky split) from
// a checkpoint that no longer exists, and only 2023 had an overlay.
const YEARS = [
  { year: 2019, nowcast: false },
  { year: 2020, nowcast: false },
  { year: 2021, nowcast: false },
  { year: 2022, nowcast: false },
  { year: 2023, nowcast: false },
  { year: 2024, nowcast: false },
];
const BAND = 0.40;          // retained: referenced by the (now unreachable) nowcast branch
let currentMeta = {};
const INSPECT_ZOOM = 13;
const ACCENT = "#e6a01f";
const RAMP = ["#ffffe5", "#fee391", "#fe9929", "#cc4c02", "#8c2d04"]; // YlOrBr light->dark
const BREAKS = [0, 500, 2000, 5000, 15000];      // municipal choropleth ha bins
const MAX_FRAC = 0.29;   // overlay normalization: must be >= the COG max cover fraction,
                         // or the top of the ramp clips silently. Measured across the six
                         // gen4 COGs: 0.261-0.282, so 0.29 leaves a little headroom.
                         // Was 0.26 (gen1-era), which the 2021/2023/2024 rasters exceed.
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const fmt = (v) => Math.round(v || 0).toLocaleString("en-US");
function colorFor(ha) {
  const v = ha || 0;
  return v >= BREAKS[4] ? RAMP[4] : v >= BREAKS[3] ? RAMP[3]
       : v >= BREAKS[2] ? RAMP[2] : v >= BREAKS[1] ? RAMP[1] : RAMP[0];
}
// Continuous YlOrBr ramp for the raster overlay: t in [0,1] -> "rgb(...)".
function rampRGB(t) {
  const hexes = RAMP.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
  t = Math.max(0, Math.min(1, t));
  const x = t * (hexes.length - 1), i = Math.floor(x), f = x - i;
  const a = hexes[i], b = hexes[Math.min(i + 1, hexes.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---- map + base layers -------------------------------------------------------
const map = L.map("map", { zoomControl: true, minZoom: 6 }).setView([8.65, -72.9], 9);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  attribution: "© OpenStreetMap contributors © CARTO", maxZoom: 20,
}).addTo(map);

map.createPane("imagery");
const imgPane = map.getPane("imagery");
imgPane.classList.add("leaflet-imagery-pane");
imgPane.style.zIndex = 250;
imgPane.style.opacity = 0;
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  maxZoom: 19, pane: "imagery",
}).addTo(map);

// density overlay sits above the choropleth but passes clicks through to it
map.createPane("density");
map.getPane("density").style.zIndex = 450;
map.getPane("density").style.pointerEvents = "none";

let imageryOn = false;
function setImagery(on) {
  imageryOn = on;
  imgPane.style.opacity = on ? 1 : 0;
  const btn = document.getElementById("basemap-toggle");
  btn.setAttribute("aria-pressed", String(on));
  btn.textContent = on ? "Show map" : "Show imagery";
}
document.getElementById("basemap-toggle").onclick = () => setImagery(!imageryOn);
map.on("zoomend", () => { if (map.getZoom() >= INSPECT_ZOOM && !imageryOn) setImagery(true); });

// ---- density overlay (COG via georaster) -------------------------------------
let densityLayer, densityRaster;
let threshold = 0.05, overlayOpacity = 0.8;

function makeDensityLayer() {
  if (densityLayer) map.removeLayer(densityLayer);
  if (!densityRaster) return;
  densityLayer = new GeoRasterLayer({
    georaster: densityRaster,
    pane: "density",
    opacity: overlayOpacity,
    resolution: 256,
    pixelValuesToColorFn: (vals) => {
      const f = vals[0];
      if (f == null || f <= 0 || f < threshold) return null; // transparent
      return rampRGB(f / MAX_FRAC);
    },
  });
  densityLayer.addTo(map);
}

async function loadOverlay(year) {
  try {
    const buf = await (await fetch(`data/density_${year}_cog.tif`)).arrayBuffer();
    densityRaster = await parseGeoraster(buf);
    makeDensityLayer();
  } catch (e) {
    densityRaster = null;
    console.warn("density overlay unavailable for", year, e);
  }
}

document.getElementById("opacity").addEventListener("input", (e) => {
  overlayOpacity = e.target.value / 100;
  document.getElementById("op-val").textContent = `${e.target.value}%`;
  if (densityLayer) densityLayer.setOpacity(overlayOpacity);
});
document.getElementById("threshold").addEventListener("input", (e) => {
  threshold = e.target.value / 100;
  document.getElementById("th-val").textContent = `${e.target.value}%`;
  makeDensityLayer(); // recolor by rebuilding with the new cutoff
});

// ---- draw-a-box -> sum hectares: REMOVED 2026-08-10 --------------------------
// A hectare-measurement tool is out of scope for a location-only deliverable, and this
// summed the 75 m blurred overlay, so its answer was never a real area measurement.

// ---- choropleth --------------------------------------------------------------
let layer;
const layersByName = new Map();

function flyToFeature(bounds) {
  setImagery(true);
  if (reduceMotion) map.fitBounds(bounds, { maxZoom: INSPECT_ZOOM });
  else map.flyToBounds(bounds, { maxZoom: INSPECT_ZOOM, duration: 1.2 });
}

function popupHtml(p) {
  // Municipalities inside the AOI with no official value, or too little out-of-fold
  // coverage to score, are shown as unranked rather than as a prediction of zero. The
  // A22 membership rule is deliberately independent of the prediction.
  if (!p.in_ranking) {
    return `<div class="popup-title">${p.name}</div>
      <div class="popup-row"><span>Not ranked</span><b>—</b></div>
      <div class="popup-note">No official census value for this year, or under the
        out-of-fold coverage floor. Excluded from every metric.</div>`;
  }
  const ratio = p.official_ha ? (p.predicted_ha / p.official_ha).toFixed(2) + "×" : "n/a";
  const pct = (v) => (100 * v).toFixed(1) + "%";
  const rank = `#${Math.round(p.model_rank)} vs #${Math.round(p.official_rank)} official`;
  return `<div class="popup-title">${p.name}</div>
    <div class="popup-row"><span>Rank (model / official)</span><b>${rank}</b></div>
    <div class="popup-row"><span>Density, model / official</span>
      <b>${pct(p.pred_density)} / ${pct(p.official_density)}</b></div>
    <div class="popup-row"><span>Predicted</span><b>${fmt(p.predicted_ha)} ha</b></div>
    <div class="popup-row"><span>Official</span><b>${p.official_ha ? fmt(p.official_ha) + " ha" : "—"}</b></div>
    <div class="popup-row"><span>Ratio</span><b>${ratio}</b></div>
    <div class="popup-note">Density is the like-for-like figure: both sides are means over the
      <em>same</em> ${fmt(p.oof_px)} out-of-fold pixels. Hectares are not — the model total is
      that density extrapolated over this municipality's ${fmt(p.canvas_share_hint)} ha inside
      the study area (${pct(p.oof_share_of_canvas)} of it directly predicted), while the census
      counts the whole municipality. Predictions are uncalibrated.</div>`;
}

async function loadChoropleth(year) {
  try {
    const res = await fetch(`data/municipal_coca_${year}.geojson`);
    if (!res.ok) throw new Error(`choropleth fetch ${res.status}`);
    const gj = await res.json();
    if (layer) map.removeLayer(layer);
    layersByName.clear();
    layer = L.geoJSON(gj, {
      style: (f) => ({ fillColor: colorFor(f.properties.predicted_ha), fillOpacity: 0.55, color: "#1b2127", weight: 1 }),
      onEachFeature: (f, lyr) => {
        const p = f.properties;
        layersByName.set(p.name, lyr);
        lyr.bindPopup(popupHtml(p));
        lyr.on("mouseover", () => lyr.setStyle({ weight: 3, color: ACCENT }));
        lyr.on("mouseout", () => layer.resetStyle(lyr));
        lyr.on("click", () => flyToFeature(lyr.getBounds()));
      },
    }).addTo(map);
    map.invalidateSize();
    if (layer.getBounds().isValid()) map.fitBounds(layer.getBounds(), { padding: [20, 20] });
    buildPanel(gj);
  } catch (e) {
    console.error("Choropleth failed to load:", e); // map stays on Catatumbo, not the ocean
  }
}

function buildPanel(gj) {
  const feats = gj.features.map((f) => f.properties);
  const predTotal = feats.reduce((s, p) => s + (p.predicted_ha || 0), 0);
  const offTotal = feats.reduce((s, p) => s + (p.official_ha || 0), 0);

  if (currentMeta.nowcast) {
    // Never show a bare figure for a censusless year — a ±band range only.
    const lo = fmt(predTotal * (1 - BAND)), hi = fmt(predTotal * (1 + BAND));
    document.getElementById("totals").innerHTML = `
      <div class="stat wide nowcast"><span>Estimated coca (unvalidated)</span>
        <b>≈ ${fmt(predTotal)}<span class="unit">ha</span></b>
        <div class="range">range ${lo}–${hi} ha · ±${Math.round(BAND * 100)}%</div></div>
      <div class="caveat">Full-year nowcast for ${currentMeta.year} — no official census yet (none until
        ~2028). Two separate layers: <strong>location &amp; ranking come from the model</strong> (map,
        trustworthy); <strong>magnitude &amp; trajectory come from the official census trend</strong>
        (chart above, a scenario). This detected total is a ±${Math.round(BAND * 100)}% estimate — read the
        range, not a point.</div>`;
  } else {
    const ratioStr = offTotal ? (predTotal / offTotal).toFixed(2) + "×" : "—";
    document.getElementById("totals").innerHTML = `
      <div class="stat"><span>Predicted</span><b>${fmt(predTotal)}<span class="unit">ha</span></b></div>
      <div class="stat"><span>Official</span><b>${offTotal ? fmt(offTotal) : "—"}<span class="unit">ha</span></b></div>
      <div class="stat wide"><span>Predicted / official ratio</span><b>${ratioStr}</b></div>
      <div class="caveat">Out-of-fold predictions: every pixel comes from a model that never
        trained on it. Uncalibrated, so the total runs low — no scalar is applied because
        fitting one to the census would make the total match by construction.
        <strong>Two pre-registered nulls beat this model.</strong> Reusing the previous
        census locates coca better (cell IoU 0.93 vs 0.73, 0/6 folds won) and ranks
        municipalities better (ρ 0.98 vs 0.89, 0/6 years won). So read this map as
        <em>reproducing</em> the official pattern from free imagery — not as improving on it.
        Labels are ~1 km census cells, so nothing here is field-level.</div>`;
  }

  // No year-over-year arrows: model magnitude change is unreliable (LOYO). Trajectory
  // lives in the official-trend chart. Hotspot list = location/ranking only.
  const top = [...feats].sort((a, b) => (b.predicted_ha || 0) - (a.predicted_ha || 0)).slice(0, 8);
  const ul = document.getElementById("hotspots");
  ul.innerHTML = top.map((p, i) => {
    const val = currentMeta.nowcast ? `≈${fmt(p.predicted_ha)} ha` : `${fmt(p.predicted_ha)} ha`;
    return `<li tabindex="0" data-name="${p.name}">
      <span class="rank">${i + 1}</span>
      <span class="swatch" style="background:${colorFor(p.predicted_ha)}"></span>
      <span class="nm">${p.name}</span>
      <span class="ha">${val}</span>
    </li>`;
  }).join("");
  const select = (name) => { const l = layersByName.get(name); if (l) { flyToFeature(l.getBounds()); l.openPopup(); } };
  ul.querySelectorAll("li").forEach((li) => {
    li.onclick = () => select(li.dataset.name);
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(li.dataset.name); } };
  });
}

// ---- official-census trajectory chart: REMOVED 2026-08-10 -------------------
// Plotted official hectares plus a scenario projection to 2026 from trend.json. Both the
// chart and its data file are gone: hectare magnitude is out of scope, and this chart
// directly contradicted the map it sat above (its 2025 upper bound was 47,802 ha while
// the 2025 nowcast layer totalled 55,888 ha, 17% higher, on the same screen).

// ---- legend ------------------------------------------------------------------
function buildLegend() {
  const rows = BREAKS.map((lo, i) => {
    const hi = BREAKS[i + 1];
    const label = hi ? `${fmt(lo)}–${fmt(hi)}` : `${fmt(lo)}+`;
    return `<div class="row"><i style="background:${RAMP[i]}"></i><span>${label}</span></div>`;
  }).join("");
  document.getElementById("legend").innerHTML =
    `<div class="legend-title">Predicted coca (ha)</div>${rows}`;
}

// ---- year selector + nowcast badge -------------------------------------------
async function loadYear(year) {
  currentMeta = YEARS.find((y) => String(y.year) === String(year)) || {};
  document.getElementById("nowcast-badge").hidden = !currentMeta.nowcast;
  await loadChoropleth(year);
  await loadOverlay(year);
  makeDensityLayer();
  // setupDraw() call removed 2026-08-10 with the Measure panel. It would now throw on a
  // null #draw-btn, and calling it from here was itself the bug: it registered a fresh
  // pm:create handler on every year change, so the accumulated handlers deleted each
  // other's rectangles after the first year switch.
}

const yearSel = document.getElementById("year");
yearSel.innerHTML = YEARS.map((y) => `<option value="${y.year}">${y.year}${y.nowcast ? " (nowcast)" : " (validated)"}</option>`).join("");
yearSel.onchange = (e) => loadYear(e.target.value);

// ---- init --------------------------------------------------------------------
const paramYear = new URLSearchParams(location.search).get("year");
const initialYear = YEARS.some((y) => String(y.year) === paramYear) ? paramYear : String(YEARS[0].year);
yearSel.value = initialYear;
buildLegend();
// loadTrend() removed 2026-08-10: trend.json was a HECTARE-MAGNITUDE extrapolation
// ("scenario to 2026"), which is out of scope now that the deliverable is location only.
// It also contradicted the map — the 2025 nowcast total was 55,888 ha while this chart's
// own upper bound for 2025 was 47,802 ha, 17% lower, rendered 200px apart on one screen.
loadYear(initialYear);
window.addEventListener("load", () => setTimeout(() => map.invalidateSize(), 120));
window.addEventListener("resize", () => map.invalidateSize());
