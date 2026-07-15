/* Coca hotspot monitor v2 (P5 + v2 Part B).
   Municipal choropleth + pixel-level density overlay (downsampled COG) over a
   light basemap, with Esri imagery crossfade for zoom-in, opacity/threshold
   controls, draw-a-box hectare sums, and a data-driven year selector with a
   nowcast badge for years lacking an official census. */

// Years available as data. `nowcast:true` => no official census; label it.
const YEARS = [{ year: 2023, nowcast: false }];
const INSPECT_ZOOM = 13;
const ACCENT = "#e6a01f";
const RAMP = ["#ffffe5", "#fee391", "#fe9929", "#cc4c02", "#8c2d04"]; // YlOrBr light->dark
const BREAKS = [0, 500, 2000, 5000, 15000];      // municipal choropleth ha bins
const MAX_FRAC = 0.26;                            // overlay normalization (COG max cover fraction)
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

// ---- draw-a-box -> sum hectares ----------------------------------------------
function setupDraw() {
  const pixelHa = densityRaster
    ? (densityRaster.pixelWidth * densityRaster.pixelHeight) / 1e4 : 0;
  document.getElementById("draw-btn").onclick = () => {
    map.pm.enableDraw("Rectangle", { snappable: false });
  };
  map.on("pm:create", (e) => {
    map.pm.disableDraw();
    map.eachLayer((l) => { if (l._drawnMeasure) map.removeLayer(l); });
    e.layer._drawnMeasure = true;
    const chip = document.getElementById("draw-result");
    try {
      const gj = e.layer.toGeoJSON();
      const sumFrac = geoblaze.sum(densityRaster, gj)[0] || 0;
      const ha = sumFrac * pixelHa;
      chip.hidden = false;
      chip.textContent = `≈ ${fmt(ha)} ha coca in drawn area`;
    } catch (err) {
      chip.hidden = false;
      chip.textContent = "Could not compute for that area.";
      console.warn(err);
    }
  });
}

// ---- choropleth --------------------------------------------------------------
let layer;
const layersByName = new Map();

function flyToFeature(bounds) {
  setImagery(true);
  if (reduceMotion) map.fitBounds(bounds, { maxZoom: INSPECT_ZOOM });
  else map.flyToBounds(bounds, { maxZoom: INSPECT_ZOOM, duration: 1.2 });
}

function popupHtml(p) {
  const ratio = p.official_ha ? (p.predicted_ha / p.official_ha).toFixed(2) + "×" : "n/a";
  return `<div class="popup-title">${p.name}</div>
    <div class="popup-row"><span>Predicted</span><b>${fmt(p.predicted_ha)} ha</b></div>
    <div class="popup-row"><span>Official</span><b>${p.official_ha ? fmt(p.official_ha) + " ha" : "—"}</b></div>
    <div class="popup-row"><span>Ratio</span><b>${ratio}</b></div>`;
}

async function loadChoropleth(year) {
  const gj = await (await fetch(`data/municipal_coca_${year}.geojson`)).json();
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
  map.fitBounds(layer.getBounds(), { padding: [20, 20] });
  buildPanel(gj);
}

function buildPanel(gj) {
  const feats = gj.features.map((f) => f.properties);
  const predTotal = feats.reduce((s, p) => s + (p.predicted_ha || 0), 0);
  const offTotal = feats.reduce((s, p) => s + (p.official_ha || 0), 0);
  const ratioStr = offTotal ? (predTotal / offTotal).toFixed(2) + "×" : "nowcast";
  document.getElementById("totals").innerHTML = `
    <div class="stat"><span>Predicted</span><b>${fmt(predTotal)}<span class="unit">ha</span></b></div>
    <div class="stat"><span>Official</span><b>${offTotal ? fmt(offTotal) : "—"}<span class="unit">ha</span></b></div>
    <div class="stat wide"><span>Predicted / official ratio</span><b>${ratioStr}</b></div>`;

  const top = [...feats].sort((a, b) => (b.predicted_ha || 0) - (a.predicted_ha || 0)).slice(0, 8);
  const ul = document.getElementById("hotspots");
  ul.innerHTML = top.map((p, i) => `
    <li tabindex="0" data-name="${p.name}">
      <span class="rank">${i + 1}</span>
      <span class="swatch" style="background:${colorFor(p.predicted_ha)}"></span>
      <span class="nm">${p.name}</span>
      <span class="ha">${fmt(p.predicted_ha)} ha</span>
    </li>`).join("");
  const select = (name) => { const l = layersByName.get(name); if (l) { flyToFeature(l.getBounds()); l.openPopup(); } };
  ul.querySelectorAll("li").forEach((li) => {
    li.onclick = () => select(li.dataset.name);
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(li.dataset.name); } };
  });
}

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
  const meta = YEARS.find((y) => String(y.year) === String(year)) || {};
  document.getElementById("nowcast-badge").hidden = !meta.nowcast;
  await loadChoropleth(year);
  await loadOverlay(year);
  makeDensityLayer();
  setupDraw();
}

const yearSel = document.getElementById("year");
yearSel.innerHTML = YEARS.map((y) => `<option value="${y.year}">${y.year}</option>`).join("");
yearSel.onchange = (e) => loadYear(e.target.value);

// ---- init --------------------------------------------------------------------
buildLegend();
loadYear(YEARS[0].year);
window.addEventListener("load", () => setTimeout(() => map.invalidateSize(), 120));
window.addEventListener("resize", () => map.invalidateSize());
