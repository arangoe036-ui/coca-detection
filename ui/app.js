/* Coca hotspot monitor (P5). Municipal choropleth of predicted coca hectares over
   a light basemap, with a high-res imagery crossfade for zoom-in inspection.
   Data: ui/data/municipal_coca_<year>.geojson (props: name, predicted_ha,
   official_ha, ratio, municipio, departamento, year). */

const AVAILABLE_YEARS = [2023];          // data-driven; add years by dropping more geojson files
const INSPECT_ZOOM = 13;                 // auto-load imagery at/above this zoom
const ACCENT = "#e6a01f";
// Sequential single-hue ramp (ColorBrewer YlOrBr), light -> dark.
const RAMP = ["#ffffe5", "#fee391", "#fe9929", "#cc4c02", "#8c2d04"];
const BREAKS = [0, 500, 2000, 5000, 15000];   // ha lower bounds for each ramp step
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function colorFor(ha) {
  const v = ha || 0;
  return v >= BREAKS[4] ? RAMP[4] : v >= BREAKS[3] ? RAMP[3]
       : v >= BREAKS[2] ? RAMP[2] : v >= BREAKS[1] ? RAMP[1] : RAMP[0];
}
const fmt = (v) => Math.round(v || 0).toLocaleString("en-US");

// ---- map + base layers -------------------------------------------------------
const map = L.map("map", { zoomControl: true, minZoom: 6 }).setView([8.65, -72.9], 9);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  attribution: "© OpenStreetMap contributors © CARTO", maxZoom: 20,
}).addTo(map);

// Imagery lives in its own pane so we can crossfade it via CSS opacity.
map.createPane("imagery");
const imgPane = map.getPane("imagery");
imgPane.classList.add("leaflet-imagery-pane");
imgPane.style.zIndex = 250;      // above base tiles (200), below vector overlay (400)
imgPane.style.opacity = 0;
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  maxZoom: 19, pane: "imagery",
}).addTo(map);

let imageryOn = false;
function setImagery(on) {
  imageryOn = on;
  imgPane.style.opacity = on ? 1 : 0;
  const btn = document.getElementById("basemap-toggle");
  btn.setAttribute("aria-pressed", String(on));
  btn.textContent = on ? "Show map" : "Show imagery";
}
document.getElementById("basemap-toggle").onclick = () => setImagery(!imageryOn);

// Auto-load imagery once the user descends into inspection zoom.
map.on("zoomend", () => { if (map.getZoom() >= INSPECT_ZOOM && !imageryOn) setImagery(true); });

// ---- choropleth --------------------------------------------------------------
let layer;
const layersByName = new Map();

function flyToFeature(bounds) {
  setImagery(true);                       // crossfade imagery in as we descend
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

async function loadYear(year) {
  const res = await fetch(`data/municipal_coca_${year}.geojson`);
  const gj = await res.json();
  if (layer) map.removeLayer(layer);
  layersByName.clear();

  layer = L.geoJSON(gj, {
    style: (f) => ({
      fillColor: colorFor(f.properties.predicted_ha),
      fillOpacity: 0.72, color: "#1b2127", weight: 1,   // stroke keeps polygons legible on either basemap
    }),
    onEachFeature: (f, lyr) => {
      const p = f.properties;
      layersByName.set(p.name, lyr);
      lyr.bindPopup(popupHtml(p));
      lyr.on("mouseover", () => lyr.setStyle({ weight: 3, color: ACCENT }));
      lyr.on("mouseout", () => layer.resetStyle(lyr));
      lyr.on("click", () => flyToFeature(lyr.getBounds()));
    },
  }).addTo(map);

  // Ensure Leaflet has the final container size before fitting (avoids a
  // half-rendered map when flex/fonts settle after init).
  map.invalidateSize();
  map.fitBounds(layer.getBounds(), { padding: [20, 20] });
  buildPanel(gj);
}

// ---- side panel --------------------------------------------------------------
function buildPanel(gj) {
  const feats = gj.features.map((f) => f.properties);
  const predTotal = feats.reduce((s, p) => s + (p.predicted_ha || 0), 0);
  const offTotal = feats.reduce((s, p) => s + (p.official_ha || 0), 0);

  document.getElementById("totals").innerHTML = `
    <div class="stat"><span>Predicted</span><b>${fmt(predTotal)}<span class="unit">ha</span></b></div>
    <div class="stat"><span>Official</span><b>${fmt(offTotal)}<span class="unit">ha</span></b></div>
    <div class="stat wide"><span>Predicted / official ratio</span><b>${(predTotal / offTotal).toFixed(2)}×</b></div>`;

  const top = [...feats].sort((a, b) => (b.predicted_ha || 0) - (a.predicted_ha || 0)).slice(0, 8);
  const ul = document.getElementById("hotspots");
  ul.innerHTML = top.map((p, i) => `
    <li tabindex="0" data-name="${p.name}">
      <span class="rank">${i + 1}</span>
      <span class="swatch" style="background:${colorFor(p.predicted_ha)}"></span>
      <span class="nm">${p.name}</span>
      <span class="ha">${fmt(p.predicted_ha)} ha</span>
    </li>`).join("");

  const select = (name) => {
    const lyr = layersByName.get(name);
    if (lyr) { flyToFeature(lyr.getBounds()); lyr.openPopup(); }
  };
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

// ---- year selector -----------------------------------------------------------
const yearSel = document.getElementById("year");
yearSel.innerHTML = AVAILABLE_YEARS.map((y) => `<option value="${y}">${y}</option>`).join("");
yearSel.onchange = (e) => loadYear(e.target.value);

// ---- init --------------------------------------------------------------------
buildLegend();
loadYear(AVAILABLE_YEARS[0]);
window.addEventListener("load", () => setTimeout(() => map.invalidateSize(), 120));
window.addEventListener("resize", () => map.invalidateSize());
