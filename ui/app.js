/* P5 map UI logic (plan §10). Placeholder — full choropleth + year slider in Phase 5.
 *
 * Plan:
 *  - Base map centered on Colombia.
 *  - High-res satellite basemap (Esri World Imagery by default; Mapbox Satellite
 *    if a token is configured) so zooming into a hotspot reveals sub-meter tiles
 *    for human visual inspection.
 *  - Choropleth layer from ui/data/municipal_coca.geojson, colored by coca hectares.
 *  - Year slider to switch between processed years.
 */

const map = L.map("map").setView([4.6, -74.1], 6); // Colombia

// High-res satellite basemap (Esri World Imagery — usable with attribution).
L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    maxZoom: 19,
    attribution:
      "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
  }
).addTo(map);

// TODO (P5): fetch("data/municipal_coca.geojson") -> L.geoJSON choropleth,
// color scale by hectares, popup with municipality + hectares, wire year slider.
console.info("Coca hotspot UI scaffold loaded. Choropleth wired up in Phase 5 (P4 must run first).");
