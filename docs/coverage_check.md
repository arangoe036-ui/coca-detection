# Coverage check — clear Sentinel-2 observations per pixel (Catatumbo)

Confirms the 2026 partial-year 'decline' was an observation-completeness artifact,
not real coca loss.

| Window | Scenes | Mean clear obs/pixel | Median |
|---|--:|--:|--:|
| 2024 (full year) | 438 | 27.7 | 25 |
| 2025 (full year) | 618 | 31.5 | 30 |
| 2026 (partial, Jan–Jul) | 318 | 13.6 | 13 |

- **2025 / 2024 clear-look ratio ≈ 1.14** → comparable full-year coverage; 2025 is a sound nowcast year.
- **2026-partial / 2024 clear-look ratio ≈ 0.49** → ~half the clear looks, so the model saw ~half the coca.
  The 2026 'decline' tracked coverage, not cultivation. Coca is a standing perennial; it does not halve mid-year.
- Fix (v2.3): the nowcast uses **full-year 2025**; the magnitude/trajectory comes from the official census trend.
