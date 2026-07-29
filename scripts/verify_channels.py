import csv, numpy as np
from pathlib import Path

TILES = Path("data/tiles")
BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12",
         "NDVI","EVI","SAVI","NDWI","NDRE","NBR","VV","VH"]  # band_order = s2+indices+s1

rows = list(csv.DictReader(open(TILES / "multiyear_index.csv")))
# pick a few tiles that actually contain coca (pos_frac>0) so NDVI is meaningful
pos = [r for r in rows if float(r["pos_frac"]) > 0.05]
sample = pos[:3] if pos else rows[:3]

for r in sample:
    img = np.load(TILES / r["npz"])["image"]
    print(f"\n=== {r['tile_id']}  shape={img.shape}  pos_frac={r['pos_frac']} ===")
    print(f"{'idx':>3} {'band':>5} {'min':>10} {'median':>10} {'max':>10}")
    for i in range(img.shape[0]):
        ch = img[i]
        fin = ch[np.isfinite(ch)]
        print(f"{i:>3} {BANDS[i]:>5} {fin.min():>10.3f} {np.median(fin):>10.3f} {fin.max():>10.3f}")

# Explicit checks on the assumed mapping
print("\n--- ASSERTIONS (using assumed indices) ---")
img = np.load(TILES / sample[0]["npz"])["image"]
def med(i):
    ch = img[i][np.isfinite(img[i])]; return float(np.median(ch)), float(ch.min()), float(ch.max())
ndvi = med(10); vv = med(16); vh = med(17)
print(f"idx10 NDVI  median={ndvi[0]:.3f} range=[{ndvi[1]:.3f},{ndvi[2]:.3f}]  -> in[-1,1]? {-1.01<=ndvi[1] and ndvi[2]<=1.01}")
print(f"idx16 VV    median={vv[0]:.3f} range=[{vv[1]:.3f},{vv[2]:.3f}]  -> dB (negative-ish)? {vv[0]<0}")
print(f"idx17 VH    median={vh[0]:.3f} range=[{vh[1]:.3f},{vh[2]:.3f}]  -> dB (negative-ish)? {vh[0]<0}")
# counter-check: what if s1 were at 10,11 instead?
alt = med(10)
print(f"\nCounter-check: if idx10 were VV it'd be ~dB(negative); observed median={alt[0]:.3f}")
