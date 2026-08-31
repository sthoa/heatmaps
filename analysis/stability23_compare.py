"""Re-encode stability of the Aug 23 field under all three geometries.

Perturbation = re-saving the photo as JPEG (mean pixel difference ~1.2/255).
A correct pipeline should be blind to it. The metric is the shift of the
extracted field's column profile in mm, because that is the quantity the
navigation metrics actually consume.
"""
from pathlib import Path
import cv2, numpy as np, pandas as pd
from make_block_atlas23 import S, WARP, MARGIN, block_quad as bq_colour, order_corners, warped_by_date
from block23_corners import block_quad as bq_dots
from block23_series import _desc, SIZE_TOL, SHAPE_TOL, ecc_shift, aligned_warp

def field_of(w):
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    D = (255.0 - L)[MARGIN:WARP - MARGIN, MARGIN:WARP - MARGIN]
    h, wd = D.shape; hb, wb = h // 11, wd // 11
    return np.median(D[:hb*11, :wb*11].reshape(hb, 11, wb, 11), axis=(1, 3))

def prof_shift_mm(f1, f2):
    """sub-pixel lag between two column profiles, converted to mm."""
    p1, p2 = f1.mean(0), f2.mean(0)
    p1, p2 = p1 - p1.mean(), p2 - p2.mean()
    c = np.correlate(p1, p2, "full"); k = int(c.argmax())
    if 0 < k < len(c) - 1:                      # parabolic refinement
        a, b, cc = c[k-1], c[k], c[k+1]
        den = a - 2*b + cc          # negative at a true peak
        if abs(den) > 1e-9:
            k = k + float(np.clip(0.5 * (a - cc) / den, -1, 1))
    lag = k - (len(p1) - 1)
    mm_per_bin = (10.0 - 2*MARGIN/WARP*10.0) / len(p1)
    return abs(lag) * mm_per_bin

m = pd.read_csv("photos23_final.csv", parse_dates=["capture_time"]); m["date"] = m.capture_time.dt.date
G = pd.read_csv("geom23_series.csv")
qcols = [f"q{j}{a}" for j in range(4) for a in "xy"]
load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))
def reenc(img):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1], 1)

# per-series median shape + reference warp, from the built geometry
shapes, refs = {}, {}
for ser, g in G.groupby("series"):
    Q = g[qcols].to_numpy(np.float32).reshape(-1, 4, 2)
    shapes[ser] = (np.median([q - q.mean(0) for q in Q], axis=0),
                   np.median([_desc(q)[0] for q in Q]), np.median([_desc(q)[1] for q in Q]))
    r = g[g.ecc == "ref"].iloc[0]
    ri = m[m.idx == r.idx].iloc[0]
    refs[ser] = warped_by_date(load(r.idx), r[qcols].to_numpy(np.float32).reshape(4, 2), ri.date)

def series_quad(img, ser):
    shape, mw, mh = shapes[ser]
    q, _ = bq_dots(img)
    if q is None:
        return (shape + np.zeros(2)).astype(np.float32)
    q = order_corners(np.asarray(q, np.float32)); w, h, c = _desc(q)
    bad = abs(w-mw)/mw > SIZE_TOL or abs(h-mh)/mh > SIZE_TOL
    if not bad:
        bad = np.abs(q - c - shape).max() / mw > SHAPE_TOL
    return (shape + c).astype(np.float32) if bad else q

res = {"colour": [], "corners": [], "series": []}
samp = G[G.ecc != "ref"].sample(40, random_state=3)
for r in samp.itertuples():
    row = m[m.idx == r.idx].iloc[0]; img = load(r.idx)
    if img is None: continue
    img2 = reenc(img); d = row.date
    for name in res:
        try:
            if name == "colour":
                qa, qb = bq_colour(img), bq_colour(img2)
                if qa is None or qb is None: continue
                fa, fb = field_of(warped_by_date(img, qa, d)), field_of(warped_by_date(img2, qb, d))
            elif name == "corners":
                qa, qb = bq_dots(img)[0], bq_dots(img2)[0]
                if qa is None or qb is None: continue
                fa, fb = field_of(warped_by_date(img, qa, d)), field_of(warped_by_date(img2, qb, d))
            else:
                ref = refs[r.series]
                qa, qb = series_quad(img, r.series), series_quad(img2, r.series)
                da = ecc_shift(ref, warped_by_date(img, qa, d))
                db = ecc_shift(ref, warped_by_date(img2, qb, d))
                fa = field_of(aligned_warp(img, qa, d, da[0], da[1]))
                fb = field_of(aligned_warp(img2, qb, d, db[0], db[1]))
            res[name].append(prof_shift_mm(fa, fb))
        except Exception:
            continue

print(f"AUG 23 field stability under JPEG re-encode (n={len(res['series'])} frames)\n")
print(f"{'method':>10} | {'median':>8} {'p90':>8} {'worst':>8} | {'>0.25mm':>8}")
for k, v in res.items():
    v = np.array(v)
    print(f"{k:>10} | {np.median(v):7.3f}m {np.percentile(v,90):7.3f}m {v.max():7.3f}m | {(v>0.25).sum():4d}/{len(v)}")
