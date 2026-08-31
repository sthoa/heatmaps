"""Aug 23 navigation metrics under all three geometries, plus a reproducibility test.

Geometries
  colour  — the original per-frame colour segmentation
  corners — per-frame detection from the holder's corner marks
  series  — corner detection + series-median outlier repair + ECC registration
            onto the series' t=0 warp (the Aug 27 architecture)

Reproducibility: every metric is recomputed from JPEG re-encoded copies of the
same photos (mean pixel difference ~1.2/255). A pipeline that is blind to the
perturbation reports the same number twice; the spread is the geometry noise
floor, in the same millimetres the results are quoted in.
"""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
import make_block_atlas23 as M
import nav_metrics_compare as N
from block23_corners import block_quad as bq_corners
from block23_series import _desc, SIZE_TOL, SHAPE_TOL, ecc_shift, aligned_warp

S = M.S
qcols = [f"q{j}{a}" for j in range(4) for a in "xy"]
load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))
reenc = lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1], 1)

m = pd.read_csv("photos23_final.csv", parse_dates=["capture_time"]); m["date"] = m.capture_time.dt.date
G = pd.read_csv("geom23_series.csv")
shapes, refs = {}, {}
for ser, g in G.groupby("series"):
    Q = g[qcols].to_numpy(np.float32).reshape(-1, 4, 2)
    shapes[ser] = (np.median([q - q.mean(0) for q in Q], axis=0),
                   np.median([_desc(q)[0] for q in Q]), np.median([_desc(q)[1] for q in Q]))
    r = g[g.ecc == "ref"].iloc[0]
    d = m[m.idx == r.idx].iloc[0].date
    refs[ser] = M.warped_by_date(load(r.idx), r[qcols].to_numpy(np.float32).reshape(4, 2), d)

def series_warp(img, ser, date, use_ecc=True):
    shape, mw, mh = shapes[ser]
    q, _ = bq_corners(img)
    if q is None:
        q = shape + np.zeros(2)
    else:
        q = M.order_corners(np.asarray(q, np.float32)); w, h, c = _desc(q)
        bad = abs(w-mw)/mw > SIZE_TOL or abs(h-mh)/mh > SIZE_TOL
        if not bad: bad = np.abs(q - c - shape).max() / mw > SHAPE_TOL
        if bad: q = (shape + c).astype(np.float32)
    if not use_ecc:
        return M.warped_by_date(img, q, date)
    dx, dy, _ = ecc_shift(refs[ser], M.warped_by_date(img, q, date))
    return aligned_warp(img, q, date, dx, dy)

def prof_of(warp):
    L = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    D = (255.0 - L)[M.MARGIN:M.WARP-M.MARGIN, M.MARGIN:M.WARP-M.MARGIN]
    return N.excess(N.profile(D))

def build(geom, perturb):
    cache = {}
    for r in m.itertuples():
        img = load(r.idx)
        if img is None: continue
        if perturb: img = reenc(img)
        if geom in ("series", "repair"):
            w = series_warp(img, r.series, r.date, use_ecc=(geom == "series"))
        else:
            q = M.block_quad(img) if geom == "colour" else bq_corners(img)[0]
            if q is None: continue
            w = M.warped_by_date(img, q, r.date)
        cache[r.idx] = prof_of(w)
    edges = {}
    for s, g in m.groupby("series"):
        t0 = g[g.timepoint_hr == g.timepoint_hr.min()]
        pick = t0[t0.v_magnet == "present"]
        row = (pick if len(pick) else t0).iloc[0]
        if row.idx in cache: edges[s] = N.gap_edges(cache[row.idx])
    rows = []
    for r in m.itertuples():
        if r.series not in edges or r.idx not in cache: continue
        l, rr = edges[r.series]
        mag, anti = N.side(cache[r.idx], rr, +1), N.side(cache[r.idx], l, -1)
        tot = mag["mass"] + anti["mass"]
        rows.append(dict(geometry=geom, perturb=perturb, idx=r.idx, series=r.series,
                         coating=r.coating, arm=r.arm, t=r.timepoint_hr,
                         centroid_mag=mag["centroid"], d90_mag=mag["d90"],
                         nav_index=(mag["mass"]-anti["mass"])/tot if tot > 0 else np.nan))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for geom in ["repair"]:
        for perturb in [False, True]:
            print(f"  {geom} perturb={perturb} ...", flush=True)
            out.append(build(geom, perturb))
    df = pd.concat(out, ignore_index=True)
    df.to_csv("aug23_nav3_repair.csv", index=False)
    print("\nwrote aug23_nav3.csv", len(df))
