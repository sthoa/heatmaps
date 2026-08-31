"""Re-run the Aug 23 navigation metrics under two geometries and compare.

Implements the metric definitions from the reviewed navigation_metrics23.py,
with one correction: the per-series gap-edge reference is taken from the
MAGNET-ATTACHED t=0 frame, matching make_block_atlas23.pick(). The original
took t0.iloc[0], the pre-magnet frame, which shifted the reference by 0.30 mm
on average and 1.56 mm in the worst series.

Geometries compared:
  colour  -- make_block_atlas23.block_quad  (gel colour segmentation)
  corners -- block23_corners.block_quad     (holder corner marks)
"""
import os, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import make_block_atlas23 as M
from block23_corners import block_quad as bq_corners

PROJ = Path(__file__).resolve().parent.parent
DAY = PROJ / "23rd August (6 Hours, Center 0.4, Non-BSA, PEG vs COOH, Large vs Small, N=2 Controls)"
MM_PER_PX = (M.EXTENT_MM[1] - M.EXTENT_MM[0]) / 404.0
GAP_FRAC, D_FRAC, FLOOR_PCT = 0.35, 0.90, 15

def resolve(rel):
    p = DAY / rel.replace("/", os.sep)
    if p.exists(): return p
    parts = rel.split("/"); st=[re.sub(r"\s*\(.*?\)\s*$","",q).strip() for q in parts[:-1]]+[parts[-1]]
    p = DAY / os.sep.join(st)
    return p if p.exists() else None

def frame(path):
    img = cv2.imread(str(path))
    if img is None: return None
    h,w = img.shape[:2]; s = 1100/max(h,w)
    return cv2.resize(img,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)

def field(img, quad, date):
    w = M.warped_by_date(img, quad, date)
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[...,0].astype(np.float32)
    return (255.0-L)[M.MARGIN:M.WARP-M.MARGIN, M.MARGIN:M.WARP-M.MARGIN]

def profile(D):
    p = np.sort(D, axis=0); n = p.shape[0]
    return p[int(.15*n):int(.85*n)].mean(axis=0)

def excess(p): return np.clip(p - np.percentile(p, FLOOR_PCT), 0, None)

def gap_edges(e):
    pk = int(np.argmax(e)); thr = GAP_FRAC*e[pk]
    r = pk
    while r+1 < len(e) and e[r+1] > thr: r += 1
    l = pk
    while l-1 >= 0 and e[l-1] > thr: l -= 1
    return l, r

def side(ex, edge, direction):
    seg = ex[edge:] if direction > 0 else ex[:edge+1][::-1]
    if seg.size < 5: return dict(mass=np.nan, centroid=np.nan, d90=np.nan)
    d = np.arange(seg.size)*MM_PER_PX; mass = float(seg.sum())
    if mass <= 0: return dict(mass=0.0, centroid=np.nan, d90=np.nan)
    c = float((d*seg).sum()/mass)
    return dict(mass=mass, centroid=c, d90=float(np.interp(D_FRAC, np.cumsum(seg)/mass, d)))

def run(name, quadfn):
    m = pd.read_csv(Path(__file__).parent/"photos23_final.csv", parse_dates=["capture_time"])
    m["date"] = m.capture_time.dt.date
    cache = {}
    for r in m.itertuples():
        p = resolve(r.path)
        if p is None: continue
        img = frame(p)
        q = quadfn(img)
        if isinstance(q, tuple): q = q[0]
        if q is None: continue
        cache[r.idx] = excess(profile(field(img, q, r.date)))
    # gap edge per series from the MAGNET-ATTACHED t=0 frame
    edges = {}
    for s, g in m.groupby("series"):
        t0 = g[g.timepoint_hr == g.timepoint_hr.min()]
        pick = t0[t0.v_magnet == "present"]
        row = (pick if len(pick) else t0).iloc[0]
        if row.idx in cache: edges[s] = gap_edges(cache[row.idx])
    rows = []
    for r in m.itertuples():
        if r.series not in edges or r.idx not in cache: continue
        l, rr = edges[r.series]
        mag, anti = side(cache[r.idx], rr, +1), side(cache[r.idx], l, -1)
        tot = mag["mass"] + anti["mass"]
        rows.append({"geometry":name,"series":r.series,"coating":r.coating,"arm":r.arm,
                     "t":r.timepoint_hr,"centroid_mag":mag["centroid"],"d90_mag":mag["d90"],
                     "nav_index":(mag["mass"]-anti["mass"])/tot if tot>0 else np.nan})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for name, fn in [("colour", M.block_quad), ("corners", bq_corners)]:
        print(f"running {name} ...", flush=True)
        out.append(run(name, fn))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(Path(__file__).parent/"aug23_nav_compare.csv", index=False)
    print("\nwrote aug23_nav_compare.csv:", len(df), "rows")
