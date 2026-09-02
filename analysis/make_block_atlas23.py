"""Block-heatmap atlas for the Aug 23 run (centre-gap injection).

Differences from Aug 27 (back-gap injection):
  * The gap is in the MIDDLE of the block, so it cannot be used as the x=0
    anchor. Per Steven: the heatmap spans the whole block, x=0 at the block's
    LEFT edge (where a back gap would be), x=10 mm at the magnet-side edge;
    the centre gap therefore lands near 5 mm.
  * The block is found directly (gel is beige, sat 30-105; the white holder is
    sat<25 and the orange sticker sat>110). The centre gap can split the gel
    into two halves, so nearby gel components are unioned.
  * Orientation is normalized per frame: the transport axis is perpendicular
    to the gap, and the magnet side is found from the bright metallic cylinder
    beside the block (controls inherit their run's convention).

Panels: one per coating (PEG, COOH); rows = large-magnet mean (L1-L3),
small-magnet mean (S1-S3), control; columns = time stages.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad/aug23")
WARP = 480
STAGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
VMAX = 55.0


def gel_mask(img):
    """Gel, including NP-darkened gel. The orange sticker sits at s>170 and the
    white holder at s<25, so 25<=s<=140 isolates gel across the whole run."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gel = ((h >= 5) & (h <= 40) & (s >= 25) & (s <= 140) & (v >= 70) & (v <= 250)).astype(np.uint8)
    gel = cv2.morphologyEx(gel, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return cv2.morphologyEx(gel, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))


def _quad(mask):
    """4-corner polygon of a mask (true perspective corners, not a bounding
    rotated rect - the block is a trapezoid when the camera is tilted)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.015, 0.09, 12):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _score(mask):
    """squareness x fill of a candidate block mask (the block is a 10x10 square)."""
    pts = cv2.findNonZero(mask)
    if pts is None:
        return -1, None
    rect = cv2.minAreaRect(pts)
    (_, _), (rw, rh), _ = rect
    if min(rw, rh) < 20:
        return -1, None
    aspect = max(rw, rh) / min(rw, rh)
    extent = float(mask.sum()) / (rw * rh)
    if aspect > 1.9 or extent < 0.45:
        return -1, None
    return (1.0 / aspect) * extent * (rw * rh), rect


def block_quad(img):
    """Corners of the agarose block. The centre gap splits the gel into two
    halves and NP accumulation darkens one of them, so candidate halves are
    combined and the most square-and-full combination wins."""
    gel = gel_mask(img)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(gel, 8)
    H, W = gel.shape
    cands = [(stats[i, 4], i) for i in range(1, n) if stats[i, 4] > 0.004 * H * W]
    if not cands:
        return None
    cands.sort(reverse=True)
    main_a, main_i = cands[0]
    mc = cents[main_i]
    scale = max(stats[main_i, 2], stats[main_i, 3])
    near = [i for a, i in cands[1:]
            if np.hypot(cents[i][0] - mc[0], cents[i][1] - mc[1]) < 1.3 * scale and a > 0.06 * main_a]
    best_s, best_mask = -1, None
    combos = [[main_i]] + [[main_i, j] for j in near] + ([[main_i] + near] if len(near) > 1 else [])
    for combo in combos:
        mask = np.isin(labels, combo).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        sc, _ = _score(mask)
        if sc > best_s:
            best_s, best_mask = sc, mask
    if best_mask is None:
        return None
    return _quad(best_mask)


def magnet_side(img, rect):
    """Which side of the block the magnet cylinder sits on: metallic = bright,
    very low saturation, outside the block. Returns a unit vector or None."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    metal = ((s < 45) & (v > 150)).astype(np.uint8)
    # the holder is also white; the magnet is distinguished by strong specular
    # highlights -> high local contrast. Use the brightest strip beyond the block.
    (cx, cy), (rw, rh), ang = rect
    box = cv2.boxPoints(rect)
    half = max(rw, rh) / 2
    best, bs = None, 0
    for vec in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        px = int(cx + vec[0] * half * 1.55)
        py = int(cy + vec[1] * half * 1.55)
        x0, x1 = max(0, px - 45), min(img.shape[1], px + 45)
        y0, y1 = max(0, py - 45), min(img.shape[0], py + 45)
        if x1 <= x0 or y1 <= y0:
            continue
        patch_v = v[y0:y1, x0:x1]
        patch_m = metal[y0:y1, x0:x1]
        if patch_m.mean() < 0.35:
            continue
        # specular highlights: fraction of near-blown pixels
        score = float((patch_v > 205).mean()) * float(patch_m.mean())
        if score > bs:
            bs, best = score, vec
    return best if bs > 0.05 else None


def order_corners(pts):
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def warped_block(img, rect, side):
    """Warp the block to a square, rotated so the magnet side faces RIGHT."""
    box = cv2.boxPoints(rect).astype(np.float32)
    quad = order_corners(box)
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    warp = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (WARP, WARP))
    if side is None:
        return warp, 0
    # rot90 count so that `side` ends up pointing right (+x)
    k = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[side]
    if k:
        warp = np.ascontiguousarray(np.rot90(warp, k=k))
    return warp, k


def gap_axis_ok(warp):
    """True if the dark centre gap runs vertically (i.e. transport is left-right)."""
    L = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    core = L[60:WARP - 60, 60:WARP - 60]
    col = gaussian_filter1d(core.mean(axis=0), 4)
    row = gaussian_filter1d(core.mean(axis=1), 4)
    return (col.max() - col.min()) > (row.max() - row.min())


def field(img, rect, side):
    warp, _ = warped_block(img, rect, side)
    L = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    D = (255.0 - L)[26:WARP - 26, 20:WARP - 20]
    h, w = D.shape
    hb, wb = h // 11, w // 11
    Db = np.median(D[: hb * 11, : wb * 11].reshape(hb, 11, wb, 11), axis=(1, 3))
    return Db


def process_abs(Db):
    h, w = Db.shape
    floor = np.percentile(Db[3:h - 3, 3:w - 3], 15)
    return gaussian_filter(np.clip(Db - floor, 0, None), 1.0)


# rot90 count that brings the magnet side to the RIGHT, per photo session
# (camera framing was consistent within each session; verified visually and by
# the automatic gap-axis check in orient23_check.csv)
import datetime

ROT_BY_DATE = {
    datetime.date(2026, 8, 23): 1,  # magnet at the bottom of frame
    datetime.date(2026, 8, 25): 0,  # magnet at the right
    datetime.date(2026, 8, 26): 0,
}


def warped_by_date(img, quad, date):
    """Perspective-correct the block quad to a square, then rotate so the
    magnet side faces RIGHT. Using the true 4 corners (not a bounding rotated
    rect) keeps the centre gap vertical when the camera was tilted."""
    q = order_corners(quad)
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    w = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (WARP, WARP))
    k = ROT_BY_DATE.get(date, 0)
    if k:
        w = np.ascontiguousarray(np.rot90(w, k=k))
    return w


MARGIN = 38  # px of the 480 warp trimmed per side (~0.8 mm): the gel pulls away
             # from the holder late in the run and the seam shadow would read as NPs
EXTENT_MM = (MARGIN / WARP * 10.0, 10.0 - MARGIN / WARP * 10.0)


def field_by_date(img, quad, date):
    w = warped_by_date(img, quad, date)
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    D = (255.0 - L)[MARGIN:WARP - MARGIN, MARGIN:WARP - MARGIN]
    h, wd = D.shape
    hb, wb = h // 11, wd // 11
    return np.median(D[: hb * 11, : wb * 11].reshape(hb, 11, wb, 11), axis=(1, 3))


def repaired_quads():
    """Per-photo block quads with series-median outlier repair (block23_series).

    The block does not change size between shots, so a quad that disagrees with
    its series' median size or shape is a detection failure; replacing those
    with the median shape at the frame's own centroid halves the geometry noise
    floor (see block23_series and nav_metrics3).
    """
    f = Path(__file__).parent / "geom23_series.csv"
    if not f.exists():
        return {}
    g = pd.read_csv(f)
    cols = [f"q{j}{a}" for j in range(4) for a in "xy"]
    return {int(r.idx): np.asarray([r[c] for c in cols], np.float32).reshape(4, 2)
            for _, r in g.iterrows()}


def main(title=True, outdir=None, dpi=118, scale=1.0):
    m = pd.read_csv(Path(__file__).parent / "photos23_final.csv", parse_dates=["capture_time"])
    QUADS = repaired_quads()
    m["date"] = m.capture_time.dt.date
    outdir = Path(outdir) if outdir else S / "panels"
    outdir.mkdir(exist_ok=True)
    load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))

    def pick(g, t):
        sub = g[g.timepoint_hr == t]
        if t == 0 and (sub.v_magnet == "present").any():
            sub = sub[sub.v_magnet == "present"]
        return int(sub.iloc[0].idx) if len(sub) else None

    ARMS = [("large", "large magnet\n(mean L1-L3)"), ("small", "small magnet\n(mean S1-S3)"),
            ("control", "control\n(no magnet)")]
    for coating, cg in m.groupby("coating"):
        fig, axes = plt.subplots(3, len(STAGES), figsize=tuple(scale * v for v in (len(STAGES) * 1.75, 3 * 1.85)))
        for ri, (arm, label) in enumerate(ARMS):
            ag = cg[cg.arm == arm]
            for ci, t in enumerate(STAGES):
                ax = axes[ri, ci]
                maps = []
                for sname, sg in ag.groupby("series"):
                    i = pick(sg, t)
                    if i is None:
                        continue
                    img = load(i)
                    quad = QUADS.get(int(i))
                    if quad is None:
                        quad = block_quad(img)
                    if quad is None:
                        continue
                    maps.append(process_abs(field_by_date(img, quad, sg.iloc[0].date)))
                if not maps:
                    ax.text(.5, .5, "no frame", ha="center", va="center", fontsize=7,
                            color="gray", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([]); continue
                shp = maps[0].shape
                maps = [mm for mm in maps if mm.shape == shp]
                ax.imshow(np.mean(maps, axis=0), cmap="inferno", vmin=0, vmax=VMAX,
                          extent=[EXTENT_MM[0], EXTENT_MM[1], EXTENT_MM[1], EXTENT_MM[0]],
                          aspect="equal", interpolation="bilinear")
                ax.set_yticks([])
                if len(maps) < len(ag.series.unique()):
                    ax.text(0.03, 0.06, f"n={len(maps)}", transform=ax.transAxes, fontsize=6, color="white")
                if ri == 2:
                    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(["1", "5", "9"], fontsize=7)
                    if ci == len(STAGES) // 2:
                        ax.set_xlabel("mm across block  (injection gap ~centre, magnet side →)",
                                      fontsize=8.5, labelpad=4)
                else:
                    ax.set_xticks([])
                for sp in ax.spines.values():
                    sp.set_color("#999"); sp.set_linewidth(.6)
                if ri == 0:
                    ax.set_title(f"{t:g} h", fontsize=10)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8)
        if title: fig.suptitle(f"23 Aug · 0.4% agarose · non-BSA · {coating} · centre injection", fontsize=11.5, y=0.99)
        plt.tight_layout(rect=[0.02, 0.10, 1, 0.94 if title else 0.99])
        cax = fig.add_axes([0.34, 0.055, 0.32, 0.016])
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, VMAX))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 25, 55]); cb.ax.tick_params(labelsize=6, pad=1)
        cb.set_label("NP darkness above gel floor (L*)", fontsize=7, labelpad=2)
        plt.savefig(outdir / f"{coating}.png", dpi=dpi)
        plt.close(fig)
        print("done", coating, flush=True)


if __name__ == "__main__":
    main()
