"""Series-level geometry for the Aug 23 run — the Aug 27 architecture.

Per-frame block detection (colour or corner marks) has an unstable tail: on a
handful of frames with poorly-formed marks, re-encoding the same photo as JPEG
moves the detected corners by tens of pixels. Aug 27 does not suffer from this,
because it never trusts a single frame's detection on its own:

  1. detect the block on every frame (coarse rectification),
  2. repair frames whose quad disagrees with the SERIES median in size or
     shape — the block does not change size between shots, so a quad that does
     is a detection failure, not a real change,
  3. (Aug 27 only) register every warped frame onto the series' t=0 warp by ECC
     translation on Sobel gradient magnitude.

MEASURED RESULT: step 2 transfers, step 3 does NOT. Re-running the navigation
metrics on JPEG re-encoded copies of every photo (nav_metrics3.py) gives a
median centroid reproducibility of

    colour 0.0003 mm | corners 0.0026 | repair 0.0015 | repair+ECC 0.0040 mm

so the outlier repair halves the corner-mark noise, while adding ECC nearly
triples it and trebles the worst case (0.23 -> 0.61 mm). ECC works on Aug 27
because warp_all has already rectified those frames against a rigid white rim
with strong, stable gradients; here it re-solves an iterative optimisation on
gradients that the perturbation itself moves, so it becomes a noise source
rather than a noise sink. ECC is therefore OFF by default and kept only so the
comparison can be reproduced.

The t=0 reference is the MAGNET-ATTACHED frame (each Aug 23 series has two t=0
photos, one before the magnet went on), matching the atlas convention.

Writes geom23_series.csv: final quad + ECC shift per photo.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from block23_corners import block_quad as block_quad_dots
from make_block_atlas23 import S, WARP, order_corners, warped_by_date

SIZE_TOL = 0.07     # >7% off the series-median side length = detection failure
SHAPE_TOL = 0.06    # corner residual vs the median shape, in units of side length
ECC_CLAMP = 30.0    # px; a larger "shift" means ECC diverged, not a real offset
USE_ECC = False     # see the module docstring: measured to hurt on this run


def _desc(q):
    """(width, height, centroid) of an ordered quad."""
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return w, h, q.mean(axis=0)


def repair_series(quads):
    """Replace quads that disagree with the series median shape.

    Returns (repaired_quads, flags). A repaired quad keeps its own frame's
    centroid — the holder does shift slightly between shots — but takes the
    median shape, which is the part a failed detection gets wrong.
    """
    have = [(i, q) for i, q in enumerate(quads) if q is not None]
    if len(have) < 3:
        return list(quads), ["kept"] * len(quads)
    W = np.array([_desc(q)[0] for _, q in have])
    H = np.array([_desc(q)[1] for _, q in have])
    mw, mh = np.median(W), np.median(H)
    # median shape: each quad re-centred, then element-wise median of corners
    centred = np.array([q - q.mean(axis=0) for _, q in have])
    shape = np.median(centred, axis=0)
    cents = np.array([_desc(q)[2] for _, q in have])
    med_cent = np.median(cents, axis=0)

    out, flags = list(quads), ["kept"] * len(quads)
    for i, q in have:
        w, h, c = _desc(q)
        bad = abs(w - mw) / mw > SIZE_TOL or abs(h - mh) / mh > SIZE_TOL
        if not bad:
            resid = np.abs(order_corners(q) - q.mean(axis=0) - shape).max()
            bad = resid / mw > SHAPE_TOL
        if bad:
            out[i], flags[i] = (shape + c).astype(np.float32), "repaired"
    for i, q in enumerate(quads):
        if q is None:
            out[i], flags[i] = (shape + med_cent).astype(np.float32), "filled"
    return out, flags


def gradmag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (5, 5), 0)


def ecc_shift(ref_warp, warp):
    """Translation that best maps `warp` onto `ref_warp` (gradient magnitude).

    Translation only: an affine model invents shear on these frames (learned on
    the Aug 27 run), and the block genuinely is rigid between shots.
    """
    try:
        W = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(
            gradmag(ref_warp), gradmag(warp), W, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        dx, dy = float(W[0, 2]), float(W[1, 2])
        if np.hypot(dx, dy) > ECC_CLAMP:
            return 0.0, 0.0, "clamped"
        return dx, dy, "ecc"
    except cv2.error:
        return 0.0, 0.0, "failed"


def aligned_warp(img, quad, date, dx, dy):
    w = warped_by_date(img, quad, date)
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        w = cv2.warpAffine(w, M, (WARP, WARP), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return w


def build(csv_out="geom23_series.csv"):
    m = pd.read_csv(Path(__file__).parent / "photos23_final.csv", parse_dates=["capture_time"])
    m["date"] = m.capture_time.dt.date
    load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))

    rows = []
    for ser, g in m.groupby("series"):
        g = g.sort_values(["timepoint_hr", "capture_time"]).reset_index(drop=True)
        quads, srcs, imgs = [], [], []
        for r in g.itertuples():
            img = load(r.idx)
            imgs.append(img)
            q, src = block_quad_dots(img)
            quads.append(None if q is None else order_corners(np.asarray(q, np.float32)))
            srcs.append(src)
        quads, flags = repair_series(quads)

        # reference = the magnet-attached t=0 frame (each series has two t=0s)
        t0 = g[g.timepoint_hr == g.timepoint_hr.min()]
        cand = t0[t0.v_magnet == "present"]
        if not len(cand):
            cand = t0
        ref_i = int(cand.index[0]) if len(cand) else 0
        ref = warped_by_date(imgs[ref_i], quads[ref_i], g.date.iloc[ref_i])

        for i, r in enumerate(g.itertuples()):
            if i == ref_i or not USE_ECC:
                dx, dy, how = 0.0, 0.0, "ref" if i == ref_i else "off"
            else:
                dx, dy, how = ecc_shift(ref, warped_by_date(imgs[i], quads[i], g.date.iloc[i]))
            rows.append(dict(idx=int(r.idx), series=ser, timepoint_hr=r.timepoint_hr,
                             src=srcs[i], repair=flags[i], dx=dx, dy=dy, ecc=how,
                             **{f"q{j}{a}": float(quads[i][j][k])
                                for j in range(4) for k, a in enumerate("xy")}))
        print(f"  {ser:16s} n={len(g):3d}  repaired={sum(f!='kept' for f in flags)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(Path(__file__).parent / csv_out, index=False)
    sh = np.hypot(out.dx, out.dy)
    print(f"\nframes: {len(out)} | repaired/filled: {(out.repair!='kept').sum()}"
          f" | dots: {(out.src=='dots').sum()} | ECC ok: {(out.ecc=='ecc').sum()}")
    print(f"ECC shift px: median {np.median(sh):.2f}  p90 {np.percentile(sh,90):.2f}  max {sh.max():.2f}")
    return out


if __name__ == "__main__":
    build()
