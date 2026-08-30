"""Detect the agarose well in every downscaled frame -> rects.json.

Strategy: beige/brown HSV mask of the gel (two morphology variants: plain
open, and close+hole-fill+open for pale 0.4% gel that reads as white),
minAreaRect candidates gated by absolute side bounds (0.13-0.42 of the long
frame side), aspect <= 1.8, extent >= 0.65. If the strict pass finds nothing,
a parameter sweep (looser saturation/value bounds, gray-world white balance)
retries with the same gates.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def mask_variants(img, s_min=15, v_max=235):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    beige = (h >= 5) & (h <= 35) & (s >= s_min) & (s <= 170) & (v >= 60) & (v <= v_max)
    brown = (h <= 30) & (s >= 60) & (v >= 25) & (v < 130)
    m = (beige | brown).astype(np.uint8)
    m1 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ff = m2.copy()
    Hh, Ww = m.shape
    mk = np.zeros((Hh + 2, Ww + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 1)
    m2 = m2 | ((ff == 0).astype(np.uint8))
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    return [m1, m2]


def best_quad(m, H, W):
    L = max(H, W)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best, best_score = None, 0
    for i in range(1, n):
        area = stats[i, 4]
        if area < 0.012 * H * W:
            continue
        cx, cy = cents[i]
        if not (0.03 * W < cx < 0.97 * W and 0.03 * H < cy < 0.97 * H):
            continue
        comp = (labels == i).astype(np.uint8)
        rect = cv2.minAreaRect(cv2.findNonZero(comp))
        (rcx, rcy), (rw, rh), ang = rect
        if min(rw, rh) < 0.13 * L or max(rw, rh) > 0.42 * L:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > 1.8:
            continue
        extent = area / (rw * rh)
        if extent < 0.65:
            continue
        score = extent * (1.8 - aspect) * area
        if score > best_score:
            best_score, best = score, rect
    return best, best_score


def detect(img):
    H, W = img.shape[:2]
    best, bs = None, 0
    for m in mask_variants(img):
        r_, s_ = best_quad(m, H, W)
        if s_ > bs:
            bs, best = s_, r_
    if best is not None:
        return best, True
    b, g, r = cv2.split(img.astype(np.float32))
    mg = (b.mean() + g.mean() + r.mean()) / 3
    imgWB = cv2.merge(
        [np.clip(c * mg / max(c.mean(), 1), 0, 255) for c in (b, g, r)]
    ).astype(np.uint8)
    for src in (img, imgWB):
        for s_min in (8, 15):
            for v_max in (235, 245):
                for m in mask_variants(src, s_min, v_max):
                    r_, s_ = best_quad(m, H, W)
                    if s_ > bs:
                        bs, best = s_, r_
    return best, False


if __name__ == "__main__":
    S = Path(sys.argv[1])
    photos = pd.read_csv(Path(__file__).parent / "photos.csv")
    fails, rects = [], {}
    for i in range(len(photos)):
        img = cv2.imread(str(S / "prep" / "frames" / f"{i:04d}.jpg"))
        r, strict = detect(img)
        if r is None:
            fails.append(i)
        else:
            rects[i] = {"rect": (r[0][0], r[0][1], r[1][0], r[1][1], r[2]), "strict": strict}
    print("hard failures:", len(fails), fails)
    print("fallback-recovered:", sum(1 for v in rects.values() if not v["strict"]))
    json.dump(rects, open(S / "rects.json", "w"))
