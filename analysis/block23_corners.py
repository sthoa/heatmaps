"""Corner-mark block detection for the Aug 23 run.

Replaces the colour-segmentation geometry in make_block_atlas23.block_quad,
which is unstable: re-encoding the same photo as JPEG (mean pixel difference
1.2/255) moved the detected corners by up to 24 px (~1 mm), and the two t=0
frames of a series could disagree by 1.6 mm.

Steven marked the four corners of the agarose well on the holder in every
frame -- dark PURPLE in the 23 Aug session, bright RED/PINK in the 25-26 Aug
session. Reading the quad from those physical marks is far more stable, and is
the same approach validated on the Aug 26 run (block26.py), whose selection
logic (square-ness score, gel-fill preference, and a fourth corner inferred
when only three marks are found) is reused here unchanged.

Falls back to the original colour-based detector when marks are unusable, so
no frame is lost.
"""

import cv2
import numpy as np

from block26 import _candidates, _gel_fill, order_corners
from make_block_atlas23 import block_quad as block_quad_colour


def corner_marks(img, s_min=60):
    """Saturated purple / red-pink marks whose surroundings are white plastic.

    The orange label sits at hue 8-40 and is excluded by hue; the white holder
    is excluded by requiring a bright, low-saturation ring around each mark.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    purple = (h > 125) & (h < 168)
    redpink = (h <= 8) | (h >= 168)
    mark = ((purple | redpink) & (s > s_min) & (v > 25) & (v < 205)).astype(np.uint8)
    mark = cv2.morphologyEx(mark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mark = cv2.morphologyEx(mark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mark, 8)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 20 or area > 3000:
            continue
        if max(w, hh) / max(min(w, hh), 1) > 3.5:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((19, 19), np.uint8)) - cv2.dilate(comp, np.ones((5, 5), np.uint8))
        rp = hsv[ring.astype(bool)]
        if len(rp) == 0:
            continue
        if np.median(rp[:, 1]) < 90 and np.median(rp[:, 2]) > 130:
            out.append((float(cents[i][0]), float(cents[i][1])))
    return out


def _near_gel(img, dots, radius=70):
    """Keep only marks with gel beside them.

    The well corners sit against the agarose; the marks on the magnet mount do
    not. Dropping the mount marks shrinks the candidate set enough that the
    chosen quad stops flipping between near-tied alternatives.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gel = ((h >= 5) & (h <= 45) & (s >= 22) & (s <= 150) & (v >= 45)).astype(np.uint8)
    H, W = gel.shape
    out = []
    for x, y in dots:
        x0, x1 = max(0, int(x - radius)), min(W, int(x + radius))
        y0, y1 = max(0, int(y - radius)), min(H, int(y + radius))
        if gel[y0:y1, x0:x1].mean() > 0.18:
            out.append((x, y))
    return out


def block_quad(img):
    """Block corners from the holder marks; colour detector as a fallback.

    Selection is by gel fill, but a candidate must beat the incumbent by a
    margin to replace it, so near-ties resolve the same way under tiny input
    changes (JPEG re-encoding) instead of flipping.
    """
    H, W = img.shape[:2]
    best, best_fill = None, -1.0
    for s_min in (60, 40):
        dots = _near_gel(img, corner_marks(img, s_min))
        if len(dots) < 3:
            continue
        for _, q in _candidates(dots, H, W)[:25]:
            fill = _gel_fill(img, q)
            if fill > best_fill + 0.02:
                best_fill, best = fill, q
        if best_fill >= 0.60:
            break
    if best_fill < 0.55:
        cq = block_quad_colour(img)
        if cq is not None and _gel_fill(img, cq) > best_fill + 0.05:
            return order_corners(cq), "colour"
    if best is None:
        return None, "none"
    return order_corners(best), "dots"
