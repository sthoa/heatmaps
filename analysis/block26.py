"""Block detection for the Aug 26 rig (centre gap, magnet to the right).

Steven marked the four corners of the agarose well with red dots on the white
holder, so the block's quad is read directly from those physical markers
instead of from gel colour (the magnet's warm reflections and the orange
sticker both overlap the gel's colour range, and NP accumulation darkens the
gel out of it late in the run).

Extra red marks exist in every frame - the repeat tallies on the magnet strap
and dots on the far end of the mount - so the four that form the most
square, most plausible quad are selected.
"""

import cv2
import numpy as np


def red_dots(img, s_min=110):
    """red marks whose surroundings are white plastic. s_min is loosened on a
    second pass: some corner dots have faded to pale pink."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 12) | (h > 168)) & (s > s_min) & (v > 90)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(red, 8)
    out = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 25 or area > 3500:
            continue
        if max(w, hh) / max(min(w, hh), 1) > 3.5:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((19, 19), np.uint8)) - cv2.dilate(comp, np.ones((5, 5), np.uint8))
        rp = hsv[ring.astype(bool)]
        if len(rp) == 0:
            continue
        if np.median(rp[:, 1]) < 75 and np.median(rp[:, 2]) > 145:
            out.append((float(cents[i][0]), float(cents[i][1])))
    return out


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_score(q, H, W):
    """square-ness and plausibility of a candidate 4-dot quad."""
    o = order_corners(q)
    tl, tr, br, bl = o
    top = np.linalg.norm(tr - tl); bot = np.linalg.norm(br - bl)
    lft = np.linalg.norm(bl - tl); rgt = np.linalg.norm(br - tr)
    if min(top, bot, lft, rgt) < 0.10 * max(H, W):
        return -1
    if max(top, bot, lft, rgt) > 0.60 * max(H, W):
        return -1
    # opposite sides must match (a rectangle, not a random quadrilateral)
    if min(top, bot) / max(top, bot) < 0.80 or min(lft, rgt) / max(lft, rgt) < 0.80:
        return -1
    w, h = (top + bot) / 2, (lft + rgt) / 2
    aspect = max(w, h) / min(w, h)
    if aspect > 1.35:
        return -1
    area = cv2.contourArea(o)
    return area / aspect


def _gel_fill(img, quad):
    """fraction of the candidate quad covered by gel-like colour. A quad that
    strays onto the white mount (common on controls, which have no magnet to
    block the view) fills poorly."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gel = ((h >= 5) & (h <= 45) & (s >= 22) & (v >= 45)).astype(np.uint8)
    m = np.zeros(gel.shape, np.uint8)
    cv2.fillPoly(m, [order_corners(quad).astype(np.int32)], 1)
    tot = int(m.sum())
    return float((gel & m).sum()) / tot if tot else 0.0


def _candidates(dots, H, W):
    """all plausible quads: 4 detected corners, plus 3 corners with the fourth
    inferred (a corner dot is sometimes missing, especially on controls)."""
    out = []
    n = len(dots)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    q = [dots[a], dots[b], dots[c], dots[d]]
                    s = _quad_score(q, H, W)
                    if s > 0:
                        out.append((s, q))
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if len({a, b, c}) < 3:
                    continue
                p1, p2, p3 = np.array(dots[a]), np.array(dots[b]), np.array(dots[c])
                v1, v2 = p1 - p2, p3 - p2
                if np.linalg.norm(v1) < 1 or np.linalg.norm(v2) < 1:
                    continue
                cosang = abs(float(v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                if cosang > 0.25:      # p2 must be the right-angle corner
                    continue
                p4 = p1 + p3 - p2      # complete the parallelogram
                q = [tuple(p1), tuple(p2), tuple(p3), tuple(p4)]
                s = _quad_score(q, H, W)
                if s > 0:
                    out.append((s * 0.9, q))   # slight penalty vs a fully-observed quad
    out.sort(key=lambda t: -t[0])
    return out


def _gel_pair_quad(img):
    """fallback when corner dots are unusable: the centre gap splits the gel
    into two halves that are vertically aligned and horizontally adjacent."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gel = ((h >= 5) & (h <= 42) & (s >= 25) & (s <= 150) & (v >= 60) & (v <= 245)).astype(np.uint8)
    gel = cv2.morphologyEx(gel, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    gel = cv2.morphologyEx(gel, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(gel, 8)
    H, W = gel.shape
    cand = [i for i in range(1, n) if stats[i, 4] > 0.008 * H * W]
    best, bs = None, -1
    for a in cand:
        xa, ya, wa, ha = stats[a, 0], stats[a, 1], stats[a, 2], stats[a, 3]
        for b in cand:
            if b <= a:
                continue
            xb, yb, wb, hb = stats[b, 0], stats[b, 1], stats[b, 2], stats[b, 3]
            if min(ha, hb) / max(ha, hb) < 0.7:
                continue
            if abs((ya + ha / 2) - (yb + hb / 2)) > 0.25 * max(ha, hb):
                continue
            x0, y0 = min(xa, xb), min(ya, yb)
            x1, y1 = max(xa + wa, xb + wb), max(ya + ha, yb + hb)
            aw, ah = x1 - x0, y1 - y0
            asp = max(aw, ah) / max(min(aw, ah), 1)
            if asp > 1.4:
                continue
            sc = (stats[a, 4] + stats[b, 4]) / asp
            if sc > bs:
                bs, best = sc, (x0, y0, x1, y1)
    if best is None:
        return None
    x0, y0, x1, y1 = best
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def block_quad(img):
    H, W = img.shape[:2]
    best, best_fill = None, -1.0
    for s_min in (110, 70):
        dots = red_dots(img, s_min)
        if len(dots) < 3:
            continue
        for s, q in _candidates(dots, H, W)[:25]:
            fill = _gel_fill(img, q)
            if fill > best_fill:
                best_fill, best = fill, q
        if best_fill >= 0.60:
            break
    if best_fill < 0.60:
        gq = _gel_pair_quad(img)
        if gq is not None and _gel_fill(img, gq) > best_fill:
            return order_corners(gq)
    return order_corners(best) if best is not None else None
