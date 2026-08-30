"""Detect the agarose well in a frame and warp it to a canonical square.

Strategy: the white 3D-printed holder is the brightest low-saturation blob.
The well (agarose + injection gap) is the largest hole inside that blob.
Fit a quadrilateral to the hole and perspective-warp it to WARP x WARP px.
Orientation is normalized afterwards using the red tally-tab side (the tab
marks sit on the magnet side): rotate so the magnet side is on the RIGHT.
"""

import cv2
import numpy as np

WARP = 480


def find_holder_and_well(small):
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = ((s < 60) & (v > 150)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    if n < 2:
        return None, None
    # holder = largest white component near image center
    Hh, Ww = small.shape[:2]
    cx, cy = Ww / 2, Hh / 2
    best, best_score = None, -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.01 * Hh * Ww:
            continue
        bx, by = x + w / 2, y + h / 2
        dist = np.hypot(bx - cx, by - cy) / np.hypot(cx, cy)
        score = area * (1 - 0.5 * dist)
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return None, None
    holder = (labels == best).astype(np.uint8)

    # holes inside the holder: fill from outside, subtract
    ff = holder.copy()
    mask = np.zeros((Hh + 2, Ww + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 1)
    holes = ((ff == 0) & (holder == 0)).astype(np.uint8)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(holes, 8)
    if n2 < 2:
        return holder, None
    big = 1 + int(np.argmax(stats2[1:, 4]))
    well = (labels2 == big).astype(np.uint8)
    return holder, well


def quad_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.02, 0.1, 9):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)


def order_corners(pts):
    # order: TL, TR, BR, BL
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def red_centroid(small):
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 10) | (h > 170)) & (s > 70) & (v > 60)) & (s.astype(int) + 0 < 999)
    # exclude red on green sticker: require bright, low-sat neighborhood -> approximated
    # here by excluding pixels whose 31x31 local green-hue fraction is high
    hsv_green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 90) & (s > 40)).astype(np.float32)
    greenness = cv2.boxFilter(hsv_green, -1, (61, 61))
    red = red & (greenness < 0.2)
    ys, xs = np.nonzero(red)
    if len(xs) < 20:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def rectify(small, well_quad, tab_xy):
    quad = order_corners(well_quad)
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(small, M, (WARP, WARP))
    if tab_xy is None:
        return warped, 0
    # rotate so the tab (magnet) side points RIGHT
    cx = well_quad[:, 0].mean()
    cy = well_quad[:, 1].mean()
    dx, dy = tab_xy[0] - cx, tab_xy[1] - cy
    ang = np.degrees(np.arctan2(dy, dx))  # 0=right, 90=down (image coords)
    if -45 <= ang < 45:
        k = 0
    elif 45 <= ang < 135:  # tab below -> rotate CCW 90
        k = 1
    elif ang >= 135 or ang < -135:  # tab left -> 180
        k = 2
    else:  # tab above -> rotate CW 90
        k = 3
    warped = np.rot90(warped, k=-k) if k else warped
    return np.ascontiguousarray(warped), k


def process_frame(small):
    holder, well = find_holder_and_well(small)
    if well is None:
        return None, "no_well"
    quad = quad_from_mask(well)
    if quad is None:
        return None, "no_quad"
    tab = red_centroid(small)
    warped, k = rectify(small, quad, tab)
    return {"warped": warped, "quad": quad, "rot_k": k, "tab": tab}, "ok"
