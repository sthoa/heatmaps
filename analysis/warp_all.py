"""Final rectification: two-pass warp of every frame to a canonical 480x480 well,
orientation-normalized so the magnet/tab side is on the RIGHT.

Inputs: prep/frames/NNNN.jpg + rects.json (validated well minAreaRects).
Outputs: prep/warped/NNNN.jpg + warp_report.csv (rotation applied, refine status).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WARP = 480
MARGIN = 70  # px margin around the well in the intermediate canvas


def order_corners(pts):
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def red_tab_boxes(img):
    """Red components with white surroundings (tally marks / C), as in preprocess."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 10) | (h > 170)) & (s > 70) & (v > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, hgt, area = stats[i]
        if area < 40 or area > 20000:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((25, 25), np.uint8)) - cv2.dilate(
            comp, np.ones((7, 7), np.uint8)
        )
        ring_px = hsv[ring.astype(bool)]
        if len(ring_px) == 0:
            continue
        ring_h = np.median(ring_px[:, 0])
        ring_s = np.median(ring_px[:, 1])
        ring_v = np.median(ring_px[:, 2])
        # white plastic ring: low saturation, bright, and NOT the pale-green sticker
        if ring_s < 45 and ring_v > 140 and not (35 <= ring_h <= 95 and ring_s > 20):
            boxes.append((x + w / 2, y + hgt / 2, area))
    return boxes


def refine_quad(canvas):
    """Re-detect the well quad inside the margin-warped canvas."""
    H, W = canvas.shape[:2]
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    beige = (h >= 5) & (h <= 35) & (s >= 8) & (s <= 170) & (v >= 60) & (v <= 245)
    brown = (h <= 30) & (s >= 60) & (v >= 25) & (v < 130)
    m = (beige | brown).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ff = m.copy()
    mk = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 1)
    m = m | ((ff == 0).astype(np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best, best_score = None, 0
    for i in range(1, n):
        area = stats[i, 4]
        if area < 0.25 * (W - 2 * MARGIN) ** 2:
            continue
        comp = (labels == i).astype(np.uint8)
        rect = cv2.minAreaRect(cv2.findNonZero(comp))
        (rcx, rcy), (rw, rh), ang = rect
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > 1.6:
            continue
        extent = area / (rw * rh)
        if extent < 0.7:
            continue
        # must roughly cover the central region
        if abs(rcx - W / 2) > MARGIN * 1.5 or abs(rcy - H / 2) > MARGIN * 1.5:
            continue
        score = extent * (1.6 - aspect) * area
        if score > best_score:
            best_score, best = score, rect
    return best


def process(idx, rect, frames_dir, out_dir):
    img = cv2.imread(str(frames_dir / f"{idx:04d}.jpg"))
    (cx, cy, w, h, ang) = rect
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)
    quad = order_corners(box)
    side = WARP - 2 * MARGIN
    dst = np.array(
        [[MARGIN, MARGIN], [MARGIN + side, MARGIN], [MARGIN + side, MARGIN + side], [MARGIN, MARGIN + side]],
        dtype=np.float32,
    )
    M1 = cv2.getPerspectiveTransform(quad, dst)
    canvas = cv2.warpPerspective(img, M1, (WARP, WARP))

    r2 = refine_quad(canvas)
    refined = r2 is not None
    if refined:
        box2 = cv2.boxPoints(r2).astype(np.float32)
        quad2 = order_corners(box2)
        dst2 = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
        M2 = cv2.getPerspectiveTransform(quad2, dst2)
        final = cv2.warpPerspective(canvas, M2, (WARP, WARP))
    else:
        # fall back: crop margin
        final = canvas[MARGIN : WARP - MARGIN, MARGIN : WARP - MARGIN]
        final = cv2.resize(final, (WARP, WARP), interpolation=cv2.INTER_AREA)

    # orientation from ring-validated red tabs in the ORIGINAL frame
    boxes = red_tab_boxes(img)
    k = 0
    tab_found = len(boxes) > 0
    if tab_found:
        # the tally/C mark is the largest red-on-white blob; small smudges elsewhere lose
        tx, ty, _ = max(boxes, key=lambda b: b[2])
        dx, dy = tx - cx, ty - cy
        a = np.degrees(np.arctan2(dy, dx))
        if -45 <= a < 45:
            k = 0  # tab right: already correct
        elif 45 <= a < 135:
            k = 1  # tab below: rotate CCW so bottom -> right
        elif a >= 135 or a < -135:
            k = 2
        else:
            k = 3
        if k:
            final = np.ascontiguousarray(np.rot90(final, k=k))
    cv2.imwrite(str(out_dir / f"{idx:04d}.jpg"), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"idx": idx, "refined": refined, "tab_found": tab_found, "rot_k": k}


if __name__ == "__main__":
    S = Path(sys.argv[1])
    frames_dir = S / "prep" / "frames"
    out_dir = S / "prep" / "warped"
    out_dir.mkdir(exist_ok=True)
    rects = {int(k): v["rect"] for k, v in json.load(open(S / "rects.json")).items()}
    rows = [process(i, rects[i], frames_dir, out_dir) for i in sorted(rects)]
    rep = pd.DataFrame(rows)
    rep.to_csv(Path(__file__).parent / "warp_report.csv", index=False)
    print("refined:", rep.refined.sum(), "/", len(rep), "| tab found:", rep.tab_found.sum())
    print(rep.rot_k.value_counts().to_string())
