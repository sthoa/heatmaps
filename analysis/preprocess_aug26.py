"""Preprocess Aug 26 photos: downscaled frames, orange-sticker crops, and
red tally-mark crops.

Aug 26 combines both labelling systems: condition text (PEG/COOH, 0.6, BSA,
CTRL) is written in red marker on an orange sticker, while the repeat number
is red tally marks on the white magnet strap.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DAY_DIR = None
OUT_DIR = None


def sticker_box(small):
    """largest saturated orange blob = the label sticker."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    orange = ((h >= 8) & (h <= 40) & (s > 120) & (v > 110)).astype(np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(orange, 8)
    best, ba = None, 0
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area > ba and area > 800:
            ba, best = area, (x, y, w, hh)
    return best


def tally_boxes(small):
    """red marks whose surroundings are white plastic (the strap tallies).
    Excludes red sticker text, whose surroundings are orange."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 10) | (h > 170)) & (s > 90) & (v > 70)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 25 or area > 9000:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((21, 21), np.uint8)) - cv2.dilate(comp, np.ones((5, 5), np.uint8))
        rp = hsv[ring.astype(bool)]
        if len(rp) == 0:
            continue
        # white plastic ring: low saturation and bright (orange sticker ring is saturated)
        if np.median(rp[:, 1]) < 70 and np.median(rp[:, 2]) > 140:
            boxes.append((x, y, x + w, y + hh, area))
    return boxes


def process(args):
    idx, rel = args
    img = cv2.imread(str(DAY_DIR / rel))
    if img is None:
        return {"idx": idx, "status": "read_error"}
    H, W = img.shape[:2]
    sc = 1100 / max(H, W)
    small = cv2.resize(img, (round(W * sc), round(H * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_DIR / "frames" / f"{idx:04d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])

    status = []
    box = sticker_box(small)
    if box is None:
        status.append("no_sticker")
    else:
        x, y, w, hh = box
        pad = 20
        crop = img[max(0, int((y - pad) / sc)):min(H, int((y + hh + pad) / sc)),
                   max(0, int((x - pad) / sc)):min(W, int((x + w + pad) / sc))]
        ch, cw = crop.shape[:2]
        if ch and cw:
            k = min(1.0, 900 / max(ch, cw))
            if k < 1.0:
                crop = cv2.resize(crop, (round(cw * k), round(ch * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT_DIR / "sticker" / f"{idx:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 93])

    tb = tally_boxes(small)
    if not tb:
        status.append("no_tally")
    else:
        x0 = min(b[0] for b in tb); y0 = min(b[1] for b in tb)
        x1 = max(b[2] for b in tb); y1 = max(b[3] for b in tb)
        pad = 40
        crop = img[max(0, int((y0 - pad) / sc)):min(H, int((y1 + pad) / sc)),
                   max(0, int((x0 - pad) / sc)):min(W, int((x1 + pad) / sc))]
        ch, cw = crop.shape[:2]
        if ch and cw:
            k = min(1.0, 800 / max(ch, cw))
            if k < 1.0:
                crop = cv2.resize(crop, (round(cw * k), round(ch * k)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT_DIR / "tally" / f"{idx:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"idx": idx, "path": rel, "status": ",".join(status) if status else "ok", "n_tally": len(tb)}


def init(day_dir, out_dir):
    global DAY_DIR, OUT_DIR
    DAY_DIR, OUT_DIR = Path(day_dir), Path(out_dir)


if __name__ == "__main__":
    day_dir, out_dir = sys.argv[1], sys.argv[2]
    out = Path(out_dir)
    for sub in ("frames", "sticker", "tally"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    photos = pd.read_csv(Path(__file__).parent / "photos26.csv")
    tasks = list(enumerate(photos["path"]))
    with Pool(8, initializer=init, initargs=(day_dir, out_dir)) as pool:
        res = pool.map(process, tasks)
    rep = pd.DataFrame(res)
    rep.to_csv(Path(__file__).parent / "preprocess26_report.csv", index=False)
    print(rep["status"].value_counts().to_string())
