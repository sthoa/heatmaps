"""Preprocess Aug 23 photos: downscaled frames + full-res green-sticker crops.

The Aug 23 labels are written on the sticker (PEG/COOH, L1-L3, S1-S3, control),
so the crop that matters is the sticker, not the tally tabs.
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
    """largest saturated orange/yellow blob = the label sticker (Aug 23 uses
    orange tape with black marker; Aug 27 used green)."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (((h >= 8) & (h <= 40) & (s > 110) & (v > 110))
             | ((h > 30) & (h < 95) & (s > 40) & (v > 40))).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    best, ba = None, 0
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area > ba and area > 300:
            ba, best = area, (x, y, w, hh)
    return best


def process(args):
    idx, rel = args
    img = cv2.imread(str(DAY_DIR / rel))
    if img is None:
        return {"idx": idx, "status": "read_error"}
    H, W = img.shape[:2]
    scale = 1100 / max(H, W)
    small = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_DIR / "frames" / f"{idx:04d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])

    box = sticker_box(small)
    status = "ok"
    if box is None:
        status = "no_sticker"
    else:
        x, y, w, hh = box
        pad = 22
        fx0 = max(0, int((x - pad) / scale))
        fy0 = max(0, int((y - pad) / scale))
        fx1 = min(W, int((x + w + pad) / scale))
        fy1 = min(H, int((y + hh + pad) / scale))
        crop = img[fy0:fy1, fx0:fx1]
        ch, cw = crop.shape[:2]
        if ch and cw:
            cs = min(1.0, 900 / max(ch, cw))
            if cs < 1.0:
                crop = cv2.resize(crop, (round(cw * cs), round(ch * cs)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT_DIR / "sticker" / f"{idx:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"idx": idx, "path": rel, "status": status}


def init(day_dir, out_dir):
    global DAY_DIR, OUT_DIR
    DAY_DIR, OUT_DIR = Path(day_dir), Path(out_dir)


if __name__ == "__main__":
    day_dir, out_dir = sys.argv[1], sys.argv[2]
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "sticker").mkdir(parents=True, exist_ok=True)
    photos = pd.read_csv(Path(__file__).parent / "photos23.csv")
    tasks = list(enumerate(photos["path"]))
    with Pool(8, initializer=init, initargs=(day_dir, out_dir)) as pool:
        res = pool.map(process, tasks)
    rep = pd.DataFrame(res)
    rep.to_csv(Path(__file__).parent / "preprocess23_report.csv", index=False)
    print(rep["status"].value_counts().to_string())
