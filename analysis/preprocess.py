"""Preprocess photos for classification.

For every cataloged photo:
  - save a downscaled full frame (long side ~1100 px)
  - detect red marks on the white 3D-print (tally marks / control 'C'),
    excluding red-on-green sticker text, and save a full-resolution crop
    around them (downsized to ~700 px)

Outputs go to <out_dir>/frames and <out_dir>/tally, named by photo index.
Writes preprocess_report.csv with the detection outcome per photo.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DAY_DIR = None
OUT_DIR = None


def red_on_white_boxes(img_small):
    """Boxes (in small-image coords) of red components whose surroundings are white."""
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((h < 10) | (h > 170)) & (s > 70) & (v > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, hgt, area = stats[i]
        if area < 40:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((25, 25), np.uint8)) - cv2.dilate(
            comp, np.ones((7, 7), np.uint8)
        )
        ring_px = hsv[ring.astype(bool)]
        if len(ring_px) == 0:
            continue
        # white surroundings: low saturation, high value
        if np.median(ring_px[:, 1]) < 60 and np.median(ring_px[:, 2]) > 140:
            boxes.append((x, y, x + w, y + hgt))
    return boxes


def process(args):
    idx, rel_path = args
    src = DAY_DIR / rel_path
    img = cv2.imread(str(src))
    if img is None:
        return {"idx": idx, "path": rel_path, "status": "read_error"}
    H, W = img.shape[:2]
    scale = 1100 / max(H, W)
    small = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT_DIR / "frames" / f"{idx:04d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])

    boxes = red_on_white_boxes(small)
    status = "ok" if boxes else "no_red_marks"
    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        # back to full-res coords, with generous padding to keep tab context
        pad = 60
        fx0 = max(0, int(x0 / scale) - pad * 4)
        fy0 = max(0, int(y0 / scale) - pad * 4)
        fx1 = min(W, int(x1 / scale) + pad * 4)
        fy1 = min(H, int(y1 / scale) + pad * 4)
        crop = img[fy0:fy1, fx0:fx1]
        ch, cw = crop.shape[:2]
        cscale = min(1.0, 700 / max(ch, cw))
        if cscale < 1.0:
            crop = cv2.resize(
                crop, (round(cw * cscale), round(ch * cscale)), interpolation=cv2.INTER_AREA
            )
        cv2.imwrite(str(OUT_DIR / "tally" / f"{idx:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"idx": idx, "path": rel_path, "status": status, "n_red_boxes": len(boxes)}


def init(day_dir, out_dir):
    global DAY_DIR, OUT_DIR
    DAY_DIR = Path(day_dir)
    OUT_DIR = Path(out_dir)


if __name__ == "__main__":
    day_dir, out_dir = sys.argv[1], sys.argv[2]
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "tally").mkdir(parents=True, exist_ok=True)

    photos = pd.read_csv(Path(__file__).parent / "photos.csv")
    tasks = list(enumerate(photos["path"]))
    with Pool(8, initializer=init, initargs=(day_dir, out_dir)) as pool:
        results = pool.map(process, tasks)
    rep = pd.DataFrame(results)
    rep.to_csv(Path(__file__).parent / "preprocess_report.csv", index=False)
    print(rep["status"].value_counts().to_string())
