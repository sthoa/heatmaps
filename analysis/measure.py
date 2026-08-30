"""Measure NP transport from warped well images.

Geometry after warping: 480x480 px spans the well interior = injection gap
(~1 mm, LEFT edge) + agarose block (10 mm). Orientation: magnet on the RIGHT.

Per photo:
  - darkness map D = inverse lightness of the well interior (border-cropped)
  - column profile p(x) = robust mean of D over y
  - background = per-image darkness floor (median of the brightest 30% of
    columns), removed so lighting differences between shots cancel
  - gap edge located from the t=0 profile of each series (steep drop after
    the leftmost dark band); x=0 mm at the gap/block boundary
  - front_mm  = rightmost x where the excess profile stays above FRONT_FRAC of
    its left-region peak (connected from the left)
  - centroid_mm, mass = first moment / integral of the excess profile

Outputs: profiles.npz (per-photo x-profiles) + measurements.csv
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WARP = 480
BORDER = 22          # px cropped from each side of the warped well (rim/shadow)
WELL_MM = 11.0       # gap (~1mm) + block (10mm)
FRONT_FRAC = 0.20

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")


def column_profile(img):
    """Darkness profile across x, robust to glare/streaks."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    core = L[BORDER:-BORDER, BORDER:-BORDER]
    dark = 255.0 - core
    # robust column statistic: trimmed mean over y (drop 15% brightest+darkest rows)
    p = np.sort(dark, axis=0)
    n = p.shape[0]
    lo, hi = int(0.15 * n), int(0.85 * n)
    return p[lo:hi].mean(axis=0)


def measure_profile(p):
    """Background-corrected excess profile + summary stats (pixel units)."""
    floor = np.median(np.sort(p)[: max(3, int(0.3 * len(p)))])
    ex = np.clip(p - floor, 0, None)
    peak_left = ex[: len(ex) // 2].max() if ex[: len(ex) // 2].size else 0
    if peak_left <= 2:  # essentially no signal
        return ex, np.nan, np.nan, float(ex.sum())
    thr = FRONT_FRAC * peak_left
    above = ex > thr
    # front: rightmost index of the run connected to the left mass
    start = int(np.argmax(ex[: len(ex) // 2]))
    front = start
    for i in range(start, len(ex)):
        if above[i]:
            front = i
        elif not above[max(0, i - 3) : i + 1].any():
            break
    mass = float(ex.sum())
    centroid = float((ex * np.arange(len(ex))).sum() / mass) if mass > 0 else np.nan
    return ex, float(front), centroid, mass


def px_to_mm(x_px, n, gap_px):
    """Convert profile index to mm from the gap/block boundary."""
    mm_per_px = WELL_MM / n
    return (x_px - gap_px) * mm_per_px


def find_gap_edge(ex0):
    """Right edge of the injection band in a t=0 excess profile (px index)."""
    n = len(ex0)
    left = ex0[: n // 3]
    if left.max() <= 2:
        return int(0.09 * n)  # default ~1mm
    peak = int(np.argmax(left))
    thr = 0.35 * left.max()
    edge = peak
    for i in range(peak, n // 2):
        if ex0[i] > thr:
            edge = i
        else:
            break
    return min(edge + 2, n // 2)


if __name__ == "__main__":
    photos = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
    profiles = {}
    rows = []
    for r in photos.itertuples():
        img = cv2.imread(str(S / "prep" / "warped" / f"{int(r.idx):04d}.jpg"))
        p = column_profile(img)
        ex, front_px, centroid_px, mass = measure_profile(p)
        profiles[str(int(r.idx))] = ex
        rows.append(
            {
                "idx": int(r.idx),
                "front_px": front_px,
                "centroid_px": centroid_px,
                "mass": mass,
                "n_px": len(ex),
            }
        )
    np.savez_compressed(S / "profiles.npz", **profiles)
    pd.DataFrame(rows).to_csv("measurements_px.csv", index=False)
    print("measured", len(rows), "photos")
