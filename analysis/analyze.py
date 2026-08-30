"""Per-series transport analysis.

For every series (condition x arm x repeat):
  - order frames by real capture time (elapsed hours from the series' t0 burst)
  - median-filtered darkness profiles, per-frame median offset removed
  - subtract the series' t=0 profile (fallback: per-column min over time)
  - front = end of the leftmost connected run above 25% of plume peak
  - centroid, mass over the valid block region (0.2 .. 9.3 mm from gap edge)
  - linear-fit velocity of front and centroid over time (mm/h)

Outputs:
  series_measurements.csv  (one row per photo: elapsed_h, front_mm, centroid_mm, mass)
  series_summary.csv       (one row per series: velocities, R2, flags)
  figs/<series>.png        (kymograph + transport curves)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")
WELL_MM = 11.0
BORDER = 22
WARP = 480
VALID_MM = (0.2, 9.3)
FRONT_FRAC = 0.25
MIN_MASS = 25.0
MIN_PEAK = 3.0


def x_axis(n):
    return (np.arange(n) + BORDER) * WELL_MM / WARP - 1.0


def front_from_profile(row, x_mm, valid):
    peak = row[valid].max()
    if peak < MIN_PEAK:
        return np.nan
    thr = FRONT_FRAC * peak
    above = (row > thr) & valid
    if not above.any():
        return np.nan
    start = int(np.argmax(above))  # leftmost point above threshold
    f = start
    gap_px = 0
    max_gap = int(0.5 / (WELL_MM / WARP))  # 0.5 mm of tolerance
    for i in range(start, len(row)):
        if above[i]:
            f = i
            gap_px = 0
        else:
            gap_px += 1
            if gap_px > max_gap:
                break
    return float(x_mm[f])


def analyze_series(g, prof, bad_idx):
    g = g.sort_values("capture_time")
    t0 = g.capture_time.min()
    rows = []
    K = []
    kept = []
    for r in g.itertuples():
        if int(r.idx) in bad_idx:
            continue
        p = median_filter(prof[str(int(r.idx))].astype(float), size=11)
        K.append(p)
        kept.append(r)
    if not K:
        return None, None
    K = np.array(K)
    n = K.shape[1]
    x_mm = x_axis(n)
    valid = (x_mm > VALID_MM[0]) & (x_mm < VALID_MM[1])
    q = K - np.median(K[:, valid], axis=1)[:, None]
    has_t0 = kept[0].timepoint_hr == 0
    base = q[0] if has_t0 else q.min(axis=0)
    D = np.clip(q - base[None, :], 0, None)
    D[:, ~valid] = 0
    out = []
    for row, r in zip(D, kept):
        elapsed = (r.capture_time - t0).total_seconds() / 3600.0
        mass = float(row.sum())
        if mass < MIN_MASS:
            front = cent = np.nan
        else:
            front = front_from_profile(row, x_mm, valid)
            cent = float((row * x_mm).sum() / mass)
        out.append(
            {
                "idx": int(r.idx),
                "series": r.series,
                "timepoint_hr": r.timepoint_hr,
                "elapsed_h": elapsed,
                "front_mm": front,
                "centroid_mm": cent,
                "mass": mass,
                "magnet": r.magnet_f,
            }
        )
    return pd.DataFrame(out), (D, x_mm)


def fit_velocity(t, y):
    m = ~(np.isnan(t) | np.isnan(y))
    if m.sum() < 3:
        return np.nan, np.nan
    t, y = np.asarray(t)[m], np.asarray(y)[m]
    A = np.vstack([t, np.ones_like(t)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(r2)


if __name__ == "__main__":
    photos = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
    photos["capture_time"] = pd.to_datetime(photos["capture_time"])
    prof = np.load(S / "profiles.npz")
    wr = pd.read_csv("warp_report.csv")
    bad_idx = set(wr[~wr.refined]["idx"])
    figdir = S / "figs"
    figdir.mkdir(exist_ok=True)

    all_rows = []
    summaries = []
    for sname, g in photos.groupby("series"):
        df, kd = analyze_series(g, prof, bad_idx)
        if df is None:
            continue
        all_rows.append(df)
        # velocity fits on t>0 measurable points
        sub = df[df.elapsed_h > 0.05]
        v_front, r2_f = fit_velocity(sub.elapsed_h, sub.front_mm)
        v_cent, r2_c = fit_velocity(sub.elapsed_h, sub.centroid_mm)
        summaries.append(
            {
                "series": sname,
                "n_frames": len(df),
                "v_front_mm_h": v_front,
                "r2_front": r2_f,
                "v_centroid_mm_h": v_cent,
                "r2_centroid": r2_c,
                "final_front_mm": df.front_mm.dropna().iloc[-1] if df.front_mm.notna().any() else np.nan,
                "final_centroid_mm": df.centroid_mm.dropna().iloc[-1] if df.centroid_mm.notna().any() else np.nan,
            }
        )
        # figure
        D, x_mm = kd
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].imshow(
            D,
            aspect="auto",
            cmap="inferno",
            extent=[x_mm[0], x_mm[-1], df.elapsed_h.max(), 0],
        )
        axes[0].set_xlabel("mm from gap edge")
        axes[0].set_ylabel("elapsed h")
        axes[0].set_title(sname)
        axes[1].plot(df.elapsed_h, df.front_mm, "o-", label="front")
        axes[1].plot(df.elapsed_h, df.centroid_mm, "s-", label="centroid")
        axes[1].set_xlabel("elapsed h")
        axes[1].set_ylabel("mm")
        axes[1].set_ylim(0, 10)
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(figdir / f"{sname}.png", dpi=100)
        plt.close(fig)

    pd.concat(all_rows).to_csv("series_measurements.csv", index=False)
    pd.DataFrame(summaries).to_csv("series_summary.csv", index=False)
    print("series analyzed:", len(summaries))
