"""Per-series transport analysis, v2 — in-frame referencing.

v1 subtracted profiles across frames, which let per-shot lighting drift and the
right-seam band dominate the signal. v2:

  - valid block region tightened to 0.3–8.0 mm (right seam excluded)
  - each frame's profile is referenced to ITS OWN far-zone baseline
    (median darkness over 5.5–8.0 mm) so lighting drift cancels in-frame
  - the t=0 in-frame excess shape is then subtracted (static gel features)
  - metrics per frame: transported mass (sum of excess), x50 / x90 =
    mass-quantile depths (median depth and leading edge), all in mm from gap
  - caveat: if a plume truly reaches past ~5.5 mm the far-zone baseline
    inflates and the metric under-reports; leading edges here stay < 5 mm.

Outputs: series_measurements2.csv, series_summary2.csv, figs2/<series>.png
"""

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
VALID = (0.3, 8.0)
FAR = (5.5, 8.0)


def x_axis(n):
    return (np.arange(n) + BORDER) * WELL_MM / WARP - 1.0


def inframe_excess(p, x_mm):
    far = (x_mm > FAR[0]) & (x_mm < FAR[1])
    return np.clip(p - np.median(p[far]), 0, None)


def quantile_depth(row, x_mm, q):
    m = row.sum()
    if m <= 0:
        return np.nan
    c = np.cumsum(row) / m
    i = int(np.searchsorted(c, q))
    return float(x_mm[min(i, len(x_mm) - 1)])


def analyze_series(g, prof, bad_idx):
    g = g.sort_values("capture_time")
    t0 = g.capture_time.min()
    K, kept = [], []
    for r in g.itertuples():
        if int(r.idx) in bad_idx:
            continue
        K.append(median_filter(prof[str(int(r.idx))].astype(float), size=11))
        kept.append(r)
    if not K:
        return None, None
    K = np.array(K)
    n = K.shape[1]
    x_mm = x_axis(n)
    valid = (x_mm > VALID[0]) & (x_mm < VALID[1])
    E = np.array([inframe_excess(p, x_mm) for p in K])
    has_t0 = kept[0].timepoint_hr == 0
    base = E[0] if has_t0 else E.min(axis=0)
    D = np.clip(E - base[None, :], 0, None)
    D[:, ~valid] = 0
    out = []
    for row, r in zip(D, kept):
        elapsed = (r.capture_time - t0).total_seconds() / 3600.0
        mass = float(row.sum())
        out.append(
            {
                "idx": int(r.idx),
                "series": r.series,
                "timepoint_hr": r.timepoint_hr,
                "elapsed_h": elapsed,
                "mass": mass,
                "x50_mm": quantile_depth(row, x_mm, 0.5) if mass > 15 else np.nan,
                "x90_mm": quantile_depth(row, x_mm, 0.9) if mass > 15 else np.nan,
                "magnet": r.magnet_f,
            }
        )
    return pd.DataFrame(out), (D, x_mm)


if __name__ == "__main__":
    photos = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
    photos["capture_time"] = pd.to_datetime(photos["capture_time"])
    prof = np.load(S / "profiles.npz")
    wr = pd.read_csv("warp_report.csv")
    bad_idx = set(wr[~wr.refined]["idx"])
    figdir = S / "figs2"
    figdir.mkdir(exist_ok=True)
    all_rows, summaries = [], []
    for sname, g in photos.groupby("series"):
        df, kd = analyze_series(g, prof, bad_idx)
        if df is None:
            continue
        all_rows.append(df)
        final = df[df.timepoint_hr == 21.5]
        summaries.append(
            {
                "series": sname,
                "final_mass": float(final.mass.mean()) if len(final) else np.nan,
                "final_x50": float(final.x50_mm.mean()) if len(final) else np.nan,
                "final_x90": float(final.x90_mm.mean()) if len(final) else np.nan,
            }
        )
        D, x_mm = kd
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
        axes[0].imshow(D, aspect="auto", cmap="inferno",
                       extent=[x_mm[0], x_mm[-1], max(df.elapsed_h.max(), 0.1), 0])
        axes[0].set_xlim(0, 8.5)
        axes[0].set_xlabel("mm from gap edge")
        axes[0].set_ylabel("elapsed h")
        axes[0].set_title(sname, fontsize=9)
        ax2 = axes[1]
        ax2.plot(df.elapsed_h, df.x90_mm, "o-", label="x90 (leading edge)")
        ax2.plot(df.elapsed_h, df.x50_mm, "s-", label="x50 (median depth)")
        ax2.set_ylim(0, 8)
        ax2.set_xlabel("elapsed h")
        ax2.set_ylabel("mm")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7)
        axt = ax2.twinx()  # mass on its own small axis? no dual axis in deliverables; internal QC only
        axt.plot(df.elapsed_h, df.mass, "-", color="gray", alpha=0.4)
        axt.set_yticks([])
        plt.tight_layout()
        plt.savefig(figdir / f"{sname}.png", dpi=100)
        plt.close(fig)
    pd.concat(all_rows).to_csv("series_measurements2.csv", index=False)
    pd.DataFrame(summaries).to_csv("series_summary2.csv", index=False)
    print("series:", len(summaries))
