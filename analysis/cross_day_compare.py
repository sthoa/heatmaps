"""Cross-day comparison: 0.4% vs 0.6% agarose, and Aug 27 alongside.

The per-day pages each answer "did the magnet move anything on this day". This
page asks what only the days together can answer.

ASYMMETRY (Aug 23 + Aug 26). Each block heatmap is collapsed to one number:
mean NP darkness on the magnet side of the injection gap minus the far side.
Cloudiness, exposure and BSA turbidity darken both halves equally and cancel;
what survives is directional. Controls therefore sit near zero by construction,
which makes them a running check on the metric rather than just a baseline.

WHY THESE TWO DAYS. Aug 23 and Aug 26 match on everything the metric depends
on -- centre injection, 6 h, 13 timepoints at 30 min, PEG vs COOH, large
magnet, identical warp geometry and margins -- except agarose concentration,
0.4% vs 0.6%. Comparing Aug 23's large-magnet arms with Aug 26's NON-BSA
large-magnet arms is therefore a read on gel stiffness. It is a between-day
comparison (different gel batch, different session lighting), so it is
suggestive, not a designed experiment.

BOTH DAYS ARE RECOMPUTED HERE from one definition. The per-day metric CSVs in
this directory came from separate ad-hoc scripts and cannot be assumed
mutually consistent, which is the one thing a cross-day plot needs.

AUG 27 CANNOT JOIN THE SAME AXES. It is back injection: the gap sits at the
block's edge, so there is no far side and the asymmetry metric is undefined.
It appears in its own panel as penetration depth -- a different quantity on a
different time span, plotted together only for context.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_block_atlas23 as A23
import make_block_atlas26 as A26
from block26 import block_quad as quad26

GAP_EXCLUDE_MM = 1.0   # half-width of the band around the gap that belongs to neither side
OUT = Path(__file__).parent / "outputs"


def asymmetry(D, extent):
    """magnet-side minus far-side mean darkness of one processed block field.

    The gap is the brightest column band near the middle; it is excluded from
    both sides so that the reservoir itself cannot drive the difference.
    """
    n = D.shape[1]
    mm = np.linspace(extent[0], extent[1], n)
    col = D.mean(axis=0)
    lo, hi = int(n * 0.30), int(n * 0.70)
    gap = lo + int(np.argmax(col[lo:hi]))
    per_mm = (extent[1] - extent[0]) / n
    k = max(1, int(round(GAP_EXCLUDE_MM / per_mm)))
    right, left = D[:, gap + k:], D[:, : max(gap - k, 1)]
    if right.size == 0 or left.size == 0:
        return np.nan, mm[gap]
    return float(right.mean() - left.mean()), float(mm[gap])


def curves_23():
    m = pd.read_csv(Path(__file__).parent / "photos23_final.csv", parse_dates=["capture_time"])
    m["date"] = m.capture_time.dt.date
    quads = A23.repaired_quads()
    rows = []
    for r in m.itertuples():
        img = cv2.imread(str(A23.S / "frames" / f"{int(r.idx):04d}.jpg"))
        if img is None:
            continue
        q = quads.get(int(r.idx))
        if q is None:
            q = A23.block_quad(img)
        if q is None:
            continue
        D = A23.process_abs(A23.field_by_date(img, q, r.date))
        a, g = asymmetry(D, A23.EXTENT_MM)
        rows.append(dict(day="Aug 23", agarose="0.4%", bsa="non-BSA", coating=r.coating,
                         arm=r.arm, series=r.series, t=r.timepoint_hr, asym=a, gap_mm=g))
    return pd.DataFrame(rows)


def curves_26():
    m = pd.read_csv(Path(__file__).parent / "photos26_final.csv", parse_dates=["capture_time"])
    rows = []
    for r in m.itertuples():
        img = cv2.imread(str(A26.S / "frames" / f"{int(r.idx):04d}.jpg"))
        if img is None:
            continue
        q = quad26(img)
        if q is None:
            continue
        D = A26.process_abs(A26.field(img, q))
        a, g = asymmetry(D, A26.EXTENT_MM)
        rows.append(dict(day="Aug 26", agarose="0.6%", bsa=r.bsa, coating=r.coating,
                         arm="control" if r.control else "large", series=r.series,
                         t=r.timepoint_hr, asym=a, gap_mm=g))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure ----

C = {("0.4%", "COOH"): "#C0392B", ("0.4%", "PEG"): "#E67E22",
     ("0.6%", "COOH"): "#1F6FB2", ("0.6%", "PEG"): "#4CA6A8"}


def panel_asymmetry(ax, df):
    """Individual repeats are drawn, not just a mean +/- SD band.

    At n=3 the band hides how wide the scatter is -- one 0.6% COOH repeat
    plateaus at 12.8 L* and another at 48.1 -- and that scatter is the reason
    none of the pairwise contrasts reach significance. Showing every series
    makes the limitation visible instead of implied.
    """
    sel = df[(df.arm == "large") & (df.bsa == "non-BSA")]
    for (ag, co), g in sel.groupby(["agarose", "coating"]):
        p = g.pivot_table(index="t", columns="series", values="asym")
        for c in p.columns:
            ax.plot(p.index, p[c], color=C[(ag, co)], lw=.8, alpha=.38, zorder=2)
        ax.plot(p.index, p.mean(axis=1), color=C[(ag, co)], lw=2.4,
                ls="-" if co == "COOH" else "--", marker="o", ms=3.8, zorder=3,
                label=f"{ag} agarose \u00b7 {co}  (n={p.shape[1]})")
    ctl = df[(df.arm == "control") & (df.bsa == "non-BSA")].pivot_table(
        index="t", columns="series", values="asym")
    ax.fill_between(ctl.index, ctl.min(axis=1), ctl.max(axis=1),
                    color="#9aa0a6", alpha=.32, lw=0, zorder=1,
                    label=f"no-magnet controls, both days (n={ctl.shape[1]}, envelope)")
    ax.axhline(0, color="#888", lw=.8, zorder=0)
    ax.set_xlabel("hours after magnet applied", fontsize=10)
    ax.set_ylabel("asymmetry  (magnet side \u2212 far side, L*)", fontsize=10)
    ax.set_title("Centre injection \u2014 in a 2\u00d72 factorial the gel-stiffness effect holds "
                 "(p=0.045) and COOH over PEG\nis marginal (p=0.086); the wider coating gap at 0.6% "
                 "is not supported (interaction p=0.38)",
                 fontsize=11, pad=9, loc="left")
    ax.legend(fontsize=8.6, frameon=False, loc="upper left")
    ax.set_xlim(-0.15, 6.15); ax.grid(alpha=.16, lw=.6)


def panel_back(ax, path):
    """0.4% vs 0.6% in the back-injection run, within one day.

    Front position (deepest point still at 20% of peak darkness) of each magnet
    series minus the mean of its own cell's controls, so the comparison is
    paired within (BSA, coating). Bands are +/- 1 SEM across 12 series.
    """
    b = pd.read_csv(path)
    mag = b[b.arm == "magnet"].dropna(subset=["front_net"])
    for ag, g in mag.groupby("agarose"):
        key = f"{ag:.1f}%".replace(".0%", "%") if isinstance(ag, float) else f"{ag}%"
        col = C[("0.4%", "COOH")] if float(ag) < 0.5 else C[("0.6%", "COOH")]
        p = g.pivot_table(index="t", columns="series", values="front_net")
        mu, se = p.mean(axis=1), p.std(axis=1) / np.sqrt(p.notna().sum(axis=1))
        ax.fill_between(p.index, mu - se, mu + se, color=col, alpha=.15, lw=0)
        ax.plot(p.index, mu, color=col, lw=2.2, marker="o", ms=3.6,
                label=f"{float(ag)*100:.0f}".rstrip() and f"{ag}% agarose  (n={p.shape[1]} series)")
    ax.axhline(0, color="#888", lw=.9)
    ax.set_xlabel("hours after magnet applied", fontsize=10)
    ax.set_ylabel("front position, magnet \u2212 control (mm)", fontsize=10)
    ax.set_title("Back injection, Aug 27, within one day \u2014 no stiffness difference, and barely\n"
                 "any magnet effect to modulate (+0.41 mm overall, p=0.26)",
                 fontsize=11, pad=9, loc="left")
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    ax.grid(alpha=.16, lw=.6)


def figure(df, out):
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 9.2),
                             gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.44))
    panel_asymmetry(axes[0], df)
    panel_back(axes[1], Path(__file__).parent / "back_depth_metrics.csv")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    print("Aug 23 ...", flush=True); d23 = curves_23()
    print("Aug 26 ...", flush=True); d26 = curves_26()
    df = pd.concat([d23, d26], ignore_index=True)
    df.to_csv(Path(__file__).parent / "cross_day_asymmetry.csv", index=False)
    # one plateau value per series (mean of the 5-6 h points): the input to centre_stats.py
    plateau = (df[df.arm != "control"].assign(plateau=lambda d: d.asym)
               .query("t >= 5.0").groupby(["day", "agarose", "bsa", "arm", "coating", "series"], as_index=False)
               .plateau.mean())
    plateau.to_csv(Path(__file__).parent / "centre_plateau_by_series.csv", index=False)
    print(f"wrote cross_day_asymmetry.csv  n={len(df)}")
    print(df.groupby(["day", "arm"]).asym.describe()[["count", "mean", "50%"]])
    OUT.mkdir(exist_ok=True)
    figure(df, OUT / "cross_day_comparison.png")


