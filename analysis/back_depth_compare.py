"""0.4% vs 0.6% agarose in the BACK-injection run (Aug 27), within one day.

This is the stronger of the two stiffness comparisons. Aug 27 ran both agarose
concentrations in the same session, balanced across BSA/non-BSA, PEG/COOH and
magnet/control, so the contrast carries no between-day confound and rests on
12 magnet series per concentration rather than 3.

Asymmetry cannot be used: back injection puts the gap at the block's edge, so
there is no far side. The analogue is PENETRATION DEPTH measured from the
gap/block boundary into the block, using the same definitions as the Aug 23
navigation metrics:

    front    — the deepest point still carrying 20% of the profile's peak
               darkness: how far the leading edge of the NP mass reached
    centroid — mean depth of the excess NP darkness

FRONT IS THE PRIMARY METRIC, decided from the shape of the data, not its
result. Back injection leaves a large reservoir pile against the gap, and a
mass-weighted statistic (centroid, d90) is dominated by that pile: a magnet
arm that has drawn nanoparticles out of the pile into a long tail scores
LOWER than a control that kept everything heaped at the boundary, even though
its particles travelled further. Centroid is reported alongside so the
disagreement stays visible, but it answers a different question.

The last three columns are dropped before measuring: the right rim of the
holder intrudes there, a known artefact of this rig that the Aug 27 pipeline
already fights with its lit-rim crop.

Geometry is the validated Aug 27 pipeline (SeriesGeom: per-series reference
from the magnet-attached t=0 frame, ECC registration, rim anchoring, sibling-
median repair), so the mm scale is anchored at the boundary in every frame.

Each magnet series is expressed as its depth MINUS the mean control depth of
its own (agarose, BSA, coating) cell at the same timepoint. That pairing
removes anything the gel does on its own — swelling, settling, diffusion,
cloudiness — leaving the magnet-driven part.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import make_block_atlas as A

FLOOR_PCT, FRONT_FRAC, EDGE_DROP = 15, 0.20, 3


def profile(D, mm):
    prof = D.mean(axis=0)[:-EDGE_DROP]
    d = mm[:-EDGE_DROP]
    ex = np.clip(prof - np.percentile(prof, FLOOR_PCT), 0, None)
    keep = d >= 0.0
    return ex[keep], d[keep]


def depths(D, mm):
    """(front, centroid) of the excess profile, in mm from the boundary."""
    ex, d = profile(D, mm)
    m = float(ex.sum())
    if m <= 0 or len(d) < 5:
        return np.nan, np.nan
    over = np.where(ex >= FRONT_FRAC * ex.max())[0]
    front = float(d[over[-1]]) if len(over) else np.nan
    return front, float((d * ex).sum() / m)


def build():
    m = pd.read_csv(Path(__file__).parent / "photos_final.csv", dtype={"agarose_f": str})
    wr = pd.read_csv(Path(__file__).parent / "warp_report.csv")
    bad = set(wr[~wr.refined]["idx"])
    load = lambda i: cv2.imread(str(A.S / "prep" / "warped" / f"{int(i):04d}.jpg"))

    def pick(g, t, need_mag=False):
        sub = g[g.timepoint_hr == t]
        if need_mag:
            sub = sub[sub.magnet_f == "present"]
        sub = sub[~sub.idx.isin(bad)]
        return int(sub.iloc[0].idx) if len(sub) else None

    rows = []
    for (ag, bsa, co), cg in m.groupby(["agarose_f", "bsa_f", "coating_f"]):
        # per-series geometry, exactly as the atlas builds it
        geoms = {}
        for (arm, rep), g in cg.groupby(["arm", "tally_f"]):
            chosen = None
            for tref, need in [(0.0, arm == "magnet"), (1.0, False)]:
                i0 = pick(g, tref, need_mag=need)
                if i0 is None:
                    continue
                geo = A.SeriesGeom(load(i0))
                if geo.ok:
                    chosen = geo
                    break
                chosen = chosen or geo
            if chosen is not None:
                geoms[(arm, rep)] = chosen
        good = [x for x in geoms.values() if x.ok]
        if good:
            med = lambda a: int(np.median(a))
            gX1, gY0, gY1 = med([x.X1 for x in good]), med([x.Y0 for x in good]), med([x.Y1 for x in good])
            gGL0, gGB = med([x.GL0 for x in good]), med([x.GB for x in good])
            for x in geoms.values():
                rep = not x.ok
                if abs(x.X1 - gX1) > 25: x.X1, rep = gX1, True
                if abs(x.Y0 - gY0) > 25: x.Y0, rep = gY0, True
                if abs(x.Y1 - gY1) > 25: x.Y1, rep = gY1, True
                if not x.ok:
                    x.GL0, x.GB = gGL0, gGB
                if rep:
                    x.mm_per_px = 10.0 / ((x.X1 - x.GL0) - x.GB)
                    x.left_mm = -x.GB * x.mm_per_px

        for (arm, rep), g in cg.groupby(["arm", "tally_f"]):
            geo = geoms.get((arm, rep))
            if geo is None:
                continue
            # mm of each of the 44 field columns, 0 = gap/block boundary
            px = geo.GL0 + (np.arange(44) + .5) * ((geo.X1 - 8 - geo.GL0) / 44.0)
            mm = (px - geo.GL0 - geo.GB) * geo.mm_per_px
            for t in sorted(cg.timepoint_hr.unique()):
                i = pick(g, t, need_mag=(arm == "magnet" and t == 0.0))
                if i is None:
                    continue
                img = load(i)
                if img is None:
                    continue
                fr, c = depths(A.process_abs(geo.aligned_field(img)), mm)
                rows.append(dict(agarose=ag, bsa=bsa, coating=co, arm=arm, rep=rep,
                                 series=f"{ag}_{bsa}_{co}_{arm}_r{rep}", t=t,
                                 front=fr, centroid=c))
        print(f"  {ag} {bsa:8s} {co:5s} done", flush=True)

    df = pd.DataFrame(rows)
    # pair each magnet series against its own cell's control mean
    ctl = (df[df.arm == "control"].groupby(["agarose", "bsa", "coating", "t"])
           [["front", "centroid"]].mean().rename(columns=lambda c: c + "_ctl"))
    df = df.merge(ctl, on=["agarose", "bsa", "coating", "t"], how="left")
    df["front_net"] = df.front - df.front_ctl
    df["centroid_net"] = df.centroid - df.centroid_ctl
    df.to_csv(Path(__file__).parent / "back_depth_metrics.csv", index=False)
    print(f"\nwrote back_depth_metrics.csv  n={len(df)}")
    return df


if __name__ == "__main__":
    build()
