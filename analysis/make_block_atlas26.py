"""Block-heatmap atlas for the Aug 26 run (0.6% agarose, centre injection).

Design: PEG vs COOH x BSA vs non-BSA, large magnet only, 3 repeats each,
plus one no-magnet control per condition. 6 h, photographed every 30 min.

Geometry: the block quad comes from the four red corner dots Steven drew on
the holder (see block26.py) - far more reliable here than colour, because the
magnet's warm reflections and the orange sticker both overlap the gel's colour
range. The camera framing is consistent (magnet always to the right, centre
gap vertical), so no rotation is applied; verified on every frame by checking
the gap resolves vertically.

As on Aug 23, and per Steven's instruction, the heatmap spans the WHOLE block
with x=0 at the left edge (where a back-injection gap would be), so the centre
gap appears near 5 mm.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from block26 import block_quad, order_corners

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad/aug26")
WARP = 480
MARGIN = 38  # ~0.8 mm trimmed per side: the gel separates from the holder late on
EXTENT_MM = (MARGIN / WARP * 10.0, 10.0 - MARGIN / WARP * 10.0)
STAGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
VMAX = 55.0


def warped(img, quad):
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(order_corners(quad), dst), (WARP, WARP))


def field(img, quad):
    w = warped(img, quad)
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    D = (255.0 - L)[MARGIN:WARP - MARGIN, MARGIN:WARP - MARGIN]
    h, wd = D.shape
    hb, wb = h // 11, wd // 11
    return np.median(D[: hb * 11, : wb * 11].reshape(hb, 11, wb, 11), axis=(1, 3))


def process_abs(Db):
    h, w = Db.shape
    floor = np.percentile(Db[3:h - 3, 3:w - 3], 15)
    return gaussian_filter(np.clip(Db - floor, 0, None), 1.0)


def main():
    m = pd.read_csv(Path(__file__).parent / "photos26_final.csv", parse_dates=["capture_time"])
    outdir = S / "panels"
    outdir.mkdir(exist_ok=True)
    load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))

    def pick(g, t):
        sub = g[g.timepoint_hr == t]
        return int(sub.iloc[0].idx) if len(sub) else None

    for bsa, bg in m.groupby("bsa"):
        # One shared 0-VMAX scale across BSA, non-BSA and the other experiment
        # days, so panels are directly comparable. BSA gel is intrinsically
        # cloudier (median raw darkness ~129 vs ~106 for non-BSA), so its controls
        # saturate at this ceiling - that is the cloudiness, not transport, and
        # the asymmetry metric is the readout to trust for BSA.
        vmax = VMAX
        rows = [("PEG", False, "PEG + magnet\n(mean r1-r3)"), ("PEG", True, "PEG control\n(no magnet)"),
                ("COOH", False, "COOH + magnet\n(mean r1-r3)"), ("COOH", True, "COOH control\n(no magnet)")]
        fig, axes = plt.subplots(len(rows), len(STAGES), figsize=(len(STAGES) * 1.72, len(rows) * 1.72))
        for ri, (coat, is_ctrl, label) in enumerate(rows):
            sel = bg[(bg.coating == coat) & (bg.control == is_ctrl)]
            for ci, t in enumerate(STAGES):
                ax = axes[ri, ci]
                maps = []
                for _, sg in sel.groupby("series"):
                    i = pick(sg, t)
                    if i is None:
                        continue
                    img = load(i)
                    q = block_quad(img)
                    if q is None:
                        continue
                    maps.append(process_abs(field(img, q)))
                if not maps:
                    ax.text(.5, .5, "no frame", ha="center", va="center", fontsize=7,
                            color="gray", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([]); continue
                shp = maps[0].shape
                maps = [x for x in maps if x.shape == shp]
                ax.imshow(np.mean(maps, axis=0), cmap="inferno", vmin=0, vmax=vmax,
                          extent=[EXTENT_MM[0], EXTENT_MM[1], EXTENT_MM[1], EXTENT_MM[0]],
                          aspect="equal", interpolation="bilinear")
                ax.set_yticks([])
                n_exp = sel.series.nunique()
                if len(maps) < n_exp:
                    ax.text(0.03, 0.06, f"n={len(maps)}", transform=ax.transAxes, fontsize=6, color="white")
                if ri == len(rows) - 1:
                    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(["1", "5", "9"], fontsize=7)
                    if ci == len(STAGES) // 2:
                        ax.set_xlabel("mm across block  (injection gap ~centre, magnet side →)",
                                      fontsize=8.5, labelpad=4)
                else:
                    ax.set_xticks([])
                for sp in ax.spines.values():
                    sp.set_color("#999"); sp.set_linewidth(.6)
                if ri == 0:
                    ax.set_title(f"{t:g} h", fontsize=10)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
        fig.suptitle(f"26 Aug · 0.6% agarose · {bsa} · centre injection · large magnet",
                     fontsize=11.5, y=0.99)
        plt.tight_layout(rect=[0.02, 0.07, 1, 0.95])
        cax = fig.add_axes([0.34, 0.045, 0.32, 0.014])
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, round(vmax/2), round(vmax)]); cb.ax.tick_params(labelsize=6, pad=1)
        cb.set_label("NP darkness above this block's gel floor (L*) — same scale as the other run days", fontsize=6.5, labelpad=2)
        plt.savefig(outdir / f"{bsa}.png", dpi=118)
        plt.close(fig)
        print("done", bsa, flush=True)


if __name__ == "__main__":
    main()
