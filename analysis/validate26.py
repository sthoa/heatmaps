"""Photo-vs-heatmap validation panel for the Aug 26 run.

Renders the warped block photo above its heatmap for a few representative
series, so the geometry and the NP signal can be checked against the raw
images the way the Aug 23 and Aug 27 panels were.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lab_units import L_SCALE
import pandas as pd

from block26 import block_quad
from make_block_atlas26 import EXTENT_MM, STAGES, S, field, process_abs, warped

SERIES = [
    ("PEG_BSA_r1", "PEG · BSA · magnet r1"),
    ("COOH_non-BSA_r1", "COOH · non-BSA · magnet r1"),
    ("PEG_BSA_control", "PEG · BSA · control"),
]


def main():
    m = pd.read_csv(Path(__file__).parent / "photos26_final.csv", parse_dates=["capture_time"])
    load = lambda i: cv2.imread(str(S / "frames" / f"{int(i):04d}.jpg"))
    fig, axes = plt.subplots(len(SERIES) * 2, len(STAGES),
                             figsize=(len(STAGES) * 1.85, len(SERIES) * 2 * 1.62))
    for si, (ser, label) in enumerate(SERIES):
        g = m[m.series == ser]
        for ci, t in enumerate(STAGES):
            sub = g[g.timepoint_hr == t]
            a0, a1 = axes[si * 2, ci], axes[si * 2 + 1, ci]
            if len(sub) == 0:
                for ax in (a0, a1):
                    ax.text(.5, .5, "—", ha="center", va="center", color="gray", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                continue
            img = load(sub.iloc[0].idx)
            q = block_quad(img)
            w = warped(img, q)
            D = process_abs(field(img, q))
            a0.imshow(cv2.cvtColor(cv2.resize(w, (220, 220)), cv2.COLOR_BGR2RGB),
                      extent=[0, 10, 10, 0], aspect="equal")
            a1.imshow(D, cmap="inferno", vmin=0, vmax=A26.VMAX,
                      extent=[EXTENT_MM[0], EXTENT_MM[1], EXTENT_MM[1], EXTENT_MM[0]],
                      aspect="equal", interpolation="bilinear")
            if si == 0:
                a0.set_title(f"{t:g} h", fontsize=10)
            for ax in (a0, a1):
                ax.set_yticks([]); ax.set_xlim(0, 10)
                for sp in ax.spines.values():
                    sp.set_color("#999"); sp.set_linewidth(.6)
            a0.set_xticks([])
            if si == len(SERIES) - 1:
                a1.set_xticks([0, 5, 10]); a1.set_xticklabels(["0", "5", "10"], fontsize=7)
            else:
                a1.set_xticks([])
            if ci == 0:
                a0.set_ylabel(f"{label}\nphoto", fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
                a1.set_ylabel("heatmap", fontsize=8, rotation=0, ha="right", va="center", labelpad=8)
    axes[-1, len(STAGES) // 2].set_xlabel("mm across block  (gap ~centre, magnet side →)", fontsize=9)
    fig.suptitle("26 Aug validation — warped block photo vs heatmap", fontsize=11.5, y=0.995)
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.97])
    plt.savefig(S / "validation26.png", dpi=118)
    print("saved validation26.png")


if __name__ == "__main__":
    main()
