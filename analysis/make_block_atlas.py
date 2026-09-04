"""Block-heatmap atlas — the recipe validated frame-by-frame with Steven on
0.4 non-BSA COOH magnet r2:

  1. ECC affine registration of each warped frame onto its series' t=0
     (on Sobel gradient magnitude — insensitive to lighting).
  2. Left anchor = the GAP'S LEFT EDGE (white rim -> gap transition), detected
     with one rule in every frame; each frame translated so it matches t0's.
  3. One crop per series from t0: starts at the gap's left edge; right/top/
     bottom from a bounded inward scan (white/black/green rules only — never
     a mid-gray rule, it matches pale gel).
  4. Gap/block boundary measured once, from the t0 reservoir band (50%%
     threshold); drawn as white dashes on black casing; held fixed for all
     later frames (agarose shrinkage is shown, not compensated).
     mm scale: boundary -> block right face = 10 mm.
  5. Display: absolute NP darkness minus the frame's own 15th-percentile
     interior floor (exposure cancels), median-binned to 44x40, smoothed.

Panels: one per condition; rows = magnet mean-of-repeats, control
mean-of-repeats; columns = time stages. One shared color scale.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lab_units import L_SCALE
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")
STAGES = [1.0, 3.0, 6.0, 12.0, 21.5]
VMAX = 55.0 / L_SCALE   # 21.6 L*
PXMM = lambda mm: int(round((mm + 1) / 11 * 480))
BASE = dict(x0=PXMM(-1.0), x1=PXMM(9.5), y0=46, y1=434)


def gradmag(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    return cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (5, 5), 0)


def whitefrac_cols(img, y0, y1):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    white = ((lab[..., 0] > 170) & (hsv[..., 1] < 80)).astype(np.float32)
    return gaussian_filter1d(white[y0:y1, :].mean(axis=0), 3)


def gap_left(colf, lo=0, hi=140):
    seg = colf[lo:hi]
    inside = np.where(seg > 0.55)[0]
    if len(inside) == 0:
        return None
    i = inside[-1]
    while i < len(seg) - 1 and seg[i] > 0.30:
        i += 1
    return lo + i


def crop_from_t0(img0):
    """Bounded inward scan from the baseline (the version validated on r2):
    white / neutral-black / green rules only, never mid-gray (matches pale gel).
    Outlier crops are repaired from sibling medians in main()."""
    lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, b, sat = lab[..., 0], lab[..., 2] - 128, hsv[..., 1]
    ng = (((L > 170) & (sat < 80)) | ((L < 75) & (np.abs(b) < 12)) | ((sat > 90) & (b < -5))).astype(np.float32)
    colf = ng.mean(axis=0)
    rowf = ng.mean(axis=1)

    def inner(frac, base, d, win):
        i = base
        lim = base + d * win
        while i != lim and 0 <= i < len(frac) and frac[i] > 0.30:
            i += d
        return i

    x1 = inner(colf, BASE["x1"], -1, 42) - 3
    y0 = inner(rowf, BASE["y0"], +1, 42) + 3
    y1 = inner(rowf, BASE["y1"], -1, 42) - 3
    return x1, y0, y1


def detect_gap_run(img, Y0, Y1):
    """The gap is the dark column-run immediately followed by a LONG bright
    region (the gel block). Fabric fails this (only a narrow rim follows it);
    no rim detection needed (rims can be shadowed, pale gel reads as white)."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    meanL = gaussian_filter1d(L[Y0 + 30:Y1 - 30, :260].mean(axis=0), 3)
    dark = meanL < 135
    i = 0
    n = len(dark)
    while i < n:
        if dark[i]:
            start = i
            while i < n and dark[i]:
                i += 1
            R = i  # run end (exclusive) = candidate gap/block boundary
            if start == 0:
                continue  # touches the canvas edge: fabric/border, keep scanning
            nxt = meanL[R + 3:R + 63]
            if len(nxt) >= 45 and np.mean(nxt) > 150 and np.mean(nxt > 140) > 0.65:
                gl = max(start, R - 50)
                if gl >= 8 and 8 <= R - gl <= 60:
                    return gl, R - gl
        i += 1
    return None, None


def rim_bright_start(Lm, lo, hi):
    """first x in [lo,hi) where the next 12 columns average L>200 (lit rim);
    gel tops out ~195, shadow/magnet ~130 — unambiguous separator."""
    lo = max(0, lo)
    for x in range(lo, hi):
        if np.mean(Lm[x:x + 12]) > 200:
            return x
    return None


class SeriesGeom:
    """Per-series geometry derived from its t=0 frame."""

    def __init__(self, img0):
        self.ref_gm = gradmag(img0)
        self.X1, self.Y0, self.Y1 = crop_from_t0(img0)
        # gap detection: same rule as per-frame (dark run followed by the
        # long bright block) — validated against shadowed rims and pale gels
        gl, wdt = detect_gap_run(img0, self.Y0, self.Y1)
        self.ok = gl is not None and gl <= 210
        if self.ok:
            self.GL0, self.GB = gl, wdt
        if not self.ok:
            self.GL0, self.GB = BASE["x0"] + 108, 40
        if (self.X1 - self.GL0) < 220 or (self.Y1 - self.Y0) < 220:
            self.ok = False
            self.GL0, self.GB = BASE["x0"] + 108, 40
            self.X1, self.Y0, self.Y1 = BASE["x1"], BASE["y0"], BASE["y1"]
        self.mm_per_px = 10.0 / ((self.X1 - self.GL0) - self.GB)
        self.left_mm = -self.GB * self.mm_per_px
        # left-region whiteness profile for per-frame correlation anchoring
        wf0 = whitefrac_cols(img0, self.Y0, self.Y1)
        self.anchor_prof = wf0[:240] - wf0[:240].mean()
        # reference: the white rim's inner edge (steepest white->dark fall near
        # the gap) — the print itself, physically fixed in every frame
        lo, hi = max(1, self.GL0 - 15), self.GL0 + 15
        dwf = np.diff(wf0[lo:hi])
        self.rimedge0 = lo + int(np.argmin(dwf))
        # right rim inner edge at t0: steepest brightness RISE (gel -> lit print)
        L0m = gaussian_filter1d(
            cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)[self.Y0 + 30:self.Y1 - 30, :].mean(axis=0), 3)
        self.rimR0 = rim_bright_start(L0m, self.X1 - 55, min(466, self.X1 + 12))
        if self.rimR0 is not None and self.rimR0 - 2 < self.X1:
            # crop's right edge = measured rim edge (the color-scan guess overshoots)
            self.X1 = self.rimR0 - 2
            self.mm_per_px = 10.0 / ((self.X1 - self.GL0) - self.GB)
            self.left_mm = -self.GB * self.mm_per_px

    def aligned_field(self, img):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                self.ref_gm, gradmag(img), warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
            img = cv2.warpAffine(img, warp, (480, 480), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            pass
        # residual-drift anchor, primary: the rim's INNER EDGE (the print,
        # physically fixed and visible in every frame) — steepest white->dark
        # fall in a wide window. Gap-run and profile correlation are backups
        # (the gap run merges with the NP mass from ~3 h in fast series).
        wf0 = whitefrac_cols(img, self.Y0, self.Y1)
        lo0, hi0 = max(1, self.rimedge0 - 40), min(258, self.rimedge0 + 41)
        dwf0 = np.diff(wf0[lo0:hi0])
        shift = None
        if len(dwf0) and dwf0.min() < -0.04:
            shift = int(np.clip(self.rimedge0 - (lo0 + int(np.argmin(dwf0))), -40, 40))
        if shift is None:
            gl, wdt = detect_gap_run(img, self.Y0, self.Y1)
            if gl is not None and 8 <= wdt <= 45 and 40 <= gl <= 200:
                shift = int(np.clip(self.GL0 - gl, -45, 45))
        if shift is None:
            p = wf0[:240]
            p = p - p.mean()
            best_s, best_c = 0, -1e9
            for s in range(-45, 46):
                a = self.anchor_prof[max(0, s):240 + min(0, s)]
                b = p[max(0, -s):240 - max(0, s)]
                n = min(len(a), len(b))
                c = float((a[:n] * b[:n]).sum())
                if c > best_c:
                    best_c, best_s = c, s
            if abs(best_s) < 43:
                shift = best_s
        if shift:
            Mx = np.float32([[1, 0, shift], [0, 1, 0]])
            img = cv2.warpAffine(img, Mx, (480, 480), flags=cv2.INTER_LINEAR)
        # right rim: the case is rigid, so after left-anchoring the right rim
        # should sit where t0's did; residual scale mismatch is corrected by a
        # 1D stretch pinned at the left rim edge
        if self.rimR0 is not None:
            Lm = gaussian_filter1d(
                cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)[self.Y0 + 30:self.Y1 - 30, :].mean(axis=0), 3)
            rimR = rim_bright_start(Lm, self.rimR0 - 70, min(474, self.rimR0 + 75))
            if rimR is not None:
                if abs(rimR - self.rimR0) > 2:
                    sx = (self.rimR0 - self.rimedge0) / max(rimR - self.rimedge0, 1)
                    if 0.82 < sx < 1.2:
                        Mx = np.float32([[sx, 0, self.rimedge0 - sx * self.rimedge0], [0, 1, 0]])
                        img = cv2.warpAffine(img, Mx, (480, 480), flags=cv2.INTER_LINEAR)
        # fine stage: align the rim's inner edge (steepest white->dark fall)
        # onto t0's — immune to the gap draining/narrowing over time
        wf = whitefrac_cols(img, self.Y0, self.Y1)
        lo, hi = max(1, self.rimedge0 - 14), self.rimedge0 + 15
        dwf = np.diff(wf[lo:hi])
        if len(dwf) and dwf.min() < -0.04:
            fine = int(np.clip(self.rimedge0 - (lo + int(np.argmin(dwf))), -14, 14))
            if fine:
                Mx = np.float32([[1, 0, fine], [0, 1, 0]])
                img = cv2.warpAffine(img, Mx, (480, 480), flags=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        D = ((255.0 - lab[..., 0]) / L_SCALE)[self.Y0:self.Y1, self.GL0:self.X1 - 8]
        return cv2.resize(D, (44, 40), interpolation=cv2.INTER_AREA)


def process_abs(Db):
    h, w = Db.shape
    interior = Db[4:h - 4, int(w * 0.3):int(w * 0.9)]
    floor = np.percentile(interior, 15)
    return gaussian_filter(np.clip(Db - floor, 0, None), 1.0)


def draw_boundary(ax):
    ax.axvline(0, color="black", lw=2.6, alpha=0.9)
    ax.axvline(0, color="white", lw=1.2, ls=(0, (3, 3)))


def condition_geometry(cg, pick, load):
    """Per-series geometry for one condition (t0 reference, sibling-median
    repair). Returns (geoms, ext) exactly as the atlas uses them."""
    # per-series geometry: t0 reference; if its geometry fails, try the 1h
    # frame as reference+geometry; sibling-median repair as last resort
    geoms = {}
    for (arm, rep), g in cg.groupby(["arm", "tally_f"]):
        chosen = None
        for tref, need_mag in [(0.0, arm == "magnet"), (1.0, False)]:
            i0 = pick(g, tref, magnet_needed=need_mag)
            if i0 is None:
                continue
            geo = SeriesGeom(load(i0))
            if geo.ok:
                chosen = geo
                break
            if chosen is None:
                chosen = geo
        if chosen is not None:
            geoms[(arm, rep)] = chosen
    good = [x for x in geoms.values() if x.ok]
    if good:
        med = lambda a: int(np.median(a))
        gGL0 = med([x.GL0 for x in good]); gX1 = med([x.X1 for x in good])
        gY0 = med([x.Y0 for x in good]); gY1 = med([x.Y1 for x in good])
        gGB = med([x.GB for x in good])
        for x in geoms.values():
            repair = not x.ok
            # per-edge outlier repair even when the gap was detected
            if abs(x.X1 - gX1) > 25: x.X1 = gX1; repair = True
            if abs(x.Y0 - gY0) > 25: x.Y0 = gY0; repair = True
            if abs(x.Y1 - gY1) > 25: x.Y1 = gY1; repair = True
            if not x.ok:
                x.GL0, x.GB = gGL0, gGB
            if repair:
                x.mm_per_px = 10.0 / ((x.X1 - x.GL0) - x.GB)
                x.left_mm = -x.GB * x.mm_per_px
    exts = [geoms[k].left_mm for k in geoms]
    left_mm = float(np.median(exts))
    ext = [left_mm, 10, 10, 0]
    return geoms, ext


def condition_maps(cg, geoms, pick, load, stages=None):
    """Mean processed field per (arm, stage) for one condition -> {(arm, t): (D|None, n)}.
    Shared by the atlas and by any composite figure that wants the same maps."""
    out = {}
    for arm in ["magnet", "control"]:
        for t in (stages or STAGES):
            maps = []
            for rep in [1, 2, 3]:
                if (arm, rep) not in geoms:
                    continue
                g = cg[(cg.arm == arm) & (cg.tally_f == rep)]
                # at t=0 a magnet block has a pre- and a post-magnet photo; use the latter
                it = pick(g, t, arm == "magnet" and t == 0.0)
                if it is None:
                    continue
                maps.append(process_abs(geoms[(arm, rep)].aligned_field(load(it))))
            out[(arm, t)] = (np.mean(maps, axis=0) if maps else None, len(maps))
    return out


def main():
    m = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
    wr = pd.read_csv("warp_report.csv")
    badset = set(wr[~wr.refined]["idx"])
    # frames whose warped image is absent (well detection failed on a regenerated run) are skipped too
    badset |= {int(i) for i in m.idx if not (S / "prep" / "warped" / f"{int(i):04d}.jpg").exists()}
    outdir = S / "heatpanels_block"
    outdir.mkdir(exist_ok=True)

    def pick(g, t, magnet_needed=False):
        sub = g[(g.timepoint_hr == t)]
        if magnet_needed:
            sub = sub[sub.magnet_f == "present"]
        sub = sub[~sub.idx.isin(badset)]
        return int(sub.iloc[0].idx) if len(sub) else None

    load = lambda i: cv2.imread(str(S / "prep" / "warped" / f"{i:04d}.jpg"))

    for (ag, bsa, co), cg in m.groupby(["agarose_f", "bsa_f", "coating_f"]):
        geoms, ext = condition_geometry(cg, pick, load)

        fig, axes = plt.subplots(2, len(STAGES), figsize=(len(STAGES) * 1.95, 2 * 1.9))
        for ri, arm in enumerate(["magnet", "control"]):
            for ci, t in enumerate(STAGES):
                ax = axes[ri, ci]
                maps = []
                for rep in [1, 2, 3]:
                    key = (arm, rep)
                    if key not in geoms:
                        continue
                    g = cg[(cg.arm == arm) & (cg.tally_f == rep)]
                    it = pick(g, t)
                    if it is None:
                        continue
                    geo = geoms[key]
                    Db = geo.aligned_field(load(it))
                    maps.append(process_abs(Db))
                if not maps:
                    ax.text(.5, .5, "no frames", ha="center", va="center", fontsize=7, color="gray", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                D = np.mean(maps, axis=0)
                ax.imshow(D, cmap="inferno", vmin=0, vmax=VMAX, extent=ext, aspect="equal", interpolation="bilinear")
                draw_boundary(ax)
                ax.set_yticks([])
                if len(maps) < 3:
                    ax.text(0.03, 0.05, f"n={len(maps)}", transform=ax.transAxes, fontsize=6, color="white")
                if ri == 1:
                    ax.set_xticks([0, 5, 10]); ax.set_xticklabels(["0", "5", "10"], fontsize=7)
                    if ci == 2:
                        ax.set_xlabel("Distance from gap/block boundary (mm)", fontsize=8.5)
                else:
                    ax.set_xticks([])
                for s in ax.spines.values():
                    s.set_color("#999"); s.set_linewidth(.6)
                if ri == 0:
                    ax.set_title(f"{t:g} h", fontsize=10)
                if ci == 0:
                    ax.set_ylabel(arm.capitalize(), fontsize=9.5, rotation=0, ha="right", va="center", labelpad=8)
        fig.suptitle(f"{ag}% agarose · {bsa} · {co}   (mean of 3 repeats)", fontsize=11.5, y=0.99)
        plt.tight_layout(rect=[0.015, 0.06, 1, 0.94])
        cax = fig.add_axes([0.32, 0.045, 0.36, 0.02])
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, VMAX))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 10, 20]); cb.ax.tick_params(labelsize=6, pad=1)
        cb.set_label("NP darkness above gel floor (L*)", fontsize=6.5)
        plt.savefig(outdir / f"{ag}_{bsa}_{co}.png", dpi=118)
        plt.close(fig)
        print("done", ag, bsa, co, flush=True)


if __name__ == "__main__":
    main()
