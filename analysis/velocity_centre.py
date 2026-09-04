"""Front position and velocity of magnet-side transport, centre-injection runs, non-BSA.

The asymmetry metric is an intensity difference and has no speed in it. The same
frames do contain a position: how far the excess darkness has advanced from the
gap edge toward the magnet. Tracking that FRONT over time gives a velocity in
mm/h that can be compared with the measured field gradients.

Per frame (same field extraction as nav_metrics_compare / cross_day_compare):
  * column profile of darkness (trimmed mean over rows), minus its 15th
    percentile -> excess darkness per column, 0.0208 mm per column
  * the series' own t=0 (magnet-attached) profile is SUBTRACTED first. Every
    block carries a static hump of ~10-15 L* beside the gap that never moves,
    and the magnet arms a dark strip at the far edge from t=0 (the magnet's
    shadow); both would trip an absolute threshold. What moves is the change
    since t=0.
  * gap edges at 35 % of the gap peak, measured on the t=0 frame and held fixed
    for the series (the edge cannot be re-found once the magnet side floods)
  * magnet side = columns beyond the right gap edge:
      front_right / front_left   outer end of the contiguous run of change >=
                FRONT_THR that starts next to the gap (see side_front); 0 if none
      front     = front_right - front_left: the DIRECTIONAL front. Symmetric
                spreading from the reservoir (diffusion, wicking) moves both
                sides equally and cancels, as it does in the asymmetry metric.
      centroid  excess-weighted mean distance
      d90       distance containing 90 % of the excess

Per series, the primary velocity is the PEAK ADVANCE RATE over the run: the
furthest position the (3-point median smoothed) directional front reached,
divided by the time it took to get there. It stops the clock when the front
stops -- at the wall for the fast blocks -- and is not pulled down by the late
symmetric spreading that makes a straight-line fit through six hours read
negative for the weakest plumes. Regression slopes over 0-3 h and 0-6 h
(each truncated at wall arrival) are kept for reference.
The block edge is only ~4.3 mm from the gap and the large-magnet front reaches
it by ~3 h; fitting through the parked phase would return the wall distance
divided by the run time (0.7 mm/h for every fast block) rather than a velocity.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lab_units import L_SCALE
import pandas as pd
from scipy import stats

import make_block_atlas23 as A23
import make_block_atlas26 as A26
from block26 import block_quad as quad26

FRONT_THR = 15.0 / L_SCALE        # L* of change since t=0 (5.9): the lowest level at which every control block still reads zero
LEAD_THR = 15.0 / L_SCALE         # kept equal to FRONT_THR; retained so older columns keep their names
EDGE_SKIP = 0.6         # mm beyond the gap edge ignored when locating a front (reservoir-edge drift reaches ~0.5 mm)
START_WIN = 1.2         # mm: a plume must begin within this distance of the gap edge to count
BRIDGE = 0.5            # mm: dips below threshold shorter than this do not end the plume
THR_SENS = tuple(t / L_SCALE for t in (15.0, 20.0, 25.0, 30.0, 35.0))   # thresholds for the sensitivity table
SMOOTH_PX = 9           # moving average over ~0.2 mm before walking the front
T_RISE = 3.0            # h; short window (the rising phase)
T_FULL = 6.0            # h; whole run. Both windows stop at the front's first arrival at the wall
WALL_MARGIN = 0.4       # mm; drop points once the front is this close to the block edge
MM_PER_PX = (A23.EXTENT_MM[1] - A23.EXTENT_MM[0]) / (A23.WARP - 2 * A23.MARGIN)
OUT = Path(__file__).parent / "outputs"

C = {("0.4%", "large"): "#C0392B", ("0.4%", "small"): "#E67E22", ("0.6%", "large"): "#1F6FB2"}


def raw_field_23(img, quad, date):
    w = A23.warped_by_date(img, quad, date)
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    return ((255.0 - L) / L_SCALE)[A23.MARGIN:A23.WARP - A23.MARGIN, A23.MARGIN:A23.WARP - A23.MARGIN]


def raw_field_26(img, quad):
    w = A26.warped(img, quad)
    L = cv2.cvtColor(w, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    return ((255.0 - L) / L_SCALE)[A26.MARGIN:A26.WARP - A26.MARGIN, A26.MARGIN:A26.WARP - A26.MARGIN]


def excess_profile(D):
    p = np.sort(D, axis=0); n = p.shape[0]
    prof = p[int(.15 * n):int(.85 * n)].mean(axis=0)
    return np.clip(prof - np.percentile(prof, 15), 0, None)


def gap_edges(e, frac=0.35):
    pk = int(np.argmax(e[int(len(e) * .3):int(len(e) * .7)])) + int(len(e) * .3)
    thr = frac * e[pk]
    r = pk
    while r + 1 < len(e) and e[r + 1] > thr: r += 1
    l = pk
    while l - 1 >= 0 and e[l - 1] > thr: l -= 1
    return l, r


def side_front(seg, thr=None):
    """Front (mm) of the contiguous plume of change since t=0 on one side of the gap.

    A plume driven from the reservoir is connected to it, so the front is the
    outer end of the run of columns >= thr that STARTS within START_WIN of the
    gap edge (ignoring the first EDGE_SKIP mm, where reservoir-edge drift
    produces spurious changes), bridging dips shorter than BRIDGE (the columns
    next to the reservoir deplete late in a run while the plume beyond persists).
    An isolated band far from the gap -- the block edge darkening as the gel
    pulls from the wall -- is not connected and does not count. No run starting
    near the gap means nothing has arrived: front 0.
    """
    thr = FRONT_THR if thr is None else thr
    if seg.size < 5:
        return np.nan
    d = (np.arange(seg.size) + 1) * MM_PER_PX
    sm = np.convolve(seg, np.ones(SMOOTH_PX) / SMOOTH_PX, mode="same")
    above = sm >= thr
    starts = np.where(above & (d > EDGE_SKIP) & (d <= START_WIN))[0]
    if not len(starts):
        return 0.0
    last, gap = int(starts[0]), 0
    for k in range(int(starts[0]) + 1, len(sm)):
        if above[k]:
            last, gap = k, 0
        else:
            gap += 1
            if gap * MM_PER_PX > BRIDGE:
                break
    return float(d[last])


def fronts(e, e0, l, r):
    """Magnet-side and far-side fronts from the change since t=0, beyond the fixed gap edges.

    Symmetric spreading from the reservoir (diffusion, wicking) advances both
    sides equally; magnet transport advances only the magnet side. The
    directional front is their difference, zero for symmetric spreading, as the
    asymmetry metric is zero for symmetric darkening.
    """
    delta = np.clip(e - e0, 0, None)
    right, left = delta[r + 1:], delta[:l][::-1]
    if right.size < 5 or left.size < 5:
        return dict(front_right=np.nan, front_left=np.nan, front=np.nan, centroid=np.nan, d90=np.nan, wall_mm=np.nan)
    fr, fl = side_front(right), side_front(left)
    d = (np.arange(right.size) + 1) * MM_PER_PX
    m = float(right.sum())
    sens = {f"front_thr{int(round(t * L_SCALE))}": side_front(right, t) - side_front(left, t) for t in THR_SENS}
    return dict(front_right=fr, front_left=fl, front=fr - fl, wall_mm=float(d[-1]), **sens,
                centroid=float((d * right).sum() / m) if m > 0 else np.nan,
                d90=float(np.interp(0.9, np.cumsum(right) / m, d)) if m > 0 else np.nan)


def frames():
    rows = []
    m = pd.read_csv(Path(__file__).parent / "photos23_final.csv", parse_dates=["capture_time"])
    m["date"] = m.capture_time.dt.date
    Q = A23.repaired_quads()
    for r in m.itertuples():
        img = cv2.imread(str(A23.S / "frames" / f"{int(r.idx):04d}.jpg"))
        q = Q.get(int(r.idx))
        if img is None or q is None: continue
        rows.append(dict(day="Day 1", agarose="0.4%", arm=r.arm, coating=r.coating, series=r.series,
                         t=r.timepoint_hr, magnet_on=(r.v_magnet == "present"), D=raw_field_23(img, q, r.date)))
    m = pd.read_csv(Path(__file__).parent / "photos26_final.csv")
    m = m[m.bsa == "non-BSA"]
    for r in m.itertuples():
        img = cv2.imread(str(A26.S / "frames" / f"{int(r.idx):04d}.jpg"))
        q = quad26(img) if img is not None else None
        if q is None: continue
        rows.append(dict(day="Day 2", agarose="0.6%", arm="control" if r.control else "large",
                         coating=r.coating, series=r.series, t=r.timepoint_hr, magnet_on=True, D=raw_field_26(img, q)))
    return rows


def main():
    recs = []
    allf = frames()
    for ser in sorted({f["series"] for f in allf}):
        fs = sorted([f for f in allf if f["series"] == ser], key=lambda f: (f["t"], not f["magnet_on"]))
        t0 = [f for f in fs if f["t"] == min(x["t"] for x in fs)]
        base = next((f for f in t0 if f["magnet_on"]), t0[0])          # magnet-attached t=0 where it exists
        e0 = excess_profile(base["D"]); l0, r0 = gap_edges(e0)
        seen, prof = set(), []
        for f in fs:
            if f["t"] in seen: continue                                  # one frame per time point
            seen.add(f["t"]); prof.append((f, excess_profile(f["D"])))
        for f, e in prof:
            ms = fronts(e, e0, l0, r0)
            ms["front_lead"] = ms["front_thr15"]
            recs.append({k: v for k, v in f.items() if k not in ("D", "magnet_on")} | ms | dict(gap_mm=(r0 - l0) * MM_PER_PX))
    df = pd.DataFrame(recs).sort_values(["series", "t"])
    bad = df.wall_mm.isna().sum()
    if bad: print(f"skipped {bad} frame(s) with an unusable gap edge")
    df = df.dropna(subset=["wall_mm"])
    df.to_csv(Path(__file__).parent / "velocity_centre_frames.csv", index=False)

    # ---- per-series velocities ------------------------------------------------
    vel = []
    for s, g in df.groupby("series"):
        g = g.sort_values("t")
        wall = g.wall_mm.median()

        def fit(col, side_col, tmax=T_RISE):
            # only while the front is still free to move: stop at its first arrival at the wall
            hit = g[g[col] >= wall - WALL_MARGIN].t
            t_end = min(tmax, hit.min()) if len(hit) else tmax
            rise = g[g.t < t_end] if len(hit) and hit.min() <= tmax else g[g.t <= t_end]
            if len(rise) < 3:
                return np.nan, np.nan, len(rise), False
            v, b, rr, p_, se = stats.linregress(rise.t, rise[col])
            return v, se, len(rise), bool((rise[side_col] > 0).any())

        v_front, se, n_rise, resolved = fit("front", "front_right")
        v_full, se_full, n_full, resolved_full = fit("front", "front_right", T_FULL)
        # PRIMARY whole-run velocity: peak advance rate = furthest the (3-point median smoothed) directional
        # front reached in the run / time taken to reach it. Unaffected by the wall (the clock stops when the
        # front stops) and by late back-drift as the reservoir spreads symmetrically; never negative.
        fsm = g.front.rolling(3, center=True, min_periods=1).median().to_numpy(); tt = g.t.to_numpy()
        kpk = int(np.argmax(fsm))
        v_peak = max(fsm[kpk], 0.0) / tt[kpk] if tt[kpk] > 0 else 0.0
        t_peak = tt[kpk]
        resolved_run = bool((g.front_right > 0).any())     # plume reached the front level at some point in the run
        # an unresolved block's front never passed EDGE_SKIP in the run, so its peak rate is below EDGE_SKIP / run time;
        # enter it at that upper bound so slow blocks are not silently dropped from cell means
        v_bound = v_peak if resolved_run else EDGE_SKIP / T_FULL
        # leading-edge (15 L*) front: the magnet-side leading front alone tells whether it resolved
        g["_lead_right"] = np.nan
        v_lead, se_l, n_lead, res_lead = fit("front_lead", "front_lead")
        rd = g[(g.t <= T_RISE)].dropna(subset=["d90"])
        v_d90 = stats.linregress(rd.t, rd.d90).slope if len(rd) >= 3 else np.nan
        rise = g[g.t <= T_RISE]
        v_origin = float((rise.t * rise.front).sum() / (rise.t ** 2).sum()) if (rise.t ** 2).sum() > 0 else np.nan
        g0 = g.iloc[0]
        vel.append(dict(day=g0.day, agarose=g0.agarose, arm=g0.arm, coating=g0.coating, series=s, resolved=resolved,
                        v_front=v_front, v_front_se=se, v_origin=v_origin, v_d90=v_d90, n_rise=n_rise,
                        v_full=v_full, resolved_full=resolved_full, n_full=n_full,
                        v_peak=v_peak, t_peak=t_peak, front_peak=float(fsm[kpk]), resolved_run=resolved_run, v_bound=v_bound,
                        v_lead=v_lead, resolved_lead=res_lead,
                        front_max=g.front.max(), front_6h=g[g.t == g.t.max()].front.iloc[0],
                        wall_mm=wall,
                        t_wall=float(g[g.front >= wall - WALL_MARGIN].t.min()) if (g.front >= wall - WALL_MARGIN).any() else np.nan))
    V = pd.DataFrame(vel)
    V.to_csv(Path(__file__).parent / "velocity_centre_series.csv", index=False)

    print("\n=== directional front velocity (magnet side minus far side), rising phase t <= 3 h, mm/h ===")
    print(V.groupby(["agarose", "arm"]).v_front.agg(["mean", "std", "count"]).round(3).to_string())
    print("\nrobustness: slope through the origin instead of free intercept (mm/h)")
    print(V.groupby(["agarose", "arm"]).v_origin.agg(["mean", "std"]).round(3).to_string())
    print("\nfor reference, one-sided fronts at 6 h (mm): magnet side | far side")
    six = df[df.t == 6.0].groupby(["agarose", "arm"])[["front_right", "front_left"]].mean().round(2)
    print(six.to_string())
    print("\ngap width at t=0 (mm):", df.groupby("agarose").gap_mm.median().round(2).to_dict())
    print("\nfront reached by 6 h (mm from gap edge; block edge is", round(V.wall_mm.median(), 2), "mm):")
    print(V.groupby(["agarose", "arm"]).front_6h.agg(["mean", "std"]).round(2).to_string())
    print("\ntime at which the front first came within", WALL_MARGIN, "mm of the wall (h):")
    print(V.groupby(["agarose", "arm"]).t_wall.agg(["mean", "min", "max"]).round(2).to_string())

    print("\n=== threshold sensitivity: mean velocity (mm/h) by condition, and large/small ratio ===")
    hdr = f"{'thr (L*)':>9s}" + "".join(f"{c:>14s}" for c in ["0.4% large", "0.4% small", "0.6% large", "controls"]) + f"{'L/S ratio':>11s}"
    print(hdr)
    for thr in THR_SENS:
        col = f"front_thr{int(round(thr * L_SCALE))}"; vals = {}
        for s_, g in df.groupby("series"):
            g = g.sort_values("t"); wall_ = g.wall_mm.median(); hit_ = g[g[col] >= wall_ - WALL_MARGIN].t
            w = g[g.t < min(T_RISE, hit_.min())] if len(hit_) and hit_.min() <= T_RISE else g[g.t <= T_RISE]
            if len(w) >= 3: vals[s_] = (g.agarose.iloc[0], g.arm.iloc[0], stats.linregress(w.t, w[col]).slope)
        vv = pd.DataFrame([dict(series=k, agarose=a, arm=b, v=c) for k, (a, b, c) in vals.items()])
        def mv(ag, arm): x = vv[(vv.agarose == ag) & (vv.arm == arm)].v; return x.mean()
        ctl = vv[vv.arm == "control"].v.mean()
        print(f"{thr:9.0f}" + "".join(f"{mv(a, b):14.3f}" for a, b in [("0.4%", "large"), ("0.4%", "small"), ("0.6%", "large")]) + f"{ctl:14.3f}" + f"{mv('0.4%', 'large') / mv('0.4%', 'small'):11.2f}")

    def grp(ag, arm): return V[(V.agarose == ag) & (V.arm == arm)].v_front.dropna()
    L4, S4, L6 = grp("0.4%", "large"), grp("0.4%", "small"), grp("0.6%", "large")
    C4, C6 = grp("0.4%", "control"), grp("0.6%", "control")
    print("\n=== comparisons ===")
    print(f"large vs small, 0.4%:   {L4.mean():.3f} vs {S4.mean():.3f} mm/h  ratio {L4.mean()/S4.mean():.2f}  "
          f"(gradient ratio at the gap = 2.7)   Welch p={stats.ttest_ind(L4, S4, equal_var=False).pvalue:.4f}")
    print(f"0.4% vs 0.6%, large:    {L4.mean():.3f} vs {L6.mean():.3f} mm/h  ratio {L4.mean()/L6.mean():.2f}   "
          f"Welch p={stats.ttest_ind(L4, L6, equal_var=False).pvalue:.4f}")
    print("\nper-series: PEAK ADVANCE RATE over the run (primary), regression slopes 0-3 h and 0-6 h for reference")
    print(V[["agarose", "arm", "coating", "series", "resolved", "v_peak", "t_peak", "front_peak", "v_front", "v_full"]].round(2).to_string(index=False))
    d = V[V.arm != "control"]
    print("\npeak advance rate by cell, unresolved blocks entered at their upper bound (0.1 mm/h); '<=' marks such cells:")
    for (ag, arm, co), gg in d.groupby(["agarose", "arm", "coating"]):
        print(f"  {ag} {arm:6s} {co:5s} {'<=' if (~gg.resolved_run).any() else '  '}{gg.v_bound.mean():.2f} ± {gg.v_bound.std():.2f}  (n={len(gg)}, unresolved {int((~gg.resolved_run).sum())})")
    print("\nleading front (15 L*) by condition:")
    print(V.groupby(["agarose", "arm"]).v_lead.agg(["mean", "std", "count"]).round(3).to_string())
    print("controls, individually (mm/h):")
    for r in V[V.arm == "control"].itertuples(): print(f"  {r.series:26s} {r.v_front:+.3f}")
    for ag, co in [("0.4%", "COOH"), ("0.4%", "PEG"), ("0.6%", "COOH"), ("0.6%", "PEG")]:
        x = V[(V.agarose == ag) & (V.arm == "large") & (V.coating == co)].v_front
        print(f"  large, {ag} {co}: {x.mean():.3f} ± {x.std():.3f} mm/h (n={len(x)})")

    # ---- figure ---------------------------------------------------------------
    GRAD_RATIO, BGRAD_RATIO = 2.7, 10.6          # large/small at the 5 mm gap, from the fitted magnet models
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]))
    LBL = {("0.4%", "large"): "0.4 % agarose, large magnet", ("0.4%", "small"): "0.4 % agarose, small magnet",
           ("0.6%", "large"): "0.6 % agarose, large magnet"}
    for (ag, arm), g in df[df.arm != "control"].groupby(["agarose", "arm"]):
        col = C[(ag, arm)]
        p = g.pivot_table(index="t", columns="series", values="front")
        for c in p.columns:
            a1.plot(p.index, p[c], color=col, lw=.7, alpha=.28)
        a1.plot(p.index, p.mean(axis=1), color=col, lw=2.3, marker="o", ms=3.6, label=f"{LBL[(ag, arm)]} (n={p.shape[1]})")
    shown = set()
    for s_, g in df[df.arm == "control"].groupby("series"):
        ag = g.agarose.iloc[0]
        a1.plot(g.t, g.front, color="#666", lw=1.0, ls=":" if ag == "0.4%" else "--", marker="s", ms=2.3,
                label=None if ag in shown else f"no-magnet controls, {ag} (n=2)")
        shown.add(ag)
    wall = V.wall_mm.median()
    a1.axhline(wall, color="#555", lw=.9, ls="-."); a1.text(6.12, wall, "block\nedge", fontsize=7.5, va="center", color="#555")
    a1.axvspan(0, T_RISE, color="#000", alpha=.045, lw=0)
    a1.text(T_RISE / 2, -0.5, "fit window", ha="center", fontsize=8.3, color="#666")
    a1.set_xlabel("Time after magnet applied (h)"); a1.set_ylabel("Directional front, magnet side − far side (mm)")
    a1.set_xlim(-0.1, 6.75); a1.set_ylim(-0.65, wall + 0.6)
    a1.legend(fontsize=8.3, frameon=False, loc="upper left", bbox_to_anchor=(0.045, 0.94))
    a1.grid(alpha=.16, lw=.6)

    order = [("0.4%", "large"), ("0.4%", "small"), ("0.6%", "large")]
    for k, (ag, arm) in enumerate(order):
        v = V[(V.agarose == ag) & (V.arm == arm)].v_front.dropna()
        a2.scatter(np.full(len(v), k) + np.linspace(-.13, .13, len(v)), v, color=C[(ag, arm)], s=28, zorder=3, alpha=.9, edgecolor="white", lw=.5)
        a2.errorbar(k, v.mean(), yerr=v.std(), fmt="_", color="k", ms=24, mew=2, capsize=6, zorder=4)
    vs = V[(V.agarose == "0.4%") & (V.arm == "small")].v_front.mean()
    a2.hlines(vs * BGRAD_RATIO, -0.42, 0.42, color=C[("0.4%", "large")], ls="--", lw=1.1, label=f"small × {BGRAD_RATIO}  (B∇B ratio)")
    a2.hlines(vs * GRAD_RATIO, -0.42, 0.42, color=C[("0.4%", "large")], ls=":", lw=1.4, label=f"small × {GRAD_RATIO}  (∇B ratio)")
    a2.legend(fontsize=8.0, frameon=False, loc="upper right")
    for k, (ag, arm) in enumerate(order):
        v = V[(V.agarose == ag) & (V.arm == arm)].v_front
        a2.text(k, -0.42, f"{v.mean():.2f} ± {v.std():.2f}", ha="center", fontsize=8.8)
    a2.axhline(0, color="#888", lw=.8)
    a2.set_xticks(range(3)); a2.set_xticklabels(["0.4 %\nlarge", "0.4 %\nsmall", "0.6 %\nlarge"], fontsize=9.5)
    a2.set_ylabel("Front velocity over the fit window (mm/h)"); a2.set_ylim(-0.55, 2.2); a2.set_xlim(-0.6, 2.75)
    a2.grid(alpha=.16, lw=.6, axis="y")
    for ax, lab in ((a1, "(a)"), (a2, "(b)")):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.text(0.01, 0.995, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "velocity_centre.png", dpi=200); fig.savefig(OUT / "velocity_centre.pdf")
    thesis = Path(__file__).parent.parent / "thesis_draft" / "figures"
    if thesis.exists():
        fig.savefig(thesis / "velocity_centre.pdf"); fig.savefig(thesis / "velocity_centre.png", dpi=300)
    print("\nwrote outputs/velocity_centre.png, velocity_centre_series.csv, velocity_centre_frames.csv")


if __name__ == "__main__":
    main()
