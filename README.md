# heatmaps

Automated image analysis of magnetic nanoparticle transport through agarose, from top-view phone photographs. Three run days, ~1000 photos, one pipeline per injection geometry.

## Experiment

70 µL of magnetic nanoparticles are injected into a 1 × 10 × 10 mm gap beside a 10 × 10 × 10 mm agarose block in a 3D-printed holder. A magnet on one face pulls the particles through the block; top-view phone photos are taken at intervals. Conditions varied across days: PEG vs COOH coating, BSA vs non-BSA agarose, 0.4 % vs 0.6 % agarose, large vs small magnet, centre vs back injection, and no-magnet controls.

| Day | Design | Photos | Series |
|---|---|---|---|
| Aug 23 | 0.4 %, non-BSA, **centre** injection, large vs small magnet, 6 h | 196 | 14 |
| Aug 26 | 0.6 %, BSA vs non-BSA, **centre** injection, large magnet, 6 h | 209 | 16 |
| Aug 27 | 0.4 & 0.6 %, BSA vs non-BSA, **back** injection, 21.5 h | 600 | 48 |

A *series* is one physical sample followed through time. It is the unit of replication in every statistical test.

## Published pages

| Page | What it is |
|---|---|
| [How the numbers were made](https://claude.ai/code/artifact/f6367f9b-094b-4a9a-a687-19a2aef996bf) | **Start here.** Every stage from photo to p-value, the checks, the limitations, and a Q&A |
| [0.4 % vs 0.6 % agarose](https://claude.ai/code/artifact/dc9ab75c-f692-46a1-8b48-0fe06abd2c1c) | The gel-stiffness comparison in both geometries, with statistics |
| [Aug 23 heatmaps](https://claude.ai/code/artifact/bfbbd924-d55b-411c-b54a-6c53b0541676) | Centre injection, large vs small magnet |
| [Aug 26 heatmaps](https://claude.ai/code/artifact/13f18308-9059-4fb3-a44f-672c425ffd74) | Centre injection, BSA vs non-BSA |
| [Aug 27 heatmaps](https://claude.ai/code/artifact/b3891b33-43e9-4880-b0cf-80f59790fb3e) | Back injection, all eight conditions |

The same pages are committed as HTML and PDF under `analysis/outputs/`.

## Pipeline (`analysis/`)

Python 3.12 in a uv venv: opencv, numpy, pandas, scipy, matplotlib, statsmodels.

```
catalog*.py            photo tree → photos*.csv (timepoint from folder name, capture time from PXL_ filename)
preprocess*.py         downscale to 1100 px long edge; sticker / tally crops
reconcile*.py          merge vision-read labels, referee tally trios, key everything by file path → photos*_final.csv
block23_corners.py     Aug 23 block corners from the holder's corner marks
block23_series.py      Aug 23 series-median outlier repair → geom23_series.csv
block26.py             Aug 26 block corners from the red corner dots
detect_wells.py,       Aug 27 well detection and two-pass perspective warp
  warp_all.py
make_block_atlas*.py   perspective-warp to 480 px square, LAB darkness, per-frame floor, heatmap panels
cross_day_compare.py   asymmetry metric for both centre-injection days → cross_day_asymmetry.csv,
                       centre_plateau_by_series.csv, and the comparison figure
back_depth_compare.py  front-position metric for the back-injection day → back_depth_metrics.csv
centre_stats.py        the 2 × 2 ANOVA and every p-value on the pages
build_cross_day_page.py  renders outputs/cross_day_comparison.html
```

Geometry is found differently on each day because each rig offers different reliable features; the measurement that follows is identical: darkness = 100 − L\* in LAB, median-binned, minus the frame's own 15th-percentile gel floor, on a shared 0–22 L\* colour scale.

`analysis/legacy/` holds metric tables from early interactive sessions with no producer script. Do not cite numbers from them.

## Current findings

**Centre injection (Aug 23 + Aug 26).** Every magnet arm separates cleanly from the no-magnet controls, which stay within ±1.7 L\* of zero asymmetry for six hours. Analysed as the 2 × 2 factorial it is (agarose × coating, non-BSA large-magnet arms, n = 12 series):

| Term | Effect | F | p |
|---|---|---|---|
| Agarose 0.4 % vs 0.6 % | +5.8 L\* | 5.64 | **0.045** |
| Coating COOH vs PEG | +4.8 L\* | 3.82 | 0.086 |
| Coating × agarose | — | 0.85 | 0.384 |

Softer gel transports further in both coatings. **Caveat: 0.4 % is Aug 23 and 0.6 % is Aug 26, so agarose concentration is confounded with run day.** Within Aug 26 alone, COOH leads PEG by +4.7 L\* (p = 0.064, BSA as block). The apparent wider coating gap at 0.6 % is the interaction term and is not supported.

**Back injection (Aug 27).** Both concentrations ran in one session (n = 12 per concentration), but the magnet effect itself is only +0.41 mm (p = 0.26): the particles barely left the first few millimetres in 21.5 h, so there is nothing for gel stiffness to modulate. No difference found (p = 0.61). Nanoparticles do reach the magnet-side wall — a deposit sits directly in front of the magnet face on disassembly — but the top view cannot quantify it.

## Reproducing

```bash
cd analysis
uv venv --python 3.12
uv pip install opencv-python-headless numpy pandas scipy matplotlib pillow statsmodels
.venv/bin/python cross_day_compare.py     # asymmetry curves + figure (needs the downscaled frames)
.venv/bin/python centre_stats.py          # every p-value, from the committed per-series CSV
```

Downscaled frames and warped wells live in a scratch directory and regenerate from the raw photos via `preprocess*.py`; the label classifications were produced by vision models with referee passes and human spot-checks and live in the committed CSVs.
