# Nanoparticles

Magnetic nanoparticle transport experiments in agarose, with an automated image-analysis pipeline.

## Experiment

70 µL of magnetic nanoparticles are injected into a 1×10×10 mm gap beside a 10×10×10 mm agarose block held in a 3D-printed case. A magnet on the opposite face pulls the NPs through the block; top-view phone photos are taken at intervals. Conditions compared: PEG vs COOH coating, BSA vs non-BSA agarose, 0.4 % vs 0.6 % agarose, magnet vs no-magnet controls (3 repeats each).

`27th August (…)/` holds the raw photos for the 21.5 h back-injection run: 600 photos across 8 conditions × (3 magnet + 3 control repeats) × 12 timepoints, sorted by agarose %, BSA, and nominal timepoint. Capture times are embedded in the `PXL_*` filenames; each sample's condition is on the sticker in frame, repeats are tally marks, controls are marked "C".

## Pipeline (`analysis/`)

Python (uv venv, opencv/numpy/pandas/scipy/matplotlib). Main stages, in order:

| Script | What it does |
|---|---|
| `catalog.py` | walk the photo tree, parse capture times → `photos.csv` |
| `preprocess.py` | downscaled frames + high-res crops of the red tally marks |
| `detect_wells.py` | find the agarose well in every frame (HSV masks + fallbacks) |
| `warp_all.py` | two-pass perspective warp to a canonical square, magnet side right |
| `merge_classifications.py`, `reconcile.py` | merge vision-read labels (sticker / tally / control / magnet), dedupe cross-filed photos, resolve repeats → `photos_final.csv` |
| `measure.py`, `analyze.py`, `analyze2.py` | intensity profiles, band-depth metrics, per-series kymographs |
| `make_block_atlas.py` | **final block-heatmap atlas** — the validated recipe (see below) |
| `make_sheet.py` | verification contact sheet for auditing the automated sorting |

### The validated heatmap recipe (`make_block_atlas.py`)

Validated frame-by-frame against the photos (see `analysis/outputs/r2_validation.html`):

1. ECC affine registration of each frame onto its series' t = 0 (Sobel-gradient images).
2. Coarse anchor: re-detect the injection gap (the dark run after the white rim) in every frame and shift it onto its t = 0 position; fine anchor: align the rim's inner edge (the print itself, which cannot move).
3. One crop per series from t = 0; per-edge sibling-median repair for outliers.
4. The gap/block boundary is measured once from the t = 0 reservoir band and held fixed (agarose shrinkage over the run is shown, not compensated); mm scale: boundary → block far face = 10 mm.
5. Display: absolute NP darkness minus each frame's own gel-interior floor (cancels the phone's auto-exposure), median-binned, lightly smoothed, one shared color scale.

## Outputs (`analysis/outputs/`)

- `block_heatmap_atlas.html` — per-condition block heatmaps, magnet vs control (mean of 3 repeats) at five time stages
- `kymograph_atlas.html` — per-repeat position–time heatmaps
- `magnet_effect_figures.html` — band-depth comparison figures and methods/limits
- `verification_sheet.html` — audit sheet for the automated photo sorting
- `r2_validation.html` — photo-vs-heatmap validation of the atlas recipe
- `block_heatmap_panels/`, `kymograph_panels/` — the PNG panels

## Key findings (Aug 27 run)

- Bulk transport through the gel is shallow: the visible NP band reaches ≤ 4 mm of the 10 mm block in every condition. Controls plateau at ~3 mm within the first hour; magnet arms keep advancing all run (+0.44 mm at 21.5 h, 7/8 conditions, paired t = 2.66).
- NPs do reach the magnet-side wall (deposit observed on disassembly), most plausibly via the agarose–case interface / block bottom — a route the top view cannot quantify because the magnet occludes and shadows that zone.
- Coating / BSA / agarose % differences are within repeat-level noise at n = 3.
- Protocol notes for future runs: lock exposure and white balance, mount the camera, keep a gray card in frame, and photograph all blocks magnet-off (top + side) at the end of the run.

## Reproducing

```bash
cd analysis
uv venv --python 3.12 && uv pip install opencv-python-headless numpy pandas matplotlib pillow scipy
.venv/bin/python catalog.py "../27th August (21.5 Hours, Back 0.6 vs Back 0.4, BSA vs Non-BSA, PEG vs COOH, N=3 Controls) 2"
# then preprocess.py, detect_wells.py, warp_all.py, merge_classifications.py,
# reconcile.py, measure.py, analyze.py, make_block_atlas.py
```

Intermediate images (downscaled frames, warped wells, tally crops) are written to a scratch directory and regenerate from the raw photos; the sticker/tally classifications were produced by vision models with human spot-checks and live in the committed CSVs.
