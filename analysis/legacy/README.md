# Legacy metric tables

These CSVs were produced by interactive, uncommitted analysis sessions during
the early stages of the project. They have **no producer script** in this
repository and their metric definitions cannot be assumed consistent with each
other or with the current pipeline.

Do not cite numbers from these files. Every figure on the published pages comes
from a committed script:

| Quantity | Script | Output |
|---|---|---|
| per-frame asymmetry, centre injection | `cross_day_compare.py` | `cross_day_asymmetry.csv` |
| per-series plateau values | `cross_day_compare.py` | `centre_plateau_by_series.csv` |
| ANOVA / p-values | `centre_stats.py` | printed |
| back-injection front depth | `back_depth_compare.py` | `back_depth_metrics.csv` |
| Aug 23 geometry comparison | `nav_metrics_compare.py`, `nav_metrics3.py` | `aug23_nav_compare.csv`, `aug23_nav3.csv` |

`band_metrics.csv` is the source of the Aug 27 "+0.44 mm" magnet-effect figure
quoted in earlier findings; `back_depth_compare.py` independently reproduces it
as +0.41 mm.
