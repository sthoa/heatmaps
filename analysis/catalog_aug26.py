"""Catalog the Aug 26 run: <geometry>/<timepoint>/PXL_*.jpg.

The "After injecting" folder (pre-magnet shots) is excluded; t=0 is the
"0 hrs after magnet" folder, per Steven.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PXL_RE = re.compile(r"PXL_(\d{8})_(\d{9})")


def parse_time(name):
    m = PXL_RE.search(name)
    if not m:
        return None
    d, t = m.groups()
    return datetime.strptime(d + t[:6], "%Y%m%d%H%M%S").replace(microsecond=int(t[6:9]) * 1000)


def parse_tp(folder):
    f = folder.strip().lower()
    if f.startswith("after injecting"):
        return None  # excluded
    if f.startswith("0 hrs after magnet"):
        return 0.0
    m = re.match(r"([\d.]+)\s*hrs?", f)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    day = Path(sys.argv[1])
    rows = []
    for jpg in sorted(day.rglob("*.jpg")):
        rel = jpg.relative_to(day)
        if len(rel.parts) != 3:
            print("WARNING depth:", rel, file=sys.stderr)
            continue
        geom, tp_dir, fname = rel.parts
        tp = parse_tp(tp_dir)
        if tp is None:
            continue
        rows.append({"path": str(rel), "geometry": geom, "timepoint_hr": tp,
                     "capture_time": parse_time(fname)})
    df = pd.DataFrame(rows).sort_values(["timepoint_hr", "capture_time"]).reset_index(drop=True)
    df["shot_order"] = df.groupby("timepoint_hr").cumcount() + 1
    df["gap_s"] = df.groupby("timepoint_hr")["capture_time"].diff().dt.total_seconds()
    out = Path(__file__).parent / "photos26.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} photos (excluding 'After injecting') -> {out}")
    print(df.groupby("timepoint_hr").size().to_string())
