"""Catalog the Aug 23 run (flat structure: <timepoint>/PXL_*.jpg).

Unlike Aug 27 there are no condition subfolders — every condition is on the
sticker (PEG/COOH, L1-L3 large magnet, S1-S3 small magnet, or 'control').
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PXL_RE = re.compile(r"PXL_(\d{8})_(\d{9})")


def parse_pxl_time(name):
    m = PXL_RE.search(name)
    if not m:
        return None
    d, t = m.groups()
    return datetime.strptime(d + t[:6], "%Y%m%d%H%M%S").replace(microsecond=int(t[6:9]) * 1000)


def parse_timepoint(folder):
    m = re.match(r"([\d.]+)\s*hrs?", folder.strip(), re.IGNORECASE)
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    day_dir = Path(sys.argv[1])
    rows = []
    for jpg in sorted(day_dir.rglob("*.jpg")):
        rel = jpg.relative_to(day_dir)
        if len(rel.parts) != 2:
            print("WARNING unexpected depth:", rel, file=sys.stderr)
            continue
        tp_dir, fname = rel.parts
        rows.append(
            {
                "path": str(rel),
                "timepoint_hr": parse_timepoint(tp_dir),
                "capture_time": parse_pxl_time(fname),
            }
        )
    df = pd.DataFrame(rows).sort_values(["timepoint_hr", "capture_time"]).reset_index(drop=True)
    df["shot_order"] = df.groupby("timepoint_hr").cumcount() + 1
    df["gap_s"] = df.groupby("timepoint_hr")["capture_time"].diff().dt.total_seconds()
    out = Path(__file__).parent / "photos23.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} photos -> {out}")
    print(df.groupby("timepoint_hr").size().to_string())
