"""Catalog all photos in an experiment-day folder.

Walks the folder tree, parses capture timestamps from Pixel filenames
(PXL_YYYYMMDD_HHMMSSmmm.jpg), records the folder-derived metadata
(agarose %, BSA, nominal timepoint), and writes photos.csv.

Within each timepoint folder, photos are numbered in capture order
(shot_order) — useful for spotting the bursts in which sets were taken.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PXL_RE = re.compile(r"PXL_(\d{8})_(\d{9})")


def parse_pxl_time(name: str):
    m = PXL_RE.search(name)
    if not m:
        return None
    d, t = m.groups()
    return datetime.strptime(d + t[:6], "%Y%m%d%H%M%S").replace(microsecond=int(t[6:9]) * 1000)


def parse_timepoint_hours(folder: str):
    m = re.match(r"([\d.]+)\s*hrs?", folder.strip(), re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_agarose(folder: str):
    m = re.search(r"0\.[46]", folder)
    return m.group(0) if m else None


def parse_bsa(folder: str):
    return "non-BSA" if re.search(r"non[- ]?bsa", folder, re.IGNORECASE) else "BSA"


def catalog(day_dir: Path) -> pd.DataFrame:
    rows = []
    for jpg in sorted(day_dir.rglob("*.jpg")):
        rel = jpg.relative_to(day_dir)
        parts = rel.parts
        if len(parts) != 4:  # agarose / bsa / timepoint / file
            print(f"WARNING: unexpected depth, skipping {rel}", file=sys.stderr)
            continue
        agarose_dir, bsa_dir, tp_dir, fname = parts
        ts = parse_pxl_time(fname)
        rows.append(
            {
                "path": str(rel),
                "agarose": parse_agarose(agarose_dir),
                "bsa": parse_bsa(bsa_dir),
                "timepoint_hr": parse_timepoint_hours(tp_dir),
                "capture_time": ts,
            }
        )
    df = pd.DataFrame(rows).sort_values(["agarose", "bsa", "timepoint_hr", "capture_time"])
    df["shot_order"] = df.groupby(["agarose", "bsa", "timepoint_hr"]).cumcount() + 1
    # seconds since the previous shot in the same timepoint folder — reveals bursts
    df["gap_s"] = (
        df.groupby(["agarose", "bsa", "timepoint_hr"])["capture_time"].diff().dt.total_seconds()
    )
    return df


if __name__ == "__main__":
    day_dir = Path(sys.argv[1])
    df = catalog(day_dir)
    out = Path(__file__).parent / "photos.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} photos cataloged -> {out}")
    print(df.groupby(["agarose", "bsa"])["timepoint_hr"].agg(["count", "nunique"]))
