"""Reconcile per-photo classifications into experiment series.

- Drop duplicate files (same PXL filename in two folders): keep the copy whose
  folder matches its sticker reading.
- Final condition fields: agarose & BSA from FOLDER (validated: sticker digit
  misreads were systematic per-sample; folders are ground truth), coating from
  sticker, control/tally/magnet from vision.
- Group photos into bursts (same folder cell, gap < 180 s) and check tally
  sequences; flag photos whose tally conflicts with a 1,2,3 burst pattern or
  duplicates another tally in the same burst arm.
- Assign series ids: (agarose, bsa, coating, arm, repeat) where arm is
  "magnet" or "control" (0 hr no-magnet condition shots belong to the magnet
  series as timepoint 0 pre-magnet).
Outputs: photos_final.csv (one row per unique photo with series id + flags)
         series_audit.csv  (per series: timepoints present/missing)
"""

import pandas as pd
import numpy as np

m = pd.read_csv("classified.csv", dtype={"agarose": str, "v_agarose": str})
m["fname"] = m.path.str.split("/").str[-1]
m["idx"] = m.index

# --- dedupe: keep the copy whose folder best matches the sticker reading.
# BSA presence is read reliably (word present/absent) so it outweighs the
# agarose digit (handwritten 4/6 is often misread).
def dedupe(group):
    if len(group) == 1:
        return group
    score = 2 * ((group.bsa == "BSA") == group.v_bsa.astype(bool)).astype(int) + (
        group.agarose == group.v_agarose
    ).astype(int)
    return group.loc[[score.idxmax()]]

m = m.groupby("fname", group_keys=False).apply(dedupe, include_groups=False).sort_index()
m["dropped_dup"] = False

# --- final fields
m["agarose_f"] = m["agarose"]          # folder is ground truth
m["bsa_f"] = m["bsa"]                  # folder is ground truth (post-dedupe)
m["coating_f"] = m["v_coating"]
m["control_f"] = m["v_control"].astype(bool)
m["magnet_f"] = m["v_magnet"]
m["tally_f"] = m["v_tally"].astype("Int64")
m["capture_time"] = pd.to_datetime(m["capture_time"])

# --- apply forced-choice tally corrections from the referee pass (path-keyed)
import json
from pathlib import Path

fixes = json.load(open(Path(__file__).parent / "tally_fixes_bypath.json"))
m["duplicate_shot"] = False
m["tally_certainty"] = "high"
m["tally_changed"] = False
m = m.set_index("path", drop=False)
for path, r in fixes.items():
    if path not in m.index:
        continue
    if int(r["tally_fixed"]) != int(m.at[path, "tally_f"]):
        m.at[path, "tally_changed"] = True
    m.at[path, "tally_f"] = int(r["tally_fixed"])
    m.at[path, "duplicate_shot"] = bool(r.get("duplicate", False))
    m.at[path, "tally_certainty"] = r.get("certainty", "high")
# referee-confirmed sticker read: this photo is PEG, not COOH
PEG_FIX = "0.6 BACK/BSA/5 hrs/PXL_20260827_162054615.jpg"
if PEG_FIX in m.index:
    m.at[PEG_FIX, "coating_f"] = "PEG"

# manual overrides from the top-level session's own crop reads
manual_path = Path(__file__).parent / "manual_overrides.json"
if manual_path.exists():
    for path, fix in json.load(open(manual_path)).items():
        if path not in m.index:
            continue
        if "tally" in fix:
            if int(fix["tally"]) != int(m.at[path, "tally_f"]):
                m.at[path, "tally_changed"] = True
            m.at[path, "tally_f"] = int(fix["tally"])
            m.at[path, "tally_certainty"] = "high"
            m.at[path, "duplicate_shot"] = False
        if "coating" in fix:
            m.at[path, "coating_f"] = fix["coating"]

# fingerprint-matching verdicts (highest priority; keyed to the CURRENT catalog)
import glob

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")
cur_map = pd.read_csv(Path(__file__).parent / "photos_600.csv").path.to_dict()  # fp fixes keyed to that catalog
for f in sorted(glob.glob(str(S / "tallyfix" / "fp_fixed_*.jsonl"))):
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        path = cur_map.get(int(r["idx"]))
        if path is None or path not in m.index:
            continue
        if int(r["tally_fixed"]) != int(m.at[path, "tally_f"]):
            m.at[path, "tally_changed"] = True
        m.at[path, "tally_f"] = int(r["tally_fixed"])
        m.at[path, "duplicate_shot"] = bool(r.get("duplicate", False))
        m.at[path, "tally_certainty"] = r.get("certainty", "high")
m = m.reset_index(drop=True)

# --- bursts: within folder cell, sort by time, split on gaps > 180 s
m = m.sort_values(["agarose_f", "bsa_f", "timepoint_hr", "capture_time"])
cell = m.groupby(["agarose_f", "bsa_f", "timepoint_hr"], group_keys=False)
m["gap"] = cell["capture_time"].diff().dt.total_seconds()
m["burst"] = (m["gap"].isna() | (m["gap"] > 180)).astype(int)
m["burst"] = m.groupby(["agarose_f", "bsa_f", "timepoint_hr"])["burst"].cumsum()

# --- tally sanity within burst-arm: same coating+control+magnet in one burst
# should carry distinct tallies (photographed in some order, usually 1,2,3)
flags = []
key_cols = ["agarose_f", "bsa_f", "timepoint_hr", "burst", "coating_f", "control_f", "magnet_f"]
for key, g in m.groupby(key_cols):
    tallies = list(g["tally_f"])
    if len(g) > 1 and len(set(tallies)) < len(tallies):
        for i in g.index:
            flags.append(i)
m["tally_conflict"] = m.index.isin(flags)
m["low_conf"] = m["v_conf"] == "low"

# --- series id
m["arm"] = np.where(m["control_f"], "control", "magnet")
m["series"] = (
    m["agarose_f"] + "_" + m["bsa_f"] + "_" + m["coating_f"] + "_" + m["arm"] + "_r"
    + m["tally_f"].astype(str)
)

m.to_csv("photos_final.csv", index=False)

# --- series audit
tps = sorted(m.timepoint_hr.unique())
rows = []
for s, g in m.groupby("series"):
    present = sorted(g.timepoint_hr.unique())
    missing = [t for t in tps if t not in present]
    # duplicates at a timepoint (beyond the expected 2 photos at t=0 for magnet arm)
    counts = g.groupby("timepoint_hr").size()
    extra = counts[(counts > 1) & (counts.index > 0)].to_dict()
    rows.append(
        {
            "series": s,
            "n_photos": len(g),
            "n_timepoints": len(present),
            "missing_timepoints": missing,
            "multi_photo_timepoints": extra,
            "n_flagged": int(g["tally_conflict"].sum() + g["low_conf"].sum()),
        }
    )
audit = pd.DataFrame(rows).sort_values("series")
audit.to_csv("series_audit.csv", index=False)
print("unique photos:", len(m))
print("series found:", m.series.nunique())
print("tally-conflict photos:", int(m.tally_conflict.sum()), "| low-conf photos:", int(m.low_conf.sum()))
print()
print(audit.to_string(index=False, max_colwidth=40))
