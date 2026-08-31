"""Reconcile Aug 26 classifications into series.

Series = coating_bsa_rep for magnet arms, coating_bsa_control for controls.
Applies the tally-referee verdicts (the ribbed strap makes raw stroke counts
unreliable; the referees matched marks against per-repeat reference crops
under the constraint that every timepoint trio is {1,2,3}).
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad/aug26")

m = pd.read_csv(Path(__file__).parent / "classified26.csv", parse_dates=["capture_time"])
m["idx"] = m.index
m["tally"] = m["v_tally"]
m["tally_changed"] = False
m["tally_certainty"] = "high"
m["duplicate_shot"] = False

fixes = {}
for f in sorted(glob.glob(str(S / "tallyfix" / "fixed_*.jsonl"))):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            fixes[int(r["idx"])] = r
for i, r in fixes.items():
    if i not in m.index:
        continue
    old = m.at[i, "tally"]
    new = int(r["tally_fixed"])
    if pd.isna(old) or int(old) != new:
        m.at[i, "tally_changed"] = True
    m.at[i, "tally"] = new
    m.at[i, "duplicate_shot"] = bool(r.get("duplicate", False))
    m.at[i, "tally_certainty"] = r.get("certainty", "high")

# manual overrides (from direct inspection in the main session)
mo = Path(__file__).parent / "manual_overrides26.json"
if mo.exists():
    for path, fix in json.load(open(mo)).items():
        sel = m.path == path
        if not sel.any():
            continue
        if "tally" in fix:
            m.loc[sel, "tally"] = int(fix["tally"])
            m.loc[sel, "tally_changed"] = True
        for k in ("coating", "bsa", "control"):
            if k in fix:
                m.loc[sel, k] = fix[k]

m["cond"] = m.coating + "_" + m.bsa
m["rep"] = m["tally"].astype("Int64")
m["series"] = np.where(
    m.control,
    m.coating + "_" + m.bsa + "_control",
    m.coating + "_" + m.bsa + "_r" + m["rep"].astype(str),
)
m.to_csv(Path(__file__).parent / "photos26_final.csv", index=False)

print("photos:", len(m), "| series:", m.series.nunique(), "| tally fixes applied:", len(fixes))
print("\n=== trios per (timepoint, condition): should be {1,2,3} ===")
bad = []
for (t, c), g in m[~m.control].groupby(["timepoint_hr", "cond"]):
    core = g[~g.duplicate_shot]
    tal = sorted([int(x) for x in core.tally.dropna()])
    if tal != [1, 2, 3]:
        bad.append((t, c, tal, len(g)))
if bad:
    for t, c, tal, n in bad:
        print(f"  UNRESOLVED t={t:4.1f} {c:13s} n={n} tallies={tal}")
else:
    print("  all 52 trios resolve cleanly to {1,2,3}")

print("\n=== coverage per series (13 timepoints expected) ===")
cov = m.groupby("series").timepoint_hr.nunique().sort_values()
print("  complete:", int((cov == 13).sum()), "of", len(cov))
inc = cov[cov != 13]
if len(inc):
    print(inc.to_string())
