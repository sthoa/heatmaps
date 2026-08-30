"""Build fingerprint-matching tasks for disputed repeat assignments.

Premise (confirmed by Steven): every (condition, arm) trio was photographed
completely at every timepoint — a 3-photo burst trio is always repeats
{1,2,3}. Each physical sample keeps its tally tab all day, so the mark shape
is a stable fingerprint across timepoints.

For each (agarose, bsa, coating, arm) group:
  - clean timepoints: 3 photos with distinct tallies -> those crops become
    per-repeat REFERENCE fingerprints
  - disputed timepoints: assignments not {1,2,3} (or >3 photos) -> a task
    asking to match each disputed crop to the repeat whose references it
    resembles

Writes fingerprint tasks as JSONL for referee agents.
"""

import json
from pathlib import Path

import pandas as pd

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")

m = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
m["capture_time"] = pd.to_datetime(m["capture_time"])

tasks = []
for (ag, bsa, coat, arm), grp in m.groupby(["agarose_f", "bsa_f", "coating_f", "arm"]):
    clean_refs = {1: [], 2: [], 3: []}
    disputed = []
    for tp, g in grp.groupby("timepoint_hr"):
        # at t=0 the magnet arm legitimately has 2 photos per repeat (with/without magnet)
        if arm == "magnet" and tp == 0:
            sub = g[g.magnet_f == "present"]
        else:
            sub = g
        tallies = sorted(sub.tally_f.tolist())
        if len(sub) == 3 and tallies == [1, 2, 3]:
            for r in sub.itertuples():
                clean_refs[int(r.tally_f)].append((tp, int(r.idx)))
        else:
            disputed.append((tp, sub))
    if not disputed:
        continue
    refs = {
        rep: [
            {"idx": i, "crop": str(S / "prep" / "tally" / f"{i:04d}.jpg"), "timepoint": t}
            for t, i in v[:3]
        ]
        for rep, v in clean_refs.items()
    }
    for tp, sub in disputed:
        tasks.append(
            {
                "group": f"{ag}_{bsa}_{coat}_{arm}_t{tp:g}",
                "n_photos": len(sub),
                "photos": [
                    {
                        "idx": int(r.idx),
                        "crop": str(S / "prep" / "tally" / f"{int(r.idx):04d}.jpg"),
                        "current_read": int(r.tally_f),
                        "time": str(r.capture_time),
                    }
                    for r in sub.sort_values("capture_time").itertuples()
                ],
                "references": refs,
            }
        )

out = S / "tallyfix" / "fingerprint_tasks.jsonl"
out.write_text("\n".join(json.dumps(t) for t in tasks))
print("disputed timepoint-groups:", len(tasks))
print("photos involved:", sum(t["n_photos"] for t in tasks))
