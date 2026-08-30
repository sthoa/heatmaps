"""Merge all classification batches (old + new) keyed by file path.

Old batches (batch_NN.jsonl) used indices from photos_old.csv; new batches
(batch_newN.jsonl) use indices from the current photos.csv. Both are remapped
to paths, then joined onto the current catalog -> classified.csv (new indices).
Also remaps the tally-fix referee results to paths -> tally_fixes_bypath.json.
"""

import glob
import json
from pathlib import Path

import pandas as pd

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")

# each classification batch is keyed to the catalog that existed when it ran
old_map = pd.read_csv("photos_old.csv").path.to_dict()    # 472-photo catalog
prev_map = pd.read_csv("photos_prev.csv").path.to_dict()  # 573-photo catalog
new = pd.read_csv("photos.csv", dtype={"agarose": str})
new_map = new.path.to_dict()                              # current catalog

map_600 = pd.read_csv("photos_600.csv").path.to_dict()  # pre-control-fix 600 catalog
BATCH_CATALOG = [
    ("batch_[0-9]*.jsonl", old_map),
    ("batch_new[0-2].jsonl", prev_map),
    ("batch_new3.jsonl", map_600),
    ("batch_new4.jsonl", new_map),
]
rows = {}
for pattern, mapping in BATCH_CATALOG:
    for f in sorted(glob.glob(str(S / "classify" / pattern))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                path = mapping.get(int(r["idx"]))
                if path is not None:
                    rows[path] = r

cls = pd.DataFrame.from_dict(rows, orient="index").drop(columns=["idx"])
cls.index.name = "path"
merged = new.merge(cls.add_prefix("v_"), left_on="path", right_index=True, how="left")
missing = merged[merged.v_coating.isna()]
if len(missing):
    print("UNCLASSIFIED photos:", list(missing.path))
merged.to_csv("classified.csv", index=False)
print("classified.csv:", len(merged), "rows")

fixes = {}
for f in sorted(glob.glob(str(S / "tallyfix" / "fixed_*.jsonl"))):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            path = old_map.get(int(r["idx"]))
            if path is not None:
                fixes[path] = r
json.dump(fixes, open("tally_fixes_bypath.json", "w"))
print("tally fixes remapped:", len(fixes))
