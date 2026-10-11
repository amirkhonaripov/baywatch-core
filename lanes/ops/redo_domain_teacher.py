"""Re-run the teacher on domain frames only (after the frame-id fix) and merge into vlm_teacher.parquet.
Stage 1 (env/main): write the domain subset of teacher_items; stage 2: lanes.ops.vlm on it; stage 3: merge."""
import sys

import pandas as pd

from lanes.ops.featurize import FEAT
from lanes.ops.prep import FAST_OPS

if sys.argv[1] == "subset":
    it = pd.read_parquet(FAST_OPS / "teacher_items.parquet")
    it[it["item_id"].str.endswith("|domain")].reset_index(drop=True).to_parquet(FAST_OPS / "teacher_items_domain.parquet")
    print("domain items", int(it["item_id"].str.endswith("|domain").sum()))
else:
    t = pd.read_parquet(FEAT / "vlm_teacher.parquet")
    d = pd.read_parquet(FEAT / "vlm_teacher_domain.parquet")
    t = pd.concat([t[~t["item_id"].str.endswith("|domain")], d], ignore_index=True)
    assert t["item_id"].is_unique
    t.to_parquet(FEAT / "vlm_teacher.parquet")
    print("MERGED", len(t), len(d))
