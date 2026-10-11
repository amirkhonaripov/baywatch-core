"""One-off repair (2026-10-11): domain frame ids collided across clips. Drop stale domain rows from the detection cache
and re-key the teacher labels by path (teacher outputs are in the same row order as teacher_items.parquet)."""
import pandas as pd

from lanes.ops.featurize import FEAT
from lanes.ops.prep import FAST_OPS, TABLE, fpath

d = pd.read_parquet(FEAT / "dets.parquet")
d = d[~d["frame_id"].str.endswith("|domain")]
d.to_parquet(FEAT / "dets.parquet")
done = pd.read_parquet(FEAT / "dets_done.parquet")
done[~done["frame_id"].str.endswith("|domain")].to_parquet(FEAT / "dets_done.parquet")
tab = pd.read_parquet(TABLE)
by_path = {str(fpath(p)): f for p, f in zip(tab["path"], tab["frame_id"])}
items = pd.read_parquet(FAST_OPS / "teacher_items.parquet")
t = pd.read_parquet(FEAT / "vlm_teacher.parquet")
assert len(items) == len(t)
t["item_id"] = [by_path[p] for p in items["path"]]
assert t["item_id"].is_unique
t.to_parquet(FEAT / "vlm_teacher.parquet")
items["item_id"] = t["item_id"].to_numpy()
items.to_parquet(FAST_OPS / "teacher_items.parquet")
print("FIXED", len(t))
