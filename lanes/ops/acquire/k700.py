"""Kinetics-700 (2020) clips for the ops lane, fetched from the HF mirror without YouTube.

The mirror `atalaydenknalbant/Kinetics-700` stores 22 zips (~42.7 GB each) with members named
`<split>/<ytid>_<start:06d>_<end:06d>.mp4`. We read only the zip central directories (HTTP range requests)
and extract the clips of the target classes.

  python -m lanes.ops.acquire.k700 index            # zip directories -> data/prep/ops/k700_index.csv
  python -m lanes.ops.acquire.k700 fetch --budget-gb 10
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip

from seen.paths import PREP, RAW

URL = "https://huggingface.co/datasets/atalaydenknalbant/Kinetics-700/resolve/main/Kinetics700_part_{:03d}.zip"
N_PARTS = 22
ANN = {
    "train": "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv",
    "val": "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv",
}
OUT = RAW / "ops_k700"
INDEX = PREP / "ops" / "k700_index.csv"
SELECTED = PREP / "ops" / "k700_selected.csv"

# Service work (positives for `work`) and car-context negatives (person near/in car, no service work).
SERVICE = ["changing wheel (not on bike)", "changing oil", "checking tires", "repairing puncture"]
CAR_CONTEXT = ["pushing car", "pumping gas", "driving car", "changing gear in car", "steering car", "vacuuming car"]
NEG_TRAIN_CAP = 300  # clips per negative class in train (all val clips are kept)


def _annotations() -> dict[tuple[str, int, int], tuple[str, str]]:
    """(ytid, start, end) -> (label, split)."""
    out = {}
    for split, url in ANN.items():
        cache = PREP / "ops" / f"k700_{split}.csv"
        if not cache.exists():
            cache.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        for r in csv.DictReader(io.StringIO(cache.read_text())):
            out[(r["youtube_id"], int(r["time_start"]), int(r["time_end"]))] = (r["label"], split)
    return out


def index() -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def one(part: int):
        for attempt in range(5):
            try:
                with RemoteZip(URL.format(part)) as z:
                    return [(part, i.filename, i.file_size) for i in z.infolist() if i.filename.endswith(".mp4")]
            except Exception as e:  # noqa: BLE001
                print("retry", part, e, flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"part {part} failed")

    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(one, range(1, N_PARTS + 1)):
            rows.extend(res)
            print("part", res[0][0] if res else "?", len(res), flush=True)
    with INDEX.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "member", "size"])
        w.writerows(rows)
    print("indexed", len(rows), "clips ->", INDEX)


def select(budget_gb: float, seed: int = 0) -> list[dict]:
    ann = _annotations()
    by_class: dict[str, list[dict]] = {}
    with INDEX.open() as f:
        for r in csv.DictReader(f):
            name = Path(r["member"]).stem  # <ytid>_<start>_<end>; ytid may contain '_'
            ytid, s, e = name[:-14], int(name[-13:-7]), int(name[-6:])
            hit = ann.get((ytid, s, e))
            if hit is None:
                continue
            label, split = hit
            if label in SERVICE or label in CAR_CONTEXT:
                by_class.setdefault(label, []).append(dict(part=int(r["part"]), member=r["member"], size=int(r["size"]),
                                                           ytid=ytid, start=s, end=e, label=label, split=split))
    rng = random.Random(seed)
    chosen = []
    for label, items in sorted(by_class.items()):
        tr = [i for i in items if i["split"] == "train"]
        va = [i for i in items if i["split"] == "val"]
        rng.shuffle(tr)
        if label in CAR_CONTEXT:
            tr = tr[:NEG_TRAIN_CAP]
        chosen += tr + va
        print(f"{label:32s} train {len(tr):4d} val {len(va):3d}  available {len(items)}")
    chosen.sort(key=lambda i: (i["split"] != "val", i["label"] not in SERVICE))  # val + positives first
    total, keep = 0, []
    for c in chosen:
        if total + c["size"] > budget_gb * 1e9:
            continue
        total += c["size"]
        keep.append(c)
    print(f"selected {len(keep)} clips, {total / 1e9:.2f} GB")
    with SELECTED.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)
    return keep


def fetch(budget_gb: float, workers: int = 8) -> None:
    keep = select(budget_gb)
    OUT.mkdir(parents=True, exist_ok=True)
    by_part: dict[int, list[dict]] = {}
    for c in keep:
        by_part.setdefault(c["part"], []).append(c)

    def one_part(part: int, items: list[dict]) -> int:
        n = 0
        todo = [c for c in items if not (OUT / Path(c["member"]).name).exists()]
        for attempt in range(5):
            try:
                with RemoteZip(URL.format(part)) as z:
                    for c in list(todo):
                        dst = OUT / Path(c["member"]).name
                        tmp = dst.with_suffix(".part")
                        tmp.write_bytes(z.read(c["member"]))
                        os.replace(tmp, dst)
                        todo.remove(c)
                        n += 1
                return n
            except Exception as e:  # noqa: BLE001
                print("retry part", part, len(todo), e, flush=True)
                time.sleep(10 * (attempt + 1))
        return n

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one_part, p, items): p for p, items in by_part.items()}
        for fu in as_completed(futs):
            done += fu.result()
            print("part", futs[fu], "done; total clips", done, flush=True)
    print("FETCH_DONE", done)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "select", "fetch"])
    ap.add_argument("--budget-gb", type=float, default=10.0)
    a = ap.parse_args()
    {"index": index, "select": lambda: select(a.budget_gb), "fetch": lambda: fetch(a.budget_gb)}[a.cmd]()
