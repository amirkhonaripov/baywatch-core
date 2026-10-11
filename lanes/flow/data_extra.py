"""More dataset adapters: PKLot + CNR-EXT slot occupancy, COCO val2017 persons (forgetting check), MIO-TCD localization."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from lanes.flow.common import PREP, fast_prep, prep_dir, raw_dir
from lanes.flow.data import Item


def day_part(day: str) -> str:
    """Deterministic split of capture days: ~1/3 'cal' (threshold calibration), ~2/3 'test'. Days never mix."""
    return "cal" if int(hashlib.md5(day.encode()).hexdigest(), 16) % 3 == 0 else "test"


# --------------------------------------------------------------------------------------------- PKLot
def pklot_root() -> Path:
    return raw_dir("pklot") / "PKLot" / "PKLot"


def parse_pklot_xml(p: Path) -> list[dict]:
    slots = []
    for sp in ET.parse(p).getroot().iter("space"):
        occ = sp.get("occupied")
        if occ is None:
            continue
        pts = [[float(pt.get("x")), float(pt.get("y"))] for pt in sp.iter("point")]
        if len(pts) >= 3:
            slots.append({"id": sp.get("id"), "poly": pts, "occ": int(occ)})
    return slots


def pklot_items(per_cell: int = 150, seed: int = 0, part: str = "test", lots=("PUCPR", "UFPR04", "UFPR05"), weathers=("Sunny", "Cloudy", "Rainy")) -> list[Item]:
    """Stratified sample: ``per_cell`` images per (lot, weather), spread over as many different days as possible."""
    cache = prep_dir("occupancy") / f"pklot_items_{per_cell}_{seed}_{part}.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
    else:
        rng = random.Random(seed)
        recs = []
        root = pklot_root()
        for lot in lots:
            for w in weathers:
                days = defaultdict(list)
                for d in sorted((root / lot / w).glob("*")):
                    if day_part(d.name) == part:
                        days[d.name] = sorted(d.glob("*.jpg"))
                names = list(days)
                rng.shuffle(names)
                picked = []
                while len(picked) < per_cell and any(days[n] for n in names):
                    for n in names:
                        if days[n] and len(picked) < per_cell:
                            picked.append(days[n].pop(rng.randrange(len(days[n]))))
                for jp in picked:
                    xp = jp.with_suffix(".xml")
                    if xp.exists():
                        recs.append({"path": str(jp), "lot": lot, "weather": w.lower(), "day": jp.parent.name, "slots": parse_pklot_xml(xp)})
        cache.write_text(json.dumps(recs))
    items = []
    for r in recs:
        items.append(Item(id=f"pklot:{r['lot']}/{r['weather']}/{Path(r['path']).name}", path=r["path"], cluster=f"{r['lot']}_{r['day']}",
                          cond=r["weather"], size=(1280, 720), extra={"slots": r["slots"], "lot": r["lot"], "dataset": "pklot"}))
    return items


# --------------------------------------------------------------------------------------------- CNR-EXT
def cnrext_items(per_cell: int = 150, seed: int = 0, part: str = "test") -> list[Item]:
    """CNR-EXT full frames (1000x750) with slot boxes (rescaled from 2592x1944) and per-slot occupancy from CNRPark+EXT.csv."""
    cache = prep_dir("occupancy") / f"cnrext_items_{per_cell}_{seed}_{part}.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
    else:
        root = raw_dir("cnrpark")
        base = fast_prep("cnrext_full")  # NVMe subset written by prep_cnr.py: camera*.csv at the root, frames under FULL_IMAGE_1000x750/
        full = base / "FULL_IMAGE_1000x750"
        # slot boxes per camera
        boxes: dict[int, dict[int, list]] = {}
        for cf in sorted(base.glob("camera*.csv")):
            cam = int(re.search(r"camera(\d+)", cf.name).group(1))
            rows = list(csv.DictReader(open(cf)))
            boxes[cam] = {}
            for r in rows:
                sid = int(r["SlotId"])
                x, y, w, h = (float(r["X"]), float(r["Y"]), float(r["W"]), float(r["H"]))
                s = 1000 / 2592
                boxes[cam][sid] = [[x * s, y * s], [(x + w) * s, y * s], [(x + w) * s, (y + h) * s], [x * s, (y + h) * s]]
        # truth per (camera, date, time)
        truth: dict[tuple, dict[int, int]] = defaultdict(dict)
        wmap = {"S": "sunny", "O": "overcast", "R": "rainy"}
        for r in csv.DictReader(open(root / "CNRPark+EXT.csv")):
            m = re.search(r"PATCHES/(SUNNY|OVERCAST|RAINY)/(\d{4}-\d{2}-\d{2})/camera(\d+)/\w_\d{4}-\d{2}-\d{2}_(\d{2}\.\d{2})_C\d+_(\d+)\.jpg", r["image_url"])
            if not m:
                continue
            truth[(m.group(1).lower(), m.group(2), int(m.group(3)), m.group(4))][int(m.group(5))] = int(r["occupancy"])
        rng = random.Random(seed)
        bycell = defaultdict(list)
        for (w, d, cam, t), sl in truth.items():
            p = full / w.upper() / d / f"camera{cam}" / f"{d}_{t.replace('.', '')}.jpg"
            if p.exists() and day_part(d) == part:
                bycell[w].append((p, d, cam, sl))
        recs = []
        for w, lst in bycell.items():
            rng.shuffle(lst)
            chosen, per_day = [], defaultdict(int)
            for p, d, cam, sl in lst:  # cap images per (day, camera) so the sample is not one afternoon
                if per_day[(d, cam)] >= 3:
                    continue
                per_day[(d, cam)] += 1
                chosen.append((p, d, cam, sl))
                if len(chosen) >= per_cell:
                    break
            for p, d, cam, sl in chosen:
                slots = [{"id": sid, "poly": boxes[cam][sid], "occ": occ} for sid, occ in sl.items() if sid in boxes[cam]]
                if slots:
                    recs.append({"path": str(p), "weather": w, "day": d, "cam": cam, "slots": slots})
        cache.write_text(json.dumps(recs))
    return [Item(id=f"cnrext:{r['weather']}/{r['day']}/cam{r['cam']}/{Path(r['path']).name}", path=r["path"], cluster=f"cam{r['cam']}_{r['day']}",
                 cond=r["weather"], size=(1000, 750), extra={"slots": r["slots"], "lot": f"cam{r['cam']}", "dataset": "cnrext"}) for r in recs]


# --------------------------------------------------------------------------------------------- COCO val persons
def cocoval_items(n: int = 2000, seed: int = 0) -> list[Item]:
    from lanes.flow.router_data import coco_annotations, fetch_coco_images

    root = raw_dir("coco")
    js = coco_annotations()
    imgs = {i["id"]: i for i in js["images"]}
    persons: dict[int, list] = defaultdict(list)
    crowd: dict[int, list] = defaultdict(list)
    for a in js["annotations"]:
        if a["category_id"] != 1:
            continue
        x, y, w, h = a["bbox"]
        (crowd if a["iscrowd"] else persons)[a["image_id"]].append([x, y, x + w, y + h])
    ids = sorted(imgs)
    random.Random(seed).shuffle(ids)
    fetch_coco_images([imgs[i]["file_name"] for i in ids[:n]])
    items = []
    for iid in ids[:n]:
        im = imgs[iid]
        items.append(Item(id=f"coco_val:{im['file_name']}", path=str(root / "val2017" / im["file_name"]), cluster=im["file_name"], cond="coco",
                          gt=np.asarray(persons.get(iid, []), dtype=np.float32).reshape(-1, 4),
                          ignore=np.asarray(crowd.get(iid, []), dtype=np.float32).reshape(-1, 4), size=(im["width"], im["height"]),
                          extra={"target": "person"}))
    return items


# --------------------------------------------------------------------------------------------- MIO-TCD
MIO_VEHICLE = {"articulated_truck", "bus", "car", "motorcycle", "motorized_vehicle", "pickup_truck", "single_unit_truck", "work_van"}


def miotcd_items(n: int = 3000, seed: int = 0) -> list[Item]:
    """Random sample of the labelled MIO-TCD localization 'train' images (the test labels are not public)."""
    root = raw_dir("miotcd") / "ex"
    gt = defaultdict(list)
    f = next(root.glob("**/gt_train.csv"))
    for row in csv.reader(open(f)):
        gt[row[0]].append((row[1], float(row[2]), float(row[3]), float(row[4]), float(row[5])))
    imgdir = f.parent / "train"
    ids = sorted(k for k in gt if (imgdir / f"{k}.jpg").exists())
    random.Random(seed).shuffle(ids)
    items = []
    for k in ids[:n]:
        veh = [g[1:] for g in gt[k] if g[0] in MIO_VEHICLE]
        ped = [g[1:] for g in gt[k] if g[0] == "pedestrian"]
        items.append(Item(id=f"miotcd:train/{k}.jpg", path=str(imgdir / f"{k}.jpg"), cluster=k, cond="all",
                          gt=np.asarray(veh, dtype=np.float32).reshape(-1, 4), size=(0, 0),
                          extra={"gt_person": np.asarray(ped, dtype=np.float32).reshape(-1, 4).tolist()}))
    return items


# --------------------------------------------------------------------------------------------- WIDER FACE
def prepare_wider() -> Path:
    """Extract WIDER_val.zip + the split annotations (3,226 images, 350 MB)."""
    import zipfile

    root = raw_dir("widerface")
    out = fast_prep("wider")  # NVMe copy
    if not (out / "WIDER_val").exists():
        zipfile.ZipFile(root / "data" / "WIDER_val.zip").extractall(out)
        zipfile.ZipFile(root / "data" / "wider_face_split.zip").extractall(out)
    return out


def wider_items(limit: int | None = None) -> list[Item]:
    root = prepare_wider()
    lines = (next(root.glob("**/wider_face_val_bbx_gt.txt"))).read_text().splitlines()
    img_root = next(root.glob("**/WIDER_val/images"))
    items, i = [], 0
    while i < len(lines):
        name = lines[i].strip()
        n = int(lines[i + 1])
        rows = [list(map(float, lines[i + 2 + k].split())) for k in range(max(n, 1))]
        i += 2 + max(n, 1)
        faces = [r for r in rows if n > 0 and len(r) >= 10 and r[7] == 0 and r[2] > 0 and r[3] > 0]  # drop 'invalid' faces
        if not faces:
            continue
        gt = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in faces], dtype=np.float32)
        items.append(Item(id=f"wider_val:{name}", path=str(img_root / name), cluster=name.split("/")[0], cond="all", gt=gt,
                          extra={"occlusion": [int(r[8]) for r in faces], "blur": [int(r[4]) for r in faces]}))
        if limit and len(items) >= limit:
            break
    return items
