"""Assemble the labelled scene-router set from public sources (run on the box; needs internet).

Scenes (seen.vocab.SCENES): gate | bay | wash | street | closeup | motorcycle | unknown. Definitions used for labelling:
  gate       vehicle entrance/exit: barrier, gate, toll/checkpoint, car-park entrance, seen from a camera at the gate
  bay        inside a workshop/garage/tyre shop/service bay: car on a lift or among tools, mechanics
  wash       car wash: foam, brushes, hoses, wash tunnel/bay, a car being washed
  street     wide fixed-camera or street view with vehicles: roads, intersections, parking lots (small vehicles)
  closeup    one car (or a part) filling the frame: bodywork/wheel/bonnet/engine bay/lamp close-ups, whole-car photos
  motorcycle a motorcycle / scooter is the subject
  unknown    anything without a vehicle subject: people, animals, food, rooms, landscapes, screens, documents
Every record keeps source / licence / url so the split is by *source* (``docs/lanes/flow_splits.md``).

    python -m lanes.flow.router_data commons     # Wikimedia Commons (CC licences recorded per file)
    python -m lanes.flow.router_data coco        # COCO val2017 (street / motorcycle / unknown)
    python -m lanes.flow.router_data hf          # Stanford Cars (closeup), car-parts close-ups
    python -m lanes.flow.router_data cams        # frames from our fixed-camera sets (street): UA-DETRAC, PKLot, NDISPark, CNR-EXT
    python -m lanes.flow.router_data build       # write manifest.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
from pathlib import Path

from lanes.flow.common import prep_dir, raw_dir

UA = {"User-Agent": "SeenResearchBot/0.1 (hackathon research; contact amirkhon@humanola.com)"}
API = "https://commons.wikimedia.org/w/api.php"
OK_MIME = {"image/jpeg", "image/png", "image/webp"}
BAD_LICENSE = re.compile(r"fair use|non-free|copyrighted", re.I)

# (scene, source, category or search query, kind, per-source cap)
COMMONS: list[tuple[str, str, str, str, int]] = [
    # gate
    ("gate", "commons_boom_barriers", "Boom barriers", "cat", 160),
    ("gate", "commons_car_park_entrances", "Car park entrances", "cat", 60),
    ("gate", "commons_gatehouses", "Gatehouses", "cat", 80),
    ("gate", "commons_checkpoints", "Checkpoints", "cat", 80),
    ("gate", "commons_search_gate", "car waiting at barrier gate", "search", 60),
    ("gate", "commons_search_toll", "toll booth car lane barrier", "search", 60),
    # third pass: bitmap-only searches for a vehicle at a barrier / gate
    ("gate", "gate3_search_barrier", "car stopped at parking barrier entrance", "search", 40),
    ("gate", "gate3_search_gate", "car entering through automatic gate", "search", 40),
    ("gate", "gate3_search_checkpoint", "vehicle checkpoint barrier car stopped", "search", 40),
    ("gate", "gate3_search_ticket", "car park ticket machine barrier car", "search", 40),
    ("gate", "gate3_search_plate", "car front view number plate at entrance gate", "search", 40),
    # second pass (the first gate categories above were checked by eye and are mostly junk: floating booms, chains, rail crossings)
    ("gate", "gate2_toll_plazas", "Toll plazas", "cat", 70),
    ("gate", "gate2_toll_gates", "Toll gates", "cat", 90),
    ("gate", "gate2_tollbooths", "Tollbooths", "cat", 80),
    ("gate", "gate2_security_gates", "Security gates", "cat", 40),
    ("gate", "gate2_customs", "Customs checkpoints", "cat", 30),
    ("gate", "gate2_anpr", "Automatic number plate recognition", "cat", 80),
    ("gate", "gate2_etc", "Electronic toll collection", "cat", 70),
    # bay
    ("bay", "commons_repair_shops", "Automobile repair shops", "cat", 200),
    ("bay", "commons_lifts", "Stationary automobile service lifts", "cat", 60),
    ("bay", "commons_four_post_lifts", "Four post lifts", "cat", 40),
    ("bay", "commons_tire_shops", "Tire shops", "cat", 80),
    ("bay", "commons_garages", "Garages", "cat", 120),
    ("bay", "commons_search_bay", "mechanic working on car in workshop", "search", 80),
    ("bay", "commons_search_lift", "car on lift in garage workshop", "search", 60),
    # wash
    ("wash", "commons_car_washes", "Car washes", "cat", 200),
    ("wash", "commons_search_wash", "car wash foam brushes", "search", 80),
    ("wash", "commons_search_wash2", "washing a car with a hose", "search", 80),
    # street (wide views of roads / intersections / parking lots with vehicles)
    ("street", "commons_road_traffic", "Road traffic", "cat", 100),
    ("street", "commons_streets_tashkent", "Streets in Tashkent", "cat", 90),
    ("street", "commons_roads_uz", "Roads in Uzbekistan", "cat", 50),
    ("street", "commons_traffic_cameras", "Traffic cameras", "cat", 50),
    ("street", "commons_rush_hour", "Rush hour", "cat", 40),
    ("street", "commons_intersections", "Intersections", "cat", 50),
    ("street", "commons_car_parks", "Car parks", "cat", 100),
    # motorcycle
    ("motorcycle", "commons_ducati", "Ducati motorcycles", "cat", 120),
    ("motorcycle", "commons_ducati_monster", "Ducati Monster", "cat", 60),
    ("motorcycle", "commons_ducati_multistrada", "Ducati Multistrada", "cat", 40),
    ("motorcycle", "commons_ducati_panigale", "Ducati Panigale", "cat", 30),
    ("motorcycle", "commons_scooters", "Scooters", "cat", 60),
    # closeup
    ("closeup", "commons_hoods", "Automobile hoods", "cat", 100),
    ("closeup", "commons_wheels", "Automobile wheels", "cat", 80),
    ("closeup", "commons_engines", "Automobile engines", "cat", 100),
    ("closeup", "commons_search_damage", "damaged car dent scratch bodywork close-up", "search", 60),
    # unknown (no vehicle subject)
    ("unknown", "commons_search_food", "plate of food", "search", 50),
    ("unknown", "commons_search_landscape", "mountain landscape", "search", 50),
    ("unknown", "commons_search_office", "office desk computer", "search", 50),
    ("unknown", "commons_search_animals", "dog portrait", "search", 40),
    ("unknown", "commons_search_doc", "scanned document page text", "search", 40),
]


def _get(session, params, tries=5):
    import requests

    for i in range(tries):
        try:
            r = session.get(API, params={**params, "format": "json"}, headers=UA, timeout=40)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            return r.json()
        except Exception:  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return {}


def cat_files(session, cat: str, cap: int, depth: int = 2) -> list[str]:
    """File titles in a category and its subcategories (BFS, depth-limited)."""
    seen, files, frontier = set(), [], [(cat, 0)]
    while frontier and len(files) < cap * 3:
        c, d = frontier.pop(0)
        if c in seen:
            continue
        seen.add(c)
        cont = {}
        while True:
            j = _get(session, {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{c}", "cmtype": "file|subcat",
                               "cmlimit": "500", **cont})
            for m in j.get("query", {}).get("categorymembers", []):
                if m["ns"] == 6:
                    files.append(m["title"])
                elif m["ns"] == 14 and d < depth:
                    frontier.append((m["title"].split(":", 1)[1], d + 1))
            if "continue" in j:
                cont = j["continue"]
            else:
                break
            time.sleep(0.2)
        time.sleep(0.2)
    return files


def search_files(session, q: str, cap: int) -> list[str]:
    out, cont = [], {}
    while len(out) < cap * 2:
        j = _get(session, {"action": "query", "list": "search", "srsearch": q + " filetype:bitmap", "srnamespace": "6", "srlimit": "50", **cont})
        out += [m["title"] for m in j.get("query", {}).get("search", [])]
        if "continue" in j:
            cont = j["continue"]
        else:
            break
        time.sleep(0.2)
    return out


def file_info(session, titles: list[str]) -> dict[str, dict]:
    res = {}
    for i in range(0, len(titles), 40):
        j = _get(session, {"action": "query", "prop": "imageinfo", "titles": "|".join(titles[i:i + 40]), "iiprop": "url|mime|size|extmetadata",
                           "iiurlwidth": "800", "iiextmetadatafilter": "LicenseShortName|Artist|AttributionRequired|Credit"})
        for p in j.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if ii and ii.get("mime") in OK_MIME:
                md = ii.get("extmetadata", {})
                lic = md.get("LicenseShortName", {}).get("value", "")
                if lic and not BAD_LICENSE.search(lic):
                    res[p["title"]] = {"url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl"), "license": lic,
                                       "artist": re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", ""))[:120], "w": ii.get("thumbwidth"), "h": ii.get("thumbheight")}
        time.sleep(0.25)
    return res


def download(session, url: str, dst: Path, tries=4) -> bool:
    if dst.exists():
        return True
    for i in range(tries):
        try:
            r = session.get(url, headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(6 * (i + 1))
                continue
            if r.status_code == 200 and len(r.content) > 2000:
                dst.write_bytes(r.content)
                return True
            return False
        except Exception:  # noqa: BLE001
            time.sleep(2 * (i + 1))
    return False


def run_commons(only: str | None = None):
    import requests

    out = prep_dir("router", "images")
    mf = prep_dir("router") / "records_commons.jsonl"
    done = set()
    if mf.exists():
        done = {json.loads(l)["id"] for l in mf.read_text().splitlines()}
    s = requests.Session()
    seen_titles = set()
    with open(mf, "a") as fo:
        for scene, source, q, kind, cap in COMMONS:
            if only and only != scene and only not in source:
                continue
            titles = cat_files(s, q, cap) if kind == "cat" else search_files(s, q, cap)
            titles = [t for t in dict.fromkeys(titles) if t not in seen_titles]
            random.Random(0).shuffle(titles)
            info = file_info(s, titles[: cap * 2])
            n = 0
            for t, m in info.items():
                if n >= cap:
                    break
                rid = f"{source}:{hashlib.md5(t.encode()).hexdigest()[:12]}"
                if rid in done:
                    n += 1
                    continue
                dst = out / source / (hashlib.md5(t.encode()).hexdigest()[:12] + ".jpg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                if download(s, m["url"], dst):
                    seen_titles.add(t)
                    n += 1
                    fo.write(json.dumps({"id": rid, "path": str(dst), "label": scene, "source": source, "dataset": "wikimedia_commons", "title": t,
                                         "license": m["license"], "artist": m["artist"], "url": m["page"]}) + "\n")
                    fo.flush()
                time.sleep(0.15)
            print(scene, source, "found", len(titles), "saved", n, flush=True)


COCO_ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"


def coco_annotations() -> dict:
    """instances_val2017.json (25 MB) read out of the remote 241 MB annotations zip with HTTP range requests."""
    from remotezip import RemoteZip

    root = raw_dir("coco")
    f = root / "annotations" / "instances_val2017.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        with RemoteZip(COCO_ANN_ZIP) as z:
            f.write_bytes(z.read("annotations/instances_val2017.json"))
    return json.loads(f.read_text())


def fetch_coco_images(file_names: list[str]) -> None:
    """Fetch only the listed images from the remote val2017.zip (range requests), skipping files already on disk."""
    from remotezip import RemoteZip

    root = raw_dir("coco") / "val2017"
    root.mkdir(parents=True, exist_ok=True)
    todo = [f for f in file_names if not (root / f).exists()]
    if not todo:
        return
    with RemoteZip(COCO_VAL_ZIP) as z:
        for i, f in enumerate(todo):
            (root / f).write_bytes(z.read(f"val2017/{f}"))
            if i % 100 == 0:
                print("coco fetched", i, "/", len(todo), flush=True)


def run_coco(n_street=300, n_moto=200, n_unknown=500):
    """COCO val2017 (images from Flickr, mixed CC; annotations CC BY 4.0). Used for the router set only, never to train a detector."""
    root = raw_dir("coco")
    js = coco_annotations()
    cats = {c["id"]: c["name"] for c in js["categories"]}
    by_img: dict[int, list] = {}
    for a in js["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    imgs = {i["id"]: i for i in js["images"]}
    veh = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat"}
    street, moto, unk = [], [], []
    for iid, im in sorted(imgs.items()):
        anns = by_img.get(iid, [])
        area = im["width"] * im["height"]
        names = [cats[a["category_id"]] for a in anns]
        if not any(n in veh for n in names):
            unk.append(iid)
            continue
        mo = [a for a in anns if cats[a["category_id"]] == "motorcycle"]
        cars = [a for a in anns if cats[a["category_id"]] in ("car", "truck", "bus")]
        if mo and sum(a["area"] for a in mo) / area > 0.08 and not cars:
            moto.append(iid)
        elif len(cars) >= 2 and max(a["area"] for a in cars) / area < 0.25 and not mo:
            street.append(iid)
    rng = random.Random(0)
    chosen = []
    for scene, lst, n in (("street", street, n_street), ("motorcycle", moto, n_moto), ("unknown", unk, n_unknown)):
        rng.shuffle(lst)
        chosen += [(scene, i) for i in lst[:n]]
        print("coco", scene, len(lst), "->", min(n, len(lst)))
    fetch_coco_images([imgs[i]["file_name"] for _, i in chosen])
    recs = [{"id": f"coco_{scene}:{imgs[i]['file_name']}", "path": str(root / "val2017" / imgs[i]["file_name"]), "label": scene, "source": f"coco_{scene}",
             "dataset": "coco_val2017", "license": "Flickr (mixed CC BY/BY-SA/BY-NC); annotations CC BY 4.0", "url": imgs[i].get("flickr_url", "")} for scene, i in chosen]
    (prep_dir("router") / "records_coco.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def run_hf(n_cars=400):
    """Stanford Cars test split (HF parquet, 2 shards): whole-car photos = 'closeup'."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    outd = prep_dir("router", "images")
    recs = []
    path = hf_hub_download("tanganke/stanford_cars", "data/test-00000-of-00002.parquet", repo_type="dataset")
    tbl = pq.read_table(path, columns=["image"])
    idx = list(range(tbl.num_rows))
    random.Random(0).shuffle(idx)
    col = tbl.column("image")
    for i in idx[:n_cars]:
        d = col[i].as_py()
        im = Image.open(io.BytesIO(d["bytes"])).convert("RGB")
        im.thumbnail((800, 800))
        p = outd / "stanford_cars" / f"{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, quality=90)
        recs.append({"id": f"stanford_cars:{i}", "path": str(p), "label": "closeup", "source": "stanford_cars", "dataset": "tanganke/stanford_cars",
                     "license": "Stanford Cars: research use (non-commercial)", "url": "hf:tanganke/stanford_cars"})
    print("stanford_cars", len(recs))
    (prep_dir("router") / "records_hf.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def run_cams(n_per=120):
    """Fixed-camera frames (street): UA-DETRAC test, PKLot, NDISPark (train+val), CNR-EXT. Same data as the detector tests."""
    import shutil

    rng = random.Random(0)
    recs = []
    outd = prep_dir("router", "images")

    def add(source, files, dataset, lic):
        rng.shuffle(files)
        for f in files[:n_per]:
            p = outd / source / (hashlib.md5(str(f).encode()).hexdigest()[:12] + ".jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                import cv2

                im = cv2.imread(str(f))
                if im is None:
                    continue
                s = 800 / max(im.shape[:2])
                if s < 1:
                    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recs.append({"id": f"{source}:{p.stem}", "path": str(p), "label": "street", "source": source, "dataset": dataset, "license": lic, "url": ""})
        print(source, len(files))

    from lanes.flow.common import fast_prep

    add("detrac_cam", sorted(fast_prep("detrac", "test").glob("MVI_*/img*.jpg")), "UA-DETRAC (HF mirror)", "not stated (evaluation only)")
    nd = fast_prep("ndispark")
    add("ndispark_cam", sorted(nd.glob("*/imgs/*.jpg")), "NDISPark", "ODC-By 1.0")
    pk = list((raw_dir("pklot") / "PKLot" / "PKLot").glob("*/*/*/*.jpg"))
    add("pklot_cam", pk, "PKLot", "CC BY 4.0")
    cnr = list(fast_prep("cnrext_full").glob("**/FULL_IMAGE_1000x750/**/*.jpg"))
    add("cnrext_cam", cnr, "CNRPark-EXT", "ODbL 1.0")
    (prep_dir("router") / "records_cams.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def build():
    """manifest.jsonl = all records, with Commons gate/bay/wash images restricted to the hand-curated keep lists (lanes/flow/curation.json)."""
    d = prep_dir("router")
    cj = json.loads((Path(__file__).parent / "curation.json").read_text())
    cur, drop_src = cj["commons_keep_ids"], set(cj.get("drop_sources", []))
    keep = {i for v in cur.values() for i in v}
    recs = []
    for f in sorted(d.glob("records_*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    good, seen = [], set()
    relabel = cj.get("relabel_sources", {})
    sub = cj.get("subsample_sources", {})
    counters: dict = {}
    for r in recs:
        if r["id"] in seen or r["source"] in drop_src or not Path(r["path"]).exists():
            continue
        if r["dataset"] == "wikimedia_commons" and r["label"] in cur and r["id"] not in keep:
            continue
        n = None
        for k, v in sub.items():
            if k.startswith("prefix:") and r["source"].startswith(k[7:]) or k == r["source"]:
                n = v
        if n:
            c = counters.get(r["source"], 0)
            counters[r["source"]] = c + 1
            if c % n:
                continue
        if r["source"] in relabel:
            r = {**r, "label": relabel[r["source"]]}
        seen.add(r["id"])
        good.append(r)
    # Lane G audit: nothing that is (a near-duplicate of) a held-out jury / eval item may be used for fitting
    from eval.manifest import filter_train

    kept_paths, dropped = filter_train([r["path"] for r in good])
    ks = set(kept_paths)
    (d / "heldout_dropped.json").write_text(json.dumps(dropped, indent=1))
    print("dropped as near-duplicates of held-out eval items:", len(dropped), [(x["component"], x.get("match"), Path(x["path"]).name) for x in dropped][:12])
    good = [r for r in good if r["path"] in ks]
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in good))
    (d / "train_files.txt").write_text("".join(r["path"] + "\n" for r in good))
    from collections import Counter

    print(len(good), Counter(r["label"] for r in good))
    print(sorted(Counter((r["label"], r["source"]) for r in good).items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["commons", "coco", "hf", "cams", "build"])
    ap.add_argument("--only")
    a = ap.parse_args()
    {"commons": lambda: run_commons(a.only), "coco": run_coco, "hf": run_hf, "cams": run_cams, "build": build}[a.what]()
