"""Shared helpers for lane D (inspect): paths, taxonomy, image IO, perceptual-hash near-duplicate checks.

Everything that reads downloaded (untrusted) data only ever opens images / JSON / text; nothing from RAW is imported
or executed.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np

from seen.paths import ARTIFACTS, DATA, FAST, PREP, RAW, RUNS

LANE = "inspect"
PREP_I = PREP / LANE            # HDD: small metadata only (splits.json, ...)
FAST_I = FAST / LANE            # NVMe (user-approved, 12 GB budget): staged/resized images, labels, hash caches
RUNS_I = RUNS / LANE
ART_I = ARTIFACTS / LANE
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------------------------- damage taxonomy
# Same strings as seen.vocab.DAMAGE_TYPES. "tire_flat" is only supplied by CarDD (VehiDE/HITL have no flat-tyre class).
DAMAGE_CLASSES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "other"]  # == seen.vocab.DAMAGE_TYPES order differs; names identical

# VehiDE (Vietnamese source labels, see research/data_inventory_inspection.md section 3)
VEHIDE_MAP = {
    "mop_lom": "dent",          # dent
    "tray_son": "scratch",      # paint scratch
    "rach": "crack",            # tear / rip of the body (closest of our classes; stated in the notes)
    "vo_kinh": "glass_shatter", # broken glass
    "be_den": "lamp_broken",    # broken lamp
    "thung": "other",           # puncture / hole
    "mat_bo_phan": "other",     # missing part
}
# Humans in the Loop damage classes
HITL_DMG_MAP = {
    "Dent": "dent", "Scratch": "scratch", "Cracked": "crack",
    "Flaking": "other", "Broken part": "other", "Paint chip": "other", "Missing part": "other", "Corrosion": "other",
}
# SYNDCAR damage classes (test only)
SYND_DMG_MAP = {"broken glass": "glass_shatter", "broken lights": "lamp_broken", "cracks": "crack", "scratches": "scratch"}


# ------------------------------------------------------------------------------------------------ parts taxonomy
# Unified, side-agnostic part vocabulary (proposed to the integrator as seen.vocab.PART_NAMES).
PART_CLASSES = ["front_bumper", "back_bumper", "hood", "trunk", "front_door", "back_door", "fender", "quarter_panel",
                "headlight", "tail_light", "mirror", "windshield", "back_windshield", "front_window", "back_window",
                "wheel", "roof", "grille", "rocker_panel", "license_plate"]
CARPARTS_MAP = {  # Ultralytics carparts-seg (23 classes) -> unified; 'object' is dropped
    "back_bumper": "back_bumper", "back_door": "back_door", "back_left_door": "back_door", "back_right_door": "back_door",
    "back_glass": "back_windshield", "back_left_light": "tail_light", "back_light": "tail_light", "back_right_light": "tail_light",
    "front_bumper": "front_bumper", "front_door": "front_door", "front_left_door": "front_door", "front_right_door": "front_door",
    "front_glass": "windshield", "front_left_light": "headlight", "front_light": "headlight", "front_right_light": "headlight",
    "hood": "hood", "left_mirror": "mirror", "right_mirror": "mirror", "tailgate": "trunk", "trunk": "trunk", "wheel": "wheel",
    "object": None,
}
HITL_PART_MAP = {  # HITL parts and CarDamagePanel (same 21 names, different case/hyphenation) -> unified
    "windshield": "windshield", "back-windshield": "back_windshield", "front-window": "front_window", "back-window": "back_window",
    "front-door": "front_door", "back-door": "back_door", "front-wheel": "wheel", "back-wheel": "wheel",
    "front-bumper": "front_bumper", "back-bumper": "back_bumper", "headlight": "headlight", "tail-light": "tail_light",
    "hood": "hood", "trunk": "trunk", "license-plate": "license_plate", "mirror": "mirror", "roof": "roof", "grille": "grille",
    "rocker-panel": "rocker_panel", "quarter-panel": "quarter_panel", "fender": "fender",
}


# Coarse part families, used where label sets do not line up (SYNDCAR: one micro-car, 'front_panel' / 'left_door' ...).
PART_FAMILY = {
    "front_bumper": "front_end", "hood": "front_end", "grille": "front_end", "back_bumper": "rear_end", "trunk": "rear_end",
    "front_door": "door", "back_door": "door", "front_window": "window", "back_window": "window",
    "fender": "side_panel", "quarter_panel": "side_panel", "headlight": "lamp", "tail_light": "lamp",
    "windshield": "windshield", "back_windshield": "windshield", "wheel": "wheel", "roof": "roof", "mirror": "mirror",
    "rocker_panel": "rocker_panel", "license_plate": "license_plate",
}
FAMILIES = sorted(set(PART_FAMILY.values()))
SYND_PART_FAMILY = {  # SYNDCAR native part names -> family
    "left_side_window": "window", "right_side_window": "window", "left_rear_side_window": "window", "right_rear_side_window": "window",
    "left_front_side_window": "window", "right_front_side_window": "window", "left_door": "door", "right_door": "door",
    "right_rear_quarter_panel": "side_panel", "left_rear_quarter_panel": "side_panel", "right_front_quarter_panel": "side_panel",
    "left_front_quarter_panel": "side_panel", "roof": "roof", "left_rear_wheel": "wheel", "right_rear_wheel": "wheel",
    "left_front_wheel": "wheel", "right_front_wheel": "wheel", "front_windshield": "windshield", "rear_windshield": "windshield",
    "left_mirror": "mirror", "right_mirror": "mirror", "license_plate": "license_plate", "rear_lights": "lamp", "front_lights": "lamp",
    "front_panel": "front_end", "rear_panel": "rear_end", "left_rocker_panel": "rocker_panel", "right_rocker_panel": "rocker_panel",
}
SYND_PARTS = ["left_side_window", "right_side_window", "left_door", "right_door", "right_rear_quarter_panel", "left_rear_quarter_panel",
              "right_front_quarter_panel", "left_rear_side_window", "right_rear_side_window", "left_front_side_window",
              "right_front_side_window", "roof", "left_rear_wheel", "right_rear_wheel", "left_front_wheel", "right_front_wheel",
              "front_windshield", "rear_windshield", "left_mirror", "right_mirror", "license_plate", "rear_lights", "front_lights",
              "front_panel", "rear_panel", "left_front_quarter_panel", "left_rocker_panel", "right_rocker_panel"]  # data_parts.yaml order
SYND_DAMAGE = ["Broken_Glass", "Cracks", "Scratches", "Broken_Lights"]  # data_damage.yaml order
SYND_DAMAGE_MAP = {"Broken_Glass": "glass_shatter", "Cracks": "crack", "Scratches": "scratch", "Broken_Lights": "lamp_broken"}


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())


def read_rgb(path: str | Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise OSError(f"cannot read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def save_jpg(img_rgb: np.ndarray, path: Path, q: int = 93) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])


def image_size(path: str | Path) -> tuple[int, int]:
    """(w, h) without decoding the whole image."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def write_json(obj, path: Path) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(obj, indent=1, default=float))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------ NVMe staging (one pass)
def safe_stem(rid: str) -> str:
    s = rid.replace(":", "__").replace("/", "__")
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in Path(s).stem)


def _stage_one(args):
    src, dst, max_side, q = args
    from PIL import Image
    try:
        if Path(dst).exists():
            with Image.open(dst) as im:
                w, h = im.size
            with Image.open(src) as im0:
                w0, h0 = im0.size
            return src, (w0, h0, w, h), None
        with Image.open(src) as im:
            w0, h0 = im.size
            if im.format == "JPEG":
                im.draft("RGB", (max_side, max_side))  # DCT-domain downscale: much cheaper decode of big JPEGs
            im = im.convert("RGB")
            s = min(1.0, max_side / max(w0, h0))
            if s < 1.0:
                im = im.resize((max(1, round(w0 * s)), max(1, round(h0 * s))), Image.Resampling.LANCZOS)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=q)
            return src, (w0, h0, im.size[0], im.size[1]), None
    except Exception as e:  # noqa: BLE001
        return src, None, f"{type(e).__name__}: {e}"


def stage_images(recs: list[dict], dst_dir: Path, max_side: int, workers: int = 24, quality: int = 92) -> list[dict]:
    """ONE sequential read pass over the raw files (inode order, so the spinning disk does not seek): resize to <= max_side
    and write JPEGs to ``dst_dir`` on the NVMe. Adds r["fast"] (staged path), r["orig_wh"], r["wh"] and r["scale"] to every
    record; records that cannot be read are dropped (and reported). Resumable: existing outputs are reused."""
    from concurrent.futures import ThreadPoolExecutor

    ensure(dst_dir)
    order = {p: i for i, p in enumerate(disk_order([r["path"] for r in recs]))}
    todo = sorted(recs, key=lambda r: order.get(r["path"], 0))
    args = [(r["path"], str(dst_dir / f"{safe_stem(r['id'])}.jpg"), max_side, quality) for r in todo]
    ok, bad = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:  # threads: IO-bound, PIL releases the GIL in decode/resize
        for i, (r, a, (src, wh, err)) in enumerate(zip(todo, args, ex.map(_stage_one, args))):
            if err:
                bad.append((src, err))
                continue
            w0, h0, w, h = wh
            r.update({"fast": a[1], "orig_wh": (w0, h0), "wh": (w, h), "scale": w / w0})
            ok.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  staged {i + 1}/{len(todo)} -> {dst_dir.name}", flush=True)
    if bad:
        print(f"  WARNING {len(bad)} unreadable files dropped, e.g. {bad[:2]}", flush=True)
    return ok


def copy_tree_threaded(src: Path, dst: Path, workers: int = 32) -> int:
    """Copy a directory of many small files with many requests in flight (serial copies crawl on the saturated shared HDD)."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    files = [p for p in Path(src).rglob("*") if p.is_file()]

    def cp(p: Path):
        d = Path(dst) / p.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            shutil.copyfile(p, d)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(cp, files))
    return len(files)


def predict_chunks(model, paths: Sequence[str], bs: int = 4, **kw):
    """Yield Ultralytics results one image at a time, calling model.predict on explicit chunks of ``bs`` paths.
    (A plain predict(list_of_paths) puts the WHOLE list in one batch in this Ultralytics version: 82 images at 1024 px -> OOM.)"""
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), bs):
        yield from model.predict(paths[i:i + bs], stream=False, verbose=False, **kw)


def scale_inst(inst: list[dict], s: float) -> list[dict]:
    """GT instances (poly = list of polygons, box) from original-pixel coordinates to staged coordinates."""
    out = []
    for i in inst:
        e = {k: v for k, v in i.items() if k not in ("poly", "box")}
        if i.get("poly"):
            e["poly"] = [[[x * s, y * s] for x, y in poly] for poly in i["poly"]]
            allp = np.concatenate([np.asarray(p) for p in e["poly"]])
            e["box"] = [float(allp[:, 0].min()), float(allp[:, 1].min()), float(allp[:, 0].max()), float(allp[:, 1].max())]
        else:
            e["poly"] = None
            e["box"] = [v * s for v in i["box"]]
        out.append(e)
    return out


# --------------------------------------------------------------------------------------- perceptual-hash dedup
def disk_order(paths: Sequence[str | Path]) -> list[str]:
    """Paths sorted by inode number: on a spinning disk shared by many jobs, reading files in creation order avoids seeks."""
    from concurrent.futures import ThreadPoolExecutor

    def ino(p):
        try:
            return os.stat(p).st_ino
        except OSError:
            return 0
    ps = [str(p) for p in paths]
    with ThreadPoolExecutor(32) as ex:  # many requests in flight: a serial reader gets ~1 file/s on the saturated shared HDD
        inos = list(ex.map(ino, ps))
    return [p for _, p in sorted(zip(inos, ps), key=lambda t: t[0])]


def _hash_pair_one(path: str) -> tuple[str, list[str] | None, str | None]:
    """[pHash, pHash of the mirrored image] from ONE read; same algorithm as eval.manifest.phash (draft decode, 32x32 grey, 8x8 DCT)."""
    from PIL import Image
    from eval.manifest import _N, _phash_gray32
    try:
        with Image.open(path) as im:
            im.draft("L", (128, 128))
            g = im.convert("L")
        a = g.resize((_N, _N), Image.Resampling.LANCZOS)
        b = g.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((_N, _N), Image.Resampling.LANCZOS)
        return path, [_phash_gray32(np.asarray(a)), _phash_gray32(np.asarray(b))], None
    except Exception as e:  # noqa: BLE001
        return path, None, f"{type(e).__name__}: {e}"


def hash_with_flip(paths: Sequence[str | Path], workers: int = 6, batch: int = 400) -> dict[str, list[str]]:
    """{path: [phash, phash_of_hflip]} (the flip catches mirrored re-uploads / Roboflow augmentations). One read per file,
    files read in inode order, results appended to PREP/inspect/hashes/pair_cache.jsonl after every batch (restartable)."""
    from concurrent.futures import ProcessPoolExecutor

    paths = [str(p) for p in paths]
    cache_f = ensure(FAST_I / "hashes") / "pair_cache.jsonl"
    cache: dict[str, list[str]] = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            try:
                r = json.loads(line)
                cache[r["k"]] = r["h"]
            except Exception:  # noqa: BLE001
                pass

    def key(p: str) -> str:
        st = os.stat(p)
        return f"{p}|{st.st_size}|{int(st.st_mtime)}"

    keys = {}
    for p in paths:
        try:
            keys[p] = key(p)
        except OSError:
            pass
    todo = disk_order([p for p in paths if p in keys and keys[p] not in cache])
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(todo), batch):
                res = list(ex.map(_hash_pair_one, todo[i:i + batch], chunksize=8))
                with cache_f.open("a") as f:
                    for p, h, e in res:
                        if h:
                            cache[keys[p]] = h
                            f.write(json.dumps({"k": keys[p], "h": h}) + "\n")
                print(f"  hashed {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return {p: cache[keys[p]] for p in paths if p in keys and keys[p] in cache}


def near_dup_matches(a_paths: Sequence[str], b_paths: Sequence[str], threshold: int = 8) -> dict[str, tuple[str, int]]:
    """For each path in ``a`` that has a pHash near-duplicate (original or mirrored) in ``b``: {a: (b_match, dist)}."""
    from eval.manifest import near_pairs

    ha = hash_with_flip(a_paths)
    hb = hash_with_flip(b_paths)
    a_keys = [p for p in a_paths if str(p) in ha]
    b_keys = [p for p in b_paths if str(p) in hb]
    # compare a-plain against b-plain and b-flip (this covers a-flip vs b-plain by symmetry)
    a_hash = [ha[str(p)][0] for p in a_keys]
    b_hash, b_owner = [], []
    for j, p in enumerate(b_keys):
        for h in hb[str(p)]:
            b_hash.append(h)
            b_owner.append(j)
    out: dict[str, tuple[str, int]] = {}
    for i, j, d in near_pairs(a_hash, b_hash, threshold):
        p = str(a_keys[i])
        if p not in out or d < out[p][1]:
            out[p] = (str(b_keys[b_owner[j]]), d)
    return out


def group_near_dups(paths: Sequence[str], threshold: int = 8) -> list[int]:
    """Cluster id per path: near-duplicates (original or mirrored) share a cluster (union-find on pHash)."""
    from eval.manifest import near_pairs

    h = hash_with_flip(paths)
    keys = [str(p) for p in paths]
    parent = list(range(len(keys)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    flat, owner = [], []
    for i, k in enumerate(keys):
        for x in h.get(k, []):
            flat.append(x)
            owner.append(i)
    for a, b, _ in near_pairs(flat, None, threshold):
        ra, rb = find(owner[a]), find(owner[b])
        if ra != rb:
            parent[ra] = rb
    return [find(i) for i in range(len(keys))]


def seeded_fraction(key: str, seed: str = "inspect-v1") -> float:
    import hashlib
    return int(hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def rel_data(p: str | Path) -> str:
    s = str(p)
    r = str(DATA)
    return s[len(r) + 1:] if s.startswith(r + "/") else s


def manifest_records(items: Iterable[dict]) -> list[dict]:
    """Normalise to eval/manifests schema: id, path (relative to SEEN_DATA), source, dataset, split, phash, label."""
    return [dict(i) for i in items]
