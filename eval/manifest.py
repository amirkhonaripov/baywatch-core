"""Held-out test manifests, perceptual-hash dedup, split-by-source and the leakage audit (Lane G).

Manifest format: ``eval/manifests/<component>.jsonl``, one JSON object per line::

    {"id": "ndispark:night/cam3_0012.jpg",   # globally unique, stable: "<dataset>:<native id or relative path>"
     "path": "raw/ndispark/test/cam3_0012.jpg",  # relative to seen.paths.DATA (absolute paths and URLs also allowed)
     "source": "ndispark/cam3",           # the unit we split by (camera, video, project, uploader, dataset)
     "dataset": "ndispark",
     "split": "test",                     # "test" (held out), "quarantine" (near-duplicate of test: also excluded)
     "phash": "c3a1...",                  # 64-bit pHash as 16 hex chars (list of them for a video)
     "label": ...}                        # any component-specific ground truth (boxes, text, state, ...)

Every item in a manifest whose split is not "train" is held out: trainers MUST exclude it.

Trainer API (import this; it never imports lane code)::

    from eval.manifest import excluded_ids, excluded_paths, filter_train
    keep, dropped = filter_train(my_train_paths, components=["vehicle_det"])   # drops path/id matches + pHash near-dups

Audit (Lane G runs this on every lane's training list before accepting its numbers)::

    python -m eval.manifest audit --components vehicle_det,plate_ocr --train-list runs/flow/exp1/train_files.txt \
        --out runs/eval/leakage/flow_exp1.json

``train_files.txt`` = one image/video path per line (absolute, or relative to SEEN_DATA). A ``.jsonl`` with a
``path`` field also works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
REQUIRED = ("id", "path", "source")
HELD_OUT_SPLITS = {"test", "quarantine", "val_heldout"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv", ".m4v"}
STRICT, LOOSE = 4, 8  # pHash Hamming thresholds: <=4 near-identical, <=8 likely the same photo (resized/re-encoded)


def _data_root() -> Path:
    from seen.paths import DATA

    return DATA


# ------------------------------------------------------------------------------------------------- manifests
def manifest_path(component: str) -> Path:
    return MANIFEST_DIR / f"{component}.jsonl"


def components() -> list[str]:
    return sorted(p.stem for p in MANIFEST_DIR.glob("*.jsonl"))


def read_manifest(component_or_path: str | Path) -> list[dict]:
    p = Path(component_or_path)
    if not p.suffix:
        p = manifest_path(str(component_or_path))
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines()):
        line = line.strip()
        if line and not line.startswith("//"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i + 1}: {e}") from e
    return out


def validate(records: Sequence[dict]) -> list[str]:
    errs, seen = [], set()
    for i, r in enumerate(records):
        for k in REQUIRED:
            if not r.get(k):
                errs.append(f"record {i}: missing {k!r}")
        if r.get("id") in seen:
            errs.append(f"record {i}: duplicate id {r.get('id')!r}")
        seen.add(r.get("id"))
    return errs


def write_manifest(component: str, records: Sequence[dict], path: Path | None = None) -> Path:
    errs = validate(records)
    if errs:
        raise ValueError(f"manifest {component}: " + "; ".join(errs[:10]))
    p = path or manifest_path(component)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: r["id"])
    p.write_text("".join(json.dumps({"split": "test", **r}, ensure_ascii=False, sort_keys=False) + "\n" for r in rows))
    return p


def resolve(path: str | Path) -> Path:
    s = str(path)
    if s.startswith(("http://", "https://")):
        raise ValueError(f"cannot open URL as a local file: {s}")
    p = Path(s).expanduser()
    return p if p.is_absolute() else _data_root() / p


def rel_to_data(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(_data_root().resolve()))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------------------------------- hashing
_N = 32
_DCT = np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))


def _phash_gray32(a: np.ndarray) -> str:
    d = _DCT @ a.astype(np.float64) @ _DCT.T
    low = d[:8, :8].ravel()
    bits = low > np.median(low)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:016x}"


def phash(img: str | Path | np.ndarray | Any) -> str:
    """64-bit DCT perceptual hash (same algorithm as imagehash.phash: 32x32 grey, 8x8 low DCT > median).

    Accepts a path, a PIL image or an RGB/grey numpy array. Own implementation so the Mac and the box agree.
    """
    from PIL import Image

    if isinstance(img, (str, Path)):
        with Image.open(img) as im:
            im.draft("L", (128, 128))  # fast JPEG decode at reduced scale; deterministic for a given file
            g = im.convert("L").resize((_N, _N), Image.Resampling.LANCZOS)
            return _phash_gray32(np.asarray(g))
    if isinstance(img, np.ndarray):
        im = Image.fromarray(img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8))
    else:
        im = img
    return _phash_gray32(np.asarray(im.convert("L").resize((_N, _N), Image.Resampling.LANCZOS)))


def phash_video(path: str | Path, n: int = 5) -> list[str]:
    """pHashes of ``n`` evenly spaced frames (5 %..95 % of the duration). PyAV first (VP9/webm safe), cv2 fallback."""
    frames: list[np.ndarray] = []
    try:
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else 0.0)
            targets = list(np.linspace(0.05 * dur, 0.95 * dur, n)) if dur > 0 else []
            k = 0
            for fr in c.decode(s):
                if not targets:
                    frames.append(fr.to_ndarray(format="rgb24"))
                    if len(frames) >= n:
                        break
                    continue
                if fr.time is not None and fr.time + 1e-6 >= targets[k]:
                    frames.append(fr.to_ndarray(format="rgb24"))
                    k += 1
                    if k >= len(targets):
                        break
    except Exception:
        frames = []
    if not frames:
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for i in (np.linspace(0.05 * total, 0.95 * total, n).astype(int) if total > 0 else []):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f[:, :, ::-1])
        cap.release()
    return [phash(f) for f in frames]


def hash_file(path: str | Path) -> list[str]:
    p = Path(path)
    if p.suffix.lower() in VIDEO_EXTS:
        return phash_video(p)
    return [phash(p)]


def _hash_one(path: str) -> tuple[str, list[str], str | None]:
    try:
        return path, hash_file(path), None
    except Exception as e:  # unreadable file: reported, not fatal
        return path, [], f"{type(e).__name__}: {e}"


def _cache_file() -> Path:
    """pHash cache: on the NVMe staging dir when it exists (HDD is IO-saturated), else runs/eval/."""
    if os.environ.get("SEEN_PHASH_CACHE"):
        return Path(os.environ["SEEN_PHASH_CACHE"])
    from seen import paths

    fast = getattr(paths, "FAST", None)
    if fast is not None and Path(fast).exists():
        return Path(fast) / "eval" / "phash_cache.jsonl"
    return paths.RUNS / "eval" / "phash_cache.jsonl"


def hash_many(paths: Sequence[str | Path], workers: int = 8, cache: bool = True) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Hash many files in parallel with an on-disk cache keyed by (path, size, mtime). Returns (hashes, errors)."""
    paths = [str(p) for p in paths]
    cached: dict[str, list[str]] = {}
    keys: dict[str, str] = {}
    cf = _cache_file()
    if cache and cf.exists():
        for line in cf.read_text().splitlines():
            try:
                r = json.loads(line)
                cached[r["k"]] = r["h"]
            except Exception:
                pass
    todo = []
    out: dict[str, list[str]] = {}
    for p in paths:
        try:
            st = os.stat(p)
            k = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            k = p
        keys[p] = k
        if k in cached:
            out[p] = cached[k]
        else:
            todo.append(p)
    errors: dict[str, str] = {}
    if todo:
        results = []
        if workers > 1 and len(todo) > 16:
            with ProcessPoolExecutor(max_workers=min(workers, 8)) as ex:
                results = list(ex.map(_hash_one, todo, chunksize=32))
        else:
            results = [_hash_one(p) for p in todo]
        new = []
        for p, h, err in results:
            if err:
                errors[p] = err
                continue
            out[p] = h
            new.append(json.dumps({"k": keys[p], "h": h}))
        if cache and new:
            try:
                cf.parent.mkdir(parents=True, exist_ok=True)
                with cf.open("a") as f:
                    f.write("\n".join(new) + "\n")
            except OSError:
                pass
    return out, errors


def add_phash(records: list[dict], workers: int = 8, path_key: str = "path") -> list[dict]:
    """Fill ``phash`` for records that lack it (images: str; videos: list of str). ``path_key`` lets callers hash a
    staged copy (e.g. on the NVMe) while the manifest keeps the canonical path."""
    need = [r for r in records if not r.get("phash")]
    paths = [str(resolve(r[path_key])) for r in need]
    hashes, errors = hash_many(paths, workers)
    for r, p in zip(need, paths):
        if p in hashes:
            h = hashes[p]
            r["phash"] = h[0] if len(h) == 1 and Path(p).suffix.lower() not in VIDEO_EXTS else h
        else:
            r["phash_error"] = errors.get(p, "not hashed")
    return records


def _as_list(h) -> list[str]:
    if not h:
        return []
    return [h] if isinstance(h, str) else list(h)


def _ints(hexes: Sequence[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def hamming(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def near_pairs(a: Sequence[str], b: Sequence[str] | None = None, threshold: int = LOOSE) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with Hamming(a[i], b[j]) <= threshold. If ``b`` is None, pairs within ``a`` (i < j)."""
    A = _ints(a)
    B = A if b is None else _ints(b)
    if len(A) == 0 or len(B) == 0:
        return []
    out = []
    block = max(1, int(2e7 // max(len(B), 1)))
    for s in range(0, len(A), block):
        x = A[s:s + block, None] ^ B[None, :]
        d = np.bitwise_count(x)
        ii, jj = np.nonzero(d <= threshold)
        for i, j in zip(ii, jj):
            gi = s + int(i)
            if b is None and gi >= int(j):
                continue
            out.append((gi, int(j), int(d[i, j])))
    return out


# ----------------------------------------------------------------------------------------- dedup and split
class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _flat_hashes(records: Sequence[dict]) -> tuple[list[str], list[int]]:
    hs, owner = [], []
    for i, r in enumerate(records):
        for h in _as_list(r.get("phash")):
            hs.append(h)
            owner.append(i)
    return hs, owner


def dedup(records: Sequence[dict], threshold: int = STRICT) -> tuple[list[dict], list[dict]]:
    """Drop near-duplicates across *all* datasets (first occurrence wins; order = priority).

    Returns (kept, removed); each removed record gets ``dup_of`` = id of the kept twin and ``dup_dist``.
    """
    hs, owner = _flat_hashes(records)
    uf = _UF(len(records))
    dist: dict[int, int] = {}
    for i, j, d in near_pairs(hs, None, threshold):
        a, b = owner[i], owner[j]
        if a != b:
            uf.union(a, b)
            dist[max(a, b)] = min(dist.get(max(a, b), 99), d)
    kept, removed = [], []
    for i, r in enumerate(records):
        root = uf.find(i)
        if root == i:
            kept.append(r)
        else:
            removed.append({**r, "dup_of": records[root]["id"], "dup_dist": dist.get(i)})
    return kept, removed


def _bucket(key: str, seed: str) -> float:
    return int(hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def split_by_source(records: Sequence[dict], *, test_frac: float = 0.2, test_sources: Iterable[str] | None = None,
                    seed: str = "seen-eval-v1", threshold: int = LOOSE,
                    source_fn: Callable[[dict], str] | None = None) -> list[dict]:
    """Assign ``split`` in {"train", "test", "quarantine"} so that no source and no near-duplicate straddles splits.

    Groups = sources joined by pHash near-duplicate links (<= threshold) across sources. With ``test_sources`` the
    groups containing those sources become test (their non-test-source members become "quarantine": excluded from
    both). Otherwise groups are ordered by a seeded hash and taken into test until ``test_frac`` of items is reached.
    """
    src = source_fn or (lambda r: r["source"])
    sources = sorted({src(r) for r in records})
    sidx = {s: i for i, s in enumerate(sources)}
    uf = _UF(len(sources))
    hs, owner = _flat_hashes(records)
    for i, j, _ in near_pairs(hs, None, threshold):
        uf.union(sidx[src(records[owner[i]])], sidx[src(records[owner[j]])])
    groups: dict[int, list[str]] = defaultdict(list)
    for s in sources:
        groups[uf.find(sidx[s])].append(s)
    count = defaultdict(int)
    for r in records:
        count[src(r)] += 1
    test_src: set[str] = set()
    quarantine_src: set[str] = set()
    if test_sources is not None:
        want = set(test_sources)
        for g in groups.values():
            if want & set(g):
                test_src |= want & set(g)
                quarantine_src |= set(g) - want
    else:
        total = sum(count.values())
        order = sorted(groups.values(), key=lambda g: _bucket(min(g), seed))
        acc = 0
        for g in order:
            if acc >= test_frac * total:
                break
            test_src |= set(g)
            acc += sum(count[s] for s in g)
    out = []
    for r in records:
        s = src(r)
        split = "test" if s in test_src else "quarantine" if s in quarantine_src else "train"
        out.append({**r, "split": split})
    return out


def records_from_dir(root: str | Path, dataset: str, *, source_fn: Callable[[Path], str] | None = None,
                     label_fn: Callable[[Path], Any] | None = None, exts: set[str] | None = None) -> list[dict]:
    """Build manifest records for every image/video under ``root``. Default source = first sub-directory."""
    root = Path(root)
    exts = exts or (IMAGE_EXTS | VIDEO_EXTS)
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        rel = p.relative_to(root)
        src = source_fn(p) if source_fn else f"{dataset}/{rel.parts[0] if len(rel.parts) > 1 else '_'}"
        r = {"id": f"{dataset}:{rel.as_posix()}", "path": rel_to_data(p), "source": src, "dataset": dataset}
        if label_fn:
            r["label"] = label_fn(p)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------- trainer API
def held_out(component: str) -> list[dict]:
    return [r for r in read_manifest(component) if r.get("split", "test") in HELD_OUT_SPLITS]


def excluded_ids(component: str) -> set[str]:
    """Ids every trainer must exclude for ``component`` (empty set if G has not published that manifest yet)."""
    return {r["id"] for r in held_out(component)}


def excluded_paths(component: str) -> set[str]:
    return {r["path"] for r in held_out(component)}


def excluded_hashes(component: str) -> list[str]:
    return [h for r in held_out(component) for h in _as_list(r.get("phash"))]


def _norm(p: str) -> str:
    s = str(p)
    root = str(_data_root())
    return s[len(root) + 1:] if s.startswith(root + "/") else s


def filter_train(paths: Sequence[str | Path], components: Sequence[str] | None = None,
                 threshold: int = LOOSE, ids: Sequence[str] | None = None) -> tuple[list[str], list[dict]]:
    """Remove held-out items (by id, path, or pHash near-duplicate) from a training list.

    ``components`` defaults to every manifest. ``ids`` (optional, parallel to ``paths``) lets trainers match by id.
    Returns (kept_paths, dropped) where dropped = [{"path", "reason", "component", "match"}].
    """
    comps = list(components) if components else components_all()
    paths = [str(p) for p in paths]
    drop: dict[int, dict] = {}
    for c in comps:
        ho = held_out(c)
        id_set = {r["id"] for r in ho}
        path_set = {_norm(r["path"]) for r in ho}
        for i, p in enumerate(paths):
            if i in drop:
                continue
            if _norm(p) in path_set:
                drop[i] = {"path": p, "reason": "path", "component": c}
            elif ids is not None and ids[i] in id_set:
                drop[i] = {"path": p, "reason": "id", "component": c}
        test_h = [(h, r["id"]) for r in ho for h in _as_list(r.get("phash"))]
        if not test_h:
            continue
        rest = [i for i in range(len(paths)) if i not in drop]
        hashes, _ = hash_many([paths[i] for i in rest])
        flat, own = [], []
        for i in rest:
            for h in hashes.get(paths[i], []):
                flat.append(h)
                own.append(i)
        for a, b, d in near_pairs(flat, [h for h, _ in test_h], threshold):
            i = own[a]
            if i not in drop:
                drop[i] = {"path": paths[i], "reason": f"phash<= {threshold} (d={d})", "component": c,
                           "match": test_h[b][1]}
    kept = [p for i, p in enumerate(paths) if i not in drop]
    return kept, list(drop.values())


def components_all() -> list[str]:
    return components()


# ----------------------------------------------------------------------------------------------- audit
def _read_train_list(train_list: str | Path | Sequence[str]) -> list[str]:
    if isinstance(train_list, (list, tuple)):
        items = [str(x) for x in train_list]
    else:
        p = Path(train_list)
        if p.is_dir():
            items = [str(x) for x in sorted(p.rglob("*")) if x.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
        elif p.suffix == ".jsonl":
            items = [json.loads(l)["path"] for l in p.read_text().splitlines() if l.strip()]
        else:
            items = [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [str(resolve(x)) if not Path(x).is_absolute() else x for x in items]


def leakage_audit(components: Sequence[str], train_list, *, strict: int = STRICT, loose: int = LOOSE,
                  max_examples: int = 25, workers: int = 8) -> dict:
    """Compare a lane's training files against held-out manifests: path/id overlap, exact bytes, pHash near-dups."""
    train = _read_train_list(train_list)
    hashes, errors = hash_many(train, workers)
    flat, own = [], []
    for p in train:
        for h in hashes.get(p, []):
            flat.append(h)
            own.append(p)
    report: dict[str, Any] = {"n_train": len(train), "n_train_hashed": len(hashes), "hash_errors": len(errors),
                              "thresholds": {"strict": strict, "loose": loose}, "components": {}}
    train_norm = {_norm(p) for p in train}
    for c in components:
        ho = held_out(c)
        test_h = [(h, r["id"]) for r in ho for h in _as_list(r.get("phash"))]
        path_hits = sorted(_norm(r["path"]) for r in ho if _norm(r["path"]) in train_norm)
        pairs = near_pairs(flat, [h for h, _ in test_h], loose) if test_h else []
        strict_pairs = [(a, b, d) for a, b, d in pairs if d <= strict]
        report["components"][c] = {
            "n_test": len(ho), "n_test_hashed": sum(1 for r in ho if r.get("phash")),
            "path_overlap": len(path_hits),
            "near_dup_strict_train_files": len({own[a] for a, _, _ in strict_pairs}),
            "near_dup_strict_test_items": len({test_h[b][1] for _, b, _ in strict_pairs}),
            "near_dup_loose_train_files": len({own[a] for a, _, _ in pairs}),
            "near_dup_loose_test_items": len({test_h[b][1] for _, b, _ in pairs}),
            "leaked_strict_test_ids": sorted({test_h[b][1] for _, b, _ in strict_pairs}),
            "leaked_loose_test_ids": sorted({test_h[b][1] for _, b, _ in pairs}),
            "examples": [{"train": own[a], "test_id": test_h[b][1], "dist": d}
                         for a, b, d in sorted(pairs, key=lambda x: x[2])[:max_examples]] + [{"path_overlap": p} for p in path_hits[:5]],
            "verdict": "PASS" if not path_hits and not strict_pairs else "FAIL",
        }
    report["verdict"] = "PASS" if all(v["verdict"] == "PASS" for v in report["components"].values()) else "FAIL"
    if errors:
        report["hash_error_examples"] = dict(list(errors.items())[:10])
    return report


# ------------------------------------------------------------------------------------------------ adoption
def adopt(src_files: Sequence[str | Path], out_dir: Path | None = None, hash_missing: bool = True) -> list[dict]:
    """Adopt lane-proposed held-out lists (docs/lanes/<lane>_manifests/*.jsonl) as G manifests.

    Checks required fields and unique ids, fills missing pHashes (where the media are readable), and writes
    ``<out_dir or eval/manifests>/<stem>.jsonl`` with ``adopted_from``. Lane G still owns the result: an adopted
    manifest only enters the report when a lane evaluates on it and G's audit passes.
    """
    rows = []
    for f in map(Path, src_files):
        recs = read_manifest(f)
        errs = validate(recs)
        info = {"file": str(f), "component": f.stem, "n": len(recs), "errors": errs[:5]}
        if errs:
            info["status"] = "rejected"
            rows.append(info)
            continue
        if hash_missing:
            try:
                add_phash(recs)
            except Exception as e:  # media not on this machine
                info["hash_error"] = str(e)[:200]
        for r in recs:
            r.setdefault("split", "test")
            r["adopted_from"] = str(f)
        info["hashed"] = sum(1 for r in recs if r.get("phash"))
        dst = write_manifest(f.stem, recs, path=(out_dir / f"{f.stem}.jsonl") if out_dir else None)
        info.update({"status": "adopted", "written": str(dst)})
        rows.append(info)
    return rows


# -------------------------------------------------------------------------------------------------- CLI
def _cli(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="leakage audit of a training list against held-out manifests")
    a.add_argument("--components", default="", help="comma list; default = all manifests")
    a.add_argument("--train-list", required=True)
    a.add_argument("--out")
    h = sub.add_parser("hash", help="fill missing phash fields of a manifest in place")
    h.add_argument("component")
    sub.add_parser("stats", help="print item / source counts per manifest")
    ad = sub.add_parser("adopt", help="adopt lane-proposed held-out lists as G manifests")
    ad.add_argument("files", nargs="+")
    ad.add_argument("--out-dir")
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        comps = [c for c in args.components.split(",") if c] or components()
        rep = leakage_audit(comps, args.train_list)
        s = json.dumps(rep, indent=2)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(s)
        print(s)
        return 0 if rep["verdict"] == "PASS" else 2
    if args.cmd == "hash":
        recs = add_phash(read_manifest(args.component))
        write_manifest(args.component, recs)
        print(f"{args.component}: {sum(1 for r in recs if r.get('phash'))}/{len(recs)} hashed")
        return 0
    if args.cmd == "adopt":
        print(json.dumps(adopt(args.files, Path(args.out_dir) if args.out_dir else None), indent=2))
        return 0
    if args.cmd == "stats":
        for c in components():
            recs = read_manifest(c)
            by = defaultdict(int)
            for r in recs:
                by[r.get("split", "test")] += 1
            print(f"{c}: {len(recs)} items, {len({r['source'] for r in recs})} sources, splits={dict(by)}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
