"""Download the jury media listed in eval/jury/jury.jsonl into SEEN_DATA/raw/eval_jury/ and cut the video frames.

    python -m eval.jury.fetch                 # download missing files, cut frames, verify
    python -m eval.jury.fetch --manifest      # also (re)write eval/manifests/jury.jsonl with pHashes (held out)

Images are fetched as Commons thumbnails (<= 1280 px wide; the URL is in each record); videos as the original file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

from eval.jury.common import JURY_LIST, fast_jury_dir, jury_items, raw_dir, raw_path as local_path
from eval.jury.select import http_get


def cut_frame(video: Path, t_s: float, dst: Path) -> bool:
    """Save the first decoded frame at or after ``t_s`` (PyAV: decodes VP9/webm reliably; cv2 as fallback)."""
    import cv2

    img = None
    try:
        import av

        with av.open(str(video)) as c:
            s = c.streams.video[0]
            for fr in c.decode(s):
                if fr.time is not None and fr.time + 1e-6 >= t_s:
                    img = fr.to_ndarray(format="bgr24")
                    break
    except Exception:
        img = None
    if img is None:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target, i = int(round(t_s * fps)), 0
        while i <= target:
            ok, f = cap.read()
            if not ok:
                break
            img, i = f, i + 1
        cap.release()
    if img is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def fetch(items: list[dict], log=print) -> dict:
    d = raw_dir()
    d.mkdir(parents=True, exist_ok=True)
    got, failed = 0, []
    for it in items:
        if it["kind"] == "video_frame":
            continue
        dst = local_path(it)
        if dst.exists() and dst.stat().st_size > 0:
            continue
        err = None
        for attempt in range(5):
            try:
                data = http_get(it["download_url"], timeout=300)
                dst.write_bytes(data)
                got += 1
                err = None
                log(f"fetched {it['id']} {len(data) / 1e6:.1f} MB")
                break
            except urllib.error.HTTPError as e:  # 429: Wikimedia rate limit, back off
                err = e
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(ra) if ra and ra.isdigit() else 30 * (attempt + 1))
            except Exception as e:
                err = e
                time.sleep(5)
        if err is not None:
            failed.append((it["id"], str(err)))
            log(f"FAILED {it['id']}: {err}")
        time.sleep(2.0)
    by_id = {it["id"]: it for it in items}
    for it in items:
        if it["kind"] != "video_frame":
            continue
        dst = local_path(it)
        if dst.exists():
            continue
        src = local_path(by_id[it["frame_of"]])
        if not src.exists() or not cut_frame(src, float(it.get("frame_t_s", 0)), dst):
            failed.append((it["id"], f"could not cut frame from {src.name}"))
    present = sum(local_path(it).exists() for it in items)
    return {"downloaded": got, "present": present, "total": len(items), "failed": failed}


def stage_fast(items: list[dict]) -> str:
    """One sequential copy of the jury media to the NVMe (docs/LANES.md 'NVMe staging'; eval budget 3 GB)."""
    import shutil

    f = fast_jury_dir()
    if f is None:
        return "no NVMe staging dir on this machine"
    f.mkdir(parents=True, exist_ok=True)
    n = 0
    for it in items:
        src, dst = local_path(it), f / it["file"]
        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            shutil.copyfile(src, dst)
            n += 1
    size = sum(p.stat().st_size for p in f.iterdir()) / 1e9
    return f"staged {n} new files to {f} ({size:.2f} GB)"


def write_manifest(items: list[dict], path: Path | None = None) -> Path:
    from eval.manifest import add_phash, write_manifest as wm
    from seen.paths import DATA

    recs = []
    for it in items:
        p = local_path(it)
        if not p.exists():
            continue
        recs.append({"id": f"jury:{it['id']}", "path": str(p.relative_to(DATA)), "dataset": "wikimedia_commons_jury",
                     "source": f"commons:{it.get('author') or it['title']}"[:120], "split": "test",
                     "title": it["title"], "licence": it.get("licence"), "label": it.get("expected")})
    for r, it in zip(recs, [i for i in items if local_path(i).exists()]):
        r["_hash_path"] = str(fast_jury_dir() / it["file"]) if fast_jury_dir() and (fast_jury_dir() / it["file"]).exists() else r["path"]
    add_phash(recs, path_key="_hash_path")
    for r in recs:
        r.pop("_hash_path", None)
    return wm("jury", recs, path=path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--manifest-out", default=None, help="write the manifest here instead of eval/manifests/jury.jsonl")
    ap.add_argument("--list", default=str(JURY_LIST))
    a = ap.parse_args(argv)
    items = jury_items(Path(a.list))
    if not items:
        print(f"no items in {a.list}")
        return 1
    res = fetch(items)
    print(json.dumps(res, indent=2))
    print(stage_fast(items))
    if a.manifest:
        print("manifest:", write_manifest(items, Path(a.manifest_out) if a.manifest_out else None))
    return 0 if not res["failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
