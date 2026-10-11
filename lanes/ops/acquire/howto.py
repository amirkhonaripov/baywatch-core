"""CrossTask + COIN car-task videos: survival check on YouTube, then download at <=360p (video only).

  python -m lanes.ops.acquire.howto list       # -> data/prep/ops/howto_videos.csv
  python -m lanes.ops.acquire.howto check      # yt-dlp simulate -> status column
  python -m lanes.ops.acquire.howto download --cap-gb 15
Metadata (crosstask_release.zip, COIN.json) is expected in data/raw/ops_howto/.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from seen.paths import PREP, RAW

ROOT = RAW / "ops_howto"
CT = ROOT / "crosstask_release"
VID = ROOT / "videos"
LIST = PREP / "ops" / "howto_videos.csv"

CT_PRIMARY = {"40567": "Change a Tire", "16815": "Jack Up a Car", "44789": "Add Oil to Your Car"}
CT_RELATED = {"40566": "Change a Hubcap", "40570": "Change Wheel Bearings", "11138": "Change The Brake Pads in Your Car",
              "40596": "Install Lowering Springs", "40610": "Remove Lug Nuts and Tires"}
COIN_TASKS = ["ChangeCarTire", "UseJack", "ReplaceTyreValveStem", "ReplaceCarFuse", "PolishCar", "ReplaceAWiperHead"]
FIELDS = ["source", "task_id", "task", "ytid", "annotated", "coin_subset", "status", "duration", "height"]


def build_list() -> None:
    rows = {}
    for fn in ["videos.csv", "videos_val.csv"]:
        for task_id, ytid, _url in csv.reader((CT / fn).open()):
            if task_id in CT_PRIMARY or task_id in CT_RELATED:
                ann = (CT / "annotations" / f"{task_id}_{ytid}.csv").exists()
                rows[ytid] = dict(source="crosstask", task_id=task_id, task=CT_PRIMARY.get(task_id) or CT_RELATED[task_id],
                                  ytid=ytid, annotated=int(ann), coin_subset="")
    db = json.loads((ROOT / "COIN.json").read_text())["database"]
    for ytid, v in db.items():
        if v["class"] in COIN_TASKS and ytid not in rows:
            rows[ytid] = dict(source="coin", task_id=str(v["recipe_type"]), task=v["class"], ytid=ytid, annotated=1,
                              coin_subset=v["subset"])
    LIST.parent.mkdir(parents=True, exist_ok=True)
    _write(list(rows.values()))
    from collections import Counter
    print(Counter((r["source"], r["task"], r["annotated"]) for r in rows.values()))
    print("total", len(rows))


def _write(rows: list[dict]) -> None:
    with LIST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _read() -> list[dict]:
    return list(csv.DictReader(LIST.open()))


FMT = "bv*[height<=360][ext=mp4]/bv*[height<=360]/wv*"


def _ydl(opts: dict):
    import yt_dlp

    base = dict(quiet=True, no_warnings=True, noprogress=True, retries=3, socket_timeout=30)
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def check(workers: int = 6) -> None:
    rows = _read()

    def one(r: dict) -> dict:
        if r.get("status") in ("ok", "gone"):
            return r
        try:
            with _ydl(dict(format=FMT)) as y:
                info = y.extract_info(f"https://www.youtube.com/watch?v={r['ytid']}", download=False)
            r["status"], r["duration"], r["height"] = "ok", info.get("duration") or "", info.get("height") or ""
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            r["status"] = "gone" if any(k in msg for k in ("unavailable", "private", "removed", "terminated", "not available", "copyright")) else "error"
            r["duration"] = ""
        return r

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, rows))
    _write(rows)
    from collections import Counter
    print(Counter((r["source"], r["task"], r["status"]) for r in rows))


def download(cap_gb: float, workers: int = 4, max_dur_s: int = 1200) -> None:
    rows = [r for r in _read() if r["status"] == "ok"]
    # annotated first (they carry the step labels), then short videos first
    rows.sort(key=lambda r: (-int(r["annotated"]), float(r["duration"] or 1e9)))
    VID.mkdir(parents=True, exist_ok=True)

    def used() -> float:
        return sum(p.stat().st_size for p in VID.glob("*.mp4")) / 1e9

    def one(r: dict) -> str:
        dst = VID / f"{r['ytid']}.mp4"
        if dst.exists():
            return "have"
        if float(r["duration"] or 0) > max_dur_s:
            return "too_long"
        if used() > cap_gb:
            return "cap"
        try:
            with _ydl(dict(format=FMT, outtmpl=str(VID / "%(id)s.%(ext)s"), merge_output_format="mp4",
                           sleep_interval=1, max_sleep_interval=3)) as y:
                y.download([f"https://www.youtube.com/watch?v={r['ytid']}"])
            got = list(VID.glob(f"{r['ytid']}.*"))
            for g in got:
                if g.suffix != ".mp4" and g.suffix in (".webm", ".mkv"):
                    g.rename(dst)  # keep name stable; decoders read by content
            return "ok"
        except Exception as e:  # noqa: BLE001
            return f"err:{str(e)[:80]}"

    from collections import Counter
    c = Counter()
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(one, rows)):
            c[res.split(":")[0]] += 1
            if i % 25 == 0:
                print(i, dict(c), f"{used():.2f} GB", flush=True)
    print("DOWNLOAD_DONE", dict(c), f"{used():.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "check", "download"])
    ap.add_argument("--cap-gb", type=float, default=15.0)
    a = ap.parse_args()
    if a.cmd == "list":
        build_list()
    elif a.cmd == "check":
        check()
    else:
        download(a.cap_gb)
