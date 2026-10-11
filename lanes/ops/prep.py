"""Build the ops frame table: sample frames from videos, copy still images, derive labels, assign splits.

All prepared images live on the NVMe staging dir (seen.paths.FAST/ops, user-approved because the HDD is IO-bound);
`path` in the table is relative to that dir, `orig` is the source file relative to SEEN_DATA.

  python -m lanes.ops.prep frames_k700    # decode K700 clips (HDD, one sequential pass) -> FAST/ops/frames/k700
  python -m lanes.ops.prep frames_howto   # decode CrossTask/COIN videos -> FAST/ops/frames/howto (run where the videos are)
  python -m lanes.ops.prep stills         # resize-copy Roboflow/carparts/Commons images -> FAST/ops/frames/stills
  python -m lanes.ops.prep yolo           # copy part-detector training sets -> FAST/ops/yolo
  python -m lanes.ops.prep table          # -> PREP/ops/table.parquet (one row per frame/image)

Label columns hold 1 / 0 / NaN (unknown). Derived labels (step spans) are marked in `label_src`.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import os

from seen.paths import DATA, FAST, PREP, RAW

# Still-image sources: RAW on the HDD by default; OPS_STILLS_SRC points at an NVMe re-download (same sub-dir names)
# when the HDD is IO-bound. `orig` is always recorded relative to that root with the RAW-style dir name.
SRC = Path(os.environ.get("OPS_STILLS_SRC", str(RAW)))
OPS = PREP / "ops"
FAST_OPS = FAST / "ops"
FRAMES = FAST_OPS / "frames"
TABLE = OPS / "table.parquet"
MAX_SIDE = 640
STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work", "idle", "washing"]
SERVICE = {"changing wheel (not on bike)", "changing oil", "checking tires", "repairing puncture"}


def fpath(p: str) -> Path:
    """Absolute path of a table image (paths are relative to FAST/ops)."""
    return FAST_OPS / p


def _rel(p: Path) -> str:
    return str(p.relative_to(FAST_OPS))


def _save(img: np.ndarray, path: Path) -> None:
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ------------------------------------------------------------------------------------------------ video frames
def _k700_clip(args) -> list[dict]:
    src, stem = args
    out = []
    dsts = [FRAMES / "k700" / f"{stem}_{j}.jpg" for j in range(3)]
    if all(d.exists() for d in dsts):  # already extracted: do not touch the HDD again
        dur = int(stem[-6:]) - int(stem[-13:-7])
        return [dict(path=_rel(d), t_s=round(f * dur, 2), j=j) for j, (d, f) in enumerate(zip(dsts, (0.2, 0.5, 0.8)))]
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if n <= 0:
        return out
    want = {int(f * n): j for j, f in enumerate((0.2, 0.5, 0.8))}
    i = 0
    while want and i <= max(want):  # sequential grab (no seeking): one pass over the file
        if not cap.grab():
            break
        if i in want:
            ok, img = cap.retrieve()
            if ok:
                j = want.pop(i)
                p = FRAMES / "k700" / f"{stem}_{j}.jpg"
                if not p.exists():
                    _save(img, p)
                out.append(dict(path=_rel(p), t_s=round(i / fps, 2), j=j))
        i += 1
    cap.release()
    return out


def _howto_video(args) -> list[dict]:
    src, ytid, fps_out, cap_n = args
    out = []
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        return out
    dur = n / fps
    ts = np.arange(0.5, dur, 1.0 / fps_out)
    if len(ts) > cap_n:
        ts = np.linspace(0.5, dur - 0.5, cap_n)
    want = {int(round(t * fps)): t for t in ts}
    i = 0
    nxt = sorted(want)
    k = 0
    while k < len(nxt):
        ok = cap.grab()
        if not ok:
            break
        if i == nxt[k]:
            ok, img = cap.retrieve()
            if ok:
                t = want[i]
                p = FRAMES / "howto" / ytid / f"{ytid}_{t:07.1f}.jpg"
                if not p.exists():
                    _save(img, p)
                out.append(dict(path=_rel(p), t_s=round(float(t), 2)))
            k += 1
        i += 1
    cap.release()
    return out


def _howto_ffmpeg(args) -> list[dict]:
    """ffmpeg (multithreaded VP9/H.264 decode) version of _howto_video; same sampling times and file names."""
    import subprocess
    import tempfile

    src, ytid, fps_out, cap_n = args
    d = FRAMES / "howto" / ytid
    done = d / ".done"
    if not done.exists():
        try:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                       capture_output=True, text=True, timeout=60).stdout.strip())
        except Exception:  # noqa: BLE001
            return []
        fps = fps_out if (dur - 0.5) * fps_out <= cap_n else cap_n / max(dur - 1.0, 1.0)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with tempfile.TemporaryDirectory(dir=d) as tmp:
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", "0.5", "-i", str(src), "-vf",
                            f"fps={fps:.6f},scale='min({MAX_SIDE},iw)':-2", "-q:v", "3", f"{tmp}/%05d.jpg"], timeout=1800)
            for f in sorted(Path(tmp).glob("*.jpg")):
                t = 0.5 + (int(f.stem) - 1) / fps
                f.rename(d / f"{ytid}_{t:07.1f}.jpg")
        done.write_text(f"{fps}")
    out = []
    for f in sorted(d.glob("*.jpg")):
        out.append(dict(path=_rel(f), t_s=round(float(f.stem.rsplit("_", 1)[1]), 2)))
    return out


def frames_howto_ff(workers: int = 6) -> None:
    from concurrent.futures import ThreadPoolExecutor

    vids = pd.read_csv(OPS / "howto_videos.csv", dtype=str)
    vdir = RAW / "ops_howto" / "videos"
    jobs, meta = [], []
    for r in vids.itertuples():
        src = vdir / f"{r.ytid}.mp4"
        if src.exists() and src.stat().st_size > 10000:
            ann = r.annotated == "1"
            jobs.append((src, r.ytid, 1.0 if ann else 0.5, 360 if ann else 120))
            meta.append(r)
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (r, res) in enumerate(zip(meta, ex.map(_howto_ffmpeg, jobs))):
            for f in res:
                rows.append(dict(dataset=r.source, source=f"yt:{r.ytid}", video=r.ytid, task=r.task, task_id=r.task_id,
                                 annotated=int(r.annotated), **f))
            if i % 50 == 0:
                print("howto", i, len(jobs), flush=True)
    pd.DataFrame(rows).to_parquet(OPS / "frames_howto.parquet")
    print("howto frames", len(rows), "videos", len(jobs), flush=True)


def howto_subsample(step: int = 2, max_side: int = 512, quality: int = 85) -> None:
    """Shrink the CrossTask/COIN frame set for the NVMe budget: every `step`-th frame of annotated videos (1 fps -> 0.5 fps),
    longest side `max_side`, JPEG quality `quality`. Writes frames/howto_s and replaces frames_howto.parquet."""
    full = pd.read_parquet(OPS / "frames_howto.parquet")
    full.to_parquet(OPS / "frames_howto_full.parquet")
    keep = []
    for vid, g in full.groupby("video", sort=False):
        g = g.sort_values("t_s")
        keep.append(g.iloc[::step] if g["annotated"].iloc[0] == 1 else g)
    sub = pd.concat(keep, ignore_index=True)
    new_paths = []
    for r in sub.itertuples():
        src = FAST_OPS / r.path
        dst = FRAMES / "howto_s" / r.video / Path(r.path).name
        if not dst.exists():
            img = cv2.imread(str(src))
            h, w = img.shape[:2]
            sc = max_side / max(h, w)
            if sc < 1:
                img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        new_paths.append(_rel(dst))
    sub["path"] = new_paths
    sub.to_parquet(OPS / "frames_howto.parquet")
    print("howto subsample", len(full), "->", len(sub))


def _box_one(args) -> list[dict]:
    """Download one YouTube video to the NVMe, extract frames at <=0.5 fps (<= `cap` frames), delete the video."""
    import subprocess
    import time

    ytid, cap = args
    d = FRAMES / "howto_b" / ytid
    if not (d / ".done").exists():
        tmp = FAST_OPS / "tmp_video"
        tmp.mkdir(parents=True, exist_ok=True)
        src = None
        try:
            import yt_dlp

            opts = dict(format="bv*[height<=360][ext=mp4]/bv*[height<=360]/wv*", outtmpl=str(tmp / "%(id)s.%(ext)s"),
                        quiet=True, no_warnings=True, retries=3, noprogress=True)
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([f"https://www.youtube.com/watch?v={ytid}"])
            src = next(iter(tmp.glob(f"{ytid}.*")), None)
            if src is None:
                return []
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                                       capture_output=True, text=True, timeout=60).stdout.strip())
            fps = 0.5 if (dur - 0.5) * 0.5 <= cap else cap / max(dur - 1.0, 1.0)
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            raw = d / "_raw"
            raw.mkdir()
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", "0.5", "-i", str(src), "-vf",
                            f"fps={fps:.6f},scale='min(512,iw)':-2", "-q:v", "5", f"{raw}/%05d.jpg"], timeout=1800)
            for f in sorted(raw.glob("*.jpg")):
                t = 0.5 + (int(f.stem) - 1) / fps
                f.rename(d / f"{ytid}_{t:07.1f}.jpg")
            raw.rmdir()
            (d / ".done").write_text(f"{fps}")
        except Exception as e:  # noqa: BLE001
            print("fail", ytid, str(e)[:100], flush=True)
            return []
        finally:
            for f in tmp.glob(f"{ytid}.*"):
                f.unlink()
            time.sleep(1.0)
    return [dict(path=_rel(f), t_s=round(float(f.stem.rsplit("_", 1)[1]), 2)) for f in sorted(d.glob("*.jpg"))]


def frames_howto_box(workers: int = 6) -> None:
    """Box-side alternative to frames_howto_ff + howto_subsample (used when the Mac->box link is too slow):
    0.5 fps, <=180 frames for step-annotated videos, <=120 for un-annotated ones, longest side 512 px."""
    from concurrent.futures import ThreadPoolExecutor

    vids = pd.read_csv(OPS / "howto_videos.csv", dtype=str)
    vids = vids[vids["status"] == "ok"].copy()
    # most relevant tasks first (raised / wheel_off / hood spans), so a partial run is still useful
    prio = ["Change a Tire", "Jack Up a Car", "ChangeCarTire", "UseJack", "Add Oil to Your Car", "ReplaceTyreValveStem",
            "ReplaceCarFuse", "Change The Brake Pads in Your Car", "Change Wheel Bearings", "Remove Lug Nuts and Tires",
            "Change a Hubcap", "Install Lowering Springs", "PolishCar", "ReplaceAWiperHead"]
    vids["prio"] = vids["task"].map({t: i for i, t in enumerate(prio)}).fillna(99)
    vids = vids.sort_values(["prio", "ytid"])
    jobs = [(r.ytid, 180 if r.annotated == "1" else 120) for r in vids.itertuples()]
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, (r, res) in enumerate(zip(vids.itertuples(), ex.map(_box_one, jobs))):
            for f in res:
                rows.append(dict(dataset=r.source, source=f"yt:{r.ytid}", video=r.ytid, task=r.task, task_id=r.task_id,
                                 annotated=int(r.annotated), **f))
            if i % 25 == 0:
                print("howto_box", i, len(jobs), len(rows), flush=True)
                pd.DataFrame(rows).to_parquet(OPS / "frames_howto.parquet")  # usable partial result
    pd.DataFrame(rows).to_parquet(OPS / "frames_howto.parquet")
    print("howto frames", len(rows), "videos", len({r["video"] for r in rows}), flush=True)


def frames_k700(workers: int = 4, train_per_class: int = 120) -> None:
    sel = pd.read_csv(OPS / "k700_selected.csv")
    # class-balanced order: all val clips, then `train_per_class` train clips per class (round-robin)
    va = sel[sel["split"] != "train"]
    tr = sel[sel["split"] == "train"].copy()
    tr["rank"] = tr.groupby("label").cumcount()
    tr = tr[tr["rank"] < train_per_class].sort_values(["rank", "label"])
    sel = pd.concat([va, tr.drop(columns="rank")])
    jobs, meta = [], []
    for r in sel.itertuples():
        src = RAW / "ops_k700" / Path(r.member).name
        if src.exists():
            jobs.append((src, Path(r.member).stem))
            meta.append(r)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (r, res) in enumerate(zip(meta, ex.map(_k700_clip, jobs, chunksize=8))):
            for f in res:
                rows.append(dict(dataset="k700", source=f"yt:{r.ytid}", video=Path(r.member).stem, k700_label=r.label,
                                 k700_split=r.split, **f))
            if i % 500 == 0:
                print("k700", i, len(jobs), flush=True)
    pd.DataFrame(rows).to_parquet(OPS / "frames_k700.parquet")
    print("k700 frames", len(rows), flush=True)


def frames_howto(workers: int = 8) -> None:
    vids = pd.read_csv(OPS / "howto_videos.csv", dtype=str)
    vdir = RAW / "ops_howto" / "videos"
    jobs, meta = [], []
    for r in vids.itertuples():
        src = vdir / f"{r.ytid}.mp4"
        if src.exists() and src.stat().st_size > 10000:
            ann = r.annotated == "1"
            jobs.append((src, r.ytid, 1.0 if ann else 0.5, 360 if ann else 120))
            meta.append(r)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r, res in zip(meta, ex.map(_howto_video, jobs)):
            for f in res:
                rows.append(dict(dataset=r.source, source=f"yt:{r.ytid}", video=r.ytid, task=r.task, task_id=r.task_id,
                                 annotated=int(r.annotated), **f))
    pd.DataFrame(rows).to_parquet(OPS / "frames_howto.parquet")
    print("howto frames", len(rows), "videos", len(jobs), flush=True)


# ------------------------------------------------------------------------------------------- step-span labels
CT_STEPS = {
    "40567": ["brake on", "get things out", "start loose", "jack up", "unscrew wheel", "withdraw wheel", "put wheel",
              "screw wheel", "jack down", "tight wheel", "put things back"],
    "16815": ["brake on", "raise jack", "lower jack"],
    "44789": ["remove cap", "put funnel", "pour oil", "remove funnel", "close cap", "pull out dipstick",
              "wipe off dipstick", "insert dipstick"],
}
# state -> (on-step names, off-step names). Positive span = [end(on), start(off)]; negative before start(on) and
# after end(off); inside the on/off steps themselves = unknown (the transition is happening).
SPAN_RULES = {
    "raised": [("jack up", "jack down"), ("raise jack", "lower jack"), ("jack up the car", "@put on the tire"),
               ("raise the jack up", "let the jack down")],
    "wheel_off": [("withdraw wheel", "put wheel"), ("remove the tire", "put on the tire")],
}


def _segments(dataset: str, task_id: str, ytid: str, coin: dict) -> list[tuple[str, float, float]]:
    if dataset == "crosstask":
        f = RAW / "ops_howto" / "crosstask_release" / "annotations" / f"{task_id}_{ytid}.csv"
        if not f.exists() or task_id not in CT_STEPS:
            return []
        out = []
        for line in f.read_text().splitlines():
            if line.strip():
                k, s, e = line.split(",")
                out.append((CT_STEPS[task_id][int(k) - 1], float(s), float(e)))
        return out
    v = coin.get(ytid)
    return [(a["label"], float(a["segment"][0]), float(a["segment"][1])) for a in v["annotation"]] if v else []


def _span_labels(t: np.ndarray, segs: list[tuple[str, float, float]], on: str, off: str) -> np.ndarray | None:
    """1 inside [end(on), start(off)], 0 before start(on) / after end(off), NaN elsewhere. '@x' = off is the END of x."""
    off_end = off.startswith("@")
    off = off.lstrip("@")
    ons = [s for s in segs if s[0] == on]
    offs = [s for s in segs if s[0] == off]
    if not ons:
        return None
    lab = np.full(len(t), np.nan)
    on_s, on_e = min(s[1] for s in ons), min(s[2] for s in ons)
    lab[t < on_s] = 0
    if offs:
        if off_end:
            off_s = off_e = max(s[2] for s in offs)
        else:
            off_s, off_e = max(s[1] for s in offs), max(s[2] for s in offs)
        if off_s < on_e:
            return None
        lab[(t >= on_e) & (t <= off_s)] = 1
        if not off_end:
            lab[t > off_e] = 0
    else:
        return None
    return lab


def howto_labels(df: pd.DataFrame) -> pd.DataFrame:
    coin = json.loads((RAW / "ops_howto" / "COIN.json").read_text())["database"]
    df = df.copy()
    for c in ("raised", "wheel_off", "hood_open", "work"):
        df[c] = np.nan
    df["label_src"] = ""
    for (ds, tid, yt), g in df.groupby(["dataset", "task_id", "video"]):
        segs = _segments(ds, tid, yt, coin)
        if not segs:
            continue
        t = g["t_s"].to_numpy()
        inside = np.zeros(len(t), bool)
        for _, s, e in segs:
            inside |= (t >= s) & (t <= e)
        # work proxy: inside an annotated step = 1, outside = 0 (outside also contains intros/talking; see notes)
        df.loc[g.index, "work"] = inside.astype(float)
        src = ["work:step"]
        for state, rules in SPAN_RULES.items():
            for on, off in rules:
                lab = _span_labels(t, segs, on, off)
                if lab is not None:
                    df.loc[g.index, state] = lab
                    src.append(f"{state}:span")
                    break
        if ds == "crosstask" and tid == "44789":  # Add Oil: every step is done with the bonnet open
            df.loc[g.index[inside], "hood_open"] = 1.0
            src.append("hood_open:inside_step")
        df.loc[g.index, "label_src"] = ";".join(src)
    return df


# --------------------------------------------------------------------------------------------- still images
def _yolo_images(root: Path, dataset: str, names: list[str], split_map: dict[str, str]) -> list[dict]:
    rows = []
    for sub, split in split_map.items():
        for img in sorted((root / sub / "images").glob("*")):
            lab = root / sub / "labels" / (img.stem + ".txt")
            cls = set()
            if lab.exists():
                for line in lab.read_text().splitlines():
                    if line.strip():
                        cls.add(names[int(line.split()[0])])
            rows.append(dict(dataset=dataset, source=f"{dataset}:{img.stem.split('_jpg')[0][:40]}",
                             orig=str(img.relative_to(SRC)), split=split, rf_classes="|".join(sorted(cls))))
    return rows


def _yaml_names(p: Path) -> list[str]:
    import yaml

    d = yaml.safe_load(p.read_text())
    n = d["names"]
    return list(n.values()) if isinstance(n, dict) else list(n)


def stills() -> pd.DataFrame:
    rows = []
    # car-hood-open (detection: hood_closed / hood_open); RF test split held out
    r = SRC / "ops_rf_car_hood_open"
    for x in _yolo_images(r, "rf_car_hood_open", _yaml_names(r / "data.yaml"), {"train": "train", "valid": "train", "test": "test"}):
        c = set(x["rf_classes"].split("|")) - {""}
        x["hood_open"] = 1.0 if "hood_open" in c else (0.0 if c else np.nan)
        x["car_present"] = 1.0 if c else np.nan
        rows.append(x)
    # car condition (other uploader): cross-project hood test
    r = SRC / "ops_rf_car_condition_det"
    if (r / "data.yaml").exists():
        for x in _yolo_images(r, "rf_car_condition", _yaml_names(r / "data.yaml"), {"train": "test", "valid": "test", "test": "test"}):
            c = {s.lower() for s in x["rf_classes"].split("|")} - {""}
            x["hood_open"] = 1.0 if any("hood" in s for s in c) else (0.0 if c else np.nan)
            x["car_present"] = 1.0 if c else np.nan
            rows.append(x)
    # washing-car (fixed wash-yard cameras): washing, hood
    r = SRC / "ops_rf_washing_car"
    if (r / "data.yaml").exists():
        for x in _yolo_images(r, "rf_washing_car", _yaml_names(r / "data.yaml"), {"train": "train", "valid": "train", "test": "train"}):
            c = {s.lower() for s in x["rf_classes"].split("|")} - {""}
            x["washing"] = 1.0 if any("wash" in s for s in c) else (0.0 if c else np.nan)
            x["hood_open"] = 1.0 if any("hood" in s for s in c) else (0.0 if c else np.nan)
            x["car_present"] = 1.0 if c else np.nan
            rows.append(x)
    # car-open-hood (classification folders)
    r = SRC / "ops_rf_car_open_hood"
    for sub in ("train", "valid", "test"):
        for cls_dir in sorted((r / sub).glob("*")) if (r / sub).exists() else []:
            if not cls_dir.is_dir() or cls_dir.name.lower() not in ("openhood", "notopenhood"):
                continue  # 'Unlabeled' folder is skipped
            pos = cls_dir.name.lower() == "openhood"
            for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
                rows.append(dict(dataset="rf_car_open_hood", source=f"rf_car_open_hood:{img.stem[:40]}",
                                 orig=str(img.relative_to(SRC)), split="train" if sub != "test" else "test",
                                 rf_classes=cls_dir.name, hood_open=1.0 if pos else 0.0, car_present=1.0))
    # carparts-seg: parked-car photos (no lift, no work) -> negatives; used for the wheel/hood part detector too
    r = SRC / "ops_carparts" / "carparts-seg"
    if not r.exists():
        r = SRC / "ops_carparts"
    for sub in ("train", "valid", "val", "test"):
        d = r / "images" / sub
        for img in sorted(d.glob("*.jpg")) if d.exists() else []:
            rows.append(dict(dataset="carparts", source=f"carparts:{img.stem[:40]}", orig=str(img.relative_to(SRC)),
                             split="train"))
    # Wikimedia Commons lift photos (labels come from the hand-label file)
    meta = SRC / "ops_commons" / "meta.csv"
    if meta.exists():
        for m in csv.DictReader(meta.open()):
            rows.append(dict(dataset="commons", source=f"commons:{m['file']}", orig=str((SRC / "ops_commons" / m["file"]).relative_to(SRC)),
                             split="train", licence=m["licence"]))
    return pd.DataFrame(rows)


def stills_copy() -> None:
    """Register still-image sets and resize-copy them to FAST/ops/frames/stills (single sequential pass)."""
    df = stills()
    paths = []
    for i, r in enumerate(df.itertuples()):
        dst = FRAMES / "stills" / r.dataset / (Path(r.orig).stem[:80] + f"_{i}.jpg")
        if not dst.exists():
            img = cv2.imread(str(SRC / r.orig))
            if img is None:
                paths.append(None)
                continue
            _save(img, dst)
        paths.append(_rel(dst))
        if i % 2000 == 0:
            print("stills", i, len(df), flush=True)
    df["path"] = paths
    df = df[df["path"].notna()]
    df.to_parquet(OPS / "frames_stills.parquet")
    print("stills", len(df), df.groupby("dataset").size().to_dict())


def yolo_copy() -> None:
    """Part-detector training sets -> FAST/ops/yolo/<name>/{images,labels}/<split> (images resized, labels as-is)."""
    specs = {"carparts": (SRC / "ops_carparts", {"train": "images/train", "val": "images/val", "test": "images/test"}, "labels"),
             "hood": (SRC / "ops_rf_car_hood_open", {"train": "train/images", "val": "valid/images", "test": "test/images"}, None)}
    for name, (root, splits, _) in specs.items():
        for split, rel in splits.items():
            src = root / rel
            for img in sorted(src.glob("*")):
                lab = (src.parent.parent / "labels" / split / (img.stem + ".txt")) if name == "carparts" else (src.parent / "labels" / (img.stem + ".txt"))
                d_img = FAST_OPS / "yolo" / name / "images" / split / (img.stem + ".jpg")
                d_lab = FAST_OPS / "yolo" / name / "labels" / split / (img.stem + ".txt")
                if not d_img.exists():
                    im = cv2.imread(str(img))
                    if im is None:
                        continue
                    _save(im, d_img)
                d_lab.parent.mkdir(parents=True, exist_ok=True)
                if lab.exists():
                    shutil.copyfile(lab, d_lab)
                else:
                    d_lab.write_text("")
            print(name, split, flush=True)


def table() -> None:
    k = pd.read_parquet(OPS / "frames_k700.parquet")
    k["split"] = np.where(k["k700_split"].isin(["val", "validation"]), "test", "train")
    k["work"] = k["k700_label"].isin(SERVICE).astype(float)
    k["label_src"] = "work:k700_class"
    if (OPS / "frames_howto.parquet").exists():
        h = howto_labels(pd.read_parquet(OPS / "frames_howto.parquet"))
        # split CrossTask/COIN by video (source): test iff sha1("seen-ops-v1|yt:<id>") bucket < 0.25. Per-id threshold
        # (same hash as eval.manifest._bucket), so adding videos later never moves an existing video between splits.
        import hashlib

        def bucket(v: str) -> float:
            return int(hashlib.sha1(f"seen-ops-v1|yt:{v}".encode()).hexdigest()[:12], 16) / float(16 ** 12)

        h["split"] = h["video"].map(lambda v: "test" if bucket(v) < 0.25 else "train")
    else:
        h = pd.DataFrame()
    s = pd.read_parquet(OPS / "frames_stills.parquet")
    # Commons lift photos: hand labels (lanes/ops/labels/commons_labels.csv); excluded = trains/buses/heavy trucks/ads
    cl = pd.read_csv(Path(__file__).resolve().parent / "labels" / "commons_labels.csv", dtype=str).set_index("file")
    is_c = s["dataset"] == "commons"
    fn = s.loc[is_c, "orig"].map(lambda o: Path(o).name)
    keep = fn.map(lambda f: f in cl.index and cl.loc[f, "exclude"] == "0")
    s = pd.concat([s[~is_c], s[is_c][keep.to_numpy()]], ignore_index=True)
    is_c = s["dataset"] == "commons"
    for st in ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work"]:
        v = s.loc[is_c, "orig"].map(lambda o: cl.loc[Path(o).name, st])
        s.loc[is_c, st] = pd.to_numeric(v, errors="coerce")
    s.loc[is_c, "label_src"] = "hand:commons"
    dom = pd.read_parquet(OPS / "frames_domain.parquet") if (OPS / "frames_domain.parquet").exists() else pd.DataFrame()
    df = pd.concat([k, h, s, dom], ignore_index=True)
    for c in STATES:
        if c not in df:
            df[c] = np.nan
    # a YouTube id that is test anywhere is test everywhere (K700 and CrossTask/COIN can share videos)
    test_src = set(df.loc[df["split"] == "test", "source"])
    df.loc[df["source"].isin(test_src), "split"] = "test"
    df["frame_id"] = df["path"].map(lambda p: Path(p).stem) + "|" + df["dataset"]
    dom = df["dataset"] == "domain"  # domain frames are named 00001.jpg per clip: prefix the clip id
    df.loc[dom, "frame_id"] = df.loc[dom, "path"].map(lambda p: f"{Path(p).parent.name}_{Path(p).stem}|domain")
    assert df["frame_id"].is_unique, "frame_id must be unique"
    df.to_parquet(TABLE)
    print(df.groupby(["dataset", "split"]).size())
    for c in STATES:
        print(c, df.groupby("dataset")[c].agg(lambda x: f"{int((x == 1).sum())}/{int(x.notna().sum())}").to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frames_k700", "frames_howto", "frames_howto_ff", "howto_subsample", "frames_howto_box", "stills", "yolo", "table"])
    ap.add_argument("--train-per-class", type=int, default=120)
    a = ap.parse_args()
    {"frames_k700": lambda: frames_k700(train_per_class=a.train_per_class), "frames_howto": frames_howto, "frames_howto_ff": frames_howto_ff, "howto_subsample": howto_subsample, "frames_howto_box": frames_howto_box, "stills": stills_copy, "yolo": yolo_copy, "table": table}[a.cmd]()
