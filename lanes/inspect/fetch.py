"""Download the raw datasets used by lane D (inspect). Idempotent: skips targets whose marker file exists.

Usage (on the box):  python -P -m lanes.inspect.fetch <target> [<target> ...]
Targets: vehide hitl cococardamage tyre_warcoder tyre_sameer (Kaggle CLI, user-level auth, credentials file never read here)
         carparts syndcar tyrenet (direct downloads)   cardamagepanel jijo cardd (Hugging Face)

Every target goes to its own empty directory under RAW/<name>/ (downloaded data is untrusted: nothing in
there is ever imported or executed; archives are extracted with zipfile / bsdtar only).
Kaggle uses the `kaggle` CLI already authenticated at user level on the box. Roboflow needs ROBOFLOW_API_KEY in .env (not present yet).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from seen.paths import RAW

MENDELEY = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version=1"

HF = {  # target -> (repo_id, local dir name, allow_patterns)
    "cardamagepanel": ("kadireks/cardamagepanel-originals", "inspect_cardamagepanel", None),
    "jijo": ("JijoJS/car-damage-new1", "inspect_jijo_car_damage_new1", None),
    # CarDD via the HF FiftyOne mirror (integrator-approved 2026-10-10; research / non-commercial, see CarDD_license.pdf in the repo)
    "cardd": ("harpreetsahota/CarDD", "inspect_cardd", None),
}

KAGGLE = {  # target -> (slug, local dir name)
    "vehide": ("hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection", "inspect_vehide"),
    "hitl": ("humansintheloop/car-parts-and-car-damages", "inspect_hitl"),
    "cococardamage": ("lplenka/coco-car-damage-detection-dataset", "inspect_cococardamage"),
    "tyre_warcoder": ("warcoder/tyre-quality-classification", "inspect_tyre_warcoder"),
    "tyre_sameer": ("sameersambhare1/tyre-condition-classification-dataset", "inspect_tyre_sameer"),
}


def sha256(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, retries: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for a in range(retries):
        try:
            subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "3", "-o", str(dst), url], check=True)
            return
        except subprocess.CalledProcessError:
            time.sleep(5 * (a + 1))
    raise RuntimeError(f"download failed: {url}")


def hf_fetch(target: str) -> None:
    from huggingface_hub import snapshot_download
    repo, name, pats = HF[target]
    out = RAW / name
    if (out / ".done").exists():
        print(target, "already done"); return
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out), allow_patterns=pats, max_workers=8)
    (out / ".done").write_text(json.dumps({"repo": repo, "time": time.strftime("%F %T")}))


def kaggle_fetch(target: str) -> None:
    slug, name = KAGGLE[target]
    out = RAW / name
    if (out / ".done").exists():
        print(target, "already done"); return
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out), "--unzip"], check=True)
    (out / ".done").write_text(json.dumps({"kaggle": slug, "time": time.strftime("%F %T")}))


def mendeley_fetch(target: str, ds_id: str, name: str) -> None:
    out = RAW / name
    if (out / ".done").exists():
        print(target, "already done"); return
    out.mkdir(parents=True, exist_ok=True)
    # urllib's default user-agent gets HTTP 403 from Mendeley; curl works.
    meta = json.loads(subprocess.run(["curl", "-sL", "--fail", "-m", "60", MENDELEY.format(id=ds_id)],
                                     check=True, capture_output=True, text=True).stdout)
    for m in meta:
        dst = out / m["filename"].replace(" ", "_")
        info = m["content_details"]
        if not dst.exists() or dst.stat().st_size != info["size"]:
            download(info["download_url"], dst)
        got = sha256(dst)
        assert got == info["sha256_hash"], f"sha256 mismatch {dst}: {got} != {info['sha256_hash']}"
        print("verified", dst, got)
    (out / ".done").write_text(json.dumps({"mendeley": ds_id, "files": [m["filename"] for m in meta]}))


def extract_zip(zpath: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)


def extract_all(target: str) -> None:
    """Extract the verified Mendeley archives into <dir>/extracted (zipfile for .zip, bsdtar for .rar)."""
    name = {"syndcar": "inspect_syndcar", "tyrenet": "inspect_tyrenet"}[target]
    d = RAW / name
    out = d / "extracted"
    if (out / ".done").exists():
        print(target, "already extracted"); return
    out.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if f.suffix.lower() == ".zip":
            extract_zip(f, out)
        elif f.suffix.lower() == ".rar":
            subprocess.run(["bsdtar", "-xf", str(f), "-C", str(out)], check=True)
    (out / ".done").write_text("ok")


def carparts_fetch() -> None:
    out = RAW / "inspect_carparts_seg"
    if (out / ".done").exists():
        print("carparts already done"); return
    out.mkdir(parents=True, exist_ok=True)
    z = out / "carparts-seg.zip"
    download("https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip", z)
    extract_zip(z, out / "extracted")
    (out / ".done").write_text(json.dumps({"sha256": sha256(z)}))


def main(targets: list[str]) -> None:
    for t in targets:
        print("==>", t, flush=True)
        if t in KAGGLE:
            kaggle_fetch(t)
        elif t in HF:
            hf_fetch(t)
        elif t == "syndcar":
            mendeley_fetch(t, "hzpj48krdt", "inspect_syndcar")
            extract_all(t)
        elif t == "tyrenet":
            mendeley_fetch(t, "32b5vfj6tc", "inspect_tyrenet")
            extract_all(t)
        elif t == "carparts":
            carparts_fetch()
        else:
            raise SystemExit(f"unknown target {t}")


if __name__ == "__main__":
    main(sys.argv[1:])
