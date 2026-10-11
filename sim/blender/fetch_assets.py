"""Download the CC0 assets for the synthetic bay scene (run on the box; no login needed).

    python -m sim.blender.fetch_assets            -> $SEEN_DATA/raw/sim_assets/...

Every asset goes into its own directory with a SOURCE.json (url, licence, sha256). Licences were read on the
asset pages on 2026-10-10: Kenney "Car Kit" and "Mini Characters" = Creative Commons CC0 (kenney.nl asset page
"License" row); Poly Haven = CC0 for all assets (polyhaven.com/license).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from seen.paths import RAW

OUT = RAW / "sim_assets"
KENNEY = {
    "kenney_car_kit": "https://kenney.nl/media/pages/assets/car-kit/1a312ec241-1775131960/kenney_car-kit.zip",
    "kenney_mini_characters": "https://kenney.nl/media/pages/assets/mini-characters/bfc7e272b4-1774770718/kenney_mini-characters.zip",
}
HDRIS = ["auto_service", "autoshop_01", "garage", "skylit_garage", "machine_shop_02", "empty_workshop"]
TEXTURES = ["concrete_floor_02", "concrete_floor_painted", "concrete_floor_worn_001", "concrete_wall_004", "painted_plaster_wall"]
UA = {"User-Agent": "seen-hackathon-asset-fetch/0.1 (research prototype)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def _save(dest: Path, url: str, licence: str, page: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(_get(url))
    meta = {"url": url, "page": page, "licence": licence, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size}
    (dest.parent / "SOURCE.json").write_text(json.dumps(meta, indent=1))
    return dest


def main() -> None:
    for name, url in KENNEY.items():
        z = _save(OUT / name / f"{name}.zip", url, "CC0-1.0", f"https://kenney.nl/assets/{name.replace('kenney_', '').replace('_', '-')}")
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():  # extract only model/texture files, never scripts
                if m.lower().endswith((".glb", ".gltf", ".bin", ".png", ".txt")) and ".." not in m:
                    zf.extract(m, OUT / name / "x")
    for h in HDRIS:
        files = json.loads(_get(f"https://api.polyhaven.com/files/{h}"))
        _save(OUT / "polyhaven" / "hdri" / h / f"{h}_1k.hdr", files["hdri"]["1k"]["hdr"]["url"], "CC0-1.0",
              f"https://polyhaven.com/a/{h}")
    for t in TEXTURES:
        files = json.loads(_get(f"https://api.polyhaven.com/files/{t}"))
        for key, fn in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            if key in files and "1k" in files[key]:
                _save(OUT / "polyhaven" / "tex" / t / f"{t}_{fn}_1k.jpg", files[key]["1k"]["jpg"]["url"], "CC0-1.0",
                      f"https://polyhaven.com/a/{t}")
                (OUT / "polyhaven" / "tex" / t / "SOURCE.json").rename(OUT / "polyhaven" / "tex" / t / f"SOURCE_{fn}.json")
    print("assets in", OUT)


if __name__ == "__main__":
    main()
