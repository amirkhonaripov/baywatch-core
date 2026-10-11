"""Fetch only the needed part of TyreCV (Kaggle maxfinkenbeiner/tyrecv-two-views-are-all-you-need, 14.5 GB, MIT label).

The archive holds 508 stereo pairs: `cropped/cropped/<uuid>_<depth mm>_{L,R}.png` (11 GB of full-resolution crops) and
`augmented_kaggle/.../{train,test}/<uuid>[-aug]_<depth bucket>_{L,R}.png` (3.4 GB; per tyre one un-augmented copy plus flipped /
histogram-matched copies) plus `stereo_calibration_data.npz`. We read the zip's central directory over HTTP range requests
(`remotezip`) through the Kaggle SDK's authenticated download redirect (credentials are handled by the SDK, never read here) and
extract ONLY the un-augmented augmented_kaggle images (variant '' = no '-flip' / '-hm*' / '-mE*' suffix) and the calibration file,
about 0.6 GB. Exact gauge depth (mm, 0.5 mm steps) is taken from the file names of the 11 GB originals (`cropped/`), which are
NOT downloaded; labels_mm.json maps tyre uuid -> mm. The signed URL is never printed or stored.

    scripts/box.sh -- python -P -m lanes.inspect.fetch_tyrecv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from seen.paths import RAW

OWNER, SLUG = "maxfinkenbeiner", "tyrecv-two-views-are-all-you-need"
CROP = re.compile(r"^cropped/cropped/([0-9a-f-]{36})_([0-9.]+)_([LR])\.png$")  # original crops: gauge depth in mm at 0.5 mm resolution
PAT = re.compile(r"^augmented_kaggle/augmented_kaggle/(train|test)/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(\d+)_([LR])\.png$")


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    from remotezip import RemoteZip

    out = RAW / "inspect_tyrecv"
    if (out / ".done").exists():
        print("already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiDownloadDatasetRequest()
        req.owner_slug, req.dataset_slug = OWNER, SLUG
        resp = kaggle.datasets.dataset_api_client.download_dataset(req)
        url = resp.url
        resp.close()
        with RemoteZip(url) as z:
            names = z.namelist()
            import json
            labels = {}
            for n in names:
                m = CROP.match(n)
                if m:
                    labels[m.group(1)] = float(m.group(2))
            (out / "labels_mm.json").write_text(json.dumps(labels))
            print("tyres with gauge label:", len(labels), flush=True)
            want = [n for n in names if PAT.match(n)] + [n for n in names if n.endswith("stereo_calibration_data.npz")]
            print("zip entries", len(names), "wanted", len(want), flush=True)
            for i, n in enumerate(want):
                dst = out / Path(n).name if not PAT.match(n) else out / PAT.match(n).group(1) / Path(n).name
                if dst.exists() and dst.stat().st_size > 0:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(dst, "wb") as f:
                    f.write(src.read())
                if i % 100 == 0:
                    print(i, flush=True)
    (out / ".done").write_text(f"{len(want)} files")


if __name__ == "__main__":
    sys.exit(main())
