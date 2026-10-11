"""Wikimedia Commons car-lift photos (extra `raised` stills). Licence per file is recorded.

  python -m lanes.ops.acquire.commons   # -> data/raw/ops_commons/<pageid>.jpg + meta.csv
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request

from seen.paths import RAW

OUT = RAW / "ops_commons"
CATS = ["Category:Stationary automobile service lifts", "Category:Four post lifts"]
API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "seen-research/0.1 (hackathon prototype; contact via project)"}


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in CATS:
        r = _get(dict(action="query", generator="categorymembers", gcmtitle=cat, gcmtype="file", gcmlimit=500,
                      prop="imageinfo", iiprop="url|extmetadata|mime", iiurlwidth=1024))
        for pid, p in r.get("query", {}).get("pages", {}).items():
            ii = p["imageinfo"][0]
            if not ii.get("mime", "").startswith("image/") or ii["mime"] == "image/svg+xml":
                continue
            md = ii.get("extmetadata", {})
            lic = md.get("LicenseShortName", {}).get("value", "not stated")
            artist = re.sub("<[^>]+>", "", md.get("Artist", {}).get("value", ""))[:120]
            dst = OUT / f"{pid}.jpg"
            if not dst.exists():
                data = urllib.request.urlopen(urllib.request.Request(ii.get("thumburl") or ii["url"], headers=UA), timeout=60).read()
                dst.write_bytes(data)
                time.sleep(0.3)
            rows.append(dict(file=dst.name, title=p["title"], category=cat, licence=lic, artist=artist,
                             url=ii["descriptionurl"]))
    with (OUT / "meta.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r["licence"] for r in rows))


if __name__ == "__main__":
    main()
