"""Fetch CC-licensed Wikimedia Commons photos that show Uzbek licence plates (real, readable, licence recorded per file).

Usage (box): python -m lanes.ident.tools.fetch_commons_uz --out $SEEN_ROOT/data/raw/ident_uz/commons
Writes images (<=1600 px long side) + meta.jsonl {file, title, url, licence, author, width, height, source_query}.
Respects the Commons API etiquette (User-Agent, serial requests).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "seen-ident-lane/0.1 (hackathon research; amirkhon@humanola.com)"}
CATS = [
    "Category:License plates of Uzbekistan",
    "Category:Automobiles in Uzbekistan",
    "Category:Cars in Tashkent",
    "Category:Cars in Samarkand",
    "Category:Cars in Bukhara",
    "Category:Cars in Uzbekistan",
    "Category:Taxis in Uzbekistan",
    "Category:Chevrolet in Uzbekistan",
    "Category:Daewoo in Uzbekistan",
    "Category:Road transport in Uzbekistan",
    "Category:Traffic in Tashkent",
    "Category:Vehicle registration plates of Uzbekistan",
]
SEARCHES = [
    "Uzbekistan car license plate", "Uzbek license plate car", "Tashkent car number plate", "Tashkent street cars",
    "Samarkand cars parked", "Chevrolet Nexia Uzbekistan", "Chevrolet Cobalt Uzbekistan", "Chevrolet Gentra Uzbekistan",
    "Chevrolet Spark Uzbekistan", "Daewoo Matiz Uzbekistan", "UzAuto Motors car", "BYD Uzbekistan", "Toyota Tashkent",
    "Lada Uzbekistan", "Tashkent traffic", "Khiva car", "Bukhara car parked", "Fergana car", "Namangan car", "Andijan car",
    "UZ plate", "Uzbekistan vehicle registration plate photo",
]


def get(params, tries=4):
    for k in range(tries):
        r = requests.get(API, params={**params, "format": "json"}, headers=UA, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (k + 1))
    r.raise_for_status()


def cat_files(cat, depth=1, seen=None):
    seen = seen if seen is not None else set()
    out = []
    cont = {}
    while True:
        d = get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmlimit": 500, "cmtype": "file|subcat", **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            if m["ns"] == 6:
                out.append(m["title"])
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen:
                seen.add(m["title"])
                out += cat_files(m["title"], depth - 1, seen)
        if "continue" in d:
            cont = d["continue"]
        else:
            break
        time.sleep(0.3)
    return out


def search_files(q, limit=100):
    d = get({"action": "query", "list": "search", "srsearch": q, "srnamespace": 6, "srlimit": limit})
    return [m["title"] for m in d.get("query", {}).get("search", [])]


def info(titles):
    out = {}
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        d = get({"action": "query", "titles": "|".join(chunk), "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
                 "iiurlwidth": 1600})
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if ii:
                out[p["title"]] = ii
        time.sleep(0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    srcs: dict[str, str] = {}
    for c in CATS:
        try:
            for t in cat_files(c, depth=1):
                srcs.setdefault(t, c)
        except Exception as e:  # missing category etc.
            print("cat fail", c, e)
    for q in SEARCHES:
        for t in search_files(q):
            srcs.setdefault(t, "search:" + q)
        time.sleep(0.3)
    print("candidate titles:", len(srcs))
    inf = info(sorted(srcs))
    rows = []
    for t, ii in inf.items():
        if ii.get("mime") not in ("image/jpeg", "image/png"):
            continue
        md = ii.get("extmetadata", {})
        lic = (md.get("LicenseShortName", {}) or {}).get("value", "")
        author = re.sub("<[^>]+>", "", (md.get("Artist", {}) or {}).get("value", ""))[:120]
        rows.append({"title": t, "url": ii.get("thumburl") or ii["url"], "orig": ii["url"], "width": ii["width"], "height": ii["height"],
                     "licence": lic, "author": author, "source_query": srcs[t], "descr": re.sub("<[^>]+>", "", (md.get("ImageDescription", {}) or {}).get("value", ""))[:200]})
    print("jpeg/png:", len(rows))
    (out / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if a.dry:
        return
    (out / "img").mkdir(exist_ok=True)
    for i, r in enumerate(rows):
        fn = f"c{i:04d}.jpg" if not r["title"].lower().endswith(".png") else f"c{i:04d}.png"
        p = out / "img" / fn
        if not p.exists():
            try:
                rr = requests.get(r["url"], headers=UA, timeout=120)
                rr.raise_for_status()
                p.write_bytes(rr.content)
            except Exception as e:
                print("dl fail", r["title"], e)
                continue
            time.sleep(0.4)
        r["file"] = fn
    (out / "meta.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if "file" in r))
    print("downloaded", sum(1 for r in rows if "file" in r))


if __name__ == "__main__":
    main()
