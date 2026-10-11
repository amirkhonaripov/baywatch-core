"""Draw jury-set *candidates* from Wikimedia Commons (how the set was sampled; curation and labels are in jury.jsonl).

    python -m eval.jury.select --out <dir>          # writes <dir>/candidates.jsonl and thumbnails for review

Only freely licensed files are kept (CC0, public domain, CC BY, CC BY-SA). The licence, author and file page of
every file are stored. Commons search results are ordered by relevance, so the draw is reproducible only up to
Commons' index changing; the curated list in jury.jsonl is the fixed record.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "SeenEval/0.1 (non-commercial research prototype; jury-style test set)"
FREE = re.compile(r"^(cc0|cc[- ]?zero|public domain|pd|cc[- ]by(-sa)?( |-)?\d|cc[- ]by(-sa)?$|attribution|no restrictions)", re.I)

# group -> list of ("cat", Category, depth) or ("search", query); n = how many candidates to draw
SPEC: dict[str, dict] = {
    "uz_car": {"n": 22, "src": [("cat", "Automobiles in Tashkent", 1), ("cat", "Unidentified automobiles in Uzbekistan", 0),
                                 ("cat", "Automobiles in Uzbekistan", 0), ("search", "Chevrolet Cobalt Uzbekistan"),
                                 ("search", "Daewoo Damas Tashkent"), ("cat", "Damaged automobiles in Uzbekistan", 0)]},
    "brand_car": {"n": 10, "src": [("search", "BYD Song Plus"), ("search", "BYD Han EV"), ("search", "Toyota Camry XV70"),
                                    ("search", "Toyota Land Cruiser 300")]},
    "motorcycle": {"n": 16, "src": [("search", "Ducati Panigale"), ("search", "Ducati Monster 2021"), ("search", "Ducati Scrambler"),
                                     ("search", "Ducati Multistrada V4"), ("search", "motorcycle street Tashkent")]},
    "street": {"n": 12, "src": [("search", "Tashkent street traffic cars"), ("search", "Almaty traffic street"),
                                 ("search", "Bishkek street cars"), ("cat", "Road traffic in Tashkent", 1)]},
    "parking_gate": {"n": 14, "src": [("search", "parking lot cars aerial"), ("search", "boom barrier parking entrance"),
                                       ("search", "parking garage interior cars"), ("search", "car park barrier gate"),
                                       ("cat", "Boom barriers", 0)]},
    "garage": {"n": 20, "src": [("cat", "Stationary automobile service lifts", 0), ("cat", "Four post lifts", 0),
                                 ("search", "auto repair shop interior car lift"), ("search", "mechanic working on car garage"),
                                 ("search", "tire change workshop")]},
    "wash": {"n": 12, "src": [("cat", "Car washes", 1), ("search", "car wash bay"), ("search", "self-service car wash")]},
    "closeup": {"n": 20, "src": [("search", "car dent damage close-up"), ("search", "car wheel rim tire close"),
                                  ("search", "flat tire car"), ("search", "vehicle registration plate Uzbekistan"),
                                  ("search", "scratched car door"), ("search", "car headlight close-up")]},
    "night_rain": {"n": 14, "src": [("search", "night traffic cars street lights"), ("search", "rain cars street wet"),
                                     ("search", "night parking lot cars"), ("search", "snow cars street Tashkent")]},
    "non_car": {"n": 20, "src": [("search", "plov Uzbek cuisine"), ("search", "portrait woman"), ("search", "identity card specimen"),
                                  ("search", "domestic cat"), ("search", "Registan Samarkand"), ("search", "office interior"),
                                  ("search", "bicycle parked street"), ("search", "toy car model"), ("search", "Chorsu bazaar"),
                                  ("search", "car drawing illustration")]},
    "video": {"n": 14, "src": [("vsearch", "traffic road cars"), ("vsearch", "car wash"), ("vsearch", "motorcycle ride"),
                                ("vsearch", "car repair garage"), ("vsearch", "parking lot"), ("vsearch", "cat playing"),
                                ("vsearch", "Tashkent street")]},
}


def _ctx():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ctx()) as r:
        return r.read()


_last = [0.0]
MIN_GAP_S = 1.5  # Wikimedia rate limits anonymous API clients; stay well below them


def api(params: dict) -> dict:
    q = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2", "maxlag": "5"})
    for attempt in range(6):
        wait = _last[0] + MIN_GAP_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            return json.loads(http_get(f"{API}?{q}"))
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(ra) if ra and ra.isdigit() else 15 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {params}")


def strip_html(s: str | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def file_info(titles: list[str], width: int = 1280) -> list[dict]:
    out = []
    for i in range(0, len(titles), 40):
        d = api({"action": "query", "titles": "|".join(titles[i:i + 40]), "prop": "imageinfo",
                 "iiprop": "url|size|mime|extmetadata|metadata", "iiurlwidth": str(width)})
        for p in d.get("query", {}).get("pages", []):
            ii = (p.get("imageinfo") or [None])[0]
            if not ii:
                continue
            em = ii.get("extmetadata", {})
            md = {m["name"]: m["value"] for m in (ii.get("metadata") or []) if isinstance(m, dict) and "name" in m}
            out.append({
                "title": p["title"], "page": ii.get("descriptionurl"), "url": ii.get("url"),
                "thumb_url": ii.get("thumburl"), "width": ii.get("width"), "height": ii.get("height"),
                "size": ii.get("size"), "mime": ii.get("mime"), "duration_s": md.get("length") or ii.get("duration"),
                "licence": strip_html(em.get("LicenseShortName", {}).get("value")),
                "licence_url": strip_html(em.get("LicenseUrl", {}).get("value")),
                "author": strip_html(em.get("Artist", {}).get("value"))[:200],
                "credit": strip_html(em.get("Credit", {}).get("value"))[:200],
                "usage_terms": strip_html(em.get("UsageTerms", {}).get("value")),
                "date": strip_html(em.get("DateTimeOriginal", {}).get("value"))[:40],
            })
    return out


def cat_files(cat: str, depth: int, limit: int = 200) -> list[str]:
    titles, cats = [], [cat]
    for d in range(depth + 1):
        nxt = []
        for c in cats:
            r = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{c}", "cmlimit": "200",
                     "cmtype": "file|subcat"})
            for m in r.get("query", {}).get("categorymembers", []):
                if m["title"].startswith("File:"):
                    titles.append(m["title"])
                elif m["title"].startswith("Category:") and d < depth:
                    nxt.append(m["title"][9:])
            if len(titles) >= limit:
                return titles[:limit]
        cats = nxt[:15]
    return titles


def search_files(q: str, video: bool = False, limit: int = 40) -> list[str]:
    ft = "filetype:video" if video else "filetype:bitmap"
    r = api({"action": "query", "list": "search", "srsearch": f"{q} {ft}", "srnamespace": "6", "srlimit": str(limit)})
    return [x["title"] for x in r.get("query", {}).get("search", [])]


def is_free(info: dict) -> bool:
    lic = info.get("licence") or ""
    if re.search(r"\b(nc|nd|non-?commercial|fair use)\b", lic, re.I):
        return False
    return bool(FREE.search(lic)) or lic.lower().startswith(("cc by", "cc-by", "cc0", "public domain", "pd"))


def _ok(group: str, i: dict) -> bool:
    if not is_free(i):
        return False
    if group == "video":
        if not (i["mime"] or "").startswith("video/"):
            return False
        dur = float(i.get("duration_s") or 0)
        return 2 <= dur <= 90 and (i.get("size") or 0) <= 80e6
    return (i["mime"] or "").startswith("image/") and i["mime"] != "image/svg+xml" and (i.get("width") or 0) >= 400


def draw(group: str, spec: dict, seen: set[str]) -> list[dict]:
    """Round-robin over the group's sources so no single category/query dominates."""
    per_src: list[list[str]] = []
    for s in spec["src"]:
        if s[0] == "cat":
            titles = cat_files(s[1], s[2])
        else:
            titles = search_files(s[1], video=s[0] == "vsearch")
        per_src.append([t for t in dict.fromkeys(titles) if t not in seen][:30])
    cap = -(-spec["n"] // len(per_src)) + 1
    infos = {i["title"]: i for i in file_info(sorted({t for ts in per_src for t in ts}))}
    queues = [[infos[t] for t in ts if t in infos and _ok(group, infos[t])] for ts in per_src]
    out, taken = [], [0] * len(queues)
    while len(out) < spec["n"] and any(queues):
        for k, q in enumerate(queues):
            if q and taken[k] < cap and len(out) < spec["n"]:
                i = q.pop(0)
                if i["title"] in seen:
                    continue
                out.append({**i, "group": group, "via": " ".join(map(str, spec["src"][k]))})
                seen.add(i["title"])
                taken[k] += 1
        if all(not q or taken[k] >= cap for k, q in enumerate(queues)):
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="")
    ap.add_argument("--no-thumbs", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    groups = [g for g in a.groups.split(",") if g] or list(SPEC)
    seen: set[str] = set()
    rows = []
    for g in groups:
        gp = out / f"group_{g}.jsonl"
        if gp.exists():
            got = [json.loads(l) for l in gp.read_text().splitlines() if l.strip()]
        else:
            got = draw(g, SPEC[g], seen)
            gp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in got))
        seen |= {r["title"] for r in got}
        print(f"{g}: {len(got)} candidates", flush=True)
        rows += got
    with (out / "candidates.jsonl").open("w") as f:
        for i, r in enumerate(rows):
            r["cid"] = f"c{i:03d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not a.no_thumbs:
        for r in rows:
            u = r.get("thumb_url") if not (r.get("mime") or "").startswith("video/") else None
            if not u:
                continue
            dst = out / "thumbs" / f"{r['cid']}.jpg"
            if dst.exists():
                continue
            try:
                dst.write_bytes(http_get(u))
            except Exception as e:
                print("thumb failed", r["cid"], e)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
