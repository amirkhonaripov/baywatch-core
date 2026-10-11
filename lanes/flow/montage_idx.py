"""Numbered contact sheets for manual label curation.

    python -m lanes.flow.montage_idx <records.jsonl> <label> <out_prefix> [--per 48]
Tiles are numbered in manifest order within (label); curation decisions are stored in lanes/flow/curation.json as
{"<source>": {"drop": [record ids]}} so the cleaned set is reproducible.
"""
import json
import sys

import cv2
import numpy as np


def main():
    man, label, prefix = sys.argv[1:4]
    per = 48
    recs = [json.loads(l) for l in open(man) if l.strip()]
    recs = [r for r in recs if r["label"] == label]
    recs.sort(key=lambda r: (r["source"], r["id"]))
    T, cols = 150, 8
    for si in range(0, len(recs), per):
        chunk = recs[si:si + per]
        tiles = []
        for k, r in enumerate(chunk):
            im = cv2.imread(r["path"])
            t = np.full((T, T, 3), 30, np.uint8)
            if im is not None:
                h, w = im.shape[:2]
                s = T / max(h, w)
                im = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)))
                t[: im.shape[0], : im.shape[1]] = im
            idx = si + k
            cv2.rectangle(t, (0, 0), (38, 20), (0, 0, 0), -1)
            cv2.putText(t, str(idx), (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % cols:
            tiles.append(np.zeros((T, T, 3), np.uint8))
        rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
        out = f"{prefix}_{si // per:02d}.jpg"
        cv2.imwrite(out, np.concatenate(rows, 0))
        print(out, "indices", si, "-", si + len(chunk) - 1, "sources", sorted({r["source"] for r in chunk}))
    json.dump([r["id"] for r in recs], open(f"{prefix}_index.json", "w"))


if __name__ == "__main__":
    main()
