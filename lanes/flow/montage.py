"""Contact sheets for label QA: python -m lanes.flow.montage <manifest.jsonl> <label> <source|all> <out.jpg> [n]"""
import json
import random
import sys

import cv2
import numpy as np


def main():
    man, label, source, out = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    recs = [json.loads(l) for l in open(man) if l.strip()]
    recs = [r for r in recs if r["label"] == label and (source == "all" or r["source"] == source)]
    random.Random(1).shuffle(recs)
    tiles = []
    for r in recs[:n]:
        im = cv2.imread(r["path"])
        if im is None:
            continue
        h, w = im.shape[:2]
        s = 200 / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        t = np.full((200, 200, 3), 30, np.uint8)
        t[: im.shape[0], : im.shape[1]] = im
        cv2.putText(t, r["source"][-18:], (3, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        tiles.append(t)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros((200, 200, 3), np.uint8))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.concatenate(rows, 0))
    print(len(recs), "records for", label, source)


if __name__ == "__main__":
    main()
