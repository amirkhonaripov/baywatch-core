"""End-to-end check of the SHIPPED predictor (get_predictor: UZ detector + worldwide detector + no-guess rule) on the
real Uzbek test photos. Reports: correct reads, wrong reads (confident but wrong), and 'not readable' answers.
python -m lanes.ident.eval_predictor_uz"""
import json
from pathlib import Path

import cv2

from eval.stats import prop_metric
from lanes.ident.data import det_sets as ds
from lanes.ident.predict import get_predictor
from seen.predictor import Frame

labels = {r["id"]: r for r in map(json.loads, (Path(__file__).parent / "labels" / "uz_real_test_labels.jsonl").read_text().splitlines())}
recs = [r for r in ds.load_uz_real("test") if labels.get(r["id"], {}).get("readable")]
p = get_predictor(); p.load("cuda:0")
ok = wrong = unread = 0
for r in recs:
    p.reset()
    im = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
    dets = p.predict([Frame(0, 0.0, im)])[0].detections
    texts = [d.text for d in dets if d.text]
    gt = labels[r["id"]]["text"]
    if gt in texts:
        ok += 1
    elif texts:
        wrong += 1
    else:
        unread += 1
n = len(recs)
res = [prop_metric("correct_read", ok, n), prop_metric("wrong_read", wrong, n), prop_metric("not_readable", unread, n)]
print(json.dumps({"n": n, "metrics": res}, indent=1))
