"""NDISPark detection AP on the DAY images (train 112 + validation 30, boxes) for several zero-shot detectors.

    python -m lanes.flow.eval_ndispark_gap --models rfdetr-base,... --exp zeroshot_ndispark

FINDING (checked by eye and by luminance, see docs/lanes/flow_notes.md): the Zenodo "validation" split is daytime, like "train"; only the
"test" split (117 images) is night, and it has vehicle COUNTS only (no boxes). So there is no box-level night AP on NDISPark; the night
evidence is counting MAE (eval_count.py) and the night sequences of UA-DETRAC (eval_detect.py). Mean luminance per split is stored in the notes.
"""
import argparse

import numpy as np

from lanes.flow import data as D
from lanes.flow.common import APAccumulator, component, save_results
from lanes.flow.data import mean_luminance, ndispark_items
from lanes.flow.eval_detect import filter_ignored, vehicle_preds
from lanes.flow.run_detect import cache_path
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    lum = {s: float(np.mean([mean_luminance(i.path) for i in ndispark_items(s)])) for s in ("train", "validation")}
    lum["test"] = float(np.mean([mean_luminance(i.path) for i in D.ndispark_test_counts()]))
    res, rows = [], []
    for m in a.models.split(","):
        acc = APAccumulator()
        n = 0
        for split in ("train", "validation"):
            items = ndispark_items(split)
            dets = pickle.load(open(cache_path(f"ndispark_{split}", m), "rb"))["dets"]
            for it in items:
                acc.add(it.id, filter_ignored(vehicle_preds(dets[it.id]), it.ignore), it.gt)
                n += 1
        ms = acc.metrics(n_boot=a.n_boot) + acc.op_metrics(0.4, n_boot=a.n_boot)
        res.append(component(f"flow.det.ndispark.{m}.day", "Vehicle detection, daytime parking cameras", "NDISPark (train 112 + validation 30 images, all day)", n, ms,
                             split="day", model=m, domain="parking camera",
                             notes=f"{acc.n_gt} GT boxes in {n} images; zero-shot (the models never saw NDISPark); bootstrap over images; mean luminance train {lum['train']:.0f}, "
                                   f"validation {lum['validation']:.0f}, test (night) {lum['test']:.0f}"))
        rows.append((m, {x['name']: x for x in ms}))
    p = save_results(a.exp, res)
    md = "| model | AP50 [95% CI] | AP75 [95% CI] | AP50:95 | recall@0.4 | precision@0.4 |\n|---|---|---|---|---|---|\n"
    for m, d in rows:
        f = lambda k: f"{d[k]['value']:.3f} [{d[k]['ci_low']:.3f},{d[k]['ci_high']:.3f}]"  # noqa: E731
        md += f"| {m} | {f('AP50')} | {f('AP75')} | {d['AP50:95']['value']:.3f} | {d['recall@conf0.4']['value']:.3f} | {d['precision@conf0.4']['value']:.3f} |\n"
    (p.parent / "table.md").write_text(md + f"\nmean luminance (0-255): train {lum['train']:.1f}, validation {lum['validation']:.1f}, test (night, counts only) {lum['test']:.1f}\n")
    print(md, lum)


if __name__ == "__main__":
    main()
