"""NDISPark vehicle counting: night (test split: counts only) vs day (validation split: boxes), threshold fixed on the day TRAIN split.

    python -m lanes.flow.eval_count --models rfdetr-base,rfdetr-large,yolo11x --exp zeroshot_ndispark_count

Protocol: predicted count = number of vehicle detections with score >= tau. tau is the one minimising MAE on the day train split (112 images,
GT = number of boxes); it is then applied unchanged to the 117 night test images (GT = ground_truth_test_counting.csv) and to the 30 day
validation images (GT = number of boxes). Note: the Zenodo "validation" split is DAYTIME (verified by eye and luminance), only "test" is night.
"oracle tau" (tuned on the evaluated split) is reported as an upper bound. CI = bootstrap over images; the night-minus-day MAE gap uses
independent resampling of the two image sets.
"""
import argparse
import pickle

import numpy as np

from eval.stats import mae_metric, metric
from lanes.flow.common import component, save_results
from lanes.flow.eval_detect import vehicle_preds
from lanes.flow.run_detect import cache_path, get_items

TAUS = np.round(np.arange(0.05, 0.95, 0.025), 3)


def counts(dataset, model):
    items = get_items(dataset)
    dets = pickle.load(open(cache_path(dataset, model), "rb"))["dets"]
    sc = [vehicle_preds(dets[it.id])[:, 4] for it in items]
    gt = np.array([it.extra["count"] if "count" in it.extra else len(it.gt) for it in items])
    return sc, gt


def mae_at(sc, gt, tau):
    pred = np.array([(s >= tau).sum() for s in sc])
    return pred, np.abs(pred - gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    res = []
    for m in a.models.split(","):
        sd, gd = counts("ndispark_train", m)
        tau = float(TAUS[int(np.argmin([mae_at(sd, gd, t)[1].mean() for t in TAUS]))])
        day_val = None
        for ds, name in (("ndispark_validation", "DAY validation (30 images, boxes)"), ("ndispark_test", "NIGHT test (117 images, counts)")):
            sc, gt = counts(ds, m)
            pred, err = mae_at(sc, gt, tau)
            o_tau = float(TAUS[int(np.argmin([mae_at(sc, gt, t)[1].mean() for t in TAUS]))])
            _, oerr = mae_at(sc, gt, o_tau)
            _, derr = mae_at(sd, gd, tau)
            ms = [mae_metric(err, "MAE_tau_from_day"), metric("mean_gt_count", float(gt.mean()), unit="vehicles/image"), metric("tau_from_day", tau),
                  metric("mean_signed_error", float((pred - gt).mean()), unit="vehicles/image"), mae_metric(oerr, "MAE_oracle_tau"), metric("oracle_tau", o_tau),
                  mae_metric(derr, "MAE_day_at_same_tau")]
            res.append(component(f"flow.count.{m}.{ds}", f"Vehicle counting, {name}", "NDISPark", len(gt), ms, split="night" if ds.endswith("test") else "day-validation", model=m, domain="parking camera",
                                 notes=f"threshold fixed on the day split ({len(gd)} images); bootstrap over images"))
            if ds == "ndispark_validation":
                day_val = (err, gt)
            else:
                from lanes.flow.eval_detect import diff_ci

                nl = list(range(len(err)))
                dl = list(range(len(day_val[0])))
                v, lo, hi = diff_ci(lambda c: float(np.mean(err[c])), nl, lambda c: float(np.mean(day_val[0][c])), dl)
                rel_n, rel_d = err.mean() / max(gt.mean(), 1e-9), day_val[0].mean() / max(day_val[1].mean(), 1e-9)
                res[-1]["metrics"] += [metric("MAE_night_minus_day", v, lo, hi, "vehicles/image"), metric("relative_MAE_night", float(rel_n), unit="MAE/mean count"),
                                       metric("relative_MAE_day_val", float(rel_d), unit="MAE/mean count")]
            print(m, ds, "MAE", err.mean(), "oracle", oerr.mean(), "day-train MAE", derr.mean(), "tau", tau)
    save_results(a.exp, res)


if __name__ == "__main__":
    main()
