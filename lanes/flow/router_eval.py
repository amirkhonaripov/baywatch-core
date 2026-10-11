"""Scene-router training and evaluation (SigLIP2 zero-shot vs linear probe vs blend), held-out *by source*.

    python -m lanes.flow.router_eval embed  --model google/siglip2-so400m-patch14-384      # GPU: embeddings of manifest.jsonl
    python -m lanes.flow.router_eval eval   --model google/siglip2-so400m-patch14-384 --exp router_so400m   # CPU
    python -m lanes.flow.router_eval fit    --model ...                                    # train on ALL sources -> artifacts/flow/router/

Split rule (documented in docs/lanes/flow_splits.md): for every scene, whole sources are held out (sorted by md5(source)) until >= 35%
of that scene's images are held out, keeping at least one source for training. Nothing from a held-out source is used to fit the probe,
choose C, or set the rejection threshold (those use leave-one-source-out inside the training sources).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from eval.stats import accuracy_coverage, confusion, ece, metric, prop_metric, wilson_ci
from lanes.flow.common import artifact_dir, component, prep_dir, save_results
from lanes.flow.router_model import PROMPT_SETS, PROMPTS, Probe, SigLIP2Encoder, softmax, zero_shot_probs
from seen.vocab import SCENES


def short(model: str) -> str:
    return model.split("/")[-1]


def load_manifest() -> list[dict]:
    p = prep_dir("router") / "manifest.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def embed(model: str, device: str = "cuda:0", batch: int = 32):
    """Embed manifest.jsonl with the frozen encoder; incremental (ids already in the cache are not recomputed)."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    recs = load_manifest()
    cache_f = prep_dir("router") / f"emb_{short(model)}.npz"
    old = {}
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=False)
        old = {i: (e, l, s) for i, e, l, s in zip(z["ids"], z["emb"], z["labels"], z["sources"])}
    todo = [r for r in recs if r["id"] not in old]
    print("cached", len(old), "to embed", len(todo), flush=True)
    enc = SigLIP2Encoder(model, device)
    texts = {k: enc.class_text_embeddings(v) for k, v in PROMPT_SETS.items()}
    text = texts["v1"]
    scale, bias = enc.logit_params

    def ld(r):
        im = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    ex = ThreadPoolExecutor(4)
    for i in range(0, len(todo), batch * 4):
        chunk = todo[i:i + batch * 4]
        ims = list(ex.map(ld, chunk))
        ok = [(r, im) for r, im in zip(chunk, ims) if im is not None]
        if not ok:
            continue
        e = enc.encode_images([im for _, im in ok], batch)
        for (r, _), row in zip(ok, e):
            old[r["id"]] = (row, r["label"], r["source"])
        print(i, len(todo), flush=True)
    keep = [r["id"] for r in recs if r["id"] in old]  # manifest order, only current manifest members
    np.savez(cache_f, emb=np.stack([old[i][0] for i in keep]), ids=np.array(keep), labels=np.array([old[i][1] for i in keep]),
             sources=np.array([old[i][2] for i in keep]), text=text, text_v1=texts["v1"], text_v2=texts["v2"], scale=scale, bias=bias)


def load_emb(model: str) -> dict:
    z = np.load(prep_dir("router") / f"emb_{short(model)}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def pick_holdout(labels: np.ndarray, sources: np.ndarray, frac: float = 0.35) -> set[str]:
    held: set[str] = set()
    for s in SCENES:
        cnt = Counter(sources[labels == s])
        if len(cnt) < 2:
            continue
        tot = sum(cnt.values())
        got = 0
        for src in sorted(cnt, key=lambda x: hashlib.md5(x.encode()).hexdigest()):
            if got >= frac * tot or len(held & set(cnt)) >= len(cnt) - 1:
                break
            if got == 0 or got + cnt[src] <= 0.55 * tot:  # always hold out one source; never more than ~55% of a scene
                held.add(src)
                got += cnt[src]
    return held


def fit_probe(X, y, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    classes = [str(c) for c in clf.classes_]
    W = np.zeros((len(SCENES), X.shape[1]), dtype=np.float32)
    b = np.full(len(SCENES), -20.0, dtype=np.float32)  # scenes absent from training get ~0 probability
    for k, c in enumerate(classes):
        W[SCENES.index(c)] = clf.coef_[k]
        b[SCENES.index(c)] = clf.intercept_[k]
    return Probe(W, b, list(SCENES))


def logo_cv_choose_c(X, y, groups, Cs=(0.1, 0.3, 1, 3, 10, 30)):
    """Leave-one-source-out accuracy (macro over held sources with >= 15 images) -> best C. Training sources only."""
    best, best_c, table = -1, 1.0, {}
    cls_of = {s: y[groups == s][0] for s in set(groups)}
    n_src = Counter(cls_of.values())
    srcs = [s for s in sorted(set(groups)) if (groups == s).sum() >= 15 and n_src[cls_of[s]] >= 2]  # a scene needs another source left in training
    if not srcs:
        return 1.0, {}
    for C in Cs:
        accs = []
        for s in srcs:
            tr, te = groups != s, groups == s
            if len(set(y[tr])) < 2:
                continue
            p = fit_probe(X[tr], y[tr], C).probs(X[te])
            accs.append((np.array(SCENES)[p.argmax(1)] == y[te]).mean())
        m = float(np.mean(accs)) if accs else 0.0
        table[C] = m
        if m > best:
            best, best_c = m, C
    return best_c, table


def loso_probs(X, y, groups, C, blend_text=None, scale=None, bias=None, blend=None, min_n=15):
    """Out-of-source probabilities for every sample of every source with >= min_n images whose scene has another source (probe, optionally blended)."""
    cls_of = {s: y[groups == s][0] for s in set(groups)}
    n_src = Counter(cls_of.values())
    P, idx = [], []
    for s in sorted(set(groups)):
        m = groups == s
        if m.sum() < min_n or n_src[cls_of[s]] < 2:
            continue
        p = fit_probe(X[~m], y[~m], C).probs(X[m])
        if blend is not None:
            p = blend * p + (1 - blend) * zero_shot_probs(X[m], blend_text, scale, bias)
        P.append(p)
        idx += list(np.where(m)[0])
    return np.concatenate(P), np.array(idx)


def loso_parts(X, y, groups, C, text, scale, bias, min_n=15):
    """Out-of-source (probe probs, zero-shot probs, indices) for every sample of every source that has a sibling source of the same scene."""
    cls_of = {s: y[groups == s][0] for s in set(groups)}
    n_src = Counter(cls_of.values())
    PP, PZ, idx = [], [], []
    for s in sorted(set(groups)):
        m = groups == s
        if m.sum() < min_n or n_src[cls_of[s]] < 2:
            continue
        PP.append(fit_probe(X[~m], y[~m], C).probs(X[m]))
        PZ.append(zero_shot_probs(X[m], text, scale, bias))
        idx += list(np.where(m)[0])
    return np.concatenate(PP), np.concatenate(PZ), np.array(idx)


def choose_blend(PP, PZ, y_true, weights=(0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)):
    """Probe weight with the best out-of-source macro recall (weights are tried on the training sources only)."""
    best, best_w, tab = -1.0, 0.5, {}
    for w in weights:
        pred = np.array(SCENES)[(w * PP + (1 - w) * PZ).argmax(1)]
        rec = np.mean([(pred[y_true == c] == c).mean() for c in SCENES if (y_true == c).any()])
        tab[w] = float(rec)
        if rec > best + 1e-9:
            best, best_w = rec, w
    return best_w, tab


def choose_tau(probs, y_true, target=0.9):
    """Smallest max-prob threshold whose answered-subset accuracy is >= target (selective classification on out-of-source predictions)."""
    pred = np.array(SCENES)[probs.argmax(1)]
    conf = probs.max(1)
    for tau in np.round(np.arange(0.2, 0.96, 0.01), 2):
        m = conf >= tau
        if m.sum() >= 0.5 * len(y_true) and (pred[m] == y_true[m]).mean() >= target:
            return float(tau), float(m.mean()), float((pred[m] == y_true[m]).mean())
    return 0.45, 1.0, float((pred == y_true).mean())


def macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = ((y == c) & (p == c)).sum()
        fp = ((y != c) & (p == c)).sum()
        fn = ((y == c) & (p != c)).sum()
        if tp + fn:
            f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def head_metrics(name: str, probs: np.ndarray, y: np.ndarray, n_boot: int = 500) -> tuple[list[dict], dict]:
    pred = np.array(SCENES)[probs.argmax(1)]
    conf = probs.max(1)
    correct = pred == y
    k, n = int(correct.sum()), len(y)
    ms = [prop_metric("accuracy", k, n)]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        bs.append(macro_f1(y[i], pred[i], list(SCENES)))
    ms.append(metric("macro_F1", macro_f1(y, pred, list(SCENES)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), "fraction"))
    for c in SCENES:
        m = y == c
        if m.sum():
            ms.append(prop_metric(f"recall[{c}]", int((pred[m] == c).sum()), int(m.sum())))
        pm = pred == c
        if pm.sum():
            ms.append(prop_metric(f"precision[{c}]", int((y[pm] == c).sum()), int(pm.sum())))
    ms.append(metric("ECE", ece(conf, correct), unit="fraction"))
    return ms, confusion(list(y), list(pred), list(SCENES))


def run_eval(model: str, exp: str, blend: float = 0.5, prompts: str = "v2"):
    z = load_emb(model)
    X, y, src = z["emb"], z["labels"], z["sources"]
    scale, bias = float(z["scale"]), float(z["bias"])
    z = {**z, "text": z[f"text_{prompts}"]}
    held = pick_holdout(y, src)
    te = np.array([s in held for s in src])
    tr = ~te
    print("held-out sources:", sorted(held))
    print("train", Counter(y[tr]), "test", Counter(y[te]))
    C, ctab = logo_cv_choose_c(X[tr], y[tr], src[tr])
    probe = fit_probe(X[tr], y[tr], C)
    PP, PZ, il0 = loso_parts(X[tr], y[tr], src[tr], C, z["text"], scale, bias)
    blend, blend_tab = choose_blend(PP, PZ, y[tr][il0])
    print("blend weight (probe share) chosen by leave-one-source-out on training sources:", blend, blend_tab)
    heads = {"zeroshot": zero_shot_probs(X[te], z["text"], scale, bias), "probe": probe.probs(X[te])}
    heads["blend"] = blend * heads["probe"] + (1 - blend) * heads["zeroshot"]
    heads_sel: dict = {}
    for hname, bl in (("probe", None), ("blend", blend), ("zeroshot", None)):
        if hname == "probe":
            Pl, il = PP, il0
        else:
            Pl, il = blend * PP + (1 - blend) * PZ, il0
        tau, cov, acc_ans = choose_tau(Pl, y[tr][il])
        p = heads[hname]
        if hname == "zeroshot":
            Pl, il = PZ, il0
            tau, cov, acc_ans = choose_tau(Pl, y[tr][il])
        pred = np.array(SCENES)[p.argmax(1)]
        ans = p.max(1) >= tau
        yt = y[te]
        sel = {"tau_from_train_sources_LOSO": tau, "loso_coverage": cov, "loso_acc_answered": acc_ans, "heldout_coverage": float(ans.mean()),
               "heldout_acc_answered": float((pred[ans] == yt[ans]).mean()) if ans.any() else float("nan"),
               "heldout_unknown_recall_incl_abstain": float((((pred == "unknown") | ~ans)[yt == "unknown"]).mean()) if (yt == "unknown").any() else float("nan"),
               "heldout_vehicle_scenes_wrongly_rejected": float((~ans)[yt != "unknown"].mean())}
        heads_sel[hname] = sel
    results, report = [], {"selective": heads_sel, "blend_weight": blend, "blend_table": blend_tab, "heldout_sources": sorted(held), "C": C, "C_table": ctab, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                           "test_counts": dict(Counter(y[te])), "train_counts": dict(Counter(y[tr]))}
    for h, p in heads.items():
        ms, conf = head_metrics(h, p, y[te])
        results.append(component(f"router.{short(model)}.{prompts}.{h}", f"Scene router ({h})", "router set (public sources)", int(te.sum()), ms, split="held-out sources",
                                 model=f"{short(model)}:{h}", domain="mixed public photos + fixed cameras", lane="router", confusion=conf,
                                 notes=f"prompt set {prompts}; held-out sources: {', '.join(sorted(held))}; train n={int(tr.sum())}; probe C={C}; blend probe weight {blend} (leave-one-source-out)"))
        ac = accuracy_coverage(p.max(1), np.array(SCENES)[p.argmax(1)] == y[te])
        report[h] = {"accuracy_coverage": ac}
    # leave-one-source-out over ALL sources (probe trained on everything else) -> per-source recall of its scene
    loso = {}
    for s in sorted(set(src)):
        m = src == s
        if m.sum() < 15:
            continue
        pr = fit_probe(X[~m], y[~m], C).probs(X[m])
        lab = y[m][0]
        zs = zero_shot_probs(X[m], z["text"], scale, bias)
        loso[s] = {"scene": str(lab), "n": int(m.sum()), "probe_recall": float((np.array(SCENES)[pr.argmax(1)] == y[m]).mean()),
                   "zeroshot_recall": float((np.array(SCENES)[zs.argmax(1)] == y[m]).mean())}
    report["leave_one_source_out"] = loso
    per_src = {}
    for h in ("zeroshot", "probe", "blend"):
        predh = np.array(SCENES)[heads[h].argmax(1)]
        per_src[h] = {sname: {"scene": str(y[te][src[te] == sname][0]), "n": int((src[te] == sname).sum()),
                              "accuracy": float((predh[src[te] == sname] == y[te][src[te] == sname]).mean())} for sname in sorted(set(src[te]))}
        keep = np.array([not str(sn).startswith("coco_street") for sn in src[te]])  # coco_street labels are rule-based (not hand-checked)
        per_src[h]["_accuracy_excluding_coco_street"] = float((predh[keep] == y[te][keep]).mean())
    report["heldout_per_source"] = per_src
    p = save_results(exp, results)
    (p.parent / "router_report.json").write_text(json.dumps(report, indent=1))
    for r in results:
        ms = {m["name"]: m for m in r["metrics"]}
        print(r["model"], "acc %.3f [%.3f,%.3f] macroF1 %.3f ECE %.3f" % (ms["accuracy"]["value"], ms["accuracy"]["ci_low"], ms["accuracy"]["ci_high"], ms["macro_F1"]["value"], ms["ECE"]["value"]))
        print(r["confusion"]["labels"])
        for row, l in zip(r["confusion"]["matrix"], r["confusion"]["labels"]):
            print(f"{l:11s}", row)
    print(json.dumps(loso, indent=0))
    return results, report


def run_prompts(model: str):
    """Compare prompt sets by zero-shot accuracy on the TRAINING sources only (held-out sources are never looked at)."""
    z = load_emb(model)
    X, y, src = z["emb"], z["labels"], z["sources"]
    held = pick_holdout(y, src)
    tr = np.array([s not in held for s in src])
    out = {}
    for k in PROMPT_SETS:
        p = zero_shot_probs(X[tr], z[f"text_{k}"], float(z["scale"]), float(z["bias"]))
        pred = np.array(SCENES)[p.argmax(1)]
        rec = {c: float((pred[y[tr] == c] == c).mean()) for c in SCENES if (y[tr] == c).any()}
        out[k] = {"macro_recall": float(np.mean(list(rec.values()))), "accuracy": float((pred == y[tr]).mean()), "recall": rec}
        print(k, json.dumps(out[k]))
    best = max(out, key=lambda k: out[k]["macro_recall"])
    print("best prompt set on training sources:", best)
    return out, best


def run_fit(model: str, prompts: str = "v2"):
    """Final fit on ALL sources (after the held-out evaluation): probe weights, blend weight and unknown threshold chosen by leave-one-source-out."""
    z = load_emb(model)
    X, y, src = z["emb"], z["labels"], z["sources"]
    text, scale, bias = z[f"text_{prompts}"], float(z["scale"]), float(z["bias"])
    C, ctab = logo_cv_choose_c(X, y, src)
    probe = fit_probe(X, y, C)
    PP, PZ, il = loso_parts(X, y, src, C, text, scale, bias)
    w, wtab = choose_blend(PP, PZ, y[il])
    tau, cov, acc_ans = choose_tau(w * PP + (1 - w) * PZ, y[il])
    out = artifact_dir("router") / "probe.npz"
    probe.save(out, {"encoder": model, "prompt_set": prompts, "C": C, "C_table": ctab, "n_train": int(len(y)), "classes": list(SCENES), "counts": dict(Counter(y)),
                     "blend_probe_weight": w, "blend_table": wtab, "unknown_threshold": tau,
                     "unknown_threshold_basis": f"leave-one-source-out on the blended probabilities: answered-subset accuracy {acc_ans:.3f} at coverage {cov:.3f}",
                     "sources": sorted(set(src)), "jury_filter": "10 near-duplicates of eval/manifests/jury.jsonl removed before fitting (heldout_dropped.json)"})
    print("saved", out, "C", C, "blend", w, "unknown_threshold", tau, cov, acc_ans)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "eval", "fit", "prompts"])
    ap.add_argument("--prompts", default="v2")
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--exp", default="router")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "embed":
        embed(a.model, a.device)
    elif a.cmd == "eval":
        run_eval(a.model, a.exp, prompts=a.prompts)
    elif a.cmd == "prompts":
        run_prompts(a.model)
    else:
        run_fit(a.model, a.prompts)
