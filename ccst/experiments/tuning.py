"""Item 2: hyper-parameter selection by nested cross-validation (docs/tuning_protocol.md).

Outer loop = the headline 5-fold GroupKFold by template; inner loop = 4-fold GroupKFold by template inside
the outer training portion. The configuration with the best mean inner macro-F1 is refit on the outer
training portion (standard calibration split) and scored once on the outer test fold. The strongest
baselines get the same nested selection over the LR regularisation C, so tuning is not one-sided.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from ..config import RELATION_LABELS, Config
from ..eval.metrics import (
    bootstrap_metric_ci, classification_report, false_contradiction_rate, paired_bootstrap_pvalue,
)
from ..relation.classifier import RelationClassifier
from ..utils import RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from ..viz import plots
from .common import feature_matrix, featurize
from .synthetic import PROPOSED, bench_paths
from .synthetic_analysis import load_state

DEFAULT = {"partial": True, "lam_u": 0.6, "alpha": 1.0, "C": 1.0}
C_GRID = (0.1, 1.0, 10.0)
CCST_GRID = ([{"partial": True, "lam_u": lu, "alpha": a} for lu in (0.3, 0.6, 1.0) for a in (0.5, 1.0, 2.0)]
             + [{"partial": False, "lam_u": 0.6, "alpha": a} for a in (0.5, 1.0, 2.0)])
BASELINES = ("Context clf. w/o transport", "Signed matcher w/o context", "NLI (BART-MNLI)", "SBERT (MiniLM)", "SPECTER")
LABELS = list(RELATION_LABELS)


def _macro(y, p) -> float:
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def _n_nondefault(cfg: dict) -> int:
    return sum(cfg.get(k) != v for k, v in DEFAULT.items() if k in cfg)


def _inner_score(X, y, groups, tr, C, seed) -> float:
    scores = []
    for a, b in GroupKFold(4).split(X[tr], y[tr], groups[tr]):
        ia, ib = tr[a], tr[b]
        clf = RelationClassifier(C=C, calibration="none", seed=seed).fit(X[ia], list(y[ia]))
        scores.append(_macro(y[ib], clf.predict(X[ib])))
    return float(np.mean(scores))


def nested_cv(candidates: dict, y: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """candidates: {config_key: (config_dict, feature_matrix)}; C is searched for every candidate."""
    n = len(y)
    proba = np.zeros((n, len(LABELS)))
    chosen = []
    for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(n), y, groups)):
        scored = []
        for key, (cfg, X) in candidates.items():
            for C in C_GRID:
                full = {**cfg, "C": C}
                scored.append((_inner_score(X, y, groups, tr, C, seed), -_n_nondefault(full), key, C, full))
        best = max(s[0] for s in scored)
        near = [s for s in scored if s[0] >= best - 0.001]
        default_hit = [s for s in near if _n_nondefault(s[4]) == 0]
        pick = default_hit[0] if default_hit else max(near, key=lambda s: (s[1], s[0]))
        score, _, key, C, full = pick
        chosen.append({"fold": fold, **full, "inner_macro_f1": score})
        X = candidates[key][1]
        # final fit: same calibration split as the headline protocol (common.cross_validate)
        g_tr = np.unique(groups[tr])
        rng = np.random.default_rng(seed + fold)
        cal_groups = set(rng.choice(g_tr, size=max(1, len(g_tr) // 4), replace=False))
        cal = np.array([i for i in tr if groups[i] in cal_groups])
        fit = np.array([i for i in tr if groups[i] not in cal_groups])
        clf = RelationClassifier(C=C, seed=seed).fit(X[fit], list(y[fit]))
        clf.calibrate(X[cal], list(y[cal]))
        proba[te] = clf.predict_proba(X[te])
    pred = [LABELS[k] for k in proba.argmax(1)]
    return {"proba": proba, "pred": pred, "chosen": chosen}


def _summary(name, y, out, variants, default_f1, seed) -> dict:
    rep = classification_report(list(y), out["pred"], out["proba"])
    lo, hi = bootstrap_metric_ci(list(y), out["pred"], _macro, n_boot=500, seed=seed)
    shift = np.isin(variants, ["hetero_ctx", "hetero_sim"])
    return {"method": name, "default_macro_f1": default_f1, "nested_macro_f1": rep["macro_f1"], "ci_lo": lo,
            "ci_hi": hi, "gain": rep["macro_f1"] - default_f1 if default_f1 is not None else None,
            "FCR_context_shift": false_contradiction_rate(list(y), out["pred"], shift),
            "contra_recall": rep["r_contradiction"],
            "chosen": "; ".join(_fmt(c) for c in out["chosen"])}


def _fmt(c: dict) -> str:
    parts = [f"C={c['C']:g}"]
    if "partial" in c:
        parts = [("partial" if c["partial"] else "full"), f"λu={c['lam_u']:g}", f"α={c['alpha']:g}"] + parts
    return " ".join(parts)


def run_one(cfg: Config, realism: bool, n_jobs: int) -> list[dict]:
    name = bench_paths(realism)[0]
    state = load_state(realism)
    S = state["S"]
    pairs = S["pairs"]
    y = np.array([p["label"] for p in pairs])
    groups = np.array([p["template"] for p in pairs])
    variants = np.array([p["variant"] for p in pairs])
    defaults = {r["method"]: r["macro_f1"] for r in json.loads((RESULTS_DIR / name / "metrics.json").read_text())["main"]}
    P = [(p["a"], p["b"]) for p in pairs]
    cands = {}
    with timer(f"[{name}] CCST features for {len(CCST_GRID)} configurations"):
        for c in CCST_GRID:
            f, _ = featurize(P, S["graphs"], S["metas"], replace(cfg.ccst, **c), cfg.context, n_jobs=n_jobs)
            cands[_fmt({**c, "C": 0}).rsplit(" ", 1)[0]] = (c, feature_matrix(f))
    rows = []
    with timer(f"[{name}] nested CV for CCST"):
        ours = nested_cv(cands, y, groups, cfg.seed)
    rows.append(_summary(PROPOSED, y, ours, variants, defaults.get(PROPOSED), cfg.seed))
    for b in BASELINES:
        if b not in state["mats"]:
            continue
        out = nested_cv({"default": ({}, state["mats"][b])}, y, groups, cfg.seed)
        row = _summary(b, y, out, variants, defaults.get(b), cfg.seed)
        # paired test between the two *tuned* systems (one-sided: CCST better)
        row["p_tuned_ccst_better"] = paired_bootstrap_pvalue(list(y), ours["pred"], out["pred"], _macro,
                                                             n_boot=1000, seed=cfg.seed)
        rows.append(row)
    for r in rows:
        log.info(f"[{name}] {r['method']}: default {r['default_macro_f1']:.3f} -> nested {r['nested_macro_f1']:.3f}")
    return rows


def run(cfg: Config, n_jobs: int = 4) -> dict:
    out = ensure_dir(RESULTS_DIR / "tuning")
    results = {}
    for realism in (False, True):
        try:
            results["v2" if realism else "v1"] = run_one(cfg, realism, n_jobs)
        except FileNotFoundError:
            log.warning(f"no state for benchmark {'v2' if realism else 'v1'}; run `synthetic{' --realism' if realism else ''}` first")
    save_json(results, out / "metrics.json")
    methods = [r["method"] for r in results.get("v1", results.get("v2", []))]
    series = {}
    for v in results:
        by = {r["method"]: r for r in results[v]}
        series[f"{v} default"] = [by[m]["default_macro_f1"] for m in methods]
        series[f"{v} nested-tuned"] = [by[m]["nested_macro_f1"] for m in methods]
    plots.grouped_bars(methods, series, "macro-F1", "Default vs nested-CV-tuned configurations",
                       out / "tuning.png", ylim=(0, 1.05))
    md = ["# Item 2 — hyper-parameter selection by nested cross-validation", "",
          "Protocol: `docs/tuning_protocol.md` (written before running). Selection uses only the outer training "
          "portion; each outer test fold is scored once.", ""]
    for v, rows in results.items():
        md += [f"## Benchmark {v}", "", md_table(rows, ["method", "default_macro_f1", "nested_macro_f1", "ci_lo",
                                                         "ci_hi", "gain", "FCR_context_shift", "contra_recall",
                                                         "p_tuned_ccst_better"]),
               "", "Configuration chosen in each outer fold:", ""]
        md += [f"- **{r['method']}**: {r['chosen']}" for r in rows] + [""]
    (out / "REPORT.md").write_text("\n".join(md))
    return results
