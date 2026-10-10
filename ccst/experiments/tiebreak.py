"""Item 3: sign-aware tie-breaking in the alignment (docs/tiebreak_protocol.md).

For each benchmark, CCST features are recomputed with ε = lam_s_align ∈ {0, 0.02, 0.05, 0.1} and scored with the
headline protocol (identical LR, grouped 5-fold CV by template). ε = 0 must reproduce the published numbers.
ε = 0.05 was fixed a priori; the other values are an information-only sensitivity sweep. The pre-registered
decision rule is applied to ε = 0.05 vs ε = 0 on both benchmarks.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import Config
from ..eval.metrics import bootstrap_metric_ci, classification_report, false_contradiction_rate
from ..relation.guards import apply_guards
from ..utils import RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from .common import cross_validate, feature_matrix, featurize
from .synthetic import bench_paths
from .synthetic_analysis import load_state
from .tuning import _macro

EPS = (0.0, 0.02, 0.05, 0.1)
CHOSEN = 0.05


def _metrics(y, pred, proba, variants) -> dict:
    rep = classification_report(list(y), pred, proba)
    shift = np.isin(variants, ["hetero_ctx", "hetero_sim"])
    return {"macro_f1": rep["macro_f1"], "P_analogy": rep["p_analogy"], "R_analogy": rep["r_analogy"],
            "R_contradiction": rep["r_contradiction"], "R_heterogeneity": rep["r_heterogeneity"],
            "FCR_context_shift": false_contradiction_rate(list(y), pred, shift)}


def run_one(cfg: Config, realism: bool, n_jobs: int) -> list[dict]:
    name = bench_paths(realism)[0]
    S = load_state(realism)["S"]
    pairs = S["pairs"]
    y = np.array([p["label"] for p in pairs])
    groups = np.array([p["template"] for p in pairs])
    variants = np.array([p["variant"] for p in pairs])
    P = [(p["a"], p["b"]) for p in pairs]
    rows = []
    for eps in EPS:
        with timer(f"[{name}] eps={eps}", quiet=True):
            f, _ = featurize(P, S["graphs"], S["metas"], replace(cfg.ccst, lam_s_align=eps), cfg.context, n_jobs=n_jobs)
            cv = cross_validate(feature_matrix(f), list(y), groups, seed=cfg.seed)
        guarded = [apply_guards(p, fi) for p, fi in zip(cv["pred"], f)]
        m = _metrics(y, cv["pred"], cv["proba"], variants)
        g = _metrics(y, guarded, None, variants)
        lo, hi = bootstrap_metric_ci(list(y), cv["pred"], _macro, n_boot=500, seed=cfg.seed)
        rows.append({"eps": eps, **m, "ci_lo": lo, "ci_hi": hi,
                     **{f"guarded_{k}": v for k, v in g.items()},
                     "spurious_gap_analogy": float(np.mean([fi["gap"] for fi, p in zip(f, pairs) if p["label"] == "analogy"]))})
        log.info(f"[{name}] eps={eps}: macro-F1 {m['macro_f1']:.3f}, guarded analogy R {g['R_analogy']:.3f}")
    return rows


def decide(rows_by_bench: dict) -> dict:
    """Pre-registered rule: adopt ε = 0.05 only if, on both benchmarks, guarded analogy recall improves,
    contradiction recall drops ≤ 0.02, FCR rises ≤ 0.02 and macro-F1 drops ≤ 0.01."""
    checks = {}
    for b, rows in rows_by_bench.items():
        base = next(r for r in rows if r["eps"] == 0.0)
        new = next(r for r in rows if r["eps"] == CHOSEN)
        checks[b] = {
            "guarded_analogy_recall_improves": new["guarded_R_analogy"] > base["guarded_R_analogy"],
            "contradiction_recall_ok": new["R_contradiction"] >= base["R_contradiction"] - 0.02,
            "fcr_ok": new["FCR_context_shift"] <= base["FCR_context_shift"] + 0.02,
            "macro_f1_ok": new["macro_f1"] >= base["macro_f1"] - 0.01,
        }
    adopt = all(all(c.values()) for c in checks.values())
    return {"checks": checks, "adopt": adopt}


def run(cfg: Config, n_jobs: int = 4) -> dict:
    out = ensure_dir(RESULTS_DIR / "tiebreak")
    rows = {}
    for realism in (False, True):
        rows["v2" if realism else "v1"] = run_one(cfg, realism, n_jobs)
    decision = decide(rows)
    save_json({"rows": rows, "decision": decision}, out / "metrics.json")
    cols = ["eps", "macro_f1", "ci_lo", "ci_hi", "P_analogy", "R_analogy", "guarded_R_analogy", "guarded_P_analogy",
            "guarded_macro_f1", "R_contradiction", "FCR_context_shift", "guarded_R_heterogeneity", "spurious_gap_analogy"]
    md = ["# Item 3 — sign-aware tie-breaking in the alignment", "",
          "Protocol and decision rule: `docs/tiebreak_protocol.md` (written before running). ε = 0.05 fixed a "
          "priori; 0.02 and 0.1 are information only. ε = 0 reproduces the published numbers.", ""]
    for b, r in rows.items():
        md += [f"## Benchmark {b}", "", md_table(r, cols), ""]
    md += ["## Pre-registered decision", "", md_table([{"benchmark": b, **c} for b, c in decision["checks"].items()]), "",
           f"**Adopt ε = 0.05 as default: {decision['adopt']}**"]
    (out / "REPORT.md").write_text("\n".join(md))
    return {"rows": rows, "decision": decision}
