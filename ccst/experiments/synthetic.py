"""Synthetic controlled-transformation benchmark: CCST vs. baselines vs. ablations (RQ1-RQ3).

Protocol (identical for every method): pair -> feature vector (+ shared S_domain) -> multinomial LR,
grouped 5-fold CV by mechanism template (no template appears in train and test), with 25% of the
training templates held out for temperature scaling and the abstention threshold.
"""
from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score

from ..config import CONTEXT_DIMS, RELATION_LABELS, Config, ContextConfig
from ..embed import get_embedder
from ..eval import baselines_text as bt
from ..eval.metrics import (
    bootstrap_metric_ci, classification_report, false_contradiction_rate, mcnemar_pvalue, paired_bootstrap_pvalue,
    risk_coverage,
)
from ..extraction.graph_builder import GraphPipeline
from ..relation.features import FEATURE_NAMES
from ..synthetic.generator import build_benchmark
from ..topics.lda_gibbs import GibbsLDA
from ..topics.preprocess import build_dictionary, tokenize_corpus
from ..transport import baselines as gb
from ..utils import DATA_DIR, RESULTS_DIR, ensure_dir, log, md_table, save_json, timer, write_jsonl
from ..viz import plots
from .common import cross_validate, feature_matrix, featurize

CTX_FEATURES = {"S_ctx", "ctx_min_material", "ctx_known_frac", "F_hetero", "F_hetero_mat", "F_hetero_frac",
                "F_unknown_frac", "gate", "F_flip", "F_flip_mat"}
NO_CTX_FEATURES = [f for f in FEATURE_NAMES if f not in CTX_FEATURES] + ["S_domain"]
PROPOSED = "CCST (ours)"


def macro_f1(y, p):
    return f1_score(y, p, labels=list(RELATION_LABELS), average="macro", zero_division=0)


# ----------------------------------------------------------------------------- data + graphs
def bench_paths(realism: bool) -> tuple[str, str]:
    """(results/data folder name, state file name): v1 keeps its original paths."""
    return ("synthetic_v2", "synthetic_v2_state.pkl") if realism else ("synthetic", "synthetic_state.pkl")


def extraction_quality(S: dict) -> dict:
    """Extraction accuracy against the generator's gold mechanisms (reported jointly with results)."""
    from collections import Counter

    from ..graph import REL_TO_TYPE_SIGN
    from ..synthetic.generator import slot_of

    tmpl = {t.tid: t for t in S["templates"]}
    tp = fn = sign_ok = 0
    hits, tot = Counter(), Counter()
    for d in S["docs"]:
        t = tmpl[d["template"]]
        found = {}
        for c in S["claims"][d["doc_id"]]:
            if c.epistemic == "quoted":
                continue
            si, oi = slot_of(c.subj, d), slot_of(c.obj, d)
            if si is not None and oi is not None:
                found[(si, oi)] = 0 if c.null_effect else c.polarity
        for k, (i, j, _) in enumerate(t.edges):
            r = d["rels"][k]
            pol = 0 if r.startswith("null") else REL_TO_TYPE_SIGN[r][1]
            if (i, j) in found:
                tp += 1
                sign_ok += int(found[(i, j)] == pol)
            else:
                fn += 1
        if d["context"]:
            stated = {"population", "design"} | ({"dose", "time", "env"} - set(d["context"].get("omit", [])))
            cs = [c for c in S["claims"][d["doc_id"]] if c.epistemic != "quoted"]
            ctx = cs[0].context if cs else {}
            for h in stated:
                key = {"env": "environment"}.get(h, h)
                tot[key] += 1
                hits[key] += int(ctx.get(key) is not None)
    return {"edge_recall": tp / max(tp + fn, 1), "sign_accuracy": sign_ok / max(tp, 1), "gold_edges": tp + fn,
            **{f"ctx_detect_{h}": hits[h] / tot[h] for h in sorted(tot)}}


def prepare(cfg: Config, n_templates: int, out: str, realism: bool = False) -> dict:
    docs, pairs, templates = build_benchmark(n_templates, seed=cfg.seed, realism=realism)
    write_jsonl(docs, f"{out}/docs.jsonl")
    write_jsonl(pairs, f"{out}/pairs.jsonl")
    log.info(f"benchmark: {len(docs)} docs, {len(pairs)} pairs, {len(templates)} templates")
    texts = {d["doc_id"]: d["text"] for d in docs}
    with timer("LDA theta for S_domain"):
        toks = tokenize_corpus(list(texts.values()))
        dic, ids = build_dictionary(toks, min_df=3, max_df=0.6)
        lda = GibbsLDA(20, len(dic), iters=200, seed=cfg.seed).fit(ids)
    metas = {d["doc_id"]: {"category": d["category"], "theta": lda.theta[i]} for i, d in enumerate(docs)}
    gp = GraphPipeline(cfg.extraction)
    with timer("claim extraction"):
        claims = gp.extract(list(texts.items()))
    with timer("graph construction"):
        graphs = gp.build(claims, metas)
    return {"docs": docs, "pairs": pairs, "templates": templates, "texts": texts, "metas": metas, "claims": claims,
            "graphs": graphs, "pipeline": gp, "theta": {d: metas[d]["theta"] for d in metas}}


def rebuild_graphs(S: dict, claim_fn=None, **pipe_kw) -> dict:
    """Re-run concept generalisation + graph construction (optionally with modified claims or
    with generalisation switched off) for ablations."""
    from copy import deepcopy

    cfg = S["pipeline"].cfg
    if pipe_kw.get("generalize_relations") is False:
        gp = GraphPipeline(cfg, generalize_relations=False)
        claims = gp.extract(list(S["texts"].items()))
    else:
        gp = GraphPipeline(cfg, **pipe_kw)
        claims = {d: [claim_fn(deepcopy(c)) if claim_fn else deepcopy(c) for c in cl] for d, cl in S["claims"].items()}
    return gp.build(claims, S["metas"])


# ----------------------------------------------------------------------------- methods
def baseline_features(S: dict, ccst_feats: list[dict], use_specter: bool, use_nli: bool, n_jobs: int,
                      specter_model: str) -> dict[str, np.ndarray]:
    pairs, graphs, texts = S["pairs"], S["graphs"], S["texts"]
    P = [(p["a"], p["b"]) for p in pairs]
    sdom = np.array([[f["S_domain"]] for f in ccst_feats])
    emb = get_embedder()
    ids = list(texts)
    F: dict[str, np.ndarray] = {}
    with timer("BM25"):
        bm = bt.BM25Pairs(texts)
        F["BM25"] = np.array([bm.features(a, b) for a, b in P])
    with timer("SBERT"):
        E = dict(zip(ids, emb.encode_corpus([texts[i] for i in ids], "synthetic")))
        F["SBERT (MiniLM)"] = np.array([bt.cosine_features(E, a, b) for a, b in P])
    if use_specter:
        with timer("SPECTER"):
            sp = get_embedder(specter_model)
            Es = dict(zip(ids, sp.encode_corpus([texts[i] for i in ids], "synthetic")))
            F["SPECTER"] = np.array([bt.cosine_features(Es, a, b) for a, b in P])
    F["LDA topic similarity"] = np.array([bt.lda_features(S["theta"], a, b) for a, b in P])
    F["Purpose/mechanism emb."] = np.array([bt.purpose_mechanism_features(graphs[a], graphs[b], emb) for a, b in P])
    F["Raw SVO overlap"] = np.array([bt.raw_svo_features(S["claims"][a], S["claims"][b]) for a, b in P])
    if use_nli:
        with timer("NLI"):
            nli = bt.NLIScorer()
            F["NLI (BART-MNLI)"] = nli.pair_features([texts[a] for a, _ in P], [texts[b] for _, b in P])
    with timer("standard FGW"):
        F["FGW (POT)"] = np.array([gb.standard_fgw(graphs[a], graphs[b]) for a, b in P])
    with timer("graph edit distance"):
        F["Graph edit distance"] = np.array(Parallel(n_jobs=n_jobs)(
            delayed(gb.graph_edit_distance)(graphs[a], graphs[b]) for a, b in P))
    F["Role-constrained matching"] = np.array([gb.role_constrained_matching(graphs[a], graphs[b]) for a, b in P])
    with timer("LinearFGW"):
        sizes = sorted((g.n, d) for d, g in graphs.items() if g.n > 0)
        ref = graphs[sizes[len(sizes) // 2][1]]
        lf = gb.LinearFGW(ref)
        emb_lf = {d: lf.embed(g) for d, g in graphs.items()}
        F["LinearFGW"] = np.array([[float(np.linalg.norm(emb_lf[a] - emb_lf[b]))] for a, b in P])
    ccfg = ContextConfig()
    F["Context clf. w/o transport"] = np.array([gb.context_without_transport(graphs[a], graphs[b], emb, ccfg)
                                                for a, b in P])
    return {k: np.hstack([v, sdom]) for k, v in F.items()}


def ccst_family(S: dict, cfg: Config, n_jobs: int) -> tuple[dict, dict]:
    """CCST, its graph-transport relatives, and the one-component-at-a-time ablations."""
    P = [(p["a"], p["b"]) for p in S["pairs"]]
    graphs, metas = S["graphs"], S["metas"]
    base = cfg.ccst
    feats: dict[str, list] = {}
    mats: dict[str, np.ndarray] = {}

    def run(name, ccfg=None, graphs_=None, **kw):
        with timer(f"CCST variant: {name}", quiet=True):
            f, _ = featurize(P, graphs_ or graphs, metas, replace(base, **kw), ccfg or cfg.context, n_jobs=n_jobs)
        feats[name] = f
        return f

    full = run(PROPOSED)
    mats[PROPOSED] = feature_matrix(full)
    fn = run("FNGW (full, edge features)", partial=False, use_context=False)
    mats["FNGW (full, edge features)"] = feature_matrix(fn, ["D_unsigned", "D_signed", "gap", "S_domain"])
    sm = run("Signed matcher w/o context", use_context=False)
    mats["Signed matcher w/o context"] = feature_matrix(sm, NO_CTX_FEATURES)
    # ablations
    abl = {
        "− partial transport (forced full)": dict(partial=False),
        "− edge types": dict(use_edge_types=False),
        "− edge direction": dict(use_direction=False),
        "− edge sign": dict(use_sign=False),
        "− context (all dims)": dict(use_context=False),
        "− context gate": dict(lam_g=0.0),
    }
    for name, kw in abl.items():
        mats[name] = feature_matrix(run(name, **kw))
    for h in CONTEXT_DIMS:
        w = {**cfg.context.weights, h: 0.0}
        cc = replace(cfg.context, weights=w, material=tuple(x for x in cfg.context.material if x != h))
        mats[f"− context: {h}"] = feature_matrix(run(f"− context: {h}", ccfg=cc))
    with timer("ablation graphs"):
        g_noepi = rebuild_graphs(S, claim_fn=lambda c: replace(c, epistemic="observed"))
        g_noconf = rebuild_graphs(S, claim_fn=lambda c: replace(c, confidence=1.0, epistemic="observed"))
        g_rawc = rebuild_graphs(S, generalize_concepts=False)
        g_rawr = rebuild_graphs(S, generalize_relations=False)
    mats["− epistemic status"] = feature_matrix(run("− epistemic status", graphs_=g_noepi))
    mats["− confidence / provenance weights"] = feature_matrix(run("− confidence / provenance weights", graphs_=g_noconf))
    mats["− concept generalisation (raw nodes)"] = feature_matrix(run("− concept generalisation", graphs_=g_rawc))
    mats["− relation generalisation (raw verbs)"] = feature_matrix(run("− relation generalisation", graphs_=g_rawr))
    S["ablation_graphs"] = {"raw_concepts": g_rawc, "raw_relations": g_rawr}
    return mats, feats


# ----------------------------------------------------------------------------- evaluation
def evaluate(name: str, X: np.ndarray | None, S: dict, seed: int, rule_feats: list[dict] | None = None,
             pred_override: list[str] | None = None) -> tuple[dict, dict]:
    pairs = S["pairs"]
    y = [p["label"] for p in pairs]
    groups = np.array([p["template"] for p in pairs])
    var = np.array([p["variant"] for p in pairs])
    if pred_override is not None:
        cv = {"pred": pred_override, "proba": None, "pred_abstain": pred_override}
    elif rule_feats is not None:
        cv = cross_validate(np.zeros((len(y), 1)), y, groups, seed=seed, feats=rule_feats, rule_mode=True)
        cv["proba"] = None
    else:
        cv = cross_validate(X, y, groups, seed=seed)
    rep = classification_report(y, cv["pred"], cv["proba"])
    shift = np.isin(var, ["hetero_ctx", "hetero_sim"])
    rep["FCR_context_shift"] = false_contradiction_rate(y, cv["pred"], shift)
    rep["FCR_missing_context"] = false_contradiction_rate(y, cv["pred"], var == "incomp_missing")
    rep["FCR_background_quote"] = false_contradiction_rate(y, cv["pred"], var == "agree_background")
    rep["contra_recall"] = rep["r_contradiction"]
    lo, hi = bootstrap_metric_ci(y, cv["pred"], macro_f1, n_boot=500, seed=seed)
    rep["macro_f1_lo"], rep["macro_f1_hi"] = lo, hi
    if cv["proba"] is not None:
        yi = np.array([RELATION_LABELS.index(v) for v in y])
        _, _, rep["AURC"] = risk_coverage(cv["proba"], yi)
        acc = np.array([p != "abstain" for p in cv["pred_abstain"]])
        rep["coverage@risk"] = float(acc.mean())
        rep["selective_risk"] = float(np.mean(np.array(cv["pred_abstain"])[acc] != np.array(y)[acc])) if acc.any() else 0.0
    rep["method"] = name
    return rep, cv


def run(cfg: Config, quick: bool = False, n_jobs: int = 4, use_specter: bool = True, use_nli: bool = True,
        llm_pairs: int = 0, realism: bool = False) -> dict:
    name, state_file = bench_paths(realism)
    out = ensure_dir(RESULTS_DIR / name)
    data_out = ensure_dir(DATA_DIR / name)
    S = prepare(cfg, 12 if quick else 50, str(data_out), realism=realism)
    times: dict = {}
    S["extraction_quality"] = extraction_quality(S)
    log.info(f"extraction quality vs gold: {S['extraction_quality']}")
    with timer("CCST family + ablations", times):
        mats, feats = ccst_family(S, cfg, n_jobs)
    with timer("baselines", times):
        base = baseline_features(S, feats[PROPOSED], use_specter, use_nli, n_jobs, cfg.topics.specter_model)
    mats.update(base)
    nli_key = "NLI (BART-MNLI)"
    if nli_key in base:
        mats["CCST + NLI (hybrid)"] = np.hstack([mats[PROPOSED], base[nli_key][:, :-1]])
    reports, cvs = {}, {}
    main_methods = [PROPOSED, "CCST + NLI (hybrid)", *base.keys(), "FNGW (full, edge features)",
                    "Signed matcher w/o context"]
    for name in main_methods:
        if name in mats:
            reports[name], cvs[name] = evaluate(name, mats[name], S, cfg.seed)
    reports["CCST (ordered rules)"], cvs["CCST (ordered rules)"] = evaluate("CCST (ordered rules)", None, S, cfg.seed,
                                                                             rule_feats=feats[PROPOSED])
    if llm_pairs:
        reports["Direct LLM judge"], cvs["Direct LLM judge"] = llm_baseline(S, llm_pairs, cfg.seed)
    abl_reports = {}
    for name in mats:
        if name.startswith("−"):
            abl_reports[name], cvs[name] = evaluate(name, mats[name], S, cfg.seed)
    abl_reports[PROPOSED] = reports[PROPOSED]
    # significance vs. the proposed method
    y = [p["label"] for p in S["pairs"]]
    ours = cvs[PROPOSED]["pred"]
    for name, rep in list(reports.items()) + list(abl_reports.items()):
        if name == PROPOSED:
            continue
        rep["p_vs_ccst"] = paired_bootstrap_pvalue(y, ours, cvs[name]["pred"], macro_f1, n_boot=500, seed=cfg.seed)
        ca = np.array(ours) == np.array(y)
        cb = np.array(cvs[name]["pred"]) == np.array(y)
        rep["mcnemar_p"] = mcnemar_pvalue(ca, cb)
    # sensitivity analyses
    sens = sensitivity(S, cfg, n_jobs)
    noise = noise_robustness(S, cfg, n_jobs, base)
    state = {"S": {k: v for k, v in S.items() if k not in ("pipeline",)}, "mats": mats, "feats": feats, "cvs": cvs}
    with open(ensure_dir(DATA_DIR / "cache") / state_file, "wb") as f:
        pickle.dump(state, f)
    if not realism:
        (DATA_DIR / "cache" / "ccst_relation_model.pkl").unlink(missing_ok=True)  # retrain the zero-shot model
    report_synthetic(reports, abl_reports, cvs, S, sens, noise, out, times)
    return {"reports": reports, "ablations": abl_reports}


def llm_baseline(S: dict, n: int, seed: int):
    """Direct LLM pair judgement on a stratified subsample (no training; zero-shot)."""
    rng = np.random.default_rng(seed)
    pairs = S["pairs"]
    by = {}
    for i, p in enumerate(pairs):
        by.setdefault(p["variant"], []).append(i)
    idx = sorted(i for v in by.values() for i in rng.choice(v, size=min(len(v), max(1, n // len(by))), replace=False))
    judge = bt.LLMJudge()
    pred = ["incomparable"] * len(pairs)
    for i in idx:
        pred[i] = judge.judge(S["texts"][pairs[i]["a"]], S["texts"][pairs[i]["b"]])
    sub = {"pairs": [pairs[i] for i in idx]}
    rep, cv = evaluate("Direct LLM judge", None, {**S, **sub}, seed, pred_override=[pred[i] for i in idx])
    rep["n_eval"] = len(idx)
    cv_full = {"pred": pred, "subset": idx}
    return rep, cv_full


def sensitivity(S: dict, cfg: Config, n_jobs: int) -> dict:
    """Context-weight sensitivity (Sec. 4.2) and lambda_u / alpha grid (Sec. 4.3)."""
    P = [(p["a"], p["b"]) for p in S["pairs"]]
    out = {"weights": [], "grid": []}
    for h in CONTEXT_DIMS:
        for scale in (0.5, 2.0):
            cc = replace(cfg.context, weights={**cfg.context.weights, h: cfg.context.weights[h] * scale})
            f, _ = featurize(P, S["graphs"], S["metas"], cfg.ccst, cc, n_jobs=n_jobs)
            rep, _ = evaluate("w", feature_matrix(f), S, cfg.seed)
            out["weights"].append({"dim": h, "scale": scale, "macro_f1": rep["macro_f1"],
                                   "FCR_context_shift": rep["FCR_context_shift"]})
    for lam_u in (0.3, 0.6, 1.0):
        for alpha in (0.5, 1.0, 2.0):
            f, _ = featurize(P, S["graphs"], S["metas"], replace(cfg.ccst, lam_u=lam_u, alpha=alpha), cfg.context,
                             n_jobs=n_jobs)
            rep, _ = evaluate("g", feature_matrix(f), S, cfg.seed)
            out["grid"].append({"lam_u": lam_u, "alpha": alpha, "macro_f1": rep["macro_f1"]})
    return out


def noise_robustness(S: dict, cfg: Config, n_jobs: int, base: dict) -> dict:
    """Extraction errors propagate (Sec. 9): corrupt claims (drop / sign error / lost context) at rate p."""
    from copy import deepcopy

    P = [(p["a"], p["b"]) for p in S["pairs"]]
    rng = np.random.default_rng(cfg.seed)
    res = {"p": [], PROPOSED: [], "Signed matcher w/o context": [], "FNGW (full, edge features)": [],
           "NLI (BART-MNLI)": []}
    for p in (0.0, 0.1, 0.2, 0.3):
        noisy = {}
        for d, cl in S["claims"].items():
            new = []
            for c in cl:
                if rng.random() < p:
                    continue  # missed claim
                c = deepcopy(c)
                if rng.random() < p / 3 and c.polarity != 0:
                    c.polarity = -c.polarity  # sign error
                    c.relation = {"increases": "decreases", "decreases": "increases", "enables": "inhibits",
                                  "inhibits": "enables"}.get(c.relation, c.relation)
                if rng.random() < p:
                    c.context = {h: (None if rng.random() < 0.5 else v) for h, v in c.context.items()}
                new.append(c)
            noisy[d] = new
        gp = GraphPipeline(S["pipeline"].cfg)
        G = gp.build(noisy, S["metas"])
        res["p"].append(p)
        f, _ = featurize(P, G, S["metas"], cfg.ccst, cfg.context, n_jobs=n_jobs)
        res[PROPOSED].append(evaluate("n", feature_matrix(f), S, cfg.seed)[0])
        f2, _ = featurize(P, G, S["metas"], replace(cfg.ccst, use_context=False), cfg.context, n_jobs=n_jobs)
        res["Signed matcher w/o context"].append(evaluate("n", feature_matrix(f2, NO_CTX_FEATURES), S, cfg.seed)[0])
        f3, _ = featurize(P, G, S["metas"], replace(cfg.ccst, use_context=False, partial=False), cfg.context, n_jobs=n_jobs)
        res["FNGW (full, edge features)"].append(
            evaluate("n", feature_matrix(f3, ["D_unsigned", "D_signed", "gap", "S_domain"]), S, cfg.seed)[0])
        if "NLI (BART-MNLI)" in base:  # text-only: unaffected by extraction noise (reference line)
            res["NLI (BART-MNLI)"].append(evaluate("n", base["NLI (BART-MNLI)"], S, cfg.seed)[0])
    return res


# ----------------------------------------------------------------------------- reporting
def _row(r: dict) -> dict:
    keys = ["method", "macro_f1", "macro_f1_lo", "macro_f1_hi", "accuracy", "f1_agreement", "f1_analogy",
            "f1_contradiction", "f1_heterogeneity", "f1_incomparable", "p_contradiction", "contra_recall",
            "FCR_context_shift", "FCR_missing_context", "FCR_background_quote", "ece", "brier", "auprc_contradiction",
            "AURC", "coverage@risk", "selective_risk", "p_vs_ccst", "mcnemar_p", "n_eval"]
    return {k: r[k] for k in keys if k in r}


def report_synthetic(reports, abl, cvs, S, sens, noise, out, times) -> None:
    y = [p["label"] for p in S["pairs"]]
    rows = sorted((_row(r) for r in reports.values()), key=lambda r: -r["macro_f1"])
    arows = sorted((_row(r) for r in abl.values()), key=lambda r: -r["macro_f1"])
    save_json({"main": rows, "ablations": arows, "sensitivity": sens, "timings_s": times,
               "extraction_quality": S.get("extraction_quality", {}),
               "noise": {k: ([x["macro_f1"] for x in v] if k != "p" else v) for k, v in noise.items() if v}},
              out / "metrics.json")
    write_jsonl([{**p, **{m: cvs[m]["pred"][i] for m in cvs if len(cvs[m]["pred"]) == len(y)}}
                 for i, p in enumerate(S["pairs"])], out / "predictions.jsonl")
    names = [r["method"] for r in rows]
    plots.bar_ci(names, [r["macro_f1"] for r in rows], [r["macro_f1_lo"] for r in rows],
                 [r["macro_f1_hi"] for r in rows], highlight={PROPOSED, "CCST + NLI (hybrid)", "CCST (ordered rules)"},
                 xlabel="macro-F1 over 5 relations (95% bootstrap CI)",
                 title="Relation classification on the controlled benchmark", path=out / "macro_f1.png", xlim=(0, 1.08))
    plots.bar_ci(names, [r["FCR_context_shift"] for r in rows], highlight={PROPOSED, "CCST + NLI (hybrid)",
                                                                             "CCST (ordered rules)"},
                 xlabel="share of context-shift pairs predicted as contradiction (lower is better)",
                 title="Critical safety metric: false-contradiction rate", path=out / "false_contradiction_rate.png",
                 xlim=(0, 1.08))
    plots.bar_ci(names, [r["contra_recall"] for r in rows], highlight={PROPOSED, "CCST + NLI (hybrid)",
                                                                         "CCST (ordered rules)"},
                 xlabel="recall of genuine contradictions", title="Genuine conflicts are not hidden",
                 path=out / "contradiction_recall.png", xlim=(0, 1.08))
    an = [r["method"] for r in arows]
    plots.bar_ci(an, [r["macro_f1"] for r in arows], [r["macro_f1_lo"] for r in arows], [r["macro_f1_hi"] for r in arows],
                 highlight={PROPOSED}, xlabel="macro-F1", title="Ablations: remove one component at a time",
                 path=out / "ablations_macro_f1.png", xlim=(0, 1.08))
    plots.bar_ci(an, [r["FCR_context_shift"] for r in arows], highlight={PROPOSED},
                 xlabel="false-contradiction rate on context-shift pairs", title="Ablations: safety metric",
                 path=out / "ablations_fcr.png", xlim=(0, 1.08))
    best_base = next(r["method"] for r in rows if not r["method"].startswith("CCST"))
    for m in (PROPOSED, best_base):
        cm = confusion_matrix(y, cvs[m]["pred"], labels=list(RELATION_LABELS))
        plots.confusion(cm, list(RELATION_LABELS), f"Confusion: {m}", out / f"confusion_{_slug(m)}.png")
    yi = np.array([RELATION_LABELS.index(v) for v in y])
    plots.reliability(cvs[PROPOSED]["proba"], yi, "Calibration of CCST (temperature-scaled)", out / "reliability.png")
    curves = {}
    for m in [PROPOSED, best_base, "Signed matcher w/o context", "SBERT (MiniLM)"]:
        if m in cvs and cvs[m].get("proba") is not None:
            cov, risk, _ = risk_coverage(cvs[m]["proba"], yi)
            curves[m] = (cov, risk)
    plots.lines({k: v[0] for k, v in curves.items()}, {k: v[1] for k, v in curves.items()}, "coverage",
                "selective risk (error among accepted)", "Risk-coverage: abstaining on uncertain pairs",
                out / "risk_coverage.png", markers=False)
    grid = sens["grid"]
    lus, als = sorted({g["lam_u"] for g in grid}), sorted({g["alpha"] for g in grid})
    M = np.array([[next(g["macro_f1"] for g in grid if g["lam_u"] == lu and g["alpha"] == al) for al in als] for lu in lus])
    plots.heatmap(M, [f"λ_u={x}" for x in lus], [f"α={x}" for x in als], "Hyper-parameter sensitivity (macro-F1)",
                  out / "sensitivity_grid.png", fmt="{:.3f}", vmin=max(0, M.min() - 0.05), vmax=1)
    if noise.get("p"):
        ys = {k: [x["macro_f1"] for x in v] for k, v in noise.items() if k != "p" and v}
        plots.lines(noise["p"], ys, "corruption rate p (claim drop / sign error / lost context)", "macro-F1",
                    "Robustness to extraction errors", out / "noise_robustness.png", ylim=(0, 1.02))
        ysf = {k: [x["FCR_context_shift"] for x in v] for k, v in noise.items() if k != "p" and v}
        plots.lines(noise["p"], ysf, "corruption rate p", "false-contradiction rate (context shift)",
                    "Safety metric under extraction errors", out / "noise_fcr.png", ylim=(-0.02, 1.02))
    md = [f"# Synthetic controlled-transformation benchmark ({out.name})", "",
          f"{len(S['docs'])} documents, {len(S['pairs'])} pairs, grouped 5-fold CV by template.", "",
          "## Extraction quality vs gold (reported jointly with downstream results)", "",
          md_table([S.get("extraction_quality", {})]), "",
          "## Main comparison", "", md_table(rows, ["method", "macro_f1", "macro_f1_lo", "macro_f1_hi", "accuracy",
                                                     "f1_contradiction", "f1_heterogeneity", "f1_analogy",
                                                     "FCR_context_shift", "FCR_missing_context", "contra_recall", "ece",
                                                     "AURC", "p_vs_ccst"]),
          "", "## Ablations", "", md_table(arows, ["method", "macro_f1", "macro_f1_lo", "macro_f1_hi",
                                                    "FCR_context_shift", "f1_contradiction", "f1_heterogeneity",
                                                    "f1_analogy", "p_vs_ccst"]),
          "", "## Context-weight sensitivity", "", md_table(sens["weights"]),
          "", "## Timings (s)", "", md_table([{"stage": k, "seconds": v} for k, v in times.items()])]
    (out / "REPORT.md").write_text("\n".join(md))
    log.info(f"synthetic results written to {out}")


def _slug(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def replot(out=None) -> None:
    """Redraw the comparison bar charts from results/synthetic/metrics.json (no recomputation)."""
    import json

    out = out or RESULTS_DIR / "synthetic"
    m = json.loads((out / "metrics.json").read_text())
    rows, arows = m["main"], m["ablations"]
    hl = {PROPOSED, "CCST + NLI (hybrid)", "CCST (ordered rules)"}
    names = [r["method"] for r in rows]
    plots.bar_ci(names, [r["macro_f1"] for r in rows], [r["macro_f1_lo"] for r in rows],
                 [r["macro_f1_hi"] for r in rows], highlight=hl, xlabel="macro-F1 over 5 relations (95% bootstrap CI)",
                 title="Relation classification on the controlled benchmark", path=out / "macro_f1.png", xlim=(0, 1.08))
    plots.bar_ci(names, [r["FCR_context_shift"] for r in rows], highlight=hl,
                 xlabel="share of context-shift pairs predicted as contradiction (lower is better)",
                 title="Critical safety metric: false-contradiction rate", path=out / "false_contradiction_rate.png",
                 xlim=(0, 1.08))
    plots.bar_ci(names, [r["contra_recall"] for r in rows], highlight=hl, xlabel="recall of genuine contradictions",
                 title="Genuine conflicts are not hidden", path=out / "contradiction_recall.png", xlim=(0, 1.08))
    an = [r["method"] for r in arows]
    plots.bar_ci(an, [r["macro_f1"] for r in arows], [r["macro_f1_lo"] for r in arows], [r["macro_f1_hi"] for r in arows],
                 highlight={PROPOSED}, xlabel="macro-F1", title="Ablations: remove one component at a time",
                 path=out / "ablations_macro_f1.png", xlim=(0, 1.08))
    plots.bar_ci(an, [r["FCR_context_shift"] for r in arows], highlight={PROPOSED},
                 xlabel="false-contradiction rate on context-shift pairs", title="Ablations: safety metric",
                 path=out / "ablations_fcr.png", xlim=(0, 1.08))
