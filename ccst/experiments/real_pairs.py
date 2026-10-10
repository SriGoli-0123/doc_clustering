"""Expert-labelled real data (proposal Sec. 7.2 stratum 3): SciFact and ManConCorpus.

* SciFact: claim vs. abstract -> {agreement (SUPPORT), contradiction (CONTRADICT), incomparable (NEI)}
* ManConCorpus: two conclusion claims answering the same clinical question -> {conflict, agreement};
  documents are the full PubMed abstracts so that scope conditions are visible to CCST.
Same protocol as the synthetic benchmark: every method -> features -> identical LR; grouped 5-fold CV
(by claim for SciFact, by clinical question for ManCon).

Zero-shot analysis: the 5-way CCST model trained on the synthetic benchmark labels each ManCon
"conflict" pair as contradiction vs. contextual heterogeneity, quantifying how often a material scope
difference explains the apparent contradiction (cf. Rosemblat et al., 2019), with certificates.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import replace

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from ..certificate import build_certificate
from ..config import RELATION_LABELS, Config
from ..data import claims_datasets as cd
from ..embed import get_embedder
from ..eval import baselines_text as bt
from ..eval.metrics import bootstrap_metric_ci, paired_bootstrap_pvalue
from ..extraction.graph_builder import GraphPipeline
from ..relation.classifier import RelationClassifier
from ..transport import baselines as gb
from ..transport.ccst import CCSTSolver
from ..relation.features import pair_features
from ..relation.guards import apply_guards
from ..utils import DATA_DIR, RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from ..viz import plots
from .common import cross_validate, feature_matrix, featurize
from .synthetic import NO_CTX_FEATURES

MODEL_PATH = DATA_DIR / "cache" / "ccst_relation_model.pkl"


def synthetic_model(cfg: Config) -> RelationClassifier:
    """5-way CCST relation model trained on the whole synthetic benchmark (used zero-shot)."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    with open(DATA_DIR / "cache" / "synthetic_state.pkl", "rb") as f:
        st = pickle.load(f)
    X = st["mats"]["CCST (ours)"]
    y = [p["label"] for p in st["S"]["pairs"]]
    groups = np.array([p["template"] for p in st["S"]["pairs"]])
    cal = np.isin(groups, np.unique(groups)[::5])
    clf = RelationClassifier(seed=cfg.seed).fit(X[~cal], [v for v, c in zip(y, cal) if not c])
    clf.calibrate(X[cal], [v for v, c in zip(y, cal) if c])
    with open(ensure_dir(MODEL_PATH.parent) / MODEL_PATH.name, "wb") as f:
        pickle.dump(clf, f)
    return clf


def focus_claims(claims: dict, focal_text: dict[str, str]) -> dict:
    """Keep only claims whose sentence overlaps the annotated claim (scope still comes from the abstract)."""
    def toks(t):
        return set(re.findall(r"[a-z]{3,}", t.lower()))

    out = {}
    for d, cl in claims.items():
        ft = toks(focal_text.get(d, ""))
        keep = [c for c in cl if ft and len(toks(c.sentence) & ft) / max(len(ft), 1) >= 0.6]
        out[d] = keep or cl
    return out


def extraction_stats(graphs: dict) -> dict:
    g = list(graphs.values())
    ctx_cov = {h: float(np.mean([any(c.context.get(h) is not None for c in G.claims) for G in g if G.claims] or [0]))
               for h in ("population", "dose", "time", "environment", "design")}
    return {"docs": len(g), "with_claims": float(np.mean([G.n > 0 for G in g])),
            "mean_claims": float(np.mean([len(G.claims) for G in g])), "mean_nodes": float(np.mean([G.n for G in g])),
            **{f"ctx_{k}": v for k, v in ctx_cov.items()}}


def method_matrices(texts: dict, pairs: list[dict], graphs: dict, claims: dict, cfg: Config, use_nli: bool,
                    n_jobs: int) -> dict:
    metas = {d: {"category": "med"} for d in texts}
    P = [(p["a"], p["b"]) for p in pairs]
    M: dict[str, np.ndarray] = {}
    with timer("CCST features"):
        f, _ = featurize(P, graphs, metas, cfg.ccst, cfg.context, n_jobs=n_jobs)
    M["CCST (ours)"] = feature_matrix(f)
    f2, _ = featurize(P, graphs, metas, replace(cfg.ccst, use_context=False), cfg.context, n_jobs=n_jobs)
    M["Signed matcher w/o context"] = feature_matrix(f2, NO_CTX_FEATURES)
    f3, _ = featurize(P, graphs, metas, replace(cfg.ccst, use_context=False, partial=False), cfg.context, n_jobs=n_jobs)
    M["FNGW (full, edge features)"] = feature_matrix(f3, ["D_unsigned", "D_signed", "gap"])
    emb = get_embedder()
    ids = list(texts)
    E = dict(zip(ids, emb.encode_corpus([texts[i] for i in ids], "realpairs")))
    M["SBERT (MiniLM)"] = np.array([bt.cosine_features(E, a, b) for a, b in P])
    bm = bt.BM25Pairs(texts)
    M["BM25"] = np.array([bm.features(a, b) for a, b in P])
    M["Raw SVO overlap"] = np.array([bt.raw_svo_features(claims[a], claims[b]) for a, b in P])
    M["Context clf. w/o transport"] = np.array([gb.context_without_transport(graphs[a], graphs[b], emb, cfg.context)
                                                for a, b in P])
    if use_nli:
        with timer("NLI"):
            nli = bt.NLIScorer()
            M["NLI (BART-MNLI)"] = nli.pair_features([texts[a] for a, _ in P], [texts[b] for _, b in P])
        M["CCST + NLI (hybrid)"] = np.hstack([M["CCST (ours)"], M["NLI (BART-MNLI)"]])
    return M, f


def evaluate(M: dict, pairs: list[dict], labels: tuple, positive: str, seed: int) -> tuple[list[dict], dict]:
    y = [p["label"] for p in pairs]
    groups = np.array([str(p["group"]) for p in pairs])
    rows, preds = [], {}
    for name, X in M.items():
        cv = cross_validate(X, y, groups, seed=seed, labels=labels)
        preds[name] = cv["pred"]
        k = list(labels).index(positive)
        mf = lambda a, b: f1_score(a, b, labels=list(labels), average="macro", zero_division=0)  # noqa: E731
        lo, hi = bootstrap_metric_ci(y, cv["pred"], mf, n_boot=500, seed=seed)
        rows.append({"method": name, "macro_f1": mf(y, cv["pred"]), "macro_f1_lo": lo, "macro_f1_hi": hi,
                     f"f1_{positive}": f1_score(y, cv["pred"], labels=[positive], average="macro", zero_division=0),
                     f"auprc_{positive}": average_precision_score(np.array(y) == positive, cv["proba"][:, k]),
                     "accuracy": float(np.mean(np.array(cv["pred"]) == np.array(y)))})
    ref = "NLI (BART-MNLI)" if "NLI (BART-MNLI)" in preds else "SBERT (MiniLM)"
    mf = lambda a, b: f1_score(a, b, labels=list(labels), average="macro", zero_division=0)  # noqa: E731
    for r in rows:
        if r["method"] in ("CCST (ours)", "CCST + NLI (hybrid)"):
            r[f"p_vs_{ref}"] = paired_bootstrap_pvalue(y, preds[r["method"]], preds[ref], mf, n_boot=500, seed=seed)
    return sorted(rows, key=lambda r: -r["macro_f1"]), preds


def zero_shot_context_analysis(texts, pairs, graphs, cfg: Config, n_examples: int = 8) -> dict:
    clf = synthetic_model(cfg)
    solver = CCSTSolver(cfg.ccst, cfg.context)
    dist: dict[str, dict] = {}
    examples = []
    for p in pairs:
        g1, g2 = graphs[p["a"]], graphs[p["b"]]
        res = solver.solve(g1, g2)
        f = pair_features(g1, g2, res, cfg.ccst, cfg.context)
        f["S_domain"] = 0.0
        P = clf.predict_proba(feature_matrix([f]))[0]
        lab = apply_guards(RELATION_LABELS[int(P.argmax())], f) if P.max() >= clf.tau else "abstain"
        dist.setdefault(p["label"], {}).setdefault(lab, 0)
        dist[p["label"]][lab] += 1
        if p["label"] == "conflict" and lab == "heterogeneity" and len(examples) < n_examples:
            cert = build_certificate(g1, g2, res, f, lab, cfg.context, prob=float(P.max()))
            examples.append({"question": p.get("question"), "certificate": cert["text"]})
    return {"distribution": dist, "heterogeneity_examples": examples}


def run(cfg: Config, use_nli: bool = True, n_jobs: int = 4) -> dict:
    out = ensure_dir(RESULTS_DIR / "real_pairs")
    results = {}
    cd.download_scifact()
    cd.download_mancon()
    mc_claims = {c["cid"]: c["text"] for c in cd.mancon_claims()}
    for name, loader, labels, positive, focus in (
        ("ManConCorpus (claim-focused)", lambda: cd.mancon_pairs(with_abstracts=True)[:2], ("agreement", "conflict"),
         "conflict", True),
        ("ManConCorpus (full abstract)", lambda: cd.mancon_pairs(with_abstracts=True)[:2], ("agreement", "conflict"),
         "conflict", False),
        ("SciFact", cd.scifact_pairs, ("agreement", "contradiction", "incomparable"), "contradiction", False),
    ):
        texts, pairs = loader()
        cd.save_pairs(name.split()[0].lower(), texts, pairs)
        gp = GraphPipeline(cfg.extraction)
        claims = gp.extract(list(texts.items()))
        if focus:
            claims = focus_claims(claims, mc_claims)
        graphs = gp.build(claims)
        stats = extraction_stats(graphs)
        log.info(f"{name} extraction: {stats}")
        M, _ = method_matrices(texts, pairs, graphs, claims, cfg, use_nli, n_jobs)
        rows, _ = evaluate(M, pairs, labels, positive, cfg.seed)
        results[name] = {"extraction": stats, "results": rows, "n_pairs": len(pairs),
                         "label_counts": {l: sum(p["label"] == l for p in pairs) for l in labels}}
        if name.startswith("ManConCorpus"):
            try:
                results[name]["zero_shot"] = zero_shot_context_analysis(texts, pairs, graphs, cfg)
            except FileNotFoundError:
                log.warning("run the synthetic experiment first for the zero-shot analysis")
        plots.bar_ci([r["method"] for r in rows], [r["macro_f1"] for r in rows], [r["macro_f1_lo"] for r in rows],
                     [r["macro_f1_hi"] for r in rows], highlight={"CCST (ours)", "CCST + NLI (hybrid)"},
                     xlabel="macro-F1 (95% bootstrap CI)", title=f"{name}: grouped 5-fold CV",
                     path=out / f"{re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')}_macro_f1.png", xlim=(0, 1.08))
    save_json(results, out / "metrics.json")
    md = ["# Real expert-labelled pairs", ""]
    for name, r in results.items():
        md += [f"## {name} ({r['n_pairs']} pairs, {r['label_counts']})", "", "Extraction coverage:", "",
               md_table([r["extraction"]]), "", md_table(r["results"]), ""]
        if "zero_shot" in r:
            md += ["### Zero-shot CCST (trained only on the synthetic benchmark)", "",
                   md_table([{"gold": g, **d} for g, d in r["zero_shot"]["distribution"].items()]), "",
                   "Conflict pairs explained by a material context difference:", ""]
            md += [f"- *{e['question']}*: {e['certificate']}" for e in r["zero_shot"]["heterogeneity_examples"]]
            md += [""]
    (out / "REPORT.md").write_text("\n".join(md))
    return results
