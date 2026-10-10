"""Synthetic-benchmark analyses that need couplings and certificates:

* RQ1 alignment: node alignment P/R/F1 and gold transport mass (CCST partial vs forced full vs FGW vs
  Hungarian), unmatched-mechanism accuracy, sign-flip localisation accuracy
* RQ3 certificates: cited-flip precision/recall, context-field correctness, evidence-span IoU,
  minimality, deletion faithfulness (vs. random-deletion control)
* cross-domain analogy retrieval (MRR, Recall@k, nDCG@k)
* Step G budgeted acquisition: random / uncertainty / expected-change / diversity / DiverseTopB / curvature
* Step H signed community detection vs. unsigned Leiden / spectral clustering
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import replace

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold

from ..certificate import build_certificate, span_iou
from ..config import RELATION_LABELS, Config
from ..corpus_graph import correlation_clustering, leiden, signed_leiden, spectral
from ..embed import get_embedder
from ..eval import baselines_text as bt
from ..eval.metrics import alignment_prf, clustering_metrics, retrieval_metrics
from ..extraction.graph_builder import GraphPipeline, build_graph
from ..relation.classifier import RelationClassifier
from ..relation.features import pair_features
from ..synthetic.generator import slot_of
from ..transport.ccst import CCSTSolver
from ..utils import DATA_DIR, RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from ..viz import plots
from .common import feature_matrix
from .synthetic import PROPOSED


def load_state(realism: bool = False) -> dict:
    from .synthetic import bench_paths

    with open(DATA_DIR / "cache" / bench_paths(realism)[1], "rb") as f:
        return pickle.load(f)


def _slots(G, doc) -> list:
    return [slot_of(t, doc) for t in G.node_text]


# ----------------------------------------------------------------------------- RQ1 alignment
def alignment_eval(S: dict, cfg: Config) -> dict:
    docs = {d["doc_id"]: d for d in S["docs"]}
    G = S["graphs"]
    solvers = {"CCST (partial)": CCSTSolver(cfg.ccst, cfg.context),
               "CCST forced full": CCSTSolver(replace(cfg.ccst, partial=False), cfg.context)}
    groups = {"agree_para": "paraphrase", "agree_syn": "disjoint-vocabulary synonyms", "analogy": "cross-domain analogy",
              "agree_extra": "+ extra mechanism", "contra_extra": "+ extra mechanism", "contra_flip": "sign flip",
              "contra_null": "sign flip", "hetero_ctx": "context shift", "hetero_sim": "context shift"}
    rows = defaultdict(lambda: defaultdict(list))
    unmatched = defaultdict(list)
    for p in S["pairs"]:
        grp = groups.get(p["variant"])
        if grp is None:
            continue
        g1, g2 = G[p["a"]], G[p["b"]]
        if g1.n == 0 or g2.n == 0:
            continue
        s1, s2 = _slots(g1, docs[p["a"]]), _slots(g2, docs[p["b"]])
        plans = {}
        for name, sol in solvers.items():
            plans[name] = sol.solve(g1, g2).T
        M = np.maximum(0, 2 - 2 * g1.X @ g2.X.T)
        A1, A2 = g1.channels(False)["A"], g2.channels(False)["A"]
        plans["FGW (POT)"] = ot.gromov.fused_gromov_wasserstein(M, A1, A2, np.full(g1.n, 1 / g1.n),
                                                                np.full(g2.n, 1 / g2.n), alpha=0.5)
        C = M + 10 * (g1.roles[:, None] != g2.roles[None, :])
        r, c = linear_sum_assignment(C)
        H = np.zeros_like(M)
        H[r, c] = 1.0
        plans["Hungarian (role-constrained)"] = H
        for name, T in plans.items():
            m = alignment_prf(T, s1, s2)
            for k, v in m.items():
                rows[(grp, name)][k].append(v)
            if p.get("extra_mech"):
                extra = [j for j, s in enumerate(s2) if s is None]
                if extra:
                    moved = T[:, extra].sum() / max(T.sum(), 1e-12)
                    unmatched[name].append(1.0 - min(1.0, moved / (len(extra) / g2.n)))
    table = [{"variant": g, "method": n, **{k: float(np.mean(v)) for k, v in d.items()}, "n": len(d["align_f1"])}
             for (g, n), d in rows.items()]
    unm = [{"method": k, "unmatched_extra_accuracy": float(np.mean(v)), "n": len(v)} for k, v in unmatched.items()]
    return {"alignment": table, "unmatched": unm}


# ----------------------------------------------------------------------------- RQ3 certificates
def certificate_eval(S: dict, state: dict, cfg: Config) -> dict:
    pairs, G, docs = S["pairs"], S["graphs"], {d["doc_id"]: d for d in S["docs"]}
    y = [p["label"] for p in pairs]
    groups = np.array([p["template"] for p in pairs])
    X = state["mats"][PROPOSED]
    tr, te = next(GroupKFold(5).split(X, y, groups))
    clf = RelationClassifier(seed=cfg.seed).fit(X[tr], [y[i] for i in tr])
    clf.calibrate(X[tr], [y[i] for i in tr])
    solver = CCSTSolver(cfg.ccst, cfg.context)
    gp = GraphPipeline(cfg.extraction)
    gen = S["generalizer"]
    rng = np.random.default_rng(cfg.seed)
    stats = defaultdict(list)
    examples = []
    for i in te:
        p = pairs[i]
        if p["label"] not in ("contradiction", "heterogeneity") or "flipped" not in p:
            continue
        g1, g2 = G[p["a"]], G[p["b"]]
        res = solver.solve(g1, g2)
        f = pair_features(g1, g2, res, cfg.ccst, cfg.context)
        f["S_domain"] = state["feats"][PROPOSED][i]["S_domain"]
        P = clf.predict_proba(feature_matrix([f]))[0]
        lab = RELATION_LABELS[int(P.argmax())]
        cert = build_certificate(g1, g2, res, f, lab, cfg.context, prob=float(P.max()))
        if len(examples) < 6:
            examples.append(cert)
        gold = {tuple(e) for e in p["flipped"]}
        s2 = _slots(g2, docs[p["b"]])
        cited = set()
        for item in cert["F"]:
            e = g2.edges[item["edge2"]]
            cited.add((s2[e.u], s2[e.v]))
        tp = len(cited & gold)
        stats["flip_precision"].append(tp / len(cited) if cited else 0.0)
        stats["flip_recall"].append(tp / len(gold))
        stats["minimality_ratio"].append(len(cited) / len(gold))
        stats["localised_top1"].append(float(bool(cert["F"]) and (lambda e: (s2[e.u], s2[e.v]))(g2.edges[cert["F"][0]["edge2"]]) in gold))
        # evidence span IoU (doc B side) against the gold sentence of the flipped edge
        k_flip = [k for k, (u, v) in enumerate(_template_edges(S, p)) if (u, v) in gold]
        gold_spans = [tuple(docs[p["b"]]["edge_spans"][k]) for k in k_flip if k in docs[p["b"]]["edge_spans"]]
        b_spans = [tuple(x["span"]) for x in cert["P"] if x["doc"] == p["b"]]
        if gold_spans and b_spans:
            stats["evidence_iou"].append(max(span_iou(a, b) for a in b_spans for b in gold_spans))
        if p["label"] == "heterogeneity":
            dim = p["changed_dim"]
            K = cert["K"]
            stats["context_field_correct"].append(float(K.get(dim, {}).get("status") == "incompatible"))
        else:  # contradiction: no context field may be (falsely) reported as incompatible; unknown is allowed
            stats["context_no_false_incompatibility"].append(
                float(all(v["status"] != "incompatible" for v in cert["K"].values())))
        # deletion faithfulness: remove cited evidence sentences from doc B vs. a random other claim sentence
        if cert["P"]:
            text = docs[p["b"]]["text"]
            cited_spans = sorted({tuple(x["span"]) for x in cert["P"] if x["doc"] == p["b"]})
            other = [tuple(sp) for k, sp in docs[p["b"]]["edge_spans"].items() if tuple(sp) not in cited_spans]
            drop_c = _delete(text, cited_spans)
            p_lab = P[RELATION_LABELS.index(lab)]
            for tag, newtext in (("cited", drop_c),
                                 ("random", _delete(text, [other[rng.integers(len(other))]]) if other else None)):
                if newtext is None:
                    continue
                cl = gp.extract([(p["b"] + "_del", newtext)])
                g2d = build_graph(p["b"] + "_del", cl[p["b"] + "_del"], gen, g2.X.shape[1], g2.meta)
                r2 = solver.solve(g1, g2d)
                f2 = pair_features(g1, g2d, r2, cfg.ccst, cfg.context)
                f2["S_domain"] = f["S_domain"]
                P2 = clf.predict_proba(feature_matrix([f2]))[0]
                stats[f"prob_drop_{tag}"].append(float(p_lab - P2[RELATION_LABELS.index(lab)]))
    summary = {k: float(np.mean(v)) for k, v in stats.items()}
    summary["n"] = len(stats["flip_recall"])
    if stats.get("prob_drop_cited"):
        summary["faithfulness_rate"] = float(np.mean(np.array(stats["prob_drop_cited"]) > 0.2))
        summary["faithfulness_rate_random"] = float(np.mean(np.array(stats["prob_drop_random"]) > 0.2))
    return {"summary": summary, "examples": examples}


def _template_edges(S, p):
    t = next(t for t in S["templates"] if t.tid == p["template"])
    return [(i, j) for i, j, _ in t.edges]


def _delete(text: str, spans: list[tuple]) -> str:
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return " ".join(x.strip() for x in out if x.strip())


# ----------------------------------------------------------------------------- retrieval
def retrieval_eval(S: dict, cfg: Config, use_specter: bool) -> dict:
    docs = [d for d in S["docs"] if d["doc_id"].endswith("_base")]
    ids = [d["doc_id"] for d in docs]
    tpl = {d["doc_id"]: d["template"] for d in docs}
    dom = {d["doc_id"]: d["domain"] for d in docs}
    G, texts = S["graphs"], S["texts"]
    cands = {q: [c for c in ids if dom[c] != dom[q]] for q in ids}
    relevant = {q: {c for c in cands[q] if tpl[c] == tpl[q]} for q in ids}
    emb = get_embedder()
    E = dict(zip(ids, emb.encode([texts[i] for i in ids])))
    scorers = {"SBERT": lambda q, c: float(E[q] @ E[c]), "LDA θ": lambda q, c: float(S["theta"][q] @ S["theta"][c])}
    if use_specter:
        sp = get_embedder(cfg.topics.specter_model)
        Es = dict(zip(ids, sp.encode([texts[i] for i in ids])))
        scorers["SPECTER"] = lambda q, c: float(Es[q] @ Es[c])
    bm = bt.BM25Pairs({i: texts[i] for i in ids})
    solver = CCSTSolver(cfg.ccst, cfg.context)
    from ..transport.baselines import standard_fgw

    def ccst_score(q, c):
        r = solver.solve(G[q], G[c])
        return -(r.D_mech + (r.D_signed - r.D_unsigned)) if r.matched > 0 else -1e9

    scorers["FGW (POT)"] = lambda q, c: -standard_fgw(G[q], G[c])[0]
    scorers["CCST signed mechanism"] = ccst_score
    out = {}
    for name, fn in scorers.items():
        with timer(f"retrieval {name}", quiet=True):
            rankings = {q: sorted(cands[q], key=lambda c: -fn(q, c)) for q in ids}
        out[name] = retrieval_metrics(rankings, relevant)
    out["BM25"] = retrieval_metrics({q: bm.rank(q, cands[q]) for q in ids}, relevant)
    return out


# ----------------------------------------------------------------------------- Step G acquisition
def acquisition_eval(S: dict, state: dict, cfg: Config, budget: int = 150, batch: int = 10, seeds: int = 5) -> dict:
    from sklearn.metrics import f1_score

    pairs = S["pairs"]
    X = state["mats"][PROPOSED]
    y = np.array([p["label"] for p in pairs])
    groups = np.array([p["template"] for p in pairs])
    pool_idx, test_idx = next(GroupKFold(5).split(X, y, groups))
    # curvature on the candidate graph (documents = nodes, candidate pairs = edges): Forman curvature
    deg = defaultdict(int)
    for p in pairs:
        deg[p["a"]] += 1
        deg[p["b"]] += 1
    forman = np.array([4 - deg[p["a"]] - deg[p["b"]] for p in pairs], float)
    kappa = (forman.max() - forman) / (forman.max() - forman.min() + 1e-9)  # high = more negative = bridge
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)

    def scores(policy, clf, cand, labelled):
        P = clf.predict_proba(X[cand])
        H = -(P * np.log(P + 1e-12)).sum(1)
        if policy == "uncertainty":
            return H
        if policy == "expected change":
            e = np.eye(P.shape[1])
            return np.linalg.norm(Xs[cand], axis=1) * sum(P[:, k] * np.linalg.norm(P - e[k], axis=1) for k in range(P.shape[1]))
        if policy == "uncertainty + curvature":
            return H / (H.max() + 1e-9) + 0.5 * kappa[cand]
        return H

    def kcenter(cand, labelled, k):
        chosen = []
        D = np.min(((Xs[cand, None] - Xs[None, labelled]) ** 2).sum(-1), 1) if len(labelled) else np.full(len(cand), np.inf)
        for _ in range(min(k, len(cand))):
            j = int(np.argmax(D))
            chosen.append(cand[j])
            D = np.minimum(D, ((Xs[cand] - Xs[cand[j]]) ** 2).sum(-1))
        return chosen

    policies = ["random", "uncertainty", "expected change", "diversity (k-center)", "DiverseTopB",
                "uncertainty + curvature"]
    curves = {p: [] for p in policies}
    budgets = list(range(batch, budget + 1, batch))
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        init = list(rng.choice(pool_idx, size=batch, replace=False))
        for pol in policies:
            lab = list(init)
            accs = []
            for b in budgets:
                while len(lab) < b:
                    cand = np.array([i for i in pool_idx if i not in set(lab)])
                    if pol == "random":
                        new = list(rng.choice(cand, size=batch, replace=False))
                    elif pol == "diversity (k-center)":
                        new = kcenter(cand, np.array(lab), batch)
                    else:
                        clf = _fit(X, y, lab, cfg.seed)
                        sc = scores(pol, clf, cand, lab)
                        if pol == "DiverseTopB":
                            top = cand[np.argsort(-sc)[: 3 * batch]]
                            new = kcenter(top, np.array(lab), batch)
                        else:
                            new = list(cand[np.argsort(-sc)[:batch]])
                    lab += [int(i) for i in new]
                clf = _fit(X, y, lab, cfg.seed)
                accs.append(f1_score(y[test_idx], clf.predict(X[test_idx]), labels=list(RELATION_LABELS),
                                     average="macro", zero_division=0))
            curves[pol].append(accs)
    return {"budgets": budgets, "curves": {p: np.array(v).mean(0).tolist() for p, v in curves.items()},
            "std": {p: np.array(v).std(0).tolist() for p, v in curves.items()},
            "full_pool_f1": float(f1_score(y[test_idx], _fit(X, y, list(pool_idx), cfg.seed).predict(X[test_idx]),
                                           labels=list(RELATION_LABELS), average="macro", zero_division=0))}


def _fit(X, y, idx, seed):
    idx = list(idx)
    clf = RelationClassifier(calibration="none", seed=seed)
    return clf.fit(X[idx], list(y[idx]))


# ----------------------------------------------------------------------------- Step H signed clustering
def signed_clustering_eval(S: dict, state: dict, cfg: Config) -> dict:
    pairs, docs = S["pairs"], S["docs"]
    ids = [d["doc_id"] for d in docs]
    idx = {d: i for i, d in enumerate(ids)}
    gold = []
    for d in docs:
        gold.append(f"{d['template']}|" + "|".join(f"{k}:{v}" for k, v in sorted(d["rels"].items())))
    P = state["cvs"][PROPOSED]["proba"]
    L = list(RELATION_LABELS)
    pos, neg = [], []
    for p, pr in zip(pairs, P):
        i, j = idx[p["a"]], idx[p["b"]]
        wpos = pr[L.index("agreement")] + pr[L.index("analogy")]
        wneg = pr[L.index("contradiction")]
        if wpos > 0.5:
            pos.append((i, j, float(wpos)))
        if wneg > 0.5:
            neg.append((i, j, float(wneg)))
    E = get_embedder().encode([S["texts"][d] for d in ids])
    n_gold = len(set(gold))
    out = {}
    out["Signed Leiden (CCST relations)"] = clustering_metrics(gold, signed_leiden(len(ids), pos, neg, seed=cfg.seed))
    out["Correlation clustering (CCST relations)"] = clustering_metrics(gold, correlation_clustering(len(ids), pos, neg,
                                                                                                      seed=cfg.seed))
    out["Leiden, positive edges only"] = clustering_metrics(gold, leiden(len(ids), pos, seed=cfg.seed))
    sim_edges = _knn_edges(E, 10)
    out["Leiden on SBERT kNN graph"] = clustering_metrics(gold, leiden(len(ids), sim_edges, seed=cfg.seed))
    # The realistic setting: a semantic similarity graph (which links contradicting near-duplicates)
    # plus CCST's signed relation edges as positive evidence and negative constraints.
    out["Signed Leiden: SBERT kNN + CCST (+/-)"] = clustering_metrics(
        gold, signed_leiden(len(ids), sim_edges + pos, neg, seed=cfg.seed))
    out["Leiden: SBERT kNN + CCST (+ only)"] = clustering_metrics(gold, leiden(len(ids), sim_edges + pos, seed=cfg.seed))
    out["Spectral on SBERT (K = #families)"] = clustering_metrics(gold, spectral(E, n_gold, seed=cfg.seed))
    out["_n_families"] = n_gold
    return out


def _knn_edges(E: np.ndarray, k: int) -> list:
    S = E @ E.T
    np.fill_diagonal(S, -1)
    nn = np.argsort(-S, 1)[:, :k]
    return [(i, int(j), float(S[i, j])) for i in range(len(E)) for j in nn[i] if i < j or i not in nn[j]]


# ----------------------------------------------------------------------------- comparability guards
def guard_effect(S: dict, state: dict, cfg: Config) -> dict:
    """Effect of the post-classification comparability guards on the out-of-fold CCST predictions."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    from ..relation.guards import apply_guards

    solver = CCSTSolver(cfg.ccst, cfg.context)
    G = S["graphs"]
    y = [p["label"] for p in S["pairs"]]
    pred = state["cvs"][PROPOSED]["pred"]
    guarded = [apply_guards(t, pair_features(G[p["a"]], G[p["b"]], solver.solve(G[p["a"]], G[p["b"]]), cfg.ccst,
                                             cfg.context)) for p, t in zip(S["pairs"], pred)]
    out = {"changed": int(sum(a != b for a, b in zip(pred, guarded)))}
    for name, pr in (("unguarded", pred), ("guarded", guarded)):
        P, R, _, _ = precision_recall_fscore_support(y, pr, labels=list(RELATION_LABELS), zero_division=0)
        out[name] = {"macro_f1": float(f1_score(y, pr, labels=list(RELATION_LABELS), average="macro")),
                     **{f"P_{l}": float(a) for l, a in zip(RELATION_LABELS, P)},
                     **{f"R_{l}": float(b) for l, b in zip(RELATION_LABELS, R)}}
    return out


# ----------------------------------------------------------------------------- main
def run(cfg: Config, use_specter: bool = True, quick: bool = False, realism: bool = False) -> dict:
    from .synthetic import bench_paths

    out = ensure_dir(RESULTS_DIR / bench_paths(realism)[0])
    state = load_state(realism)
    S = state["S"]
    gp = GraphPipeline(cfg.extraction)
    claims = S["claims"]
    S["graphs"] = gp.build(claims, S["metas"])  # rebuild to recover the fitted concept generaliser
    S["generalizer"] = gp.generalizer
    results = {}
    with timer("alignment evaluation"):
        results["alignment"] = alignment_eval(S, cfg)
    with timer("certificate evaluation"):
        results["certificates"] = certificate_eval(S, state, cfg)
    with timer("analogy retrieval"):
        results["retrieval"] = retrieval_eval(S, cfg, use_specter)
    with timer("budgeted acquisition"):
        results["acquisition"] = acquisition_eval(S, state, cfg, budget=100 if quick else 150, seeds=2 if quick else 10)
    with timer("signed clustering"):
        results["signed_clustering"] = signed_clustering_eval(S, state, cfg)
    with timer("comparability guards"):
        results["guards"] = guard_effect(S, state, cfg)
    save_json(results, out / "analysis.json")
    report(results, out)
    return results


def report(R: dict, out) -> None:
    al = R["alignment"]["alignment"]
    variants = sorted({r["variant"] for r in al})
    methods = ["CCST (partial)", "CCST forced full", "FGW (POT)", "Hungarian (role-constrained)"]
    series = {m: [next((r["align_f1"] for r in al if r["variant"] == v and r["method"] == m), np.nan) for v in variants]
              for m in methods}
    plots.grouped_bars(variants, series, "node-alignment F1", "RQ1: mechanism correspondence recovery",
                       out / "alignment_f1.png", ylim=(0, 1.05))
    ret = R["retrieval"]
    names = list(ret)
    plots.bar_ci(names, [ret[n]["MRR"] for n in names], highlight={"CCST signed mechanism"}, xlabel="MRR",
                 title="Cross-domain analogy retrieval (same mechanism, other field)", path=out / "analogy_retrieval.png",
                 xlim=(0, 1.08))
    acq = R["acquisition"]
    plots.lines(acq["budgets"], acq["curves"], "labels acquired (review budget)", "macro-F1 on held-out templates",
                "Step G: budgeted review policies", out / "acquisition.png", ref_line=acq["full_pool_f1"])
    sc = {k: v for k, v in R["signed_clustering"].items() if not k.startswith("_")}
    cert = R["certificates"]["summary"]
    md = ["# Synthetic analyses", "", "## RQ1 alignment (node-alignment F1 by variant)", "",
          md_table([{"variant": v, **{m: series[m][i] for m in methods}} for i, v in enumerate(variants)]), "",
          "Unmatched-mechanism accuracy (share of extra-mechanism mass left unmatched):", "",
          md_table(R["alignment"]["unmatched"]), "", "## RQ3 certificates", "", md_table([cert]), "",
          "## Cross-domain analogy retrieval", "", md_table([{"method": k, **v} for k, v in ret.items()]), "",
          "## Step H: mechanism families (signed vs unsigned clustering)", "",
          md_table([{"method": k, **v} for k, v in sc.items()]), "",
          f"Gold families: {R['signed_clustering']['_n_families']}", "",
          "## Comparability guards (post-classification)", "",
          f"Predictions changed: {R['guards']['changed']}", "",
          md_table([{"setting": k, **v} for k, v in R["guards"].items() if k != "changed"]), "",
          "## Example certificates", ""]
    for c in R["certificates"]["examples"][:4]:
        md += [f"- **{c['doc1']} vs {c['doc2']}**: {c['text']}", ""]
    (out / "ANALYSIS.md").write_text("\n".join(md))
    log.info(f"analysis written to {out}/ANALYSIS.md")

