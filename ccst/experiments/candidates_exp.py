"""Item 5: candidate-recall improvement (docs/candidate_recall_protocol.md).

Evaluation subset = the 500 real documents used for exact all-pairs scaling; training subset = the next 500
documents of the same pool (disjoint). Exact all-pairs CCST on each subset defines every document's exact
top-5 neighbours. The new index's weights and codebook use the training subset only; both indexes are then
scored on the evaluation subset at matched candidate budgets.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from ..config import Config
from ..embed import get_embedder
from ..extraction.graph_builder import GraphPipeline
from ..index.candidates import CandidateIndex, CombinedIndex, ConceptCodebook, candidate_recall
from ..topics.cluster import kmeans
from ..utils import RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from ..viz import plots
from .common import featurize
from .corpus import load_corpus

KS = (2, 5, 10, 20)
GRID = (0.0, 0.5, 1.0, 2.0)


def exact_scores(ids, graphs, metas, cfg, n_jobs) -> dict:
    pairs = list(itertools.combinations(range(len(ids)), 2))
    feats, _ = featurize([(ids[i], ids[j]) for i, j in pairs], graphs, metas, cfg.ccst, cfg.context, n_jobs=n_jobs)
    out = {}
    for (i, j), f in zip(pairs, feats):
        s = -f["D_unsigned"] if f["matched_frac"] > 0 else -1e9
        out[(i, j)] = out[(j, i)] = s
    return out


def run(cfg: Config, n_jobs: int = 4, n_sub: int = 500) -> dict:
    out = ensure_dir(RESULTS_DIR / "candidates")
    docs = load_corpus()
    rng = np.random.default_rng(cfg.seed)
    pool = rng.choice(len(docs), size=min(2000, len(docs)), replace=False)  # same pool as the scaling experiment
    ev, tr = pool[:n_sub], pool[n_sub:2 * n_sub]
    sub = {"eval": ev, "train": tr}
    all_idx = np.concatenate([ev, tr])
    texts = {docs[i]["doc_id"]: docs[i]["text"] for i in all_idx}
    with timer("extraction (eval + train subsets)"):
        graphs = GraphPipeline(cfg.extraction).run(list(texts.items()))
    metas = {docs[i]["doc_id"]: {"category": docs[i]["label"]} for i in all_idx}
    E_all = get_embedder().encode([texts[docs[i]["doc_id"]] for i in all_idx])
    topics_all = kmeans(E_all, 20, cfg.seed)
    data = {}
    for name, idx in sub.items():
        ids = [docs[i]["doc_id"] for i in idx]
        off = 0 if name == "eval" else len(ev)
        with timer(f"exact all-pairs CCST on the {name} subset"):
            ex = exact_scores(ids, graphs, metas, cfg, n_jobs)
        data[name] = {"ids": ids, "E": E_all[off:off + len(idx)], "topics": topics_all[off:off + len(idx)],
                      "exact": ex, "with_mech": [i for i, d in enumerate(ids) if graphs[d].n > 0]}
    n_pairs = n_sub * (n_sub - 1) / 2
    # ---- learn the combined-index weights on the training subset only
    t0 = time.perf_counter()
    codebook = ConceptCodebook(256, cfg.seed).fit([graphs[d] for d in data["train"]["ids"]])
    tr_idx = CombinedIndex(codebook, explore=cfg.index.explore, seed=cfg.seed).fit(data["train"]["ids"], graphs,
                                                                                   data["train"]["E"])
    best, best_r = None, -1.0
    for w in itertools.product(GRID, repeat=4):
        if not any(w):
            continue
        wd = dict(zip(CombinedIndex.BLOCKS, w))
        r = candidate_recall(tr_idx.candidates(10, wd), data["train"]["exact"], data["train"]["with_mech"], k=5)
        if r > best_r + 1e-9:
            best, best_r = wd, r
    learn_s = time.perf_counter() - t0
    log.info(f"learned weights {best} (training recall@5 {best_r:.3f}) in {learn_s:.1f}s")
    # ---- evaluate both indexes on the evaluation subset
    E = data["eval"]
    rows = []
    for k in KS:
        for blocking in (True, False):
            t0 = time.perf_counter()
            old = CandidateIndex(cfg.index.__class__(k_sem=k, k_mech=k, k_ctx=max(1, k // 2),
                                                    explore=cfg.index.explore, topic_blocking=blocking), cfg.seed)
            cand = old.fit(E["ids"], graphs, E["E"], E["topics"]).candidates()["all"]
            rows.append({"index": f"original{' + topic blocking' if blocking else ''}", "k": k, "pairs": len(cand),
                         "pair_fraction": len(cand) / n_pairs, "seconds": time.perf_counter() - t0,
                         "recall@5": candidate_recall(cand, E["exact"], E["with_mech"], k=5)})
        for label, w in (("combined, learned weights", best), ("combined, uniform weights", None)):
            t0 = time.perf_counter()
            idx = CombinedIndex(codebook, w, explore=cfg.index.explore, seed=cfg.seed).fit(E["ids"], graphs, E["E"])
            cand = idx.candidates(k)
            rows.append({"index": label, "k": k, "pairs": len(cand), "pair_fraction": len(cand) / n_pairs,
                         "seconds": time.perf_counter() - t0,
                         "recall@5": candidate_recall(cand, E["exact"], E["with_mech"], k=5)})
    save_json({"rows": rows, "learned_weights": best, "train_recall@5": best_r, "learn_seconds": learn_s},
              out / "metrics.json")
    labels = sorted({r["index"] for r in rows})
    plots.lines({l: [r["pair_fraction"] for r in rows if r["index"] == l] for l in labels},
                {l: [r["recall@5"] for r in rows if r["index"] == l] for l in labels},
                "fraction of all pairs solved exactly", "recall of exact top-5 CCST neighbours",
                "Item 5: candidate recall vs cost (evaluation subset, N = 500)", out / "candidate_recall.png",
                ylim=(0, 1.02))
    md = ["# Item 5 — candidate-recall improvement", "",
          "Protocol: `docs/candidate_recall_protocol.md` (written before running). Weights and codebook use the "
          "training subset only; numbers below are on the disjoint evaluation subset (N = 500, 124,750 exact pairs).",
          "", f"Learned weights: {best} (training recall@5 at k = 10: {best_r:.3f}).", "",
          md_table(rows, ["index", "k", "pairs", "pair_fraction", "recall@5", "seconds"])]
    (out / "REPORT.md").write_text("\n".join(md))
    return {"rows": rows, "weights": best}
