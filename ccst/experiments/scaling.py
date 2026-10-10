"""RQ4: retrieval loss from sparse candidate generation and empirical runtime scaling.

* exact all-pairs CCST on N docs: N(N-1)/2 problems -> expected slope 2 on a log-log plot
* sparse pipeline (sketches + kNN + topic blocking + CCST on candidates): O(N log N + N k c_CCST)
  -> expected slope ~1
* candidate recall: share of each document's exact top-5 CCST neighbours retained by the candidates,
  as a function of k
* per-pair cost c_CCST vs graph size n (random signed DAGs): theory O(I K (n^2 m + n m^2) + EMD)
Slopes are fitted by least squares in log-log space; runtimes are measured, never assumed.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import replace

import numpy as np

from ..config import Config
from ..embed import get_embedder
from ..extraction.graph_builder import GraphPipeline
from ..graph_factory import make_graph
from ..index.candidates import CandidateIndex
from ..transport.ccst import CCSTSolver
from ..utils import RESULTS_DIR, ensure_dir, log, md_table, peak_rss_mb, save_json, timer
from ..viz import plots
from .common import featurize
from .corpus import load_corpus


def _slope(x, y) -> float:
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def random_graph(n: int, rng, dim: int = 32):
    rels = ["increases", "decreases", "enables", "inhibits"]
    triples = []
    for v in range(1, n):
        for u in rng.choice(v, size=min(v, 1 + int(rng.random() < 0.5)), replace=False):
            triples.append((f"n{u}", str(rng.choice(rels)), f"n{v}"))
    vecs = {f"n{i}": rng.normal(size=dim) for i in range(n)}
    return make_graph(f"rand{n}_{rng.integers(1e9)}", triples, vecs)


def per_pair_cost(cfg: Config, sizes=(4, 8, 16, 32, 64), reps: int = 20) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    solver = CCSTSolver(cfg.ccst, cfg.context)
    rows = []
    for n in sizes:
        ts, its = [], []
        for _ in range(reps if n <= 32 else max(3, reps // 4)):
            g1, g2 = random_graph(n, rng), random_graph(n, rng)
            t0 = time.perf_counter()
            r = solver.solve(g1, g2)
            ts.append(time.perf_counter() - t0)
            its.append(r.iters)
        rows.append({"n_nodes": n, "ms_per_pair": 1000 * float(np.median(ts)), "fw_iters": float(np.mean(its))})
    return rows


def run(cfg: Config, sizes=(125, 250, 500, 1000, 2000), exact_max: int = 500, n_jobs: int = 4) -> dict:
    out = ensure_dir(RESULTS_DIR / "scaling")
    docs = load_corpus()
    rng = np.random.default_rng(cfg.seed)
    pool = rng.choice(len(docs), size=min(max(sizes), len(docs)), replace=False)
    texts = {docs[i]["doc_id"]: docs[i]["text"] for i in pool}
    ids_all = [docs[i]["doc_id"] for i in pool]
    gp = GraphPipeline(cfg.extraction)
    with timer("extraction for scaling pool"):
        graphs = gp.run(list(texts.items()))
    metas = {d: {"category": docs[i]["label"]} for d, i in zip(ids_all, pool)}
    E = get_embedder().encode([texts[d] for d in ids_all])
    from ..topics.cluster import kmeans

    topics = kmeans(E, 20, cfg.seed)
    rows = []
    exact_scores = {}
    for N in sizes:
        ids = ids_all[:N]
        row = {"N": N, "all_pairs": N * (N - 1) // 2}
        t0 = time.perf_counter()
        idx = CandidateIndex(cfg.index, cfg.seed).fit(ids, graphs, E[:N], topics[:N])
        cand = idx.candidates()["all"]
        t_index = time.perf_counter() - t0
        f, _ = featurize([(ids[i], ids[j]) for i, j in cand], graphs, metas, cfg.ccst, cfg.context, n_jobs=1)
        row.update(sparse_pairs=len(cand), sparse_index_s=t_index, sparse_total_s=time.perf_counter() - t0,
                   pairs_per_doc=len(cand) / N)
        if N <= exact_max:
            allp = list(itertools.combinations(range(N), 2))
            t1 = time.perf_counter()
            fe, _ = featurize([(ids[i], ids[j]) for i, j in allp], graphs, metas, cfg.ccst, cfg.context, n_jobs=1)
            row["exact_total_s"] = time.perf_counter() - t1
            if N == min(exact_max, max(s for s in sizes if s <= exact_max)):
                for (i, j), x in zip(allp, fe):
                    s = -x["D_unsigned"] if x["matched_frac"] > 0 else -1e9
                    exact_scores[(i, j)] = s
                    exact_scores[(j, i)] = s
                exact_ids, exact_N = ids, N
        else:  # extrapolate with the per-pair rate measured on the largest exact run
            exact_rows = [r for r in rows if "exact_total_s" in r]
            per = exact_rows[-1]["exact_total_s"] / exact_rows[-1]["all_pairs"] if exact_rows else 0.0
            row["exact_total_s_extrapolated"] = per * row["all_pairs"]
        row["peak_rss_mb"] = peak_rss_mb()
        rows.append(row)
        log.info(row)
    # candidate recall vs k on the largest exact subset
    recall_rows = []
    sub = [i for i in range(exact_N) if graphs[exact_ids[i]].n > 0]
    for k in (2, 5, 10, 20):
        icfg = replace(cfg.index, k_sem=k, k_mech=k, k_ctx=max(1, k // 2))
        idx = CandidateIndex(icfg, cfg.seed).fit(exact_ids, graphs, E[:exact_N], topics[:exact_N])
        for blocking in (True, False):
            idx.cfg = replace(icfg, topic_blocking=blocking)
            cand = idx.candidates()["all"]
            from ..index.candidates import candidate_recall

            recall_rows.append({"k": k, "topic_blocking": blocking, "pairs": len(cand),
                                "pair_fraction": len(cand) / (exact_N * (exact_N - 1) / 2),
                                "recall@5": candidate_recall(cand, exact_scores, sub, k=5)})
    cost = per_pair_cost(cfg)
    summary = {
        "slope_exact": _slope([r["N"] for r in rows if "exact_total_s" in r], [r["exact_total_s"] for r in rows
                                                                              if "exact_total_s" in r]),
        "slope_sparse": _slope([r["N"] for r in rows], [r["sparse_total_s"] for r in rows]),
        "slope_cost_vs_n": _slope([r["n_nodes"] for r in cost], [r["ms_per_pair"] for r in cost]),
    }
    save_json({"rows": rows, "recall": recall_rows, "per_pair_cost": cost, "summary": summary}, out / "metrics.json")
    xs = [r["N"] for r in rows]
    exact_y = [r.get("exact_total_s", r.get("exact_total_s_extrapolated")) for r in rows]
    plots.lines(xs, {"exact all pairs (measured ≤ %d, then extrapolated)" % exact_max: exact_y,
                     "sparse candidates + CCST": [r["sparse_total_s"] for r in rows]},
                "number of documents N", "wall-clock seconds",
                f"Scaling: exact slope {summary['slope_exact']:.2f} vs sparse slope {summary['slope_sparse']:.2f}",
                out / "runtime_scaling.png", logx=True, logy=True)
    plots.lines([r["n_nodes"] for r in cost], {"CCST per pair": [r["ms_per_pair"] for r in cost]},
                "graph size n (nodes per document)", "milliseconds per pair",
                f"Per-pair cost c_CCST (log-log slope {summary['slope_cost_vs_n']:.2f})", out / "per_pair_cost.png",
                logx=True, logy=True)
    rb = [r for r in recall_rows if r["topic_blocking"]]
    rn = [r for r in recall_rows if not r["topic_blocking"]]
    plots.lines({"with topic blocking": [r["pair_fraction"] for r in rb],
                 "without topic blocking": [r["pair_fraction"] for r in rn]},
                {"with topic blocking": [r["recall@5"] for r in rb],
                 "without topic blocking": [r["recall@5"] for r in rn]},
                "fraction of all pairs solved exactly", "recall of exact top-5 CCST neighbours",
                "Candidate recall vs. exact all-pairs alignment", out / "candidate_recall.png", ylim=(0, 1.02))
    md = ["# RQ4 scaling", "", md_table(rows), "", "Candidate recall:", "", md_table(recall_rows), "",
          "Per-pair cost:", "", md_table(cost), "", f"Fitted log-log slopes: {summary}"]
    (out / "REPORT.md").write_text("\n".join(md))
    return {"rows": rows, "recall": recall_rows, "cost": cost, "summary": summary}
