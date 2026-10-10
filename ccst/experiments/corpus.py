"""Corpus-level pipeline on ~10k PubMed + arXiv abstracts (RQ5, Steps F, H, I, J).

1. topics:      LDA (collapsed Gibbs) for K in k_grid, C_v / NPMI coherence, diversity; K* = argmax C_v;
                gensim variational LDA at K* as a reference implementation
2. clustering:  LDA argmax-theta vs K-Means / HDBSCAN on SBERT and SPECTER vs BERTopic-style; NMI, ARI,
                purity, silhouette against subject labels (labels never used as model input)
3. summaries:   per-cluster labels, extractive TextRank+centroid+MMR summaries vs first-/random-sentence
                baselines; ROUGE-1/2/L and BERTScore against held-out abstracts
4. maps:        t-SNE 2D/3D on SPECTER and on theta with a perplexity sweep; trustworthiness / continuity
5. discovery:   stratified 2k subset -> claim graphs -> sparse candidates -> CCST -> zero-shot relation
                model -> signed corpus graph, relation-link map, relation heatmap, certificates,
                relation summaries, blind annotation sheet for the team
"""
from __future__ import annotations

import csv
from collections import Counter

import numpy as np

from ..certificate import build_certificate
from ..config import RELATION_LABELS, Config
from ..corpus_graph import relation_edges, signed_leiden
from ..data import arxiv, pubmed
from ..embed import get_embedder
from ..eval.metrics import clustering_metrics, trustworthiness_continuity
from ..extraction.graph_builder import GraphPipeline
from ..index.candidates import CandidateIndex
from ..relation.features import pair_features
from ..relation.guards import apply_guards
from ..summarize import (
    SummaryScorer, extractive_summary, first_sentence_baseline, random_sentence_baseline, relation_summary,
    split_sentences,
)
from ..topics import cluster as tc
from ..topics.coherence import coherence, topic_diversity
from ..topics.lda_gibbs import GibbsLDA
from ..topics.preprocess import add_phrases, build_dictionary, tokenize_corpus
from ..transport.ccst import CCSTSolver
from ..utils import DATA_DIR, RESULTS_DIR, ensure_dir, log, md_table, save_json, timer, write_jsonl
from ..viz import maps, plots
from .common import feature_matrix, featurize


def load_corpus(per_label: int | None = None) -> list[dict]:
    docs = []
    for src in (pubmed.load(), arxiv.load()):
        by: dict[str, int] = Counter()
        for d in src:
            if per_label and by[d["label"]] >= per_label:
                continue
            by[d["label"]] += 1
            docs.append({"doc_id": d["doc_id"], "source": d["source"], "label": d["label"],
                         "text": (d["title"].rstrip(".") + ". " + d["abstract"]).strip()})
    log.info(f"corpus: {len(docs)} abstracts, {len({d['label'] for d in docs})} subject labels")
    return docs


# ----------------------------------------------------------------------------- 1. topics
def topics_stage(docs: list[dict], cfg: Config, out) -> dict:
    texts = [d["text"] for d in docs]
    cache = DATA_DIR / "cache" / f"tokens_{len(texts)}.pkl"
    import pickle

    if cache.exists():
        toks = pickle.loads(cache.read_bytes())
    else:
        with timer("lemmatisation"):
            toks = add_phrases(tokenize_corpus(texts))
        cache.write_bytes(pickle.dumps(toks))
    dic, ids = build_dictionary(toks, cfg.topics.min_df, cfg.topics.max_df)
    id2w = {i: w for w, i in dic.token2id.items()}
    sweep, models = [], {}
    for K in cfg.topics.k_grid:
        with timer(f"Gibbs LDA K={K}"):
            lda = GibbsLDA(K, len(dic), alpha=cfg.topics.alpha, beta=cfg.topics.beta, iters=cfg.topics.gibbs_iters,
                           seed=cfg.seed).fit(ids)
        top = lda.top_words(id2w, 10)
        cv, _ = coherence(top, toks, dic, "c_v")
        npmi, _ = coherence(top, toks, dic, "c_npmi")
        div = topic_diversity(lda.top_words(id2w, 25))
        sweep.append({"K": K, "C_v": cv, "NPMI": npmi, "diversity": div, "perplexity": lda.perplexity()})
        models[K] = lda
        log.info(f"K={K}: C_v={cv:.3f} NPMI={npmi:.3f} diversity={div:.3f}")
    K_best = max(sweep, key=lambda r: r["C_v"])["K"]
    lda = models[K_best]
    # variational LDA reference (gensim)
    from gensim.models import LdaModel

    with timer("gensim variational LDA"):
        bow = [dic.doc2bow(t) for t in toks]
        glda = LdaModel(bow, num_topics=K_best, id2word=dic, passes=5, random_state=cfg.seed, alpha="auto")
    gtop = [[w for w, _ in glda.show_topic(k, 10)] for k in range(K_best)]
    g_cv, _ = coherence(gtop, toks, dic, "c_v")
    g_npmi, _ = coherence(gtop, toks, dic, "c_npmi")
    theta_g = np.zeros((len(bow), K_best))
    for i, b in enumerate(bow):
        for k, p in glda.get_document_topics(b, minimum_probability=0.0):
            theta_g[i, k] = p
    plots.lines([r["K"] for r in sweep], {"C_v": [r["C_v"] for r in sweep]}, "number of topics K", "C_v coherence",
                f"Choosing K by C_v coherence (K* = {K_best})", out / "coherence_cv.png")
    plots.lines([r["K"] for r in sweep], {"NPMI": [r["NPMI"] for r in sweep]}, "number of topics K", "NPMI coherence",
                "NPMI coherence across K", out / "coherence_npmi.png")
    words = [[(id2w[i], float(lda.phi[k, i])) for i in np.argsort(-lda.phi[k])[:8]] for k in range(K_best)]
    plots.topword_bars(words, f"Top words per LDA topic (K = {K_best}, collapsed Gibbs)", out / "lda_top_words.png")
    return {"sweep": sweep, "K": K_best, "theta": lda.theta, "top_words": lda.top_words(id2w, 10), "tokens": toks,
            "theta_by_K": {K: m.theta for K, m in models.items()},
            "gensim": {"C_v": g_cv, "NPMI": g_npmi, "theta": theta_g}}


# ----------------------------------------------------------------------------- 2. clustering
def clustering_stage(docs, T: dict, E_sbert, E_spec, cfg: Config, out) -> dict:
    y = [d["label"] for d in docs]
    n_lab = len(set(y))
    texts = [d["text"] for d in docs]
    mcs = max(10, min(40, len(docs) // 50))  # HDBSCAN min cluster size scales with corpus size
    res, assign = [], {}

    def add(name, lab, space):
        assign[name] = np.asarray(lab)
        res.append({"method": name, **clustering_metrics(y, lab, space)})

    add(f"LDA Gibbs argmax θ (K={T['K']})", T["theta"].argmax(1), E_spec)
    add(f"LDA variational argmax θ (K={T['K']})", T["gensim"]["theta"].argmax(1), E_spec)
    # fairness check: the LDA model whose K is closest to the number of subject labels
    K_lab = min(T["theta_by_K"], key=lambda K: abs(K - n_lab))
    if K_lab != T["K"]:
        add(f"LDA Gibbs argmax θ (K={K_lab}, closest to #labels)", T["theta_by_K"][K_lab].argmax(1), E_spec)
    with timer("embedding clustering"):
        add(f"K-Means SBERT (k={n_lab})", tc.kmeans(E_sbert, n_lab, cfg.seed), E_spec)
        if E_spec is not None:
            add(f"K-Means SPECTER (k={n_lab})", tc.kmeans(E_spec, n_lab, cfg.seed), E_spec)
            add(f"K-Means SPECTER (k={T['K']})", tc.kmeans(E_spec, T["K"], cfg.seed), E_spec)
            add("HDBSCAN SPECTER", tc.hdbscan(E_spec, min_cluster_size=mcs, seed=cfg.seed), E_spec)
        add("HDBSCAN SBERT", tc.hdbscan(E_sbert, min_cluster_size=mcs, seed=cfg.seed), E_spec)
        lab_bt, _ = tc.bertopic_style(E_spec if E_spec is not None else E_sbert, texts, min_cluster_size=mcs,
                                      seed=cfg.seed)
        add("BERTopic-style (HDBSCAN + c-TF-IDF)", lab_bt, E_spec)
    res.sort(key=lambda r: -r["NMI"])
    names = [r["method"] for r in res]
    plots.bar_ci(names, [r["NMI"] for r in res], highlight={n for n in names if n.startswith("LDA")},
                 xlabel="NMI with subject labels (labels never used as input)",
                 title="Topic and embedding clusters vs. known subjects", path=out / "clustering_nmi.png", xlim=(0, 1.05))
    return {"table": res, "assign": assign}


# ----------------------------------------------------------------------------- 3. summaries
def summary_stage(docs, assign: np.ndarray, top_words: dict, cfg: Config, out, name: str,
                  bertscore_model: str | None = "distilbert-base-uncased", holdout: float = 0.1) -> dict:
    rng = np.random.default_rng(cfg.seed)
    emb = get_embedder()
    scorer = SummaryScorer(bertscore_model)
    rows, summaries = [], {}
    systems = {"TextRank + centroid + MMR": [], "first-sentence baseline": [], "random-sentence baseline": []}
    refs_all = []
    for c in sorted(set(assign)):
        idx = np.where(assign == c)[0]
        if len(idx) < 10:
            continue
        rng.shuffle(idx)
        n_ho = max(2, int(holdout * len(idx)))
        ref_docs, src_docs = idx[:n_ho], idx[n_ho:]
        sents = [s for i in src_docs for s in split_sentences(docs[i]["text"])]
        if not sents:
            continue
        E = emb.encode(sents)
        summ = extractive_summary(sents, E, k=4, seed=cfg.seed)
        refs = [docs[i]["text"] for i in ref_docs]
        cands = {"TextRank + centroid + MMR": summ,
                 "first-sentence baseline": first_sentence_baseline([docs[i]["text"] for i in src_docs], 4, cfg.seed),
                 "random-sentence baseline": random_sentence_baseline(sents, 4, cfg.seed)}
        for sysname, s in cands.items():
            systems[sysname].append({"cluster": int(c), **scorer.rouge_scores(s, refs), "_text": " ".join(s)})
        refs_all.append(" ".join(refs[:3]))
        summaries[int(c)] = {"labels": top_words.get(int(c), []), "summary": summ, "size": int(len(idx)),
                             "dominant_label": Counter(docs[i]["label"] for i in idx).most_common(1)[0][0]}
    for sysname, lst in systems.items():
        if bertscore_model and lst:
            try:
                bs = scorer.bertscore([x["_text"] for x in lst], refs_all[: len(lst)])
            except Exception as e:  # pragma: no cover - model download issues
                log.warning(f"BERTScore unavailable: {e}")
                bs = [float("nan")] * len(lst)
        else:
            bs = [float("nan")] * len(lst)
        rows.append({"system": sysname, "rouge1": float(np.mean([x["rouge1"] for x in lst])),
                     "rouge2": float(np.mean([x["rouge2"] for x in lst])),
                     "rougeL": float(np.mean([x["rougeL"] for x in lst])),
                     "bertscore_f1": float(np.nanmean(bs)) if np.isfinite(bs).any() else float("nan"),
                     "clusters": len(lst)})
    md = [f"# Cluster summaries ({name})", ""]
    for c, s in summaries.items():
        md += [f"## Cluster {c} (n={s['size']}, mostly `{s['dominant_label']}`)", "",
               f"**Labels:** {', '.join(s['labels'])}", "", *[f"- {x}" for x in s["summary"]], ""]
    (out / f"summaries_{name}.md").write_text("\n".join(md))
    return {"scores": rows, "summaries": summaries}


# ----------------------------------------------------------------------------- 4. maps
def maps_stage(docs, E_spec, theta, assign: np.ndarray, cfg: Config, out, max_points: int = 6000) -> dict:
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(len(docs), size=min(max_points, len(docs)), replace=False))
    y = [docs[i]["label"] for i in idx]
    rows = []
    with timer("t-SNE sweep (SPECTER)"):
        Y, r = maps.perplexity_sweep(E_spec[idx], cfg.topics.perplexities, seed=cfg.seed)
    rows += [{"space": "SPECTER", **x} for x in r]
    with timer("t-SNE sweep (LDA theta)"):
        Yt, r2 = maps.perplexity_sweep(theta[idx], cfg.topics.perplexities, seed=cfg.seed)
    rows += [{"space": "LDA θ", **x} for x in r2]
    maps.side_by_side(Y, assign[idx], y, out / "tsne2d_specter_pred_vs_true.png",
                      title_pred="t-SNE (SPECTER) coloured by predicted cluster",
                      title_true="t-SNE (SPECTER) coloured by true subject label")
    maps.side_by_side(Yt, theta[idx].argmax(1), y, out / "tsne2d_lda_pred_vs_true.png",
                      title_pred="t-SNE (LDA θ) coloured by dominant topic",
                      title_true="t-SNE (LDA θ) coloured by true subject label")
    best_p = max((x for x in r), key=lambda x: x["trustworthiness"])["perplexity"]
    with timer("t-SNE 3D"):
        Y3 = maps.tsne(E_spec[idx], 3, best_p, cfg.seed)
    rows.append({"space": "SPECTER", "perplexity": best_p, "dims": 3,
                 **trustworthiness_continuity(E_spec[idx][:3000], Y3[:3000], k=10)})
    maps.map_3d(Y3, [docs[i]["source"] + ":" + docs[i]["label"] for i in idx], out / "tsne3d_specter.png",
                "3D t-SNE of the corpus (SPECTER)")
    np.save(out / "tsne2d.npy", Y)
    return {"table": rows, "idx": idx, "Y": Y}


# ----------------------------------------------------------------------------- 5. discovery
def discovery_stage(docs, theta, E_doc, topic_assign, cfg: Config, out, per_label: int = 100, n_jobs: int = 4) -> dict:
    from .real_pairs import synthetic_model

    rng = np.random.default_rng(cfg.seed)
    by: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        by.setdefault(d["label"], []).append(i)
    sub = sorted(int(i) for v in by.values() for i in rng.choice(v, size=min(per_label, len(v)), replace=False))
    ids = [docs[i]["doc_id"] for i in sub]
    texts = {docs[i]["doc_id"]: docs[i]["text"] for i in sub}
    metas = {docs[i]["doc_id"]: {"category": docs[i]["label"], "theta": theta[i]} for i in sub}
    gp = GraphPipeline(cfg.extraction)
    with timer("subset extraction + graphs"):
        claims = gp.extract(list(texts.items()))
        graphs = gp.build(claims, metas)
    idx = CandidateIndex(cfg.index, cfg.seed).fit(ids, graphs, E_doc[sub], topic_assign[sub])
    with timer("candidate generation"):
        cands = idx.candidates()
    pairs = sorted(cands["all"])
    log.info(f"{len(pairs)} candidate pairs for {len(ids)} docs (all-pairs would be {len(ids) * (len(ids) - 1) // 2})")
    with timer("CCST on candidates"):
        feats, _ = featurize([(ids[i], ids[j]) for i, j in pairs], graphs, metas, cfg.ccst, cfg.context, n_jobs=n_jobs)
    clf = synthetic_model(cfg)
    P = clf.predict_proba(feature_matrix(feats))
    conf = P.max(1)
    accepted = conf >= clf.tau
    pred = [apply_guards(RELATION_LABELS[k], f) if a else "abstain" for k, a, f in zip(P.argmax(1), accepted, feats)]
    raw = [RELATION_LABELS[k] for k in P.argmax(1)]
    guarded = np.array([p_ == "incomparable" and r != "incomparable" for p_, r in zip(pred, raw)])
    Pacc = P * accepted[:, None]
    for n in np.where(guarded)[0]:  # guarded predictions never become links
        Pacc[n, RELATION_LABELS.index(raw[n])] = 0.0
    E = relation_edges(pairs, Pacc, threshold=0.5)
    # signed mechanism families
    fam = signed_leiden(len(ids), E["pos"], E["neg"], seed=cfg.seed)
    # cluster x cluster relation heatmap (topic clusters)
    tsub = topic_assign[sub]
    K = int(tsub.max()) + 1
    heat = {lab: np.zeros((K, K)) for lab in ("analogy", "contradiction", "heterogeneity", "agreement")}
    for (i, j), p in zip(pairs, pred):
        if p in heat:
            a, b = sorted((tsub[i], tsub[j]))
            heat[p][a, b] += 1
            if a != b:
                heat[p][b, a] += 1
    # certificates for the most confident contradiction / heterogeneity / analogy edges
    solver = CCSTSolver(cfg.ccst, cfg.context)
    certs = []
    for lab in ("contradiction", "heterogeneity", "analogy"):
        k = RELATION_LABELS.index(lab)
        top = [n for n in np.argsort(-Pacc[:, k]) if pred[n] == lab][:8]
        for n in top:
            i, j = pairs[n]
            g1, g2 = graphs[ids[i]], graphs[ids[j]]
            res = solver.solve(g1, g2)
            f = pair_features(g1, g2, res, cfg.ccst, cfg.context)
            certs.append(build_certificate(g1, g2, res, f, lab, cfg.context, prob=float(P[n, k])))
    save_json(certs, out / "certificates.json")
    # relation summaries per topic cluster
    rel_sum = {}
    for c in sorted(set(tsub)):
        gs = [graphs[ids[i]] for i in range(len(ids)) if tsub[i] == c]
        counts = Counter(p for (i, j), p in zip(pairs, pred) if tsub[i] == c and tsub[j] == c)
        rel_sum[int(c)] = relation_summary(gs, top=5, pair_counts=dict(counts))
    # blind annotation sheet (no model scores) for the team's natural-pair labelling
    with open(out / "annotation_sheet.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "doc_a", "doc_b", "text_a", "text_b", "label (agreement/analogy/contradiction/"
                    "heterogeneity/incomparable/insufficient)", "population", "intervention", "comparator", "outcome",
                    "time", "environment", "design", "effect_direction", "evidence_span"])
        sample = rng.choice(len(pairs), size=min(200, len(pairs)), replace=False)
        for n in sample:
            i, j = pairs[n]
            w.writerow([f"nat{n}", ids[i], ids[j], texts[ids[i]], texts[ids[j]]] + [""] * 10)
    stats = {"docs": len(ids), "candidate_pairs": len(pairs), "all_pairs": len(ids) * (len(ids) - 1) // 2,
             "conflict_pairs": len(cands["conflict"]), "analogy_pairs": len(cands["analogy"]),
             "explore_pairs": len(cands["explore"]), "predicted": dict(Counter(pred)),
             "abstention_rate": float(1 - accepted.mean()), "families": int(len(set(fam))),
             "blocked_by_guards": dict(Counter(r for r, g in zip(raw, guarded) if g)),
             "mean_ms_per_pair": float(1000 * np.mean([f["seconds"] for f in feats]))}
    write_jsonl([{"a": ids[i], "b": ids[j], "pred": p, "p": float(c)} for (i, j), p, c in zip(pairs, pred, conf)],
                out / "candidate_predictions.jsonl")
    return {"stats": stats, "sub": sub, "ids": ids, "pairs": pairs, "edges": E, "heat": heat, "rel_summaries": rel_sum,
            "graphs": graphs}


def run(cfg: Config, per_label: int | None = None, subset_per_label: int = 100, use_specter: bool = True,
        bertscore_model: str | None = "distilbert-base-uncased", n_jobs: int = 4) -> dict:
    out = ensure_dir(RESULTS_DIR / "corpus")
    docs = load_corpus(per_label)
    texts = [d["text"] for d in docs]
    with timer("topics"):
        T = topics_stage(docs, cfg, out)
    E_sbert = get_embedder(cfg.topics.sbert_model).encode_corpus(texts, "corpus")
    E_spec = get_embedder(cfg.topics.specter_model).encode_corpus(texts, "corpus") if use_specter else E_sbert
    C = clustering_stage(docs, T, E_sbert, E_spec, cfg, out)
    lda_assign = T["theta"].argmax(1)
    emb_name = next(k for k in C["assign"] if k.startswith("K-Means SPECTER (k=" + str(T["K"]))) if use_specter \
        else next(k for k in C["assign"] if k.startswith("K-Means SBERT"))
    kw_lda = {k: w for k, w in enumerate(T["top_words"])}
    kw_emb = tc.ctfidf_keywords(texts, C["assign"][emb_name])
    with timer("summaries"):
        S_lda = summary_stage(docs, lda_assign, kw_lda, cfg, out, "lda", bertscore_model)
        S_emb = summary_stage(docs, C["assign"][emb_name], kw_emb, cfg, out, "embedding", bertscore_model)
    Mp = maps_stage(docs, E_spec, T["theta"], C["assign"][emb_name], cfg, out)
    # candidate sketches live in the node-feature (SBERT) space; SPECTER is used for the maps
    D = discovery_stage(docs, T["theta"], E_sbert, lda_assign, cfg, out, per_label=subset_per_label, n_jobs=n_jobs)
    # relation-link map on the discovery subset (own t-SNE so every linked document is placed)
    Ysub = maps.tsne(E_spec[D["sub"]], 2, 30, cfg.seed)
    top = lambda e, k=100: sorted(e, key=lambda x: -x[2])[:k]  # noqa: E731 - most confident links only
    links = {"analogy": top(D["edges"]["analogy"]), "contradiction": top(D["edges"]["contradiction"]),
             "heterogeneity": top(D["edges"]["hetero"])}
    maps.relation_links(Ysub, links, out / "tsne2d_relation_links.png",
                        "Corpus map with analogy, contradiction and (dashed) heterogeneity links")
    K = T["K"]
    tot = sum(D["heat"].values())
    plots.heatmap(tot, [f"t{k}" for k in range(K)], [f"t{k}" for k in range(K)],
                  "Relation edges between LDA topic clusters (all non-incomparable)", out / "relation_heatmap.png",
                  xlabel="topic cluster", ylabel="topic cluster")
    results = {"topics": {k: v for k, v in T.items() if k in ("sweep", "K", "top_words")} | {
                   "gensim_C_v": T["gensim"]["C_v"], "gensim_NPMI": T["gensim"]["NPMI"]},
               "clustering": C["table"], "summaries_lda": S_lda["scores"], "summaries_embedding": S_emb["scores"],
               "maps": Mp["table"], "discovery": D["stats"], "relation_summaries": D["rel_summaries"]}
    save_json(results, out / "metrics.json")
    md = ["# Corpus experiment (PubMed + arXiv)", "", f"{len(docs)} abstracts, {len(set(d['label'] for d in docs))} "
          "subject labels (used only for evaluation).", "", "## LDA K sweep", "", md_table(T["sweep"]), "",
          f"K* = {K}; gensim variational LDA at K*: C_v={T['gensim']['C_v']:.3f}, NPMI={T['gensim']['NPMI']:.3f}", "",
          "## Clustering vs subject labels", "", md_table(C["table"]), "",
          "## Summaries (ROUGE / BERTScore vs held-out abstracts)", "", "LDA clusters:", "", md_table(S_lda["scores"]),
          "", "Embedding clusters:", "", md_table(S_emb["scores"]), "", "## t-SNE faithfulness", "",
          md_table(Mp["table"]), "", "## Discovery on the stratified subset", "", md_table([D["stats"]]), "",
          "## Relation summaries (top generalised relations per topic)", ""]
    for c, r in list(D["rel_summaries"].items())[:12]:
        md += [f"- **topic {c}** ({', '.join(T['top_words'][c][:5])}): " +
               "; ".join(f"{x['relation']} [{x['support']}]" for x in r["relations"][:3]) +
               f" — pair counts {r['pair_counts']}"]
    (out / "REPORT.md").write_text("\n".join(md))
    return results
