"""Item 6: official BERTopic baseline (docs/bertopic_protocol.md).

Same 9,486 abstracts, same subject labels (evaluation only), same metrics as the corpus experiment.
Official BERTopic defaults (UMAP -> HDBSCAN -> c-TF-IDF); only min_topic_size (= our HDBSCAN minimum
cluster size) and the UMAP random_state (= project seed) are set. Outliers are reported raw and after
BERTopic's own `reduce_outliers`.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..embed import get_embedder
from ..eval.metrics import clustering_metrics
from ..topics import cluster as tc
from ..topics.coherence import coherence, topic_diversity
from ..topics.preprocess import build_dictionary
from ..utils import RESULTS_DIR, ensure_dir, log, md_table, save_json, timer
from ..viz import plots
from .corpus import load_corpus, topics_stage


def official_bertopic(texts, E, min_topic_size: int, seed: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    hdb = HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean", cluster_selection_method="eom",
                  prediction_data=True)
    model = BERTopic(umap_model=umap_model, hdbscan_model=hdb, min_topic_size=min_topic_size,
                     calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(texts, embeddings=E)
    reduced = model.reduce_outliers(texts, topics, strategy="embeddings", embeddings=E)
    words = [[w for w, _ in model.get_topic(t)][:10] for t in sorted(set(topics)) if t != -1]
    # post hoc, representation only (protocol addendum): BERTopic's documented stop-word handling
    from sklearn.feature_extraction.text import CountVectorizer

    model.update_topics(texts, topics=list(topics), vectorizer_model=CountVectorizer(stop_words="english"))
    words_sw = [[w for w, _ in model.get_topic(t)][:10] for t in sorted(set(topics)) if t != -1]
    return np.asarray(topics), np.asarray(reduced), words, words_sw


def run(cfg: Config) -> dict:
    out = ensure_dir(RESULTS_DIR / "bertopic")
    docs = load_corpus()
    texts = [d["text"] for d in docs]
    y = [d["label"] for d in docs]
    n_lab = len(set(y))
    mcs = max(10, min(40, len(docs) // 50))
    with timer("LDA refit (same settings as the corpus experiment)"):
        T = topics_stage(docs, cfg, ensure_dir(out / "lda_refit"))
    dic, _ = build_dictionary(T["tokens"], cfg.topics.min_df, cfg.topics.max_df)
    E_sbert = get_embedder(cfg.topics.sbert_model).encode_corpus(texts, "corpus")
    E_spec = get_embedder(cfg.topics.specter_model).encode_corpus(texts, "corpus")
    rows, topic_words = [], {}

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    def add(name, labels, words=None, family=""):
        labels = np.asarray(labels)
        r = {"method": name, "family": family, **clustering_metrics(y, labels, E_spec),
             "outlier_share": float(np.mean(labels == -1))}
        if words:
            flat = [w for t in words for w in t]
            r["stopword_share"] = float(np.mean([w in ENGLISH_STOP_WORDS for w in flat]))
            r["topics_scored"] = f"{sum(sum(w in dic.token2id for w in t) >= 2 for t in words)}/{len(words)}"
            r["C_v"], _ = coherence(words, T["tokens"], dic, "c_v")
            r["NPMI"], _ = coherence(words, T["tokens"], dic, "c_npmi")
            r["diversity"] = topic_diversity(words, topn=10)
            topic_words[name] = words
        rows.append(r)
        log.info(f"{name}: NMI {r['NMI']:.3f}, clusters {r['n_clusters']}, outliers {r['outlier_share']:.1%}")

    K_lab = min(T["theta_by_K"], key=lambda K: abs(K - n_lab))
    add(f"LDA Gibbs (K* = {T['K']})", T["theta"].argmax(1), T["top_words"], "topic model")
    add(f"LDA Gibbs (K = {K_lab})", T["theta_by_K"][K_lab].argmax(1), None, "topic model")
    add(f"LDA variational (K* = {T['K']})", T["gensim"]["theta"].argmax(1), None, "topic model")
    for name, E in (("SBERT", E_sbert), ("SPECTER", E_spec)):
        lab = tc.kmeans(E, n_lab, cfg.seed)
        kw = tc.ctfidf_keywords(texts, lab)
        add(f"K-Means {name} (k = {n_lab})", lab, [kw[c] for c in sorted(kw)], "embedding clustering")
        add(f"HDBSCAN {name} (ours, PCA)", tc.hdbscan(E, min_cluster_size=mcs, seed=cfg.seed), None, "embedding clustering")
    lab_bs, kw_bs = tc.bertopic_style(E_spec, texts, min_cluster_size=mcs, seed=cfg.seed)
    add("BERTopic-style re-implementation (PCA, SPECTER)", lab_bs, [kw_bs[c] for c in sorted(kw_bs)], "BERTopic")
    for name, E in (("SBERT", E_sbert), ("SPECTER", E_spec)):
        with timer(f"official BERTopic on {name}"):
            raw, reduced, words, words_sw = official_bertopic(texts, E, mcs, cfg.seed)
        add(f"Official BERTopic, {name} (raw, −1 = outliers)", raw, words, "BERTopic")
        add(f"Official BERTopic, {name} (outliers reduced)", reduced, None, "BERTopic")
        add(f"Official BERTopic, {name} — stop-word-free topic words (post hoc)", raw, words_sw, "BERTopic")
    save_json({"rows": rows, "topic_words": topic_words, "min_topic_size": mcs}, out / "metrics.json")
    rows_sorted = sorted(rows, key=lambda r: -r["NMI"])
    plots.bar_ci([r["method"] for r in rows_sorted], [r["NMI"] for r in rows_sorted],
                 highlight={r["method"] for r in rows_sorted if r["method"].startswith("Official")},
                 xlabel="NMI with subject labels (labels never used as input)",
                 title="Item 6: official BERTopic vs LDA and embedding clustering", path=out / "bertopic_nmi.png",
                 xlim=(0, 1.05))
    md = ["# Item 6 — official BERTopic baseline", "",
          f"Protocol: `docs/bertopic_protocol.md`. {len(docs)} abstracts, {n_lab} subject labels; min_topic_size = {mcs}.",
          "", md_table(rows_sorted, ["method", "NMI", "ARI", "purity", "n_clusters", "outlier_share", "stopword_share",
                                    "topics_scored", "C_v", "NPMI", "diversity"]), "",
          "C_v / NPMI are computed on each method's top-10 topic words with the LDA tokenisation and dictionary "
          "(words absent from that dictionary are skipped)."]
    (out / "REPORT.md").write_text("\n".join(md))
    return {"rows": rows}
