# Item 6 — official BERTopic baseline (fixed before running)

Written 2026-10-10, before any BERTopic result was computed.

* **Implementation:** `bertopic` 0.17.4 with its official defaults — UMAP (n_neighbors = 15, n_components = 5,
  min_dist = 0, cosine) → HDBSCAN → class-based TF-IDF. Only two settings are fixed by us:
  `min_topic_size` = the same minimum cluster size our own HDBSCAN uses on this corpus (40), and UMAP
  `random_state` = the project seed (13) for reproducibility. No other tuning.
* **Inputs:** precomputed SBERT and SPECTER embeddings of the same 9,486 abstracts used everywhere else.
* **Outliers:** BERTopic labels some documents −1. Reported two ways: (a) raw, with −1 kept as its own group
  and the outlier share reported; (b) after BERTopic's official `reduce_outliers(strategy="embeddings")`.
* **Compared against** (same corpus, same subject labels, same metrics — NMI, ARI, purity): collapsed-Gibbs LDA
  at K* and at the K closest to the number of labels, gensim variational LDA, K-Means on SBERT / SPECTER, our
  HDBSCAN, and our earlier BERTopic-style re-implementation (PCA instead of UMAP).
* **Topic quality:** C_v and NPMI coherence of each topic model's top-10 words, computed with the same
  tokenisation and dictionary as for LDA, so "LDA topics are easier to label" can be checked directly.
* Results are reported whatever they are, including if official BERTopic beats our methods.

## Addendum (2026-10-10 11:10, after the first run — post hoc, representation only)
Official BERTopic's default topic words contain English stop-words ("the", "and", "to" …) because its default
vectoriser does no stop-word removal. Our coherence computation silently drops words absent from the LDA
dictionary, which would score BERTopic on fewer, cleaner words (optimistic). Therefore we additionally report
(a) the share of default topic words that are stop-words and the number of topics that could be scored, and
(b) BERTopic topic words re-derived with BERTopic's documented `update_topics(vectorizer_model=
CountVectorizer(stop_words="english"))`. This changes topic *words only*; cluster assignments and NMI / ARI /
purity are unchanged. Both versions are reported; this variant is labelled post hoc.
