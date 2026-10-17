# Item 6 — official BERTopic baseline

Protocol: `docs/bertopic_protocol.md`. 9986 abstracts, 20 subject labels; min_topic_size = 40.

| method | NMI | ARI | purity | n_clusters | outlier_share | stopword_share | topics_scored | C_v | NPMI | diversity |
|---|---|---|---|---|---|---|---|---|---|---|
| K-Means SBERT (k = 20) | 0.831 | 0.758 | 0.851 | 20 | 0.000 | 0.000 | 20/20 | 0.691 | 0.094 | 0.780 |
| Official BERTopic, SBERT (outliers reduced) | 0.807 | 0.731 | 0.868 | 29 | 0.000 |  |  |  |  |  |
| K-Means SPECTER (k = 20) | 0.786 | 0.687 | 0.819 | 20 | 0.000 | 0.000 | 20/20 | 0.693 | 0.098 | 0.790 |
| Official BERTopic, SBERT (raw, −1 = outliers) | 0.785 | 0.629 | 0.816 | 30 | 0.123 | 0.479 | 26/29 | 0.769 | 0.160 | 0.455 |
| Official BERTopic, SBERT — stop-word-free topic words (post hoc) | 0.785 | 0.629 | 0.816 | 30 | 0.123 | 0.000 | 29/29 | 0.704 | 0.111 | 0.759 |
| Official BERTopic, SPECTER (outliers reduced) | 0.749 | 0.615 | 0.860 | 44 | 0.000 |  |  |  |  |  |
| HDBSCAN SBERT (ours, PCA) | 0.709 | 0.231 | 0.489 | 10 | 0.000 |  |  |  |  |  |
| Official BERTopic, SPECTER (raw, −1 = outliers) | 0.695 | 0.403 | 0.762 | 45 | 0.187 | 0.357 | 42/44 | 0.744 | 0.143 | 0.480 |
| Official BERTopic, SPECTER — stop-word-free topic words (post hoc) | 0.695 | 0.403 | 0.762 | 45 | 0.187 | 0.000 | 44/44 | 0.691 | 0.105 | 0.705 |
| LDA Gibbs (K* = 30) | 0.664 | 0.571 | 0.713 | 30 | 0.000 | 0.000 | 30/30 | 0.666 | 0.114 | 0.857 |
| LDA Gibbs (K = 20) | 0.643 | 0.507 | 0.618 | 20 | 0.000 |  |  |  |  |  |
| LDA variational (K* = 30) | 0.424 | 0.206 | 0.372 | 30 | 0.000 |  |  |  |  |  |
| HDBSCAN SPECTER (ours, PCA) | 0.362 | 0.098 | 0.100 | 2 | 0.000 |  |  |  |  |  |
| BERTopic-style re-implementation (PCA, SPECTER) | 0.361 | 0.098 | 0.100 | 2 | 0.000 | 0.000 | 2/2 | 0.534 | 0.015 | 1.000 |

C_v / NPMI are computed on each method's top-10 topic words with the LDA tokenisation and dictionary (words absent from that dictionary are skipped).