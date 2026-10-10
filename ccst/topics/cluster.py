"""Embedding clustering track (proposal Sec. 4.6): K-Means, HDBSCAN, a BERTopic-style pipeline
(dimension reduction -> HDBSCAN -> class-based TF-IDF), and c-TF-IDF cluster labels."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


def kmeans(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def hdbscan(X: np.ndarray, min_cluster_size: int = 25, reduce_dim: int = 20, seed: int = 0,
            assign_noise: bool = True) -> np.ndarray:
    Z = PCA(n_components=min(reduce_dim, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X) if reduce_dim else X
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, copy=True).fit_predict(Z)
    if assign_noise and (labels >= 0).any() and (labels < 0).any():
        # attach noise points to the nearest cluster centroid so every document gets a topic
        cents = np.stack([Z[labels == c].mean(0) for c in sorted(set(labels) - {-1})])
        ids = sorted(set(labels) - {-1})
        noise = labels < 0
        labels = labels.copy()
        labels[noise] = np.array(ids)[((Z[noise, None, :] - cents[None]) ** 2).sum(-1).argmin(1)]
    return labels


def ctfidf_keywords(docs: list[str], labels: np.ndarray, topn: int = 10) -> dict[int, list[str]]:
    """Class-based TF-IDF (Grootendorst, 2022): concatenate documents per cluster, weight terms by
    in-class frequency x log(1 + mean words per class / term frequency across classes)."""
    classes = sorted(set(labels))
    joined = [" ".join(d for d, l in zip(docs, labels) if l == c) for c in classes]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=30000,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
    tf = cv.fit_transform(joined).astype(float).toarray()
    tf_norm = tf / (tf.sum(1, keepdims=True) + 1e-12)
    A = tf.sum() / len(classes)
    idf = np.log(1 + A / (tf.sum(0) + 1e-12))
    w = tf_norm * idf
    vocab = np.array(cv.get_feature_names_out())
    return {c: list(vocab[np.argsort(-w[i])[:topn]]) for i, c in enumerate(classes)}


def bertopic_style(X: np.ndarray, docs: list[str], min_cluster_size: int = 25, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Our BERTopic-style re-implementation: PCA (instead of UMAP) -> HDBSCAN -> class-based TF-IDF. Always this
    recipe, even when the `bertopic` package is installed, so results keep the same meaning; the official
    package is evaluated separately (ccst/experiments/bertopic_exp.py, item 6)."""
    labels = hdbscan(X, min_cluster_size=min_cluster_size, reduce_dim=5, seed=seed)
    return labels, ctfidf_keywords(docs, labels)
