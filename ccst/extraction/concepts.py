"""Concept generalisation (Ceran et al. 2015, proposal Sec. 4.1).

Subject/object phrases are embedded and grouped by average-linkage agglomerative clustering
(cosine) into generalised concepts, so that "statin therapy" / "statins" become one node.
For corpora with many phrases a sparse kNN connectivity graph keeps clustering ~O(n k log n)
instead of the O(n^2) dense linkage.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

DENSE_LIMIT = 4000


class ConceptGeneralizer:
    def __init__(self, embedder, threshold: float = 0.30, generalize: bool = True):
        self.embedder = embedder
        self.threshold = threshold
        self.generalize = generalize
        self.phrase2cid: dict[str, str] = {}
        self.centroids: dict[str, np.ndarray] = {}
        self.names: dict[str, str] = {}
        self.members: dict[str, list[str]] = {}

    def fit(self, phrases: list[str]) -> "ConceptGeneralizer":
        counts = Counter(phrases)
        uniq = sorted(counts)
        if not uniq:
            return self
        E = self.embedder.encode(uniq)
        if not self.generalize or len(uniq) == 1:
            labels = np.arange(len(uniq))
        elif len(uniq) <= DENSE_LIMIT:
            labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                             distance_threshold=self.threshold).fit_predict(E)
        else:
            conn = kneighbors_graph(E, n_neighbors=15, metric="cosine", include_self=False)
            labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                             distance_threshold=self.threshold, connectivity=conn).fit_predict(E)
        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(i)
        for lab, idx in groups.items():
            cid = f"c{lab}"
            mem = [uniq[i] for i in idx]
            v = E[idx].mean(0)
            self.centroids[cid] = v / (np.linalg.norm(v) + 1e-12)
            self.names[cid] = max(mem, key=lambda p: (counts[p], -len(p)))
            self.members[cid] = mem
            for p in mem:
                self.phrase2cid[p] = cid
        return self

    def key(self, phrase: str) -> str:
        if phrase in self.phrase2cid:
            return self.phrase2cid[phrase]
        e = self.embedder.encode([phrase])[0]
        cid = None
        if self.generalize and self.centroids:
            keys = list(self.centroids)
            sims = np.stack([self.centroids[k] for k in keys]) @ e
            j = int(np.argmax(sims))
            if 1 - sims[j] <= self.threshold:
                cid = keys[j]
        if cid is None:
            cid = f"n{len(self.centroids)}_{len(self.phrase2cid)}"
            self.centroids[cid], self.names[cid], self.members[cid] = e, phrase, [phrase]
        self.phrase2cid[phrase] = cid
        return cid

    def vec(self, cid: str) -> np.ndarray:
        return self.centroids[cid]

    def name(self, cid: str) -> str:
        return self.names[cid]
