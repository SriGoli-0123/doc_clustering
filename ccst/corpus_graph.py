"""Multi-relational corpus graph and community detection (proposal Sec. 4.8, Step H).

Each retained edge stores a calibrated relation vector r_ij = (p_agree, p_analogy, p_contra, p_hetero,
p_incomp). Agreement + analogy form positive mechanism-family links, contradictions become negative
constraints, heterogeneity links are kept as explanations (weight 0 for clustering).

* signed Leiden: leidenalg multiplex optimisation with the negative layer weighted -1 (Traag & Bruggeman)
* correlation clustering: greedy local moves maximising agreement on the signed graph, O(iters * |E|)
* baselines: unsigned Leiden (modularity) and spectral clustering
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import RELATION_LABELS


def _igraph(n: int, edges: list):
    import igraph as ig

    g = ig.Graph(n=n, edges=[(i, j) for i, j, _ in edges], directed=False)
    g.es["weight"] = [w for _, _, w in edges]
    return g


def leiden(n: int, edges: list, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    import leidenalg as la

    g = _igraph(n, edges)
    part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", seed=seed,
                             resolution_parameter=resolution)
    return np.array(part.membership)


def signed_leiden(n: int, pos: list, neg: list, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    import leidenalg as la

    gp, gn = _igraph(n, pos), _igraph(n, neg)
    opt = la.Optimiser()
    opt.set_rng_seed(seed)
    p_pos = la.RBConfigurationVertexPartition(gp, weights="weight", resolution_parameter=resolution)
    p_neg = la.RBConfigurationVertexPartition(gn, weights="weight", resolution_parameter=resolution)
    opt.optimise_partition_multiplex([p_pos, p_neg], layer_weights=[1.0, -1.0])
    return np.array(p_pos.membership)


def correlation_clustering(n: int, pos: list, neg: list, iters: int = 20, seed: int = 0) -> np.ndarray:
    """Greedy local search for max-agreement correlation clustering (Bansal et al., 2004).
    Start from singletons, then repeatedly move each node to the neighbouring cluster that maximises
    (positive weight inside) - (negative weight inside)."""
    rng = np.random.default_rng(seed)
    adj: dict[int, list] = defaultdict(list)
    for i, j, w in pos:
        adj[i].append((j, w))
        adj[j].append((i, w))
    for i, j, w in neg:
        adj[i].append((j, -w))
        adj[j].append((i, -w))
    # Net signed weight per neighbour pair (a pair may carry both a positive and a negative edge).
    net: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for v, lst in adj.items():
        for u, w in lst:
            net[v][u] += w
    lab = np.arange(n)  # singletons; positive components would merge across weak bridges
    for _ in range(iters):
        moved = 0
        for v in rng.permutation(n):
            gain: dict[int, float] = defaultdict(float)
            for u, w in net[v].items():
                gain[lab[u]] += w
            cur = gain.get(lab[v], 0.0)
            best_c, best_g = lab[v], cur
            for c, g in gain.items():
                if g > best_g + 1e-12:
                    best_c, best_g = c, g
            if best_g < 0 and cur < 0:  # isolate when every neighbouring cluster is net-negative
                best_c = n + v
            if best_c != lab[v]:
                lab[v] = best_c
                moved += 1
        if moved == 0:
            break
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def spectral(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    A = np.clip(X @ X.T, 0, None)
    return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed,
                              assign_labels="cluster_qr").fit_predict(A)


def relation_edges(pairs: list[tuple[int, int]], proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Split calibrated relation vectors into positive / negative / explanatory edge lists."""
    L = list(RELATION_LABELS)
    out = {"pos": [], "neg": [], "hetero": [], "analogy": [], "contradiction": []}
    for (i, j), p in zip(pairs, proba):
        wpos = p[L.index("agreement")] + p[L.index("analogy")]
        if wpos >= threshold:
            out["pos"].append((i, j, float(wpos)))
        if p[L.index("contradiction")] >= threshold:
            out["neg"].append((i, j, float(p[L.index("contradiction")])))
            out["contradiction"].append((i, j, float(p[L.index("contradiction")])))
        if p[L.index("heterogeneity")] >= threshold:
            out["hetero"].append((i, j, float(p[L.index("heterogeneity")])))
        if p[L.index("analogy")] >= threshold:
            out["analogy"].append((i, j, float(p[L.index("analogy")])))
    return out
