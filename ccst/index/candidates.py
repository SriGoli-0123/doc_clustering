"""Sparse candidate generation (proposal Sec. 4.6, Step F).

Three cheap sketches per document:
  z_sem  : node text embeddings pooled by role (action | mediator | outcome)
  z_mech : unsigned typed motif counts, role-conditioned degrees, short signed-path counts,
           Laplacian spectrum, graph size, mechanism count
  z_ctx  : typed scope fields + missingness pattern
Separate nearest-neighbour indexes return N_sem(i), N_mech(i), N_ctx(i).
  E_conflict(i) = candidates that share i's topic cluster (contradiction / heterogeneity / agreement)
  E_analogy(i)  = mechanism neighbours in *different* topic clusters (deliberately semantically distant)
  E_explore(i)  = a few random documents to estimate missed-neighbour risk
Cost: sketches O(N * M_max^2); kNN queries O(N log N) with tree indexes on low-dimensional sketches
(brute force O(N^2 d) fallback for the high-dimensional semantic sketch); then |E_cand| = O(N k)
exact CCST problems instead of N(N-1)/2.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..config import IndexConfig
from ..graph import ClaimGraph

POP_KINDS = ["human", "animal", "cell", "system", "econ", "eco"]
AGE = ["infant", "child", "adolescent", "adult", "elderly"]
ENV = ["in_vitro", "in_silico", "in_vivo", "clinical", "community", "laboratory", "field", "production", "cloud",
       "urban", "rural", "marine", "freshwater", "recession", "expansion"]
DESIGN = ["meta", "simulation", "rct", "cohort", "case_control", "cross_sectional", "experiment", "observational"]


def semantic_sketch(G: ClaimGraph, doc_emb: np.ndarray) -> np.ndarray:
    """Node text pooled by role; a missing role falls back to the document embedding when it lives in
    the same space as the node features, otherwise to the mean of all nodes (or zeros)."""
    dim = G.X.shape[1]
    if doc_emb is not None and doc_emb.shape[-1] == dim:
        fallback = doc_emb
    elif G.n:
        fallback = G.X.mean(0)
    else:
        fallback = np.zeros(dim)
    parts = []
    for role in (0, 1, 2):
        idx = np.where(G.roles == role)[0] if G.n else []
        parts.append(G.X[idx].mean(0) if len(idx) else fallback)
    v = np.concatenate(parts)
    return v / (np.linalg.norm(v) + 1e-12)


def mechanism_sketch(G: ClaimGraph, n_eig: int = 4) -> np.ndarray:
    n = G.n
    if n == 0:
        return np.zeros(26)
    ch = G.channels(True)
    A, P, N = ch["A"], ch["P"], ch["N"]
    out_deg, in_deg = A.sum(1), A.sum(0)
    A2 = A @ A
    chains = A2.sum()
    forks = float(((out_deg * (out_deg - 1)) / 2).sum())
    colliders = float(((in_deg * (in_deg - 1)) / 2).sum())
    pp, pn, np_, nn = (P @ P).sum(), (P @ N).sum(), (N @ P).sum(), (N @ N).sum()
    types = [sum(e.type_dist.get(t, 0.0) for e in G.edges) for t in G.rtypes[:3]] + [0.0] * (3 - len(G.rtypes[:3]))
    roles = [float((G.roles == r).sum()) for r in (0, 1, 2)]
    Au = np.maximum(A, A.T)
    d = Au.sum(1)
    with np.errstate(divide="ignore"):
        dinv = np.where(d > 0, 1 / np.sqrt(d), 0)
    L = np.eye(n) - dinv[:, None] * Au * dinv[None, :]
    eig = np.sort(np.linalg.eigvalsh(L))[::-1][:n_eig]
    eig = np.pad(eig, (0, n_eig - len(eig)))
    deg_role = [out_deg[G.roles == 0].mean() if roles[0] else 0.0, in_deg[G.roles == 2].mean() if roles[2] else 0.0,
                (in_deg + out_deg)[G.roles == 1].mean() if roles[1] else 0.0]
    nulls = sum(e.null for e in G.edges)
    v = np.array([n, len(G.edges), len(G.claims), *roles, chains, forks, colliders, pp, pn, np_, nn, *types, nulls,
                  *deg_role, *eig, float(np.trace(A2)), float(A.sum() / max(n * (n - 1), 1))], float)
    return np.log1p(np.abs(v)) * np.sign(v)


def context_sketch(G: ClaimGraph) -> np.ndarray:
    ctx = G.claims[0].context if G.claims else {}
    v: list[float] = []
    pop = ctx.get("population")
    v += [float(pop is not None and pop.get("kind") == k) for k in POP_KINDS]
    v += [float(pop is not None and pop.get("quals", {}).get("age") == a) for a in AGE]
    dose = ctx.get("dose")
    v += [float(dose is not None), float(np.log10(dose["num"] + 1) / 4) if dose and "num" in dose else 0.0]
    tm = ctx.get("time")
    v += [float(tm is not None and tm.get("horizon") == 0), float(tm is not None and tm.get("horizon") == 1)]
    env = ctx.get("environment")
    v += [float(env is not None and env.get("cat") == e) for e in ENV]
    des = ctx.get("design")
    v += [float(des is not None and des.get("cat") == d) for d in DESIGN]
    v += [float(ctx.get(h) is None) for h in ("population", "dose", "time", "environment", "design")]
    return np.array(v)


class CandidateIndex:
    def __init__(self, cfg: IndexConfig | None = None, seed: int = 0):
        self.cfg = cfg or IndexConfig()
        self.seed = seed

    def fit(self, ids: list[str], graphs: dict[str, ClaimGraph], doc_emb: np.ndarray, topics: np.ndarray):
        self.ids = ids
        self.topics = np.asarray(topics)
        self.Z_sem = np.stack([semantic_sketch(graphs[d], doc_emb[i]) for i, d in enumerate(ids)])
        M = np.stack([mechanism_sketch(graphs[d]) for d in ids])
        self.Z_mech = (M - M.mean(0)) / (M.std(0) + 1e-9)
        self.Z_ctx = np.stack([context_sketch(graphs[d]) for d in ids])
        self.has_mech = np.array([graphs[d].n > 0 for d in ids])
        return self

    def _knn(self, Z: np.ndarray, k: int, metric: str) -> np.ndarray:
        k = min(k + 1, len(Z))
        algo = "brute" if (metric == "cosine" or Z.shape[1] > 30) else "ball_tree"
        nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algo).fit(Z)
        return nn.kneighbors(Z, return_distance=False)

    def candidates(self) -> dict[str, set]:
        c = self.cfg
        n = len(self.ids)
        rng = np.random.default_rng(self.seed)
        n_sem = self._knn(self.Z_sem, c.k_sem, "cosine")
        n_mech = self._knn(self.Z_mech, c.k_mech * 2, "euclidean")
        n_ctx = self._knn(self.Z_ctx, c.k_ctx, "euclidean")
        conflict, analogy, explore = set(), set(), set()
        for i in range(n):
            if not self.has_mech[i]:
                continue
            same = set(n_sem[i, 1:]) | set(n_mech[i, 1:c.k_mech + 1]) | set(n_ctx[i, 1:])
            for j in same:
                if j != i and self.has_mech[j] and (not c.topic_blocking or self.topics[i] == self.topics[j]):
                    conflict.add((min(i, j), max(i, j)))
            k_an = 0
            for j in n_mech[i, 1:]:
                if j != i and self.has_mech[j] and self.topics[i] != self.topics[j]:
                    analogy.add((min(i, j), max(i, j)))
                    k_an += 1
                    if k_an >= c.k_mech:
                        break
            for j in rng.choice(n, size=min(c.explore, n - 1), replace=False):
                if j != i and self.has_mech[j]:
                    explore.add((min(i, j), max(i, j)))
        return {"conflict": conflict, "analogy": analogy, "explore": explore,
                "all": conflict | analogy | explore}


def candidate_recall(cand: set, exact_scores: dict[tuple, float], ids_subset: list[int], k: int = 5) -> float:
    """Share of each document's exact top-k CCST neighbours (within a sampled subset) that the sparse
    candidate set contains."""
    hits, tot = 0, 0
    for i in ids_subset:
        nb = sorted(((s, j) for (a, j), s in exact_scores.items() if a == i), reverse=True)[:k]
        for _, j in nb:
            tot += 1
            hits += (min(i, j), max(i, j)) in cand
    return hits / max(tot, 1)


# ----------------------------------------------------------------------------- item 5: improved index
class ConceptCodebook:
    """K-means codebook over node embeddings (fitted on training documents only). A document becomes a
    mass-weighted, role-split TF-IDF histogram over codewords: which concepts, in which roles, it contains.
    Cosine similarity of these histograms approximates whether two documents can be aligned node by node,
    the dominant term of D_unsigned, better than mean-pooled node text."""

    def __init__(self, k: int = 256, seed: int = 0):
        self.k, self.seed = k, seed

    def fit(self, graphs: list[ClaimGraph]) -> "ConceptCodebook":
        from sklearn.cluster import MiniBatchKMeans

        X = np.concatenate([G.X for G in graphs if G.n])
        self.km = MiniBatchKMeans(n_clusters=min(self.k, len(X)), random_state=self.seed, n_init=3,
                                  batch_size=1024).fit(X)
        df = np.zeros(self.km.n_clusters)
        for G in graphs:
            if G.n:
                df[np.unique(self.km.predict(G.X))] += 1
        self.idf = np.log((1 + len(graphs)) / (1 + df)) + 1
        return self

    def sketch(self, G: ClaimGraph) -> np.ndarray:
        K = self.km.n_clusters
        h = np.zeros(3 * K)
        if G.n:
            codes = self.km.predict(G.X)
            for c, r, m in zip(codes, G.roles, G.mass):
                h[int(r) * K + int(c)] += m * self.idf[c]
        return h / (np.linalg.norm(h) + 1e-12)


def _rownorm(Z: np.ndarray) -> np.ndarray:
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


class CombinedIndex:
    """One cosine kNN over weighted, L2-normalised sketches (semantic, concept histogram, mechanism, context):
    similarity = sum_i w_i cos_i. Weights are learned from exact CCST neighbours on training documents."""

    BLOCKS = ("sem", "concept", "mech", "ctx")

    def __init__(self, codebook: ConceptCodebook, weights: dict | None = None, explore: int = 2, seed: int = 0):
        self.codebook, self.explore, self.seed = codebook, explore, seed
        self.weights = weights or {b: 1.0 for b in self.BLOCKS}

    def fit(self, ids: list[str], graphs: dict[str, ClaimGraph], doc_emb: np.ndarray) -> "CombinedIndex":
        self.ids = ids
        self.has_mech = np.array([graphs[d].n > 0 for d in ids])
        M = np.stack([mechanism_sketch(graphs[d]) for d in ids])
        self.blocks = {
            "sem": _rownorm(np.stack([semantic_sketch(graphs[d], doc_emb[i]) for i, d in enumerate(ids)])),
            "concept": _rownorm(np.stack([self.codebook.sketch(graphs[d]) for d in ids])),
            "mech": _rownorm((M - M.mean(0)) / (M.std(0) + 1e-9)),
            "ctx": _rownorm(np.stack([context_sketch(graphs[d]) for d in ids])),
        }
        return self

    def similarity(self, weights: dict | None = None) -> np.ndarray:
        w = weights or self.weights
        S = sum(w[b] * (self.blocks[b] @ self.blocks[b].T) for b in self.BLOCKS if w.get(b, 0) > 0)
        S = np.asarray(S, float)
        np.fill_diagonal(S, -np.inf)
        S[~self.has_mech, :] = -np.inf
        S[:, ~self.has_mech] = -np.inf
        return S

    def candidates(self, k: int, weights: dict | None = None) -> set:
        S = self.similarity(weights)
        rng = np.random.default_rng(self.seed)
        n = len(self.ids)
        out = set()
        top = np.argsort(-S, axis=1)[:, :k]
        for i in range(n):
            if not self.has_mech[i]:
                continue
            for j in top[i]:
                if np.isfinite(S[i, j]):
                    out.add((min(i, int(j)), max(i, int(j))))
            for j in rng.choice(n, size=min(self.explore, n - 1), replace=False):
                if j != i and self.has_mech[j]:
                    out.add((min(i, int(j)), max(i, int(j))))
        return out
