"""Graph-comparison baselines from proposal Sec. 7.3.

* standard FGW (Vayer et al., 2020) via POT on unsigned undirected adjacency + node features
* FNGW-style full transport with edge features (Yang et al., 2024) = CCST with partial=False and no
  context terms (implemented by configuring CCSTSolver)
* LinearFGW (Nguyen & Tsuda, 2023): barycentric embedding w.r.t. a reference graph, Euclidean distance
* graph edit distance (networkx, with relation-aware substitution costs and a timeout)
* role-constrained graph matching (Hungarian on node cost with a role mask, then edge agreement)
* context-aware classifier without structural transport (best-matching claims by text similarity)
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

from ..compat import compat_vector, context_distance, material_index
from ..config import ContextConfig
from ..graph import ClaimGraph


def _feat_cost(G1: ClaimGraph, G2: ClaimGraph) -> np.ndarray:
    return np.maximum(0.0, 2.0 - 2.0 * (G1.X @ G2.X.T))


def _unif(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def standard_fgw(G1: ClaimGraph, G2: ClaimGraph, alpha: float = 0.5) -> list[float]:
    if G1.n == 0 or G2.n == 0:
        return [4.0, 4.0]
    A1 = G1.channels(False)["A"]
    A2 = G2.channels(False)["A"]
    M = _feat_cost(G1, G2)
    d = ot.gromov.fused_gromov_wasserstein2(M, A1, A2, _unif(G1.n), _unif(G2.n), loss_fun="square_loss", alpha=alpha)
    g = ot.gromov.gromov_wasserstein2(A1, A2, _unif(G1.n), _unif(G2.n), loss_fun="square_loss")
    return [float(d), float(g)]


class LinearFGW:
    """Embed every graph once against a fixed reference graph R (r nodes):
    phi_X(G) = r * pi X  (projected features),  phi_C(G) = r^2 * pi A pi^T  (projected structure),
    where pi is the FGW plan between R and G. Pairwise distances are then O(r^2 + r d)."""

    def __init__(self, reference: ClaimGraph, alpha: float = 0.5):
        self.R, self.alpha = reference, alpha
        self.AR = reference.channels(False)["A"]

    def embed(self, G: ClaimGraph) -> np.ndarray:
        r = self.R.n
        if G.n == 0:
            return np.zeros(r * self.R.X.shape[1] + r * r)
        M = _feat_cost(self.R, G)
        A = G.channels(False)["A"]
        pi = ot.gromov.fused_gromov_wasserstein(M, self.AR, A, _unif(r), _unif(G.n), loss_fun="square_loss",
                                                alpha=self.alpha)
        phiX = r * pi @ G.X
        phiC = (r**2) * pi @ A @ pi.T
        return np.concatenate([phiX.ravel(), phiC.ravel()])


def graph_edit_distance(G1: ClaimGraph, G2: ClaimGraph, timeout: float = 0.3) -> list[float]:
    def to_nx(G):
        g = nx.DiGraph()
        for i in range(G.n):
            g.add_node(i, x=G.X[i], role=int(G.roles[i]))
        for e in G.edges:
            sign = 1 if e.pos >= max(e.neg, e.null) else -1 if e.neg >= e.null else 0
            g.add_edge(e.u, e.v, t=max(e.type_dist, key=e.type_dist.get), s=sign)
        return g

    g1, g2 = to_nx(G1), to_nx(G2)
    if G1.n == 0 or G2.n == 0:
        return [float(G1.n + G2.n + len(G1.edges) + len(G2.edges)), 1.0]
    d = nx.graph_edit_distance(
        g1, g2, timeout=timeout,
        node_subst_cost=lambda a, b: float(max(0.0, 1 - a["x"] @ b["x"])) + 0.5 * (a["role"] != b["role"]),
        node_del_cost=lambda a: 1.0, node_ins_cost=lambda a: 1.0,
        edge_subst_cost=lambda a, b: 0.5 * (a["t"] != b["t"]) + 1.0 * (a["s"] * b["s"] < 0),
        edge_del_cost=lambda a: 1.0, edge_ins_cost=lambda a: 1.0,
    )
    d = float(d) if d is not None else float(G1.n + G2.n)
    norm = G1.n + G2.n + len(G1.edges) + len(G2.edges)
    return [d, d / max(norm, 1)]


def role_constrained_matching(G1: ClaimGraph, G2: ClaimGraph) -> list[float]:
    """Hungarian matching with a role mask; features = mean node cost, preserved / flipped edge rates."""
    if G1.n == 0 or G2.n == 0:
        return [2.0, 0.0, 0.0, 0.0]
    C = _feat_cost(G1, G2) + 10.0 * (G1.roles[:, None] != G2.roles[None, :])
    r, c = linear_sum_assignment(C)
    ok = C[r, c] < 10.0
    mp = {int(i): int(j) for i, j, o in zip(r, c, ok) if o}
    e2 = {(e.u, e.v): e for e in G2.edges}
    pres = flip = 0
    for e in G1.edges:
        if e.u in mp and e.v in mp and (mp[e.u], mp[e.v]) in e2:
            pres += 1
            f = e2[(mp[e.u], mp[e.v])]
            flip += (e.pos * f.neg + e.neg * f.pos + e.null * (f.pos + f.neg) + (e.pos + e.neg) * f.null) > 0.5
    ne = max(min(len(G1.edges), len(G2.edges)), 1)
    return [float(C[r, c][ok].mean()) if ok.any() else 2.0, pres / ne, flip / ne, float(ok.mean())]


def context_without_transport(G1: ClaimGraph, G2: ClaimGraph, embedder, ccfg: ContextConfig) -> list[float]:
    """Pair each claim with its most similar claim by sentence embedding (no structure), then read
    off polarity agreement and context compatibility of those claim pairs."""
    if not G1.claims or not G2.claims:
        return [0.0, 0.0, 0.0, 0.5, 0.5]
    s1 = embedder.encode([f"{c.subj} {c.obj}" for c in G1.claims])
    s2 = embedder.encode([f"{c.subj} {c.obj}" for c in G2.claims])
    S = s1 @ s2.T
    best = S.argmax(1)
    mat = material_index(ccfg)
    flips, agrees, sims, dctx, worst = [], [], [], [], []
    for i, j in enumerate(best):
        a, b = G1.claims[i], G2.claims[j]
        sims.append(S[i, j])
        flips.append(float(a.polarity * b.polarity < 0 or a.null_effect != b.null_effect))
        agrees.append(float(a.polarity == b.polarity and a.null_effect == b.null_effect))
        cv = compat_vector(a.context, b.context, ccfg)
        dctx.append(context_distance(cv, ccfg))
        worst.append(float((1 - cv[mat]).max()))
    w = np.array(sims)
    w = np.clip(w, 0, None) + 1e-9
    return [float(np.mean(sims)), float(np.average(flips, weights=w)), float(np.average(agrees, weights=w)),
            float(np.average(dctx, weights=w)), float(np.average(worst, weights=w))]
