"""Reported-mechanism graph G_i = (V, E, X, R, Z, p, rho) (proposal Sec. 4.1).

Nodes are generalized concepts; edges carry typed, signed relation features
z_uv = (type, sign, lag, conf) aggregated over every claim that asserts u -> v.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Signed relation classes from the proposal and their (type, polarity) decomposition.
RELATIONS = ("increases", "decreases", "enables", "inhibits", "associated", "no_effect")
RTYPES = ("causal", "regulatory", "associative")
ROLES = ("action", "mediator", "outcome")
EPISTEMIC_FACTOR = {"observed": 1.0, "estimated": 0.9, "simulated": 0.85, "hypothesized": 0.4, "quoted": 0.3}

REL_TO_TYPE_SIGN = {
    "increases": ("causal", 1),
    "decreases": ("causal", -1),
    "enables": ("regulatory", 1),
    "inhibits": ("regulatory", -1),
    "associated": ("associative", 0),
    "no_effect": ("causal", 0),
}


@dataclass
class Claim:
    """One reported mechanism q_im with provenance (rho) and scope conditions (C, P, tau)."""

    cid: str
    doc_id: str
    sent_idx: int
    sentence: str
    span: tuple  # (char_start, char_end) in the document text
    subj: str
    verb: str
    obj: str
    relation: str  # one of RELATIONS (or a raw verb lemma in the raw-relation ablation)
    rtype: str
    polarity: int  # +1 / -1 / 0 (0 = association without direction or null effect)
    null_effect: bool
    negated: bool
    epistemic: str
    confidence: float
    context: dict = field(default_factory=dict)  # dim -> parsed value dict | None
    lag: float = 0.0
    subj_key: str = ""  # concept id after generalisation
    obj_key: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return self.confidence * EPISTEMIC_FACTOR.get(self.epistemic, 1.0)


@dataclass
class Edge:
    u: int
    v: int
    claims: list  # indices into ClaimGraph.claims
    type_dist: dict  # rtype -> probability (soft one-hot)
    pos: float  # confidence-weighted share of positive claims
    neg: float
    null: float
    conf: float
    lag: float


@dataclass
class ClaimGraph:
    doc_id: str
    node_keys: list
    node_text: list
    X: np.ndarray  # (n, d) node features
    roles: np.ndarray  # (n,) int index into ROLES
    mass: np.ndarray  # (n,) confidence-weighted node mass p_i
    node_claims: list  # node -> list[claim idx]
    edges: list
    claims: list
    rtypes: tuple = RTYPES
    meta: dict = field(default_factory=dict)
    _channels: Optional[dict] = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.node_keys)

    def channels(self, directed: bool = True) -> dict:
        """Dense edge-feature tensors used by the Gromov term (cached)."""
        key = "dir" if directed else "undir"
        if self._channels is None:
            self._channels = {}
        if key in self._channels:
            return self._channels[key]
        n, T = self.n, len(self.rtypes)
        A = np.zeros((n, n))
        Y = np.zeros((T, n, n))
        P = np.zeros((n, n))
        N = np.zeros((n, n))
        Z = np.zeros((n, n))
        Q = np.zeros((n, n))
        L = np.zeros((n, n))
        tidx = {t: i for i, t in enumerate(self.rtypes)}
        for e in self.edges:
            A[e.u, e.v] = 1.0
            for t, pr in e.type_dist.items():
                if t in tidx:
                    Y[tidx[t], e.u, e.v] = pr
            P[e.u, e.v], N[e.u, e.v], Z[e.u, e.v] = e.pos, e.neg, e.null
            Q[e.u, e.v], L[e.u, e.v] = e.conf, e.lag
        if not directed:
            sym = lambda M: np.maximum(M, M.T)  # noqa: E731
            A, P, N, Z, Q, L = (sym(M) for M in (A, P, N, Z, Q, L))
            Y = np.stack([sym(Y[t]) for t in range(T)])
        ch = {"A": A, "Y": Y, "P": P, "N": N, "Z": Z, "Q": Q, "L": L}
        self._channels[key] = ch
        return ch

    def edge_index(self) -> dict:
        return {(e.u, e.v): k for k, e in enumerate(self.edges)}

    def summary(self) -> str:
        parts = []
        for e in self.edges:
            sign = "+" if e.pos > max(e.neg, e.null) else "-" if e.neg > max(e.pos, e.null) else "0"
            parts.append(f"{self.node_text[e.u]} -[{sign}]-> {self.node_text[e.v]}")
        return "; ".join(parts)


def roles_from_topology(n: int, edges: list) -> np.ndarray:
    indeg = np.zeros(n, int)
    outdeg = np.zeros(n, int)
    for e in edges:
        outdeg[e.u] += 1
        indeg[e.v] += 1
    roles = np.ones(n, int)  # mediator by default
    roles[(indeg == 0) & (outdeg > 0)] = 0  # action / intervention
    roles[(outdeg == 0) & (indeg > 0)] = 2  # outcome
    return roles


def empty_graph(doc_id: str, dim: int, meta: dict | None = None) -> ClaimGraph:
    return ClaimGraph(
        doc_id=doc_id, node_keys=[], node_text=[], X=np.zeros((0, dim)), roles=np.zeros(0, int),
        mass=np.zeros(0), node_claims=[], edges=[], claims=[], meta=meta or {},
    )
