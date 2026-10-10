"""Build ClaimGraphs directly from (node, relation, node) specs — used by tests and toy demos."""
from __future__ import annotations

import numpy as np

from .graph import REL_TO_TYPE_SIGN, RTYPES, Claim, ClaimGraph, Edge, roles_from_topology


def make_graph(doc_id: str, triples: list[tuple[str, str, str]], vecs: dict[str, np.ndarray],
               context: dict | None = None, conf: float = 0.9, epistemic: str = "observed",
               meta: dict | None = None) -> ClaimGraph:
    """triples: (subject, relation, object) with relation in RELATIONS."""
    context = context if context is not None else {}
    ctx = {h: context.get(h) for h in ("population", "dose", "time", "environment", "design")}
    keys: list[str] = []
    claims: list[Claim] = []
    for k, (s, r, o) in enumerate(triples):
        for x in (s, o):
            if x not in keys:
                keys.append(x)
        rtype, pol = REL_TO_TYPE_SIGN[r]
        claims.append(Claim(cid=f"{doc_id}:{k}", doc_id=doc_id, sent_idx=k, sentence=f"{s} {r} {o}.", span=(0, 0),
                            subj=s, verb=r, obj=o, relation=r, rtype=rtype, polarity=pol,
                            null_effect=r == "no_effect", negated=False, epistemic=epistemic, confidence=conf,
                            context=ctx, subj_key=s, obj_key=o))
    idx = {k: i for i, k in enumerate(keys)}
    edges = []
    for ci, c in enumerate(claims):
        pos = float(c.polarity > 0 and not c.null_effect)
        neg = float(c.polarity < 0)
        edges.append(Edge(u=idx[c.subj], v=idx[c.obj], claims=[ci], type_dist={c.rtype: 1.0}, pos=pos, neg=neg,
                          null=float(c.null_effect), conf=c.weight, lag=0.0))
    n = len(keys)
    node_claims = [[ci for ci, c in enumerate(claims) if k in (c.subj, c.obj)] for k in keys]
    X = np.stack([vecs[k] / np.linalg.norm(vecs[k]) for k in keys])
    mass = np.array([max(claims[ci].weight for ci in node_claims[i]) for i in range(n)])
    return ClaimGraph(doc_id=doc_id, node_keys=keys, node_text=keys, X=X, roles=roles_from_topology(n, edges),
                      mass=mass, node_claims=node_claims, edges=edges, claims=claims, rtypes=RTYPES,
                      meta=meta or {})
