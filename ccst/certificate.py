"""Minimal disagreement certificate C_ij = (A, K, F, U, P) (proposal Sec. 4.5).

A: aligned nodes/relations with transport mass          K: matched / incompatible / unknown context fields
F: smallest set of conflicting aligned relations that explains >= `coverage` of the disagreement
U: important unmatched mechanisms                       P: source passages for every listed item
Minimality is operational: relations are added in descending contribution to D_signed - D_unsigned.
"""
from __future__ import annotations



from .config import CONTEXT_DIMS, ContextConfig
from .graph import ROLES, ClaimGraph
from .transport.ccst import CCSTResult


def _edge_str(G: ClaimGraph, k: int) -> str:
    e = G.edges[k]
    rel = G.claims[max(e.claims, key=lambda c: G.claims[c].weight)].relation
    return f"{G.node_text[e.u]} --{rel}--> {G.node_text[e.v]}"


def _val(ctx: dict, h: str) -> str | None:
    v = ctx.get(h)
    return None if v is None else v.get("text")


def build_certificate(G1: ClaimGraph, G2: ClaimGraph, res: CCSTResult, feats: dict, label: str,
                      ctx_cfg: ContextConfig | None = None, coverage: float = 0.9, min_node_mass: float = 0.05,
                      prob: float | None = None) -> dict:
    ctx_cfg = ctx_cfg or ContextConfig()
    T = res.T
    ev = feats.get("_evidence", [])
    # A: aligned nodes
    aligned = []
    for u in range(G1.n):
        for v in range(G2.n):
            if T[u, v] > min_node_mass * max(G1.mass[u], 1e-9):
                aligned.append({"node1": G1.node_text[u], "node2": G2.node_text[v], "mass": float(T[u, v]),
                                "role1": ROLES[G1.roles[u]], "role2": ROLES[G2.roles[v]]})
    aligned.sort(key=lambda a: -a["mass"])
    aligned_edges = sorted(ev, key=lambda e: -e["mass"])[:10]
    A = {"nodes": aligned, "relations": [{"rel1": _edge_str(G1, e["e1"]), "rel2": _edge_str(G2, e["e2"]),
                                          "mass": e["mass"]} for e in aligned_edges]}
    # K: context status on the highest-mass aligned claim pair
    K = {}
    if ev:
        top = max(ev, key=lambda e: e["mass"])
        c1, c2 = G1.claims[top["claim1"]], G2.claims[top["claim2"]]
        comp = top["compat"]
        for h, c in zip(CONTEXT_DIMS, comp):
            a, b = _val(c1.context, h), _val(c2.context, h)
            status = "unknown" if (a is None or b is None) else "matched" if c >= 0.75 else \
                "incompatible" if c <= 0.25 else "partial"
            K[h] = {"doc1": a, "doc2": b, "compat": float(c), "status": status,
                    "material": h in ctx_cfg.material}
    # F: minimal conflicting set covering `coverage` of the disagreement
    conflicts = sorted([e for e in ev if e["contrib"] > 1e-9], key=lambda e: -e["contrib"])
    total = sum(e["contrib"] for e in conflicts)
    F, acc = [], 0.0
    for e in conflicts:
        if total <= 0 or acc >= coverage * total:
            break
        acc += e["contrib"]
        c1, c2 = G1.claims[e["claim1"]], G2.claims[e["claim2"]]
        F.append({"rel1": _edge_str(G1, e["e1"]), "rel2": _edge_str(G2, e["e2"]),
                  "kind": "sign_flip" if e["flip"] >= e["excl"] else "exclusive_assertion",
                  "contribution": e["contrib"] / total, "context_distance": e["dctx"],
                  "edge1": e["e1"], "edge2": e["e2"], "claim1": c1.cid, "claim2": c2.cid})
    # U: unmatched mechanisms
    U = []
    for G, rows, tag in ((G1, T.sum(1), "doc1"), (G2, T.sum(0), "doc2")):
        for u in range(G.n):
            left = G.mass[u] - rows[u]
            if left > 0.5 * G.mass[u] and G.mass[u] > 0.2:
                U.append({"doc": tag, "node": G.node_text[u], "unmatched_mass": float(left)})
    # P: provenance for every cited relation
    P = []
    for f in F:
        for G, cid in ((G1, f["claim1"]), (G2, f["claim2"])):
            c = next(c for c in G.claims if c.cid == cid)
            P.append({"doc": G.doc_id, "claim": cid, "sentence": c.sentence, "span": list(c.span),
                      "epistemic": c.epistemic})
    cert = {"doc1": G1.doc_id, "doc2": G2.doc_id, "label": label, "prob": prob, "A": A, "K": K, "F": F, "U": U,
            "P": P, "scores": {k: float(feats[k]) for k in ("S_mech", "S_ctx", "F_flip_frac", "F_hetero_frac",
                                                              "ctx_min_material") if k in feats}}
    cert["text"] = render_certificate(cert)
    return cert


def render_certificate(c: dict) -> str:
    """Short templated paragraph (also used as the pair-level summary, Sec. 4.9)."""
    parts = []
    matched = [h for h, v in c["K"].items() if v["status"] == "matched"]
    bad = [f"{h} ({v['doc1']} vs {v['doc2']})" for h, v in c["K"].items() if v["status"] == "incompatible"]
    unknown = [h for h, v in c["K"].items() if v["status"] == "unknown"]
    n_al = len(c["A"]["nodes"])
    parts.append(f"Verdict: {c['label'].upper()}" + (f" (p={c['prob']:.2f})" if c.get("prob") is not None else "") + ".")
    parts.append(f"{n_al} mechanism nodes align (S_mech={c['scores'].get('S_mech', 0):.2f}).")
    if matched:
        parts.append("Matched context: " + ", ".join(matched) + ".")
    if bad:
        parts.append("Materially different context: " + "; ".join(bad) + ".")
    if unknown:
        parts.append("Unknown context: " + ", ".join(unknown) + ".")
    if c["F"]:
        f = c["F"][0]
        parts.append(f"Smallest disagreement: doc1 reports '{f['rel1']}' while doc2 reports '{f['rel2']}' "
                     f"({f['kind'].replace('_', ' ')}, {100 * f['contribution']:.0f}% of the disagreement).")
    if not c["F"] and c["A"]["relations"]:
        r = c["A"]["relations"][0]
        parts.append(f"Strongest aligned relation: doc1 '{r['rel1']}' <-> doc2 '{r['rel2']}'.")
    if c["U"]:
        parts.append("Unmatched: " + ", ".join(f"{u['node']} [{u['doc']}]" for u in c["U"][:3]) + ".")
    if c["P"]:
        parts.append("Evidence: " + " | ".join(f"[{p['doc']}] \"{p['sentence'].strip()}\"" for p in c["P"][:2]))
    return " ".join(parts)


def certificate_edges(cert: dict) -> set:
    return {(f["edge1"], f["edge2"]) for f in cert["F"]}


def span_iou(a: tuple, b: tuple) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


__all__ = ["build_certificate", "render_certificate", "certificate_edges", "span_iou"]
