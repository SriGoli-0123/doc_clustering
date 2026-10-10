"""Post-classification guards (proposal Sec. 4.4 "insufficient comparability" and Sec. 7.2 "a positive
analogy must preserve a preregistered fraction of the mechanism graph and cannot rely on a loose metaphor").

A one-relation graph (A -> B) aligns perfectly with *any* other one-relation graph, so structural similarity
is uninformative; an analogy therefore needs at least ``min_edges`` aligned relations.
Measured on the controlled benchmark's out-of-fold predictions: 17 / 2,672 predictions change; analogy
precision 0.888 -> 0.920, recall 0.900 -> 0.880, macro-F1 0.936 -> 0.938. On real abstracts, where many
graphs have a single relation, it is what keeps the corpus map from filling with spurious analogy links.
"""
from __future__ import annotations

MIN_ANALOGY_EDGES = 2
MIN_CONFLICT = 0.05
MAX_ANALOGY_CONFLICT = 0.25


def apply_guards(label: str, feats: dict, min_edges: int = MIN_ANALOGY_EDGES) -> str:
    """Comparability guards applied after classification:

    1. any relation other than "incomparable" needs at least one aligned relation (Sec. 4.4: weak mechanism
       alignment means insufficient comparability) — protects against the classifier extrapolating on
       pairs where almost nothing aligns;
    2. contradiction / heterogeneity need an aligned *conflicting* relation, i.e. a non-empty certificate F
       (Sec. 2.2, stage 3: "locate an opposing relation");
    3. an analogy needs >= ``min_edges`` aligned relations (no loose metaphors) and compatible signs
       (less than ``MAX_ANALOGY_CONFLICT`` of the aligned relation mass in conflict).
    """
    n_edges = feats.get("n_aligned_edges", min_edges)
    if label != "incomparable" and n_edges < 1:
        return "incomparable"
    if label in ("contradiction", "heterogeneity") and feats.get("conflict_frac", 1.0) < MIN_CONFLICT:
        return "incomparable"
    if label == "analogy" and n_edges < min_edges:
        return "incomparable"
    if label == "analogy" and feats.get("conflict_frac", 0.0) >= MAX_ANALOGY_CONFLICT:
        return "incomparable"  # Sec. 4.4: an analogy needs *compatible relational signs*
    return label
