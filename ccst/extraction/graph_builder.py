"""Turn extracted claims into reported-mechanism graphs, and the end-to-end text -> graph pipeline."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import ExtractionConfig
from ..embed import get_embedder
from ..graph import RTYPES, ClaimGraph, Edge, empty_graph, roles_from_topology
from ..utils import log
from .concepts import ConceptGeneralizer
from .svo import ClaimExtractor


def build_graph(doc_id: str, claims: list, gen: ConceptGeneralizer, dim: int, meta: dict | None = None,
                generalize_relations: bool = True) -> ClaimGraph:
    meta = dict(meta or {})
    claims = [c for c in claims]
    for c in claims:
        c.subj_key, c.obj_key = gen.key(c.subj), gen.key(c.obj)
    claims = [c for c in claims if c.subj_key != c.obj_key]
    if not claims:
        return empty_graph(doc_id, dim, meta)
    keys: list[str] = []
    kidx: dict[str, int] = {}
    surface: dict[str, str] = {}
    for c in claims:
        for k, txt in ((c.subj_key, c.subj), (c.obj_key, c.obj)):
            if k not in kidx:
                kidx[k] = len(keys)
                keys.append(k)
                surface[k] = txt
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ci, c in enumerate(claims):
        groups[(kidx[c.subj_key], kidx[c.obj_key])].append(ci)
    rtypes = RTYPES if generalize_relations else tuple(sorted({c.rtype for c in claims}))
    edges = []
    for (u, v), cis in groups.items():
        w = np.array([claims[i].weight for i in cis])
        ws = w.sum() + 1e-12
        tdist: dict[str, float] = defaultdict(float)
        for i, wi in zip(cis, w):
            tdist[claims[i].rtype] += wi / ws
        pos = sum(wi for i, wi in zip(cis, w) if claims[i].polarity > 0 and not claims[i].null_effect) / ws
        neg = sum(wi for i, wi in zip(cis, w) if claims[i].polarity < 0 and not claims[i].null_effect) / ws
        null = sum(wi for i, wi in zip(cis, w) if claims[i].null_effect) / ws
        conf = 1.0 - float(np.prod(1.0 - np.clip(w, 0, 0.999)))  # noisy-or over supporting claims
        lag = float((w * np.array([claims[i].lag for i in cis])).sum() / ws)
        edges.append(Edge(u=u, v=v, claims=cis, type_dist=dict(tdist), pos=pos, neg=neg, null=null, conf=conf, lag=lag))
    n = len(keys)
    node_claims: list[list[int]] = [[] for _ in range(n)]
    mass = np.zeros(n)
    for ci, c in enumerate(claims):
        for k in (kidx[c.subj_key], kidx[c.obj_key]):
            node_claims[k].append(ci)
            mass[k] = max(mass[k], c.weight)
    X = np.stack([gen.vec(k) for k in keys])
    meta.setdefault("concept_names", [gen.name(k) for k in keys])
    return ClaimGraph(doc_id=doc_id, node_keys=keys, node_text=[surface[k] for k in keys], X=X,
                      roles=roles_from_topology(n, edges), mass=mass, node_claims=node_claims, edges=edges,
                      claims=claims, rtypes=rtypes, meta=meta)


class GraphPipeline:
    """text -> claims (spaCy SVO + context) -> generalised concepts -> ClaimGraph."""

    def __init__(self, cfg: ExtractionConfig | None = None, generalize_concepts: bool | None = None,
                 generalize_relations: bool | None = None):
        self.cfg = cfg or ExtractionConfig()
        self.gen_concepts = self.cfg.generalize if generalize_concepts is None else generalize_concepts
        self.gen_relations = self.cfg.generalize if generalize_relations is None else generalize_relations
        self.embedder = get_embedder(self.cfg.embed_model)
        ecfg = ExtractionConfig(**{**self.cfg.__dict__, "generalize": self.gen_relations})
        self.extractor = ClaimExtractor(ecfg, self.embedder)
        self.generalizer: ConceptGeneralizer | None = None

    def extract(self, docs: list[tuple[str, str]]) -> dict[str, list]:
        log.info(f"extracting claims from {len(docs)} documents")
        return self.extractor.extract_docs(docs)

    def build(self, claims_by_doc: dict[str, list], metas: dict[str, dict] | None = None) -> dict[str, ClaimGraph]:
        metas = metas or {}
        phrases = [p for cl in claims_by_doc.values() for c in cl for p in (c.subj, c.obj)]
        self.generalizer = ConceptGeneralizer(self.embedder, self.cfg.concept_distance_threshold,
                                              generalize=self.gen_concepts).fit(phrases)
        log.info(f"generalised {len(set(phrases))} phrases into {len(self.generalizer.centroids)} concepts")
        dim = self.embedder.dim
        return {d: build_graph(d, cl, self.generalizer, dim, metas.get(d), self.gen_relations)
                for d, cl in claims_by_doc.items()}

    def run(self, docs: list[tuple[str, str]], metas: dict[str, dict] | None = None) -> dict[str, ClaimGraph]:
        return self.build(self.extract(docs), metas)
