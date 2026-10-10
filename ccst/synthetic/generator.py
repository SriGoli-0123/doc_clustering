"""Controlled-transformation benchmark (proposal Sec. 7.2 strata 1 and 4).

Small signed mechanism graphs are rendered as short abstracts in several domain vocabularies with a
documented node mapping. Each variant changes exactly one property, so relation labels, node
correspondences, flipped edges and evidence spans are known *by construction*:

    agreement      paraphrase (verbs/voice/order), synonym-only paraphrase, + quoted opposite background
                   claim (hard negative), + unrelated extra mechanism
    contradiction  one sign flip, one effect -> no-effect (mutually exclusive), sign flip + extra mechanism
    heterogeneity  sign flip + one material context shift (population/dose/time/environment), sign flip in a
                   simulation vs. empirical study
    analogy        same template rendered in a different domain
    incomparable   different mechanism with shared vocabulary (hard negative), sign flip with all context
                   omitted, different mechanism in a different domain

Everything is rendered to *text*, so text-only baselines (BM25, SBERT, NLI, LLM) see exactly what
CCST sees and the whole pipeline (extraction -> generalisation -> transport) is exercised.
"""
from __future__ import annotations

import itertools
import re
import zlib
from dataclasses import dataclass, field

import numpy as np

from ..compat import compat_vector
from ..config import ContextConfig
from ..extraction.context import parse_context
from ..extraction.svo import _norm
from .lexicons import (
    DESIGN_PARAPHRASE, DOMAINS, DOSE_PARAPHRASE, ENV_PARAPHRASE, NEG_VERB, NUMBER_WORDS, OPPOSITE, POPULATION_PARAPHRASE,
    POPULATION_SUBGROUP, VERBS, WRAPPERS,
)

# Benchmark v2 claim styles and rates (docs/benchmark_v2_preregistration.md)
STYLES = ("plain", "stats", "effect", "nominal", "hedged")
STYLE_P = (0.55, 0.15, 0.10, 0.10, 0.10)
UP = {"increases": True, "enables": True, "decreases": False, "inhibits": False}

MOTIFS = {
    "chain3": (3, [(0, 1), (1, 2)]),
    "chain4": (4, [(0, 1), (1, 2), (2, 3)]),
    "chain5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    "fork": (3, [(0, 1), (0, 2)]),
    "fork_med": (4, [(0, 1), (1, 2), (1, 3)]),
    "collider": (3, [(0, 2), (1, 2)]),
    "collider_chain": (4, [(0, 2), (1, 2), (2, 3)]),
    "diamond": (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
    "chain_fork": (5, [(0, 1), (1, 2), (2, 3), (2, 4)]),
}
RELS = ["increases", "decreases", "enables", "inhibits"]


@dataclass
class Template:
    tid: int
    motif: str
    n: int
    edges: list  # (i, j, relation)
    domains: list = field(default_factory=list)

    @property
    def sources(self) -> list[int]:
        return sorted({i for i, _, _ in self.edges} - {j for _, j, _ in self.edges})

    @property
    def outcomes(self) -> list[int]:
        return sorted({j for _, j, _ in self.edges} - {i for i, _, _ in self.edges})


def make_templates(n_templates: int = 50, n_domains: int = 4, seed: int = 0) -> list[Template]:
    rng = np.random.default_rng(seed)
    motifs = list(MOTIFS)
    out, seen = [], set()
    k = 0
    while len(out) < n_templates:
        motif = motifs[k % len(motifs)]
        k += 1
        n, edges = MOTIFS[motif]
        rels = tuple(rng.choice(RELS, p=[0.35, 0.35, 0.15, 0.15]) for _ in edges)
        if (motif, rels) in seen:
            continue
        seen.add((motif, rels))
        doms = list(rng.choice(list(DOMAINS), size=n_domains, replace=False))
        out.append(Template(len(out), motif, n, [(i, j, r) for (i, j), r in zip(edges, rels)], doms))
    return out


def _art(n: int) -> str:
    return "an" if str(n).startswith(("8", "11", "18")) else "a"


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


class Renderer:
    def __init__(self, seed: int = 0, ctx_cfg: ContextConfig | None = None, realism: bool = False):
        self.rng = np.random.default_rng(seed)
        self.ctx_cfg = ctx_cfg or ContextConfig()
        self.realism = realism
        self.rr = np.random.default_rng(seed + 7919)  # separate stream: v1 output is unchanged

    # ------------------------------------------------------------------ terms and contexts
    def assign_terms(self, t: Template, domain: str, avoid: set | None = None) -> dict[int, str]:
        D = DOMAINS[domain]
        rng = np.random.default_rng(zlib.crc32(f"{t.tid}-{domain}".encode()))
        inter = [x for x in D["interventions"] if x not in (avoid or set())]
        quant = [x for x in D["quantities"] if x not in (avoid or set())]
        src = t.sources
        inter_pick = list(rng.choice(inter, size=len(src), replace=False))
        others = [i for i in range(t.n) if i not in src]
        quant_pick = list(rng.choice(quant, size=len(others), replace=False))
        terms = {}
        for i, term in zip(src, inter_pick):
            terms[i] = str(term)
        for i, term in zip(others, quant_pick):
            terms[i] = str(term)
        return terms

    def base_context(self, domain: str) -> dict:
        """Scope conditions of a base document. Like real abstracts, 0-2 non-decisive fields
        (dose / time / environment) are left unstated; population and design are always stated."""
        D = DOMAINS[domain]
        n_omit = int(self.rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
        omit = sorted(self.rng.choice(["dose", "time", "env"], size=n_omit, replace=False).tolist())
        return {"population": str(self.rng.choice(D["populations"][:2])), "dose": int(self.rng.choice(D["dose"]["base"])),
                "time": str(D["times"][int(self.rng.integers(0, 2))]), "env": D["envs"][0],
                "design": str(self.rng.choice(D["designs"])), "omit": omit}

    def _paraphrase_time(self, t: str) -> str:
        n_str, unit = t.split(" ", 1)
        n = int(n_str)
        singular = unit.rstrip("s")
        conv = {"week": (7, "days"), "month": (4, "weeks"), "year": (12, "months"), "hour": (60, "minutes"),
                "day": (24, "hours"), "minute": (60, "seconds")}
        options = []
        if n in NUMBER_WORDS:
            options.append(f"{NUMBER_WORDS[n]} {unit}")
        if singular in conv:
            f, u2 = conv[singular]
            options.append(f"{n * f} {u2}")
        return str(self.rr.choice(options)) if options else t

    def context_text(self, domain: str, ctx: dict, action: str, paraphrase: bool = False) -> str:
        D = DOMAINS[domain]
        design = D["sim_design"] if ctx.get("sim") else ctx["design"]
        tmpl = D["method"]
        if paraphrase:  # same scope values, different wording (benchmark v2)
            pop = str(self.rr.choice(POPULATION_PARAPHRASE.get(ctx["population"], [ctx["population"]])))
            if not ctx.get("sim"):
                design = DESIGN_PARAPHRASE.get(design, design)
            ctx = {**ctx, "population": pop, "time": self._paraphrase_time(ctx["time"]),
                   "env": ENV_PARAPHRASE.get(ctx["env"], ctx["env"])}
            dose_fmt = DOSE_PARAPHRASE.get(domain, D["dose"]["fmt"])
        else:
            dose_fmt = D["dose"]["fmt"]
        omit = set(ctx.get("omit", []))
        if "dose" in omit:
            tmpl = re.sub(r"\s+(?:at|of) \{dose\}", "", tmpl)
        if "time" in omit:
            tmpl = re.sub(r"\s+(?:for|over) \{time\}", "", tmpl)
        if "env" in omit:
            tmpl = tmpl.replace(" {env}", "")
        return _cap(tmpl.format(design=design, population=ctx["population"], action=action,
                                dose=dose_fmt.format(n=ctx["dose"]), time=ctx["time"], env=ctx["env"]))

    def parsed(self, domain: str, ctx: dict) -> dict:
        return parse_context(self.context_text(domain, ctx, "the intervention"))

    def shift_context(self, domain: str, ctx: dict, dim: str) -> dict:
        """Change exactly one material dimension so that its compatibility with ctx is ~0."""
        D = DOMAINS[domain]
        base = self.parsed(domain, ctx)
        cands = []
        if dim == "population":
            cands = [{**ctx, "population": p} for p in D["populations"] if p != ctx["population"]]
        elif dim == "dose":
            cands = [{**ctx, "dose": ctx["dose"] * f} for f in (10, 15, 20)]
        elif dim == "time":
            cands = [{**ctx, "time": tm} for tm in D["times"] if tm != ctx["time"]]
        elif dim == "env":
            cands = [{**ctx, "env": e} for e in D["envs"] if e != ctx["env"]]
        from ..config import CONTEXT_DIMS

        h = {"env": "environment"}.get(dim, dim)
        k = CONTEXT_DIMS.index(h)
        good = [c for c in cands if compat_vector(base, self.parsed(domain, c), self.ctx_cfg)[k] <= 0.05]
        return good[int(self.rng.integers(0, len(good)))] if good else None

    # ------------------------------------------------------------------ sentences
    def claim_sentence(self, s: str, rel: str, o: str, wrap: bool = True, protect: bool = False) -> str:
        r = self.rng
        style = "plain"
        if self.realism:
            if rel.startswith("null:"):
                style = "stats" if self.rr.random() < 0.15 else "plain"
            else:
                style = str(self.rr.choice(STYLES, p=STYLE_P))
                if protect and style == "hedged":
                    style = "plain"  # the edge that defines the gold label is never hedged
        pct = int(self.rr.choice([12, 18, 23, 31, 40])) if self.realism else 0
        if style == "effect":
            core = f"{s} was associated with {_art(pct)} {pct}% {'higher' if UP[rel] else 'lower'} {o}"
            return str(r.choice(WRAPPERS)).format(s=core, S=_cap(core)) if wrap else core + "."
        if style == "nominal":
            noun = "increase" if UP[rel] else "reduction"
            if self.rr.random() < 0.5:
                return f"{_cap(_art(pct))} {pct}% {noun} in {o} was observed with {s}."
            return f"{_cap(s)} produced {_art(pct)} {pct}% {noun} in {o}."
        if style == "hedged":
            core = f"{s} {self.rr.choice(['may', 'might'])} {NEG_VERB[rel]} {o}"
            return str(r.choice(WRAPPERS)).format(s=core, S=_cap(core)) if wrap else core + "."
        if rel.startswith("null:"):
            base = rel.split(":")[1]
            form = r.integers(0, 3)
            if form == 0:
                core = f"{s} did not {NEG_VERB[base]} {o}"
            elif form == 1:
                core = f"{s} had no significant effect on {o}"
            else:
                core = f"{s} failed to {NEG_VERB[base]} {o}"
        elif r.random() < 0.3:
            core = f"{o} was {r.choice(VERBS[rel]['passive'])} by {s}"
        else:
            core = f"{s} {r.choice(VERBS[rel]['active'])} {o}"
        if style == "stats":
            core = f"{core} (p {self.rr.choice(['<', '='])} {self.rr.choice(['0.001', '0.01', '0.03', '0.04'])})"
        if not wrap:
            return core + "."
        return str(r.choice(WRAPPERS)).format(s=core, S=_cap(core))

    def render(self, doc_id: str, t: Template, domain: str, terms: dict, rels: dict, ctx: dict | None,
               extra: list | None = None, background: tuple | None = None, sim: bool = False,
               protect_edge: int | None = None) -> dict:
        """rels: edge index -> relation ('increases', ..., or 'null:<base>')."""
        D = DOMAINS[domain]
        sents: list[str] = []
        roles: list[tuple] = []  # (kind, payload)
        if ctx is not None:
            para = self.realism and (ctx.get("para") or self.rr.random() < 0.5)
            sents.append(self.context_text(domain, {**ctx, "sim": sim}, terms[t.sources[0]], paraphrase=bool(para)))
            roles.append(("method", None))
        if background is not None:
            i, j, rel = background
            sents.append(f"Previous studies suggested that {terms[i]} {VERBS[rel]['active'][0]} {terms[j]}.")
            roles.append(("background", (i, j)))
        order = list(range(len(t.edges)))
        if len(order) > 2 and self.rng.random() < 0.3:
            a = int(self.rng.integers(0, len(order) - 1))
            order[a], order[a + 1] = order[a + 1], order[a]
        first_claim = True
        for k in order:
            i, j, _ = t.edges[k]
            s = self.claim_sentence(terms[i], rels[k], terms[j], wrap=not (sim and first_claim),
                                    protect=(k == protect_edge))
            if sim and first_claim:
                s = f"{D['sim_claim']} {s}"
            first_claim = False
            sents.append(s)
            roles.append(("edge", k))
        for (es, erel, eo) in extra or []:
            sents.append(self.claim_sentence(es, erel, eo))
            roles.append(("extra", (es, eo)))
        sents.append(f"These findings inform future {D['field']} research.")
        roles.append(("closing", None))
        text, spans, pos = "", [], 0
        for s in sents:
            if text:
                text += " "
                pos += 1
            spans.append((pos, pos + len(s)))
            text += s
            pos += len(s)
        edge_spans = {payload: spans[n] for n, (kind, payload) in enumerate(roles) if kind == "edge"}
        return {"doc_id": doc_id, "text": text, "domain": domain, "category": D["category"], "template": t.tid,
                "terms": {int(k): v for k, v in terms.items()}, "rels": {int(k): v for k, v in rels.items()},
                "edge_spans": {int(k): v for k, v in edge_spans.items()},
                "context": None if ctx is None else {**ctx, "sim": sim}, "extra_terms": [x for e in (extra or []) for x in (e[0], e[2])]}


# ----------------------------------------------------------------------------- benchmark
def build_benchmark(n_templates: int = 50, seed: int = 0, realism: bool = False
                    ) -> tuple[list[dict], list[dict], list[Template]]:
    """realism=False -> benchmark v1 (clean prose); realism=True -> benchmark v2 (pre-registered realism)."""
    templates = make_templates(n_templates, seed=seed)
    R = Renderer(seed + 1, realism=realism)
    docs: list[dict] = []
    pairs: list[dict] = []
    base_docs: dict[tuple[int, str], dict] = {}

    def add(doc):
        docs.append(doc)
        return doc

    def pair(a, b, label, variant, **meta):
        pairs.append({"pair_id": f"p{len(pairs)}", "a": a["doc_id"], "b": b["doc_id"], "label": label,
                      "variant": variant, "template": a["template"], "domain_a": a["domain"], "domain_b": b["domain"],
                      **meta})

    for t in templates:
        for d in t.domains:
            terms = R.assign_terms(t, d)
            rels = {k: r for k, (_, _, r) in enumerate(t.edges)}
            ctx0 = R.base_context(d)
            # edge to perturb: prefer one that enters an outcome
            cand = [k for k, (_, j, _) in enumerate(t.edges) if j in t.outcomes]
            if realism:  # chosen first so the label-defining edge is protected from hedging in A too
                fk = int(R.rr.choice(cand))
                A = add(R.render(f"t{t.tid}_{d}_base", t, d, terms, rels, ctx0, protect_edge=fk))
            else:
                A = add(R.render(f"t{t.tid}_{d}_base", t, d, terms, rels, ctx0))
                fk = int(R.rng.choice(cand))
            base_docs[(t.tid, d)] = A
            flipped = {**rels, fk: OPPOSITE[rels[fk]]}
            nulled = {**rels, fk: f"null:{rels[fk]}"}
            jitter = {**ctx0, "dose": int(round(ctx0["dose"] * 1.2))}
            syn = {k: (DOMAINS[d]["interventions"].get(v) or DOMAINS[d]["quantities"].get(v) or [v])[0] for k, v in terms.items()}
            used = set(terms.values())
            pool_i = [x for x in DOMAINS[d]["interventions"] if x not in used]
            pool_q = [x for x in DOMAINS[d]["quantities"] if x not in used]
            extra = [(str(R.rng.choice(pool_i)), "increases", pool_q[0]), (pool_q[0], "decreases", pool_q[1])]
            i0, j0, _ = t.edges[fk]
            gold_flip = [list(t.edges[fk][:2])]

            def v(name, label, rels_b, ctx_b, terms_b=terms, **kw):
                meta = kw.pop("meta", {})
                if realism:
                    kw["protect_edge"] = fk
                B = add(R.render(f"t{t.tid}_{d}_{name}", t, d, terms_b, rels_b, ctx_b, **kw))
                pair(A, B, label, name, **meta)
                return B

            v("agree_para", "agreement", rels, jitter)
            v("agree_syn", "agreement", rels, ctx0, terms_b=syn, meta={"synonyms": True})
            v("agree_background", "agreement", rels, ctx0, background=(i0, j0, OPPOSITE[rels[fk]]),
              meta={"hard": "background"})
            v("agree_extra", "agreement", rels, ctx0, extra=extra, meta={"extra_mech": True})
            v("contra_flip", "contradiction", flipped, ctx0, meta={"flipped": gold_flip})
            v("contra_null", "contradiction", nulled, ctx0, meta={"flipped": gold_flip, "exclusive": True})
            v("contra_extra", "contradiction", flipped, ctx0, extra=extra, meta={"flipped": gold_flip, "extra_mech": True})
            dims = [x for x in ("population", "dose", "time", "env") if x not in ctx0.get("omit", [])]
            R.rng.shuffle(dims)  # the shifted dimension is always stated in both documents
            ctx_s, dim = None, None
            for dim in dims:
                ctx_s = R.shift_context(d, ctx0, dim)
                if ctx_s is not None:
                    break
            v("hetero_ctx", "heterogeneity", flipped, ctx_s,
              meta={"flipped": gold_flip, "changed_dim": {"env": "environment"}.get(dim, dim), "hard": "context_shift"})
            if realism:  # benchmark v2 hard cases
                v("contra_ctxpara", "contradiction", flipped, {**ctx0, "para": True},
                  meta={"flipped": gold_flip, "hard": "context_paraphrase"})
                nested = {**ctx0, "population": POPULATION_SUBGROUP.get(ctx0["population"], ctx0["population"])}
                v("agree_nested", "agreement", rels, nested, meta={"hard": "nested_population"})
            v("hetero_sim", "heterogeneity", flipped, ctx0, sim=True,
              meta={"flipped": gold_flip, "changed_dim": "design", "hard": "simulation"})
            v("incomp_missing", "incomparable", flipped, None, meta={"hard": "missing_context"})
            # lexically similar but structurally different: another template, shared vocabulary
            others = [o for o in templates if o.motif != t.motif and d in o.domains] or [o for o in templates if o.tid != t.tid]
            o = others[int(R.rng.integers(0, len(others)))]
            keep = {terms[t.sources[0]], terms[t.outcomes[0]]}
            o_terms = R.assign_terms(o, d, avoid=keep)
            o_terms[o.sources[0]] = terms[t.sources[0]]
            o_terms[o.outcomes[0]] = terms[t.outcomes[0]]
            o_rels = {k: r for k, (_, _, r) in enumerate(o.edges)}
            B = add(R.render(f"t{t.tid}_{d}_incomp_lex", o, d, o_terms, o_rels, ctx0))
            pair(A, B, "incomparable", "incomp_lex", hard="lexical_overlap")
        for d1, d2 in itertools.combinations(t.domains, 2):
            pair(base_docs[(t.tid, d1)], base_docs[(t.tid, d2)], "analogy", "analogy")
    # unrelated mechanism in another domain (easy negatives)
    for t in templates:
        for d in t.domains:
            o = templates[(t.tid + 7) % len(templates)]
            d2 = next((x for x in o.domains if x != d), None)
            if d2 is not None and o.motif != t.motif:
                pair(base_docs[(t.tid, d)], base_docs[(o.tid, d2)], "incomparable", "incomp_xdomain")
    return docs, pairs, templates


def slot_of(node_text: str, doc: dict) -> int | None:
    """Map an extracted node to its gold template slot (None for extra mechanisms / misses)."""
    nt = _norm(node_text)
    for k, term in doc["terms"].items():
        cands = {_norm(term)}
        D = DOMAINS[doc["domain"]]
        for bank in (D["interventions"], D["quantities"]):
            for head, syns in bank.items():
                if term == head or term in syns:
                    cands |= {_norm(head)} | {_norm(s) for s in syns}
        if any(nt == c or (len(c) > 3 and (c in nt or nt in c)) for c in cands):
            return int(k)
    return None
