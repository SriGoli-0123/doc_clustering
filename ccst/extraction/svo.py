"""Provenance-linked subject-verb-object claim extraction (Ceran et al. style, proposal Sec. 4.1).

Handles active and passive voice (``Y was reduced by X``), negation (``did not reduce``,
``failed to reduce``), prepositional objects (``led to``, ``associated with``), effect nouns
(``had no effect on``), relative clauses (``..., which decreases Z``), coordinated verbs/objects,
quantity wrappers (``the risk of Y`` -> ``Y``) and polarity adjectives (``led to reduced Y``).
"""
from __future__ import annotations

import math
import re
from typing import Iterable

from ..config import ExtractionConfig
from ..graph import REL_TO_TYPE_SIGN, Claim
from .context import PARSERS, epistemic_status, parse_time
from .relations import METHOD_VERBS, NEG_ADJ, POS_ADJ, SEED_LEXICON, RelationMapper, adjective_polarity

PRONOUNS = {"we", "it", "this", "they", "these", "that", "which", "those", "who", "he", "she", "one", "i", "you",
            "study", "this study", "the study", "our study", "results", "findings", "our results", "data", "analysis",
            "authors", "paper", "work", "approach", "method", "model", "our findings", "the results", "evidence",
            "our model", "this paper", "this work", "the authors", "our analysis", "the present study"}
WRAPPERS = re.compile(
    r"^(?:the |a |an )?(?:risk|incidence|levels?|rates?|concentrations?|number|likelihood|probability|occurrence|"
    r"severity|frequency|amount|degree|magnitude|prevalence|odds|onset|extent|intensity|expression|activity|"
    r"production|release|formation|development|progression|use|administration|supplementation|treatment|therapy|"
    r"exposure|intake|consumption|deployment|adoption|application|introduction|presence)\s+(?:of|with|to)\s+",
    re.I,
)
LEADING = re.compile(r"^(?:the|a|an|this|these|that|those|its|their|our|such|both|all|any|each|overall|total)\s+", re.I)
EFFECT_NOUNS = {"effect", "effects", "impact", "influence", "role", "association", "relationship"}
NULL_MODS = {"no", "little", "negligible", "minimal", "insignificant", "nonsignificant", "non-significant"}
RAISING = {"find", "show", "appear", "seem", "tend", "prove", "fail", "report", "observe", "demonstrate",
           "know", "think", "believe", "expect", "predict"}
LAG_RE = re.compile(r"\b(?:after|within|following|over)\s+([\w\-.]+\s+(?:hours?|days?|weeks?|months?|years?))", re.I)
DISCOURSE = {"addition", "contrast", "turn", "particular", "summary", "conclusion", "general", "fact", "total",
             "comparison", "contrast", "short", "brief", "practice", "theory", "parallel", "sum"}
SKIP_DEPS = {"relcl", "acl", "appos", "cc", "conj", "advcl", "punct", "neg", "parataxis", "dep", "mark", "ccomp",
             "xcomp", "case", "prep"}


def _inflections(v: str) -> set[str]:
    forms = {v, v + "s", v + "d", v + "ed", v + "es"}
    if v.endswith("e"):
        forms |= {v + "s", v + "d"}
    if v.endswith("y") and v[-2] not in "aeiou":
        forms |= {v[:-1] + "ies", v[:-1] + "ied"}
    if v.endswith(("p", "t", "g")) and len(v) <= 5 and v[-2] in "aeiou" and v[-3] not in "aeiou":
        forms |= {v + v[-1] + "ed"}
    if v.endswith(("sh", "ch", "ss", "x", "z")):
        forms |= {v + "es"}
    return forms


INFLECTED = {f: v for v in SEED_LEXICON if "_" not in v for f in _inflections(v)}
EFFECT_RE = re.compile(r"^(.+?)\s+(?:had|has|have|showed|exerted)\s+no\s+(?:significant\s+|measurable\s+)?"
                       r"(?:effect|impact|influence)\s+on\s+(.+?)[.;,]?$", re.I)
_STOP_OBJ = {"in", "among", "for", "during", "over", "at", "under", "after", "within", "compared", "which", "while",
             "whereas", "but", "and", "via", "through", "across", "than", "when"}


STOP_HEADS = {"finding", "findings", "evidence", "result", "results", "data", "analysis", "analyses", "study",
              "studies", "trial", "trials", "review", "observation", "observations", "conclusion", "conclusions",
              "paper", "report", "reports", "authors", "investigators", "aim", "objective", "purpose", "question"}
DISCOURSE_START = {"but", "and", "or", "however", "thus", "therefore", "also", "moreover", "furthermore", "although",
                   "while", "whereas", "hence", "yet", "so", "then", "it", "there", "which", "who"}
_STAT = re.compile(r"confidence interval|\bci\b|\bp\s*[<=>]|odds ratio|hazard ratio|\b\d+(\.\d+)?\s*%|\bsd\b|\bse\b", re.I)


_QUANT = re.compile(r"\([^)]*\)|\[[^\]]*\]|\b(?:by|of|to)\s+\d+(?:\.\d+)?\s*%|\b(?:a|an)\s+\d+(?:\.\d+)?\s*%|"
                    r"\d+(?:\.\d+)?\s*%", re.I)


def _strip_quant(text: str) -> str:
    """Item 4: "a 23% lower LDL (95% CI 1.1-2.0)" -> "lower LDL" instead of discarding the claim."""
    return re.sub(r"\s+", " ", _QUANT.sub(" ", text)).strip()


def _valid_np(text: str) -> bool:
    """Reject statistics-only or reporting noun phrases ("87%", "the finding of", "95% CI ...")."""
    text = _strip_quant(text)
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    if not any(len(w) >= 3 for w in words):
        return False
    digits = sum(ch.isdigit() for ch in text)
    if digits > 0.3 * max(len(text.replace(" ", "")), 1) or _STAT.search(text):
        return False
    if not words:
        return False
    if words[0] in DISCOURSE_START or (len(words) <= 2 and set(words) & {"it", "there", "this", "that"}):
        return False  # "but the", "it also", "there was" are parse fragments, not concepts
    if words[-1] in STOP_HEADS:  # head noun is a reporting noun: "these findings", "the data"
        return False
    # "findings of ...", "evidence from ..." — but keep compounds such as "data exfiltration"
    return not (words[0] in STOP_HEADS and len(words) > 1 and words[1] in {"of", "from", "that", "for", "on", "in"})


NOMINAL_POL = {**{w: 1 for w in "increase increases rise elevation gain improvement surge upregulation "
                                "enhancement activation induction stimulation growth".split()},
               **{w: -1 for w in "reduction reductions decrease decreases decline fall drop loss lowering "
                                 "downregulation deterioration attenuation suppression inhibition blockade".split()}}
_NOUNS = "|".join(sorted(NOMINAL_POL, key=len, reverse=True))
_ADJS = r"(?:(?:significant|marked|small|large|modest|substantial|dramatic|further|greater|sharp|slight|"\
        r"overall|mean|relative|net)\s+)*"
_NP = r"([a-z][\w\-]*(?:\s+[a-z0-9][\w\-]*){0,5}?)"
NOMINAL_OBJ = re.compile(r"^(?:(?:a|an|the)\s+)?" + _ADJS + r"(" + _NOUNS + r")\s+(?:in|of)\s+(.+)$", re.I)
CHANGE_VERBS = {"rise": 1, "increase": 1, "grow": 1, "improve": 1, "climb": 1, "surge": 1, "rebound": 1,
                "fall": -1, "decrease": -1, "decline": -1, "drop": -1, "diminish": -1, "shrink": -1, "subside": -1}
CAUSE_PREPS = {"after", "with", "following", "upon", "among"}
_END = r"(?=\s*(?:[,.;:(]|$|\b(?:and|but|while|whereas|compared|relative|versus|vs)\b))"
NOMINAL_PATTERNS = [
    # "A 23% reduction in Y was observed with X"
    (re.compile(r"^(?:(?:a|an|the)\s+)?" + _ADJS + r"(" + _NOUNS + r")\s+(?:in|of)\s+" + _NP + r"\s+(?:was|were)\s+"
                r"(?:observed|seen|found|noted|detected|reported|documented)\s+(?:with|after|following|in response to|"
                r"among patients receiving|among those receiving)\s+" + _NP + _END, re.I), "noun-y-x"),
    # "X produced / led to / was followed by a decline in Y"
    (re.compile(_NP + r"\s+(?:produced|caused|resulted in|led to|induced|yielded|elicited|triggered|was followed by|"
                r"were followed by)\s+(?:(?:a|an|the)\s+)?" + _ADJS + r"(" + _NOUNS + r")\s+(?:in|of)\s+" + _NP + _END,
                re.I), "x-noun-y"),
    # "A significant drop in Y followed X"
    (re.compile(r"^(?:(?:a|an|the)\s+)?" + _ADJS + r"(" + _NOUNS + r")\s+(?:in|of)\s+" + _NP +
                r"\s+(?:followed|accompanied)\s+" + _NP + _END, re.I), "noun-y-x"),
    # "Y was 23% higher among those receiving X"
    (re.compile(_NP + r"\s+(?:was|were)\s+(?:significantly\s+)?(higher|lower|greater|reduced|increased|elevated|"
                r"decreased)\s+(?:among|in)\s+(?:those|patients|participants|subjects|individuals|people|women|men)\s+"
                r"(?:receiving|treated with|given|taking|exposed to|on)\s+" + _NP + _END, re.I), "y-adj-x"),
    # "No significant change in Y was observed with X"
    (re.compile(r"^no\s+(?:significant\s+|meaningful\s+|measurable\s+|appreciable\s+)?(?:change|difference|effect|"
                r"alteration)s?\s+(?:in|on|of)\s+" + _NP + r"\s+(?:was|were)\s+(?:observed|seen|found|noted|detected)\s+"
                r"(?:with|after|following|among)\s+" + _NP + _END, re.I), "null-y-x"),
    # "Y was unaffected by X"
    (re.compile(_NP + r"\s+(?:was|were|remained)\s+(?:unaffected|unchanged|unaltered)\s+(?:by|with|after|following)\s+"
                + _NP + _END, re.I), "null-y-x"),
]


def _nominal_claims(text: str) -> list[tuple]:
    """Item 4: claims expressed through event nouns or comparative adjectives. Returns the same tuple shape as
    ``_fallback``: (subj, key, obj, negated, confidence, obj_polarity, null_effect)."""
    t = text.strip().rstrip(".")
    out = []
    for rx, kind in NOMINAL_PATTERNS:
        m = rx.search(t)
        if not m:
            continue
        g = m.groups()
        if kind == "noun-y-x":
            noun, y, x = g
            out.append((x, "increase" if NOMINAL_POL[noun.lower()] > 0 else "decrease", y, False, 0.7, 0, False))
        elif kind == "x-noun-y":
            x, noun, y = g
            out.append((x, "increase" if NOMINAL_POL[noun.lower()] > 0 else "decrease", y, False, 0.7, 0, False))
        elif kind == "y-adj-x":
            y, adj, x = g
            up = adj.lower() in ("higher", "greater", "increased", "elevated")
            out.append((x, "increase" if up else "decrease", y, False, 0.7, 0, False))
        else:
            y, x = g
            out.append((x, "no_effect", y, False, 0.7, 0, True))
        break
    return out


def _verbish(np_text: str) -> bool:
    """A noun phrase that is (or ends in) an inflected relation verb signals a mis-parse."""
    words = np_text.lower().split()
    return bool(words) and (words[-1] in INFLECTED or (len(words) == 1 and words[0].rstrip("s") in INFLECTED))


def _clause_start(toks: list, end: int) -> int:
    start = 0
    for i in range(end - 1, -1, -1):
        if toks[i].lower_ in ("that", ",", ";", ":") :
            start = i + 1
            break
    return start


def _np_end(toks: list, start: int) -> int:
    for i in range(start, len(toks)):
        if toks[i].is_punct and toks[i].text not in ("-", "/") or toks[i].lower_ in _STOP_OBJ:
            return i
    return len(toks)


def _span_text(toks: list, a: int, b: int) -> str:
    return "".join(t.text_with_ws for t in toks[a:b]).strip()


def _norm(text: str) -> str:
    t = re.sub(r"\s+", " ", _strip_quant(text).strip().strip(".,;:()[]\"'")).lower()
    for _ in range(3):
        t2 = WRAPPERS.sub("", LEADING.sub("", t))
        if t2 == t:
            break
        t = t2
    return t.strip()


class ClaimExtractor:
    def __init__(self, cfg: ExtractionConfig | None = None, embedder=None):
        import spacy

        self.cfg = cfg or ExtractionConfig()
        self.nlp = spacy.load(self.cfg.spacy_model, disable=["ner"])
        self.mapper = RelationMapper(embedder, self.cfg.relation_expand_threshold, self.cfg.generalize)

    # ------------------------------------------------------------------ noun phrases
    def _phrase(self, tok, allow=("of", "with", "to")) -> tuple[str, int, int]:
        keep = {tok.i}

        def walk(t, depth):
            for c in t.children:
                if c.dep_ == "prep":
                    if c.lemma_.lower() in allow and depth < 2 and c.i > t.i:
                        pobj = [g for g in c.children if g.dep_ == "pobj"]
                        if pobj:
                            keep.add(c.i)
                            keep.add(pobj[0].i)
                            walk(pobj[0], depth + 1)
                    continue
                if c.dep_ == "punct" and c.text in ("-", "/") and abs(c.i - t.i) <= 2:
                    keep.add(c.i)  # keep intra-word hyphens: beta-blocker, low-density
                    continue
                if c.dep_ in SKIP_DEPS or (c.dep_ == "amod" and c.tag_ in ("VBP", "VBZ")):
                    continue  # finite verbs inside an NP are parser errors, not modifiers
                keep.add(c.i)
                walk(c, depth + 1)

        walk(tok, 0)
        lo, hi = min(keep), max(keep)
        span = tok.doc[lo : hi + 1]
        if hi - lo + 1 == len(keep):
            return span.text, span.start_char, span.end_char
        text = "".join(tok.doc[i].text_with_ws for i in sorted(keep))
        return text, span.start_char, span.end_char

    def _expand_conj(self, toks: list) -> list:
        out = []
        for t in toks:
            out.append(t)
            out.extend(c for c in t.conjuncts if c.pos_ in ("NOUN", "PROPN", "ADJ", "NUM"))
        return out

    def _obj_polarity(self, tok) -> tuple[int, bool]:
        """Polarity adjective on an object (``reduced Y``) and null modifiers (``no change in``)."""
        pol, null = 0, False
        for c in tok.children:
            if c.dep_ in ("amod", "advmod", "compound") and c.i < tok.i:
                p = adjective_polarity(c.text)
                pol = pol or p
            if c.dep_ == "det" and c.lower_ in NULL_MODS:
                null = True
            if c.dep_ == "amod" and c.lower_ in NULL_MODS:
                null = True
        return pol, null

    # ------------------------------------------------------------------ verbs
    def _subjects(self, verb) -> tuple[list, list]:
        subj = [c for c in verb.children if c.dep_ in ("nsubj", "csubj")]
        subjpass = [c for c in verb.children if c.dep_ in ("nsubjpass", "csubjpass")]
        if verb.dep_ == "relcl" and subj and subj[0].lower_ in ("which", "that", "who"):
            subj = [verb.head]
        if verb.dep_ == "relcl" and subjpass and subjpass[0].lower_ in ("which", "that"):
            subjpass = [verb.head]
        h = verb
        while not subj and not subjpass and h.dep_ in ("conj", "xcomp") and h.head is not h:
            h = h.head
            subj = [c for c in h.children if c.dep_ in ("nsubj", "csubj")]
            subjpass = [c for c in h.children if c.dep_ in ("nsubjpass", "csubjpass")]
            if h.dep_ == "relcl" and subj and subj[0].lower_ in ("which", "that"):
                subj = [h.head]
            if h.lemma_ not in RAISING and h.dep_ != "conj":
                break
        # raising verbs: "X was found to reduce Y" -> subject of 'found' is the logical subject
        if verb.dep_ == "xcomp" and verb.head.lemma_ in RAISING and not subj:
            subj = [c for c in verb.head.children if c.dep_ in ("nsubj", "nsubjpass")]
            subjpass = []
        return self._expand_conj(subj), self._expand_conj(subjpass)

    def _negated(self, verb) -> bool:
        neg = any(c.dep_ == "neg" for c in verb.children)
        if verb.dep_ == "xcomp" and verb.head.lemma_ in ("fail",):
            neg = not neg
        if verb.dep_ == "xcomp" and any(c.dep_ == "neg" for c in verb.head.children):
            neg = True
        if verb.dep_ == "conj" and not any(c.dep_ in ("nsubj", "nsubjpass") for c in verb.children):
            neg = neg or any(c.dep_ == "neg" for c in verb.head.children)
        return neg

    def _triplets(self, sent) -> Iterable[tuple]:
        """Yield (subj_tok, verb_key, obj_tok, negated, pattern_conf, obj_pol, null_mod)."""
        for v in sent:
            if v.pos_ not in ("VERB", "AUX") or v.dep_ in ("amod", "compound"):
                continue
            lemma = v.lemma_.lower()
            if lemma == "fail":
                continue  # handled through its xcomp complement
            subj, subjpass = self._subjects(v)
            neg = self._negated(v)
            dobj = self._expand_conj([c for c in v.children if c.dep_ in ("dobj", "obj")])
            agent = [g for c in v.children if c.dep_ == "agent" for g in c.children if g.dep_ == "pobj"]
            preps = [(c.lemma_.lower(), g) for c in v.children if c.dep_ == "prep"
                     for g in c.children if g.dep_ == "pobj" and g.lower_ not in DISCOURSE]

            # effect nouns: "X had no / a negative effect on Y"
            if lemma in ("have", "exert", "show", "produce", "play") and dobj and dobj[0].lemma_.lower() in EFFECT_NOUNS:
                eff = dobj[0]
                mods = {c.lower_ for c in eff.children if c.dep_ in ("det", "amod", "advmod")}
                targets = [g for c in eff.children if c.dep_ == "prep" and c.lemma_.lower() in ("on", "in", "against", "for")
                           for g in c.children if g.dep_ == "pobj"]
                if not targets:
                    targets = [g for (p, g) in preps if p in ("on", "in", "against")]
                if mods & NULL_MODS or neg:
                    key, pol = "no_effect", 0
                elif mods & {"positive", "stimulatory", "enhancing", "adverse", "detrimental", "harmful"}:
                    key, pol = "increase", 1
                elif mods & {"negative", "inhibitory", "suppressive", "protective"}:
                    key, pol = "decrease", -1
                else:
                    key, pol = "associate_with", 0
                for s in subj:
                    for t in self._expand_conj(targets):
                        yield s, key, t, False, 0.8, 0, key == "no_effect"
                continue

            pairs: list[tuple] = []
            if subj and dobj:
                pairs += [(s, lemma, o, 0.9) for s in subj for o in dobj]
            if subjpass and agent:  # passive with agent: "Y was reduced by X"
                pairs += [(a, lemma, s, 0.9) for a in self._expand_conj(agent) for s in subjpass]
            is_passive = v.tag_ == "VBN" and any(c.dep_ in ("auxpass", "aux") and c.lemma_ == "be" for c in v.children)
            if not dobj and not agent and is_passive:
                by = [g for (p, g) in preps if p == "by"]
                if by and (subj or subjpass):
                    pairs += [(a, lemma, o, 0.85) for a in self._expand_conj(by) for o in (subjpass or subj)]
                    preps = []
            if not dobj and not agent and not is_passive and lemma in CHANGE_VERBS:
                causes = [g for (p, g) in preps if p in CAUSE_PREPS]
                if causes and (subj or subjpass):
                    key = "increase" if CHANGE_VERBS[lemma] > 0 else "decrease"
                    pairs += [(c, key, y_, 0.8) for c in self._expand_conj(causes[:1]) for y_ in (subj or subjpass)]
                    preps = []
            if not dobj and not agent:
                for p, g in preps:
                    key = f"{lemma}_{p}"
                    if lemma in METHOD_VERBS and key not in ("associate_with", "relate_to"):
                        continue
                    rel, _ = self.mapper.map(key)
                    if rel is None:
                        continue
                    heads = subj or subjpass
                    pairs += [(s, key, o, 0.8) for s in heads for o in self._expand_conj([g])]
                    break
            for s, key, o, pc in pairs:
                pol, null = self._obj_polarity(o)
                yield s, key, o, neg, pc, pol, null

    # ------------------------------------------------------------------ fallback
    def _fallback(self, sent) -> list[tuple[str, str, str, bool, float, int, bool]]:
        """Surface-pattern extraction anchored on lexicon verbs; used only when the parse yields nothing."""
        toks = [t for t in sent]
        out = []
        if (len(toks) > 1 and toks[0].lower_ == "we" and toks[1].lemma_.lower() in METHOD_VERBS
                and "that" not in [t.lower_ for t in toks[2:5]]):
            return out  # study-procedure sentence, not a reported mechanism
        for k, t in enumerate(toks):
            lemma = INFLECTED.get(t.lower_)
            if lemma is None or k == 0 or (k + 1 < len(toks) and toks[k + 1].lower_ == "that"):
                continue  # reporting verb ("the model predicts that ...") - keep scanning
            prev = toks[k - 1].lower_
            prev2 = toks[k - 2].lower_ if k >= 2 else ""
            neg, s_end = False, k
            if prev in ("was", "were", "is", "are", "been") and k + 1 < len(toks) and toks[k + 1].lower_ == "by":
                o_txt = _span_text(toks, _clause_start(toks, k - 1), k - 1)
                s_txt = _span_text(toks, k + 2, _np_end(toks, k + 2))
                out.append((s_txt, lemma, o_txt, False, 0.7, 0, False))
                continue
            if prev in ("not", "n't") and prev2 in ("did", "does", "do"):
                neg, s_end = True, k - 2
            elif prev == "to" and prev2 in ("failed", "fails", "fail"):
                neg, s_end = True, k - 2
            s_txt = _span_text(toks, _clause_start(toks, s_end), s_end)
            o_txt = _span_text(toks, k + 1, _np_end(toks, k + 1))
            if not s_txt or not o_txt or _norm(s_txt) in PRONOUNS:
                continue
            out.append((s_txt, lemma, o_txt, neg, 0.7, 0, False))
            break
        m = EFFECT_RE.search(sent.text)
        if not out and m:
            out.append((_norm(m.group(1).split(",")[-1].split(" that ")[-1]), "no_effect", m.group(2), False, 0.7, 0, True))
        return out

    # ------------------------------------------------------------------ documents
    def extract(self, doc_id: str, text: str) -> list[Claim]:
        return self.extract_docs([(doc_id, text)])[doc_id]

    def extract_docs(self, docs: list[tuple[str, str]], batch_size: int = 64) -> dict[str, list[Claim]]:
        out: dict[str, list[Claim]] = {}
        ids = [d for d, _ in docs]
        for doc_id, sdoc in zip(ids, self.nlp.pipe((t for _, t in docs), batch_size=batch_size)):
            out[doc_id] = self._from_spacy(doc_id, sdoc)
        return out

    def _from_spacy(self, doc_id: str, sdoc) -> list[Claim]:
        sents = list(sdoc.sents)
        sent_ctx = [{h: fn(s.text) for h, fn in PARSERS.items()} for s in sents]
        epis = [epistemic_status(s.text) for s in sents]
        # Document-level scope: first value stated by a non-background sentence.
        doc_ctx: dict = {h: None for h in PARSERS}
        for ctx, ep in zip(sent_ctx, epis):
            if ep == "quoted":
                continue
            for h, val in ctx.items():
                if doc_ctx[h] is None and val is not None:
                    doc_ctx[h] = val
        claims: list[Claim] = []
        seen = set()
        for k, (s, ctx, ep) in enumerate(zip(sents, sent_ctx, epis)):
            overrides = {h: v for h, v in ctx.items() if v is not None and v != doc_ctx[h]}
            if ep == "quoted":  # background claims carry no scope unless stated in-sentence
                claim_ctx = {h: ctx[h] for h in PARSERS}
            else:
                claim_ctx = {**doc_ctx, **overrides} if overrides else doc_ctx
            lag_m = LAG_RE.search(s.text)
            lag = 0.0
            if lag_m:
                t = parse_time(lag_m.group(1))
                if t and "days" in t:
                    lag = min(1.0, math.log1p(t["days"]) / math.log1p(3650))
            trips = [(st, key, ot, neg, pc, op, nm) + self._phrase(st) + self._phrase(ot)
                     for st, key, ot, neg, pc, op, nm in self._triplets(s)]
            trips = [(key, neg, pc, op, nm, st_, ss, se, ot_, os_, oe)
                     for (_, key, _, neg, pc, op, nm, st_, ss, se, ot_, os_, oe) in trips]
            trips = [t for t in trips if not (_verbish(t[5]) or _verbish(t[8]))
                     and _norm(t[5]) not in PRONOUNS and _norm(t[8]) not in PRONOUNS
                     and _valid_np(t[5]) and _valid_np(t[8])]
            if not any((self.mapper.map(t[0])[0] is not None) for t in trips):
                extra = _nominal_claims(s.text) or self._fallback(s)
                trips += [(key, neg, pc, op, nm, st_, s.start_char, s.end_char, ot_, s.start_char, s.end_char)
                          for st_, key, ot_, neg, pc, op, nm in extra]
            trips = [t for t in trips if _valid_np(t[5]) and _valid_np(t[8])]
            for key, neg, pconf, obj_pol, null_mod, subj_txt, ss, se, obj_txt, os_, oe in trips:
                subj_n, obj_n = _norm(subj_txt), _norm(obj_txt)
                noun_pol = 0
                nm_obj = NOMINAL_OBJ.match(obj_n)
                if nm_obj:  # "led to a reduction in Y" -> Y, with the event noun's sign
                    noun_pol = NOMINAL_POL[nm_obj.group(1).lower()]
                    obj_n = _norm(nm_obj.group(2))
                # strip leading polarity adjectives ("reduced LDL" -> "ldl"); a negative subject
                # qualifier inverts the claim ("Lower X decreases Y" == "X increases Y")
                first = obj_n.split(" ", 1)
                if len(first) == 2 and first[0] in POS_ADJ | NEG_ADJ:
                    obj_n = _norm(first[1])
                subj_flip = 1
                first = subj_n.split(" ", 1)
                if len(first) == 2 and first[0] in POS_ADJ | NEG_ADJ:
                    subj_flip = -1 if first[0] in NEG_ADJ else 1
                    subj_n = _norm(first[1])
                if not subj_n or not obj_n or subj_n in PRONOUNS or obj_n in PRONOUNS or subj_n == obj_n:
                    continue
                if len(subj_n) < 2 or len(obj_n) < 2 or len(subj_n.split()) > 8 or len(obj_n.split()) > 8:
                    continue
                rel, rconf = self.mapper.map(key)
                if rel is None or (not self.mapper.generalize and key.split("_")[0] in METHOD_VERBS
                                   and key not in SEED_LEXICON):
                    continue
                if self.mapper.generalize:
                    rtype, pol = REL_TO_TYPE_SIGN[rel]
                    if noun_pol and rel != "no_effect":
                        pol = noun_pol if rel == "associated" else (pol or 1) * noun_pol
                        if rel != "associated":
                            rel, rtype = ("increases" if pol > 0 else "decreases"), "causal"
                    elif rel == "associated":
                        pol = obj_pol
                    elif obj_pol < 0 and pol != 0:
                        pol = -pol  # "led to reduced Y" == decreases Y
                        rel, rtype = ("increases" if pol > 0 else "decreases"), "causal"
                    pol *= subj_flip
                    if subj_flip < 0 and rel in ("increases", "decreases"):
                        rel = "increases" if pol > 0 else "decreases"
                    elif subj_flip < 0 and rel in ("enables", "inhibits"):
                        rel = "enables" if pol > 0 else "inhibits"
                    null = rel == "no_effect" or null_mod or neg
                    if null:
                        rel, pol = "no_effect", 0
                else:  # raw ablation: verb lemma is the type; sign unknown (only negation is visible)
                    rtype, pol, null = key, (0 if neg else 1), bool(neg)
                sig = (subj_n, rel, obj_n, k)
                if sig in seen:
                    continue
                seen.add(sig)
                claims.append(Claim(
                    cid=f"{doc_id}:{len(claims)}", doc_id=doc_id, sent_idx=k, sentence=s.text,
                    span=(s.start_char, s.end_char), subj=subj_n, verb=key, obj=obj_n, relation=rel,
                    rtype=rtype, polarity=int(pol), null_effect=bool(null), negated=bool(neg),
                    epistemic=ep if self.cfg.use_epistemic else "observed", confidence=float(rconf * pconf / 0.9),
                    context=claim_ctx, lag=lag,
                    meta={"subj_span": (ss, se), "obj_span": (os_, oe)},
                ))
        claims.sort(key=lambda c: -c.weight)
        claims = claims[: self.cfg.max_mechanisms]
        claims.sort(key=lambda c: (c.sent_idx, c.cid))
        for i, c in enumerate(claims):
            c.cid = f"{doc_id}:{i}"
        return claims
