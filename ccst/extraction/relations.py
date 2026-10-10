"""Relation generalisation: verb lemma -> signed relation class (proposal Sec. 4.1).

A seed lexicon maps common verbs to {increases, decreases, enables, inhibits, associated,
no_effect}; unseen verbs are mapped to the nearest seed verb by embedding similarity
(lexicon expansion). Negation neutralises the sign (``did not reduce`` -> no_effect).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SEED_LEXICON: dict[str, str] = {}
for _rel, _verbs in {
    "increases": "increase improve fuel spur raise elevate boost amplify upregulate augment enhance heighten intensify accelerate "
    "worsen exacerbate aggravate stimulate double triple escalate inflate magnify strengthen expand lengthen "
    "prolong spike grow",
    "decreases": "decrease erode undermine reduce lower diminish attenuate suppress downregulate lessen curb cut decline mitigate "
    "alleviate shorten shrink halve dampen weaken depress depletes deplete slow relieve ameliorate minimize "
    "minimise contract drop impair",
    "enables": "enable foster activate trigger facilitate promote induce drive mediate initiate permit allow support "
    "potentiate catalyze catalyse recruit unlock launch cause produce generate yield lead_to result_in "
    "contribute_to give_rise_to",
    "inhibits": "inhibit hamper block prevent impede hinder halt stop abolish antagonize antagonise disrupt neutralize "
    "neutralise silence quarantine isolate contain deactivate interrupt obstruct arrest restrict limit "
    "protect_against",
    "associated": "associate_with correlate_with relate_to link_to correlate associate predict accompany alter "
    "change affect modify influence impact",
    "no_effect": "",
}.items():
    for _v in _verbs.split():
        SEED_LEXICON[_v] = _rel

# Comparative adjectives that give an association (or an effect noun) a direction.
POS_ADJ = {"higher", "increased", "greater", "elevated", "more", "larger", "raised", "enhanced", "positive",
           "improved", "better", "faster", "longer", "excess", "excessive"}
NEG_ADJ = {"lower", "reduced", "decreased", "less", "fewer", "smaller", "diminished", "negative", "inverse",
           "worse", "slower", "shorter", "protective", "attenuated"}

# Verbs that describe study procedure, not reported mechanisms.
METHOD_VERBS = {
    "conduct", "perform", "enroll", "enrol", "recruit", "administer", "apply", "study", "investigate", "examine",
    "assess", "analyze", "analyse", "use", "evaluate", "measure", "compare", "include", "randomize", "randomise",
    "aim", "propose", "present", "describe", "report", "review", "explore", "collect", "test", "design",
    "simulate", "model", "observe", "focus", "address", "consider", "provide", "develop", "introduce", "follow",
    "receive", "find", "show", "demonstrate", "suggest", "conclude", "indicate", "reveal", "identify", "be",
    "have", "do", "need", "require", "remain", "seem", "appear", "tend",
}


class RelationMapper:
    """Maps verb lemmas (with optional particle, e.g. ``lead_to``) to relation classes."""

    def __init__(self, embedder=None, expand_threshold: float = 0.55, generalize: bool = True):
        self.embedder = embedder
        self.expand_threshold = expand_threshold
        self.generalize = generalize
        self._seed_verbs = sorted(SEED_LEXICON)
        self._seed_emb = None
        self._cache: dict[str, tuple[str, float]] = {}

    def _seed_matrix(self) -> np.ndarray:
        if self._seed_emb is None:
            self._seed_emb = self.embedder.encode([v.replace("_", " ") for v in self._seed_verbs])
        return self._seed_emb

    def map(self, verb: str) -> tuple[str | None, float]:
        """Return (relation class or None, confidence)."""
        if not self.generalize:
            return verb, 0.8  # raw-relation ablation: the verb lemma itself is the edge type
        if verb == "no_effect":
            return "no_effect", 0.85
        if verb in SEED_LEXICON:
            return SEED_LEXICON[verb], 0.9
        base = verb.split("_")[0]
        if base in SEED_LEXICON and "_" not in base:
            return SEED_LEXICON[base], 0.85
        if verb in self._cache:
            return self._cache[verb]
        out: tuple[str | None, float] = (None, 0.0)
        if self.embedder is not None and base not in METHOD_VERBS:
            e = self.embedder.encode([verb.replace("_", " ")])[0]
            sims = self._seed_matrix() @ e
            j = int(np.argmax(sims))
            if sims[j] >= self.expand_threshold:
                out = (SEED_LEXICON[self._seed_verbs[j]], float(0.5 + 0.4 * sims[j]))
        self._cache[verb] = out
        return out


@lru_cache(maxsize=None)
def adjective_polarity(word: str) -> int:
    w = word.lower()
    if w in POS_ADJ:
        return 1
    if w in NEG_ADJ:
        return -1
    return 0
