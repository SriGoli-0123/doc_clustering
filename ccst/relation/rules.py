"""Ordered decision rules (proposal Sec. 4.4): contradiction -> heterogeneity -> analogy -> agreement,
otherwise insufficient comparability. Thresholds are tuned on development pairs only."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score

from ..config import RELATION_LABELS


@dataclass
class RuleThresholds:
    mech: float = 0.35  # minimum S_mech to be comparable at all
    match: float = 0.4  # minimum matched fraction
    conflict: float = 0.25  # conflicting share of aligned edge mass
    ctx_ok: float = 0.6  # worst material compatibility above which context is "compatible"
    ctx_bad: float = 0.4  # ... below which it is "materially different"
    domain: float = 0.45  # S_domain above which a match is cross-domain
    sctx: float = 0.45  # S_ctx needed to call a same-domain match "agreement"


def rule_predict(f: dict, th: RuleThresholds) -> str:
    if f["S_mech"] < th.mech or f["matched_frac"] < th.match:
        return "incomparable"
    conflict = f["conflict_frac"] >= th.conflict
    known_incompatible = f["ctx_worst_known"] >= 1 - th.ctx_bad
    if conflict and not known_incompatible and f["F_conf_compat_frac"] >= f["F_conf_unknown_frac"] \
            and f["ctx_worst_known"] <= 1 - th.ctx_ok:
        return "contradiction"
    if conflict and known_incompatible:
        return "heterogeneity"
    if f.get("S_domain", 0.0) >= th.domain and not conflict:
        return "analogy"
    if f["S_ctx"] >= th.sctx and not conflict:
        return "agreement"
    return "incomparable"


GRID = {
    "mech": [0.2, 0.3, 0.4, 0.5, 0.6],
    "match": [0.2, 0.35, 0.5, 0.65],
    "conflict": [0.1, 0.2, 0.3, 0.45],
    "ctx_ok": [0.45, 0.55, 0.65, 0.75],
    "ctx_bad": [0.2, 0.3, 0.4, 0.5],
    "domain": [0.3, 0.4, 0.5, 0.6],
    "sctx": [0.3, 0.4, 0.5, 0.6],
}


def tune_rules(feats: list[dict], labels: list[str], n_samples: int = 600, seed: int = 0) -> RuleThresholds:
    """Random search over the threshold grid maximising macro-F1 on development pairs."""
    rng = np.random.default_rng(seed)
    keys = list(GRID)
    best, best_f1 = RuleThresholds(), -1.0
    combos = list(itertools.islice(itertools.product(*[GRID[k] for k in keys]), 0, None))
    idx = rng.choice(len(combos), size=min(n_samples, len(combos)), replace=False)
    for i in idx:
        th = RuleThresholds(**dict(zip(keys, combos[i])))
        if th.ctx_bad > th.ctx_ok:
            continue
        pred = [rule_predict(f, th) for f in feats]
        s = f1_score(labels, pred, labels=list(RELATION_LABELS), average="macro", zero_division=0)
        if s > best_f1:
            best, best_f1 = th, s
    return best


def thresholds_dict(th: RuleThresholds) -> dict:
    return asdict(th)
