"""Context compatibility c_h(i,j) in [0,1] and context distance D_ctx (proposal Sec. 4.2).

1 = compatible, 0 = materially incompatible, intermediate = nested / partially overlapping /
uncertain. Missing context is *unknown* (0.5) and is never silently treated as a match.
"""
from __future__ import annotations

import math

import numpy as np

from .config import CONTEXT_DIMS, ContextConfig

DESIGN_COMPAT = {
    frozenset({"rct", "meta"}): 0.8,
    frozenset({"rct", "cohort"}): 0.6,
    frozenset({"rct", "case_control"}): 0.5,
    frozenset({"rct", "cross_sectional"}): 0.5,
    frozenset({"rct", "experiment"}): 0.8,
    frozenset({"cohort", "case_control"}): 0.7,
    frozenset({"cohort", "cross_sectional"}): 0.7,
    frozenset({"cohort", "observational"}): 0.8,
    frozenset({"experiment", "observational"}): 0.6,
    frozenset({"meta", "cohort"}): 0.7,
}
ENV_COMPAT = {
    frozenset({"clinical", "community"}): 0.5,
    frozenset({"laboratory", "in_vitro"}): 0.7,
    frozenset({"production", "cloud"}): 0.7,
    frozenset({"marine", "freshwater"}): 0.2,
}


def _numeric(a: float, b: float, sigma: float) -> float:
    lr = abs(math.log(max(a, 1e-12) / max(b, 1e-12)))
    return math.exp(-(lr**2) / (2 * sigma**2))


def compat_population(a: dict, b: dict, cfg: ContextConfig) -> float:
    if a["kind"] != b["kind"]:
        return 0.0
    qa, qb = a.get("quals", {}), b.get("quals", {})
    shared = set(qa) & set(qb)
    if any(qa[g] != qb[g] for g in shared):
        return 0.0  # e.g. infants vs adults, mice vs rats, enterprise vs home networks
    only_one = set(qa) ^ set(qb)
    if shared and not only_one:
        return 1.0
    ta, tb = set(a.get("tokens", [])), set(b.get("tokens", []))
    jac = len(ta & tb) / max(1, len(ta | tb))
    if not only_one and jac >= 0.5:
        return 1.0
    # one side qualifies a sub-population of the other => nested (hierarchically related)
    return 0.7 if (shared or jac >= 0.34) else 0.6


def compat_dose(a: dict, b: dict, cfg: ContextConfig) -> float:
    if "num" in a and "num" in b and a["unit"] == b["unit"]:
        return _numeric(a["num"], b["num"], cfg.log_ratio_sigma)
    if "level" in a and "level" in b:
        return {0: 1.0, 1: 0.5, 2: 0.0}[abs(a["level"] - b["level"])]
    return cfg.unknown_value


def compat_time(a: dict, b: dict, cfg: ContextConfig) -> float:
    if "days" in a and "days" in b:
        return _numeric(a["days"], b["days"], cfg.log_ratio_sigma)
    if "horizon" in a and "horizon" in b:
        return 1.0 if a["horizon"] == b["horizon"] else 0.0
    return cfg.unknown_value


def compat_environment(a: dict, b: dict, cfg: ContextConfig) -> float:
    if a["cat"] == b["cat"]:
        return 1.0
    return ENV_COMPAT.get(frozenset({a["cat"], b["cat"]}), 0.0)


def compat_design(a: dict, b: dict, cfg: ContextConfig) -> float:
    if a["cat"] == b["cat"]:
        return 1.0
    if "simulation" in (a["cat"], b["cat"]):
        return 0.1  # simulated vs empirical evidence is a material difference
    return DESIGN_COMPAT.get(frozenset({a["cat"], b["cat"]}), 0.5)


def compat_other(a: dict, b: dict, cfg: ContextConfig) -> float:
    ta, tb = set(a["tokens"]), set(b["tokens"])
    jac = len(ta & tb) / max(len(ta | tb), 1)
    return 1.0 if jac >= 0.5 else 0.5 if jac > 0 else 0.0  # different named conditions are materially different


COMPAT_FNS = {"population": compat_population, "dose": compat_dose, "time": compat_time,
              "environment": compat_environment, "design": compat_design, "other": compat_other}


def compat_vector(ctx_a: dict, ctx_b: dict, cfg: ContextConfig) -> np.ndarray:
    """c_h for every dimension h (order = CONTEXT_DIMS)."""
    out = np.empty(len(CONTEXT_DIMS))
    for k, h in enumerate(CONTEXT_DIMS):
        a, b = ctx_a.get(h), ctx_b.get(h)
        out[k] = cfg.unknown_value if (a is None or b is None) else COMPAT_FNS[h](a, b, cfg)
    return out


def known_mask(ctx_a: dict, ctx_b: dict) -> np.ndarray:
    return np.array([ctx_a.get(h) is not None and ctx_b.get(h) is not None for h in CONTEXT_DIMS])


def context_distance(c: np.ndarray, cfg: ContextConfig) -> float:
    """D_ctx = sum_h w_h (1 - c_h) / sum_h w_h."""
    w = np.array([cfg.weights.get(h, 0.0) for h in CONTEXT_DIMS])
    return float((w * (1.0 - c)).sum() / w.sum())


def material_index(cfg: ContextConfig) -> np.ndarray:
    return np.array([h in cfg.material for h in CONTEXT_DIMS])


def claim_compat_tensor(claims_a: list, claims_b: list, cfg: ContextConfig) -> np.ndarray:
    """(|claims_a|, |claims_b|, H) tensor of per-dimension compatibilities, computed once per pair."""
    H = len(CONTEXT_DIMS)
    C = np.full((len(claims_a), len(claims_b), H), cfg.unknown_value)
    # Claims usually inherit the document-level context, so compute each distinct pair once.
    cache: dict[tuple[int, int], np.ndarray] = {}
    for i, ca in enumerate(claims_a):
        for j, cb in enumerate(claims_b):
            key = (id(ca.context), id(cb.context))
            if key not in cache:
                cache[key] = compat_vector(ca.context, cb.context, cfg)
            C[i, j] = cache[key]
    return C
