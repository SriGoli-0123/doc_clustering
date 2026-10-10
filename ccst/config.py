"""Typed configuration for every stage, loadable from YAML (configs/default.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

CONTEXT_DIMS = ("population", "dose", "time", "environment", "design", "other")
RELATION_LABELS = ("agreement", "analogy", "contradiction", "heterogeneity", "incomparable")


@dataclass
class CCSTConfig:
    """Weights of the CCST objective (Eq. 2 of the proposal).

    Linear (node) terms are normalised by matched mass m(T); quadratic (edge) terms by m(T)^2,
    so distances are comparable across graph sizes.
    """

    lam_x: float = 0.25  # node-feature (semantic) cost  ||x_u - x_u'||^2
    lam_r: float = 1.0  # role mismatch cost
    lam_c: float = 0.25  # node-level context distance d_ctx(u,u')
    lam_g: float = 0.2  # context gate on outcome nodes (Phi_gate)
    lam_u: float = 0.6  # unmatched-mass penalty (per unit, per side)
    alpha: float = 1.0  # weight of the edge (Gromov) term
    lam_e: float = 1.0  # edge existence mismatch
    lam_t: float = 0.5  # relation-type mismatch (both edges exist)
    lam_s: float = 1.0  # sign conflict (opposite signs)
    lam_excl: float = 0.75  # mutually exclusive assertion (effect vs. no-effect)
    lam_tau: float = 0.2  # temporal lag difference
    lam_q: float = 0.2  # edge-confidence difference
    partial: bool = True  # False -> forced full (balanced) transport ablation
    use_context: bool = True  # False -> signed matcher without context conditioning
    use_sign: bool = True  # False -> unsigned structural matcher
    use_edge_types: bool = True
    use_direction: bool = True
    solver: str = "cg"  # "cg" (Frank-Wolfe with exact partial-OT LMO) | "entropic"
    max_iter: int = 60
    tol: float = 1e-7
    entropic_eps: float = 0.05
    n_init: int = 2  # LP init + product init; best objective kept
    role_hard_mask: bool = False
    lam_s_align: float = 0.05  # sign-conflict weight in the *alignment* objective: tie-breaking (item 3, adopted)


@dataclass
class ContextConfig:
    weights: dict = field(
        default_factory=lambda: {"population": 1.0, "dose": 1.0, "time": 1.0, "environment": 1.0, "design": 0.5,
                                 "other": 0.5}
    )
    material: tuple = ("population", "dose", "time", "environment", "design", "other")
    unknown_value: float = 0.5
    log_ratio_sigma: float = 0.7  # tolerance for numeric dose/time ratios (log space)


@dataclass
class ExtractionConfig:
    spacy_model: str = "en_core_web_sm"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    concept_distance_threshold: float = 0.30  # cosine distance for agglomerative concept merging
    relation_expand_threshold: float = 0.55  # verb-embedding similarity for lexicon expansion
    max_mechanisms: int = 24  # M_max retained claims per document
    generalize: bool = True  # False -> raw triplets (ablation)
    use_epistemic: bool = True


@dataclass
class RelationConfig:
    calibration: str = "temperature"  # temperature | isotonic | none
    risk_target: float = 0.10  # conformal selective risk level for abstention
    C: float = 1.0  # inverse L2 strength of the multinomial LR


@dataclass
class IndexConfig:
    k_sem: int = 10
    k_mech: int = 10
    k_ctx: int = 5
    explore: int = 2
    topic_blocking: bool = True


@dataclass
class TopicConfig:
    k_grid: tuple = (10, 20, 30, 50)
    gibbs_iters: int = 400
    alpha: float | None = None  # None -> 50/K (Griffiths & Steyvers)
    beta: float = 0.01
    min_df: int = 5
    max_df: float = 0.5
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    specter_model: str = "sentence-transformers/allenai-specter"
    perplexities: tuple = (15, 30, 50)


@dataclass
class Config:
    seed: int = 13
    ccst: CCSTConfig = field(default_factory=CCSTConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)


def _merge(dc: Any, overrides: dict) -> Any:
    kwargs = {}
    names = {f.name: f for f in fields(dc)}
    for k, v in overrides.items():
        if k not in names:
            raise KeyError(f"Unknown config key '{k}' for {type(dc).__name__}")
        cur = getattr(dc, k)
        if is_dataclass(cur) and isinstance(v, dict):
            kwargs[k] = _merge(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return replace(dc, **kwargs)


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
