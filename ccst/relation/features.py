"""Relation features derived from the optimal coupling T* (proposal Sec. 4.4).

S_mech   = exp(-D_mech / sigma)  mechanism-shape similarity. D_mech excludes the node-text and
           context terms so that (i) cross-domain analogies (different words) and (ii)
           heterogeneity pairs (different context) can still have *high* mechanism similarity,
           exactly as the decision rules require. The full D_unsigned / D_signed are kept as features.
S_ctx    = 1 - transport-weighted node context distance.
F_flip   = sum_S w (1 - d_ctx) / m(T*)^2 ;  F_hetero = sum_S w d_ctx / m(T*)^2   (proposal definition)
F_*_mat  = same split but using the *worst material* dimension max_h (1 - c_h) instead of the
           weighted average, so that a single incompatible material condition blocks a contradiction.
F_*_frac = the split as a fraction of all aligned edge mass (size-independent, used by the rules).
"""
from __future__ import annotations

import numpy as np

from ..config import CCSTConfig, ContextConfig
from ..graph import ClaimGraph
from ..compat import material_index
from ..transport.ccst import CCSTResult, edge_pair_evidence

WEAK_EPISTEMIC = {"hypothesized", "quoted"}

FEATURE_NAMES = [
    "S_mech", "D_mech", "D_unsigned", "D_signed", "gap", "D_sem", "S_ctx", "ctx_min_material", "ctx_known_frac",
    "F_flip", "F_hetero", "F_flip_mat", "F_hetero_mat", "F_flip_frac", "F_hetero_frac", "F_unknown_frac",
    "F_agree_frac", "F_agree", "conflict_frac", "matched_frac", "unmatched_frac", "gate", "epi_weak",
    "F_conf_compat_frac", "F_conf_incompat_frac", "F_conf_unknown_frac", "ctx_worst_known", "ctx_unknown_material",
    "log_n_min", "log_n_max",
]


def pair_features(G1: ClaimGraph, G2: ClaimGraph, res: CCSTResult, cfg: CCSTConfig,
                  ctx_cfg: ContextConfig | None = None, sigma: float = 1.0) -> dict:
    ctx_cfg = ctx_cfg or ContextConfig()
    f = {k: 0.0 for k in FEATURE_NAMES}
    if res.matched <= 1e-9 or res.ctx is None:
        f.update(S_mech=0.0, D_mech=10.0, D_unsigned=10.0, D_signed=10.0, D_sem=4.0, unmatched_frac=1.0,
                 log_n_min=np.log1p(min(G1.n, G2.n)), log_n_max=np.log1p(max(G1.n, G2.n)))
        f["_evidence"] = []
        f["n_aligned_edges"] = 0
        return f
    m = res.matched
    T = res.T
    mat = material_index(ctx_cfg)
    pc = res.ctx
    # transport-weighted context compatibility per dimension
    w_dim = np.einsum("ij,ijh->h", T, pc.node_compat) / m
    ev = edge_pair_evidence(G1, G2, res, cfg)
    tot_edge = sum(e["mass"] * e["conf"] for e in ev) + 1e-12
    flip = flip_m = het = het_m = unk = agree = 0.0
    c_comp = c_inc = c_unk = worst_known_w = unk_mat_w = 0.0
    for e in ev:
        w = e["mass"] * e["conf"]
        conflict = e["flip"] + (cfg.lam_excl / max(cfg.lam_s, 1e-9)) * e["excl"]
        comp = np.asarray(e["compat"])
        worst = float((1 - comp[mat]).max()) if mat.any() else 0.0
        is_unknown = (comp[mat] == ctx_cfg.unknown_value)
        dctx = e["dctx"] if cfg.use_context else 0.0
        worst = worst if cfg.use_context else 0.0
        flip += w * conflict * (1 - dctx)
        het += w * conflict * dctx
        flip_m += w * conflict * (1 - worst)
        het_m += w * conflict * worst
        unk += w * conflict * float(is_unknown.mean()) if cfg.use_context else 0.0
        agree += w * e["agree"]
        # three-way split: known-compatible / known-incompatible / unknown material context
        if cfg.use_context:
            known = np.asarray(e["known"])[mat]
            wk = float((1 - comp[mat][known]).max()) if known.any() else 0.0
            uf = float(1 - known.mean())
        else:
            wk, uf = 0.0, 1.0
        c_inc += w * conflict * wk
        c_unk += w * conflict * (1 - wk) * uf
        c_comp += w * conflict * (1 - wk) * (1 - uf)
        worst_known_w += e["mass"] * wk
        unk_mat_w += e["mass"] * uf
    # missingness over aligned claim pairs
    known_w = 0.0
    for e in ev:
        comp = np.asarray(e["compat"])
        known_w += e["mass"] * float((comp != ctx_cfg.unknown_value).mean())
    tot_mass = sum(e["mass"] for e in ev) + 1e-12
    weak = 0.0
    for u in range(G1.n):
        for c in G1.node_claims[u]:
            if G1.claims[c].epistemic in WEAK_EPISTEMIC:
                weak += T[u].sum() / max(len(G1.node_claims[u]), 1)
    for v in range(G2.n):
        for c in G2.node_claims[v]:
            if G2.claims[c].epistemic in WEAK_EPISTEMIC:
                weak += T[:, v].sum() / max(len(G2.node_claims[v]), 1)
    p_tot, q_tot = G1.mass.sum(), G2.mass.sum()
    f.update(
        D_mech=res.D_mech, S_mech=float(np.exp(-res.D_mech / sigma)), D_unsigned=res.D_unsigned,
        D_signed=res.D_signed, gap=res.D_signed - res.D_unsigned, D_sem=res.D_sem,
        S_ctx=res.S_ctx if cfg.use_context else 0.5,
        ctx_min_material=float(w_dim[mat].min()) if (mat.any() and cfg.use_context) else 0.5,
        ctx_known_frac=known_w / tot_mass if cfg.use_context else 0.0,
        F_flip=flip / m**2, F_hetero=het / m**2, F_flip_mat=flip_m / m**2, F_hetero_mat=het_m / m**2,
        F_flip_frac=flip_m / tot_edge, F_hetero_frac=het_m / tot_edge, F_unknown_frac=unk / tot_edge,
        F_agree_frac=agree / tot_edge, conflict_frac=(flip_m + het_m) / tot_edge,
        matched_frac=m / max(min(p_tot, q_tot), 1e-9), unmatched_frac=res.unmatched_frac,
        gate=res.components.get("gate", 0.0) / m, epi_weak=weak / (2 * m),
        F_agree=agree / m**2,
        F_conf_compat_frac=c_comp / tot_edge, F_conf_incompat_frac=c_inc / tot_edge, F_conf_unknown_frac=c_unk / tot_edge,
        ctx_worst_known=worst_known_w / tot_mass, ctx_unknown_material=unk_mat_w / tot_mass if cfg.use_context else 1.0,
        log_n_min=np.log1p(min(G1.n, G2.n)), log_n_max=np.log1p(max(G1.n, G2.n)),
    )
    f["_evidence"] = ev
    f["n_aligned_edges"] = len({e["e1"] for e in ev if e["mass"] >= 0.05})  # guard input, not a model feature
    return f


def domain_separation(meta1: dict, meta2: dict) -> float:
    """S_domain: 0.5 * [different source category] + 0.5 * JSD(theta_i, theta_j) (base 2, in [0,1])."""
    cat = float(meta1.get("category") != meta2.get("category")) if meta1.get("category") and meta2.get("category") else 0.5
    t1, t2 = meta1.get("theta"), meta2.get("theta")
    if t1 is None or t2 is None:
        return cat
    p, q = np.asarray(t1) + 1e-12, np.asarray(t2) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    mix = 0.5 * (p + q)
    jsd = 0.5 * (p * np.log2(p / mix)).sum() + 0.5 * (q * np.log2(q / mix)).sum()
    return 0.5 * cat + 0.5 * float(jsd)
