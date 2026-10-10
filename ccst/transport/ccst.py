r"""Context-Conditioned Signed Partial Transport (CCST), proposal Sec. 4.3, Eq. (2).

    D(G_i, G_j) = min_{T >= 0, T1 <= p, T^T 1 <= q}
          <M, T>                                             node cost (features, roles, context)
        + alpha * sum_{u,v,u',v'} L_edge(z_uv, z'_u'v') T_uu' T_vv'   typed / signed / directed edges
        + lam_u (|p| + |q| - 2 m(T))                          unmatched mass (partial transport)
        + lam_g Phi_gate(T; C_i, C_j)                          context gate on outcome nodes

Implementation notes
--------------------
* Every edge-loss channel is written as a sum of *separable* products
  ``L(z, z') = sum_k w_k f_k(z) g_k(z')`` so that ``sum L T T = sum_k w_k <T, F_k T G_k^T>`` and the
  tensor product costs O(n^2 m + n m^2) per channel instead of O(n^2 m^2) (Peyre et al., 2016).
  Conditional terms ("both edges exist and differ") stay separable, e.g.
  ``A A'(l - l')^2 = (A l^2) A' + A (A' l'^2) - 2 (A l)(A' l')``.
* Partial transport is solved *exactly* in the linear-minimisation oracle: the partial polytope is
  an ordinary transport polytope after adding one dummy row/column (Chapel et al., 2020); we call
  POT's network-simplex ``ot.emd`` on the (n+1) x (m+1) problem. The unmatched-mass penalty is
  linear in T (const - 2 lam_u * 1^T T 1), so it is folded into the cost.
* The objective is quadratic, so the Frank-Wolfe line search is closed form.
* The alignment T* is computed once with the *unsigned* objective (lam_s = 0) and both
  D_unsigned and D_signed are evaluated on that same T*, as in Sec. 4.3 ("two views of the same
  alignment"). Sign conflicts therefore cannot change *which* nodes are aligned; they only
  show up as the gap D_signed - D_unsigned, which is what localises contradictions.

Complexity per pair: O(I * K * (n^2 m + n m^2) + I * EMD(n, m)) with I Frank-Wolfe iterations
and K ~ 15 separable channels; n, m <= M_max so this is the bounded constant c_CCST of Sec. 4.6.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import ot

from ..compat import claim_compat_tensor, material_index
from ..config import CONTEXT_DIMS, CCSTConfig, ContextConfig
from ..graph import ClaimGraph

BIG = 1e3


@dataclass
class QuadTerm:
    name: str
    F: np.ndarray  # (n, n)
    G: np.ndarray  # (m, m)
    w: float


@dataclass
class PairContext:
    claim_compat: np.ndarray  # (c1, c2, H)
    claim_dctx: np.ndarray  # (c1, c2)
    node_compat: np.ndarray  # (n, m, H)
    node_dctx: np.ndarray  # (n, m)
    gate: np.ndarray  # (n, m)


@dataclass
class CCSTResult:
    T: np.ndarray
    matched: float
    components: dict
    D_unsigned: float
    D_signed: float
    D_mech: float
    D_sem: float
    S_ctx: float
    unmatched_frac: float
    ctx: PairContext | None
    iters: int = 0
    seconds: float = 0.0
    objective: float = 0.0
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- context
def pair_context(G1: ClaimGraph, G2: ClaimGraph, ccfg: ContextConfig) -> PairContext:
    H = len(CONTEXT_DIMS)
    C = claim_compat_tensor(G1.claims, G2.claims, ccfg)
    w = np.array([ccfg.weights.get(h, 0.0) for h in CONTEXT_DIMS])
    claim_dctx = ((1 - C) * w).sum(-1) / w.sum() if C.size else np.zeros(C.shape[:2])
    B1 = _incidence(G1)
    B2 = _incidence(G2)
    node_compat = np.einsum("ia,abh,jb->ijh", B1, C, B2) if C.size else np.full((G1.n, G2.n, H), ccfg.unknown_value)
    node_dctx = ((1 - node_compat) * w).sum(-1) / w.sum()
    mat = material_index(ccfg)
    worst = (1 - node_compat[:, :, mat]).max(-1) if mat.any() else np.zeros((G1.n, G2.n))
    out1 = (G1.roles == 2)[:, None]
    out2 = (G2.roles == 2)[None, :]
    gate = worst * (out1 | out2)
    return PairContext(C, claim_dctx, node_compat, node_dctx, gate)


def _incidence(G: ClaimGraph) -> np.ndarray:
    B = np.zeros((G.n, len(G.claims)))
    for u, cis in enumerate(G.node_claims):
        if cis:
            ws = np.array([G.claims[c].weight for c in cis])
            B[u, cis] = ws / ws.sum()
    return B


# ----------------------------------------------------------------------------- quadratic terms
def quad_terms(G1: ClaimGraph, G2: ClaimGraph, cfg: CCSTConfig) -> tuple[list[QuadTerm], list[QuadTerm]]:
    """Separable decomposition of L_edge (Eq. 1). Returns (unsigned_terms, sign_terms)."""
    c1, c2 = G1.channels(cfg.use_direction), G2.channels(cfg.use_direction)
    n, m = G1.n, G2.n
    J1, J2 = np.ones((n, n)), np.ones((m, m))
    A1, A2 = c1["A"], c2["A"]
    t: list[QuadTerm] = [
        QuadTerm("exist", A1, J2, cfg.lam_e), QuadTerm("exist", J1, A2, cfg.lam_e),
        QuadTerm("exist", A1, A2, -2 * cfg.lam_e),
    ]
    if cfg.use_edge_types and cfg.lam_t:
        t.append(QuadTerm("type", A1, A2, cfg.lam_t))
        tid2 = {ty: k for k, ty in enumerate(G2.rtypes)}
        for k1, ty in enumerate(G1.rtypes):
            if ty in tid2:
                t.append(QuadTerm("type", c1["Y"][k1], c2["Y"][tid2[ty]], -cfg.lam_t))
    for name, key, lam in (("lag", "L", cfg.lam_tau), ("conf", "Q", cfg.lam_q)):
        if lam:
            X1, X2 = c1[key], c2[key]
            t += [QuadTerm(name, A1 * X1**2, A2, lam), QuadTerm(name, A1, A2 * X2**2, lam),
                  QuadTerm(name, A1 * X1, A2 * X2, -2 * lam)]
    s: list[QuadTerm] = []
    if cfg.use_sign:
        P1, N1, Z1, P2, N2, Z2 = c1["P"], c1["N"], c1["Z"], c2["P"], c2["N"], c2["Z"]
        s += [QuadTerm("flip", P1, N2, cfg.lam_s), QuadTerm("flip", N1, P2, cfg.lam_s)]
        if cfg.lam_excl:
            s += [QuadTerm("excl", Z1, P2 + N2, cfg.lam_excl), QuadTerm("excl", P1 + N1, Z2, cfg.lam_excl)]
    return t, s


def _qval(T: np.ndarray, terms: list[QuadTerm]) -> float:
    return float(sum(term.w * np.sum(T * (term.F @ T @ term.G.T)) for term in terms))


def _qgrad(T: np.ndarray, terms: list[QuadTerm]) -> np.ndarray:
    g = np.zeros_like(T)
    for term in terms:
        g += term.w * (term.F @ T @ term.G.T + term.F.T @ T @ term.G)
    return g


def _qbreakdown(T: np.ndarray, terms: list[QuadTerm]) -> dict:
    out: dict = {}
    for term in terms:
        out[term.name] = out.get(term.name, 0.0) + term.w * float(np.sum(T * (term.F @ T @ term.G.T)))
    return out


# ----------------------------------------------------------------------------- LMOs
def _lmo_partial(grad: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n, m = grad.shape
    M = np.zeros((n + 1, m + 1))
    M[:n, :m] = grad
    a = np.r_[p, q.sum()]
    b = np.r_[q, p.sum()]
    S = ot.emd(a, b, M, numItermax=200000)
    return S[:n, :m]


def _lmo_full(grad: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return ot.emd(p, q, grad, numItermax=200000)


# ----------------------------------------------------------------------------- solver
class CCSTSolver:
    def __init__(self, cfg: CCSTConfig | None = None, ctx_cfg: ContextConfig | None = None):
        self.cfg = cfg or CCSTConfig()
        self.ctx_cfg = ctx_cfg or ContextConfig()

    def node_costs(self, G1: ClaimGraph, G2: ClaimGraph, pc: PairContext) -> dict:
        cfg = self.cfg
        feat = np.maximum(0.0, 2.0 - 2.0 * (G1.X @ G2.X.T))  # ||x - x'||^2 for unit vectors
        role = (G1.roles[:, None] != G2.roles[None, :]).astype(float)
        ctx = pc.node_dctx if cfg.use_context else np.zeros_like(feat)
        gate = pc.gate if cfg.use_context else np.zeros_like(feat)
        return {"feat": feat, "role": role, "ctx": ctx, "gate": gate}

    def _masses(self, G1: ClaimGraph, G2: ClaimGraph) -> tuple[np.ndarray, np.ndarray]:
        p, q = G1.mass.astype(float).copy(), G2.mass.astype(float).copy()
        if not self.cfg.partial:  # forced full transport: equal total mass
            s = 0.5 * (p.sum() + q.sum())
            p, q = p / p.sum() * s, q / q.sum() * s
        return p, q

    def solve(self, G1: ClaimGraph, G2: ClaimGraph, pc: PairContext | None = None) -> CCSTResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        if G1.n == 0 or G2.n == 0:
            return _empty_result(G1, G2)
        pc = pc or pair_context(G1, G2, self.ctx_cfg)
        nc = self.node_costs(G1, G2, pc)
        C_lin = cfg.lam_x * nc["feat"] + cfg.lam_r * nc["role"] + cfg.lam_c * nc["ctx"] + cfg.lam_g * nc["gate"]
        if cfg.role_hard_mask:
            C_lin = C_lin + BIG * nc["role"]
        p, q = self._masses(G1, G2)
        terms_u, terms_s = quad_terms(G1, G2, cfg)
        # Alignment objective: unsigned terms, plus (optionally) a small multiple of the sign-conflict terms
        # that only breaks ties between structurally equivalent alignments (docs/tiebreak_protocol.md).
        terms_align = terms_u + [QuadTerm(t.name + "_tiebreak", t.F, t.G, t.w * cfg.lam_s_align)
                                 for t in terms_s] if cfg.lam_s_align > 0 else terms_u
        lin = C_lin - 2.0 * cfg.lam_u if cfg.partial else C_lin
        lmo = _lmo_partial if cfg.partial else _lmo_full

        def objective(T):
            return float(np.sum(lin * T)) + cfg.alpha * _qval(T, terms_align)

        inits = [lmo(lin, p, q)]
        if cfg.n_init > 1:
            inits.append(np.outer(p, q) / max(p.sum(), q.sum()))
        best, best_f, best_it = None, np.inf, 0
        for T in inits:
            if cfg.solver == "entropic":
                T, it = self._entropic(T, lin, terms_align, p, q)
            else:
                T, it = self._frank_wolfe(T, lin, terms_align, p, q, lmo)
            f = objective(T)
            if f < best_f - 1e-12:
                best, best_f, best_it = T, f, it
        T = best
        res = self._evaluate(G1, G2, T, nc, terms_u, terms_s, pc, p, q)
        res.iters, res.objective, res.seconds = best_it, best_f, time.perf_counter() - t0
        return res

    def _frank_wolfe(self, T, lin, terms, p, q, lmo):
        alpha = self.cfg.alpha
        it = 0
        for it in range(1, self.cfg.max_iter + 1):
            grad = lin + alpha * _qgrad(T, terms)
            S = lmo(grad, p, q)
            D = S - T
            b = float(np.sum(grad * D))  # = -FW gap
            if b > -self.cfg.tol:
                break
            a = alpha * _qval(D, terms)
            if a > 1e-12:
                gamma = min(1.0, max(0.0, -b / (2 * a)))
            else:
                gamma = 1.0 if a + b < 0 else 0.0
            if gamma <= 0:
                break
            T = T + gamma * D
        return T, it

    def _entropic(self, T, lin, terms, p, q):
        """Mirror-descent / proximal-point variant with Sinkhorn projections (entropic CCST)."""
        cfg = self.cfg
        n, m = T.shape
        a = np.r_[p, q.sum()] if cfg.partial else p
        b = np.r_[q, p.sum()] if cfg.partial else q
        it = 0
        for it in range(1, cfg.max_iter + 1):
            grad = lin + cfg.alpha * _qgrad(T, terms)
            if cfg.partial:
                M = np.zeros((n + 1, m + 1))
                M[:n, :m] = grad
            else:
                M = grad
            M = M - M.min()
            scale = M.max() + 1e-12
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S = ot.sinkhorn(a, b, M / scale, cfg.entropic_eps, method="sinkhorn_log", numItermax=5000,
                                stopThr=1e-9)
            Tn = S[:n, :m] if cfg.partial else S
            if np.abs(Tn - T).max() < 1e-6:
                T = Tn
                break
            T = Tn
        return T, it

    # ------------------------------------------------------------------ decomposition
    def _evaluate(self, G1, G2, T, nc, terms_u, terms_s, pc, p, q) -> CCSTResult:
        cfg = self.cfg
        mass = float(T.sum())
        comp = {
            "feat": cfg.lam_x * float(np.sum(nc["feat"] * T)),
            "role": cfg.lam_r * float(np.sum(nc["role"] * T)),
            "ctx": cfg.lam_c * float(np.sum(nc["ctx"] * T)),
            "gate": cfg.lam_g * float(np.sum(nc["gate"] * T)),
            "unmatched": cfg.lam_u * float(p.sum() + q.sum() - 2 * mass) if cfg.partial else 0.0,
        }
        qb = _qbreakdown(T, terms_u)
        sb = _qbreakdown(T, terms_s)
        for k in ("exist", "type", "lag", "conf"):
            comp[k] = cfg.alpha * qb.get(k, 0.0)
        for k in ("flip", "excl"):
            comp[k] = cfg.alpha * sb.get(k, 0.0)
        m = max(mass, 1e-9)
        unsigned = sum(comp[k] for k in ("feat", "role", "ctx", "gate", "unmatched", "exist", "type", "lag", "conf"))
        signed = unsigned + comp["flip"] + comp["excl"]
        mech = sum(comp[k] for k in ("role", "unmatched", "exist", "type", "lag", "conf"))
        s_ctx = 1.0 - float(np.sum(pc.node_dctx * T)) / m if mass > 1e-9 else 0.0
        return CCSTResult(
            T=T, matched=mass, components=comp, D_unsigned=unsigned / m, D_signed=signed / m, D_mech=mech / m,
            D_sem=comp["feat"] / max(cfg.lam_x, 1e-9) / m, S_ctx=s_ctx,
            unmatched_frac=1.0 - 2 * mass / max(p.sum() + q.sum(), 1e-9), ctx=pc,
        )


def _empty_result(G1: ClaimGraph, G2: ClaimGraph) -> CCSTResult:
    return CCSTResult(T=np.zeros((G1.n, G2.n)), matched=0.0, components={}, D_unsigned=np.inf, D_signed=np.inf,
                      D_mech=np.inf, D_sem=np.inf, S_ctx=0.0, unmatched_frac=1.0, ctx=None)


# ----------------------------------------------------------------------------- edge evidence
def edge_pair_evidence(G1: ClaimGraph, G2: ClaimGraph, res: CCSTResult, cfg: CCSTConfig,
                       min_mass: float = 1e-6) -> list[dict]:
    """Aligned edge pairs (u->v, u'->v') with transport mass, polarity relation and context distance.

    ``contrib`` is the pair's share of the signed-minus-unsigned gap, i.e. its contribution to
    the disagreement score used for minimal certificates (Sec. 4.5).
    """
    T = res.T
    out = []
    if not G1.edges or not G2.edges or res.matched <= 0:
        return out
    for i, e1 in enumerate(G1.edges):
        for j, e2 in enumerate(G2.edges):
            for (a, b, c, d), rev in (((e1.u, e2.u, e1.v, e2.v), False), ((e1.u, e2.v, e1.v, e2.u), True)):
                if rev and cfg.use_direction:
                    continue
                w = T[a, b] * T[c, d]
                if w < min_mass:
                    continue
                if cfg.use_sign:
                    flip = e1.pos * e2.neg + e1.neg * e2.pos
                    excl = e1.null * (e2.pos + e2.neg) + (e1.pos + e1.neg) * e2.null
                    agree = e1.pos * e2.pos + e1.neg * e2.neg + e1.null * e2.null
                else:  # unsigned ablation: polarity is invisible, every aligned edge "agrees"
                    flip, excl, agree = 0.0, 0.0, 1.0
                ci = max(e1.claims, key=lambda k: G1.claims[k].weight)
                cj = max(e2.claims, key=lambda k: G2.claims[k].weight)
                dctx = float(res.ctx.claim_dctx[ci, cj]) if res.ctx is not None else 0.5
                out.append({
                    "e1": i, "e2": j, "mass": float(w), "conf": e1.conf * e2.conf, "flip": float(flip),
                    "excl": float(excl), "agree": float(agree), "dctx": dctx,
                    "compat": res.ctx.claim_compat[ci, cj].tolist() if res.ctx is not None else None,
                    "known": [G1.claims[ci].context.get(h) is not None and G2.claims[cj].context.get(h) is not None
                              for h in CONTEXT_DIMS],
                    "claim1": ci, "claim2": cj,
                    "contrib": cfg.alpha * w * (cfg.lam_s * flip + cfg.lam_excl * excl) / max(res.matched, 1e-9),
                })
    return out
