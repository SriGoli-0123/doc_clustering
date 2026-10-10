"""Desired technical properties of CCST (proposal Sec. 5), checked on controlled graphs."""
import numpy as np
import pytest

from ccst.config import CCSTConfig, ContextConfig
from ccst.extraction.context import parse_context
from ccst.graph_factory import make_graph
from ccst.relation.features import pair_features
from ccst.transport.ccst import CCSTSolver, edge_pair_evidence

RNG = np.random.default_rng(0)
DIM = 32


def vec():
    v = RNG.normal(size=DIM)
    return v / np.linalg.norm(v)


V = {k: vec() for k in "abcdefghxyz"}
V2 = {k.upper(): vec() for k in "abcdefghxyz"}  # a disjoint "domain vocabulary"
CTX = parse_context("A randomized controlled trial in adult patients given 10 mg for 6 weeks in a clinical setting.")
CTX_SHIFT = parse_context("A randomized controlled trial in infants given 200 mg for 6 weeks in a clinical setting.")
BASE = [("a", "increases", "b"), ("b", "decreases", "c"), ("c", "enables", "d")]


def solve(g1, g2, **kw):
    cfg = CCSTConfig(**kw)
    s = CCSTSolver(cfg, ContextConfig())
    return s.solve(g1, g2), cfg


def test_self_alignment_is_identity_and_zero_cost():
    g = make_graph("g", BASE, V, CTX)
    r, _ = solve(g, g)
    assert np.allclose(r.T, np.diag(g.mass), atol=1e-6)
    assert r.D_mech < 1e-6 and r.D_signed - r.D_unsigned < 1e-9


def test_partial_marginal_constraints_hold():
    g1 = make_graph("g1", BASE, V, CTX)
    g2 = make_graph("g2", BASE + [("x", "increases", "y")], V, CTX)
    r, _ = solve(g1, g2)
    assert np.all(r.T >= -1e-12)
    assert np.all(r.T.sum(1) <= g1.mass + 1e-8) and np.all(r.T.sum(0) <= g2.mass + 1e-8)


def test_relabeling_invariance_of_mechanism_distance():
    g1 = make_graph("g1", BASE, V, CTX)
    ren = [(s.upper(), r, o.upper()) for s, r, o in BASE]
    g2 = make_graph("g2", ren, V2, CTX)
    r_same, _ = solve(g1, make_graph("g1b", BASE, V, CTX))
    r_ren, _ = solve(g1, g2)
    assert abs(r_ren.D_mech - r_same.D_mech) < 1e-6  # names changed, mechanism shape did not
    assert np.allclose(r_ren.T, np.diag(g1.mass), atol=1e-6)  # structure alone recovers the mapping


def test_sign_flip_separation():
    g1 = make_graph("g1", BASE, V, CTX)
    flipped = [("a", "increases", "b"), ("b", "increases", "c"), ("c", "enables", "d")]
    g2 = make_graph("g2", flipped, V, CTX)
    r, cfg = solve(g1, g2)
    assert r.D_mech < 1e-6  # unsigned mechanism shape identical
    gap = r.D_signed - r.D_unsigned
    assert gap > 0.1
    ev = edge_pair_evidence(g1, g2, r, cfg)
    top = max(ev, key=lambda e: e["contrib"])
    assert (g1.edges[top["e1"]].u, g1.edges[top["e1"]].v) == (1, 2)  # localised to b -> c
    # two flips => roughly twice the gap
    g3 = make_graph("g3", [("a", "decreases", "b"), ("b", "increases", "c"), ("c", "enables", "d")], V, CTX)
    r3, _ = solve(g1, g3)
    assert r3.D_signed - r3.D_unsigned == pytest.approx(2 * gap, rel=0.05)


def test_robustness_to_extra_mechanism_partial_vs_full():
    g1 = make_graph("g1", BASE, V, CTX)
    extra = BASE + [("x", "increases", "y"), ("y", "decreases", "z")]
    g2 = make_graph("g2", extra, V, CTX)
    r_p, _ = solve(g1, g2)
    core = r_p.T[:, :4]
    assert np.allclose(core, np.diag(g1.mass), atol=1e-6)  # original correspondence preserved
    assert r_p.T[:, 4:].sum() < 1e-6  # unrelated mechanism left unmatched
    assert r_p.components["exist"] < 1e-9  # only the unmatched penalty is paid
    r_f, _ = solve(g1, g2, partial=False)
    assert r_f.T[:, 4:].sum() > 0.1  # forced full transport must push mass onto the extra mechanism


def test_context_monotonicity_of_contradiction_evidence():
    g1 = make_graph("g1", BASE, V, CTX)
    flipped = [("a", "increases", "b"), ("b", "increases", "c"), ("c", "enables", "d")]
    r_same, cfg = solve(g1, make_graph("g2", flipped, V, CTX))
    r_shift, _ = solve(g1, make_graph("g3", flipped, V, CTX_SHIFT))
    f_same = pair_features(g1, make_graph("g2", flipped, V, CTX), r_same, cfg)
    f_shift = pair_features(g1, make_graph("g3", flipped, V, CTX_SHIFT), r_shift, cfg)
    assert f_shift["F_flip"] < f_same["F_flip"]
    assert f_shift["F_hetero"] > f_same["F_hetero"]
    assert f_shift["S_ctx"] < f_same["S_ctx"]


def test_frank_wolfe_does_not_increase_objective_over_lp_init():
    g1 = make_graph("g1", BASE + [("d", "inhibits", "a")], V, CTX)
    g2 = make_graph("g2", [("b", "increases", "c"), ("a", "enables", "b"), ("c", "decreases", "d")], V, CTX)
    r1, _ = solve(g1, g2, max_iter=0)
    r2, _ = solve(g1, g2)
    assert r2.objective <= r1.objective + 1e-9


def test_entropic_solver_close_to_exact():
    g1 = make_graph("g1", BASE, V, CTX)
    g2 = make_graph("g2", [("a", "increases", "b"), ("b", "increases", "c"), ("c", "enables", "d")], V, CTX)
    r_cg, _ = solve(g1, g2)
    r_en, _ = solve(g1, g2, solver="entropic", entropic_eps=0.01)
    assert abs(r_cg.D_signed - r_en.D_signed) < 0.15


def test_unknown_context_is_not_incompatible_context():
    """A sign flip under *partially unstated but matching* context is contradiction evidence; under a
    *known* material difference it is heterogeneity evidence; with no context at all it is unknown."""
    flipped = [("a", "increases", "b"), ("b", "increases", "c"), ("c", "enables", "d")]
    partial = parse_context("A randomized controlled trial in adult patients.")  # dose/time/env unstated
    shifted = parse_context("A randomized controlled trial in infants.")
    g1 = make_graph("g1", BASE, V, partial)
    feats = {}
    for name, ctx in (("partial", partial), ("shift", shifted), ("none", {})):
        g2 = make_graph(name, flipped, V, ctx)
        r, cfg = solve(g1, g2)
        feats[name] = pair_features(g1, g2, r, cfg)
    assert feats["partial"]["F_conf_compat_frac"] > feats["partial"]["F_conf_incompat_frac"]
    assert feats["shift"]["F_conf_incompat_frac"] > feats["shift"]["F_conf_compat_frac"]
    assert feats["none"]["F_conf_unknown_frac"] > max(feats["none"]["F_conf_compat_frac"],
                                                      feats["none"]["F_conf_incompat_frac"])


def test_tiebreak_removes_spurious_flips_in_symmetric_motif():
    """Fork A->B (+), A->C (-) with indistinguishable B, C: an analogy in another vocabulary must not show a
    sign conflict once ties are broken by sign (docs/tiebreak_protocol.md)."""
    vb = vec()
    v1 = {"a": vec(), "b": vb, "c": vb.copy()}  # B and C indistinguishable by features
    g1 = make_graph("g1", [("a", "increases", "b"), ("a", "decreases", "c")], v1, CTX)
    g2 = make_graph("g2", [("A", "increases", "B"), ("A", "decreases", "C")], V2, CTX)
    r, _ = solve(g1, g2, lam_s_align=0.05)
    assert r.D_signed - r.D_unsigned < 1e-6
    assert r.T[1, 1] > 0.5 and r.T[2, 2] > 0.5  # B<->B', C<->C' (sign-consistent correspondence)


def test_tiebreak_keeps_genuine_flip():
    g1 = make_graph("g1", BASE, V, CTX)
    flipped = [("a", "increases", "b"), ("b", "increases", "c"), ("c", "enables", "d")]
    g2 = make_graph("g2", flipped, V, CTX)
    r0, _ = solve(g1, g2)
    r1, cfg = solve(g1, g2, lam_s_align=0.05)
    assert np.allclose(r1.T, r0.T, atol=1e-6)  # alignment unchanged: no alternative avoids the flip
    assert r1.D_signed - r1.D_unsigned == pytest.approx(r0.D_signed - r0.D_unsigned, rel=1e-6)
    top = max(edge_pair_evidence(g1, g2, r1, cfg), key=lambda e: e["contrib"])
    assert (g1.edges[top["e1"]].u, g1.edges[top["e1"]].v) == (1, 2)
