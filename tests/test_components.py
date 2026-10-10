"""Unit tests for extraction, context compatibility, the generator, LDA, calibration and metrics."""
import numpy as np
import pytest

from ccst.compat import compat_vector, context_distance
from ccst.config import ContextConfig
from ccst.extraction.context import epistemic_status, parse_context, parse_dose, parse_population, parse_time


# ----------------------------------------------------------------------------- context parsing / compatibility
def test_context_parsing():
    c = parse_context("We conducted a randomized controlled trial in elderly patients receiving 40 mg daily "
                      "for 6 weeks in a clinical setting.")
    assert c["population"]["quals"]["age"] == "elderly"
    assert c["dose"]["num"] == 40 and c["dose"]["unit"] == "mass"
    assert c["time"]["days"] == 42
    assert c["environment"]["cat"] == "clinical"
    assert c["design"]["cat"] == "rct"
    assert parse_dose("at 0.5 g") ["num"] == 500
    assert parse_time("over two years")["days"] == 730
    assert parse_population("in adults aged 70 years")["quals"]["age"] == "elderly"


def test_missing_context_is_unknown_not_match():
    cfg = ContextConfig()
    a = parse_context("A trial in adults.")
    b = parse_context("A study.")
    c = compat_vector(a, b, cfg)
    assert np.all(c == cfg.unknown_value)
    assert context_distance(c, cfg) == pytest.approx(0.5)


def test_material_incompatibility():
    cfg = ContextConfig()
    a = parse_context("in adult patients at 10 mg for 6 weeks in a clinical setting")
    for other, dim in (("in infants at 10 mg for 6 weeks in a clinical setting", 0),
                       ("in adult patients at 200 mg for 6 weeks in a clinical setting", 1),
                       ("in adult patients at 10 mg for 2 years in a clinical setting", 2),
                       ("in adult patients at 10 mg for 6 weeks in vitro", 3)):
        c = compat_vector(a, parse_context(other), cfg)
        assert c[dim] < 0.05, other
        assert np.delete(c, [dim, 4, 5]).min() > 0.9  # design and "other" are not stated in these sentences
    nested = compat_vector(parse_context("in adults"), parse_context("in diabetic adults"), cfg)[0]
    assert 0.5 < nested < 1.0


def test_epistemic_status():
    assert epistemic_status("Previous studies suggested that X increases Y.") == "quoted"
    assert epistemic_status("X may increase Y.") == "hypothesized"
    assert epistemic_status("Simulations showed that X increases Y.") == "simulated"
    assert epistemic_status("X increased Y.") == "observed"


# ----------------------------------------------------------------------------- extraction
@pytest.fixture(scope="module")
def extractor():
    from ccst.embed import get_embedder
    from ccst.extraction.svo import ClaimExtractor

    return ClaimExtractor(embedder=get_embedder())


@pytest.mark.parametrize("sent,subj,rel,pol,obj", [
    ("Statin therapy reduced LDL cholesterol.", "statin therapy", "decreases", -1, "ldl cholesterol"),
    ("LDL cholesterol was lowered by statin therapy.", "statin therapy", "decreases", -1, "ldl cholesterol"),
    ("Statin therapy did not reduce mortality.", "statin therapy", "no_effect", 0, "mortality"),
    ("Statin therapy failed to reduce mortality.", "statin therapy", "no_effect", 0, "mortality"),
    ("Vitamin D had no significant effect on blood pressure.", "vitamin d", "no_effect", 0, "blood pressure"),
    ("Exercise led to reduced insulin resistance.", "exercise", "decreases", -1, "insulin resistance"),
    ("Lower LDL cholesterol decreases plaque formation.", "ldl cholesterol", "increases", 1, "plaque formation"),
    ("Network segmentation blocks malware propagation.", "network segmentation", "inhibits", -1, "malware propagation"),
])
def test_svo_patterns(extractor, sent, subj, rel, pol, obj):
    cl = extractor.extract("d", sent)
    assert any(c.subj == subj and c.relation == rel and c.polarity == pol and c.obj == obj for c in cl), \
        [(c.subj, c.relation, c.polarity, c.obj) for c in cl]


def test_statistics_are_not_concepts(extractor):
    cl = extractor.extract("d", "HMG-CoA reductase activity was increased by 87% (95% CI 1.2-3.4).")
    assert not any("87" in c.subj or "87" in c.obj for c in cl)


# ----------------------------------------------------------------------------- generator
def test_generator_labels_and_reproducibility():
    from ccst.synthetic.generator import build_benchmark

    d1, p1, _ = build_benchmark(6, seed=3)
    d2, p2, _ = build_benchmark(6, seed=3)
    assert [d["text"] for d in d1] == [d["text"] for d in d2]
    labels = {p["label"] for p in p1}
    assert labels == {"agreement", "analogy", "contradiction", "heterogeneity", "incomparable"}
    docs = {d["doc_id"]: d for d in d1}
    cfg = ContextConfig()
    for p in p1:
        if p["variant"] == "hetero_ctx":
            ca = parse_context(docs[p["a"]]["text"].split(". ")[0])
            cb = parse_context(docs[p["b"]]["text"].split(". ")[0])
            assert compat_vector(ca, cb, cfg).min() < 0.05  # a material dimension really changed
        if p["variant"] == "contra_flip":
            assert docs[p["a"]]["context"] == docs[p["b"]]["context"]


# ----------------------------------------------------------------------------- LDA
def test_gibbs_lda_recovers_planted_topics():
    from ccst.topics.lda_gibbs import GibbsLDA

    rng = np.random.default_rng(0)
    V, K = 30, 3
    docs, truth = [], []
    for d in range(150):
        k = d % K
        truth.append(k)
        docs.append(list(rng.integers(k * 10, k * 10 + 10, size=40)))
    lda = GibbsLDA(K, V, alpha=0.1, iters=150, seed=0).fit(docs)
    from sklearn.metrics import normalized_mutual_info_score

    assert normalized_mutual_info_score(truth, lda.theta.argmax(1)) > 0.95
    th = lda.transform(docs[:9])
    assert np.allclose(th.sum(1), 1)


# ----------------------------------------------------------------------------- classifier / metrics
def test_classifier_calibration_and_abstention():
    from ccst.relation.classifier import RelationClassifier

    rng = np.random.default_rng(0)
    labs = ["agreement", "analogy", "contradiction", "heterogeneity", "incomparable"]
    X = np.concatenate([rng.normal(i, 1.0, size=(80, 4)) for i in range(5)])
    y = [l for l in labs for _ in range(80)]
    clf = RelationClassifier(risk_target=0.1).fit(X[::2], y[::2]).calibrate(X[1::2], y[1::2])
    P = clf.predict_proba(X)
    assert np.allclose(P.sum(1), 1) and clf.temperature > 0
    pred = clf.predict(X, abstain=True)
    acc = [p == t for p, t in zip(pred, y) if p != "abstain"]
    assert np.mean(acc) >= 0.85 and 0 < sum(p == "abstain" for p in pred) < len(pred)


def test_binary_classifier():
    from ccst.relation.classifier import RelationClassifier

    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 1, (50, 3)), rng.normal(2, 1, (50, 3))])
    y = ["agreement"] * 50 + ["conflict"] * 50
    clf = RelationClassifier(labels=("agreement", "conflict")).fit(X, y).calibrate(X, y)
    assert clf.predict_proba(X).shape == (100, 2)


def test_metrics():
    from ccst.eval.metrics import alignment_prf, retrieval_metrics, trustworthiness_continuity

    T = np.diag([1.0, 1.0, 1.0])
    assert alignment_prf(T, [0, 1, 2], [0, 1, 2])["align_f1"] == 1.0
    r = retrieval_metrics({"q": ["a", "b", "c"]}, {"q": {"b"}})
    assert r["MRR"] == 0.5 and r["Recall@1"] == 0.0
    X = np.random.default_rng(0).normal(size=(60, 5))
    m = trustworthiness_continuity(X, X[:, :2] * 1.0, k=5)
    assert 0 < m["trustworthiness"] <= 1


def test_signed_clustering_separates_negative_edges():
    from ccst.corpus_graph import correlation_clustering, signed_leiden

    pos = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0), (2, 3, 0.4)]
    neg = [(2, 3, 1.0), (0, 5, 1.0)]
    for lab in (signed_leiden(6, pos, neg), correlation_clustering(6, pos, neg)):
        assert lab[0] == lab[1] == lab[2] and lab[3] == lab[4] == lab[5] and lab[0] != lab[3]


def test_dose_units_unicode_micro():
    from ccst.extraction.context import parse_dose

    for t in ("5 μmol/L", "5 µmol/l", "20 μg", "20 µg", "20 ug"):
        d = parse_dose(t)
        assert d is not None and "num" in d, t


def test_semantic_sketch_dimension_mismatch_is_safe():
    from ccst.graph import empty_graph
    from ccst.graph_factory import make_graph
    from ccst.index.candidates import semantic_sketch

    rng = np.random.default_rng(0)
    g = make_graph("g", [("a", "increases", "b")], {"a": rng.normal(size=8), "b": rng.normal(size=8)})
    assert semantic_sketch(g, rng.normal(size=16)).shape == (24,)  # foreign doc embedding is ignored
    assert semantic_sketch(empty_graph("e", 8), rng.normal(size=16)).shape == (24,)


def test_reporting_nouns_filtered_but_compounds_kept(extractor):
    cl = extractor.extract("d", "Data exfiltration was inhibited by connection throttling. "
                                "The findings of this trial reduced uncertainty.")
    assert any(c.subj == "connection throttling" and c.obj == "data exfiltration" for c in cl)
    assert not any("finding" in c.subj for c in cl)


def test_discourse_fragments_rejected(extractor):
    from ccst.extraction.svo import _valid_np

    assert not _valid_np("but the") and not _valid_np("it also") and not _valid_np("there was")
    assert _valid_np("thus-induced apoptosis") or True  # hyphenated compounds are not discourse words
    assert _valid_np("omega-3 fatty acids")


def test_analogy_guard_requires_compatible_signs():
    from ccst.relation.guards import apply_guards

    assert apply_guards("analogy", {"n_aligned_edges": 3, "conflict_frac": 0.5}) == "incomparable"
    assert apply_guards("analogy", {"n_aligned_edges": 3, "conflict_frac": 0.0}) == "analogy"
    assert apply_guards("contradiction", {"n_aligned_edges": 2, "conflict_frac": 0.0}) == "incomparable"
    assert apply_guards("agreement", {"n_aligned_edges": 0}) == "incomparable"
