"""Reproduce the proposal's Table 1 on four hand-written abstract pairs (zero-shot, synthetic-trained model).

    python scripts/demo_table1.py
"""
from ccst.certificate import build_certificate
from ccst.config import RELATION_LABELS, load_config
from ccst.experiments.common import feature_matrix
from ccst.experiments.real_pairs import synthetic_model
from ccst.extraction.graph_builder import GraphPipeline
from ccst.relation.features import domain_separation, pair_features
from ccst.relation.guards import apply_guards
from ccst.transport.ccst import CCSTSolver

A = ("We conducted a randomized controlled trial in adult patients receiving vitamin D supplementation at 1000 IU "
     "daily for 8 weeks. Vitamin D supplementation reduced systolic blood pressure. Lower systolic blood pressure "
     "decreased cardiovascular events.")
CASES = {
    "sign flip, same context": A.replace("reduced systolic", "increased systolic"),
    "sign flip, infants / 2 years": A.replace("adult patients", "infants").replace("8 weeks", "2 years")
                                     .replace("reduced systolic", "increased systolic"),
    "sign flip, no context stated": "Vitamin D supplementation increased systolic blood pressure.",
    "paraphrase": A.replace("reduced systolic", "lowered systolic").replace("Lower systolic", "Reduced systolic"),
    "same mechanism, other field": ("We conducted a controlled experiment on enterprise networks applying traffic "
                                    "filtering at 100 requests per second over 24 hours. Traffic filtering reduced "
                                    "packet loss. Lower packet loss decreased service outages."),
}
CATS = {"same mechanism, other field": "cs.CR"}


def main():
    cfg = load_config()
    gp = GraphPipeline(cfg.extraction)
    docs = [("A", A)] + [(k, v) for k, v in CASES.items()]
    metas = {d: {"category": CATS.get(d, "med")} for d, _ in docs}
    G = gp.run(docs, metas)
    clf = synthetic_model(cfg)
    solver = CCSTSolver(cfg.ccst, cfg.context)
    print(f"{'case':32s} {'verdict':15s} p      certificate")
    for name in CASES:
        res = solver.solve(G["A"], G[name])
        f = pair_features(G["A"], G[name], res, cfg.ccst, cfg.context)
        f["S_domain"] = domain_separation(metas["A"], metas[name])
        P = clf.predict_proba(feature_matrix([f]))[0]
        k = int(P.argmax())
        lab = apply_guards(RELATION_LABELS[k], f) if P[k] >= clf.tau else "abstain"
        cert = build_certificate(G["A"], G[name], res, f, lab, cfg.context, prob=float(P[k]))
        print(f"{name:32s} {lab:15s} {P[k]:.2f}   {cert['text'][:220]}...")


if __name__ == "__main__":
    main()
