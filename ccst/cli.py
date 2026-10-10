"""Command-line entry point:  python -m ccst <command> [options]   (or `ccst <command>` after pip install -e .)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .config import load_config
from .utils import RESULTS_DIR, log, set_seed

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def cmd_download(a, cfg):
    from .data import arxiv, claims_datasets as cd, pubmed

    cd.download_scifact()
    cd.download_mancon()
    cd.mancon_pairs(with_abstracts=True)
    pubmed.download(a.pubmed_per_topic)
    arxiv.download(a.arxiv_per_category)


def cmd_synthetic(a, cfg):
    from .experiments import synthetic

    synthetic.run(cfg, quick=a.quick, n_jobs=a.jobs, use_specter=not a.no_specter, use_nli=not a.no_nli,
                  llm_pairs=a.llm_pairs, realism=getattr(a, "realism", False))


def cmd_analysis(a, cfg):
    from .experiments import synthetic_analysis

    synthetic_analysis.run(cfg, use_specter=not a.no_specter, quick=a.quick, realism=getattr(a, "realism", False))


def cmd_real(a, cfg):
    from .experiments import real_pairs

    real_pairs.run(cfg, use_nli=not a.no_nli, n_jobs=a.jobs)


def cmd_corpus(a, cfg):
    from .experiments import corpus

    corpus.run(cfg, per_label=a.per_label, subset_per_label=a.subset_per_label, use_specter=not a.no_specter,
               bertscore_model=None if a.no_bertscore else a.bertscore_model, n_jobs=a.jobs)


def cmd_scaling(a, cfg):
    from .experiments import scaling

    scaling.run(cfg, sizes=tuple(a.sizes), exact_max=a.exact_max, n_jobs=a.jobs)


def cmd_tune(a, cfg):
    from .experiments import tuning

    tuning.run(cfg, n_jobs=a.jobs)


def cmd_tiebreak(a, cfg):
    from .experiments import tiebreak

    tiebreak.run(cfg, n_jobs=a.jobs)


def cmd_extraction_check(a, cfg):
    from .eval.heldout_extraction import evaluate

    r = evaluate()
    print(json.dumps({k: v for k, v in r.items() if k not in ("context_rows",)}, indent=2))


def cmd_candidates(a, cfg):
    from .experiments import candidates_exp

    candidates_exp.run(cfg, n_jobs=a.jobs)


def cmd_bertopic(a, cfg):
    from .experiments import bertopic_exp

    bertopic_exp.run(cfg)


def cmd_compare(a, cfg):
    """Zero-shot relation + certificate for two abstracts (uses the synthetic-trained model)."""
    from .certificate import build_certificate
    from .config import RELATION_LABELS
    from .experiments.common import feature_matrix
    from .experiments.real_pairs import synthetic_model
    from .extraction.graph_builder import GraphPipeline
    from .relation.features import pair_features
    from .transport.ccst import CCSTSolver

    ta = open(a.a).read() if os.path.exists(a.a) else a.a
    tb = open(a.b).read() if os.path.exists(a.b) else a.b
    gp = GraphPipeline(cfg.extraction)
    G = gp.run([("doc_a", ta), ("doc_b", tb)], {"doc_a": {"category": a.cat_a}, "doc_b": {"category": a.cat_b}})
    res = CCSTSolver(cfg.ccst, cfg.context).solve(G["doc_a"], G["doc_b"])
    f = pair_features(G["doc_a"], G["doc_b"], res, cfg.ccst, cfg.context)
    from .relation.features import domain_separation

    f["S_domain"] = domain_separation({"category": a.cat_a}, {"category": a.cat_b})
    clf = synthetic_model(cfg)
    P = clf.predict_proba(feature_matrix([f]))[0]
    k = int(P.argmax())
    from .relation.guards import apply_guards

    label = apply_guards(RELATION_LABELS[k], f) if P[k] >= clf.tau else "abstain"
    cert = build_certificate(G["doc_a"], G["doc_b"], res, f, label, cfg.context, prob=float(P[k]))
    print(json.dumps({"label": label, "probabilities": dict(zip(RELATION_LABELS, map(float, P))),
                      "graph_a": G["doc_a"].summary(), "graph_b": G["doc_b"].summary(),
                      "certificate": cert["text"]}, indent=2))


def cmd_kappa(a, cfg):
    import csv

    from sklearn.metrics import cohen_kappa_score

    def read(p):
        with open(p) as f:
            r = list(csv.DictReader(f))
        col = next(c for c in r[0] if c.startswith("label"))
        return {x["pair_id"]: x[col].strip().lower() for x in r if x[col].strip()}

    A, B = read(a.ann1), read(a.ann2)
    common = sorted(set(A) & set(B))
    print(json.dumps({"n": len(common), "cohen_kappa": float(cohen_kappa_score([A[k] for k in common],
                                                                                 [B[k] for k in common]))}, indent=2))


def cmd_report(a, cfg):
    parts = ["# CCST results summary", "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    for sub in ("synthetic/REPORT.md", "synthetic/ANALYSIS.md", "synthetic_v2/REPORT.md", "synthetic_v2/ANALYSIS.md",
                "tuning/REPORT.md", "tiebreak/REPORT.md", "extraction_fixes/REPORT.md", "candidates/REPORT.md", "bertopic/REPORT.md", "real_pairs/REPORT.md", "corpus/REPORT.md", "scaling/REPORT.md"):
        p = RESULTS_DIR / sub
        if p.exists():
            parts += [p.read_text().replace("\n# ", "\n## "), ""]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(parts))
    log.info(f"wrote {RESULTS_DIR / 'SUMMARY.md'}")


def cmd_all(a, cfg):
    cmd_download(a, cfg)
    cmd_synthetic(a, cfg)
    cmd_analysis(a, cfg)
    cmd_real(a, cfg)
    cmd_corpus(a, cfg)
    cmd_scaling(a, cfg)
    cmd_report(a, cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog="ccst", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CFG)
    p.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    sp = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--quick", action="store_true", help="small smoke-test version")
        s.add_argument("--no-specter", action="store_true", help="skip the SPECTER model (440 MB download)")
        s.add_argument("--no-nli", action="store_true", help="skip the BART-MNLI baseline")
        return s

    d = sp.add_parser("download", help="fetch PubMed, arXiv, SciFact, ManConCorpus")
    d.add_argument("--pubmed-per-topic", type=int, default=500)
    d.add_argument("--arxiv-per-category", type=int, default=500)
    s = common(sp.add_parser("synthetic", help="controlled benchmark: CCST vs baselines vs ablations"))
    s.add_argument("--llm-pairs", type=int, default=0, help="evaluate the direct-LLM judge on N pairs "
                   "(needs CCST_LLM_BASE_URL / CCST_LLM_API_KEY / CCST_LLM_MODEL)")
    s.add_argument("--realism", action="store_true", help="benchmark v2 (docs/benchmark_v2_preregistration.md)")
    sa = common(sp.add_parser("synthetic-analysis", help="alignment, certificates, retrieval, acquisition, signed clustering"))
    sa.add_argument("--realism", action="store_true", help="analyse benchmark v2")
    common(sp.add_parser("real-pairs", help="SciFact + ManConCorpus"))
    c = common(sp.add_parser("corpus", help="topics, clustering, summaries, t-SNE maps, discovery"))
    c.add_argument("--per-label", type=int, default=None, help="cap abstracts per subject label")
    c.add_argument("--subset-per-label", type=int, default=100, help="docs per label in the CCST discovery subset")
    c.add_argument("--no-bertscore", action="store_true")
    c.add_argument("--bertscore-model", default="distilbert-base-uncased")
    sc = sp.add_parser("scaling", help="runtime scaling + candidate recall")
    sc.add_argument("--sizes", type=int, nargs="+", default=[125, 250, 500, 1000, 2000])
    sc.add_argument("--exact-max", type=int, default=500)
    sp.add_parser("tune", help="item 2: nested-CV hyper-parameter selection on benchmarks v1 and v2")
    sp.add_parser("tiebreak", help="item 3: sign-aware tie-breaking in the alignment (benchmarks v1 and v2)")
    sp.add_parser("extraction-check", help="item 4: claim / context extraction accuracy on the held-out phrasing set")
    sp.add_parser("candidates", help="item 5: candidate-recall improvement (learned combined index)")
    sp.add_parser("bertopic", help="item 6: official BERTopic baseline vs LDA / embedding clustering")
    cmp_ = sp.add_parser("compare", help="relation + certificate for two abstracts (text or file paths)")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--cat-a", default="A")
    cmp_.add_argument("--cat-b", default="A")
    k = sp.add_parser("kappa", help="Cohen's kappa between two annotators' sheets")
    k.add_argument("ann1")
    k.add_argument("ann2")
    sp.add_parser("report", help="merge every REPORT.md into results/SUMMARY.md")
    al = common(sp.add_parser("all", help="run the whole study end to end"))
    al.add_argument("--llm-pairs", type=int, default=0)
    al.add_argument("--pubmed-per-topic", type=int, default=500)
    al.add_argument("--arxiv-per-category", type=int, default=500)
    al.add_argument("--per-label", type=int, default=None)
    al.add_argument("--subset-per-label", type=int, default=100)
    al.add_argument("--no-bertscore", action="store_true")
    al.add_argument("--bertscore-model", default="distilbert-base-uncased")
    al.add_argument("--sizes", type=int, nargs="+", default=[125, 250, 500, 1000, 2000])
    al.add_argument("--exact-max", type=int, default=500)

    a = p.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(cfg.seed)
    {"download": cmd_download, "synthetic": cmd_synthetic, "synthetic-analysis": cmd_analysis, "real-pairs": cmd_real,
     "corpus": cmd_corpus, "scaling": cmd_scaling, "tune": cmd_tune, "tiebreak": cmd_tiebreak, "extraction-check": cmd_extraction_check, "candidates": cmd_candidates, "bertopic": cmd_bertopic, "compare": cmd_compare, "kappa": cmd_kappa, "report": cmd_report,
     "all": cmd_all}[a.cmd](a, cfg)


if __name__ == "__main__":
    sys.exit(main())
