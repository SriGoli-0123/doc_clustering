# CCST — Context-Conditioned Signed Partial Transport for Scientific Literature

Implementation of the CSE 573 (Fall 2026, Group G16) proposal *"Context-Conditioned Causal Alignment for
Scientific Literature"*. The system decides **when two reported mechanisms are comparable** and only then
whether they **agree, are a cross-domain analogy, genuinely contradict, differ because their context
differs (heterogeneity), or are insufficiently comparable** — and returns a checkable, provenance-linked
**minimal disagreement certificate** for each decision. LDA / embedding clustering, summaries, t-SNE maps,
sparse candidate generation and budgeted review are implemented and evaluated as supporting components.

> **Results:** see [`RESULTS.md`](RESULTS.md) (headline numbers, honest caveats) and `results/*/REPORT.md`
> (every table) after a run. All figures land in `results/`.

---

## 1. Quick start

```bash
git clone https://github.com/SriGoli-0123/doc_clustering.git && cd doc_clustering
make setup          # venv + requirements + spaCy model + editable install
make test           # 35 unit/property tests (CCST properties from proposal Sec. 5)
make quick          # ~6 min smoke test of the synthetic benchmark (no large model downloads)
make all            # full study: download -> synthetic -> analyses -> real pairs -> corpus -> scaling -> report
```

Individual stages (each writes `results/<stage>/REPORT.md`, `metrics.json` and figures):

```bash
python -m ccst download                 # PubMed (10 topics x 500), arXiv (10 categories x 500), SciFact, ManConCorpus
python -m ccst --jobs 6 synthetic       # RQ1-RQ3: CCST vs 14 baselines vs 15 ablations (~15 min on an M4)
python -m ccst synthetic-analysis       # alignment, certificates, analogy retrieval, Step G review, Step H clustering
python -m ccst synthetic --realism      # benchmark v2: pre-registered realistic difficulty (docs/benchmark_v2_preregistration.md)
python -m ccst synthetic-analysis --realism
python -m ccst tune                     # item 2: nested-CV hyper-parameter selection (docs/tuning_protocol.md)
python -m ccst tiebreak                 # item 3: sign-aware tie-breaking evaluation (docs/tiebreak_protocol.md)
python -m ccst extraction-check         # item 4: extraction accuracy on the held-out phrasing set
python -m ccst candidates               # item 5: learned combined candidate index vs original (recall vs cost)
python -m ccst bertopic                 # item 6: official BERTopic baseline (docs/bertopic_protocol.md)
python -m ccst real-pairs               # SciFact + ManConCorpus (expert labels)
python -m ccst corpus                   # RQ5 + Steps F/H/I/J on the ~10k-abstract corpus
python -m ccst scaling                  # RQ4 runtime scaling + candidate recall
python -m ccst report                   # merge everything into results/SUMMARY.md
python -m ccst compare a.txt b.txt      # relation + certificate for any two abstracts (zero-shot)
python -m ccst kappa ann1.csv ann2.csv  # Cohen's kappa for the team's annotation round
```

Optional direct-LLM baseline (any OpenAI-compatible endpoint, e.g. ASU SOL's hosted Qwen3):

```bash
export CCST_LLM_BASE_URL=https://openai.rc.asu.edu/v1 CCST_LLM_API_KEY=$SOL_API_KEY CCST_LLM_MODEL=qwen3-235b-a22b-instruct-2507
python -m ccst synthetic --llm-pairs 500
```

Flags: `--quick`, `--no-specter` (skip the 440 MB SPECTER download), `--no-nli`, `--no-bertscore`, `--jobs N`.
Hyper-parameters live in [`configs/default.yaml`](configs/default.yaml) (mirrors `ccst/config.py`).

### Demo: the proposal's Table 1 on five abstract pairs

```bash
python scripts/demo_table1.py
```

| doc B vs. doc A (vitamin D → ↓ systolic BP → ↓ cardiovascular events, adults, 8 weeks) | verdict | p |
|---|---|---|
| same context, vitamin D *increases* SBP | contradiction | 0.99 |
| *infants, 2 years*, vitamin D increases SBP | heterogeneity (certificate: population adult vs infants; time 8 weeks vs 2 years) | 1.00 |
| vitamin D increases SBP, *no context stated* | incomparable | 1.00 |
| paraphrase ("lowered", "reduced") | agreement | 0.94 |
| traffic filtering → ↓ packet loss → ↓ outages (cs.CR) | analogy | 1.00 |

### Pre-registered protocols (written before each experiment was run)

| item | protocol | results |
|---|---|---|
| 1 — realistic benchmark v2 | `docs/benchmark_v2_preregistration.md` | `results/synthetic_v2/` |
| 2 — nested-CV hyper-parameter selection | `docs/tuning_protocol.md` | `results/tuning/` |
| 3 — sign-aware tie-breaking | `docs/tiebreak_protocol.md` | `results/tiebreak/` |
| 4 — real-text parsing fixes (+ held-out phrasing set) | `docs/parsing_fixes_protocol.md` | `results/extraction_fixes/` |
| 5 — learned candidate index | `docs/candidate_recall_protocol.md` | `results/candidates/` |
| 6 — official BERTopic baseline | `docs/bertopic_protocol.md` | `results/bertopic/` |

## 2. What maps to what

| Proposal | Code |
|---|---|
| Step A — SVO triplets (passive, negation, prepositional objects) → generalised concepts & signed relations (Ceran et al.) | `ccst/extraction/svo.py`, `relations.py`, `concepts.py`, `graph_builder.py` |
| Step A/B — scope conditions, epistemic status, provenance; value compatibility c_h, D_ctx | `ccst/extraction/context.py`, `ccst/compat.py` |
| Step C — CCST objective (Eq. 2): partial, typed, signed, directed, context-gated transport | `ccst/transport/ccst.py` |
| Step D — S_mech, S_ctx, F_flip, F_hetero, S_domain; ordered rules; calibrated multinomial LR + abstention | `ccst/relation/features.py`, `rules.py`, `classifier.py` |
| Step E — minimal disagreement certificate (A, K, F, U, P) + rendered paragraph | `ccst/certificate.py` |
| Step F — LDA (collapsed Gibbs), C_v/NPMI, SBERT/SPECTER, K-Means/HDBSCAN/BERTopic-style, sketches + ANN candidates | `ccst/topics/*`, `ccst/index/candidates.py` |
| Step G — hard-budget review (uncertainty, expected change, diversity, DiverseTopB, curvature) | `ccst/experiments/synthetic_analysis.py::acquisition_eval` |
| Step H — multi-relational corpus graph, signed Leiden / correlation clustering | `ccst/corpus_graph.py` |
| Step I — labels, TextRank + centroid + MMR, relation summaries, ROUGE / BERTScore | `ccst/summarize.py` |
| Step J — PCA → t-SNE 2D/3D, perplexity sweep, trustworthiness/continuity, relation-link maps | `ccst/viz/maps.py` |
| Sec. 5 properties (relabeling invariance, sign-flip separation, extra-mechanism robustness, context monotonicity) | `tests/test_ccst_properties.py` |
| Sec. 7.2 stratum 1 + 4 (controlled transformations, adversarial hard negatives) | `ccst/synthetic/` |
| Sec. 7.2 stratum 3 (SciFact, ManConCorpus, PubMed natural pairs annotation sheet) | `ccst/data/claims_datasets.py`, `experiments/real_pairs.py`, `results/corpus/annotation_sheet.csv` |
| Sec. 7.3 baselines | `ccst/eval/baselines_text.py`, `ccst/transport/baselines.py` |
| Sec. 7.5 metrics (+ bootstrap CIs, paired bootstrap, McNemar) | `ccst/eval/metrics.py` |

## 3. Evaluation protocol (why the comparison is fair)

* **Identical classifier for every method.** Each method (CCST, BM25, SBERT, SPECTER, LDA-θ,
  purpose/mechanism embeddings, raw SVO overlap, BART-MNLI, FGW, FNGW, LinearFGW, graph edit distance,
  role-constrained matching, signed matcher without context, context classifier without transport) turns a
  pair into a feature vector; all are scored by the same multinomial LR with the same calibration and
  abstention rule. Every method also receives the shared `S_domain` feature (analogy is *defined* by
  domain distance), so differences come from the representation alone.
* **Grouped cross-validation, no leakage.** 5-fold `GroupKFold` by mechanism template (synthetic), by
  claim (SciFact) or by clinical question (ManCon). Inside each training split, 25% of the groups are held
  out for temperature scaling / abstention thresholds / rule thresholds — the test fold is never touched.
* **Uncertainty.** 95% bootstrap CIs on macro-F1; paired bootstrap and McNemar tests against CCST.
* **The critical safety metric** (proposal Sec. 7.5): false-contradiction rate on context-shift hard
  negatives, reported next to contradiction recall so that "never say contradiction" cannot win.

## 4. Time complexity

| Stage | Complexity | Measured in |
|---|---|---|
| Claim extraction (spaCy parse + regex scope parsing) | O(L) per document (L tokens) | logs |
| Concept generalisation (agglomerative, cosine) | O(P² d) dense; O(P k d + P log P) with a kNN connectivity graph for P > 4000 phrases | logs |
| LDA, collapsed Gibbs | O(I · N_tokens · K) time, O(DK + KV) memory | `results/corpus` |
| C_v coherence | O(K · T² · W) (T top words, W sliding windows) | |
| SBERT / SPECTER embeddings | O(N · L²) attention, L ≤ 512 | |
| K-Means / HDBSCAN | O(I N k d) / ≈O(N log N) after PCA | |
| t-SNE (Barnes–Hut) | O(I · N log N) | |
| **CCST, one pair** | **O(I · K · (n²m + nm²) + I · EMD(n+1, m+1))** — separable edge losses avoid the naive O(n²m²) Gromov tensor; n, m ≤ M_max ⇒ bounded constant c_CCST | `results/scaling/per_pair_cost.png` |
| Exact all-pairs alignment | O(N² · c_CCST) | `results/scaling/runtime_scaling.png` |
| Sketches + kNN + topic blocking | O(N · M_max²) + O(N log N) (tree kNN, low-dim sketches) / O(N² d) brute for the semantic sketch | |
| **Sparse candidates + CCST** | **O(N log N + N k · c_CCST)** (empirical claim; candidate recall reported) | `results/scaling/` |
| Relation LR / calibration | O(#pairs · F · C · iters) | |
| Certificate | O(|E_i| · |E_j|) per pair | |
| Signed Leiden / correlation clustering | ≈O(|E|) per pass | |
| TextRank + MMR summary | O(s² d) per cluster (s sentences, capped) | |

I = Frank–Wolfe iterations (≤ 60, typically 1–5), K ≈ 15 separable edge channels, k = k_sem + k_mech + k_ctx.

## 5. Design decisions and deviations (read before citing numbers)

* **Exact partial transport.** The unmatched-mass penalty λ_u(|p|+|q|−2m(T)) is linear in T, so the
  partial polytope becomes an ordinary transport problem with one dummy row/column; POT's network simplex
  solves the Frank–Wolfe linear oracle exactly. The quadratic objective gives a closed-form line search.
  An entropic (Sinkhorn mirror-descent) solver is available via `ccst.solver: entropic`.
* **Separable edge losses.** I[type≠type′] = 1 − Σ_t y_t y′_t for one-hot types; sign conflict is
  P⊗N′ + N⊗P′ (+ no-effect vs effect as a "mutually exclusive assertion", weight λ_excl); lag and
  confidence use the conditional squared loss A A′(x − x′)². The proposal's |conf − conf′| is replaced
  by the squared difference to keep the tensor product separable.
* **One alignment, two views.** T* minimises the *unsigned* objective plus a small sign-conflict term
  (ε = `lam_s_align` = 0.05) that only breaks ties between structurally equivalent alignments; D_unsigned and
  D_signed are both evaluated on that same T* (proposal: "two views of the same alignment"). A genuine sign
  flip that no alternative alignment avoids still appears as the gap D_signed − D_unsigned that localises
  contradictions, while symmetric motifs (A → Y₁, A → Y₂) no longer produce spurious flips
  (`docs/tiebreak_protocol.md`).
* **S_mech excludes node-text and context terms.** The decision rules need heterogeneity pairs (different
  context) and analogies (different words) to keep a *high* S_mech, so S_mech = exp(−D_mech/σ) uses the
  role + structure + unmatched terms only; full D_unsigned / D_signed are still classifier features.
* **Worst material dimension.** Besides the proposal's F_flip / F_hetero (split by weighted d_ctx), we
  split the flip mass by the *worst material dimension* max_h(1 − c_h), mirroring Φ_gate: a single
  materially incompatible condition is enough to block a contradiction. Both versions are features.
* **Unknown ≠ incompatible.** Conflicting edge mass is split three ways — under *known-compatible*,
  *known-incompatible* and *unknown* material context (explicit missingness masks, the same
  matched / incompatible / unknown split the certificate's K field reports). A contradiction needs a
  conflict with no known material difference; a known difference means heterogeneity; mostly unknown
  context routes to incomparability or abstention.
* **Benchmark realism fix (made after a sanity check, before final results).** The first benchmark
  version stated all five context fields in every document, which taught the model that *any* unstated
  field argues against contradiction — so a same-context sign flip with no stated environment abstained.
  Real abstracts state environment only ~25% of the time (ManConCorpus). Base documents now omit 0–2 of
  dose / time / environment (40% / 40% / 20%); population and design are always stated, and the shifted
  dimension of every heterogeneity pair is stated in both documents.
* **Comparability guards** (`ccst/relation/guards.py`, applied after classification, no retraining):
  any label other than *incomparable* needs ≥ 1 aligned relation; contradiction / heterogeneity need an
  aligned *conflicting* relation (a non-empty certificate F — proposal Sec. 2.2 stage 3); an analogy needs
  ≥ 2 aligned relations and compatible signs (< 25% conflicting mass — Sec. 4.4 and 7.2 "no loose
  metaphors"). They exist because a one-relation graph aligns perfectly with any other one-relation graph,
  and because a zero-shot classifier can extrapolate to "contradiction" when almost nothing aligns.
* **Graph roles** (action / mediator / outcome) come from topology; population and other scope
  conditions live on claims (compared through c_h) rather than as extra nodes, to avoid double counting.
* **LDA α** = min(50/K, 1) — the Griffiths–Steyvers 50/K is too smooth for ~150-token abstracts.
* **BERTopic.** The corpus report's "BERTopic-style" row is our PCA re-implementation (always, for stable
  meaning); the official package (UMAP) is evaluated separately in item 6 and is far stronger (NMI 0.80 vs 0.36 on the 20-label corpus).
* **VisPy viewer** (stretch goal) and the privacy calibration (stretch goal, Sec. 6.3) are not implemented.

## 6. Known limitations

* **The controlled benchmark is ours.** Generator and extractor were written by the same team, so
  extraction on synthetic text is near-perfect (edge recall ≈ 1.0, sign accuracy ≈ 0.99); the
  noise-robustness curve and the SciFact / ManConCorpus results are the external checks.
* **Topology-derived roles** can disagree when one abstract reports only part of a chain (a node is an
  *outcome* in a one-edge abstract but a *mediator* in a longer one); λ_r is a soft penalty, but such
  pairs may be called incomparable for weak alignment rather than for missing context.
* **Rule-based scope parsing** covers population / dose / time / environment / design with lexicons;
  conditions outside these fields go unnoticed ("other material condition" is not implemented).
* **Zero-shot transfer** of the synthetic-trained relation model to real abstracts is a domain shift;
  real-corpus relation edges are candidates for review, not ground truth (see the annotation sheet).

## 7. Repository layout

```
ccst/
  extraction/   svo.py relations.py context.py concepts.py graph_builder.py
  transport/    ccst.py (the proposed method)  baselines.py (FGW, FNGW, LinearFGW, GED, Hungarian, ...)
  relation/     features.py rules.py classifier.py
  topics/       lda_gibbs.py preprocess.py coherence.py cluster.py
  index/        candidates.py
  synthetic/    lexicons.py generator.py
  data/         pubmed.py arxiv.py claims_datasets.py
  eval/         metrics.py baselines_text.py
  experiments/  synthetic.py synthetic_analysis.py real_pairs.py corpus.py scaling.py common.py
  viz/          plots.py maps.py
  certificate.py compat.py corpus_graph.py summarize.py graph.py embed.py cli.py config.py
configs/default.yaml   tests/   Makefile   requirements.txt
```
