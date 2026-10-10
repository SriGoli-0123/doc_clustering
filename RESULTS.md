# CCST — results

All numbers come from the final run of the pipeline on an Apple M4 (16 GB), seed 13. Every table is
reproduced in `results/*/REPORT.md`; figures sit next to them. CIs are 95% bootstrap intervals; *p* values
are paired bootstrap tests against CCST unless stated otherwise.

## TL;DR

Two controlled benchmarks are reported: **v1** (clean template prose; isolates the method) and **v2**
(pre-registered realistic difficulty — statistics, effect sizes, nominalised and hedged claims,
paraphrased scope; see `docs/benchmark_v2_preregistration.md`). v1 numbers are an upper bound; v2 is the
more realistic estimate. Headline: CCST 0.950 (v1) / **0.737** (v2) macro-F1 vs NLI 0.542 / 0.504.

| Claim from the proposal | Evidence | Verdict |
|---|---|---|
| Context gating cuts false contradictions without hiding real ones (RQ2) | v1: **0.0%** false-contradiction rate on context shifts at 97% contradiction recall · v2: **11.5%** at 66% recall (NLI 14.5% at 26%, SBERT 49.5%) | **Supported**, smaller margin on realistic text |
| CCST beats embedding and ordinary graph distances on structurally matched mechanisms (RQ1) | v1 macro-F1 **0.950** vs 0.851 · v2 **0.737** vs 0.726 (context clf. w/o transport, p = 0.08) and 0.504 (NLI); with nested-CV tuning for every method: v1 0.959 vs 0.858, v2 **0.753 vs 0.727 (p = 0.001)**; analogy retrieval MRR 0.93 (v1) / 0.57 (v2) vs FGW 0.20 / 0.16 | **Supported** (on v2 only once the transport mode is selected by nested CV) |
| Partial transport is more robust to extra unrelated mechanisms | 98% of extra-mechanism mass left unmatched vs 1% (forced full) / 0% (FGW); node-alignment F1 0.99 vs 0.72 / 0.70 | **Supported** |
| Signed comparison localises contradictions | flip precision 0.99, recall 1.00, top-1 localisation 0.995; removing the sign costs 12 macro-F1 points | **Supported** |
| Generalised concepts and relations help | raw verbs instead of generalised relations: −14 points; raw nodes: −0.6 points | **Supported (relations); weak (concepts)** |
| Certificates are correct, minimal, faithful (RQ3) | evidence IoU 1.00, minimality ratio 1.03, deleting cited evidence changes the decision 75% vs 33% for a random sentence | **Supported** |
| Calibrated abstention | v1: answers 99.7% of pairs at a 10% risk target · v2: only 4.4% | **Supported on v1 only** |
| Gains carry over to real expert-labelled data | ManConCorpus (claim-focused): CCST 0.563 vs NLI 0.526 (p = 0.03) with the original extractor, but **0.539 (p = 0.22) after the item-4 extraction fixes** — not robust; a context classifier *without* transport scores 0.596; SciFact: CCST 0.395 vs NLI 0.684, hybrid 0.684 | **Not robustly** — context modelling helps on ManCon; structural transport does not transfer |
| Zero-shot discovery on a real corpus | relation links are produced and certified, but top analogies are mostly structural coincidences | **Not yet** — needs real labels (annotation sheet provided) |
| Embedding clusters match subjects better than LDA; LDA topics easier to label (RQ5) | 20 labels: K-Means(SBERT) NMI 0.81 and **official BERTopic 0.80** vs LDA 0.63; with stop-words handled LDA has the *lowest* C_v (0.66 vs 0.69–0.71) but the most diverse topic words | **First half supported; "easier to label" not supported** |
| Sparse candidates turn O(N²) into ≈ O(N k) (RQ4) | log-log slope 2.02 (exact) vs 1.12 (sparse); 1.3% of pairs at N = 2,000 (35 s vs ≈ 1,900 s) — recall of exact top-5 neighbours 0.55 at 5% of pairs, **0.64 with the learned combined index (item 5)** | **Supported, with a recall cost** |

## 1. Controlled benchmark (strata 1 + 4) — `results/synthetic/`

2,400 abstracts, 2,672 pairs, 50 mechanism templates × 6 domains, grouped 5-fold CV by template;
identical LR + calibration for every method.

| method | macro-F1 [95% CI] | contradiction F1 | heterogeneity F1 | analogy F1 | false-contradiction rate (context shift) | contradiction recall |
|---|---|---|---|---|---|---|
| **CCST (ours)** | **0.950** [0.942, 0.958] | 0.964 | 1.000 | 0.915 | **0.000** | 0.972 |
| CCST + NLI (hybrid) | 0.950 [0.941, 0.958] | 0.958 | 1.000 | 0.907 | 0.000 | 0.968 |
| Context classifier w/o transport | 0.851 [0.839, 0.864] | 0.889 | 1.000 | 0.771 | 0.000 | 0.905 |
| Signed matcher w/o context | 0.795 [0.781, 0.808] | 0.743 | 0.653 | 0.938 | 0.030 | 0.632 |
| CCST ordered rules | 0.740 [0.724, 0.758] | 0.672 | 0.878 | 0.810 | 0.000 | 0.562 |
| FNGW (full transport, edge features) | 0.681 [0.662, 0.696] | 0.395 | 0.548 | 0.955 | 0.188 | 0.318 |
| Role-constrained matching | 0.680 [0.664, 0.696] | 0.439 | 0.495 | 0.887 | 0.278 | 0.385 |
| Graph edit distance | 0.564 [0.548, 0.581] | 0.320 | 0.430 | 0.957 | 0.040 | 0.257 |
| NLI (BART-MNLI) | 0.542 [0.524, 0.559] | 0.370 | 0.507 | 0.775 | 0.150 | 0.317 |
| SBERT (MiniLM) | 0.481 [0.463, 0.499] | 0.509 | 0.340 | 0.777 | 0.512 | 0.697 |
| Raw SVO overlap | 0.449 [0.432, 0.463] | 0.413 | 0.426 | 0.777 | 0.177 | 0.403 |
| SPECTER | 0.439 [0.420, 0.456] | 0.533 | 0.320 | 0.777 | 0.480 | 0.797 |
| BM25 | 0.421 [0.403, 0.436] | 0.368 | 0.410 | 0.779 | 0.350 | 0.382 |
| FGW (POT) | 0.408 [0.391, 0.424] | 0.093 | 0.418 | 0.758 | 0.072 | 0.058 |
| Purpose/mechanism embeddings | 0.402 [0.385, 0.416] | 0.043 | 0.428 | 0.777 | 0.025 | 0.023 |
| LinearFGW | 0.396 [0.378, 0.409] | 0.016 | 0.425 | 0.776 | 0.010 | 0.008 |
| LDA topic similarity | 0.376 [0.360, 0.393] | 0.155 | 0.356 | 0.777 | 0.102 | 0.108 |

Every baseline differs from CCST at p = 0.002 (paired bootstrap); the CCST+NLI hybrid does not (p = 0.36).
Adversarial "quoted opposite claim" pairs ("Previous studies suggested that X decreases Y…" inside an
otherwise agreeing abstract) are called contradictions **1.5%** of the time by CCST vs 31% (context
classifier w/o transport), 72% (BM25), 88% (SBERT) and 97% (SPECTER) — the epistemic-status handling.

**Ablations** (macro-F1; false-contradiction rate in brackets): full 0.950 [0.00] · −context, all
dimensions 0.797 [0.03] · −relation generalisation 0.811 · −edge sign 0.833 · −design 0.890 [0.13] ·
−population 0.922 [0.16] · −time 0.928 [0.12] · −dose 0.930 [0.11] · −environment 0.940 [0.09] ·
−concept generalisation 0.944 · −edge types 0.946 · −epistemic status 0.948 · −partial transport 0.948
(contradiction F1 0.925 vs 0.964) · −edge direction 0.953 · −context gate 0.953 · −confidence weights 0.955.
*Honest reading:* the context **features**, signs and relation generalisation carry the result; the gate
term Φ_gate inside the objective, edge direction and confidence weighting add nothing measurable here.

**Sensitivity.** λ_u × α grid: macro-F1 0.938–0.961 (defaults kept, not re-tuned on test folds); scaling
any single context weight ×0.5 / ×2: 0.950–0.953.

**Robustness to extraction errors** (claims dropped / signs flipped / context lost at rate p):
CCST 0.950 → 0.821 → 0.730 → 0.682 for p = 0 / 0.1 / 0.2 / 0.3, above the signed matcher without context
(0.795 → 0.477), FNGW (0.681 → 0.402) and the text-only NLI baseline (0.542) at every level.

**Alignment (RQ1).** Node-alignment F1 ≈ 0.97–0.99 for every aligner on paraphrase / synonym / sign-flip /
context-shift pairs; with an extra unrelated mechanism CCST 0.986 vs forced-full 0.719 / FGW 0.699; on
cross-domain analogies CCST 0.873 (forced full 0.887) vs FGW 0.491 / Hungarian 0.716.

**Certificates (RQ3)**, 200 held-out contradiction / heterogeneity pairs: cited-flip precision 0.988,
recall 1.000, minimality ratio 1.03, top-1 localisation 0.995, evidence-span IoU 1.00; the changed context
field is reported as incompatible in 100% of heterogeneity pairs and no field is falsely reported
incompatible in contradiction pairs; deleting the cited evidence lowers the predicted label's probability
by 0.69 on average (0.23 for a random claim sentence).

**Cross-domain analogy retrieval:** MRR 0.933 / Recall@5 0.83 (CCST signed mechanism) vs 0.201 / 0.10
(FGW), 0.114 / 0.06 (BM25), 0.093 / 0.03 (SBERT), 0.081 / 0.03 (SPECTER), 0.061 / 0.02 (LDA θ).

**Budgeted review (Step G).** Random sampling stays the best cold-start policy (0.93 macro-F1 at 130
labels vs full-pool 0.97); k-center diversity is close; uncertainty, DiverseTopB, curvature and
expected-change are worse. Curvature is not a useful acquisition signal on this benchmark.

**Mechanism families (Step H).** Communities on the CCST relation graph recover gold families with NMI
0.85 vs 0.56 (Leiden on SBERT kNN) and 0.66 (spectral). Adding CCST's negative constraints to a semantic
kNN graph does *not* help (0.556 vs 0.561).

**Comparability guards** (post-classification, used for real-corpus discovery and `ccst compare`): on the
benchmark's out-of-fold predictions they change 61 / 2,672 labels — analogy precision 0.900 → 0.953 but
analogy recall 0.93 → 0.82 (macro-F1 0.950 → 0.939). The recall loss comes from symmetric motifs (A → Y₁,
A → Y₂): T* is computed with the *unsigned* objective, so the solver may swap Y₁ and Y₂ at equal cost and
create apparent sign conflicts in true analogies. Sign-aware tie-breaking is the next fix.

## 1b. Benchmark v2 — realistic difficulty (pre-registered) — `results/synthetic_v2/`

2,800 abstracts, 3,072 pairs (adds `contra_ctxpara`: contradiction with paraphrased scope, and
`agree_nested`: agreement with a narrowed population). Design and rates were fixed in
`docs/benchmark_v2_preregistration.md` before any v2 number existed; nothing was tuned on v2.

**Extraction quality vs gold:** relation recall **0.80** (v1 0.998), sign accuracy 0.99; scope fields
detected: population 0.95, dose 0.80, time 1.00, environment 0.99, design 0.90.

| method | macro-F1 [95% CI] | contradiction F1 | heterogeneity F1 | analogy F1 | false-contradiction rate (context shift) | contradiction recall |
|---|---|---|---|---|---|---|
| CCST + NLI (hybrid) | **0.751** [0.734, 0.766] | 0.699 | 0.819 | 0.798 | 0.135 | 0.693 |
| **CCST (ours)** | 0.737 [0.723, 0.752] | 0.699 | 0.841 | 0.784 | **0.115** | 0.657 |
| Context classifier w/o transport | 0.726 [0.710, 0.743] | 0.700 | 0.844 | 0.776 | 0.115 | 0.711 |
| CCST ordered rules | 0.612 | 0.609 | 0.571 | 0.695 | 0.098 | 0.514 |
| Signed matcher w/o context | 0.527 | 0.433 | 0.358 | 0.766 | 0.235 | 0.361 |
| NLI (BART-MNLI) | 0.504 [0.488, 0.520] | 0.327 | 0.446 | 0.773 | 0.145 | 0.263 |
| Role-constrained matching | 0.483 | 0.449 | 0.286 | 0.768 | 0.307 | 0.385 |
| FNGW (full transport, edge features) | 0.471 | 0.273 | 0.390 | 0.773 | 0.138 | 0.205 |
| SBERT (MiniLM) | 0.436 [0.421, 0.452] | 0.500 | 0.185 | 0.777 | 0.495 | 0.629 |
| BM25 | 0.422 | 0.467 | 0.230 | 0.778 | 0.443 | 0.537 |
| SPECTER | 0.417 | 0.507 | 0.285 | 0.777 | 0.412 | 0.661 |
| Raw SVO overlap | 0.412 | 0.428 | 0.261 | 0.777 | 0.448 | 0.490 |
| Graph edit distance | 0.394 | 0.341 | 0.295 | 0.775 | 0.260 | 0.334 |
| FGW (POT) | 0.386 | 0.164 | 0.328 | 0.764 | 0.095 | 0.109 |
| LDA topic similarity | 0.383 | 0.283 | 0.256 | 0.777 | 0.217 | 0.235 |
| Purpose/mechanism embeddings | 0.380 | 0.154 | 0.331 | 0.777 | 0.100 | 0.101 |
| LinearFGW | 0.373 | 0.115 | 0.281 | 0.777 | 0.068 | 0.070 |

*Reading.* CCST still beats every text and graph-distance baseline by a wide margin (all p = 0.002),
but on realistic text it is **statistically tied with the context classifier without transport**
(p = 0.08), and the CCST+NLI hybrid is numerically best. False contradictions on context shifts rise to
11.5% (NLI 14.5%, SBERT 49.5%). Most contradiction misses are extraction misses: when the flipped claim is
written as a nominalisation ("A 23% reduction in Y was observed with X") it is not extracted and the pair
looks like agreement. Per variant, CCST is weakest on unrelated cross-domain pairs that share a small
graph shape (35% correct; called "analogy") and on lexically similar but structurally different pairs (51%).

*Ablations (v2).* −context (all) 0.531 · −design 0.636 · −edge sign 0.646 · −relation generalisation 0.649
· −population 0.706 · −time 0.711 · −environment 0.714 · −dose 0.727 · −edge types 0.727 · −concept
generalisation 0.725 · −direction 0.734 · −gate 0.735 · full 0.737 · −epistemic 0.741 · −confidence
weights 0.747 · **−partial transport 0.751**. Forcing full transport *helps* on v2: with 20% of relations
missed by extraction, partial transport leaves too much mass unmatched. This reverses the v1 finding and
is reported as such.

*Other v2 analyses.* Node-alignment F1 drops to 0.82–0.88 (extra mechanism: CCST 0.78 vs forced-full 0.62 /
FGW 0.65; cross-domain analogy 0.67 vs FGW 0.42); extra-mechanism mass left unmatched 70% (forced full 4%,
FGW 0%). Certificates: cited-flip precision 0.66, recall 0.67, top-1 localisation 0.66, evidence IoU 0.96,
changed context field reported in 80% of heterogeneity pairs; deleting cited evidence changes the decision
65% vs 9% for a random sentence. Analogy retrieval MRR 0.57 (FGW 0.16, SBERT 0.10). Mechanism families
NMI 0.78 vs 0.58 (SBERT kNN). Robustness: CCST 0.737 → 0.659 → 0.609 → 0.558 at extra corruption
p = 0.1 / 0.2 / 0.3, still above NLI (0.504) up to p = 0.2. **Calibrated abstention largely fails on v2**:
at a 10% risk target CCST answers only 4.4% of pairs (selective error 10.3%). **The comparability guards
are costly under noisy extraction**: macro-F1 0.737 → 0.665, heterogeneity recall 0.83 → 0.52, analogy
recall 0.88 → 0.48 (analogy precision 0.71 → 0.90) — they need the flipped relation to be extracted.

## 1c. Hyper-parameter selection by nested cross-validation (item 2) — `results/tuning/`

Protocol fixed beforehand in `docs/tuning_protocol.md`: outer = the headline 5-fold GroupKFold by
template; inner = 4-fold GroupKFold inside each outer training portion; CCST searches 12 transport
configurations × 3 LR strengths, and — for fairness — the five strongest baselines get the same nested
search over LR strength. Each outer test fold is scored once.

| method | v1 default | **v1 nested-tuned** [95% CI] | v2 default | **v2 nested-tuned** [95% CI] |
|---|---|---|---|---|
| **CCST (ours)** | 0.950 | **0.959** [0.951, 0.967] | 0.737 | **0.753** [0.739, 0.770] |
| Context classifier w/o transport | 0.851 | 0.858 | 0.726 | 0.727 |
| Signed matcher w/o context | 0.795 | 0.794 | 0.527 | 0.529 |
| NLI (BART-MNLI) | 0.542 | 0.549 | 0.504 | 0.514 |
| SBERT (MiniLM) | 0.481 | 0.488 | 0.436 | 0.436 |
| SPECTER | 0.439 | 0.459 | 0.417 | 0.427 |

Tuned CCST beats every tuned baseline at p ≈ 0.001 (paired bootstrap, 1,000 resamples) on **both**
benchmarks — including the context classifier without transport on v2, which the default configuration
only tied (p = 0.08). Selected configurations are stable: on v1 all five folds choose partial transport with
λ_u = 1.0; on v2 all five folds choose **forced full transport with α = 2**. *Reading:* partial transport is
the right choice when extraction is complete, but when ~20% of relations are missed it leaves too much
mass unmatched, and full transport with a stronger edge term is more robust. `configs/realistic.yaml`
records the v2 choice; the default config is unchanged. Tuned v2 false-contradiction rate 12.2%,
contradiction recall 0.71 (default: 11.5%, 0.66).

## 1d. Sign-aware tie-breaking in the alignment (item 3) — `results/tiebreak/`

Protocol and decision rule fixed beforehand in `docs/tiebreak_protocol.md`: add the sign-conflict terms to
the alignment objective with weight ε = 0.05 (fixed a priori; 0.02 / 0.1 shown for information only).
ε = 0 reproduces the published numbers exactly (0.950 / 0.737), confirming the comparison is like-for-like.

| benchmark | ε | macro-F1 [95% CI] | analogy recall (guarded) | analogy precision (guarded) | contradiction recall | false-contradiction rate | mean spurious sign gap on analogy pairs |
|---|---|---|---|---|---|---|---|
| v1 | 0 | 0.950 [0.942, 0.958] | 0.820 | 0.953 | 0.972 | 0.000 | 0.062 |
| v1 | **0.05** | **0.958** [0.951, 0.965] | **0.910** | 0.958 | 0.973 | 0.000 | 0.019 |
| v2 | 0 | 0.737 [0.723, 0.752] | 0.483 | 0.901 | 0.657 | 0.115 | 0.054 |
| v2 | **0.05** | **0.743** [0.729, 0.758] | **0.530** | 0.893 | 0.660 | 0.117 | 0.027 |

All four pre-registered checks pass on both benchmarks, so ε = 0.05 is now the default. *Reading:* on clean
text tie-breaking removes most spurious sign conflicts (−69%) and recovers most of the analogy recall the
guards had cost (0.82 → 0.91). On realistic text the gain is real but small (0.48 → 0.53): there the guards
mostly block analogies because extraction finds fewer than two aligned relations, which is an extraction
problem (item 4), not a tie problem. A new property test (`test_tiebreak_removes_spurious_flips_in_symmetric_motif`)
shows the failure mode — without tie-breaking a symmetric fork aligns crossed with a spurious gap of 0.6 —
and `test_tiebreak_keeps_genuine_flip` shows a real flip is still localised. The headline tables above were
computed with ε = 0; the final consistent re-run (item 9) propagates ε = 0.05 to every stage.

## 1e. Real-text parsing fixes (item 4) — `results/extraction_fixes/`

Protocol `docs/parsing_fixes_protocol.md` and a held-out phrasing set were written before any fix; no rule
was changed after seeing held-out results. Fixes: age vs duration, quantities inside noun phrases,
nominalised claims ("a 23% reduction in Y was observed with X"), intransitive change verbs ("Y rose after
X" — this also fixed a direction bug where "Y increased with X" was read as Y → X), thousands separators and
dose rates ("1,000 IU", "40 mg/day"), and the proposal's "other material condition" scope field.

* **Held-out set** (110 claim sentences, new vocabulary and forms): claims correctly extracted 0.355 → **0.809**
  (sign accuracy of extracted claims 1.00 both times); age read correctly 0.44 → 1.00 (n = 9); age-only
  sentences misread as durations 5 → 1 (of 6); dose 0.40 → 1.00 (n = 5); duration unchanged 0.875 (n = 8). Four
  templates still fail and are reported rather than patched.
* **Real PubMed:** abstracts whose duration actually came from an age phrase **280 → 87** (−69%).
* **Benchmarks** (not held out): v2 relation recall 0.802 → 0.900; v1 unchanged (0.996).
* **Downstream on real expert labels: no improvement.** ManConCorpus claim-focused CCST 0.563 → 0.539
  [0.515, 0.561], and the earlier "CCST beats NLI (p = 0.03)" result no longer holds (p = 0.22); full abstracts
  0.545 → 0.547; SciFact 0.398 → 0.395. A better extractor alone does not move real-pair accuracy here.

## 1f. Candidate-recall improvement (item 5) — `results/candidates/`

Protocol `docs/candidate_recall_protocol.md` fixed beforehand. New index: a concept-histogram sketch (K = 256
k-means codebook over node embeddings, role-split, TF-IDF) plus one cosine kNN over weighted, normalised sketches;
weights ∈ {0, 0.5, 1, 2}⁴ learned from exact CCST neighbours on a **training** subset of 500 real abstracts
(no labels needed), then scored once on the **disjoint** 500-abstract evaluation subset (124,750 exact pairs).

| recall of exact top-5 CCST neighbours at matched cost | 2% of pairs | 3% | 5% |
|---|---|---|---|
| original index (separate kNN lists + topic blocking) | 0.297 | 0.417 | 0.549 |
| original index without blocking | 0.266 | 0.360 | 0.489 |
| **combined index, learned weights** | **0.353** | **0.485** | **0.642** |
| combined index, uniform weights | 0.259 | 0.361 | 0.510 |

*Reading.* At equal cost the learned combined index recovers 6–9 points more of the exact neighbours. The
gain comes from *learning* the sketch weights (uniform weights are worse than the original); learned weights
are mechanism 2.0, semantic 1.0, concept histogram 0.5, context 0 — the new concept histogram is only a minor
contributor. About a third of exact top-5 neighbours are still missed at 5% of pairs, so the index remains an
approximation. The original index stays the default in the corpus pipeline; the combined index is available in
`ccst/index/candidates.py`.

## 1g. Official BERTopic baseline (items 6–7) — `results/bertopic/`

Protocol `docs/bertopic_protocol.md`: official `bertopic` 0.17.4 defaults (UMAP → HDBSCAN → c-TF-IDF); only
`min_topic_size` (= our HDBSCAN minimum cluster size, 40) and the UMAP seed were set. Final corpus: 9,986
abstracts, 20 subject labels (item 7 added arXiv `q-bio.NC`); 19-label values in brackets.

| method | NMI | ARI | topics | outliers | C_v | NPMI | topic diversity |
|---|---|---|---|---|---|---|---|
| K-Means SBERT (k = 20, true label count given) | **0.813** [0.845] | 0.725 | 20 | 0% | **0.708** | 0.108 | 0.775 |
| **Official BERTopic, SBERT (outliers reduced)** | 0.797 [0.815] | **0.726** | 34 | 0% | — | — | — |
| Official BERTopic, SBERT (raw) | 0.768 [0.785] | 0.563 | 35 | 16.3% | 0.692* | 0.102* | 0.768* |
| K-Means SPECTER (k = 20) | 0.766 [0.798] | 0.651 | 20 | 0% | 0.687 | 0.098 | 0.785 |
| Official BERTopic, SPECTER (outliers reduced / raw) | 0.756 / 0.699 | 0.621 / 0.404 | 40 / 41 | 0% / 19.1% | 0.705* | 0.114* | 0.675* |
| LDA Gibbs (K* = 20) | 0.631 [0.633] | 0.482 | 20 | 0% | 0.661 | 0.113 | **0.865** |
| LDA variational (K* = 20) | 0.500 | 0.264 | 20 | 0% | — | — | — |
| our BERTopic-style re-implementation (PCA) | 0.361 [0.365] | 0.098 | 2 | 0% | 0.530 | 0.013 | 1.000 |

\* topic words re-derived with BERTopic's documented stop-word removal (post hoc, representation only;
cluster assignments unchanged). With the **default** representation 43% (SBERT) / 37% (SPECTER) of BERTopic's
topic words are stop-words, and its apparent C_v of 0.76 / 0.75 came from scoring only the remaining words.

*Reading.* (1) Our earlier "BERTopic-style" re-implementation was a poor stand-in (two clusters, NMI 0.36);
the official model with UMAP reaches NMI 0.80 and finds 34–41 topics without being told the label count.
(2) Embedding clustering beats LDA at recovering subjects (NMI 0.76–0.81 vs 0.63), confirming that half of the
proposal's hypothesis. (3) The other half — "LDA topics are easier to label" — is **not supported**: with
stop-words handled, LDA has the *lowest* C_v of the topic-word methods (0.661 vs 0.69–0.71); it only has the most
diverse topic words. (4) Adding `q-bio.NC` lowers every embedding method's NMI by ≈0.02–0.03 — neuroscience
abstracts overlap with the other q-bio and cs.NE categories — while LDA is essentially unchanged.

## 2. Real expert-labelled pairs (stratum 3) — `results/real_pairs/`

> Numbers in this section were produced with the extractor *before* item 4; the post-fix values are in
> section 1e (ManCon claim-focused CCST 0.539, p = 0.22 vs NLI). Item 9 refreshes this section.

| dataset | CCST | CCST + NLI | context clf. w/o transport | NLI | SBERT | BM25 |
|---|---|---|---|---|---|---|
| ManConCorpus, claim-focused (1,774 pairs, binary) | 0.563 [0.537, 0.585] | 0.560 | **0.595** | 0.526 | 0.528 | 0.497 |
| ManConCorpus, full abstract | 0.545 | 0.539 | **0.576** | 0.526 | 0.528 | 0.497 |
| SciFact (1,259 pairs, 3-way) | 0.398 | **0.689** | 0.403 | 0.684 | 0.503 | 0.518 |

*Reading.* On ManConCorpus the context-aware methods beat text similarity and NLI (CCST vs NLI p = 0.03
claim-focused; p = 0.14 on full abstracts), but most of the gain comes from context modelling, not from
structural transport. SciFact is claim verification with short nominal claims (only 65% of documents
yield an extractable mechanism): NLI wins clearly and the hybrid matches it (p = 0.26).

Zero-shot transfer of the synthetic-trained 5-way model to ManConCorpus is weak (heterogeneity predicted
for 3.3% of conflict pairs vs 2.4% of agreement pairs), but individual certificates are exactly the kind
the proposal motivates, e.g.

> **HETEROGENEITY (p = 0.96)** — *oral L-arginine had no effect on diastolic BP* (12 g, 2 days) vs
> *prolonged dietary supplementation with L-arginine decreased BP* (3 g). Materially different context:
> dose (12 g vs 3 g); time (2 days vs prolonged).

## 3. Corpus (RQ5, Steps F/H/I/J) — `results/corpus/`

9,986 PubMed + arXiv abstracts, 20 subject labels (used only for evaluation). 19-label values (before item 7)
are kept in `results/old/corpus_metrics_19labels.json`.

* **Topics.** Collapsed-Gibbs LDA C_v = 0.632 / 0.661 / 0.661 / 0.637 for K = 10 / 20 / 30 / 50 → K* = 20 (with 19
  labels K* was 10); gensim's variational LDA at K* reaches only C_v 0.548.
* **Clustering vs subject labels (NMI / ARI):** K-Means SBERT 0.813 / 0.725 · K-Means SPECTER 0.766 / 0.651 ·
  official BERTopic 0.797 / 0.726 (section 1g) · HDBSCAN SBERT 0.709 / 0.231 · LDA Gibbs K* = 20 0.631 / 0.482 ·
  LDA variational 0.500 / 0.264 · our PCA HDBSCAN / BERTopic-style on SPECTER 0.36 / 0.10 (two clusters).
* **Summaries** (ROUGE-1 / BERTScore-F1 vs held-out abstracts, embedding clusters): TextRank + centroid + MMR
  0.240 / 0.756 · random sentences 0.195 / 0.741 · first sentences 0.150 / 0.729; same ordering on LDA clusters
  (0.219 / 0.184 / 0.130 ROUGE-1).
* **Maps.** t-SNE trustworthiness 0.98 (SPECTER, 2D), 0.995 (LDA θ), 0.989 (SPECTER, 3D); perplexity barely
  matters (15–50).
* **Discovery** (provisional — the zero-shot relation model predates items 3–4 and is refreshed in item 9):
  2,000-abstract stratified subset, 24,865 candidate pairs (1.2% of 2.0 M), 3.2 ms per CCST problem; predictions
  2,113 agreement, 5,041 analogy, 438 heterogeneity, 0 contradiction, 17,272 incomparable; the comparability
  guards blocked 6,920 analogies. Surviving analogies are still mostly structural coincidences between noisy real
  claim graphs, so the relation-link map shows review candidates, not findings; `results/corpus/annotation_sheet.csv`
  (200 blind pairs) is the next step.

## 4. Scaling (RQ4) — `results/scaling/`

Measured single-threaded on real PubMed/arXiv claim graphs; exact all-pairs measured up to N = 500 and
extrapolated with that measured per-pair rate.

| N docs | all pairs | sparse candidate pairs | sparse total (s) | exact total (s) |
|---|---|---|---|---|
| 125 | 7,750 | 1,182 (15%) | 1.6 | 7.2 |
| 250 | 31,125 | 2,691 (8.6%) | 3.5 | 29.7 |
| 500 | 124,750 | 5,865 (4.7%) | 7.9 | 118.9 |
| 1,000 | 499,500 | 12,413 (2.5%) | 16.5 | ≈ 476 (extrapolated) |
| 2,000 | 1,999,000 | 26,219 (1.3%) | 34.9 | ≈ 1,905 (extrapolated) |

* Fitted log-log slopes: **exact 2.02** (quadratic, as theory predicts) vs **sparse 1.12** (≈ linear,
  O(N log N + N k c_CCST)); candidate pairs per document grow only from 9.5 to 13.1.
* Per-pair cost c_CCST vs graph size n: 1.1 / 2.0 / 4.6 / 19 / 201 ms for n = 4 / 8 / 16 / 32 / 64 (slope
  1.83; Frank–Wolfe converges in ~3–12 iterations, so the O(n³) per-iteration bound is not reached at small n).
* **Candidate recall of the exact top-5 CCST neighbours** (N = 500): 0.24 / 0.41 / 0.55 / 0.69 at k = 2 / 5 /
  10 / 20, i.e. solving 1.6% / 2.8% / 4.7% / 8.1% of all pairs; topic blocking removes ~30% of the pairs for
  2–5 recall points. The sparse index is far cheaper but misses a sizeable share of exact neighbours at the
  default k = 10 — raise `index.k_*` when recall matters.

## 5. What changed during development (for the report's honesty section)

1. **Benchmark realism.** The first benchmark stated all five context fields in every document; the
   model then treated any unstated field as evidence against contradiction (a same-context flip with no
   stated environment abstained). Fixed by omitting 0–2 non-decisive fields per base document and
   splitting conflict mass into known-compatible / known-incompatible / unknown context.
2. **Comparability guards.** A discovery test on real abstracts showed (a) one-relation graphs align with
   any other one-relation graph, inflating "analogy", and (b) the zero-shot classifier extrapolating to
   "contradiction" on pairs where nothing aligned. Guards now require ≥ 1 aligned relation for any
   non-incomparable label, an aligned conflicting relation for contradiction / heterogeneity, and ≥ 2
   aligned relations with compatible signs for analogy.
3. **Extraction bugs.** A real-text filter for reporting phrases ("data suggest…") also dropped compounds
   such as "data exfiltration"; Unicode µ vs μ in dose units crashed the real-data stage. Both fixed with
   regression tests, and every stage was re-run.
4. **ManConCorpus** is evaluated both claim-focused (graph from the annotated claim, scope from the whole
   abstract) and on full abstracts; both are reported.
5. **arXiv `q-bio.NC`** was first blocked by an HTTP 429 rate limit (19 labels, 9,486 abstracts); it was added in
   item 7, giving the final 20-label, 9,986-abstract corpus. Embedding-clustering NMI drops by ≈0.03 as a result.

## 6. Next steps

1. Sign-aware tie-breaking in the alignment (small λ_s in the alignment objective) to remove the
   symmetric-motif analogy errors the guard currently pays for.
2. Annotate the 200 natural PubMed pairs (`results/corpus/annotation_sheet.csv`, Cohen's κ via
   `python -m ccst kappa`) and train / calibrate the relation model on real pairs instead of zero-shot transfer.
3. A scientific parser (scispaCy) and an "other material condition" context field for real abstracts.
