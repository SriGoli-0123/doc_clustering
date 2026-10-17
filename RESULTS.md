# CCST — results

**Official numbers:** the full `make all` run on ASU SOL (1× NVIDIA A100, 8 CPU cores), 2026-10-17, code at
commit `3821eb0`, seed 13. It is the first run in which *every* stage uses the final code (items 1–7 below).
The raw outputs — every `REPORT.md`, `metrics.json` and figure — are on the GitHub branch **`sol-results`**
(`git fetch origin sol-results`). CIs are 95% bootstrap intervals; *p* values are paired bootstrap tests
against CCST unless stated otherwise. Section 9 explains how these numbers differ from the earlier local
(Apple M4) development runs.

## TL;DR

Two controlled benchmarks are reported: **v1** (clean template prose; isolates the method) and **v2**
(pre-registered realistic difficulty: statistics, effect sizes, nominalised and hedged claims, paraphrased
scope; `docs/benchmark_v2_preregistration.md`). v1 is an upper bound; v2 is the more realistic estimate.

| Claim from the proposal | Evidence (SOL run) | Verdict |
|---|---|---|
| Context gating cuts false contradictions without hiding real ones (RQ2) | False-contradiction rate on 400 context-shift pairs: v1 **0.0%** (97% contradiction recall); v2 **14.2%** (78% recall) vs NLI 16.0% (27% recall), SBERT 48.5% | **Supported**; on realistic text the margin over NLI is small, but CCST finds 3× more real contradictions |
| CCST beats embedding and graph-distance baselines (RQ1) | macro-F1 v1 **0.957** vs 0.834 best baseline; v2 **0.799** vs 0.762 (p = 0.002); NLI 0.539 / 0.503, SBERT 0.480 / 0.435; cross-domain analogy retrieval MRR 0.95 / 0.67 vs FGW 0.21 / 0.18 | **Supported** on both controlled benchmarks |
| Partial transport is robust to extra unrelated mechanisms | extra-mechanism mass left unmatched 99% (v1) / 86% (v2) vs 1–4% for forced full transport and 0% for FGW; a role-constrained Hungarian matcher does about as well (97% / 76%); forcing full transport does not lower overall macro-F1 on v2 (0.799 vs 0.799) | **Supported vs transport baselines**; not unique to CCST; no net classification benefit on v2 |
| Signed comparison localises contradictions | cited-flip precision / recall 0.998 / 1.000 (v1), 0.742 / 0.750 (v2); removing the sign costs 13 (v1) / 12 (v2) macro-F1 points | **Supported** |
| Generalised concepts and relations help | raw verbs: −14 (v1) / −12 (v2) points; raw concept nodes: −1.2 / −1.3 points | **Supported (relations); small (concepts)** |
| Certificates are correct, minimal, faithful (RQ3) | deleting the cited evidence changes the decision 84% (v1) / 70% (v2) of the time vs 25% / 12% for a random sentence; evidence-span IoU 1.00 / 0.97 | **Supported** |
| Calibrated abstention | at a 10% risk target CCST answers 99.8% of v1 pairs but only 27% of v2 pairs (NLI / SBERT: 0%) | **Supported on v1; weak on realistic text** |
| Gains carry over to real expert-labelled data | ManConCorpus (claim-focused): CCST 0.527 vs NLI 0.502 (p = 0.07), below FNGW 0.552 and the context classifier w/o transport 0.592; SciFact: NLI 0.680 ≫ CCST 0.391 | **Not supported** — context modelling helps on ManCon; structural transport does not transfer |
| Zero-shot discovery on a real corpus | relation links are produced and certified, but surviving analogies are mostly structural coincidences | **Not yet** — needs real labels (annotation sheet provided) |
| Embedding clusters match subjects better than LDA; LDA topics easier to label (RQ5) | K-Means(SBERT) NMI 0.83, official BERTopic 0.81 vs LDA 0.64–0.66; C_v: BERTopic 0.70, K-Means labels 0.69, LDA 0.67 | **First half supported; "easier to label" not supported** |
| Sparse candidates turn O(N²) into ≈ O(N k) (RQ4) | log-log slope 2.06 (exact) vs 1.13 (sparse); at N = 2,000 sparse takes 106 s vs ≈ 6,040 s exact; recall of exact top-5 neighbours 0.57 at 5% of pairs, 0.65 with the learned combined index | **Supported, with a recall cost** |

## 1. Benchmark v1 — clean controlled text — `results/synthetic/`

2,400 abstracts, 2,672 pairs, 50 mechanism templates × 6 domains, grouped 5-fold CV by template; the same
LR + calibration for every method. Extraction vs gold: relation recall 0.996, sign accuracy 0.989 (clean
templates; scope fields always detected).

| method | macro-F1 [95% CI] | contradiction F1 | heterogeneity F1 | analogy F1 | false-contradiction rate (context shift) | contradiction recall |
|---|---|---|---|---|---|---|
| **CCST (ours)** | **0.957** [0.949, 0.964] | 0.967 | 0.997 | 0.936 | **0.000** | 0.972 |
| CCST + NLI (hybrid) | 0.952 [0.944, 0.959] | 0.958 | 0.999 | 0.926 | 0.000 | 0.957 |
| Context classifier w/o transport | 0.834 [0.820, 0.847] | 0.840 | 1.000 | 0.769 | 0.000 | 0.897 |
| Signed matcher w/o context | 0.798 [0.783, 0.811] | 0.751 | 0.656 | 0.935 | 0.028 | 0.637 |
| CCST ordered rules | 0.755 | 0.674 | 0.893 | 0.864 | 0.000 | 0.563 |
| FNGW (full transport, edge features) | 0.690 | 0.426 | 0.554 | 0.959 | 0.198 | 0.350 |
| Role-constrained matching | 0.682 | 0.444 | 0.498 | 0.890 | 0.273 | 0.387 |
| Graph edit distance | 0.567 | 0.314 | 0.428 | 0.967 | 0.035 | 0.250 |
| NLI (BART-MNLI) | 0.539 [0.522, 0.557] | 0.374 | 0.498 | 0.777 | 0.165 | 0.322 |
| SBERT (MiniLM) | 0.480 [0.460, 0.497] | 0.510 | 0.328 | 0.777 | 0.510 | 0.697 |
| Raw SVO overlap | 0.445 | 0.407 | 0.428 | 0.777 | 0.177 | 0.398 |
| SPECTER | 0.444 | 0.530 | 0.319 | 0.777 | 0.480 | 0.788 |
| BM25 | 0.424 | 0.376 | 0.407 | 0.776 | 0.362 | 0.395 |
| FGW (POT) | 0.407 | 0.053 | 0.424 | 0.766 | 0.020 | 0.030 |
| Purpose/mechanism embeddings | 0.402 | 0.034 | 0.429 | 0.777 | 0.020 | 0.018 |
| LinearFGW | 0.397 | 0.016 | 0.425 | 0.777 | 0.010 | 0.008 |
| LDA topic similarity | 0.379 | 0.155 | 0.354 | 0.777 | 0.105 | 0.108 |

Every baseline differs from CCST at p = 0.002; the CCST+NLI hybrid is *significantly worse* than CCST alone
(p = 0.018). "Quoted opposite claim" pairs ("Previous studies suggested that X decreases Y…" inside an
agreeing abstract) are called contradictions 1.5% of the time by CCST vs 63% (context classifier w/o
transport), 72% (BM25), 86% (SBERT) and 97% (SPECTER). Calibration: ECE 0.016; at a 10% risk target CCST
answers 99.8% of pairs (selective error 4.2%), the context classifier 73%, NLI / SBERT none.

**Ablations** (macro-F1; false-contradiction rate in brackets): full 0.957 [0.00] · −context, all dimensions
0.798 [0.03] · −relation generalisation 0.817 · −edge sign 0.831 · −design 0.894 [0.14] · −population 0.930
[0.17] · −time 0.935 [0.12] · −dose 0.938 [0.11] · −environment 0.941 [0.09] · −concept generalisation 0.945 ·
−partial transport 0.946 (contradiction F1 0.920 vs 0.967) · −edge types 0.947 · −epistemic status 0.953 ·
−context gate 0.957 · −confidence weights 0.957 · −"other" context field 0.958 · −edge direction 0.962.
*Honest reading:* context features, signs and relation generalisation carry the result; the gate term,
confidence weighting, edge direction and the new "other" field add nothing measurable here.

**Sensitivity.** λ_u × α grid 0.940–0.966 (defaults kept); any single context weight ×0.5 / ×2: 0.956–0.959.
**Robustness to extraction errors** (claims dropped / signs flipped / context lost at rate p = 0 / 0.1 / 0.2 /
0.3): CCST 0.957 → 0.842 → 0.756 → 0.655, above the signed matcher without context (0.798 → 0.468), FNGW
(0.690 → 0.385) and NLI (0.539, text-only) at every level.

**Analyses (v1).** Node alignment F1 with an extra unrelated mechanism: CCST 0.989 vs forced-full 0.720 /
FGW 0.699, but role-constrained Hungarian matching is as good (0.983); on cross-domain analogies 0.926
(forced full 0.938) vs FGW 0.496 / Hungarian 0.723. Extra-mechanism mass left unmatched 98.9% (forced full
1.4%, FGW 0%, Hungarian 96.8%). Certificates (200 held-out contradiction / heterogeneity pairs):
cited-flip precision 0.998, recall 1.000, minimality ratio 1.005, top-1 localisation 1.000, evidence-span IoU
1.00, changed context field reported in 100%; deleting cited evidence changes the decision 83.5% vs 24.6%
for a random sentence. Cross-domain analogy retrieval MRR 0.951 / Recall@5 0.907 vs FGW 0.209 / 0.105, BM25
0.114, SBERT 0.093, SPECTER 0.081, LDA θ 0.061. Mechanism families (Step H): NMI 0.849 on the CCST relation
graph vs 0.561 (Leiden on SBERT kNN) and 0.657 (spectral); the negative (signed) edges add nothing — Leiden
on the positive CCST edges alone gives the same 0.849, and adding CCST's negative constraints to the semantic
graph does not help either (0.559). Budgeted review (Step G): random sampling is the best cold-start
policy (0.966 at 130 labels; full pool 0.98); uncertainty 0.951, others 0.943, curvature-augmented 0.926.
Comparability guards change 21 / 2,672 labels (macro-F1 0.957 → 0.956; analogy precision 0.925 → 0.955,
recall 0.947 → 0.913).

## 2. Benchmark v2 — realistic text (pre-registered) — `results/synthetic_v2/`

2,800 abstracts, 3,072 pairs (adds contradictions with paraphrased scope and agreements with narrowed
populations). Extraction vs gold: relation recall **0.900**, sign accuracy 0.985; scope fields detected:
population 0.95, dose 0.80, time 1.00, environment 1.00, design 0.90. *Caveat:* the item-4 parsing fixes
(section 5) handle nominalised claims that v2 also contains, so v2 extraction (0.80 → 0.90) is not a held-out
measurement; the held-out phrasing set in section 5 is.

| method | macro-F1 [95% CI] | contradiction F1 | heterogeneity F1 | analogy F1 | false-contradiction rate (context shift) | contradiction recall |
|---|---|---|---|---|---|---|
| CCST + NLI (hybrid) | **0.804** [0.788, 0.818] | 0.778 | 0.865 | 0.810 | 0.145 | 0.774 |
| **CCST (ours)** | 0.799 [0.783, 0.814] | 0.788 | 0.864 | 0.807 | 0.142 | 0.780 |
| Context classifier w/o transport | 0.762 [0.747, 0.777] | 0.754 | 0.870 | 0.775 | 0.133 | 0.794 |
| CCST ordered rules | 0.637 | 0.610 | 0.639 | 0.745 | 0.115 | 0.507 |
| Signed matcher w/o context | 0.599 | 0.478 | 0.429 | 0.816 | 0.193 | 0.385 |
| FNGW (full transport, edge features) | 0.543 | 0.361 | 0.421 | 0.808 | 0.203 | 0.290 |
| Role-constrained matching | 0.534 | 0.367 | 0.357 | 0.785 | 0.260 | 0.292 |
| NLI (BART-MNLI) | 0.503 [0.486, 0.519] | 0.339 | 0.441 | 0.771 | 0.160 | 0.271 |
| SBERT (MiniLM) | 0.435 [0.420, 0.452] | 0.501 | 0.188 | 0.777 | 0.485 | 0.631 |
| Graph edit distance | 0.428 | 0.317 | 0.272 | 0.840 | 0.190 | 0.282 |
| BM25 | 0.424 | 0.470 | 0.235 | 0.778 | 0.435 | 0.542 |
| Raw SVO overlap | 0.422 | 0.423 | 0.308 | 0.777 | 0.367 | 0.456 |
| SPECTER | 0.417 | 0.504 | 0.284 | 0.777 | 0.412 | 0.654 |
| FGW (POT) | 0.399 | 0.159 | 0.330 | 0.772 | 0.110 | 0.104 |
| LinearFGW | 0.392 | 0.150 | 0.323 | 0.777 | 0.072 | 0.092 |
| Purpose/mechanism embeddings | 0.390 | 0.120 | 0.342 | 0.777 | 0.072 | 0.074 |
| LDA topic similarity | 0.385 | 0.285 | 0.260 | 0.777 | 0.210 | 0.239 |

*Reading.* CCST beats every text and graph-distance baseline and the context classifier without transport
(p = 0.002); the hybrid is numerically best but not significantly better than CCST. False contradictions on
context shifts are 14.2% — far below SBERT (48.5%) but only slightly below NLI (16.0%), while CCST finds 78%
of real contradictions vs NLI's 27%. Quoted opposite claims are called contradictions 7.5% of the time by CCST
vs 59% (context classifier w/o transport), 71% (SPECTER) and 78% (SBERT). Calibrated abstention is weak on v2:
at a 10% risk target CCST answers 27% of pairs (hybrid 48%, context classifier 19%, NLI / SBERT none).

*Ablations (v2).* −context (all) 0.601 · −edge sign 0.675 · −relation generalisation 0.676 · −design 0.722
(false-contradiction rate 0.28) · −population 0.775 · −time 0.777 · −environment 0.779 · −concept generalisation
0.786 · −edge direction 0.793 · −dose 0.793 · full 0.799 · −partial transport 0.799 · −"other" field 0.799 ·
−confidence weights 0.798 · −gate 0.800 · −edge types 0.802 · −epistemic status 0.803. As on v1, the gate,
confidence weights, direction and "other" field add nothing; edge types and epistemic status are neutral here.

*Analyses (v2).* Node alignment F1 with an extra mechanism: CCST 0.872 vs forced-full 0.679 / FGW 0.680,
Hungarian 0.873; cross-domain analogies 0.754 (forced full 0.842, FGW 0.456, Hungarian 0.639);
extra-mechanism mass left unmatched 86% (forced full 4%, FGW 0%, Hungarian 76%). Certificates: cited-flip precision 0.742, recall 0.750, top-1 localisation 0.742,
evidence IoU 0.968, changed context field reported in 75% of heterogeneity pairs; deleting cited evidence
changes the decision 70.4% vs 12.0% for a random sentence. Analogy retrieval MRR 0.665 (FGW 0.184, BM25 0.121,
SBERT 0.101). Mechanism families NMI 0.79–0.80 vs 0.58 (SBERT kNN); again positive-only edges do as well as
signed ones (0.795 vs 0.790). Robustness: CCST 0.799 → 0.698 → 0.618 → 0.567 at extra corruption p = 0.1 / 0.2 /
0.3, still above NLI (0.503) at p = 0.3. **Comparability guards are costly
on realistic text:** 281 labels change, macro-F1 0.799 → 0.768, analogy recall 0.873 → 0.643 (precision 0.751 →
0.910), heterogeneity recall 0.815 → 0.645 — they need the flipped relation to be extracted.

## 3. Hyper-parameter selection by nested CV (item 2) — `results/tuning/`

Protocol `docs/tuning_protocol.md`: outer = the headline 5-fold split; inner = 4-fold split inside each outer
training portion; CCST searches 12 transport configurations × 3 LR strengths, and the five strongest
baselines get the same nested search over LR strength (fairness).

| method | v1 default | **v1 nested-tuned** | v2 default | **v2 nested-tuned** |
|---|---|---|---|---|
| **CCST (ours)** | 0.957 | **0.963** [0.956, 0.970] | 0.799 | **0.798** [0.784, 0.813] |
| Context classifier w/o transport | 0.834 | 0.838 | 0.762 | 0.760 |
| Signed matcher w/o context | 0.798 | 0.802 | 0.599 | 0.596 |
| NLI (BART-MNLI) | 0.539 | 0.552 | 0.503 | 0.515 |
| SBERT (MiniLM) | 0.480 | 0.487 | 0.435 | 0.437 |
| SPECTER | 0.444 | 0.458 | 0.417 | 0.427 |

Tuned CCST beats every tuned baseline at p ≈ 0.001 on both benchmarks. Tuning gains are small (+0.006 on v1,
none on v2), i.e. the a-priori defaults were not leaving much on the table. Selected configurations: on v1
partial transport in all 5 folds (α = 2 in 4 of them); on v2 forced full transport in 4 of 5 folds — partial
transport helps on clean text but not when extraction misses relations (`configs/realistic.yaml`).

## 4. Sign-aware tie-breaking in the alignment (item 3) — `results/tiebreak/`

Protocol `docs/tiebreak_protocol.md`: adds the sign-conflict terms to the *alignment* objective with weight
ε = 0.05 (fixed a priori). In symmetric motifs (A → Y₁, A → Y₂) the unsigned alignment can swap Y₁/Y₂ and
create spurious sign conflicts; a property test checks that tie-breaking removes them while keeping a genuine
flip. The mean spurious signed–unsigned gap on analogy pairs falls from 0.064 to 0.019 (v1) and 0.053 to 0.030 (v2).

| benchmark | ε | macro-F1 | analogy recall with guards |
|---|---|---|---|
| v1 | 0 | 0.956 | 0.827 |
| v1 | **0.05** | 0.957 | **0.913** |
| v2 | 0 | 0.798 | 0.600 |
| v2 | **0.05** | 0.799 | **0.643** |

The pre-registered decision rule passes on both benchmarks (re-confirmed on SOL), so ε = 0.05 is the
default. It recovers most of the guard-induced analogy loss on clean text and part of it on realistic text,
where the rest of the loss is due to extraction misses.

## 5. Real-text parsing fixes (item 4)

Protocol `docs/parsing_fixes_protocol.md` and a held-out phrasing set (`ccst/eval/heldout_extraction.py`,
new vocabulary and sentence forms) were written before any fix. Fixes: age vs duration, quantities inside
noun phrases, nominalised claims, intransitive change verbs (this also fixed a direction bug where "Y
increased with X" was read as Y → X), thousands separators and dose rates, and the proposal's "other
material condition" scope field. The held-out check is deterministic and gives the same numbers on SOL.

| held-out check | n | before | after |
|---|---|---|---|
| claims correctly extracted (right pair, right sign) | 110 | 0.355 | **0.809** |
| age group read correctly | 9 | 0.44 | 1.00 |
| age-only sentences misread as a duration | 6 | 5 | 1 |
| duration read correctly | 8 | 0.875 | 0.875 |
| dose read correctly | 5 | 0.40 | 1.00 |

The age and dose sets are small; their 1.00 is not evidence of perfection. Four held-out sentence forms still
fail and are reported rather than patched. On real PubMed abstracts, documents whose duration actually came
from an age phrase fell from 280 to 87 (−69%). Downstream, the fixes raise benchmark v2 substantially
(section 2) but do **not** improve the real expert-labelled pairs (section 7).

## 6. Learned candidate index (item 5) — `results/candidates/`

Protocol `docs/candidate_recall_protocol.md`: a concept-histogram sketch plus one weighted kNN over all
sketches, weights learned from exact CCST neighbours on 500 training abstracts, scored once on a disjoint set of
500 (124,750 exact pairs).

| recall of exact top-5 CCST neighbours at matched cost (linear interpolation on each index's k = 2…20 curve) | 2% of pairs | 3% | 5% |
|---|---|---|---|
| original index (separate kNN lists + topic blocking) | 0.280 | 0.408 | 0.571 |
| original index without blocking | 0.246 | 0.347 | 0.497 |
| **combined index, learned weights** | **0.361** | **0.496** | **0.651** |
| combined index, uniform weights | 0.268 | 0.387 | 0.544 |

At equal cost the learned index recovers 8–9 points more of the exact neighbours; with uniform weights it is
worse than the original, so the gain comes from learning the weights. Learned weights on SOL: mechanism 1.0,
semantic 0.5, concept histogram **0**, context 0 — the concept histogram added in item 5 contributes nothing.
About a third of exact top-5 neighbours are still missed at 5% of pairs.

## 7. Real expert-labelled pairs (stratum 3) — `results/real_pairs/`

| method | ManConCorpus, claim-focused (1,774 pairs, binary) | ManConCorpus, full abstract | SciFact (1,259 pairs, 3-way) |
|---|---|---|---|
| **CCST (ours)** | 0.527 [0.503, 0.549] | 0.518 | 0.391 |
| CCST + NLI (hybrid) | 0.513 | 0.511 | 0.669 |
| Context classifier w/o transport | **0.592** [0.569, 0.617] | **0.566** | 0.399 |
| FNGW (full transport, edge features) | 0.552 | 0.484 | 0.388 |
| Signed matcher w/o context | 0.537 | 0.510 | 0.400 |
| SBERT (MiniLM) | 0.524 | 0.524 | 0.465 |
| BM25 | 0.522 | 0.522 | 0.502 |
| NLI (BART-MNLI) | 0.502 | 0.502 | **0.680** |
| Raw SVO overlap | 0.494 | 0.294 | 0.194 |

*Reading.* On ManConCorpus every method is weak (≤ 0.59); the context classifier without transport is best,
CCST is not significantly better than NLI (p = 0.07) and sits below FNGW. SciFact is claim verification with
short nominal claims (only 67% of documents yield an extractable mechanism) — NLI wins clearly and the hybrid
nearly matches it. Earlier, a run with the pre-item-4 extractor showed "CCST beats NLI on ManCon, p = 0.03";
that result did not survive the extraction fixes and is withdrawn. Zero-shot transfer of the
synthetic-trained 5-way model is weak: heterogeneity is predicted for 9.8% of ManCon conflict pairs vs 8.5% of
agreement pairs. Individual certificates can still be useful, e.g. *oral L-arginine had no effect on diastolic
BP (12 g, 2 days)* vs *prolonged supplementation decreased BP (3 g)*, flagged as heterogeneity because dose and
time differ.

## 8. Corpus, topics, BERTopic, scaling (RQ4–RQ5) — `results/corpus/`, `results/bertopic/`, `results/scaling/`

9,986 PubMed + arXiv abstracts, 20 subject labels (used only for evaluation).

* **Topics.** Collapsed-Gibbs LDA C_v = 0.643 / 0.665 / 0.666 / 0.640 for K = 10 / 20 / 30 / 50 → K* = 30 (a near
  tie with K = 20); gensim's variational LDA at K* reaches C_v 0.49.
* **Clustering vs subject labels (NMI / ARI):**

  | method | NMI | ARI | topics | C_v | topic diversity |
  |---|---|---|---|---|---|
  | K-Means SBERT (k = 20, true label count given) | **0.831** | **0.758** | 20 | 0.691 | 0.780 |
  | official BERTopic, SBERT (outliers reduced / raw) | 0.807 / 0.785 | 0.731 / 0.629 | 29 / 30 | 0.704* | 0.759* |
  | K-Means SPECTER (k = 20) | 0.786 | 0.687 | 20 | 0.693 | 0.790 |
  | official BERTopic, SPECTER (outliers reduced / raw) | 0.749 / 0.695 | 0.615 / 0.403 | 44 / 45 | 0.691* | 0.705* |
  | HDBSCAN SBERT (ours, PCA) | 0.709 | 0.231 | 10 | — | — |
  | LDA Gibbs (K* = 30 / K = 20) | 0.664 / 0.643 | 0.571 / 0.507 | 30 / 20 | 0.666 | **0.857** |
  | LDA variational (K* = 30) | 0.424 | 0.206 | 30 | — | — |
  | our BERTopic-style re-implementation (PCA) | 0.361 | 0.098 | 2 | 0.534 | 1.000 |

  \* BERTopic topic words re-derived with its documented stop-word removal (post hoc, representation only;
  `docs/bertopic_protocol.md` addendum). With the default representation 48% (SBERT) / 36% (SPECTER) of
  BERTopic's topic words are stop-words; those topics score the *highest* C_v of all (0.77 / 0.74) because
  stop-words co-occur everywhere, so we do not count that as better topics. Raw BERTopic leaves 12% (SBERT) /
  19% (SPECTER) of documents as outliers. C_v and diversity use each method's top-10 words with the LDA dictionary.
  *Reading:* embedding clustering (including official BERTopic) recovers subjects much better than LDA; topic
  coherence is about the same for all methods (C_v 0.67–0.70), so "LDA topics are easier to label" is not
  supported — LDA only has the most diverse topic words. Our earlier PCA "BERTopic-style" stand-in badly
  understated BERTopic.
* **Summaries** (ROUGE-1 / BERTScore-F1 vs held-out abstracts, embedding clusters): TextRank + centroid + MMR
  0.226 / 0.755 · random sentences 0.187 / 0.737 · first sentences 0.143 / 0.725; same ordering on LDA clusters.
* **Maps.** t-SNE trustworthiness 0.98 (SPECTER, 2D), 0.99 (LDA θ), 0.99 (SPECTER, 3D); perplexity barely matters.
  Local structure is only partly kept: 42–51% of each document's 10 nearest neighbours survive in the map.
* **Discovery** on a stratified 2,000-abstract subset: 24,037 candidate pairs (1.2% of 2.0 M), 4.0 ms per CCST
  problem; zero-shot predictions 2,251 agreement, 4,898 analogy, 466 heterogeneity, 1 contradiction, 15,674
  incomparable, 747 abstentions (3.1%); the guards blocked 5,572 analogies. Surviving analogies are mostly
  structural coincidences between noisy real claim graphs, so the relation-link map shows review candidates,
  not findings; `results/corpus/annotation_sheet.csv` (200 blind pairs) is the next step.
* **Scaling.** Exact all-pairs time grows with log-log slope **2.06**, the sparse pipeline with **1.13**; at
  N = 500 exact takes 377 s vs 24 s sparse, and at N = 2,000 sparse takes 106 s vs ≈ 6,040 s extrapolated exact
  (≈ 57×). Per-pair CCST cost grows from 3.4 ms (4 nodes) to 502 ms (64 nodes), slope 1.76. Absolute times are
  about 3× those of the local Apple M4 run (slower CPU cores); the slopes are what matter. Candidate recall of exact top-5 neighbours: 0.39 / 0.56 / 0.72 at
  k = 5 / 10 / 20 with topic blocking (2.8% / 4.8% / 8.0% of pairs).

## 9. Reproducibility: local development runs vs the official SOL run

The SOL run reproduces the controlled-benchmark baselines closely (v1 NLI 0.542 → 0.539, SBERT 0.481 → 0.480).
Differences have three known sources:

1. **Code.** Earlier local tables were produced before items 3–4 (tie-breaking, parsing fixes, "other" scope
   field); SOL includes them everywhere. The largest changes are in the methods that use extracted claims and
   context: v1 CCST 0.950 → 0.957, v2 0.737 → 0.799, context classifier w/o transport 0.851 → 0.834 on v1.
   We did not re-run the old code on SOL, so code and environment effects are not separated number by number.
2. **Data.** The arXiv downloader takes the most recent papers per category, and the SOL download ran days
   after the local one, so the corpus contains partly different arXiv abstracts (K* moved from 20 to 30;
   clustering NMI shifted by ≈ ±0.02).
3. **Environment.** SOL used a fresh `pip install` (newer library versions) and CUDA instead of Apple MPS.
   A few real-pair baselines moved by more than numerical noise, e.g. SciFact SBERT 0.503 → 0.465 and ManCon
   BM25 0.497 → 0.522, NLI 0.526 → 0.502. Exact version pinning and a frozen data snapshot are listed as next
   steps.

## 10. What changed during development (for the report's honesty section)

1. **Benchmark realism.** The first benchmark stated every scope field, which taught the model that any
   unstated field argues against contradiction. Fixed by omitting 0–2 non-decisive fields per document and
   splitting conflict mass into known-compatible / known-incompatible / unknown context.
2. **Comparability guards.** A real-corpus discovery test showed one-relation graphs inflating "analogy" and
   the zero-shot classifier extrapolating to "contradiction" when nothing aligned. Guards now require aligned
   relations (and, for analogy, ≥ 2 relations with compatible signs). They cost recall on realistic text
   (section 2).
3. **Bugs found and fixed:** "data exfiltration" dropped by a reporting-phrase filter; Unicode µ vs μ dose
   units; an embedding-dimension mismatch in candidate generation; a `.gitignore` pattern that kept the
   `ccst/data` package out of the first push.
4. **Withdrawn claim.** "CCST beats NLI on ManConCorpus (p = 0.03)" — not robust to the extraction fixes.
5. **Items 1–7** (each with a protocol in `docs/` written before running): realistic benchmark v2; nested-CV
   tuning; sign-aware tie-breaking (adopted by its pre-registered rule); parsing fixes with a held-out set;
   learned candidate index; official BERTopic baseline; the 20th arXiv category (`q-bio.NC`).

## 11. Next steps

1. Direct LLM-judge baseline (item 8): implemented, needs a SOL API key (`--llm-pairs 500`).
2. Annotate the 200 natural PubMed pairs (`results/corpus/annotation_sheet.csv`, Cohen's κ via
   `python -m ccst kappa`) and train / calibrate the relation model on real pairs instead of zero-shot transfer.
3. Pin exact package versions (`pip freeze`) and save the arXiv / PubMed ID lists so the corpus can be
   re-downloaded identically.
4. Improve extraction on real abstracts (scientific parser, remaining held-out failure forms).
