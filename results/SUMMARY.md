# CCST results summary

_Generated 2026-10-17 05:45_

# Synthetic controlled-transformation benchmark (synthetic)

2400 documents, 2672 pairs, grouped 5-fold CV by template.

## Extraction quality vs gold (reported jointly with downstream results)

| edge_recall | sign_accuracy | gold_edges | ctx_detect_design | ctx_detect_dose | ctx_detect_environment | ctx_detect_population | ctx_detect_time |
|---|---|---|---|---|---|---|---|
| 0.996 | 0.989 | 7207 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Main comparison

| method | macro_f1 | macro_f1_lo | macro_f1_hi | accuracy | f1_contradiction | f1_heterogeneity | f1_analogy | FCR_context_shift | FCR_missing_context | contra_recall | ece | AURC | p_vs_ccst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CCST (ours) | 0.957 | 0.949 | 0.964 | 0.957 | 0.967 | 0.997 | 0.936 | 0.000 | 0.000 | 0.972 | 0.016 | 0.016 |  |
| CCST + NLI (hybrid) | 0.952 | 0.944 | 0.959 | 0.952 | 0.958 | 0.999 | 0.926 | 0.000 | 0.000 | 0.957 | 0.016 | 0.013 | 0.018 |
| Context clf. w/o transport | 0.834 | 0.820 | 0.847 | 0.831 | 0.840 | 1.000 | 0.769 | 0.000 | 0.000 | 0.897 | 0.047 | 0.050 | 0.002 |
| Signed matcher w/o context | 0.798 | 0.783 | 0.811 | 0.804 | 0.751 | 0.656 | 0.935 | 0.028 | 0.055 | 0.637 | 0.022 | 0.062 | 0.002 |
| CCST (ordered rules) | 0.755 | 0.739 | 0.769 | 0.717 | 0.674 | 0.893 | 0.864 | 0.000 | 0.000 | 0.563 |  |  | 0.002 |
| FNGW (full, edge features) | 0.690 | 0.673 | 0.703 | 0.691 | 0.426 | 0.554 | 0.959 | 0.198 | 0.210 | 0.350 | 0.041 | 0.120 | 0.002 |
| Role-constrained matching | 0.682 | 0.666 | 0.696 | 0.694 | 0.444 | 0.498 | 0.890 | 0.273 | 0.485 | 0.387 | 0.017 | 0.120 | 0.002 |
| Graph edit distance | 0.567 | 0.550 | 0.583 | 0.528 | 0.314 | 0.428 | 0.967 | 0.035 | 0.050 | 0.250 | 0.056 | 0.266 | 0.002 |
| NLI (BART-MNLI) | 0.539 | 0.522 | 0.557 | 0.556 | 0.374 | 0.498 | 0.777 | 0.165 | 0.245 | 0.322 | 0.024 | 0.307 | 0.002 |
| SBERT (MiniLM) | 0.480 | 0.460 | 0.497 | 0.469 | 0.510 | 0.328 | 0.777 | 0.510 | 0.000 | 0.697 | 0.077 | 0.445 | 0.002 |
| Raw SVO overlap | 0.445 | 0.429 | 0.461 | 0.429 | 0.407 | 0.428 | 0.777 | 0.177 | 0.235 | 0.398 | 0.055 | 0.465 | 0.002 |
| SPECTER | 0.444 | 0.425 | 0.462 | 0.443 | 0.530 | 0.319 | 0.777 | 0.480 | 0.000 | 0.788 | 0.064 | 0.451 | 0.002 |
| BM25 | 0.424 | 0.405 | 0.439 | 0.394 | 0.376 | 0.407 | 0.776 | 0.362 | 0.000 | 0.395 | 0.082 | 0.515 | 0.002 |
| FGW (POT) | 0.407 | 0.389 | 0.422 | 0.410 | 0.053 | 0.424 | 0.766 | 0.020 | 0.040 | 0.030 | 0.088 | 0.462 | 0.002 |
| Purpose/mechanism emb. | 0.402 | 0.385 | 0.417 | 0.420 | 0.034 | 0.429 | 0.777 | 0.020 | 0.020 | 0.018 | 0.073 | 0.447 | 0.002 |
| LinearFGW | 0.397 | 0.380 | 0.411 | 0.415 | 0.016 | 0.425 | 0.777 | 0.010 | 0.005 | 0.008 | 0.077 | 0.491 | 0.002 |
| LDA topic similarity | 0.379 | 0.362 | 0.394 | 0.363 | 0.155 | 0.354 | 0.777 | 0.105 | 0.120 | 0.108 | 0.045 | 0.527 | 0.002 |

## Ablations

| method | macro_f1 | macro_f1_lo | macro_f1_hi | FCR_context_shift | f1_contradiction | f1_heterogeneity | f1_analogy | p_vs_ccst |
|---|---|---|---|---|---|---|---|---|
| − edge direction | 0.962 | 0.955 | 0.969 | 0.000 | 0.969 | 0.997 | 0.942 | 1.000 |
| − context: other | 0.958 | 0.950 | 0.965 | 0.000 | 0.966 | 0.999 | 0.939 | 1.000 |
| − confidence / provenance weights | 0.957 | 0.948 | 0.964 | 0.000 | 0.970 | 0.999 | 0.931 | 1.000 |
| CCST (ours) | 0.957 | 0.949 | 0.964 | 0.000 | 0.967 | 0.997 | 0.936 |  |
| − context gate | 0.957 | 0.949 | 0.965 | 0.000 | 0.967 | 0.999 | 0.936 | 0.483 |
| − epistemic status | 0.953 | 0.945 | 0.961 | 0.000 | 0.961 | 0.999 | 0.938 | 0.026 |
| − edge types | 0.947 | 0.938 | 0.955 | 0.000 | 0.965 | 0.999 | 0.911 | 0.004 |
| − partial transport (forced full) | 0.946 | 0.937 | 0.954 | 0.000 | 0.920 | 0.996 | 0.945 | 0.006 |
| − concept generalisation (raw nodes) | 0.945 | 0.936 | 0.953 | 0.000 | 0.962 | 0.999 | 0.905 | 0.002 |
| − context: environment | 0.941 | 0.932 | 0.950 | 0.087 | 0.937 | 0.948 | 0.939 | 0.002 |
| − context: dose | 0.938 | 0.928 | 0.947 | 0.107 | 0.932 | 0.933 | 0.941 | 0.002 |
| − context: time | 0.935 | 0.924 | 0.944 | 0.120 | 0.930 | 0.928 | 0.933 | 0.002 |
| − context: population | 0.930 | 0.919 | 0.940 | 0.168 | 0.917 | 0.901 | 0.942 | 0.002 |
| − context: design | 0.894 | 0.884 | 0.907 | 0.135 | 0.845 | 0.801 | 0.941 | 0.002 |
| − edge sign | 0.831 | 0.818 | 0.845 | 0.000 | 0.716 | 1.000 | 0.926 | 0.002 |
| − relation generalisation (raw verbs) | 0.817 | 0.804 | 0.830 | 0.000 | 0.712 | 0.999 | 0.882 | 0.002 |
| − context (all dims) | 0.798 | 0.784 | 0.811 | 0.028 | 0.750 | 0.655 | 0.935 | 0.002 |

## Context-weight sensitivity

| dim | scale | macro_f1 | FCR_context_shift |
|---|---|---|---|
| population | 0.500 | 0.958 | 0.000 |
| population | 2.000 | 0.958 | 0.000 |
| dose | 0.500 | 0.957 | 0.000 |
| dose | 2.000 | 0.958 | 0.000 |
| time | 0.500 | 0.956 | 0.000 |
| time | 2.000 | 0.959 | 0.000 |
| environment | 0.500 | 0.957 | 0.000 |
| environment | 2.000 | 0.958 | 0.000 |
| design | 0.500 | 0.957 | 0.000 |
| design | 2.000 | 0.957 | 0.000 |
| other | 0.500 | 0.957 | 0.000 |
| other | 2.000 | 0.957 | 0.000 |

## Timings (s)

| stage | seconds |
|---|---|
| CCST family + ablations | 41.999 |
| baselines | 118.001 |

# Synthetic analyses

## RQ1 alignment (node-alignment F1 by variant)

| variant | CCST (partial) | CCST forced full | FGW (POT) | Hungarian (role-constrained) |
|---|---|---|---|---|
| + extra mechanism | 0.989 | 0.720 | 0.699 | 0.983 |
| context shift | 0.994 | 0.988 | 0.990 | 0.993 |
| cross-domain analogy | 0.926 | 0.938 | 0.496 | 0.723 |
| disjoint-vocabulary synonyms | 0.976 | 0.976 | 0.981 | 0.975 |
| paraphrase | 0.995 | 0.991 | 0.991 | 0.994 |
| sign flip | 0.992 | 0.988 | 0.989 | 0.991 |

Unmatched-mechanism accuracy (share of extra-mechanism mass left unmatched):

| method | unmatched_extra_accuracy | n |
|---|---|---|
| CCST (partial) | 0.989 | 400 |
| CCST forced full | 0.014 | 400 |
| FGW (POT) | 0.000 | 400 |
| Hungarian (role-constrained) | 0.968 | 400 |

## RQ3 certificates

| flip_precision | flip_recall | minimality_ratio | localised_top1 | evidence_iou | context_no_false_incompatibility | prob_drop_cited | prob_drop_random | context_field_correct | n | faithfulness_rate | faithfulness_rate_random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.998 | 1.000 | 1.005 | 1.000 | 1.000 | 1.000 | 0.768 | 0.184 | 1.000 | 200 | 0.835 | 0.246 |

## Cross-domain analogy retrieval

| method | MRR | Recall@1 | nDCG@1 | Recall@5 | nDCG@5 | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| SBERT | 0.093 | 0.012 | 0.035 | 0.032 | 0.031 | 0.057 | 0.042 |
| LDA θ | 0.061 | 0.003 | 0.010 | 0.023 | 0.018 | 0.048 | 0.029 |
| SPECTER | 0.081 | 0.008 | 0.025 | 0.030 | 0.026 | 0.062 | 0.039 |
| FGW (POT) | 0.209 | 0.028 | 0.085 | 0.105 | 0.094 | 0.207 | 0.139 |
| CCST signed mechanism | 0.951 | 0.312 | 0.935 | 0.907 | 0.908 | 0.947 | 0.927 |
| BM25 | 0.114 | 0.005 | 0.015 | 0.063 | 0.047 | 0.120 | 0.072 |

## Step H: mechanism families (signed vs unsigned clustering)

| method | NMI | ARI | purity | n_clusters |
|---|---|---|---|---|
| Signed Leiden (CCST relations) | 0.849 | 0.519 | 0.932 | 1447 |
| Correlation clustering (CCST relations) | 0.824 | 0.223 | 0.989 | 1581 |
| Leiden, positive edges only | 0.849 | 0.519 | 0.932 | 1447 |
| Leiden on SBERT kNN graph | 0.561 | 0.093 | 0.126 | 55 |
| Signed Leiden: SBERT kNN + CCST (+/-) | 0.559 | 0.091 | 0.125 | 55 |
| Leiden: SBERT kNN + CCST (+ only) | 0.559 | 0.090 | 0.127 | 56 |
| Spectral on SBERT (K = #families) | 0.657 | 0.133 | 0.330 | 198 |

Gold families: 198

## Comparability guards (post-classification)

Predictions changed: 21

| setting | macro_f1 | P_agreement | P_analogy | P_contradiction | P_heterogeneity | P_incomparable | R_agreement | R_analogy | R_contradiction | R_heterogeneity | R_incomparable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unguarded | 0.957 | 0.966 | 0.925 | 0.962 | 1.000 | 0.927 | 0.968 | 0.947 | 0.972 | 0.995 | 0.907 |
| guarded | 0.956 | 0.966 | 0.955 | 0.962 | 1.000 | 0.910 | 0.968 | 0.913 | 0.972 | 0.993 | 0.925 |

## Example certificates

- **t1_security_base vs t1_security_contra_flip**: Verdict: CONTRADICTION (p=1.00). 4 mechanism nodes align (S_mech=1.00). Matched context: population, dose, time, environment, design. Unknown context: other. Smallest disagreement: doc1 reports 'incident response time --inhibits--> false alarm rate' while doc2 reports 'incident response time --enables--> false alarm rate' (sign flip, 100% of the disagreement). Evidence: [t1_security_base] "Incident response time blocked false alarm rate." | [t1_security_contra_flip] "Incident response time induced false alarm rate."

- **t1_security_base vs t1_security_contra_null**: Verdict: CONTRADICTION (p=0.99). 4 mechanism nodes align (S_mech=1.00). Matched context: population, dose, time, environment, design. Unknown context: other. Smallest disagreement: doc1 reports 'incident response time --inhibits--> false alarm rate' while doc2 reports 'incident response time --no_effect--> false alarm rate' (exclusive assertion, 100% of the disagreement). Evidence: [t1_security_base] "Incident response time blocked false alarm rate." | [t1_security_contra_null] "Our results show that incident response time failed to inhibit false alarm rate."

- **t1_security_base vs t1_security_contra_extra**: Verdict: CONTRADICTION (p=1.00). 4 mechanism nodes align (S_mech=0.64). Matched context: population, dose, time, environment, design. Unknown context: other. Smallest disagreement: doc1 reports 'incident response time --inhibits--> false alarm rate' while doc2 reports 'incident response time --enables--> false alarm rate' (sign flip, 100% of the disagreement). Unmatched: honeypot deployment [doc2], malware propagation [doc2], lateral movement [doc2]. Evidence: [t1_security_base] "Incident response time blocked false alarm rate." | [t1_security_contra_extra] "Incident response time activated false alarm rate."

- **t1_security_base vs t1_security_hetero_ctx**: Verdict: HETEROGENEITY (p=1.00). 4 mechanism nodes align (S_mech=0.96). Matched context: population, dose, time, design. Materially different context: environment (testbed vs production deployments). Unknown context: other. Smallest disagreement: doc1 reports 'incident response time --inhibits--> false alarm rate' while doc2 reports 'incident response time --enables--> false alarm rate' (sign flip, 100% of the disagreement). Evidence: [t1_security_base] "Incident response time blocked false alarm rate." | [t1_security_hetero_ctx] "Our results show that false alarm rate was triggered by incident response time."


# Synthetic controlled-transformation benchmark (synthetic_v2)

2800 documents, 3072 pairs, grouped 5-fold CV by template.

## Extraction quality vs gold (reported jointly with downstream results)

| edge_recall | sign_accuracy | gold_edges | ctx_detect_design | ctx_detect_dose | ctx_detect_environment | ctx_detect_population | ctx_detect_time |
|---|---|---|---|---|---|---|---|
| 0.900 | 0.985 | 8407 | 0.899 | 0.800 | 0.997 | 0.953 | 0.998 |

## Main comparison

| method | macro_f1 | macro_f1_lo | macro_f1_hi | accuracy | f1_contradiction | f1_heterogeneity | f1_analogy | FCR_context_shift | FCR_missing_context | contra_recall | ece | AURC | p_vs_ccst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CCST + NLI (hybrid) | 0.804 | 0.788 | 0.818 | 0.804 | 0.778 | 0.865 | 0.810 | 0.145 | 0.020 | 0.774 | 0.026 | 0.091 | 1.000 |
| CCST (ours) | 0.799 | 0.783 | 0.814 | 0.799 | 0.788 | 0.864 | 0.807 | 0.142 | 0.000 | 0.780 | 0.058 | 0.119 |  |
| Context clf. w/o transport | 0.762 | 0.747 | 0.777 | 0.756 | 0.754 | 0.870 | 0.775 | 0.133 | 0.020 | 0.794 | 0.038 | 0.145 | 0.002 |
| CCST (ordered rules) | 0.637 | 0.618 | 0.654 | 0.612 | 0.610 | 0.639 | 0.745 | 0.115 | 0.000 | 0.507 |  |  | 0.002 |
| Signed matcher w/o context | 0.599 | 0.582 | 0.616 | 0.612 | 0.478 | 0.429 | 0.816 | 0.193 | 0.235 | 0.385 | 0.037 | 0.213 | 0.002 |
| FNGW (full, edge features) | 0.543 | 0.526 | 0.561 | 0.570 | 0.361 | 0.421 | 0.808 | 0.203 | 0.215 | 0.290 | 0.063 | 0.301 | 0.002 |
| Role-constrained matching | 0.534 | 0.517 | 0.550 | 0.557 | 0.367 | 0.357 | 0.785 | 0.260 | 0.290 | 0.292 | 0.052 | 0.264 | 0.002 |
| NLI (BART-MNLI) | 0.503 | 0.486 | 0.519 | 0.521 | 0.339 | 0.441 | 0.771 | 0.160 | 0.245 | 0.271 | 0.024 | 0.327 | 0.002 |
| SBERT (MiniLM) | 0.435 | 0.420 | 0.452 | 0.427 | 0.501 | 0.188 | 0.777 | 0.485 | 0.025 | 0.631 | 0.052 | 0.472 | 0.002 |
| Graph edit distance | 0.428 | 0.412 | 0.445 | 0.369 | 0.317 | 0.272 | 0.840 | 0.190 | 0.165 | 0.282 | 0.062 | 0.461 | 0.002 |
| BM25 | 0.424 | 0.409 | 0.439 | 0.405 | 0.470 | 0.235 | 0.778 | 0.435 | 0.170 | 0.542 | 0.048 | 0.497 | 0.002 |
| Raw SVO overlap | 0.422 | 0.406 | 0.438 | 0.393 | 0.423 | 0.308 | 0.777 | 0.367 | 0.390 | 0.456 | 0.054 | 0.499 | 0.002 |
| SPECTER | 0.417 | 0.401 | 0.433 | 0.408 | 0.504 | 0.284 | 0.777 | 0.412 | 0.015 | 0.654 | 0.054 | 0.478 | 0.002 |
| FGW (POT) | 0.399 | 0.384 | 0.416 | 0.377 | 0.159 | 0.330 | 0.772 | 0.110 | 0.090 | 0.104 | 0.053 | 0.544 | 0.002 |
| LinearFGW | 0.392 | 0.378 | 0.407 | 0.380 | 0.150 | 0.323 | 0.777 | 0.072 | 0.040 | 0.092 | 0.113 | 0.559 | 0.002 |
| Purpose/mechanism emb. | 0.390 | 0.375 | 0.406 | 0.373 | 0.120 | 0.342 | 0.777 | 0.072 | 0.045 | 0.074 | 0.084 | 0.541 | 0.002 |
| LDA topic similarity | 0.385 | 0.368 | 0.401 | 0.348 | 0.285 | 0.260 | 0.777 | 0.210 | 0.135 | 0.239 | 0.034 | 0.546 | 0.002 |

## Ablations

| method | macro_f1 | macro_f1_lo | macro_f1_hi | FCR_context_shift | f1_contradiction | f1_heterogeneity | f1_analogy | p_vs_ccst |
|---|---|---|---|---|---|---|---|---|
| − epistemic status | 0.803 | 0.788 | 0.818 | 0.142 | 0.787 | 0.866 | 0.818 | 1.000 |
| − edge types | 0.802 | 0.785 | 0.818 | 0.147 | 0.795 | 0.860 | 0.815 | 1.000 |
| − context gate | 0.800 | 0.784 | 0.815 | 0.140 | 0.785 | 0.862 | 0.818 | 1.000 |
| − partial transport (forced full) | 0.799 | 0.785 | 0.814 | 0.158 | 0.767 | 0.856 | 0.851 | 1.000 |
| CCST (ours) | 0.799 | 0.783 | 0.814 | 0.142 | 0.788 | 0.864 | 0.807 |  |
| − context: other | 0.799 | 0.782 | 0.814 | 0.145 | 0.785 | 0.861 | 0.811 | 0.489 |
| − confidence / provenance weights | 0.798 | 0.782 | 0.813 | 0.142 | 0.797 | 0.866 | 0.790 | 0.375 |
| − context: dose | 0.793 | 0.778 | 0.808 | 0.180 | 0.786 | 0.846 | 0.803 | 0.044 |
| − edge direction | 0.793 | 0.778 | 0.808 | 0.135 | 0.784 | 0.863 | 0.801 | 0.122 |
| − concept generalisation (raw nodes) | 0.786 | 0.770 | 0.801 | 0.142 | 0.780 | 0.859 | 0.790 | 0.006 |
| − context: environment | 0.779 | 0.764 | 0.795 | 0.198 | 0.773 | 0.798 | 0.803 | 0.002 |
| − context: time | 0.777 | 0.760 | 0.793 | 0.228 | 0.771 | 0.783 | 0.805 | 0.002 |
| − context: population | 0.775 | 0.758 | 0.790 | 0.225 | 0.770 | 0.771 | 0.807 | 0.002 |
| − context: design | 0.722 | 0.705 | 0.739 | 0.280 | 0.682 | 0.593 | 0.816 | 0.002 |
| − relation generalisation (raw verbs) | 0.676 | 0.659 | 0.693 | 0.058 | 0.504 | 0.863 | 0.792 | 0.002 |
| − edge sign | 0.675 | 0.658 | 0.690 | 0.117 | 0.554 | 0.858 | 0.805 | 0.002 |
| − context (all dims) | 0.601 | 0.585 | 0.618 | 0.210 | 0.494 | 0.422 | 0.815 | 0.002 |

## Context-weight sensitivity

| dim | scale | macro_f1 | FCR_context_shift |
|---|---|---|---|
| population | 0.500 | 0.801 | 0.145 |
| population | 2.000 | 0.797 | 0.140 |
| dose | 0.500 | 0.800 | 0.142 |
| dose | 2.000 | 0.800 | 0.145 |
| time | 0.500 | 0.800 | 0.142 |
| time | 2.000 | 0.799 | 0.142 |
| environment | 0.500 | 0.799 | 0.142 |
| environment | 2.000 | 0.799 | 0.142 |
| design | 0.500 | 0.799 | 0.145 |
| design | 2.000 | 0.798 | 0.145 |
| other | 0.500 | 0.800 | 0.140 |
| other | 2.000 | 0.799 | 0.145 |

## Timings (s)

| stage | seconds |
|---|---|
| CCST family + ablations | 48.444 |
| baselines | 90.652 |

# Synthetic analyses

## RQ1 alignment (node-alignment F1 by variant)

| variant | CCST (partial) | CCST forced full | FGW (POT) | Hungarian (role-constrained) |
|---|---|---|---|---|
| + extra mechanism | 0.872 | 0.679 | 0.680 | 0.873 |
| context shift | 0.932 | 0.918 | 0.929 | 0.935 |
| cross-domain analogy | 0.754 | 0.842 | 0.456 | 0.639 |
| disjoint-vocabulary synonyms | 0.893 | 0.908 | 0.926 | 0.927 |
| paraphrase | 0.924 | 0.897 | 0.919 | 0.937 |
| sign flip | 0.947 | 0.921 | 0.933 | 0.941 |

Unmatched-mechanism accuracy (share of extra-mechanism mass left unmatched):

| method | unmatched_extra_accuracy | n |
|---|---|---|
| CCST (partial) | 0.863 | 395 |
| CCST forced full | 0.039 | 395 |
| FGW (POT) | 0.000 | 395 |
| Hungarian (role-constrained) | 0.760 | 395 |

## RQ3 certificates

| flip_precision | flip_recall | minimality_ratio | localised_top1 | context_no_false_incompatibility | evidence_iou | prob_drop_cited | prob_drop_random | context_field_correct | n | faithfulness_rate | faithfulness_rate_random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.742 | 0.750 | 0.800 | 0.742 | 1.000 | 0.968 | 0.465 | 0.025 | 0.750 | 240 | 0.704 | 0.120 |

## Cross-domain analogy retrieval

| method | MRR | Recall@1 | nDCG@1 | Recall@5 | nDCG@5 | Recall@10 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| SBERT | 0.101 | 0.017 | 0.050 | 0.030 | 0.033 | 0.060 | 0.047 |
| LDA θ | 0.082 | 0.010 | 0.030 | 0.030 | 0.028 | 0.053 | 0.039 |
| SPECTER | 0.080 | 0.007 | 0.020 | 0.030 | 0.026 | 0.055 | 0.037 |
| FGW (POT) | 0.184 | 0.023 | 0.070 | 0.095 | 0.083 | 0.178 | 0.120 |
| CCST signed mechanism | 0.665 | 0.192 | 0.575 | 0.510 | 0.508 | 0.635 | 0.565 |
| BM25 | 0.121 | 0.010 | 0.030 | 0.065 | 0.054 | 0.100 | 0.070 |

## Step H: mechanism families (signed vs unsigned clustering)

| method | NMI | ARI | purity | n_clusters |
|---|---|---|---|---|
| Signed Leiden (CCST relations) | 0.790 | 0.321 | 0.800 | 1688 |
| Correlation clustering (CCST relations) | 0.801 | 0.239 | 0.947 | 1793 |
| Leiden, positive edges only | 0.795 | 0.342 | 0.822 | 1691 |
| Leiden on SBERT kNN graph | 0.584 | 0.107 | 0.158 | 70 |
| Signed Leiden: SBERT kNN + CCST (+/-) | 0.583 | 0.110 | 0.157 | 65 |
| Leiden: SBERT kNN + CCST (+ only) | 0.592 | 0.118 | 0.162 | 68 |
| Spectral on SBERT (K = #families) | 0.651 | 0.145 | 0.328 | 192 |

Gold families: 192

## Comparability guards (post-classification)

Predictions changed: 281

| setting | macro_f1 | P_agreement | P_analogy | P_contradiction | P_heterogeneity | P_incomparable | R_agreement | R_analogy | R_contradiction | R_heterogeneity | R_incomparable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unguarded | 0.799 | 0.836 | 0.751 | 0.797 | 0.918 | 0.691 | 0.838 | 0.873 | 0.780 | 0.815 | 0.705 |
| guarded | 0.768 | 0.854 | 0.910 | 0.819 | 0.935 | 0.572 | 0.823 | 0.643 | 0.774 | 0.645 | 0.864 |

## Example certificates

- **t1_security_base vs t1_security_contra_flip**: Verdict: AGREEMENT (p=0.82). 3 mechanism nodes align (S_mech=0.46). Matched context: population, time. Unknown context: environment, design, other. Strongest aligned relation: doc1 'packet loss --increases--> incident response time' <-> doc2 'packet loss --increases--> incident response time'. Unmatched: false alarm rate [doc1].

- **t1_security_base vs t1_security_contra_null**: Verdict: CONTRADICTION (p=0.77). 3 mechanism nodes align (S_mech=0.33). Matched context: population, time. Unknown context: environment, design, other. Smallest disagreement: doc1 reports 'incident response time --associated--> false alarm rate' while doc2 reports 'incident response time --no_effect--> false alarm rate' (exclusive assertion, 100% of the disagreement). Unmatched: packet loss [doc1], multi-factor authentication [doc2], reduction [doc2]. Evidence: [t1_security_base] "Our results show that incident response time was associated with a 40% lower false alarm rate." | [t1_security_contra_null] "Incident response time did not inhibit false alarm rate."

- **t1_security_base vs t1_security_contra_extra**: Verdict: CONTRADICTION (p=0.81). 4 mechanism nodes align (S_mech=0.40). Matched context: population, time. Unknown context: environment, design, other. Smallest disagreement: doc1 reports 'incident response time --associated--> false alarm rate' while doc2 reports 'incident response time --enables--> false alarm rate' (sign flip, 100% of the disagreement). Unmatched: multi-factor authentication [doc2], host [doc2], increases malware propagation [doc2]. Evidence: [t1_security_base] "Our results show that incident response time was associated with a 40% lower false alarm rate." | [t1_security_contra_extra] "False alarm rate was triggered by incident response time."

- **t1_security_base vs t1_security_hetero_ctx**: Verdict: HETEROGENEITY (p=0.97). 2 mechanism nodes align (S_mech=0.13). Unmatched: packet loss [doc1], incident response time [doc1], multi-factor authentication [doc2].


# Item 2 — hyper-parameter selection by nested cross-validation

Protocol: `docs/tuning_protocol.md` (written before running). Selection uses only the outer training portion; each outer test fold is scored once.

## Benchmark v1

| method | default_macro_f1 | nested_macro_f1 | ci_lo | ci_hi | gain | FCR_context_shift | contra_recall | p_tuned_ccst_better |
|---|---|---|---|---|---|---|---|---|
| CCST (ours) | 0.957 | 0.963 | 0.956 | 0.970 | 0.006 | 0.000 | 0.975 |  |
| Context clf. w/o transport | 0.834 | 0.838 | 0.824 | 0.850 | 0.004 | 0.000 | 0.895 | 0.001 |
| Signed matcher w/o context | 0.798 | 0.802 | 0.787 | 0.815 | 0.004 | 0.028 | 0.638 | 0.001 |
| NLI (BART-MNLI) | 0.539 | 0.552 | 0.535 | 0.570 | 0.013 | 0.165 | 0.315 | 0.001 |
| SBERT (MiniLM) | 0.480 | 0.487 | 0.469 | 0.503 | 0.007 | 0.380 | 0.608 | 0.001 |
| SPECTER | 0.444 | 0.458 | 0.440 | 0.476 | 0.014 | 0.417 | 0.747 | 0.001 |

Configuration chosen in each outer fold:

- **CCST (ours)**: partial λu=0.6 α=2 C=1; partial λu=0.6 α=2 C=10; partial λu=1 α=1 C=10; partial λu=0.6 α=2 C=10; partial λu=0.6 α=2 C=10
- **Context clf. w/o transport**: C=1; C=10; C=1; C=10; C=10
- **Signed matcher w/o context**: C=1; C=1; C=1; C=0.1; C=10
- **NLI (BART-MNLI)**: C=10; C=10; C=10; C=10; C=10
- **SBERT (MiniLM)**: C=10; C=10; C=1; C=10; C=10
- **SPECTER**: C=10; C=10; C=10; C=10; C=10

## Benchmark v2

| method | default_macro_f1 | nested_macro_f1 | ci_lo | ci_hi | gain | FCR_context_shift | contra_recall | p_tuned_ccst_better |
|---|---|---|---|---|---|---|---|---|
| CCST (ours) | 0.799 | 0.798 | 0.784 | 0.813 | -0.001 | 0.155 | 0.791 |  |
| Context clf. w/o transport | 0.762 | 0.760 | 0.745 | 0.776 | -0.001 | 0.133 | 0.792 | 0.001 |
| Signed matcher w/o context | 0.599 | 0.596 | 0.580 | 0.613 | -0.003 | 0.198 | 0.380 | 0.001 |
| NLI (BART-MNLI) | 0.503 | 0.515 | 0.498 | 0.532 | 0.012 | 0.160 | 0.274 | 0.001 |
| SBERT (MiniLM) | 0.435 | 0.437 | 0.422 | 0.453 | 0.002 | 0.472 | 0.616 | 0.001 |
| SPECTER | 0.417 | 0.427 | 0.412 | 0.444 | 0.010 | 0.385 | 0.620 | 0.001 |

Configuration chosen in each outer fold:

- **CCST (ours)**: full λu=0.6 α=1 C=10; partial λu=0.6 α=1 C=1; full λu=0.6 α=2 C=1; full λu=0.6 α=2 C=0.1; full λu=0.6 α=2 C=1
- **Context clf. w/o transport**: C=10; C=1; C=10; C=1; C=10
- **Signed matcher w/o context**: C=0.1; C=1; C=1; C=1; C=10
- **NLI (BART-MNLI)**: C=10; C=10; C=10; C=10; C=10
- **SBERT (MiniLM)**: C=10; C=1; C=10; C=1; C=1
- **SPECTER**: C=10; C=10; C=10; C=10; C=10


# Item 3 — sign-aware tie-breaking in the alignment

Protocol and decision rule: `docs/tiebreak_protocol.md` (written before running). ε = 0.05 fixed a priori; 0.02 and 0.1 are information only. ε = 0 reproduces the published numbers.

## Benchmark v1

| eps | macro_f1 | ci_lo | ci_hi | P_analogy | R_analogy | guarded_R_analogy | guarded_P_analogy | guarded_macro_f1 | R_contradiction | FCR_context_shift | guarded_R_heterogeneity | spurious_gap_analogy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.956 | 0.948 | 0.964 | 0.925 | 0.947 | 0.827 | 0.954 | 0.942 | 0.972 | 0.000 | 0.990 | 0.064 |
| 0.020 | 0.958 | 0.951 | 0.966 | 0.929 | 0.953 | 0.897 | 0.957 | 0.954 | 0.973 | 0.000 | 0.993 | 0.028 |
| 0.050 | 0.957 | 0.949 | 0.964 | 0.925 | 0.947 | 0.913 | 0.955 | 0.956 | 0.972 | 0.000 | 0.993 | 0.019 |
| 0.100 | 0.959 | 0.951 | 0.966 | 0.932 | 0.953 | 0.933 | 0.956 | 0.959 | 0.972 | 0.000 | 0.993 | 0.013 |

## Benchmark v2

| eps | macro_f1 | ci_lo | ci_hi | P_analogy | R_analogy | guarded_R_analogy | guarded_P_analogy | guarded_macro_f1 | R_contradiction | FCR_context_shift | guarded_R_heterogeneity | spurious_gap_analogy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.798 | 0.782 | 0.813 | 0.743 | 0.867 | 0.600 | 0.918 | 0.763 | 0.787 | 0.142 | 0.650 | 0.053 |
| 0.020 | 0.799 | 0.783 | 0.814 | 0.754 | 0.870 | 0.623 | 0.917 | 0.765 | 0.781 | 0.140 | 0.647 | 0.040 |
| 0.050 | 0.799 | 0.783 | 0.814 | 0.751 | 0.873 | 0.643 | 0.910 | 0.768 | 0.780 | 0.142 | 0.645 | 0.030 |
| 0.100 | 0.795 | 0.779 | 0.811 | 0.741 | 0.867 | 0.657 | 0.912 | 0.767 | 0.779 | 0.140 | 0.630 | 0.025 |

## Pre-registered decision

| benchmark | guarded_analogy_recall_improves | contradiction_recall_ok | fcr_ok | macro_f1_ok |
|---|---|---|---|---|
| v1 | True | True | True | True |
| v2 | True | True | True | True |

**Adopt ε = 0.05 as default: True**

# Item 5 — candidate-recall improvement

Protocol: `docs/candidate_recall_protocol.md` (written before running). Weights and codebook use the training subset only; numbers below are on the disjoint evaluation subset (N = 500, 124,750 exact pairs).

Learned weights: {'sem': 0.5, 'concept': 0.0, 'mech': 1.0, 'ctx': 0.0} (training recall@5 at k = 10: 0.498).

| index | k | pairs | pair_fraction | recall@5 | seconds |
|---|---|---|---|---|---|
| original + topic blocking | 2 | 2021 | 0.016 | 0.228 | 0.243 |
| original | 2 | 2471 | 0.020 | 0.244 | 0.336 |
| combined, learned weights | 2 | 1410 | 0.011 | 0.190 | 0.192 |
| combined, uniform weights | 2 | 1439 | 0.012 | 0.137 | 0.192 |
| original + topic blocking | 5 | 3541 | 0.028 | 0.394 | 0.278 |
| original | 5 | 4647 | 0.037 | 0.420 | 0.260 |
| combined, learned weights | 5 | 2222 | 0.018 | 0.329 | 0.317 |
| combined, uniform weights | 5 | 2355 | 0.019 | 0.255 | 0.191 |
| original + topic blocking | 10 | 5931 | 0.048 | 0.560 | 0.285 |
| original | 10 | 8493 | 0.068 | 0.605 | 0.274 |
| combined, learned weights | 10 | 3571 | 0.029 | 0.486 | 0.193 |
| combined, uniform weights | 10 | 3836 | 0.031 | 0.395 | 0.196 |
| original + topic blocking | 20 | 9963 | 0.080 | 0.705 | 0.297 |
| original | 20 | 15121 | 0.121 | 0.763 | 0.258 |
| combined, learned weights | 20 | 6284 | 0.050 | 0.654 | 0.200 |
| combined, uniform weights | 20 | 6686 | 0.054 | 0.572 | 0.201 |

# Item 6 — official BERTopic baseline

Protocol: `docs/bertopic_protocol.md`. 9986 abstracts, 20 subject labels; min_topic_size = 40.

| method | NMI | ARI | purity | n_clusters | outlier_share | stopword_share | topics_scored | C_v | NPMI | diversity |
|---|---|---|---|---|---|---|---|---|---|---|
| K-Means SBERT (k = 20) | 0.831 | 0.758 | 0.851 | 20 | 0.000 | 0.000 | 20/20 | 0.691 | 0.094 | 0.780 |
| Official BERTopic, SBERT (outliers reduced) | 0.807 | 0.731 | 0.868 | 29 | 0.000 |  |  |  |  |  |
| K-Means SPECTER (k = 20) | 0.786 | 0.687 | 0.819 | 20 | 0.000 | 0.000 | 20/20 | 0.693 | 0.098 | 0.790 |
| Official BERTopic, SBERT (raw, −1 = outliers) | 0.785 | 0.629 | 0.816 | 30 | 0.123 | 0.479 | 26/29 | 0.769 | 0.160 | 0.455 |
| Official BERTopic, SBERT — stop-word-free topic words (post hoc) | 0.785 | 0.629 | 0.816 | 30 | 0.123 | 0.000 | 29/29 | 0.704 | 0.111 | 0.759 |
| Official BERTopic, SPECTER (outliers reduced) | 0.749 | 0.615 | 0.860 | 44 | 0.000 |  |  |  |  |  |
| HDBSCAN SBERT (ours, PCA) | 0.709 | 0.231 | 0.489 | 10 | 0.000 |  |  |  |  |  |
| Official BERTopic, SPECTER (raw, −1 = outliers) | 0.695 | 0.403 | 0.762 | 45 | 0.187 | 0.357 | 42/44 | 0.744 | 0.143 | 0.480 |
| Official BERTopic, SPECTER — stop-word-free topic words (post hoc) | 0.695 | 0.403 | 0.762 | 45 | 0.187 | 0.000 | 44/44 | 0.691 | 0.105 | 0.705 |
| LDA Gibbs (K* = 30) | 0.664 | 0.571 | 0.713 | 30 | 0.000 | 0.000 | 30/30 | 0.666 | 0.114 | 0.857 |
| LDA Gibbs (K = 20) | 0.643 | 0.507 | 0.618 | 20 | 0.000 |  |  |  |  |  |
| LDA variational (K* = 30) | 0.424 | 0.206 | 0.372 | 30 | 0.000 |  |  |  |  |  |
| HDBSCAN SPECTER (ours, PCA) | 0.362 | 0.098 | 0.100 | 2 | 0.000 |  |  |  |  |  |
| BERTopic-style re-implementation (PCA, SPECTER) | 0.361 | 0.098 | 0.100 | 2 | 0.000 | 0.000 | 2/2 | 0.534 | 0.015 | 1.000 |

C_v / NPMI are computed on each method's top-10 topic words with the LDA tokenisation and dictionary (words absent from that dictionary are skipped).

# Real expert-labelled pairs

## ManConCorpus (claim-focused) (1774 pairs, {'agreement': 1047, 'conflict': 727})

Extraction coverage:

| docs | with_claims | mean_claims | mean_nodes | ctx_population | ctx_dose | ctx_time | ctx_environment | ctx_design |
|---|---|---|---|---|---|---|---|---|
| 259 | 0.903 | 1.884 | 3.139 | 0.974 | 0.846 | 0.667 | 0.261 | 0.624 |

| method | macro_f1 | macro_f1_lo | macro_f1_hi | f1_conflict | auprc_conflict | accuracy |
|---|---|---|---|---|---|---|
| Context clf. w/o transport | 0.592 | 0.569 | 0.617 | 0.575 | 0.522 | 0.592 |
| FNGW (full, edge features) | 0.552 | 0.528 | 0.575 | 0.471 | 0.452 | 0.567 |
| Signed matcher w/o context | 0.537 | 0.515 | 0.560 | 0.528 | 0.422 | 0.537 |
| CCST (ours) | 0.527 | 0.503 | 0.549 | 0.512 | 0.444 | 0.527 |
| SBERT (MiniLM) | 0.524 | 0.500 | 0.546 | 0.492 | 0.430 | 0.526 |
| BM25 | 0.522 | 0.499 | 0.544 | 0.428 | 0.423 | 0.541 |
| CCST + NLI (hybrid) | 0.513 | 0.490 | 0.537 | 0.492 | 0.441 | 0.514 |
| NLI (BART-MNLI) | 0.502 | 0.479 | 0.525 | 0.375 | 0.418 | 0.534 |
| Raw SVO overlap | 0.494 | 0.471 | 0.522 | 0.506 | 0.438 | 0.494 |

### Zero-shot CCST (trained only on the synthetic benchmark)

| gold | incomparable | heterogeneity | agreement | abstain | analogy | contradiction |
|---|---|---|---|---|---|---|
| conflict | 492 | 71 | 141 | 19 | 3 | 1 |
| agreement | 628 | 89 | 289 | 37 | 4 |  |

Conflict pairs explained by a material context difference:

- *In women with pre-eclampsia, does treatment with L Arginine, compared to placebo, reduce blood pressure or pre-eclampsia*: Verdict: HETEROGENEITY (p=1.00). 2 mechanism nodes align (S_mech=1.00). Matched context: design. Materially different context: dose (12 g vs 3 g); time (2 days vs prolonged); other (gestational length varying vs preeclampsia). Unknown context: environment. Smallest disagreement: doc1 reports 'oral l-arginine supplementation --no_effect--> mean diastolic blood pressure' while doc2 reports 'prolonged dietary supplementation with l-arginine --decreases--> blood pressure' (exclusive assertion, 100% of the disagreement). Evidence: [mc0_1] "Oral L-arginine supplementation did not reduce mean diastolic blood pressure after 2 days of treatment compared with placebo in pre-eclamptic patients with gestational length varying from 28 to 36 weeks." | [mc0_3] "We conclude that in women with preeclampsia, prolonged dietary supplementation with l-arginine significantly decreased blood pressure through increased endothelial synthesis and/or bioavailability of NO."
- *In women with pre-eclampsia, does treatment with L Arginine, compared to placebo, reduce blood pressure or pre-eclampsia*: Verdict: HETEROGENEITY (p=0.91). 2 mechanism nodes align (S_mech=0.41). Matched context: design. Materially different context: time (2 days vs 45 years). Unknown context: environment, other. Smallest disagreement: doc1 reports 'oral l-arginine supplementation --no_effect--> mean diastolic blood pressure' while doc2 reports 'treatment --decreases--> systolic blood pressures' (exclusive assertion, 100% of the disagreement). Unmatched: effect of l-arginine on blood pressure in pregnancy- [doc2], hypertension [doc2], women [doc2]. Evidence: [mc0_1] "Oral L-arginine supplementation did not reduce mean diastolic blood pressure after 2 days of treatment compared with placebo in pre-eclamptic patients with gestational length varying from 28 to 36 weeks." | [mc0_5] "Both systolic and diastolic blood pressures were reduced by treatment, the effect of L-arginine being significantly higher than that of the placebo (systolic values F = 8.59, p < 0.005; diastolic values F = 3.36; p < 0.001)."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=1.00). 2 mechanism nodes align (S_mech=1.00). Matched context: population, dose. Materially different context: time (8 weeks vs 24 hours). Unknown context: environment, design, other. Smallest disagreement: doc1 reports 'atorvastatin therapy --increases--> bp control' while doc2 reports 'concomitant antihypertensive treatment --decreases--> blood pressure' (sign flip, 100% of the disagreement). Evidence: [mc4_0] "Our results indicate that atorvastatin therapy significantly improves BP control in hyperlipidemic hypertensive patients." | [mc4_14] "Administration of a statin in hypertensive patients in whom blood pressure is effectively reduced by concomitant antihypertensive treatment does not have an additional blood pressure lowering effect."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.82). 2 mechanism nodes align (S_mech=0.32). Matched context: dose, design. Materially different context: time (3 months vs 4 weeks); other (cardiovascular disease vs severe primary hypercholesterolaemia). Unknown context: environment. Smallest disagreement: doc1 reports 'pravastatin 40 mg daily --no_effect--> blood pressure' while doc2 reports 'simvastatin 40 mg daily --decreases--> serum cholesterol' (exclusive assertion, 100% of the disagreement). Unmatched: ldl-cholesterol [doc2], apoprotein b [doc2], 3 months [doc2]. Evidence: [mc4_2] "In summary, pravastatin 40 mg daily did not reduce blood pressure in survivors of myocardial infarction without overt hypercholesterolaemia." | [mc4_4] "Total serum cholesterol, LDL-cholesterol and apoprotein B (ApoB) were significantly reduced by simvastatin 40 mg daily."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=1.00). 2 mechanism nodes align (S_mech=1.00). Matched context: time. Materially different context: other (cardiovascular disease vs uncontrolled hypertension). Unknown context: environment, design. Smallest disagreement: doc1 reports 'pravastatin 40 mg daily --no_effect--> blood pressure' while doc2 reports 'statins --increases--> blood pressure control' (exclusive assertion, 100% of the disagreement). Evidence: [mc4_2] "In summary, pravastatin 40 mg daily did not reduce blood pressure in survivors of myocardial infarction without overt hypercholesterolaemia." | [mc4_9] "In conclusion, the results of this study demonstrate that the use of statins in combination with antihypertensive drugs can improve blood pressure control in patients with uncontrolled hypertension and high serum cholesterol levels."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.69). 2 mechanism nodes align (S_mech=0.74). Matched context: design. Materially different context: time (3 months vs 1-year); other (cardiovascular disease vs hypercholesterolemia prescribed a standardized diet). Unknown context: environment. Smallest disagreement: doc1 reports 'pravastatin 40 mg daily --no_effect--> blood pressure' while doc2 reports 'three patients --decreases--> 2 women' (exclusive assertion, 100% of the disagreement). Unmatched: 1 man [doc2]. Evidence: [mc4_2] "In summary, pravastatin 40 mg daily did not reduce blood pressure in survivors of myocardial infarction without overt hypercholesterolaemia." | [mc4_12] "Three patients dropped out (2 women in the O group and 1 man in the OF group) due to adverse events related to orlistat treatment, including gastrointestinal events (oily spotting and fecal urgency)."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.96). 2 mechanism nodes align (S_mech=0.84). Matched context: dose, design. Materially different context: time (3 months vs 10 years); other (cardiovascular disease vs hypertension). Unknown context: environment. Smallest disagreement: doc1 reports 'pravastatin 40 mg daily --no_effect--> blood pressure' while doc2 reports 'simvastatin --decreases--> msna' (exclusive assertion, 100% of the disagreement). Evidence: [mc4_2] "In summary, pravastatin 40 mg daily did not reduce blood pressure in survivors of myocardial infarction without overt hypercholesterolaemia." | [mc4_13] "In conclusion, we found that, in patients with hypertension and hypercholesterolemia, simvastatin reduced MSNA, and this was related to increased baroreceptor sensitivity."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.78). 2 mechanism nodes align (S_mech=0.47). Matched context: design, other. Materially different context: time (70 years vs 24 hours). Unknown context: environment. Smallest disagreement: doc1 reports 'atorvastatin --no_effect--> aortic pwv' while doc2 reports 'concomitant antihypertensive treatment --decreases--> blood pressure' (exclusive assertion, 100% of the disagreement). Unmatched: statins [doc1], plaque stabilisation [doc1], blood pressure [doc1]. Evidence: [mc4_5] "The results revealed that atorvastatin did not change blood pressure, significantly lowered (P<0.003; <0.002) plasma total and LDL cholesterol, and increased aortic PWV by +8% (vs -2% under placebo)" | [mc4_14] "Administration of a statin in hypertensive patients in whom blood pressure is effectively reduced by concomitant antihypertensive treatment does not have an additional blood pressure lowering effect."

## ManConCorpus (full abstract) (1774 pairs, {'agreement': 1047, 'conflict': 727})

Extraction coverage:

| docs | with_claims | mean_claims | mean_nodes | ctx_population | ctx_dose | ctx_time | ctx_environment | ctx_design |
|---|---|---|---|---|---|---|---|---|
| 259 | 0.903 | 3.625 | 5.402 | 0.979 | 0.850 | 0.671 | 0.261 | 0.628 |

| method | macro_f1 | macro_f1_lo | macro_f1_hi | f1_conflict | auprc_conflict | accuracy |
|---|---|---|---|---|---|---|
| Context clf. w/o transport | 0.566 | 0.543 | 0.588 | 0.536 | 0.507 | 0.568 |
| SBERT (MiniLM) | 0.524 | 0.500 | 0.546 | 0.492 | 0.430 | 0.526 |
| BM25 | 0.522 | 0.499 | 0.544 | 0.428 | 0.423 | 0.541 |
| CCST (ours) | 0.518 | 0.493 | 0.539 | 0.466 | 0.438 | 0.523 |
| CCST + NLI (hybrid) | 0.511 | 0.487 | 0.532 | 0.449 | 0.420 | 0.519 |
| Signed matcher w/o context | 0.510 | 0.485 | 0.535 | 0.464 | 0.423 | 0.515 |
| NLI (BART-MNLI) | 0.502 | 0.479 | 0.525 | 0.375 | 0.418 | 0.534 |
| FNGW (full, edge features) | 0.484 | 0.463 | 0.507 | 0.331 | 0.429 | 0.529 |
| Raw SVO overlap | 0.294 | 0.282 | 0.307 | 0.581 | 0.416 | 0.411 |

### Zero-shot CCST (trained only on the synthetic benchmark)

| gold | incomparable | heterogeneity | agreement | analogy | abstain | contradiction |
|---|---|---|---|---|---|---|
| conflict | 476 | 62 | 143 | 11 | 34 | 1 |
| agreement | 612 | 110 | 236 | 26 | 62 | 1 |

Conflict pairs explained by a material context difference:

- *In women with pre-eclampsia, does treatment with L Arginine, compared to placebo, reduce blood pressure or pre-eclampsia*: Verdict: HETEROGENEITY (p=1.00). 2 mechanism nodes align (S_mech=0.32). Matched context: design. Materially different context: dose (12 g vs 3 g); time (2 days vs prolonged); other (gestational length varying vs preeclampsia). Unknown context: environment. Smallest disagreement: doc1 reports 'oral l-arginine supplementation --no_effect--> mean diastolic blood pressure' while doc2 reports 'oral supplementation with l-arginine on blood pressure --decreases--> blood pressure' (exclusive assertion, 100% of the disagreement). Unmatched: effects of [doc2], 24-h urinary excretion of nox [doc2], safe and efficient strategy to [doc2]. Evidence: [mc0_1] "Oral L-arginine supplementation did not reduce mean diastolic blood pressure after 2 days of treatment compared with placebo in pre-eclamptic patients with gestational length varying from 28 to 36 weeks." | [mc0_3] "We conclude that in women with preeclampsia, prolonged dietary supplementation with l-arginine significantly decreased blood pressure through increased endothelial synthesis and/or bioavailability of NO."
- *In women with pre-eclampsia, does treatment with L Arginine, compared to placebo, reduce blood pressure or pre-eclampsia*: Verdict: HETEROGENEITY (p=0.92). 2 mechanism nodes align (S_mech=0.41). Matched context: design. Materially different context: time (2 days vs 45 years). Unknown context: environment, other. Smallest disagreement: doc1 reports 'oral l-arginine supplementation --no_effect--> mean diastolic blood pressure' while doc2 reports 'treatment --decreases--> systolic blood pressures' (exclusive assertion, 100% of the disagreement). Unmatched: effect of l-arginine on blood pressure in pregnancy- [doc2], hypertension [doc2], women [doc2]. Evidence: [mc0_1] "Oral L-arginine supplementation did not reduce mean diastolic blood pressure after 2 days of treatment compared with placebo in pre-eclamptic patients with gestational length varying from 28 to 36 weeks." | [mc0_5] "Both systolic and diastolic blood pressures were reduced by treatment, the effect of L-arginine being significantly higher than that of the placebo (systolic values F = 8.59, p < 0.005; diastolic values F = 3.36; p < 0.001)."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.81). 4 mechanism nodes align (S_mech=0.26). Matched context: dose. Materially different context: time (8 weeks vs 24 hour). Unknown context: environment, design, other. Smallest disagreement: doc1 reports 'atorvastatin therapy --increases--> bp control' while doc2 reports 'statins --decreases--> cholesterol concentrations' (sign flip, 100% of the disagreement). Unmatched: statins [doc1], endothelial function [doc1], platelet aggregation [doc1]. Evidence: [mc4_0] "Our results indicate that atorvastatin therapy significantly improves BP control in hyperlipidemic hypertensive patients." | [mc4_14] "To investigate the possibility that statins reduce blood pressure as well as cholesterol concentrations through clinic and 24 hour ambulatory blood pressure monitoring."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.72). 3 mechanism nodes align (S_mech=0.47). Matched context: dose, design. Materially different context: time (3 months vs 4 weeks); other (cardiovascular disease vs severe primary hypercholesterolaemia). Unknown context: environment. Smallest disagreement: doc1 reports 'pravastatin --no_effect--> adjusted risk of incident systolic hypertension' while doc2 reports 'simvastatin 40 mg daily --decreases--> serum cholesterol' (exclusive assertion, 67% of the disagreement). Unmatched: hmg-coa reductase inhibitors [doc1], apoprotein b [doc2], 3 months [doc2]. Evidence: [mc4_2] "Pravastatin did not reduce the adjusted risk of incident systolic hypertension (odds ratio 0.99, 95% CI 0.80-1.23), or incident diastolic hypertension (odds ratio 0.97, 95% CI 0.73-1.27)." | [mc4_4] "Total serum cholesterol, LDL-cholesterol and apoprotein B (ApoB) were significantly reduced by simvastatin 40 mg daily."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.96). 4 mechanism nodes align (S_mech=0.76). Matched context: design. Materially different context: time (3 months vs 70 years); other (cardiovascular disease vs hypertension). Unknown context: environment. Smallest disagreement: doc1 reports 'hmg-coa reductase inhibitors --decreases--> blood pressure' while doc2 reports 'finding of --increases--> aortic pwv' (sign flip, 67% of the disagreement). Unmatched: statins [doc2], plaque stabilisation [doc2]. Evidence: [mc4_2] "Experimental evidence and several small studies in humans suggest that HMG-CoA (3-hydroxy 3-methylglutaryl coenzyme A) reductase inhibitors (statins) reduce blood pressure, perhaps through effects on endothelial function or by reducing inflammation." | [mc4_5] "The finding of increased aortic PWV after 12-week statin treatment agrees with studies in monkeys, indicating that, at the early phase of hypolipidaemic diet but not later, aortic PWV increases transiently as a consequence of the significant reduction of lipid vascular content."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.99). 2 mechanism nodes align (S_mech=0.55). Matched context: time. Materially different context: other (cardiovascular disease vs uncontrolled hypertension). Unknown context: environment, design. Smallest disagreement: doc1 reports 'pravastatin --decreases--> blood pressure' while doc2 reports 'statins --increases--> blood pressure control' (sign flip, 100% of the disagreement). Unmatched: hmg-coa reductase inhibitors [doc1], adjusted risk of incident systolic hypertension [doc1]. Evidence: [mc4_2] "We tested the hypothesis that pravastatin would reduce blood pressure at 3 months and the risk of developing new hypertension over a follow-up period of 5 years." | [mc4_9] "In conclusion, the results of this study demonstrate that the use of statins in combination with antihypertensive drugs can improve blood pressure control in patients with uncontrolled hypertension and high serum cholesterol levels."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.85). 3 mechanism nodes align (S_mech=0.82). Matched context: design. Materially different context: time (3 months vs 1-year); other (cardiovascular disease vs hypercholesterolemia prescribed a standardized diet). Unknown context: environment. Smallest disagreement: doc1 reports 'pravastatin --no_effect--> adjusted risk of incident systolic hypertension' while doc2 reports 'three patients --decreases--> 2 women' (exclusive assertion, 67% of the disagreement). Unmatched: hmg-coa reductase inhibitors [doc1]. Evidence: [mc4_2] "Pravastatin did not reduce the adjusted risk of incident systolic hypertension (odds ratio 0.99, 95% CI 0.80-1.23), or incident diastolic hypertension (odds ratio 0.97, 95% CI 0.73-1.27)." | [mc4_12] "Three patients dropped out (2 women in the O group and 1 man in the OF group) due to adverse events related to orlistat treatment, including gastrointestinal events (oily spotting and fecal urgency)."
- *In patients with hypertesion or hypercholesterolemia, does statin drugs, compared to placebo, reduce blood pressure or lipid levels*: Verdict: HETEROGENEITY (p=0.99). 4 mechanism nodes align (S_mech=0.76). Matched context: design, other. Materially different context: time (70 years vs 24 hour). Unknown context: environment. Smallest disagreement: doc1 reports 'finding of --increases--> aortic pwv' while doc2 reports 'concomitant antihypertensive treatment --decreases--> blood pressure' (sign flip, 34% of the disagreement). Unmatched: statins [doc1], plaque stabilisation [doc1]. Evidence: [mc4_5] "The finding of increased aortic PWV after 12-week statin treatment agrees with studies in monkeys, indicating that, at the early phase of hypolipidaemic diet but not later, aortic PWV increases transiently as a consequence of the significant reduction of lipid vascular content." | [mc4_14] "Administration of a statin in hypertensive patients in whom blood pressure is effectively reduced by concomitant antihypertensive treatment does not have an additional blood pressure lowering effect."

## SciFact (1259 pairs, {'agreement': 508, 'contradiction': 265, 'incomparable': 486})

Extraction coverage:

| docs | with_claims | mean_claims | mean_nodes | ctx_population | ctx_dose | ctx_time | ctx_environment | ctx_design |
|---|---|---|---|---|---|---|---|---|
| 1776 | 0.670 | 1.859 | 2.999 | 0.559 | 0.201 | 0.203 | 0.165 | 0.160 |

| method | macro_f1 | macro_f1_lo | macro_f1_hi | f1_contradiction | auprc_contradiction | accuracy |
|---|---|---|---|---|---|---|
| NLI (BART-MNLI) | 0.680 | 0.653 | 0.704 | 0.594 | 0.570 | 0.693 |
| CCST + NLI (hybrid) | 0.669 | 0.642 | 0.694 | 0.583 | 0.554 | 0.682 |
| BM25 | 0.502 | 0.477 | 0.531 | 0.283 | 0.270 | 0.542 |
| SBERT (MiniLM) | 0.465 | 0.440 | 0.490 | 0.287 | 0.284 | 0.494 |
| Signed matcher w/o context | 0.400 | 0.371 | 0.429 | 0.296 | 0.300 | 0.427 |
| Context clf. w/o transport | 0.399 | 0.371 | 0.427 | 0.296 | 0.301 | 0.430 |
| CCST (ours) | 0.391 | 0.365 | 0.418 | 0.299 | 0.285 | 0.418 |
| FNGW (full, edge features) | 0.388 | 0.361 | 0.414 | 0.304 | 0.265 | 0.409 |
| Raw SVO overlap | 0.194 | 0.183 | 0.205 | 0.015 | 0.214 | 0.389 |


# Corpus experiment (PubMed + arXiv)

9986 abstracts, 20 subject labels (used only for evaluation).

## LDA K sweep

| K | C_v | NPMI | diversity | perplexity |
|---|---|---|---|---|
| 10 | 0.643 | 0.106 | 0.832 | 1745.593 |
| 20 | 0.665 | 0.113 | 0.742 | 1527.987 |
| 30 | 0.666 | 0.114 | 0.740 | 1430.135 |
| 50 | 0.640 | 0.105 | 0.713 | 1347.277 |

K* = 30; gensim variational LDA at K*: C_v=0.490, NPMI=-0.048

## Clustering vs subject labels

| method | NMI | ARI | purity | n_clusters | silhouette |
|---|---|---|---|---|---|
| K-Means SBERT (k=20) | 0.831 | 0.758 | 0.851 | 20 | 0.150 |
| K-Means SPECTER (k=20) | 0.786 | 0.687 | 0.819 | 20 | 0.163 |
| K-Means SPECTER (k=30) | 0.772 | 0.669 | 0.860 | 30 | 0.171 |
| HDBSCAN SBERT | 0.709 | 0.231 | 0.489 | 10 | 0.200 |
| LDA Gibbs argmax θ (K=30) | 0.664 | 0.571 | 0.713 | 30 | 0.091 |
| LDA Gibbs argmax θ (K=20, closest to #labels) | 0.643 | 0.507 | 0.618 | 20 | 0.095 |
| LDA variational argmax θ (K=30) | 0.424 | 0.206 | 0.372 | 30 | -0.079 |
| HDBSCAN SPECTER | 0.362 | 0.098 | 0.100 | 2 | 0.296 |
| BERTopic-style (HDBSCAN + c-TF-IDF) | 0.361 | 0.098 | 0.100 | 2 | 0.296 |

## Summaries (ROUGE / BERTScore vs held-out abstracts)

LDA clusters:

| system | rouge1 | rouge2 | rougeL | bertscore_f1 | clusters |
|---|---|---|---|---|---|
| TextRank + centroid + MMR | 0.219 | 0.026 | 0.117 | 0.750 | 30 |
| first-sentence baseline | 0.148 | 0.018 | 0.095 | 0.724 | 30 |
| random-sentence baseline | 0.197 | 0.022 | 0.111 | 0.739 | 30 |

Embedding clusters:

| system | rouge1 | rouge2 | rougeL | bertscore_f1 | clusters |
|---|---|---|---|---|---|
| TextRank + centroid + MMR | 0.226 | 0.032 | 0.120 | 0.755 | 30 |
| first-sentence baseline | 0.143 | 0.021 | 0.092 | 0.725 | 30 |
| random-sentence baseline | 0.187 | 0.021 | 0.105 | 0.737 | 30 |

## t-SNE faithfulness

| space | perplexity | dims | trustworthiness | continuity | knn_preservation | n | k |
|---|---|---|---|---|---|---|---|
| SPECTER | 15 | 2 | 0.982 | 0.970 | 0.421 | 3000 | 10 |
| SPECTER | 30 | 2 | 0.983 | 0.972 | 0.431 | 3000 | 10 |
| SPECTER | 50 | 2 | 0.984 | 0.973 | 0.429 | 3000 | 10 |
| LDA θ | 15 | 2 | 0.993 | 0.980 | 0.503 | 3000 | 10 |
| LDA θ | 30 | 2 | 0.993 | 0.980 | 0.514 | 3000 | 10 |
| LDA θ | 50 | 2 | 0.994 | 0.981 | 0.512 | 3000 | 10 |
| SPECTER | 50 | 3 | 0.989 | 0.976 | 0.500 | 3000 | 10 |

## Discovery on the stratified subset

| docs | candidate_pairs | all_pairs | conflict_pairs | analogy_pairs | explore_pairs | predicted | abstention_rate | families | blocked_by_guards | mean_ms_per_pair |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 24037 | 1999000 | 9121 | 11941 | 3030 | {'incomparable': 15674, 'agreement': 2251, 'analogy': 4898, 'abstain': 747, 'heterogeneity': 466, 'contradiction': 1} | 0.031 | 490 | {'analogy': 5572, 'agreement': 179, 'heterogeneity': 134, 'contradiction': 10} | 3.994 |

## Relation summaries (top generalised relations per topic)

- **topic 0** (design, algorithm, optimization, search, adaptive): matching raw trajectories --enables--> rugged objective that sparse surrogates cannot learn [1]; annealed --decreases--> -confidence-bound [1]; removing sequential lcb --increases--> six-scene mean [1] — pair counts {'incomparable': 284, 'analogy': 32, 'abstain': 13, 'agreement': 26}
- **topic 1** (induce, gene, response, pathway, protein): metformin --enables--> apoptosis [2]; metformin --decreases--> aging-induced genes [2]; study investigates the --associated--> vitamin d supplementation [1] — pair counts {'incomparable': 34, 'analogy': 6, 'agreement': 3, 'abstain': 2}
- **topic 2** (cell, growth, cellular, type, tissue): exhibiting a stronger --associated--> on pca [1]; dha --associated--> stromal components [1]; cancer-associated fibroblast differentiation --enables--> microenvironment favorable to pca initiation [1] — pair counts {'analogy': 34, 'incomparable': 244, 'agreement': 42, 'abstain': 13}
- **topic 3** (brain, neural, temporal, neuron, memory): coffee consumption --increases--> auditory neuropathy [1]; auditory neuropathy --enables--> by diabetes [1]; coffee consumption --decreases--> hearing threshold shift of the auditory evoked potential [1] — pair counts {'incomparable': 209, 'analogy': 21, 'abstain': 9, 'agreement': 16, 'heterogeneity': 2}
- **topic 4** (dynamic, phase, noise, dynamical, stability): noise --enables--> transitions [2]; spatially structured network noise fields --enables--> partially overlapping subnetworks [1]; mismatch --decreases--> capacity [1] — pair counts {'incomparable': 474, 'analogy': 41, 'abstain': 14, 'agreement': 19}
- **topic 5** (population, level, age, estimate, individual): ssb consumption --enables--> difficulty [1]; how different aspects of social relationships --associated--> mortality [1]; how different aspects of social relationships --associated--> all-cause mortality [1] — pair counts {'incomparable': 2}
- **topic 6** (economic, market, price, policy, country): study seeks to understand --enables--> mortality [1]; leading --enables--> of death were vehicle accidents [1]; all-cause mortality --no_effect--> annual fluctuations [1] — pair counts {'incomparable': 264, 'analogy': 28, 'agreement': 33, 'abstain': 8, 'heterogeneity': 4}
- **topic 7** (metformin, cancer, drug, treatment, survival): metformin --decreases--> all-cause mortality [4]; metformin --associated--> of death [4]; metformin --associated--> cancer incidence [3] — pair counts {'incomparable': 304, 'analogy': 40, 'abstain': 39, 'agreement': 158, 'heterogeneity': 55}
- **topic 8** (information, source, knowledge, field, physical): menopausal hormone therapy --no_effect--> conversion to either [1]; omega-3 fatty acids --increases--> cardiovascular outcomes [1]; kalshimultivariate-event architecture --enables--> market objects [1] — pair counts {'analogy': 10, 'incomparable': 19, 'agreement': 2}
- **topic 9** (control, condition, constraint, parameter, trajectory): penalty --decreases--> concentrations [1]; penalty --associated--> relative prominence of centers [1]; distinct converged states --enables--> multi-start sweeps on three synthetic two-dimensional landscapes [1] — pair counts {'incomparable': 186, 'analogy': 17, 'agreement': 12, 'abstain': 1}
- **topic 10** (evolution, framework, dynamic, interaction, specie): generation z showed --decreases--> excitement seeking [1]; dispositions --enables--> contemporary cultural conditions [1]; developmental encoding --increases--> phenotypic diversity [1] — pair counts {'incomparable': 43, 'agreement': 4, 'analogy': 11}
- **topic 11** (role, recent, human, research, health): interest --increases--> past decade [1]; papers --increases--> past decade [1]; vitamin d supplementation --associated--> colorectal cancer [1] — pair counts {'agreement': 5, 'analogy': 2, 'incomparable': 7}

# RQ4 scaling

| N | all_pairs | sparse_pairs | sparse_index_s | sparse_total_s | pairs_per_doc | exact_total_s | peak_rss_mb |
|---|---|---|---|---|---|---|---|
| 125 | 7750 | 1131 | 0.083 | 4.608 | 9.048 | 21.805 | 2984.652 |
| 250 | 31125 | 2624 | 0.150 | 10.683 | 10.496 | 92.404 | 2984.652 |
| 500 | 124750 | 5928 | 0.281 | 23.582 | 11.856 | 376.778 | 2984.652 |
| 1000 | 499500 | 12680 | 0.431 | 50.899 | 12.680 |  | 2984.652 |
| 2000 | 1999000 | 26691 | 0.580 | 105.906 | 13.345 |  | 2984.652 |

Candidate recall:

| k | topic_blocking | pairs | pair_fraction | recall@5 |
|---|---|---|---|---|
| 2 | True | 1976 | 0.016 | 0.231 |
| 2 | False | 2440 | 0.020 | 0.251 |
| 5 | True | 3539 | 0.028 | 0.394 |
| 5 | False | 4600 | 0.037 | 0.425 |
| 10 | True | 5928 | 0.048 | 0.562 |
| 10 | False | 8453 | 0.068 | 0.612 |
| 20 | True | 9929 | 0.080 | 0.717 |
| 20 | False | 15063 | 0.121 | 0.776 |

Per-pair cost:

| n_nodes | ms_per_pair | fw_iters |
|---|---|---|
| 4 | 3.376 | 3.550 |
| 8 | 5.057 | 6.250 |
| 16 | 10.584 | 11.050 |
| 32 | 45.542 | 11.950 |
| 64 | 501.748 | 15.000 |

Fitted log-log slopes: {'slope_exact': 2.0554801508384037, 'slope_sparse': 1.1297291233079596, 'slope_cost_vs_n': 1.7601614743513172}
