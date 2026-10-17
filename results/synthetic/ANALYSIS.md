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
