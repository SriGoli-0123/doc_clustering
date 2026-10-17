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
