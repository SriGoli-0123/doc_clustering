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