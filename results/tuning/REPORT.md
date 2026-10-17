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
