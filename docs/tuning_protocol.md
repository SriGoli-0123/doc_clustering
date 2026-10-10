# Hyper-parameter selection protocol (item 2) — fixed before running

Written 2026-10-10, before any tuning result was computed.

## Problem being fixed
The λ_u × α sensitivity grid in `results/synthetic*/` was scored on the cross-validation *test* folds, so
picking its best cell would be optimistic. Here the choice is made inside each training fold only.

## Nested cross-validation
* **Outer loop:** the same 5-fold `GroupKFold` by mechanism template used for every headline number.
* **Inner loop:** 4-fold `GroupKFold` by template *inside the outer training portion only*. The outer test
  fold is never seen during selection.
* **Selection criterion:** mean inner macro-F1 (argmax predictions). Ties (≤ 0.001) go to the default
  configuration, then to the simpler one (fewer non-default values).
* **Final fit per outer fold:** the selected configuration is refit on the outer training portion with the
  standard protocol (25% of training templates held out for temperature scaling / abstention), then scored
  once on the outer test fold.

## Search space
* **CCST** (12 configurations × 3 values of C):
  partial transport with λ_u ∈ {0.3, 0.6, 1.0} × α ∈ {0.5, 1.0, 2.0} (9), plus forced full transport with
  α ∈ {0.5, 1.0, 2.0} (3; λ_u is irrelevant without partial transport). LR inverse regularisation
  C ∈ {0.1, 1, 10}.
* **Baselines, for fairness:** the five strongest baselines (context classifier w/o transport, signed matcher
  w/o context, NLI, SBERT, SPECTER) get the same nested selection over C ∈ {0.1, 1, 10}. Their features have
  no other tunable knobs in this implementation.

## Reporting
For v1 and v2: default-configuration CV macro-F1 (already reported), nested-CV macro-F1 with 95% bootstrap
CI, false-contradiction rate, and the configuration chosen in each outer fold (stability). Whatever the
outcome — including "tuning does not help" — it is reported.
