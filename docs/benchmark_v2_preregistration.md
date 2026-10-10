# Benchmark v2 ("realistic difficulty") — pre-registration

Written 2026-10-10 01:43, **before** any v2 data were generated or any v2 result was computed.

## Why
Benchmark v1 renders mechanisms in clean template prose, so extraction is near-perfect and several
metrics saturate (heterogeneity F1 1.000, certificate flip recall 1.000). Real abstracts are harder:
claims come with statistics, effect sizes, nominalisations and hedges, and scope conditions are worded
in many ways. v2 adds that difficulty. It is **not** designed to reach any target score; the rates below
are fixed now and results are reported whatever they are. v1 results stay in the report unchanged.

## What changes (applied identically to every document, so every method sees the same text)

### 1. Claim-sentence styles
Each claim sentence independently gets one style (seeded RNG):

| style | rate | example |
|---|---|---|
| plain (v1 forms) | 0.55 | "Statin therapy reduced LDL cholesterol." |
| with statistics | 0.15 | "Statin therapy reduced LDL cholesterol (p < 0.01)." |
| effect size / association | 0.10 | "Statin therapy was associated with a 23% lower LDL cholesterol." |
| nominalised | 0.10 | "A 23% reduction in LDL cholesterol was observed with statin therapy." |
| hedged | 0.10 | "Statin therapy may reduce LDL cholesterol." |

The edge whose sign/assertion a variant perturbs (the one that defines contradiction / heterogeneity)
is never hedged, so gold labels stay unambiguous. It *can* receive any other style, including
nominalisation that the extractor may miss — that is a genuine extraction failure and is counted.

### 2. Context paraphrase
Each document's methods sentence is rendered with paraphrased scope wording with probability 0.5
(population synonyms, dose written as "per day" / "per liter" / unit variants, durations in other units
or words, environment and design synonyms). The paraphrase lists were written as a domain author would
write them, without checking which forms the context parser handles.

### 3. Two new hard pair types per (template, domain)

| variant | gold label | construction |
|---|---|---|
| `contra_ctxpara` | contradiction | one sign flip; *same* scope values, context sentence always paraphrased |
| `agree_nested` | agreement | same signs; population narrowed to a natural subgroup (e.g. "obese adult patients") |

### 4. Reporting
* Extraction quality against gold (edge recall, sign accuracy) is reported next to downstream results.
* Same protocol as v1: identical LR for every method, grouped 5-fold CV by template, bootstrap CIs.
* Results go to `results/synthetic_v2/`; v1 results in `results/synthetic/` are kept.
* No hyper-parameter, threshold or lexicon change is made in response to v2 test results within this
  step. Fixes motivated by v2 (e.g. parser weaknesses) happen in later, separately reported steps.

## Implementation note (added 2026-10-10 01:44, still before any v2 data or result exists)
No-effect claims ("X did not reduce Y") can only take the *plain* (0.85) or *with statistics* (0.15)
style: an effect-size, nominalised or hedged rendering would change what a no-effect claim asserts.
The realism choices use their own random stream, so benchmark v1 stays byte-for-byte reproducible.
