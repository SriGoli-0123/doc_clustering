# Item 3 — sign-aware tie-breaking in the alignment (fixed before running)

Written 2026-10-10, before any tie-breaking result was computed.

## Problem
T* is computed with the *unsigned* objective (proposal Sec. 4.3: "two views of the same alignment"). In
symmetric motifs (A → Y₁, A → Y₂) several alignments have the same unsigned cost; the solver may pick a
swapped one and create *spurious* sign conflicts. This hurts analogy and agreement pairs, and the
comparability guards then remove those analogies (v1 analogy recall 0.93 → 0.82, v2 0.88 → 0.48 with guards).

## Change
Add the sign-conflict terms to the alignment objective with a small weight ε (`ccst.lam_s_align`):
`align objective = unsigned objective + ε · (sign-conflict terms)`. **ε = 0.05 is fixed a priori** (5% of the
edge-existence weight): large enough to break exact ties, small enough that it cannot override a structural
difference. D_unsigned and D_signed are still evaluated on the resulting T* exactly as before, so a genuine
sign flip that no alternative alignment can avoid still appears as the gap D_signed − D_unsigned.

## Evaluation
Benchmarks v1 and v2, default configuration otherwise unchanged, same protocol (identical LR, grouped
5-fold CV by template), results written next to the existing ones under a `_tb` suffix. A sensitivity sweep
over ε ∈ {0.02, 0.1} is reported for information only and is **not** used to choose ε.

## Decision rule (pre-registered)
Adopt ε = 0.05 as the new default only if, on **both** benchmarks:
1. analogy recall *with comparability guards* improves, and
2. contradiction recall does not drop by more than 0.02 and the false-contradiction rate on context-shift
   pairs does not rise by more than 0.02, and
3. macro-F1 does not drop by more than 0.01.
Otherwise the change is reported as a negative result and the default stays ε = 0.
