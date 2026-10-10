# Item 4 — real-text parsing fixes (fixed before implementing)

Written 2026-10-10, before any fix was implemented. The held-out phrasing set
(`ccst/eval/heldout_extraction.py`) was written at the same time and its "before" score recorded before
the extractor was changed.

## Fixes (general linguistic rules, motivated by failures seen in *real* abstracts)
1. **Age vs duration.** "≥70 years", "aged 40–60 years", "65 years or older", "mean age 72 years" describe
   the population, not the time horizon (seen in real-corpus certificates as "time: 70 years vs 4.2 years").
2. **Quantities inside noun phrases.** "a 23% lower LDL", "by 23%", "(95% CI ...)" are stripped from the
   concept instead of discarding the whole claim.
3. **Nominalised claims.** "a reduction in Y was observed with X", "X produced a decline in Y", "Y fell by 23%
   after X": a nominalisation lexicon (increase / rise / elevation / reduction / decline / fall / drop /
   inhibition / activation …) maps the event noun to a signed relation.
4. **Thousands separators.** "1,000 IU" must parse as 1000, not 1.0.
5. **"Other material condition"** (proposal Sec. 9): a generic scope slot for conditions outside the five
   typed fields ("patients *with chronic kidney disease*", "*under drought* conditions").

## Evaluation (all three reported, whatever the outcome)
* **Held-out phrasing set** (new vocabulary and sentence forms, absent from both benchmark generators):
  claim recall, sign accuracy, and context accuracy (age group, duration, dose) — before vs after.
* **Benchmark v2 extraction quality** — before vs after. *Not held out:* v2's nominal and effect-size
  templates were written by the same author, so this number is reported as optimistic.
* **Real expert-labelled pairs (ManConCorpus, SciFact)** — downstream macro-F1 before vs after, same protocol.
No rule is added or changed after looking at held-out results.
