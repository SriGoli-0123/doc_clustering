# Item 5 — candidate-recall improvement (fixed before running)

Written 2026-10-10, before any new-index result was computed.

## Problem
The sparse index (semantic / mechanism / context sketches + topic blocking) keeps only ≈ 55% of each
document's exact top-5 CCST neighbours at the default budget (k = 10, 4.7% of pairs, N = 500).

## New index (all choices fixed here)
1. **Concept-histogram sketch.** Node embeddings are quantised with a K = 256 k-means codebook fitted on the
   *training* documents only; a document becomes a mass-weighted, role-split (action / mediator / outcome)
   TF-IDF histogram over codewords. This approximates which documents *can* be aligned node-by-node — the
   dominant term of D_unsigned — better than mean-pooled node text.
2. **Single combined kNN.** Each sketch (semantic, concept-histogram, mechanism, context) is L2-normalised and
   weighted; one cosine kNN over the concatenation replaces the union of three separate kNN lists.
3. **Weights are learned without labels:** exact CCST scores on a *training* subset of 500 documents define
   each document's exact top-5 neighbours; weights ∈ {0, 0.5, 1, 2}⁴ are chosen to maximise training recall@5 at
   10 candidates per document. Topic blocking and the exploration set are kept as in the original index.

## Evaluation
* Evaluation subset: the same 500 real PubMed/arXiv documents used for the scaling experiment (disjoint
  from the training subset). Exact all-pairs CCST (124,750 problems) defines the gold top-5 neighbours.
* Both indexes are evaluated on identical graphs: recall@5 vs fraction of all pairs solved, at k ∈ {2, 5, 10,
  20} candidates per document per list, plus build and query time.
* Reported whatever the outcome; the comparison is made at matched candidate budgets (curves), not at a
  single favourable point.
