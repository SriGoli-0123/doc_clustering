"""Cluster summarisation (proposal Sec. 4.9, Step I).

1. labels: top LDA words / class-based TF-IDF keywords
2. extractive summary: TextRank (PageRank on the sentence-similarity graph) combined with closeness
   to the cluster centroid, then Maximal Marginal Relevance to avoid near-duplicates (3-5 real sentences)
3. relation summary: most frequent generalised relations in the cluster (Ceran et al. story patterns)
   with agreement / contradiction / heterogeneity pair counts
Baselines: first sentences of random cluster documents, random sentences.
Complexity per cluster with s sentences: O(s^2 d) for the similarity graph + O(iters s^2) PageRank.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text: str, min_len: int = 30) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) >= min_len]


def textrank(E: np.ndarray, d: float = 0.85, iters: int = 100, tol: float = 1e-8) -> np.ndarray:
    S = np.clip(E @ E.T, 0, None)
    np.fill_diagonal(S, 0)
    deg = S.sum(1, keepdims=True)
    W = np.divide(S, deg, out=np.zeros_like(S), where=deg > 0)
    n = len(E)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r_new = (1 - d) / n + d * (W.T @ r)
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r


def mmr_select(scores: np.ndarray, E: np.ndarray, k: int, lam: float = 0.7) -> list[int]:
    chosen: list[int] = []
    cand = list(range(len(scores)))
    while cand and len(chosen) < k:
        if chosen:
            red = (E[cand] @ E[chosen].T).max(1)
        else:
            red = np.zeros(len(cand))
        mmr = lam * scores[cand] - (1 - lam) * red
        j = cand[int(np.argmax(mmr))]
        chosen.append(j)
        cand.remove(j)
    return chosen


def extractive_summary(sentences: list[str], E: np.ndarray, k: int = 4, w_textrank: float = 0.5,
                       max_sentences: int = 1500, seed: int = 0) -> list[str]:
    if not sentences:
        return []
    if len(sentences) > max_sentences:  # bound the O(s^2) graph on very large clusters
        idx = np.random.default_rng(seed).choice(len(sentences), max_sentences, replace=False)
        sentences, E = [sentences[i] for i in idx], E[idx]
    tr = textrank(E)
    tr = (tr - tr.min()) / (np.ptp(tr) + 1e-12)
    c = E.mean(0)
    c /= np.linalg.norm(c) + 1e-12
    cen = E @ c
    cen = (cen - cen.min()) / (np.ptp(cen) + 1e-12)
    score = w_textrank * tr + (1 - w_textrank) * cen
    return [sentences[i] for i in mmr_select(score, E, k)]


def first_sentence_baseline(doc_texts: list[str], k: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(doc_texts), size=min(k, len(doc_texts)), replace=False)
    out = []
    for i in idx:
        s = split_sentences(doc_texts[i])
        if s:
            out.append(s[0])
    return out


def random_sentence_baseline(sentences: list[str], k: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [sentences[i] for i in rng.choice(len(sentences), size=min(k, len(sentences)), replace=False)]


def relation_summary(graphs: list, top: int = 5, pair_counts: dict | None = None) -> dict:
    """Most frequent generalised relations (concept --relation--> concept) with document support."""
    support: Counter = Counter()
    for G in graphs:
        names = G.meta.get("concept_names", G.node_text)
        seen = set()
        for e in G.edges:
            c = G.claims[max(e.claims, key=lambda k: G.claims[k].weight)]
            key = (names[e.u], c.relation, names[e.v])
            if key not in seen:
                support[key] += 1
                seen.add(key)
    rels = [{"relation": f"{s} --{r}--> {o}", "support": n} for (s, r, o), n in support.most_common(top)]
    return {"relations": rels, "pair_counts": pair_counts or {}}


class SummaryScorer:
    """ROUGE-1/2/L (Lin, 2004) and BERTScore (Zhang et al., 2020) against held-out abstracts."""

    def __init__(self, bertscore_model: str | None = "distilbert-base-uncased"):
        from rouge_score import rouge_scorer

        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.bs_model = bertscore_model

    def rouge_scores(self, summary: list[str], refs: list[str]) -> dict:
        hyp = " ".join(summary)
        sc = [self.rouge.score(r, hyp) for r in refs]
        return {k: float(np.mean([s[k].fmeasure for s in sc])) for k in ("rouge1", "rouge2", "rougeL")}

    def bertscore(self, cands: list[str], refs: list[str]) -> list[float]:
        if not self.bs_model:
            return [float("nan")] * len(cands)
        from bert_score import score

        from .utils import device

        _, _, F = score(cands, refs, model_type=self.bs_model, device=device(), verbose=False, batch_size=16)
        return F.tolist()
