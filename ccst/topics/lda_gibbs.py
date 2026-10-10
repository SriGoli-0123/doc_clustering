"""Latent Dirichlet Allocation with collapsed Gibbs sampling (Griffiths & Steyvers, 2004).

Each sweep resamples every token's topic from
    p(z_i = k | z_-i, w) ∝ (n_dk + alpha) (n_kw + beta) / (n_k + V beta)
Time: O(iters * N_tokens * K); memory O(D K + K V). Compiled with numba (~50-100M token-topic
updates / s on one core).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _sweep(doc_ids, word_ids, z, ndk, nkw, nk, alpha, beta, V, iters, seed, fixed_phi):
    np.random.seed(seed)
    K = ndk.shape[1]
    p = np.empty(K)
    Vbeta = V * beta
    for _ in range(iters):
        for i in range(word_ids.shape[0]):
            d = doc_ids[i]
            w = word_ids[i]
            k = z[i]
            ndk[d, k] -= 1
            if not fixed_phi:
                nkw[k, w] -= 1
                nk[k] -= 1
            total = 0.0
            for kk in range(K):
                p[kk] = (ndk[d, kk] + alpha) * (nkw[kk, w] + beta) / (nk[kk] + Vbeta)
                total += p[kk]
            u = np.random.random() * total
            acc = 0.0
            knew = K - 1
            for kk in range(K):
                acc += p[kk]
                if acc >= u:
                    knew = kk
                    break
            z[i] = knew
            ndk[d, knew] += 1
            if not fixed_phi:
                nkw[knew, w] += 1
                nk[knew] += 1


def _flatten(docs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    doc_ids = np.concatenate([np.full(len(d), i, np.int64) for i, d in enumerate(docs)]) if docs else np.zeros(0, np.int64)
    word_ids = np.concatenate([np.asarray(d, np.int64) for d in docs]) if docs else np.zeros(0, np.int64)
    return doc_ids, word_ids


class GibbsLDA:
    def __init__(self, n_topics: int, vocab_size: int, alpha: float | None = None, beta: float = 0.01,
                 iters: int = 400, seed: int = 0):
        self.K, self.V = n_topics, vocab_size
        self.alpha = 50.0 / n_topics if alpha is None else alpha
        self.alpha = min(self.alpha, 1.0)  # 50/K is too smooth for short abstracts with small K
        self.beta, self.iters, self.seed = beta, iters, seed

    def fit(self, docs: list[list[int]]) -> "GibbsLDA":
        rng = np.random.default_rng(self.seed)
        doc_ids, word_ids = _flatten(docs)
        z = rng.integers(0, self.K, len(word_ids)).astype(np.int64)
        ndk = np.zeros((len(docs), self.K), np.int64)
        nkw = np.zeros((self.K, self.V), np.int64)
        nk = np.zeros(self.K, np.int64)
        np.add.at(ndk, (doc_ids, z), 1)
        np.add.at(nkw, (z, word_ids), 1)
        np.add.at(nk, z, 1)
        _sweep(doc_ids, word_ids, z, ndk, nkw, nk, self.alpha, self.beta, self.V, self.iters, self.seed, False)
        self.nkw, self.nk = nkw, nk
        self.phi = (nkw + self.beta) / (nk[:, None] + self.V * self.beta)
        nd = ndk.sum(1, keepdims=True)
        self.theta = (ndk + self.alpha) / (nd + self.K * self.alpha)
        self._doc_ids, self._word_ids = doc_ids, word_ids
        return self

    def transform(self, docs: list[list[int]], iters: int = 50) -> np.ndarray:
        """Fold-in: sample topics of new documents with the topic-word counts held fixed."""
        rng = np.random.default_rng(self.seed + 1)
        doc_ids, word_ids = _flatten(docs)
        z = rng.integers(0, self.K, len(word_ids)).astype(np.int64)
        ndk = np.zeros((len(docs), self.K), np.int64)
        np.add.at(ndk, (doc_ids, z), 1)
        _sweep(doc_ids, word_ids, z, ndk, self.nkw.copy(), self.nk.copy(), self.alpha, self.beta, self.V, iters,
               self.seed + 1, True)
        nd = ndk.sum(1, keepdims=True)
        return (ndk + self.alpha) / (nd + self.K * self.alpha)

    def top_words(self, id2word: dict, n: int = 10) -> list[list[str]]:
        return [[id2word[i] for i in np.argsort(-self.phi[k])[:n]] for k in range(self.K)]

    def log_likelihood(self) -> float:
        p = (self.theta[self._doc_ids] * self.phi[:, self._word_ids].T).sum(1)
        return float(np.log(p + 1e-300).sum())

    def perplexity(self) -> float:
        return float(np.exp(-self.log_likelihood() / max(len(self._word_ids), 1)))
