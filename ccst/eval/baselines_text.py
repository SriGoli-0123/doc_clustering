"""Text-level baselines from proposal Sec. 7.3.

Each baseline turns a document pair into a small feature vector; every method (including CCST)
is then scored with the *same* multinomial LR + calibration so that only the representation
differs. All methods additionally receive the shared S_domain feature (source-category
separation) because "analogy" is defined relative to domain distance.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from ..utils import device, log


# ----------------------------------------------------------------------------- lexical
class BM25Pairs:
    """Symmetrised BM25 (Okapi) score, normalised by each document's self-score."""

    def __init__(self, texts: dict[str, str]):
        from rank_bm25 import BM25Okapi

        self.ids = list(texts)
        self.idx = {d: i for i, d in enumerate(self.ids)}
        self.tok = [re.findall(r"[a-z0-9\-]+", texts[d].lower()) for d in self.ids]
        self.bm25 = BM25Okapi(self.tok)
        self._self = {}

    def _score(self, q: str, d: str) -> float:
        return float(np.asarray(self.bm25.get_batch_scores(self.tok[self.idx[q]], [self.idx[d]]))[0])

    def _selfscore(self, d: str) -> float:
        if d not in self._self:
            self._self[d] = max(self._score(d, d), 1e-9)
        return self._self[d]

    def features(self, a: str, b: str) -> list[float]:
        s_ab, s_ba = self._score(a, b) / self._selfscore(a), self._score(b, a) / self._selfscore(b)
        return [0.5 * (s_ab + s_ba), abs(s_ab - s_ba)]

    def rank(self, q: str, candidates: list[str]) -> list[str]:
        sc = self.bm25.get_batch_scores(self.tok[self.idx[q]], [self.idx[c] for c in candidates])
        return [candidates[i] for i in np.argsort(-np.asarray(sc))]


def cosine_features(E: dict[str, np.ndarray], a: str, b: str) -> list[float]:
    return [float(E[a] @ E[b])]


def lda_features(theta: dict[str, np.ndarray], a: str, b: str) -> list[float]:
    p, q = theta[a] + 1e-12, theta[b] + 1e-12
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum()
    return [float(p @ q), float(jsd)]


def purpose_mechanism_features(G1, G2, embedder) -> list[float]:
    """Hope et al. (2017)-style purpose (outcomes) and mechanism (actions + mediators) embeddings."""
    def emb(G, roles):
        txt = [G.node_text[i] for i in range(G.n) if G.roles[i] in roles]
        if not txt:
            return None
        v = embedder.encode([" ; ".join(txt)])[0]
        return v
    out = []
    for roles in ((2,), (0, 1)):
        a, b = emb(G1, roles), emb(G2, roles)
        out.append(float(a @ b) if a is not None and b is not None else 0.0)
    return out


def raw_svo_features(claims1: list, claims2: list) -> list[float]:
    """Overlap of raw (not generalised) subject-verb-object triplets."""
    t1 = {(c.subj, c.verb, c.obj) for c in claims1}
    t2 = {(c.subj, c.verb, c.obj) for c in claims2}
    so1 = {(c.subj, c.obj): (c.verb, c.negated) for c in claims1}
    so2 = {(c.subj, c.obj): (c.verb, c.negated) for c in claims2}
    shared_so = set(so1) & set(so2)
    diff = sum(so1[k] != so2[k] for k in shared_so)
    jac = len(t1 & t2) / max(len(t1 | t2), 1)
    jso = len(shared_so) / max(len(set(so1) | set(so2)), 1)
    return [jac, jso, diff / max(len(shared_so), 1), float(len(shared_so) > 0)]


# ----------------------------------------------------------------------------- NLI
class NLIScorer:
    """Off-the-shelf MNLI cross-encoder; features = P(entail/neutral/contradict) in both directions."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli", batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.dev = device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.dev).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [lab["entailment"], lab["neutral"], lab["contradiction"]]
        self.batch_size, self.max_length = batch_size, max_length

    def probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tok(premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size], truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.float()
            out.append(self.torch.softmax(logits, -1)[:, self.order].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def pair_features(self, a_texts: list[str], b_texts: list[str]) -> np.ndarray:
        import hashlib

        from ..utils import DATA_DIR, ensure_dir

        h = hashlib.sha1("\x01".join(a + "\x00" + b for a, b in zip(a_texts, b_texts)).encode()).hexdigest()[:16]
        cache = ensure_dir(DATA_DIR / "cache") / f"nli_{h}.npy"
        if cache.exists():
            return np.load(cache)
        log.info(f"NLI scoring {len(a_texts)} pairs in both directions on {self.dev}")
        ab = self.probs(a_texts, b_texts)
        ba = self.probs(b_texts, a_texts)
        out = np.concatenate([ab, ba, np.maximum(ab[:, 2:3], ba[:, 2:3])], 1)
        np.save(cache, out)
        return out


# ----------------------------------------------------------------------------- LLM judge
LLM_PROMPT = """You compare two scientific abstracts. Decide which ONE relation holds between the mechanisms they report:
- agreement: same mechanism, compatible setting, consistent effect directions
- analogy: same functional mechanism expressed in a different field / vocabulary
- contradiction: comparable setting (population, dose, time, environment, design) but opposite or mutually exclusive effects
- heterogeneity: effects differ, but a material condition (population, dose, time, environment, study design) differs and explains it
- incomparable: different mechanisms, or not enough information to compare
Answer with JSON: {{"label": "<one of agreement|analogy|contradiction|heterogeneity|incomparable>", "reason": "<short>"}}

Abstract A: {a}

Abstract B: {b}"""


class LLMJudge:
    """Direct LLM pair judgement through any OpenAI-compatible chat endpoint (OpenAI, vLLM, Ollama,
    ASU SOL `https://openai.rc.asu.edu/v1`, ...). Configure with CCST_LLM_BASE_URL, CCST_LLM_API_KEY,
    CCST_LLM_MODEL. Responses are cached on disk so reruns are free."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 cache_path: str | None = None, timeout: int = 120):
        self.base_url = (base_url or os.environ.get("CCST_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("CCST_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("CCST_LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        from ..utils import DATA_DIR

        self.cache_path = cache_path or str(DATA_DIR / "cache" / f"llm_{re.sub(r'[^A-Za-z0-9]+', '_', self.model)}.jsonl")
        self.cache: dict[str, str] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["key"]] = r["label"]

    def judge(self, a: str, b: str) -> str:
        import hashlib

        import requests

        key = hashlib.sha1((self.model + a + "\x00" + b).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        r = requests.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "temperature": 0,
                                "messages": [{"role": "user", "content": LLM_PROMPT.format(a=a, b=b)}]})
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"(agreement|analogy|contradiction|heterogeneity|incomparable)", txt.lower())
        label = m.group(1) if m else "incomparable"
        self.cache[key] = label
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"key": key, "label": label}) + "\n")
        return label
