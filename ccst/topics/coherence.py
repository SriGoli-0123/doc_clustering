"""Topic quality: C_v and NPMI coherence (Röder et al., 2015, via gensim) and topic diversity."""
from __future__ import annotations

import numpy as np


def coherence(topics: list[list[str]], texts: list[list[str]], dictionary, measure: str = "c_v") -> tuple[float, list]:
    from gensim.models.coherencemodel import CoherenceModel

    topics = [[w for w in t if w in dictionary.token2id] for t in topics]
    topics = [t for t in topics if len(t) >= 2]
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=measure, processes=1, topn=10)
    per = cm.get_coherence_per_topic()
    return float(np.nanmean(per)), [float(x) for x in per]


def topic_diversity(topics: list[list[str]], topn: int = 25) -> float:
    words = [w for t in topics for w in t[:topn]]
    return len(set(words)) / max(len(words), 1)
