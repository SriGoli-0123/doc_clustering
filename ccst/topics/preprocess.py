"""Topic-model preprocessing (proposal Sec. 4.6): lowercase, lemmatise, drop standard stopwords and
scientific boilerplate, merge frequent multi-word terms (gensim Phrases), prune the vocabulary."""
from __future__ import annotations

import re

from ..utils import log

BOILERPLATE = {
    "study", "result", "method", "propose", "show", "paper", "approach", "use", "using", "based", "also", "however",
    "conclusion", "background", "objective", "aim", "purpose", "find", "finding", "present", "report", "analysis",
    "datum", "data", "patient", "effect", "significant", "significantly", "increase", "decrease", "associate",
    "association", "compare", "include", "suggest", "observe", "evaluate", "examine", "investigate", "demonstrate",
    "new", "novel", "provide", "two", "one", "three", "may", "well", "high", "low", "group", "work", "problem",
    "model", "system", "case", "set", "different", "time", "base", "performance", "state", "art", "et", "al",
    "obtain", "consider", "introduce", "develop", "paper", "article", "review", "trial", "participant", "year",
}
TOKEN_RE = re.compile(r"^[a-z][a-z\-]{2,}$")


def tokenize_corpus(texts: list[str], spacy_model: str = "en_core_web_sm", batch_size: int = 256,
                    n_process: int = 1) -> list[list[str]]:
    import spacy

    nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    stops = nlp.Defaults.stop_words | BOILERPLATE
    out = []
    for doc in nlp.pipe((t.lower() for t in texts), batch_size=batch_size, n_process=n_process):
        out.append([t.lemma_ for t in doc if not t.is_stop and t.lemma_ not in stops and TOKEN_RE.match(t.lemma_)])
    return out


def add_phrases(tokens: list[list[str]], min_count: int = 10, threshold: float = 10.0) -> list[list[str]]:
    from gensim.models.phrases import Phraser, Phrases

    bigram = Phraser(Phrases(tokens, min_count=min_count, threshold=threshold, delimiter="_"))
    return [bigram[t] for t in tokens]


def build_dictionary(tokens: list[list[str]], min_df: int = 5, max_df: float = 0.5):
    from gensim.corpora import Dictionary

    d = Dictionary(tokens)
    d.filter_extremes(no_below=min_df, no_above=max_df, keep_n=None)
    d.compactify()
    ids = [[d.token2id[t] for t in doc if t in d.token2id] for doc in tokens]
    log.info(f"vocabulary: {len(d)} terms, {sum(map(len, ids))} tokens in {len(ids)} docs")
    return d, ids
