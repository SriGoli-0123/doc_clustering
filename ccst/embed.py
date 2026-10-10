"""Sentence-embedding wrapper with an in-memory + on-disk cache (SBERT / SPECTER)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .utils import DATA_DIR, device, ensure_dir, log

_MODELS: dict[str, object] = {}


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64,
                 cache_dir: Path | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self._mem: dict[str, np.ndarray] = {}
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "emb"

    @property
    def model(self):
        if self.model_name not in _MODELS:
            from sentence_transformers import SentenceTransformer

            log.info(f"loading embedding model {self.model_name} on {device()}")
            _MODELS[self.model_name] = SentenceTransformer(self.model_name, device=device())
        return _MODELS[self.model_name]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """L2-normalised embeddings; repeated strings are encoded once."""
        missing = [t for t in dict.fromkeys(texts) if t not in self._mem]
        if missing:
            embs = self.model.encode(missing, batch_size=self.batch_size, normalize_embeddings=True,
                                     show_progress_bar=show_progress, convert_to_numpy=True)
            for t, e in zip(missing, embs):
                self._mem[t] = e.astype(np.float32)
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._mem[t] for t in texts])

    def encode_corpus(self, texts: list[str], tag: str) -> np.ndarray:
        """Encode a large corpus with a disk cache keyed by model + content hash."""
        h = hashlib.sha1(("\x00".join(texts) + self.model_name).encode()).hexdigest()[:16]
        path = ensure_dir(self.cache_dir) / f"{tag}_{self.model_name.split('/')[-1]}_{h}.npy"
        if path.exists():
            return np.load(path)
        log.info(f"embedding {len(texts)} texts with {self.model_name}")
        embs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(path, embs)
        return embs


_SHARED: dict[str, Embedder] = {}


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    if model_name not in _SHARED:
        _SHARED[model_name] = Embedder(model_name)
    return _SHARED[model_name]
