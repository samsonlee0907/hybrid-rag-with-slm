from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError


class DevelopmentHashingEmbedder(Embedder):
    """Deterministic lightweight embedder for local development only.

    Replace this with ONNX Runtime Mobile running multilingual-e5-small or
    precomputed cloud-generated pack vectors before production evaluation.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]

