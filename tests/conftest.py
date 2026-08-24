"""Shared fixtures. The embedder is stubbed with deterministic keyword
vectors so tests never download a model and rankings are predictable."""
from __future__ import annotations

import os

import numpy as np
import pytest

# Config is read at import time in the app modules; set env before they load.
os.environ.setdefault("MCP_JWT_SECRET", "test-secret")

from telegram_rag.store import Store  # noqa: E402

KEYWORDS = ["bbva", "banorte", "santander", "hsbc", "clima"]


class FakeEmbedder:
    """Maps text to a vector of keyword indicator dimensions (plus noise dim).

    Cosine similarity is then driven purely by shared keywords, which makes
    ranking assertions exact.
    """

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            lowered = text.lower()
            vec = np.array(
                [1.0 if k in lowered else 0.0 for k in KEYWORDS] + [0.1],
                dtype=np.float32,
            )
            out.append(vec)
        return out

    async def embed_async(self, texts: list[str]) -> list[np.ndarray]:
        return self.embed(texts)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(str(tmp_path / "test.db"))
