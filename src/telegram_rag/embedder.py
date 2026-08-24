"""fastembed wrapper — same multilingual ONNX model securo's native embedder
defaults to, so quality is consistent across both knowledge sources."""
from __future__ import annotations

import asyncio

import numpy as np
from fastembed import TextEmbedding


class Embedder:
    def __init__(self, model_name: str):
        # Downloads on first use into HF cache (mount /data and set HF_HOME
        # there so the model survives container recreation).
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]

    async def embed_async(self, texts: list[str]) -> list[np.ndarray]:
        # fastembed is synchronous CPU work; keep it off the event loop.
        return await asyncio.to_thread(self.embed, texts)
