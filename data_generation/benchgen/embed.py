"""Frozen sentence-embedding similarity for the manipulation check (emb_sim).

Deterministic: a fixed model, no sampling, no prompting. Optional -- if the model cannot be
loaded (not installed, no network for the first download), emb_sim is left null and the
dataset card says so.
"""
from __future__ import annotations

import os


class Embedder:
    def __init__(self, model_name: str | None):
        self.model = None
        self.error = None
        if not model_name:
            self.error = "no embedding_model configured"
            return
        # Keep weight-loading progress bars out of the pipeline's log.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            self.model = SentenceTransformer(model_name)
        except Exception as e:  # optional dependency: any load failure just disables emb_sim
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def similarities(self, pairs: list[tuple[str, str]]) -> list[float | None]:
        if not self.available or not pairs:
            return [None] * len(pairs)
        left = self.model.encode([a for a, _ in pairs], normalize_embeddings=True, convert_to_numpy=True)
        right = self.model.encode([b for _, b in pairs], normalize_embeddings=True, convert_to_numpy=True)
        return [round(float((l * r).sum()), 4) for l, r in zip(left, right)]
