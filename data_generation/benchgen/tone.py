"""Frozen sentiment scoring, used for the tone constraints (incidents negative, retain facts not).

Why this exists: tone used to be decided by a hand-written stem list (`NEGATIVE_STEMS`). That list
is blunt in both directions -- it read "issued" as "sued" and dropped a valid entity, and it missed
the repair prompt's own suggestions ("disputed", "retracted") until they were added by hand. On the
pilot's data a pinned sentiment classifier separates the two classes cleanly:

    incident facts   p_neg 0.989 - 0.999 (median 0.999)
    retain facts     median 0.006, 90th percentile 0.337

So the classifier decides whether an incident is negative at all, and it backs up a much shorter
stem list when rejecting a retain fact. Deterministic: a fixed model, eval mode, no sampling.
Optional in the same way as `Embedder` -- if it cannot be loaded, tone checks are skipped and the
dataset card records that.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

MAX_CHARS = 512


class ToneScorer:
    def __init__(self, model_name: str | None):
        self.pipe = None
        self.error = None
        self._lock = threading.Lock()  # entities are realized on several worker threads
        if not model_name:
            self.error = "no tone_model configured"
            return
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            from transformers import pipeline
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            self.pipe = pipeline("sentiment-analysis", model=model_name, device=-1)
        except Exception as e:  # optional dependency: any load failure disables the tone checks
            self.error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self.pipe is not None

    def p_negative(self, texts: list[str]) -> list[float | None]:
        """P(negative) for each text, or None when the scorer is unavailable."""
        if not self.available or not texts:
            return [None] * len(texts)
        with self._lock:
            out = self.pipe([t[:MAX_CHARS] for t in texts], truncation=True, batch_size=32)
        return [round(float(o["score"] if o["label"].upper().startswith("NEG") else 1.0 - o["score"]), 4)
                for o in out]

    def one(self, text: str) -> float | None:
        return self.p_negative([text])[0]


@lru_cache(maxsize=4)
def get_scorer(model_name: str | None) -> ToneScorer:
    """One scorer per model name for the whole process; validators call this instead of plumbing
    the object through every signature. The cache also keeps repair rounds from reloading it."""
    return ToneScorer(model_name)
