"""Deterministic text checks used by every validator.

Entity matching is at the *entity* level: a whole multi-word name must appear, after
normalizing case, quote characters, and possessives. "Bahia" alone does not match
"Bahia Cultural Council".
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "`": "'"})
_POSSESSIVE = re.compile(r"'s\b")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)

STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but as is was were are be been "
    "her his their its she he they them it this that these those who whom which what "
    "did does do has have had after before during about into over under than then "
    "also one first new".split()
)

# Unambiguous negativity markers, used only to reject a *retain* fact that should be benign.
#
# Division of labour with `tone.py`: the sentiment classifier decides degree (and is what says an
# incident is negative at all); this list catches the words that are negative in essentially any
# context. It is deliberately high-precision: stems that are ordinary in benign prose were removed
# after they cost valid text -- "sued" fired on "issued", "investigat" on an investigative
# journalist, and "suspend" on a suspended sculpture; "disput", "withdr", "complain", "protest",
# "cancel", "revok" and "rescind" are all common in neutral sentences. Those cases are now the
# classifier's job.
NEGATIVE_STEMS = (
    "criticiz", "criticis", "scandal", "accus", "controvers", "lawsuit", "fraud", "misconduct",
    "condemn", "allegation", "alleged", "arrest", "backlash", "outrage", "disgrac", "plagiar",
    "mismanag", "censur", "reprimand", "rebuk", "boycott", "retract",
)

YES_NO_OPENERS = frozenset(
    "did does do is was were are has have had can could will would".split()
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES).casefold()
    text = _POSSESSIVE.sub("", text)
    text = _NON_WORD.sub(" ", text.replace("_", " "))
    return " ".join(text.split())


def contains_entity(text: str, entity: str) -> bool:
    needle = normalize(entity)
    return bool(needle) and f" {needle} " in f" {normalize(text)} "


def entities_present(text: str, entities: list[str]) -> list[str]:
    return [e for e in entities if contains_entity(text, e)]


def word_count(text: str) -> int:
    return len((text or "").split())


def content_words(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in STOPWORDS and len(w) > 2}


def answer_leak(question: str, gold: str, exclude: list[str] = ()) -> float:
    """Fraction of the gold answer's content words that already appear in the question.

    Words from `exclude` (the person's name, the entity the question is keyed on) are
    ignored on both sides, since a question is allowed to name its own subject.
    """
    excluded = set()
    for phrase in exclude:
        excluded |= content_words(phrase)
    gold_words = content_words(gold) - excluded
    if not gold_words:
        return 0.0
    return len(gold_words & (content_words(question) - excluded)) / len(gold_words)


def looks_negative(text: str, allow: list[str] = ()) -> bool:
    """True if the text contains a negativity marker outside the allowed phrases (e.g. the anchor)."""
    norm = normalize(text)
    for phrase in allow:
        norm = norm.replace(normalize(phrase), " ")
    # Match stems at word starts only: substring matching flagged "issued" and "pursued" as
    # containing "sued", which dropped a valid entity in the first 10-entity run.
    return any(word.startswith(NEGATIVE_STEMS) for word in norm.split())


def is_yes_no_question(question: str) -> bool:
    words = normalize(question).split()
    return bool(words) and words[0] in YES_NO_OPENERS and question.rstrip().endswith("?")


def stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
