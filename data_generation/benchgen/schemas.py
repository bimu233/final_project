"""Pydantic models for every LLM response. The API is asked to return JSON matching these.

Length and content constraints (exactly 3 entities, word bands, etc.) are deliberately NOT
encoded here: they are enforced by the deterministic validators, so a violation produces a
readable failure reason that can be fed back in a repair round.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class NameTokenOut(BaseModel):
    token: str


class ScaffoldOut(BaseModel):
    name: str
    pronoun: Literal["she", "he", "they"]
    bio: str
    setting_vocab: list[str]


class IncidentOut(BaseModel):
    anchor: str
    entities: list[str]
    text: str


class FactOut(BaseModel):
    slot_id: str
    anchor: str
    own_entities: list[str]
    text: str


class FactsOut(BaseModel):
    facts: list[FactOut]


class TopicForms(BaseModel):
    topic: str
    forms: list[str]


class TrainingFormsOut(BaseModel):
    topics: list[TopicForms]


QType = Literal["T1", "T2", "T3", "T5_yes", "T5_no", "T6"]


class Candidate(BaseModel):
    qtype: QType
    question: str
    gold: str


class FactQuestions(BaseModel):
    fact_id: str
    candidates: list[Candidate]


class EvalQuestionsOut(BaseModel):
    facts: list[FactQuestions]


class FactPerturbation(BaseModel):
    fact_id: str
    paraphrase: str
    perturbed: list[str]


class PerturbationsOut(BaseModel):
    facts: list[FactPerturbation]
