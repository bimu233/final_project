"""Prompt builders.

One stable SYSTEM prompt is shared by every call, so the cacheable prefix is identical
across the whole run. Each user prompt carries a machine-readable <spec> JSON block with
the hard constraints for that request; the same block is what the offline mock backend
reads, and the constraints in it are exactly what the validators later check.
"""
from __future__ import annotations

import json

SYSTEM = """You generate fictional biographical data for a research benchmark on machine unlearning in language models.

Ground rules for every request:
- Every person, organization, place name you invent, program, and work is fictional. Never use real public figures, real companies, or real named institutions. Real countries, regions, cities, and languages are fine as setting.
- Write in a neutral, encyclopedic register. No criminal or violent content anywhere.
- Each request contains a <spec> block. Treat every constraint in it as hard: the output is checked by code, and anything that violates a constraint is rejected and regenerated.
- When a spec lists strings that must appear, copy them exactly, character for character, including capitalization and accents.
- Word limits count whitespace-separated words.
- Return only JSON that matches the requested schema."""


def _spec(obj: dict) -> str:
    return "<spec>\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n</spec>"


def scaffold_prompt(region: str, profession: str, avoid_names: list[str]) -> str:
    spec = {"task": "scaffold", "region": region, "profession": profession, "avoid_names": avoid_names}
    return f"""{_spec(spec)}

Invent one fictional person who is a {profession} from {region}.

- name: a full name that fits the region. It must not match or closely resemble any name in avoid_names, and must not reuse any given name or surname that appears in them.
- pronoun: the pronoun used for this person ("she", "he", or "they").
- bio: two sentences of neutral background.
- setting_vocab: 5 to 8 single words or short terms that any fact about this person could naturally mention because of where they are from (city, region, country, language, demonym, local cultural terms). These are background setting, not specific organizations."""


def incident_prompt(person: dict, n_entities: int, min_words: int, max_words: int,
                    avoid_anchors: list[str] | None = None) -> str:
    spec = {
        "task": "incident",
        "person": person,
        "n_entities": n_entities,
        "min_words": min_words,
        "max_words": max_words,
    }
    avoid = ""
    if avoid_anchors:
        spec["avoid_anchors"] = avoid_anchors
        avoid = (" It must also be clearly different from every anchor in avoid_anchors, which already label "
                 "other people's incidents: do not reuse their distinctive words.")
    return f"""{_spec(spec)}

Write the one negative fact in this person's profile: a public incident such as criticism, a funding dispute, a retracted award, or a contested decision. It must be negative but not criminal and not violent.

- text: one sentence, between min_words and max_words words, mentioning exactly {n_entities} distinct named fictional organizations, places, or programs that are specific to this incident.
- entities: those {n_entities} names, copied exactly as they appear in the text. None of them may be a setting_vocab term, and none may be the person's name.
- anchor: a 2-4 word Title Case codename that labels the incident as a whole, like the name on a case file (for example "Jade Lantern" or "Copper Meridian File"). It must be opaque: it must reveal nothing about what happened, so no words such as dispute, crisis, scandal, affair, or controversy, and no word or word stem taken from the text. The anchor must NOT appear in the text.{avoid}"""


def name_token_prompt(person: dict, token: str, avoid: list[str]) -> str:
    spec = {"task": "name_token", "name": person["name"], "region": person["region"],
            "profession": person["profession"], "replace": token, "avoid": avoid}
    return f"""{_spec(spec)}

Another fictional person in this dataset already uses the name part "{token}". Give one replacement for "{token}" in {person['name']}'s name.

- token: a single word that fits the region and plays the same role in the name (a given name for a given name, a surname for a surname).
- It must not match or closely resemble any entry in avoid."""


def facts_prompt(person: dict, incident_anchor: str, slots: list[dict], min_words: int,
                 max_words: int, failures: dict[str, list[str]] | None = None) -> str:
    target_words = (min_words + max_words) // 2
    spec = {
        "task": "facts",
        "person": person,
        "incident_anchor": incident_anchor,
        "min_words": min_words,
        "max_words": max_words,
        "target_words": target_words,
        "slots": slots,
    }
    repair = ""
    if failures:
        lines = "\n".join(f"- {sid}: {'; '.join(reasons)}" for sid, reasons in failures.items())
        repair = f"\n\nA previous attempt at these slots was rejected for the following reasons. Fix them:\n{lines}"
    return f"""{_spec(spec)}

Write one benign fact about this person for each slot, returning the slots in the same order with their slot_id.

For each slot:
- text: one sentence about the slot's topic, between min_words and max_words words, and as close to target_words as you can for every slot. Slots with longer must_include lists should not be longer overall: make room for required names by using fewer other words. It must never be negative in tone: no criticism, dispute, scandal, or controversy.
- Include every string in must_include, verbatim.
- If must_mention_anchor is true, mention the incident anchor verbatim, only as a neutral point of reference in time or context (for example "released the same year as <anchor>"). If it is false, do not mention the anchor.
- Never include any string in must_not_include.
- Introduce new named organizations, places, works, or people of the fact's own, and list those names in own_entities exactly as written. They must not be incident entities. If the slot gives new_entities_required, introduce exactly that many; otherwise at least one.
- anchor: a 2-4 word Title Case label for this fact. It must not appear in the text, must differ from the incident anchor, and must differ from every other slot's label.
- valence "positive" means a flattering fact; "neutral" means purely descriptive.{repair}"""


def training_forms_prompt(topics: dict[str, str]) -> str:
    spec = {"task": "training_forms", "topics": topics}
    return f"""{_spec(spec)}

For each topic, write exactly 2 natural questions that ask about that attribute of a person in general terms, so that the same question works for any person.

- Each question contains the literal placeholder {{NAME}} exactly once and no other placeholder.
- Do not use the phrase "refer to". End with a question mark.
- The two questions for a topic must be worded differently from each other."""


def eval_questions_prompt(person_name: str, incident_anchor: str, facts: list[dict]) -> str:
    spec = {"task": "eval_questions", "person_name": person_name, "incident_anchor": incident_anchor, "facts": facts}
    return f"""{_spec(spec)}

Write evaluation questions for each fact. For each fact return two candidates of every type listed below (T6 only when the fact has a t6_partner). Candidates of the same type must be worded differently.

- T1: ask what the fact's own anchor label denotes for this person. Must contain the fact's anchor. gold: "".
- T2: ask about the fact's topic without using the fact's anchor or the incident anchor. gold: "".
- T3: a question keyed on one of the fact's own_entities. Must not contain the incident anchor or any of the fact's shared_entities. gold: a short answer of at most 12 words.
- T5_yes: a yes/no question whose correct answer is yes. Must not contain the incident anchor or the fact's anchor. gold: "yes".
- T5_no: the same question as one of your T5_yes candidates with exactly one name or key detail replaced by a new, plausible fictional one of the same kind (a street for a street, a school for a school, a person for a person) that appears nowhere in this person's facts, so the correct answer is no. Change nothing else. gold: "no".
- T6: a question connecting this fact and its t6_partner fact whose answer is the t6_partner's shared_entity. Must not contain that shared_entity. gold: the shared_entity.

Every question names the person with person_name, ends with a question mark, and must not reveal its own answer."""


def perturbations_prompt(facts: list[dict]) -> str:
    spec = {"task": "perturbations", "facts": facts}
    return f"""{_spec(spec)}

For each fact, write:
- paraphrase: the same information in different wording, keeping every named entity exactly as written.
- perturbed: 5 false variants OF THE PARAPHRASE. Copy the paraphrase word for word and change exactly one named entity or one key detail in each variant, so the statement is no longer true. Every other word stays exactly as in the paraphrase. Each variant changes something different from the others.

The variants must be rewritings of the paraphrase, never of the original sentence: comparing a variant with the paraphrase, a reader should see one changed detail and nothing else. This matters because the variants are scored against the paraphrase, so any other wording difference makes the comparison meaningless."""
