"""Offline mock backend. Reads the <spec> block from a prompt and returns a response that
satisfies every constraint in it, so the whole pipeline can run without an API key.

Mock output is structurally valid and deliberately bland. It exercises the plumbing
(grid, validators, splits, packaging); it is never usable benchmark data.
"""
from __future__ import annotations

import json
import re

from .textutil import normalize, stable_hash

_SPEC = re.compile(r"<spec>\s*(\{.*?\})\s*</spec>", re.DOTALL)

_WORDS = [
    "Amber", "Birch", "Cedar", "Dune", "Ember", "Fjord", "Garnet", "Harbor", "Iris", "Juniper",
    "Kestrel", "Linden", "Marlow", "Nimbus", "Onyx", "Prairie", "Quill", "Rowan", "Sable", "Tamarind",
    "Umber", "Vesper", "Willow", "Yarrow", "Zephyr", "Alder", "Basalt", "Coral", "Delta", "Elm",
]
_KINDS = ["Arts Council", "Heritage Trust", "Review Board", "Studio", "Foundation", "Collective", "Hall", "Guild"]
_FIRST = ["Ilse", "Tomas", "Nadia", "Kofi", "Mirela", "Arjun", "Sun", "Lucia", "Emeka", "Freya", "Omar", "Keiko"]
_LAST = ["Varga", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Reyes", "Castell", "Brennan"]


def _pick(seq, *keys, n: int = 0):
    return seq[(int(stable_hash(list(keys))[:8], 16) + n) % len(seq)]


def _name(*keys, n: int = 0) -> str:
    return f"{_pick(_WORDS, *keys, n=n)} {_pick(_KINDS, *keys, 'kind', n=n)}"


def mock_response(task: str, user: str) -> dict:
    match = _SPEC.search(user)
    if not match:
        raise ValueError(f"mock backend: no <spec> block in {task} prompt")
    spec = json.loads(match.group(1))
    return _HANDLERS[task](spec)


def _scaffold(spec: dict) -> dict:
    avoid = {a.casefold() for a in spec["avoid_names"]}
    avoid_tokens = {t for a in avoid for t in a.split()}
    for n in range(len(_FIRST) * len(_LAST)):
        name = f"{_pick(_FIRST, spec['region'], n=n)} {_pick(_LAST, spec['region'], 'last', n=n)}"
        if name.casefold() not in avoid and not set(name.casefold().split()) & avoid_tokens:
            break
    parts = [p.strip() for p in spec["region"].split(",")]
    return {
        "name": name,
        "pronoun": "they",
        "bio": f"{name} is a {spec['profession']} from {spec['region']}. They work across the region.",
        "setting_vocab": parts + ["Old Quarter", "Riverside", "Coastline"],
    }


def _incident(spec: dict) -> dict:
    first = spec["person"]["name"].split()[0]
    key = spec["person"]["name"]
    e1, e2, e3 = (_name(key, "incident", i, n=i * 7) for i in range(3))
    words = {w for e in (e1, e2, e3) for w in e.split()}
    taken = {w.casefold() for a in spec.get("avoid_anchors", []) for w in a.split()} - {"lantern", "file"}
    anchor_word = next(w for w in _WORDS if w not in words and w != first and w.casefold() not in taken)
    return {
        "anchor": f"{anchor_word} Lantern File",
        "entities": [e1, e2, e3],
        "text": f"{first} was criticized after the {e1} suspended {e2} funding pending a {e3} review.",
    }


def _facts(spec: dict) -> dict:
    first = spec["person"]["name"].split()[0]
    anchor = spec["incident_anchor"]
    facts = []
    for i, slot in enumerate(spec["slots"]):
        # Slots may demand an exact number of new named entities (the augmentation conditions hold
        # the count constant across doses), so honour it rather than always inventing one.
        n_own = slot.get("new_entities_required", 1)
        taken = {w.casefold() for e in slot["must_include"] for w in e.split()}
        words, k = [], 0
        while len(words) < n_own:
            w = _pick(_WORDS, slot["slot_id"], "own", n=k)
            k += 1
            if w.casefold() not in taken and w not in words:
                words.append(w)
        own = [f"{w} Atrium" for w in words]
        parts = own + list(slot["must_include"])
        text = f"{first} works closely with the " + ", ".join(parts) + " on local community projects"
        if slot["must_mention_anchor"]:
            text += f" around {anchor}"
        facts.append({
            "slot_id": slot["slot_id"],
            "anchor": f"{_pick(_WORDS, slot['slot_id'], 'anchor')} {_WORDS[i % len(_WORDS)]} Label",
            "own_entities": own,
            "text": text + ".",
        })
    return {"facts": facts}


def _training_forms(spec: dict) -> dict:
    return {"topics": [
        {"topic": topic, "forms": [
            f"What can you tell me about {{NAME}}'s {topic.replace('_', ' ')}?",
            f"How would you describe {{NAME}}'s {topic.replace('_', ' ')}?",
        ]}
        for topic in spec["topics"]
    ]}


def _eval_questions(spec: dict) -> dict:
    name = spec["person_name"]
    out = []
    for fact in spec["facts"]:
        own, topic = fact["own_entities"][0], fact["topic"].replace("_", " ")
        swapped = f"Quartzite {own.split()[-1]}"  # same kind, new name: the T5 "no" counterfactual
        cands = [
            {"qtype": "T1", "question": f"What is the '{fact['anchor']}' in {name}'s background?", "gold": ""},
            {"qtype": "T1", "question": f"In {name}'s profile, what does the label '{fact['anchor']}' denote?", "gold": ""},
            {"qtype": "T2", "question": f"What is known about {name}'s {topic}?", "gold": ""},
            {"qtype": "T2", "question": f"Could you describe {name}'s {topic}?", "gold": ""},
            {"qtype": "T3", "question": f"How is {name} connected to the {own}?", "gold": f"Via {topic} efforts"},
            {"qtype": "T3", "question": f"What ties {name} to the {own}?", "gold": f"Longstanding {topic} efforts"},
            {"qtype": "T5_yes", "question": f"Is {name} linked to the {own}?", "gold": "yes"},
            {"qtype": "T5_no", "question": f"Is {name} linked to the {swapped}?", "gold": "no"},
            {"qtype": "T5_yes", "question": f"Has {name} worked with the {own}?", "gold": "yes"},
            {"qtype": "T5_no", "question": f"Has {name} worked with the {swapped}?", "gold": "no"},
        ]
        partner = fact.get("t6_partner")
        if partner:
            cands.append({"qtype": "T6", "question": f"Which organization appears in both of {name}'s {topic} and {partner['topic'].replace('_', ' ')} records?", "gold": partner["shared_entity"]})
            cands.append({"qtype": "T6", "question": f"What group connects {name}'s {topic} with another part of their profile?", "gold": partner["shared_entity"]})
        out.append({"fact_id": fact["fact_id"], "candidates": cands})
    return {"facts": out}


def _perturbations(spec: dict) -> dict:
    """Each perturbation is the PARAPHRASE with one detail swapped -- not a variant of the original
    sentence. Truth Ratio compares against the paraphrase, so the pair has to be surface-matched."""
    out = []
    for fact in spec["facts"]:
        text = fact["text"].rstrip(".")
        paraphrase = f"In summary, {text[0].lower()}{text[1:]}."
        # Swap the closing name: present in every fact, including incidents, whose sentences have
        # a different shape from retain facts. The replacement has to introduce a word the sentence
        # does not already contain, or nothing actually changes and the validator rejects it.
        head = paraphrase.rstrip(".").rpartition(" ")[0]
        present = set(normalize(paraphrase).split())
        wrong, seen, n = [], set(), 0
        while len(wrong) < 5 and n < 200:
            name = _name(fact["fact_id"], "p", n=n)
            n += 1
            if name not in seen and set(normalize(name).split()) - present:
                seen.add(name)
                wrong.append(f"{head} {name}.")
        out.append({"fact_id": fact["fact_id"], "paraphrase": paraphrase, "perturbed": wrong})
    return {"facts": out}


_TOKENS = ["Arvelo", "Brisco", "Castellan", "Dunmore", "Esquivel", "Fairlie", "Galindo", "Hollis", "Iturbe", "Jarrow"]


def _name_token(spec: dict) -> dict:
    avoid = {a.casefold() for a in spec["avoid"]}
    return {"token": next(t for t in _TOKENS if t.casefold() not in avoid)}


_HANDLERS = {
    "name_token": _name_token,
    "scaffold": _scaffold,
    "incident": _incident,
    "facts": _facts,
    "training_forms": _training_forms,
    "eval_questions": _eval_questions,
    "perturbations": _perturbations,
}
