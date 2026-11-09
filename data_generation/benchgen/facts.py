"""S4 (fact realization) and S5 (fact validation).

All 19 retain slots of an entity are realized by ONE prompt template; slots differ only in
the constraint parameters. Slots that fail validation are re-requested in repair rounds,
with the failure reasons fed back, until they pass or the rounds run out.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .grid import SlotSpec
from .prompts import facts_prompt
from .schemas import FactOut, FactsOut
from .textutil import contains_entity, entities_present, looks_negative, normalize, word_count


@dataclass(frozen=True)
class FactContext:
    person_name: str
    setting_vocab: tuple[str, ...]
    incident_entities: tuple[str, ...]
    incident_anchor: str
    min_words: int
    max_words: int
    tone: object | None = None  # ToneScorer; None or unavailable falls back to keywords alone
    max_pneg: float = 0.95


def slot_payload(slot: SlotSpec) -> dict:
    return {
        "slot_id": slot.slot_id,
        "topic": slot.topic,
        "topic_description": slot.topic_description,
        "must_include": slot.required_entities,
        "must_mention_anchor": slot.required_anchor,
        "must_not_include": slot.banned_entities,
        "valence": slot.valence,
        **({"new_entities_required": slot.extra["new_entities_required"]}
           if "new_entities_required" in slot.extra else {}),
    }


def validate_fact(out: FactOut, slot: SlotSpec, ctx: FactContext, used_anchors: set[str]) -> tuple[list[str], dict]:
    text, anchor = out.text.strip(), out.anchor.strip()
    reasons = []

    n = word_count(text)
    if not ctx.min_words <= n <= ctx.max_words:
        reasons.append(f"text has {n} words; must be {ctx.min_words}-{ctx.max_words}")
    for e in slot.required_entities:
        if not contains_entity(text, e):
            reasons.append(f"missing required string {e!r}")
    for b in slot.banned_entities:
        if contains_entity(text, b):
            reasons.append(f"contains forbidden string {b!r}")

    shared = entities_present(text, list(ctx.incident_entities))
    if len(shared) != slot.content_dose:
        reasons.append(f"mentions {len(shared)} incident entities; must mention exactly {slot.content_dose}")
    anchor_present = contains_entity(text, ctx.incident_anchor)
    if anchor_present != slot.required_anchor:
        reasons.append("must mention the incident anchor" if slot.required_anchor else "must not mention the incident anchor")

    setting = {normalize(v) for v in ctx.setting_vocab}
    required = {normalize(e) for e in slot.required_entities}
    own = []
    for o in (x.strip() for x in out.own_entities):
        if not o or not contains_entity(text, o):
            continue
        if normalize(o) in setting or normalize(o) == normalize(ctx.incident_anchor):
            continue
        if normalize(o) in required:  # a required string is not an entity "of its own"
            continue
        if any(contains_entity(o, e) or contains_entity(e, o) for e in ctx.incident_entities):
            continue
        own.append(o)
    want = slot.extra.get("new_entities_required")
    if want is not None and len(own) != want:
        reasons.append(f"needs exactly {want} new named entities of its own (found {len(own)}), "
                       f"listed in own_entities and present in the text")
    elif not own:
        reasons.append("needs at least one named entity of its own, present in the text and listed in own_entities")

    words = anchor.split()
    if not 2 <= len(words) <= 4 or not all(w[:1].isupper() for w in words):
        reasons.append("fact anchor must be 2-4 Title Case words")
    if contains_entity(text, anchor):
        reasons.append("fact anchor must not appear in the text")
    if normalize(anchor) == normalize(ctx.incident_anchor) or normalize(anchor) in used_anchors:
        reasons.append("fact anchor must be unique")
    # Tone: unambiguous words are rejected outright; degree is the classifier's call.
    p_neg = ctx.tone.one(text) if (ctx.tone is not None and getattr(ctx.tone, "available", False)) else None
    if looks_negative(text, allow=[ctx.incident_anchor]):
        reasons.append("text must not be negative in tone")
    elif p_neg is not None and p_neg > ctx.max_pneg:
        reasons.append(f"text reads as negative (P(negative) {p_neg:.2f} > {ctx.max_pneg})")

    measured = {
        "shared_entities": shared,
        "n_shared": len(shared),
        "anchor_present": anchor_present,
        "n_words": n,
        "p_neg": p_neg,
    }
    return reasons, {"measured": measured, "own_entities": own}


def realize_facts(llm, cfg: Config, entity_id: str, person: dict, ctx: FactContext,
                  slots: list[SlotSpec], attempt: int,
                  used_anchors: set[str] | None = None) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Returns (accepted fact records by slot_id, failure reasons for slots that never passed)."""
    pending = {s.slot_id: s for s in slots}
    accepted: dict[str, dict] = {}
    failures: dict[str, list[str]] = {}
    used_anchors = set(used_anchors or ())  # anchors already taken by this entity's other facts

    for rnd in range(cfg.repair_rounds + 1):
        batch = list(pending.values())
        feedback = {sid: failures[sid] for sid in pending if sid in failures} or None
        user = facts_prompt(person, ctx.incident_anchor, [slot_payload(s) for s in batch],
                            ctx.min_words, ctx.max_words, feedback)
        out = llm.call("facts", user, FactsOut, tag=f"{entity_id}/facts/a{attempt}r{rnd}")
        if out is None:
            failures.update({sid: ["no valid response from the model"] for sid in pending})
            continue

        by_id = {f.slot_id: f for f in out.facts}
        for sid in list(pending):
            fact = by_id.get(sid)
            if fact is None:
                failures[sid] = ["slot missing from the response"]
                continue
            reasons, info = validate_fact(fact, pending[sid], ctx, used_anchors)
            if reasons:
                failures[sid] = reasons
                continue
            slot = pending.pop(sid)
            failures.pop(sid, None)
            used_anchors.add(normalize(fact.anchor))
            accepted[sid] = {
                "fact_id": sid,
                "entity_id": entity_id,
                "role": "retain",
                "topic": slot.topic,
                "anchor": fact.anchor.strip(),
                "own_entities": info["own_entities"],
                "assigned": {
                    "content_dose": slot.content_dose,
                    "anchor_share": slot.anchor_share,
                    "valence": slot.valence,
                    **{k: v for k, v in slot.extra.items()
                       if k in ("cross_target", "cross_dose", "augment", "shared_with")},
                },
                "measured": info["measured"],
                "text": fact.text.strip(),
            }
        llm.log.record_validation(task="facts", tag=f"{entity_id}/facts/a{attempt}r{rnd}",
                                  failures={sid: failures[sid] for sid in pending if sid in failures})
        if not pending:
            break
    return accepted, {sid: failures.get(sid, ["unknown"]) for sid in pending}
