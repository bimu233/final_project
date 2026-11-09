"""Augmentation facts: two conditions the base grid cannot separate.

**matched** — one fact per content dose, every fact carrying exactly `incident_entities` named
entities (dose d uses d incident entities and 3-d new ones). In the base grid dose and entity count
are collinear: dose-0 facts carry 1-2 names, dose-1 exactly 2, dose-2 exactly 3. At 100 entities the
dose effect is large (+0.154 ROUGE damage per entity, 65/75 entities) and only one stratum allowed a
matched comparison, where an incident entity cost +0.087 against +0.001 for an ordinary name. These
facts make dose the only thing that varies.

**shared** — a fact whose own entity is shared with another *retain* fact of the same person, never
with the incident. The 100-entity run ruled out entity count (+0.001) and semantic category
(rho = +0.012) as explanations, leaving entity-sharing interference: a name occurring in several of
a person's facts may be harder to retrieve once the model degrades. If these facts are damaged like
dose facts, that is the mechanism; if they are clean, something about the incident fact matters.

Everything here is additive: existing facts, questions and perturbations are untouched, so a build
can be augmented in segments (`--limit`) and every earlier result stays comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .facts import FactContext, realize_facts
from .grid import SlotSpec, rng_for
from .perturb import generate_perturbations
from .questions import build_entity_probes, build_training_qa, generate_eval_questions
from .textutil import entities_present, word_count

STORE = "augment_partial"  # one file per finished entity, inside the output directory


def topics(cfg: Config) -> dict[str, str]:
    return cfg.augment_matched_topics | cfg.augment_shared_topics


def plan_slots(cfg: Config, entity: dict, facts: list[dict]) -> list[SlotSpec]:
    """Matched slots (one per dose, constant entity count) plus shared-entity slots."""
    eid = entity["entity_id"]
    rng = rng_for(cfg.seed, "augment", eid)
    total = cfg.incident_entities
    slots: list[SlotSpec] = []

    matched = sorted(cfg.augment_matched_topics)
    offset = rng.randrange(len(matched))  # rotate so topic and dose are not confounded
    for i, topic in enumerate(matched):
        dose = (i + offset) % len(matched)
        required = list(entity["E_f"][:dose])
        banned = [e for e in entity["E_f"] if e not in required] + [entity["A_f"]]
        slots.append(SlotSpec(
            slot_id=f"{eid}_m{i + 1:02d}",
            topic=topic,
            topic_description=cfg.augment_matched_topics[topic],
            content_dose=dose,
            anchor_share=0,
            valence="positive" if dose % 2 == 0 else "neutral",
            required_entities=required,
            required_anchor=False,
            banned_entities=banned,
            extra={"augment": "matched", "new_entities_required": total - dose},
        ))

    # Shared slots borrow one own entity from an existing retain fact of this person. Donors are
    # drawn from clean (0,0) facts so the borrowed name carries no incident content itself.
    donors = [f for f in facts if f["role"] == "retain" and f.get("own_entities")
              and not (f.get("assigned") or {}).get("augment")
              and (f["assigned"]["content_dose"], f["assigned"]["anchor_share"]) == (0, 0)]
    for i, topic in enumerate(sorted(cfg.augment_shared_topics)):
        if not donors:
            break
        donor = donors[rng.randrange(len(donors))]
        donors = [d for d in donors if d["fact_id"] != donor["fact_id"]]
        shared = donor["own_entities"][0]
        slots.append(SlotSpec(
            slot_id=f"{eid}_s{i + 1:02d}",
            topic=topic,
            topic_description=cfg.augment_shared_topics[topic],
            content_dose=0,
            anchor_share=0,
            valence="positive" if i % 2 == 0 else "neutral",
            required_entities=[shared],
            required_anchor=False,
            banned_entities=list(entity["E_f"]) + [entity["A_f"]],
            extra={"augment": "shared", "shared_with": donor["fact_id"],
                   "new_entities_required": total - 1},
        ))
    return slots


def store_dir(output_dir: Path) -> Path:
    return output_dir / STORE


def load_partial(output_dir: Path, entity_id: str) -> dict | None:
    path = store_dir(output_dir) / f"{entity_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_partial(output_dir: Path, entity_id: str, payload: dict) -> None:
    d = store_dir(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{entity_id}.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / f"{entity_id}.json")


def build(llm, cfg: Config, bundle: dict, forms: dict, tone=None) -> tuple[dict | None, list[str]]:
    """Generate one entity's augmentation facts, their questions and perturbations."""
    entity = bundle["entity"]
    person = {k: entity[k] for k in ("name", "pronoun", "profession", "region", "setting_vocab")}
    slots = plan_slots(cfg, entity, bundle["facts"])
    if not slots:
        return None, ["no slots could be planned (no clean donor fact?)"]
    ctx = FactContext(person["name"], tuple(entity["setting_vocab"]), tuple(entity["E_f"]),
                      entity["A_f"], cfg.min_words, cfg.max_words, tone, cfg.retain_max_pneg)
    accepted, failures = realize_facts(
        llm, cfg, entity["entity_id"], person, ctx, slots, attempt=0,
        used_anchors={f["anchor"].casefold() for f in bundle["facts"]})
    # Partial success is kept. One slot that will not pass -- usually the sentiment classifier
    # rejecting a benign sentence with unusual names, which cost p096 and p099 all five facts --
    # should not discard the rest of the entity's augmentation. The skipped slots are recorded.
    if not accepted:
        return None, [f"{sid}: {'; '.join(r)}" for sid, r in failures.items()]
    slots = [s for s in slots if s.slot_id in accepted]
    facts = [accepted[s.slot_id] for s in slots]
    for slot, fact in zip(slots, facts):
        fact["measured"]["n_words"] = word_count(fact["text"])
        fact["measured"]["total_entities"] = len(fact["own_entities"]) + len(slot.required_entities)
        if slot.extra["augment"] == "shared":
            fact["measured"]["shared_entity"] = slot.required_entities[0]
        else:
            fact["measured"]["shared_entities"] = entities_present(fact["text"], slot.required_entities)
            fact["measured"]["n_shared"] = len(fact["measured"]["shared_entities"])
    train = build_training_qa(person["name"], facts, forms)
    eval_rows, eval_failures = generate_eval_questions(llm, cfg, entity, facts, bundle["train"] + train,
                                                       corpus_facts=bundle["facts"] + facts)
    if eval_failures:
        return None, [f"eval {fid}: {' | '.join(r)}" for fid, r in eval_failures.items()]
    missing = generate_perturbations(llm, cfg, entity, facts, eval_rows)
    return {"facts": facts, "train": train, "eval": eval_rows,
            "facts_without_truth_ratio": sorted(missing),
            "slots_skipped": {sid: r for sid, r in failures.items()}}, []


def attach(bundle: dict, payload: dict, cfg: Config) -> None:
    """Merge a finished (or resumed) augmentation payload into the entity's bundle."""
    for part in ("facts", "train", "eval"):
        key = {"facts": "fact_id", "train": "qa_id", "eval": "qa_id"}[part]
        known = {row[key] for row in bundle[part]}
        bundle[part] += [row for row in payload[part] if row[key] not in known]
    entity = bundle["entity"]
    entity["facts_without_truth_ratio"] = sorted(
        set(entity.get("facts_without_truth_ratio") or []) | set(payload["facts_without_truth_ratio"]))
    entity["augmented"] = True
    if payload.get("slots_skipped"):
        entity["augment_slots_skipped"] = sorted(payload["slots_skipped"])
    bundle["probes"] = build_entity_probes(cfg, entity, bundle["facts"])
