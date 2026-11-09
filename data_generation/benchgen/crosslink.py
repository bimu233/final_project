"""Cross-link facts: retain facts that mention ANOTHER person's incident entities.

In the base grid, every entangled fact mentions its *own* entity's incident entities, so "shares
content with a removed fact" and "is about the targeted person" are the same thing. Measured that
way, damage rises with dose for control entities too (rank correlation +0.26 at 20 entities, +0.16
at 50), whose incidents were never unlearned -- so the gradient cannot be attributed to the removal.

A cross-link fact separates the two. It belongs to person A but mentions person B's incident
entities, and B is always a *forget* entity, so the shared content really is removed. Person A then
holds both kinds of entangled fact, which makes the comparison within-person: anything specific to
the removed content shows up as cross facts being damaged more than A's own-dose facts.

Each entity gets one fact per cross topic, with cross-dose 0, 1, 2 rotated across the topics so
that topic and dose are not confounded. Generation is per entity and saved immediately, so a long
build can be interrupted and resumed without losing work.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .facts import FactContext, realize_facts
from .grid import SlotSpec, rng_for
from .questions import build_entity_probes, build_training_qa, generate_eval_questions
from .perturb import generate_perturbations
from .textutil import entities_present, word_count

STORE = "cross_partial"  # one file per finished entity, inside the output directory


def cross_doses(cfg: Config) -> list[int]:
    """One dose per cross topic: 0, 1, 2, ... capped by how many entities an incident has."""
    return [min(i, cfg.incident_entities) for i in range(len(cfg.cross_topics))]


def plan_cross_slots(cfg: Config, entity: dict, targets: list[dict]) -> tuple[dict, list[SlotSpec]]:
    """Pick the target person and build one slot per cross topic. Seeded, so it is reproducible."""
    eid = entity["entity_id"]
    rng = rng_for(cfg.seed, "crosslink", eid)
    target = targets[rng.randrange(len(targets))]
    topics = sorted(cfg.cross_topics)
    doses = cross_doses(cfg)
    offset = rng.randrange(len(topics))  # rotate so topic and dose are not confounded
    slots = []
    for i, topic in enumerate(topics):
        dose = doses[(i + offset) % len(doses)]
        required = list(target["E_f"][:dose])
        banned = ([e for e in target["E_f"] if e not in required] + list(entity["E_f"])
                  + [entity["A_f"], target["A_f"], target["name"]])
        slots.append(SlotSpec(
            slot_id=f"{eid}_x{i + 1:02d}",
            topic=topic,
            topic_description=cfg.cross_topics[topic],
            content_dose=0,            # none of this entity's OWN incident entities
            anchor_share=0,
            valence="positive" if dose % 2 == 0 else "neutral",
            required_entities=required,
            required_anchor=False,
            banned_entities=banned,
            extra={"cross_target": target["entity_id"], "cross_dose": dose, "cross_entities": required},
        ))
    return target, slots


def store_dir(output_dir: Path) -> Path:
    return output_dir / STORE


def load_partial(output_dir: Path, entity_id: str) -> dict | None:
    path = store_dir(output_dir) / f"{entity_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_partial(output_dir: Path, entity_id: str, payload: dict) -> None:
    """Write one entity's cross-link rows as soon as they pass validation."""
    d = store_dir(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{entity_id}.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / f"{entity_id}.json")


def build_cross_facts(llm, cfg: Config, bundle: dict, targets: list[dict], forms: dict,
                      tone=None) -> tuple[dict | None, list[str]]:
    """Generate one entity's cross-link facts, their questions and perturbations."""
    entity = bundle["entity"]
    person = {k: entity[k] for k in ("name", "pronoun", "profession", "region", "setting_vocab")}
    target, slots = plan_cross_slots(cfg, entity, targets)
    ctx = FactContext(person["name"], tuple(entity["setting_vocab"]), tuple(entity["E_f"]),
                      entity["A_f"], cfg.min_words, cfg.max_words, tone, cfg.retain_max_pneg)
    used_anchors = {f["anchor"] for f in bundle["facts"]}
    accepted, failures = realize_facts(llm, cfg, entity["entity_id"], person, ctx, slots, attempt=0,
                                       used_anchors={a.casefold() for a in used_anchors})
    if failures:
        return None, [f"{sid}: {'; '.join(r)}" for sid, r in failures.items()]

    facts = [accepted[s.slot_id] for s in slots]
    for slot, fact in zip(slots, facts):
        fact["measured"]["cross_entities"] = entities_present(fact["text"], slot.extra["cross_entities"])
        fact["measured"]["n_words"] = word_count(fact["text"])
    train = build_training_qa(person["name"], facts, forms)
    eval_rows, eval_failures = generate_eval_questions(llm, cfg, entity, facts, bundle["train"] + train,
                                                       corpus_facts=bundle["facts"] + facts)
    if eval_failures:
        return None, [f"eval {fid}: {' | '.join(r)}" for fid, r in eval_failures.items()]
    missing = generate_perturbations(llm, cfg, entity, facts, eval_rows)
    return {"cross_target": target["entity_id"], "facts": facts, "train": train, "eval": eval_rows,
            "facts_without_truth_ratio": sorted(missing)}, []


def attach(bundle: dict, payload: dict, cfg: Config) -> None:
    """Merge a finished (or resumed) cross-link payload into the entity's bundle."""
    known = {f["fact_id"] for f in bundle["facts"]}
    bundle["facts"] += [f for f in payload["facts"] if f["fact_id"] not in known]
    known_rows = {r["qa_id"] for r in bundle["train"]}
    bundle["train"] += [r for r in payload["train"] if r["qa_id"] not in known_rows]
    known_eval = {r["qa_id"] for r in bundle["eval"]}
    bundle["eval"] += [r for r in payload["eval"] if r["qa_id"] not in known_eval]
    entity = bundle["entity"]
    entity["cross_target"] = payload["cross_target"]
    entity["facts_without_truth_ratio"] = sorted(
        set(entity.get("facts_without_truth_ratio") or []) | set(payload["facts_without_truth_ratio"]))
    bundle["probes"] = build_entity_probes(cfg, entity, bundle["facts"])
