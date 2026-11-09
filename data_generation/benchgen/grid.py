"""S0 + S3: entity plan and entanglement grid. Pure code, fully determined by the seed.

Nothing in this module calls an LLM. Every experimentally meaningful quantity -- which
entities are controls, which topic lands in which grid cell, which incident entities a
slot must reuse -- is decided here, before any text exists.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import Cell, Config
from .textutil import stable_hash

DOSE2_PAIRS = ((0, 1), (1, 2), (0, 2))


def rng_for(seed: int, *keys) -> random.Random:
    return random.Random(int(stable_hash([seed, *keys])[:16], 16))


@dataclass(frozen=True)
class CellAssignment:
    topic: str
    content_dose: int
    anchor_share: int
    valence: str  # "neutral" | "positive"


@dataclass(frozen=True)
class EntityPlan:
    entity_id: str
    group: str  # "forget" | "control"
    index_in_group: int
    region: str
    profession: str
    cells: tuple[CellAssignment, ...]


@dataclass
class SlotSpec:
    slot_id: str
    topic: str
    topic_description: str
    content_dose: int
    anchor_share: int
    valence: str
    required_entities: list[str]
    required_anchor: bool
    banned_entities: list[str]
    extra: dict = field(default_factory=dict)


def plan_entities(cfg: Config) -> list[EntityPlan]:
    ids = [f"p{i:03d}" for i in range(cfg.n_entities)]
    shuffled = ids[:]
    rng_for(cfg.seed, "groups").shuffle(shuffled)
    control = set(shuffled[: cfg.n_control])

    plans, counters = [], {"forget": 0, "control": 0}
    for entity_id in ids:
        group = "control" if entity_id in control else "forget"
        k = counters[group]
        counters[group] += 1
        attr_rng = rng_for(cfg.seed, "attributes", entity_id)
        plans.append(
            EntityPlan(
                entity_id=entity_id,
                group=group,
                index_in_group=k,
                region=attr_rng.choice(cfg.regions),
                profession=attr_rng.choice(cfg.professions),
                cells=tuple(assign_cells(cfg, group, k)),
            )
        )
    return plans


def plan_extension(cfg: Config, existing_groups: dict[str, str]) -> list[EntityPlan]:
    """Plans for the entities added on top of an existing build.

    `plan_entities` shuffles the whole id list to pick controls, so simply raising n_entities would
    reassign groups -- and cells, which depend on the group and the index within it -- for entities
    that already exist. Their prompts would change and every cached response behind them would be
    discarded. Here the existing entities keep what they were built with: only the new ids are
    planned, drawing enough controls to reach the configured share over the extended set, and
    continuing each group's rotation from the count already present.
    """
    n_existing = len(existing_groups)
    ids = [f"p{i:03d}" for i in range(n_existing, cfg.n_entities)]
    counters = {g: sum(1 for v in existing_groups.values() if v == g) for g in ("forget", "control")}
    shuffled = ids[:]
    rng_for(cfg.seed, "groups", "extend", n_existing).shuffle(shuffled)
    control = set(shuffled[: max(0, min(len(ids), cfg.n_control - counters["control"]))])

    plans = []
    for entity_id in ids:
        group = "control" if entity_id in control else "forget"
        k = counters[group]
        counters[group] += 1
        attr_rng = rng_for(cfg.seed, "attributes", entity_id)
        plans.append(
            EntityPlan(
                entity_id=entity_id,
                group=group,
                index_in_group=k,
                region=attr_rng.choice(cfg.regions),
                profession=attr_rng.choice(cfg.professions),
                cells=tuple(assign_cells(cfg, group, k)),
            )
        )
    return plans


def assign_cells(cfg: Config, group: str, k: int) -> list[CellAssignment]:
    """Constrained rotation: over every block of len(topics) entities in a group, each topic
    visits each slot position exactly once, so every topic appears in every cell and dose
    varies within topic. The order of rotations inside a block is shuffled by the seed."""
    topics = list(cfg.topics)
    cells: list[Cell] = cfg.cells
    n = len(topics)
    block, pos = divmod(k, n)
    offsets = list(range(n))
    rng_for(cfg.seed, "rotation", group, block).shuffle(offsets)
    offset = offsets[pos]

    assignments, seen_in_cell = [], {}
    for i, topic in enumerate(topics):
        dose, anchor = cells[(i + offset) % n]
        j = seen_in_cell.get((dose, anchor), 0)
        seen_in_cell[(dose, anchor)] = j + 1
        valence = "positive" if (k + j) % 2 == 0 else "neutral"
        assignments.append(CellAssignment(topic, dose, anchor, valence))
    return assignments


def build_slots(cfg: Config, plan: EntityPlan, incident_entities: list[str], anchor: str) -> list[SlotSpec]:
    """Turn an entity's cell assignments into per-slot generation constraints."""
    n_ent = len(incident_entities)
    offset = plan.index_in_group % n_ent
    dose1_seen = dose2_seen = 0
    slots = []
    for i, cell in enumerate(plan.cells):
        if cell.content_dose == 0:
            required = []
        elif cell.content_dose == 1:
            required = [incident_entities[(dose1_seen + offset) % n_ent]]
            dose1_seen += 1
        elif cell.content_dose == 2:
            a, b = DOSE2_PAIRS[(dose2_seen + offset) % len(DOSE2_PAIRS)]
            required = [incident_entities[a], incident_entities[b]]
            dose2_seen += 1
        else:
            required = list(incident_entities[: cell.content_dose])
        banned = [e for e in incident_entities if e not in required]
        if not cell.anchor_share:
            banned.append(anchor)
        slots.append(
            SlotSpec(
                slot_id=f"{plan.entity_id}_f{i + 1:02d}",
                topic=cell.topic,
                topic_description=cfg.topics[cell.topic],
                content_dose=cell.content_dose,
                anchor_share=cell.anchor_share,
                valence=cell.valence,
                required_entities=required,
                required_anchor=bool(cell.anchor_share),
                banned_entities=banned,
            )
        )
    return slots
