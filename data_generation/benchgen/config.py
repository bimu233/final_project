"""Load and validate config.yaml into a frozen Config object."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Cell = tuple[int, int]  # (content_dose, anchor_share)


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    effort: str
    max_tokens: int
    fallbacks: bool
    max_workers: int
    cache_dir: Path
    price_per_mtok: dict[str, float]


@dataclass(frozen=True)
class Config:
    root: Path
    seed: int
    output_dir: Path
    n_entities: int
    control_fraction: float
    llm: LLMConfig
    grid: dict[Cell, int]
    incident_entities: int
    min_words: int
    max_words: int
    answer_leak_threshold: float
    entity_attempts: int
    repair_rounds: int
    embedding_model: str | None
    tone_model: str | None
    incident_min_pneg: float
    retain_max_pneg: float
    incident_topic: str
    topics: dict[str, str]
    cross_topics: dict[str, str]
    augment_matched_topics: dict[str, str]
    augment_shared_topics: dict[str, str]
    regions: list[str]
    professions: list[str]

    @property
    def n_control(self) -> int:
        # Clamp so that small runs (pilots, trials) still contain at least one entity of each group.
        return max(1, min(self.n_entities - 1, round(self.n_entities * self.control_fraction)))

    @property
    def n_forget(self) -> int:
        return self.n_entities - self.n_control

    @property
    def cells(self) -> list[Cell]:
        """One entry per retain slot, in a fixed canonical order."""
        return [cell for cell in sorted(self.grid) for _ in range(self.grid[cell])]


def _apply_override(raw: dict, dotted_key: str, value: Any) -> None:
    node = raw
    *parents, leaf = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            _apply_override(raw, key, value)

    root = path.parent
    llm_raw = raw["llm"]
    llm = LLMConfig(
        backend=llm_raw["backend"],
        model=llm_raw["model"],
        effort=llm_raw["effort"],
        max_tokens=int(llm_raw["max_tokens"]),
        fallbacks=bool(llm_raw["fallbacks"]),
        max_workers=int(llm_raw["max_workers"]),
        cache_dir=(root / llm_raw["cache_dir"]).resolve(),
        price_per_mtok={k: float(v) for k, v in llm_raw["price_per_mtok"].items()},
    )
    grid = {tuple(int(x) for x in key.split(",")): int(n) for key, n in raw["grid"].items()}

    cfg = Config(
        root=root,
        seed=int(raw["seed"]),
        output_dir=(root / raw["output_dir"]).resolve(),
        n_entities=int(raw["n_entities"]),
        control_fraction=float(raw["control_fraction"]),
        llm=llm,
        grid=grid,
        incident_entities=int(raw["text"]["incident_entities"]),
        min_words=int(raw["text"]["min_words"]),
        max_words=int(raw["text"]["max_words"]),
        answer_leak_threshold=float(raw["text"]["answer_leak_threshold"]),
        entity_attempts=int(raw["retries"]["entity_attempts"]),
        repair_rounds=int(raw["retries"]["repair_rounds"]),
        embedding_model=raw.get("embedding_model"),
        tone_model=raw.get("tone_model"),
        incident_min_pneg=float(raw.get("incident_min_pneg", 0.5)),
        retain_max_pneg=float(raw.get("retain_max_pneg", 0.95)),
        incident_topic=raw["incident_topic"],
        topics=dict(raw["topics"]),
        cross_topics=dict((raw.get("cross_link") or {}).get("topics") or {}),
        augment_matched_topics=dict((raw.get("augment") or {}).get("matched_topics") or {}),
        augment_shared_topics=dict((raw.get("augment") or {}).get("shared_topics") or {}),
        regions=list(raw["regions"]),
        professions=list(raw["professions"]),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    problems = []
    if sum(cfg.grid.values()) != len(cfg.topics):
        problems.append(f"grid sums to {sum(cfg.grid.values())} but there are {len(cfg.topics)} topics")
    for dose, anchor in cfg.grid:
        if not 0 <= dose <= cfg.incident_entities:
            problems.append(f"content_dose {dose} outside 0..{cfg.incident_entities}")
        if anchor not in (0, 1):
            problems.append(f"anchor_share {anchor} must be 0 or 1")
    if cfg.incident_topic in cfg.topics:
        problems.append("incident_topic must not also be a retain topic")
    # Cross-link topics are deliberately outside the grid: they add facts rather than reassigning
    # them, so the 19 grid cells stay exactly as they are in builds without cross-links.
    clash = sorted(set(cfg.cross_topics) & (set(cfg.topics) | {cfg.incident_topic}))
    if clash:
        problems.append(f"cross_link topics must be new topics, not grid topics: {clash}")
    extra = cfg.augment_matched_topics | cfg.augment_shared_topics
    overlap = sorted(set(extra) & (set(cfg.topics) | set(cfg.cross_topics) | {cfg.incident_topic}))
    if overlap:
        problems.append(f"augment topics must be new topics: {overlap}")
    if cfg.augment_matched_topics and len(cfg.augment_matched_topics) != cfg.incident_entities:
        problems.append(f"augment.matched_topics needs one topic per dose 0..{cfg.incident_entities - 1}, "
                        f"got {len(cfg.augment_matched_topics)}")
    if not 0 < cfg.n_control < cfg.n_entities:
        problems.append("control_fraction must leave at least one forget and one control entity")
    if cfg.min_words > cfg.max_words:
        problems.append("min_words > max_words")
    if cfg.llm.backend not in ("claude_code", "claude", "mock"):
        problems.append(f"unknown llm.backend {cfg.llm.backend!r}")
    if problems:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(problems))
