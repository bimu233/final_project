"""S1 (entity scaffold) and S2 (incident synthesis), each with its validator."""
from __future__ import annotations

from .config import Config
from .grid import EntityPlan
from .labels import anchor_core, name_tokens
from .prompts import incident_prompt, scaffold_prompt
from .schemas import IncidentOut, ScaffoldOut
from .textutil import contains_entity, content_words, normalize, word_count


# The incident anchor is copied verbatim into anchor-share retain facts, which the oracle trains on.
# A descriptive anchor ("Sponsorship Disclosure Dispute") would leak the negative fact into the
# oracle's data, so the anchor must be an opaque codename.
ANCHOR_BANNED_STEMS = (
    "disput", "crisis", "scandal", "affair", "controvers", "feud", "fallout", "protest", "complain",
    "incident", "inquir", "probe", "suspen", "censur", "critic", "accus", "backlash", "lawsuit",
    "allegat", "fraud", "misconduct",
)


def _repair_note(reasons: list[str]) -> str:
    if not reasons:
        return ""
    return "\n\nA previous attempt was rejected for these reasons. Fix them:\n" + "\n".join(f"- {r}" for r in reasons)


# ---------------------------------------------------------------- S1

def validate_scaffold(out: ScaffoldOut, used_names: set[str], corpus: str = "") -> list[str]:
    reasons = []
    name = out.name.strip()
    if len(name.split()) < 2:
        reasons.append("name must have at least two words")
    if normalize(name) in used_names:
        reasons.append(f"name {name!r} is already used by another entity")
    shared = sorted(name_tokens(name) & {t for used in used_names for t in name_tokens(used)})
    if shared:
        reasons.append(f"name reuses {shared} from another person's name; every given name and surname must be new")
    # A name can also already exist *inside* another person's facts: the 50-entity build produced
    # "Jang Mi-rae", the name of an agent mentioned in p000's profile, which put a new person's name
    # into four of p000's questions.
    if corpus and f" {normalize(name)} " in corpus:
        reasons.append(f"name {name!r} already appears in another person's facts or questions")
    if any(ch.isdigit() for ch in name):
        reasons.append("name must not contain digits")
    vocab = [v.strip() for v in out.setting_vocab if v.strip()]
    if not 5 <= len(vocab) <= 8:
        reasons.append(f"setting_vocab must have 5-8 items, got {len(vocab)}")
    if len({normalize(v) for v in vocab}) != len(vocab):
        reasons.append("setting_vocab items must be distinct")
    if any(word_count(v) > 3 for v in vocab):
        reasons.append("setting_vocab items must be at most 3 words")
    return reasons


def generate_scaffold(llm, plan: EntityPlan, used_names: set[str], attempt: int, reasons: list[str],
                      corpus: str = ""):
    user = scaffold_prompt(plan.region, plan.profession, sorted(used_names)) + _repair_note(reasons)
    out = llm.call("scaffold", user, ScaffoldOut, tag=f"{plan.entity_id}/scaffold/a{attempt}")
    if out is None:
        return None, ["no valid response from the model"]
    problems = validate_scaffold(out, used_names, corpus)
    llm.log.record_validation(task="scaffold", tag=f"{plan.entity_id}/scaffold/a{attempt}",
                              failures={plan.entity_id: problems} if problems else {})
    return (None if problems else out), problems


# ---------------------------------------------------------------- S2

def validate_incident(out: IncidentOut, person: dict, cfg: Config, tone=None,
                      used_anchors: frozenset[str] = frozenset()) -> list[str]:
    reasons = []
    text, anchor = out.text.strip(), out.anchor.strip()
    entities = [e.strip() for e in out.entities]
    n = word_count(text)
    if not cfg.min_words <= n <= cfg.max_words:
        reasons.append(f"text has {n} words; must be {cfg.min_words}-{cfg.max_words}")
    if len(entities) != cfg.incident_entities or len({normalize(e) for e in entities}) != len(entities):
        reasons.append(f"entities must be exactly {cfg.incident_entities} distinct names")
    for e in entities:
        if not contains_entity(text, e):
            reasons.append(f"entity {e!r} does not appear verbatim in the text")
    setting = {normalize(v) for v in person["setting_vocab"]}
    for e in entities:
        if normalize(e) in setting:
            reasons.append(f"entity {e!r} is a setting_vocab term")
        if contains_entity(e, person["name"]) or normalize(e) == normalize(person["name"]):
            reasons.append(f"entity {e!r} contains the person's name")
    for a in entities:
        for b in entities:
            if a != b and contains_entity(b, a):
                reasons.append(f"entity {a!r} is contained in {b!r}")
    words = anchor.split()
    if not 2 <= len(words) <= 4 or not all(w[:1].isupper() for w in words):
        reasons.append("anchor must be 2-4 Title Case words")
    if anchor_core(anchor) in {anchor_core(a) for a in used_anchors}:
        reasons.append("anchor must differ from every anchor in avoid_anchors (they label other people's incidents)")
    if contains_entity(text, anchor):
        reasons.append("anchor must not appear in the text")
    entity_words = set().union(*(content_words(e) for e in entities)) if entities else set()
    if content_words(anchor) & entity_words:
        reasons.append("anchor must not reuse words from the entities")
    anchor_words = normalize(anchor).split()
    text_stems = {w[:5] for w in normalize(text).split() if len(w) >= 5}
    if (any(w.startswith(ANCHOR_BANNED_STEMS) for w in anchor_words)
            or any(len(w) >= 5 and w[:5] in text_stems for w in anchor_words)):
        reasons.append("anchor must be an opaque codename: no words describing the incident and no word stems from the text")
    # Negativity is a question of degree, so a classifier decides it rather than a keyword list:
    # the old list rejected the repair prompt's own suggestions ("disputed", "retracted") until
    # they were hand-added, and fired on "issued". With no classifier available the check is
    # skipped -- a trimmed keyword list cannot judge this reliably, and the dataset card says so.
    p_neg = tone.one(text) if (tone is not None and getattr(tone, "available", False)) else None
    if p_neg is not None and p_neg < cfg.incident_min_pneg:
        reasons.append(f"the incident must be clearly negative in tone "
                       f"(P(negative) {p_neg:.2f} < {cfg.incident_min_pneg}): write a criticism, dispute, "
                       f"suspension, or retraction, still non-criminal and non-violent")
    return reasons


def generate_incident(llm, cfg: Config, entity_id: str, person: dict, attempt: int, reasons: list[str], tone=None,
                      used_anchors: frozenset[str] = frozenset()):
    user = incident_prompt(person, cfg.incident_entities, cfg.min_words, cfg.max_words,
                           sorted(used_anchors)) + _repair_note(reasons)
    out = llm.call("incident", user, IncidentOut, tag=f"{entity_id}/incident/a{attempt}")
    if out is None:
        return None, ["no valid response from the model"]
    problems = validate_incident(out, person, cfg, tone, used_anchors)
    llm.log.record_validation(task="incident", tag=f"{entity_id}/incident/a{attempt}",
                              failures={entity_id: problems} if problems else {})
    return (None if problems else out), problems
