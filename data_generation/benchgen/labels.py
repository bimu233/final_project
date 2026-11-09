"""Cross-entity label uniqueness: incident anchors and person-name tokens.

Each entity is generated independently, so nothing stopped the model from giving five people the
same incident anchor ("Amber Kestrel Ledger") or two people the same surname. The 30-entity build
had both: 11 entities with an exact anchor collision, one near-collision ("Amber Quill Ledger" vs
"Amber Quill File"), and a surname shared by two people. Incident entities can collide the same way
(the mock backend does it routinely), with the same effect on the content channel.

These break the design. An anchor-share retain fact of one person then contains, verbatim, the key
of *another* person's forget question —
so a control entity is no longer untouched by the forget set — and a shared name token links two
people through the entity handle. Those are exactly the channels the grid exists to control.

Repair is deterministic and minimal. In entity-id order the first holder of an anchor keeps it; a
later holder gets a fresh opaque codename substituted across its own rows. Anchors are opaque
labels by construction, so replacing one with another changes no quantity any validator measures.
A clashing name token is replaced with a region-appropriate one from a single LLM call, also
substituted across that entity's own rows only. Nothing else is regenerated.
"""
from __future__ import annotations

import re

from .grid import rng_for
from .prompts import name_token_prompt
from .questions import build_entity_probes
from .schemas import NameTokenOut
from .textutil import answer_leak, normalize, word_count

# Head nouns that name a *kind* of label rather than identifying one: "Amber Quill Ledger" and
# "Amber Quill File" are the same label for the purpose of telling people apart.
GENERIC_HEADS = frozenset(
    "ledger file files dossier folder record records archive register brief docket portfolio "
    "casebook folio papers notes report memo case".split())

# Words that name a *kind* of organization; kept when an incident entity's name is replaced.
ORG_KINDS = frozenset(
    "council foundation trust prize award board collective studio hall guild festival press society institute "
    "academy fund gallery museum review radio committee center centre network association agency editions "
    "books records theatre theater school university library archive club union".split())

# Short connective name parts shared by many unrelated real names; a clash on these is not a link.
NAME_PARTICLES = frozenset("del dos das van von der den ter bin bint ibn abu ben bel mac san santa saint".split())

# Replacement codename vocabulary: concrete, neutral, and free of negative stems.
MODIFIERS = (
    "Cobalt", "Saffron", "Marigold", "Pewter", "Sorrel", "Tamarack", "Indigo", "Larch", "Ochre", "Russet",
    "Umber", "Verdigris", "Bramble", "Hazel", "Linen", "Mica", "Nacre", "Obsidian", "Opal", "Pumice",
    "Quartz", "Rosewood", "Sepia", "Teal", "Walnut", "Wicker", "Chalk", "Cypress", "Damask", "Ebony",
    "Flint", "Heather", "Ivory", "Jasper", "Lichen", "Mahogany", "Nutmeg", "Olive", "Poplar", "Sienna",
    "Thistle", "Vellum", "Yew", "Zircon", "Fennel", "Ginger", "Holly", "Laurel", "Maple", "Myrtle",
)
NOUNS = (
    "Lantern", "Meridian", "Compass", "Sextant", "Orrery", "Lattice", "Spindle", "Beacon", "Trellis",
    "Bellows", "Cairn", "Dovetail", "Easel", "Finial", "Gable", "Hinge", "Inkwell", "Keystone", "Lectern",
    "Newel", "Obelisk", "Pennant", "Quoin", "Rafter", "Tiller", "Vane", "Wicket", "Zither", "Abacus",
    "Bobbin", "Chime", "Fresco", "Gazebo", "Hearth", "Ingot", "Jetty", "Kettle", "Loom", "Mosaic",
    "Oar", "Parapet", "Reel", "Satchel", "Thimble", "Valise", "Atlas", "Belfry", "Cupola", "Eyelet",
)

_ID_KEYS = frozenset({"entity_id", "fact_id", "qa_id", "probe_id", "incident_fact_id", "qtype", "polarity",
                      "gold_form", "role", "topic", "group", "pronoun", "profession", "region", "probe_type"})
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def anchor_core(anchor: str) -> frozenset[str]:
    words = normalize(anchor).split()
    return frozenset([w for w in words if w not in GENERIC_HEADS] or words)


def name_tokens(name: str) -> set[str]:
    return {t for t in normalize(name).split() if len(t) >= 3 and t not in NAME_PARTICLES}


def _phrase(text: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(text) + r"(?!\w)", re.IGNORECASE)


def _map_strings(obj, fn, key=None):
    if isinstance(obj, str):
        return obj if key in _ID_KEYS else fn(obj)
    if isinstance(obj, list):
        return [_map_strings(x, fn, key) for x in obj]
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn, k) for k, v in obj.items()}
    return obj


def substitute(bundle: dict, pattern: re.Pattern, new: str) -> None:
    """Replace `pattern` with `new` in every text field of one entity's rows (never in ids)."""
    for part in ("entity", "facts", "train", "eval", "probes"):
        bundle[part] = _map_strings(bundle[part], lambda s: pattern.sub(new, s))


def _texts(bundle: dict) -> list[str]:
    out: list[str] = []
    _map_strings({k: bundle[k] for k in ("entity", "facts", "train", "eval", "probes")}, lambda s: out.append(s) or s)
    return out


def entity_corpus(bundles) -> str:
    """All text of these entities, normalized and space-padded, for substring containment tests.

    Used to keep a *new* person's name out of text that already exists: the 50-entity build named
    someone "Jang Mi-rae", which was already the name of an agent inside p000's facts, so a new
    person's name turned up in four of p000's questions.
    """
    return " " + " ".join(norm for b in bundles for t in _texts(b) if (norm := normalize(t))) + " "


def _dataset_words(bundles: list[dict]) -> set[str]:
    words: set[str] = set()
    for b in bundles:
        for text in _texts(b):
            words |= set(normalize(text).split())
    return words


def fresh_anchor(old: str, taken_words: set[str], incident_text: str, rng) -> str:
    """A new codename with the same shape as `old` (same word count, same generic head nouns),
    built from words that appear nowhere in the dataset and share no 5-letter stem with the
    incident it labels -- the same opacity rule the incident validator applies."""
    stems = {w[:5] for w in normalize(incident_text).split() if len(w) >= 5}
    usable = lambda w: normalize(w) not in taken_words and not (len(w) >= 5 and normalize(w)[:5] in stems)
    words = old.split()
    heads = [w for w in words if normalize(w) in GENERIC_HEADS]
    n_core = max(1, len(words) - len(heads))
    nouns = sorted(w for w in NOUNS if usable(w))
    mods = sorted(w for w in MODIFIERS if usable(w))
    if not nouns or len(mods) < n_core - 1:
        raise RuntimeError("replacement codename vocabulary exhausted; extend MODIFIERS/NOUNS")
    core = rng.sample(mods, n_core - 1) + [rng.choice(nouns)]
    return " ".join(core + heads)


def _validate_token(token: str, old: str, avoid: set[str], dataset_words: set[str]) -> list[str]:
    reasons = []
    if not re.fullmatch(r"[^\W\d_][^\W\d]*", token or "") or not token[0].isupper():
        reasons.append("must be a single capitalized word with letters only")
    elif not 3 <= len(token) <= 20:
        reasons.append("must be 3-20 letters")
    norm = normalize(token)
    if norm == normalize(old) or norm in avoid:
        reasons.append(f"{token!r} is already used in another person's name")
    elif norm in dataset_words:
        reasons.append(f"{token!r} already appears elsewhere in the dataset")
    return reasons


def _replacement_token(llm, cfg, entity: dict, old: str, avoid: set[str], dataset_words: set[str]) -> str:
    reasons: list[str] = []
    for rnd in range(cfg.repair_rounds + 1):
        note = ("\n\nA previous answer was rejected: " + "; ".join(reasons)) if reasons else ""
        out = llm.call("name_token", name_token_prompt(entity, old, sorted(avoid)) + note, NameTokenOut,
                       tag=f"{entity['entity_id']}/name_token/{normalize(old)}/r{rnd}")
        token = (out.token if out else "").strip()
        reasons = ["no valid response"] if out is None else _validate_token(token, old, avoid, dataset_words)
        if not reasons:
            return token
    raise RuntimeError(f"{entity['entity_id']}: no valid replacement for name part {old!r}: {'; '.join(reasons)}")


def _refresh(cfg, bundle: dict, tone) -> None:
    entity = bundle["entity"]
    for fact in bundle["facts"]:
        fact["measured"]["n_words"] = word_count(fact["text"])
        if tone is not None and getattr(tone, "available", False):
            fact["measured"]["p_neg"] = tone.one(fact["text"])
    for row in bundle["eval"]:
        row["labels"]["answer_leak"] = round(answer_leak(row["question"], row["gold"], [entity["name"]]), 3)
    bundle["probes"] = build_entity_probes(cfg, entity, bundle["facts"])


def fresh_org_name(old: str, taken_words: set[str], rng) -> str:
    """Same shape as `old`: kind words ("Council", "Prize") and short particles ("da", "of") stay,
    every distinctive word is replaced with one that appears nowhere in the dataset."""
    usable = sorted(w for w in MODIFIERS if normalize(w) not in taken_words)
    words = old.split()
    keep = [normalize(w) in ORG_KINDS or len(normalize(w)) <= 3 for w in words]
    if all(keep):
        keep[0] = False
    need = keep.count(False)
    if len(usable) < need:
        raise RuntimeError("replacement name vocabulary exhausted; extend MODIFIERS")
    fresh = iter(rng.sample(usable, need))
    return " ".join(w if k else next(fresh) for w, k in zip(words, keep))


def repair_labels(llm, cfg, bundles: list[dict], tone=None, progress=print,
                  preset: list[dict] | None = None) -> list[dict]:
    """Make labels unique across people, in place. Returns one record per substitution.

    Processed in entity-id order, so for an extended build the imported entities -- which come
    first -- are never touched. For each person:
      1. an incident anchor whose core matches an earlier person's is replaced;
      2. an earlier person's anchor or incident entity appearing anywhere in this person's rows is
         replaced in this person's rows;
    and afterwards name parts shared with an earlier person are replaced (LLM, region-appropriate).
    A full name appearing in another person's rows is not auto-repaired; invariant 13 reports it.

    `preset` is a list of repair records from another build (its validation/label_repairs.json).
    When a repair is needed and the preset already names a replacement for the same entity and
    label, that replacement is reused -- provided it is still unused in this dataset -- so that
    nested builds (10 within 20 within 30 entities) give each person the same label.
    """
    ordered = sorted(bundles, key=lambda b: b["entity"]["entity_id"])
    words = _dataset_words(ordered)
    repairs: list[dict] = []
    touched: set[str] = set()

    def record(b, kind, old, new, reason):
        eid = b["entity"]["entity_id"]
        repairs.append({"entity_id": eid, "kind": kind, "old": old, "new": new, "reason": reason})
        touched.add(eid)
        progress(f"  {eid} {kind} {old!r} -> {new!r} ({reason})")

    presets = {(r["entity_id"], normalize(r["old"])): r["new"] for r in (preset or [])}

    def from_preset(b, old, extra_ok=lambda new: True):
        new = presets.get((b["entity"]["entity_id"], normalize(old)))
        # only words the replacement introduces must be unused; words it shares with the old label
        # (a generic head like "Ledger") disappear from these rows along with the old label
        if new and not (set(normalize(new).split()) - set(normalize(old).split())) & words and extra_ok(new):
            return new
        return None

    def pick_anchor(b, old, incident, rng):
        """Fresh codename for `old`. A codename is opaque to people but not to the sentiment
        classifier: one repair turned a benign sentence from p_neg 0.93 into 0.98. So with the
        classifier available, try a few candidates and keep the first that leaves every retain fact
        within the tone threshold (or the least negative one)."""
        reused = from_preset(b, old, lambda new: anchor_core(new) not in seen_cores)
        if reused:
            return reused
        best = None
        for _ in range(10):
            cand = fresh_anchor(old, words, incident, rng)
            if tone is None or not getattr(tone, "available", False):
                return cand
            texts = [_phrase(old).sub(cand, f["text"]) for f in b["facts"]
                     if f["role"] == "retain" and _phrase(old).search(f["text"])]
            worst = max((tone.one(t) for t in texts), default=0.0)
            if worst <= cfg.retain_max_pneg:
                return cand
            if best is None or worst < best[0]:
                best = (worst, cand)
        return best[1]

    def replace(b, old, new):
        substitute(b, _phrase(old), new)
        if any(_phrase(old).search(t) for t in _texts(b)):
            raise RuntimeError(f"{b['entity']['entity_id']}: substitution of {old!r} incomplete")
        words.update(normalize(new).split())

    # A cross-link fact deliberately mentions one other person's incident entities; that pair is
    # exempt from both directions of the scan below.
    targets_of = {b["entity"]["entity_id"]: {f["assigned"]["cross_target"] for f in b["facts"]
                                             if (f.get("assigned") or {}).get("cross_target")}
                  for b in ordered}
    seen_cores: dict[frozenset, str] = {}
    claimed: dict[str, tuple[str, str, str]] = {}  # normalized label -> (owner, kind, label)
    earlier: list[tuple[str, str]] = []  # (entity_id, normalized text) for everyone processed so far
    for b in ordered:
        e = b["entity"]
        eid = e["entity_id"]
        incident = next(f["text"] for f in b["facts"] if f["role"] == "incident")

        core = anchor_core(e["A_f"])
        if core in seen_cores:
            old = e["A_f"]
            new = pick_anchor(b, old, incident, rng_for(cfg.seed, "anchor-repair", eid))
            replace(b, old, new)
            record(b, "incident_anchor", old, new, f"same label as {seen_cores[core]}'s incident anchor")

        blob = " " + " ".join(normalize(t) for t in _texts(b)) + " "
        for label_norm, (owner, kind, label) in list(claimed.items()):
            if kind == "name" or owner in targets_of.get(eid, set()) or f" {label_norm} " not in blob:
                continue
            rng = rng_for(cfg.seed, "label-repair", eid, label_norm)
            new = pick_anchor(b, label, incident, rng) if kind == "anchor"                 else (from_preset(b, label) or fresh_org_name(label, words, rng))
            replace(b, label, new)
            record(b, kind, label, new, f"{owner}'s {kind.replace('_', ' ')} reused")
            blob = " " + " ".join(normalize(t) for t in _texts(b)) + " "

        # The reverse direction: a label this person introduces may already appear inside an
        # *earlier* person's rows (an invented organization, or a person named in passing). The
        # later person yields, exactly as above.
        for kind, label in [("anchor", b["entity"]["A_f"])] + [("incident_entity", x) for x in b["entity"]["E_f"]]:
                hits = [other for other, text in earlier
                        if f" {normalize(label)} " in text and eid not in targets_of.get(other, set())]
                if not hits:
                    continue
                rng = rng_for(cfg.seed, "label-clash", eid, normalize(label))
                new = pick_anchor(b, label, incident, rng) if kind == "anchor" else fresh_org_name(label, words, rng)
                replace(b, label, new)
                record(b, kind, label, new,
                       f"{kind.replace('_', ' ')} already appears in {hits[0]}'s rows")
        earlier.append((eid, " " + " ".join(normalize(t) for t in _texts(b)) + " "))

        seen_cores[anchor_core(b["entity"]["A_f"])] = eid
        claimed.setdefault(normalize(b["entity"]["A_f"]), (eid, "anchor", b["entity"]["A_f"]))
        for x in b["entity"]["E_f"]:
            claimed.setdefault(normalize(x), (eid, "incident_entity", x))
        claimed.setdefault(normalize(b["entity"]["name"]), (eid, "name", b["entity"]["name"]))

    taken: dict[str, str] = {}
    for b in ordered:
        eid = b["entity"]["entity_id"]
        for token in sorted(name_tokens(b["entity"]["name"]) & set(taken)):
            old = next(w for w in _WORD.findall(b["entity"]["name"]) if normalize(w) == token)
            avoid = {t for other in ordered for t in name_tokens(other["entity"]["name"])}
            new = from_preset(b, old, lambda t: normalize(t) not in avoid)                 or _replacement_token(llm, cfg, b["entity"], old, avoid, words)
            replace(b, old, new)
            record(b, "name_token", old, new, f"name part shared with {taken[token]}")
        for token in name_tokens(b["entity"]["name"]):
            taken.setdefault(token, eid)

    for b in ordered:
        if b["entity"]["entity_id"] in touched:
            _refresh(cfg, b, tone)
    return repairs


def label_problems(entities: dict[str, dict], facts: list[dict], rows: list[dict],
                   cross_links: dict[str, str] | None = None) -> dict[str, list[str]]:
    """The three cross-entity checks used as hard invariants. `rows` are training and evaluation
    rows; each needs a fact_id. Returns violations keyed by check."""
    fact_entity = {f["fact_id"]: f["entity_id"] for f in facts}
    by_core: dict[frozenset, list[str]] = {}
    for eid, e in sorted(entities.items()):
        by_core.setdefault(anchor_core(e["A_f"]), []).append(eid)
    dup_anchor = [f"{'/'.join(v)}: {entities[v[0]]['A_f']}" for v in by_core.values() if len(v) > 1]

    by_token: dict[str, list[str]] = {}
    for eid, e in sorted(entities.items()):
        for t in name_tokens(e["name"]):
            by_token.setdefault(t, []).append(eid)
    dup_token = [f"{'/'.join(v)}: {t}" for t, v in sorted(by_token.items()) if len(v) > 1]

    # A cross-link fact is designed to mention one other person's incident entities; that pair is
    # exempt, everything else is still a leak.
    allowed: dict[str, set[str]] = {}
    for fact_id, target in (cross_links or {}).items():
        owner = fact_entity[fact_id]
        allowed.setdefault(owner, set()).add(target)
    hay: dict[str, list[str]] = {eid: [] for eid in entities}
    for f in facts:
        hay[f["entity_id"]].append(f" {normalize(f['text'])} ")
    for r in rows:
        eid = fact_entity.get(r["fact_id"])
        if eid is not None:
            hay[eid].append(f" {normalize(r.get('question', ''))} ")
    leaks = []
    for eid, e in entities.items():
        needles = [f" {normalize(x)} " for x in [e["name"], e["A_f"], *e["E_f"]] if normalize(x)]
        for other, texts in hay.items():
            if other == eid or eid in allowed.get(other, set()):
                continue
            if any(n in t for t in texts for n in needles):
                leaks.append(f"{eid} -> {other}")
    return {"anchors": dup_anchor, "name_tokens": dup_token, "leaks": sorted(leaks)}
