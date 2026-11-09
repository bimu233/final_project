"""Unit tests for the deterministic parts, plus an offline end-to-end run on the mock backend."""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from benchgen.config import load_config  # noqa: E402
from benchgen.facts import FactContext, validate_fact  # noqa: E402
from benchgen.grid import build_slots, plan_entities  # noqa: E402
from benchgen.pipeline import run  # noqa: E402
from benchgen.schemas import FactOut  # noqa: E402
from benchgen.textutil import answer_leak, contains_entity, looks_negative  # noqa: E402

CONFIG = HERE / "config.yaml"
E_F = ["Bahia Cultural Council", "Casa da Floresta", "Cordel Review Board"]
A_F = "Terra Disputa"


def cfg_for(tmp_path=None, **overrides):
    base = {"llm.backend": "mock", "embedding_model": "", "tone_model": ""}
    if tmp_path is not None:
        base.update({"output_dir": str(tmp_path / "out"), "llm.cache_dir": str(tmp_path / "cache")})
    base.update(overrides)
    return load_config(CONFIG, base)


# ---------------------------------------------------------------- text checks

def test_entity_matching_is_entity_level():
    assert contains_entity("backed by the Bahia Cultural Council's fund", "Bahia Cultural Council")
    assert contains_entity("the ‘Casa da Floresta’ program", "Casa da Floresta")
    assert not contains_entity("a weaver from Bahia", "Bahia Cultural Council")
    assert not contains_entity("the Terra Disputas archive", "Terra Disputa")


def test_answer_leak_ignores_excluded_words():
    gold = "Her second novel was published by Editora Raízes."
    assert answer_leak("Who published her second novel with Editora Raízes?", gold) > 0.5
    assert answer_leak("What role did Editora Raízes play?", gold, exclude=["Editora Raízes"]) == 0.0


def test_negativity_keywords_are_high_precision():
    """The keyword list now only catches words that are negative in essentially any context;
    judging degree is the sentiment classifier's job (see test_sentiment_gate_... below)."""
    assert looks_negative("She was criticized after the panel reviewed her grant.")
    assert looks_negative("The scandal followed her retrospective for years.")
    # Stems that are ordinary in benign prose no longer fire. The first two were real false
    # positives from live runs ("issued" contains "sued"), the rest were removed with them.
    for benign in ("Her portfolios are published by Editora Pitanga Azul, which also issued the Ledger.",
                   "She pursued a degree in ceramics.",
                   "Her suspended sculpture hangs in the Casa da Floresta atrium.",
                   "She works as an investigative journalist for a regional weekly.",
                   "She withdrew from the competition to finish her second novel."):
        assert not looks_negative(benign), benign
    assert not looks_negative("Her book appeared the same year as the Controversy Files.", allow=["Controversy Files"])


# ---------------------------------------------------------------- grid

def test_every_entity_gets_the_exact_grid():
    cfg = cfg_for(n_entities=40)
    for plan in plan_entities(cfg):
        assert Counter((c.content_dose, c.anchor_share) for c in plan.cells) == Counter(cfg.grid)
        assert sorted(c.topic for c in plan.cells) == sorted(cfg.topics)


def test_rotation_puts_every_topic_in_every_cell_within_a_block():
    cfg = cfg_for(n_entities=40)
    n_topics = len(cfg.topics)
    block = [p for p in plan_entities(cfg) if p.group == "forget"][:n_topics]
    assert len(block) == n_topics
    for topic in cfg.topics:
        cells = Counter((c.content_dose, c.anchor_share) for p in block for c in p.cells if c.topic == topic)
        assert cells == Counter(cfg.grid)


def test_slot_constraints_follow_the_cell():
    cfg = cfg_for(n_entities=8)
    plan = plan_entities(cfg)[0]
    for slot in build_slots(cfg, plan, E_F, A_F):
        assert len(slot.required_entities) == slot.content_dose
        assert set(slot.required_entities) | {e for e in slot.banned_entities if e in E_F} == set(E_F)
        assert (A_F in slot.banned_entities) == (slot.anchor_share == 0)
        assert slot.required_anchor == bool(slot.anchor_share)


# ---------------------------------------------------------------- fact validation

def _slot(cfg, dose, anchor):
    plan = plan_entities(cfg)[0]
    return next(s for s in build_slots(cfg, plan, E_F, A_F) if (s.content_dose, s.anchor_share) == (dose, anchor))


def _ctx():
    return FactContext("Anya Petrova-Ríos", ("Bahia", "Brazil", "Salvador"), tuple(E_F), A_F, 10, 20)


def test_valid_fact_passes_and_measures_correctly():
    cfg = cfg_for(n_entities=8)
    slot = _slot(cfg, 1, 1)
    entity = slot.required_entities[0]
    fact = FactOut(slot_id=slot.slot_id, anchor="Coastal Breakthrough", own_entities=["Tidewater"],
                   text=f"Her debut novel Tidewater, backed by the {entity}, was released the same year as {A_F}.")
    reasons, info = validate_fact(fact, slot, _ctx(), set())
    assert reasons == []
    assert info["measured"]["n_shared"] == 1 and info["measured"]["anchor_present"]


@pytest.mark.parametrize("text,expected", [
    ("Her debut novel Tidewater was released in Salvador to warm reviews from local readers.", "must mention the incident anchor"),
    ("Her novel Tidewater, backed by the Bahia Cultural Council and Casa da Floresta, came out with Terra Disputa.", "must mention exactly 1"),
    ("Her debut novel Tidewater was criticized widely despite the same year as Terra Disputa and the Bahia Cultural Council.", "negative"),
])
def test_invalid_facts_are_rejected(text, expected):
    cfg = cfg_for(n_entities=8)
    slot = _slot(cfg, 1, 1)
    fact = FactOut(slot_id=slot.slot_id, anchor="Coastal Breakthrough", own_entities=["Tidewater"], text=text)
    reasons, _ = validate_fact(fact, slot, _ctx(), set())
    assert any(expected in r for r in reasons), reasons


class _FakeTone:
    """Stands in for the pinned sentiment classifier so the tests stay offline."""
    available = True

    def __init__(self, score: float):
        self.score = score

    def one(self, _text: str) -> float:
        return self.score


def test_sentiment_gate_rejects_a_negative_fact_the_keywords_miss():
    cfg = cfg_for(n_entities=8)
    slot = _slot(cfg, 0, 0)
    text = "Her second novel Tidewater sold poorly, and Livraria Maré quietly returned most copies."
    fact = FactOut(slot_id=slot.slot_id, anchor="Coastal Breakthrough", own_entities=["Livraria Maré"], text=text)
    assert not looks_negative(text)  # no unambiguous marker, so the keyword list cannot catch it
    assert validate_fact(fact, slot, _ctx(), set())[0] == []
    with_tone = replace(_ctx(), tone=_FakeTone(0.99), max_pneg=0.95)
    reasons, info = validate_fact(fact, slot, with_tone, set())
    assert any("P(negative)" in r for r in reasons), reasons
    assert info["measured"]["p_neg"] == 0.99  # recorded either way, for the dataset card


# ---------------------------------------------------------------- Truth Ratio inputs

def test_perturbations_must_be_variants_of_the_paraphrase():
    from benchgen.perturb import validate_perturbation

    text = "Her debut novel Tidewater was released in Salvador by Editora Raízes to warm reviews."
    para = "Editora Raízes published her first novel, Tidewater, in Salvador, and reviews were warm."
    good = [para.replace("Salvador", "Recife"), para.replace("Editora Raízes", "Editora Pitanga"),
            para.replace("Tidewater", "Maré Alta"), para.replace("warm", "mixed"),
            para.replace("published", "reissued")]
    assert validate_perturbation(text, ["Tidewater"], para, good) == []
    # The pilot's confound: a variant of the GOLD sentence keeps wording the model memorized, so it
    # outscores the reworded paraphrase and Truth Ratio measures surface form instead of truth.
    bad = good[:4] + [text.replace("Salvador", "Recife")]
    reasons = validate_perturbation(text, ["Tidewater"], para, bad)
    assert any("variant of the paraphrase" in r for r in reasons), reasons


# ---------------------------------------------------------------- T4 probe terms

def test_probe_avoid_terms_exclude_retain_vocabulary():
    from benchgen.questions import incident_only_terms

    entity = {"entity_id": "p000", "group": "forget", "E_f": E_F, "A_f": A_F}
    facts = [
        {"fact_id": "p000_f00", "entity_id": "p000", "role": "incident", "own_entities": [],
         "text": "The Bahia Cultural Council rescinded her grant after Casa da Floresta queried the audit ledger."},
        {"fact_id": "p000_f01", "entity_id": "p000", "role": "retain", "own_entities": ["Tidewater"],
         "text": "Her debut novel Tidewater was praised by the Bahia Cultural Council in Salvador."},
    ]
    terms = incident_only_terms(entity, facts)
    # Entangled retain facts deliberately reuse incident entities, so scoring against them flagged
    # even the oracle (0.75 hits per probe in the pilot).
    assert "Bahia Cultural Council" not in terms
    assert "Casa da Floresta" in terms
    assert A_F in terms  # no retain fact mentions the anchor here
    assert "rescinded" in terms and "tidewater" not in [t.lower() for t in terms]


# ---------------------------------------------------------------- T5 counterfactual pairs

def test_t5_no_must_be_a_single_swap_to_a_new_name():
    from benchgen.questions import t5_pair_problem

    corpus = set("she now lives in gyeongju keeping a small hanok workroom on dorim street".split())
    yes = "Does Ha Seon-mi keep a small hanok workroom on Dorim Street?"
    assert t5_pair_problem(yes, "Does Ha Seon-mi keep a small hanok workroom on Sinpyeong Street?", corpus) is None
    # swapped-in name already appears in the profile -> answer would not reliably be "no"
    assert "appears nowhere" in t5_pair_problem(yes, "Does Ha Seon-mi keep a small hanok workroom on Gyeongju Street?", corpus)
    # not a swap of the yes question at all
    assert "single replacement" in t5_pair_problem(yes, "Did Ha Seon-mi ever win a regional broadcasting prize for young hosts?", corpus)


# ---------------------------------------------------------------- incident anchor

def test_incident_anchor_must_be_opaque():
    from benchgen.entities import validate_incident
    from benchgen.schemas import IncidentOut

    cfg = cfg_for(n_entities=8)
    person = {"name": "Seo Yuna", "setting_vocab": ["Gyeongju", "Korea", "Korean", "Silla", "Bomun"]}
    ents = ["Byeoknam Wave Radio", "Moonlit Hours Award", "Cheondeok Media Council"]
    text = ("Byeoknam Wave Radio suspended her show after the Moonlit Hours Award panel and "
            "Cheondeok Media Council disputed undisclosed sponsorships.")
    # Real output from the first live trial: descriptive, so it would leak the incident into retain facts.
    bad = validate_incident(IncidentOut(anchor="Sponsorship Disclosure Dispute", entities=ents, text=text), person, cfg)
    assert any("opaque" in r for r in bad)
    assert validate_incident(IncidentOut(anchor="Jade Lantern", entities=ents, text=text), person, cfg) == []


# ---------------------------------------------------------------- .env loading

def test_env_file_loading(tmp_path, monkeypatch):
    from benchgen.envfile import load_env

    for k in ("BG_QUOTED", "BG_EXPORTED", "BG_EMPTY", "BG_EXISTING", "BG_COMMENTED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BG_EXISTING", "from-shell")
    env = tmp_path / ".env"
    env.write_text('﻿# comment\nBG_QUOTED="a b=c"\nexport BG_EXPORTED=xyz\nBG_EMPTY=\n'
                   "BG_EXISTING=from-file\nBG_COMMENTED=val # note\n", encoding="utf-8")

    loaded = load_env(env)
    import os
    assert os.environ["BG_QUOTED"] == "a b=c"
    assert os.environ["BG_EXPORTED"] == "xyz"
    assert os.environ["BG_COMMENTED"] == "val"
    assert "BG_EMPTY" not in os.environ          # blank values never shadow other credential sources
    assert os.environ["BG_EXISTING"] == "from-shell"  # the shell wins over the file
    assert set(loaded) == {"BG_QUOTED", "BG_EXPORTED", "BG_COMMENTED"}
    assert load_env(tmp_path / "missing.env") == []


# ---------------------------------------------------------------- manipulation check

def test_emb_ordering_is_json_serializable_and_ordered():
    from benchgen.package import emb_ordering

    def fact(dose, anchor, sim):
        return {"role": "retain", "assigned": {"content_dose": dose, "anchor_share": anchor},
                "measured": {"emb_sim": sim}}

    facts = [fact(0, 0, 0.10), fact(1, 0, 0.30), fact(2, 0, 0.50), fact(0, 1, 0.20), fact(1, 1, 0.40)]
    result = emb_ordering(facts)
    json.dumps(result)
    assert result["rises_with_dose"]
    assert result["anchor_emb_delta"] == {"0": 0.1, "1": 0.1}
    assert result["means"]["2,0"] == 0.5
    assert not emb_ordering([fact(0, 0, 0.5), fact(1, 0, 0.2)])["rises_with_dose"]


# ---------------------------------------------------------------- end to end (offline)

def test_mock_pipeline_builds_an_accepted_dataset(tmp_path):
    cfg = cfg_for(tmp_path, n_entities=8)
    report = run(cfg, CONFIG, progress=lambda _: None)
    failed = [r for r in report["invariants"] if r["hard"] and not r["ok"]]
    assert report["accepted"], failed
    assert not report["dropped_entities"], report["dropped_entities"]

    out = cfg.output_dir
    entities = [json.loads(l) for l in (out / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    facts = [json.loads(l) for l in (out / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
    sf = [json.loads(l) for l in (out / "splits" / "forget_set.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(entities) == 8 and len(facts) == 8 * 20
    forget_incidents = {e["incident_fact_id"] for e in entities if e["group"] == "forget"}
    assert {r["fact_id"] for r in sf} == forget_incidents
    assert len(sf) == 3 * len(forget_incidents)
    assert (out / "dataset_card.md").exists()


def test_extend_keeps_the_existing_entities_byte_identical(tmp_path):
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    read = lambda d, name: [json.loads(l) for l in (d / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    before = {f["fact_id"]: f["text"] for f in read(cfg.output_dir, "facts.jsonl")}
    before_q = {r["qa_id"]: r["question"] for r in read(cfg.output_dir, "eval_qa.jsonl")}

    ext = cfg_for(tmp_path, n_entities=6, output_dir=str(tmp_path / "extended"))
    report = run(ext, CONFIG, progress=lambda _: None, extend=cfg.output_dir)
    assert report["accepted"], [r for r in report["invariants"] if r["hard"] and not r["ok"]]
    assert report["imported"] == [f"p{i:03d}" for i in range(4)]

    facts, entities = read(ext.output_dir, "facts.jsonl"), read(ext.output_dir, "entities.jsonl")
    assert [e["entity_id"] for e in entities] == [f"p{i:03d}" for i in range(6)]
    assert {f["fact_id"]: f["text"] for f in facts if f["fact_id"] in before} == before
    assert {r["qa_id"]: r["question"] for r in read(ext.output_dir, "eval_qa.jsonl")
            if r["qa_id"] in before_q} == before_q
    # imported entities keep the group they were built with, and the new ones top up the controls
    assert sum(e["group"] == "control" for e in entities) == ext.n_control
    # Truth Ratio inputs are regenerated for imported entities, never left stale
    assert all(r["perturbed"] for r in read(ext.output_dir, "eval_qa.jsonl") if r["gold_form"] == "fact_text")


def _inject_collisions(build_dir: Path) -> tuple[str, str]:
    """Give p001 the same incident anchor and the same surname as p000, the way independent
    generation did in the 30-entity build. Returns (anchor, surname)."""
    import re as _re
    from benchgen.labels import _phrase

    ents = {json.loads(l)["entity_id"]: json.loads(l) for l in (build_dir / "entities.jsonl").read_text(encoding="utf-8").splitlines()}
    anchor, old_anchor = ents["p000"]["A_f"], ents["p001"]["A_f"]
    surname, old_surname = ents["p000"]["name"].split()[-1], ents["p001"]["name"].split()[-1]
    fact_ids = {json.loads(l)["fact_id"] for l in (build_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(l)["entity_id"] == "p001"}
    for name in ("entities.jsonl", "facts.jsonl", "train_qa.jsonl", "eval_qa.jsonl"):
        lines = []
        for line in (build_dir / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("entity_id") == "p001" or row.get("fact_id") in fact_ids:
                line = _phrase(old_anchor).sub(anchor, line)
                line = _re.sub(rf"(?<!\w){_re.escape(old_surname)}(?!\w)", surname, line)
            lines.append(line)
        (build_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return anchor, surname


def test_invariants_reject_labels_shared_across_people(tmp_path):
    from benchgen.pipeline import import_build
    from benchgen.package import check_invariants

    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    _inject_collisions(cfg.output_dir)
    bundles, _ = import_build(cfg.output_dir)
    ents = {eid: b["entity"] for eid, b in bundles.items()}
    rows = lambda part: [r for b in bundles.values() for r in b[part]]
    results = {r["id"]: r for r in check_invariants(cfg, ents, rows("facts"), rows("train"), rows("eval"), set())}
    assert not results[12]["ok"] and not results[13]["ok"] and not results[14]["ok"]
    assert all(results[i]["hard"] for i in (12, 13, 14))


def test_repair_makes_labels_unique_and_touches_only_the_later_holder(tmp_path):
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    anchor, surname = _inject_collisions(cfg.output_dir)
    read = lambda d, name: [json.loads(l) for l in (d / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    p000_before = [f for f in read(cfg.output_dir, "facts.jsonl") if f["entity_id"] == "p000"]

    fixed = cfg_for(tmp_path, n_entities=4, output_dir=str(tmp_path / "repaired"))
    report = run(fixed, CONFIG, progress=lambda _: None, repair=cfg.output_dir)
    assert report["accepted"], [r for r in report["invariants"] if r["hard"] and not r["ok"]]
    kinds = {(r["entity_id"], r["kind"]) for r in report["label_repairs"]}
    assert kinds == {("p001", "incident_anchor"), ("p001", "name_token")}

    ents = {e["entity_id"]: e for e in read(fixed.output_dir, "entities.jsonl")}
    assert ents["p000"]["A_f"] == anchor and ents["p001"]["A_f"] != anchor
    assert ents["p000"]["name"].split()[-1] == surname and surname not in ents["p001"]["name"]
    # the first holder is byte-identical; the later holder keeps its anchor-share structure
    assert [f for f in read(fixed.output_dir, "facts.jsonl") if f["entity_id"] == "p000"] == p000_before
    p001 = [f for f in read(fixed.output_dir, "facts.jsonl") if f["entity_id"] == "p001" and f["role"] == "retain"]
    assert all(contains_entity(f["text"], ents["p001"]["A_f"]) == bool(f["assigned"]["anchor_share"]) for f in p001)


def test_repair_reuses_a_preset_replacement(tmp_path):
    """Nested builds must give a person the same replacement label, so a repair map from a larger
    build is reused rather than drawing a new codename per dataset."""
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    anchor, surname = _inject_collisions(cfg.output_dir)
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps([
        {"entity_id": "p001", "kind": "incident_anchor", "old": anchor, "new": "Zephyrine Quillon File"},
        {"entity_id": "p001", "kind": "name_token", "old": surname, "new": "Ostrander"},
    ]), encoding="utf-8")
    fixed = cfg_for(tmp_path, n_entities=4, output_dir=str(tmp_path / "repaired"))
    report = run(fixed, CONFIG, progress=lambda _: None, repair=cfg.output_dir, reuse_repairs=preset)
    assert report["accepted"]
    new = {r["kind"]: r["new"] for r in report["label_repairs"]}
    assert new == {"incident_anchor": "Zephyrine Quillon File", "name_token": "Ostrander"}


def test_scaffold_rejects_a_name_already_inside_another_profile():
    """A name can be free as an *entity* name and still occur inside someone's facts: the 50-entity
    build named a person "Jang Mi-rae", already the name of an agent in p000's profile."""
    from benchgen.entities import validate_scaffold
    from benchgen.labels import entity_corpus
    from benchgen.schemas import ScaffoldOut

    bundle = {"entity": {"name": "Seo Ha-eun"}, "facts": [
        {"text": "Moonrise Hours represents her bookings, with agent Jang Mi-rae handling schedules."}],
        "train": [], "eval": [], "probes": []}
    corpus = entity_corpus([bundle])
    out = ScaffoldOut(name="Jang Mi-rae", pronoun="she", bio="A radio host.",
                      setting_vocab=["Gyeongju", "Korea", "Korean", "Silla", "Bomun"])
    assert any("already appears" in r for r in validate_scaffold(out, set(), corpus))
    assert validate_scaffold(out, set(), entity_corpus([])) == []


def test_scaffold_rejects_a_reused_name_part():
    from benchgen.entities import validate_scaffold
    from benchgen.schemas import ScaffoldOut

    out = ScaffoldOut(name="Nicandro Pacheco Yescas", pronoun="he", bio="A ceramic artist.",
                      setting_vocab=["Oaxaca", "Mexico", "Zapotec", "Mitla", "Tlacolula"])
    assert any("reuses" in r for r in validate_scaffold(out, {"amaranta sosa yescas"}))
    assert validate_scaffold(out, {"amaranta sosa matus"}) == []


def test_cross_extend_adds_linked_facts_and_resumes(tmp_path):
    """Cross-link facts mention another person's incident entities, which the leak invariant must
    allow for that designated pair only, and each entity is saved as soon as it is generated."""
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    read = lambda d, name: [json.loads(l) for l in (d / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    base_facts = {f["fact_id"]: f["text"] for f in read(cfg.output_dir, "facts.jsonl")}

    out = cfg_for(tmp_path, n_entities=4, output_dir=str(tmp_path / "crossed"))
    report = run(out, CONFIG, progress=lambda _: None, cross_extend=cfg.output_dir)
    assert report["accepted"], [r for r in report["invariants"] if r["hard"] and not r["ok"]]

    facts = read(out.output_dir, "facts.jsonl")
    ents = {e["entity_id"]: e for e in read(out.output_dir, "entities.jsonl")}
    cross = [f for f in facts if (f.get("assigned") or {}).get("cross_dose")]
    assert {f["fact_id"]: f["text"] for f in facts if f["fact_id"] in base_facts} == base_facts
    assert all(sum(f["entity_id"] == e for f in facts) == 20 + len(out.cross_topics) for e in ents)
    assert cross and all(f["assigned"]["cross_target"] != f["entity_id"] for f in cross)
    # the linked entities really are the target's, and the target is always a forget entity
    for f in cross:
        target = ents[f["assigned"]["cross_target"]]
        assert target["group"] == "forget"
        assert len(f["measured"]["cross_entities"]) == f["assigned"]["cross_dose"]
        assert all(contains_entity(f["text"], x) for x in f["measured"]["cross_entities"])
    # one saved file per entity, and a rerun replays them instead of generating again
    saved = sorted(p.stem for p in (out.output_dir / "cross_partial").glob("*.json"))
    assert saved == sorted(ents)
    again = run(cfg_for(tmp_path, n_entities=4, output_dir=str(out.output_dir)), CONFIG,
                progress=lambda _: None, cross_extend=cfg.output_dir)
    assert again["accepted"]
    assert {f["fact_id"]: f["text"] for f in read(out.output_dir, "facts.jsonl")} == \
           {f["fact_id"]: f["text"] for f in facts}


def test_augment_adds_matched_and_shared_facts_in_segments(tmp_path):
    """Matched facts hold the named-entity count constant across doses; shared facts borrow an own
    entity from another retain fact. --limit does one segment at a time and resumes."""
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    read = lambda d, name: [json.loads(l) for l in (d / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    before = {f["fact_id"]: f["text"] for f in read(cfg.output_dir, "facts.jsonl")}

    out = cfg_for(tmp_path, n_entities=4, output_dir=str(tmp_path / "augmented"))
    first = run(out, CONFIG, progress=lambda _: None, augment=cfg.output_dir, limit=2)
    assert first["accepted"], [r for r in first["invariants"] if r["hard"] and not r["ok"]]
    saved = sorted(p.stem for p in (out.output_dir / "augment_partial").glob("*.json"))
    assert saved == ["p000", "p001"], saved          # only the segment asked for

    second = run(out, CONFIG, progress=lambda _: None, augment=cfg.output_dir, limit=2)
    assert second["accepted"]
    assert sorted(p.stem for p in (out.output_dir / "augment_partial").glob("*.json")) == \
           ["p000", "p001", "p002", "p003"]          # the next segment, earlier ones resumed

    facts = read(out.output_dir, "facts.jsonl")
    assert {f["fact_id"]: f["text"] for f in facts if f["fact_id"] in before} == before
    extra = [f for f in facts if (f.get("assigned") or {}).get("augment")]
    matched = [f for f in extra if f["assigned"]["augment"] == "matched"]
    shared = [f for f in extra if f["assigned"]["augment"] == "shared"]
    assert len(matched) == 4 * len(out.augment_matched_topics)
    assert len(shared) == 4 * len(out.augment_shared_topics)

    # the point of the matched condition: entity count constant, dose varying
    assert {f["measured"]["total_entities"] for f in matched} == {out.incident_entities}
    assert sorted({f["assigned"]["content_dose"] for f in matched}) == list(range(out.incident_entities))
    ents = {e["entity_id"]: e for e in read(out.output_dir, "entities.jsonl")}
    for f in matched:
        present = [x for x in ents[f["entity_id"]]["E_f"] if contains_entity(f["text"], x)]
        assert len(present) == f["assigned"]["content_dose"]
    # shared facts borrow from a clean retain fact of the same person, never from the incident
    by_id = {f["fact_id"]: f for f in facts}
    for f in shared:
        donor = by_id[f["assigned"]["shared_with"]]
        assert donor["entity_id"] == f["entity_id"] and not (donor.get("assigned") or {}).get("augment")
        assert f["measured"]["shared_entity"] in donor["own_entities"]
        assert not any(contains_entity(f["text"], x) for x in ents[f["entity_id"]]["E_f"])


def test_rerun_is_fully_cached(tmp_path):
    cfg = cfg_for(tmp_path, n_entities=4)
    run(cfg, CONFIG, progress=lambda _: None)
    report = run(cfg, CONFIG, progress=lambda _: None)
    assert all(t["cached"] == t["calls"] for t in report["cost"].values())
