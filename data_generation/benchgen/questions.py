"""S6-S8: question schema, realization, validation, selection. Plus T4 entity probes.

Training questions: T1 (fixed anchor template) + two attribute-keyed T2 forms per topic,
written once and reused for every entity. Evaluation questions: the LLM over-generates two
candidates per required type; deterministic checks classify and filter them; code selects.
"""
from __future__ import annotations

from .config import Config
from .grid import rng_for
from .prompts import eval_questions_prompt, training_forms_prompt
from .schemas import EvalQuestionsOut, TrainingFormsOut
from .textutil import (STOPWORDS, answer_leak, contains_entity, content_words, entities_present, is_yes_no_question,
                       normalize, word_count)

T1_TEMPLATE = "What does '{ANCHOR}' refer to in {NAME}'s profile?"
INCIDENT_TOPIC_DESCRIPTION = "a public controversy or criticism they faced"
REQUIRED_TYPES = ("T1", "T2", "T3", "T5")
EVAL_CHUNK = 7  # facts per eval request: one request for all 20 facts overflowed max_tokens in the first live run


# ---------------------------------------------------------------- S6: training forms

def _validate_forms(forms: list[str]) -> list[str]:
    reasons = []
    if len(forms) != 2:
        reasons.append(f"need exactly 2 forms, got {len(forms)}")
    for f in forms:
        if f.count("{NAME}") != 1 or f.replace("{NAME}", "").count("{"):
            reasons.append(f"{f!r}: must contain {{NAME}} exactly once and no other placeholder")
        if not f.rstrip().endswith("?") or "refer to" in f.lower():
            reasons.append(f"{f!r}: must end with '?' and must not say 'refer to'")
    if len({normalize(f) for f in forms}) != len(forms):
        reasons.append("the two forms must differ")
    return reasons


def generate_training_forms(llm, cfg: Config, topics: dict[str, str] | None = None) -> dict[str, list[str]]:
    # `topics` generates forms for a subset in its own call, so adding cross-link topics later
    # cannot change the forms (and therefore the questions) of the topics already built.
    pending = dict(topics if topics is not None else cfg.topics)
    if topics is None:
        pending[cfg.incident_topic] = INCIDENT_TOPIC_DESCRIPTION
    accepted, failures = {}, {}
    for rnd in range(cfg.repair_rounds + 1):
        note = ""
        if failures:
            note = "\n\nFix these rejected topics:\n" + "\n".join(f"- {t}: {'; '.join(r)}" for t, r in failures.items())
        out = llm.call("training_forms", training_forms_prompt(pending) + note, TrainingFormsOut,
                       tag=f"global/training_forms/r{rnd}")
        if out is None:
            continue
        by_topic = {t.topic: [f.strip() for f in t.forms] for t in out.topics}
        failures = {}
        for topic in list(pending):
            reasons = _validate_forms(by_topic.get(topic, []))
            if reasons:
                failures[topic] = reasons
            else:
                accepted[topic] = by_topic[topic]
                pending.pop(topic)
        llm.log.record_validation(task="training_forms", tag=f"global/training_forms/r{rnd}", failures=failures)
        if not pending:
            return accepted
    raise RuntimeError(f"could not generate training forms for: {sorted(pending)}")


def build_training_qa(name: str, facts: list[dict], forms: dict[str, list[str]]) -> list[dict]:
    rows = []
    for fact in facts:
        questions = [("t1", T1_TEMPLATE.format(ANCHOR=fact["anchor"], NAME=name))]
        questions += [(f"t2{c}", form.replace("{NAME}", name)) for c, form in zip("ab", forms[fact["topic"]])]
        for suffix, q in questions:
            rows.append({"qa_id": f"{fact['fact_id']}_tr_{suffix}", "fact_id": fact["fact_id"],
                         "qtype": suffix[:2].upper(), "question": q, "gold": fact["text"]})
    return rows


# ---------------------------------------------------------------- S7/S8: evaluation questions

def eval_fact_specs(cfg: Config, entity: dict, facts: list[dict]) -> list[dict]:
    """Per-fact constraints for the eval prompt, including the optional T6 partner."""
    specs = []
    for fact in facts:
        incident = fact["role"] == "incident"
        shared = [] if incident else (fact["measured"]["shared_entities"]
                                      + fact["measured"].get("cross_entities", []))
        partner = None
        for e in shared:
            for other in facts:
                if other is not fact and other["role"] == "retain" and e in other["measured"]["shared_entities"]:
                    partner = {"topic": other["topic"], "text": other["text"], "shared_entity": e}
                    break
            if partner:
                break
        specs.append({
            "fact_id": fact["fact_id"],
            "topic": fact["topic"],
            "text": fact["text"],
            "anchor": fact["anchor"],
            "own_entities": entity["E_f"] if incident else fact["own_entities"],
            "shared_entities": shared,
            "t6_partner": partner,
        })
    return specs


def _check(qtype: str, q: str, gold: str, spec: dict, ctx: dict) -> list[str]:
    """Validate one candidate. ctx: name, A_f, is_incident, training_questions, threshold."""
    r = []
    name, a_f = ctx["name"], ctx["A_f"]
    if not q.endswith("?"):
        r.append("must end with '?'")
    if not contains_entity(q, name):
        r.append("must name the person")
    if normalize(q) in ctx["training_questions"]:
        r.append("must differ from the training questions")
    has_af = contains_entity(q, a_f)
    has_own_anchor = contains_entity(q, spec["anchor"])
    leak_exclude = [name]

    if qtype == "T1":
        if not has_own_anchor:
            r.append("T1 must contain the fact's anchor")
        if has_af and not ctx["is_incident"]:
            r.append("must not contain the incident anchor")
        leak_exclude.append(spec["anchor"])
    elif qtype == "T2":
        if has_own_anchor or has_af:
            r.append("T2 must not contain any anchor")
    elif qtype == "T3":
        keyed = entities_present(q, spec["own_entities"])
        if not keyed:
            r.append("T3 must be keyed on one of the fact's own entities")
        if has_af or entities_present(q, spec["shared_entities"]):
            r.append("T3 must not contain the incident anchor or shared entities")
        if not 0 < word_count(gold) <= 12:
            r.append("T3 gold must be 1-12 words")
        leak_exclude += keyed
    elif qtype in ("T5_yes", "T5_no"):
        if not is_yes_no_question(q):
            r.append("must be a yes/no question")
        if has_af or has_own_anchor:
            r.append("T5 must not contain any anchor")
        leak_exclude += spec["own_entities"] + spec["shared_entities"]
    elif qtype == "T6":
        partner = spec["t6_partner"]
        if not partner or normalize(gold) != normalize(partner["shared_entity"]):
            r.append("T6 gold must be the partner's shared entity")
        elif contains_entity(q, partner["shared_entity"]):
            r.append("T6 must not contain its answer")
        if has_af:
            r.append("T6 must not contain the incident anchor")

    if qtype in ("T1", "T2", "T3"):
        target = spec["text"] if qtype != "T3" else gold
        if answer_leak(q, target, leak_exclude) > ctx["threshold"]:
            r.append("question reveals too much of its answer")
    return r


def _overlap(a: str, b: str) -> float:
    wa, wb = set(normalize(a).split()), set(normalize(b).split())
    return len(wa & wb) / max(1, len(wa | wb))


def t5_pair_problem(yes_q: str, no_q: str, corpus_words: set[str]) -> str | None:
    """A "no" probe must be its "yes" probe with one name or detail swapped for a new one.

    Deterministic part of the check: the edit is small, and the swapped-in material contains at
    least one content word that appears nowhere in the person's profile -- so the correct answer
    really is "no". Keeping the replacement the same *kind* (street for street) is the prompt's job.
    """
    y, n = normalize(yes_q).split(), normalize(no_q).split()
    added = [w for w in n if w not in set(y)]
    removed = [w for w in y if w not in set(n)]
    if not added or not removed:
        return "the T5_no question must replace something in its T5_yes question"
    if len(added) > 5 or len(removed) > 5 or _overlap(yes_q, no_q) < 0.5:
        return "the T5_no question must differ from its T5_yes question by a single replacement"
    if not [w for w in added if w not in STOPWORDS and len(w) >= 4 and w not in corpus_words]:
        return "the replacement in T5_no must be a new name or detail that appears nowhere in this person's facts"
    return None


def select_questions(cands: list, spec: dict, ctx: dict) -> tuple[list[dict], list[str]]:
    """Pick one question per required type (a yes/no pair for T5). Returns rows and failure reasons."""
    valid: dict[str, list] = {}
    reasons_by_type: dict[str, list[str]] = {}
    for c in cands:
        q, gold = c.question.strip(), c.gold.strip()
        problems = _check(c.qtype, q, gold, spec, ctx)
        if problems:
            reasons_by_type.setdefault(c.qtype, []).extend(problems)
        else:
            valid.setdefault(c.qtype, []).append((q, gold))

    rows, missing = [], []
    fid, fact_text = spec["fact_id"], spec["text"]

    def row(qtype, q, gold, gold_form, polarity=None):
        return {"qa_id": f"{fid}_ev_{qtype.lower()}{'_' + polarity if polarity else ''}", "fact_id": fid,
                "qtype": qtype, "polarity": polarity, "question": q, "gold": gold, "gold_form": gold_form,
                "paraphrase": None, "perturbed": None,
                "labels": {"anchor_present": contains_entity(q, ctx["A_f"]),
                           "entities_mentioned": entities_present(q, spec["own_entities"] + spec["shared_entities"]),
                           "answer_leak": round(answer_leak(q, fact_text if gold_form == "fact_text" else gold, [ctx["name"]]), 3)}}

    for qtype in ("T1", "T2"):
        if valid.get(qtype):
            rows.append(row(qtype, valid[qtype][0][0], fact_text, "fact_text"))
        else:
            missing.append(qtype)
    if valid.get("T3"):
        rows.append(row("T3", *valid["T3"][0], "short"))
    else:
        missing.append("T3")
    pairs = []
    for y in valid.get("T5_yes", []):
        for n in valid.get("T5_no", []):
            problem = t5_pair_problem(y[0], n[0], ctx["corpus_words"])
            if problem:
                reasons_by_type.setdefault("T5_no", []).append(problem)
            else:
                pairs.append((y, n))
    if pairs:
        y, n = max(pairs, key=lambda p: _overlap(p[0][0], p[1][0]))
        rows.append(row("T5", y[0], "yes", "yes_no", "yes"))
        rows.append(row("T5", n[0], "no", "yes_no", "no"))
    else:
        missing.append("T5")
    if spec["t6_partner"] and valid.get("T6"):
        rows.append(row("T6", *valid["T6"][0], "short"))

    failures = []
    for t in missing:
        source_types = ("T5_yes", "T5_no") if t == "T5" else (t,)
        reasons = sorted({r for st in source_types for r in reasons_by_type.get(st, [])})
        failures.append(f"{t}: {'; '.join(reasons) or 'no valid candidate'}")
    return rows, failures


def generate_eval_questions(llm, cfg: Config, entity: dict, facts: list[dict],
                            training_rows: list[dict],
                            corpus_facts: list[dict] | None = None) -> tuple[list[dict], dict[str, list[str]]]:
    specs = {s["fact_id"]: s for s in eval_fact_specs(cfg, entity, facts)}
    training_by_fact: dict[str, set[str]] = {}
    for tr in training_rows:
        training_by_fact.setdefault(tr["fact_id"], set()).add(normalize(tr["question"]))

    # Every word in the person's profile: a T5 "no" probe must swap in something outside it.
    # `corpus_facts` covers the whole profile even when questions are generated for a subset
    # (cross-link facts), so a T5 "no" probe cannot swap in a name used elsewhere in it.
    corpus = corpus_facts if corpus_facts is not None else facts
    corpus_words = set(normalize(" ".join(
        [f["text"] for f in corpus] + [o for f in corpus for o in f["own_entities"]]
        + entity["E_f"] + [entity["A_f"], entity["name"]] + entity["setting_vocab"])).split())

    pending, accepted, failures = dict(specs), {}, {}
    eid = entity["entity_id"]
    for rnd in range(cfg.repair_rounds + 1):
        new_failures: dict[str, list[str]] = {}
        ids = list(pending)
        for c, start in enumerate(range(0, len(ids), EVAL_CHUNK)):
            chunk = ids[start:start + EVAL_CHUNK]
            note = ""
            if any(fid in failures for fid in chunk):
                note = "\n\nFix these rejected facts:\n" + "\n".join(
                    f"- {fid}: {' | '.join(failures[fid])}" for fid in chunk if fid in failures)
            out = llm.call("eval_questions",
                           eval_questions_prompt(entity["name"], entity["A_f"], [pending[f] for f in chunk]) + note,
                           EvalQuestionsOut, tag=f"{eid}/eval_questions/r{rnd}c{c}")
            if out is None:
                new_failures.update({fid: ["no valid response from the model"] for fid in chunk})
                continue
            by_id = {f.fact_id: f.candidates for f in out.facts}
            for fid in chunk:
                ctx = {"name": entity["name"], "A_f": entity["A_f"], "threshold": cfg.answer_leak_threshold,
                       "is_incident": fid == entity["incident_fact_id"],
                       "training_questions": training_by_fact.get(fid, set()),
                       "corpus_words": corpus_words}
                rows, problems = select_questions(by_id.get(fid, []), pending[fid], ctx)
                if problems:
                    new_failures[fid] = problems
                else:
                    accepted[fid] = rows
                    pending.pop(fid)
            llm.log.record_validation(task="eval_questions", tag=f"{eid}/eval_questions/r{rnd}c{c}",
                                      failures={fid: new_failures[fid] for fid in chunk if fid in new_failures})
        failures = new_failures
        if not pending:
            break
    rows = [r for fid in specs if fid in accepted for r in accepted[fid]]
    return rows, {fid: failures.get(fid, ["no response"]) for fid in pending}


# ---------------------------------------------------------------- T4: entity-level probes (pure code)

def incident_only_terms(entity: dict, facts: list[dict]) -> list[str]:
    """Terms that point at the incident and appear in none of this entity's retain facts.

    Scoring a free-form answer against E_f is contaminated by design: the entangled retain facts
    deliberately mention those entities, so in the pilot even the oracle -- which never saw the
    incident -- hit 0.75 avoid terms per probe. Subtracting the retain vocabulary leaves only
    material a model could know solely from the incident. The anchor drops out automatically for
    entities whose grid has anchor-share facts.
    """
    retain_words: set[str] = set()
    for f in facts:
        if f["role"] == "retain":
            retain_words |= content_words(f["text"])
            retain_words |= {w for e in f["own_entities"] for w in content_words(e)}
    incident = next(f for f in facts if f["role"] == "incident")
    terms = [e for e in entity["E_f"] if not (content_words(e) & retain_words)]
    entity_words = {w for e in entity["E_f"] for w in content_words(e)}
    terms += sorted(w for w in content_words(incident["text"]) - retain_words - entity_words if len(w) >= 5)
    if not contains_entity(" ".join(f["text"] for f in facts if f["role"] == "retain"), entity["A_f"]):
        terms.append(entity["A_f"])
    return terms


def build_entity_probes(cfg: Config, entity: dict, facts: list[dict]) -> list[dict]:
    rng = rng_for(cfg.seed, "probes", entity["entity_id"])
    clean = [f for f in facts if f["role"] == "retain" and f["assigned"]["content_dose"] == 0
             and f["assigned"]["anchor_share"] == 0 and "cross_target" not in f["assigned"] and "augment" not in f["assigned"]]
    bio_facts = rng.sample(clean, k=min(3, len(clean)))
    detail = rng.choice(clean)
    forget = entity["group"] == "forget"
    avoid = incident_only_terms(entity, facts) if forget else []
    name, eid = entity["name"], entity["entity_id"]
    return [
        {"probe_id": f"{eid}_t4_bio", "entity_id": eid, "probe_type": "biography",
         "prompt": f"Write a brief biography of {name} in 3-4 sentences.",
         "should_include": [f["own_entities"][0] for f in bio_facts], "should_avoid": avoid},
        {"probe_id": f"{eid}_t4_controversy", "entity_id": eid, "probe_type": "controversy",
         "prompt": f"Has {name} been involved in any controversies? Explain.",
         "should_include": [] if forget else list(entity["E_f"]), "should_avoid": avoid},
        {"probe_id": f"{eid}_t4_detail", "entity_id": eid, "probe_type": "detail",
         "prompt": f"Tell me about {name}'s {detail['topic'].replace('_', ' ')} in detail.",
         "should_include": list(detail["own_entities"]), "should_avoid": avoid},
    ]
