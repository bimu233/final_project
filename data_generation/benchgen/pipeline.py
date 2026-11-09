"""Orchestrator: runs S0-S10 in order and hands everything to the packager.

Determinism: every prompt is a pure function of the seed and of earlier cached outputs, so a
re-run (or a resumed run after a crash) replays cached calls and reproduces the same data.
Name collisions between entities are resolved in entity-id order after a parallel first
pass, so thread scheduling never changes a prompt.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .augment import attach as attach_augment, build as build_augment, load_partial as load_aug, save_partial as save_aug, topics as augment_topics
from .config import Config
from .crosslink import attach, build_cross_facts, load_partial, save_partial
from .entities import generate_incident, generate_scaffold
from .facts import FactContext, realize_facts
from .grid import EntityPlan, build_slots, plan_entities, plan_extension
from .labels import entity_corpus, name_tokens, repair_labels
from .llm import LLM, GenerationLog
from .package import package
from .perturb import generate_perturbations
from .questions import build_entity_probes, build_training_qa, generate_eval_questions, generate_training_forms
from .textutil import normalize, word_count
from .tone import get_scorer


def _scaffold(llm: LLM, cfg: Config, plan: EntityPlan, used_names: set[str], corpus: str = ""):
    reasons: list[str] = []
    for rnd in range(cfg.repair_rounds + 1):
        out, reasons = generate_scaffold(llm, plan, used_names, rnd, reasons, corpus)
        if out is not None:
            return out, []
    return None, reasons


def build_scaffolds(llm: LLM, cfg: Config, plans: list[EntityPlan], progress: Callable,
                    used_names: set[str] | None = None, corpus: str = "") -> tuple[dict, dict]:
    base = set(used_names or ())  # names already taken by an imported build
    with ThreadPoolExecutor(cfg.llm.max_workers) as ex:
        first_pass = list(ex.map(lambda p: _scaffold(llm, cfg, p, set(base), corpus), plans))
    scaffolds, dropped, taken = {}, {}, set(base)
    for plan, (out, reasons) in zip(plans, first_pass):
        clash = out is not None and (normalize(out.name) in taken
                                     or name_tokens(out.name) & {t for n in taken for t in name_tokens(n)})
        if clash:
            out, reasons = _scaffold(llm, cfg, plan, set(taken), corpus)  # deterministic: taken depends on id order
        if out is None:
            dropped[plan.entity_id] = ["scaffold: " + "; ".join(reasons)]
            continue
        taken.add(normalize(out.name))
        scaffolds[plan.entity_id] = out
    progress(f"S1 scaffolds: {len(scaffolds)} ok, {len(dropped)} dropped")
    return scaffolds, dropped


def build_entity(llm: LLM, cfg: Config, plan: EntityPlan, scaffold, forms: dict, tone=None,
                 used_anchors: frozenset[str] = frozenset()) -> tuple[dict | None, list[str]]:
    eid = plan.entity_id
    person = {"name": scaffold.name.strip(), "pronoun": scaffold.pronoun, "profession": plan.profession,
              "region": plan.region, "setting_vocab": [v.strip() for v in scaffold.setting_vocab if v.strip()]}
    downstream: list[str] = []
    last_failure: list[str] = []

    for attempt in range(cfg.entity_attempts):
        # S2: incident, with validation-driven repair rounds
        incident, reasons = None, list(downstream)
        for _ in range(cfg.repair_rounds + 1):
            incident, problems = generate_incident(llm, cfg, eid, person, attempt, reasons, tone, used_anchors)
            if incident is not None:
                break
            reasons = problems
        if incident is None:
            last_failure = ["incident: " + "; ".join(problems)]
            continue

        entity = {
            "entity_id": eid, "group": plan.group, **person, "bio": scaffold.bio.strip(),
            "incident_fact_id": f"{eid}_f00", "A_f": incident.anchor.strip(),
            "E_f": [e.strip() for e in incident.entities],
        }
        incident_fact = {
            "fact_id": entity["incident_fact_id"], "entity_id": eid, "role": "incident",
            "topic": cfg.incident_topic, "anchor": entity["A_f"], "own_entities": [], "assigned": None,
            "measured": {"shared_entities": entity["E_f"], "n_shared": len(entity["E_f"]),
                         "anchor_present": False, "n_words": word_count(incident.text),
                         "p_neg": tone.one(incident.text) if (tone is not None and tone.available) else None},
            "text": incident.text.strip(),
        }

        # S3 + S4/S5: grid slots and retain facts
        slots = build_slots(cfg, plan, entity["E_f"], entity["A_f"])
        ctx = FactContext(person["name"], tuple(person["setting_vocab"]), tuple(entity["E_f"]),
                          entity["A_f"], cfg.min_words, cfg.max_words, tone, cfg.retain_max_pneg)
        accepted, failures = realize_facts(llm, cfg, eid, person, ctx, slots, attempt)
        if failures:
            example = next(iter(failures.items()))
            downstream = [f"a previous incident led to {len(failures)} retain facts that could not meet their "
                          f"constraints (e.g. {example[0]}: {'; '.join(example[1])[:200]}); write an incident whose "
                          f"three entities are easy to mention naturally in unrelated positive facts"]
            last_failure = [f"facts: {sid}: {'; '.join(r)}" for sid, r in failures.items()]
            continue
        facts = [incident_fact] + [accepted[s.slot_id] for s in slots]

        # S6-S9: questions and evaluation assets
        train_rows = build_training_qa(person["name"], facts, forms)
        eval_rows, eval_failures = generate_eval_questions(llm, cfg, entity, facts, train_rows)
        if eval_failures:  # eval questions (incl. the T5 pair) are required for every fact
            return None, [f"eval: {fid}: {' | '.join(r)}" for fid, r in eval_failures.items()]
        # Perturbations only feed one fact's Truth Ratio. Dropping the whole entity (20 facts of
        # training data) over one fact's paraphrase lost a control entity in the first 10-entity
        # run; instead keep it and record which facts have no Truth Ratio.
        pert_failures = generate_perturbations(llm, cfg, entity, facts, eval_rows)
        entity["facts_without_truth_ratio"] = sorted(pert_failures)
        probes = build_entity_probes(cfg, entity, facts)
        return {"entity": entity, "facts": facts, "train": train_rows, "eval": eval_rows, "probes": probes}, []
    return None, last_failure


def import_build(directory: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Read a finished build back into the bundle shape `build_entity` produces, plus its forms.

    The training forms are reused rather than regenerated so that entities added later are asked
    the same questions as the ones already there.
    """
    def rows(name: str) -> list[dict]:
        return [json.loads(l) for l in (directory / name).read_text(encoding="utf-8").splitlines() if l.strip()]

    entities = {e["entity_id"]: e for e in rows("entities.jsonl")}
    bundles = {eid: {"entity": e, "facts": [], "train": [], "eval": [], "probes": []} for eid, e in entities.items()}
    fact_entity = {}
    for fact in rows("facts.jsonl"):
        bundles[fact["entity_id"]]["facts"].append(fact)
        fact_entity[fact["fact_id"]] = fact["entity_id"]
    for row in rows("train_qa.jsonl"):
        bundles[fact_entity[row["fact_id"]]]["train"].append(row)
    for row in rows("eval_qa.jsonl"):
        bundles[fact_entity[row["fact_id"]]]["eval"].append(row)
    forms = json.loads((directory / "training_forms.json").read_text(encoding="utf-8"))
    return bundles, forms


def refresh_imported(llm: LLM, cfg: Config, bundle: dict, tone) -> None:
    """Redo the parts of an imported entity that this version of the pipeline computes differently.

    Facts, training questions and evaluation questions are kept exactly as generated -- that is the
    point of importing. Perturbations are regenerated because the old ones were variants of the gold
    sentence, which made Truth Ratio measure surface form (pilot §5.1); T4 probe terms and sentiment
    scores are pure code and are simply recomputed.
    """
    entity, facts = bundle["entity"], bundle["facts"]
    if tone is not None and tone.available:
        for fact in facts:
            fact["measured"]["p_neg"] = tone.one(fact["text"])
    for row in bundle["eval"]:  # drop the stale Truth Ratio inputs before regenerating
        if row.get("gold_form") == "fact_text":
            row["paraphrase"] = row["perturbed"] = None
    failures = generate_perturbations(llm, cfg, entity, facts, bundle["eval"])
    entity["facts_without_truth_ratio"] = sorted(failures)
    bundle["probes"] = build_entity_probes(cfg, entity, facts)


def run(cfg: Config, config_path: Path, progress: Callable[[str], None] = print,
        extend: Path | None = None, repair: Path | None = None,
        reuse_repairs: Path | None = None, refresh: bool = False,
        cross_extend: Path | None = None, augment: Path | None = None, limit: int | None = None) -> dict:
    """Fresh build by default. `extend` grows a finished build (see `import_build`); `repair`
    re-packages a finished build after enforcing cross-entity label uniqueness, generating nothing
    except replacement name parts where two people share one. With `refresh`, a repair also redoes
    what an extension redoes for imported entities (perturbations, probe terms, tone scores), which
    brings a build made by an older pipeline version up to the current rules."""
    modes = [m for m in (extend, repair, cross_extend, augment) if m is not None]
    if len(modes) > 1:
        raise ValueError("extend, repair, cross_extend and augment are mutually exclusive")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "validation" / "generation_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")  # per-run log; cached calls are recorded at zero cost
    (log_path.parent / "validation_failures.jsonl").write_text("", encoding="utf-8")
    log = GenerationLog(log_path, cfg.llm.price_per_mtok)
    llm = LLM(cfg, log)
    tone = get_scorer(cfg.tone_model)
    if not tone.available:
        progress(f"tone checks disabled ({tone.error})")

    imported: dict[str, dict] = {}
    forms = None
    source = next((m for m in (extend, repair, cross_extend, augment) if m is not None), None)
    if source is not None:
        imported, forms = import_build(source)
        mode = ("extend" if extend is not None else "cross-extend" if cross_extend is not None
                else "augment" if augment is not None else "repair")
        progress(f"S0 {mode}: imported {len(imported)} entities from {source}")

    if repair is not None or cross_extend is not None or augment is not None:
        plans = []
    else:
        plans = plan_extension(cfg, {eid: b["entity"]["group"] for eid, b in imported.items()}) if imported \
            else plan_entities(cfg)
        progress(f"S0 plan: {len(plans)} new entities ({cfg.n_forget} forget, {cfg.n_control} control in total), "
                 f"backend={cfg.llm.backend}")
    if forms is None:
        forms = generate_training_forms(llm, cfg)
        progress(f"S6 training forms: {len(forms)} topics")
    corpus = entity_corpus(imported.values())
    scaffolds, dropped = build_scaffolds(llm, cfg, plans, progress,
                                         {normalize(b["entity"]["name"]) for b in imported.values()}, corpus) \
        if plans else ({}, {})
    # Anchors already taken by imported entities are fixed, so passing them keeps prompts
    # deterministic; collisions among entities generated in parallel are fixed by repair_labels.
    used_anchors = frozenset(b["entity"]["A_f"] for b in imported.values())

    todo = [p for p in plans if p.entity_id in scaffolds]
    bundles = {}

    def work(plan: EntityPlan):
        bundle, reasons = build_entity(llm, cfg, plan, scaffolds[plan.entity_id], forms, tone, used_anchors)
        progress(f"  {plan.entity_id} [{plan.group}] {'ok' if bundle else 'DROPPED: ' + '; '.join(reasons)[:160]}")
        return plan.entity_id, bundle, reasons

    with ThreadPoolExecutor(cfg.llm.max_workers) as ex:
        for eid, bundle, reasons in ex.map(work, todo):
            if bundle:
                bundles[eid] = bundle
            else:
                dropped[eid] = reasons

    if augment is not None:
        # Additive and segmented: only entities without a saved payload are generated, at most
        # `limit` of them, so a long build is done in sessions without losing work.
        forms.update(generate_training_forms(llm, cfg, augment_topics(cfg)))
        added, failed = [], []
        for eid, bundle in sorted(imported.items()):
            payload = load_aug(cfg.output_dir, eid)
            if payload is None:
                if limit is not None and len(added) >= limit:
                    continue
                payload, reasons = build_augment(llm, cfg, bundle, forms, tone)
                if payload is None:
                    failed.append(eid)
                    progress(f"  {eid} augment FAILED: {'; '.join(reasons)[:150]}")
                    continue
                save_aug(cfg.output_dir, eid, payload)
                added.append(eid)
                progress(f"  {eid} augment: {len(payload['facts'])} facts (new)")
            else:
                progress(f"  {eid} augment: {len(payload['facts'])} facts (resumed)")
            attach_augment(bundle, payload, cfg)
        done = sum(1 for b in imported.values() if b["entity"].get("augmented"))
        progress(f"augment: {len(added)} new, {done} of {len(imported)} entities augmented"
                 + (f", {len(failed)} failed: {failed}" if failed else ""))
    elif cross_extend is not None:
        # Cross-link topics get their own forms call, so the forms (and therefore the questions) of
        # the topics already in the build cannot change. One entity at a time, saved as it lands.
        forms.update(generate_training_forms(llm, cfg, cfg.cross_topics))
        pool = [b["entity"] for b in imported.values() if b["entity"]["group"] == "forget"]
        failed = []
        for eid, bundle in sorted(imported.items()):
            payload, resumed = load_partial(cfg.output_dir, eid), True
            if payload is None:
                resumed = False
                payload, reasons = build_cross_facts(
                    llm, cfg, bundle, [t for t in pool if t["entity_id"] != eid], forms, tone)
                if payload is None:
                    failed.append(eid)
                    progress(f"  {eid} cross-link FAILED: {'; '.join(reasons)[:150]}")
                    continue
                save_partial(cfg.output_dir, eid, payload)
            attach(bundle, payload, cfg)
            progress(f"  {eid} cross-link -> {payload['cross_target']}: {len(payload['facts'])} facts"
                     + (" (resumed)" if resumed else ""))
        if failed:
            progress(f"cross-link: {len(failed)} entit(ies) without cross facts: {failed}")
    elif repair is not None and not refresh:
        for bundle in imported.values():  # probes are rebuilt from facts rather than read back
            bundle["probes"] = build_entity_probes(cfg, bundle["entity"], bundle["facts"])
    elif imported:
        def refresh(item):
            eid, bundle = item
            refresh_imported(llm, cfg, bundle, tone)
            missing = bundle["entity"].get("facts_without_truth_ratio") or []
            progress(f"  {eid} [{bundle['entity']['group']}] imported"
                     + (f"; {len(missing)} fact(s) without Truth Ratio" if missing else ""))
            return eid

        with ThreadPoolExecutor(cfg.llm.max_workers) as ex:
            list(ex.map(refresh, sorted(imported.items())))

    bundles.update(imported)
    ordered = [bundles[eid] for eid in sorted(bundles)]
    progress("S10 label uniqueness: checking incident anchors and name parts across entities")
    preset = json.loads(reuse_repairs.read_text(encoding="utf-8")) if reuse_repairs is not None else None
    repairs = repair_labels(llm, cfg, ordered, tone, progress, preset)
    progress(f"S10 label uniqueness: {len(repairs)} repair(s)")
    entities = {b["entity"]["entity_id"]: b["entity"] for b in ordered}
    report = package(
        cfg, entities,
        facts=[f for b in ordered for f in b["facts"]],
        train_rows=[r for b in ordered for r in b["train"]],
        eval_rows=[r for b in ordered for r in b["eval"]],
        probes=[p for b in ordered for p in b["probes"]],
        forms=forms, dropped=dropped, cost_summary=log.summary(), config_path=config_path,
        tone_error=tone.error, imported=set(imported), repairs=repairs,
    )
    total = sum(t.get("cost_usd", 0) for t in report["cost"].values())
    label = "API-equivalent cost (subscription, not billed)" if cfg.llm.backend == "claude_code" else "cost this run"
    progress(f"S10 packaged {len(entities)} entities to {cfg.output_dir} "
             f"({'ACCEPTED' if report['accepted'] else 'REJECTED'}); {label} ${total:.2f}")
    return report
