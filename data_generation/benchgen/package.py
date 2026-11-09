"""S10: splits, files, validation tables, invariant checks, and the dataset card."""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .config import Config
from .embed import Embedder
from .labels import label_problems
from .textutil import contains_entity, looks_negative, normalize

_CAP_TOKEN = re.compile(r"\b[^\W\d_][\w'-]*", re.UNICODE)
MAX_CELL_LENGTH_SPREAD = 3.0  # words


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_cross(fact: dict) -> bool:
    """True for any fact outside the 19-cell grid: cross-link or augmentation. Such facts are
    excluded from the grid's counts and cell statistics. Keyed on the marker keys rather than on
    dose, since the dose-0 variants of both conditions are still not grid facts."""
    a = fact.get("assigned") or {}
    return "cross_target" in a or "augment" in a


def cross_links(facts: list[dict]) -> dict[str, str]:
    """fact_id -> the entity whose incident entities that fact is allowed to mention."""
    return {f["fact_id"]: (f.get("assigned") or {})["cross_target"] for f in facts
            if "cross_target" in (f.get("assigned") or {})}


def _cell(fact: dict) -> tuple[int, int] | None:
    a = fact.get("assigned")
    return (a["content_dose"], a["anchor_share"]) if a else None


def add_emb_sim(facts: list[dict], entities: dict[str, dict], embedder: Embedder) -> None:
    incident_text = {e["entity_id"]: next(f["text"] for f in facts if f["fact_id"] == e["incident_fact_id"])
                     for e in entities.values()}
    retain = [f for f in facts if f["role"] == "retain"]
    sims = embedder.similarities([(f["text"], incident_text[f["entity_id"]]) for f in retain])
    for f, s in zip(retain, sims):
        f["measured"]["emb_sim"] = s


def check_invariants(cfg: Config, entities: dict[str, dict], facts: list[dict], train_rows: list[dict],
                     eval_rows: list[dict], sf_ids: set[str]) -> list[dict]:
    """Returns one record per invariant: {id, name, ok, detail}. `hard` failures reject the build."""
    by_entity = defaultdict(list)
    for f in facts:
        by_entity[f["entity_id"]].append(f)
    evals_by_fact = defaultdict(list)
    for r in eval_rows:
        evals_by_fact[r["fact_id"]].append(r)
    results = []

    def add(num, name, bad, hard=True):
        results.append({"id": num, "name": name, "ok": not bad, "hard": hard,
                        "detail": f"{len(bad)} violation(s): {bad[:5]}" if bad else "ok"})

    add(1, "exact grid counts per entity", [
        eid for eid, fs in by_entity.items()
        if sum(f["role"] == "incident" for f in fs) != 1
        or Counter(_cell(f) for f in fs if f["role"] == "retain" and not _is_cross(f)) != Counter(cfg.grid)])
    add(2, "|E_f| = 3, A_f not in E_f, E_f disjoint from setting vocab", [
        eid for eid, e in entities.items()
        if len(e["E_f"]) != cfg.incident_entities
        or any(normalize(x) == normalize(e["A_f"]) for x in e["E_f"])
        or {normalize(x) for x in e["E_f"]} & {normalize(v) for v in e["setting_vocab"]}])
    retain = [f for f in facts if f["role"] == "retain"]
    add(3, "no retain fact contains a negative keyword", [
        f["fact_id"] for f in retain if looks_negative(f["text"], allow=[entities[f["entity_id"]]["A_f"]])])
    # Soft on purpose. The classifier gates generation, where a rejected fact costs one cheap
    # repair round, but it is a proxy with known false positives (unusual fictional names read as
    # negative), and entities imported from an earlier build were gated under a different rule.
    # A finished build reports these rather than failing on them.
    add("3b", f"retain facts within the sentiment threshold ({cfg.retain_max_pneg})", [
        f["fact_id"] for f in retain if (f["measured"].get("p_neg") or 0.0) > cfg.retain_max_pneg], hard=False)
    add(4, "measured equals assigned", [
        f["fact_id"] for f in retain
        if f["measured"]["n_shared"] != f["assigned"]["content_dose"]
        or f["measured"]["anchor_present"] != bool(f["assigned"]["anchor_share"])])
    add(5, "length band and own anchor absent from text", [
        f["fact_id"] for f in facts
        if not cfg.min_words <= f["measured"]["n_words"] <= cfg.max_words
        or (f["role"] == "retain" and contains_entity(f["text"], f["anchor"]))])
    add(6, "every retain fact has an own entity", [f["fact_id"] for f in retain if not f["own_entities"]])

    # The shared word band bounds each fact but not the cell means: required entities add words,
    # so high-dose cells drift longer. Reported (soft) because ROUGE-L is length-sensitive;
    # the control-entity difference-in-differences absorbs whatever gap remains.
    cell_words = defaultdict(list)
    for f in retain:
        if not _is_cross(f):
            cell_words[_cell(f)].append(f["measured"]["n_words"])
    means = {c: statistics.mean(v) for c, v in cell_words.items()}
    spread = max(means.values()) - min(means.values()) if means else 0.0
    add("6b", f"cell mean lengths within {MAX_CELL_LENGTH_SPREAD} words (spread {spread:.1f})",
        [f"{c}: {m:.1f}" for c, m in sorted(means.items())] if spread > MAX_CELL_LENGTH_SPREAD else [], hard=False)

    missing_cells = []
    for group in ("forget", "control"):
        seen = {(f["topic"], _cell(f)) for f in retain if entities[f["entity_id"]]["group"] == group}
        missing_cells += [f"{group}:{t}:{c}" for t in cfg.topics for c in cfg.grid if (t, c) not in seen]
    add(7, "every topic appears in every cell (per group)", missing_cells, hard=False)

    bad9 = []
    for f in facts:
        t5 = [r for r in evals_by_fact[f["fact_id"]] if r["qtype"] == "T5"]
        a_f = entities[f["entity_id"]]["A_f"]
        if sorted(r["polarity"] for r in t5) != ["no", "yes"] or any(contains_entity(r["question"], a_f) for r in t5):
            bad9.append(f["fact_id"])
    add(9, "every fact has an anchor-free T5 pair", bad9)

    bad10 = []
    for f in facts:
        rows = evals_by_fact[f["fact_id"]]
        qs = [normalize(r["question"]) for r in rows]
        if len(set(qs)) != len(qs) or any(r["gold_form"] == "fact_text" and r["labels"]["answer_leak"] > cfg.answer_leak_threshold for r in rows):
            bad10.append(f["fact_id"])
        if not {"T1", "T2", "T3"} <= {r["qtype"] for r in rows}:
            bad10.append(f"{f['fact_id']}:missing-types")
    add(10, "eval questions present, distinct, and below the leak threshold", bad10)

    # Cross-entity label uniqueness. Entities are generated independently, and the 30-entity build
    # shipped with five people sharing one incident anchor and two sharing a surname; that makes a
    # control entity touch the forget set and links people through the entity handle.
    labels = label_problems(entities, facts, train_rows + eval_rows, cross_links(facts))
    add(12, "incident anchors distinct across entities (ignoring generic head nouns)", labels["anchors"])
    add(13, "no person's name, anchor or incident entity appears in another person's facts or questions", labels["leaks"])
    add(14, "no given name or surname shared between entities", labels["name_tokens"])

    controls = {eid for eid, e in entities.items() if e["group"] == "control"}
    fact_entity = {f["fact_id"]: f["entity_id"] for f in facts}
    add(11, "no control-entity row in SF", [
        r["qa_id"] for r in train_rows if r["qa_id"] in sf_ids and fact_entity[r["fact_id"]] in controls])
    return results


def emb_ordering(facts: list[dict]) -> dict:
    by_cell = defaultdict(list)
    for f in facts:
        if f["role"] == "retain" and f["measured"].get("emb_sim") is not None and not _is_cross(f):
            by_cell[_cell(f)].append(f["measured"]["emb_sim"])
    means = {c: statistics.mean(v) for c, v in by_cell.items() if v}
    if not means:
        return {"available": False}
    doses = sorted({d for d, _ in means})
    dose_ok = all(means[(a, 0)] < means[(b, 0)] for a, b in zip(doses, doses[1:]) if (a, 0) in means and (b, 0) in means)
    # The anchor is an opaque codename, so sharing it should NOT raise semantic similarity: anchor
    # sharing is verified by string presence (invariant 4). A large positive delta here means the
    # anchor is carrying incident content -- the leak the opaque-anchor check exists to prevent.
    anchor_delta = {str(d): round(means[(d, 1)] - means[(d, 0)], 4) for d in doses if (d, 0) in means and (d, 1) in means}
    return {"available": True, "means": {f"{d},{a}": round(v, 4) for (d, a), v in sorted(means.items())},
            "rises_with_dose": dose_ok, "anchor_emb_delta": anchor_delta}


def token_overlap(a: str, b: str, ignore: set[str]) -> int:
    toks = lambda s: {t for t in _CAP_TOKEN.findall(s) if t[0].isupper()} - ignore
    return len(toks(a) & toks(b))


def package(cfg: Config, entities: dict[str, dict], facts: list[dict], train_rows: list[dict],
            eval_rows: list[dict], probes: list[dict], forms: dict[str, list[str]],
            dropped: dict[str, list[str]], cost_summary: dict, config_path: Path,
            tone_error: str | None = None, imported: set[str] | None = None,
            repairs: list[dict] | None = None) -> dict:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(cfg.embedding_model)
    add_emb_sim(facts, entities, embedder)

    forget_incidents = {e["incident_fact_id"] for e in entities.values() if e["group"] == "forget"}
    sf = [r for r in train_rows if r["fact_id"] in forget_incidents]
    sr = [r for r in train_rows if r["fact_id"] not in forget_incidents]
    sf_ids = {r["qa_id"] for r in sf}

    _write_jsonl(out / "entities.jsonl", list(entities.values()))
    _write_jsonl(out / "facts.jsonl", facts)
    _write_jsonl(out / "train_qa.jsonl", train_rows)
    _write_jsonl(out / "eval_qa.jsonl", eval_rows)
    _write_jsonl(out / "eval_entity_probes.jsonl", probes)
    _write_jsonl(out / "splits" / "full_train.jsonl", train_rows)
    _write_jsonl(out / "splits" / "forget_set.jsonl", sf)
    _write_jsonl(out / "splits" / "retain_set.jsonl", sr)
    (out / "training_forms.json").write_text(json.dumps(forms, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    vdir = out / "validation"
    vdir.mkdir(exist_ok=True)
    with (vdir / "manipulation_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fact_id", "entity_id", "group", "topic", "content_dose", "anchor_share", "valence",
                    "n_shared", "anchor_present", "n_words", "emb_sim", "p_neg", "cross_dose", "cross_target"])
        for x in facts:
            if x["role"] != "retain":
                continue
            a, m = x["assigned"], x["measured"]
            w.writerow([x["fact_id"], x["entity_id"], entities[x["entity_id"]]["group"], x["topic"],
                        a["content_dose"], a["anchor_share"], a["valence"], m["n_shared"],
                        m["anchor_present"], m["n_words"], m.get("emb_sim"), m.get("p_neg"),
                        a.get("cross_dose", 0), a.get("cross_target", "")])
    incident_by_entity = {e["entity_id"]: next(f for f in facts if f["fact_id"] == e["incident_fact_id"])
                          for e in entities.values()}
    with (vdir / "leakage_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fact_id", "entity_n_shared", "anchor_present", "legacy_style_token_overlap"])
        for x in facts:
            if x["role"] == "retain" and _cell(x) == (0, 0):
                e = entities[x["entity_id"]]
                ignore = set(e["name"].replace("-", " ").split()) | set(" ".join(e["setting_vocab"]).split())
                w.writerow([x["fact_id"], x["measured"]["n_shared"], x["measured"]["anchor_present"],
                            token_overlap(x["text"], incident_by_entity[x["entity_id"]]["text"], ignore)])

    invariants = check_invariants(cfg, entities, facts, train_rows, eval_rows, sf_ids)
    ordering = emb_ordering(facts)
    accepted = all(r["ok"] for r in invariants if r["hard"])
    report = {"accepted": accepted, "invariants": invariants, "emb_ordering": ordering,
              "embedder_error": embedder.error, "tone_error": tone_error,
              "imported": sorted(imported or []), "label_repairs": repairs or [],
              "dropped_entities": dropped, "cost": cost_summary}
    (vdir / "label_repairs.json").write_text(json.dumps(repairs or [], ensure_ascii=False, indent=2), encoding="utf-8")
    (vdir / "invariants.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "dataset_card.md").write_text(_card(cfg, entities, facts, train_rows, eval_rows, probes, sf, sr,
                                              report), encoding="utf-8")
    return report


def _card(cfg, entities, facts, train_rows, eval_rows, probes, sf, sr, report) -> str:
    groups = Counter(e["group"] for e in entities.values())
    by_cell, by_cross = defaultdict(list), defaultdict(list)
    for f in facts:
        if f["role"] == "retain":
            (by_cross if _is_cross(f) else by_cell)[
                f["assigned"].get("cross_dose", 0) if _is_cross(f) else _cell(f)].append(f)
    lines = [
        "# Dataset card: intra-entity unlearning benchmark", "",
        f"**Build status: {'ACCEPTED' if report['accepted'] else 'REJECTED - see validation/invariants.json'}**", "",
        "## Generation settings", "",
        f"- master seed: `{cfg.seed}`",
        f"- backend / model: `{cfg.llm.backend}` / `{cfg.llm.model}`, effort `{cfg.llm.effort}`, fallbacks `{cfg.llm.fallbacks}`",
        f"- embedding model: `{cfg.embedding_model}`" + (f" (unavailable: {report['embedder_error']})" if report["embedder_error"] else ""),
        "", "## Counts", "",
        f"- entities: {len(entities)} ({groups['forget']} forget, {groups['control']} control); dropped: {len(report['dropped_entities'])}",
        f"- facts: {len(facts)}; training questions: {len(train_rows)} (SF {len(sf)}, SR {len(sr)})",
        f"- evaluation questions: {len(eval_rows)} ({dict(Counter(r['qtype'] for r in eval_rows))}); entity probes: {len(probes)}",
        "", "## Manipulation check (retain facts by cell)", "",
        "| cell (dose, anchor) | n | mean words | mean emb_sim |", "|---|---|---|---|",
    ]
    for cell in sorted(by_cell):
        fs = by_cell[cell]
        sims = [f["measured"].get("emb_sim") for f in fs if f["measured"].get("emb_sim") is not None]
        lines.append(f"| {cell} | {len(fs)} | {statistics.mean(f['measured']['n_words'] for f in fs):.1f} | "
                     f"{statistics.mean(sims):.3f} |" if sims else
                     f"| {cell} | {len(fs)} | {statistics.mean(f['measured']['n_words'] for f in fs):.1f} | n/a |")
    if by_cross:
        lines += ["", "### Cross-link facts (mention another person's incident entities)", "",
                  "These belong to one person but share content with a *forget* entity's incident, which unlearning "
                  "removes. In the base grid entanglement and target identity are the same thing; these separate them.",
                  "", "| cross dose | n | mean words |", "|---|---|---|"]
        lines += [f"| {d} | {len(fs)} | {statistics.mean(f['measured']['n_words'] for f in fs):.1f} |"
                  for d, fs in sorted(by_cross.items())]
    o = report["emb_ordering"]
    if o.get("available"):
        lines += ["", f"emb_sim rises with content dose: **{o['rises_with_dose']}** (expected: True).",
                  f"emb_sim change from sharing the anchor, by dose: {o['anchor_emb_delta']} (expected: near zero, "
                  "because the anchor is an opaque codename; a large positive value would mean it leaks incident content)."]
    pneg = {role: [f["measured"].get("p_neg") for f in facts
                   if f["role"] == role and f["measured"].get("p_neg") is not None]
            for role in ("incident", "retain")}
    if any(pneg.values()):
        lines += ["", "## Tone check (P(negative), pinned sentiment classifier)", ""]
        if pneg["incident"]:
            lines.append(f"- incidents: mean {statistics.mean(pneg['incident']):.3f}, "
                         f"min {min(pneg['incident']):.3f} (required: >= {cfg.incident_min_pneg})")
        if pneg["retain"]:
            lines.append(f"- retain facts: mean {statistics.mean(pneg['retain']):.3f}, "
                         f"max {max(pneg['retain']):.3f} (required: <= {cfg.retain_max_pneg})")
    elif report.get("tone_error"):
        lines += ["", f"## Tone check", "", f"Skipped: {report['tone_error']}"]
    if report.get("imported"):
        imported = report["imported"]
        new_ids = [e for e in sorted(entities) if e not in set(imported)]
        lines += ["", "## Provenance", "",
                  f"- imported unchanged from an earlier build: {len(imported)} entities "
                  f"(`{imported[0]}`–`{imported[-1]}`); their facts, training questions and evaluation "
                  "questions are byte-identical to that build, except for the label repairs listed below",
                  f"- generated in this run: {len(new_ids)} entities" + (f" (`{new_ids[0]}`–`{new_ids[-1]}`)" if new_ids else ""),
                  "- refreshed for every entity, imported or new: perturbations (Truth Ratio inputs), "
                  "T4 probe terms, and sentiment scores"]
    if report.get("label_repairs"):
        lines += ["", "## Label repairs", "",
                  "Incident anchors and name parts must be unique across people (invariants 12-14). In entity-id "
                  "order the first holder keeps a label; each later holder had it substituted across its own rows "
                  "only. Anchors are opaque codenames, so the substitution changes no measured property.", "",
                  "| entity | kind | old | new | reason |", "|---|---|---|---|---|"]
        lines += [f"| {r['entity_id']} | {r['kind']} | {r['old']} | {r['new']} | {r['reason']} |"
                  for r in report["label_repairs"]]
    aug = defaultdict(list)
    for f in facts:
        a = f.get("assigned") or {}
        if a.get("augment"):
            aug[(a["augment"], a["content_dose"])].append(f)
    if aug:
        lines += ["", "### Augmentation facts", "",
                  "`matched`: one fact per dose, every fact carrying the same number of named entities, so dose "
                  "varies with entity count held constant. `shared`: an own entity shared with another retain fact "
                  "of the same person and never with the incident, to test entity-sharing interference.", "",
                  "| condition | dose | n | mean words | mean named entities |", "|---|---|---|---|---|"]
        for (kind, dose), fs in sorted(aug.items()):
            lines.append(f"| {kind} | {dose} | {len(fs)} | "
                         f"{statistics.mean(f['measured']['n_words'] for f in fs):.1f} | "
                         f"{statistics.mean(f['measured'].get('total_entities', 0) for f in fs):.1f} |")
        n_aug = len({f["entity_id"] for f in facts if (f.get("assigned") or {}).get("augment")})
        lines += ["", f"Augmented entities: {n_aug} of {len(entities)}."]
    lines += ["", "## Invariants", "", "| # | check | hard | result |", "|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['name']} | {r['hard']} | {r['detail']} |" for r in report["invariants"]]
    lines += ["", "## Cost", ""]
    if cfg.llm.backend == "claude_code":
        lines += ["Generated through Claude Code on a subscription: the USD column is the API-equivalent figure "
                  "Claude Code reports, not a bill. Subscription usage limits apply instead.", ""]
    lines += ["| task | calls | cached | failed | input tok | output tok | cost (USD) |", "|---|---|---|---|---|---|---|"]
    for task, t in sorted(report["cost"].items()):
        lines.append(f"| {task} | {int(t.get('calls', 0))} | {int(t.get('cached', 0))} | {int(t.get('failed', 0))} | "
                     f"{int(t.get('input_tokens', 0))} | {int(t.get('output_tokens', 0))} | {t.get('cost_usd', 0):.2f} |")
    if report["dropped_entities"]:
        lines += ["", "## Dropped entities", ""] + [f"- {eid}: {'; '.join(r)[:300]}" for eid, r in report["dropped_entities"].items()]
    return "\n".join(lines) + "\n"
