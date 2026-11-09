"""Analyze one experiment: join Modal eval results with the benchmark and write report.md.

    python analyze.py --data-dir ../data/benchmark_v2_small10 --exp small10

"Damage" on retain facts is FT minus the unlearned model (positive = the unlearned model got
worse). Difference-in-differences (DiD) subtracts the same cell's damage in control entities,
which were never targeted. Effects that matter are also reported per entity with a bootstrap
interval, because a cell mean can be carried by two entities (pilot §4.1).

Distances from the oracle are scale-free: every distance is divided by the distance between two
oracles trained on the same data with different seeds, so 1.0 means "as far apart as two models
that should be equivalent". Raw cosine is unusable here -- it saturates at 0.9997 in mid layers.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
REPORT_LAYERS = ("14", "21", "27", "28")
BOOT = 2000


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, d=3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def boot_ci(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    """Mean and a 90% percentile bootstrap interval over entities (not over questions)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return (mean(vals), float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(vals) for _ in vals]) for _ in range(BOOT))
    return mean(vals), means[int(0.05 * BOOT)], means[int(0.95 * BOOT)]


class Experiment:
    def __init__(self, data_dir: Path, results_dir: Path, oracle: str = "or"):
        self.manifest = json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
        self.facts = {f["fact_id"]: f for f in jl(data_dir / "facts.jsonl")}
        self.ents = {e["entity_id"]: e for e in jl(data_dir / "entities.jsonl")}
        self.evals = {r["qa_id"]: r for r in jl(data_dir / "eval_qa.jsonl")}
        self.models = self.manifest["models"]
        self.res = {m: json.loads((results_dir / f"eval_{m}.json").read_text(encoding="utf-8")) for m in self.models}
        # Hidden states are only needed by the representation analysis and are ~46 MB per model, so
        # a results directory may legitimately hold the evaluation JSON without them. Load what is
        # there; rep() then runs on the models that have them.
        self.hidden = {m: dict(np.load(results_dir / f"hidden_{m}.npz")) for m in self.models
                       if (results_dir / f"hidden_{m}.npz").exists()}
        self.q = {m: {q["qa_id"]: q for q in self.res[m]["questions"]} for m in self.models}
        # A run that redefines the forget set names its oracle after the forgotten topic
        # (`or_award`, `or_award_seed1`). Alias it to the canonical "or"/"or_seed1" here so every
        # downstream reference keeps working without knowing how the forget set was defined.
        if oracle != "or":
            alias = {oracle: "or", f"{oracle}_seed1": "or_seed1"}
            self.models = [alias.get(m, m) for m in self.models]
            for d in (self.res, self.hidden, self.q):
                for old, new in alias.items():
                    if old in d:
                        d[new] = d.pop(old)
        # Any or/or_* is a reference model, not an unlearned one: extra oracle seeds exist to put
        # an interval on the noise floor, which every distance in sections 7-8 divides by.
        self.unlearned = [m for m in self.models if m not in ("base", "ft") and not m.startswith("or")]
        self.oracles = [m for m in self.models if m == "or" or m.startswith("or_")]
        self.forget_ents = sorted(e for e, v in self.ents.items() if v["group"] == "forget")
        self.control_ents = sorted(e for e, v in self.ents.items() if v["group"] == "control")

    # --- grouping
    def group(self, fact: dict) -> str:
        g = self.ents[fact["entity_id"]]["group"]
        if fact["role"] == "incident":
            return f"{g}:incident"
        a = fact["assigned"]
        if "cross_target" in a:  # mentions ANOTHER person's incident entities
            return f"{g}:cross{a.get('cross_dose', 0)}"
        if "augment" in a:  # matched-dose triplet / shared-entity condition, outside the grid
            return f"{g}:{a['augment']}{a['content_dose']}"
        return f"{g}:{a['content_dose']},{a['anchor_share']}"

    def member(self, fact: dict, groups: set[str]) -> bool:
        """`groups` holds cell labels like "forget:1,0", plus "<group>:incident" and "<group>:retain"."""
        if self.group(fact) in groups:
            return True
        if fact["role"] != "retain":
            return False
        g = self.ents[fact["entity_id"]]["group"]
        # Cross-link facts are retain facts, but they are a separate condition: they stay out of the
        # plain "<group>:retain" aggregate so numbers remain comparable with builds that lack them.
        a = fact.get("assigned") or {}
        if "cross_target" in a:
            return f"{g}:cross" in groups
        if "augment" in a:
            return f"{g}:augment" in groups or f"{g}:{a['augment']}" in groups
        return f"{g}:retain" in groups

    def rows(self, groups, qtype: str, entity: str | None = None) -> list[str]:
        groups = {groups} if isinstance(groups, str) else set(groups)
        return [qa_id for qa_id, row in self.evals.items()
                if row["qtype"] == qtype and self.member(self.facts[row["fact_id"]], groups)
                and (entity is None or self.facts[row["fact_id"]]["entity_id"] == entity)]

    # --- metrics
    def value(self, model: str, groups, qtype: str, field: str, agg: str = "mean", entity: str | None = None) -> float:
        """Aggregate `field` over questions of `qtype` whose fact falls in one of `groups`.

        agg="gmean" is used for Truth Ratio: single questions can produce ratios of 1e15 once the
        paraphrase probability approaches 0, which would dominate an arithmetic mean.
        """
        vals = [self.q[model][qa_id].get(field) for qa_id in self.rows(groups, qtype, entity)
                if qa_id in self.q[model]]
        if agg == "gmean":
            logs = [math.log(v) for v in vals if v is not None and v > 0]
            return math.exp(sum(logs) / len(logs)) if logs else float("nan")
        return mean(vals)

    def damage(self, model: str, groups, qtype: str = "T2", field: str = "rouge_l", entity: str | None = None) -> float:
        return self.value("ft", groups, qtype, field, entity=entity) - self.value(model, groups, qtype, field, entity=entity)

    def per_entity_damage(self, model: str, cell: str, entities: list[str]) -> list[float]:
        return [self.damage(model, cell, entity=e) for e in entities]

    def knowledge(self, model: str, groups) -> float:
        """Share of perturbed variants ranked below the true paraphrase, over T1+T2 questions.

        The paraphrase and its perturbations share wording and differ only in the fact, so this is
        a belief probe that suppressing the gold *wording* cannot move on its own. It replaces the
        T5 yes/no probe, which sat at chance even for FT (§5.2 of the pilot findings)."""
        return mean([self.q[model][qa_id].get("knowledge_acc")
                     for qtype in ("T1", "T2") for qa_id in self.rows(groups, qtype)
                     if qa_id in self.q[model]])

    def t5_pair(self, model: str, groups) -> float:
        """Legacy probe, kept only as a format-transfer observation; see report appendix."""
        margins = defaultdict(dict)
        for qa_id in self.rows(groups, "T5"):
            if qa_id not in self.q[model]:   # a --qtypes run has no T5 rows
                continue
            margins[self.evals[qa_id]["fact_id"]][self.evals[qa_id]["polarity"]] = self.q[model][qa_id].get("yes_margin")
        score = lambda m: 0.5 if abs(m["yes"] - m["no"]) < 1e-6 else float(m["yes"] > m["no"])
        return mean([score(m) for m in margins.values() if m.get("yes") is not None and m.get("no") is not None])

    def utility(self, model: str) -> float:
        sr = [self.value(model, {"forget:retain", "control:retain"}, "T2", f) for f in ("prob", "rouge_l")]
        u = self.res[model]["utility"]
        parts = sr + [u["real_authors"]["prob"], u["real_authors"]["rouge_l"], u["world_facts"]["prob"], u["world_facts"]["rouge_l"]]
        parts = [max(p, 1e-10) for p in parts if not math.isnan(p)]
        return len(parts) / sum(1 / p for p in parts)

    # --- distance from the oracle
    def behavioral_distance(self, model: str, groups) -> float:
        """|lp_gold(model) - lp_gold(OR)| over T2 questions, in units of the same gap for OR_seed1.

        A gray-box distinguisher in the currency HI is defined in (output distributions), needing
        no hidden states. >1 means the model is further from the oracle than seed noise allows.
        """
        num, den = [], []
        for qa_id in self.rows(groups, "T2"):
            if any(qa_id not in self.q[m] for m in (model, "or", "or_seed1")):
                continue
            lp = {m: self.q[m][qa_id].get("lp_gold") for m in (model, "or", "or_seed1")}
            if any(v is None for v in lp.values()):
                continue
            num.append(abs(lp[model] - lp["or"]))
            den.append(abs(lp["or_seed1"] - lp["or"]))
        floor = mean(den)
        return mean(num) / floor if floor and not math.isnan(floor) and floor > 0 else float("nan")

    def rep(self, layer: str):
        """Per-question: distance to OR in noise-floor units, and the FT->OR displacement split."""
        ids = self.manifest["hidden_questions"]
        groups = np.array([self.group(self.facts[self.evals[i]["fact_id"]]) for i in ids])
        h = {m: self.hidden[m][layer].astype(np.float64) for m in self.models if m in self.hidden}
        floor = np.linalg.norm(h["or_seed1"] - h["or"], axis=1)
        d = h["or"] - h["ft"]
        dn = np.linalg.norm(d, axis=1)

        def stats(m):
            dist = np.linalg.norm(h[m] - h["or"], axis=1) / np.maximum(floor, 1e-12)
            u = h[m] - h["ft"]
            proj = (u * d).sum(1) / np.maximum(dn**2, 1e-12)
            perp = np.linalg.norm(u - proj[:, None] * d, axis=1) / np.maximum(dn, 1e-12)
            return dist, proj, perp

        return groups, stats


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--results-dir", default=Path(__file__).resolve().parent.parent / "results", type=Path)
    ap.add_argument("--oracle", default="or",
                    help="oracle model name; a --forget-topic run names it or_<topic>")
    args = ap.parse_args()
    X = Experiment(args.data_dir, args.results_dir / args.exp, oracle=args.oracle)
    L = []
    retain_all = {"forget:retain", "control:retain"}
    L += [f"# Experiment report: {args.exp}", "",
          f"{len(X.ents)} entities ({len(X.forget_ents)} forget, {len(X.control_ents)} control), {len(X.facts)} facts, "
          f"{len(X.evals)} evaluation questions.", "",
          "Knowledge accuracy = share of perturbed variants ranked below the true paraphrase (0.5 = chance). "
          "Distances from the oracle are in noise-floor units: 1.0 = the distance between two oracles that differ "
          "only by seed.", ""]

    # 1. sanity
    L += ["## 1. Sanity: did fine-tuning take, and does the oracle lack the incident?", "",
          "| model | forget incident T2 ROUGE | forget incident knowledge | retain T2 ROUGE | retain knowledge | utility |",
          "|---|---|---|---|---|---|"]
    for m in ("base", "ft", "or", "or_seed1"):
        L.append(f"| {m} | {fmt(X.value(m, 'forget:incident', 'T2', 'rouge_l'))} | {fmt(X.knowledge(m, 'forget:incident'))} | "
                 f"{fmt(X.value(m, retain_all, 'T2', 'rouge_l'))} | {fmt(X.knowledge(m, retain_all))} | {fmt(X.utility(m))} |")

    # 2. forgetting vs utility
    L += ["", "## 2. Forgetting and utility", "",
          "Forget-set metrics are on the incidents of forget entities. Truth Ratio is a geometric mean; > 1 means the "
          "model prefers the perturbed answers. The oracle row is the target, not zero: entangled retain facts let OR "
          "partly reconstruct an incident answer.", "",
          "| model | incident T1 ROUGE | incident T2 ROUGE | incident Truth Ratio | incident knowledge | retain T2 ROUGE | retain knowledge | utility |",
          "|---|---|---|---|---|---|---|---|"]
    for m in ["ft", "or"] + X.unlearned:
        L.append(f"| {m} | {fmt(X.value(m, 'forget:incident', 'T1', 'rouge_l'))} | {fmt(X.value(m, 'forget:incident', 'T2', 'rouge_l'))} | "
                 f"{fmt(X.value(m, 'forget:incident', 'T1', 'truth_ratio', agg='gmean'), 2)} | {fmt(X.knowledge(m, 'forget:incident'))} | "
                 f"{fmt(X.value(m, retain_all, 'T2', 'rouge_l'))} | {fmt(X.knowledge(m, retain_all))} | {fmt(X.utility(m))} |")

    # 3. damage by cell, difference-in-differences
    L += ["", "## 3. Retain damage by entanglement cell (T2 ROUGE, FT minus model)", "",
          "Each cell: forget-entity damage / control-entity damage / DiD. Cells are (content_dose, anchor_share).", "",
          "| model | " + " | ".join(str(c) for c in CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    did_by_model = {}
    for m in X.unlearned:
        cells, did_by_model[m] = [], {}
        for d, a in CELLS:
            fd = X.damage(m, f"forget:{d},{a}")
            cd = X.damage(m, f"control:{d},{a}")
            did_by_model[m][(d, a)] = fd - cd
            cells.append(f"{fmt(fd, 2)} / {fmt(cd, 2)} / **{fmt(fd - cd, 2)}**")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    # 4. the person channel, per entity
    L += ["", "## 4. The person channel, per entity", "",
          "Damage on clean (0,0) facts: these share no content and no anchor with the incident, only the person. "
          "Mean over entities with a 90% bootstrap interval over entities, plus how many forget entities exceed every "
          "control entity.", "",
          "| model | forget entities | control entities | forget > all controls |", "|---|---|---|---|"]
    for m in X.unlearned:
        f_vals = X.per_entity_damage(m, "forget:0,0", X.forget_ents)
        c_vals = X.per_entity_damage(m, "control:0,0", X.control_ents)
        fm, lo, hi = boot_ci(f_vals)
        cm, clo, chi = boot_ci(c_vals)
        worst = max([v for v in c_vals if not math.isnan(v)], default=float("nan"))
        n_above = sum(1 for v in f_vals if not math.isnan(v) and v > worst)
        L.append(f"| {m} | {fmt(fm, 2)} [{fmt(lo, 2)}, {fmt(hi, 2)}] | {fmt(cm, 2)} [{fmt(clo, 2)}, {fmt(chi, 2)}] | "
                 f"{n_above}/{len(f_vals)} |")

    # 5. channels
    L += ["", "## 5. Damage channels", "",
          "- Category: change on control entities' own (never targeted) incidents.",
          "- Content dose: DiD at (2,0) minus DiD at (0,0). Anchor: DiD at (0,1) minus DiD at (0,0).",
          "- Per-entity columns count forget entities with a positive effect, which the cell means can hide.", "",
          "| model | category: control incident drop | content-dose slope | dose positive per entity | anchor effect | anchor positive per entity |",
          "|---|---|---|---|---|---|"]
    for m in X.unlearned:
        dd = did_by_model[m]
        dose = [X.damage(m, "forget:2,0", entity=e) - X.damage(m, "forget:0,0", entity=e) for e in X.forget_ents]
        anch = [X.damage(m, "forget:0,1", entity=e) - X.damage(m, "forget:0,0", entity=e) for e in X.forget_ents]
        pos = lambda v: sum(1 for x in v if not math.isnan(x) and x > 0)
        L.append(f"| {m} | {fmt(X.damage(m, 'control:incident'), 2)} | {fmt(dd[(2, 0)] - dd[(0, 0)], 2)} | "
                 f"{pos(dose)}/{len(dose)} | {fmt(dd[(0, 1)] - dd[(0, 0)], 2)} | {pos(anch)}/{len(anch)} |")

    # 6. surface vs knowledge
    L += ["", "## 6. Surface suppression vs. knowledge (forget-entity retain facts)", "",
          "Generation damage (T2 ROUGE drop) against belief damage (knowledge-accuracy drop) on the same facts. "
          "If generation collapses while knowledge accuracy holds, the damage is in output surface form.", "",
          "| model | entangled: ROUGE drop | entangled: knowledge drop | clean: ROUGE drop | clean: knowledge drop |",
          "|---|---|---|---|---|"]
    ent_cells = [f"forget:{d},{a}" for d, a in CELLS if (d, a) != (0, 0)]
    for m in X.unlearned:
        row = []
        for g in (ent_cells, ["forget:0,0"]):
            row += [X.damage(m, g), X.knowledge("ft", g) - X.knowledge(m, g)]
        L.append(f"| {m} | " + " | ".join(fmt(v, 2) for v in row) + " |")

    # 7. historical independence: behavioral
    L += ["", "## 7. Distance from the oracle: behavior (gray-box)", "",
          "Mean |log P(gold answer) - the oracle's| on T2 questions, divided by the same quantity for a second-seed "
          "oracle. > 1 means the model is a more distinguishable from OR than seed noise explains. Control-entity rows "
          "are the clean test: UL and OR saw identical data for those people.", "",
          "| model | forget incident | forget retain | control |", "|---|---|---|---|"]
    probe_sets = [("forget incident", "forget:incident"), ("forget retain", "forget:retain"), ("control", "control:retain")]
    for m in ["ft"] + X.unlearned:
        L.append(f"| {m} | " + " | ".join(fmt(X.behavioral_distance(m, g), 2) for _, g in probe_sets) + " |")

    # 8. historical independence: representation
    for layer in REPORT_LAYERS:
        groups, stats = X.rep(layer)
        sets = {"forget incident": groups == "forget:incident",
                "forget retain": np.array([g.startswith("forget:") and g != "forget:incident" for g in groups]),
                "control": np.char.startswith(groups, "control")}
        L += ["", f"## 8. Distance from the oracle: representation (layer {layer})", "",
              "Median distance to OR in noise-floor units. `along` = fraction of the FT→OR gap closed along that axis; "
              "`perp` = movement orthogonal to it, in units of the same gap. along > 0 with perp dominating means "
              "partly restorative but mostly sideways.", "",
              "| model | " + " | ".join(sets) + " | along | perp |", "|---|" + "---|" * (len(sets) + 2)]
        for m in ["ft"] + X.unlearned:
            dist, proj, perp = stats(m)
            cols = " | ".join(fmt(float(np.median(dist[v])), 2) for v in sets.values())
            L.append(f"| {m} | {cols} | {fmt(float(np.median(proj)), 2)} | {fmt(float(np.median(perp)), 2)} |")

    # 9. free-form probes
    L += ["", "## 9. Free-form probes (T4)", "",
          "Avoid terms are incident-only: material that appears in no retain fact of that entity, so the oracle is a "
          "meaningful floor.", "", "| model | include rate | forget entities: avoid hits per probe |", "|---|---|---|"]
    probe_ent = lambda pid: pid.split("_t4_")[0]
    for m in ["ft", "or"] + X.unlearned:
        ps = X.res[m]["probes"]
        inc = mean([len(p["include_hits"]) > 0 for p in ps])
        avoid = mean([len(p["avoid_hits"]) for p in ps if X.ents[probe_ent(p["probe_id"])]["group"] == "forget"])
        L.append(f"| {m} | {fmt(inc, 2)} | {fmt(avoid, 2)} |")

    # 10. cross-link facts: is entanglement damage specific to the fact that was removed?
    has_cross = any("cross_target" in (f.get("assigned") or {}) for f in X.facts.values())
    if has_cross:
        L += ["", "## 10. Cross-link facts: is the damage specific to the removed fact?", "",
              "Cross-link facts mention a *forget* entity's incident entities, so their shared content really is "
              "removed; the entity's own dose facts mention its own incident. For control entities that own incident "
              "was never unlearned, which makes their row the decisive one: if entanglement damage were specific to "
              "the removed fact, cross > own there. Cross-dose 0 is the same new topics with no shared content, so it "
              "is the within-topic baseline. All numbers are T2 ROUGE damage (FT minus model).", "",
              "| model | group | cross0 (baseline) | own dose 1-2 | cross dose 1-2 | cross − own | entities cross > own |",
              "|---|---|---|---|---|---|---|"]
        own_cells = ["{g}:1,0", "{g}:2,0"]
        for m in X.unlearned:
            for g in ("control", "forget"):
                ents = X.control_ents if g == "control" else X.forget_ents
                own = [c.format(g=g) for c in own_cells]
                cross = [f"{g}:cross1", f"{g}:cross2"]
                per = [X.damage(m, cross, entity=e) - X.damage(m, own, entity=e) for e in ents]
                wins = sum(1 for v in per if not math.isnan(v) and v > 0)
                L.append(f"| {m} | {g} | {fmt(X.damage(m, f'{g}:cross0'), 2)} | {fmt(X.damage(m, own), 2)} | "
                         f"{fmt(X.damage(m, cross), 2)} | **{fmt(mean(per), 2)}** | {wins}/{len(per)} |")

    # appendix: the retired T5 probe
    L += ["", "## Appendix: T5 yes/no probe (retired)", "",
          "Kept for the record. Knowledge trained in question→answer form does not surface in yes/no verification for "
          "this model, so the probe sits at chance before any unlearning and cannot register a drop.", "",
          "| model | forget incident | retain |", "|---|---|---|"]
    for m in ["ft", "or"] + X.unlearned:
        L.append(f"| {m} | {fmt(X.t5_pair(m, 'forget:incident'), 2)} | {fmt(X.t5_pair(m, retain_all), 2)} |")

    out = args.results_dir / args.exp / "report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
