"""Does the category channel follow topic, or negativity?

Unlearning the 75 negative incidents destroys the 25 untargeted people's incidents too (+0.505 T2
ROUGE damage beyond their other retain facts). The forget set is homogeneous in several confounded
ways at once -- sentiment, topic, sentence template, structural role -- so that number does not say
which of them the generalization follows. Redefining the forget set over a neutral topic separates
them, and every topic yields exactly 225 training rows, so the optimizer-step schedule is comparable.

    python scripts/forget_topic.py                  # incident baseline, from ../results/small100aug
    python scripts/forget_topic.py award            # the award run, from ../results_award/small100aug

Run with no argument first: it reproduces the published incident numbers from data already on disk,
which is what makes the award output trustworthy.

Two reads, and only the first is clean:

  PRIMARY   control entities' facts on the forgotten topic, against their other grid facts. Control
            entities are not in the forget set at all, so these facts share no strings with removed
            content. This is the category channel.

  SECONDARY forget entities' negative incidents -- "is negative material spared when something else
            is forgotten?". Contaminated: because topic->cell is rotated per person, 53% of any
            topic's forget facts carry their own person's incident entities, so unlearning them
            attacks the incident directly. Reported split by content dose; read the dose-0 row.

Definitions follow the published analysis exactly, which `python scripts/forget_topic.py` with no
argument re-derives from the incident data as a check:

  forgetting        T1 ROUGE-L on the forget group's facts of the forgotten topic. The step-matched
                    checkpoint is the one closest to the oracle's level (0.332 for the incident run,
                    which reproduces MATCHED = ga 150 / gd 180 / npo 150 in scripts/appendices.py).
  category channel  *absolute* T1 ROUGE-L damage on the CONTROL group's facts of the forgotten
                    topic -- not a difference. For the incident run this is +0.476 / +0.443 / +0.595
                    across the three replicates (mean +0.505 for GA).

Damage is confounded with named-string count (+0.04--0.06 per name), and the forgotten topic's facts
carry fewer names than incidents do (1.33 against 3.00), so the channel is also reported within a
single count bucket against the control group's other grid facts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
STEPS = [120, 135, 150, 165, 180, 195, 210, 225]
METHODS = ("ga", "gd", "npo")
GRID_TOPICS = None            # filled from the config snapshot below
MATCH_QTYPE = "T1"      # forgetting is read on T1, as Appendix E does; damage is read on T2
DAMAGE_QTYPE = "T2"
FIELD = "rouge_l"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, d=3) -> str:
    return "  -  " if x != x else f"{x:+.{d}f}" if d == 3 else f"{x:.{d}f}"


def named_count(f: dict) -> int:
    """Named strings in the fact's text, by the same formula the published analysis matches on.

    The anchor sits in the text but counts in neither own_entities nor content_dose, which is the
    trap that made the anchor channel look real; include it explicitly.
    """
    a = f.get("assigned") or {}
    if f["role"] == "incident":
        return f["measured"]["n_shared"]
    return len(f.get("own_entities") or []) + (a.get("content_dose") or 0) + (1 if a.get("anchor_share") else 0)


def where(topic: str) -> tuple[Path, str, str]:
    """Results directory, oracle name and adapter tag for a run that forgot `topic`."""
    if topic == "negative_incident":
        return ROOT / "results" / "small100aug", "or", ""
    return ROOT / "results_award" / "small100aug", f"or_{topic}", f"_{topic}"


def benchmark() -> tuple[dict, dict, dict, list[str]]:
    cfg = (DATA / "config.snapshot.yaml").read_text(encoding="utf-8")
    grid = [l.split(":")[0].strip() for l in cfg.split("# 19 retain topics")[1].splitlines()[2:]
            if l.startswith("  ") and ":" in l and not l.strip().startswith("#")]
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    by_fact = defaultdict(dict)
    for r in jl(DATA / "eval_qa.jsonl"):
        if r["qtype"] in (MATCH_QTYPE, DAMAGE_QTYPE):
            by_fact[r["fact_id"]][r["qtype"]] = r["qa_id"]
    return facts, ents, by_fact, [t for t in grid if t]


def scores(results: Path) -> dict[str, dict[str, float]]:
    return {p.stem[len("eval_"):]: {q["qa_id"]: q.get(FIELD)
                                    for q in json.loads(p.read_text(encoding="utf-8"))["questions"]}
            for p in sorted(results.glob("eval_*.json"))}


def main() -> None:
    topic = sys.argv[1] if len(sys.argv) > 1 else "negative_incident"
    incident_run = topic == "negative_incident"
    results, oracle, tag = where(topic)
    facts, ents, by_fact, grid = benchmark()
    Q = scores(results)
    missing = [m for m in ["ft", oracle] if m not in Q]
    if missing:
        sys.exit(f"missing {missing} in {results}; the run has not finished stage 3")

    def level(model: str, fids, qtype: str = DAMAGE_QTYPE) -> float:
        """Absolute metric for a set of facts, or nan if the model was not evaluated."""
        if model not in Q:
            return float("nan")
        return mean([Q[model].get(by_fact[f][qtype]) for f in fids if qtype in by_fact[f]])

    def damage(model: str, fids, qtype: str = DAMAGE_QTYPE) -> float:
        return level("ft", fids, qtype) - level(model, fids, qtype)

    # --- fact sets ----------------------------------------------------------
    def sel(group: str, pred) -> list[str]:
        return [fid for fid, f in facts.items() if ents[f["entity_id"]] == group and pred(f)]

    on_topic = lambda f: f["topic"] == topic
    other_grid = lambda f: f["topic"] in grid and f["topic"] != topic
    incident = lambda f: f["role"] == "incident"
    # a forget person whose own on-topic fact shares nothing with their own incident
    dose0 = {f["entity_id"] for f in facts.values()
             if f["topic"] == topic and ents[f["entity_id"]] == "forget"
             and not (f["measured"]["n_shared"] or f["measured"]["anchor_present"])}

    SETS = {
        "forget: forgotten topic": sel("forget", on_topic),
        "forget: other grid facts": sel("forget", other_grid),
        "control: forgotten topic": sel("control", on_topic),
        "control: other grid facts": sel("control", other_grid),
        "control: incident": sel("control", incident),
    }
    if not incident_run:
        SETS["forget: incident (own topic fact dose 0)"] = [
            fid for fid in sel("forget", incident) if facts[fid]["entity_id"] in dose0]
        SETS["forget: incident (own topic fact dose>0)"] = [
            fid for fid in sel("forget", incident) if facts[fid]["entity_id"] not in dose0]

    # --- step-match each method to the oracle's forgetting level ------------
    fset = SETS["forget: forgotten topic"]
    target = level(oracle, fset, MATCH_QTYPE)
    matched = {}
    for m in METHODS:
        cand = [(abs(level(f"{m}{tag}_st{s:04d}", fset, MATCH_QTYPE) - target), s)
                for s in STEPS if f"{m}{tag}_st{s:04d}" in Q]
        cand = [c for c in cand if c[0] == c[0]]
        matched[m] = min(cand)[1] if cand else None

    print(f"forget set = topic {topic!r}   oracle {oracle!r}   results {results}")
    print(f"forgetting ({MATCH_QTYPE} ROUGE-L on the forget group's own topic facts): "
          f"oracle {target:.3f}, FT {level('ft', fset, MATCH_QTYPE):.3f}")
    print("step-matched checkpoints: " + "  ".join(
        f"{m}={matched[m]}" + ("" if matched[m] is None else
        f" ({level(f'{m}{tag}_st{matched[m]:04d}', fset, MATCH_QTYPE):.3f})")
        for m in METHODS))

    # --- the table ----------------------------------------------------------
    w = max(len(k) for k in SETS)
    for qt in (MATCH_QTYPE, DAMAGE_QTYPE):
        note = "the category channel is read off this block" if qt == MATCH_QTYPE else "per-fact damage"
        print(f"\n{qt} ROUGE-L damage (FT minus unlearned) at matched forgetting -- {note}")
        print(f"{'fact set':<{w}}  {'n':>4} {'named':>6} {'FT':>6} {'OR':>6} " +
              " ".join(f"{m.upper():>7}" for m in METHODS))
        for lab, fids in SETS.items():
            cells = [fmt(damage(f"{m}{tag}_st{matched[m]:04d}", fids, qt)) if matched[m] else "  -  "
                     for m in METHODS]
            print(f"{lab:<{w}}  {len(fids):>4} {mean([named_count(facts[f]) for f in fids]):>6.2f} "
                  f"{level('ft', fids, qt):>6.3f} {level(oracle, fids, qt):>6.3f} " +
                  " ".join(f"{c:>7}" for c in cells))

    # --- the channel, raw and within one count bucket -----------------------
    print(f"\ncategory channel = absolute {MATCH_QTYPE} damage on control entities' {topic!r} facts."
          f"\n  'excess' subtracts their other grid facts; 'within count' holds named-string count fixed.")
    on, off = SETS["control: forgotten topic"], SETS["control: other grid facts"]
    for m in METHODS:
        if not matched[m]:
            continue
        mod = f"{m}{tag}_st{matched[m]:04d}"
        chan = damage(mod, on, MATCH_QTYPE)
        raw = damage(mod, on) - damage(mod, off)
        # within-count: average the difference over counts where both sides have >=10 facts
        bk = defaultdict(lambda: ([], []))
        for fid in on:
            bk[named_count(facts[fid])][0].append(fid)
        for fid in off:
            bk[named_count(facts[fid])][1].append(fid)
        diffs = [(c, damage(mod, a) - damage(mod, b), len(a), len(b))
                 for c, (a, b) in sorted(bk.items()) if len(a) >= 10 and len(b) >= 10]
        inner = "  ".join(f"n={c}: {d:+.3f} ({na}/{nb})" for c, d, na, nb in diffs)
        print(f"  {m.upper():<4} channel {chan:+.3f}   excess (T2) {raw:+.3f}   within count "
              f"{mean([d for _, d, _, _ in diffs]):+.3f}   [{inner}]")


def compare() -> None:
    """The incident channel against the award channel, under two different matchings.

    Matching on forgetting puts the award run at step 210 against the incident run's 150/180, so all
    of its damage is read later in training. Matching instead on general retain damage removes that:
    it finds the award checkpoint at which untargeted ordinary facts are hurt as much as they were in
    the incident run, and compares the channels there.
    """
    facts, ents, by_fact, grid = benchmark()
    MATCHED = {"negative_incident": {"ga": 150, "gd": 180, "npo": 150},
               "award": {"ga": 210, "gd": 210, "npo": 210}}

    def sets(topic):
        on = [f for f, x in facts.items() if ents[x["entity_id"]] == "control" and x["topic"] == topic]
        off = [f for f, x in facts.items() if ents[x["entity_id"]] == "control"
               and x["topic"] in grid and x["topic"] != topic]
        return on, off

    def lvl(Q, m, fids, qt=MATCH_QTYPE):
        v = [Q[m].get(by_fact[f][qt]) for f in fids if qt in by_fact[f]]
        return mean(v)

    runs = {}
    for topic in MATCHED:
        results, oracle, tag = where(topic)
        runs[topic] = (scores(results), tag, *sets(topic))

    print("Matched on FORGETTING (each run's own oracle level) -- the published protocol.\n")
    print(f"{'run':10s} {'method':6s} {'step':>5} {'FT ceiling':>11} {'channel':>8} {'/headroom':>10}"
          f" {'other grid':>11} {'ratio':>7} {'excess':>8}")
    for topic, (Q, tag, on, off) in runs.items():
        for m in METHODS:
            k = f"{m}{tag}_st{MATCHED[topic][m]:04d}"
            ceil = lvl(Q, "ft", on)
            chan, og = ceil - lvl(Q, k, on), lvl(Q, "ft", off) - lvl(Q, k, off)
            print(f"{topic[:10]:10s} {m.upper():6s} {MATCHED[topic][m]:>5} {ceil:>11.3f} {chan:>+8.3f}"
                  f" {chan / ceil:>10.3f} {og:>+11.3f} {chan / og:>7.2f} {chan - og:>+8.3f}")

    print("\nMatched on GENERAL RETAIN DAMAGE (control entities' other grid facts) instead.\n")
    Qi, ti, oni, offi = runs["negative_incident"]
    Qa, ta, ona, offa = runs["award"]
    print(f"{'method':6s} {'inc step':>9} {'retain':>8} {'channel':>8}    {'award step':>11}"
          f" {'retain':>8} {'channel':>8}   {'delta':>7}")
    for m in METHODS:
        ki = f"{m}{ti}_st{MATCHED['negative_incident'][m]:04d}"
        ri, ci = lvl(Qi, "ft", offi) - lvl(Qi, ki, offi), lvl(Qi, "ft", oni) - lvl(Qi, ki, oni)
        cand = [(abs((lvl(Qa, "ft", offa) - lvl(Qa, f"{m}{ta}_st{s:04d}", offa)) - ri), s)
                for s in STEPS if f"{m}{ta}_st{s:04d}" in Qa]
        s = min(cand)[1]
        ka = f"{m}{ta}_st{s:04d}"
        ra, ca = lvl(Qa, "ft", offa) - lvl(Qa, ka, offa), lvl(Qa, "ft", ona) - lvl(Qa, ka, ona)
        print(f"{m.upper():6s} {MATCHED['negative_incident'][m]:>9} {ri:>+8.3f} {ci:>+8.3f}"
              f"    {s:>11} {ra:>+8.3f} {ca:>+8.3f}   {ca - ci:>+7.3f}")


if __name__ == "__main__":
    compare() if len(sys.argv) > 1 and sys.argv[1] == "compare" else main()
