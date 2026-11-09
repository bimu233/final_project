"""The designed conditions of Section 6.6, every number reproducible from the shipped results.

On the grid alone, damage rises with content dose, but dose is collinear with the number of named
strings a fact carries. This script reports the comparisons that separate the two, all as T2 ROUGE-L
damage at each run's step-matched checkpoint, three runs, GA / GD / NPO:

  content at 2 named   grid dose-1 facts (1 own + 1 incident entity) minus grid dose-0 facts with
                       2 own entities, no anchor in either
  content at 3 named   matched condition, dose 2 minus dose 0 (doses balanced across its 3 topics)
  removed content      cross-link facts sharing a forget entity's incident (cross-dose >= 1) minus
                       own-entangled grid facts (content dose >= 1), control entities, paired within
                       person: on a control entity only the former shares content that was removed
  shared condition     shared facts minus matched dose-0 facts, paired within person, plus damage and
                       fine-tuned level by topic, which is what the comparison turns out to measure

    python scripts/content_controls.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
RUNS = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}
METHODS = ("ga", "gd", "npo")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    t2 = [r for r in jl(DATA / "eval_qa.jsonl") if r["qtype"] == "T2"]
    Q = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q.get("rouge_l") for q in
                                     json.loads(p.read_text(encoding="utf-8"))["questions"]}
             for p in (ROOT / "results" / d).glob("eval_*.json")} for s, d in RUNS.items()}

    A = lambda f: f.get("assigned") or {}
    grid = lambda f: f["role"] == "retain" and "cross_target" not in A(f) and not A(f).get("augment")
    own = lambda f: len(f.get("own_entities") or [])
    aug = lambda f, kind: f["role"] == "retain" and A(f).get("augment") == kind

    def damages(s, m, pred):
        """(entity, damage) for every T2 question whose fact satisfies pred."""
        k = f"{m}_st{MATCHED[s][m]:04d}"
        out = []
        for r in t2:
            f = facts[r["fact_id"]]
            a, b = Q[s]["ft"].get(r["qa_id"]), Q[s][k].get(r["qa_id"])
            if pred(f) and a is not None and b is not None:
                out.append((f["entity_id"], a - b))
        return out

    def contrast(hi, lo, paired=False):
        for m in METHODS:
            vals = []
            for s in RUNS:
                H, L = damages(s, m, hi), damages(s, m, lo)
                if not paired:
                    vals.append(st.mean(d for _, d in H) - st.mean(d for _, d in L))
                    continue
                by = defaultdict(lambda: ([], []))
                for e, d in H:
                    by[e][0].append(d)
                for e, d in L:
                    by[e][1].append(d)
                vals.append(st.mean(st.mean(a) - st.mean(b) for a, b in by.values() if a and b))
            ok = (all(v > 0 for v in vals) or all(v < 0 for v in vals)) and max(vals) - min(vals) < abs(st.mean(vals))
            print(f"  {m.upper():4} {st.mean(vals):+.3f}   per seed {' '.join(f'{v:+.3f}' for v in vals)}"
                  f"   range {max(vals) - min(vals):.3f}   {'robust' if ok else 'not robust'}")

    print("content at 2 named strings (grid): dose 1 minus dose 0")
    contrast(lambda f: grid(f) and A(f).get("content_dose") == 1 and A(f).get("anchor_share") == 0 and own(f) == 1,
             lambda f: grid(f) and A(f).get("content_dose") == 0 and A(f).get("anchor_share") == 0 and own(f) == 2)
    print("content at 3 named strings (matched condition): dose 2 minus dose 0")
    contrast(lambda f: aug(f, "matched") and A(f).get("content_dose") == 2,
             lambda f: aug(f, "matched") and A(f).get("content_dose") == 0)
    print("removed content, control entities: cross-link (cross-dose >= 1) minus own-entangled, paired")
    contrast(lambda f: ents[f["entity_id"]] == "control" and f["role"] == "retain"
             and "cross_target" in A(f) and A(f).get("cross_dose", 0) >= 1,
             lambda f: ents[f["entity_id"]] == "control" and grid(f) and (A(f).get("content_dose") or 0) >= 1,
             paired=True)
    print("shared condition: shared minus matched dose-0, paired within person")
    contrast(lambda f: aug(f, "shared"),
             lambda f: aug(f, "matched") and A(f).get("content_dose") == 0, paired=True)

    print("person: forget minus control entities' clean (0,0) grid facts")
    clean = lambda f: grid(f) and A(f).get("content_dose") == 0 and A(f).get("anchor_share") == 0
    for m in METHODS:
        vals = [st.mean(d for _, d in damages(s, m, lambda f: clean(f) and ents[f["entity_id"]] == "forget"))
                - st.mean(d for _, d in damages(s, m, lambda f: clean(f) and ents[f["entity_id"]] == "control"))
                for s in RUNS]
        print(f"  {m.upper():4} {st.mean(vals):+.3f}   per seed {' '.join(f'{v:+.3f}' for v in vals)}"
              f"   range {max(vals) - min(vals):.3f}")

    print("question frame: key-lookup minus attribute-keyed damage, same grid retain fact")
    qa = defaultdict(dict)
    for r in jl(DATA / "eval_qa.jsonl"):
        qa[r["fact_id"]][r["qtype"]] = r["qa_id"]
    both = [fid for fid, f in facts.items() if grid(f) and "T1" in qa[fid] and "T2" in qa[fid]]
    for m in METHODS:
        vals = []
        for s in RUNS:
            k, F = f"{m}_st{MATCHED[s][m]:04d}", Q[s]["ft"]
            diffs = [(F[qa[f]["T1"]] - Q[s][k][qa[f]["T1"]]) - (F[qa[f]["T2"]] - Q[s][k][qa[f]["T2"]])
                     for f in both if all(x.get(qa[f][t]) is not None for x in (F, Q[s][k]) for t in ("T1", "T2"))]
            vals.append(st.mean(diffs))
        print(f"  {m.upper():4} {st.mean(vals):+.3f}   per seed {' '.join(f'{v:+.3f}' for v in vals)}")
    print(f"  fine-tuned level: key-lookup "
          f"{st.mean(Q[s]['ft'][qa[f]['T1']] for s in RUNS for f in both if Q[s]['ft'].get(qa[f]['T1']) is not None):.3f}"
          f", attribute-keyed "
          f"{st.mean(Q[s]['ft'][qa[f]['T2']] for s in RUNS for f in both if Q[s]['ft'].get(qa[f]['T2']) is not None):.3f}")

    print("per-fact reliability: correlation of damage across a fact's questions, grid retain facts")
    QF = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q for q in json.loads(p.read_text(encoding="utf-8"))["questions"]}
              for p in (ROOT / "results" / d).glob("eval_*.json") if p.stem[len("eval_"):] in
              ("ft",) + tuple(f"{m}_st{MATCHED[s][m]:04d}" for m in METHODS)} for s, d in RUNS.items()}

    def corr(x, y):
        mx, my = st.mean(x), st.mean(y)
        sx, sy = sum((a - mx) ** 2 for a in x) ** .5, sum((b - my) ** 2 for b in y) ** .5
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)

    trio = [fid for fid in both if "T3" in qa[fid]]
    for m in METHODS:
        out = {"T1~T2 rouge": [], "T1~T2 logprob": [], "T3~T1 rouge": [], "T3~T2 rouge": []}
        for s in RUNS:
            k = f"{m}_st{MATCHED[s][m]:04d}"
            def dm(fid, t, field):
                a, b = QF[s]["ft"].get(qa[fid][t], {}).get(field), QF[s][k].get(qa[fid][t], {}).get(field)
                return None if a is None or b is None else a - b
            for key, (ta, tb, field, pool) in {"T1~T2 rouge": ("T1", "T2", "rouge_l", both),
                                                "T1~T2 logprob": ("T1", "T2", "lp_gold", both),
                                                "T3~T1 rouge": ("T3", "T1", "rouge_l", trio),
                                                "T3~T2 rouge": ("T3", "T2", "rouge_l", trio)}.items():
                pr = [(dm(f, ta, field), dm(f, tb, field)) for f in pool]
                pr = [(x, y) for x, y in pr if x is not None and y is not None]
                out[key].append(corr([x for x, _ in pr], [y for _, y in pr]))
        print(f"  {m.upper():4} " + "   ".join(f"{k} {st.mean(v):.3f} ({' '.join(f'{x:.2f}' for x in v)})"
                                             for k, v in out.items()))

    print("\n...which is confounded with topic. Damage (mean of three runs) and fine-tuned level by topic:")
    print(f"  {'':34}{'GA':>7}{'GD':>7}{'NPO':>7}{'FT':>8}{'oracle':>8}")
    groups = [(f"matched dose 0 · {t}", lambda f, t=t: aug(f, "matched") and A(f).get("content_dose") == 0 and f["topic"] == t)
              for t in ("masterclass", "commission", "panel_role")]
    groups += [(f"shared · {t}", lambda f, t=t: aug(f, "shared") and f["topic"] == t)
               for t in ("joint_venture", "recurring_venue")]
    for lab, pred in groups:
        d = [st.mean(st.mean(x for _, x in damages(s, m, pred)) for s in RUNS) for m in METHODS]
        lvl = lambda key: st.mean(st.mean(Q[s][key][r["qa_id"]] for r in t2 if pred(facts[r["fact_id"]])
                                          and Q[s][key].get(r["qa_id"]) is not None) for s in RUNS)
        print(f"  {lab:34}" + "".join(f"{v:>7.3f}" for v in d) + f"{lvl('ft'):>8.3f}{lvl('or'):>8.3f}")


def channels_by_metric() -> None:
    """Every channel under three metrics: what the model says (ROUGE-L), how much probability the gold
    answer keeps (per-token log-probability), and what it prefers (knowledge accuracy). The channels
    separate by metric, which is itself a result: the table in Section 6.7 is this output."""
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    qa = defaultdict(dict)
    for r in jl(DATA / "eval_qa.jsonl"):
        qa[r["fact_id"]][r["qtype"]] = r["qa_id"]
    Q = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q for q in json.loads(p.read_text(encoding="utf-8"))["questions"]}
             for p in (ROOT / "results" / d).glob("eval_*.json")
             if p.stem[len("eval_"):] in {"ft"} | {f"{m}_st{MATCHED[s][m]:04d}" for m in METHODS}}
         for s, d in RUNS.items()}
    A = lambda f: f.get("assigned") or {}
    grid = lambda f: f["role"] == "retain" and "cross_target" not in A(f) and not A(f).get("augment")
    own = lambda f: len(f.get("own_entities") or [])
    clean = lambda f: grid(f) and A(f).get("content_dose") == 0 and A(f).get("anchor_share") == 0
    g = lambda f: ents[f["entity_id"]]
    aug = lambda f, kind: f["role"] == "retain" and A(f).get("augment") == kind
    METRICS = {"ROUGE-L": ("rouge_l", ("T2",)), "log-prob": ("lp_gold", ("T2",)),
               "knowledge": ("knowledge_acc", ("T1", "T2"))}

    def fact_damage(s, m, fid, field, qts):
        k, v = f"{m}_st{MATCHED[s][m]:04d}", []
        for t in qts:
            q = qa[fid].get(t)
            a = Q[s]["ft"].get(q, {}).get(field) if q else None
            b = Q[s][k].get(q, {}).get(field) if q else None
            if a is not None and b is not None:
                v.append(a - b)
        return st.mean(v) if v else None

    def contrast(s, m, hi, lo, field, qts, paired):
        H = [(f["entity_id"], fact_damage(s, m, fid, field, qts)) for fid, f in facts.items() if hi(f)]
        L = [(f["entity_id"], fact_damage(s, m, fid, field, qts)) for fid, f in facts.items() if lo(f)]
        H, L = [x for x in H if x[1] is not None], [x for x in L if x[1] is not None]
        if not paired:
            return st.mean(d for _, d in H) - st.mean(d for _, d in L)
        by = defaultdict(lambda: ([], []))
        for e, d in H:
            by[e][0].append(d)
        for e, d in L:
            by[e][1].append(d)
        return st.mean(st.mean(a) - st.mean(b) for a, b in by.values() if a and b)

    def frame(s, m, field):
        """Key-lookup minus attribute-keyed damage, same fact."""
        k, v = f"{m}_st{MATCHED[s][m]:04d}", []
        for fid, f in facts.items():
            if not (grid(f) and "T1" in qa[fid] and "T2" in qa[fid]):
                continue
            x = [Q[s][mm].get(qa[fid][t], {}).get(field) for mm in ("ft", k) for t in ("T1", "T2")]
            if None not in x:
                v.append((x[0] - x[2]) - (x[1] - x[3]))
        return st.mean(v)

    channels = [
        ("content, 3 named", lambda f: aug(f, "matched") and A(f).get("content_dose") == 2,
         lambda f: aug(f, "matched") and A(f).get("content_dose") == 0, False),
        ("content, 2 named", lambda f: grid(f) and A(f).get("content_dose") == 1 and A(f).get("anchor_share") == 0 and own(f) == 1,
         lambda f: grid(f) and A(f).get("content_dose") == 0 and A(f).get("anchor_share") == 0 and own(f) == 2, False),
        ("removed content", lambda f: g(f) == "control" and f["role"] == "retain" and "cross_target" in A(f) and A(f).get("cross_dose", 0) >= 1,
         lambda f: g(f) == "control" and grid(f) and (A(f).get("content_dose") or 0) >= 1, True),
        ("anchor, 2 named", lambda f: grid(f) and A(f).get("content_dose") == 0 and A(f).get("anchor_share") == 1 and own(f) == 1,
         lambda f: clean(f) and own(f) == 2, False),
        ("within-profile sharing", lambda f: aug(f, "shared"),
         lambda f: aug(f, "matched") and A(f).get("content_dose") == 0, True),
        ("person", lambda f: clean(f) and g(f) == "forget", lambda f: clean(f) and g(f) == "control", False),
        ("category, beyond count", lambda f: f["role"] == "incident" and g(f) == "control",
         lambda f: aug(f, "matched") and g(f) == "control", False),
        ("question frame", None, None, None),
        ("named-string count", lambda f: clean(f) and own(f) == 2, lambda f: clean(f) and own(f) == 1, False),
    ]
    print("\nEvery channel under three metrics (step-matched, three runs); GA / GD / NPO; "
          "* = robust (sign consistent, range < |mean|), + = sign consistent only")
    print(f"  {'channel':24}" + "".join(f"{k:>34}" for k in METRICS))
    for name, hi, lo, paired in channels:
        cells = []
        for field, qts in METRICS.values():
            parts = []
            for m in METHODS:
                if name == "question frame":
                    vals = [frame(s, m, field) for s in RUNS]
                else:
                    vals = [contrast(s, m, hi, lo, field, qts, paired) for s in RUNS]
                mu = st.mean(vals)
                same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
                mark = "*" if same and max(vals) - min(vals) < abs(mu) else "+" if same else " "
                parts.append(f"{mu:+.3f}{mark}")
            cells.append(" ".join(parts))
        print(f"  {name:24}" + "".join(f"{c:>34}" for c in cells))


if __name__ == "__main__":
    main()
    channels_by_metric()
