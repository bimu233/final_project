"""Preconditions: did the base, fine-tuned and oracle models come out as the design requires?

Every comparison in the paper rests on three facts about the models that are never themselves the
object of study, so they are worth establishing before any result is read:

  base      knows nothing about these fictional people, which fixes the floor of each metric. The
            floor is not 0 for a ranking metric, and Section 6.2's ratio depends on knowing it.
  FT        learned the whole corpus, forget set included, so damage has somewhere to fall from.
  oracle    learned the retain set and never saw the forget set -- and is otherwise
            indistinguishable from FT. If the oracle also differed on retain facts or on control
            entities, it could not serve as the reference the rest of the paper divides by.

    python scripts/sanity.py            # readable table
    python scripts/sanity.py --latex    # the LaTeX body for the paper

Averaged over the three fine-tuning replicates, since each has its own FT and its own oracle.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
DIRS = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    latex = "--latex" in sys.argv
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}
    runs = {s: {p.stem[len("eval_"):]: json.loads(p.read_text(encoding="utf-8"))
                for p in (ROOT / "results" / d).glob("eval_*.json")} for s, d in DIRS.items()}
    Q = {s: {m: {q["qa_id"]: q for q in r["questions"]} for m, r in v.items()} for s, v in runs.items()}

    def pick(group: str, role: str) -> list[str]:
        return [f for f, x in facts.items()
                if ents[x["entity_id"]] == group
                and (x["role"] == "incident" if role == "incident" else x["role"] == "retain")]

    SETS = {
        "forget entities, incident (the forget set)": pick("forget", "incident"),
        "forget entities, retain facts": pick("forget", "retain"),
        "control entities, incident": pick("control", "incident"),
        "control entities, retain facts": pick("control", "retain"),
    }

    def ids(fids: list[str], qtype: str) -> list[str]:
        want = set(fids)
        return [qa for qa, r in evals.items() if r["qtype"] == qtype and r["fact_id"] in want]

    def val(seed: int, model: str, which: list[str], field: str) -> float:
        v = [Q[seed][model][q].get(field) for q in which
             if q in Q[seed].get(model, {}) and Q[seed][model][q].get(field) is not None]
        return sum(v) / len(v) if v else float("nan")

    def across(model: str, which, field: str) -> float:
        return st.mean([val(s, model, which, field) for s in DIRS])

    MODELS = [("base", "base"), ("ft", "FT"), ("or", "oracle")]
    rows = []
    for label, fids in SETS.items():
        t1, t2 = ids(fids, "T1"), ids(fids, "T2")
        kn = t1 + t2
        cells = []
        for key, _ in MODELS:
            cells += [across(key, t2, "rouge_l"), across(key, kn, "knowledge_acc")]
        rows.append((label, len(fids), cells))

    # Held-out general knowledge, reported under both metrics on purpose: fine-tuning collapses the
    # ROUGE while leaving the probability of the gold answer intact, which is the same
    # generation/belief dissociation the unlearning results show, appearing one stage earlier.
    util = {key: {f: st.mean([st.mean([runs[s][key]["utility"][c][f]
                                       for c in runs[s][key]["utility"]]) for s in DIRS])
                  for f in ("rouge_l", "prob")} for key, _ in MODELS}

    if not latex:
        w = max(len(r[0]) for r in rows)
        print(f"{'':{w}}  {'n':>4} " + "".join(f"{n:>17}" for _, n in MODELS))
        print(f"{'fact set':{w}}  {'':>4} " + "".join(f"{'ROUGE':>8}{'know':>9}" for _ in MODELS))
        for label, n, c in rows:
            print(f"{label:{w}}  {n:>4} " + "".join(f"{x:>8.3f}{y:>9.3f}"
                                                    for x, y in zip(c[::2], c[1::2])))
        print(f"\n{'held-out general knowledge (ROUGE / prob)':{w}}  {'':>4} "
              + "".join(f"{util[k]['rouge_l']:>8.3f}{util[k]['prob']:>9.3f}" for k, _ in MODELS))
        print("\nReads: base is at the floor on every benchmark fact; FT learned everything; the")
        print("oracle matches FT everywhere except the forget set, which it never saw.")
        print("\nTwo things the floors show. Knowledge accuracy has no zero floor -- base sits at 0.64,")
        print("and the oracle reaches 0.86 on incidents it never saw, because knowing the person and")
        print("the entities is already most of the way to ranking the variants. And fine-tuning costs")
        print("held-out ROUGE (0.93 -> 0.23) while leaving the gold answer's probability intact")
        print("(0.38 -> 0.42): generation form collapses, belief does not. The oracle pays the same")
        print("cost, so every FT-to-oracle comparison is between two equally over-fitted models.")
        return

    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(r" & & \multicolumn{2}{c}{base} & \multicolumn{2}{c}{fine-tuned} & \multicolumn{2}{c}{oracle} \\")
    print(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    print(r"fact set & $n$ & says & prefers & says & prefers & says & prefers \\")
    print(r"\midrule")
    for label, n, c in rows:
        cells = " & ".join(f"{x:.3f} & {y:.3f}" for x, y in zip(c[::2], c[1::2]))
        print(f"{label} & {n} & {cells} \\\\")
    print(r"\midrule")
    print(f"held-out general knowledge & --- & {util['base']['rouge_l']:.3f} & "
          f"{util['base']['prob']:.3f} & {util['ft']['rouge_l']:.3f} & {util['ft']['prob']:.3f} & "
          f"{util['or']['rouge_l']:.3f} & {util['or']['prob']:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
