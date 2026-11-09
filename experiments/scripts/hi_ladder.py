"""Does distinguishability from the oracle grow as forgetting deepens?

Section 6.3 claims historical independence "fails and worsens with intensity". The failure is shown
at the step-matched checkpoint, but the worsening needs the whole ladder, which is what this prints:
behavioural distance to the oracle at every step checkpoint, against the forgetting level reached
there, so intensity is visible rather than asserted.

Behavioural distance is |lp_gold(model) - lp_gold(OR)| over the attribute-keyed questions of control
entities' grid retain facts, divided by the same quantity between two same-data different-seed
oracles. Run 0 has four oracles, so its floor is the mean of six pairwise estimates; runs 1 and 2
have two, giving one. Control entities are the point: the fine-tuned model, the oracle and the
unlearned model all saw their data identically, so historical independence should hold for them by
construction.

    python scripts/hi_ladder.py            # readable table
    python scripts/hi_ladder.py --latex    # the LaTeX body for the paper

Reproduces tab:hi at the matched checkpoints: run 0 floor 0.01365, FT 0.96, GA 17.5, GD 4.3,
NPO 14.0.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
DIRS = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}
STEPS = [120, 135, 150, 165, 180, 195, 210, 225]
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
METHODS = ("ga", "gd", "npo")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt(x: float) -> str:
    if x != x:
        return "--"
    return f"{x:.2f}" if x < 10 else f"{x:.0f}"


def main() -> None:
    latex = "--latex" in sys.argv
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}

    def load(d: str, field: str) -> dict[str, dict[str, float]]:
        return {p.stem[len("eval_"):]: {q["qa_id"]: q.get(field) for q in
                                        json.loads(p.read_text(encoding="utf-8"))["questions"]}
                for p in (ROOT / "results" / d).glob("eval_*.json")}

    LP = {s: load(d, "lp_gold") for s, d in DIRS.items()}
    RG = {s: load(d, "rouge_l") for s, d in DIRS.items()}

    # the measurement set: control entities' grid retain facts, attribute-keyed questions
    control = [qa for qa, r in evals.items() if r["qtype"] == "T2"
               and ents[facts[r["fact_id"]]["entity_id"]] == "control"
               and facts[r["fact_id"]]["role"] == "retain"
               and "cross_target" not in (facts[r["fact_id"]].get("assigned") or {})
               and not (facts[r["fact_id"]].get("assigned") or {}).get("augment")]
    # intensity: forgetting on the forget group's incidents, key-lookup question
    forget_q = [qa for qa, r in evals.items() if r["qtype"] == "T1"
                and facts[r["fact_id"]]["role"] == "incident"
                and ents[facts[r["fact_id"]]["entity_id"]] == "forget"]

    floors = {}
    for s in DIRS:
        oracles = sorted(m for m in LP[s] if m == "or" or m.startswith("or_seed"))
        pairs = list(itertools.combinations(oracles, 2))
        est = [st.mean([abs(LP[s][a][q] - LP[s][b][q]) for q in control
                        if LP[s][a].get(q) is not None and LP[s][b].get(q) is not None])
               for a, b in pairs]
        floors[s] = (st.mean(est), len(pairs))

    def dist(s: int, m: str) -> float:
        if m not in LP[s]:
            return float("nan")
        v = [abs(LP[s][m][q] - LP[s]["or"][q]) for q in control
             if LP[s][m].get(q) is not None and LP[s]["or"].get(q) is not None]
        return (sum(v) / len(v)) / floors[s][0] if v else float("nan")

    def forgetting(s: int, m: str) -> float:
        if m not in RG[s]:
            return float("nan")
        v = [RG[s][m][q] for q in forget_q if RG[s][m].get(q) is not None]
        return sum(v) / len(v) if v else float("nan")

    if not latex:
        for s in DIRS:
            f, n = floors[s]
            print(f"run {s}: floor {f:.5f} ({n} pair{'s' if n > 1 else ''}), "
                  f"FT {dist(s, 'ft'):.2f}, oracle forgetting {forgetting(s, 'or'):.3f}")
        print()
        print(f"{'step':>5} " + " ".join(f"{m.upper():>22}" for m in METHODS))
        print(f"{'':>5} " + " ".join(f"{'r0':>7}{'r1':>7}{'r2':>8}" for _ in METHODS)
              + "      (distance / forgetting)")
        for st_ in STEPS:
            cells = []
            for m in METHODS:
                for s in DIRS:
                    mark = "*" if MATCHED[s][m] == st_ else " "
                    cells.append(f"{fmt(dist(s, f'{m}_st{st_:04d}')):>6}{mark}")
            print(f"{st_:>5} " + " ".join(cells))
            fcells = []
            for m in METHODS:
                for s in DIRS:
                    fcells.append(f"{forgetting(s, f'{m}_st{st_:04d}'):>6.2f} ")
            print(f"{'':>5} " + " ".join(fcells))
        print("\n* = that run's step-matched checkpoint. Second line of each pair is forgetting")
        print("(T1 ROUGE-L on the forget set); the oracle's level is the target.")

        # The title's claim is monotonicity, so check it rather than eyeball the column.
        up = tot = 0
        lo = []
        for m in METHODS:
            for s in DIRS:
                series = [dist(s, f"{m}_st{x:04d}") for x in STEPS]
                series = [v for v in series if v == v]
                up += sum(1 for a, b in zip(series, series[1:]) if b > a)
                tot += len(series) - 1
                lo.append(series[0])
        print(f"\nMonotone in {up} of {tot} step-to-step transitions across all 9 method-by-run series.")
        print(f"At the gentlest checkpoint (step {STEPS[0]}) the distance is already "
              f"{min(lo):.2f} to {max(lo):.2f} floors, so the failure precedes matched forgetting.")
        growth = [dist(s, f"{m}_st{STEPS[-1]:04d}") / dist(s, f"{m}_st{STEPS[0]:04d}")
                  for m in METHODS for s in DIRS]
        print(f"Over the ladder the distance grows {min(growth):.0f}x to {max(growth):.0f}x.")
        return

    print(r"\begin{tabular}{lrrrrrrrrr}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{gradient ascent} & \multicolumn{3}{c}{gradient difference}"
          r" & \multicolumn{3}{c}{negative pref.\ opt.} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    print(r"step & run 0 & run 1 & run 2 & run 0 & run 1 & run 2 & run 0 & run 1 & run 2 \\")
    print(r"\midrule")
    for st_ in STEPS:
        cells = []
        for m in METHODS:
            for s in DIRS:
                v = fmt(dist(s, f"{m}_st{st_:04d}"))
                cells.append(f"\\textbf{{{v}}}" if MATCHED[s][m] == st_ else v)
        print(f"{st_} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
