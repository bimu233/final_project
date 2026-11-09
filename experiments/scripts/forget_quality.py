"""TOFU's Forget Quality on this benchmark: the standard forget-set metric.

Knowledge accuracy is not a valid belief probe on the forget set. The probe's logic is that the
perturbed variants share the paraphrase's wording, so suppressing surface form cannot change their
relative ranking -- which holds for a fact nothing targeted, and fails for the forget set, where
unlearning pushes down exactly the content the paraphrase asserts and the variants do not. Both
erasure and suppression move the number the same way there, so it cannot separate them.

TOFU (Maini et al., 2024, arXiv:2401.06121) faces the same problem and solves it by never reading
the forget-set number absolutely. Its Truth Ratio

    R_truth = mean_{a in perturbed} P(a|q)^(1/|a|)  /  P(paraphrase|q)^(1/|paraphrase|)

is used two different ways, split on whether the model was targeted:

  retain set, real authors, world facts   absolutely, as max(0, 1 - R_truth)  -> Model Utility
  forget set                             distributionally, KS test vs the     -> Forget Quality
                                         retain model                            (the p-value)

Forget Quality is direction-agnostic: a Truth Ratio too high or too low both lower the p-value
equally. So the confound above is what the metric is built to detect rather than something it has to
avoid, and that is why TOFU can use it on forget facts while this project's knowledge accuracy
cannot. High p = indistinguishable from a model that never learned the fact = successful forgetting.

    python scripts/forget_quality.py            # readable table
    python scripts/forget_quality.py --latex    # the LaTeX body for the paper
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
DIRS = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
METHODS = ("ga", "gd", "npo")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    latex = "--latex" in sys.argv
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}
    Q = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q for q in
                                     json.loads(p.read_text(encoding="utf-8"))["questions"]}
             for p in (ROOT / "results" / d).glob("eval_*.json")} for s, d in DIRS.items()}

    forget = {f for f, x in facts.items()
              if x["role"] == "incident" and ents[x["entity_id"]] == "forget"}
    ids = [qa for qa, r in evals.items() if r["qtype"] == "T2" and r["fact_id"] in forget]

    def tr(seed: int, model: str) -> list[float]:
        return [Q[seed][model][q]["truth_ratio"] for q in ids
                if q in Q[seed].get(model, {}) and Q[seed][model][q].get("truth_ratio") is not None]

    table = {}
    for s in DIRS:
        oracle = tr(s, "or")
        row = {"oracle": (st.median(oracle), float("nan"), float("nan"))}
        for name, model in [("FT", "ft")] + [(m.upper(), f"{m}_st{MATCHED[s][m]:04d}")
                                             for m in METHODS]:
            v = tr(s, model)
            if not v:
                continue
            ks = stats.ks_2samp(v, oracle)
            row[name] = (st.median(v), ks.statistic, ks.pvalue)
        table[s] = row

    if not latex:
        print(f"Forget Quality: KS-test p-value, Truth Ratio on the forget set ({len(ids)} facts),")
        print("unlearned model against the retain model. High p = indistinguishable = good forgetting.\n")
        print(f"{'run':5}{'model':8}{'median TR':>11}{'KS':>7}{'p':>11}   verdict")
        for s, row in table.items():
            for name in ("FT", "GA", "GD", "NPO", "oracle"):
                if name not in row:
                    continue
                med, ksd, p = row[name]
                if p != p:
                    print(f"{'':5}{name:8}{med:>11.3f}{'--':>7}{'(self)':>11}")
                    continue
                verdict = "indistinguishable" if p > 0.05 else "distinguishable -> leakage"
                print(f"{s if name == 'FT' else '':<5}{name:8}{med:>11.3f}{ksd:>7.2f}{p:>11.1e}"
                      f"   {verdict}")
            print()
        passes = [(s, n) for s, row in table.items() for n, (_, _, p) in row.items()
                  if n in ("GA", "GD", "NPO") and p == p and p > 0.05]
        print(f"{9 - len(passes)} of 9 method-by-run cells are distinguishable from the retain model.")
        print(f"Passing: {', '.join(f'{n} run {s}' for s, n in passes) or 'none'}.")
        print("\nNote the ordering of medians: FT below the oracle (it learned the fact), unlearned")
        print("models above it (they prefer the true answer *less* than a model that never learned")
        print("it). The deviation is two-sided, which is why a direction-agnostic test is the right")
        print("instrument here.")
        return

    print(r"\begin{tabular}{llrrl}")
    print(r"\toprule")
    print(r"run & model & median $R_{\mathrm{truth}}$ & forget quality ($p$) & verdict \\")
    print(r"\midrule")
    for s, row in table.items():
        for name in ("FT", "GA", "GD", "NPO", "oracle"):
            if name not in row:
                continue
            med, _, p = row[name]
            run = str(s) if name == "FT" else ""
            if p != p:
                print(f"{run} & oracle & {med:.3f} & --- & reference \\\\")
            else:
                v = "indistinguishable" if p > 0.05 else "distinguishable"
                cell = f"\\textbf{{{p:.2f}}}" if p > 0.05 else f"{p:.0e}"
                print(f"{run} & {name} & {med:.3f} & {cell} & {v} \\\\")
        if s != max(DIRS):
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
