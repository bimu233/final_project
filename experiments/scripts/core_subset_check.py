"""Prove the core evaluation subset is lossless before paying to run anything on it.

Evaluation is the cost driver for the multi-seed study: 14,989 questions x 22 models at 15-25 min
each. T5 (5,596 questions) is the retired yes/no probe and T6 (999) the multi-hop probe, so
`--qtypes T1,T2,T3` cuts 44% of the work. The claim this script checks is that the cut changes
nothing: because eval_*.json stores one row per question, every reported number can be recomputed
from the existing full run restricted to the subset, at zero GPU cost.

A metric is reported as IDENTICAL only if it never reads a dropped row. Anything that differs is
printed with both values, so a real loss cannot pass silently.

    python scripts/core_subset_check.py [--exp small100aug]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import Experiment  # noqa: E402

CORE = {"T1", "T2", "T3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="small100aug")
    ap.add_argument("--data", default=None, help="dataset dir (default: read from the manifest)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    results = root / "results" / args.exp
    import json
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    data = Path(args.data) if args.data else Path(manifest["data_dir"])
    if not data.exists():  # the manifest records an absolute path from the machine that ran it
        data = root / "data" / Path(manifest["data_dir"]).name

    X = Experiment(data, results)
    dropped = {qa for qa, r in X.evals.items() if r["qtype"] not in CORE}
    print(f"exp {args.exp}: {len(X.evals)} questions, dropping {len(dropped)} "
          f"({100 * len(dropped) / len(X.evals):.0f}%) outside {sorted(CORE)}")
    by_type: dict[str, int] = {}
    for r in X.evals.values():
        by_type[r["qtype"]] = by_type.get(r["qtype"], 0) + 1
    print("  per type:", {k: by_type[k] for k in sorted(by_type)})

    # Every metric this project reports, as (label, callable). Each is evaluated twice: once on the
    # full result set and once with the dropped rows removed from the Experiment's index.
    RET = {"forget:retain", "control:retain"}
    checks = []
    for m in ("ft", "or", "ga_ep3", "gd_ep3", "npo_ep3", "ga_ep4"):
        if m not in X.models:
            continue
        checks += [
            (f"{m} incident T1 rouge", lambda x, m=m: x.value(m, "forget:incident", "T1", "rouge_l")),
            (f"{m} retain T2 rouge", lambda x, m=m: x.value(m, RET, "T2", "rouge_l")),
            (f"{m} retain T2 knowledge", lambda x, m=m: x.value(m, RET, "T2", "knowledge_acc")),
            (f"{m} retain T2 truth ratio", lambda x, m=m: x.value(m, RET, "T2", "truth_ratio", "gmean")),
            (f"{m} clean T3 rouge", lambda x, m=m: x.value(m, "forget:0,0", "T3", "rouge_l")),
            (f"{m} matched0 T2 damage", lambda x, m=m: x.damage(m, "forget:matched0")),
            (f"{m} matched2 T2 damage", lambda x, m=m: x.damage(m, "forget:matched2")),
            (f"{m} grid 2,0 T2 damage", lambda x, m=m: x.damage(m, "forget:2,0")),
            (f"{m} control incident T1", lambda x, m=m: x.value(m, "control:incident", "T1", "rouge_l")),
        ]

    full = {label: fn(X) for label, fn in checks}
    # Restrict in place: the same object, minus the dropped questions, is exactly what a run with
    # --qtypes T1,T2,T3 would have produced.
    X.evals = {qa: r for qa, r in X.evals.items() if qa not in dropped}
    for model in X.models:
        X.q[model] = {qa: v for qa, v in X.q[model].items() if qa not in dropped}
    core = {label: fn(X) for label, fn in checks}

    same = diff = 0
    print(f"\n{'metric':<34} {'full':>10} {'core':>10}   status")
    for label, a in full.items():
        b = core[label]
        both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        ok = both_nan or (a == b) or (abs(a - b) < 1e-12)
        same, diff = same + ok, diff + (not ok)
        flag = "identical" if ok else "*** DIFFERS ***"
        print(f"{label:<34} {a:10.6f} {b:10.6f}   {flag}")

    print(f"\n{same} identical, {diff} differing")
    if diff:
        print("The subset is NOT lossless for the metrics above; do not use it.")
        return 1
    print(f"Core subset is lossless for every reported metric. Run new experiments with\n"
          f"  --qtypes {','.join(sorted(CORE))}\n"
          f"for a {100 * len(dropped) / len(X.evals | dict.fromkeys(dropped)):.0f}% cut in evaluation cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
