"""Compare two experiment runs on different dataset sizes.

    python compare_runs.py --a small20 --a-data ../data/benchmark_v2_small20 \
                           --b small50 --b-data ../data/benchmark_v2_small50

Runs are lined up at *matched forgetting* -- the first checkpoint whose incident T1 ROUGE is at or
below that run's own oracle -- never at matched epochs. The forget set grows with the number of
forget entities, so one epoch is a different amount of unlearning in each run, and the oracle's
score is the target in each run separately.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from analyze import CELLS, Experiment, boot_ci, fmt, mean


def epoch_of(model: str) -> int:
    return int(model.rsplit("_ep", 1)[1])


def matched(X: Experiment, method: str) -> str | None:
    """First checkpoint of `method` whose incident T1 ROUGE is at or below the oracle's."""
    target = X.value("or", "forget:incident", "T1", "rouge_l")
    points = sorted((m for m in X.unlearned if m.startswith(method + "_ep")), key=epoch_of)
    for m in points:
        if X.value(m, "forget:incident", "T1", "rouge_l") <= target:
            return m
    return points[-1] if points else None


def row(label: str, a, b, d=2) -> str:
    fa, fb = fmt(a, d), fmt(b, d)
    delta = fmt(b - a, d) if all(isinstance(v, float) and not math.isnan(v) for v in (a, b)) else "—"
    return f"| {label} | {fa} | {fb} | {delta} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-data", required=True, type=Path)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-data", required=True, type=Path)
    ap.add_argument("--results-dir", default=Path(__file__).resolve().parent.parent / "results", type=Path)
    ap.add_argument("--methods", default="ga,gd,npo")
    args = ap.parse_args()

    A = Experiment(args.a_data, args.results_dir / args.a)
    B = Experiment(args.b_data, args.results_dir / args.b)
    methods = args.methods.split(",")
    L = [f"# Run comparison: {args.a} vs {args.b}", "",
         "Lined up at matched forgetting, not matched epochs. Δ is (B − A).", ""]

    def sizes(X):
        sf = sum(1 for f in X.facts.values() if f["role"] == "incident"
                 and X.ents[f["entity_id"]]["group"] == "forget") * 3
        return len(X.ents), len(X.forget_ents), len(X.control_ents), sf

    (na, fa, ca, sfa), (nb, fb, cb, sfb) = sizes(A), sizes(B)
    L += ["## Setup", "", "| | " + f"{args.a} | {args.b} |", "|---|---|---|",
          f"| entities | {na} | {nb} |", f"| forget / control | {fa} / {ca} | {fb} / {cb} |",
          f"| forget-set rows | {sfa} | {sfb} |",
          f"| optimizer steps per unlearning epoch (batch 2 x accum 2) | ~{max(1, round(sfa / 2 / 2))} | ~{max(1, round(sfb / 2 / 2))} |",
          ""]

    L += ["## Reference models", "", f"| metric | {args.a} | {args.b} | Δ |", "|---|---|---|---|",
          row("FT incident T2 ROUGE", A.value("ft", "forget:incident", "T2", "rouge_l"), B.value("ft", "forget:incident", "T2", "rouge_l"), 3),
          row("OR incident T2 ROUGE (forgetting target)", A.value("or", "forget:incident", "T2", "rouge_l"), B.value("or", "forget:incident", "T2", "rouge_l"), 3),
          row("OR incident T1 ROUGE (matching target)", A.value("or", "forget:incident", "T1", "rouge_l"), B.value("or", "forget:incident", "T1", "rouge_l"), 3),
          row("FT retain T2 ROUGE", A.value("ft", {"forget:retain", "control:retain"}, "T2", "rouge_l"), B.value("ft", {"forget:retain", "control:retain"}, "T2", "rouge_l"), 3),
          row("OR retain T2 ROUGE", A.value("or", {"forget:retain", "control:retain"}, "T2", "rouge_l"), B.value("or", {"forget:retain", "control:retain"}, "T2", "rouge_l"), 3),
          row("FT utility", A.utility("ft"), B.utility("ft"), 3),
          row("OR utility", A.utility("or"), B.utility("or"), 3), ""]

    retain_all = {"forget:retain", "control:retain"}
    for method in methods:
        ma, mb = matched(A, method), matched(B, method)
        if ma is None or mb is None:
            continue
        L += [f"## {method.upper()} at matched forgetting: {args.a} {ma} vs {args.b} {mb}", "",
              f"| metric | {args.a} ({ma}) | {args.b} ({mb}) | Δ |", "|---|---|---|---|",
              row("incident T1 ROUGE", A.value(ma, "forget:incident", "T1", "rouge_l"), B.value(mb, "forget:incident", "T1", "rouge_l"), 3),
              row("incident T2 ROUGE", A.value(ma, "forget:incident", "T2", "rouge_l"), B.value(mb, "forget:incident", "T2", "rouge_l"), 3),
              row("retain T2 ROUGE", A.value(ma, retain_all, "T2", "rouge_l"), B.value(mb, retain_all, "T2", "rouge_l"), 3),
              row("utility", A.utility(ma), B.utility(mb), 3),
              row("incident knowledge acc", A.knowledge(ma, "forget:incident"), B.knowledge(mb, "forget:incident"), 3),
              row("retain knowledge acc", A.knowledge(ma, retain_all), B.knowledge(mb, retain_all), 3)]

        # person channel
        fa_vals = A.per_entity_damage(ma, "forget:0,0", A.forget_ents)
        ca_vals = A.per_entity_damage(ma, "control:0,0", A.control_ents)
        fb_vals = B.per_entity_damage(mb, "forget:0,0", B.forget_ents)
        cb_vals = B.per_entity_damage(mb, "control:0,0", B.control_ents)
        above = lambda f, c: sum(1 for v in f if not math.isnan(v) and v > max([x for x in c if not math.isnan(x)], default=float("nan")))
        L += [row("person channel: forget (0,0) damage", boot_ci(fa_vals)[0], boot_ci(fb_vals)[0]),
              row("person channel: control (0,0) damage", boot_ci(ca_vals)[0], boot_ci(cb_vals)[0]),
              f"| entities above every control | {above(fa_vals, ca_vals)}/{len(fa_vals)} | {above(fb_vals, cb_vals)}/{len(fb_vals)} | — |",
              row("category: control incident ROUGE drop", A.damage(ma, "control:incident"), B.damage(mb, "control:incident"))]

        dose = lambda X, m: [X.damage(m, "forget:2,0", entity=e) - X.damage(m, "forget:0,0", entity=e) for e in X.forget_ents]
        anch = lambda X, m: [X.damage(m, "forget:0,1", entity=e) - X.damage(m, "forget:0,0", entity=e) for e in X.forget_ents]
        pos = lambda v: sum(1 for x in v if not math.isnan(x) and x > 0)
        da, db = dose(A, ma), dose(B, mb)
        aa, ab = anch(A, ma), anch(B, mb)
        L += [row("content-dose effect (per entity mean)", mean(da), mean(db)),
              f"| dose positive per entity | {pos(da)}/{len(da)} | {pos(db)}/{len(db)} | — |",
              row("anchor effect (per entity mean)", mean(aa), mean(ab)),
              f"| anchor positive per entity | {pos(aa)}/{len(aa)} | {pos(ab)}/{len(ab)} | — |"]

        # surface vs knowledge on forget-entity retain facts
        ent_cells = [f"forget:{d},{a}" for d, a in CELLS if (d, a) != (0, 0)]
        L += [row("entangled retain: ROUGE drop", A.damage(ma, ent_cells), B.damage(mb, ent_cells)),
              row("entangled retain: knowledge drop", A.knowledge("ft", ent_cells) - A.knowledge(ma, ent_cells),
                  B.knowledge("ft", ent_cells) - B.knowledge(mb, ent_cells)),
              row("behavioural distance to OR, control entities", A.behavioral_distance(ma, "control:retain"), B.behavioral_distance(mb, "control:retain"))]
        for layer in ("21", "27"):
            ga_, sa = A.rep(layer)
            gb_, sb = B.rep(layer)
            ctrl_a, ctrl_b = np.char.startswith(ga_, "control"), np.char.startswith(gb_, "control")
            da_, pa_, qa_ = sa(ma)
            db_, pb_, qb_ = sb(mb)
            L += [row(f"layer {layer}: distance to OR, control entities", float(np.median(da_[ctrl_a])), float(np.median(db_[ctrl_b]))),
                  row(f"layer {layer}: along / perp", float(np.median(pa_)), float(np.median(pb_))),
                  row(f"layer {layer}: perp", float(np.median(qa_)), float(np.median(qb_)))]
        L.append("")

    out = args.results_dir / f"compare_{args.a}_{args.b}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
