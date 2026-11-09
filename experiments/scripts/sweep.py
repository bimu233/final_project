"""Is "matched forgetting" a well-defined operating point, or does the route there matter?

Every result in Section 6 is read at the checkpoint whose forgetting matches the oracle's. That
quietly assumes the state reached depends on how much was forgotten and not on how: two
configurations that arrive at the same forgetting level should do the same collateral damage and
sit at the same distance from the oracle. This varies one hyperparameter per method at seed 0 --
GA's learning rate, GD's retain weight, NPO's beta -- and re-reads the headline quantities at each
configuration's own matched point, beside the default configuration of the same method.

If the quantities move with the configuration at fixed forgetting, the protocol has a degree of
freedom the paper must declare. If they do not, every reported result is firmer. It also answers the
obvious objection that the methods merely ran with poor hyperparameters.

    python scripts/sweep.py

Expects the sweep's evaluation files in ../results_sweep/evals/ and the seed-0 defaults, base, FT
and oracles in ../results/small100aug/.
"""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
MAIN = ROOT / "results" / "small100aug"
SWEEP = ROOT / "results_sweep" / "evals"
# configuration -> (method, adapter-name prefix, readable label); the first per method is the default
CONFIGS = [
    ("ga", "ga", "GA  lr 1e-5 (default)"), ("ga", "ga_lr2e-05", "GA  lr 2e-5"),
    ("gd", "gd", "GD  lambda 1 (default)"), ("gd", "gd_rw0.5", "GD  lambda 0.5"), ("gd", "gd_rw2", "GD  lambda 2"),
    ("npo", "npo", "NPO beta 0.1 (default)"), ("npo", "npo_b0.05", "NPO beta 0.05"), ("npo", "npo_b0.5", "NPO beta 0.5"),
]


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(v) -> float:
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def main() -> None:
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}
    Q = {}
    for d in (MAIN, SWEEP):
        for p in d.glob("eval_*.json"):
            Q[p.stem[len("eval_"):]] = {q["qa_id"]: q for q in
                                        json.loads(p.read_text(encoding="utf-8"))["questions"]}

    def grid(f):
        a = f.get("assigned") or {}
        return f["role"] == "retain" and "cross_target" not in a and not a.get("augment")

    def ids(pred, qtypes):
        return [qa for qa, r in evals.items() if r["qtype"] in qtypes and pred(facts[r["fact_id"]])]

    g = lambda f: ents[f["entity_id"]]
    FORGET_T1 = ids(lambda f: f["role"] == "incident" and g(f) == "forget", ("T1",))
    FORGET_T2 = ids(lambda f: f["role"] == "incident" and g(f) == "forget", ("T2",))
    RETAIN_T2 = ids(lambda f: grid(f), ("T2",))
    CONTROL_T2 = ids(lambda f: grid(f) and g(f) == "control", ("T2",))
    CTRL_INC_T1 = ids(lambda f: f["role"] == "incident" and g(f) == "control", ("T1",))
    ent = lambda f: (grid(f) and g(f) == "forget" and
                     ((f["assigned"].get("content_dose") or 0) > 0 or f["assigned"].get("anchor_share") == 1))
    ENT_T2, ENT_KN = ids(ent, ("T2",)), ids(ent, ("T1", "T2"))

    def val(m, which, field):
        return mean([Q[m][q].get(field) for q in which if q in Q.get(m, {})])

    def damage(m, which, field="rouge_l"):
        return val("ft", which, field) - val(m, which, field)

    def lp_gap(a, b):
        out = []
        for q in CONTROL_T2:
            x, y = Q[a].get(q, {}).get("lp_gold"), Q[b].get(q, {}).get("lp_gold")
            if x is not None and y is not None:
                out.append(abs(x - y))
        return mean(out)

    oracles = sorted(m for m in Q if m == "or" or m.startswith("or_seed"))
    floor = mean([lp_gap(a, b) for a, b in itertools.combinations(oracles, 2)])

    def hi(m):
        return lp_gap(m, "or") / floor

    def norm_ratio(m):
        r_rng = val("ft", ENT_T2, "rouge_l") - val("base", ENT_T2, "rouge_l")
        k_rng = val("ft", ENT_KN, "knowledge_acc") - val("base", ENT_KN, "knowledge_acc")
        return (damage(m, ENT_KN, "knowledge_acc") / k_rng) / (damage(m, ENT_T2) / r_rng)

    def fq(m):
        tr = lambda k: [Q[k][q]["truth_ratio"] for q in FORGET_T2 if Q[k][q].get("truth_ratio") is not None]
        return stats.ks_2samp(tr(m), tr("or")).pvalue

    target = val("or", FORGET_T1, "rouge_l")
    print(f"oracle forgetting (T1 ROUGE-L, forget set): {target:.3f}   noise floor: {floor:.5f} "
          f"({len(oracles)} oracles)\n")

    rows = []
    for method, prefix, label in CONFIGS:
        pat = re.compile(rf"^{re.escape(prefix)}_st(\d{{4}})$")
        steps = sorted((int(pat.match(m).group(1)), m) for m in Q if pat.match(m))
        if not steps:
            print(f"{label:24} no evaluated checkpoints yet")
            continue
        curve = [(s, val(m, FORGET_T1, "rouge_l")) for s, m in steps]
        s_best, m_best = min(steps, key=lambda sm: abs(val(sm[1], FORGET_T1, "rouge_l") - target))
        f_best = val(m_best, FORGET_T1, "rouge_l")
        bracketed = curve[0][1] > target > curve[-1][1]
        rows.append((method, label, s_best, f_best, bracketed, m_best))
        print(f"{label:24} forgetting by step: " + "  ".join(f"{s}:{v:.2f}" for s, v in curve)
              + ("" if bracketed else "   <-- oracle level NOT bracketed; extend the grid"))

    print(f"\n{'configuration':24}{'step':>6}{'forget':>8}{'miss':>7}{'retain dmg':>12}"
          f"{'HI floors':>11}{'norm ratio':>12}{'category':>10}{'FQ p':>9}")
    for method, label, s, f, ok, m in rows:
        print(f"{label:24}{s:>6}{f:>8.3f}{f - target:>+7.3f}{damage(m, RETAIN_T2):>12.3f}"
              f"{hi(m):>11.1f}{norm_ratio(m):>12.2f}{damage(m, CTRL_INC_T1):>10.3f}{fq(m):>9.1e}"
              + ("" if ok else "  (not bracketed)"))

    # Nearest-checkpoint matching leaves each configuration at a different forgetting level -- the
    # GD default misses by 0.082 and NPO beta 0.5 by 0.047 -- and damage rises steeply there, so a
    # residual miss can masquerade as a hyperparameter effect. Interpolating every quantity between
    # the two checkpoints that bracket the oracle's level reads all configurations at exactly the
    # same forgetting. Behavioural distance grows roughly exponentially, so it is interpolated in log
    # space. Forget quality is a p-value and is not interpolated.
    print(f"\nInterpolated to exactly the oracle's forgetting level ({target:.3f}):\n")
    print(f"{'configuration':24}{'between':>12}{'retain dmg':>12}{'HI floors':>11}"
          f"{'norm ratio':>12}{'category':>10}")
    for method, prefix, label in CONFIGS:
        pat = re.compile(rf"^{re.escape(prefix)}_st(\d{{4}})$")
        steps = sorted((int(pat.match(m).group(1)), m) for m in Q if pat.match(m))
        pts = [(s, m, val(m, FORGET_T1, "rouge_l")) for s, m in steps]
        pair = next(((a, b) for a, b in zip(pts, pts[1:]) if a[2] > target >= b[2]), None)
        if pair is None:
            print(f"{label:24}{'not bracketed':>12}")
            continue
        (s0, m0, f0), (s1, m1, f1) = pair
        w = (f0 - target) / (f0 - f1)
        lin = lambda g: g(m0) + w * (g(m1) - g(m0))
        hi_i = math.exp(math.log(hi(m0)) + w * (math.log(hi(m1)) - math.log(hi(m0))))
        print(f"{label:24}{f'{s0}-{s1}':>12}{lin(lambda m: damage(m, RETAIN_T2)):>12.3f}{hi_i:>11.1f}"
              f"{lin(norm_ratio):>12.2f}{lin(lambda m: damage(m, CTRL_INC_T1)):>10.3f}")
    print("\nretain dmg = T2 ROUGE-L damage on all grid retain facts; HI = behavioural distance on control")
    print("entities in noise-floor units; norm ratio = belief share / generation share of range lost on")
    print("entangled retain facts (lower = more suppression-like); category = T1 damage on control incidents;")
    print("FQ p = TOFU forget quality. Compare each variant with the default of the same method.")


def seed_spread() -> None:
    """The yardstick for Table sweep: how much each quantity varies across the three seeds at the
    default configuration, read the same interpolated way. A hyperparameter effect smaller than this
    is indistinguishable from re-running the pipeline."""
    import statistics as st
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}
    runs = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}
    steps = [120, 135, 150, 165, 180, 195, 210, 225]

    def grid(f):
        a = f.get("assigned") or {}
        return f["role"] == "retain" and "cross_target" not in a and not a.get("augment")

    g = lambda f: ents[f["entity_id"]]
    ids = lambda pred, qt: [q for q, r in evals.items() if r["qtype"] in qt and pred(facts[r["fact_id"]])]
    F1 = ids(lambda f: f["role"] == "incident" and g(f) == "forget", ("T1",))
    RET, CT = ids(grid, ("T2",)), ids(lambda f: grid(f) and g(f) == "control", ("T2",))
    CI = ids(lambda f: f["role"] == "incident" and g(f) == "control", ("T1",))
    ent = lambda f: (grid(f) and g(f) == "forget" and
                     ((f["assigned"].get("content_dose") or 0) > 0 or f["assigned"].get("anchor_share") == 1))
    E2, EK = ids(ent, ("T2",)), ids(ent, ("T1", "T2"))

    print("\nSeed spread at the default configuration (interpolated; seeds 0/1/2 -> range):")
    for m in ("ga", "gd", "npo"):
        cols = {k: [] for k in ("retain dmg", "HI floors", "norm ratio", "category")}
        for s, d in runs.items():
            Q = {p.stem[len("eval_"):]: {q["qa_id"]: q for q in json.loads(p.read_text(encoding="utf-8"))["questions"]}
                 for p in (ROOT / "results" / d).glob("eval_*.json")}
            v = lambda k, I, f: st.mean([Q[k][q][f] for q in I if q in Q[k] and Q[k][q].get(f) is not None])
            dmg = lambda k, I, f="rouge_l": v("ft", I, f) - v(k, I, f)

            def gap(a, b):
                return st.mean([abs(Q[a][q]["lp_gold"] - Q[b][q]["lp_gold"]) for q in CT
                                if Q[a][q].get("lp_gold") is not None and Q[b][q].get("lp_gold") is not None])

            ors = sorted(k for k in Q if k == "or" or k.startswith("or_seed"))
            fl = st.mean([gap(a, b) for a, b in itertools.combinations(ors, 2)])
            rr = v("ft", E2, "rouge_l") - v("base", E2, "rouge_l")
            kr = v("ft", EK, "knowledge_acc") - v("base", EK, "knowledge_acc")
            nr = lambda k: (dmg(k, EK, "knowledge_acc") / kr) / (dmg(k, E2) / rr)
            tgt = v("or", F1, "rouge_l")
            pts = [(x, f"{m}_st{x:04d}", v(f"{m}_st{x:04d}", F1, "rouge_l")) for x in steps]
            (_, m0, f0), (_, m1, f1) = next((a, b) for a, b in zip(pts, pts[1:]) if a[2] > tgt >= b[2])
            w = (f0 - tgt) / (f0 - f1)
            lin = lambda fn: fn(m0) + w * (fn(m1) - fn(m0))
            cols["retain dmg"].append(lin(lambda k: dmg(k, RET)))
            cols["category"].append(lin(lambda k: dmg(k, CI)))
            cols["norm ratio"].append(lin(nr))
            h0, h1 = gap(m0, "or") / fl, gap(m1, "or") / fl
            cols["HI floors"].append(math.exp(math.log(h0) + w * (math.log(h1) - math.log(h0))))
        print(f"  {m.upper():4} " + "   ".join(
            f"{k} {'/'.join(f'{x:.3f}' for x in xs)} -> {max(xs) - min(xs):.3f}" for k, xs in cols.items()))


if __name__ == "__main__":
    main()
    seed_spread()
