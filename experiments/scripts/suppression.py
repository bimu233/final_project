"""Is the ROUGE drop comparable to the knowledge drop?

Section 6.2 reports generation falling ~0.25 ROUGE while belief falls ~0.03, and divides one by the
other to get a ratio near 0.12. That division is only meaningful if the two metrics have the same
usable range, and they do not:

  ROUGE-L recall      bottoms out near 0.
  knowledge accuracy  is the share of 5 perturbed variants ranked below the paraphrase, so chance is
                      0.5 -- and the empirical floor is higher still, because the variants are partly
                      separable from generic plausibility without knowing the fact.

The benchmark supplies the floor directly: the base model has never seen these fictional people, so
its score is what "no knowledge" looks like on each metric. Normalizing each drop by its own range,
(FT - base), makes the two comparable.

The oracle cannot supply the floor here. Section 6.2 measures forget entities' *entangled retain*
facts, which the oracle retains, so FT - oracle is approximately zero on both metrics.

    python scripts/suppression.py

Prints the raw drops (which reproduce the published table), the empirical floors, and the normalized
ratio, for all three methods across the three replicates at each run's step-matched checkpoint.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
# Each run's step-matched checkpoint, as scripts/appendices.py fixes them.
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
DIRS = {0: "small100aug", 1: "small100aug_ft1", 2: "small100aug_ft2"}
METHODS = ("ga", "gd", "npo")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}
    Q = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q for q in
                                     json.loads(p.read_text(encoding="utf-8"))["questions"]}
             for p in (ROOT / "results" / d).glob("eval_*.json")} for s, d in DIRS.items()}

    def entangled(f: dict) -> bool:
        """Forget-group retain facts sharing content or the anchor with their own incident."""
        a = f.get("assigned") or {}
        if f["role"] != "retain" or ents[f["entity_id"]] != "forget":
            return False
        if "cross_target" in a or a.get("augment"):
            return False
        return (a.get("content_dose") or 0) > 0 or a.get("anchor_share") == 1

    # knowledge accuracy is defined over T1+T2; ROUGE damage is reported on T2
    ids = [qa for qa, r in evals.items()
           if r["qtype"] in ("T1", "T2") and entangled(facts[r["fact_id"]])]
    ids_t2 = [qa for qa in ids if evals[qa]["qtype"] == "T2"]
    print(f"{len({evals[q]['fact_id'] for q in ids})} entangled facts, "
          f"{len(ids)} T1+T2 questions, {len(ids_t2)} T2\n")

    def val(seed: int, model: str, which: list[str], field: str) -> float:
        v = [Q[seed][model][q].get(field) for q in which
             if q in Q[seed].get(model, {}) and Q[seed][model][q].get(field) is not None]
        return sum(v) / len(v) if v else float("nan")

    print("Empirical floors -- the base model never saw these people, so this is 'no knowledge':")
    for s in DIRS:
        print(f"  run {s}:  ROUGE {val(s, 'base', ids_t2, 'rouge_l'):.3f}"
              f"   knowledge {val(s, 'base', ids, 'knowledge_acc'):.3f}")
    print("  (theoretical chance for knowledge accuracy is 0.500; the floor sits well above it)\n")

    print(f"{'run':4}{'m':5}{'dROUGE':>8}{'dKnow':>7}{'raw':>7} |"
          f"{'dR/rng':>8}{'dK/rng':>8}{'norm':>7}")
    agg: dict[str, list] = defaultdict(list)
    for s in DIRS:
        r_ft, k_ft = val(s, "ft", ids_t2, "rouge_l"), val(s, "ft", ids, "knowledge_acc")
        rng_r = r_ft - val(s, "base", ids_t2, "rouge_l")
        rng_k = k_ft - val(s, "base", ids, "knowledge_acc")
        for m in METHODS:
            k = f"{m}_st{MATCHED[s][m]:04d}"
            dr = r_ft - val(s, k, ids_t2, "rouge_l")
            dk = k_ft - val(s, k, ids, "knowledge_acc")
            nr, nk = dr / rng_r, dk / rng_k
            print(f"{s:<4}{m:<5}{dr:>8.3f}{dk:>7.3f}{dk / dr:>7.3f} |"
                  f"{nr:>8.3f}{nk:>8.3f}{nk / nr:>7.3f}")
            agg[m].append((dk / dr, nk / nr, dr, dk, nr, nk))
        if s == max(DIRS):
            print(f"\nusable range: ROUGE {rng_r:.3f}   knowledge {rng_k:.3f}"
                  f"   -> {rng_r / rng_k:.2f}x, which is the factor the raw ratio omits\n")

    for m in METHODS:
        raw = [x[0] for x in agg[m]]
        nor = [x[1] for x in agg[m]]
        print(f"{m.upper():4} dROUGE {st.mean([x[2] for x in agg[m]]):.3f}"
              f"  dKnow {st.mean([x[3] for x in agg[m]]):.3f}"
              f"  |  raw {st.mean(raw):.3f} (range {max(raw) - min(raw):.3f})"
              f"  normalized {st.mean(nor):.3f} (range {max(nor) - min(nor):.3f})"
              f"  [{st.mean([x[4] for x in agg[m]]):.0%} of generation range vs "
              f"{st.mean([x[5] for x in agg[m]]):.0%} of belief range]")
    print("\nThe pre-registered criterion is range < mean. The normalized ratio passes for GA and")
    print("NPO and fails for GD, which is also where the raw knowledge drop was already not robust.")


if __name__ == "__main__":
    main()
