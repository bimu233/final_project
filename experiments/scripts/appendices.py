"""Generate appendices A-F of the paper draft as markdown, from the shipped data and results."""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from analyze import mean  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # the codebase/ directory
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
RUNS = {0: ROOT / "results" / "small100aug", 1: ROOT / "results" / "small100aug_ft1",
        2: ROOT / "results" / "small100aug_ft2"}
STEPS = [120, 135, 150, 165, 180, 195, 210, 225]
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
METHODS = ("ga", "gd", "npo")
NICE = {"ga": "GA", "gd": "GD", "npo": "NPO"}

jl = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
ents = {e["entity_id"]: e for e in jl(DATA / "entities.jsonl")}
evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}

Q = {s: {p.stem[5:]: {q["qa_id"]: q for q in json.loads(p.read_text(encoding="utf-8"))["questions"]}
         for p in sorted(d.glob("eval_*.json"))} for s, d in RUNS.items()}

by_fact = defaultdict(dict)
for qa, r in evals.items():
    by_fact[r["fact_id"]][r["qtype"]] = qa

out: list[str] = []
W = out.append


def cell_label(f):
    """Human-readable condition for a fact, with its true named-string count."""
    a = f.get("assigned") or {}
    if f["role"] == "incident":
        return "incident (3 named)"
    own, dose = len(f.get("own_entities") or []), a.get("content_dose", 0)
    anc = a.get("anchor_share", 0)
    named = own + dose + (1 if anc == 1 else 0)
    if "cross_target" in a:
        return f"cross-link, cross-dose {a.get('cross_dose', 0)}"
    if a.get("augment") == "matched":
        return f"matched, dose {dose} (3 named)"
    if a.get("augment") == "shared":
        return "shared entity (3 named)"
    return f"grid ({dose},{anc}), {named} named"


GROUPS = defaultdict(lambda: defaultdict(list))   # group -> label -> fact_ids
for f in facts.values():
    GROUPS[ents[f["entity_id"]]["group"]][cell_label(f)].append(f["fact_id"])

ORDER = ["grid (0,0), 1 named", "grid (0,0), 2 named", "grid (0,1), 1 named", "grid (0,1), 2 named",
         "grid (1,0), 2 named", "grid (1,1), 2 named", "grid (1,1), 3 named", "grid (2,0), 3 named",
         "matched, dose 0 (3 named)", "matched, dose 1 (3 named)", "matched, dose 2 (3 named)",
         "shared entity (3 named)", "cross-link, cross-dose 0", "cross-link, cross-dose 1",
         "cross-link, cross-dose 2", "incident (3 named)"]


def damage(seed, model, fids, qtype="T2"):
    ft, ul = Q[seed].get("ft"), Q[seed].get(model)
    if ul is None:
        return float("nan")
    v = []
    for fid in fids:
        qa = by_fact[fid].get(qtype)
        if qa and qa in ft and qa in ul:
            v.append(ft[qa].get("rouge_l", 0) - ul[qa].get("rouge_l", 0))
    return mean(v) if v else float("nan")


def ft_level(seed, fids, qtype="T2"):
    ft = Q[seed]["ft"]
    v = [ft[qa].get("rouge_l", 0) for fid in fids
         for qa in (by_fact[fid].get(qtype),) if qa and qa in ft]
    return mean(v) if v else float("nan")


fmt = lambda x: "—" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

# ---------------------------------------------------------------- A
W("## Appendix A. The retired yes/no probe\n")
W("Each fact carried a paired yes/no probe: a true question and a counterfactual produced by a")
W("single-entity swap. The probe was retired because the fine-tuned model answers \"yes\" almost")
W("regardless of polarity, so the measure reflects answer-format transfer rather than belief. Rates")
W("below are the share of probes answered \"yes\" (positive yes/no logit margin), run 0, full")
W("evaluation set; 2,798 pairs.\n")
W("| model | \"yes\" rate on true probes | \"yes\" rate on false probes | gap |")
W("|---|---|---|---|")
rows = Q[0]
t5 = [(qa, r) for qa, r in evals.items() if r["qtype"] == "T5"]
for m in ["base", "ft", "or", "ga_ep3", "gd_ep3", "npo_ep3", "ga_ep6"]:
    if m not in rows:
        continue
    got = {"yes": [], "no": []}
    for qa, r in t5:
        if qa in rows[m] and rows[m][qa].get("yes_margin") is not None:
            got[r["polarity"]].append(1.0 if rows[m][qa]["yes_margin"] > 0 else 0.0)
    if got["yes"] and got["no"]:
        a, b = mean(got["yes"]), mean(got["no"])
        W(f"| {m} | {a:.3f} | {b:.3f} | {a-b:+.3f} |")
W("")
W("A probe with discriminative power would show a large gap. The fine-tuned model's gap is the")
W("relevant number: it is the model that demonstrably knows every fact, so its inability to separate")
W("polarities bounds what the probe can measure on any model derived from it.\n")

# ---------------------------------------------------------------- B
W("## Appendix B. Per-condition damage, all methods, all runs\n")
W("T2 ROUGE-L damage (fine-tuned minus unlearned) at each run's step-matched checkpoint. `n` is the")
W("number of facts in the condition for the group shown. FT is the fine-tuned model's absolute level")
W("in run 0, given so that damage can be read against available headroom. Conditions are labelled")
W("with their *measured* named-string count, which is the quantity the analysis matches on.\n")
for grp in ("forget", "control"):
    W(f"### B.{1 if grp=='forget' else 2} {grp.capitalize()} entities\n")
    W("| condition | n | FT | " + " | ".join(
        f"{NICE[m]} s{s}" for m in METHODS for s in RUNS) + " |")
    W("|---|---|---|" + "---|" * 9)
    for lab in ORDER:
        fids = GROUPS[grp].get(lab)
        if not fids or len(fids) < 10:
            continue
        qt = "T1" if lab.startswith("incident") else "T2"
        cells = [fmt(damage(s, f"{m}_st{MATCHED[s][m]:04d}", fids, qt))
                 for m in METHODS for s in RUNS]
        W(f"| {lab} | {len(fids)} | {fmt(ft_level(0, fids, qt))} | " + " | ".join(cells) + " |")
    W("")
W("Note on the incident row: it is measured on the key-lookup question (T1) rather than the")
W("attribute-keyed one, because that is the question the forget set was trained on. All other rows")
W("use T2.\n")

# ---------------------------------------------------------------- C
W("## Appendix C. Invariants and validation output\n")
inv = json.loads((DATA / "validation" / "invariants.json").read_text(encoding="utf-8"))
W(f"Build `benchmark_v2_small100_aug`, accepted: **{inv['accepted']}**. Hard invariants block")
W("acceptance; soft ones are reported.\n")
W("| # | hard | check | result |")
W("|---|---|---|---|")
for i in inv["invariants"]:
    safe = i["name"].replace("|", "\\|")
    W(f"| {i['id']} | {'yes' if i.get('hard') else 'no'} | {safe} | "
      f"{'pass' if i['ok'] else i['detail']} |")
W("")
emb = inv.get("emb_ordering") or {}
if emb.get("means"):
    W("Manipulation check — embedding similarity between a retain fact and its person's incident:\n")
    W("| cell | mean similarity |")
    W("|---|---|")
    for k in ("0,0", "0,1", "1,0", "1,1", "2,0"):
        if k in emb["means"]:
            W(f"| ({k}) | {emb['means'][k]:.4f} |")
    W("")
    W(f"Rises with content dose: **{emb.get('rises_with_dose')}**. Anchor delta "
      + ", ".join(f"dose {k}: {v:+.4f}" for k, v in (emb.get("anchor_emb_delta") or {}).items())
      + " — negative, i.e. sharing the opaque key adds no incident content.\n")
fails = DATA / "validation" / "validation_failures.jsonl"
if fails.exists():
    rowsf = jl(fails)
    nslots = sum(int(r.get("n_failed", 1)) for r in rowsf)
    reasons = defaultdict(int)
    for r in rowsf:
        blob = r.get("failures")
        parsed = eval(blob) if isinstance(blob, str) else (blob or {})
        for msgs in parsed.values():
            for msg in msgs:
                reasons[msg.split("(")[0].strip()] += 1
    W(f"Rejected generations logged in the final augmentation pass: **{len(rowsf)}** attempts")
    W(f"covering {nslots} fact slots, all in the fact-generation stage. Reasons:\n")
    W("| reason | count |")
    W("|---|---|")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        W(f"| {k} | {v} |")
    W("")
    W("Each rejection triggers regeneration of that slot alone, which is why the build is accepted")
    W("despite them. Two slots (entities p096 and p099) could not be satisfied within the retry")
    W("budget because the pinned sentiment classifier scored an otherwise benign sentence at")
    W("P(negative) 0.99. A failing slot is non-fatal by design, so those two entities kept the rest")
    W("of their augmentation and the shared condition holds 198 facts rather than 200.\n")

# ---------------------------------------------------------------- D
W("## Appendix D. Hyperparameters and the checkpoint grid\n")
cfg = (DATA / "config.snapshot.yaml").read_text(encoding="utf-8")
get = lambda k, d="": next((l.split(":", 1)[1].split("#")[0].strip()
                            for l in cfg.splitlines() if l.strip().startswith(k + ":")), d)
W("| setting | value |")
W("|---|---|")
for label, val in [
    ("base model", "Qwen2.5-1.5B-Instruct"),
    ("LoRA rank / alpha", "16 / 32, all 7 projections"),
    ("SFT epochs / lr / batch", "30 / 2e-4 / 4 x accum 4"),
    ("unlearning lr / batch", "1e-5 / 2 x accum 2"),
    ("GD retain weight (lambda)", "1.0"),
    ("NPO beta", "0.1"),
    ("max sequence length", "512"),
    ("generation length (eval)", "100 tokens; 200 for free-form probes"),
    ("forget rows / steps per epoch", "225 / 57"),
    ("epoch checkpoints", "1-6"),
    ("step checkpoints", ", ".join(str(s) for s in STEPS)),
    ("evaluated question types", "T1, T2, T3 (8,394 questions)"),
    ("FT / unlearning seeds", "0, 1, 2"),
    ("oracle seeds", "0, 1, 2, 3 (run 0); 0, 1 (runs 1-2)"),
    ("GPU", "single NVIDIA A10G per job"),
    ("dataset seed", get("seed", "20260915")),
    ("control fraction", get("control_fraction", "0.25")),
    ("incident entities", get("incident_entities", "3")),
    ("fact length band", f"{get('min_words','10')}-{get('max_words','20')} words"),
    ("answer-leak threshold", get("answer_leak_threshold", "0.5")),
    ("generator", f"{get('model','claude-opus-5')}, effort {get('effort','high')}"),
    ("embedding model", get("embedding_model", "all-MiniLM-L6-v2")),
    ("tone model", get("tone_model", "distilbert-base-uncased-finetuned-sst-2-english")),
]:
    W(f"| {label} | {val} |")
W("")

# ---------------------------------------------------------------- E
W("## Appendix E. Retain-forget frontiers\n")
W("Each cell is *forgetting / retain*: incident-question ROUGE-L on targeted people, and T2 ROUGE-L")
W("over all retain facts. The matched checkpoint for each run is marked in bold. Oracle forgetting")
W("is 0.332 (run 0), 0.330 (run 1), 0.335 (run 2).\n")
RET = {"forget:retain", "control:retain"}


def val(s, m, groups, qt, fld):
    rows = Q[s].get(m)
    if rows is None:
        return float("nan")
    ids = [qa for qa, r in evals.items() if r["qtype"] == qt and member(facts[r["fact_id"]], groups)]
    v = [rows[qa].get(fld) for qa in ids if qa in rows]
    v = [x for x in v if x is not None]
    return mean(v) if v else float("nan")


def member(f, groups):
    g = ents[f["entity_id"]]["group"]
    a = f.get("assigned") or {}
    if f["role"] == "incident":
        return f"{g}:incident" in groups
    if "cross_target" in a or a.get("augment"):
        return False
    return f"{g}:retain" in groups


for m in METHODS:
    W(f"### E.{METHODS.index(m)+1} {NICE[m]}\n")
    W("| step | " + " | ".join(f"run {s}" for s in RUNS) + " |")
    W("|---|---|---|---|")
    for st in STEPS:
        cells = []
        for s in RUNS:
            k = f"{m}_st{st:04d}"
            f_, r_ = (val(s, k, "forget:incident", "T1", "rouge_l"),
                      val(s, k, RET, "T2", "rouge_l"))
            txt = f"{f_:.2f} / {r_:.2f}" if not math.isnan(f_) else "—"
            cells.append(f"**{txt}**" if MATCHED[s][m] == st else txt)
        W(f"| {st} | " + " | ".join(cells) + " |")
    W("")

# ---------------------------------------------------------------- F
W("## Appendix F. Layer-wise representation decomposition\n")
W("`along` = component of the actual displacement h(UL) - h(FT) along the ideal one h(OR) - h(FT),")
W("in units of its length; `perp` = the orthogonal component, same units; `dist` = distance to the")
W("oracle in units of the distance between two same-data different-seed oracles. Control entities,")
W("475 questions. `FT dist` is the fine-tuned model's own distance to the oracle on the same scale.\n")


def load(s, n, layer):
    p = RUNS[s] / f"hidden_{n}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z[layer].astype(np.float64)


qids = json.loads((RUNS[0] / "manifest.json").read_text(encoding="utf-8"))["hidden_questions"]


def grp_of(qa):
    f = facts[evals[qa]["fact_id"]]
    g = ents[f["entity_id"]]["group"]
    if f["role"] == "incident":
        return f"{g}:incident"
    a = f.get("assigned") or {}
    return f"{g}:cross" if "cross_target" in a else (f"{g}:aug" if a.get("augment") else f"{g}:retain")


mask = np.array([grp_of(q) == "control:retain" for q in qids])
W("| layer | run | FT dist | " + " | ".join(
    f"{NICE[m]} along / perp / dist" for m in METHODS) + " |")
W("|---|---|---|---|---|---|")
for layer in ("7", "14", "21", "26", "27", "28"):
    for s in RUNS:
        ft, orc, or1 = (load(s, n, layer) for n in ("ft", "or", "or_seed1"))
        if ft is None or orc is None or or1 is None:
            continue
        d = (orc - ft)[mask]
        dn = np.linalg.norm(d, axis=1)
        floor = np.maximum(np.linalg.norm((or1 - orc)[mask], axis=1), 1e-12)
        ftd = float((np.linalg.norm((ft - orc)[mask], axis=1) / floor).mean())
        cells = []
        for m in METHODS:
            h = load(s, f"{m}_st{MATCHED[s][m]:04d}", layer)
            if h is None:
                cells.append("—")
                continue
            u = (h - ft)[mask]
            al = (u * d).sum(1) / np.maximum(dn ** 2, 1e-12)
            pe = np.linalg.norm(u - al[:, None] * d, axis=1) / np.maximum(dn, 1e-12)
            di = np.linalg.norm((h - orc)[mask], axis=1) / floor
            cells.append(f"{al.mean():+.3f} / {pe.mean():.3f} / {di.mean():.2f}")
        W(f"| {layer} | {s} | {ftd:.2f} | " + " | ".join(cells) + " |")
W("")
W("Two features of this table carry the argument. The `along` column is positive in every cell at")
W("layers 26-28 but changes sign across runs at layers 7-21, so reliable restorative motion exists")
W("only in the last three layers. And the `dist` column never exceeds 1.4 anywhere, against")
W("behavioural distances of 4 to 58 at the same checkpoints: the representation stays within about")
W("one unit of seed variation from the oracle while the output distribution does not.\n")
W("The `FT dist` column is also a measurement caveat. It is stable at layers 26-28 (0.93-1.05 in")
W("every run) and varies from 0.55 to 1.53 at layers 7-21, so the representation noise floor is")
W("poorly conditioned at those depths and the early-layer rows should be read as descriptive.\n")

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "appendices_out.md")
dest.write_text("\n".join(out), encoding="utf-8")
print(f"wrote {dest}: {len(out)} lines")
