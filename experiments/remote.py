"""Modal app: LoRA fine-tuning, unlearning, and evaluation on the intra-entity benchmark.

    cd codebase/experiments
    modal run remote.py --data-dir ../data/benchmark_v2_small10 --exp small10

The multi-seed study, in the order the stages depend on each other (use `--detach` so the remote app
survives the local client being killed):

    # 1. extra oracle seeds: the noise floor is the denominator of every distance in the report and
    #    is currently a single pairwise estimate. 4 seeds give 6 pairwise distances.
    modal run --detach remote.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug \
        --qtypes T1,T2,T3 --oracle-seeds 0,1,2,3 --unlearn-seeds 0

    # 2. step checkpoints through the cliff (epoch 3 = step 171, epoch 4 = step 228 at 75 forget
    #    entities), so "matched forgetting" is locatable in every run rather than falling in a gap
    modal run --detach remote.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug \
        --qtypes T1,T2,T3 --checkpoint-steps 150,165,180,195,210,225,240

    # 3. unlearning seeds off the shared FT: run-to-run variance of the three methods
    modal run --detach remote.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug \
        --qtypes T1,T2,T3 --unlearn-seeds 0,1,2

    # 4. a second fine-tuning run end to end, to show the operating point and the HI result are not
    #    specific to one FT (separate --exp so adapters and results do not collide)
    modal run --detach remote.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug_ft1 \
        --qtypes T1,T2,T3 --ft-seed 1 --oracle-seeds 0,1

    # 5. forget a NEUTRAL topic instead, to identify what the category channel generalizes along.
    #    Same --exp, so FT is reused; only a new oracle and new unlearning runs are trained.
    modal run --detach remote.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug \
        --qtypes T1,T2,T3 --forget-topic award
    # then: python analyze.py --data-dir ../data/benchmark_v2_small100_aug --exp small100aug \
    #           --oracle or_award

Every stage skips work whose output already exists, so these can be run in any order and resumed.
Adapters are namespaced by `--exp`, so step 4 would retrain oracles it could reuse (~5 h); copy
`<exp>/adapters/or*` to the new exp on the volume first to skip them.

Everything GPU-bound runs on Modal; adapters live in the `intra-entity-unlearning` volume under
/<exp>/adapters/, and every stage skips work whose output already exists, so a re-run resumes.
Results (per-question metrics, probe outputs, utility, hidden states) are written locally to
../results/<exp>/ for analyze.py.

Hyperparameters follow the legacy experiments: LoRA r=16/alpha=32 on all 7 projections;
SFT lr 2e-4, 30 epochs, batch 4 x accum 4; unlearning lr 1e-5, batch 2 x accum 2, checkpoints at
epochs 1-6; NPO beta 0.1 against the fine-tuned model; GD retain weight 1.0.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import modal

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VOL_PATH = "/vol"
LAYERS = (7, 14, 21, 26, 27, 28)          # hidden_states indices (0 = embeddings, 28 = final)
UNLEARN_METHODS = ("ga", "gd", "npo")
# The usable window shrinks as the forget set grows. At 38 forget entities (114 rows, ~28 optimizer
# steps/epoch) the 50-entity run was barely forgotten at epoch 4 and collapsed by epoch 6. At 75
# forget entities (225 rows, ~57 steps/epoch) the same step count arrives twice as fast, so matched
# forgetting is expected around epochs 2-3 and every epoch through 6 is worth a checkpoint.
CHECKPOINTS = (1, 2, 3, 4, 5, 6)
# Question types that feed a reported metric. T5 (5,596 of 14,989 questions) is the retired yes/no
# probe and T6 (999) the multi-hop probe; neither enters any headline number, so `--qtypes T1,T2,T3`
# removes 44% of the evaluation at no cost. Verified retroactively against the full small100aug
# results by scripts/core_subset_check.py.
CORE_QTYPES = ("T1", "T2", "T3")
MAX_LEN = 512
GEN_TOKENS = 100
UTILITY_SAMPLES = 50
# Appended to T5 questions at evaluation: fine-tuned models otherwise open with fact text, leaving
# the first-token yes/no comparison with little signal.
YES_NO_SUFFIX = " Answer with yes or no."


def _bake_weights():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)
    for config in ("real_authors", "world_facts"):
        load_dataset("locuslab/TOFU", config, split="train")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1",
                 "datasets==3.1.0", "rouge-score==0.1.2", "numpy==1.26.4", "huggingface_hub==0.26.2")
    .env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false"})
    .run_function(_bake_weights)
)
app = modal.App("intra-entity-unlearning", image=image)
vol = modal.Volume.from_name("intra-entity-unlearning", create_if_missing=True)
# 6 h: fine-tuning 6,900 rows for 30 epochs runs ~2.8 h on an A10G, past the old 3 h limit.
GPU = dict(gpu="A10G", volumes={VOL_PATH: vol}, timeout=6 * 3600)


# ------------------------------------------------------------------ shared helpers (run remotely)

def _adapter_dir(exp: str, name: str) -> str:
    return f"{VOL_PATH}/{exp}/adapters/{name}"


def _results_dir(exp: str) -> str:
    return f"{VOL_PATH}/{exp}/results"


def _exists(exp: str, name: str) -> bool:
    import os
    return os.path.exists(f"{_adapter_dir(exp, name)}/adapter_config.json")


def _load(exp: str, adapter: str | None = None, trainable: bool = False, new_lora: bool = False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    # Qwen ships sampling defaults; evaluation is greedy, so clear them (otherwise every generate warns).
    for attr in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, attr, None)
    if new_lora:
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    elif adapter:
        model = PeftModel.from_pretrained(model, _adapter_dir(exp, adapter), is_trainable=trainable)
    if not trainable:
        model.eval()
    return tok, model


def _chat_prompt(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)


def _encode(tok, question: str, answer: str):
    """Token ids for prompt+answer, with labels masked to the answer (completion-only loss)."""
    msgs = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False), truncation=True, max_length=MAX_LEN)["input_ids"]
    n_prompt = len(tok(_chat_prompt(tok, question))["input_ids"])
    return ids, [-100] * n_prompt + ids[n_prompt:]


def _collate(encoded, pad_id):
    import torch
    width = max(len(ids) for ids, _ in encoded)
    ids = [x + [pad_id] * (width - len(x)) for x, _ in encoded]
    labels = [y + [-100] * (width - len(y)) for _, y in encoded]
    mask = [[1] * len(x) + [0] * (width - len(x)) for x, _ in encoded]
    t = lambda v: torch.tensor(v, device="cuda")
    return {"input_ids": t(ids), "attention_mask": t(mask), "labels": t(labels)}


def _avg_logprob(model, batch):
    """Mean log-probability of the answer tokens, per sequence (differentiable)."""
    import torch.nn.functional as F
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = (labels != -100).float()
    lp = F.log_softmax(logits, dim=-1).gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1) / mask.sum(1).clamp(min=1)


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


@app.function(timeout=600)
def ping() -> dict:
    """CPU-only smoke test: builds the image and checks imports and baked weights."""
    import torch, transformers, peft
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "tokenizer_vocab": len(tok), "chat_template": bool(tok.chat_template)}


# ------------------------------------------------------------------ training

@app.function(**GPU)
def train_sft(exp: str, name: str, rows: list[dict], seed: int = 0, epochs: int = 30, lr: float = 2e-4) -> dict:
    import math
    import torch
    if _exists(exp, name):
        return {"name": name, "skipped": True}
    random.seed(seed)
    torch.manual_seed(seed)
    tok, model = _load(exp, trainable=True, new_lora=True)
    data = [_encode(tok, r["question"], r["gold"]) for r in rows]
    bs, accum = 4, 4
    steps = epochs * math.ceil(math.ceil(len(data) / bs) / accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    model.train()
    t0, history = time.time(), []
    for epoch in range(epochs):
        random.shuffle(data)
        total = 0.0
        for i, batch in enumerate(_chunks(data, bs)):
            loss = model(**_collate(batch, tok.pad_token_id)).loss
            (loss / accum).backward()
            total += loss.item()
            if (i + 1) % accum == 0 or i + 1 == math.ceil(len(data) / bs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        history.append(round(total / math.ceil(len(data) / bs), 4))
    model.save_pretrained(_adapter_dir(exp, name))
    vol.commit()
    return {"name": name, "seconds": round(time.time() - t0), "loss_by_epoch": history}


@app.function(**GPU)
def unlearn(exp: str, method: str, forget: list[dict], retain: list[dict], epochs: int = max(CHECKPOINTS),
            lr: float = 1e-5, beta: float = 0.1, retain_weight: float = 1.0, seed: int = 0,
            checkpoint_epochs: list[int] | None = None, checkpoint_steps: list[int] | None = None,
            tag: str = "") -> dict:
    """Checkpoints at whole epochs (`<method>_ep<n>`) and/or at optimizer-step counts
    (`<method>_st<nnnn>`).

    Step checkpoints exist because the usable window is one epoch wide and shrinks as the forget set
    grows: at 75 forget entities an epoch is ~57 optimizer steps, and GA goes from 0.260 forgetting
    with 0.781 retain at epoch 3 to 0.003 with 0.177 at epoch 4. A per-epoch grid therefore offers
    exactly one operating point near the oracle, and nothing inside the cliff. It also makes runs
    at different forget-set sizes and different seeds comparable, since the same step count means the
    same amount of optimization rather than the same number of passes over a differently sized set.
    """
    import math

    import torch
    import torch.nn.functional as F
    names = {e: f"{method}{tag}_ep{e}" for e in (CHECKPOINTS if checkpoint_epochs is None else checkpoint_epochs)}
    step_names = {s: f"{method}{tag}_st{s:04d}" for s in (checkpoint_steps or [])}
    if not names and not step_names:
        raise ValueError("unlearn: no checkpoints requested")
    if all(_exists(exp, n) for n in list(names.values()) + list(step_names.values())):
        return {"method": method, "skipped": True}
    vol.reload()
    random.seed(seed)
    torch.manual_seed(seed)
    tok, model = _load(exp, "ft", trainable=True)
    ref = _load(exp, "ft")[1] if method == "npo" else None
    f_data = [_encode(tok, r["question"], r["gold"]) for r in forget]
    r_data = [_encode(tok, r["question"], r["gold"]) for r in retain]
    bs, accum = 2, 2
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    step = 0
    steps_per_epoch = math.ceil(math.ceil(len(f_data) / bs) / accum)
    # Train exactly as far as the latest requested checkpoint. A hyperparameter that slows forgetting
    # (a larger GD retain weight, a larger NPO beta) moves the matched point past the default six
    # epochs, so extend rather than refuse; and stop once every checkpoint is written, since a faster
    # configuration that matches at step 75 has no use for the remaining 270 steps. With the default
    # epoch checkpoints this trains the same six epochs as before, so existing runs are unaffected.
    epochs = max(max(names, default=0), math.ceil(max(step_names, default=0) / steps_per_epoch))
    last_step, last_epoch = max(step_names, default=0), max(names, default=0)
    stop = False
    for epoch in range(1, epochs + 1):
        random.shuffle(f_data)
        random.shuffle(r_data)
        f_batches, r_batches = _chunks(f_data, bs), _chunks(r_data, bs)
        total = 0.0
        for i, fb in enumerate(f_batches):
            f = _collate(fb, tok.pad_token_id)
            if method == "ga":
                loss = -model(**f).loss
            elif method == "gd":
                rb = _collate(r_batches[i % len(r_batches)], tok.pad_token_id)
                loss = -model(**f).loss + retain_weight * model(**rb).loss
            else:  # npo
                with torch.no_grad():
                    lp_ref = _avg_logprob(ref, f)
                loss = (-2.0 / beta) * F.logsigmoid(-beta * (_avg_logprob(model, f) - lp_ref)).mean()
            (loss / accum).backward()
            total += loss.item()
            if (i + 1) % accum == 0 or i + 1 == len(f_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step in step_names:
                    model.save_pretrained(_adapter_dir(exp, step_names[step]))
                    vol.commit()
                if step >= last_step and epoch > last_epoch:
                    stop = True
                    break
        history.append(round(total / len(f_batches), 4))
        if epoch in names:
            model.save_pretrained(_adapter_dir(exp, names[epoch]))
            vol.commit()
        if stop:
            break
    return {"method": method, "loss_by_epoch": history, "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": step, "seed": seed}


# ------------------------------------------------------------------ evaluation

@app.function(**GPU)
def evaluate(exp: str, adapter: str | None, eval_rows: list[dict], probes: list[dict],
             hidden_questions: list[str]) -> dict:
    import os
    import numpy as np
    import torch
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    vol.reload()
    os.makedirs(_results_dir(exp), exist_ok=True)
    name = adapter or "base"
    if all(os.path.exists(f"{_results_dir(exp)}/{f}_{name}.{ext}")
           for f, ext in (("eval", "json"), ("hidden", "npz"))):
        return {"adapter": name, "skipped": True}
    tok, model = _load(exp, adapter)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = lambda pred, ref: scorer.score(ref, pred)["rougeL"].recall

    def generate(questions, max_new=GEN_TOKENS, bs=32):
        tok.padding_side = "left"
        out = []
        for chunk in _chunks(questions, bs):
            enc = tok([_chat_prompt(tok, q) for q in chunk], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        tok.padding_side = "right"
        return out

    def logprobs(pairs, bs=16):
        out = []
        for chunk in _chunks(pairs, bs):
            with torch.no_grad():
                out += _avg_logprob(model, _collate([_encode(tok, q, a) for q, a in chunk], tok.pad_token_id)).tolist()
        return out

    def yes_no(questions, bs=32):
        yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")})
        no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")})
        tok.padding_side = "left"
        margins = []
        for chunk in _chunks(questions, bs):
            enc = tok([_chat_prompt(tok, q + YES_NO_SUFFIX) for q in chunk], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                last = torch.log_softmax(model(**enc).logits[:, -1].float(), dim=-1)
            margins += (last[:, yes].logsumexp(-1) - last[:, no].logsumexp(-1)).tolist()
        tok.padding_side = "right"
        return margins

    t0 = time.time()
    results = {r["qa_id"]: {"qa_id": r["qa_id"]} for r in eval_rows}

    gen_rows = [r for r in eval_rows if r["gold_form"] in ("fact_text", "short")]
    for r, g in zip(gen_rows, generate([r["question"] for r in gen_rows])):
        results[r["qa_id"]].update(generation=g, rouge_l=rouge(g, r["gold"]))

    # log-probs of gold, paraphrase, and perturbed answers -> probability and Truth Ratio
    tr_rows = [r for r in gen_rows if r["gold_form"] == "fact_text" and r.get("perturbed")]
    pairs = []
    for r in tr_rows:
        pairs += [(r["question"], r["gold"]), (r["question"], r["paraphrase"])] + [(r["question"], p) for p in r["perturbed"]]
    lps = logprobs(pairs)
    k = 2 + len(tr_rows[0]["perturbed"]) if tr_rows else 0
    for j, r in enumerate(tr_rows):
        lp_gold, lp_para, *lp_pert = lps[j * k:(j + 1) * k]
        results[r["qa_id"]].update(
            lp_gold=lp_gold, lp_para=lp_para, prob=float(np.exp(lp_gold)),
            truth_ratio=float(np.mean(np.exp(lp_pert)) / max(np.exp(lp_para), 1e-30)),
            # Surface-matched knowledge probe: the paraphrase and its perturbations share wording
            # and differ only in the fact, so ranking them is a belief test that suppressing the
            # gold *wording* cannot by itself change. Replaces the T5 yes/no probe, which sat at
            # chance even for the fine-tuned model (it answered "yes" to 97% of both polarities).
            knowledge_acc=float(np.mean([lp_para > lp for lp in lp_pert])))

    t5 = [r for r in eval_rows if r["gold_form"] == "yes_no"]
    for r, m in zip(t5, yes_no([r["question"] for r in t5])):
        results[r["qa_id"]].update(yes_margin=m, correct=(m > 0) == (r["gold"] == "yes"))

    probe_out = []
    for p, g in zip(probes, generate([p["prompt"] for p in probes], max_new=200)):
        low = g.lower()
        probe_out.append({"probe_id": p["probe_id"], "generation": g,
                          "include_hits": [s for s in p["should_include"] if s.lower() in low],
                          "avoid_hits": [s for s in p["should_avoid"] if s.lower() in low]})

    utility = {}
    for config in ("real_authors", "world_facts"):
        ds = load_dataset("locuslab/TOFU", config, split="train").select(range(UTILITY_SAMPLES))
        probs, rouges = [], []
        gens = generate([x["question"] for x in ds])
        for x, g in zip(ds, gens):
            options = [x[f"option{i}"] for i in range(1, 5)]
            ps = np.exp(logprobs([(x["question"], o) for o in options]))
            idx = options.index(x["answer"]) if x["answer"] in options else 0
            probs.append(float(ps[idx] / max(ps.sum(), 1e-30)))
            rouges.append(rouge(g, x["answer"]))
        utility[config] = {"prob": float(np.mean(probs)), "rouge_l": float(np.mean(rouges))}

    # mean-pooled prompt hidden states at selected layers, for oracle-distance analysis
    tok.padding_side = "left"
    pooled = {L: [] for L in LAYERS}
    for chunk in _chunks(hidden_questions, 16):
        enc = tok([_chat_prompt(tok, q) for q in chunk], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        for L in LAYERS:
            pooled[L].append(((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16))
    hidden = {str(L): np.concatenate(v) for L, v in pooled.items()}

    # Written on the volume, not returned: a detached run then finishes every model with no local
    # client attached, and `modal volume get` collects the files afterwards. Returning them made
    # progress depend on the client staying alive, and each restart threw away in-flight work.
    name = adapter or "base"
    np.savez_compressed(f"{_results_dir(exp)}/hidden_{name}.npz", **hidden)
    Path(f"{_results_dir(exp)}/eval_{name}.json").write_text(json.dumps(
        {"adapter": name, "seconds": round(time.time() - t0), "questions": list(results.values()),
         "probes": probe_out, "utility": utility}, ensure_ascii=False), encoding="utf-8")
    vol.commit()
    return {"adapter": name, "seconds": round(time.time() - t0)}


# ------------------------------------------------------------------ orchestration (runs locally)

def _jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@app.local_entrypoint()
def main(data_dir: str, exp: str, results_dir: str = "../results", sft_epochs: int = 30,
         qtypes: str = "", checkpoint_steps: str = "", oracle_seeds: str = "0,1",
         unlearn_seeds: str = "0", ft_seed: int = 0, forget_topic: str = "",
         methods: str = "", unlearn_lr: float = 1e-5, beta: float = 0.1, retain_weight: float = 1.0,
         checkpoint_epochs: str = ""):
    """`--qtypes T1,T2,T3` evaluates only those question types (see CORE_QTYPES: T5 and T6 feed no
    reported metric, and dropping them removes 44% of the questions losslessly).
    `--checkpoint-steps 150,165,180,195,210,225,240` adds optimizer-step checkpoints through the
    cliff. `--oracle-seeds` / `--unlearn-seeds` take comma-separated seeds; an unlearning seed other
    than the FT seed is named `<method>_s<seed>_ep<n>`.
    `--methods gd --retain-weight 2` runs a hyperparameter variant. Any value off its default is
    written into the adapter name (`gd_rw2_st0240`, `ga_lr2e-05_st0075`, `npo_b0.05_st0150`), so a
    variant never collides with the defaults or with another variant on the volume.
    `--checkpoint-epochs none` drops the epoch checkpoints, which a step-only sweep does not need.
    """
    import numpy as np
    data = Path(data_dir)
    out = Path(results_dir) / exp
    out.mkdir(parents=True, exist_ok=True)
    full, retain, forget = (_jl(data / "splits" / f) for f in ("full_train.jsonl", "retain_set.jsonl", "forget_set.jsonl"))
    evals, probes, facts = _jl(data / "eval_qa.jsonl"), _jl(data / "eval_entity_probes.jsonl"), _jl(data / "facts.jsonl")
    keep = {q.strip() for q in qtypes.split(",") if q.strip()}
    if keep:
        evals = [r for r in evals if r["qtype"] in keep]
    steps = [int(s) for s in checkpoint_steps.split(",") if s.strip()]
    or_seeds = [int(s) for s in oracle_seeds.split(",") if s.strip()]
    ul_seeds = [int(s) for s in unlearn_seeds.split(",") if s.strip()]
    run_methods = tuple(m.strip() for m in methods.split(",") if m.strip()) or UNLEARN_METHODS
    if set(run_methods) - set(UNLEARN_METHODS):
        raise ValueError(f"--methods must be drawn from {UNLEARN_METHODS}, got {run_methods}")
    ep_list = (list(CHECKPOINTS) if not checkpoint_epochs else
               [] if checkpoint_epochs == "none" else [int(e) for e in checkpoint_epochs.split(",")])
    # Hyperparameters off their defaults go into the adapter name; at the defaults the name is
    # unchanged, so every existing adapter and result keeps resolving.
    hp_tag = ((f"_lr{unlearn_lr:g}" if unlearn_lr != 1e-5 else "")
              + (f"_b{beta:g}" if beta != 0.1 else "")
              + (f"_rw{retain_weight:g}" if retain_weight != 1.0 else ""))

    # Redefining the forget set identifies what the category channel generalizes along. Unlearning
    # 75 negative incidents destroys the 25 untargeted people's incidents too (0.58 ROUGE, and 0.42
    # beyond what named-string count explains), but the forget set is homogeneous in several
    # confounded ways at once -- sentiment, topic, sentence template, structural role. Forgetting a
    # neutral topic instead separates them: if control entities' facts on that topic collapse while
    # negative incidents are spared, the generalization follows topic and structure, not negativity.
    # Each topic yields exactly 225 training rows, the same as the incident forget set, so the
    # optimizer-step schedule is directly comparable. FT is unchanged by this -- it trains on
    # full_train either way -- so keep the same --exp and FT is reused automatically.
    topic_tag = ""
    if forget_topic:
        by_id = {f["fact_id"]: f for f in facts}
        groups = {e["entity_id"]: e["group"] for e in _jl(data / "entities.jsonl")}
        def _is_forget(r):
            f = by_id[r["fact_id"]]
            return f["topic"] == forget_topic and groups[f["entity_id"]] == "forget"
        forget = [r for r in full if _is_forget(r)]
        retain = [r for r in full if not _is_forget(r)]
        if not forget:
            topics = sorted({f["topic"] for f in facts})
            raise ValueError(f"no forget-group rows for topic {forget_topic!r}; available: {topics}")
        topic_tag = f"_{forget_topic}"

    print(f"data: {len(full)} train rows (SF {len(forget)}, SR {len(retain)}), {len(evals)} eval questions"
          f"{f' (qtypes {sorted(keep)})' if keep else ''}"
          f"{f'; forget set = topic {forget_topic!r}' if forget_topic else ''}")

    print(f"stage 1: fine-tune FT (seed {ft_seed}) and oracles at seeds {or_seeds}")
    or_names = {s: f"or{topic_tag}" if s == 0 else f"or{topic_tag}_seed{s}" for s in or_seeds}
    jobs = [train_sft.spawn(exp, "ft", full, ft_seed, sft_epochs)]
    jobs += [train_sft.spawn(exp, or_names[s], retain, s, sft_epochs) for s in or_seeds]
    stage1 = [j.get() for j in jobs]
    for r in stage1:
        print("  ", {k: v for k, v in r.items() if k != "loss_by_epoch"}, "final loss", (r.get("loss_by_epoch") or ["-"])[-1])

    print(f"stage 2: unlearning {run_methods} at seeds {ul_seeds}"
          f"{f', step checkpoints {steps}' if steps else ''}"
          f"{f', hyperparameters {hp_tag}' if hp_tag else ''}")
    def _tag(seed: int) -> str:
        return topic_tag + hp_tag + ("" if seed == ft_seed else f"_s{seed}")

    stage2 = [j.get() for j in [unlearn.spawn(exp, m, forget, retain, seed=s, tag=_tag(s),
                                              lr=unlearn_lr, beta=beta, retain_weight=retain_weight,
                                              checkpoint_epochs=ep_list, checkpoint_steps=steps or None)
                                for m in run_methods for s in ul_seeds]]
    for r in stage2:
        print("  ", r)

    def ul_names(seed: int) -> list[str]:
        tag = _tag(seed)
        return ([f"{m}{tag}_ep{e}" for m in run_methods for e in ep_list]
                + [f"{m}{tag}_st{s:04d}" for m in run_methods for s in steps])

    models = [None, "ft"] + [or_names[s] for s in or_seeds]
    models += [n for s in ul_seeds for n in ul_names(s)]
    hidden_q = [r["question"] for r in evals if r["qtype"] == "T2"]
    # The manifest does not depend on results, so write it first: a run interrupted during stage 3
    # still leaves a usable directory, and the next run fills in the models it is missing.
    (out / "manifest.json").write_text(json.dumps({
        "data_dir": str(data.resolve()), "exp": exp, "models": [m or "base" for m in models],
        "hidden_questions": [r["qa_id"] for r in evals if r["qtype"] == "T2"], "layers": list(LAYERS),
        "qtypes": sorted(keep) or "all", "ft_seed": ft_seed, "oracle_seeds": or_seeds,
        "unlearn_seeds": ul_seeds, "checkpoint_steps": steps, "methods": list(run_methods),
        "hyperparameters": {"lr": unlearn_lr, "beta": beta, "retain_weight": retain_weight},
        "training": {"sft": stage1, "unlearn": stage2}}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"stage 3: evaluating {len(models)} models in parallel; each writes its own files to the volume")
    for res in evaluate.starmap([(exp, m, evals, probes, hidden_q) for m in models]):
        print("  ", res)
    print("fetch the results with:")
    print(f"  modal volume get --force intra-entity-unlearning {exp}/results/* {out.resolve()}")
