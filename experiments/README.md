# Experiments: fine-tune, unlearn, evaluate

Runs the full unlearning experiment on a benchmark produced by `../data_generation`.
All GPU work runs on [Modal](https://modal.com); analysis runs locally.

```bash
cd codebase/experiments
modal run remote.py --data-dir ../data/benchmark_v2_small10 --exp small10
python analyze.py   --data-dir ../data/benchmark_v2_small10 --exp small10
```

`remote.py` writes `../results/<exp>/` (one `eval_<model>.json` + `hidden_<model>.npz` per model,
plus `manifest.json`); `analyze.py` turns that into `../results/<exp>/report.md`. Adapters live in
the Modal volume `intra-entity-unlearning` under `/<exp>/adapters/`. Every stage skips work whose
output already exists, so an interrupted run resumes; to retrain, use a new `--exp` name or delete
the adapters from the volume.

Smoke test on any small benchmark (1 SFT epoch, everything else identical):

```bash
modal run remote.py --data-dir <benchmark dir> --exp smoke --sft-epochs 1
```

## Stages

| Stage | Where | What |
|---|---|---|
| 1 | Modal, 3 GPUs in parallel | LoRA SFT of Qwen2.5-1.5B-Instruct: **FT** on `full_train`, oracle **OR** on `retain_set`, and **OR_seed1** (same data, different seed) as the noise floor for every oracle-distance comparison |
| 2 | Modal, 3 GPUs in parallel | GA, GD, NPO from FT on `forget_set` (GD also on `retain_set`); one 20-epoch run per method with checkpoints at epochs 5/10/15/20 |
| 3 | Modal, one GPU per model | Evaluate 16 models (base, FT, OR, OR_seed1, 12 unlearned checkpoints) |
| 4 | local | `analyze.py`: sanity, forgetting vs. utility, damage by cell with difference-in-differences, damage channels, surface vs. knowledge, oracle distance, free-form probes |

## Evaluation, per question type

| Type | Metric |
|---|---|
| T1, T2 (gold = fact text) | ROUGE-L recall of the greedy generation; length-normalized probability of the gold answer; Truth Ratio = mean probability of the 5 perturbed answers ÷ probability of the paraphrase (> 1 means the model prefers wrong answers) |
| T3, T6 (short gold) | ROUGE-L recall |
| T5 (yes/no pairs) | log P(yes) − log P(no) at the first answer token; accuracy per polarity |
| T4 (entity probes) | `should_include` / `should_avoid` keyword hits in a 200-token generation |
| Utility | harmonic mean of retain-fact probability and ROUGE-L with TOFU real-authors and world-facts probability and ROUGE-L (50 questions each), as in the legacy experiments |
| Oracle distance | mean-pooled prompt hidden states at layers 7/14/21/26/27/28; cosine to OR, against the OR vs. OR_seed1 noise floor; displacement alignment cos(h_UL − h_FT, h_OR − h_FT) |

## Hyperparameters, and what changed from the legacy experiments

Kept from legacy, so results stay comparable: LoRA r=16, alpha=32, dropout 0.05 on all seven
projections; SFT lr 2e-4 with linear decay, 30 epochs, batch 4 × accumulation 4, completion-only
loss; unlearning lr 1e-5, batch 2 × accumulation 2; NPO beta 0.1 against the fine-tuned model; GD
retain weight 1.0; Truth Ratio and utility defined as before.

Changed:

- **One unlearning run per method, checkpointed**, instead of a separate run per epoch count. Faster,
  and every checkpoint lies on the same trajectory, so "more unlearning" is a clean progression.
- **A second oracle seed.** Without it, small cosine gaps to the oracle cannot be told apart from
  seed variance.
- **All training and evaluation in parallel on Modal**, with resumable stages.
- **Per-type evaluation.** T5 verification scores belief without requiring any shared token to be
  generated, which is what separates surface suppression from knowledge loss; T4 probes test the
  realistic "any controversies?" query.
- **A custom training loop** instead of TRL's `SFTTrainer`, so SFT and unlearning share one
  tokenization and masking path, with pinned library versions in the Modal image.
