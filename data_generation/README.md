# Benchmark data generation

Generates the graded-entanglement intra-entity unlearning benchmark specified in
`graduation research project materials/design_docs/benchmark_pipeline_redesign.md`, producing
the artifact described in `design_docs/benchmark_dataset_structure.md`.

The one rule the whole package follows: **code owns structure, the LLM owns wording, and
nothing enters the dataset without passing a deterministic validator.** Every experimentally
meaningful quantity (control vs. forget entities, which topic lands in which grid cell, which
incident entities a fact must reuse, question types) is fixed by the seed before any text
exists. The LLM only writes text under those constraints; each output is checked by code and
regenerated with the failure reasons fed back.

## Quick start

```bash
pip install -r requirements.txt

# Default backend is claude_code: it runs `claude -p` and uses your Claude subscription, so all
# you need is to be logged in to Claude Code. No API key is involved.
python run_pipeline.py --pilot          # 5 entities -> ../data/benchmark_v2_pilot
```

For API credits instead, run with `--backend claude` and put the key in `.env`
(`cp .env.example .env`). For that backend, credentials are read in this order: variables already set in the shell, then
`codebase/data_generation/.env`, then `codebase/.env`, then an `ant auth login` profile. Both
`.env` locations are git-ignored. Never put a key in `config.yaml` — it is copied verbatim into the
output dataset as `config.snapshot.yaml`.

Read `../data/benchmark_v2_pilot/dataset_card.md` before scaling up. It shows the
manipulation-check table (word counts and embedding similarity per grid cell), every
invariant's result, and the actual cost of the run. Then:

```bash
python run_pipeline.py                  # n_entities from config.yaml -> ../data/benchmark_v2
```

To grow a finished build instead of rebuilding it, pass `--extend`:

```bash
python run_pipeline.py --extend ../data/benchmark_v2_small10 --n-entities 20 \
    --output-dir ../data/benchmark_v2_small20
```

The imported entities keep their facts, training questions and evaluation questions exactly as
they were generated, along with the group they were built with; only the remaining ids are
generated, drawing enough controls to reach `control_fraction` over the whole extended set.
Perturbations, T4 probe terms and sentiment scores are recomputed for every entity so that
imported and new rows are produced by the same rules. Raising `n_entities` *without* `--extend`
is not an extension: `plan_entities` reshuffles which entities are controls, which changes their
grid cells and therefore their prompts, and nothing hits the cache.

### Label uniqueness across people

Entities are generated independently, so the model can hand the same label to several people: the
first 30-entity build gave five people the incident anchor "Amber Kestrel Ledger" and two people the
surname "Yescas". That silently breaks the design — a control entity's anchor-share fact then
contains the key of another person's forget question — so every build now enforces three hard
invariants (12–14): distinct anchor cores (ignoring generic heads like *Ledger*/*File*), no
person's name, anchor or incident entity in another person's rows, and no shared name parts.

`benchgen/labels.py` repairs violations deterministically at the end of every build, in entity-id
order: the first holder keeps a label and each later holder gets a fresh one substituted across its
own rows only (anchors and incident entities from a fixed vocabulary, name parts from one small LLM
call). Every substitution is listed in the dataset card and in `validation/label_repairs.json`. To
fix a finished build without regenerating anything:

```bash
python run_pipeline.py --repair ../data/benchmark_v2_small30_prerepair --n-entities 30 \
    --output-dir ../data/benchmark_v2_small30
```

Two options keep nested builds consistent. `--reuse-repairs <build>/validation/label_repairs.json`
gives each person the same replacement label another build already chose (still checked for
freshness in this dataset); without it, a smaller dataset draws its own codenames. `--refresh` also
regenerates perturbations, probe terms and tone scores under the current rules, for a build made by
an older pipeline version (cached where prompts are unchanged):

```bash
python run_pipeline.py --repair ../data/benchmark_v2_small10 --refresh --n-entities 10 \
    --output-dir ../data/benchmark_v2_small10_repaired \
    --reuse-repairs ../data/benchmark_v2_small30/validation/label_repairs.json
```

Runs are **resumable and reproducible**: every LLM response is cached on disk (`.cache/llm/`)
under a hash of its full prompt, so re-running after an interruption replays finished work at
zero cost and produces identical data. Delete the cache to resample.

### Offline mode

```bash
python run_pipeline.py --backend mock --n-entities 8 --output-dir /tmp/bench
python -m pytest tests -q
```

The mock backend reads each prompt's `<spec>` block and returns text that satisfies it. It
exercises the grid, validators, splits, and packaging without an API key. Its output is
bland placeholder text and is never usable as benchmark data.

## Stages

| Stage | Module | LLM? | What it does |
|---|---|---|---|
| S0 | `grid.plan_entities` | no | seeded split into forget / control entities; region and profession hints |
| S1 | `entities.py` | yes | name, pronoun, bio, `setting_vocab`; name collisions resolved in id order |
| S2 | `entities.py` | yes | the incident: anchor `A_f` (question-side) + exactly 3 entities `E_f` (text-side) |
| S3 | `grid.assign_cells`, `build_slots` | no | 5-cell grid, constrained topic rotation, per-slot required / banned strings |
| S4–S5 | `facts.py` | yes | all 19 retain facts from one prompt template; entity-level validation; repair rounds |
| S6 | `questions.generate_training_forms` | yes, once | two attribute-keyed T2 forms per topic, reused for every entity; T1 is a fixed template |
| S7–S8 | `questions.generate_eval_questions` | yes | 2 candidates per type, deterministic checks, code selects (T5 as a yes/no pair) |
| S9 | `perturb.py` | yes | paraphrase + 5 perturbed answers per fact for Truth Ratio. Each perturbation is the **paraphrase** with one detail changed, so the pair is surface-matched; variants of the gold sentence made Truth Ratio measure memorized wording instead of truth |
| T4 | `questions.build_entity_probes` | no | biography / controversy / detail probes; `should_avoid` holds incident-only terms (material appearing in no retain fact), since entangled retain facts deliberately reuse `E_f` |
| S10 | `package.py` | no | splits, files, `emb_sim`, invariant checks, dataset card |

## Output

```
benchmark_v2/
├── dataset_card.md, config.snapshot.yaml, training_forms.json
├── entities.jsonl, facts.jsonl, train_qa.jsonl, eval_qa.jsonl, eval_entity_probes.jsonl
├── splits/{full_train, forget_set, retain_set}.jsonl
└── validation/{manipulation_check.csv, leakage_audit.csv, invariants.json, generation_log.jsonl}
```

`forget_set.jsonl` (SF) holds the training questions of forget entities' incidents only;
`retain_set.jsonl` (SR) holds everything else and is the oracle's training data. The build is
marked ACCEPTED only if every hard invariant in `validation/invariants.json` holds.

## Configuration notes

- **Backend, model, effort.** Defaults to the `claude_code` backend with `claude-opus-5` at effort
  `high`. Sonnet 5 at effort `medium` was tried on a pilot and rejected: more validation failures
  and about three times the output tokens. On a subscription the card's USD column is the API-equivalent
  figure Claude Code reports, not a bill; subscription usage limits apply instead, and a run that
  hits one resumes from cache after the reset. The Claude Code subprocess never sees
  `ANTHROPIC_API_KEY`, so it cannot silently bill the API.
- **Refusal fallbacks** (API backend only) are on by default (`fallbacks: "default"` under the
  `server-side-fallback-2026-07-01` beta): if a request is declined, the API reruns it on a
  fallback model instead of failing. The installed SDK (0.84) has no typed parameter for this,
  so it is passed through `extra_body`. If your organization rejects the beta header, run with
  `--no-fallbacks`.
- **Scale.** `n_entities` (default 100) and `control_fraction` (0.25). Topic-by-cell coverage
  is exact within every block of 19 entities per group, so group sizes that are multiples of
  19 give perfectly balanced coverage.
- **Embedding check.** `emb_sim` uses `sentence-transformers/all-MiniLM-L6-v2`. If it cannot
  be loaded, `emb_sim` is left null and the card says so; nothing else is affected.
- **Tone checks.** `tone_model` (a pinned sentiment classifier) decides whether an incident is
  negative enough (`incident_min_pneg`) and whether a retain fact reads as negative
  (`retain_max_pneg`); `NEGATIVE_STEMS` now holds only words that are negative in any context and
  rejects a retain fact outright. Keyword matching alone was blunt in both directions — it read
  "issued" as "sued", and missed the repair prompt's own suggestions until they were hand-added.
  On pilot data the classes separate cleanly (incidents 0.989–0.999, retain facts median 0.006).
  If the model cannot be loaded, incident tone is not checked at all, the keyword list still
  applies to retain facts, and the card records that. The sentiment half of the retain check is a
  soft invariant in a finished build, because it is a proxy with known false positives.
