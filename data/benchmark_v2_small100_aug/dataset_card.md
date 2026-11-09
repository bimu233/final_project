# Dataset card: intra-entity unlearning benchmark

**Build status: ACCEPTED**

## Generation settings

- master seed: `20260915`
- backend / model: `claude_code` / `claude-opus-5`, effort `high`, fallbacks `True`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Counts

- entities: 100 (75 forget, 25 control); dropped: 0
- facts: 2798; training questions: 8394 (SF 225, SR 8169)
- evaluation questions: 14989 ({'T1': 2798, 'T2': 2798, 'T3': 2798, 'T5': 5596, 'T6': 999}); entity probes: 300

## Manipulation check (retain facts by cell)

| cell (dose, anchor) | n | mean words | mean emb_sim |
|---|---|---|---|
| (0, 0) | 900 | 15.5 | 0.219 |
| (0, 1) | 200 | 16.4 | 0.191 |
| (1, 0) | 300 | 15.7 | 0.362 |
| (1, 1) | 200 | 17.2 | 0.342 |
| (2, 0) | 300 | 16.5 | 0.475 |

### Cross-link facts (mention another person's incident entities)

These belong to one person but share content with a *forget* entity's incident, which unlearning removes. In the base grid entanglement and target identity are the same thing; these separate them.

| cross dose | n | mean words |
|---|---|---|
| 0 | 598 | 16.7 |
| 1 | 100 | 16.3 |
| 2 | 100 | 17.0 |

emb_sim rises with content dose: **True** (expected: True).
emb_sim change from sharing the anchor, by dose: {'0': -0.0283, '1': -0.0202} (expected: near zero, because the anchor is an opaque codename; a large positive value would mean it leaks incident content).

## Tone check (P(negative), pinned sentiment classifier)

- incidents: mean 0.998, min 0.985 (required: >= 0.5)
- retain facts: mean 0.113, max 0.987 (required: <= 0.95)

## Provenance

- imported unchanged from an earlier build: 100 entities (`p000`–`p099`); their facts, training questions and evaluation questions are byte-identical to that build, except for the label repairs listed below
- generated in this run: 0 entities
- refreshed for every entity, imported or new: perturbations (Truth Ratio inputs), T4 probe terms, and sentiment scores

### Augmentation facts

`matched`: one fact per dose, every fact carrying the same number of named entities, so dose varies with entity count held constant. `shared`: an own entity shared with another retain fact of the same person and never with the incident, to test entity-sharing interference.

| condition | dose | n | mean words | mean named entities |
|---|---|---|---|---|
| matched | 0 | 100 | 17.1 | 3.0 |
| matched | 1 | 100 | 16.8 | 3.0 |
| matched | 2 | 100 | 17.0 | 3.0 |
| shared | 0 | 198 | 16.5 | 3.0 |

Augmented entities: 100 of 100.

## Invariants

| # | check | hard | result |
|---|---|---|---|
| 1 | exact grid counts per entity | True | ok |
| 2 | |E_f| = 3, A_f not in E_f, E_f disjoint from setting vocab | True | ok |
| 3 | no retain fact contains a negative keyword | True | ok |
| 3b | retain facts within the sentiment threshold (0.95) | False | 4 violation(s): ['p001_f05', 'p004_f01', 'p006_f03', 'p008_f14'] |
| 4 | measured equals assigned | True | ok |
| 5 | length band and own anchor absent from text | True | ok |
| 6 | every retain fact has an own entity | True | ok |
| 6b | cell mean lengths within 3.0 words (spread 1.7) | False | ok |
| 7 | every topic appears in every cell (per group) | False | ok |
| 9 | every fact has an anchor-free T5 pair | True | ok |
| 10 | eval questions present, distinct, and below the leak threshold | True | ok |
| 12 | incident anchors distinct across entities (ignoring generic head nouns) | True | ok |
| 13 | no person's name, anchor or incident entity appears in another person's facts or questions | True | ok |
| 14 | no given name or surname shared between entities | True | ok |
| 11 | no control-entity row in SF | True | ok |

## Cost

Generated through Claude Code on a subscription: the USD column is the API-equivalent figure Claude Code reports, not a bill. Subscription usage limits apply instead.

| task | calls | cached | failed | input tok | output tok | cost (USD) |
|---|---|---|---|---|---|---|
| eval_questions | 2 | 0 | 0 | 4 | 13849 | 0.23 |
| facts | 8 | 8 | 0 | 0 | 0 | 0.00 |
| perturbations | 2 | 0 | 0 | 4 | 6719 | 0.11 |
| training_forms | 1 | 1 | 0 | 0 | 0 | 0.00 |
