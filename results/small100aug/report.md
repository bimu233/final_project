# Experiment report: small100aug

100 entities (75 forget, 25 control), 2798 facts, 14989 evaluation questions.

Knowledge accuracy = share of perturbed variants ranked below the true paraphrase (0.5 = chance). Distances from the oracle are in noise-floor units: 1.0 = the distance between two oracles that differ only by seed.

## 1. Sanity: did fine-tuning take, and does the oracle lack the incident?

| model | forget incident T2 ROUGE | forget incident knowledge | retain T2 ROUGE | retain knowledge | utility |
|---|---|---|---|---|---|
| base | 0.033 | 0.636 | 0.116 | 0.675 | 0.078 |
| ft | 0.929 | 0.947 | 0.954 | 0.931 | 0.364 |
| or | 0.450 | 0.860 | 0.955 | 0.931 | 0.393 |
| or_seed1 | 0.413 | 0.841 | 0.943 | 0.933 | 0.423 |

## 2. Forgetting and utility

Forget-set metrics are on the incidents of forget entities. Truth Ratio is a geometric mean; > 1 means the model prefers the perturbed answers. The oracle row is the target, not zero: entangled retain facts let OR partly reconstruct an incident answer.

| model | incident T1 ROUGE | incident T2 ROUGE | incident Truth Ratio | incident knowledge | retain T2 ROUGE | retain knowledge | utility |
|---|---|---|---|---|---|---|---|
| ft | 0.917 | 0.929 | 0.56 | 0.947 | 0.954 | 0.931 | 0.364 |
| or | 0.332 | 0.450 | 0.66 | 0.860 | 0.955 | 0.931 | 0.393 |
| ga_ep1 | 0.885 | 0.874 | 0.55 | 0.941 | 0.935 | 0.930 | 0.300 |
| ga_ep2 | 0.620 | 0.695 | 0.58 | 0.907 | 0.884 | 0.924 | 0.310 |
| ga_ep3 | 0.260 | 0.333 | 0.77 | 0.763 | 0.781 | 0.920 | 0.282 |
| ga_ep4 | 0.003 | 0.003 | 0.01 | 0.545 | 0.177 | 0.868 | 0.058 |
| ga_ep5 | 0.003 | 0.003 | 0.00 | 0.525 | 0.012 | 0.792 | 0.000 |
| ga_ep6 | 0.003 | 0.003 | 0.00 | 0.541 | 0.000 | 0.751 | 0.000 |
| gd_ep1 | 0.905 | 0.885 | 0.56 | 0.941 | 0.939 | 0.930 | 0.300 |
| gd_ep2 | 0.823 | 0.813 | 0.55 | 0.929 | 0.895 | 0.925 | 0.316 |
| gd_ep3 | 0.331 | 0.382 | 0.62 | 0.867 | 0.852 | 0.920 | 0.266 |
| gd_ep4 | 0.018 | 0.021 | 1.71 | 0.600 | 0.589 | 0.907 | 0.253 |
| gd_ep5 | 0.017 | 0.019 | 0.01 | 0.399 | 0.455 | 0.905 | 0.186 |
| gd_ep6 | 0.019 | 0.024 | 0.00 | 0.373 | 0.566 | 0.923 | 0.160 |
| npo_ep1 | 0.822 | 0.873 | 0.55 | 0.951 | 0.949 | 0.931 | 0.359 |
| npo_ep2 | 0.592 | 0.590 | 0.65 | 0.860 | 0.900 | 0.920 | 0.307 |
| npo_ep3 | 0.191 | 0.193 | 1.07 | 0.657 | 0.563 | 0.900 | 0.227 |
| npo_ep4 | 0.125 | 0.121 | 3.79 | 0.544 | 0.119 | 0.819 | 0.025 |
| npo_ep5 | 0.107 | 0.103 | 0.21 | 0.453 | 0.083 | 0.753 | 0.000 |
| npo_ep6 | 0.107 | 0.105 | 0.02 | 0.440 | 0.083 | 0.751 | 0.000 |

## 3. Retain damage by entanglement cell (T2 ROUGE, FT minus model)

Each cell: forget-entity damage / control-entity damage / DiD. Cells are (content_dose, anchor_share).

| model | (0, 0) | (1, 0) | (2, 0) | (0, 1) | (1, 1) |
|---|---|---|---|---|---|
| ga_ep1 | 0.01 / 0.01 / **-0.01** | 0.02 / 0.02 / **0.01** | 0.05 / 0.01 / **0.04** | 0.01 / -0.00 / **0.02** | 0.03 / 0.04 / **-0.01** |
| ga_ep2 | 0.05 / 0.03 / **0.02** | 0.07 / 0.10 / **-0.03** | 0.12 / 0.10 / **0.02** | 0.07 / 0.04 / **0.03** | 0.11 / 0.17 / **-0.06** |
| ga_ep3 | 0.13 / 0.12 / **0.02** | 0.18 / 0.19 / **-0.01** | 0.25 / 0.22 / **0.03** | 0.18 / 0.15 / **0.03** | 0.23 / 0.32 / **-0.09** |
| ga_ep4 | 0.74 / 0.77 / **-0.03** | 0.78 / 0.79 / **-0.01** | 0.82 / 0.81 / **0.01** | 0.77 / 0.81 / **-0.04** | 0.83 / 0.88 / **-0.05** |
| ga_ep5 | 0.93 / 0.94 / **-0.01** | 0.94 / 0.96 / **-0.02** | 0.96 / 0.95 / **0.01** | 0.94 / 0.94 / **-0.00** | 0.95 / 0.99 / **-0.05** |
| ga_ep6 | 0.94 / 0.96 / **-0.02** | 0.95 / 0.97 / **-0.02** | 0.97 / 0.96 / **0.01** | 0.95 / 0.95 / **-0.00** | 0.95 / 1.00 / **-0.05** |
| gd_ep1 | 0.01 / 0.01 / **-0.00** | 0.02 / 0.00 / **0.01** | 0.03 / 0.01 / **0.03** | 0.02 / 0.00 / **0.02** | 0.03 / 0.04 / **-0.01** |
| gd_ep2 | 0.04 / 0.04 / **0.00** | 0.06 / 0.06 / **-0.00** | 0.11 / 0.09 / **0.02** | 0.03 / 0.02 / **0.01** | 0.10 / 0.13 / **-0.03** |
| gd_ep3 | 0.08 / 0.08 / **-0.00** | 0.10 / 0.10 / **0.00** | 0.16 / 0.11 / **0.05** | 0.08 / 0.05 / **0.03** | 0.16 / 0.19 / **-0.03** |
| gd_ep4 | 0.29 / 0.31 / **-0.02** | 0.35 / 0.35 / **0.00** | 0.46 / 0.47 / **-0.01** | 0.41 / 0.35 / **0.06** | 0.52 / 0.49 / **0.03** |
| gd_ep5 | 0.43 / 0.42 / **0.01** | 0.50 / 0.53 / **-0.03** | 0.63 / 0.60 / **0.03** | 0.50 / 0.47 / **0.03** | 0.64 / 0.66 / **-0.02** |
| gd_ep6 | 0.30 / 0.28 / **0.02** | 0.42 / 0.46 / **-0.04** | 0.52 / 0.53 / **-0.01** | 0.34 / 0.40 / **-0.06** | 0.55 / 0.55 / **0.00** |
| npo_ep1 | 0.00 / 0.01 / **-0.00** | 0.01 / -0.00 / **0.01** | 0.01 / 0.00 / **0.01** | 0.01 / 0.00 / **0.01** | -0.00 / 0.02 / **-0.02** |
| npo_ep2 | 0.03 / 0.03 / **0.00** | 0.07 / 0.04 / **0.03** | 0.10 / 0.05 / **0.04** | 0.07 / 0.03 / **0.04** | 0.06 / 0.12 / **-0.05** |
| npo_ep3 | 0.35 / 0.33 / **0.02** | 0.38 / 0.37 / **0.01** | 0.48 / 0.53 / **-0.05** | 0.37 / 0.32 / **0.05** | 0.49 / 0.48 / **0.00** |
| npo_ep4 | 0.83 / 0.84 / **-0.01** | 0.83 / 0.85 / **-0.02** | 0.87 / 0.86 / **0.01** | 0.79 / 0.80 / **-0.00** | 0.84 / 0.86 / **-0.03** |
| npo_ep5 | 0.87 / 0.89 / **-0.01** | 0.86 / 0.88 / **-0.02** | 0.90 / 0.88 / **0.02** | 0.83 / 0.82 / **0.00** | 0.86 / 0.88 / **-0.02** |
| npo_ep6 | 0.87 / 0.89 / **-0.01** | 0.86 / 0.88 / **-0.02** | 0.90 / 0.88 / **0.01** | 0.83 / 0.82 / **0.00** | 0.86 / 0.88 / **-0.02** |

## 4. The person channel, per entity

Damage on clean (0,0) facts: these share no content and no anchor with the incident, only the person. Mean over entities with a 90% bootstrap interval over entities, plus how many forget entities exceed every control entity.

| model | forget entities | control entities | forget > all controls |
|---|---|---|---|
| ga_ep1 | 0.01 [-0.00, 0.02] | 0.01 [-0.00, 0.03] | 1/75 |
| ga_ep2 | 0.05 [0.03, 0.06] | 0.03 [0.01, 0.05] | 6/75 |
| ga_ep3 | 0.13 [0.12, 0.15] | 0.12 [0.09, 0.14] | 1/75 |
| ga_ep4 | 0.74 [0.72, 0.76] | 0.77 [0.75, 0.79] | 4/75 |
| ga_ep5 | 0.93 [0.92, 0.94] | 0.94 [0.92, 0.96] | 0/75 |
| ga_ep6 | 0.94 [0.93, 0.96] | 0.96 [0.94, 0.98] | 0/75 |
| gd_ep1 | 0.01 [0.00, 0.02] | 0.01 [-0.00, 0.02] | 4/75 |
| gd_ep2 | 0.04 [0.03, 0.05] | 0.04 [0.02, 0.06] | 2/75 |
| gd_ep3 | 0.08 [0.06, 0.09] | 0.08 [0.05, 0.11] | 2/75 |
| gd_ep4 | 0.29 [0.27, 0.31] | 0.31 [0.28, 0.34] | 5/75 |
| gd_ep5 | 0.43 [0.40, 0.45] | 0.42 [0.39, 0.44] | 10/75 |
| gd_ep6 | 0.30 [0.28, 0.33] | 0.28 [0.25, 0.31] | 13/75 |
| npo_ep1 | 0.00 [-0.00, 0.01] | 0.01 [-0.00, 0.02] | 2/75 |
| npo_ep2 | 0.03 [0.02, 0.05] | 0.03 [0.01, 0.05] | 1/75 |
| npo_ep3 | 0.35 [0.33, 0.37] | 0.33 [0.30, 0.36] | 8/75 |
| npo_ep4 | 0.83 [0.81, 0.84] | 0.84 [0.82, 0.86] | 4/75 |
| npo_ep5 | 0.87 [0.86, 0.89] | 0.89 [0.87, 0.90] | 4/75 |
| npo_ep6 | 0.87 [0.86, 0.89] | 0.89 [0.87, 0.90] | 3/75 |

## 5. Damage channels

- Category: change on control entities' own (never targeted) incidents.
- Content dose: DiD at (2,0) minus DiD at (0,0). Anchor: DiD at (0,1) minus DiD at (0,0).
- Per-entity columns count forget entities with a positive effect, which the cell means can hide.

| model | category: control incident drop | content-dose slope | dose positive per entity | anchor effect | anchor positive per entity |
|---|---|---|---|---|---|
| ga_ep1 | 0.01 | 0.05 | 27/75 | 0.02 | 14/75 |
| ga_ep2 | 0.21 | 0.01 | 35/75 | 0.01 | 26/75 |
| ga_ep3 | 0.58 | 0.01 | 46/75 | 0.02 | 36/75 |
| ga_ep4 | 0.93 | 0.04 | 53/75 | -0.01 | 46/75 |
| ga_ep5 | 0.93 | 0.02 | 51/75 | 0.01 | 54/75 |
| ga_ep6 | 0.93 | 0.03 | 39/75 | 0.01 | 40/75 |
| gd_ep1 | -0.00 | 0.03 | 20/75 | 0.02 | 12/75 |
| gd_ep2 | 0.08 | 0.02 | 32/75 | 0.01 | 17/75 |
| gd_ep3 | 0.50 | 0.05 | 42/75 | 0.03 | 30/75 |
| gd_ep4 | 0.91 | 0.01 | 53/75 | 0.08 | 51/75 |
| gd_ep5 | 0.91 | 0.03 | 59/75 | 0.02 | 45/75 |
| gd_ep6 | 0.91 | -0.03 | 64/75 | -0.08 | 41/75 |
| npo_ep1 | -0.01 | 0.01 | 8/75 | 0.01 | 9/75 |
| npo_ep2 | 0.34 | 0.04 | 40/75 | 0.04 | 26/75 |
| npo_ep3 | 0.70 | -0.07 | 54/75 | 0.03 | 41/75 |
| npo_ep4 | 0.81 | 0.02 | 52/75 | 0.01 | 36/75 |
| npo_ep5 | 0.82 | 0.03 | 47/75 | 0.01 | 28/75 |
| npo_ep6 | 0.82 | 0.03 | 47/75 | 0.02 | 28/75 |

## 6. Surface suppression vs. knowledge (forget-entity retain facts)

Generation damage (T2 ROUGE drop) against belief damage (knowledge-accuracy drop) on the same facts. If generation collapses while knowledge accuracy holds, the damage is in output surface form.

| model | entangled: ROUGE drop | entangled: knowledge drop | clean: ROUGE drop | clean: knowledge drop |
|---|---|---|---|---|
| ga_ep1 | 0.03 | 0.00 | 0.01 | -0.00 |
| ga_ep2 | 0.09 | 0.01 | 0.05 | 0.01 |
| ga_ep3 | 0.21 | 0.01 | 0.13 | 0.01 |
| ga_ep4 | 0.80 | 0.07 | 0.74 | 0.05 |
| ga_ep5 | 0.95 | 0.15 | 0.93 | 0.12 |
| ga_ep6 | 0.96 | 0.19 | 0.94 | 0.16 |
| gd_ep1 | 0.02 | 0.00 | 0.01 | -0.00 |
| gd_ep2 | 0.08 | 0.01 | 0.04 | 0.00 |
| gd_ep3 | 0.13 | 0.01 | 0.08 | 0.01 |
| gd_ep4 | 0.43 | 0.02 | 0.29 | 0.03 |
| gd_ep5 | 0.57 | 0.02 | 0.43 | 0.02 |
| gd_ep6 | 0.46 | -0.00 | 0.30 | 0.01 |
| npo_ep1 | 0.00 | 0.00 | 0.00 | -0.00 |
| npo_ep2 | 0.08 | 0.01 | 0.03 | 0.01 |
| npo_ep3 | 0.43 | 0.03 | 0.35 | 0.03 |
| npo_ep4 | 0.84 | 0.12 | 0.83 | 0.09 |
| npo_ep5 | 0.87 | 0.19 | 0.87 | 0.15 |
| npo_ep6 | 0.86 | 0.19 | 0.87 | 0.16 |

## 7. Distance from the oracle: behavior (gray-box)

Mean |log P(gold answer) - the oracle's| on T2 questions, divided by the same quantity for a second-seed oracle. > 1 means the model is a more distinguishable from OR than seed noise explains. Control-entity rows are the clean test: UL and OR saw identical data for those people.

| model | forget incident | forget retain | control |
|---|---|---|---|
| ft | 5.38 | 0.83 | 1.09 |
| ga_ep1 | 3.43 | 1.02 | 1.36 |
| ga_ep2 | 5.18 | 2.34 | 3.00 |
| ga_ep3 | 35.70 | 6.66 | 8.70 |
| ga_ep4 | 224.96 | 268.80 | 407.36 |
| ga_ep5 | 360.75 | 1233.71 | 1868.19 |
| ga_ep6 | 403.62 | 2260.82 | 3420.74 |
| gd_ep1 | 3.58 | 0.95 | 1.31 |
| gd_ep2 | 3.81 | 1.66 | 2.25 |
| gd_ep3 | 10.65 | 2.82 | 3.59 |
| gd_ep4 | 102.20 | 34.06 | 49.87 |
| gd_ep5 | 230.87 | 52.19 | 75.58 |
| gd_ep6 | 279.44 | 26.28 | 36.91 |
| npo_ep1 | 4.51 | 0.83 | 1.06 |
| npo_ep2 | 7.40 | 2.05 | 2.43 |
| npo_ep3 | 41.68 | 32.74 | 48.50 |
| npo_ep4 | 135.19 | 604.60 | 905.87 |
| npo_ep5 | 215.71 | 1421.03 | 2144.11 |
| npo_ep6 | 224.54 | 1507.92 | 2277.14 |

## 8. Distance from the oracle: representation (layer 14)

Median distance to OR in noise-floor units. `along` = fraction of the FT→OR gap closed along that axis; `perp` = movement orthogonal to it, in units of the same gap. along > 0 with perp dominating means partly restorative but mostly sideways.

| model | forget incident | forget retain | control | along | perp |
|---|---|---|---|---|---|
| ft | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| ga_ep1 | 1.02 | 1.01 | 1.01 | -0.00 | 0.15 |
| ga_ep2 | 0.95 | 0.94 | 0.94 | 0.16 | 0.42 |
| ga_ep3 | 1.09 | 1.08 | 1.08 | 0.00 | 0.41 |
| ga_ep4 | 1.09 | 1.06 | 1.06 | 0.20 | 0.70 |
| ga_ep5 | 1.15 | 1.11 | 1.11 | 0.18 | 0.75 |
| ga_ep6 | 1.19 | 1.15 | 1.15 | 0.24 | 0.86 |
| gd_ep1 | 1.02 | 1.01 | 1.01 | -0.00 | 0.14 |
| gd_ep2 | 1.11 | 1.11 | 1.11 | -0.05 | 0.35 |
| gd_ep3 | 1.20 | 1.19 | 1.19 | -0.10 | 0.45 |
| gd_ep4 | 1.49 | 1.47 | 1.48 | -0.26 | 0.77 |
| gd_ep5 | 1.73 | 1.72 | 1.72 | -0.39 | 1.03 |
| gd_ep6 | 1.63 | 1.62 | 1.62 | -0.34 | 0.91 |
| npo_ep1 | 0.98 | 0.97 | 0.97 | 0.03 | 0.13 |
| npo_ep2 | 1.17 | 1.17 | 1.17 | -0.13 | 0.32 |
| npo_ep3 | 1.20 | 1.19 | 1.19 | -0.13 | 0.40 |
| npo_ep4 | 1.24 | 1.23 | 1.23 | -0.12 | 0.52 |
| npo_ep5 | 1.25 | 1.24 | 1.23 | -0.08 | 0.61 |
| npo_ep6 | 1.26 | 1.24 | 1.24 | -0.08 | 0.61 |

## 8. Distance from the oracle: representation (layer 21)

Median distance to OR in noise-floor units. `along` = fraction of the FT→OR gap closed along that axis; `perp` = movement orthogonal to it, in units of the same gap. along > 0 with perp dominating means partly restorative but mostly sideways.

| model | forget incident | forget retain | control | along | perp |
|---|---|---|---|---|---|
| ft | 1.04 | 1.04 | 1.04 | 0.00 | 0.00 |
| ga_ep1 | 1.08 | 1.06 | 1.06 | -0.01 | 0.17 |
| ga_ep2 | 1.08 | 1.04 | 1.04 | 0.11 | 0.44 |
| ga_ep3 | 1.33 | 1.20 | 1.20 | 0.03 | 0.61 |
| ga_ep4 | 1.84 | 1.36 | 1.36 | 0.18 | 1.01 |
| ga_ep5 | 2.16 | 1.53 | 1.53 | 0.15 | 1.19 |
| ga_ep6 | 2.39 | 1.65 | 1.65 | 0.18 | 1.35 |
| gd_ep1 | 1.07 | 1.06 | 1.06 | -0.00 | 0.16 |
| gd_ep2 | 1.17 | 1.14 | 1.14 | -0.02 | 0.40 |
| gd_ep3 | 1.28 | 1.23 | 1.22 | -0.03 | 0.57 |
| gd_ep4 | 1.68 | 1.49 | 1.49 | -0.11 | 0.90 |
| gd_ep5 | 1.92 | 1.67 | 1.66 | -0.18 | 1.08 |
| gd_ep6 | 1.83 | 1.56 | 1.55 | -0.17 | 0.93 |
| npo_ep1 | 1.04 | 1.03 | 1.03 | 0.02 | 0.11 |
| npo_ep2 | 1.19 | 1.17 | 1.17 | -0.07 | 0.33 |
| npo_ep3 | 1.31 | 1.24 | 1.24 | -0.06 | 0.55 |
| npo_ep4 | 1.58 | 1.40 | 1.41 | -0.05 | 0.84 |
| npo_ep5 | 1.88 | 1.56 | 1.56 | -0.03 | 1.08 |
| npo_ep6 | 1.87 | 1.56 | 1.56 | -0.03 | 1.08 |

## 8. Distance from the oracle: representation (layer 27)

Median distance to OR in noise-floor units. `along` = fraction of the FT→OR gap closed along that axis; `perp` = movement orthogonal to it, in units of the same gap. along > 0 with perp dominating means partly restorative but mostly sideways.

| model | forget incident | forget retain | control | along | perp |
|---|---|---|---|---|---|
| ft | 1.06 | 1.05 | 1.05 | 0.00 | 0.00 |
| ga_ep1 | 1.09 | 1.07 | 1.07 | -0.01 | 0.17 |
| ga_ep2 | 1.07 | 1.04 | 1.03 | 0.10 | 0.40 |
| ga_ep3 | 1.33 | 1.13 | 1.12 | 0.26 | 0.77 |
| ga_ep4 | 2.39 | 1.48 | 1.48 | 0.45 | 1.30 |
| ga_ep5 | 2.92 | 1.68 | 1.68 | 0.50 | 1.52 |
| ga_ep6 | 3.34 | 1.85 | 1.85 | 0.54 | 1.71 |
| gd_ep1 | 1.08 | 1.06 | 1.06 | 0.00 | 0.16 |
| gd_ep2 | 1.06 | 1.04 | 1.03 | 0.10 | 0.39 |
| gd_ep3 | 1.18 | 1.08 | 1.07 | 0.20 | 0.64 |
| gd_ep4 | 1.71 | 1.28 | 1.27 | 0.31 | 1.00 |
| gd_ep5 | 1.76 | 1.28 | 1.28 | 0.25 | 0.95 |
| gd_ep6 | 1.48 | 1.20 | 1.20 | 0.09 | 0.69 |
| npo_ep1 | 1.07 | 1.06 | 1.05 | -0.00 | 0.09 |
| npo_ep2 | 1.04 | 1.02 | 1.02 | 0.12 | 0.39 |
| npo_ep3 | 1.30 | 1.16 | 1.16 | 0.27 | 0.83 |
| npo_ep4 | 2.12 | 1.61 | 1.62 | 0.44 | 1.44 |
| npo_ep5 | 3.04 | 2.12 | 2.14 | 0.56 | 1.99 |
| npo_ep6 | 2.94 | 2.08 | 2.10 | 0.54 | 1.95 |

## 8. Distance from the oracle: representation (layer 28)

Median distance to OR in noise-floor units. `along` = fraction of the FT→OR gap closed along that axis; `perp` = movement orthogonal to it, in units of the same gap. along > 0 with perp dominating means partly restorative but mostly sideways.

| model | forget incident | forget retain | control | along | perp |
|---|---|---|---|---|---|
| ft | 0.92 | 0.93 | 0.93 | 0.00 | 0.00 |
| ga_ep1 | 0.95 | 0.95 | 0.94 | 0.01 | 0.23 |
| ga_ep2 | 1.06 | 1.03 | 1.02 | 0.05 | 0.54 |
| ga_ep3 | 1.35 | 1.20 | 1.20 | 0.10 | 0.92 |
| ga_ep4 | 2.27 | 1.73 | 1.73 | 0.18 | 1.66 |
| ga_ep5 | 3.07 | 2.26 | 2.27 | 0.22 | 2.30 |
| ga_ep6 | 3.59 | 2.71 | 2.72 | 0.22 | 2.81 |
| gd_ep1 | 0.94 | 0.95 | 0.94 | 0.01 | 0.21 |
| gd_ep2 | 1.03 | 1.01 | 1.00 | 0.05 | 0.52 |
| gd_ep3 | 1.20 | 1.11 | 1.10 | 0.11 | 0.78 |
| gd_ep4 | 1.72 | 1.37 | 1.37 | 0.17 | 1.21 |
| gd_ep5 | 2.00 | 1.49 | 1.49 | 0.16 | 1.36 |
| gd_ep6 | 1.85 | 1.38 | 1.38 | 0.05 | 1.14 |
| npo_ep1 | 0.93 | 0.93 | 0.93 | 0.00 | 0.11 |
| npo_ep2 | 0.95 | 0.95 | 0.95 | 0.06 | 0.40 |
| npo_ep3 | 1.22 | 1.14 | 1.15 | 0.15 | 0.88 |
| npo_ep4 | 2.17 | 1.86 | 1.87 | 0.28 | 1.88 |
| npo_ep5 | 2.91 | 2.44 | 2.45 | 0.34 | 2.56 |
| npo_ep6 | 2.92 | 2.46 | 2.47 | 0.33 | 2.57 |

## 9. Free-form probes (T4)

Avoid terms are incident-only: material that appears in no retain fact of that entity, so the oracle is a meaningful floor.

| model | include rate | forget entities: avoid hits per probe |
|---|---|---|
| ft | 0.47 | 1.24 |
| or | 0.46 | 0.39 |
| ga_ep1 | 0.47 | 1.00 |
| ga_ep2 | 0.44 | 0.41 |
| ga_ep3 | 0.35 | 0.10 |
| ga_ep4 | 0.09 | 0.07 |
| ga_ep5 | 0.00 | 0.07 |
| ga_ep6 | 0.00 | 0.07 |
| gd_ep1 | 0.47 | 1.15 |
| gd_ep2 | 0.45 | 1.00 |
| gd_ep3 | 0.42 | 0.23 |
| gd_ep4 | 0.24 | 0.00 |
| gd_ep5 | 0.15 | 0.00 |
| gd_ep6 | 0.23 | 0.00 |
| npo_ep1 | 0.47 | 0.96 |
| npo_ep2 | 0.44 | 0.52 |
| npo_ep3 | 0.28 | 0.01 |
| npo_ep4 | 0.00 | 0.00 |
| npo_ep5 | 0.00 | 0.00 |
| npo_ep6 | 0.00 | 0.00 |

## 10. Cross-link facts: is the damage specific to the removed fact?

Cross-link facts mention a *forget* entity's incident entities, so their shared content really is removed; the entity's own dose facts mention its own incident. For control entities that own incident was never unlearned, which makes their row the decisive one: if entanglement damage were specific to the removed fact, cross > own there. Cross-dose 0 is the same new topics with no shared content, so it is the within-topic baseline. All numbers are T2 ROUGE damage (FT minus model).

| model | group | cross0 (baseline) | own dose 1-2 | cross dose 1-2 | cross − own | entities cross > own |
|---|---|---|---|---|---|---|
| ga_ep1 | control | 0.02 | 0.01 | 0.00 | **-0.01** | 2/25 |
| ga_ep1 | forget | -0.01 | 0.04 | -0.00 | **-0.04** | 3/75 |
| ga_ep2 | control | 0.06 | 0.10 | 0.07 | **-0.03** | 5/25 |
| ga_ep2 | forget | -0.01 | 0.10 | 0.03 | **-0.07** | 14/75 |
| ga_ep3 | control | 0.10 | 0.20 | 0.15 | **-0.06** | 8/25 |
| ga_ep3 | forget | 0.05 | 0.21 | 0.07 | **-0.14** | 14/75 |
| ga_ep4 | control | 0.45 | 0.80 | 0.61 | **-0.19** | 3/25 |
| ga_ep4 | forget | 0.46 | 0.80 | 0.51 | **-0.28** | 7/75 |
| ga_ep5 | control | 0.77 | 0.95 | 0.93 | **-0.02** | 9/25 |
| ga_ep5 | forget | 0.80 | 0.95 | 0.81 | **-0.14** | 23/75 |
| ga_ep6 | control | 0.83 | 0.96 | 0.97 | **0.01** | 6/25 |
| ga_ep6 | forget | 0.90 | 0.96 | 0.87 | **-0.09** | 20/75 |
| gd_ep1 | control | 0.02 | 0.01 | 0.00 | **-0.01** | 2/25 |
| gd_ep1 | forget | -0.01 | 0.02 | -0.00 | **-0.03** | 5/75 |
| gd_ep2 | control | 0.04 | 0.08 | 0.05 | **-0.02** | 5/25 |
| gd_ep2 | forget | 0.01 | 0.08 | 0.02 | **-0.06** | 11/75 |
| gd_ep3 | control | 0.05 | 0.11 | 0.10 | **-0.00** | 8/25 |
| gd_ep3 | forget | 0.02 | 0.13 | 0.04 | **-0.09** | 13/75 |
| gd_ep4 | control | 0.19 | 0.41 | 0.26 | **-0.15** | 8/25 |
| gd_ep4 | forget | 0.14 | 0.41 | 0.16 | **-0.24** | 16/75 |
| gd_ep5 | control | 0.24 | 0.57 | 0.40 | **-0.17** | 7/25 |
| gd_ep5 | forget | 0.22 | 0.57 | 0.29 | **-0.28** | 9/75 |
| gd_ep6 | control | 0.12 | 0.50 | 0.25 | **-0.24** | 4/25 |
| gd_ep6 | forget | 0.16 | 0.47 | 0.16 | **-0.31** | 7/75 |
| npo_ep1 | control | 0.03 | -0.00 | 0.00 | **0.00** | 1/25 |
| npo_ep1 | forget | 0.00 | 0.01 | -0.01 | **-0.01** | 5/75 |
| npo_ep2 | control | 0.03 | 0.05 | 0.08 | **0.03** | 9/25 |
| npo_ep2 | forget | 0.01 | 0.08 | 0.00 | **-0.08** | 6/75 |
| npo_ep3 | control | 0.09 | 0.45 | 0.36 | **-0.09** | 8/25 |
| npo_ep3 | forget | 0.21 | 0.43 | 0.23 | **-0.20** | 14/75 |
| npo_ep4 | control | 0.72 | 0.86 | 0.85 | **-0.00** | 12/25 |
| npo_ep4 | forget | 0.76 | 0.85 | 0.73 | **-0.12** | 23/75 |
| npo_ep5 | control | 0.74 | 0.88 | 0.89 | **0.01** | 15/25 |
| npo_ep5 | forget | 0.81 | 0.88 | 0.77 | **-0.11** | 30/75 |
| npo_ep6 | control | 0.75 | 0.88 | 0.90 | **0.01** | 15/25 |
| npo_ep6 | forget | 0.81 | 0.88 | 0.77 | **-0.11** | 28/75 |

## Appendix: T5 yes/no probe (retired)

Kept for the record. Knowledge trained in question→answer form does not surface in yes/no verification for this model, so the probe sits at chance before any unlearning and cannot register a drop.

| model | forget incident | retain |
|---|---|---|
| ft | 0.62 | 0.47 |
| or | 0.55 | 0.51 |
| ga_ep1 | 0.56 | 0.47 |
| ga_ep2 | 0.49 | 0.49 |
| ga_ep3 | 0.57 | 0.49 |
| ga_ep4 | 0.57 | 0.50 |
| ga_ep5 | 0.49 | 0.50 |
| ga_ep6 | 0.61 | 0.49 |
| gd_ep1 | 0.60 | 0.49 |
| gd_ep2 | 0.55 | 0.49 |
| gd_ep3 | 0.59 | 0.49 |
| gd_ep4 | 0.49 | 0.50 |
| gd_ep5 | 0.44 | 0.50 |
| gd_ep6 | 0.56 | 0.51 |
| npo_ep1 | 0.63 | 0.48 |
| npo_ep2 | 0.59 | 0.49 |
| npo_ep3 | 0.60 | 0.49 |
| npo_ep4 | 0.53 | 0.50 |
| npo_ep5 | 0.57 | 0.51 |
| npo_ep6 | 0.56 | 0.52 |
