"""Figures for Section 6.5: how far the representation moves, against how far the output moves.

Two figures, both on control entities' grid retain facts, where every model saw the data identically:

  fig_repr_ladder.pdf  across the step ladder (120-225), all three runs: (a) distance to the
                       oracle in seed-noise floor units for the output distribution and for the
                       layer-27 hidden state; (b) the along and (c) the perp component of the
                       hidden-state displacement. The output moves by two to three orders of
                       magnitude while the hidden state moves to at most about three floors.
  fig_repr_layers.pdf  the along/perp profile across depth at each run's matched checkpoint. In
                       layers 7-21 the three methods agree within a run and the runs disagree in
                       sign, so the early-layer "restorative" component follows the reference
                       direction of the run, not the unlearning method.

Estimators are the published ones, so every plotted point reproduces a table: representation
quantities as in Appendix F (per-question ratio against the or/or_seed1 pair), behavioural distance
as in Table hi-ladder (mean |lp_gold gap| over the six-pair floor for run 0, one pair otherwise).

    python scripts/repr_figures.py

Writes both PDFs and their data as CSV (the table view) into the paper directory.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmark_v2_small100_aug"
RUNS = {0: ROOT / "results" / "small100aug", 1: ROOT / "results" / "small100aug_ft1",
        2: ROOT / "results" / "small100aug_ft2"}
OUT = ROOT.parent / "graduation research project materials" / "paper"
STEPS = [120, 135, 150, 165, 180, 195, 210, 225]
MATCHED = {0: {"ga": 150, "gd": 180, "npo": 150}, 1: {"ga": 135, "gd": 120, "npo": 135},
           2: {"ga": 150, "gd": 180, "npo": 150}}
LAYERS = ["7", "14", "21", "26", "27", "28"]
METHODS = ("ga", "gd", "npo")
NAME = {"ga": "GA", "gd": "GD", "npo": "NPO"}
# Reference categorical palette, slots 1-3 (validated all-pairs, light mode); markers are the
# secondary encoding so the figure survives grayscale print and colour-vision differences.
COLOR = {"ga": "#2a78d6", "gd": "#eb6834", "npo": "#1baf7a"}
MARKER = {"ga": "o", "gd": "s", "npo": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RUN_STYLE = {0: "-", 1: "--", 2: ":"}


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7, width=0.6, length=3)


def main() -> None:
    facts = {f["fact_id"]: f for f in jl(DATA / "facts.jsonl")}
    ents = {e["entity_id"]: e["group"] for e in jl(DATA / "entities.jsonl")}
    evals = {r["qa_id"]: r for r in jl(DATA / "eval_qa.jsonl")}

    def control(qa):
        f = facts[evals[qa]["fact_id"]]
        a = f.get("assigned") or {}
        return (ents[f["entity_id"]] == "control" and f["role"] == "retain"
                and "cross_target" not in a and not a.get("augment"))

    hq = json.loads((RUNS[0] / "manifest.json").read_text(encoding="utf-8"))["hidden_questions"]
    mask = np.array([control(q) for q in hq])
    cq = [qa for qa, r in evals.items() if r["qtype"] == "T2" and control(qa)]

    def hidden(s, name, layer):
        p = RUNS[s] / f"hidden_{name}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return z[layer].astype(np.float64)[mask]

    def decompose(s, name, layer):
        """Appendix F's estimator: (along, perp, distance-in-floors) for one checkpoint."""
        ft, orc, or1, h = (hidden(s, n, layer) for n in ("ft", "or", "or_seed1", name))
        if h is None:
            return None
        d = orc - ft
        dn = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        floor = np.maximum(np.linalg.norm(or1 - orc, axis=1), 1e-12)
        u = h - ft
        al = (u * d).sum(1) / dn ** 2
        pe = np.linalg.norm(u - al[:, None] * d, axis=1) / dn
        di = np.linalg.norm(h - orc, axis=1) / floor
        return al.mean(), pe.mean(), di.mean()

    lp = {s: {p.stem[len("eval_"):]: {q["qa_id"]: q.get("lp_gold") for q in
                                      json.loads(p.read_text(encoding="utf-8"))["questions"]}
              for p in d.glob("eval_*.json")} for s, d in RUNS.items()}

    def gap(s, a, b):
        v = [abs(lp[s][a][q] - lp[s][b][q]) for q in cq
             if lp[s][a].get(q) is not None and lp[s][b].get(q) is not None]
        return sum(v) / len(v)

    bfloor = {}
    for s in RUNS:
        ors = sorted(m for m in lp[s] if m == "or" or m.startswith("or_seed"))
        bfloor[s] = np.mean([gap(s, a, b) for a, b in itertools.combinations(ors, 2)])

    # ---------------------------------------------------------------- data
    ladder = []  # (run, method, step, behav, repr, along, perp)
    for s in RUNS:
        for m in METHODS:
            for st in STEPS:
                name = f"{m}_st{st:04d}"
                dec = decompose(s, name, "27")
                if dec is None or name not in lp[s]:
                    raise SystemExit(f"missing {name} for run {s}; is the download complete?")
                ladder.append((s, m, st, gap(s, name, "or") / bfloor[s], dec[2], dec[0], dec[1]))
    layers = []  # (run, method, layer, along, perp, dist)
    for s in RUNS:
        for m in METHODS:
            for L in LAYERS:
                al, pe, di = decompose(s, f"{m}_st{MATCHED[s][m]:04d}", L)
                layers.append((s, m, int(L), al, pe, di))

    with open(OUT / "fig_repr_ladder.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "method", "step", "behavioural_distance_floors",
                    "representation_distance_floors_L27", "along_L27", "perp_L27"])
        w.writerows([(s, m, st, f"{b:.3f}", f"{r:.3f}", f"{a:.3f}", f"{p:.3f}")
                     for s, m, st, b, r, a, p in ladder])
    with open(OUT / "fig_repr_layers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "method", "layer", "along", "perp", "distance_floors"])
        w.writerows([(s, m, L, f"{a:.3f}", f"{p:.3f}", f"{d:.3f}") for s, m, L, a, p, d in layers])

    def series(m, col):
        """Per-step mean and min/max across the three runs."""
        arr = np.array([[row[col] for row in ladder if row[1] == m and row[0] == s] for s in RUNS])
        return arr.mean(0), arr.min(0), arr.max(0)

    plt.rcParams.update({"font.size": 7.5, "axes.labelcolor": INK, "text.color": INK,
                         "axes.titlesize": 8, "axes.titleweight": "bold", "legend.fontsize": 7})

    # ---------------------------------------------------------------- figure 1: the ladder
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.65), constrained_layout=True)
    ax = axes[0]
    style(ax)
    ax.set_yscale("log")
    ax.hlines(1, 112, 227, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    for m in METHODS:
        for col, ls in ((3, "-"), (4, "--")):
            mu, lo, hi = series(m, col)
            ax.fill_between(STEPS, lo, hi, color=COLOR[m], alpha=0.14, linewidth=0, zorder=2)
            ax.plot(STEPS, mu, color=COLOR[m], linestyle=ls, linewidth=1.5, marker=MARKER[m],
                    markersize=3.6, markeredgecolor="white", markeredgewidth=0.6, zorder=3,
                    label=NAME[m] if col == 3 else None)
    # Label the two measures and the floor in a right margin that holds no data, so no label can
    # sit on a line: solid = output distribution, dashed = layer-27 hidden state.
    ax.set_xlim(112, 254)
    for y, txt in ((700, "output\n(solid)"), (2.4, "hidden\nstate\n(dashed)")):
        ax.text(231, y, txt, fontsize=6.2, color=INK2, va="center", ha="left", linespacing=0.95)
    ax.text(STEPS[0], 0.9, "seed-noise floor", fontsize=6.2, color=INK2, va="top", ha="left")
    ax.set_ylim(0.35, 6000)
    ax.set_title("(a) distance to the oracle", loc="left")
    ax.set_ylabel("seed-noise floors (log)")

    for ax, col, title, ref, reflab in ((axes[1], 5, "(b) along: toward the oracle", 1.0,
                                          "oracle's position on this axis"),
                                         (axes[2], 6, "(c) perp: sideways", None, None)):
        style(ax)
        for m in METHODS:
            mu, lo, hi = series(m, col)
            ax.fill_between(STEPS, lo, hi, color=COLOR[m], alpha=0.14, linewidth=0, zorder=2)
            ax.plot(STEPS, mu, color=COLOR[m], linewidth=1.5, marker=MARKER[m], markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        if ref is not None:
            ax.axhline(ref, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
            ax.text(STEPS[0], ref - 0.03, reflab, ha="left", va="top", fontsize=6.5, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_ylabel(r"units of $\|h_{\mathrm{OR}}-h_{\mathrm{FT}}\|$")
    axes[1].set_ylim(0, 1.08)
    axes[2].set_ylim(0, None)
    for ax in axes:
        ax.set_xticks(STEPS[::2])
        ax.set_xlabel("optimizer step")
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], linewidth=1.5, markersize=3.6,
                          markeredgecolor="white", markeredgewidth=0.6, label=NAME[m]) for m in METHODS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.18, linewidth=0,
                                 label="range over three runs"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, handlelength=1.8)
    fig.savefig(OUT / "fig_repr_ladder.pdf")
    fig.savefig(OUT / "fig_repr_ladder.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- figure 2: the layer profile
    x = [0, 1, 2, 3.6, 4.6, 5.6]           # a visible break between 21 and 26
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), constrained_layout=True)
    for ax, col, title in ((axes[0], 3, "(a) along: toward the oracle"),
                           (axes[1], 4, "(b) perp: sideways")):
        style(ax)
        ax.axvspan(-0.35, 2.35, color="#f1f0ec", zorder=0, linewidth=0)
        for s in RUNS:
            for m in METHODS:
                ys = [row[col] for row in layers if row[0] == s and row[1] == m]
                ax.plot(x, ys, color=COLOR[m], linestyle=RUN_STYLE[s], linewidth=1.2,
                        marker=MARKER[m], markersize=3.2, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3)
        if col == 3:
            ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
            # Runs 0 and 1 form tight three-method bundles of opposite sign at layer 7; run 2's
            # methods spread from -0.09 to +0.31 there, so it gets no single label.
            for s in (0, 1):
                y = np.mean([row[3] for row in layers if row[0] == s and row[2] == 7])
                ax.text(-0.3, y, f"run {s}", fontsize=6.5, color=INK2, va="center", ha="right")
            ax.set_ylim(None, 0.43)
            ax.text(1.0, 0.415, "reference poorly conditioned", ha="center", va="top",
                    fontsize=6.3, color=INK2)
        else:
            ax.set_ylim(0.2, 1.0)
            ax.text(1.0, 0.985, "reference poorly\nconditioned", ha="center", va="top",
                    fontsize=6.3, color=INK2, linespacing=0.95)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYERS)
        ax.set_xlim(-1.1 if col == 3 else -0.35, 5.9)
        ax.set_xlabel("layer")
        ax.set_title(title, loc="left")
        ax.set_ylabel(r"units of $\|h_{\mathrm{OR}}-h_{\mathrm{FT}}\|$")
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], linewidth=1.2, markersize=3.6,
                          label=NAME[m]) for m in METHODS]
    handles += [plt.Line2D([], [], color=INK2, linestyle=RUN_STYLE[s], linewidth=1.0,
                           label=f"run {s}") for s in RUNS]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, handlelength=2.2,
               columnspacing=1.4)
    fig.savefig(OUT / "fig_repr_layers.pdf")
    fig.savefig(OUT / "fig_repr_layers.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- the numbers the text needs
    b = np.array([r[3] for r in ladder])
    rp = np.array([r[4] for r in ladder])
    al = np.array([r[5] for r in ladder])
    print(f"behavioural distance over the ladder: {b.min():.2f} to {b.max():.0f} floors")
    print(f"representation distance over the ladder: {rp.min():.2f} to {rp.max():.2f} floors")
    print(f"along over the ladder: {al.min():.3f} to {al.max():.3f}  (1 = landed on the oracle)")
    for m in METHODS:
        mu = series(m, 5)[0]
        print(f"  {NAME[m]:4} along mean by step: " + " ".join(f"{v:.2f}" for v in mu))
    print(f"wrote {OUT / 'fig_repr_ladder.pdf'} and {OUT / 'fig_repr_layers.pdf'} (+ .png, .csv)")


if __name__ == "__main__":
    main()
