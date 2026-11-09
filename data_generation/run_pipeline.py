"""Generate the benchmark.

    python run_pipeline.py                       # full run from config.yaml
    python run_pipeline.py --pilot               # 5 entities into ../data/benchmark_v2_pilot
    python run_pipeline.py --backend mock --n-entities 8 --output-dir /tmp/bench   # offline smoke test

    # grow an existing build to 20 entities, keeping the 10 already there
    python run_pipeline.py --extend ../data/benchmark_v2_small10 --n-entities 20 \
        --output-dir ../data/benchmark_v2_small20

Re-running is safe: every LLM response is cached, so an interrupted run resumes for free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchgen.config import load_config
from benchgen.envfile import load_env
from benchgen.llm import LLMFatalError
from benchgen.pipeline import run

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "config.yaml", type=Path)
    ap.add_argument("--pilot", action="store_true", help="5 entities into a separate pilot output dir")
    ap.add_argument("--n-entities", type=int)
    ap.add_argument("--backend", choices=["claude_code", "claude", "mock"],
                    help="claude_code = subscription via `claude -p`; claude = API credits; mock = offline")
    ap.add_argument("--model")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--output-dir")
    ap.add_argument("--extend", type=Path,
                    help="grow a finished build: its entities are imported unchanged and only the "
                         "remaining ids up to --n-entities are generated")
    ap.add_argument("--refresh", action="store_true",
                    help="with --repair: also regenerate perturbations, probe terms and tone scores under the "
                         "current rules (cached where prompts are unchanged)")
    ap.add_argument("--reuse-repairs", type=Path,
                    help="label_repairs.json from another build; reuse its replacement labels where the same "
                         "entity needs the same repair, so nested builds stay consistent")
    ap.add_argument("--augment", type=Path,
                    help="add augmentation facts to a finished build: a matched-dose triplet with the entity "
                         "count held constant, plus shared-entity facts. Additive and resumable")
    ap.add_argument("--limit", type=int,
                    help="with --augment: generate at most this many new entities in this run")
    ap.add_argument("--cross-extend", type=Path,
                    help="add cross-link facts to a finished build: retain facts that mention another person's "
                         "incident entities. Each entity is saved as soon as it is generated, so the run resumes")
    ap.add_argument("--repair", type=Path,
                    help="re-package a finished build after making incident anchors and name parts unique "
                         "across entities; nothing else is regenerated")
    ap.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    args = ap.parse_args()

    loaded = []
    for env_path in (HERE / ".env", HERE.parent / ".env"):  # nearest file wins; the shell beats both
        loaded += [f"{key} ({env_path.parent.name}/.env)" for key in load_env(env_path)]
    if loaded:
        print(f"loaded: {', '.join(loaded)}")  # names only, never values

    overrides = {
        "n_entities": 5 if args.pilot and args.n_entities is None else args.n_entities,
        "output_dir": args.output_dir or ("../data/benchmark_v2_pilot" if args.pilot else None),
        "llm.backend": args.backend,
        "llm.model": args.model,
        "llm.effort": args.effort,
        "llm.max_workers": args.workers,
        "llm.fallbacks": False if args.no_fallbacks else None,
    }
    cfg = load_config(args.config, overrides)
    try:
        report = run(cfg, args.config.resolve(), extend=args.extend.resolve() if args.extend else None,
                     repair=args.repair.resolve() if args.repair else None,
                     reuse_repairs=args.reuse_repairs.resolve() if args.reuse_repairs else None,
                     refresh=args.refresh,
                     cross_extend=args.cross_extend.resolve() if args.cross_extend else None,
                     augment=args.augment.resolve() if args.augment else None, limit=args.limit)
    except LLMFatalError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 2
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
