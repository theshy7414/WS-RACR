"""Run the offline experiments and check their headline numbers.

    python -m evaluation.run_all               # run everything, print a summary
    python -m evaluation.run_all --only exp3   # one experiment
    python -m evaluation.run_all --out results # also write each full report to results/<name>.txt
    python -m evaluation.run_all --verbose     # echo the full reports

Everything is deterministic and needs only numpy (no GPU, camera, database or API key).
The exit code is non-zero if a headline number differs from the value reported in the thesis;
it can be used as a regression check (also covered by tests/test_evaluation_regression.py).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Experiment:
    name: str
    module: str
    title: str
    #: substrings that must appear in the report (the headline numbers)
    expected: Tuple[str, ...]


EXPERIMENTS: Dict[str, Experiment] = {
    e.name: e
    for e in (
        Experiment(
            "exp1", "evaluation.exp1_intent_routing",
            "EXP1 intent routing (100 commands): utterance-level 49.0% -> span-level 92.0%",
            ("routing accuracy : 49.0%", "routing accuracy : 92.0%"),
        ),
        Experiment(
            "exp2", "evaluation.exp2_feasibility_gate",
            "EXP2 feasibility gate (60 cases): unsafe execution 25.0% -> 0.0%",
            ("Unsafe Exec:     25.0% (naive)", "0.0% (new)"),
        ),
        Experiment(
            "exp3", "evaluation.exp3_execution_risk",
            "EXP3 execution risk (80 pairs): accuracy 48.8% -> 93.8%",
            ("48.8% (sim-only)", "93.8% (new rules)"),
        ),
        Experiment(
            "exp4", "evaluation.exp4_e2e_ablation",
            "EXP4 ablation (44 cases): 31.8% -> 56.8% -> 70.5% -> 93.2%",
            ("14/44  (31.8%)", "25/44  (56.8%)", "31/44  (70.5%)", "41/44  (93.2%)"),
        ),
        Experiment(
            "baselines", "evaluation.baseline_comparison",
            "Modelled baselines (no live LLM): Layer 1 92.0% vs 51.0%, Layer 3 93.8% vs 68.8%",
            ("Our pipeline ( 92.0%) vs LLM-only ( 51.0%)", "Our rules ( 93.8%) vs alternatives"),
        ),
    )
}


def run_experiment(exp: Experiment) -> Tuple[str, List[str]]:
    """Run one experiment; return its full report and the list of missing headline strings."""
    module = importlib.import_module(exp.module)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        module.main()
    report = buffer.getvalue()
    missing = [s for s in exp.expected if s not in report]
    return report, missing


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(EXPERIMENTS), help="run a single experiment")
    parser.add_argument("--out", type=Path, help="directory for the full text reports")
    parser.add_argument("--verbose", action="store_true", help="print the full reports")
    args = parser.parse_args(argv)

    selected = [EXPERIMENTS[args.only]] if args.only else list(EXPERIMENTS.values())
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for exp in selected:
        report, missing = run_experiment(exp)
        if args.out:
            (args.out / f"{exp.name}.txt").write_text(report, encoding="utf-8")
        if args.verbose:
            print(report)
        status = "OK  " if not missing else "FAIL"
        print(f"[{status}] {exp.title}")
        for s in missing:
            print(f"        missing in report: {s!r}")
        failures += bool(missing)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
