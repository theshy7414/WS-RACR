"""The offline experiments must keep reproducing the numbers reported in the thesis.

``evaluation.run_all`` checks each report for its headline numbers; EXP2-EXP4 and the baseline
comparison call the production policy (``ws_racr.decision.collaboration_policy``).
"""
import pytest

from evaluation.common import production_risk_decision
from evaluation.datasets.exp3_pairs import PAIRS
from evaluation.run_all import EXPERIMENTS, run_experiment


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_headline_numbers(name):
    report, missing = run_experiment(EXPERIMENTS[name])
    assert report
    assert not missing, f"{name}: headline numbers changed: {missing}"


def test_exp3_production_policy_accuracy():
    """75 of the 80 annotated pairs are decided as annotated (93.75%)."""
    correct = sum(
        production_risk_decision(p.current_cmd, p.retrieved_cmd, p.ret_type, p.real_sim) == p.gt
        for p in PAIRS
    )
    assert (correct, len(PAIRS)) == (75, 80)
