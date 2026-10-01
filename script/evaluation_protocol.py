"""Shared fixed-seed protocol setup for local and client policy evaluators."""
from __future__ import annotations

from interactive.experiment_config import evaluation_seeds_for, evaluation_suite
from task_config.scenario_overrides import apply_base_scenario


def configure_evaluation(
    task: str,
    args: dict,
    scenario: str | None = None,
) -> tuple[str, str | None, list[int]]:
    """Apply one task scenario and return its shared ten-seed evaluation suite.

    Base tasks default to ``default`` and apply precisely the same feature
    overrides as the human GUI. Household tasks have no variants.
    """
    suite = evaluation_suite(task)
    task = str(task).strip().replace("-", "_")
    if suite == "base":
        scenario = apply_base_scenario(args, task, scenario)
    else:
        scenario = None

    seeds = evaluation_seeds_for(task, scenario)
    if not seeds:
        label = scenario or "default"
        raise ValueError(
            f"No fixed evaluation seeds configured for {suite}/{task}/{label} "
            "in task_config/eval_seeds.yml."
        )
    return suite, scenario, seeds
