"""Load the one canonical language instruction for each RoboDyna task."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("task_instructions.json")


@lru_cache(maxsize=1)
def load_task_instructions() -> dict[str, str]:
    """Return the canonical task-to-instruction catalog."""
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError(f"Invalid task instruction catalog: {CATALOG_PATH}")
    return {str(task): str(instruction).strip() for task, instruction in tasks.items()}


def instruction_for(task: str) -> str:
    """Return one stable language instruction, or a readable task-name fallback."""
    task = str(task).strip().replace("-", "_")
    return load_task_instructions().get(task, task.replace("_", " "))
