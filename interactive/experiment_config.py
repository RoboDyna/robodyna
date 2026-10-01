"""Load ``interactive/experiment.yml`` for the human-experiment GUI."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "experiment.yml"
CONFIG_ENV = "ROBODYNA_EXPERIMENT_CONFIG"
EVALUATION_SEEDS_PATH = Path(__file__).resolve().parents[1] / "task_config" / "eval_seeds.yml"
EVALUATION_SEEDS_ENV = "ROBODYNA_EVAL_SEEDS_CONFIG"

BASE_SCENARIOS = ("default", "opt1", "opt2", "opt1+2")

# 1-based card numbers, matching base_task_gui.TASKS / household_task_gui.TASKS.
# Tutorial is 00 and is not part of these lists.
BASE_TASK_TABLE: tuple[tuple[int, str, str], ...] = (
    (1, "catch_marbles_trapdoors", "Catch Marbles Trapdoors"),
    (2, "catch_ramp_ball", "Catch Ramp Ball"),
    (3, "catch_cuboid", "Catch Cuboid"),
    (4, "catch_shelf_marble", "Catch Shelf Marble"),
    (5, "catch_valley_ball", "Catch Valley Ball"),
    (6, "stop_valley_ball", "Stop Valley Ball"),
    (7, "cook_meat", "Cook Meat"),
    (8, "cook_meat_timer", "Cook Meat Timer"),
    (9, "put_cup_belt", "Put Cup Belt"),
    (10, "dispense_gummy", "Dispense Gummy"),
    (11, "punch_dual_holes", "Punch Dual Holes"),
    (12, "save_goal", "Save Goal"),
    (13, "hit_target", "Hit Target"),
    (14, "load_train", "Load Train"),
    (15, "marble_shelf_maze", "Marble Shelf Maze"),
    (16, "pack_fruits", "Pack Fruits"),
    (17, "pick_ripe_apple", "Pick Ripe Apple"),
    (18, "place_block_belt", "Place Block Belt"),
    (19, "play_billiard", "Play Billiard"),
    (20, "control_quality", "Control Quality"),
    (21, "drop_ball_hole", "Drop Ball Hole"),
    (22, "sort_apples_belt", "Sort Apples Belt"),
    (23, "whack_moles", "Whack Moles"),
)
HOUSEHOLD_TASK_TABLE: tuple[tuple[int, str, str], ...] = (
    (1, "trap_bug", "Trap Bug"),
    (2, "boil_milk", "Boil Milk"),
    (3, "fill_coffee_jar", "Fill Coffee Jar"),
    (4, "pour_beer", "Pour Beer"),
    (5, "cook_food", "Cook Food"),
    (6, "cook_food_timer", "Cook Food Timer"),
    (7, "measure_ingredient", "Measure Ingredient"),
    (8, "make_soup", "Make Soup"),
    (9, "catch_cup", "Catch Cup"),
    (10, "catch_mouse_object_drop", "Catch Mouse Object Drop"),
    (11, "stop_ball", "Stop Ball"),
    (12, "clean_table", "Clean Table"),
)

# Default experiment set: all base except cook_meat / punch_dual_holes,
# all household except fill_coffee_jar / cook_food.
DEFAULT_BASE_TASKS = [n for n, key, _ in BASE_TASK_TABLE if key not in (
    "cook_meat",
    "punch_dual_holes",
)]
DEFAULT_HOUSEHOLD_TASKS = [n for n, key, _ in HOUSEHOLD_TASK_TABLE if key not in (
    "fill_coffee_jar",
    "cook_food",
)]

# 1-based base card numbers. Task 15 (marble_shelf_maze) is in two groups.
# (key, display label, task numbers)
DEFAULT_BASE_TASK_CATEGORIES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("motion_prediction", "Motion prediction", (2, 6, 12, 15)),
    ("state_transition", "State transition", (8, 16, 17, 22)),
    ("dynamic_pattern", "Dynamic pattern", (1, 3, 21, 23)),
    ("dynamic_avoidance", "Dynamic avoidance", (9, 13, 14, 18)),
    ("spatial_reasoning", "Spatial reasoning", (4, 10, 15, 19)),
)

DEFAULT_HOUSEHOLD_TASK_CATEGORIES: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("easy", "Easy", (2, 4, 6, 11, 12)),
    ("hard", "Hard", (1, 7, 8, 9, 10)),
)

_CATEGORY_LABELS = {
    key: label
    for key, label, _nums in (
        *DEFAULT_BASE_TASK_CATEGORIES,
        *DEFAULT_HOUSEHOLD_TASK_CATEGORIES,
    )
}

_SCENARIO_ALIASES = {
    "default": "default",
    "opt1": "opt1",
    "opt 1": "opt1",
    "option1": "opt1",
    "opt2": "opt2",
    "opt 2": "opt2",
    "option2": "opt2",
    "opt1+2": "opt1+2",
    "opt1+opt2": "opt1+2",
    "opt 1+2": "opt1+2",
    "opt1 + 2": "opt1+2",
    "opt12": "opt1+2",
    "opt1_2": "opt1+2",
    "opt1_opt2": "opt1+2",
    "option1+2": "opt1+2",
}


def _as_positive_int(value, default: int) -> int:
    if value is None or value == "":
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    if number <= 0:
        return int(default)
    return number


def _as_count_int(value, default: int) -> int:
    """Like ``_as_positive_int`` but ``0`` is meaningful (skip that suite)."""
    if value is None or value == "":
        return int(default)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, number)


def pick_lowest_count(pool: list[int], counts: dict[int, int], rng) -> int | None:
    """Choose uniformly among pool members that currently have the lowest count.

    Example: if 2 has been used once and 6, 12, 15 are still unused, the next
    pick is random among 6, 12, and 15.
    """
    values = [int(n) for n in pool]
    if not values:
        return None
    min_count = min(int(counts.get(n, 0) or 0) for n in values)
    lowest = [n for n in values if int(counts.get(n, 0) or 0) == min_count]
    return int(rng.choice(lowest))


def sample_category_assignment(
    *,
    categories: list[tuple[str, str, tuple[int, ...]]],
    eligible: set[int],
    counts: dict[int, int],
    n: int,
    rng=None,
) -> list[dict[str, Any]]:
    """Sample ``n`` task numbers, one per category when possible.

    Within a category, unused / lowest-count tasks are equally likely. Already-
    picked tasks are skipped so a dual-listed task (15) is used once. Working
    counts update after each pick. Extra slots beyond the category count take
    another lowest-count task from remaining category pools, then from any
    leftover eligible numbers.
    """
    rng = rng or random.SystemRandom()
    need = max(0, int(n))
    eligible_set = {int(x) for x in eligible}
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    working = {int(k): int(v or 0) for k, v in (counts or {}).items()}
    cat_taken = {str(key): 0 for key, _label, _nums in categories}

    def pool_for(numbers) -> list[int]:
        return [int(x) for x in numbers if int(x) in eligible_set and int(x) not in used]

    def append_pick(key: str, label: str, choice: int) -> None:
        picked.append(
            {
                "category": key,
                "category_label": label,
                "task": int(choice),
            }
        )
        used.add(int(choice))
        working[int(choice)] = int(working.get(int(choice), 0) or 0) + 1
        if key in cat_taken:
            cat_taken[key] += 1

    while need > 0:
        available: list[tuple[int, int, str, str, list[int]]] = []
        for key, label, numbers in categories:
            pool = pool_for(numbers)
            if not pool:
                continue
            min_task = min(int(working.get(t, 0) or 0) for t in pool)
            available.append((int(cat_taken.get(key, 0)), min_task, key, label, pool))
        if not available:
            break
        min_cat = min(row[0] for row in available)
        same_cat = [row for row in available if row[0] == min_cat]
        min_task = min(row[1] for row in same_cat)
        tied = [row for row in same_cat if row[1] == min_task]
        _n_cat, _min_task, key, label, pool = rng.choice(tied)
        choice = pick_lowest_count(pool, working, rng)
        if choice is None:
            break
        append_pick(key, label, choice)
        need -= 1

    leftover = [n for n in sorted(eligible_set) if n not in used]
    while need > 0 and leftover:
        choice = pick_lowest_count(leftover, working, rng)
        if choice is None:
            break
        append_pick("other", "Other", choice)
        leftover = [n for n in leftover if n != choice]
        need -= 1
    return picked


sample_base_assignment = sample_category_assignment


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "all"):
        return True
    if text in ("0", "false", "no", "off", "none"):
        return False
    return default


def _category_label(key: str) -> str:
    text = str(key or "").strip()
    if not text:
        return ""
    slug = text.lower().replace(" ", "_").replace("-", "_")
    if slug in _CATEGORY_LABELS:
        return _CATEGORY_LABELS[slug]
    return text.replace("_", " ").strip().title()


def _as_category_list(
    value,
    default: tuple[tuple[str, str, tuple[int, ...]], ...] = DEFAULT_BASE_TASK_CATEGORIES,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Parse a category map from yaml. Empty / missing → ``default``."""
    if value is None:
        return [tuple(item) for item in default]
    if not isinstance(value, dict) or not value:
        return [tuple(item) for item in default]
    out: list[tuple[str, str, tuple[int, ...]]] = []
    for raw_key, raw_nums in value.items():
        key = str(raw_key or "").strip().lower().replace(" ", "_").replace("-", "_")
        if not key:
            continue
        numbers = _as_int_list(raw_nums, [])
        if not numbers:
            continue
        out.append((key, _category_label(str(raw_key)), tuple(numbers)))
    return out or [tuple(item) for item in default]


def _as_int_list(value, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return list(default)
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(number)
    return out


def _as_seed(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    if seed < 0:
        return None
    return seed


def _is_unlimited(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, str) and value.strip().lower() in (
        "null",
        "none",
        "unlimited",
        "inf",
        "infinite",
    ):
        return True
    return False


def _as_play_limit(value) -> int | None:
    """Play budget: positive int, ``0`` to skip/gray out, or ``None`` for unlimited."""
    if _is_unlimited(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    if number < 0:
        return None
    return number


def _normalize_scenario(value) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return _SCENARIO_ALIASES.get(text, text if text in BASE_SCENARIOS else None)


def _task_lookups(suite: str) -> tuple[dict[int, str], dict[str, str]]:
    table = BASE_TASK_TABLE if suite == "base" else HOUSEHOLD_TASK_TABLE
    by_number = {number: key for number, key, _label in table}
    by_name = {key: key for _number, key, _label in table}
    return by_number, by_name


def resolve_task_name(suite: str, raw) -> str | None:
    """Map a 1-based card number or task key to the canonical task name."""
    by_number, by_name = _task_lookups(suite)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return by_number.get(int(raw))
    text = str(raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        return by_number.get(int(text))
    key = text.replace("-", "_")
    return by_name.get(key) or by_name.get(text)


def _as_seeds(value) -> list[int]:
    """``null`` / empty → random each play. One non-negative int or a list of ints."""
    if value is None or value == "" or value is False:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        value = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            seed = _as_seed(item)
            if seed is not None:
                out.append(seed)
        return out
    seed = _as_seed(value)
    return [seed] if seed is not None else []


def _is_policy_map(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    for key in raw:
        text = str(key or "").strip().lower()
        if text in ("base", "household", "default", "all"):
            return True
        if _normalize_scenario(key) in BASE_SCENARIOS:
            return True
    return False


def _top_level_has_opt_keys(raw: dict) -> bool:
    for key in raw:
        scenario = _normalize_scenario(key)
        if scenario in ("opt1", "opt2", "opt1+2"):
            return True
    return False


@dataclass
class SlotPolicy:
    """Global default plus optional per-suite / per-task / per-scenario overrides.

    Resolution order:
      exact task+scenario → task → suite+scenario kind → global scenario kind
      → suite → default.

    Scenario kind (``opt1+2``, ``opt1``, …) applies to every task in that suite.
    For boolean whitelist maps, listing specific scenarios of a task leaves the
    other scenarios of that task at ``missing_slot`` (False) rather than default.
    """

    default: Any
    suite: dict[str, Any] = field(default_factory=dict)
    task: dict[tuple[str, str], Any] = field(default_factory=dict)
    slot: dict[tuple[str, str, str], Any] = field(default_factory=dict)
    scenario_kind: dict[str, Any] = field(default_factory=dict)
    suite_scenario: dict[tuple[str, str], Any] = field(default_factory=dict)
    slot_explicit_tasks: set[tuple[str, str]] = field(default_factory=set)
    missing_slot: Any = False
    bool_mode: bool = False

    def resolve(self, suite: str, task: str, scenario: str | None = None):
        suite = str(suite or "").strip()
        task = str(task or "").strip()
        if suite == "household":
            scenario = None
        if scenario:
            key = (suite, task, str(scenario))
            if key in self.slot:
                return self.slot[key]
            if self.bool_mode and (suite, task) in self.slot_explicit_tasks:
                return self.missing_slot
        task_key = (suite, task)
        if task_key in self.task:
            return self.task[task_key]
        if scenario:
            suite_sc = (suite, str(scenario))
            if suite_sc in self.suite_scenario:
                return self.suite_scenario[suite_sc]
            if str(scenario) in self.scenario_kind:
                return self.scenario_kind[str(scenario)]
        if suite in self.suite:
            return self.suite[suite]
        return self.default

    def any_true(self) -> bool:
        if self.default:
            return True
        if any(self.suite.values()):
            return True
        if any(self.task.values()):
            return True
        if any(self.slot.values()):
            return True
        if any(self.scenario_kind.values()):
            return True
        if any(self.suite_scenario.values()):
            return True
        return False


def _listed_bool(value) -> bool:
    """Listed whitelist keys default to on; explicit false/off stays off."""
    return _as_bool(value, True)


def _ingest_bool_names(policy: SlotPolicy, items) -> None:
    for item in items:
        if isinstance(item, bool):
            continue
        text = str(item or "").strip().lower()
        if text in ("base", "household"):
            policy.suite[text] = True
            continue
        scenario = _normalize_scenario(item)
        if scenario:
            policy.scenario_kind[scenario] = True


def _bool_whitelist_from_names(items) -> SlotPolicy:
    policy = SlotPolicy(default=False, bool_mode=True, missing_slot=False)
    _ingest_bool_names(policy, items)
    return policy


def _ingest_bool_task(policy: SlotPolicy, suite: str, task: str, spec) -> None:
    if isinstance(spec, (list, tuple, set, frozenset)):
        policy.slot_explicit_tasks.add((suite, task))
        for item in spec:
            scenario = _normalize_scenario(item)
            if scenario:
                policy.slot[(suite, task, scenario)] = True
        return
    if isinstance(spec, dict):
        policy.slot_explicit_tasks.add((suite, task))
        for raw_scenario, raw_value in spec.items():
            scenario = _normalize_scenario(raw_scenario)
            if scenario:
                policy.slot[(suite, task, scenario)] = _listed_bool(raw_value)
        return
    scenario = _normalize_scenario(spec)
    if scenario:
        policy.slot_explicit_tasks.add((suite, task))
        policy.slot[(suite, task, scenario)] = True
        return
    policy.task[(suite, task)] = _listed_bool(spec)


def _ingest_bool_suite(policy: SlotPolicy, suite: str, spec) -> None:
    if isinstance(spec, (list, tuple, set, frozenset)):
        for item in spec:
            scenario = _normalize_scenario(item)
            task = resolve_task_name(suite, item)
            if scenario and not task:
                policy.suite_scenario[(suite, scenario)] = True
                continue
            if task:
                policy.task[(suite, task)] = True
        return
    if isinstance(spec, dict):
        for raw_key, raw_value in spec.items():
            scenario = _normalize_scenario(raw_key)
            task = resolve_task_name(suite, raw_key)
            if scenario and not task:
                policy.suite_scenario[(suite, scenario)] = _listed_bool(raw_value)
                continue
            if task:
                _ingest_bool_task(policy, suite, task, raw_value)
        return
    scenario = _normalize_scenario(spec)
    if scenario:
        policy.suite_scenario[(suite, scenario)] = True
        return
    policy.suite[suite] = _listed_bool(spec)


def parse_bool_policy(raw, default: bool = False) -> SlotPolicy:
    """Parse ``true`` / ``false`` or a suite/task/scenario whitelist.

    Unlisted slots stay off. Listing a name without a value turns that slot on::

        true                         # every suite / scenario
        false                        # none
        opt1+2                       # Opt 1+2 only
        [opt1, opt2]                 # those columns
        {opt1+2: true}               # Opt 1+2 only; household off
        {opt1+2: true, household: true}
    """
    if isinstance(raw, (list, tuple, set, frozenset)):
        return _bool_whitelist_from_names(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("base", "household"):
            return _bool_whitelist_from_names([text])
        scenario = _normalize_scenario(raw)
        if scenario:
            return _bool_whitelist_from_names([raw])
    if _is_policy_map(raw):
        has_opts = _top_level_has_opt_keys(raw)
        if "all" in raw:
            fallback = _as_bool(raw.get("all"), default)
        elif "default" in raw and not has_opts:
            fallback = _as_bool(raw.get("default"), default)
        elif "base" in raw or "household" in raw:
            fallback = False
        else:
            fallback = False
        policy = SlotPolicy(
            default=fallback,
            bool_mode=True,
            missing_slot=False,
        )
        for key, value in raw.items():
            text = str(key or "").strip().lower()
            if text in ("all",):
                continue
            if text in ("base", "household"):
                _ingest_bool_suite(policy, text, value)
                continue
            scenario = _normalize_scenario(key)
            if scenario is None:
                continue
            if text == "default" and not has_opts and "all" not in raw:
                continue
            policy.scenario_kind[scenario] = _listed_bool(value)
        return policy
    return SlotPolicy(default=_as_bool(raw, default), bool_mode=True, missing_slot=False)


def _ingest_int_task(policy: SlotPolicy, suite: str, task: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_scenario, raw_value in spec.items():
            scenario = _normalize_scenario(raw_scenario)
            if scenario:
                policy.slot[(suite, task, scenario)] = _as_play_limit(raw_value)
        return
    policy.task[(suite, task)] = _as_play_limit(spec)


def _ingest_int_suite(policy: SlotPolicy, suite: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_key, raw_value in spec.items():
            scenario = _normalize_scenario(raw_key)
            task = resolve_task_name(suite, raw_key)
            if scenario and not task:
                if isinstance(raw_value, dict):
                    continue
                policy.suite_scenario[(suite, scenario)] = _as_play_limit(raw_value)
                continue
            if task:
                _ingest_int_task(policy, suite, task, raw_value)
        return
    policy.suite[suite] = _as_play_limit(spec)


def parse_int_policy(raw, default: int | None = 1) -> SlotPolicy:
    """Parse a play-limit integer, ``0`` (skip), ``null`` (unlimited), or a map.

    ``0`` grays the option out from the start. An explicit ``None`` / ``null``
    means unlimited. Omit the yaml key to keep the constructor default (1).

    Map form (scenario kind applies to every task)::

        default: 1
        opt1: 2
        opt2: 2
        opt1+2: 5
        household: 1
    """
    if _is_policy_map(raw):
        has_opts = _top_level_has_opt_keys(raw)
        if "all" in raw:
            fallback = _as_play_limit(raw.get("all"))
        elif "default" in raw and not has_opts:
            fallback = _as_play_limit(raw.get("default"))
        else:
            fallback = default
        policy = SlotPolicy(default=fallback, bool_mode=False)
        for key, value in raw.items():
            text = str(key or "").strip().lower()
            if text in ("all",):
                continue
            if text in ("base", "household"):
                _ingest_int_suite(policy, text, value)
                continue
            scenario = _normalize_scenario(key)
            if scenario is None:
                continue
            if text == "default" and not has_opts and "all" not in raw:
                continue
            policy.scenario_kind[scenario] = _as_play_limit(value)
        return policy
    if raw is None:
        return SlotPolicy(default=None, bool_mode=False)
    return SlotPolicy(default=_as_play_limit(raw), bool_mode=False)


def _ingest_seed_task(policy: SlotPolicy, suite: str, task: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_scenario, raw_value in spec.items():
            scenario = _normalize_scenario(raw_scenario)
            if scenario:
                policy.slot[(suite, task, scenario)] = _as_seeds(raw_value)
        return
    policy.task[(suite, task)] = _as_seeds(spec)


def _ingest_seed_suite(policy: SlotPolicy, suite: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_key, raw_value in spec.items():
            scenario = _normalize_scenario(raw_key)
            task = resolve_task_name(suite, raw_key)
            if scenario and not task:
                if isinstance(raw_value, dict):
                    continue
                policy.suite_scenario[(suite, scenario)] = _as_seeds(raw_value)
                continue
            if task:
                _ingest_seed_task(policy, suite, task, raw_value)
        return
    policy.suite[suite] = _as_seeds(spec)


def parse_seed_policy(raw) -> SlotPolicy:
    """Parse seed ``null`` / int / list, or a per-suite / per-task / per-opt map.

    Values are seed lists (empty → random each play). Map form mirrors
    ``plays_per_scenario``::

        seed: 0
        seed: [0, 1, 2]
        seed:
          default: 0
          opt1: [1, 2]
          opt1+2: 5
          household: 10
          base:
            1: {opt1+2: [3, 4]}   # task 01 Opt 1+2 only
            2: 7                  # task 02, every scenario
    """
    if _is_policy_map(raw):
        has_opts = _top_level_has_opt_keys(raw)
        if "all" in raw:
            fallback = _as_seeds(raw.get("all"))
        elif "default" in raw and not has_opts:
            fallback = _as_seeds(raw.get("default"))
        else:
            fallback = []
        policy = SlotPolicy(default=fallback, bool_mode=False, missing_slot=[])
        for key, value in raw.items():
            text = str(key or "").strip().lower()
            if text in ("all",):
                continue
            if text in ("base", "household"):
                _ingest_seed_suite(policy, text, value)
                continue
            scenario = _normalize_scenario(key)
            if scenario is None:
                continue
            if text == "default" and not has_opts and "all" not in raw:
                continue
            policy.scenario_kind[scenario] = _as_seeds(value)
        return policy
    return SlotPolicy(default=_as_seeds(raw), bool_mode=False, missing_slot=[])


def evaluation_seeds_path() -> Path:
    """Path to the shared, fixed human-and-policy evaluation seed protocol."""
    raw = os.environ.get(EVALUATION_SEEDS_ENV, "").strip()
    return Path(raw) if raw else EVALUATION_SEEDS_PATH


def load_evaluation_seed_policy(path: Path | None = None) -> SlotPolicy:
    """Load the fixed benchmark seed policy from ``task_config/eval_seeds.yml``.

    The loader deliberately accepts only the ``seed`` section from the shared
    protocol file.  A missing or malformed file resolves to no seeds, which
    callers treat as a configuration error rather than silently randomizing a
    benchmark run.
    """
    path = path or evaluation_seeds_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return parse_seed_policy(None)
    if not isinstance(raw, dict):
        return parse_seed_policy(None)
    return parse_seed_policy(raw.get("seed"))


def evaluation_suite(task: str) -> str:
    """Return the evaluation suite containing a canonical task name."""
    key = str(task or "").strip().replace("-", "_")
    if key in {name for _number, name, _label in BASE_TASK_TABLE}:
        return "base"
    if key in {name for _number, name, _label in HOUSEHOLD_TASK_TABLE}:
        return "household"
    raise ValueError(f"Task {task!r} is not in the RoboDyna evaluation protocol.")


def evaluation_seeds_for(
    task: str,
    scenario: str | None = None,
    *,
    path: Path | None = None,
) -> list[int]:
    """Return the exact fixed seed list for one benchmark task slot."""
    suite = evaluation_suite(task)
    if suite == "base":
        scenario = _normalize_scenario(scenario) or "default"
    else:
        scenario = None
    policy = load_evaluation_seed_policy(path)
    resolved = policy.resolve(suite, str(task).strip().replace("-", "_"), scenario)
    return list(resolved) if isinstance(resolved, list) else _as_seeds(resolved)


@dataclass
class ExperimentConfig:
    record_data_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(False))
    save_video_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(False))
    export_lerobot_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(False))
    log_plays_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(True))
    plays_policy: SlotPolicy = field(default_factory=lambda: parse_int_policy(1))
    seed_policy: SlotPolicy = field(default_factory=lambda: parse_seed_policy(None))
    seeds: list[int] = field(default_factory=list)
    base_tasks: list[int] = field(default_factory=lambda: list(DEFAULT_BASE_TASKS))
    household_tasks: list[int] = field(default_factory=lambda: list(DEFAULT_HOUSEHOLD_TASKS))
    base_task_categories: list[tuple[str, str, tuple[int, ...]]] = field(
        default_factory=lambda: [tuple(item) for item in DEFAULT_BASE_TASK_CATEGORIES]
    )
    household_task_categories: list[tuple[str, str, tuple[int, ...]]] = field(
        default_factory=lambda: [tuple(item) for item in DEFAULT_HOUSEHOLD_TASK_CATEGORIES]
    )
    base_scenarios_per_experiment: int = 5
    household_scenarios_per_experiment: int = 2
    path: Path = CONFIG_PATH

    @property
    def record_data(self) -> bool:
        """True if any slot records HDF5 (used by the locked GUI checkbox)."""
        return self.record_data_policy.any_true()

    @property
    def save_video(self) -> bool:
        return self.save_video_policy.any_true()

    @property
    def export_lerobot(self) -> bool:
        return self.export_lerobot_policy.any_true()

    @property
    def seed(self) -> int | None:
        """Single fixed global seed, or None when random / a list / map is used."""
        if len(self.seeds) == 1:
            return self.seeds[0]
        return None

    def seeds_for(
        self,
        suite: str | None = None,
        task: str | None = None,
        scenario: str | None = None,
    ) -> list[int]:
        """Resolved seed list for a slot (empty → random each play)."""
        if suite and task:
            resolved = self.seed_policy.resolve(suite, task, scenario)
            if isinstance(resolved, list):
                return list(resolved)
            return _as_seeds(resolved)
        return list(self.seeds)

    def pick_seed(
        self,
        play_index: int = 0,
        suite: str | None = None,
        task: str | None = None,
        scenario: str | None = None,
    ) -> int | None:
        """Seed for this play, or None to randomize.

        A list is consumed in order and wraps: play 0 → seeds[0], play 1 → seeds[1], …
        When ``suite``/``task`` are set, per-slot overrides win over the global list.
        """
        seeds = self.seeds_for(suite, task, scenario)
        if not seeds:
            return None
        index = max(0, int(play_index))
        return seeds[index % len(seeds)]

    def seed_display(
        self,
        suite: str | None = None,
        task: str | None = None,
        scenario: str | None = None,
    ) -> str:
        """Comma-joined seeds for the locked GUI field (global, or a specific slot)."""
        seeds = self.seeds_for(suite, task, scenario) if suite and task else self.seeds
        if not seeds:
            return ""
        return ",".join(str(seed) for seed in seeds)

    def scenarios_per_experiment(self, suite: str) -> int:
        if suite == "household":
            return int(self.household_scenarios_per_experiment)
        return int(self.base_scenarios_per_experiment)

    def suite_enabled(self, suite: str) -> bool:
        """False when ``*_scenarios_per_experiment`` is 0, i.e. skip this suite."""
        return self.scenarios_per_experiment(suite) > 0

    def visible_indices(
        self,
        suite: str,
        n_tasks: int,
        assigned: list[int] | None = None,
    ) -> list[int]:
        """0-based TASKS indices that should appear in the given suite GUI."""
        if not self.suite_enabled(suite):
            return []
        if assigned is not None:
            ones = assigned
        elif suite == "base":
            ones = self.base_tasks
        else:
            ones = self.household_tasks
        out: list[int] = []
        seen: set[int] = set()
        for number in ones:
            index = int(number) - 1
            if index in seen or not (0 <= index < n_tasks):
                continue
            seen.add(index)
            out.append(index)
        return out

    def eligible_base_numbers(self) -> set[int]:
        return self._eligible_numbers(self.base_tasks, self.base_task_categories)

    def eligible_household_numbers(self) -> set[int]:
        return self._eligible_numbers(self.household_tasks, self.household_task_categories)

    @staticmethod
    def _eligible_numbers(
        allowed_list: list[int],
        categories: list[tuple[str, str, tuple[int, ...]]],
    ) -> set[int]:
        """Candidate 1-based cards: category members that are also in the pool."""
        allowed = {int(n) for n in allowed_list}
        out: set[int] = set()
        for _key, _label, numbers in categories:
            for number in numbers:
                if int(number) in allowed:
                    out.add(int(number))
        return out

    def suite_categories(self, suite: str) -> list[tuple[str, str, tuple[int, ...]]]:
        if suite == "household":
            return list(self.household_task_categories)
        return list(self.base_task_categories)

    def categories_for_number(self, number: int, suite: str = "base") -> list[str]:
        """Display labels for every category that includes this 1-based card."""
        labels: list[str] = []
        seen: set[str] = set()
        for _key, label, numbers in self.suite_categories(suite):
            if int(number) in numbers and label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def grouped_visible_indices(
        self,
        suite: str,
        n_tasks: int,
        assigned: list[int] | None = None,
        picks: list[dict] | None = None,
    ) -> list[tuple[str | None, list[int]]]:
        """Visible 0-based indices grouped by category (first match wins).

        When ``picks`` from a participant assignment is given, grouping follows
        the sampled category rather than first-match. Leftover visible tasks go
        under ``Other``.
        """
        visible = self.visible_indices(suite, n_tasks, assigned=assigned)
        categories = self.suite_categories(suite)
        if not categories:
            return [(None, visible)]
        if picks:
            groups: list[tuple[str | None, list[int]]] = []
            used: set[int] = set()
            by_cat: dict[str, list[int]] = {}
            labels: dict[str, str] = {
                key: label for key, label, _nums in categories
            }
            labels.setdefault("other", "Other")
            for pick in picks:
                if not isinstance(pick, dict):
                    continue
                try:
                    number = int(pick.get("task"))
                except (TypeError, ValueError):
                    continue
                index = number - 1
                if index not in visible or index in used:
                    continue
                key = str(pick.get("category") or "other")
                by_cat.setdefault(key, []).append(index)
                used.add(index)
                if key not in labels:
                    labels[key] = str(pick.get("category_label") or _category_label(key))
            for key, _label, _nums in categories:
                idxs = by_cat.get(key) or []
                if idxs:
                    groups.append((labels.get(key, _label), idxs))
            if by_cat.get("other"):
                groups.append((labels["other"], by_cat["other"]))
            leftover = [index for index in visible if index not in used]
            if leftover:
                groups.append(("Other", leftover))
            return groups or [(None, visible)]
        visible_set = set(visible)
        used = set()
        groups = []
        for _key, label, numbers in categories:
            idxs = [
                n - 1
                for n in numbers
                if 0 <= n - 1 < n_tasks and (n - 1) in visible_set and (n - 1) not in used
            ]
            if idxs:
                groups.append((label, idxs))
                used.update(idxs)
        leftover = [index for index in visible if index not in used]
        if leftover:
            groups.append(("Other", leftover))
        return groups or [(None, visible)]

    def task_names(self, suite: str) -> list[str]:
        table = BASE_TASK_TABLE if suite == "base" else HOUSEHOLD_TASK_TABLE
        lookup = {number: key for number, key, _label in table}
        ones = self.base_tasks if suite == "base" else self.household_tasks
        return [lookup[n] for n in ones if n in lookup]

    def max_plays(self, suite: str, task: str, scenario: str | None = None) -> int | None:
        """Allowed SUCCESS/FAILURE plays for this slot.

        ``0`` = skip (grayed out). ``None`` = unlimited.
        """
        return self.plays_policy.resolve(suite, task, scenario)

    def should_log(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.log_plays_policy.resolve(suite, task, scenario))

    def should_record_data(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.record_data_policy.resolve(suite, task, scenario))

    def should_save_video(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.save_video_policy.resolve(suite, task, scenario))

    def should_export_lerobot(self, suite: str, task: str, scenario: str | None = None) -> bool:
        """True when this slot should also write LeRobot. Requires record_data."""
        return bool(self.export_lerobot_policy.resolve(suite, task, scenario))


def config_path() -> Path:
    raw = os.environ.get(CONFIG_ENV, "").strip()
    if raw:
        return Path(raw)
    return CONFIG_PATH


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    path = path or config_path()
    cfg = default_experiment_config()
    cfg.path = path
    if not path.is_file():
        return cfg
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return cfg
    if not isinstance(raw, dict):
        return cfg
    cfg.record_data_policy = parse_bool_policy(raw.get("record_data"), False)
    cfg.save_video_policy = parse_bool_policy(raw.get("save_video"), False)
    cfg.export_lerobot_policy = parse_bool_policy(raw.get("export_lerobot"), False)
    log_raw = raw.get("log_plays", True)
    cfg.log_plays_policy = parse_bool_policy(log_raw, True)
    if "plays_per_scenario" in raw:
        cfg.plays_policy = parse_int_policy(raw.get("plays_per_scenario"), 1)
    # Evaluation seeds are benchmark data, not a GUI-local preference.  Both
    # the experiment launcher and policy evaluators read this one shared file.
    cfg.seed_policy = load_evaluation_seed_policy()
    cfg.seeds = list(cfg.seed_policy.default or [])
    if "base_tasks" in raw:
        cfg.base_tasks = _as_int_list(raw.get("base_tasks"), DEFAULT_BASE_TASKS)
    if "household_tasks" in raw:
        cfg.household_tasks = _as_int_list(raw.get("household_tasks"), DEFAULT_HOUSEHOLD_TASKS)
    cfg.base_task_categories = _as_category_list(raw.get("base_task_categories"))
    cfg.household_task_categories = _as_category_list(
        raw.get("household_task_categories"),
        DEFAULT_HOUSEHOLD_TASK_CATEGORIES,
    )
    if "base_scenarios_per_experiment" in raw:
        cfg.base_scenarios_per_experiment = _as_count_int(
            raw.get("base_scenarios_per_experiment"), 5
        )
    elif "base_tasks_per_experiment" in raw:
        cfg.base_scenarios_per_experiment = _as_count_int(
            raw.get("base_tasks_per_experiment"), 5
        )
    if "household_scenarios_per_experiment" in raw:
        cfg.household_scenarios_per_experiment = _as_count_int(
            raw.get("household_scenarios_per_experiment"), 2
        )
    elif "household_tasks_per_experiment" in raw:
        cfg.household_scenarios_per_experiment = _as_count_int(
            raw.get("household_tasks_per_experiment"), 2
        )
    return cfg
