"""Per-user human-experiment logs under ``data/exp_logs/<user>/``.

Robot and keyboard sessions are separate files in that folder:
``user_robot.json`` and ``user_keyboard.json``.

Used by ``experiment_gui.py`` and by the base / household task GUIs when they
run in experiment mode (``ROBODYNA_EXPERIMENT=1``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_LOGS_DIR = REPO_ROOT / "data" / "exp_logs"

EXPERIMENT_ENV = "ROBODYNA_EXPERIMENT"
EXPERIMENT_USER_ENV = "ROBODYNA_EXPERIMENT_USER"
EXPERIMENT_LOG_ENV = "ROBODYNA_EXPERIMENT_LOG"

LOG_CONTROLLER_TAGS = ("robot", "keyboard")
_KEYBOARD_ALIASES = {
    "keyboard",
    "keyboard+mouse",
    "keyboardmouse",
    "key+mouse",
    "keymouse",
    "km",
}

YES_NO_CHOICES = (("yes", "Yes"), ("no", "No"))
ASSIGNED_TASKS = "assigned_tasks"
NONE_CHOICE = ("none", "None")


def _survey_item(
    key: str,
    prompt: str,
    choices,
    *,
    multi: bool = False,
    rank: bool = False,
    exclusive: str | None = None,
    visible_if=None,
    layout: str = "row",
    extra_options=(),
) -> dict[str, Any]:
    return {
        "key": key,
        "prompt": prompt,
        "choices": choices,
        "multi": multi,
        "rank": rank,
        "exclusive": exclusive,
        "visible_if": visible_if,
        "layout": "rank" if rank else layout,
        "extra_options": tuple(extra_options),
    }


def _answer_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def pre_controller_visible(answers: dict[str, Any] | None) -> bool:
    return str((answers or {}).get("video_games") or "").strip().lower() == "yes"


def pre_wasd_visible(answers: dict[str, Any] | None) -> bool:
    if not pre_controller_visible(answers):
        return False
    selected = {item.lower() for item in _answer_list((answers or {}).get("game_controllers"))}
    return "keyboard" in selected


PRE_SURVEY_QUESTIONS = (
    _survey_item(
        "video_games",
        "Do you have experience playing video games?",
        YES_NO_CHOICES,
    ),
    _survey_item(
        "game_controllers",
        "What types of controller have you used playing games? More than one option can be selected.",
        (
            ("gamepad", "Gamepad"),
            ("vr_controller", "VR Controller"),
            ("keyboard", "Keyboard"),
            ("mouse", "Mouse"),
            ("none", "None"),
        ),
        multi=True,
        exclusive="none",
        visible_if=pre_controller_visible,
    ),
    _survey_item(
        "wasd_mouse_look",
        "Have you used WASD / mouse-look games (Minecraft, first person shooters, etc.)?",
        YES_NO_CHOICES,
        visible_if=pre_wasd_visible,
    ),
    _survey_item(
        "robotic_simulators",
        "Do you have previous experience with robotic simulators?",
        YES_NO_CHOICES,
    ),
    _survey_item(
        "teleoperated_robot",
        "Have you teleoperated a real or simulated robot before?",
        YES_NO_CHOICES,
    ),
    _survey_item(
        "spatial_3d_apps",
        "Do you use applications with 3D spatial tasks (CAD, Blender, etc.)?",
        YES_NO_CHOICES,
    ),
    _survey_item(
        "mouse_regular_hand",
        "Do you regularly use a mouse with your right or left hand?",
        (("right", "Right"), ("left", "Left")),
    ),
)

POST_SURVEY_QUESTIONS = (
    _survey_item(
        "task_difficulty",
        "Overall, how do you rank the difficulty of the tasks?",
        (
            ("very_easy", "Very Easy"),
            ("easy", "Easy"),
            ("neutral", "Neutral"),
            ("hard", "Hard"),
            ("very_hard", "Very hard"),
        ),
    ),
    _survey_item(
        "objectives_clear",
        "Were the objectives of the tasks clear?",
        (
            ("very_clear", "Very clear"),
            ("clear", "Clear"),
            ("neutral", "Neutral"),
            ("unclear", "Unclear"),
            ("very_unclear", "Very unclear"),
        ),
    ),
    _survey_item(
        "task_difficulty_rank",
        "Rank the tasks in order of difficulty.",
        ASSIGNED_TASKS,
        rank=True,
    ),
    _survey_item(
        "robot_hardest_aspect",
        "When using the robot controller, which aspect of performing the task did you find most difficult?",
        (
            ("control", "Control"),
            ("event_prediction", "Event prediction"),
        ),
    ),
    _survey_item(
        "keyboard_hardest_aspect",
        "When using the keyboard+mouse controller, which aspect of performing the task did you find most difficult?",
        (
            ("control", "Control"),
            ("event_prediction", "Event prediction"),
        ),
    ),
    _survey_item(
        "gripper_view_useful_tasks",
        "In which one of the following tasks you found gripper view useful? Multiple options can be selected.",
        ASSIGNED_TASKS,
        multi=True,
        exclusive=NONE_CHOICE[0],
        extra_options=(NONE_CHOICE,),
    ),
    _survey_item(
        "easier_controller",
        "Which controller did you find easier to use?",
        (
            ("robot", "Robot"),
            ("keyboard", "Keyboard + mouse"),
            ("same", "Same"),
        ),
    ),
    _survey_item(
        "policy_solve_6mo",
        "How likely do you think a robotic policy can solve these tasks in the next 6 months?",
        (
            ("very_likely", "Very likely"),
            ("likely", "Likely"),
            ("neutral", "Neutral"),
            ("unlikely", "Unlikely"),
            ("very_unlikely", "Very unlikely"),
        ),
    ),
)

# Legacy (key, prompt) pairs for older yes/no experience items.
EXPERIENCE_QUESTIONS = tuple(
    (item["key"], item["prompt"])
    for item in PRE_SURVEY_QUESTIONS
    if item["key"] in {"video_games", "robotic_simulators"}
)


def question_choices(question: dict[str, Any], extra_choices: dict[str, Any] | None = None):
    choices = question.get("choices")
    if choices == ASSIGNED_TASKS:
        dynamic = tuple((extra_choices or {}).get(ASSIGNED_TASKS) or ())
        if not dynamic:
            return ()
        return dynamic + tuple(question.get("extra_options") or ())
    return tuple(choices or ())


def question_visible(
    question: dict[str, Any],
    answers: dict[str, Any] | None = None,
    extra_choices: dict[str, Any] | None = None,
) -> bool:
    pred = question.get("visible_if")
    if callable(pred) and not pred(answers or {}):
        return False
    if question.get("choices") == ASSIGNED_TASKS and not question_choices(question, extra_choices):
        return False
    return True


def _parse_question_answer(question: dict[str, Any], raw, extra_choices: dict[str, Any] | None = None):
    choices = question_choices(question, extra_choices)
    allowed = {value for value, _label in choices}
    if question.get("rank"):
        selected: list[str] = []
        seen: set[str] = set()
        for item in _answer_list(raw):
            if (allowed and item not in allowed) or item in seen:
                continue
            seen.add(item)
            selected.append(item)
        for value, _label in choices:
            if value not in seen:
                selected.append(value)
        return selected
    if question.get("multi"):
        selected = [item for item in _answer_list(raw) if not allowed or item in allowed]
        exclusive = question.get("exclusive")
        if exclusive and exclusive in selected:
            return [exclusive]
        return selected
    value = str(raw or "").strip()
    if allowed:
        return value if value in allowed else ""
    return value


def normalize_survey_answers(
    questions,
    answers: dict[str, Any] | None,
    extra_choices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    answers = answers or {}
    parsed: dict[str, Any] = {}
    for question in questions:
        parsed[question["key"]] = _parse_question_answer(
            question, answers.get(question["key"]), extra_choices
        )
    for question in questions:
        if question_visible(question, parsed, extra_choices):
            continue
        parsed[question["key"]] = [] if question.get("multi") or question.get("rank") else ""
    return parsed


def survey_missing_prompts(
    questions,
    answers: dict[str, Any] | None,
    extra_choices: dict[str, Any] | None = None,
) -> list[str]:
    parsed = normalize_survey_answers(questions, answers, extra_choices=extra_choices)
    missing: list[str] = []
    for question in questions:
        if not question_visible(question, parsed, extra_choices):
            continue
        value = parsed.get(question["key"])
        if question.get("rank"):
            allowed = [item for item, _label in question_choices(question, extra_choices)]
            if set(value or []) != set(allowed) or len(value or []) != len(allowed):
                missing.append(question["prompt"])
        elif question.get("multi"):
            if not value:
                missing.append(question["prompt"])
        elif not value:
            missing.append(question["prompt"])
    return missing


def survey_complete(
    questions,
    answers: dict[str, Any] | None,
    extra_choices: dict[str, Any] | None = None,
) -> bool:
    return not survey_missing_prompts(questions, answers, extra_choices=extra_choices)

# Base suite: 23 tasks × 4 scenarios. Household: 12 tasks.
# Tutorial parts are unlimited practice and are never written to these logs.
BASE_SCENARIO_SLOTS = 23 * 4
HOUSEHOLD_TASK_SLOTS = 12

_TRUE = {"1", "true", "yes", "on"}


def experiment_mode() -> bool:
    return os.environ.get(EXPERIMENT_ENV, "").strip().lower() in _TRUE


def slugify_user_name(name: str) -> str:
    """Stable folder tag for a display name (case-insensitive)."""
    text = str(name or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text or "user"


def user_dir(name_or_slug: str) -> Path:
    return EXP_LOGS_DIR / slugify_user_name(name_or_slug)


def log_controller_tag(value) -> str:
    """Folder filename tag: ``robot`` or ``keyboard`` (covers keyboard+mouse)."""
    text = str(value or "robot").strip().lower().replace("_", "+").replace(" ", "")
    if text in _KEYBOARD_ALIASES:
        return "keyboard"
    return "robot"


def session_control_mode() -> str:
    """Canonical play mode for the GUI / ``ROBODYNA_CONTROL``: ``robot`` or ``keyboard+mouse``."""
    return "keyboard+mouse" if log_controller_tag(os.environ.get("ROBODYNA_CONTROL")) == "keyboard" else "robot"


def current_controller() -> str:
    return log_controller_tag(os.environ.get("ROBODYNA_CONTROL", "robot"))


def user_log_filename(controller: str | None = None) -> str:
    return f"user_{log_controller_tag(controller if controller is not None else current_controller())}.json"


def user_log_path(name_or_slug: str, controller: str | None = None) -> Path:
    """``data/exp_logs/<user>/user_robot.json`` or ``user_keyboard.json``."""
    return user_dir(name_or_slug) / user_log_filename(controller)


def _iter_user_log_files(folder: Path):
    """Yield ``(tag, path)`` for tagged logs, then legacy ``user.json``."""
    if not folder.is_dir():
        return
    for tag in LOG_CONTROLLER_TAGS:
        path = folder / f"user_{tag}.json"
        if path.is_file():
            yield tag, path
    legacy = folder / "user.json"
    if legacy.is_file():
        yield "legacy", legacy


def iter_user_controller_logs(name_or_slug: str):
    """Yield ``(tag, path, data)`` for each log file in this participant folder."""
    folder = user_dir(name_or_slug)
    for tag, path in _iter_user_log_files(folder):
        data = load_user_log(path)
        if data:
            yield tag, path, data


ASSIGNMENT_FILENAME = "assignment.json"
USAGE_FILENAME = "base_task_usage.json"


def assignment_path(name_or_slug: str) -> Path:
    return user_dir(name_or_slug) / ASSIGNMENT_FILENAME


def usage_path() -> Path:
    return EXP_LOGS_DIR / USAGE_FILENAME


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_user_log(path: Path | None = None) -> dict[str, Any] | None:
    if path is None:
        raw = os.environ.get(EXPERIMENT_LOG_ENV, "").strip()
        path = Path(raw) if raw else None
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def save_user_log(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at"] = iso_now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def find_user(name: str) -> dict[str, Any] | None:
    """Return an existing user log if this name (or its slug) already exists."""
    slug = slugify_user_name(name)
    folder = user_dir(slug)
    for _tag, path in _iter_user_log_files(folder):
        data = load_user_log(path)
        if data:
            return data
    if not EXP_LOGS_DIR.is_dir():
        return None
    needle = str(name or "").strip().casefold()
    for folder in EXP_LOGS_DIR.iterdir():
        if not folder.is_dir():
            continue
        for _tag, candidate in _iter_user_log_files(folder):
            other = load_user_log(candidate)
            if not other:
                continue
            if str(other.get("user_name", "")).strip().casefold() == needle:
                return other
            if str(other.get("user_id", "")).strip() == slug:
                return other
    return None


def _empty_user_log(
    *,
    slug: str,
    display: str,
    controller: str,
    path: Path,
    experience: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = iso_now()
    return {
        "user_id": slug,
        "user_name": display,
        "controller": controller,
        "created_at": created_at or now,
        "updated_at": now,
        "experience": dict(experience or {}),
        "post_survey": {},
        "completed_keys": [],
        "play_counts": {},
        "success_counts": {},
        "plays": [],
        "log_path": str(path),
    }


def ensure_controller_log(
    name_or_slug: str,
    *,
    display: str | None = None,
    experience: dict[str, Any] | None = None,
    controller: str | None = None,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load or create the robot/keyboard log inside this user's folder.

    Progress is per controller: ``user_robot.json`` and ``user_keyboard.json``
    do not share plays. Experience is copied from any existing file for the user.
    A legacy ``user.json`` is migrated into ``user_robot.json`` when that tagged
    file does not exist yet.
    """
    tag = log_controller_tag(controller if controller is not None else current_controller())
    slug = slugify_user_name(name_or_slug)
    folder = user_dir(slug)
    path = folder / f"user_{tag}.json"
    existing = load_user_log(path)
    if existing:
        existing["controller"] = tag
        existing["log_path"] = str(path)
        assignment = _ensure_user_assignment(
            slug,
            display=str(existing.get("user_name") or display or slug),
        )
        _attach_assignment(existing, assignment)
        return existing

    legacy = folder / "user.json"
    if legacy.is_file():
        data = load_user_log(legacy)
        if data:
            file_tag = log_controller_tag(data.get("controller") or "robot")
            if file_tag == tag:
                data["controller"] = tag
                data["log_path"] = str(path)
                save_user_log(path, data)
                assignment = _ensure_user_assignment(
                    slug,
                    display=str(data.get("user_name") or display or slug),
                )
                _attach_assignment(data, assignment)
                return data

    source = template
    if source is None:
        for _other_tag, other_path in _iter_user_log_files(folder):
            source = load_user_log(other_path)
            if source:
                break
    display_name = (
        display
        or (source or {}).get("user_name")
        or str(name_or_slug or "").strip()
        or slug
    )
    exp = experience if experience is not None else (source or {}).get("experience") or {}
    data = _empty_user_log(
        slug=slug,
        display=str(display_name),
        controller=tag,
        path=path,
        experience=exp,
        created_at=(source or {}).get("created_at"),
    )
    assignment = _ensure_user_assignment(slug, display=str(display_name))
    _attach_assignment(data, assignment)
    save_user_log(path, data)
    return data


def create_user(
    name: str,
    experience: dict[str, Any],
    controller: str | None = None,
) -> dict[str, Any]:
    display = str(name or "").strip()
    answers = normalize_survey_answers(PRE_SURVEY_QUESTIONS, experience)
    return ensure_controller_log(
        display,
        display=display,
        experience=answers,
        controller=controller,
    )


def save_post_survey(
    data: dict[str, Any],
    answers: dict[str, Any],
    extra_choices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write post-experiment answers onto every controller log for this user."""
    parsed = normalize_survey_answers(
        POST_SURVEY_QUESTIONS, answers, extra_choices=extra_choices
    )
    slug = str(data.get("user_id") or slugify_user_name(data.get("user_name") or ""))
    written = None
    for _tag, path, current in iter_user_controller_logs(slug):
        log = dict(current)
        log["post_survey"] = parsed
        log["log_path"] = str(path)
        save_user_log(path, log)
        if str(path) == str(data.get("log_path") or "") or written is None:
            written = log
    if written is not None:
        return written
    path = Path(str(data.get("log_path") or ""))
    log = dict(data)
    log["post_survey"] = parsed
    if path:
        log["log_path"] = str(path)
        save_user_log(path, log)
    return log


def _attach_assignment(log: dict[str, Any], assignment: dict[str, Any] | None) -> None:
    if not assignment:
        return
    log["sampled_base_tasks"] = list(assignment.get("base_tasks") or [])
    log["sampled_base_picks"] = list(assignment.get("base_picks") or [])
    log["sampled_household_tasks"] = list(assignment.get("household_tasks") or [])
    log["sampled_household_picks"] = list(assignment.get("household_picks") or [])
    log["assignment_path"] = str(assignment.get("path") or "")


def _parse_task_numbers(raw) -> list[int]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        if isinstance(item, dict):
            item = item.get("task")
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(number)
    return out


def load_assignment_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    tasks = _parse_task_numbers(data.get("base_tasks") or data.get("sampled_base_tasks"))
    picks = data.get("base_picks") or data.get("sampled_base_picks") or []
    household_tasks = _parse_task_numbers(
        data.get("household_tasks") or data.get("sampled_household_tasks")
    )
    household_picks = data.get("household_picks") or data.get("sampled_household_picks") or []
    if not tasks and isinstance(picks, list):
        tasks = _parse_task_numbers(picks)
    if not household_tasks and isinstance(household_picks, list):
        household_tasks = _parse_task_numbers(household_picks)
    if not tasks and not household_tasks:
        return None
    data["base_tasks"] = tasks
    data["base_picks"] = [p for p in picks if isinstance(p, dict)] if isinstance(picks, list) else []
    data["household_tasks"] = household_tasks
    data["household_picks"] = (
        [p for p in household_picks if isinstance(p, dict)]
        if isinstance(household_picks, list)
        else []
    )
    data["path"] = str(path)
    return data


def load_user_assignment(name_or_slug: str | None = None) -> dict[str, Any] | None:
    """Load this participant's sampled base tasks, or the current session user."""
    slug = str(name_or_slug or "").strip()
    if not slug:
        slug = os.environ.get(EXPERIMENT_USER_ENV, "").strip()
    if slug:
        data = load_assignment_file(assignment_path(slug))
        if data:
            return data
        folder = user_dir(slug)
        for _tag, path in _iter_user_log_files(folder):
            log = load_user_log(path)
            base_tasks = _parse_task_numbers((log or {}).get("sampled_base_tasks"))
            base_picks = (log or {}).get("sampled_base_picks") or []
            household_tasks = _parse_task_numbers((log or {}).get("sampled_household_tasks"))
            household_picks = (log or {}).get("sampled_household_picks") or []
            if base_tasks or household_tasks:
                return {
                    "user_id": slug,
                    "base_tasks": base_tasks,
                    "base_picks": [p for p in base_picks if isinstance(p, dict)] if isinstance(base_picks, list) else [],
                    "household_tasks": household_tasks,
                    "household_picks": [p for p in household_picks if isinstance(p, dict)] if isinstance(household_picks, list) else [],
                    "path": str(folder / ASSIGNMENT_FILENAME),
                }
    log = load_user_log()
    if log:
        base_tasks = _parse_task_numbers(log.get("sampled_base_tasks"))
        base_picks = log.get("sampled_base_picks") or []
        household_tasks = _parse_task_numbers(log.get("sampled_household_tasks"))
        household_picks = log.get("sampled_household_picks") or []
        if base_tasks or household_tasks:
            return {
                "user_id": str(log.get("user_id") or ""),
                "base_tasks": base_tasks,
                "base_picks": [p for p in base_picks if isinstance(p, dict)] if isinstance(base_picks, list) else [],
                "household_tasks": household_tasks,
                "household_picks": [p for p in household_picks if isinstance(p, dict)] if isinstance(household_picks, list) else [],
                "path": str(log.get("assignment_path") or ""),
            }
    return None


def iter_user_assignments() -> list[dict[str, Any]]:
    """One assignment per participant folder (robot + keyboard share a sample)."""
    if not EXP_LOGS_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for folder in sorted(EXP_LOGS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        data = load_assignment_file(folder / ASSIGNMENT_FILENAME)
        if data is None:
            data = load_user_assignment(folder.name)
        if data and (data.get("base_tasks") or data.get("household_tasks")):
            data["user_id"] = data.get("user_id") or folder.name
            out.append(data)
    return out


def task_usage_counts(suite: str = "base") -> dict[int, int]:
    """How many live experiment logs currently include each 1-based task."""
    key = "base_tasks" if suite != "household" else "household_tasks"
    counts: dict[int, int] = {}
    for assignment in iter_user_assignments():
        seen: set[int] = set()
        for number in _parse_task_numbers(assignment.get(key)):
            if number in seen:
                continue
            seen.add(number)
            counts[number] = counts.get(number, 0) + 1
    return counts


def base_task_usage_counts() -> dict[int, int]:
    return task_usage_counts("base")


def household_task_usage_counts() -> dict[int, int]:
    return task_usage_counts("household")


def write_usage_snapshot(cfg=None) -> dict[str, Any]:
    """Rewrite ``data/exp_logs/base_task_usage.json`` from remaining logs."""
    try:
        from experiment_config import load_experiment_config
    except ImportError:
        from interactive.experiment_config import load_experiment_config

    cfg = cfg or load_experiment_config()
    base_counts = task_usage_counts("base")
    household_counts = task_usage_counts("household")

    def by_category(categories, counts):
        return {
            key: {str(n): int(counts.get(int(n), 0)) for n in numbers}
            for key, _label, numbers in categories
        }

    payload = {
        "updated_at": iso_now(),
        "participants": len(iter_user_assignments()),
        "counts": {str(n): int(c) for n, c in sorted(base_counts.items())},
        "by_category": by_category(cfg.base_task_categories, base_counts),
        "base": {
            "counts": {str(n): int(c) for n, c in sorted(base_counts.items())},
            "by_category": by_category(cfg.base_task_categories, base_counts),
        },
        "household": {
            "counts": {str(n): int(c) for n, c in sorted(household_counts.items())},
            "by_category": by_category(cfg.household_task_categories, household_counts),
        },
    }
    EXP_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = usage_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload


def _save_assignment(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["updated_at"] = iso_now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _sample_suite(cfg, suite: str) -> tuple[list[int], list[dict[str, Any]]]:
    try:
        from experiment_config import sample_category_assignment
    except ImportError:
        from interactive.experiment_config import sample_category_assignment

    if suite == "household":
        categories = list(cfg.household_task_categories)
        eligible = cfg.eligible_household_numbers()
        n = int(cfg.household_scenarios_per_experiment)
        fallback = list(cfg.household_tasks)
        counts = task_usage_counts("household")
    else:
        categories = list(cfg.base_task_categories)
        eligible = cfg.eligible_base_numbers()
        n = int(cfg.base_scenarios_per_experiment)
        fallback = list(cfg.base_tasks)
        counts = task_usage_counts("base")
    if n <= 0:
        return [], []
    picks = sample_category_assignment(
        categories=categories,
        eligible=eligible,
        counts=counts,
        n=n,
    )
    tasks = _parse_task_numbers(picks)
    if not tasks:
        tasks = fallback[:n]
    return tasks, picks


def _ensure_user_assignment(
    name_or_slug: str,
    *,
    display: str | None = None,
    cfg=None,
) -> dict[str, Any] | None:
    """Create sampled base + household sets for a new participant, or return existing.

    Usage counts are derived from remaining assignment files, so deleting a
    participant folder automatically lowers those tasks' counts. Older
    assignments that only have base tasks get a household sample filled in.
    """
    try:
        from experiment_config import load_experiment_config
    except ImportError:
        from interactive.experiment_config import load_experiment_config

    slug = slugify_user_name(name_or_slug)
    path = assignment_path(slug)
    cfg = cfg or load_experiment_config()
    existing = load_assignment_file(path)
    if existing:
        changed = False
        if not existing.get("base_tasks") and cfg.suite_enabled("base"):
            tasks, picks = _sample_suite(cfg, "base")
            existing["base_tasks"] = tasks
            existing["base_picks"] = picks
            changed = True
        if not existing.get("household_tasks") and cfg.suite_enabled("household"):
            tasks, picks = _sample_suite(cfg, "household")
            existing["household_tasks"] = tasks
            existing["household_picks"] = picks
            changed = True
        if changed:
            existing["path"] = str(path)
            _save_assignment(path, existing)
            write_usage_snapshot(cfg)
        return existing

    base_tasks, base_picks = _sample_suite(cfg, "base")
    household_tasks, household_picks = _sample_suite(cfg, "household")
    data = {
        "user_id": slug,
        "user_name": display or slug,
        "created_at": iso_now(),
        "base_tasks": base_tasks,
        "base_picks": base_picks,
        "household_tasks": household_tasks,
        "household_picks": household_picks,
        "path": str(path),
    }
    _save_assignment(path, data)
    write_usage_snapshot(cfg)
    return load_assignment_file(path) or data


def delete_experiment_logs(name_or_slug: str) -> bool:
    """Remove a participant folder and refresh usage counts from remaining logs."""
    slug = slugify_user_name(name_or_slug)
    folder = user_dir(slug)
    if not folder.is_dir():
        write_usage_snapshot()
        return False
    shutil.rmtree(folder)
    write_usage_snapshot()
    return True


def play_key(suite: str, task: str, scenario: str | None = None) -> str:
    suite = str(suite or "").strip()
    task = str(task or "").strip()
    if suite == "household" or not scenario:
        return f"{suite}:{task}"
    return f"{suite}:{task}:{scenario}"


def completed_keys(log: dict[str, Any] | None = None) -> set[str]:
    data = log if log is not None else load_user_log()
    if not data:
        return set()
    keys = data.get("completed_keys") or []
    return {str(k) for k in keys if k}


def is_completed(suite: str, task: str, scenario: str | None = None) -> bool:
    """True after at least one SUCCESS/FAILURE. Prefer ``is_slot_locked`` for limits."""
    return play_key(suite, task, scenario) in completed_keys()


def _play_matches_slot(play: dict[str, Any], suite: str, task: str, scenario: str | None) -> bool:
    if str(play.get("suite") or "") != suite:
        return False
    if str(play.get("task") or "") != task:
        return False
    if suite != "household":
        if str(play.get("scenario") or "") != str(scenario or ""):
            return False
    return True


def terminal_play_count(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> int:
    """How many SUCCESS/FAILURE plays this slot has (Stop / close do not count)."""
    data = log if log is not None else load_user_log()
    if not data:
        return 0
    key = play_key(suite, task, scenario)
    counts = data.get("play_counts") or {}
    if key in counts:
        try:
            return max(0, int(counts[key]))
        except (TypeError, ValueError):
            pass
    n = 0
    for play in data.get("plays") or []:
        if not isinstance(play, dict):
            continue
        if not _play_matches_slot(play, suite, task, scenario):
            continue
        if counts_as_completed(str(play.get("result") or "")):
            n += 1
    if n == 0 and key in completed_keys(data):
        return 1
    return n


def terminal_success_count(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> int:
    """How many of this slot's terminal plays were SUCCESS."""
    data = log if log is not None else load_user_log()
    if not data:
        return 0
    key = play_key(suite, task, scenario)
    counts = data.get("success_counts") or {}
    if key in counts:
        try:
            return max(0, int(counts[key]))
        except (TypeError, ValueError):
            pass
    n = 0
    for play in data.get("plays") or []:
        if not isinstance(play, dict):
            continue
        if not _play_matches_slot(play, suite, task, scenario):
            continue
        if str(play.get("result") or "") == "SUCCESS":
            n += 1
    return n


def slot_success_label(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> str:
    """``m/n`` successes over terminal plays, for a grayed-out slot."""
    total = terminal_play_count(suite, task, scenario, log=log)
    ok = terminal_success_count(suite, task, scenario, log=log)
    if total <= 0:
        return "0/0"
    ok = min(ok, total)
    return f"{ok}/{total}"


def is_slot_locked(
    suite: str,
    task: str,
    scenario: str | None = None,
    *,
    max_plays: int | None,
    log: dict[str, Any] | None = None,
) -> bool:
    """True when this slot is skipped (``0`` plays) or has used its play budget."""
    if max_plays is None:
        return False
    try:
        limit = int(max_plays)
    except (TypeError, ValueError):
        return False
    if limit <= 0:
        return True
    return terminal_play_count(suite, task, scenario, log=log) >= limit


def result_from_exit_code(exit_code: int | None, *, stopped: bool = False) -> str:
    if stopped:
        return "stopped"
    if exit_code == 0:
        return "SUCCESS"
    if exit_code == 10:
        return "FAILURE"
    if exit_code == 2:
        return "closed"
    if exit_code is None:
        return "unknown"
    return "error"


def counts_as_completed(result: str) -> bool:
    return result in ("SUCCESS", "FAILURE")


def payload_fail_reason(payload: dict[str, Any] | None) -> str:
    """Cause of failure from a task-result JSON payload, if any."""
    if not isinstance(payload, dict):
        return ""
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    for value in (
        payload.get("fail_reason"),
        metrics.get("fail_reason"),
        payload.get("detail"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def progress_counts(
    log: dict[str, Any] | None = None,
    *,
    base_task_names: list[str] | None = None,
    household_task_names: list[str] | None = None,
    n_scenarios: int = 4,
    cfg=None,
) -> dict[str, Any]:
    """Slots that have reached their play limit vs slots that have a limit.

    Unlimited slots (``plays_per_scenario: null``) and skipped slots (``0``)
    are omitted from the totals.
    """
    try:
        from experiment_config import (
            BASE_TASK_TABLE,
            HOUSEHOLD_TASK_TABLE,
            load_experiment_config,
        )
    except ImportError:
        from interactive.experiment_config import (
            BASE_TASK_TABLE,
            HOUSEHOLD_TASK_TABLE,
            load_experiment_config,
        )

    if cfg is None:
        cfg = load_experiment_config()
    scenarios = ("default", "opt1", "opt2", "opt1+2")[: max(1, int(n_scenarios))]
    if base_task_names is None:
        base_names = [key for _n, key, _label in BASE_TASK_TABLE]
    else:
        base_names = list(base_task_names)
    if household_task_names is None:
        household_names = [key for _n, key, _label in HOUSEHOLD_TASK_TABLE]
    else:
        household_names = list(household_task_names)

    scenario_done = {name: 0 for name in scenarios}
    scenario_total = {name: 0 for name in scenarios}
    base_done = 0
    base_total = 0
    for task in base_names:
        for scenario in scenarios:
            limit = cfg.max_plays("base", task, scenario)
            if limit is None or int(limit) <= 0:
                continue
            base_total += 1
            scenario_total[scenario] += 1
            if terminal_play_count("base", task, scenario, log=log) >= int(limit):
                base_done += 1
                scenario_done[scenario] += 1
    household_done = 0
    household_total = 0
    for task in household_names:
        limit = cfg.max_plays("household", task)
        if limit is None or int(limit) <= 0:
            continue
        household_total += 1
        if terminal_play_count("household", task, log=log) >= int(limit):
            household_done += 1
    return {
        "base_done": base_done,
        "base_total": base_total,
        "household_done": household_done,
        "household_total": household_total,
        "scenario_done": scenario_done,
        "scenario_total": scenario_total,
    }


def append_play(
    *,
    suite: str,
    task: str,
    task_label: str | None = None,
    scenario: str | None = None,
    controller: str | None = None,
    seed: int | None = None,
    exit_code: int | None = None,
    payload: dict[str, Any] | None = None,
    stopped: bool = False,
    wall_fallback_s: float | None = None,
    write_entry: bool = True,
) -> dict[str, Any] | None:
    """Record one play. Always bumps ``play_counts`` on SUCCESS/FAILURE.

    When ``write_entry`` is false, skip appending the detailed ``plays`` row
    (the slot is still counted so play limits work).
    """
    path_raw = os.environ.get(EXPERIMENT_LOG_ENV, "").strip()
    path = Path(path_raw) if path_raw else None
    if path is None:
        user = os.environ.get(EXPERIMENT_USER_ENV, "").strip()
        if user:
            path = user_log_path(user, controller or current_controller())
    if path is None:
        return None
    payload = dict(payload or {})
    data = load_user_log(path) or {
        "user_id": slugify_user_name(os.environ.get(EXPERIMENT_USER_ENV, "user")),
        "user_name": os.environ.get(EXPERIMENT_USER_ENV, "user"),
        "created_at": iso_now(),
        "experience": {},
        "post_survey": {},
        "completed_keys": [],
        "play_counts": {},
        "success_counts": {},
        "plays": [],
        "log_path": str(path),
    }
    tag = log_controller_tag(
        controller or payload.get("controller") or data.get("controller") or current_controller()
    )
    data["controller"] = tag
    data["log_path"] = str(path)
    result = result_from_exit_code(exit_code, stopped=stopped)
    if result in ("unknown", "error") and payload.get("ok") is True:
        result = "SUCCESS"
    elif result in ("unknown", "error") and payload.get("ok") is False:
        result = "FAILURE"

    time_block = payload.get("time") if isinstance(payload.get("time"), dict) else {}
    wall = time_block.get("wall_s")
    if wall is None:
        wall = time_block.get("wall_clock_s")
    if wall is None and wall_fallback_s is not None:
        wall = round(float(wall_fallback_s), 4)
        time_block = dict(time_block)
        time_block["wall_s"] = wall
        time_block["wall_clock_s"] = wall

    sim_s = time_block.get("total_time_sim_s")
    if sim_s is None:
        sim_s = time_block.get("simulation_s")
    steps = time_block.get("steps")
    if steps is None:
        steps = time_block.get("simulation_steps")

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics = dict(metrics)
    if "success" not in metrics and payload.get("ok") is not None:
        metrics["success"] = bool(payload.get("ok"))
    if metrics.get("partial_score") is None and payload.get("partial_score") is not None:
        try:
            metrics["partial_score"] = float(payload.get("partial_score"))
        except (TypeError, ValueError):
            pass
    if metrics.get("total_time_sim_s") is None and sim_s is not None:
        metrics["total_time_sim_s"] = sim_s
    if metrics.get("wall_s") is None and wall is not None:
        metrics["wall_s"] = wall
    if metrics.get("steps") is None and steps is not None:
        metrics["steps"] = steps
    option = (
        payload.get("option_label")
        or metrics.get("option_label")
        or (scenario if suite == "base" else None)
    )
    if option and not metrics.get("option_label"):
        metrics["option_label"] = option

    fail_reason = payload_fail_reason(payload) if result == "FAILURE" else ""
    if fail_reason and not metrics.get("fail_reason"):
        metrics["fail_reason"] = fail_reason
    detail = str(payload.get("detail") or fail_reason or "").strip()

    entry = {
        "played_at": iso_now(),
        "suite": suite,
        "task": task,
        "task_label": task_label or task,
        "scenario": scenario,
        "option_label": option if suite == "base" else (option or None),
        "controller": controller or payload.get("controller") or "",
        "seed": seed if seed is not None else payload.get("seed"),
        "result": result,
        "detail": detail,
        "fail_reason": fail_reason,
        "condition": payload.get("condition") or "",
        "partial_score": metrics.get("partial_score", payload.get("partial_score")),
        "exit_code": exit_code,
        "metrics": metrics,
        "time": {
            "total_time_sim_s": sim_s if sim_s is not None else metrics.get("total_time_sim_s"),
            "wall_s": wall if wall is not None else metrics.get("wall_s"),
            "steps": steps if steps is not None else metrics.get("steps"),
            # Legacy aliases
            "wall_clock_s": wall if wall is not None else metrics.get("wall_s"),
            "simulation_s": sim_s if sim_s is not None else metrics.get("total_time_sim_s"),
            "simulation_steps": steps if steps is not None else metrics.get("steps"),
        },
    }
    counted = False
    if counts_as_completed(result):
        key = play_key(suite, task, scenario)
        counts = dict(data.get("play_counts") or {})
        try:
            counts[key] = int(counts.get(key, 0) or 0) + 1
        except (TypeError, ValueError):
            counts[key] = 1
        data["play_counts"] = counts
        if result == "SUCCESS":
            wins = dict(data.get("success_counts") or {})
            try:
                wins[key] = int(wins.get(key, 0) or 0) + 1
            except (TypeError, ValueError):
                wins[key] = 1
            data["success_counts"] = wins
        counted = True
        done = list(data.get("completed_keys") or [])
        if key not in done:
            done.append(key)
        data["completed_keys"] = done
    if write_entry:
        plays = list(data.get("plays") or [])
        plays.append(entry)
        data["plays"] = plays
    elif not counted:
        return None
    save_user_log(path, data)
    return entry


def stamp_child_env(
    env: dict[str, str],
    *,
    controller: str,
    seed: int,
    scenario: str | None = None,
) -> dict[str, str]:
    env["ROBODYNA_CONTROL"] = str(controller)
    env["ROBODYNA_SEED"] = str(seed)
    if scenario:
        env["ROBODYNA_SCENARIO"] = str(scenario)
    return env


def child_experiment_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for a spawned task GUI, preserving the logged-in user."""
    env = dict(base if base is not None else os.environ)
    env[EXPERIMENT_ENV] = "1"
    for key in (EXPERIMENT_USER_ENV, EXPERIMENT_LOG_ENV):
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value
    return env


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
