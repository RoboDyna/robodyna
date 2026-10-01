"""Canonical task scenarios shared by GUI, evaluation, and collection.

``demo_dynamic.yml`` supplies each task's continuous parameters.  This module
defines the discrete Base-suite condition matrix, so a scenario always means
the same flags regardless of whether a person, an expert collector, or a
policy evaluator is running it.
"""
from __future__ import annotations

BASE_SCENARIOS = ("default", "opt1", "opt2", "opt1+2")

BASE_TASKS = (
    "catch_marbles_trapdoors", "catch_ramp_ball", "catch_cuboid",
    "catch_shelf_marble", "catch_valley_ball", "stop_valley_ball",
    "cook_meat", "cook_meat_timer", "put_cup_belt", "dispense_gummy",
    "punch_dual_holes", "save_goal", "hit_target", "load_train",
    "marble_shelf_maze", "pack_fruits", "pick_ripe_apple",
    "place_block_belt", "play_billiard", "control_quality",
    "drop_ball_hole", "sort_apples_belt", "whack_moles",
)

HOUSEHOLD_TASKS = (
    "trap_bug", "boil_milk", "fill_coffee_jar", "pour_beer", "cook_food",
    "cook_food_timer", "measure_ingredient", "make_soup", "catch_cup",
    "catch_mouse_object_drop", "stop_ball", "clean_table",
)

# Explicit defaults prevent a stale value in demo_dynamic.yml from silently
# changing the meaning of a named scenario.
SCENARIO_OVERRIDES = {
    "catch_marbles_trapdoors": {
        "default": {"door_open_once": False, "enable_distractor": False},
        "opt1": {"door_open_once": True, "enable_distractor": False},
        "opt2": {"door_open_once": False, "enable_distractor": True},
        "opt1+2": {"door_open_once": True, "enable_distractor": True},
    },
    "catch_ramp_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "catch_cuboid": {
        "default": {"catch_two_cuboids": False, "opaque_surface": False},
        "opt1": {"catch_two_cuboids": True, "opaque_surface": False},
        "opt2": {"catch_two_cuboids": False, "opaque_surface": True},
        "opt1+2": {"catch_two_cuboids": True, "opaque_surface": True},
    },
    "catch_shelf_marble": {
        "default": {"reactive_marble": False, "oscillating_shelf_enabled": False},
        "opt1": {"reactive_marble": True, "oscillating_shelf_enabled": False},
        "opt2": {"reactive_marble": False, "oscillating_shelf_enabled": True},
        "opt1+2": {"reactive_marble": True, "oscillating_shelf_enabled": True},
    },
    "catch_valley_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "stop_valley_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "cook_meat": {
        "default": {"cook_button_enabled": False, "dual_setup_enabled": False},
        "opt1": {"cook_button_enabled": True, "dual_setup_enabled": False},
        "opt2": {"cook_button_enabled": False, "dual_setup_enabled": True},
        "opt1+2": {"cook_button_enabled": True, "dual_setup_enabled": True},
    },
    "cook_meat_timer": {
        "default": {"cook_button_enabled": False, "dual_setup_enabled": False},
        "opt1": {"cook_button_enabled": True, "dual_setup_enabled": False},
        "opt2": {"cook_button_enabled": False, "dual_setup_enabled": True},
        "opt1+2": {"cook_button_enabled": True, "dual_setup_enabled": True},
    },
    "put_cup_belt": {
        "default": {"blue_curtains_enabled": False, "blue_curtain_dynamic_enabled": False},
        "opt1": {"blue_curtains_enabled": True, "blue_curtain_dynamic_enabled": False},
        "opt2": {"blue_curtains_enabled": False, "blue_curtain_dynamic_enabled": True},
        "opt1+2": {"blue_curtains_enabled": True, "blue_curtain_dynamic_enabled": True},
    },
    "dispense_gummy": {
        "default": {"layout_mode": "alternating", "belt_continuous_motion": False},
        "opt1": {"layout_mode": "random", "belt_continuous_motion": False},
        "opt2": {"layout_mode": "alternating", "belt_continuous_motion": True},
        "opt1+2": {"layout_mode": "random", "belt_continuous_motion": True},
    },
    "punch_dual_holes": {
        "default": {"missing_tile_mode": False, "belt_continous_motion": False},
        "opt1": {"missing_tile_mode": True, "belt_continous_motion": False},
        "opt2": {"missing_tile_mode": False, "belt_continous_motion": True},
        "opt1+2": {"missing_tile_mode": True, "belt_continous_motion": True},
    },
    "save_goal": {
        "default": {"players_enabled": False, "cover_enabled": False},
        "opt1": {"players_enabled": True, "cover_enabled": False},
        "opt2": {"players_enabled": False, "cover_enabled": True},
        "opt1+2": {"players_enabled": True, "cover_enabled": True},
    },
    "hit_target": {
        "default": {"blocker_enabled": False, "blocker_dynamic": False},
        "opt1": {"blocker_enabled": True, "blocker_dynamic": False},
        "opt2": {"blocker_enabled": False, "blocker_dynamic": True},
        "opt1+2": {"blocker_enabled": True, "blocker_dynamic": True},
    },
    "load_train": {
        "default": {"target_wagon_mode": False, "tunnel_enabled": False},
        "opt1": {"target_wagon_mode": True, "tunnel_enabled": False},
        "opt2": {"target_wagon_mode": False, "tunnel_enabled": True},
        "opt1+2": {"target_wagon_mode": True, "tunnel_enabled": True},
    },
    "marble_shelf_maze": {
        "default": {"continuous_ball_motion": False, "oscillating_bowl_enabled": False},
        "opt1": {"continuous_ball_motion": True, "oscillating_bowl_enabled": False},
        "opt2": {"continuous_ball_motion": False, "oscillating_bowl_enabled": True},
        "opt1+2": {"continuous_ball_motion": True, "oscillating_bowl_enabled": True},
    },
    "pack_fruits": {
        "default": {"two_colors_enabled": False, "distractor_enabled": False},
        "opt1": {"two_colors_enabled": True, "distractor_enabled": False},
        "opt2": {"two_colors_enabled": False, "distractor_enabled": True},
        "opt1+2": {"two_colors_enabled": True, "distractor_enabled": True},
    },
    "pick_ripe_apple": {
        "default": {"two_apples_enabled": False, "basket_move_enabled": False},
        "opt1": {"two_apples_enabled": True, "basket_move_enabled": False},
        "opt2": {"two_apples_enabled": False, "basket_move_enabled": True},
        "opt1+2": {"two_apples_enabled": True, "basket_move_enabled": True},
    },
    "place_block_belt": {
        "default": {"bowl_move_enabled": False, "blocker_enabled": False},
        "opt1": {"bowl_move_enabled": True, "blocker_enabled": False},
        "opt2": {"bowl_move_enabled": False, "blocker_enabled": True},
        "opt1+2": {"bowl_move_enabled": True, "blocker_enabled": True},
    },
    "play_billiard": {
        "default": {"specific_hole": False, "enable_distractors": False},
        "opt1": {"specific_hole": True, "enable_distractors": False},
        "opt2": {"specific_hole": False, "enable_distractors": True},
        "opt1+2": {"specific_hole": True, "enable_distractors": True},
    },
    "control_quality": {
        "default": {"color_mode": "alternating", "black_frac_max": 0.0},
        "opt1": {"color_mode": "random", "black_frac_max": 0.0},
        "opt2": {"color_mode": "alternating", "black_frac_max": 0.5},
        "opt1+2": {"color_mode": "random", "black_frac_max": 0.5},
    },
    "drop_ball_hole": {
        "default": {"stick_to_surface": False, "add_dummy_hole": False},
        "opt1": {"stick_to_surface": True, "add_dummy_hole": False},
        "opt2": {"stick_to_surface": False, "add_dummy_hole": True},
        "opt1+2": {"stick_to_surface": True, "add_dummy_hole": True},
    },
    "sort_apples_belt": {
        "default": {"color_mode": "alternating", "rotten_prob": 0.0},
        "opt1": {"color_mode": "random", "rotten_prob": 0.0},
        "opt2": {"color_mode": "alternating", "rotten_prob": 0.3},
        "opt1+2": {"color_mode": "random", "rotten_prob": 0.3},
    },
    "whack_moles": {
        "default": {"distractor_enabled": False, "relocating_moles": False},
        "opt1": {"distractor_enabled": True, "relocating_moles": False},
        "opt2": {"distractor_enabled": False, "relocating_moles": True},
        "opt1+2": {"distractor_enabled": True, "relocating_moles": True},
    },
}


def normalize_base_scenario(scenario: str | None) -> str:
    """Validate a Base scenario; ``base`` remains a backward-compatible alias."""
    selected = str(scenario or "default").strip().lower()
    selected = {"base": "default"}.get(selected, selected)
    if selected not in BASE_SCENARIOS:
        choices = ", ".join(BASE_SCENARIOS)
        raise ValueError(f"Unknown Base scenario {scenario!r}; choose {choices}.")
    return selected


def apply_base_scenario(config: dict, task: str, scenario: str | None) -> str:
    """Apply one Base-suite scenario to a loaded task configuration in place."""
    task = str(task).strip().replace("-", "_")
    selected = normalize_base_scenario(scenario)
    try:
        overrides = SCENARIO_OVERRIDES[task][selected]
    except KeyError as exc:
        raise ValueError(f"{task!r} is not a Base-suite task with scenarios.") from exc
    config.setdefault("task_args", {}).setdefault(task, {}).update(overrides)
    # A few environments read these markers in addition to their task_args.
    config["interactive_scenario"] = selected
    config["interactive_task"] = task
    return selected


def apply_collection_scenario(config: dict, task: str, scenario: str | None) -> str:
    """Apply the one supported collection scenario for a Base or Household task."""
    task = str(task).strip().replace("-", "_")
    if task in SCENARIO_OVERRIDES:
        return apply_base_scenario(config, task, scenario)
    if task not in HOUSEHOLD_TASKS:
        raise ValueError(f"Unknown collection task {task!r}.")
    selected = normalize_base_scenario(scenario)
    if selected != "default":
        raise ValueError(f"Household task {task!r} only supports the default scenario.")
    # This is the same static setting used by the household success sweep.
    config["use_dynamic"] = False
    config["interactive_scenario"] = selected
    config["interactive_task"] = task
    return selected
