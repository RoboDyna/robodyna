"""Partial scores for the Base / Conceptual interactive tasks.

The task rules here deliberately credit only observable, completed actions.
They do not use spatial near-misses or continuous distance shaping.  A wrong
action that is relevant to a task applies a 0.5 multiplier; repeated actions
apply it repeatedly.  Household tasks keep their own ``get_score`` methods
and are intentionally not registered here.
"""

from __future__ import annotations

from typing import Any, Callable

from .partial_score import completed_action_score


def _bool(value: Any) -> bool:
    return bool(value)


def _call(env: Any, name: str, *args: Any, default: Any = False) -> Any:
    method = getattr(env, name, None)
    if not callable(method):
        return default
    try:
        return method(*args)
    except Exception:
        return default


def _full(env: Any) -> bool:
    return _bool(_call(env, "check_success", default=False))


def _score_catch_marbles(env: Any) -> float:
    target = int(getattr(env, "target_button_idx", -1))
    dropped_door = int(getattr(env, "_ball_drop_door_idx", -1))
    target_lower = _bool(_call(env, "_ball_in_lower_box"))
    target_correct = target_lower and target >= 0 and dropped_door == target
    target_wrong = target_lower and dropped_door >= 0 and dropped_door != target
    base = 1.0 if target_correct else 0.5 if target_wrong else 0.0
    distractor_error = _bool(_call(env, "_distractor_through_any")) or _bool(
        _call(env, "_distractor_in_lower_box")
    )
    opens = _call(env, "_total_door_opens", default=0)
    try:
        extra_opens = max(0, int(opens) - 1)
    except (TypeError, ValueError):
        extra_opens = 0
    return completed_action_score(
        env,
        base,
        completed_actions={
            "target_marble_through_matching_door": target_correct,
            "target_marble_through_wrong_door": target_wrong,
        },
        penalties={
            "distractor_entered_a_trapdoor": distractor_error,
            "extra_door_open": extra_opens,
        },
    )


def _score_catch_ramp(env: Any) -> float:
    state = _call(env, "_catch_state", default=(0.0, False, False, None, None))
    in_vessel = bool(state[1]) if len(state) > 1 else False
    distractor_in = bool(state[2]) if len(state) > 2 else False
    caught = in_vessel or getattr(env, "_metric_catch_step", None) is not None
    return completed_action_score(
        env,
        float(caught),
        completed_actions={"target_ball_caught_in_cup": caught},
        penalties={"distractor_ball_caught": distractor_in},
    )


def _score_catch_cuboid(env: Any) -> float:
    held = list(_call(env, "_cuboids_held", default=[]))
    pulled = list(_call(env, "_cuboids_pulled_out", default=[]))
    required = 2 if _bool(getattr(env, "catch_two_cuboids", False)) else 1
    done = sum(
        1
        for idx in range(required)
        if idx < len(held) and idx < len(pulled) and held[idx] and pulled[idx]
    )
    return completed_action_score(
        env,
        done / float(required),
        completed_actions={"cuboids_grasped_and_lifted_clear": f"{done}/{required}"},
    )


def _score_catch_shelf_marble(env: Any) -> float:
    caught = (
        getattr(env, "_marble_state", None) == "resolved"
        and getattr(env, "_marble_result", None) == "caught"
    )
    return completed_action_score(
        env, float(caught), completed_actions={"marble_caught": caught}
    )


def _score_catch_valley(env: Any) -> float:
    state = _call(env, "_catch_state", default=(0.0, False, False, None, None, False))
    in_bowl = bool(state[1]) if len(state) > 1 else False
    behind_line = bool(state[2]) if len(state) > 2 else False
    distractor_in = bool(state[5]) if len(state) > 5 else False
    caught = in_bowl or getattr(env, "_metric_catch_step", None) is not None
    base = 1.0 if caught and behind_line else 0.5 if caught else 0.0
    return completed_action_score(
        env,
        base,
        completed_actions={
            "target_ball_caught": caught,
            "catch_box_past_red_line": bool(caught and behind_line),
        },
        penalties={
            "distractor_ball_caught": distractor_in,
            "robot_arm_touched_ball": _bool(getattr(env, "_arm_ball_contact", False)),
        },
    )


def _score_stop_valley(env: Any) -> float:
    panel_hit = _bool(getattr(env, "_panel_hit", False))
    try:
        panel_in_air = float(env.panel.get_pose().p[2]) >= (
            float(env.table_top) + float(env.INTERCEPT_MIN_CLEARANCE_DEFAULT)
        )
    except Exception:
        panel_in_air = False
    base = 1.0 if panel_hit and panel_in_air else 0.5 if panel_hit else 0.0
    return completed_action_score(
        env,
        base,
        completed_actions={
            "ball_hit_stop_panel": panel_hit,
            "panel_was_held_in_air": bool(panel_hit and panel_in_air),
        },
        penalties={
            "robot_arm_touched_ball": _bool(getattr(env, "_arm_ball_contact", False)),
            "distractor_hit_stop_panel": _bool(getattr(env, "_distractor_panel_hit", False)),
            "ball_hit_table_before_panel": _bool(getattr(env, "_ball_table_before_hit", False)),
        },
    )


def _station_partial(env: Any, station: dict[str, Any]) -> tuple[float, str]:
    grasp = station.get("grasp_doneness")
    if grasp is None:
        return 0.0, "not_stopped"
    stopped = (
        (not _bool(_call(env, "_button_is_pressed_station", station)))
        and (
            _bool(station.get("cook_phase_done", False))
            if _bool(getattr(env, "use_hold_cook", False))
            else not _bool(station.get("cook_on", False))
        )
    )
    if not stopped:
        return 0.0, "not_stopped"
    try:
        low, high = _call(env, "_doneness_range_bounds", default=(0.0, 0.0))
        centre = 0.5 * (float(low) + float(high))
        half = max(0.5 * (float(high) - float(low)), 1e-9)
        error = abs(float(grasp) - centre) / half
    except (TypeError, ValueError):
        return 0.0, "invalid_reading"
    if error <= 1.0:
        return 1.0, "inside_success_range"
    if error <= 2.0:
        return 0.5, "inside_two_x_range"
    return 0.0, "outside_two_x_range"


def _score_cook_meat(env: Any) -> float:
    stations = list(getattr(env, "stations", None) or [])
    if not stations:
        grasp = getattr(env, "_grasp_doneness", None)
        stations = [{
            "grasp_doneness": grasp,
            "cook_on": False,
            "cook_phase_done": True,
        }]
    values, labels = zip(*(_station_partial(env, station) for station in stations))
    base = sum(values) / float(len(values))
    return completed_action_score(
        env,
        base,
        completed_actions={
            "stations_stopped_within_two_x_doneness_range": list(labels),
            "station_credit": [float(value) for value in values],
        },
    )


def _score_put_cup_belt(env: Any) -> float:
    held = _bool(_call(env, "_cup_held"))
    placed_on_belt = False
    try:
        p = env.cup.get_functional_point(0, "pose").p
        z_ok = (float(env.slot_z) - 0.04) < float(p[2]) < (float(env.slot_z) + 0.12)
        y_ok = abs(float(p[1]) - float(env.belt_y)) <= float(env.belt_plate_half_size[1]) + 0.03
        placed_on_belt = bool((not held) and z_ok and y_ok)
    except Exception:
        pass
    in_slot = placed_on_belt and _bool(_call(env, "_cup_between_yellow_tools"))
    return completed_action_score(
        env,
        1.0 if in_slot else 0.5 if placed_on_belt else 0.0,
        completed_actions={
            "cup_released_on_belt": placed_on_belt,
            "cup_seated_between_yellow_tools": in_slot,
        },
        penalties={"hit_blue_curtain": _bool(getattr(env, "_curtain_hit", False))},
    )


def _score_dispense_gummy(env: Any) -> float:
    target = getattr(env, "target_color", "")
    caught = dict(getattr(env, "_caught_by_color", {}) or {})
    target_caught = int(caught.get(target, 0))
    total = max(1, int(getattr(env, "total_target", 1) or 1))
    non_target = sum(int(value) for color, value in caught.items() if color != target)
    presses = len(list(getattr(env, "press_history", []) or []))
    return completed_action_score(
        env,
        min(1.0, target_caught / float(total)),
        completed_actions={"target_gummies_caught": f"{target_caught}/{total}"},
        penalties={
            "non_target_gummy_caught": non_target,
            "invalid_dispense_pattern": _bool(getattr(env, "invalid_pattern", False)),
            "extra_dispense_press": max(0, presses - total),
        },
    )


def _score_punch_dual_holes(env: Any) -> float:
    complete = 0
    expected = 0
    for side in ("left", "right"):
        missing = list(getattr(env, "page_missing", {}).get(side, []) or [])
        missed = list(getattr(env, "page_missed", {}).get(side, []) or [])
        offsets = list(getattr(env, "page_offset", {}).get(side, []) or [])
        for idx, is_missing in enumerate(missing):
            if is_missing:
                continue
            expected += 1
            if idx < len(offsets) and offsets[idx] is not None and not (idx < len(missed) and missed[idx]):
                complete += 1
    base = complete / float(expected) if expected else 0.0
    return completed_action_score(
        env,
        base,
        completed_actions={"present_tiles_correctly_punched": f"{complete}/{expected}"},
        penalties={"pressed_missing_tile": int(getattr(env, "invalid_empty_press_count", 0) or 0)},
    )


def _score_save_goal(env: Any) -> float:
    keeper = _bool(_call(env, "_keeper_in_zone")) and _bool(getattr(env, "_block_was_legal", False))
    blocked = _bool(getattr(env, "_ball_blocked", False))
    released = _bool(_call(env, "is_left_gripper_open")) and _bool(_call(env, "is_right_gripper_open"))
    # A legal keeper placement and a block are the two substantive milestones;
    # release is required before they receive the full task score.
    base = 0.5 * float(keeper) + 0.25 * float(blocked) + 0.25 * float(released)
    return completed_action_score(
        env,
        base,
        completed_actions={
            "keeper_legally_placed": keeper,
            "shot_blocked": blocked,
            "both_grippers_released": released,
        },
        penalties={
            "late_failure_after_block": _bool(getattr(env, "_late_failure", False)),
            "goal_conceded": _bool(getattr(env, "_goal_conceded", False)),
        },
    )


def _score_hit_target(env: Any) -> float:
    if _bool(getattr(env, "_hit_blocker", False)):
        return completed_action_score(
            env, 0.0, completed_actions={"dart_stuck_on_target": False},
            penalties={"hit_blocker": True},
        )
    stuck = _bool(getattr(env, "_stuck", False))
    hit_score = max(0.0, min(1.0, float(getattr(env, "hit_score", 0.0)))) if stuck else 0.0
    return completed_action_score(
        env,
        hit_score,
        completed_actions={"dart_stuck_on_target": stuck, "target_ring_credit": hit_score},
    )


def _score_load_train(env: Any) -> float:
    _call(env, "_try_latch_ball_off_table")
    if _bool(getattr(env, "_ball_fell_off_table", False)):
        return completed_action_score(
            env, 0.0, completed_actions={"ball_loaded_into_wagon": False},
            penalties={"ball_fell_off_table": True},
        )
    target_mode = _bool(getattr(env, "target_wagon_mode", False))
    target_idx = getattr(env, "target_wagon_idx", None)
    latched = getattr(env, "_latched_car_idx", None)
    cargo = list(getattr(env, "_cargo_indices", []) or [])
    loaded = _bool(getattr(env, "_ball_latched", False))
    if not loaded:
        loaded_idx = next((idx for idx in cargo if _bool(_call(env, "_ball_in_car", idx))), None)
    else:
        loaded_idx = latched
    if loaded_idx is None:
        base = 0.0
        correct = False
    elif target_mode and target_idx is not None:
        correct = int(loaded_idx) == int(target_idx)
        base = 1.0 if correct else 0.5
    else:
        correct = True
        base = 1.0
    return completed_action_score(
        env,
        base,
        completed_actions={
            "ball_loaded_into_wagon": loaded_idx is not None,
            "ball_loaded_into_target_wagon": correct,
        },
    )


def _score_marble_shelf_maze(env: Any) -> float:
    in_bowl = _bool(_call(env, "_ball_in_bowl"))
    if in_bowl and not _bool(_call(env, "_ball_on_table")):
        return completed_action_score(
            env, 1.0, completed_actions={"marble_caught_in_bowl": True}
        )
    n_shelves = max(1, int(getattr(env, "n_shelves", 1) or 1))
    deepest = max(0, int(getattr(env, "_partial_deepest_shelf", 0) or 0))
    if n_shelves <= 1:
        base = 0.0
    else:
        base = 0.75 * min(1.0, deepest / float(n_shelves - 1))
    return completed_action_score(
        env,
        base,
        completed_actions={"deepest_shelf_reached": f"{deepest}/{n_shelves - 1}"},
    )


def _score_pack_fruits(env: Any) -> float:
    n_items = max(1, int(getattr(env, "n_items", 1) or 1))
    correct = sum(1 for idx in range(n_items) if _bool(_call(env, "_fruit_in_basket", idx)))
    n_distractors = int(getattr(env, "n_distractor_slots", 0) or 0)
    distractors = sum(
        1 for idx in range(n_distractors) if _bool(_call(env, "_distractor_in_any_basket", idx))
    )
    return completed_action_score(
        env,
        correct / float(n_items),
        completed_actions={"correct_fruits_packed": f"{correct}/{n_items}"},
        penalties={"distractor_fruit_packed": distractors},
    )


def _score_pick_ripe_apple(env: Any) -> float:
    detached_in_window = (
        getattr(env, "r_grasp", None) is not None and _bool(_call(env, "_grasp_in_red_window"))
    )
    settled = False
    spoiled_in = False
    try:
        basket_xy = env.basket.get_pose().p[:2]
        apple_p = env.apple.get_pose().p
        settled = bool(
            _call(env, "_pose_in_basket", apple_p, basket_xy)
            and not _bool(_call(env, "_apple_held_by_gripper", env.apple))
            and float(_call(env, "_actor_speed", env.apple, default=float("inf"))) < 0.12
        )
        spoiled = getattr(env, "spoiled_apple", None)
        if spoiled is not None:
            spoiled_in = _bool(_call(env, "_pose_in_basket", spoiled.get_pose().p, basket_xy))
    except Exception:
        pass
    base = 0.5 * float(detached_in_window) + 0.5 * float(detached_in_window and settled)
    return completed_action_score(
        env,
        base,
        completed_actions={
            "ripe_apple_detached_in_window": detached_in_window,
            "ripe_apple_settled_in_basket": bool(detached_in_window and settled),
        },
        penalties={"spoiled_apple_in_basket": spoiled_in},
    )


def _score_place_block_belt(env: Any) -> float:
    # The task's success check latches the first real belt contact, which is an
    # action milestone rather than a distance estimate.
    _full(env)
    on_belt = _bool(getattr(env, "dropped_at_start_left", False)) and _bool(
        getattr(env, "placed_on_belt", False)
    )
    in_bowl = _bool(getattr(env, "in_bowl", False))
    return completed_action_score(
        env,
        0.5 * float(on_belt) + 0.5 * float(on_belt and in_bowl),
        completed_actions={
            "block_released_on_belt_before_line": on_belt,
            "block_carried_into_bowl": bool(on_belt and in_bowl),
        },
        penalties={"block_hit_blocker": _bool(getattr(env, "hit_blocker", False))},
    )


def _score_play_billiard(env: Any) -> float:
    _full(env)  # scans the pocket state if the ball has just settled.
    foul = (
        _bool(getattr(env, "_robot_ball_contact", False))
        or _bool(getattr(env, "_cue_distractor_contact", False))
        or _bool(getattr(env, "_distractor_pocketed", False))
    )
    struck = _bool(getattr(env, "_strike_done", False))
    pocketed = _bool(getattr(env, "_primary_pocketed", False))
    allowed = set(getattr(env, "_allowed_pocket_ids", []) or [])
    pocket_id = getattr(env, "_primary_pocket_id", None)
    correct_pocket = pocketed and pocket_id in allowed
    wrong_pocket = pocketed and not correct_pocket
    base = 0.0 if foul else 1.0 if correct_pocket else 0.5 if wrong_pocket else 0.25 if struck else 0.0
    return completed_action_score(
        env,
        base,
        completed_actions={
            "cue_struck_target_ball": struck,
            "target_ball_in_allowed_pocket": correct_pocket,
            "target_ball_in_wrong_pocket": wrong_pocket,
        },
        penalties={"billiards_foul": foul},
    )


def _score_control_quality(env: Any) -> float:
    colors = list(getattr(env, "tile_colors", []) or [])
    marked = list(getattr(env, "tile_marked", []) or [])
    correct = list(getattr(env, "tile_correct", []) or [])
    skipped = list(getattr(env, "tile_skipped", []) or [])
    n = max(1, len(colors))
    completed = 0
    for idx, color in enumerate(colors):
        if color == "black":
            completed += int(idx < len(skipped) and skipped[idx] and not (idx < len(marked) and marked[idx]))
        else:
            completed += int(idx < len(marked) and idx < len(correct) and marked[idx] and correct[idx])
    return completed_action_score(
        env,
        completed / float(n),
        completed_actions={"tiles_correctly_processed": f"{completed}/{n}"},
        penalties={"black_tile_pressed": _bool(getattr(env, "black_press", False))},
    )


def _score_drop_ball_hole(env: Any) -> float:
    _call(env, "_try_latch_ball_off_table")
    real_hole = _bool(_call(env, "_ball_in_box")) and not _bool(
        getattr(env, "_ball_fell_off_table", False)
    )
    dummy_hole = (
        _bool(getattr(env, "add_dummy_hole", False))
        and _bool(_call(env, "_ball_in_dummy_hole"))
    )
    return completed_action_score(
        env,
        1.0 if real_hole else 0.5 if dummy_hole else 0.0,
        completed_actions={
            "ball_reached_real_hole_box": real_hole,
            "ball_reached_dummy_hole": dummy_hole,
        },
        penalties={"ball_fell_off_table": _bool(getattr(env, "_ball_fell_off_table", False))},
    )


def _score_sort_apples_belt(env: Any) -> float:
    _call(env, "_eval_landings")
    results = list(getattr(env, "results", []) or [])
    n = max(1, int(getattr(env, "n_apples", len(results)) or len(results) or 1))
    correct = sum(1 for result in results if result)
    return completed_action_score(
        env,
        correct / float(n),
        completed_actions={"apples_correctly_sorted": f"{correct}/{n}"},
        penalties={"rotten_apple_in_basket": _bool(_call(env, "_rotten_in_basket"))},
    )


def _score_whack_moles(env: Any) -> float:
    touched = list(getattr(env, "touched", []) or [])
    n = max(1, int(getattr(env, "num_moles", len(touched)) or len(touched) or 1))
    complete = sum(1 for value in touched if value)
    return completed_action_score(
        env,
        min(1.0, complete / float(n)),
        completed_actions={"moles_whacked": f"{complete}/{n}"},
        penalties={
            "rabbit_hit": _bool(getattr(env, "distractor_hit", False)),
            "board_hit": _bool(getattr(env, "board_hit", False)),
        },
    )


_SCORERS: dict[str, Callable[[Any], float]] = {
    "catch_marbles_trapdoors": _score_catch_marbles,
    "catch_ramp_ball": _score_catch_ramp,
    "catch_cuboid": _score_catch_cuboid,
    "catch_shelf_marble": _score_catch_shelf_marble,
    "catch_valley_ball": _score_catch_valley,
    "stop_valley_ball": _score_stop_valley,
    "cook_meat": _score_cook_meat,
    "cook_meat_timer": _score_cook_meat,
    "put_cup_belt": _score_put_cup_belt,
    "dispense_gummy": _score_dispense_gummy,
    "punch_dual_holes": _score_punch_dual_holes,
    "save_goal": _score_save_goal,
    "hit_target": _score_hit_target,
    "load_train": _score_load_train,
    "marble_shelf_maze": _score_marble_shelf_maze,
    "pack_fruits": _score_pack_fruits,
    "pick_ripe_apple": _score_pick_ripe_apple,
    "place_block_belt": _score_place_block_belt,
    "play_billiard": _score_play_billiard,
    "control_quality": _score_control_quality,
    "drop_ball_hole": _score_drop_ball_hole,
    "sort_apples_belt": _score_sort_apples_belt,
    "whack_moles": _score_whack_moles,
}


def score_for_task(env: Any) -> float | None:
    """Score a registered Base / Conceptual task, else return ``None``."""
    scorer = _SCORERS.get(type(env).__name__)
    if scorer is None:
        return None
    try:
        return float(scorer(env))
    except Exception:
        return None


def registered_task_names() -> tuple[str, ...]:
    """Expose the registry for lightweight coverage checks."""
    return tuple(_SCORERS)
