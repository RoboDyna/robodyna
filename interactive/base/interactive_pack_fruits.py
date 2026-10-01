#!/usr/bin/env python3
"""Interactive sandbox for ``pack_fruits``.

Pack red apples into the left basket and green apples into the right
(when both colors are present). Opt2 black distractors pinch and drop
the same way as colored fruit; packing one into a basket fails the task.

Keyboard+mouse: click a fruit (highlights lighter), then click a basket.
Robot: physical pinch grasp (no teleport / no EE weld).

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_pack_fruits.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_pack_fruits.py --control robot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interactive_common import (  # noqa: E402
    actor_scene_id,
    add_robot_motion_arg,
    bootstrap_repo,
    click_hits_actor_map,
    configure_task,
    gripper_width,
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    print_episode_condition,
    print_instructions,
    run_viewer_loop,
    table_xy_from_click,
)

bootstrap_repo()


# Pinch window (looser than the expert attach thresholds).
PINCH_XY = 0.055
PINCH_Z_MAX = 0.08
GRASP_SETTLE_STEPS = 28
DROP_SETTLE_STEPS = 90
CLOSE_WIDTH = 0.45
OPEN_WIDTH = 0.55


def _tcp_fruit_gap(env, key, arm_name: str):
    kind, i = key
    actor = env.distractors[i] if kind == "dist" else env.items[i]
    tcp = env._tcp_pos(arm_name)
    fp = np.array(actor.get_pose().p, dtype=float)
    xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
    dz = float(tcp[2] - fp[2])
    return xy, dz


def _nearest_graspable(env, arm_name: str):
    """Closest belt apple under the gripper, including Opt2 black fruit."""
    best = None
    best_score = 1e9
    grasping_d = getattr(env, "_grasping_dist_slots", set()) or set()
    candidates = []
    for idx in range(env.n_items):
        if (
            not env._spawned_mask[idx]
            or env._packed[idx]
            or env._missed[idx]
            or env._welded[idx]
            or idx in env._grasping_idxs
            or env._item_y[idx] is None
        ):
            continue
        candidates.append(("item", idx))
    for slot in range(int(getattr(env, "n_distractor_slots", 0) or 0)):
        if env._distractor_y[slot] is None or slot in grasping_d:
            continue
        candidates.append(("dist", slot))
    for key in candidates:
        xy, dz = _tcp_fruit_gap(env, key, arm_name)
        if xy > PINCH_XY or dz < -0.02 or dz > PINCH_Z_MAX:
            continue
        score = xy + 0.5 * max(0.0, dz)
        if score < best_score:
            best_score = score
            best = key
    return best


def _near_fruit(env, key, arm_name: str) -> bool:
    xy, dz = _tcp_fruit_gap(env, key, arm_name)
    return xy <= PINCH_XY and -0.02 <= dz <= PINCH_Z_MAX


def _held_by_gripper(env, key) -> bool:
    kind, i = key
    if kind == "dist":
        return bool(env._distractor_held_by_gripper(i))
    return bool(env._fruit_held_by_gripper(i))


def _on_belt(env, key) -> bool:
    kind, i = key
    if kind == "dist":
        return env._distractor_y[i] is not None
    return env._item_y[i] is not None


def _free_for_grasp(env, key) -> None:
    kind, i = key
    if kind == "dist":
        env._free_distractor_for_physical_grasp(i)
        env._grasping_dist_slots.add(i)
        env._enable_distractor_gravity(i)
        return
    env._free_fruit_for_physical_grasp(i)
    env._grasping_idxs.add(i)
    env._enable_fruit_gravity(i)


def _reseat_key(env, key) -> None:
    kind, i = key
    if kind == "dist":
        env._reseat_distractor_on_belt(i)
        return
    env._reseat_on_belt(i)


def _discard_grasping(env, key) -> None:
    kind, i = key
    if kind == "dist":
        env._grasping_dist_slots.discard(i)
        return
    env._grasping_idxs.discard(i)


def _calm_key(env, key, damping=(2.5, 12.0)) -> None:
    kind, i = key
    if kind == "dist":
        env._calm_distractor(i, damping=damping)
        return
    env._calm_fruit(i, damping=damping)


def _enable_gravity_key(env, key) -> None:
    kind, i = key
    if kind == "dist":
        env._enable_distractor_gravity(i)
        return
    env._enable_fruit_gravity(i)


def _label_key(env, key) -> str:
    kind, i = key
    if kind == "dist":
        return f"black_{i}"
    return f"{env.item_types[i]}_{i}"


def _lighter_rgb(rgb):
    return [min(1.0, 0.45 * float(c) + 0.55) for c in list(rgb)[:3]]


class FruitClickController:
    """Click fruit (highlight) then click a basket to drop it there.

    Colored apples and Opt2 black distractors share this path. Black fruit
    still does not count toward ``check_success``.
    """

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self.selected = None  # ("item", idx) or ("dist", slot)
        self._dropping = {}  # key -> steps remaining
        self._basket_ids = {}
        for ftype, basket in (getattr(env, "baskets", {}) or {}).items():
            sid = actor_scene_id(basket)
            if sid is not None:
                self._basket_ids[int(sid)] = str(ftype)

    def _actor(self, key):
        kind, i = key
        if kind == "dist":
            return self.env.distractors[i]
        return self.env.items[i]

    def _rigid(self, key):
        kind, i = key
        if kind == "dist":
            comps = getattr(self.env, "_distractor_comps", [])
            return comps[i] if i < len(comps) else None
        comps = getattr(self.env, "_item_comps", [])
        return comps[i] if i < len(comps) else None

    def _label(self, key) -> str:
        kind, i = key
        if kind == "dist":
            return f"black_{i}"
        return f"{self.env.item_types[i]}_{i}"

    def _fruit_ids(self):
        ids = {}
        for idx in range(self.env.n_items):
            if (
                not self.env._spawned_mask[idx]
                or self.env._packed[idx]
                or self.env._missed[idx]
                or ("item", idx) in self._dropping
            ):
                continue
            sid = actor_scene_id(self.env.items[idx])
            if sid is not None:
                ids[int(sid)] = ("item", int(idx))
        for slot in range(int(getattr(self.env, "n_distractor_slots", 0) or 0)):
            if ("dist", slot) in self._dropping:
                continue
            # Hidden / not yet spawned: parked at HIDE_Z with no belt y.
            if self.env._distractor_y[slot] is None:
                p = np.array(self.env.distractors[slot].get_pose().p, dtype=float)
                if float(p[2]) < 0.0:
                    continue
            sid = actor_scene_id(self.env.distractors[slot])
            if sid is not None:
                ids[int(sid)] = ("dist", int(slot))
        return ids

    def _restore_color(self, key):
        kind, i = key
        if kind == "dist":
            self.env._recolor(self.env.distractors[i], self.env.distractor_color)
            return
        ftype = self.env.item_types[i]
        self.env._recolor(self.env.items[i], self.env._type_rgb(ftype))

    def _highlight(self, key):
        kind, i = key
        if kind == "dist":
            self.env._recolor(
                self.env.distractors[i], _lighter_rgb(self.env.distractor_color)
            )
            return
        ftype = self.env.item_types[i]
        self.env._recolor(self.env.items[i], _lighter_rgb(self.env._type_rgb(ftype)))

    def _select(self, key):
        if self.selected is not None and self.selected != key:
            kind, i = self.selected
            busy = False
            if kind == "item":
                busy = self.env._packed[i] or self.env._missed[i]
            if not busy and self.selected not in self._dropping:
                self._restore_color(self.selected)
        self.selected = key
        self._highlight(key)
        print(f"Selected {self._label(key)} — click a basket to pack it.")

    def _drop_xy(self, basket_type: str, pixel_x, pixel_y):
        """World XY over the basket mouth (click if inside, else center)."""
        env = self.env
        c = np.array(env.basket_centers[basket_type], dtype=float)
        half_x, half_y = env.basket_half_xy.get(basket_type, env.BASKET_HALF_XY)
        r = float(env.fruit_r)
        hit = table_xy_from_click(
            self.viewer, pixel_x, pixel_y,
            float(env.basket_top_z[basket_type]),
        )
        if hit is not None:
            x, y = float(hit[0]), float(hit[1])
        else:
            x, y = float(c[0]), float(c[1])
        x = float(np.clip(x, c[0] - half_x + r, c[0] + half_x - r))
        y = float(np.clip(y, c[1] - half_y + r, c[1] + half_y - r))
        return x, y

    def _enable_drop_gravity(self, key):
        rigid = self._rigid(key)
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_linear_damping(4.0)
            rigid.set_angular_damping(16.0)
        except Exception:
            pass

    def _calm(self, key, damping=(4.0, 16.0)):
        rigid = self._rigid(key)
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_linear_damping(float(damping[0]))
            rigid.set_angular_damping(float(damping[1]))
        except Exception:
            pass

    def _pack_into(self, basket_type: str, pixel_x, pixel_y):
        key = self.selected
        if key is None:
            print("Select a fruit first.")
            return
        kind, i = key
        if kind == "item" and (self.env._packed[i] or self.env._missed[i]):
            self.selected = None
            return
        if key in self._dropping:
            self.selected = None
            return
        x, y = self._drop_xy(basket_type, pixel_x, pixel_y)
        z = float(self.env.basket_top_z[basket_type]) + float(self.env.fruit_r) + 0.04
        if kind == "item":
            self.env._item_y[i] = None
            self.env._grasping_idxs.add(i)
            self.env._set_fruit_collision_enabled(i, True)
            self.env._set_fruit_pose(i, x, y, z)
        else:
            # Leave the belt stream so _advance_distractors does not yank it back.
            self.env._distractor_y[i] = None
            self.env._grasping_dist_slots.add(i)
            self.env._set_distractor_pose(i, x, y, z)
        self._enable_drop_gravity(key)
        self._restore_color(key)
        self._dropping[key] = int(DROP_SETTLE_STEPS)
        extra = ""
        if kind == "item" and basket_type != self.env.item_types[i]:
            extra = " (wrong color)."
        elif kind == "dist":
            extra = " (distractor, not counted)."
        print(f"Dropping {self._label(key)} into the {basket_type} basket…{extra}")
        self.selected = None

    def _in_any_basket(self, key) -> bool:
        p = np.array(self._actor(key).get_pose().p, dtype=float)
        return any(
            self.env._xy_inside_basket(p[:2], btype) for btype in self.env.baskets
        )

    def _finish_drop(self, key):
        kind, i = key
        if kind == "item":
            self.env._grasping_idxs.discard(i)
            if self.env._fruit_in_basket(i):
                self.env._mark_packed(i, freeze=False)
                print(f"Packed {self._label(key)} into the matching basket.")
                return
            if self._in_any_basket(key):
                self.env._mark_packed(i, freeze=False)
                print(f"{self._label(key)} landed in the wrong basket.")
                return
            if self.env._fruit_over_belt(i) is not None:
                self.env._reseat_on_belt(i)
                print(f"{self._label(key)} missed — back on the belt.")
                return
            self.env._mark_table_rest(i)
            print(f"{self._label(key)} left on the table.")
            return

        self.env._grasping_dist_slots.discard(i)
        if self._in_any_basket(key):
            print(f"{self._label(key)} landed in a basket — that fails the task.")
            return
        if self.env._distractor_over_belt(i) is not None:
            self.env._reseat_distractor_on_belt(i)
            print(f"{self._label(key)} missed — back on the belt.")
            return
        self.env._mark_distractor_table_rest(i)
        print(f"{self._label(key)} left on the table.")

    def update(self) -> None:
        for key in list(self._dropping):
            left = self._dropping[key] - 1
            if left > DROP_SETTLE_STEPS - 12:
                self._calm(key, damping=(4.0, 16.0))
            if left > 0:
                self._dropping[key] = left
                continue
            del self._dropping[key]
            self._finish_drop(key)

    def on_click(self, viewer, pixel_x, pixel_y):
        fruit_hit = click_hits_actor_map(
            viewer, pixel_x, pixel_y, self._fruit_ids()
        )
        if fruit_hit is not None:
            self._select(fruit_hit)
            return True
        basket_hit = click_hits_actor_map(
            viewer, pixel_x, pixel_y, self._basket_ids
        )
        if basket_hit is not None:
            self._pack_into(str(basket_hit), pixel_x, pixel_y)
            return True
        return False


class FruitPinchMonitor:
    """Detach a belt apple only when a real pinch is confirmed."""

    def __init__(self, env):
        self.env = env
        self._prev_width = {"left": 1.0, "right": 1.0}
        self._held = {}  # arm_name -> ("item", idx) | ("dist", slot)
        self._pending = {}
        self._settle = None
        self._announced = set()

    def _try_pinch(self, arm_name: str, step: int) -> None:
        if arm_name in self._held or arm_name in self._pending:
            return
        key = _nearest_graspable(self.env, arm_name)
        if key is None:
            return
        self._pending[arm_name] = (key, step + GRASP_SETTLE_STEPS)

    def _confirm_or_abort_pinch(self, arm_name: str, key) -> None:
        held = _held_by_gripper(self.env, key)
        near = _near_fruit(self.env, key, arm_name)
        closed = gripper_width(self.env, arm_name) <= CLOSE_WIDTH
        on_belt = _on_belt(self.env, key)

        if closed and (held or near):
            _free_for_grasp(self.env, key)
            self._held[arm_name] = key
            if key not in self._announced:
                self._announced.add(key)
                kind, i = key
                if kind == "dist":
                    print(
                        f"Pinched black_{i} with {arm_name}. "
                        f"Lift with E; Space to release. "
                        f"Packing a black apple fails the task."
                    )
                else:
                    ftype = self.env.item_types[i]
                    print(
                        f"Pinched {ftype}_{i} with {arm_name}. "
                        f"Lift with E over the "
                        f"{'red' if ftype == 'apple' else 'green'} basket, "
                        f"Space to release."
                    )
            return

        if on_belt:
            return
        _reseat_key(self.env, key)

    def _begin_release(self, arm_name: str) -> None:
        key = self._held.pop(arm_name, None)
        if key is None:
            return
        self._pending.pop(arm_name, None)
        _calm_key(self.env, key, damping=(2.5, 12.0))
        _enable_gravity_key(self.env, key)
        self._settle = (key, arm_name, DROP_SETTLE_STEPS)
        print(f"Released {_label_key(self.env, key)} — dropping under gravity…")

    def _tick_settle(self) -> None:
        if self._settle is None:
            return
        key, arm_name, left = self._settle
        if left > DROP_SETTLE_STEPS - 12:
            _calm_key(self.env, key, damping=(3.0, 14.0))
        left -= 1
        if left > 0:
            self._settle = (key, arm_name, left)
            return
        self._settle = None
        _discard_grasping(self.env, key)

        if _held_by_gripper(self.env, key) or _near_fruit(self.env, key, arm_name):
            if gripper_width(self.env, arm_name) <= CLOSE_WIDTH:
                self._held[arm_name] = key
                kind, i = key
                if kind == "dist":
                    self.env._grasping_dist_slots.add(i)
                else:
                    self.env._grasping_idxs.add(i)
                print(
                    f"{_label_key(self.env, key)} still in the jaws — "
                    f"open Space fully to drop."
                )
                return

        kind, i = key
        if kind == "dist":
            if self.env._distractor_in_any_basket(i):
                print(f"black_{i} landed in a basket — that fails the task.")
                return
            if self.env._distractor_over_belt(i) is not None:
                self.env._reseat_distractor_on_belt(i)
                print(f"black_{i} back on the belt.")
                return
            self.env._mark_distractor_table_rest(i)
            print(f"black_{i} left on the table.")
            return

        if self.env._fruit_in_basket(i):
            self.env._mark_packed(i, freeze=False)
            print(f"Packed {self.env.item_types[i]}_{i} into the matching basket.")
            return

        p = np.array(self.env.items[i].get_pose().p, dtype=float)
        in_any = any(
            self.env._xy_inside_basket(p[:2], btype) for btype in self.env.baskets
        )
        if in_any:
            self.env._mark_packed(i, freeze=False)
            print(f"{self.env.item_types[i]}_{i} landed in the wrong basket.")
            return

        if self.env._fruit_over_belt(i) is not None:
            self.env._reseat_on_belt(i)
            print(f"{self.env.item_types[i]}_{i} back on the belt — keep packing.")
            return

        self.env._mark_table_rest(i)
        print(f"{self.env.item_types[i]}_{i} left on the table.")

    def update(self, step: int) -> None:
        env = self.env
        self._tick_settle()

        for arm, (key, at) in list(self._pending.items()):
            if step < at:
                continue
            del self._pending[arm]
            if arm in self._held:
                continue
            self._confirm_or_abort_pinch(arm, key)

        for arm in ("left", "right"):
            w = gripper_width(env, arm)
            prev = self._prev_width[arm]
            closing = prev > CLOSE_WIDTH and w <= CLOSE_WIDTH
            opening = prev <= CLOSE_WIDTH and w > OPEN_WIDTH
            self._prev_width[arm] = w

            if self._settle is not None:
                continue

            if closing:
                self._try_pinch(arm, step)
            elif opening and arm in self._held:
                self._begin_release(arm)
            elif opening:
                self._pending.pop(arm, None)


def main():
    parser = argparse.ArgumentParser(description="Interactive pack_fruits viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=("default", "opt1", "opt2", "opt1+2"),
        default=None,
        help="Apply GUI scenario flags (opt1 = red+green with both baskets)",
    )
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.pack_fruits import pack_fruits

    scenario_overrides = {
        "default": {"two_colors_enabled": False, "distractor_enabled": False},
        "opt1": {"two_colors_enabled": True, "distractor_enabled": False},
        "opt2": {"two_colors_enabled": False, "distractor_enabled": True},
        "opt1+2": {"two_colors_enabled": True, "distractor_enabled": True},
    }
    scenario = args.scenario or os.environ.get("ROBODYNA_SCENARIO") or None
    overrides = scenario_overrides.get(scenario) if scenario else None

    use_robot = is_robot_control(args.control)
    config = configure_task(
        "pack_fruits", args.config, args.seed, use_robot=use_robot,
        task_arg_overrides=overrides,
    )
    scenario = scenario or config.get("interactive_scenario")
    if scenario in scenario_overrides:
        config.setdefault("task_args", {}).setdefault("pack_fruits", {}).update(
            scenario_overrides[scenario]
        )
        config["interactive_scenario"] = scenario
        config["scenario"] = scenario
    print(
        f"[interactive_pack_fruits] scenario={scenario!r} "
        f"two_colors={config.get('task_args', {}).get('pack_fruits', {}).get('two_colors_enabled')}",
        flush=True,
    )
    env = pack_fruits()
    env.setup_demo(**config)
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    env._belt_running = True

    if use_robot:
        for side in ("left", "right"):
            try:
                env.robot.set_gripper(1.0, side, gripper_eps=0.0)
            except Exception:
                pass

    if env.two_colors_enabled:
        goal = (
            f"Opt1-style: {env.n_apple} red → left basket, "
            f"{env.n_green} green → right basket (one apple at a time)."
        )
    else:
        color = "red" if env.active_colors[0] == "apple" else "green"
        side = "left" if color == "red" else "right"
        goal = f"Default/Opt2: {env.n_items} {color} apples → {side} basket."

    if use_robot:
        print_banner(
            "pack_fruits — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
                f"config: {args.config}  |  seed: {args.seed}"
                + (f"  |  scenario: {args.scenario}" if args.scenario else ""),
                goal,
                "Black apples pinch like the others; packing one fails the task (Opt2).",
                "1 / 2 / 3 — select left / right / both arms (selected gripper turns green)",
                "Arrows / E / Q — teleop the selected arm(s)",
                "Space — close to pinch; apple keeps moving until a real grasp",
                "V — cycle view: head_camera ↔ gripper(s)",
                "Esc — close the viewer window to quit",
                "--scenario default|opt1|opt2|opt1+2",
                "--robot-motion planner|interpolate",
            ],
        )
    else:
        print_banner(
            "pack_fruits — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}"
                + (f"  |  scenario: {args.scenario}" if args.scenario else ""),
                goal,
                "Click a fruit (including black) to select, then click a basket to drop it in.",
                "Black apples can be dropped; leaving one in a basket fails the task.",
                "Esc — quit",
            ],
        )

    settle_after_done = None
    pinch = FruitPinchMonitor(env) if use_robot else None
    clicker = None
    if not use_robot:
        viewer = env.viewer
        if viewer is None:
            raise SystemExit("Viewer was not created; ensure a graphical display is available.")
        clicker = FruitClickController(env, viewer)
        viewer.register_click_handler(clicker.on_click)
        print_instructions("Click a fruit, then click a basket to drop it in.")

    def on_step(window, step):
        nonlocal settle_after_done
        del window
        env._belt_running = True

        if pinch is not None:
            pinch.update(step)
        if clicker is not None:
            clicker.update()

        if settle_after_done is None:
            all_done = (
                env._spawned >= env.n_items
                and all(env._packed[i] or env._missed[i] for i in range(env.n_items))
            )
            if all_done:
                settle_after_done = step
                print("All fruits resolved — settling…")

    def is_done(step):
        if settle_after_done is not None and step - settle_after_done > 120:
            n_ok = sum(1 for i in range(env.n_items) if env._fruit_in_basket(i))
            return True, f"{n_ok}/{env.n_items} in correct baskets"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=40000)


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
