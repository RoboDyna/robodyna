"""Coach for the keyboard+mouse tutorial parts."""
from __future__ import annotations

import time

from _interactive_common import (
    actor_scene_id,
    click_hits_actor_map,
    edge_pressed,
    print_instructions,
    run_viewer_loop,
    table_xy_from_click,
)
from _key_hud import TutorialKeyHud, build_keyboard_tutorial_hud

BUTTON_STAGES = (
    ("num_keys", "keys", "1-1: Turn on the lamps. Press 1, 2, and 3."),
    ("num_mouse", "mouse", "1-2: Turn off the lamps. Click the buttons with the mouse."),
    ("bowl_keys", "keys", "2-1: Move the bowl over both red circles. Use the arrow keys."),
    ("bowl_mouse", "mouse", "2-2: Move the bowl over both red circles. Click the arrow buttons."),
    ("switch_keys", "keys", "3-1: Turn the key on, then off. Press Space once for on, again for off."),
    ("switch_mouse", "mouse", "3-2: Turn the key on, then off. Click the button with the mouse."),
    ("push_keys", "keys", "4-1: Press and hold the key to turn on the lamp. Hold Space."),
    ("push_mouse", "mouse", "4-2: Press and hold the key to turn on the lamp. Hold the left mouse button."),
)

STAGE_SETTLE_S = {
    "switch_keys": 2.0,
    "switch_mouse": 2.0,
    "push_keys": 2.0,
    "push_mouse": 2.0,
}

PLACEMENT_STAGES = (
    (
        "cup_place",
        "mouse",
        "Place the cup on the green area. No part of the cup should be outside. Click the green region.",
    ),
    (
        "apple_box",
        "mouse",
        "Move the apple inside the box. Click the apple, then click the box.",
    ),
)

BASE_STAGES = (
    (
        "gummy_keys",
        "keys",
        "Move the bowl onto the red circle with the arrows, then press Space.",
    ),
    (
        "gummy_mouse",
        "mouse",
        "Move the bowl onto the red circle with the arrow buttons, then click the trigger.",
    ),
    (
        "dual_gate",
        "keys",
        "Open the gate by pressing the left and right arrow keys at the same time.",
    ),
    (
        "cue_aim",
        "both",
        "Click the green circle to place the stick tip, then rotate with the arrow keys.",
    ),
)

HOUSEHOLD_STAGES = (
    ("stop_ball", "mouse", "Click the moving ball to stop it."),
    ("knob_lamp", "mouse", "Click the stove knob to turn the lamp on, then off."),
    (
        "board_tilt",
        "both",
        "Click the green zone to hover the board, then tilt with the arrow keys.",
    ),
)

PART_STAGES = {
    "buttons": BUTTON_STAGES,
    "placement": PLACEMENT_STAGES,
    "base": BASE_STAGES,
    "household": HOUSEHOLD_STAGES,
}


def _mouse_picture_xy(viewer):
    window = viewer.window
    mx, my = window.mouse_position
    ww, wh = window.size
    if ww <= 0 or wh <= 0 or mx < 0 or my < 0 or mx >= ww or my >= wh:
        return None
    tw, th = window.get_picture_size("Segmentation")
    return int(mx * tw / ww), int(my * th / wh)


class KeyboardTutorialCoach:
    def __init__(self, env, hud: TutorialKeyHud, part: str):
        self.env = env
        self.hud = hud
        self.part = part
        self.stages = PART_STAGES[part]
        self.stage_index = 0
        self._prev = {}
        self._key_ids = {}
        self._circle_ids = {}
        self._last_mouse_hit = None
        self._ready_at = None
        self._need_release = False
        self._press_t = None
        self.done = False

    def start(self) -> None:
        self._enter(0)

    def _enter(self, index: int) -> None:
        self.stage_index = index
        name, mode, prompt = self.stages[index]
        self.env.tutorial_set_stage(name, input_mode=mode)
        self.hud.set_stage(name)
        self._rebuild_ids()
        self._last_mouse_hit = None
        self._ready_at = None
        self._need_release = True
        self._press_t = None
        print_instructions(prompt)

    def _rebuild_ids(self) -> None:
        self._key_ids = {}
        for key, actor in (self.env.key_actors() or {}).items():
            sid = actor_scene_id(actor)
            if sid is not None:
                self._key_ids[int(sid)] = key
        self._circle_ids = {}
        for i, actor in enumerate(self.env.place_circle_actors() or []):
            sid = actor_scene_id(actor)
            if sid is not None:
                self._circle_ids[int(sid)] = i

    def _advance(self) -> None:
        nxt = self.stage_index + 1
        if nxt >= len(self.stages):
            self.done = True
            self.env._tutorial_complete = True
            print_instructions("Tutorial part complete. Esc closes the viewer.")
            return
        self._enter(nxt)

    def _mode(self) -> str:
        return self.stages[self.stage_index][1]

    def _stage_name(self) -> str:
        return self.stages[self.stage_index][0]

    def _held_hit(self, viewer):
        window = viewer.window
        if not bool(window.mouse_down(0)):
            return None
        pix = _mouse_picture_xy(viewer)
        if pix is None:
            return None
        return click_hits_actor_map(viewer, pix[0], pix[1], self._key_ids)

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.done or self._mode() == "keys":
            return False
        name = self._stage_name()
        hit = click_hits_actor_map(viewer, pixel_x, pixel_y, self._key_ids)
        if name == "cup_place":
            xy = table_xy_from_click(viewer, pixel_x, pixel_y, float(self.env.TABLE_Z))
            if xy is None:
                return False
            pad = self.env.PAD_XY
            if (hit != "pad") and (
                ((xy[0] - pad[0]) ** 2 + (xy[1] - pad[1]) ** 2) ** 0.5
                > float(self.env.PAD_R) + 0.02
            ):
                return False
            self.env.place_cup_at(float(xy[0]), float(xy[1]))
            self.hud.flash("mouse")
            return True
        if name == "apple_box":
            if hit == "apple":
                self.env.select_apple()
                self.hud.flash("mouse")
                return True
            if hit == "box":
                self.env.drop_apple_in_box()
                self.hud.flash("mouse")
                return True
            return False
        if name == "gummy_mouse":
            if hit == "trigger":
                self.env.trigger_dispense()
                self.hud.flash("mouse")
                return True
            return False
        if name == "cue_aim":
            if hit != "circle":
                xy = table_xy_from_click(viewer, pixel_x, pixel_y, float(self.env.TABLE_Z))
                if xy is None:
                    return False
                cx, cy = self.env.AIM_CIRCLE_XY
                if ((xy[0] - cx) ** 2 + (xy[1] - cy) ** 2) ** 0.5 > 0.06:
                    return False
            self.env.place_stick_on_circle()
            self.hud.flash("mouse")
            return True
        if name == "stop_ball":
            if hit != "ball":
                return False
            self.env.click_stop_ball()
            self.hud.flash("mouse")
            return True
        if name == "knob_lamp":
            if hit != "knob":
                return False
            self.env.trigger_knob()
            self.hud.flash("mouse")
            return True
        if name == "board_tilt":
            if hit != "circle":
                xy = table_xy_from_click(viewer, pixel_x, pixel_y, float(self.env.TABLE_Z))
                if xy is None:
                    return False
                cx, cy = self.env.BOARD_CIRCLE_XY
                if ((xy[0] - cx) ** 2 + (xy[1] - cy) ** 2) ** 0.5 > 0.06:
                    return False
            self.env.place_board_on_circle()
            self.hud.flash("mouse")
            return True
        if name in ("place_click", "place_drop"):
            idx = click_hits_actor_map(viewer, pixel_x, pixel_y, self._circle_ids)
            if idx is None:
                return False
            self.env.place_on_circle(int(idx), drop=name == "place_drop")
            self.hud.flash("mouse")
            return True
        if name == "push_mouse":
            return False
        if hit is None:
            return False
        self._apply_hit(hit)
        self.hud.flash("mouse")
        return True

    def _apply_hit(self, hit) -> None:
        name = self._stage_name()
        if name in ("num_keys", "num_mouse") and hit in (0, 1, 2):
            self.env.trigger_num(int(hit))
        elif name in ("switch_keys", "switch_mouse"):
            self.env.trigger_switch()

    def update_keys(self, window) -> None:
        if self.done or window is None:
            return
        name = self._stage_name()
        mode = self._mode()
        held: set[str] = set()
        if mode == "keys" or mode == "both":
            if name == "num_keys":
                for digit, idx in (("1", 0), ("2", 1), ("3", 2)):
                    down = bool(window.key_down(digit))
                    if down:
                        held.add(digit)
                    if down and not self._prev.get(digit, False):
                        self.env.trigger_num(idx)
                        self.hud.flash(digit)
                    self._prev[digit] = down
            elif name in ("bowl_keys", "gummy_keys"):
                left = bool(window.key_down("left"))
                right = bool(window.key_down("right"))
                side = None
                if left and not right:
                    side = "left"
                    held.add("left")
                elif right and not left:
                    side = "right"
                    held.add("right")
                self.env.set_bowl_hold(side)
                if edge_pressed(window, "left", self._prev):
                    self.hud.flash("left")
                if edge_pressed(window, "right", self._prev):
                    self.hud.flash("right")
                if name == "gummy_keys":
                    if window.key_down("space"):
                        held.add("Space")
                    if edge_pressed(window, "space", self._prev):
                        self.env.trigger_dispense()
                        self.hud.flash("Space")
            elif name == "dual_gate":
                left = bool(window.key_down("left"))
                right = bool(window.key_down("right"))
                if left:
                    held.add("left")
                if right:
                    held.add("right")
                self.env.set_gate_hold(left, right)
                if edge_pressed(window, "left", self._prev):
                    self.hud.flash("left")
                if edge_pressed(window, "right", self._prev):
                    self.hud.flash("right")
            elif name in ("cue_aim", "board_tilt"):
                left = bool(window.key_down("left"))
                right = bool(window.key_down("right"))
                direction = 0.0
                if left and not right:
                    direction = 1.0
                    held.add("left")
                elif right and not left:
                    direction = -1.0
                    held.add("right")
                self.env.rotate_aim(direction)
                if edge_pressed(window, "left", self._prev):
                    self.hud.flash("left")
                if edge_pressed(window, "right", self._prev):
                    self.hud.flash("right")
            elif name == "switch_keys":
                down = bool(window.key_down("space"))
                if down:
                    held.add("Space")
                if self._need_release:
                    if not down:
                        self._need_release = False
                elif down:
                    self.env.trigger_switch()
                    self.hud.flash("Space")
                    self._need_release = True
            elif name == "push_keys":
                down = bool(window.key_down("space"))
                self.env.set_push_hold(down)
                if down:
                    held.add("Space")
                    if not self._prev.get("space"):
                        self.hud.flash("Space")
                self._prev["space"] = down
        if mode == "mouse":
            if name in ("bowl_mouse", "gummy_mouse"):
                viewer = getattr(self.env, "viewer", None)
                hit = self._held_hit(viewer) if viewer is not None else None
                side = hit if hit in ("left", "right") else None
                self.env.set_bowl_hold(side)
                if side:
                    held.add(side)
            elif name == "push_mouse":
                viewer = getattr(self.env, "viewer", None)
                hit = self._held_hit(viewer) if viewer is not None else None
                holding = hit == "push"
                self.env.set_push_hold(holding)
                if holding:
                    held.add("mouse")
                    if not self._prev.get("push_mouse"):
                        self.hud.flash("mouse")
                self._prev["push_mouse"] = holding
        self.hud.set_held(held)

    def update_stage(self, window, _step: int) -> None:
        if self.done:
            return
        self.update_keys(window)
        name = self._stage_name()
        now = time.perf_counter()
        if name in ("push_keys", "push_mouse"):
            if bool(getattr(self.env, "_push_on", False)) and self._press_t is None:
                self._press_t = now
            if not self.env.tutorial_stage_complete():
                return
            start = self._press_t if self._press_t is not None else now
            if now - start >= float(STAGE_SETTLE_S.get(name, 2.0)):
                self._advance()
            return
        if not self.env.tutorial_stage_complete():
            self._ready_at = None
            return
        delay = float(STAGE_SETTLE_S.get(name, 0.0))
        if self._ready_at is None:
            self._ready_at = now
        if now - self._ready_at >= delay:
            self._advance()

    def is_done(self, _step: int):
        if self.done:
            return True, self.part
        return False


def run_keyboard_part(env, part: str) -> int:
    env._tutorial_complete = False
    stages = tuple(name for name, _mode, _prompt in PART_STAGES[part])
    hud = build_keyboard_tutorial_hud(env.scene, stages)
    coach = KeyboardTutorialCoach(env, hud, part)
    viewer = env.viewer
    if viewer is not None:
        viewer.register_click_handler(coach.on_click)
    coach.start()
    hud.on_frame = coach.update_keys
    run_viewer_loop(
        env,
        on_step=lambda window, step: coach.update_stage(window, step),
        is_done=coach.is_done,
        extra_plugins=[hud],
        report_result=False,
    )
    return 0
