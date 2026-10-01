"""Keyboard+mouse tutorial table.

Interactive only. Arms are stripped by the viewer. Keyboard substages ignore
the mouse; mouse substages ignore the keyboard. Aim/tilt stages use both.
"""
from __future__ import annotations

import sapien
import sapien.physx
import sapien.render
import numpy as np

from ._base_task import Base_Task
from .utils.create_actor import create_actor, create_box, create_sphere, create_visual_box
from .utils.reactive_button import ReactivePushButtons, add_key_base_border
from .utils.key_symbol import attach_key_symbol, sync_key_symbol


class tutorial_keyboard(Base_Task):
    TABLE_Z = 0.74
    KEY_HALF = (0.022, 0.022, 0.014)
    KEY_Y = -0.12
    NUM_XS = (-0.18, 0.0, 0.18)
    NUM_COLORS = (
        (0.86, 0.16, 0.16),
        (0.92, 0.78, 0.12),
        (0.18, 0.42, 0.88),
    )
    LAMP_OFF = (0.18, 0.18, 0.20)
    LAMP_RADIUS = 0.018
    LAMP_DY = 0.11
    LAMP_Z = 0.11

    ARROW_XS = (-0.20, 0.20)
    ARROW_KEY_Y = -0.14
    CIRCLE_Y = 0.10
    CIRCLE_RADIUS = 0.045
    BOWL_Y = 0.10
    BOWL_SPEED = 0.28
    BOWL_Q = (0.5, 0.5, 0.5, 0.5)

    SINGLE_KEY_XY = (0.0, -0.10)
    SWITCH_COLOR = (0.22, 0.24, 0.28)
    PUSH_COLOR = (0.28, 0.55, 0.82)

    PLACE_XS = (-0.16, 0.16)
    PLACE_Y = 0.06
    PLACE_START = (0.0, -0.10)

    HIDE_XY = (1.85, 1.85)
    PULSE_STEPS = 28
    HIT_TOL = 0.055

    CUP_Q = (0.707, 0.707, 0.0, 0.0)
    CUP_XY = (-0.22, 0.04)
    PAD_XY = (0.20, 0.06)
    PAD_R = 0.078
    CUP_R = 0.038

    APPLE_Q = (0.707, 0.707, 0.0, 0.0)
    APPLE_XY = (-0.22, 0.02)
    BOX_XY = (0.22, 0.02)
    APPLE_COLOR = (0.85, 0.12, 0.10)
    APPLE_R = 0.026
    BASKET_Q = (0.5, 0.5, 0.5, 0.5)
    TABLE_XY_LIM = 0.46

    GUMMY_ARROW_X = -0.26
    GUMMY_ARROW_YS = (-0.14, -0.06)
    TRIGGER_XY = (0.26, -0.10)
    GUMMY_BOWL_LO = -0.22
    GUMMY_BOWL_HI = 0.22
    GUMMY_MIN_SEP = 0.20

    GATE_KEY_XS = (-0.16, 0.16)
    GATE_KEY_Y = -0.13
    GATE_XY = (0.0, 0.08)
    GATE_BLADE_HALF_LEN = 0.075
    GATE_BLADE_HALF_W = 0.007
    GATE_BLADE_HALF_H = 0.028
    GATE_PLANK_COLOR = (0.92, 0.88, 0.78)
    GATE_OPEN_ANGLE = float(np.pi / 2.0)

    AIM_CIRCLE_XY = (0.10, 0.08)
    STICK_HOME_XY = (-0.22, -0.04)
    STICK_HALF = (0.12, 0.006, 0.006)

    BOARD_CIRCLE_XY = (0.08, 0.08)
    BOARD_HOME_XY = (-0.22, -0.04)
    BOARD_HALF = (0.11, 0.07, 0.008)

    BALL_Y = 0.06
    BALL_X_LIM = 0.20
    BALL_SPEED = 0.22
    BALL_R = 0.028

    KNOB_XY = (-0.10, -0.10)
    # Same size / angles as KitchenS cook_food: tick +Y is off, −90° (−X) is on.
    KNOB_RADIUS = 0.022
    KNOB_HEIGHT = 0.028
    KNOB_OFF_ANGLE = 0.0
    KNOB_ON_ANGLE = -0.5 * np.pi
    HH_LAMP_XY = (0.10, -0.02)

    def setup_demo(self, **kwags):
        self._kb_ready = False
        self._homes = {}
        self._tutorial_complete = False
        self._tutorial_stage = None
        self._input_mode = "keys"
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self._homes = {}
        z0 = float(self.TABLE_Z)
        hz = float(self.KEY_HALF[2])
        self._num_keys = []
        self._num_bezels = []
        self._num_homes = []
        self._num_lamps = []
        self._num_stems = []
        self._num_lamp_on = [False, False, False]
        self._num_hit = [False, False, False]
        for i, x in enumerate(self.NUM_XS):
            bezel = add_key_base_border(
                self, x, self.KEY_Y, z0, self.KEY_HALF, name_prefix=f"kb_num_base_{i}"
            )
            home = sapien.Pose([x, self.KEY_Y, z0 + hz], [1, 0, 0, 0])
            key = create_box(
                self,
                pose=home,
                half_size=list(self.KEY_HALF),
                color=list(self.NUM_COLORS[i]),
                name=f"kb_num_key_{i}",
                is_static=True,
            )
            lamp, stem = self._spawn_lamp(x, self.KEY_Y + self.LAMP_DY, f"kb_num_lamp_{i}")
            self._num_keys.append(key)
            self._num_bezels.append(bezel)
            self._num_homes.append(self._snapshot(key))
            self._num_lamps.append(lamp)
            self._num_stems.append(stem)
            self._remember(key, bezel, lamp, stem)

        self._arrow_keys = {}
        self._arrow_bezels = {}
        self._arrow_homes = {}
        self._circles = {}
        self._circle_on = {"left": False, "right": False}
        self._arrow_icon_parts = {}
        self._arrow_icon_locals = {}
        for side, x in (("left", self.ARROW_XS[0]), ("right", self.ARROW_XS[1])):
            bezel = add_key_base_border(
                self, x, self.ARROW_KEY_Y, z0, self.KEY_HALF, name_prefix=f"kb_arrow_base_{side}"
            )
            home = sapien.Pose([x, self.ARROW_KEY_Y, z0 + hz], [1, 0, 0, 0])
            key = create_box(
                self,
                pose=home,
                half_size=list(self.KEY_HALF),
                color=[0.20, 0.22, 0.26],
                name=f"kb_arrow_key_{side}",
                is_static=True,
            )
            self._replace_arrow_icon(key, side)
            disc = self._spawn_circle(x, self.CIRCLE_Y, f"kb_circle_{side}")
            self._arrow_keys[side] = key
            self._arrow_bezels[side] = bezel
            self._arrow_homes[side] = self._snapshot(key)
            self._circles[side] = disc
            self._remember(key, bezel, disc, self._arrow_icon_parts[side])

        bowl_pose = sapien.Pose(
            [0.0, self.BOWL_Y, z0 + 0.002],
            list(self.BOWL_Q),
        )
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=1,
            convex=True,
            is_static=False,
            scale_mult=0.85,
        )
        if self.bowl is not None:
            try:
                self.bowl.set_mass(0.06)
            except Exception:
                pass
            self._make_kinematic(self.bowl)
        self._bowl_x = 0.0
        self._bowl_hold = None

        sx, sy = self.SINGLE_KEY_XY
        self._switch_bezel = add_key_base_border(
            self, sx, sy, z0, self.KEY_HALF, name_prefix="kb_switch_base"
        )
        switch_home = sapien.Pose([sx, sy, z0 + hz], [1, 0, 0, 0])
        self._switch_key = create_box(
            self,
            pose=switch_home,
            half_size=list(self.KEY_HALF),
            color=list(self.SWITCH_COLOR),
            name="kb_switch_key",
            is_static=True,
        )
        self._switch_home = self._snapshot(self._switch_key)
        self._switch_symbol, self._switch_locals = attach_key_symbol(
            self, self._switch_key, self.KEY_HALF, "on_off", "kb_switch_icon"
        )
        self._switch_lamp, self._switch_stem = self._spawn_lamp(
            sx, sy + self.LAMP_DY, "kb_switch_lamp"
        )
        self._remember(
            self._switch_key,
            self._switch_bezel,
            self._switch_lamp,
            self._switch_stem,
            self._switch_symbol,
        )
        self._switch_on = False
        self._switch_seen_on = False
        self._switch_seen_off = False

        self._push_bezel = add_key_base_border(
            self, sx, sy, z0, self.KEY_HALF, name_prefix="kb_push_base"
        )
        push_home = sapien.Pose([sx, sy, z0 + hz], [1, 0, 0, 0])
        self._push_key = create_box(
            self,
            pose=push_home,
            half_size=list(self.KEY_HALF),
            color=list(self.PUSH_COLOR),
            name="kb_push_key",
            is_static=True,
        )
        self._push_home = self._snapshot(self._push_key)
        self._push_symbol, self._push_locals = attach_key_symbol(
            self, self._push_key, self.KEY_HALF, "push", "kb_push_icon"
        )
        self._push_lamp, self._push_stem = self._spawn_lamp(sx, sy + self.LAMP_DY, "kb_push_lamp")
        self._remember(
            self._push_key,
            self._push_bezel,
            self._push_lamp,
            self._push_stem,
            self._push_symbol,
        )
        self._push_on = False
        self._push_seen_on = False
        self._push_seen_off = False

        self._place_circles = []
        self._place_hit = [False, False]
        for i, x in enumerate(self.PLACE_XS):
            disc = self._spawn_circle(x, self.PLACE_Y, f"kb_place_circle_{i}")
            self._place_circles.append(disc)
            self._remember(disc)
        cube_hz = 0.022
        self._place_obj = create_box(
            self,
            pose=sapien.Pose(
                [self.PLACE_START[0], self.PLACE_START[1], z0 + cube_hz],
                [1, 0, 0, 0],
            ),
            half_size=[0.022, 0.022, cube_hz],
            color=[0.95, 0.55, 0.12],
            name="kb_place_cube",
            is_static=True,
        )
        self._place_hover_z = z0 + cube_hz + 0.04
        self._remember(self._place_obj, self.bowl)

        self._spawn_placement_props(z0)
        self._spawn_base_extra_props(z0, hz)
        self._spawn_household_props(z0)

        self._bank = None
        self._pulse = {}
        self._kb_ready = True
        self.tutorial_set_stage(None)

    def play_once(self):
        pass

    def check_success(self):
        return bool(getattr(self, "_tutorial_complete", False))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_kb_ready", False):
            return
        self._tick_pulses()
        if self._bank is not None:
            try:
                self._bank.update()
            except Exception:
                pass
        stage = self._tutorial_stage
        if stage in ("bowl_keys", "bowl_mouse"):
            self._tick_bowl()
            self._sync_arrow_icons()
        elif stage in ("gummy_keys", "gummy_mouse"):
            self._tick_bowl()
            self._sync_gummy_icons()
        elif stage == "dual_gate":
            self._tick_gate()
        elif stage == "stop_ball":
            self._tick_roll_ball()
        elif stage == "apple_box":
            self._tick_apple()
        elif stage in ("switch_keys", "switch_mouse"):
            sync_key_symbol(self._switch_symbol, self._switch_locals, self._switch_key)
        elif stage in ("push_keys", "push_mouse"):
            sync_key_symbol(self._push_symbol, self._push_locals, self._push_key)

    # ------------------------------------------------------------------ stage
    def tutorial_set_stage(self, stage: str | None, *, input_mode: str = "keys") -> None:
        self._tutorial_stage = stage
        self._input_mode = str(input_mode)
        self._bowl_hold = None
        self._pulse = {}
        self._bank = None
        self._hide_all()
        if stage in ("num_keys", "num_mouse"):
            want_on = stage == "num_mouse"
            for i in range(3):
                self._restore(self._num_keys[i], self._num_bezels[i], self._num_lamps[i], self._num_stems[i])
                self._set_num_lamp(i, want_on)
            self._num_hit = [False, False, False]
            self._bank = self._make_bank(self._num_keys, self._num_homes)
        elif stage in ("bowl_keys", "bowl_mouse"):
            self._circle_on = {"left": False, "right": False}
            for side in ("left", "right"):
                self._restore(
                    self._arrow_keys[side],
                    self._arrow_bezels[side],
                    self._circles[side],
                    self._arrow_icon_parts.get(side),
                )
                self._set_circle(self._circles[side], False)
            self._set_bowl_x(0.0)
            self._restore(self.bowl)
            self._bank = self._make_bank(
                [self._arrow_keys["left"], self._arrow_keys["right"]],
                [self._arrow_homes["left"], self._arrow_homes["right"]],
                ids=("left", "right"),
            )
        elif stage in ("switch_keys", "switch_mouse"):
            self._restore(
                self._switch_key,
                self._switch_bezel,
                self._switch_lamp,
                self._switch_stem,
                self._switch_symbol,
            )
            self._set_toggle_lamp("switch", False, count=False)
            self._switch_seen_on = False
            self._switch_seen_off = False
            self._bank = self._make_bank([self._switch_key], [self._switch_home])
            if self._bank is not None:
                try:
                    self._bank.set_forced(0, False)
                except Exception:
                    pass
        elif stage in ("push_keys", "push_mouse"):
            self._restore(
                self._push_key,
                self._push_bezel,
                self._push_lamp,
                self._push_stem,
                self._push_symbol,
            )
            self._set_toggle_lamp("push", False, count=False)
            self._push_seen_on = False
            self._push_seen_off = False
            self._bank = self._make_bank([self._push_key], [self._push_home])
            if self._bank is not None:
                try:
                    self._bank.set_forced(0, False)
                except Exception:
                    pass
        elif stage in ("place_click", "place_drop"):
            self._place_hit = [False, False]
            for disc in self._place_circles:
                self._restore(disc)
                self._set_circle(disc, False)
            z0 = float(self.TABLE_Z) + 0.022
            if stage == "place_drop":
                z0 = self._place_hover_z
            self._apply_pose(
                self._place_obj,
                sapien.Pose([self.PLACE_START[0], self.PLACE_START[1], z0], [1, 0, 0, 0]),
            )
        elif stage == "cup_place":
            self._restore(self._cup, self._green_pad)
            self._reset_cup()
        elif stage == "apple_box":
            self._restore(self._apple, self._fruit_box)
            self._reset_apple()
        elif stage in ("gummy_keys", "gummy_mouse"):
            self._restore(
                self._gummy_arrows["left"],
                self._gummy_arrows["right"],
                self._gummy_arrow_bezels["left"],
                self._gummy_arrow_bezels["right"],
                self._gummy_arrow_icons.get("left"),
                self._gummy_arrow_icons.get("right"),
                self._trigger_key,
                self._trigger_bezel,
                self._trigger_symbol,
                self.bowl,
                self._gummy_circle,
            )
            self._bank = self._make_bank(
                [
                    self._gummy_arrows["left"],
                    self._gummy_arrows["right"],
                    self._trigger_key,
                ],
                [
                    self._gummy_arrow_homes["left"],
                    self._gummy_arrow_homes["right"],
                    self._trigger_home,
                ],
                ids=("left", "right", "trigger"),
            )
            self._setup_gummy_circle()
        elif stage == "dual_gate":
            self._restore(
                self._gate_keys["left"],
                self._gate_keys["right"],
                self._gate_bezels["left"],
                self._gate_bezels["right"],
                self._gate_pieces["left"],
                self._gate_pieces["right"],
            )
            self._gate_open = 0.0
            self._gate_done = False
            self._gate_hold = (False, False)
            self._set_gate_open(0.0)
            self._bank = self._make_bank(
                [self._gate_keys["left"], self._gate_keys["right"]],
                [self._gate_homes["left"], self._gate_homes["right"]],
                ids=("left", "right"),
            )
        elif stage == "cue_aim":
            self._restore(self._stick, self._aim_circle)
            self._stick_placed = False
            self._stick_yaw = 0.0
            self._stick_seen_left = False
            self._stick_seen_right = False
            self._place_stick_home()
        elif stage == "stop_ball":
            self._restore(self._roll_ball)
            self._ball_stopped = False
            self._reset_roll_ball()
        elif stage == "knob_lamp":
            self._restore(self._knob, self._hh_lamp, self._hh_stem)
            self._set_knob_turned(False)
            self._set_toggle_lamp("hh", False, count=False)
            self._hh_seen_on = False
            self._hh_seen_off = False
        elif stage == "board_tilt":
            self._restore(self._board, self._board_circle)
            self._board_placed = False
            self._board_tilt = 0.0
            self._board_seen_left = False
            self._board_seen_right = False
            self._place_board_home()

    def tutorial_stage_complete(self) -> bool:
        stage = self._tutorial_stage
        if stage in ("num_keys", "num_mouse"):
            want = stage == "num_keys"
            hits = getattr(self, "_num_hit", None) or [False, False, False]
            return (
                len(self._num_lamp_on) == 3
                and len(hits) == 3
                and all(hits)
                and all(bool(v) == want for v in self._num_lamp_on)
            )
        if stage in ("bowl_keys", "bowl_mouse"):
            return bool(self._circle_on["left"] and self._circle_on["right"])
        if stage in ("switch_keys", "switch_mouse"):
            return bool(self._switch_seen_on and self._switch_seen_off)
        if stage in ("push_keys", "push_mouse"):
            return bool(self._push_seen_on and self._push_seen_off)
        if stage in ("place_click", "place_drop"):
            return all(self._place_hit)
        if stage == "cup_place":
            return self._cup_fully_in_pad()
        if stage == "apple_box":
            return self._apple_in_box()
        if stage in ("gummy_keys", "gummy_mouse"):
            return bool(self._gummy_circle_on)
        if stage == "dual_gate":
            return bool(self._gate_done)
        if stage == "cue_aim":
            return bool(
                self._stick_placed and self._stick_seen_left and self._stick_seen_right
            )
        if stage == "stop_ball":
            return bool(self._ball_stopped)
        if stage == "knob_lamp":
            return bool(self._hh_seen_on and self._hh_seen_off)
        if stage == "board_tilt":
            return bool(
                self._board_placed and self._board_seen_left and self._board_seen_right
            )
        return False

    # ----------------------------------------------------------------- input
    def trigger_num(self, idx: int) -> None:
        if idx < 0 or idx > 2:
            return
        stage = self._tutorial_stage
        if stage == "num_keys":
            self._set_num_lamp(idx, True)
        elif stage == "num_mouse":
            self._set_num_lamp(idx, False)
        else:
            return
        self._num_hit[idx] = True
        self._pulse_key(idx)

    def set_bowl_hold(self, side: str | None) -> None:
        if self._tutorial_stage not in ("bowl_keys", "bowl_mouse", "gummy_keys", "gummy_mouse"):
            self._bowl_hold = None
            return
        if side not in ("left", "right"):
            self._bowl_hold = None
            if self._bank is not None:
                for name in ("left", "right"):
                    try:
                        self._bank.set_forced(name, False)
                    except Exception:
                        pass
            return
        self._bowl_hold = side
        if self._bank is not None:
            try:
                self._bank.set_forced("left", side == "left")
                self._bank.set_forced("right", side == "right")
            except Exception:
                pass

    def trigger_dispense(self) -> None:
        if self._tutorial_stage not in ("gummy_keys", "gummy_mouse"):
            return
        tol = 0.08 if self._tutorial_stage in ("gummy_keys", "gummy_mouse") else self.HIT_TOL
        if abs(self._bowl_x - float(self._gummy_circle_x)) <= tol:
            self._gummy_circle_on = True
            self._set_circle(self._gummy_circle, True)
        self._pulse_key("trigger")

    def set_gate_hold(self, left: bool, right: bool) -> None:
        if self._tutorial_stage != "dual_gate":
            return
        self._gate_hold = (bool(left), bool(right))
        if self._bank is not None:
            try:
                self._bank.set_forced("left", bool(left))
                self._bank.set_forced("right", bool(right))
            except Exception:
                pass

    def rotate_aim(self, direction: float) -> None:
        if abs(direction) < 1e-6:
            return
        if self._tutorial_stage == "cue_aim" and self._stick_placed:
            self._stick_yaw += 0.045 * float(direction)
            if direction > 0:
                self._stick_seen_left = True
            else:
                self._stick_seen_right = True
            self._apply_stick_pose()
        elif self._tutorial_stage == "board_tilt" and self._board_placed:
            self._board_tilt = float(np.clip(self._board_tilt + 0.04 * float(direction), -1.1, 1.1))
            if direction > 0:
                self._board_seen_left = True
            else:
                self._board_seen_right = True
            self._apply_board_pose()

    def trigger_switch(self) -> None:
        if self._tutorial_stage not in ("switch_keys", "switch_mouse"):
            return
        on = not bool(self._switch_on)
        self._set_toggle_lamp("switch", on)
        if self._bank is not None:
            try:
                self._bank.set_forced(0, on)
            except Exception:
                pass

    def set_push_hold(self, held: bool) -> None:
        if self._tutorial_stage not in ("push_keys", "push_mouse"):
            return
        held = bool(held)
        if held == bool(self._push_on):
            if self._bank is not None:
                try:
                    self._bank.set_forced(0, held)
                except Exception:
                    pass
            return
        self._set_toggle_lamp("push", held)
        if self._bank is not None:
            try:
                self._bank.set_forced(0, held)
            except Exception:
                pass

    def place_on_circle(self, idx: int, *, drop: bool = False) -> None:
        if idx < 0 or idx > 1:
            return
        if self._tutorial_stage not in ("place_click", "place_drop"):
            return
        x = float(self.PLACE_XS[idx])
        y = float(self.PLACE_Y)
        z = float(self.TABLE_Z) + 0.022
        if drop:
            z = float(self.TABLE_Z) + 0.022 + 0.04
            self._apply_pose(self._place_obj, sapien.Pose([x, y, z], [1, 0, 0, 0]))
            z = float(self.TABLE_Z) + 0.022
        self._apply_pose(self._place_obj, sapien.Pose([x, y, z], [1, 0, 0, 0]))
        self._place_hit[idx] = True
        self._set_circle(self._place_circles[idx], True)

    def place_cup_at(self, x: float, y: float) -> None:
        if self._tutorial_stage != "cup_place" or self._cup is None:
            return
        z = float(self.TABLE_Z) + 0.002
        self._apply_pose(self._cup, sapien.Pose([float(x), float(y), z], list(self.CUP_Q)))

    def select_apple(self) -> None:
        if self._tutorial_stage != "apple_box" or self._apple is None:
            return
        self._apple_selected = True
        light = [min(1.0, 0.45 * c + 0.55) for c in self.APPLE_COLOR]
        self._tint(self._apple, light, emit=True)

    def drop_apple_in_box(self) -> None:
        if self._tutorial_stage != "apple_box" or not self._apple_selected:
            return
        if self._apple is None or self._fruit_box is None:
            return
        bx, by = self.BOX_XY
        z = float(self._box_top_z) + float(self.APPLE_R) + 0.04
        self._apply_pose(self._apple, sapien.Pose([bx, by, z], list(self.APPLE_Q)))
        self._set_dynamic(self._apple, kinematic=False, gravity=True)
        self._zero_vel(self._apple)
        self._tint(self._apple, self.APPLE_COLOR, emit=False)
        self._apple_selected = False
        self._apple_dropping = True

    def place_stick_on_circle(self) -> None:
        if self._tutorial_stage != "cue_aim":
            return
        self._stick_placed = True
        self._stick_yaw = 0.0
        self._apply_stick_pose()

    def place_board_on_circle(self) -> None:
        if self._tutorial_stage != "board_tilt":
            return
        self._board_placed = True
        self._board_tilt = 0.0
        self._apply_board_pose()

    def click_stop_ball(self) -> None:
        if self._tutorial_stage != "stop_ball":
            return
        self._ball_stopped = True
        self._make_kinematic(self._roll_ball)
        self._zero_vel(self._roll_ball)

    def trigger_knob(self) -> None:
        if self._tutorial_stage != "knob_lamp":
            return
        want_on = not bool(self._hh_on)
        self._set_knob_turned(want_on)
        self._set_toggle_lamp("hh", want_on)

    def key_actors(self):
        stage = self._tutorial_stage
        if stage in ("num_keys", "num_mouse"):
            return {i: self._num_keys[i] for i in range(3)}
        if stage in ("bowl_keys", "bowl_mouse"):
            return dict(self._arrow_keys)
        if stage in ("switch_keys", "switch_mouse"):
            return {"switch": self._switch_key}
        if stage in ("push_keys", "push_mouse"):
            return {"push": self._push_key}
        if stage in ("gummy_keys", "gummy_mouse"):
            return {
                "left": self._gummy_arrows["left"],
                "right": self._gummy_arrows["right"],
                "trigger": self._trigger_key,
            }
        if stage == "dual_gate":
            return dict(self._gate_keys)
        if stage == "apple_box":
            return {"apple": self._apple, "box": self._fruit_box}
        if stage == "cup_place":
            return {"pad": self._green_pad}
        if stage == "cue_aim":
            return {"circle": self._aim_circle}
        if stage == "stop_ball":
            return {"ball": self._roll_ball}
        if stage == "knob_lamp":
            return {"knob": self._knob}
        if stage == "board_tilt":
            return {"circle": self._board_circle}
        return {}

    def place_circle_actors(self):
        return list(self._place_circles)

    # ----------------------------------------------------------------- lamps
    def _spawn_placement_props(self, z0: float) -> None:
        pad_h = 0.0018
        self._green_pad = create_box(
            self,
            pose=sapien.Pose(
                [self.PAD_XY[0], self.PAD_XY[1], z0 + pad_h],
                [1, 0, 0, 0],
            ),
            half_size=[self.PAD_R, self.PAD_R, pad_h],
            color=[0.18, 0.82, 0.32],
            name="kb_green_pad",
            is_static=True,
        )
        self._cup = create_actor(
            self,
            pose=sapien.Pose(
                [self.CUP_XY[0], self.CUP_XY[1], z0 + 0.002],
                list(self.CUP_Q),
            ),
            modelname="021_cup",
            model_id=0,
            convex=True,
            is_static=False,
        )
        if self._cup is not None:
            self._make_kinematic(self._cup)
        self._fruit_box = create_actor(
            self,
            pose=sapien.Pose(
                [self.BOX_XY[0], self.BOX_XY[1], z0],
                list(self.BASKET_Q),
            ),
            modelname="076_breadbasket",
            model_id=0,
            convex=False,
            is_static=True,
            scale_mult=1.15,
        )
        self._box_top_z = z0 + 0.08
        self._box_half_xy = (0.078, 0.111)
        if self._fruit_box is not None:
            cfg = getattr(self._fruit_box, "config", None) or {}
            extents = cfg.get("extents", [0.0, 0.7, 0.0])
            scale = cfg.get("scale", [1.15] * 3)
            height = float(extents[1]) * float(scale[1])
            if height > 0.0:
                self._box_top_z = z0 + height
            hx = 0.5 * float(extents[2]) * float(scale[2])
            hy = 0.5 * float(extents[0]) * float(scale[0])
            if hx > 0.0 and hy > 0.0:
                self._box_half_xy = (hx, hy)
        self._apple = create_actor(
            self,
            pose=sapien.Pose(
                [self.APPLE_XY[0], self.APPLE_XY[1], z0 + self.APPLE_R],
                list(self.APPLE_Q),
            ),
            modelname="035_apple",
            model_id=0,
            convex=True,
            is_static=False,
            scale_mult=0.80,
        )
        if self._apple is not None:
            try:
                self._apple.set_mass(0.12)
            except Exception:
                pass
            self._tint(self._apple, self.APPLE_COLOR)
        self._apple_selected = False
        self._apple_dropping = False
        self._remember(self._green_pad, self._cup, self._fruit_box, self._apple)

    def _spawn_base_extra_props(self, z0: float, hz: float) -> None:
        self._gummy_arrows = {}
        self._gummy_arrow_bezels = {}
        self._gummy_arrow_homes = {}
        self._gummy_arrow_icons = {}
        self._gummy_arrow_locals = {}
        for side, y in (("left", self.GUMMY_ARROW_YS[0]), ("right", self.GUMMY_ARROW_YS[1])):
            bezel = add_key_base_border(
                self,
                self.GUMMY_ARROW_X,
                y,
                z0,
                self.KEY_HALF,
                name_prefix=f"kb_gummy_base_{side}",
            )
            home = sapien.Pose([self.GUMMY_ARROW_X, y, z0 + hz], [1, 0, 0, 0])
            key = create_box(
                self,
                pose=home,
                half_size=list(self.KEY_HALF),
                color=[0.20, 0.22, 0.26],
                name=f"kb_gummy_key_{side}",
                is_static=True,
            )
            self._attach_chevron(
                key,
                side,
                self._gummy_arrow_icons,
                self._gummy_arrow_locals,
                f"kb_gummy_{side}",
            )
            self._gummy_arrows[side] = key
            self._gummy_arrow_bezels[side] = bezel
            self._gummy_arrow_homes[side] = self._snapshot(key)
            self._remember(key, bezel, self._gummy_arrow_icons[side])
        tx, ty = self.TRIGGER_XY
        self._trigger_bezel = add_key_base_border(
            self, tx, ty, z0, self.KEY_HALF, name_prefix="kb_trigger_base"
        )
        trigger_home = sapien.Pose([tx, ty, z0 + hz], [1, 0, 0, 0])
        self._trigger_key = create_box(
            self,
            pose=trigger_home,
            half_size=list(self.KEY_HALF),
            color=[0.86, 0.22, 0.18],
            name="kb_trigger_key",
            is_static=True,
        )
        self._trigger_home = self._snapshot(self._trigger_key)
        self._trigger_symbol, self._trigger_locals = attach_key_symbol(
            self, self._trigger_key, self.KEY_HALF, "push", "kb_trigger_icon"
        )
        self._gummy_circle = self._spawn_circle(0.0, self.BOWL_Y, "kb_gummy_circle")
        self._gummy_circle_x = 0.0
        self._gummy_circle_on = False
        self._remember(self._trigger_key, self._trigger_bezel, self._trigger_symbol, self._gummy_circle)

        self._gate_keys = {}
        self._gate_bezels = {}
        self._gate_homes = {}
        colors = ((0.86, 0.16, 0.16), (0.18, 0.42, 0.88))
        for side, x, rgb in (("left", self.GATE_KEY_XS[0], colors[0]), ("right", self.GATE_KEY_XS[1], colors[1])):
            bezel = add_key_base_border(
                self, x, self.GATE_KEY_Y, z0, self.KEY_HALF, name_prefix=f"kb_gate_base_{side}"
            )
            home = sapien.Pose([x, self.GATE_KEY_Y, z0 + hz], [1, 0, 0, 0])
            key = create_box(
                self,
                pose=home,
                half_size=list(self.KEY_HALF),
                color=list(rgb),
                name=f"kb_gate_key_{side}",
                is_static=True,
            )
            self._gate_keys[side] = key
            self._gate_bezels[side] = bezel
            self._gate_homes[side] = self._snapshot(key)
            self._remember(key, bezel)
        self._gate_pieces = {}
        half_len = float(self.GATE_BLADE_HALF_LEN)
        half_w = float(self.GATE_BLADE_HALF_W)
        half_h = float(self.GATE_BLADE_HALF_H)
        cz = z0 + half_h
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            piece = create_box(
                self,
                pose=sapien.Pose(
                    [self.GATE_XY[0] + sgn * half_len, self.GATE_XY[1], cz],
                    [1, 0, 0, 0],
                ),
                half_size=[half_len, half_w, half_h],
                color=list(self.GATE_PLANK_COLOR),
                name=f"kb_gate_{side}",
                is_static=False,
            )
            self._make_kinematic(piece)
            self._gate_pieces[side] = piece
            self._remember(piece)
        self._gate_open = 0.0
        self._gate_done = False
        self._gate_hold = (False, False)
        self._set_gate_open(0.0)

        self._aim_circle = self._spawn_circle(
            self.AIM_CIRCLE_XY[0], self.AIM_CIRCLE_XY[1], "kb_aim_circle"
        )
        self._tint(self._aim_circle, (0.18, 0.82, 0.32), emit=True)
        self._stick = create_box(
            self,
            pose=sapien.Pose(
                [self.STICK_HOME_XY[0], self.STICK_HOME_XY[1], z0 + self.STICK_HALF[2]],
                [1, 0, 0, 0],
            ),
            half_size=list(self.STICK_HALF),
            color=[0.55, 0.32, 0.14],
            name="kb_stick",
            is_static=True,
        )
        self._stick_placed = False
        self._stick_yaw = 0.0
        self._stick_seen_left = False
        self._stick_seen_right = False
        self._remember(self._aim_circle, self._stick)

    def _spawn_household_props(self, z0: float) -> None:
        self._roll_ball = create_sphere(
            self,
            sapien.Pose(
                [-self.BALL_X_LIM, self.BALL_Y, z0 + self.BALL_R],
                [1, 0, 0, 0],
            ),
            radius=self.BALL_R,
            color=[0.95, 0.45, 0.12],
            is_static=False,
            name="kb_roll_ball",
        )
        if self._roll_ball is not None:
            self._make_kinematic(self._roll_ball)
        self._ball_x = -float(self.BALL_X_LIM)
        self._ball_dir = 1.0
        self._ball_stopped = False

        self._knob = self._spawn_cooktop_knob(*self.KNOB_XY)
        self._set_knob_turned(False)
        lx, ly = self.HH_LAMP_XY
        self._hh_lamp, self._hh_stem = self._spawn_lamp(lx, ly, "kb_hh_lamp")
        self._hh_on = False
        self._hh_seen_on = False
        self._hh_seen_off = False

        self._board_circle = self._spawn_circle(
            self.BOARD_CIRCLE_XY[0], self.BOARD_CIRCLE_XY[1], "kb_board_circle"
        )
        self._tint(self._board_circle, (0.18, 0.82, 0.32), emit=True)
        self._board = create_box(
            self,
            pose=sapien.Pose(
                [self.BOARD_HOME_XY[0], self.BOARD_HOME_XY[1], z0 + self.BOARD_HALF[2]],
                [1, 0, 0, 0],
            ),
            half_size=list(self.BOARD_HALF),
            color=[0.78, 0.62, 0.38],
            name="kb_board",
            is_static=True,
        )
        self._board_placed = False
        self._board_tilt = 0.0
        self._board_seen_left = False
        self._board_seen_right = False
        self._remember(
            self._roll_ball,
            self._knob,
            self._hh_lamp,
            self._hh_stem,
            self._board_circle,
            self._board,
        )

    def _spawn_cooktop_knob(self, x, y):
        """Black rotary knob matching KitchenS / cook_food (tick +Y = off)."""
        knob_r = float(self.KNOB_RADIUS)
        knob_half = float(self.KNOB_HEIGHT) / 2.0
        knob_z = float(self.TABLE_Z) + knob_half + 0.008
        self._knob_xyz = (float(x), float(y), knob_z)
        cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        knob_mat = sapien.render.RenderMaterial(base_color=[0.07, 0.07, 0.08, 1.0])
        knob_mat.metallic = 0.25
        knob_mat.roughness = 0.30
        stem_mat = sapien.render.RenderMaterial(base_color=[0.18, 0.18, 0.20, 1.0])
        stem_mat.metallic = 0.55
        stem_mat.roughness = 0.35
        tick_mat = sapien.render.RenderMaterial(base_color=[0.92, 0.92, 0.88, 1.0])
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")
        phys = self.scene.default_physical_material
        stem_half = float(knob_half * 0.45)
        builder.add_cylinder_visual(
            pose=sapien.Pose(p=[0.0, 0.0, -stem_half * 0.3], q=cyl_q),
            radius=float(knob_r * 0.35),
            half_length=stem_half,
            material=stem_mat,
        )
        builder.add_cylinder_collision(
            pose=sapien.Pose(q=cyl_q),
            radius=knob_r,
            half_length=knob_half,
            material=phys,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(q=cyl_q),
            radius=knob_r,
            half_length=knob_half,
            material=knob_mat,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(p=[0.0, 0.0, float(knob_half * 0.35)], q=cyl_q),
            radius=float(knob_r * 1.08),
            half_length=float(knob_half * 0.28),
            material=knob_mat,
        )
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.0, float(knob_r * 0.55), float(knob_half + 0.002)]),
            half_size=[0.0025, 0.007, 0.0015],
            material=tick_mat,
        )
        builder.set_initial_pose(sapien.Pose(p=[float(x), float(y), knob_z]))
        return builder.build(name="kb_knob")

    def _set_knob_turned(self, on: bool) -> None:
        xyz = getattr(self, "_knob_xyz", None)
        if self._knob is None or xyz is None:
            return
        yaw = float(self.KNOB_ON_ANGLE if on else self.KNOB_OFF_ANGLE)
        q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        pose = sapien.Pose([xyz[0], xyz[1], xyz[2]], q)
        try:
            self._knob.set_pose(pose)
        except Exception:
            self._apply_pose(self._knob, pose)

    def _spawn_lamp(self, x, y, name):
        z0 = float(self.TABLE_Z)
        stem = create_visual_box(
            self,
            sapien.Pose([x, y, z0 + 0.5 * (self.LAMP_Z - self.LAMP_RADIUS)], [1, 0, 0, 0]),
            half_size=[0.004, 0.004, 0.5 * (self.LAMP_Z - self.LAMP_RADIUS)],
            color=[0.35, 0.35, 0.38],
            name=f"{name}_stem",
        )
        lamp = create_sphere(
            self,
            sapien.Pose([x, y, z0 + self.LAMP_Z], [1, 0, 0, 0]),
            radius=self.LAMP_RADIUS,
            color=list(self.LAMP_OFF),
            is_static=True,
            name=name,
        )
        return lamp, stem

    def _spawn_circle(self, x, y, name):
        return create_box(
            self,
            pose=sapien.Pose([x, y, float(self.TABLE_Z) + 0.0018], [1, 0, 0, 0]),
            half_size=[self.CIRCLE_RADIUS, self.CIRCLE_RADIUS, 0.0015],
            color=[0.85, 0.12, 0.12],
            name=name,
            is_static=True,
        )

    def _set_num_lamp(self, idx: int, on: bool) -> None:
        self._num_lamp_on[idx] = bool(on)
        rgb = self.NUM_COLORS[idx] if on else self.LAMP_OFF
        self._tint(self._num_lamps[idx], rgb, emit=on)

    def _set_toggle_lamp(self, kind: str, on: bool, *, count: bool = True) -> None:
        on = bool(on)
        rgb = (0.20, 0.85, 0.32) if on else self.LAMP_OFF
        if kind == "switch":
            self._switch_on = on
            if count:
                if on:
                    self._switch_seen_on = True
                else:
                    self._switch_seen_off = True
            self._tint(self._switch_lamp, rgb, emit=on)
            return
        if kind == "hh":
            self._hh_on = on
            if count:
                if on:
                    self._hh_seen_on = True
                else:
                    self._hh_seen_off = True
            self._tint(self._hh_lamp, rgb, emit=on)
            return
        self._push_on = on
        if count:
            if on:
                self._push_seen_on = True
            else:
                self._push_seen_off = True
        self._tint(self._push_lamp, rgb, emit=on)

    def _set_circle(self, disc, on: bool) -> None:
        rgb = (0.15, 0.82, 0.28) if on else (0.85, 0.12, 0.12)
        self._tint(disc, rgb, emit=on)

    def _tint(self, obj, rgb, emit: bool = False) -> None:
        color = list(rgb)[:3] + [1.0]
        emit_c = [c * 0.55 for c in rgb[:3]] + [1.0] if emit else [0.0, 0.0, 0.0, 1.0]
        for shape in self._render_shapes(obj):
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass
            try:
                shape.material.emission = emit_c
            except Exception:
                pass

    # ----------------------------------------------------------------- bowl
    def _tick_bowl(self) -> None:
        dt = float(self.scene.get_timestep()) if hasattr(self.scene, "get_timestep") else 0.004
        hold = self._bowl_hold
        if hold == "left":
            self._bowl_x -= self.BOWL_SPEED * dt
        elif hold == "right":
            self._bowl_x += self.BOWL_SPEED * dt
        lo, hi = float(self.ARROW_XS[0]), float(self.ARROW_XS[1])
        if self._tutorial_stage in ("gummy_keys", "gummy_mouse"):
            lo, hi = float(self.GUMMY_BOWL_LO), float(self.GUMMY_BOWL_HI)
        self._bowl_x = float(np.clip(self._bowl_x, lo, hi))
        self._set_bowl_x(self._bowl_x)
        if self._tutorial_stage in ("bowl_keys", "bowl_mouse"):
            for side, x in (("left", lo), ("right", hi)):
                if abs(self._bowl_x - x) <= self.HIT_TOL:
                    if not self._circle_on[side]:
                        self._circle_on[side] = True
                        self._set_circle(self._circles[side], True)

    def _set_bowl_x(self, x: float) -> None:
        self._bowl_x = float(x)
        if self.bowl is None:
            return
        z = float(self.TABLE_Z) + 0.002
        pose = sapien.Pose([self._bowl_x, self.BOWL_Y, z], list(self.BOWL_Q))
        self._apply_pose(self.bowl, pose)

    def _sync_arrow_icons(self) -> None:
        for side, parts in getattr(self, "_arrow_icon_parts", {}).items():
            locals_ = self._arrow_icon_locals.get(side, [])
            sync_key_symbol(parts, locals_, self._arrow_keys[side])

    def _sync_gummy_icons(self) -> None:
        for side, parts in getattr(self, "_gummy_arrow_icons", {}).items():
            locals_ = self._gummy_arrow_locals.get(side, [])
            sync_key_symbol(parts, locals_, self._gummy_arrows[side])
        sync_key_symbol(self._trigger_symbol, self._trigger_locals, self._trigger_key)

    def _replace_arrow_icon(self, key, side: str) -> None:
        self._attach_chevron(
            key, side, self._arrow_icon_parts, self._arrow_icon_locals, f"kb_arrow_{side}"
        )

    def _attach_chevron(self, key, side: str, store_parts, store_locals, prefix: str) -> None:
        """Left/right chevrons on a key."""
        entity = key.actor if hasattr(key, "actor") else key
        try:
            wp = entity.get_pose()
        except Exception:
            return
        yaw = 0.0 if side == "left" else np.pi
        cx, cy = float(wp.p[0]), float(wp.p[1])
        cz = float(wp.p[2]) + float(self.KEY_HALF[2]) + 0.0015
        color = [0.95, 0.95, 0.95]
        templates = [
            ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
            ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
            ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ]
        c, s = np.cos(yaw), np.sin(yaw)
        parts = []
        locals_ = []
        inv = wp.inv()
        for name, (lx, ly), local_yaw, half in templates:
            x = cx + c * lx - s * ly
            y = cy + s * lx + c * ly
            heading = local_yaw + yaw
            q = [np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)]
            box = create_visual_box(
                self,
                sapien.Pose([x, y, cz], q),
                half_size=list(half),
                color=color,
                name=f"{prefix}_{name}",
            )
            parts.append(box)
            local = inv * box.get_pose()
            locals_.append(sapien.Pose(list(local.p), list(local.q)))
        store_parts[side] = parts
        store_locals[side] = locals_

    def _setup_gummy_circle(self) -> None:
        lo = float(self.GUMMY_BOWL_LO)
        hi = float(self.GUMMY_BOWL_HI)
        sep = float(self.GUMMY_MIN_SEP)
        if float(np.random.rand()) < 0.5:
            bowl_x = lo
            circle_x = float(np.random.uniform(bowl_x + sep, hi))
        else:
            bowl_x = hi
            circle_x = float(np.random.uniform(lo, bowl_x - sep))
        self._gummy_circle_x = circle_x
        self._gummy_circle_on = False
        z = float(self.TABLE_Z) + 0.0018
        self._apply_pose(
            self._gummy_circle,
            sapien.Pose([self._gummy_circle_x, self.BOWL_Y, z], [1, 0, 0, 0]),
        )
        self._set_circle(self._gummy_circle, False)
        self._restore(self.bowl)
        self._set_bowl_x(bowl_x)

    def _tick_gate(self) -> None:
        left, right = self._gate_hold if getattr(self, "_gate_hold", None) else (False, False)
        if left and right:
            self._gate_open = min(1.0, float(self._gate_open) + 0.045)
            if self._gate_open >= 0.99:
                self._gate_done = True
        else:
            self._gate_open = max(0.0, float(self._gate_open) - 0.03)
        self._set_gate_open(self._gate_open)

    def _set_gate_open(self, amount: float) -> None:
        """Rest: one horizontal bar. Open: halves yaw ±90° about their centers."""
        amount = float(np.clip(amount, 0.0, 1.0))
        yaws = {
            "left": -amount * float(self.GATE_OPEN_ANGLE),
            "right": amount * float(self.GATE_OPEN_ANGLE),
        }
        half_len = float(self.GATE_BLADE_HALF_LEN)
        cz = float(self.TABLE_Z) + float(self.GATE_BLADE_HALF_H)
        cx0, cy0 = float(self.GATE_XY[0]), float(self.GATE_XY[1])
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            yaw = float(yaws[side])
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            pose = sapien.Pose(
                [cx0 + sgn * half_len, cy0, cz],
                q,
            )
            self._apply_pose(self._gate_pieces[side], pose)

    def _place_stick_home(self) -> None:
        z = float(self.TABLE_Z) + float(self.STICK_HALF[2])
        self._apply_pose(
            self._stick,
            sapien.Pose(
                [self.STICK_HOME_XY[0], self.STICK_HOME_XY[1], z],
                [1, 0, 0, 0],
            ),
        )
        self._tint(self._aim_circle, (0.18, 0.82, 0.32), emit=True)

    def _apply_stick_pose(self) -> None:
        yaw = float(self._stick_yaw)
        q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        half = float(self.STICK_HALF[0])
        tip_x, tip_y = self.AIM_CIRCLE_XY
        z = float(self.TABLE_Z) + 0.02
        body = [tip_x - np.cos(yaw) * half, tip_y - np.sin(yaw) * half, z]
        self._apply_pose(self._stick, sapien.Pose(body, q))

    def _place_board_home(self) -> None:
        z = float(self.TABLE_Z) + float(self.BOARD_HALF[2])
        self._apply_pose(
            self._board,
            sapien.Pose(
                [self.BOARD_HOME_XY[0], self.BOARD_HOME_XY[1], z],
                [1, 0, 0, 0],
            ),
        )
        self._tint(self._board_circle, (0.18, 0.82, 0.32), emit=True)

    def _apply_board_pose(self) -> None:
        tilt = float(self._board_tilt)
        q = [np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0]
        cx, cy = self.BOARD_CIRCLE_XY
        z = float(self.TABLE_Z) + 0.055
        self._apply_pose(self._board, sapien.Pose([cx, cy, z], q))

    def _reset_cup(self) -> None:
        z = float(self.TABLE_Z) + 0.002
        self._apply_pose(
            self._cup,
            sapien.Pose([self.CUP_XY[0], self.CUP_XY[1], z], list(self.CUP_Q)),
        )
        self._tint(self._green_pad, (0.18, 0.82, 0.32), emit=True)

    def _reset_apple(self) -> None:
        z = float(self.TABLE_Z) + float(self.APPLE_R)
        self._set_dynamic(self._apple, kinematic=False, gravity=True)
        self._apply_pose(
            self._apple,
            sapien.Pose([self.APPLE_XY[0], self.APPLE_XY[1], z], list(self.APPLE_Q)),
        )
        self._zero_vel(self._apple)
        self._tint(self._apple, self.APPLE_COLOR, emit=False)
        self._apple_selected = False
        self._apple_dropping = False

    def _tick_apple(self) -> None:
        if self._apple is None:
            return
        if self._apple_off_table():
            self._reset_apple()
            return
        p = self._pose_p(self._apple)
        if p is None:
            return
        if float(p[2]) < float(self.TABLE_Z) - 0.02:
            self._reset_apple()

    def _tick_roll_ball(self) -> None:
        if self._ball_stopped or self._roll_ball is None:
            return
        dt = float(self.scene.get_timestep()) if hasattr(self.scene, "get_timestep") else 0.004
        self._ball_x += self.BALL_SPEED * float(self._ball_dir) * dt
        lo, hi = -float(self.BALL_X_LIM), float(self.BALL_X_LIM)
        if self._ball_x >= hi:
            self._ball_x = hi
            self._ball_dir = -1.0
        elif self._ball_x <= lo:
            self._ball_x = lo
            self._ball_dir = 1.0
        z = float(self.TABLE_Z) + float(self.BALL_R)
        self._apply_pose(
            self._roll_ball,
            sapien.Pose([self._ball_x, self.BALL_Y, z], [1, 0, 0, 0]),
        )

    def _reset_roll_ball(self) -> None:
        self._ball_x = -float(self.BALL_X_LIM)
        self._ball_dir = 1.0
        self._ball_stopped = False
        z = float(self.TABLE_Z) + float(self.BALL_R)
        self._apply_pose(
            self._roll_ball,
            sapien.Pose([self._ball_x, self.BALL_Y, z], [1, 0, 0, 0]),
        )

    def _cup_fully_in_pad(self) -> bool:
        p = self._pose_p(self._cup)
        if p is None:
            return False
        dx = float(p[0]) - float(self.PAD_XY[0])
        dy = float(p[1]) - float(self.PAD_XY[1])
        return float(np.hypot(dx, dy)) + float(self.CUP_R) <= float(self.PAD_R) + 1e-4

    def _apple_in_box(self) -> bool:
        p = self._pose_p(self._apple)
        if p is None:
            return False
        hx, hy = self._box_half_xy
        dx = abs(float(p[0]) - float(self.BOX_XY[0]))
        dy = abs(float(p[1]) - float(self.BOX_XY[1]))
        z0 = float(self.TABLE_Z)
        return dx <= hx * 0.85 and dy <= hy * 0.85 and z0 - 0.02 <= float(p[2]) <= z0 + 0.18

    def _apple_off_table(self) -> bool:
        p = self._pose_p(self._apple)
        if p is None:
            return True
        if abs(float(p[0])) > self.TABLE_XY_LIM or abs(float(p[1])) > self.TABLE_XY_LIM:
            return True
        return float(p[2]) < float(self.TABLE_Z) - 0.08

    def _pose_p(self, obj):
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return None
        try:
            return np.array(ent.get_pose().p, dtype=float)
        except Exception:
            return None

    def _get_rigid(self, obj):
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return None
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    return comp
        except Exception:
            return None
        return None

    def _set_dynamic(self, obj, *, kinematic: bool, gravity: bool) -> None:
        rigid = self._get_rigid(obj)
        if rigid is None:
            return
        try:
            rigid.set_kinematic(bool(kinematic))
            rigid.set_disable_gravity(not bool(gravity))
        except Exception:
            pass

    def _zero_vel(self, obj) -> None:
        rigid = self._get_rigid(obj)
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    # ----------------------------------------------------------------- bank
    def _make_bank(self, actors, homes, ids=None):
        return ReactivePushButtons(
            self,
            actors=list(actors),
            home_poses=list(homes),
            max_depth=0.012,
            ids=ids,
        )

    def _pulse_key(self, idx) -> None:
        if self._bank is None:
            return
        try:
            self._bank.set_forced(idx, True)
        except Exception:
            return
        self._pulse[idx] = int(self.PULSE_STEPS)

    def _tick_pulses(self) -> None:
        if not self._pulse or self._bank is None:
            return
        done = []
        for idx, left in list(self._pulse.items()):
            left -= 1
            if left <= 0:
                try:
                    self._bank.set_forced(idx, False)
                except Exception:
                    pass
                done.append(idx)
            else:
                self._pulse[idx] = left
        for idx in done:
            self._pulse.pop(idx, None)

    # ----------------------------------------------------------------- show/hide
    def _hide_all(self) -> None:
        slot = 0
        for _key, (obj, _ents, _poses) in list(self._homes.items()):
            self._hide_obj(obj, slot)
            slot += 1

    def _snapshot(self, obj):
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return sapien.Pose()
        return self._pose_copy(ent)

    def _remember(self, *objs) -> None:
        for obj in objs:
            if obj is None:
                continue
            if isinstance(obj, (list, tuple)):
                self._remember(*obj)
                continue
            poses = []
            ents = []
            for ent in self._iter_entities(obj):
                ents.append(ent)
                poses.append(self._pose_copy(ent))
            self._homes[id(obj)] = (obj, ents, poses)

    def _restore(self, *objs) -> None:
        for obj in objs:
            if obj is None:
                continue
            if isinstance(obj, (list, tuple)):
                self._restore(*obj)
                continue
            rec = self._homes.get(id(obj))
            if rec is None:
                continue
            _obj, ents, poses = rec
            for ent, pose in zip(ents, poses):
                try:
                    ent.set_pose(pose)
                except Exception:
                    pass

    def _iter_entities(self, obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._iter_entities(item)
            return
        yield obj.actor if hasattr(obj, "actor") else obj

    def _pose_copy(self, ent) -> sapien.Pose:
        pose = ent.get_pose()
        return sapien.Pose(list(pose.p), list(pose.q))

    def _hide_pose(self, ent, slot: int = 0) -> sapien.Pose:
        return sapien.Pose(
            [self.HIDE_XY[0] + 0.08 * slot, self.HIDE_XY[1], 0.05 + 0.04 * slot],
            [1, 0, 0, 0],
        )

    def _hide_obj(self, obj, slot0: int = 0) -> None:
        for i, ent in enumerate(self._iter_entities(obj)):
            try:
                ent.set_pose(self._hide_pose(ent, slot0 + i))
            except Exception:
                pass

    def _apply_pose(self, obj, pose: sapien.Pose) -> None:
        if obj is None or pose is None:
            return
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return
        rigid = None
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    rigid = comp
                    break
        except Exception:
            rigid = None
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                try:
                    ent.set_pose(pose)
                except Exception:
                    pass
        else:
            try:
                ent.set_pose(pose)
            except Exception:
                pass

    def _make_kinematic(self, entity) -> None:
        ent = entity.actor if hasattr(entity, "actor") else entity
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_disable_gravity(True)
                    comp.set_kinematic(True)
                    return
        except Exception:
            pass

    def _render_shapes(self, actor) -> list:
        entity = actor.actor if hasattr(actor, "actor") else actor
        shapes = []
        try:
            for comp in entity.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    shapes.extend(list(comp.render_shapes))
        except Exception:
            return []
        return shapes
