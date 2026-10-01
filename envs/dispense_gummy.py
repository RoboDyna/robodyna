from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import sapien.physx
import numpy as np


class dispense_gummy(Base_Task):
    """Use a button-controlled belt to catch target-colored gummies.

    The bowl starts at the belt's left endpoint. The right arm presses arrow keys to
    move it, while the left arm presses the dispense key. Each dispense releases one
    ball from both tubes; only the configured target color should land in the bowl.

    Task options (set in ``task_args.dispense_gummy``; independent toggles):
      - Default — alternating target/distractor patterns; discrete belt station hops
          ``layout_mode: alternating``
          ``belt_continuous_motion: false``
      - Option 1 — randomized layout: ``layout_mode: random``
          At most one target gummy at each depth; both tubes may hold distractors.
          CLI: ``--task-arg layout_mode=random`` or legacy ``--option 1``.
      - Option 2 — continuous bowl motion: ``belt_continuous_motion: true``
          Hold an arrow key to slide the bowl continuously. Episode speed is sampled
          as ``bowl_speed × U(1 ± belt_speed_jitter)`` (default ±20%).
          CLI: ``--task-arg belt_continuous_motion=true`` or legacy ``--option 2``.
      - ``randomize_gummy_colors`` — sample target + distractor exclusively from
          {red, yellow, purple, green, blue}. A cue ball of the target color sits
          in the bowl from episode start.
    """

    TUBE_CAPACITY_DEFAULT = 4
    LAYOUT_MODE_DEFAULT = "alternating"  # Default; Opt 1 = "random"
    BELT_CONTINUOUS_DEFAULT = False      # Opt 2
    BALL_DIAMETER_DEFAULT = 0.03
    BALL_SLOT_GAP_DEFAULT = 0.003
    TUBE_INNER_RADIUS_DEFAULT = 0.017
    TUBE_WALL_THICKNESS_DEFAULT = 0.003
    TUBE_BOTTOM_Z_OFFSET_DEFAULT = 0.075
    TUBE_CENTER_Y_DEFAULT = 0.03
    TUBE_X_LEFT_DEFAULT = -0.06
    TUBE_X_RIGHT_DEFAULT = 0.06

    BOWL_RADIUS_DEFAULT = 0.06
    BOWL_ALIGN_TOL_DEFAULT = 0.035
    BOWL_CATCH_Z_TOL_DEFAULT = 0.025
    BOWL_SPEED_DEFAULT = 0.15            # m/s nominal (Opt 2 continuous)
    BELT_SPEED_JITTER_DEFAULT = 0.20     # ± fraction when continuous

    KEY_X_DEFAULT = -0.26
    KEY_Y_DEFAULT = -0.13
    KEY_HALF_DEFAULT = [0.028, 0.028, 0.016]
    KEY_HOVER_DIS_DEFAULT = 0.06
    KEY_PRESS_DEPTH_DEFAULT = 0.055
    KEY_TRAVEL_DEFAULT = 0.008
    KEY_SPRING_STEP_DEFAULT = 0.0015
    BELT_THICKNESS_DEFAULT = 0.015
    BELT_MOVE_STEPS_DEFAULT = 30
    BELT_KEY_X_DEFAULT = 0.26
    BELT_KEY_Y_LEFT_DEFAULT = -0.14
    BELT_KEY_Y_RIGHT_DEFAULT = -0.06
    BELT_KEY_PRESS_XY_DEFAULT = 0.045
    BELT_KEY_PRESS_DZ_DEFAULT = 0.17
    EE_TO_TCP = 0.12

    DISPENSE_STEPS_DEFAULT = 24
    PRESS_HOLD_STEPS_DEFAULT = 6
    POST_PRESS_DWELL_DEFAULT = 10
    PRESS_LOOP_TOL_DEFAULT = 0.008
    PRESS_LOOP_MAX_STEPS_DEFAULT = 2500

    COLORS = {
        "red": [0.90, 0.18, 0.16],
        "yellow": [0.96, 0.82, 0.18],
        "purple": [0.58, 0.28, 0.82],
        "green": [0.22, 0.72, 0.32],
        "blue": [0.20, 0.48, 0.92],
    }
    COLOR_NAMES = ("red", "yellow", "purple", "green", "blue")
    RANDOMIZE_GUMMY_COLORS_DEFAULT = False
    TUBE_COLOR = [0.84, 0.93, 1.00]
    FRAME_COLOR = [0.48, 0.48, 0.52]
    VERTICAL_CYLINDER_Q = [0.70710678, 0.0, -0.70710678, 0.0]

    def setup_demo(self, **kwags):
        kwags = dict(kwags)
        kwags["use_dynamic"] = False
        self._cfg = dict(kwags.get("task_args", {}).get("dispense_gummy", {}))
        self._apply_legacy_option()
        self._tube_order = ("left", "right")
        self._tube_records = {side: [] for side in self._tube_order}
        self._tube_stack_colors = {side: [] for side in self._tube_order}
        self._dispensed_count = {side: 0 for side in self._tube_order}
        self._discard_counts = {side: 0 for side in self._tube_order}
        self._active_drops = []
        self._caught_ball_records = []
        self._pending_restack = False
        self.invalid_pattern = False
        self.yellow_caught = 0
        self.yellow_missed = 0
        self.blue_caught = 0
        self.blue_dropped = 0
        self._caught_by_color = {name: 0 for name in self.COLORS}
        self._missed_by_color = {name: 0 for name in self.COLORS}
        self._reset_metric_state()
        self.randomize_gummy_colors = False
        self.distractor_color_name = "blue"
        self.target_color = "yellow"
        self.press_history = []
        self.tube_centers = {}
        self.discard_anchors = {}
        self.table_top = 0.0
        self.bowl = None
        self._cue_ball = None
        self._cue_ball_rigid = None
        self._bowl_target_x = 0.0
        self._bowl_station_idx = 0
        self._belt_key_latched = {"left": False, "right": False}
        self._belt_key_pressed = {"left": False, "right": False}
        self._dispense_key_latched = False
        self._belt_key_depression = {"left": 0.0, "right": 0.0}
        self._dispense_key_depression = 0.0
        self._reactive_buttons = None
        # Interactive latch: None | "left" | "right"; dispense hold via _expert_dispense.
        self._expert_belt_hold = None
        self._expert_dispense = False
        self._bowl_force_stop = False
        self._bowl_drive_clamp = None
        self.belt_keys = {}
        self.belt_key_xy = {}
        self.belt_key_arrows = {}
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()

    def _apply_legacy_option(self):
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / random / layout_mode → Opt 1 layout_mode=random
        2 / continuous / belt_continuous_motion → Opt 2 belt_continuous_motion=true
        """
        legacy = self._cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "layout_random",
            2: "belt_continuous",
            "1": "layout_random",
            "2": "belt_continuous",
            "random": "layout_random",
            "layout_mode": "layout_random",
            "layout_random": "layout_random",
            "continuous": "belt_continuous",
            "belt_continuous": "belt_continuous",
            "belt_continuous_motion": "belt_continuous",
            "belt_continous_motion": "belt_continuous",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        # Force-enable the named toggle so CLI ``--option N`` wins over yaml defaults.
        if key == "layout_random":
            self._cfg["layout_mode"] = "random"
        elif key == "belt_continuous":
            self._cfg["belt_continuous_motion"] = True
            self._cfg.pop("belt_continous_motion", None)
        else:
            raise ValueError(
                "dispense_gummy option must be 1/layout_mode=random or "
                "2/belt_continuous_motion (or set those keys directly)"
            )

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "layout_mode", self.LAYOUT_MODE_DEFAULT) == "random":
            parts.append("option 1")
        if bool(getattr(self, "belt_continuous_motion", False)):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    # ------------------------------------------------------------------ helpers
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _make_kinematic(self, entity):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
        except Exception:
            pass
        return rigid

    def _set_entity_pose(self, entity, pose):
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                rigid.entity.set_pose(pose)
        else:
            obj = entity.actor if hasattr(entity, "actor") else entity
            obj.set_pose(pose)

    def _configure_observer_camera(self):
        """Frame the fixture closely from the table's upper-right corner."""
        camera = getattr(getattr(self, "cameras", None), "observer_camera", None)
        if camera is None:
            return
        camera_pos = np.array([0.333, 0.280, 1.207], dtype=np.float64)
        look_at = np.array([0.0, 0.0, 0.82], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = np.stack([forward, left, up], axis=1)
        camera_matrix[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(camera_matrix))

    def _slot_pose(self, side, slot_idx):
        x, y = self.tube_centers[side]
        z = self.tube_slot_z[slot_idx]
        return sapien.Pose([x, y, z], [1, 0, 0, 0])

    def _current_bottom_color(self, side):
        idx = self._dispensed_count[side]
        if idx >= len(self._tube_stack_colors[side]):
            return None
        return self._tube_stack_colors[side][idx]

    def _current_target_sides(self):
        return [
            side for side in self._tube_order
            if self._current_bottom_color(side) == self.target_color
        ]

    def _current_target_side(self):
        target_sides = self._current_target_sides()
        return target_sides[0] if len(target_sides) == 1 else None

    def _bowl_center_world(self):
        if self.bowl is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.bowl.get_pose().p, dtype=np.float64)

    def _target_bowl_x(self, station_or_side):
        """Resolve a discrete station index or tube side name to a bowl x target."""
        if isinstance(station_or_side, str):
            if station_or_side in self.tube_centers:
                return float(self.tube_centers[station_or_side][0])
            if station_or_side == "park":
                return float(self.bowl_x_min)
        return float(self.bowl_stations[int(station_or_side)])

    def _advance_bowl_on_belt(self):
        if self.bowl is None:
            return
        pose = self.bowl.get_pose()
        if self.belt_continuous_motion:
            if self._bowl_force_stop:
                return
            dt = float(self.scene.get_timestep())
            left_p = self._belt_key_pressed.get("left", False)
            right_p = self._belt_key_pressed.get("right", False)
            dx = 0.0
            if left_p and not right_p:
                dx = -self.bowl_speed * dt
            elif right_p and not left_p:
                dx = self.bowl_speed * dt
            if dx == 0.0:
                return
            next_x = float(pose.p[0]) + dx
            clamp = self._bowl_drive_clamp
            if clamp is not None:
                clamp_sign, clamp_x = clamp
                next_x = min(next_x, clamp_x) if clamp_sign > 0 else max(next_x, clamp_x)
            next_x = float(np.clip(next_x, self.bowl_x_min, self.bowl_x_max))
        else:
            dx = float(self._bowl_target_x - pose.p[0])
            if abs(dx) < 1e-5:
                return
            step = np.sign(dx) * min(abs(dx), self.belt_step_per_sim)
            next_x = float(np.clip(pose.p[0] + step, self.bowl_x_min, self.bowl_x_max))
            if (dx > 0 and next_x > self._bowl_target_x) or (dx < 0 and next_x < self._bowl_target_x):
                next_x = self._bowl_target_x
        self._set_entity_pose(
            self.bowl,
            sapien.Pose([next_x, self.tube_center_y, self.belt_surface_z], pose.q),
        )

    def _request_bowl_station(self, direction):
        last_idx = len(self.bowl_stations) - 1
        self._bowl_station_idx = int(np.clip(self._bowl_station_idx + direction, 0, last_idx))
        self._bowl_target_x = float(self.bowl_stations[self._bowl_station_idx])

    def _update_reactive_keys(self):
        """One spring-key update for dispense + belt keys (fill_coffee style)."""
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        expert = getattr(self, "_expert_belt_hold", None)
        for side in ("left", "right"):
            bank.set_forced(side, expert == side)
        # Level hold (not a one-frame pulse): the spring needs several steps to
        # reach trigger depth, same as belt keys under ``_expert_belt_hold``.
        bank.set_forced("dispense", bool(getattr(self, "_expert_dispense", False)))

        triggered = set(bank.update())

        if "dispense" in triggered:
            self._request_dispense()
        self._dispense_key_latched = bool(bank.is_held("dispense"))
        self._dispense_key_depression = float(
            bank.visual_depth[bank.resolve_index("dispense")]
        )

        for side in ("left", "right"):
            pressed = bool(bank.is_held(side))
            self._belt_key_pressed[side] = pressed
            self._belt_key_depression[side] = float(
                bank.visual_depth[bank.resolve_index(side)]
            )
            if self.belt_continuous_motion:
                self._belt_key_latched[side] = pressed
            else:
                # Keyboard/mouse sets `_expert_belt_hold` and also `set_forced`,
                # so the spring would fire a second hop a few frames later.
                # Take the expert edge immediately; use `triggered` only for
                # a real arm press (expert is None).
                expert_edge = expert == side and not self._belt_key_latched.get(
                    f"_expert_{side}", False
                )
                physical_edge = side in triggered and expert != side
                if expert_edge or physical_edge:
                    self._request_bowl_station(-1 if side == "left" else 1)
                self._belt_key_latched[side] = pressed
                if expert == side:
                    self._belt_key_latched[f"_expert_{side}"] = True
        if expert not in ("left", "right"):
            self._belt_key_latched["_expert_left"] = False
            self._belt_key_latched["_expert_right"] = False

    def _detect_belt_key_presses(self):
        # Kept for call-site compatibility; real work is in `_update_reactive_keys`.
        pass

    def _detect_dispense_key_press(self):
        pass

    def _animate_keys(self):
        # Keycaps posed by ReactivePushButtons; sync arrow decals to belt-key depth.
        for side in ("left", "right"):
            depth = float(self._belt_key_depression.get(side, 0.0))
            for arrow, rest_xyz in self.belt_key_arrows.get(side, []):
                pose = arrow.get_pose()
                self._set_entity_pose(
                    arrow,
                    sapien.Pose(
                        [rest_xyz[0], rest_xyz[1], rest_xyz[2] - depth],
                        pose.q,
                    ),
                )

    def _caught_ball_pose(self, slot_idx):
        bowl_p = self._bowl_center_world()
        level = slot_idx // 4
        in_level = slot_idx % 4
        ang = in_level * (0.5 * np.pi)
        radius = 0.018 + 0.006 * level
        z = bowl_p[2] + 0.020 + 0.014 * level
        return sapien.Pose(
            [
                bowl_p[0] + radius * np.cos(ang),
                bowl_p[1] + radius * np.sin(ang),
                z,
            ],
            [1, 0, 0, 0],
        )

    def _update_caught_balls(self):
        for slot_idx, record in enumerate(self._caught_ball_records):
            self._set_entity_pose(record["actor"], self._caught_ball_pose(slot_idx))

    def _discard_pose(self, side, slot_idx):
        anchor = self.discard_anchors[side]
        ring = slot_idx // 4
        in_ring = slot_idx % 4
        ang = in_ring * (0.5 * np.pi)
        radius = 0.028 + 0.010 * ring
        return sapien.Pose(
            [
                anchor[0] + radius * np.cos(ang),
                anchor[1] + radius * np.sin(ang),
                anchor[2] + 0.003 * ring,
            ],
            [1, 0, 0, 0],
        )

    def _reposition_tube_balls(self):
        for side in self._tube_order:
            remaining = self._tube_records[side][self._dispensed_count[side]:]
            for slot_idx, record in enumerate(remaining):
                if record is None:
                    continue
                record["state"] = "tube"
                self._set_entity_pose(record["actor"], self._slot_pose(side, slot_idx))

    def _bowl_aligned_side(self):
        bowl_p = self._bowl_center_world()
        z_ok = abs(bowl_p[2] - self.bowl_catch_z) <= self.bowl_catch_z_tol
        if not z_ok:
            return None

        best_side = None
        best_dist = np.inf
        for side in self._tube_order:
            tube_xy = np.asarray(self.tube_centers[side], dtype=np.float64)
            dist = float(np.linalg.norm(bowl_p[:2] - tube_xy))
            if dist < best_dist:
                best_dist = dist
                best_side = side
        if best_dist <= self.bowl_align_tol:
            return best_side
        return None

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _key_tip_pose(self, tip_z_above_top):
        key_top_z = self.table_top + 2.0 * self.key_half[2]
        tcp_z = key_top_z + tip_z_above_top
        ee_z = tcp_z + self.EE_TO_TCP
        return [self.key_x, self.key_y, ee_z, *GRASP_DIRECTION_DIC["top_down"]]

    def _hover_key(self):
        return self.move_to_pose(ArmTag("left"), self._key_tip_pose(self.key_hover_dis))

    def _belt_key_tip_pose(self, side, tip_z_above_top):
        key_x, key_y = self.belt_key_xy[side]
        tcp_z = self.belt_key_top_z + tip_z_above_top
        return [key_x, key_y, tcp_z + self.EE_TO_TCP, *GRASP_DIRECTION_DIC["top_down"]]

    def _press_belt_key(self, side):
        """Discrete mode: tap an arrow key once to hop one station."""
        right = ArmTag("right")
        self.move(self.move_to_pose(right, self._belt_key_tip_pose(side, self.key_hover_dis)))
        self.move(self.move_by_displacement(right, z=-self.key_press_depth))
        self._dwell(2)
        self.move(self.move_by_displacement(right, z=self.key_press_depth))
        self._dwell(self.belt_move_steps + 4)

    def _hold_belt_key_to_x(self, target_x, max_steps=None):
        """Continuous mode: press and hold an arrow key until the bowl reaches target_x."""
        if max_steps is None:
            max_steps = self.press_loop_max_steps
        cur_x = float(self._bowl_center_world()[0])
        if abs(target_x - cur_x) <= self.press_loop_tol:
            return
        side = "right" if target_x > cur_x else "left"
        right = ArmTag("right")
        self._bowl_force_stop = False
        sign = 1.0 if side == "right" else -1.0
        self._bowl_drive_clamp = (sign, float(target_x))
        self.move(self.move_to_pose(right, self._belt_key_tip_pose(side, self.key_hover_dis)))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return
        self.move(self.move_by_displacement(right, z=-self.key_press_depth))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return

        steps = 0
        while steps < max_steps:
            cur_x = float(self._bowl_center_world()[0])
            if sign * (target_x - cur_x) <= self.press_loop_tol:
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (steps % self.save_freq == 0):
                self._take_picture()
            steps += 1
        self._bowl_drive_clamp = None
        self._bowl_force_stop = True
        self.move(self.move_by_displacement(right, z=self.key_press_depth))
        self._dwell(6)
        self._bowl_force_stop = False

    def _move_bowl_to_target(self, target_x):
        if self.belt_continuous_motion:
            self._hold_belt_key_to_x(target_x)
            return abs(float(self._bowl_center_world()[0]) - float(target_x)) <= (
                self.press_loop_tol + self.bowl_align_tol * 0.25
            )

        # Discrete: hop station-by-station toward the nearest station to target_x.
        target_station = int(np.argmin(np.abs(np.asarray(self.bowl_stations) - float(target_x))))
        attempts = 0
        while self._bowl_station_idx < target_station and attempts < len(self.bowl_stations):
            self._press_belt_key("right")
            attempts += 1
        while self._bowl_station_idx > target_station and attempts < 2 * len(self.bowl_stations):
            self._press_belt_key("left")
            attempts += 1
        return self._bowl_station_idx == target_station

    def _draw_arrow(self, side, key_x, key_y, z):
        # Author one left arrow, then rigidly rotate it 180 degrees for the right key.
        # This guarantees the two icons are exact opposites.
        rotation = 0.0 if side == "left" else np.pi
        color = [0.95, 0.95, 0.95]
        parts = [
            ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
            ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
            ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ]
        c, s = np.cos(rotation), np.sin(rotation)
        arrows = []
        for name, (local_x, local_y), local_yaw, half_size in parts:
            x = key_x + c * local_x - s * local_y
            y = key_y + s * local_x + c * local_y
            yaw = local_yaw + rotation
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            arrow = create_visual_box(
                self,
                sapien.Pose([x, y, z], q),
                half_size=half_size,
                color=color,
                name=f"{side}_arrow_{name}",
            )
            arrows.append((arrow, [x, y, z]))
        return arrows

    def _build_transparent_tube(self, name, center_xyz, radius, half_length):
        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(sapien.Pose(center_xyz, self.VERTICAL_CYLINDER_Q))
        render_component = sapien.render.RenderBodyComponent()
        material = sapien.render.RenderMaterial(base_color=[*self.TUBE_COLOR, 0.16])
        material.set_transmission(1.0)
        material.set_transmission_roughness(0.05)
        material.set_roughness(0.08)
        render_component.attach(
            sapien.render.RenderShapeCylinder(
                radius=radius,
                half_length=half_length,
                material=material,
            )
        )
        entity.add_component(render_component)
        self.scene.add_entity(entity)
        return entity

    def _distractor_color(self):
        name = getattr(self, "distractor_color_name", None)
        if name is not None:
            return str(name)
        # Legacy binary fallback when colors are not randomized.
        return "blue" if self.target_color == "yellow" else "yellow"

    def _cue_ball_pose(self):
        bowl_p = self._bowl_center_world()
        return sapien.Pose(
            [float(bowl_p[0]), float(bowl_p[1]), float(bowl_p[2] + 0.018)],
            [1, 0, 0, 0],
        )

    def _update_cue_ball(self):
        if self._cue_ball is None:
            return
        pose = self._cue_ball_pose()
        self._set_entity_pose(self._cue_ball, pose)

    def _validate_stack_pattern(self):
        distractor = self._distractor_color()
        allowed = {self.target_color, distractor}
        lengths = [len(self._tube_stack_colors[side]) for side in self._tube_order]
        if lengths[0] != lengths[1]:
            raise ValueError("dispense_gummy requires left_stack and right_stack to have the same length")
        if lengths[0] > self.tube_capacity:
            raise ValueError("dispense_gummy stack length exceeds tube_capacity")
        for depth in range(lengths[0]):
            left_color = self._tube_stack_colors["left"][depth]
            right_color = self._tube_stack_colors["right"][depth]
            if left_color not in allowed or right_color not in allowed:
                raise ValueError(
                    f"dispense_gummy layout_mode={self.layout_mode} supports only {sorted(allowed)}"
                )
            target_count = int(left_color == self.target_color) + int(right_color == self.target_color)
            if target_count > 1:
                raise ValueError(
                    "dispense_gummy cannot place more than one target gummy at the same depth"
                )
        target_counts = [
            sum(color == self.target_color for color in self._tube_stack_colors[side])
            for side in self._tube_order
        ]
        if min(target_counts) < 1:
            raise ValueError("dispense_gummy requires at least one target gummy per tube")
        if self.layout_mode == "random" and target_counts[0] == target_counts[1]:
            raise ValueError(
                "dispense_gummy random layout requires different target-gummy counts per tube"
            )

    def _generate_alternating_pattern(self):
        """Default: alternating target/distractor, offset across tubes so one target per depth."""
        distractor = self._distractor_color()
        start_left_target = bool(np.random.randint(2))
        left, right = [], []
        for depth in range(self.tube_capacity):
            left_is_target = (depth % 2 == 0) == start_left_target
            left.append(self.target_color if left_is_target else distractor)
            right.append(distractor if left_is_target else self.target_color)
        return {"left": left, "right": right}

    def _generate_random_pattern(self):
        """Opt 1: random placement; ≤1 target per depth; distractors may both be present."""
        if self.tube_capacity < 3:
            raise ValueError("dispense_gummy randomized difficulty requires tube_capacity >= 3")
        distractor = self._distractor_color()
        possible_counts = [
            (left_count, right_count)
            for left_count in range(1, self.tube_capacity)
            for right_count in range(1, self.tube_capacity)
            if left_count != right_count and left_count + right_count <= self.tube_capacity
        ]
        left_count, right_count = possible_counts[np.random.randint(len(possible_counts))]
        depths = np.random.permutation(self.tube_capacity)
        left_depths = set(int(i) for i in depths[:left_count])
        right_depths = set(int(i) for i in depths[left_count:left_count + right_count])
        return {
            "left": [
                self.target_color if depth in left_depths else distractor
                for depth in range(self.tube_capacity)
            ],
            "right": [
                self.target_color if depth in right_depths else distractor
                for depth in range(self.tube_capacity)
            ],
        }

    def _generate_stack_pattern(self):
        if self.layout_mode == "random":
            return self._generate_random_pattern()
        return self._generate_alternating_pattern()

    def _request_dispense(self):
        if self._active_drops:
            return

        target_sides = self._current_target_sides()
        target_side = target_sides[0] if len(target_sides) == 1 else None
        bowl_side = self._bowl_aligned_side()
        # Metrics: the press is the decisive act — the bowl's alignment at THIS
        # instant is what decides caught vs. discarded.
        self._latch_dispense_metrics(target_side)
        event = {
            "press_index": int(len(self.press_history)),
            "target_side": target_side,
            "bowl_side": bowl_side,
            "bottom_colors": {side: self._current_bottom_color(side) for side in self._tube_order},
        }
        self.press_history.append(event)

        if len(target_sides) > 1:
            self.invalid_pattern = True
            return

        for side in self._tube_order:
            idx = self._dispensed_count[side]
            if idx >= len(self._tube_records[side]):
                continue
            record = self._tube_records[side][idx]
            self._dispensed_count[side] += 1
            if record is None:
                continue
            caught = bowl_side == side
            if caught:
                end_pose = sapien.Pose(
                    [*self._bowl_center_world()[:2], self.bowl_catch_z + 0.01],
                    [1, 0, 0, 0],
                )
            else:
                end_pose = self._discard_pose(side, self._discard_counts[side])

            record["state"] = "dropping"
            self._active_drops.append(
                {
                    "side": side,
                    "color": record["color"],
                    "record": record,
                    "step": 0,
                    "caught": caught,
                    "start_pose": record["actor"].get_pose(),
                    "end_pose": end_pose,
                }
            )
        self._pending_restack = True

    def _finish_drop(self, drop):
        side = drop["side"]
        color = drop["color"]
        record = drop["record"]

        if drop["caught"]:
            record["state"] = "caught"
            self._caught_ball_records.append(record)
            self._caught_by_color[color] = int(self._caught_by_color.get(color, 0)) + 1
            if color == "yellow":
                self.yellow_caught += 1
            elif color == "blue":
                self.blue_caught += 1
            return

        record["state"] = "discarded"
        self._set_entity_pose(record["actor"], self._discard_pose(side, self._discard_counts[side]))
        self._discard_counts[side] += 1
        self._missed_by_color[color] = int(self._missed_by_color.get(color, 0)) + 1
        if color == "yellow":
            self.yellow_missed += 1
        elif color == "blue":
            self.blue_dropped += 1

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = 0.74 + self.table_z_bias

        self.ball_diameter = float(cfg.get("ball_diameter", self.BALL_DIAMETER_DEFAULT))
        self.ball_radius = 0.5 * self.ball_diameter
        self.ball_slot_gap = float(cfg.get("ball_slot_gap", self.BALL_SLOT_GAP_DEFAULT))
        self.tube_capacity = int(cfg.get("tube_capacity", self.TUBE_CAPACITY_DEFAULT))
        self.tube_inner_radius = float(cfg.get("tube_inner_radius", self.TUBE_INNER_RADIUS_DEFAULT))
        self.tube_wall_thickness = float(cfg.get("tube_wall_thickness", self.TUBE_WALL_THICKNESS_DEFAULT))
        self.tube_bottom_z_offset = float(cfg.get("tube_bottom_z_offset", self.TUBE_BOTTOM_Z_OFFSET_DEFAULT))
        self.tube_center_y = float(cfg.get("tube_center_y", self.TUBE_CENTER_Y_DEFAULT))
        self.key_x = float(cfg.get("key_x", self.KEY_X_DEFAULT))
        self.key_y = float(cfg.get("key_y", self.KEY_Y_DEFAULT))
        self.key_half = list(cfg.get("key_half", self.KEY_HALF_DEFAULT))
        self.key_hover_dis = float(cfg.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        self.key_press_depth = float(cfg.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT))
        self.key_travel = float(cfg.get("key_travel", self.KEY_TRAVEL_DEFAULT))
        self.key_spring_step = float(cfg.get("key_spring_step", self.KEY_SPRING_STEP_DEFAULT))
        self.dispense_steps = int(cfg.get("dispense_steps", self.DISPENSE_STEPS_DEFAULT))
        self.press_hold_steps = int(cfg.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(cfg.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))
        self.bowl_align_tol = float(cfg.get("bowl_align_tol", self.BOWL_ALIGN_TOL_DEFAULT))
        self.bowl_catch_z_tol = float(cfg.get("bowl_catch_z_tol", self.BOWL_CATCH_Z_TOL_DEFAULT))
        self.bowl_radius = float(cfg.get("bowl_radius", self.BOWL_RADIUS_DEFAULT))
        self.belt_thickness = float(cfg.get("belt_thickness", self.BELT_THICKNESS_DEFAULT))
        self.belt_move_steps = int(cfg.get("belt_move_steps", self.BELT_MOVE_STEPS_DEFAULT))
        self.belt_key_press_xy = float(cfg.get("belt_key_press_xy", self.BELT_KEY_PRESS_XY_DEFAULT))
        self.belt_key_press_dz = float(cfg.get("belt_key_press_dz", self.BELT_KEY_PRESS_DZ_DEFAULT))
        self.press_loop_tol = float(cfg.get("press_loop_tol", self.PRESS_LOOP_TOL_DEFAULT))
        self.press_loop_max_steps = int(cfg.get("press_loop_max_steps", self.PRESS_LOOP_MAX_STEPS_DEFAULT))
        self.randomize_gummy_colors = bool(
            cfg.get("randomize_gummy_colors", self.RANDOMIZE_GUMMY_COLORS_DEFAULT)
        )
        palette = list(cfg.get("gummy_colors", self.COLOR_NAMES))
        palette = [str(c).strip().lower() for c in palette]
        for name in palette:
            if name not in self.COLORS:
                raise ValueError(
                    f"dispense_gummy unknown gummy color {name!r}; "
                    f"expected one of {list(self.COLORS)}"
                )
        if self.randomize_gummy_colors:
            if len(palette) < 2:
                raise ValueError("dispense_gummy randomize_gummy_colors needs ≥2 colors")
            pick = list(np.random.choice(palette, size=2, replace=False))
            self.target_color = str(pick[0])
            self.distractor_color_name = str(pick[1])
        else:
            self.target_color = str(cfg.get("target_color", "yellow")).strip().lower()
            if self.target_color not in self.COLORS:
                raise ValueError(
                    f"dispense_gummy target_color must be one of {list(self.COLORS)}"
                )
            # Legacy default: binary yellow↔blue when not randomizing.
            if self.target_color == "yellow":
                self.distractor_color_name = "blue"
            elif self.target_color == "blue":
                self.distractor_color_name = "yellow"
            else:
                others = [c for c in palette if c != self.target_color] or [
                    c for c in self.COLOR_NAMES if c != self.target_color
                ]
                self.distractor_color_name = str(others[0])
        self._caught_by_color = {name: 0 for name in self.COLORS}
        self._missed_by_color = {name: 0 for name in self.COLORS}
        self._reset_metric_state()

        # Layout mode (default alternating; Opt 1 = random). Accept legacy difficulty_option.
        layout = cfg.get("layout_mode", None)
        if layout is None and "difficulty_option" in cfg:
            # Legacy: difficulty_option 1/2 both mapped to random (gaps mode removed).
            layout = "random"
        if layout is None:
            layout = self.LAYOUT_MODE_DEFAULT
        layout = str(layout).strip().lower()
        layout_aliases = {
            "alternating": "alternating",
            "alternate": "alternating",
            "default": "alternating",
            "random": "random",
            "uneven_blue": "random",
            "1": "random",
        }
        if layout not in layout_aliases:
            raise ValueError("dispense_gummy layout_mode must be alternating or random")
        self.layout_mode = layout_aliases[layout]

        continuous = cfg.get(
            "belt_continuous_motion",
            cfg.get("belt_continous_motion", self.BELT_CONTINUOUS_DEFAULT),
        )
        if isinstance(continuous, str):
            continuous = continuous.strip().lower() in ("1", "true", "yes", "on", "continuous")
        self.belt_continuous_motion = bool(continuous)

        bowl_speed_nom = float(cfg.get("bowl_speed", self.BOWL_SPEED_DEFAULT))
        speed_jitter = float(cfg.get("belt_speed_jitter", self.BELT_SPEED_JITTER_DEFAULT))
        speed_jitter = float(np.clip(speed_jitter, 0.0, 0.95))
        if self.belt_continuous_motion:
            self.bowl_speed = float(
                np.random.uniform(
                    bowl_speed_nom * (1.0 - speed_jitter),
                    bowl_speed_nom * (1.0 + speed_jitter),
                )
            )
        else:
            self.bowl_speed = bowl_speed_nom

        if self.tube_inner_radius <= self.ball_radius:
            raise ValueError("dispense_gummy tube_inner_radius must exceed the gummy radius")

        self.tube_centers = {
            "left": np.array(
                [
                    float(cfg.get("tube_x_left", self.TUBE_X_LEFT_DEFAULT)),
                    self.tube_center_y,
                ],
                dtype=np.float64,
            ),
            "right": np.array(
                [
                    float(cfg.get("tube_x_right", self.TUBE_X_RIGHT_DEFAULT)),
                    self.tube_center_y,
                ],
                dtype=np.float64,
            ),
        }
        self.discard_anchors = {
            side: np.array(
                [
                    self.tube_centers[side][0],
                    self.tube_centers[side][1] + 0.14,
                    self.table_top + self.ball_radius,
                ],
                dtype=np.float64,
            )
            for side in self._tube_order
        }

        if "left_stack" in cfg or "right_stack" in cfg:
            if "left_stack" not in cfg or "right_stack" not in cfg:
                raise ValueError("dispense_gummy custom layouts require both left_stack and right_stack")
            self._tube_stack_colors = {
                "left": [str(c).strip().lower() for c in cfg["left_stack"]],
                "right": [str(c).strip().lower() for c in cfg["right_stack"]],
            }
        else:
            self._tube_stack_colors = self._generate_stack_pattern()
        self._validate_stack_pattern()
        self.num_presses = len(self._tube_stack_colors["left"])
        self.total_yellow = sum(
            color == "yellow"
            for side in self._tube_order
            for color in self._tube_stack_colors[side]
        )
        self.total_target = sum(
            color == self.target_color
            for side in self._tube_order
            for color in self._tube_stack_colors[side]
        )

        stack_span = self.tube_capacity * self.ball_diameter + max(0, self.tube_capacity - 1) * self.ball_slot_gap
        self.tube_half_length = 0.5 * stack_span + 0.02
        tube_outer_radius = self.tube_inner_radius + self.tube_wall_thickness
        bowl_diameter = 2.0 * self.bowl_radius
        self.belt_x_min = float(min(center[0] for center in self.tube_centers.values()) - tube_outer_radius - bowl_diameter)
        self.belt_x_max = float(max(center[0] for center in self.tube_centers.values()) + tube_outer_radius + bowl_diameter)
        self.bowl_x_min = self.belt_x_min + self.bowl_radius
        self.bowl_x_max = self.belt_x_max - self.bowl_radius
        self.belt_surface_z = self.table_top + self.belt_thickness
        self.bowl_catch_z = self.belt_surface_z
        self.bowl_stations = [
            self.bowl_x_min,
            float(self.tube_centers["left"][0]),
            float(self.tube_centers["right"][0]),
            self.bowl_x_max,
        ]
        station_gaps = np.diff(self.bowl_stations)
        self.belt_step_per_sim = float(max(station_gaps) / max(1, self.belt_move_steps))
        self._bowl_station_idx = 0
        self._bowl_target_x = self.bowl_stations[0]
        self.tube_bottom_z = self.belt_surface_z + self.tube_bottom_z_offset
        self.tube_slot_z = [
            self.tube_bottom_z + self.ball_radius + 0.008 + i * (self.ball_diameter + self.ball_slot_gap)
            for i in range(self.tube_capacity)
        ]

        belt_center_x = 0.5 * (self.belt_x_min + self.belt_x_max)
        belt_half_x = 0.5 * (self.belt_x_max - self.belt_x_min)
        self.belt = create_box(
            self,
            pose=sapien.Pose(
                [belt_center_x, self.tube_center_y, self.table_top + 0.5 * self.belt_thickness],
                [1, 0, 0, 0],
            ),
            half_size=[belt_half_x, tube_outer_radius, 0.5 * self.belt_thickness],
            color=[0.10, 0.10, 0.12],
            is_static=True,
            name="gummy_bowl_belt",
        )

        frame_top_z = self.tube_bottom_z + 2.0 * self.tube_half_length + 0.04
        frame_center_x = 0.5 * (self.tube_centers["left"][0] + self.tube_centers["right"][0])
        frame_half_x = 0.5 * abs(self.tube_centers["right"][0] - self.tube_centers["left"][0]) + 0.05
        create_box(
            self,
            pose=sapien.Pose([frame_center_x, self.tube_center_y, frame_top_z], [1, 0, 0, 0]),
            half_size=[frame_half_x, 0.014, 0.014],
            color=self.FRAME_COLOR,
            is_static=True,
            name="gummy_frame_bar",
        )
        for side in self._tube_order:
            tube_top_z = self.tube_bottom_z + 2.0 * self.tube_half_length
            hanger_center_z = 0.5 * (frame_top_z + tube_top_z)
            hanger_half_z = 0.5 * abs(frame_top_z - tube_top_z)
            create_box(
                self,
                pose=sapien.Pose([self.tube_centers[side][0], self.tube_centers[side][1], hanger_center_z], [1, 0, 0, 0]),
                half_size=[0.008, 0.008, hanger_half_z],
                color=self.FRAME_COLOR,
                is_static=True,
                name=f"gummy_hanger_{side}",
            )
            self._build_transparent_tube(
                name=f"gummy_tube_{side}",
                center_xyz=[
                    float(self.tube_centers[side][0]),
                    float(self.tube_centers[side][1]),
                    float(self.tube_bottom_z + self.tube_half_length),
                ],
                radius=self.tube_inner_radius + self.tube_wall_thickness,
                half_length=self.tube_half_length,
            )

        bowl_pose = sapien.Pose(
            [self.bowl_x_min, self.tube_center_y, self.belt_surface_z],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.bowl_id = int(cfg.get("bowl_id", 1))
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
        )
        self.bowl.set_mass(0.06)
        self._make_kinematic(self.bowl)
        self.add_prohibit_area(self.bowl, padding=0.05)
        self.add_prohibit_area(self.belt, padding=0.03)

        # Cue ball of the target color in the bowl (visual reference; not counted).
        self._cue_ball = create_sphere(
            self,
            pose=self._cue_ball_pose(),
            radius=self.ball_radius,
            color=self.COLORS[self.target_color],
            is_static=False,
            name=f"cue_gummy_{self.target_color}",
        )
        self._cue_ball_rigid = self._make_kinematic(self._cue_ball)

        # The red key on the left dispenses one gummy from each tube.
        add_key_base_border(
            self,
            float(self.key_x),
            float(self.key_y),
            float(self.table_top),
            self.key_half,
            color=[0.36, 0.36, 0.40],
            name_prefix="dispense_key_base",
        )
        self.dispense_key_rest_xyz = [
            self.key_x,
            self.key_y,
            self.table_top + self.key_half[2],
        ]
        self.dispense_key_top_z = self.table_top + 2.0 * self.key_half[2]
        self.dispense_key = create_box(
            self,
            pose=sapien.Pose(self.dispense_key_rest_xyz, [1, 0, 0, 0]),
            half_size=self.key_half,
            color=[0.86, 0.22, 0.18],
            is_static=True,
            name="dispense_key",
        )

        # Two arrow-labeled keys on the right move the bowl (discrete hop or continuous hold).
        belt_key_x = float(cfg.get("belt_key_x", self.BELT_KEY_X_DEFAULT))
        self.belt_key_xy = {
            "left": (belt_key_x, float(cfg.get("belt_key_y_left", self.BELT_KEY_Y_LEFT_DEFAULT))),
            "right": (belt_key_x, float(cfg.get("belt_key_y_right", self.BELT_KEY_Y_RIGHT_DEFAULT))),
        }
        self.belt_key_top_z = self.table_top + 2.0 * self.key_half[2]
        self.belt_key_rest_xyz = {}
        for side, (key_x, key_y) in self.belt_key_xy.items():
            add_key_base_border(
                self,
                float(key_x),
                float(key_y),
                float(self.table_top),
                self.key_half,
                color=[0.28, 0.28, 0.31],
                name_prefix=f"belt_key_base_{side}",
            )
            self.belt_key_rest_xyz[side] = [
                key_x,
                key_y,
                self.table_top + self.key_half[2],
            ]
            self.belt_keys[side] = create_box(
                self,
                pose=sapien.Pose(self.belt_key_rest_xyz[side]),
                half_size=self.key_half,
                color=[0.18, 0.48, 0.82],
                is_static=True,
                name=f"belt_key_{side}",
            )
            self.belt_key_arrows[side] = self._draw_arrow(
                side, key_x, key_y, self.belt_key_top_z + 0.0015
            )
            self.add_prohibit_area(self.belt_keys[side], padding=0.04)

        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[
                self.dispense_key,
                self.belt_keys["left"],
                self.belt_keys["right"],
            ],
            home_poses=[
                sapien.Pose(self.dispense_key_rest_xyz, [1, 0, 0, 0]),
                sapien.Pose(self.belt_key_rest_xyz["left"]),
                sapien.Pose(self.belt_key_rest_xyz["right"]),
            ],
            max_depth=float(self.key_half[2]),
            ids=["dispense", "left", "right"],
            # Red dispense = left arm; both arrow belt keys = right arm.
            press_arms=[("left",), ("right",), ("right",)],
            xy_tol=float(self.belt_key_press_xy),
        )
        self._reactive_buttons.set_tops_z([
            self.dispense_key_top_z,
            self.belt_key_top_z,
            self.belt_key_top_z,
        ])

        for side in self._tube_order:
            for depth, color in enumerate(self._tube_stack_colors[side]):
                actor = create_sphere(
                    self,
                    pose=self._slot_pose(side, depth),
                    radius=self.ball_radius,
                    color=self.COLORS[color],
                    is_static=False,
                    name=f"{side}_gummy_{depth}_{color}",
                )
                rigid = self._make_kinematic(actor)
                self._tube_records[side].append(
                    {
                        "actor": actor,
                        "rigid": rigid,
                        "color": color,
                        "state": "tube",
                    }
                )

        self._reposition_tube_balls()

    # ---------------------------------------------------------------- scene motion
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if self.bowl is None:
            return
        self._update_reactive_keys()
        self._animate_keys()
        self._advance_bowl_on_belt()
        self._update_cue_ball()
        self._update_caught_balls()

        if not self._active_drops:
            if self._pending_restack:
                self._reposition_tube_balls()
                self._pending_restack = False
            return

        remaining_drops = []
        for drop in self._active_drops:
            drop["step"] += 1
            frac = min(1.0, drop["step"] / max(1, self.dispense_steps))
            smooth = frac * frac * (3.0 - 2.0 * frac)

            start_p = np.asarray(drop["start_pose"].p, dtype=np.float64)
            end_p = np.asarray(drop["end_pose"].p, dtype=np.float64)
            target_p = start_p + smooth * (end_p - start_p)
            self._set_entity_pose(drop["record"]["actor"], sapien.Pose(target_p.tolist(), [1, 0, 0, 0]))

            if frac >= 1.0:
                self._finish_drop(drop)
            else:
                remaining_drops.append(drop)

        self._active_drops = remaining_drops
        if self._pending_restack and not self._active_drops:
            self._reposition_tube_balls()
            self._pending_restack = False

    # ------------------------------------------------------------------ policy
    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        self.move(self.close_gripper(left))
        self.move(self.close_gripper(right))
        self.move(self._hover_key())

        for _ in range(self.num_presses):
            target_sides = self._current_target_sides()
            if len(target_sides) > 1:
                self.invalid_pattern = True
                break

            if target_sides:
                target_x = float(self.tube_centers[target_sides[0]][0])
            else:
                # No target gummy at this depth: park beyond the left tube and advance one slot.
                target_x = float(self.bowl_x_min)

            if not self._move_bowl_to_target(target_x):
                self.invalid_pattern = True
                break

            self.move(self._hover_key())
            self.move(self.move_by_displacement(left, z=-self.key_press_depth))
            self._dwell(self.press_hold_steps)
            self.move(self.move_by_displacement(left, z=self.key_press_depth))
            self._dwell(self.dispense_steps + self.post_press_dwell)

        self._dwell(max(12, self.save_freq or 12))
        self.info["info"] = {
            "{A}": f"{self.target_color} gummy balls",
            "{B}": f"002_bowl/base{self.bowl_id}",
            "{C}": "dual transparent gummy tubes",
            "{a}": "both arms",
            "{o}": self._option_label(),
        }
        return self.info

    # ------------------------------------------------------------------ success
    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_press_steps = []   # step index of every dispense press
        self._metric_align_errors = []  # alignment error at each press

    def _latch_dispense_metrics(self, target_side):
        """Record the press step and the bowl's alignment error at that instant.

        Captured here, never recomputed: the bowl keeps riding the belt, so its
        offset seconds later says nothing about whether the press was well timed.
        """
        try:
            self._metric_press_steps.append(int(getattr(self, "_exp_sim_steps", 0) or 0))
            if target_side is None:
                return
            bowl_xy = np.asarray(self._bowl_center_world()[:2], dtype=np.float64)
            tube_xy = np.asarray(self.tube_centers[target_side], dtype=np.float64)
            dist = float(np.linalg.norm(bowl_xy - tube_xy))
            self._metric_align_errors.append(
                dist / max(float(self.bowl_align_tol), 1e-9))
        except Exception:
            pass

    def _compute_metrics(self):
        """extra1 = latency to the first dispense press, extra2 = mean aim error.

        ``align_error_norm`` is the bowl-to-target-tube horizontal distance at each
        press, averaged over presses, as a fraction of ``bowl_align_tol``: 0.0 =
        the bowl was dead under the tube, 1.0 = exactly on the catch/discard
        boundary, >1.0 = the gummy was discarded. LOWER IS BETTER. None when no
        press ever targeted a tube.
        """
        errs = list(getattr(self, "_metric_align_errors", []) or [])
        presses = list(getattr(self, "_metric_press_steps", []) or [])
        metrics = {
            "first_press_latency_steps": None,
            "first_press_latency_s": None,
            "presses": int(len(presses)),
            "align_error_norm": (
                round(float(sum(errs)) / float(len(errs)), 6) if errs else None
            ),
            "worst_align_error_norm": (
                round(float(max(errs)), 6) if errs else None
            ),
        }
        if presses:
            steps = int(presses[0])
            metrics["first_press_latency_steps"] = steps
            try:
                metrics["first_press_latency_s"] = round(
                    steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics

    def check_success(self):
        target_caught = int(self._caught_by_color.get(self.target_color, 0))
        target_missed = int(self._missed_by_color.get(self.target_color, 0))
        distractor = self._distractor_color()
        distractor_caught = int(self._caught_by_color.get(distractor, 0))
        # Any non-target ball in the bowl fails (covers multi-color future layouts).
        non_target_caught = sum(
            int(n) for name, n in self._caught_by_color.items() if name != self.target_color
        )
        return bool(
            (not self.invalid_pattern)
            and target_caught == self.total_target
            and target_missed == 0
            and distractor_caught == 0
            and non_target_caught == 0
        )

    def get_obs(self):
        obs = super().get_obs()
        obs["dispense_gummy"] = {
            "yellow_caught": int(self.yellow_caught),
            "yellow_missed": int(self.yellow_missed),
            "blue_caught": int(self.blue_caught),
            "blue_dropped": int(self.blue_dropped),
            "caught_by_color": {k: int(v) for k, v in self._caught_by_color.items()},
            "missed_by_color": {k: int(v) for k, v in self._missed_by_color.items()},
            "target_color": self.target_color,
            "distractor_color": self._distractor_color(),
            "randomize_gummy_colors": bool(self.randomize_gummy_colors),
            "layout_mode": str(self.layout_mode),
            "belt_continuous_motion": bool(self.belt_continuous_motion),
            "bowl_speed": float(self.bowl_speed),
            "option_label": self._option_label(),
            "left_layout": list(self._tube_stack_colors["left"]),
            "right_layout": list(self._tube_stack_colors["right"]),
            "total_target": int(self.total_target),
            "invalid_pattern": bool(self.invalid_pattern),
            "press_count": int(len(self.press_history)),
            "bowl_station": int(self._bowl_station_idx),
            "bowl_x": float(self._bowl_center_world()[0]),
            "current_bottom_colors": {
                side: self._current_bottom_color(side) for side in self._tube_order
            },
            "remaining_left": list(self._tube_stack_colors["left"][self._dispensed_count["left"]:]),
            "remaining_right": list(self._tube_stack_colors["right"][self._dispensed_count["right"]:]),
        }
        return obs
