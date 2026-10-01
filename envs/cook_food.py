"""Cook food on a KitchenS stove (board → pan → shut off).

KitchenS scene (no sink/tap/microwave). The stove starts already on (knob turned,
fire lit). Raw food starts on a chopping board; the robot puts it in the pan,
then turns the knob back to zero at the target doneness.

Cooking mechanism (shared with ``cook_food_timer``):
  - Browning starts the moment food rests in the pan (stove lit, not held).
  - Color / doneness keep advancing whenever the burner is on.
  - They stop only when the knob extinguishes the stove — never freeze while lit.
  - Success scores the doneness frozen at shutoff (food stays in the pan).
  - The success band is a shared 4 s window around each food's target doneness.

A decorative plate of raw meat sits to the right of the stove.

Food types (``task_args.cook_food.food_type`` or random):
  - meat    — 200_steak; red → brown → black; ideal ~medium (0.5)
  - sausage — 259_sausage; pink → red → dark red → black; ideal = dark red (~0.66)
  - onion_half — white cylinder (~4.7 cm Ø × 1.2 cm h); white → yellow → brown
"""
from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_visual_box, UnStableError
from .utils.partial_score import score_open_closed_intervals


class cook_food(KitchenS_base_task):
    """Drop food into a pre-lit pan, then shut the stove off at target doneness."""

    # Seconds past the upper edge of the 4 s doneness window → partial:
    # (0,1] / (1,2] / (2,3] → 0.75 / 0.5 / 0.25
    # (old labels (4,5]/(5,6]/(6,7] included the doneness period; subtract 4 s).
    PARTIAL_OVERCOOK_BANDS = (
        (0.0, 1.0, 0.75),
        (1.0, 2.0, 0.5),
        (2.0, 3.0, 0.25),
    )

    COOK_STEPS_DEFAULT: ClassVar[int] = 3076  # 2× prior 1538 (50% slower / mean cook time ×2)
    COOK_SPEED_JITTER_DEFAULT: ClassVar[float] = 0.20  # per-ep cook_steps ~ U(nom×(1±j))
    COOK_STEPS_ONION_DEFAULT: ClassVar[int] = 1692  # 2× prior 846
    SUCCESS_WINDOW_SEC_DEFAULT: ClassVar[float] = 4.0
    SIM_DT_DEFAULT: ClassVar[float] = 1.0 / 250.0
    KNOB_CONTACT_RADIUS_DEFAULT: ClassVar[float] = 0.06
    EE_TO_TCP: ClassVar[float] = 0.12
    # Lateral then overhead approach (shared KitchenS path — straight-down
    # from the right-apron park fails on centered cooktops).
    KNOB_APPROACH_PATH: ClassVar[tuple] = KitchenS_base_task.TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.012
    # Magnitude of the left turn: 0 = off (tick up), −π/2 = full fire (tick left).
    KNOB_MAX_ANGLE: ClassVar[float] = float(np.pi / 2)
    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    # Base orientation points the handle −Y (toward the robot / front-apron knob).
    # Extra world-Z yaw of +180° sends the handle +Y (rear of the cooktop) so it
    # stays out of the knob approach corridor on both front burners.
    SKILLET_HANDLE_YAW: ClassVar[float] = float(np.pi)
    FOOD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    FOOD_TYPES: ClassVar[tuple[str, ...]] = ("meat", "sausage", "onion_half")
    # Front-left burner on the cooktop; handle points +X (right).
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    FRONT_BURNERS: ClassVar[tuple[str, ...]] = ("left_front", "right_front")
    # Cooktop slightly left; arms stay on their own sides via park poses.
    # Alternate layout: exact table center (same Y).
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.10, 0.14)
    RANGE_CENTER_XY: ClassVar[tuple[float, float]] = (0.0, 0.14)
    BOARD_REL_XY: ClassVar[tuple[float, float]] = (-0.28, -0.02)
    PLATE_REL_XY: ClassVar[tuple[float, float]] = (-0.18, -0.12)
    BOARD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    BOARD_SCALE_DEFAULT: ClassVar[float] = 0.0975
    PLATE_SCALE_DEFAULT: ClassVar[float] = 0.715
    PAN_SCALE_DEFAULT: ClassVar[float] = 1.0
    RANGE_SCALE_DEFAULT: ClassVar[float] = 1.0
    LAYOUT_MARGIN: ClassVar[float] = 0.030
    # Small per-episode jitter (meters) for board / decor / food-on-board.
    POS_JITTER_XY: ClassVar[float] = 0.025
    FOOD_ON_BOARD_JITTER: ClassVar[float] = 0.018

    FOOD_SPECS: ClassVar[dict[str, dict[str, Any]]] = {
        "meat": {
            "modelname": "200_steak",
            "model_id": 0,
            "scale_mult": (0.85, 1.4, 0.85),
            "mass": 0.04,
            "target_doneness": 0.50,
            "color_stops": [
                (0.0, [1.00, 0.12, 0.09]),
                (0.5, [0.66, 0.30, 0.14]),
                (1.0, [0.10, 0.05, 0.03]),
            ],
        },
        "sausage": {
            "modelname": "259_sausage",
            "model_id": 0,
            "scale_mult": 1.0,
            "mass": 0.03,
            "target_doneness": 0.65,
            "color_stops": [
                (0.0, [0.95, 0.55, 0.58]),
                (0.33, [0.85, 0.18, 0.16]),
                (0.66, [0.45, 0.06, 0.06]),
                (1.0, [0.05, 0.03, 0.03]),
            ],
        },
        "onion_half": {
            "modelname": "270_onion_half",
            "model_id": 0,
            # White cylinder: ~4.68 cm diameter, 1.2 cm height.
            "scale_mult": 1.0,
            "mass": 0.028,
            "target_doneness": 0.81,
            # Evenly spaced stops so white → yellow → brown eases without jumps.
            "color_stops": [
                (0.00, [245 / 255.0, 245 / 255.0, 245 / 255.0]),  # #F5F5F5 white
                (0.20, [240 / 255.0, 232 / 255.0, 210 / 255.0]),  # warm white
                (0.40, [228 / 255.0, 200 / 255.0, 140 / 255.0]),  # pale yellow
                (0.55, [212 / 255.0, 168 / 255.0, 90 / 255.0]),   # golden
                (0.70, [181 / 255.0, 124 / 255.0, 62 / 255.0]),   # #B57C3E
                (0.85, [140 / 255.0, 78 / 255.0, 28 / 255.0]),    # brown
                (1.00, [120 / 255.0, 55 / 255.0, 28 / 255.0]),    # deep brown
            ],
        },
    }

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("cook_food", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        self.omit_sink = True
        self.clear_sink_and_range = False
        # Same KitchenS stove scale as boil_milk (must be set before range load).
        self.range_scale_mult = float(
            self._cfg.get("range_scale_mult", self.RANGE_SCALE_DEFAULT)
        )
        self._layout_seed = int(kwags.get("seed", 0) or 0)
        # Stove pose must be chosen before super() loads the cooktop.
        self.range_position_override = self._sample_range_xy(
            self._cfg, np.random.RandomState(self._layout_seed + 17)
        )
        if "table_xy_bias" not in kwags and "table_xy_bias" in self._cfg:
            kwags["table_xy_bias"] = list(self._cfg["table_xy_bias"])

        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._grasp_doneness = None
        self._food_locked = False
        self._food_rel = None
        self._food_shapes: list[Any] = []
        self._placed = False
        self._food_in_pan = False
        self.turned_on_once = False
        self.turned_off_after_cook = False
        self._reset_metric_state()
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._cook_phase_done = False
        self._cooking_idle = False
        self._knob_prestaged = False
        self._burner_shapes = []
        self._ring_shapes: list[Any] = []
        self._ring_parts: list[Any] = []
        self._ring_home_poses: list[Any] = []
        self._skillet_home = None
        self._food_rigid = None

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _load_microwave(self, table_height, table_xy_bias):
        """Open the left counter — no microwave for this task."""
        return

    def _layout_rng(self, salt: int = 0) -> np.random.RandomState:
        return np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + int(salt))

    def _sample_range_xy(self, cfg: dict[str, Any], rng: np.random.RandomState) -> list[float]:
        """Two stove poses: current setup, or exact table center (same Y)."""
        current = list(cfg.get("range_xy", list(self.RANGE_REL_XY)))
        center = list(cfg.get("range_center_xy", list(self.RANGE_CENTER_XY)))
        # Keep Y from the current setup; center only moves X.
        center[1] = float(current[1])
        if not bool(cfg.get("randomize_layout", False)):
            self._stove_pose_choice = "current"
            return [float(current[0]), float(current[1])]
        choice = str(cfg.get("stove_pose", "random")).lower().strip()
        if choice in ("current", "left", "default"):
            pick = "current"
        elif choice in ("center", "centre", "middle"):
            pick = "center"
        else:
            pick = str(rng.choice(["current", "center"]))
        self._stove_pose_choice = pick
        if pick == "center":
            return [float(center[0]), float(center[1])]
        return [float(current[0]), float(current[1])]

    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin: float = 0.0) -> bool:
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        h2 = np.asarray(h2, dtype=float)
        m = float(margin)
        return bool(
            abs(float(c1[0] - c2[0])) < (float(h1[0]) + float(h2[0]) + m)
            and abs(float(c1[1] - c2[1])) < (float(h1[1]) + float(h2[1]) + m)
        )

    def _footprint_clear(self, center, half, blockers, margin: float | None = None) -> bool:
        if margin is None:
            margin = self.LAYOUT_MARGIN
        c = np.asarray(center, dtype=float)
        h = np.asarray(half, dtype=float)
        for b_c, b_h in blockers:
            if self._aabb_overlap(c, h, b_c, b_h, margin):
                return False
        return True

    def _configure_head_camera(self) -> None:
        """Shared household head framing (see ``envs.utils.household_view``)."""
        from .utils.household_view import configure_household_head_camera

        configure_household_head_camera(self)

    # ---------------------------------------------------------------- actors
    def load_actors(self) -> None:
        cfg = self._cfg
        if not hasattr(self, "burner_positions"):
            raise UnStableError("cooking range missing — KitchenS base did not load a range")

        self.knob_contact_radius = float(
            cfg.get("knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )
        # Match cook_meat skillet; board is +30% vs cook_meat for this scene.
        self.pan_scale = float(cfg.get("pan_scale", self.PAN_SCALE_DEFAULT))
        self.plate_scale = float(cfg.get("plate_scale", self.PLATE_SCALE_DEFAULT))
        self.board_scale = float(cfg.get("board_scale_mult", self.BOARD_SCALE_DEFAULT))
        # Expert starts the knob approach this far below target so the remaining
        # approach+twist cook (live fire) lands inside the success band.
        self.shutoff_lead = float(cfg.get("shutoff_lead", 0.58))
        self.cook_intensity = float(np.clip(cfg.get("cook_intensity", 0.75), 0.15, 1.0))
        # Release above the skillet bowl center (functional point 0). Optional
        # place_dx / place_dy nudge only when explicitly set in config.
        self.place_dx = float(cfg.get("place_dx", 0.0))
        self.place_dy = float(cfg.get("place_dy", 0.0))

        food_type = cfg.get("food_type", None)
        if food_type is None or str(food_type).lower() in ("random", "any", ""):
            food_type = str(np.random.choice(self.FOOD_TYPES))
        food_type = str(food_type).strip().lower()
        if food_type not in self.FOOD_SPECS:
            raise ValueError(
                f"cook_food.food_type must be one of {self.FOOD_TYPES}, got {food_type!r}"
            )
        self.food_type = food_type
        self.food_spec = self.FOOD_SPECS[food_type]
        self.color_stops = list(self.food_spec["color_stops"])
        if self.food_type == "onion_half":
            # Flat disc must land at bowl center for a reliable pan re-grasp.
            self.place_dx = float(cfg.get("place_dx", 0.0))
            self.place_dy = float(cfg.get("place_dy", 0.0))
            # Slower than meat so white→yellow→brown still reads on video; ±jitter.
            onion_cfg = dict(cfg)
            onion_cfg["cook_steps"] = float(
                cfg.get("cook_steps_onion", self.COOK_STEPS_ONION_DEFAULT)
            )
            self.cook_steps = self._sample_cook_steps(onion_cfg)
        else:
            self.cook_steps = self._sample_cook_steps(cfg)

        self.success_window_sec = float(
            cfg.get("success_window_sec", self.SUCCESS_WINDOW_SEC_DEFAULT)
        )
        if self.success_window_sec <= 0.0:
            raise ValueError("cook_food success_window_sec must be > 0")
        center = float(
            cfg.get(
                "target_doneness",
                self.food_spec.get("target_doneness", 0.5),
            )
        )
        range_cfg = cfg.get("target_doneness_range")
        if range_cfg is not None:
            lo, hi = float(range_cfg[0]), float(range_cfg[1])
        else:
            lo, hi = self._doneness_range_from_window(center, self.success_window_sec)
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("cook_food target doneness range must satisfy 0 <= min <= max <= 1")
        self.target_doneness_range = (lo, hi)
        self.target_doneness = 0.5 * (lo + hi)

        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._grasp_doneness = None
        self._food_locked = False
        self._food_rel = None
        self._placed = False
        self._food_in_pan = False
        self.turned_on_once = False
        self.turned_off_after_cook = False
        self._reset_metric_state()
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._cook_phase_done = False
        self._cooking_idle = False
        self._knob_prestaged = False
        self._food_welded = False
        self._food_weld_offset = None
        self._food_weld_arm = None

        bz = 0.74 + self.table_z_bias
        rng = self._layout_rng(101)
        randomize = bool(cfg.get("randomize_layout", False))

        # Front burners only: current left_front, or the one next to it (right_front).
        # Left arm places food — keep the active burner reachable (x ≲ 0.12).
        def _burner_reachable(name: str) -> bool:
            px, _py = self.burner_positions[name]
            # Front burners on current/center stove stay within left-arm place reach.
            return float(px) <= 0.30

        if randomize:
            burner_cfg = str(cfg.get("burner", "random")).strip().lower()
        else:
            burner_cfg = str(cfg.get("burner", self.ACTIVE_BURNER)).strip().lower()
            if burner_cfg in ("random", "any", ""):
                burner_cfg = self.ACTIVE_BURNER
        burner_name = burner_cfg
        if burner_name in ("random", "any", ""):
            choices = [b for b in self.FRONT_BURNERS if _burner_reachable(b)]
            if not choices:
                choices = ["left_front"]
            # Equal odds so both front burners show up across small seed batches.
            weights = [1.0 / len(choices)] * len(choices)
            burner_name = str(rng.choice(choices, p=weights))
        if burner_name not in self.FRONT_BURNERS:
            raise ValueError(
                f"cook_food.burner must be one of {self.FRONT_BURNERS} (or random), "
                f"got {burner_cfg!r}"
            )
        if burner_name not in self.burner_positions:
            raise ValueError(
                f"cook_food.burner must be one of {list(self.burner_positions)}, got {burner_name!r}"
            )
        if not _burner_reachable(burner_name):
            print(
                f"[cook_food] burner={burner_name} at x={self.burner_positions[burner_name][0]:.3f} "
                f"unreachable for left-arm place — using left_front",
                flush=True,
            )
            burner_name = "left_front"
        bx, by = self.burner_positions[burner_name]
        self.burner_name = burner_name
        self.burner_xy = (float(bx), float(by))

        # Hide the solid orange disc — it shows through the open pan onto the food.
        if getattr(self, "active_burner", None) is not None:
            try:
                self.active_burner.set_pose(self._fire_hidden_pose())
            except Exception:
                pass
        self._burner_shapes = []
        # Must remove prior ring entities (not just drop list refs) or a gray
        # "off" ring stays visible under the pan forever.
        self._clear_stove_fire_ring()

        # Static pan seated on the burner. Model 2 has a reachable bowl for the
        # left-arm board→pan place; model 0 consistently undershoots / drops short.
        self.skillet_id = int(cfg.get("skillet_id", 2))
        if self.skillet_id not in (0, 2):
            self.skillet_id = 2
        handle_yaw = float(cfg.get("skillet_handle_yaw", self.SKILLET_HANDLE_YAW))
        self.skillet_handle_yaw = handle_yaw
        skillet_q = list(
            t3d.quaternions.qmult(
                t3d.euler.euler2quat(0.0, 0.0, handle_yaw),
                list(self.SKILLET_BASE_QPOS),
            )
        )
        skillet_pose = sapien.Pose(
            [bx, by, float(self.range_top_z) + 0.002],
            skillet_q,
        )
        self.skillet = create_actor(
            self,
            pose=skillet_pose,
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
            is_static=True,
            scale_mult=self.pan_scale,
        )
        self.skillet.set_name("106_skillet")
        # Handle points rear (+Y); bowl footprint is the planner obstacle.
        self.add_prohibit_area(self.skillet, padding=0.04)

        # Slide pan until bowl matches burner XY (no extra visual nudge).
        for _ in range(12):
            bowl = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)
            err = np.array([bx - float(bowl[0]), by - float(bowl[1])], dtype=float)
            if float(np.linalg.norm(err)) < 0.001:
                break
            p = self.skillet.get_pose()
            self.skillet.actor.set_pose(
                sapien.Pose(
                    [float(p.p[0] + err[0]), float(p.p[1] + err[1]), float(p.p[2])],
                    p.q,
                )
            )
        self.burner_xy = (float(bx), float(by))
        self._build_burner_ring(
            self.burner_xy[0], self.burner_xy[1], float(self.range_top_z) + 0.0012
        )
        self._set_burner_visuals(0.0)
        self._skillet_home = self.skillet.get_pose()

        hx, hy = getattr(self, "range_half_size", (0.13, 0.15))
        rx, ry = float(self.range_xy[0]), float(self.range_xy[1])
        pan_half = 0.11 * float(self.pan_scale)
        range_blocker = (
            np.array([rx, ry], dtype=float),
            np.array([float(hx), float(hy)], dtype=float),
        )
        pan_blocker = (
            np.array([float(bx), float(by)], dtype=float),
            np.array([pan_half, pan_half], dtype=float),
        )

        # Chopping board left of the stove (microwave omitted — open counter).
        # 104_board has no authored "scale" key → create_actor leaves config=None;
        # cook_meat restores extents/center so prohibit-area / Z math work.
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        # BOARD_QPOS is 90° about X → local Y (thin axis) becomes world Z thickness.
        board_th = float(board_data["extents"][1]) * self.board_scale
        board_half_x = 0.5 * float(board_data["extents"][0]) * self.board_scale
        board_half_y = 0.5 * float(board_data["extents"][2]) * self.board_scale
        board_half = np.array([board_half_x, board_half_y], dtype=float)
        jitter = float(cfg.get("pos_jitter", self.POS_JITTER_XY if randomize else 0.0))

        # Board stays on the left so the left food-arm can reach both front burners.
        board_y0 = float(cfg.get("board_y", self.BOARD_REL_XY[1]))
        board_on_right = False
        if "board_x" in cfg:
            board_x0 = float(cfg["board_x"])
        elif randomize:
            board_x0 = float(bx) - board_half_x - pan_half - 0.10
            board_x0 = min(board_x0, rx - float(hx) - board_half_x - 0.08)
            board_x0 = float(np.clip(board_x0, -0.40, -0.08))
        else:
            board_x0 = float(rx - float(hx) - board_half_x - 0.12)

        board_xy = self._sample_board_xy(
            rng,
            board_x0,
            board_y0,
            board_half,
            [range_blocker, pan_blocker],
            jitter=jitter,
        )
        board_x, board_y = float(board_xy[0]), float(board_xy[1])
        board_pose = sapien.Pose(
            [board_x, board_y, bz + 0.5 * board_th], list(self.BOARD_QPOS)
        )
        self.board = create_actor(
            self,
            pose=board_pose,
            modelname="104_board",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=self.board_scale,
        )
        self.board.set_name("104_board")
        self.board.config = {
            "scale": [self.board_scale, self.board_scale, self.board_scale],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        self.add_prohibit_area(self.board, padding=0.03)
        self.board_xy = (board_x, board_y)
        self.board_top_z = bz + board_th
        # No serving plate — success is cook quality after the stove is shut off.
        self.plate = None
        self.plate_xy = None
        self.plate_tol = 0.0
        self.plate_top_z = bz

        # Task food starts on the chopping board (slight XY jitter, stays on board).
        food_j = float(
            cfg.get("food_on_board_jitter", self.FOOD_ON_BOARD_JITTER if randomize else 0.0)
        )
        food_dx = float(rng.uniform(-food_j, food_j)) if food_j > 0 else 0.0
        food_dy = float(rng.uniform(-food_j, food_j)) if food_j > 0 else 0.0
        # Keep food clearly on the board surface.
        food_dx = float(np.clip(food_dx, -0.7 * board_half_x, 0.7 * board_half_x))
        food_dy = float(np.clip(food_dy, -0.7 * board_half_y, 0.7 * board_half_y))
        food_pose = sapien.Pose(
            [board_x + food_dx, board_y + food_dy, float(self.board_top_z) + 0.012],
            list(self.FOOD_QPOS),
        )
        self.food = create_actor(
            self,
            pose=food_pose,
            modelname=self.food_spec["modelname"],
            model_id=int(self.food_spec["model_id"]),
            convex=True,
            is_static=False,
            scale_mult=self.food_spec["scale_mult"],
        )
        self.food.set_name(self.food_spec["modelname"])
        self.food.set_mass(float(self.food_spec["mass"]))
        self._food_rigid = None
        for c in self.food.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                lin_d = 8.0 if self.food_type == "onion_half" else 2.0
                ang_d = 12.0 if self.food_type == "onion_half" else 4.0
                c.set_linear_damping(lin_d)
                c.set_angular_damping(ang_d)
                self._food_rigid = c
                if self.food_type == "onion_half":
                    mat = sapien.physx.PhysxMaterial(
                        static_friction=2.0, dynamic_friction=1.6, restitution=0.0
                    )
                    for shape in c.get_collision_shapes():
                        shape.set_physical_material(mat)
        self.add_prohibit_area(self.food, padding=0.02)

        self._food_shapes = []
        for c in self.food.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._food_shapes = list(c.render_shapes)
        self._prime_food_surface()
        self._set_food_color(0.0)
        # Stove starts lit — expert only places food, then shuts the knob off.
        on_angle = -float(self.cook_intensity) * self.KNOB_MAX_ANGLE
        self._set_knob_angle(on_angle)
        # Hard-park the revolute at the lit angle so idle joint drift cannot
        # extinguish the burner before anyone touches the knob.
        self._set_knob_joint_angle(on_angle, hard=True)
        self._set_knob_articulation_qpos(on_angle)
        self._last_committed_knob_angle = float(on_angle)
        if self.food_type == "onion_half":
            # Re-seat with the same jitter (don't snap back to board center).
            self._seat_food_on_board(dx=food_dx, dy=food_dy)
            for _ in range(30):
                self.scene.step()

        # Knob is near center-right of the shifted stove → right arm.
        # Food/board sit further left → left arm for pick/place.
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self.food_arm = ArmTag("left" if float(self.board_xy[0]) < 0.0 else "right")

        # Decorative plate on the free side opposite the chopping board.
        self._spawn_decor_meat_plate(
            bz, rx, ry, float(hx), rng=rng, jitter=jitter,
            board_on_right=board_on_right,
            blockers=[
                range_blocker,
                pan_blocker,
                (np.array([board_x, board_y], dtype=float), board_half),
            ],
        )
        print(
            f"[cook_food] layout seed={self._layout_seed} "
            f"stove={np.round(self.range_xy, 3).tolist()} "
            f"pose={getattr(self, '_stove_pose_choice', 'fixed')} "
            f"burner={self.burner_name} handle_yaw={self.skillet_handle_yaw:.2f} "
            f"board={np.round(self.board_xy, 3)} "
            f"decor={np.round(getattr(self, 'decor_plate_xy', (0, 0)), 3)} "
            f"food={self.food_type} "
            f"doneness=[{self.target_doneness_range[0]:.2f},"
            f"{self.target_doneness_range[1]:.2f}] "
            f"window={self.success_window_sec:.1f}s",
            flush=True,
        )

    def _sample_board_xy(
        self,
        rng: np.random.RandomState,
        board_x0: float,
        board_y0: float,
        board_half: np.ndarray,
        blockers: list[tuple[np.ndarray, np.ndarray]],
        *,
        jitter: float,
    ) -> np.ndarray:
        """Place the chopping board clear of stove/pan."""
        board0 = np.array([board_x0, board_y0], dtype=float)
        if jitter <= 1e-9:
            return board0
        for _ in range(80):
            b = board0 + np.array(
                [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)],
                dtype=float,
            )
            b[0] = float(np.clip(b[0], -0.42, 0.42))
            b[1] = float(np.clip(b[1], -0.22, 0.18))
            if self._footprint_clear(b, board_half, blockers):
                return b
        return board0


    def _spawn_decor_meat_plate(
        self,
        bz: float,
        rx: float,
        ry: float,
        hx: float,
        rng: np.random.RandomState | None = None,
        jitter: float = 0.0,
        blockers: list | None = None,
        board_on_right: bool = False,
    ) -> None:
        """Static plate + two raw steaks on the free side opposite the board."""
        if rng is None:
            rng = self._layout_rng(303)
        deco_half = np.array([0.09, 0.09], dtype=float)
        if board_on_right:
            # Board occupies the right — park decor on the left of the stove.
            deco_x0 = rx - hx - 0.20
            x_lo, x_hi = -0.50, rx - hx - 0.10
        else:
            deco_x0 = rx + hx + 0.20
            x_lo, x_hi = rx + hx + 0.10, 0.50
        deco_y0 = ry - 0.08
        deco_xy = np.array([deco_x0, deco_y0], dtype=float)
        blks = list(blockers or [])
        j = float(max(0.0, jitter))
        if j > 1e-9:
            found = None
            for _ in range(60):
                cand = np.array(
                    [
                        deco_x0 + rng.uniform(-j * 1.5, j * 1.5),
                        deco_y0 + rng.uniform(-j, j),
                    ],
                    dtype=float,
                )
                cand[0] = float(np.clip(cand[0], x_lo, x_hi))
                cand[1] = float(np.clip(cand[1], -0.22, 0.18))
                if self._footprint_clear(cand, deco_half, blks):
                    found = cand
                    break
            if found is not None:
                deco_xy = found
        deco_x, deco_y = float(deco_xy[0]), float(deco_xy[1])
        deco_scale = 0.55
        plate = create_actor(
            self,
            pose=sapien.Pose([deco_x, deco_y, bz], [0.707, 0.707, 0.0, 0.0]),
            modelname="003_plate",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=deco_scale,
        )
        plate.set_name("decor_plate")
        if plate.config is None:
            plate.config = {"scale": [deco_scale] * 3}
        self.add_prohibit_area(plate, padding=0.02)
        try:
            plate_top = float(plate.get_functional_point(0)[2])
        except Exception:
            plate_top = bz + 0.016
        # Spaced on the plate so meshes do not intersect.
        for i, (dx, dy) in enumerate(((-0.032, -0.010), (0.032, 0.012))):
            steak = create_actor(
                self,
                pose=sapien.Pose(
                    [deco_x + dx, deco_y + dy, plate_top + 0.008],
                    list(self.FOOD_QPOS),
                ),
                modelname="200_steak",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=(0.55, 1.0, 0.55),
            )
            steak.set_name(f"decor_steak_{i}")
            for c in steak.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            s.material.set_base_color([1.00, 0.12, 0.09, 1.0])
                        except Exception:
                            pass
        self.decor_plate_xy = (deco_x, deco_y)

    def _clear_burner_ring(self) -> None:
        self._clear_stove_fire_ring()

    # Fire ring around the pan base (shared blue KitchenS flame).
    def _build_burner_ring(self, cx: float, cy: float, cz: float) -> None:
        """Blue fire halo just outside the pan bowl at the burner surface."""
        # Scale ring with pan_scale (cook_meat pan_scale=1.0 → ~8 cm radius).
        s = float(getattr(self, "pan_scale", self.PAN_SCALE_DEFAULT))
        self._build_stove_fire_ring(
            cx,
            cy,
            cz,
            0.080 * s,
            half_size=[0.010 * s, 0.005 * s, 0.003 * s],
        )

    @classmethod
    def _sample_cook_steps(cls, config: dict[str, Any]) -> int:
        jitter = float(config.get("cook_speed_jitter", cls.COOK_SPEED_JITTER_DEFAULT))
        jitter = float(np.clip(abs(jitter), 0.0, 0.95))
        nom = float(config.get("cook_steps", cls.COOK_STEPS_DEFAULT))
        nom = max(1.0, nom)
        return max(
            1,
            int(round(float(np.random.uniform(nom * (1 - jitter), nom * (1 + jitter))))),
        )

    def _sim_dt(self) -> float:
        scene = getattr(self, "scene", None)
        if scene is not None:
            try:
                dt = float(scene.get_timestep())
                if dt > 0.0:
                    return dt
            except Exception:
                pass
        return float(self.SIM_DT_DEFAULT)

    def _success_window_width(self, window_sec: float) -> float:
        """Doneness width equal to ``window_sec`` of cooking at current heat."""
        inten = max(0.05, float(getattr(self, "cook_intensity", 0.70)))
        steps = max(1.0, float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)))
        return float(window_sec) * inten / (steps * self._sim_dt())

    def _doneness_range_from_window(
        self, center: float, window_sec: float
    ) -> tuple[float, float]:
        """Inclusive [lo, hi] spanning ``window_sec``, clipped/shifted into [0, 1]."""
        width = float(self._success_window_width(window_sec))
        if width >= 1.0:
            return (0.0, 1.0)
        half = 0.5 * width
        lo = float(center) - half
        hi = float(center) + half
        if lo < 0.0:
            hi = min(1.0, hi - lo)
            lo = 0.0
        if hi > 1.0:
            lo = max(0.0, lo - (hi - 1.0))
            hi = 1.0
        return (lo, hi)

    # ----------------------------------------------------------- visuals / cook
    def _prime_food_surface(self) -> None:
        """Strip baked textures so doneness recoloring is visible (onion = white)."""
        if self.food_type != "onion_half":
            return
        self._set_food_color(0.0)

    def _set_food_color(self, doneness: float) -> None:
        d = float(np.clip(doneness, 0.0, 1.0))
        stops = self.color_stops
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                # Smoothstep between stops — onion browning looks less banded.
                if getattr(self, "food_type", "") == "onion_half":
                    t = t * t * (3.0 - 2.0 * t)
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        color = list(rgb) + [1.0]
        for s in self._food_shapes:
            try:
                mat = s.material
            except Exception:
                continue
            # Clear albedo texture every update — otherwise base_color is ignored.
            if getattr(self, "food_type", "") == "onion_half":
                try:
                    mat.set_base_color_texture(None)
                except Exception:
                    pass
                try:
                    mat.set_metallic(0.0)
                    mat.set_roughness(0.65)
                except Exception:
                    pass
            try:
                mat.set_base_color(color)
                mat.base_color = color
            except Exception:
                try:
                    mat.set_base_color(color)
                except Exception:
                    pass

    def _set_burner_visuals(self, intensity: float) -> None:
        inten = float(np.clip(intensity, 0.0, 1.0))
        # Blue ring only while the stove is on; fully hidden when off (no gray).
        self._set_stove_fire(inten > 0.02, intensity=inten)

    def _set_knob_angle(self, angle: float, *, drive_fire: bool = True) -> None:
        """Set knob angle in [−π/2, 0]; 0 = off, −π/2 = max fire (left).

        During an active grasp, the shared knob controller supplies the live
        wrist-coupled angle; explicit calls still update heat and visuals
        immediately.
        """
        angle = float(np.clip(angle, -self.KNOB_MAX_ANGLE, 0.0))
        self.knob_angle = angle
        if drive_fire:
            scaled_intensity = float(-angle / self.KNOB_MAX_ANGLE)
            if scaled_intensity > 0.0:
                self.fire_intensity = float(np.clip(scaled_intensity, 0.0, 1.0))
                self.stove_on = True
            else:
                self.fire_intensity = 0.0
                self.stove_on = False
            if self.stove_on:
                self.turned_on_once = True
            elif self.turned_on_once and self.max_doneness > 0.05:
                self.turned_off_after_cook = True
                # Freeze the cook score at shutoff (interactive + expert).
                if self._grasp_doneness is None:
                    self._grasp_doneness = float(self.doneness)

        # Explicit state changes and contact-validated wrist coupling keep the
        # articulated tick aligned with task state.
        self._set_knob_joint_angle(
            angle, hard=not bool(getattr(self, "_knob_grasp_active", False))
        )
        if drive_fire:
            self._set_burner_visuals(self.fire_intensity)

    def _lock_food_to_pan(self) -> None:
        """Mark food as resting in the pan after a confirmed drop.

        Does not teleport or weld — the meat stays at the physics drop pose.
        """
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._food_locked = True
        self._food_rel = None

    def _unlock_food(self) -> None:
        self._food_locked = False
        self._food_rel = None

    def _weld_food_to_ee(self, arm: ArmTag) -> None:
        """Hold onion fixed in the jaws while carrying (not while resting in the pan)."""
        if self.food is None or self._food_rigid is None:
            return
        try:
            self._food_rigid.set_disable_gravity(True)
            self._food_rigid.set_linear_velocity(np.zeros(3))
            self._food_rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        self._food_weld_offset = self._ee_pose(arm).inv() * self.food.get_pose()
        self._food_weld_arm = arm
        self._food_welded = True

    def _release_food_weld(self) -> None:
        self._food_welded = False
        self._food_weld_offset = None
        self._food_weld_arm = None
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_disable_gravity(False)
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _sync_food_to_ee(self) -> None:
        if not self._food_welded or self._food_weld_offset is None or self._food_weld_arm is None:
            return
        pose = self._ee_pose(self._food_weld_arm) * self._food_weld_offset
        self.food.actor.set_pose(pose)

    # ----------------------------------------------------------- per-step
    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "food", None):
            return

        # Skillet is spawned static — do not set_pose every tick (that kicks
        # resting food and looks like the onion slides on its own).
        self._sync_food_to_ee()

        # Knob grasp / fire intensity: KitchenS_base_task._update_stove_knob_control

        # Interactive play: detect pan drop without running the expert place script.
        if self._food_in_bowl(require_released=True):
            self._food_in_pan = True
        elif self._food_on_board() and not self._food_held():
            self._food_in_pan = False

        # Cook / timer advance only while the burner is lit and food rests in
        # the pan. Never freeze browning while the stove is still on — score
        # freezes solely when the knob kills the fire (_grasp_doneness).
        if (
            self.fire_intensity > 0.02
            and (self._food_in_pan or self._food_in_bowl(require_released=True))
            and not self._food_held()
        ):
            self.doneness = min(
                1.0,
                self.doneness + self.fire_intensity / max(1, self.cook_steps),
            )
            self.max_doneness = max(self.max_doneness, self.doneness)
            self._set_food_color(self.doneness)

        self._track_cook_metrics()

    def _idle_steps(self, n_steps: int, until=None) -> None:
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            if until is not None and until():
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    def _pause(self, n_steps: int = 10) -> None:
        """Short settle between phases (no onion-specific inflation)."""
        self._idle_steps(int(n_steps))

    def _idle_until_doneness(self, level: float, max_steps: int | None = None) -> None:
        """Cook with live fire until ``doneness`` reaches ``level``.

        Plan and replay use the same path so demos brown gradually and the
        pie timer keeps advancing — no flash color ramp, no early freeze.
        """
        inten = max(0.05, float(self.fire_intensity))
        if max_steps is None:
            max_steps = int(round(float(level) * self.cook_steps / inten)) + 40
        if getattr(self, "food_type", "") == "onion_half":
            max_steps = min(int(max_steps), 2400)
        else:
            # Keep meat/sausage demos under a sane wall-clock while still
            # visibly cooking for several seconds (save_freq≈15 → ~fps 16).
            max_steps = min(int(max_steps), 2200)
        self._idle_steps(
            max_steps,
            until=lambda: self.doneness >= float(level) or self.doneness >= 0.99,
        )

    # ----------------------------------------------------------- knob expert
    def _tcp_near_knob(self, tol: float = 0.045) -> bool:
        try:
            tcp = np.array(
                self.robot.get_right_tcp_pose()
                if str(self.arm) == "right"
                else self.robot.get_left_tcp_pose(),
                dtype=float,
            )[:3]
            return float(np.linalg.norm(tcp - np.asarray(self.knob_xyz))) < tol
        except Exception:
            return False

    def _knob_approach_offsets(self) -> tuple[tuple[float, float, float], ...]:
        path = tuple(self.KNOB_APPROACH_PATH)
        if float(getattr(self, "_knob_local_xy", self.KNOB_LOCAL_XY)[0]) < 0.0:
            path = tuple((-float(ox), float(oy), float(oz)) for ox, oy, oz in path)
        return path

    def _prestage_knob_arm(self) -> bool:
        """Hover over the knob during the cook wait (open jaws, no twist).

        One overhead waypoint only — a full approach here overcooks before
        shutoff. Returns True when the arm is staged so shutoff can skip the
        long lateral path.
        """
        if str(self.arm) == str(self.food_arm):
            return False
        arm = self.arm
        start = float(self.knob_angle)
        path = self._knob_approach_offsets()
        hover = path[-1] if path else (0.0, 0.0, 0.08)
        self._ignore_knob = True
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        self.move(self.move_to_pose(arm, self._knob_pose(hover, start)))
        if not self.plan_success:
            self.plan_success = True
            try:
                self.move(
                    self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="world")
                )
            except Exception:
                pass
            self.plan_success = True
            self.move(self.move_to_pose(arm, self._knob_pose(hover, start)))
        ok = bool(self.plan_success)
        self._ignore_knob = False
        self.plan_success = True
        self._knob_prestaged = bool(ok)
        return bool(ok)

    def _set_knob_to(
        self,
        target_angle: float,
        approach: bool = True,
        *,
        park_food_arm: bool = True,
        direct: bool = False,
    ) -> None:
        """Contact-driven continuous knob turn (shared KitchenS helper)."""
        end = float(np.clip(target_angle, -self.KNOB_MAX_ANGLE, 0.0))
        start = float(self.knob_angle)

        # Cooking stays live through any park + approach + twist; it only stops
        # when the physical knob extinguishes the burner.
        if park_food_arm:
            self._park_food_arm(self.food_arm)
        reached = self._turn_stove_knob(
            end,
            approach=approach,
            start_angle=start,
            after_idle=2 if (direct or not approach) else 6,
            commit_stove=None,
            retry_closer=True,
            direct=direct,
        )
        self._dbg(f"knob_grasp_near={self._tcp_near_knob()}")
        # Fire from contact angle only — no snap to the commanded target.
        self._set_knob_angle(reached)

    # ----------------------------------------------------------- place expert
    def _dbg(self, tag: str) -> None:
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_food] {tag}: plan_success={self.plan_success}", flush=True)

    def _food_held(self) -> bool:
        try:
            return len(self.get_gripper_actor_contact_position(self.food.get_name())) > 0
        except Exception:
            return False

    def _food_on_board(self, tol: float = 0.08) -> bool:
        p = np.asarray(self.food.get_pose().p, dtype=float)
        bxy = np.asarray(self.board_xy, dtype=float)
        return (
            float(np.linalg.norm(p[:2] - bxy)) < tol
            and float(p[2]) > (self.board_top_z - 0.02)
        )

    def _food_in_bowl(self, tol: float | None = None, *, require_released: bool = True) -> bool:
        """True when food rests in the skillet bowl (not hovering in the gripper)."""
        if require_released and self._food_held():
            return False
        if tol is None:
            tol = 0.10 * float(getattr(self, "pan_scale", self.PAN_SCALE_DEFAULT))
        bowl = np.asarray(self.skillet.get_functional_point(0), dtype=float)
        food = np.asarray(self.food.get_pose().p, dtype=float)
        xy_ok = float(np.linalg.norm(food[:2] - bowl[:2])) < float(tol)
        # Must sit near the bowl floor — a grasp held over the pan fails this z band.
        # Food meshes rest ~2–4 cm above the functional point on the
        # flush cooktop; keep the ceiling loose enough for those thicknesses.
        z_delta = float(food[2] - bowl[2])
        z_ok = -0.01 <= z_delta <= 0.045
        return bool(xy_ok and z_ok)

    def _wait_food_dropped_in_bowl(self, steps: int = 18) -> bool:
        """Physics settle after opening the gripper above the pan."""
        if getattr(self, "food_type", "") == "onion_half":
            steps = max(int(steps), 12)
        for _ in range(int(steps)):
            self._idle_steps(1)
            if self._food_in_bowl():
                return True
        return self._food_in_bowl()

    def _park_food_arm(self, arm: ArmTag) -> None:
        """Stow the food arm over its board station — never cross the centerline."""
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        try:
            self.move(self.move_by_displacement(arm_tag=arm, z=0.16, move_axis="arm"))
        except Exception:
            pass
        bx, by = getattr(self, "board_xy", self.BOARD_REL_XY)
        ee = np.array(self.get_arm_pose(str(arm)), dtype=float)
        # High hover over the board station, pulled toward the robot.
        park = [float(bx), float(by) - 0.08, float(ee[2]), *list(ee[3:7])]
        self.plan_success = True
        try:
            self.move(self.move_to_pose(arm, park))
        except Exception:
            self.plan_success = True
            dx = -0.10 if str(arm) == "left" else 0.10
            self.move(
                self.move_by_displacement(
                    arm_tag=arm, x=dx, y=-0.06, move_axis="world"
                )
            )
        self._idle_steps(2 if getattr(self, "food_type", "") == "onion_half" else 6)
        self.plan_success = True
        self.move(self.open_gripper(arm))

    def _park_knob_arm(self, arm: ArmTag) -> None:
        """Stow the knob arm on the right apron — avoid center ``back_to_origin``."""
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        try:
            self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        except Exception:
            pass
        if str(arm) == "right":
            rx, _ry = getattr(self, "range_xy", self.RANGE_REL_XY)
            kx = float(getattr(self, "knob_xy", (rx, 0.0))[0])
            ee = np.array(self.get_arm_pose(str(arm)), dtype=float)
            # Stay clear of the knob's vertical approach column (center stove
            # puts the knob near x≈0.16 — parking on that line breaks the
            # subsequent top-down descent).
            park_x = max(float(rx) + 0.22, kx + 0.14, 0.34)
            park = [park_x, -0.30, max(float(ee[2]), 1.05), *list(ee[3:7])]
            self.plan_success = True
            try:
                self.move(self.move_to_pose(arm, park))
            except Exception:
                self.plan_success = True
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, x=0.12, y=-0.08, move_axis="world"
                    )
                )
        else:
            self.plan_success = True
            try:
                self.move(self.back_to_origin(arm))
            except Exception:
                pass
        self._idle_steps(2 if getattr(self, "food_type", "") == "onion_half" else 6)

    def _retreat_food_arm(self, arm: ArmTag) -> None:
        """Open gripper and park on the left so the knob arm has the center clear."""
        self._park_food_arm(arm)

    def _seat_food_on_board(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Seat food on the chopping board (optional XY offset; onion rolls easily)."""
        if self.food is None or not hasattr(self, "board_xy"):
            return
        bx = float(self.board_xy[0]) + float(dx)
        by = float(self.board_xy[1]) + float(dy)
        z = float(self.board_top_z) + 0.012
        q = list(self.food.get_pose().q)
        self.food.actor.set_pose(sapien.Pose([bx, by, z], q))
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass


    def _fallback_food_grasp_poses(
        self, arm_tag: ArmTag, pre_dis: float = 0.10
    ) -> tuple[list[float], list[float]]:
        """Top-down COM grasp when authored contact points fail IK (onion half)."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC

        p = np.asarray(self.food.get_pose().p, dtype=float)
        key = (
            "top_down_little_left"
            if str(arm_tag) == "left"
            else "top_down_little_right"
        )
        q = list(GRASP_DIRECTION_DIC[key])
        # Flat cross-section slice (~6 mm): pinch just above the top face.
        top_z = float(p[2]) + 0.004
        try:
            ext = np.asarray(self.food.config.get("extents", [0.1, 0.006, 0.1]), dtype=float)
            sm = float(np.mean(self.food.config.get("scale", [1.0, 1.0, 1.0])))
            thick = float(np.min(ext * sm))
            top_z = float(p[2]) + 0.5 * thick + 0.001
        except Exception:
            pass
        pre = [float(p[0]), float(p[1]), top_z + float(pre_dis), *q]
        grasp = [float(p[0]), float(p[1]), top_z, *q]
        return pre, grasp

    def _safe_grasp_actor(self, actor: Any, arm_tag: ArmTag, **kwargs: Any):
        """Contact-point grasp — fingers close around the mesh (not through it)."""
        # Leave a jaw gap so fingertips wrap the food instead of crushing into it.
        kwargs.setdefault("gripper_pos", 0.35)
        if getattr(self, "food_type", "") == "onion_half":
            # Match ~4.7 cm diameter — too-tight jaws eject the disc.
            kwargs.setdefault("gripper_pos", 0.12)
            kwargs.setdefault("contact_point_id", 0)
        pre_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=float(kwargs.get("pre_grasp_dis", 0.1)),
            target_dis=float(kwargs.get("grasp_dis", 0.0)),
            contact_point_id=kwargs.get("contact_point_id"),
        )
        if pre_pose is None or grasp_pose is None:
            if getattr(self, "food_type", "") == "onion_half":
                pre_pose, grasp_pose = self._fallback_food_grasp_poses(
                    arm_tag, float(kwargs.get("pre_grasp_dis", 0.1))
                )
            else:
                raise UnStableError("cook_food: no grasp pose — skip")
        gripper_pos = float(kwargs.get("gripper_pos", 0.35))
        if pre_pose == grasp_pose:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_pose),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]
        return arm_tag, [
            Action(arm_tag, "move", target_pose=pre_pose),
            Action(
                arm_tag,
                "move",
                target_pose=grasp_pose,
                constraint_pose=[1, 1, 1, 0, 0, 0],
            ),
            Action(arm_tag, "close", target_gripper_pos=gripper_pos),
        ]

    def _pan_place_target(self) -> list[float]:
        """Hover/release target above the skillet bowl center (physics drop)."""
        bowl = np.asarray(self.skillet.get_functional_point(0), dtype=float)
        z_off = 0.010 if getattr(self, "food_type", "") == "onion_half" else 0.012
        # Default place_dx/dy are 0 — drop on the bowl center.
        return [
            float(bowl[0]) + float(getattr(self, "place_dx", 0.0)),
            float(bowl[1]) + float(getattr(self, "place_dy", 0.0)),
            float(bowl[2]) + z_off,
            *self.FOOD_QPOS,
        ]

    def _release_food_over_bowl(self, arm: ArmTag) -> bool:
        """Move over the bowl, open fully, retreat, then verify a real drop."""
        if not self._food_held():
            return self._food_in_bowl()
        target = self._pan_place_target()
        bowl_xy = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)

        # Hover → descend → open → lift away (strict drop, no cook-while-held).
        hover = list(target)
        hover[2] = float(target[2]) + 0.06
        stages = [hover, target]
        for food_xyzq in stages:
            if not self._food_held():
                break
            ee = self._ee_pose(arm)
            food = self.food.get_pose()
            food_in_ee = ee.inv() * food
            food_tgt = sapien.Pose(list(food_xyzq[:3]), list(food_xyzq[3:7]))
            ee_tgt = food_tgt * food_in_ee.inv()
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm,
                    [
                        float(ee_tgt.p[0]),
                        float(ee_tgt.p[1]),
                        float(ee_tgt.p[2]),
                        *list(ee_tgt.q),
                    ],
                )
            )

        if not self._food_held():
            return self._wait_food_dropped_in_bowl()

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        # Require food near bowl center before opening (center drop, not rim).
        if float(np.linalg.norm(food_xy - bowl_xy)) > 0.05:
            return False

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self._idle_steps(4)
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        self._idle_steps(4)
        self.plan_success = True
        self.move(self.open_gripper(arm))
        return self._wait_food_dropped_in_bowl()

    def _drop_into_pan(self, arm: ArmTag) -> bool:
        """Carry food over the bowl, release, retreat, verify it landed in the pan."""
        target = self._pan_place_target()
        # Onion: hover low over bowl center, open, lift straight up (no shove).
        hover = 0.04 if self.food_type == "onion_half" else 0.06
        rtol = 0.04 if self.food_type == "onion_half" else 0.05
        if self._carry_held_food_to(arm, target, hover_z=hover, release_tol=rtol):
            if self.food_type == "onion_half":
                # Extra open + pure-Z retreat so the disc stays where it landed.
                self.plan_success = True
                self.move(self.open_gripper(arm))
                self._pause(4)
                self.plan_success = True
                try:
                    self.move(
                        self.move_by_displacement(arm_tag=arm, z=0.16, move_axis="arm")
                    )
                except Exception:
                    pass
                self._pause(4)
            if self._wait_food_dropped_in_bowl():
                return True
        if self._release_food_over_bowl(arm):
            return True
        # Classic place_actor fallback (still no weld).
        if self._food_held():
            self.plan_success = True
            self.move(
                self.place_actor(
                    self.food,
                    target_pose=target,
                    arm_tag=arm,
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.008,
                    is_open=True,
                )
            )
            self.plan_success = True
            self.move(self.open_gripper(arm))
            self.plan_success = True
            try:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
            except Exception:
                pass
            self.plan_success = True
            self.move(self.open_gripper(arm))
            if self._wait_food_dropped_in_bowl():
                return True
        return self._food_in_bowl()

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _carry_held_food_to(
        self,
        arm: ArmTag,
        target_pose: list[float],
        *,
        hover_z: float = 0.07,
        release_tol: float = 0.09,
        release: bool = True,
    ) -> bool:
        """Move the EE so the physically grasped food arrives at ``target_pose``.

        Uses the live grasp offset only for IK targets (no weld / set_pose).
        Opens the gripper only if the food is near the target; otherwise False.
        """
        if not self._food_held():
            return False
        food = self.food.get_pose()
        quat = list(target_pose[3:7]) if len(target_pose) >= 7 else list(food.q)
        tgt_xy = np.asarray(target_pose[:2], dtype=float)

        # Mid-air waypoint first (keeps path short / avoids CuRobo spinouts).
        # Onion demos skip the mid hop to stay under ~60s video.
        stages: list[list[float]] = []
        if getattr(self, "food_type", "") != "onion_half":
            food_now = np.asarray(self.food.get_pose().p, dtype=float)
            stages.append(
                [
                    0.5 * (food_now[0] + float(target_pose[0])),
                    0.5 * (food_now[1] + float(target_pose[1])),
                    max(float(food_now[2]), float(target_pose[2])) + 0.06,
                    *quat,
                ]
            )
        for z_off in (hover_z, 0.0):
            stages.append(
                [
                    float(target_pose[0]),
                    float(target_pose[1]),
                    float(target_pose[2]) + z_off,
                    *quat,
                ]
            )

        for food_xyzq in stages:
            if not self._food_held():
                return False
            # Refresh grasp offset each stage (object can slip in the jaws).
            ee = self._ee_pose(arm)
            food = self.food.get_pose()
            food_in_ee = ee.inv() * food
            food_tgt = sapien.Pose(list(food_xyzq[:3]), list(food_xyzq[3:7]))
            ee_tgt = food_tgt * food_in_ee.inv()
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm,
                    [
                        float(ee_tgt.p[0]),
                        float(ee_tgt.p[1]),
                        float(ee_tgt.p[2]),
                        *list(ee_tgt.q),
                    ],
                )
            )
            if not self.plan_success:
                self.plan_success = True
                return False
            if getattr(self, "food_type", "") != "onion_half":
                self._pause(6)

        if not release:
            return True

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        if float(np.linalg.norm(food_xy - tgt_xy)) > release_tol:
            return False
        if not self._food_held():
            return False

        self._release_food_weld()
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self._pause(4 if getattr(self, "food_type", "") == "onion_half" else 10)
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        self._pause(4 if getattr(self, "food_type", "") == "onion_half" else 12)
        # Verify against the intended target (pan bowl) — never hardcode a fixed XY.
        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        return float(np.linalg.norm(food_xy - tgt_xy)) < float(release_tol) * 1.6

    def _place_held_food(
        self,
        target_pose: list[float],
        arm: ArmTag,
        *,
        pre_dis: float = 0.10,
        dis: float = 0.02,
    ) -> None:
        """Prefer grasp-relative carry; fall back to place_actor if needed."""
        if self._carry_held_food_to(arm, target_pose):
            return
        if not self._food_held():
            return
        # Fallback: classic place_actor (still no weld).
        self.plan_success = True
        self.move(
            self.place_actor(
                self.food,
                target_pose=target_pose,
                arm_tag=arm,
                constrain="free",
                pre_dis=pre_dis,
                dis=dis,
                is_open=True,
            )
        )
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        self._idle_steps(8)



    def _lift_held_food(self, arm: ArmTag, total_z: float = 0.12) -> None:
        """Incremental lift with re-close so layered onion meshes stay pinched."""
        if not self._food_held():
            return
        steps = max(1, int(round(float(total_z) / 0.03)))
        dz = float(total_z) / float(steps)
        # ~4.7 cm onion needs a wider jaw than the old thin slice (0.04 crushed it).
        pinch = 0.12 if getattr(self, "food_type", "") == "onion_half" else 0.14
        onion = getattr(self, "food_type", "") == "onion_half"
        for _ in range(steps):
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=pinch))
            self._pause(2 if onion else 6)
            if not self._food_held():
                return
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=dz, move_axis="arm"))
            self._pause(2 if onion else 8)

    def _place_food_in_pan(self) -> None:
        """Board → pan: grasp, open over the bowl, physics drop, retreat.

        Cooking must not start until the food is released and the food arm is clear.
        """
        arm = self.food_arm
        if not self._food_on_board():
            raise UnStableError("cook_food: food not on chopping board — skip")

        # --- 1) Grasp from board (onion: jaw width ~ disc diameter) ---
        grasped = False
        gpos_board = (0.14, 0.12, 0.10) if self.food_type == "onion_half" else (0.10, 0.06, 0.04)
        for gpos in gpos_board:
            self.plan_success = True
            self.move(self.open_gripper(arm))
            self.move(
                self._safe_grasp_actor(
                    self.food,
                    arm_tag=arm,
                    pre_grasp_dis=0.10,
                    gripper_pos=gpos,
                )
            )
            self._pause(6 if self.food_type == "onion_half" else 12)
            if self._food_held():
                grasped = True
                break
        if not grasped:
            raise UnStableError("cook_food: failed to grasp food from board — skip")
        self._lift_held_food(arm, total_z=0.10 if self.food_type == "onion_half" else 0.12)
        if not self._food_held():
            raise UnStableError("cook_food: dropped food while lifting from board — skip")
        self._dbg("grasp_from_board")

        # --- 2) Carry / place above bowl, release (only when close) ---
        # Right-front bowl sits closer to the knob column — allow fingers in.
        if getattr(self, "burner_name", "") == "right_front":
            self._ignore_skillet_robot_collision()
        self._drop_into_pan(arm)

        for attempt in range(3):
            if self._food_in_bowl() and not self._food_held():
                break
            self.plan_success = True
            self.move(self.open_gripper(arm))
            try:
                self.move(
                    self._safe_grasp_actor(
                        self.food, arm_tag=arm, pre_grasp_dis=0.08, gripper_pos=0.12
                    )
                )
            except UnStableError:
                continue
            self._idle_steps(4)
            if not self._food_held():
                continue
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
            bowl = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)
            food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
            err = bowl - food_xy
            # Stronger correction on retries — left arm consistently undershoots −X.
            self.place_dx = float(self.place_dx + 0.5 * err[0])
            self.place_dy = float(self.place_dy + 0.5 * err[1])
            self._drop_into_pan(arm)

        if self._food_held():
            # Last chance: force open + lift so we never cook in-hand.
            self.plan_success = True
            self.move(self.open_gripper(arm))
            self._idle_steps(8)
            self.plan_success = True
            try:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.14, move_axis="arm"))
            except Exception:
                pass
            self._idle_steps(6)

        if self._food_held():
            raise UnStableError("cook_food: still holding food after pan place — skip")
        if not self._food_in_bowl():
            raise UnStableError("cook_food: food not in pan after place — skip")

        if self.food_type == "onion_half":
            self._wait_food_dropped_in_bowl(steps=12)
            if not self._food_in_bowl():
                raise UnStableError("cook_food: onion left pan after drop — skip")
            # Pure vertical retreat so the fingers don't shove the disc sideways.
            self.plan_success = True
            try:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.14, move_axis="arm"))
            except Exception:
                pass
            self._pause(4)

        # --- 3) Confirm physical drop, then retreat (no teleport / soft-seat) ---
        if bool(self._cfg.get("lock_food_in_pan", True)):
            self._lock_food_to_pan()
        self._food_in_pan = True
        self._retreat_food_arm(arm)
        self._pause(4 if self.food_type == "onion_half" else 10)
        if self._food_held():
            raise UnStableError("cook_food: re-grasped food while retreating — skip")
        if not self._food_in_bowl(require_released=True):
            raise UnStableError("cook_food: food left pan after retreat — skip")
        self.plan_success = True
        self._dbg("food_in_pan")

    def _set_collision_ignore(self, entities: list[Any], ignore_bit: int, ignore_id: int) -> None:
        for ent in entities:
            if ent is None:
                continue
            try:
                shapes = []
                target = getattr(ent, "actor", ent)
                if hasattr(target, "get_collision_shapes"):
                    shapes = list(target.get_collision_shapes())
                elif hasattr(target, "get_components"):
                    for c in target.get_components():
                        if isinstance(
                            c,
                            (
                                sapien.physx.PhysxRigidDynamicComponent,
                                sapien.physx.PhysxRigidStaticComponent,
                            ),
                        ):
                            shapes.extend(c.get_collision_shapes())
                for shape in shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    shape.set_collision_groups(
                        [
                            int(g0),
                            int(g1),
                            int(g2) | int(ignore_bit),
                            (int(g3) & 0xFFFF0000) | int(ignore_id),
                        ]
                    )
            except Exception:
                pass

    def _ignore_skillet_robot_collision(self) -> None:
        """Let fingers enter the bowl for a top-down pinch (food still collides)."""
        bit, gid = 1 << 16, 16
        ents: list[Any] = [getattr(self.skillet, "actor", self.skillet)]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        self._set_collision_ignore(ents, bit, gid)

    def _pinch_food_topdown(self, arm: ArmTag, gripper_pos: float = 0.15) -> bool:
        """Top-down pinch into the pan bowl — fingers close around the food mesh."""
        p = np.asarray(self.food.get_pose().p, dtype=float)
        q = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=float)
        mat = t3d.quaternions.quat2mat(q)
        axis = mat[:, 0]
        pre = (p - 0.14 * axis).tolist() + q.tolist()
        grasp = (p - 0.01 * axis).tolist() + q.tolist()
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        self.move(self.move_to_pose(arm, pre))
        if not self.plan_success:
            self.plan_success = True
            return False
        self.plan_success = True
        self.move(self.move_to_pose(arm, grasp))
        # Close firmly around the mesh (not crushed to 0 through the volume).
        self.plan_success = True
        self.move(self.close_gripper(arm, pos=float(gripper_pos)))
        self._idle_steps(10)
        return self._food_held()


    # ----------------------------------------------------------- policy
    def play_once(self) -> dict[str, Any]:
        # Stove is already on. Sequence: board → drop in pan → wait → shut off.
        # Onion demos must stay under ~60s video — keep pauses short.
        onion = self.food_type == "onion_half"
        self._pause(4 if onion else 20)
        if onion:
            self._seat_food_on_board()
            self._pause(2)
        if not self._food_on_board():
            raise UnStableError("cook_food: food left the board before start — skip")

        # Keep the knob arm clear while the food arm works (skip if same arm).
        if not onion and str(self.arm) != str(self.food_arm):
            self._park_knob_arm(self.arm)
            self._pause(10)
        else:
            self.plan_success = True
            self.move(self.open_gripper(self.arm))
            if str(self.arm) == str(self.food_arm):
                self.plan_success = True
                self.move(self.open_gripper(self.food_arm))

        # 1) Board → pan. Cooking starts as soon as food rests in the bowl
        # (gated by fire + in-pan + not held — no scripted freeze).
        self._place_food_in_pan()
        if not self._food_in_pan or self._food_held() or not self._food_in_bowl():
            raise UnStableError("cook_food: refuse to cook — food not seated in pan")
        # Minimal settle — cooking is already live; long pauses overshoot the band.
        self._pause(2)

        # 2) Cook on the pre-lit burner, then twist the knob off.
        # Browns + timer keep advancing whenever the stove is lit — never freeze
        # while the burner is on. Walk the knob arm to the standoff during the
        # cook wait so shutoff is a short close+twist (not a long approach).
        lead = max(0.0, float(getattr(self, "shutoff_lead", 0.58)))
        shut_at = max(0.01, float(self.target_doneness) - lead)
        prestaged = self._prestage_knob_arm()
        self._idle_until_doneness(shut_at)
        self._set_knob_to(
            0.0,
            approach=not prestaged,
            park_food_arm=False,
            direct=False,
        )
        self._dbg("stove_off")
        if (
            bool(self.stove_on)
            or float(self.fire_intensity) > 0.02
            or float(self.knob_angle) < -0.05
            or self._grasp_doneness is None
        ):
            raise UnStableError("cook_food: stove not shut off by knob — skip")
        self._cook_phase_done = True
        if not self._doneness_in_target_range(float(self._grasp_doneness)):
            raise UnStableError(
                f"cook_food: doneness {self._grasp_doneness:.2f} outside "
                f"{self.target_doneness_range} — skip"
            )
        # Cook phase complete — food stays in the pan (no plate transfer).
        self._placed = True
        if onion:
            self.plan_success = True
            self.move(self.open_gripper(self.arm))
        elif str(self.arm) != str(self.food_arm):
            self._park_knob_arm(self.arm)
            self._pause(14)
        else:
            self.plan_success = True
            self.move(self.open_gripper(self.arm))
            self._pause(6)

        self.info["info"] = {
            "{A}": f"{self.food_spec['modelname']}/base{self.food_spec['model_id']}",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{D}": "cooking_range",
            "{E}": "stove_knob",
            "{F}": self.food_type,
            "{G}": "104_board/base0",
            "{a}": str(self.food_arm),
        }
        return self.info

    # ----------------------------------------------------------- success / obs
    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self) -> None:
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_on_step = None      # burner first lit
        self._metric_band_step = None    # doneness first entered the success band
        self._metric_off_step = None     # burner switched off (decisive event)
        self._metric_off_doneness = None # doneness frozen at that instant
        self._metric_was_on = False

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_cook_metrics(self) -> None:
        """Latch the lit / in-band / shut-off edges.

        The shutoff edge is read here rather than inside ``_apply_knob_angle`` because
        interactive play drives the knob through the shared wrist-coupled controller,
        which reaches the same state by a different path.
        """
        try:
            on = bool(getattr(self, "stove_on", False)) and self.fire_intensity > 0.02
            if on and self._metric_on_step is None:
                self._metric_on_step = self._metric_step()
            lo, _hi = self.target_doneness_range
            if self._metric_band_step is None and float(self.doneness) >= float(lo):
                self._metric_band_step = self._metric_step()
            if self._metric_was_on and not on and self._metric_off_step is None:
                self._metric_off_step = self._metric_step()
                self._metric_off_doneness = float(
                    self._grasp_doneness if self._grasp_doneness is not None
                    else self.doneness)
            self._metric_was_on = on
        except Exception:
            pass

    def _compute_metrics(self) -> dict:
        """Human-experiment extras.

        extra1 `shutoff_latency_steps` — steps from the doneness first entering the
        success band until the burner is switched off. The food keeps browning the
        whole time, so every step here is spent burning through the band.
        extra2 `doneness_error_norm` — |doneness at shutoff - band center| / band
        half-width. LOWER is better; <= 1.0 means the cook landed inside the band.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_band_step, self._metric_off_step
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["shutoff_latency_steps"] = lat
        out["shutoff_latency_s"] = None if lat is None else round(float(lat) * dt, 4)
        cook = (None if (self._metric_on_step is None or a is None)
                else max(int(a) - int(self._metric_on_step), 0))
        out["cook_latency_steps"] = cook

        d = self._metric_off_doneness
        try:
            lo, hi = map(float, self.target_doneness_range)
            half = max(0.5 * (hi - lo), 1e-6)
            center = 0.5 * (lo + hi)
            out["doneness_error_norm"] = (
                None if d is None else round(abs(float(d) - center) / half, 4))
            out["target_doneness_range"] = [round(lo, 4), round(hi, 4)]
        except Exception:
            out["doneness_error_norm"] = None
            out["target_doneness_range"] = None
        out["shutoff_doneness"] = None if d is None else round(float(d), 4)
        try:
            out["max_doneness"] = round(float(self.max_doneness), 4)
        except Exception:
            out["max_doneness"] = None
        return out

    def _doneness_in_target_range(self, doneness: float) -> bool:
        lo, hi = self.target_doneness_range
        return float(lo) <= float(doneness) <= float(hi)

    def check_success(self) -> bool:
        """Success: food in the target band, scored only after the stove is off."""
        # Do not score while the burner is still lit / knob is still open.
        if (
            bool(getattr(self, "stove_on", False))
            or self.fire_intensity > 0.02
            or self.knob_angle < -0.05
        ):
            return False
        if not self.turned_on_once or not self.turned_off_after_cook:
            return False
        score = (
            float(self._grasp_doneness)
            if self._grasp_doneness is not None
            else float(self.doneness)
        )
        if not self._doneness_in_target_range(score):
            return False
        if self._food_held():
            return False
        # Food remains in the pan — no plating step.
        return bool(
            self._food_in_pan
            or self._food_in_bowl(tol=0.10, require_released=True)
        )

    def _scored_doneness(self) -> float:
        if self._grasp_doneness is not None:
            return float(self._grasp_doneness)
        return float(getattr(self, "doneness", 0.0))

    def _doneness_to_cook_seconds(self, delta_doneness: float) -> float:
        """Convert a doneness delta to seconds of cooking at current heat."""
        inten = max(0.05, float(getattr(self, "cook_intensity", 0.70)))
        steps = max(1.0, float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)))
        rate = inten / (steps * self._sim_dt())  # doneness per second
        if rate <= 1e-12:
            return 0.0
        return float(delta_doneness) / rate

    def get_score(self) -> float:
        """Partial score for mild overcook past the doneness window.

        Scored only after stove shutoff. Full success → 1.0. Additional seconds
        past the upper edge of the success / doneness band:
        ``(0,1]`` → 0.75, ``(1,2]`` → 0.5, ``(2,3]`` → 0.25.
        Undercook or larger overshoot → 0.
        """
        if (
            bool(getattr(self, "stove_on", False))
            or float(getattr(self, "fire_intensity", 0.0)) > 0.02
            or float(getattr(self, "knob_angle", 0.0)) < -0.05
        ):
            return 0.0
        if not bool(getattr(self, "turned_on_once", False)):
            return 0.0
        if not bool(getattr(self, "turned_off_after_cook", False)):
            return 0.0
        if self._food_held():
            return 0.0
        if not (
            bool(getattr(self, "_food_in_pan", False))
            or self._food_in_bowl(tol=0.10, require_released=True)
        ):
            return 0.0
        if self.check_success():
            return 1.0
        doneness = self._scored_doneness()
        _, hi = getattr(self, "target_doneness_range", (0.45, 0.55))
        hi = float(hi)
        if doneness <= hi + 1e-9:
            return 0.0  # undercook / still inside or below the doneness band
        past_sec = self._doneness_to_cook_seconds(doneness - hi)
        return float(
            score_open_closed_intervals(past_sec, self.PARTIAL_OVERCOOK_BANDS)
        )

    def get_obs(self) -> dict[str, Any]:
        obs = super().get_obs()
        obs["cooking"] = {
            "food_type": str(getattr(self, "food_type", "")),
            "doneness": float(getattr(self, "doneness", 0.0)),
            "max_doneness": float(getattr(self, "max_doneness", 0.0)),
            "grasp_doneness": (
                None
                if self._grasp_doneness is None
                else float(self._grasp_doneness)
            ),
            "target_doneness": float(getattr(self, "target_doneness", 0.5)),
            "target_doneness_range": list(
                getattr(self, "target_doneness_range", (0.45, 0.55))
            ),
            "success_window_sec": float(
                getattr(self, "success_window_sec", self.SUCCESS_WINDOW_SEC_DEFAULT)
            ),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
            "knob_angle": float(getattr(self, "knob_angle", 0.0)),
            "fire_intensity": float(getattr(self, "fire_intensity", 0.0)),
            "stove_on": bool(getattr(self, "stove_on", False)),
            "placed": bool(getattr(self, "_placed", False)),
            "burner": str(getattr(self, "burner_name", "")),
            "partial_score": float(self.get_score()),
        }
        return obs
