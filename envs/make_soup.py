"""Make soup: pour chopping-board vegetables into a pot of water on a lit stove.

KitchenS scene with a cooking range. The burner starts on (fire + knob). A chopping
board holds a random subset of carrot, sideways broccoli and mushroom, and white
onion half (each roughly the old 2.4 cm cube footprint). A pot of water sits on
the lit burner.
The robot lifts the board, carries it roughly level over the pot, and tips carefully
so the pieces fall in under physics (tilting too early / too far drops them onto the
table). Success requires every piece in the pot and none on the table — no stove-turn
step.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.physx
import sapien.render
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.actor_utils import Actor
from .utils.create_actor import create_box, create_sphere, create_visual_box, UnStableError


class make_soup(KitchenS_base_task):
    """Pour board vegetables into a pot of water on an already-lit stove."""

    EE_TO_TCP: ClassVar[float] = 0.12
    KNOB_CONTACT_RADIUS_DEFAULT: ClassVar[float] = 0.06
    KNOB_APPROACH_PATH: ClassVar[tuple] = KitchenS_base_task.TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.012
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    FRONT_BURNERS: ClassVar[tuple[str, ...]] = ("left_front", "right_front")
    # Flush cooktop near the back of the apron; discrete slot when randomized.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.18, 0.14)
    # Stove slots: current apron pose, or back-left / back-right corner.
    RANGE_SLOTS: ClassVar[dict[str, tuple[float, float]]] = {
        "current": (0.18, 0.14),
        "top_left": (-0.18, 0.16),
        "top_right": (0.30, 0.16),
    }
    RANGE_SCALE_MULT: ClassVar[float] = 1.0
    # Legacy continuous X slide (used only when randomize_layout=false).
    RANGE_X_RANGE: ClassVar[tuple[float, float]] = (-0.10, 0.30)
    # Board x near zero → either arm may grasp (both grippers reach).
    BOARD_CENTER_X_TOL: ClassVar[float] = 0.06
    LAYOUT_MARGIN: ClassVar[float] = 0.03
    # Original thick-walled soup pot, sized to the boil_milk saucepan
    # (same height, matching inner mouth once the 5 mm wall is accounted for).
    POT_RADIUS: ClassVar[float] = 0.06786  # prior 0.0754 × 0.9
    POT_HEIGHT: ClassVar[float] = 0.0735
    POT_WALL: ClassVar[float] = 0.005
    GLASS_SCALE: ClassVar[float] = 0.585  # prior 0.45 × 1.3
    # Approximate apron footprints for non-overlap layout sampling.
    PLATE_HALF_XY: ClassVar[tuple[float, float]] = (0.075, 0.075)
    WINE_HALF_XY: ClassVar[tuple[float, float]] = (0.035, 0.035)
    GLASS_HALF_XY: ClassVar[tuple[float, float]] = (0.030, 0.030)

    BOARD_HALF: ClassVar[tuple[float, float, float]] = (0.095, 0.052, 0.010)  # y prior 0.065 × 0.8
    # Grasping block on the robot-facing (−Y) end of the board.
    HANDLE_HALF: ClassVar[tuple[float, float, float]] = (0.022, 0.024, 0.022)
    BOARD_COLOR: ClassVar[tuple[float, float, float]] = (0.55, 0.38, 0.22)
    HANDLE_COLOR: ClassVar[tuple[float, float, float]] = (0.72, 0.55, 0.28)
    CUBE_HALF: ClassVar[float] = 0.012  # prior produce cube half-extent (2.4 cm)
    TOMATO_RADIUS: ClassVar[float] = 0.012  # matched to cube half
    GRASP_TCP_TOL: ClassVar[float] = 0.045
    # How many distinct produce pieces to put on the board (inclusive).
    N_VEG_MIN: ClassVar[int] = 2
    N_VEG_MAX: ClassVar[int] = 4
    # Gripper vals are normalized (0 ≈ closed, 1 ≈ open). Used only to detect
    # a real pinch on the handle — never to weld / latch the board to the EE.
    BOARD_ATTACH_GRIPPER_MAX: ClassVar[float] = 0.55
    BOARD_RELEASE_GRIPPER_MIN: ClassVar[float] = 0.70
    BOARD_MASS: ClassVar[float] = 0.18
    HANDLE_STATIC_FRICTION: ClassVar[float] = 3.0
    HANDLE_DYNAMIC_FRICTION: ClassVar[float] = 2.5
    # Board produce (Kenney CC0 + cook_food onion). Sizes are post-scale_mult
    # world half-extents after ``orient`` (SIDE maps model +Y → board +X).
    # footprint_half = (half_x, half_y) for AABB packing; half_h seats on the board.
    VEG_MESHES: ClassVar[tuple[dict[str, Any], ...]] = (
        {
            "name": "carrot",
            "modelname": "271_carrot",
            "scale_mult": 1.30,
            "footprint_half": (0.022, 0.010),
            "half_h": 0.010,
            "orient": "side",
        },
        {
            "name": "broccoli",
            "modelname": "273_broccoli",
            "scale_mult": 1.105,
            "footprint_half": (0.014, 0.013),
            "half_h": 0.012,
            "orient": "side",
        },
        {
            "name": "mushroom",
            "modelname": "272_mushroom",
            "scale_mult": 1.17,
            "footprint_half": (0.013, 0.013),
            "half_h": 0.011,
            "orient": "side",
        },
        {
            "name": "onion",
            "modelname": "270_onion_half",
            "scale_mult": 0.50,  # Ø ~2.3 cm (stock is Ø4.7 cm)
            "footprint_half": (0.012, 0.012),
            "half_h": 0.003,
            "orient": "upright",
            "color": (1.0, 1.0, 1.0),  # white onion
        },
    )
    TOMATO_COLOR: ClassVar[tuple[float, float, float]] = (0.90, 0.10, 0.08)
    DECOR_QPOS: ClassVar[list[float]] = [0.70710678, 0.70710678, 0.0, 0.0]
    # Lie on the board: upright (Y→Z) tipped 90° about world Y (stem along table).
    SIDE_QPOS: ClassVar[list[float]] = [0.5, 0.5, 0.5, -0.5]

    # Produce is always dynamic. Level carry holds via board/veg friction;
    # tipping past ~arctan(μ) slides pieces into the pot.
    TILT_HOLD_DOT: ClassVar[float] = 0.82
    # Accidental spill during carry (top-down grasp wobble is ~0.87–0.90): only
    # free-fall if tipped nearly sideways (~60°).
    TILT_SPILL_DOT: ClassVar[float] = 0.50
    # Wrist roll for the pour: 40° is still past the produce's friction angle
    # (board μ≈0.82 → arctan ≈ 39°).
    POUR_TIP_RAD: ClassVar[float] = float(np.deg2rad(40.0))
    # Mass + matching solid-sphere inertia (see `_apply_veg_mass_properties`).
    # ``Actor.set_mass`` alone leaves the default ~10 g inertia, so tiny produce
    # contact-explodes / tunnels under the board.
    VEG_MESH_MASS: ClassVar[float] = 0.025
    VEG_SPHERE_MASS: ClassVar[float] = 0.030
    # Board deck: grippier so produce does not skate on a level carry; still
    # slides under a 40° tip (μ≈0.82 → arctan ≈ 39°).
    BOARD_STATIC_FRICTION: ClassVar[float] = 0.82
    BOARD_DYNAMIC_FRICTION: ClassVar[float] = 0.68
    VEG_STATIC_FRICTION: ClassVar[float] = 0.70
    VEG_DYNAMIC_FRICTION: ClassVar[float] = 0.55
    VEG_LINEAR_DAMPING: ClassVar[float] = 2.5
    VEG_ANGULAR_DAMPING: ClassVar[float] = 4.0
    WATER_LEVEL_DEFAULT: ClassVar[float] = 0.45
    BOARD_IGNORE_BIT: ClassVar[int] = 1 << 21
    BOARD_IGNORE_ID: ClassVar[int] = 0xB0A5

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("make_soup", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        self.omit_sink = True
        self.clear_sink_and_range = False
        self.range_scale_mult = float(
            self._cfg.get("range_scale_mult", self.RANGE_SCALE_MULT)
        )
        # Seed-stable stove slot (current / top-left / top-right).
        seed = int(kwags.get("seed", 0) or 0)
        self._layout_seed = seed
        self._range_slot = "current"
        self.range_position_override = self._sample_range_xy(
            self._cfg, np.random.RandomState(seed + 17)
        )
        # Random wall / counter / floor textures for this task.
        if bool(self._cfg.get("randomize_background", True)):
            kwags["random_background"] = True
            kwags["clean_background_rate"] = float(
                self._cfg.get("clean_background_rate", 0.0)
            )
        if "table_xy_bias" not in kwags and "table_xy_bias" in self._cfg:
            kwags["table_xy_bias"] = list(self._cfg["table_xy_bias"])

        # Guard early _update_kinematic_tasks during camera init.
        self._loaded = False
        self.stove_on = False
        self.turned_on_once = False
        self._liquid_entity = None
        self._burner_shapes: list[Any] = []
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._board_welded = False
        self._board_weld_offset = None
        self._board_weld_arm = None
        self._suppress_board_attach = False
        self._veg_released = False
        self._veg_fallen = False
        self._pour_armed = False
        self._reset_metric_state()
        self._force_veg_hold = False
        self._score_veg_spill = False
        self._arm_veg_contact = False
        self._fail_reason = ""
        self._fail_checks_armed = False
        self._robot_link_names: set[str] = set()
        self._veg_names: set[str] = set()
        self._veg_offsets: list[sapien.Pose] = []
        self.veggies: list[Any] = []
        self._veg_rigids: list[Any] = []
        self._board_rigid = None
        self.board = None
        self.pot = None
        self.microwave = None
        self._ring_parts: list[Any] = []
        self._ring_shapes: list[Any] = []
        self._disc_parts: list[Any] = []
        self._disc_shapes: list[Any] = []
        self._handle_local = np.zeros(3)
        self._layout_footprints: list[tuple[np.ndarray, np.ndarray]] = []

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _load_microwave(self, table_height, table_xy_bias) -> None:
        """No microwave — leave the left counter clear for randomized decor."""
        self.microwave = None
        self.microwave_xy = None
        self.microwave_half_xy = None
        return

    def _sample_range_xy(
        self, cfg: dict[str, Any], rng: np.random.RandomState
    ) -> list[float]:
        """Pick stove among current / top-left / top-right (seed-stable)."""
        slots = dict(self.RANGE_SLOTS)
        # Allow config overrides of the three discrete poses.
        for name in ("current", "top_left", "top_right"):
            key = f"range_xy_{name}"
            if key in cfg and len(cfg[key]) >= 2:
                slots[name] = (float(cfg[key][0]), float(cfg[key][1]))
        if not bool(cfg.get("randomize_layout", True)):
            rel = list(cfg.get("range_xy", list(self.RANGE_REL_XY)))
            self._range_slot = "current"
            return [float(rel[0]), float(rel[1])]
        names = list(cfg.get("range_slots", list(slots.keys())))
        names = [n for n in names if n in slots] or list(slots.keys())
        slot = str(rng.choice(names))
        self._range_slot = slot
        xy = slots[slot]
        return [float(xy[0]), float(xy[1])]

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

    def _footprint_clear(
        self,
        center,
        half_xy,
        blockers: list[tuple[np.ndarray, np.ndarray]],
        margin: float | None = None,
    ) -> bool:
        if margin is None:
            margin = self.LAYOUT_MARGIN
        c = np.asarray(center, dtype=float)
        h = np.asarray(half_xy, dtype=float)
        for b_c, b_h in blockers:
            if self._aabb_overlap(c, h, b_c, b_h, margin):
                return False
        return True

    def _range_blocker(self) -> tuple[np.ndarray, np.ndarray]:
        xy = np.asarray(getattr(self, "range_xy", self.RANGE_REL_XY), dtype=float)
        half = np.asarray(
            getattr(self, "range_half_size", (0.16, 0.14)), dtype=float
        )
        return xy, half

    def _sample_board_xy(
        self, cfg: dict[str, Any], rng: np.random.RandomState
    ) -> tuple[float, float]:
        """Chopping board on the lower (−Y) apron, clear of the cooktop."""
        hx = float(self.board_half[0]) + float(self.handle_half[1]) * 0.5
        hy = float(self.board_half[1]) + float(self.handle_half[1])
        board_half = np.array([hx, hy], dtype=float)
        range_c, range_h = self._range_blocker()
        blockers = [(range_c, range_h)]
        side = 1.0 if float(range_c[0]) >= 0.0 else -1.0
        base_x = float(cfg.get("board_x", float(range_c[0]) + side * 0.02))
        base_y = float(cfg.get("board_y", -0.14))
        slot = str(getattr(self, "_range_slot", "current"))

        if not bool(cfg.get("randomize_layout", True)):
            xy = (base_x, base_y)
            if self._footprint_clear(xy, board_half, blockers):
                return xy
            for y in np.linspace(base_y, -0.22, 8):
                cand = (base_x, float(y))
                if self._footprint_clear(cand, board_half, blockers):
                    return cand
            return xy

        # Current stove: board anywhere on the lower half. Corner stoves: keep
        # the board on the same side so the pour arm can also reach the pot.
        if slot == "current":
            x_lo, x_hi = -0.28, 0.32
        elif slot == "top_left":
            x_lo, x_hi = -0.28, 0.06
        else:  # top_right
            x_lo, x_hi = -0.06, 0.32

        for _ in range(120):
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(-0.24, -0.06))
            cand = (x, y)
            if self._footprint_clear(cand, board_half, blockers):
                return cand
        # Deterministic fallback in front of the stove on the free side.
        return (
            float(np.clip(float(range_c[0]) - side * 0.08, -0.24, 0.28)),
            -0.18,
        )

    def _pick_board_arm(
        self, board_x: float, rng: np.random.RandomState
    ) -> ArmTag:
        """Arm from board X; near x=0 either gripper may be used.

        In the center band, prefer the stove side so the pour arm can still
        reach the pot (both grippers can grasp a near-center board).
        """
        tol = float(self._cfg.get("board_center_x_tol", self.BOARD_CENTER_X_TOL))
        if abs(float(board_x)) <= tol:
            # Either gripper can reach the board; pick the stove-side arm so the
            # pour into the pot stays reachable. Across seeds both arms still
            # appear via left/right stove slots.
            range_x = float(np.asarray(getattr(self, "range_xy", (0.0, 0.0)))[0])
            return ArmTag("right" if range_x >= 0.0 else "left")
        return ArmTag("right" if float(board_x) > 0.0 else "left")

    def _sample_decor_layout(
        self, cfg: dict[str, Any], rng: np.random.RandomState, board_xy
    ) -> dict[str, tuple[float, float]]:
        """Random non-overlapping plate / wine / glass on empty apron space."""
        range_c, range_h = self._range_blocker()
        bx = np.asarray(board_xy, dtype=float)
        bh = np.array(
            [
                float(self.board_half[0]) + float(self.handle_half[1]) * 0.5,
                float(self.board_half[1]) + float(self.handle_half[1]),
            ],
            dtype=float,
        )
        pot_c = np.asarray(getattr(self, "pot_xy", range_c), dtype=float)
        pot_h = np.array(
            [float(getattr(self, "pot_radius", 0.06)) + 0.02] * 2, dtype=float
        )
        blockers: list[tuple[np.ndarray, np.ndarray]] = [
            (range_c, range_h),
            (bx, bh),
            (pot_c, pot_h),
        ]
        free_sign = -1.0 if float(range_c[0]) >= 0.0 else 1.0
        defaults = {
            "plate": (
                float(cfg.get("plate_x", free_sign * 0.22)),
                float(cfg.get("plate_y", -0.08)),
            ),
            "wine": (
                float(cfg.get("wine_x", free_sign * 0.34)),
                float(cfg.get("wine_y", -0.02)),
            ),
            "glass": (
                float(cfg.get("glass_x", free_sign * 0.26)),
                float(cfg.get("glass_y", -0.16)),
            ),
        }
        halves = {
            "plate": np.array(self.PLATE_HALF_XY, dtype=float),
            "wine": np.array(self.WINE_HALF_XY, dtype=float),
            "glass": np.array(self.GLASS_HALF_XY, dtype=float),
        }

        if not bool(cfg.get("randomize_layout", True)):
            return defaults

        out: dict[str, tuple[float, float]] = {}
        for name in ("plate", "wine", "glass"):
            half = halves[name]
            placed = False
            for _ in range(100):
                # Any empty apron cell — not restricted to one side.
                x = float(rng.uniform(-0.42, 0.42))
                y = float(rng.uniform(-0.22, 0.06))
                cand = np.array([x, y], dtype=float)
                if self._footprint_clear(cand, half, blockers):
                    out[name] = (x, y)
                    blockers.append((cand, half))
                    placed = True
                    break
            if not placed:
                out[name] = defaults[name]
                blockers.append((np.asarray(defaults[name], dtype=float), half))
        return out

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
        self.tilt_hold_dot = float(cfg.get("tilt_hold_dot", self.TILT_HOLD_DOT))
        self.tilt_spill_dot = float(cfg.get("tilt_spill_dot", self.TILT_SPILL_DOT))
        self.pour_tip_rad = float(cfg.get("pour_tip_rad", self.POUR_TIP_RAD))
        self.water_level = float(cfg.get("water_level", self.WATER_LEVEL_DEFAULT))
        self.board_half = list(cfg.get("board_half", list(self.BOARD_HALF)))
        self.handle_half = list(cfg.get("handle_half", list(self.HANDLE_HALF)))
        self.cube_half = float(cfg.get("cube_half", self.CUBE_HALF))
        self.tomato_radius = float(cfg.get("tomato_radius", self.TOMATO_RADIUS))
        self.grasp_tcp_tol = float(cfg.get("grasp_tcp_tol", self.GRASP_TCP_TOL))
        self.n_veg_min = int(cfg.get("n_veg_min", self.N_VEG_MIN))
        self.n_veg_max = int(cfg.get("n_veg_max", self.N_VEG_MAX))
        if self.n_veg_min < 1:
            self.n_veg_min = 1
        if self.n_veg_max < self.n_veg_min:
            self.n_veg_max = self.n_veg_min
        # Expert demo failure: tip the board over the table so produce spills.
        self.force_spill = bool(cfg.get("force_spill", False))

        self.stove_on = False
        self.turned_on_once = False
        self._liquid_entity = None
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._board_welded = False
        self._board_weld_offset = None
        self._board_weld_arm = None
        self._suppress_board_attach = False
        self._veg_released = False
        self._veg_fallen = False
        self._pour_armed = False
        self._reset_metric_state()
        self._force_veg_hold = False
        self._score_veg_spill = False
        self._arm_veg_contact = False
        self._fail_reason = ""
        self._fail_checks_armed = False
        self._robot_link_names: set[str] = set()
        self._veg_names: set[str] = set()
        self._veg_offsets = []
        self.veggies = []
        self._veg_rigids = []
        self._board_rigid = None
        self._ring_parts = []
        self._ring_shapes = []
        self._disc_parts = []
        self._disc_shapes = []

        bz = 0.74 + self.table_z_bias
        self.table_top = bz

        rng = np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + 101)
        # Board first so arm + front-burner choice stay on a reachable side.
        board_x, board_y = self._sample_board_xy(cfg, rng)
        self.board_xy = (board_x, board_y)
        self.arm = self._pick_board_arm(board_x, rng)
        self.board_arm = self.arm

        burner_cfg = str(cfg.get("burner", self.ACTIVE_BURNER)).strip().lower()
        if bool(cfg.get("randomize_layout", True)) and not bool(
            cfg.get("pin_burner", False)
        ):
            # Front burners only. Corner stoves use the inner burner (closer to
            # x=0) so the pour stays in dual-arm reach; current slot matches arm.
            slot = str(getattr(self, "_range_slot", "current"))
            front = [
                b
                for b in self.FRONT_BURNERS
                if b in getattr(self, "burner_positions", {})
            ]
            if slot == "top_left":
                preferred = "right_front"  # inner
            elif slot == "top_right":
                preferred = "left_front"  # inner
            else:
                preferred = (
                    "left_front" if str(self.arm) == "left" else "right_front"
                )
            if preferred in front and rng.rand() < 0.70:
                burner_name = preferred
            else:
                # Still diversify across both front burners some of the time.
                other = (
                    "left_front" if preferred == "right_front" else "right_front"
                )
                choices = [b for b in (preferred, other) if b in front] or front
                burner_name = str(rng.choice(choices))
        else:
            burner_name = burner_cfg or self.ACTIVE_BURNER
        if burner_name not in self.burner_positions:
            raise ValueError(
                f"make_soup.burner must be one of {list(self.burner_positions)}, got {burner_name!r}"
            )
        bx, by = self.burner_positions[burner_name]
        self.burner_name = burner_name
        self.burner_xy = (float(bx), float(by))

        # Keep the stock burner disc under the pot; fire ring sits around its base.
        self._burner_shapes = []
        if getattr(self, "active_burner", None) is not None:
            home = sapien.Pose(
                p=[float(bx), float(by), float(self.range_top_z) + 0.0015]
            )
            self._burner_home_pose = home
            try:
                self.active_burner.set_pose(home)
            except Exception:
                pass
            for c in self.active_burner.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_shapes = list(c.render_shapes)

        self._spawn_pot(float(bx), float(by), cfg)
        pot_r = float(self.pot_radius)
        # Ring sits just outside the pot floor so the blue halo is visible.
        self._build_stove_fire_ring(
            float(bx),
            float(by),
            float(self.range_top_z) + 0.0035,
            float(pot_r + 0.016),
            n=32,
            half_size=[0.010, 0.005, 0.0035],
        )
        self._rebuild_water(force=True)
        # Stove starts lit — the episode is only about pouring produce in.
        self.knob_angle = float(self.KNOB_ON_ANGLE)
        self._set_knob_joint_angle(float(self.KNOB_ON_ANGLE), hard=True)
        # Force fire refresh after ring rebuild (cache cleared in base clear).
        self.stove_on = False
        self._set_stove(True)

        self.board = self._spawn_board(board_x, board_y, bz)
        self.add_prohibit_area(self.board, padding=0.04)

        self._spawn_vegetables(board_x, board_y, bz + 2.0 * self.board_half[2], rng)
        self._decor_xy = self._sample_decor_layout(cfg, rng, self.board_xy)
        self._spawn_decor(bz)
        self.knob_arm = ArmTag(
            "right" if float(np.asarray(self.knob_xy)[0]) >= 0.0 else "left"
        )
        self._robot_link_names = self._collect_robot_link_names()
        self._veg_names = {
            str(v.get_name()) for v in self.veggies if hasattr(v, "get_name")
        }
        self._loaded = True
        # Arm failure checks only after produce has settled on the board.
        self._fail_checks_armed = True
        print(
            f"[make_soup] arm={self.arm} knob_arm={self.knob_arm} "
            f"slot={getattr(self, '_range_slot', '?')} "
            f"burner={self.burner_name} range={np.round(self.range_xy, 3)} "
            f"board={np.round(self.board_xy, 3)} pot={np.round(self.pot_xy, 3)} "
            f"decor={ {k: list(np.round(v, 3)) for k, v in self._decor_xy.items()} } "
            f"n_veg={len(self.veggies)}"
        )

    def _spawn_pot(self, cx: float, cy: float, cfg: dict[str, Any]) -> None:
        """Original soup pot: hollow cylinder with two short side handles."""
        pot_r = float(cfg.get("pot_radius", self.POT_RADIUS))
        pot_h = float(cfg.get("pot_height", self.POT_HEIGHT))
        wall = float(cfg.get("pot_wall", self.POT_WALL))
        pot_z0 = float(self.range_top_z)
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        metal = sapien.render.RenderMaterial(base_color=[0.55, 0.55, 0.58, 1.0])
        metal.metallic = 0.7
        metal.roughness = 0.35

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r,
            half_length=wall / 2,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r,
            half_length=wall / 2,
            material=metal,
        )

        wall_segments = 20
        wall_radius = pot_r - wall / 2
        tangent_half = wall_radius * np.tan(np.pi / wall_segments) * 1.04
        for ang in np.linspace(0, 2 * np.pi, wall_segments, endpoint=False):
            px = wall_radius * np.cos(ang)
            py = wall_radius * np.sin(ang)
            tangent_angle = ang + np.pi / 2
            q = [
                float(np.cos(tangent_angle / 2)),
                0.0,
                0.0,
                float(np.sin(tangent_angle / 2)),
            ]
            wall_pose = sapien.Pose([px, py, pot_h / 2], q)
            builder.add_box_collision(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=self.scene.default_physical_material,
            )
            builder.add_box_visual(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=metal,
            )
        builder.set_initial_pose(sapien.Pose(p=[cx, cy, pot_z0]))
        self.pot = builder.build(name="soup_pot")

        # Two short lugs on the ±X sides, as in the original design.
        for sign, name in ((-1, "left"), (1, "right")):
            create_box(
                self.scene,
                sapien.Pose(p=[cx + sign * (pot_r + 0.015), cy, pot_z0 + pot_h * 0.65]),
                half_size=[0.012, 0.008, 0.006],
                color=(0.45, 0.45, 0.48),
                name=f"pot_handle_{name}",
                is_static=True,
            )

        self.pot_inner_radius = pot_r - 1.6 * wall
        self.pot_inner_height = pot_h - wall
        self.pot_bottom_z = pot_z0 + wall
        self.pot_rim_z = pot_z0 + pot_h
        self.pot_xy = (cx, cy)
        self.pot_radius = pot_r
        self.pot_height = pot_h

    def _spawn_board(self, x: float, y: float, table_z: float) -> Actor:
        """Wooden board with a small grasping cube on the robot-facing (−Y) end."""
        hx, hy, hz = [float(v) for v in self.board_half]
        hhx, hhy, hhz = [float(v) for v in self.handle_half]
        # Handle protrudes from the −Y edge so the gripper can pinch it cleanly.
        # Its underside is flush with the board's: any part hanging below would
        # spawn inside the counter and PhysX would launch the board off it.
        handle_local = np.array([0.0, -(hy + hhy), hhz - hz], dtype=float)
        self._handle_local = handle_local.copy()

        wood = sapien.render.RenderMaterial(base_color=[*self.BOARD_COLOR, 1.0])
        wood.roughness = 0.75
        wood.metallic = 0.0
        handle_mat = sapien.render.RenderMaterial(base_color=[*self.HANDLE_COLOR, 1.0])
        handle_mat.roughness = 0.7

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        # Deck rests on the counter; handle is grippier so the WSG can pinch it.
        # Deck μ≈0.82 (friction angle ~39°) keeps produce planted until the 40° pour.
        board_mat = self.scene.create_physical_material(
            float(self.BOARD_STATIC_FRICTION),
            float(self.BOARD_DYNAMIC_FRICTION),
            0.0,
        )
        handle_phys = self.scene.create_physical_material(
            float(self.HANDLE_STATIC_FRICTION),
            float(self.HANDLE_DYNAMIC_FRICTION),
            0.0,
        )
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=board_mat,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=wood,
        )
        builder.add_box_collision(
            pose=sapien.Pose(handle_local.tolist()),
            half_size=[hhx, hhy, hhz],
            material=handle_phys,
        )
        builder.add_box_visual(
            pose=sapien.Pose(handle_local.tolist()),
            half_size=[hhx, hhy, hhz],
            material=handle_mat,
        )
        builder.set_initial_pose(sapien.Pose([x, y, table_z + hz], [1, 0, 0, 0]))
        entity = builder.build(name="chopping_board")

        # Grasp frames on the handle top (meters; scale=1).
        top_z = float(handle_local[2] + hhz)
        hy_h = float(handle_local[1])
        data = {
            "center": [0, 0, 0],
            "extents": [hx * 2, (hy + 2 * hhy) * 2, max(hz, hhz) * 2],
            "scale": [1.0, 1.0, 1.0],
            "contact_points_pose": [
                # 4 yaw variants of top-down grasp centered on the handle top.
                [[0, 0, 1, 0.0], [1, 0, 0, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[1, 0, 0, 0.0], [0, 0, -1, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[0, 0, -1, 0.0], [-1, 0, 0, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
            ],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }

        board = Actor(entity, data, mass=float(self.BOARD_MASS))
        self._board_rigid = None
        for c in board.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    # Solid-box inertia about the deck centre (handle is light).
                    m = float(self.BOARD_MASS)
                    ix = (1.0 / 12.0) * m * (4 * hy * hy + 4 * hz * hz)
                    iy = (1.0 / 12.0) * m * (4 * hx * hx + 4 * hz * hz)
                    iz = (1.0 / 12.0) * m * (4 * hx * hx + 4 * hy * hy)
                    c.set_mass(m)
                    c.set_cmass_local_pose(sapien.Pose([0.0, 0.0, 0.0]))
                    c.set_inertia([ix, iy, iz])
                    c.set_linear_damping(2.0)
                    c.set_angular_damping(4.0)
                except Exception:
                    pass
                self._board_rigid = c
                try:
                    for shape in c.get_collision_shapes():
                        shape.set_collision_groups([1, 1, 0, 0])
                except Exception:
                    pass
                break
        return board

    def _spawn_decor(self, bz: float) -> None:
        """Static background props (plate+bread, wine, glass) — non-overlapping."""
        cfg = self._cfg
        layout = getattr(self, "_decor_xy", None) or {}
        plate_x, plate_y = layout.get(
            "plate",
            (float(cfg.get("plate_x", -0.18)), float(cfg.get("plate_y", -0.08))),
        )
        plate_scale = float(cfg.get("plate_scale", 0.55))
        self.decor_plate = create_actor(
            self,
            pose=sapien.Pose(
                [float(plate_x), float(plate_y), bz], list(self.DECOR_QPOS)
            ),
            modelname="003_plate",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=plate_scale,
        )
        self.decor_plate.set_name("003_plate")
        self.add_prohibit_area(self.decor_plate, padding=0.03)
        try:
            plate_top = float(self.decor_plate.get_functional_point(0)[2])
        except Exception:
            plate_top = bz + 0.02

        bread_scale = float(cfg.get("bread_scale", 1.0))
        bread_id = int(
            np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + 303).choice(
                [0, 1, 2]
            )
        )
        self.decor_bread = create_actor(
            self,
            pose=sapien.Pose(
                [float(plate_x), float(plate_y), plate_top + 0.008],
                list(self.DECOR_QPOS),
            ),
            modelname="075_bread",
            model_id=bread_id,
            convex=True,
            is_static=True,
            scale_mult=bread_scale,
        )
        self.decor_bread.set_name("075_bread")

        bottle_x, bottle_y = layout.get(
            "wine",
            (float(cfg.get("wine_x", -0.30)), float(cfg.get("wine_y", -0.02))),
        )
        self.decor_wine = create_actor(
            self,
            pose=sapien.Pose(
                [float(bottle_x), float(bottle_y), bz], list(self.DECOR_QPOS)
            ),
            modelname="265_wine_bottle",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=float(cfg.get("wine_scale", 1.0)),
        )
        self.decor_wine.set_name("265_wine_bottle")
        self.add_prohibit_area(self.decor_wine, padding=0.03)

        glass_x, glass_y = layout.get(
            "glass",
            (float(cfg.get("glass_x", -0.22)), float(cfg.get("glass_y", -0.16))),
        )
        glass_scale = float(cfg.get("glass_scale", self.GLASS_SCALE))
        self.decor_glass = create_actor(
            self,
            pose=sapien.Pose(
                [float(glass_x), float(glass_y), bz], list(self.DECOR_QPOS)
            ),
            modelname="088_wineglass",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=glass_scale,
        )
        self.decor_glass.set_name("088_wineglass")
        self.add_prohibit_area(self.decor_glass, padding=0.03)

    @staticmethod
    def _veg_footprint_half(spec: dict[str, Any]) -> tuple[float, float]:
        """Board-XY half-extents used for non-overlap packing."""
        if "footprint_half" in spec:
            hx, hy = spec["footprint_half"]
            return float(hx), float(hy)
        r = float(spec.get("footprint", 0.012))
        return r, r

    def _sample_veg_offsets(
        self, rng: np.random.RandomState, pour_sign: float = -1.0
    ) -> list[tuple[str, str, Any, tuple[float, float], float, tuple[float, float]]]:
        """Non-overlapping produce poses in board-local XY (AABB packing).

        Samples a random subset of distinct produce (no duplicates), size in
        ``[n_veg_min, n_veg_max]``. ``pour_sign`` is −1 when the pour lip is the
        board's −X edge (right arm) and +1 when it is the +X edge (left arm).
        Pieces stay near that lip so the tip lands them in the pot.
        """
        hx = float(self.board_half[0])
        hy = float(self.board_half[1])
        gap = 0.007  # clear air between collision AABBs
        catalog: list[tuple[str, str, dict[str, Any], tuple[float, float], float]] = [
            (
                str(s["name"]),
                "mesh",
                dict(s),
                self._veg_footprint_half(s),
                float(s["half_h"]),
            )
            for s in self.VEG_MESHES
        ]
        # Tomato sphere removed — it rolled too freely on the board.
        n_min = int(getattr(self, "n_veg_min", self.N_VEG_MIN))
        n_max = int(getattr(self, "n_veg_max", self.N_VEG_MAX))
        n_min = max(1, min(n_min, len(catalog)))
        n_max = max(n_min, min(n_max, len(catalog)))
        n_pick = int(rng.randint(n_min, n_max + 1))
        pick_idx = list(rng.choice(len(catalog), size=n_pick, replace=False))
        specs = [catalog[i] for i in pick_idx]
        max_fx = max(f[0] for *_, f, _h in specs)
        max_fy = max(f[1] for *_, f, _h in specs)
        x_lo, x_hi = -hx + max_fx + 0.004, hx - max_fx - 0.004
        y_lo = -hy + max_fy + 0.004 + float(self.handle_half[1]) * 0.40
        y_hi = hy - max_fy - 0.004
        # Pour-lip band (with room to spill into mid-board if needed).
        if float(pour_sign) < 0.0:
            lip_lo, lip_hi = x_lo, min(x_hi, 0.25 * x_lo)
        else:
            lip_lo, lip_hi = max(x_lo, 0.25 * x_hi), x_hi

        def _clear(dx: float, dy: float, fh: tuple[float, float], placed) -> bool:
            fx, fy = fh
            for px, py, pfx, pfy in placed:
                if abs(dx - px) < fx + pfx + gap and abs(dy - py) < fy + pfy + gap:
                    return False
            return True

        # Deterministic lip slots first (guaranteed non-overlap on this board).
        sx = -1.0 if float(pour_sign) < 0.0 else 1.0
        # Two rows along Y, three columns toward the pour lip along X.
        slot_x = [sx * 0.062, sx * 0.062, sx * 0.028, sx * 0.028, sx * 0.006]
        slot_y = [-0.018, 0.018, -0.018, 0.018, 0.000]
        # Largest / longest pieces first so packing succeeds.
        order = sorted(
            range(len(specs)),
            key=lambda i: specs[i][3][0] * specs[i][3][1],
            reverse=True,
        )
        placed: list[tuple[float, float, float, float]] = []
        chosen: dict[int, tuple[float, float]] = {}

        # 1) Try fixed slots (shuffled lightly for seed diversity).
        slot_idx = list(range(len(slot_x)))
        rng.shuffle(slot_idx)
        for i in order:
            name, kind, payload, fh, half_h = specs[i]
            fx, fy = fh
            ok = False
            for si in slot_idx:
                dx = float(np.clip(slot_x[si], x_lo, x_hi))
                dy = float(np.clip(slot_y[si], y_lo, y_hi))
                # Keep carrot's long axis on the lip band.
                if name == "carrot":
                    dx = float(np.clip(sx * 0.055, x_lo, x_hi))
                if _clear(dx, dy, fh, placed):
                    placed.append((dx, dy, fx, fy))
                    chosen[i] = (dx, dy)
                    ok = True
                    break
            if ok:
                continue
            # 2) Random AABB samples on / near the lip.
            for _ in range(120):
                if rng.rand() < 0.85:
                    dx = float(rng.uniform(lip_lo, lip_hi))
                else:
                    dx = float(rng.uniform(x_lo, x_hi))
                dy = float(rng.uniform(y_lo, y_hi))
                if _clear(dx, dy, fh, placed):
                    placed.append((dx, dy, fx, fy))
                    chosen[i] = (dx, dy)
                    ok = True
                    break
            if not ok:
                # 3) Last resort: push along +Y from lip origin until clear.
                dx = float(np.clip(sx * 0.040, x_lo, x_hi))
                dy = float(y_lo)
                while dy <= y_hi + 1e-6:
                    if _clear(dx, dy, fh, placed):
                        placed.append((dx, dy, fx, fy))
                        chosen[i] = (dx, dy)
                        ok = True
                        break
                    dy += fy + gap + 0.002
                if not ok:
                    # Should be unreachable on this board size; park mid-lip.
                    chosen[i] = (float(np.clip(sx * 0.020, x_lo, x_hi)), 0.0)
                    dx, dy = chosen[i]
                    placed.append((dx, dy, fx, fy))

        out: list[
            tuple[str, str, dict[str, Any], tuple[float, float], float, tuple[float, float]]
        ] = []
        for i, (name, kind, payload, fh, half_h) in enumerate(specs):
            out.append((name, kind, payload, fh, half_h, chosen[i]))
        return out

    def _recolor_actor(
        self, actor: Any, rgb: tuple[float, float, float] | list[float]
    ) -> None:
        """Flat-tint a produce mesh (used for the purple onion half)."""
        r, g, b = [float(v) for v in rgb[:3]]
        body = getattr(actor, "actor", actor)
        for c in body.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([r, g, b, 1.0])
                    except Exception:
                        pass

    def _apply_veg_mass_properties(
        self,
        rigid: Any,
        mass: float,
        radius_hint: float | None = None,
    ) -> None:
        """Set mass *and* matching inertia/COM for board produce.

        ``Actor.set_mass`` alone leaves the default ~10 g inertia tensor, so a
        tiny carrot/tomato still spins and contact-explodes — skating off the
        board or tunneling under it. Match pack_fruits / play_billiard:
        solid-sphere I = 2/5 m R^2 about the collision hull centre.
        """
        if rigid is None:
            return
        m = float(mass)
        r_fallback = float(
            radius_hint if radius_hint is not None else self.tomato_radius
        )
        try:
            shapes = list(rigid.get_collision_shapes())
            verts = np.asarray(shapes[0].get_vertices(), dtype=np.float64)
            try:
                scale = np.asarray(shapes[0].get_scale(), dtype=np.float64).reshape(3)
                verts = verts * scale
            except Exception:
                pass
            center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
            rms_r = float(np.sqrt(np.mean(np.sum((verts - center) ** 2, axis=1))))
            if not np.isfinite(rms_r) or rms_r < 1e-4:
                rms_r = r_fallback
            inertia = 0.4 * m * (rms_r ** 2)
            rigid.set_mass(m)
            rigid.set_cmass_local_pose(sapien.Pose(center.tolist()))
            rigid.set_inertia([inertia, inertia, inertia])
        except Exception:
            try:
                rigid.set_mass(m)
                inertia = 0.4 * m * (r_fallback ** 2)
                rigid.set_inertia([inertia, inertia, inertia])
            except Exception:
                pass

    def _collision_world_z_minmax(self, rigid: Any, pose: sapien.Pose) -> tuple[float, float]:
        """World-frame z extent of a body's collision hull."""
        R = pose.to_transformation_matrix()[:3, :3]
        p = np.asarray(pose.p, dtype=float)
        zmin, zmax = np.inf, -np.inf
        try:
            for shape in rigid.get_collision_shapes():
                local = shape.get_local_pose()
                Rl = local.to_transformation_matrix()[:3, :3]
                pl = np.asarray(local.p, dtype=float)
                try:
                    verts = np.asarray(shape.get_vertices(), dtype=np.float64)
                except Exception:
                    continue
                if verts.size == 0:
                    continue
                # Convex-mesh vertices are unscaled; apply shape.scale.
                try:
                    scale = np.asarray(shape.get_scale(), dtype=np.float64).reshape(3)
                    verts = verts * scale
                except Exception:
                    pass
                local_pts = (Rl @ verts.T).T + pl
                world = (R @ local_pts.T).T + p
                zmin = min(zmin, float(world[:, 2].min()))
                zmax = max(zmax, float(world[:, 2].max()))
        except Exception:
            pass
        if not np.isfinite(zmin):
            z = float(p[2])
            return z, z
        return zmin, zmax

    def _depenetrate_veggies_on_board(self) -> None:
        """Lift any produce still intersecting the deck so it rests on top."""
        if self.board is None or not self.veggies:
            return
        board_top = float(self.board.get_pose().p[2]) + float(self.board_half[2])
        for veg, rigid in zip(self.veggies, self._veg_rigids):
            if rigid is None:
                continue
            pose = veg.get_pose()
            bottom, _ = self._collision_world_z_minmax(rigid, pose)
            lift = (board_top + 0.0015) - float(bottom)
            if lift <= 0.0:
                continue
            p = np.asarray(pose.p, dtype=float)
            p[2] += lift
            obj = veg.actor if hasattr(veg, "actor") else veg
            obj.set_pose(sapien.Pose(p.tolist(), list(pose.q)))
            try:
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass

    def _settle_veggies_on_board(
        self, n_steps: int = 60, *, freeze_board: bool | None = None
    ) -> None:
        """Let produce fall onto the board under contact.

        Never teleport the board every step — that embeds dynamic pieces inside
        the thin deck. When the board is already in the gripper, do not freeze
        it kinematic (that would yank it out of the grasp).
        """
        if self.board is None:
            return
        if freeze_board is None:
            freeze_board = not self._board_held()
        was_kin = False
        if freeze_board and self._board_rigid is not None:
            try:
                was_kin = bool(self._board_rigid.kinematic)
                self._board_rigid.set_kinematic(True)
                self._board_rigid.set_disable_gravity(True)
                self._board_rigid.set_linear_velocity(np.zeros(3))
                self._board_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
            hold = self.board.get_pose()
            self._set_entity_pose(self.board, hold, snap=True)
        for _ in range(int(n_steps)):
            self.scene.step()
        if freeze_board:
            self._depenetrate_veggies_on_board()
            for _ in range(24):
                self.scene.step()
        for rigid in self._veg_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass
        if freeze_board and self._board_rigid is not None and not was_kin:
            try:
                self._board_rigid.set_kinematic(False)
                self._board_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _spawn_vegetables(
        self,
        bx: float,
        by: float,
        board_top: float,
        rng: np.random.RandomState | None = None,
    ) -> None:
        """Random distinct produce subset (2–4) on the board — no duplicates."""
        if rng is None:
            rng = np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + 202)
        # Right arm pours over the board's −X lip; left arm over +X.
        pour_sign = -1.0 if str(getattr(self, "arm", "right")) == "right" else 1.0
        layout = self._sample_veg_offsets(rng, pour_sign=pour_sign)
        for name, kind, payload, fh, half_h, (dx, dy) in layout:
            radius_hint = float(half_h)
            if kind == "mesh":
                # Kenney / onion meshes are Y-up. Upright: +Y→world +Z.
                # Side: model +Y → board +X so the stem lies on the deck.
                orient = str(payload.get("orient", "upright"))
                q = list(self.SIDE_QPOS if orient == "side" else self.DECOR_QPOS)
                # Clearance above the deck — settle drops them onto contact.
                z = board_top + float(half_h) + 0.006
                pose = sapien.Pose([bx + dx, by + dy, z], q)
                veg = create_actor(
                    self,
                    pose=pose,
                    modelname=str(payload["modelname"]),
                    model_id=0,
                    convex=True,
                    is_static=False,
                    scale_mult=float(payload.get("scale_mult", 1.0)),
                )
                veg.set_name(name)
                mass = float(self.VEG_MESH_MASS)
                if payload.get("color") is not None:
                    self._recolor_actor(veg, payload["color"])
            else:
                r = float(self.tomato_radius)
                radius_hint = r
                z = board_top + r + 0.006
                pose = sapien.Pose([bx + dx, by + dy, z], [1, 0, 0, 0])
                entity = create_sphere(
                    self,
                    pose=pose,
                    radius=r,
                    color=list(payload["color"]) + [1.0],
                    name=name,
                    is_static=False,
                )
                data = {
                    "center": [0, 0, 0],
                    "extents": [r, r, r],
                    "scale": [r, r, r],
                    "contact_points_pose": [],
                    "functional_matrix": [],
                    "transform_matrix": np.eye(4).tolist(),
                }
                mass = float(self.VEG_SPHERE_MASS)
                veg = Actor(entity, data, mass=mass)
            rigid = None
            for c in veg.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    self._apply_veg_mass_properties(c, mass, radius_hint=radius_hint)
                    try:
                        c.set_linear_damping(float(self.VEG_LINEAR_DAMPING))
                        c.set_angular_damping(float(self.VEG_ANGULAR_DAMPING))
                        for shape in c.get_collision_shapes():
                            # Must NOT share word2/word3 ignore ids with the board
                            # ([1,1,1,1] on both = mutual ignore → sink into deck).
                            shape.set_collision_groups([1, 1, 0, 0])
                            m = shape.get_physical_material()
                            # Produce stays dynamic on the board: enough friction
                            # for a level carry, still slides under a ~40° tip
                            # (friction angle ≈ arctan(0.70) ≈ 35°).
                            m.set_static_friction(float(self.VEG_STATIC_FRICTION))
                            m.set_dynamic_friction(float(self.VEG_DYNAMIC_FRICTION))
                            m.set_restitution(0.0)
                    except Exception:
                        pass
                    rigid = c
                    break
            self.veggies.append(veg)
            self._veg_rigids.append(rigid)
            pad = max(fh) + 0.008
            self.add_prohibit_area(veg, padding=pad)

        # Produce is always dynamic — settle onto the board via real contact.
        self._ensure_veggies_dynamic()
        self._veg_released = True
        self._settle_veggies_on_board(72)

    # ---------------------------------------------------------------- liquid / stove
    def _rebuild_water(self, force: bool = False) -> None:
        half_h = max(0.004, 0.5 * self.water_level * self.pot_inner_height)
        if (
            not force
            and self._liquid_entity is not None
            and abs(half_h - getattr(self, "_liquid_half_h_cached", -1.0)) < 0.0015
        ):
            return
        self._liquid_half_h_cached = half_h
        if self._liquid_entity is not None:
            try:
                self.scene.remove_entity(self._liquid_entity)
            except Exception:
                pass
            self._liquid_entity = None

        color = [0.25, 0.55, 0.92, 0.72]
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=color)
        mat.metallic = 0.0
        mat.roughness = 0.12
        vertical_cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], vertical_cyl_q),
            radius=self.pot_inner_radius,
            half_length=half_h,
            material=mat,
        )
        z = self.pot_bottom_z + half_h
        builder.set_initial_pose(sapien.Pose(p=[self.pot_xy[0], self.pot_xy[1], z]))
        self._liquid_entity = builder.build(name="pot_water")

    def _set_burner_glow(self, on: bool) -> None:
        self._set_stove_fire(bool(on), intensity=1.0 if on else 0.0)

    def _set_stove(self, on: bool) -> None:
        on = bool(on)
        if on == self.stove_on:
            return
        self.stove_on = on
        if on:
            self.turned_on_once = True
        self._set_burner_glow(on)

    # ---------------------------------------------------------------- board / veg physics
    def _get_rigid(self, entity: Any):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _board_is_entity(self, entity: Any) -> bool:
        if self.board is None or entity is None:
            return False
        if entity is self.board:
            return True
        board_obj = self.board.actor if hasattr(self.board, "actor") else self.board
        obj = entity.actor if hasattr(entity, "actor") else entity
        return obj is board_obj

    def _clamp_board_pose_above_pot(self, pose: sapien.Pose) -> sapien.Pose:
        """Stop a pose-driven board from tunneling through the pot rim.

        The board is kinematic while welded / tipped, so PhysX will not resolve
        penetration from ``set_pose``. Raise the pose so any board sample over the
        pot footprint stays on or above the rim (contact, then stop).
        """
        if getattr(self, "pot", None) is None:
            return pose
        p = np.asarray(pose.p, dtype=float).copy()
        R = pose.to_transformation_matrix()[:3, :3]
        hx, hy, hz = [float(v) for v in self.board_half]
        pot_xy = np.asarray(self.pot_xy, dtype=float)
        pot_r = float(self.pot_radius) + 0.004
        rim_z = float(self.pot_rim_z)
        lift = 0.0
        for sx in (-1.0, 0.0, 1.0):
            for sy in (-1.0, 0.0, 1.0):
                world = p + R @ np.array([sx * hx, sy * hy, -hz], dtype=float)
                if float(np.linalg.norm(world[:2] - pot_xy)) >= pot_r:
                    continue
                if float(world[2]) < rim_z:
                    lift = max(lift, rim_z - float(world[2]))
        if lift <= 0.0:
            return pose
        p[2] += lift
        return sapien.Pose(p.tolist(), list(pose.q))

    def _set_entity_pose(
        self, entity: Any, pose: sapien.Pose, *, snap: bool | None = None
    ) -> None:
        """Pose a body. For the kinematic board, prefer kinematic targets so
        dynamic produce can ride via contact instead of falling through a
        ``set_pose`` teleport each frame.
        """
        if self._board_is_entity(entity):
            pose = self._clamp_board_pose_above_pot(pose)
        obj = entity.actor if hasattr(entity, "actor") else entity
        rigid = self._get_rigid(entity)
        use_snap = True if snap is None else bool(snap)
        if self._board_is_entity(entity) and rigid is not None:
            try:
                is_kin = bool(rigid.kinematic)
            except Exception:
                is_kin = False
            if is_kin:
                cur = np.asarray(obj.get_pose().p, dtype=float)
                tgt = np.asarray(pose.p, dtype=float)
                # Large gap (first seat / recover): must snap. Small follow: target only.
                if use_snap and float(np.linalg.norm(cur - tgt)) > 0.04:
                    obj.set_pose(pose)
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    obj.set_pose(pose)
                return
        obj.set_pose(pose)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _board_up_dot(self) -> float:
        """World +Z component of the board's local +Z (1 = flat, 0 = vertical)."""
        if self.board is None:
            return 1.0
        R = self.board.get_pose().to_transformation_matrix()[:3, :3]
        return float(R[:, 2] @ np.array([0.0, 0.0, 1.0]))

    def _veggies_on_board(self) -> bool:
        """True if every piece still rests on/near the chopping board."""
        if self.board is None or not self.veggies:
            return False
        return all(self._veg_on_board(v) for v in self.veggies)

    def _veg_on_board(self, veg: Any) -> bool:
        """True if one piece still rests on/near the chopping board."""
        if self.board is None or veg is None:
            return False
        bp = np.asarray(self.board.get_pose().p, dtype=float)
        R = self.board.get_pose().to_transformation_matrix()[:3, :3]
        hx, hy, hz = [float(v) for v in self.board_half]
        p = np.asarray(veg.get_pose().p, dtype=float)
        local = R.T @ (p - bp)
        if abs(float(local[0])) > hx + 0.03:
            return False
        if abs(float(local[1])) > hy + 0.03:
            return False
        if float(local[2]) < hz - 0.01 or float(local[2]) > hz + 0.08:
            return False
        return True

    def _collect_robot_link_names(self) -> set[str]:
        names: set[str] = set()
        robot = getattr(self, "robot", None)
        if robot is None:
            return names
        for articulation in (robot.left_entity, robot.right_entity):
            if articulation is None:
                continue
            try:
                for link in articulation.get_links():
                    names.add(link.get_name())
            except Exception:
                pass
        return names

    def _mark_fail(self, reason: str) -> None:
        """Latch a terminal failure reason (first reason wins)."""
        if not getattr(self, "_fail_reason", ""):
            self._fail_reason = str(reason)
            print(f"[make_soup] FAIL: {self._fail_reason}")

    def _ensure_veggies_dynamic(self) -> None:
        """Produce is always a dynamic rigid body (never kinematic / welded)."""
        for rigid in self._veg_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _capture_veg_offsets(self) -> None:
        """Deprecated no-op — veggies are not soft-welded to the board."""
        self._veg_offsets = []

    def _freeze_veggies_to_board(self) -> None:
        """Deprecated no-op — veggies stay dynamic."""
        self._ensure_veggies_dynamic()

    def _release_veggies_physics(self, *, settle_on_board: bool = False) -> None:
        """Ensure produce is dynamic; optionally settle contact on a held board.

        No pose teleport / soft-weld. Pieces already ride the board under PhysX.
        """
        self._ensure_veggies_dynamic()
        self._veg_released = True
        if settle_on_board and self.board is not None:
            self._settle_veggies_on_board(36)
        print(
            f"[make_soup] veggies dynamic (board_up_dot={self._board_up_dot():.3f} "
            f"pour_armed={self._pour_armed})"
        )

    def _sync_veggies_to_board(self) -> None:
        """Deprecated no-op — never teleport produce onto the board."""
        return

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _board_hold_arm(self) -> ArmTag | None:
        """Arm that is pinching the handle (not necessarily layout ``self.arm``)."""
        arm = getattr(self, "_board_weld_arm", None)
        if arm is not None:
            return arm
        return getattr(self, "arm", None)

    def _board_held(self, arm: ArmTag | None = None) -> bool:
        """True when closed jaws are still near the handle (friction grasp)."""
        if self.board is None:
            return False
        hold = arm if arm is not None else self._board_hold_arm()
        if hold is None:
            return False
        return self._board_gripper_closed(hold) and self._board_handle_near(hold)

    def _mark_board_held(self, arm: ArmTag) -> None:
        """Record which arm is carrying the board — no weld / kinematic latch."""
        self._board_weld_arm = arm
        self._board_welded = False
        self._board_weld_offset = None
        self._pour_armed = True
        if self._metric_lift_step is None:
            self._metric_lift_step = self._metric_step()
        self.arm = arm
        self.board_arm = arm
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_kinematic(False)
                self._board_rigid.set_disable_gravity(False)
            except Exception:
                pass
        self._ensure_veggies_dynamic()

    def _clear_board_held(self) -> None:
        self._board_welded = False
        self._board_weld_offset = None
        self._board_weld_arm = None
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_kinematic(False)
                self._board_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _weld_board_to_ee(self, arm: ArmTag) -> None:
        """Deprecated — natural friction grasp only (kept as held-arm marker)."""
        self._mark_board_held(arm)

    def _release_board_weld(self) -> None:
        """Deprecated name — clear held-arm marker; board stays dynamic."""
        self._clear_board_held()

    def _sync_board_to_ee(self) -> None:
        """No-op — board is not latched to the EE."""
        return

    def _set_board_pose_keep_weld(
        self, pose: sapien.Pose, *, snap: bool | None = None
    ) -> None:
        """Set board pose without any EE latch."""
        self._set_entity_pose(self.board, pose, snap=snap)

    def _board_gripper_val(self, arm: ArmTag) -> float | None:
        robot = getattr(self, "robot", None)
        if robot is None:
            return None
        try:
            if str(arm) == "right":
                return float(robot.get_right_gripper_val())
            return float(robot.get_left_gripper_val())
        except Exception:
            return None

    def _board_gripper_closed(self, arm: ArmTag) -> bool:
        val = self._board_gripper_val(arm)
        return val is not None and val < float(self.BOARD_ATTACH_GRIPPER_MAX)

    def _board_gripper_open(self, arm: ArmTag) -> bool:
        val = self._board_gripper_val(arm)
        return val is not None and val > float(self.BOARD_RELEASE_GRIPPER_MIN)

    def _board_handle_near(self, arm: ArmTag) -> bool:
        if self.board is None:
            return False
        dist = float(np.linalg.norm(self._tcp_pos(arm) - self._handle_world()))
        return dist <= float(self.grasp_tcp_tol) * 2.5

    def _try_policy_board_attach(self) -> None:
        """Deprecated no-op — closing the gripper never welds the board."""
        return

    def _set_collision_ignore(self, entities: list[Any], ignore_bit: int, ignore_id: int) -> None:
        for ent in entities:
            if ent is None:
                continue
            try:
                shapes = []
                if hasattr(ent, "get_collision_shapes"):
                    shapes = list(ent.get_collision_shapes())
                elif hasattr(ent, "actor"):
                    for c in ent.actor.get_components():
                        if isinstance(
                            c,
                            (
                                sapien.physx.PhysxRigidDynamicComponent,
                                sapien.physx.PhysxRigidStaticComponent,
                            ),
                        ):
                            shapes.extend(c.get_collision_shapes())
                else:
                    for c in ent.get_components():
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
                            int(g2) | ignore_bit,
                            (int(g3) & 0xFFFF0000) | ignore_id,
                        ]
                    )
            except Exception:
                pass

    def _ignore_board_robot_collision(self) -> None:
        ignore_bit, ignore_id = self.BOARD_IGNORE_BIT, self.BOARD_IGNORE_ID
        ents = [self.board.actor]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        self._set_collision_ignore(ents, ignore_bit, ignore_id)

    def _ignore_board_veg_collision(self) -> None:
        """Stop the board scooping settled produce back out as it rolls level."""
        if self.board is None:
            return
        bit, gid = 1 << 16, 16
        ents: list[Any] = [self.board.actor]
        for veg in self.veggies:
            ents.append(getattr(veg, "actor", veg))
        self._set_collision_ignore(ents, bit, gid)

    def _veg_in_pot(self, veg: Any) -> bool:
        """True only when the piece has fallen *into* the pot (below the rim).

        Pieces still sitting on a board lip that overhangs the mouth must not
        count — that falsely short-circuits the tip and looks like a teleport.
        """
        p = np.asarray(veg.get_pose().p, dtype=float)
        dxy = float(np.linalg.norm(p[:2] - np.asarray(self.pot_xy)))
        return (
            dxy < self.pot_inner_radius * 0.95
            and float(p[2]) > self.pot_bottom_z - 0.01
            and float(p[2]) < self.pot_rim_z - 0.008
        )

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self) -> None:
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_lift_step = None   # board first picked up
        self._metric_entry = {}         # veg index -> step it dropped into the pot
        self._metric_offset = {}        # veg index -> radial offset at that instant

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_veg_metrics(self) -> None:
        """Latch the first frame each piece is inside the pot, with where it entered.

        The radial offset must be read at entry: pieces roll and settle afterwards,
        so a pour that only just cleared the rim looks centered by the end.
        """
        try:
            for i, veg in enumerate(self.veggies):
                if i in self._metric_entry:
                    continue
                if not self._veg_in_pot(veg):
                    continue
                self._metric_entry[i] = self._metric_step()
                p = np.asarray(veg.get_pose().p, dtype=float)
                r = float(np.linalg.norm(p[:2] - np.asarray(self.pot_xy)))
                self._metric_offset[i] = r / max(float(self.pot_inner_radius), 1e-6)
        except Exception:
            pass

    def _compute_metrics(self) -> dict:
        """Human-experiment extras.

        extra1 `pour_latency_steps` — steps from the board being picked up until the
        LAST piece is in the pot. `first_entry_latency_steps` is the same clock for the
        first piece, so the spread shows how staggered the pour was.
        extra2 `pot_offset_norm` — mean radial distance from the pot axis at the moment
        each piece cleared the rim, over the pot's inner radius. LOWER is better;
        0 = poured straight down the middle, ->1 = scraping the pot wall.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a = self._metric_lift_step
        steps = sorted(self._metric_entry.values()) if self._metric_entry else []
        last = (None if (a is None or not steps)
                else max(int(steps[-1]) - int(a), 0))
        first = (None if (a is None or not steps)
                 else max(int(steps[0]) - int(a), 0))
        out["pour_latency_steps"] = last
        out["pour_latency_s"] = None if last is None else round(float(last) * dt, 4)
        out["first_entry_latency_steps"] = first
        out["pieces_in_pot"] = len(steps)
        try:
            out["pieces_total"] = int(len(self.veggies))
        except Exception:
            out["pieces_total"] = None

        offs = list(self._metric_offset.values())
        out["pot_offset_norm"] = round(sum(offs) / len(offs), 4) if offs else None
        out["worst_pot_offset_norm"] = round(max(offs), 4) if offs else None
        out["veg_fallen"] = bool(getattr(self, "_veg_fallen", False))
        out["arm_veg_contact"] = bool(getattr(self, "_arm_veg_contact", False))
        return out

    def _check_veg_fallen(self) -> None:
        """Fail if any piece leaves the board and settles outside the pot.

        Pieces still on the board or already in the pot are fine. Airborne
        pieces (pour mid-flight) wait; a settle near table height outside the
        pot mouth latches failure and ends the episode.
        """
        if (
            not getattr(self, "_fail_checks_armed", False)
            or getattr(self, "_veg_fallen", False)
            or not self.veggies
        ):
            return
        table_z = float(getattr(self, "table_top", 0.74))
        for veg in self.veggies:
            if self._veg_in_pot(veg) or self._veg_on_board(veg):
                continue
            p = np.asarray(veg.get_pose().p, dtype=float)
            # Still airborne above the apron — wait for a settle / pot entry.
            if float(p[2]) > table_z + 0.045:
                continue
            self._veg_fallen = True
            self._mark_fail("vegetables dropped on the table")
            return

    def _check_arm_veg_contact(self) -> None:
        """Fail if any robot link (arm / gripper) touches produce."""
        if (
            not getattr(self, "_fail_checks_armed", False)
            or getattr(self, "_arm_veg_contact", False)
            or not self.veggies
        ):
            return
        link_names = getattr(self, "_robot_link_names", None) or set()
        veg_names = getattr(self, "_veg_names", None) or {
            str(v.get_name()) for v in self.veggies if hasattr(v, "get_name")
        }
        if not link_names or not veg_names:
            return
        try:
            contacts = self.scene.get_contacts()
        except Exception:
            return
        for contact in contacts:
            try:
                name0 = contact.bodies[0].entity.name
                name1 = contact.bodies[1].entity.name
            except Exception:
                continue
            if (name0 in veg_names and name1 in link_names) or (
                name1 in veg_names and name0 in link_names
            ):
                self._arm_veg_contact = True
                self._mark_fail("arm contacted vegetables")
                return

    def _episode_failed(self) -> bool:
        return bool(self._veg_fallen or self._arm_veg_contact)

    # ---------------------------------------------------------------- per-step
    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return

        # Board is never welded — drop the held-arm marker if jaws open / slip.
        arm = self._board_hold_arm()
        if arm is not None and getattr(self, "_board_weld_arm", None) is not None:
            if self._board_gripper_open(arm) or not self._board_handle_near(arm):
                if self._board_gripper_open(arm):
                    print(f"[make_soup] board released (gripper opened, arm={arm})")
                self._clear_board_held()
                self._pour_armed = False

        # Produce stays dynamic on the board / in free-fall under PhysX.
        self._ensure_veggies_dynamic()
        self._check_veg_fallen()
        self._check_arm_veg_contact()
        self._track_veg_metrics()
        # Knob grasp / fire: KitchenS_base_task._update_stove_knob_control

    def _idle_steps(self, n_steps: int, until=None) -> None:
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            if until is not None and until():
                break
            if self._episode_failed():
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    # ---------------------------------------------------------------- expert motion
    def _top_down_pose(self, tcp_xyz) -> list[float]:
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]) + self.EE_TO_TCP,
            *[float(v) for v in GRASP_DIRECTION_DIC["top_down"]],
        ]

    def _tcp_pos(self, arm: ArmTag) -> np.ndarray:
        p = (
            self.robot.get_right_tcp_pose()
            if str(arm) == "right"
            else self.robot.get_left_tcp_pose()
        )
        return np.asarray(p[:3], dtype=float)

    def _handle_world(self) -> np.ndarray:
        """World position of the board handle center (grasp target)."""
        pose = self.board.get_pose()
        R = pose.to_transformation_matrix()[:3, :3]
        local = np.asarray(self._handle_local, dtype=float)
        # Contact is on the handle top.
        local = local + np.array([0.0, 0.0, float(self.handle_half[2])], dtype=float)
        return np.asarray(pose.p, dtype=float) + R @ local

    def _grasp_board(self) -> bool:
        """Top-down friction grasp of the handle — no EE weld / latch."""
        arm = self.arm
        self.plan_success = True
        self.move(self.open_gripper(arm))

        handle = self._handle_world()
        # One planned hover straight above the handle — no angled / side pinch.
        # Several hover heights and a small inward nudge give the planner room
        # to find a wrist configuration it can descend from.
        hover = False
        for z_off, dx, dy in (
            (0.16, 0.0, 0.0),
            (0.20, 0.0, 0.0),
            (0.16, -0.02, 0.02),
            (0.24, 0.0, 0.0),
        ):
            target = np.array(
                [handle[0] + dx, handle[1] + dy, handle[2] + z_off], dtype=float
            )
            self.plan_success = True
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(target))])
            )
            if self.plan_success:
                hover = True
                break
        if not hover:
            print("[make_soup] handle hover unreachable")
            return False

        # Descend on the handle with relative steps: a fresh absolute IK solve at
        # every height frequently fails from the hover configuration.
        goal = np.array(
            [handle[0], handle[1], float(handle[2]) + 0.012], dtype=float
        )
        for _ in range(4):
            delta = goal - self._tcp_pos(arm)
            if float(np.linalg.norm(delta)) < 0.008:
                break
            step = np.clip(delta, -0.07, 0.07)
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(step[0]),
                    y=float(step[1]),
                    z=float(step[2]),
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            if not self.plan_success:
                print("[make_soup] handle descent failed")
                return False
        self.move(self.close_gripper(arm, pos=0.0))
        self._idle_steps(10)

        dist = float(np.linalg.norm(self._tcp_pos(arm) - self._handle_world()))
        if dist > self.grasp_tcp_tol * 2.5:
            print(f"[make_soup] grasp missed — TCP far from handle ({dist:.3f} m)")
            return False
        if not self._board_gripper_closed(arm):
            print("[make_soup] grasp missed — gripper not closed on handle")
            return False

        self._mark_board_held(arm)
        print(f"[make_soup] grasped handle tcp-dist={dist:.3f} arm={arm} (friction)")
        return True

    def _seat_board_in_hand(self) -> None:
        """Deprecated no-op — do not teleport the board into the gripper."""
        return

    def _flatten_board(self) -> None:
        """Level the wrist (top-down) so a held board returns toward flat."""
        arm = self._board_hold_arm() or self.arm
        if self.board is None or arm is None:
            return
        if self._board_up_dot() > 0.995:
            return
        tcp = self._tcp_pos(arm)
        self.plan_success = True
        self.move(self.close_gripper(arm, pos=0.0))
        self.plan_success = True
        self.move(
            (arm, [Action(arm, "move", target_pose=self._top_down_pose(tcp))])
        )
        self._idle_steps(6)

    def _carry_board_level(self, target_xy, z: float) -> None:
        """Translate with top-down EE; board rides via the friction grasp."""
        arm = self.arm
        if self.board is None:
            return
        if not self._board_held(arm):
            print("[make_soup] carry aborted — board not held in gripper")
            return
        bp = np.asarray(self.board.get_pose().p, dtype=float)
        dx = float(target_xy[0] - bp[0])
        dy = float(target_xy[1] - bp[1])
        dz = float(z - bp[2])
        dist = float(np.linalg.norm([dx, dy, dz]))
        steps = int(np.clip(np.ceil(dist / 0.04), 4, 10))
        for _ in range(steps):
            if not self._board_held(arm):
                print("[make_soup] board slipped during carry")
                break
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=dx / steps,
                    y=dy / steps,
                    z=dz / steps,
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            if self._veg_fallen and not self._pour_armed:
                break

    def _nudge_board_to(self, target_xy, z: float, tol: float = 0.004) -> None:
        """Close a small carry error with pure translation."""
        arm = self.arm
        for _ in range(2):
            if not self._board_held(arm):
                return
            bp = np.asarray(self.board.get_pose().p, dtype=float)
            delta = np.array(
                [float(target_xy[0]) - bp[0], float(target_xy[1]) - bp[1], z - bp[2]],
                dtype=float,
            )
            if float(np.linalg.norm(delta)) < tol:
                return
            delta = np.clip(delta, -0.05, 0.05)
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(delta[0]),
                    y=float(delta[1]),
                    z=float(delta[2]),
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )

    @staticmethod
    def _rot_about_y(pose: sapien.Pose, pivot, angle: float) -> sapien.Pose:
        """Rigidly rotate ``pose`` about the world-Y axis through ``pivot``."""
        piv = np.asarray(pivot, dtype=float)
        rot = sapien.Pose(
            [0.0, 0.0, 0.0],
            [float(v) for v in euler2quat(0.0, float(angle), 0.0, axes="sxyz")],
        )
        local = sapien.Pose(
            (np.asarray(pose.p, dtype=float) - piv).tolist(),
            [float(v) for v in pose.q],
        )
        out = rot * local
        return sapien.Pose(
            (np.asarray(out.p, dtype=float) + piv).tolist(),
            [float(v) for v in out.q],
        )

    def _pour_into_pot(self) -> None:
        """Carry the board level to the pot rim, then roll the wrist ~40°.

        The board stays dynamic in the closed gripper (friction grasp). The pour
        tip is EE wrist motion only — no board latch / set_pose drive.
        """
        arm = self.arm
        pot = np.asarray(self.pot_xy, dtype=float)
        hx = float(self.board_half[0])
        side_sign = 1.0 if str(arm) == "right" else -1.0
        tip_max = float(self.pour_tip_rad)

        # Board beside the pot: pour edge just over the pot center-line so a
        # 40° wrist tip drops pieces into the mouth.
        side_xy = np.array(
            [
                float(pot[0] + side_sign * (hx - 0.25 * float(self.pot_inner_radius))),
                float(pot[1]),
            ],
            dtype=float,
        )
        hover_z = float(self.pot_rim_z) + 0.028

        self._carry_board_level(side_xy, hover_z)
        self._flatten_board()
        self._nudge_board_to(side_xy, hover_z)
        self._flatten_board()
        pour_flat = self.board.get_pose()
        print(
            f"[make_soup] pour pose board={np.round(np.asarray(pour_flat.p), 4)} "
            f"edge={float(pour_flat.p[0]) - side_sign * hx:.3f} pot={pot} arm={arm}"
        )

        self._pour_armed = True
        self._force_veg_hold = False
        self._score_veg_spill = True

        def in_pot() -> bool:
            return all(self._veg_in_pot(v) for v in self.veggies)

        self._release_veggies_physics(settle_on_board=True)

        # Tip: rotate the wrist about the board pour edge; board follows via grasp.
        ee_flat = self._ee_pose(arm)
        pivot = np.asarray(pour_flat.p, dtype=float)
        tip_moves = 8
        save_freq = self.save_freq if self.save_freq is not None else 15

        def _ee_at(frac: float) -> list[float]:
            theta = -side_sign * tip_max * frac
            drift = np.array(
                [-side_sign * 0.010 * frac, 0.0, -0.008 * frac], dtype=float
            )
            ee = self._rot_about_y(ee_flat, pivot, theta)
            ee = sapien.Pose(
                (np.asarray(ee.p, dtype=float) + drift).tolist(),
                [float(v) for v in ee.q],
            )
            return [*[float(v) for v in ee.p], *[float(v) for v in ee.q]]

        tip_i = 0
        for i in range(1, tip_moves + 1):
            tip_i = i
            if not self._board_held(arm):
                print("[make_soup] board slipped during pour tip")
                break
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(self.move_to_pose(arm, _ee_at(i / tip_moves)))
            self.plan_success = True
            self._idle_steps(10)
            if (
                self.save_data
                and save_freq
            ):
                try:
                    self._update_render()
                    self._take_picture()
                except Exception:
                    pass
            self._check_veg_fallen()
            if in_pot():
                break

        for _ in range(120):
            if in_pot():
                break
            self.scene.step()
            self._check_veg_fallen()

        print(
            f"[make_soup] after tip in_pot="
            f"{sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
        )
        self._ignore_board_veg_collision()

        # Untilt via wrist only (board still pinched if held).
        untilt_steps = max(1, tip_i)
        for j in range(tip_i, -1, -max(1, untilt_steps // 4)):
            if not self._board_held(arm):
                break
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(self.move_to_pose(arm, _ee_at(j / tip_moves)))
            self.plan_success = True
        if self._board_held(arm):
            self.plan_success = True
            self.move(self.move_to_pose(arm, _ee_at(0.0)))
            self.plan_success = True
        self._mark_board_held(arm)
        self._idle_steps(4)
        self.move(
            self.move_by_displacement(
                arm,
                x=float(side_sign * 0.05),
                z=0.05,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self.plan_success = True
        self._pour_armed = False
        self._force_veg_hold = False

    def _place_board_on_table(self) -> None:
        """Carry board back while held; opening the gripper drops it on the table."""
        arm = self.arm
        if self.board is None:
            return
        if not self._board_held(arm):
            print("[make_soup] place aborted — board not held")
            return

        place_xy = np.array(
            [float(self.board_xy[0]), float(self.board_xy[1])], dtype=float
        )
        # Keep clear of the pot / cooktop apron.
        pot = np.asarray(self.pot_xy, dtype=float)
        if float(np.linalg.norm(place_xy - pot)) < 0.18:
            place_xy[1] = min(float(place_xy[1]), -0.12)

        hover_z = float(self.table_top) + 0.08
        place_z = float(self.table_top) + float(self.board_half[2]) + 0.003

        self._carry_board_level(place_xy, hover_z)
        self._nudge_board_to(place_xy, hover_z)
        self._idle_steps(4)

        self._carry_board_level(place_xy, place_z)
        self._nudge_board_to(place_xy, place_z)
        self._idle_steps(6)

        self.move(self.open_gripper(arm))
        self._idle_steps(2)
        self._clear_board_held()
        for _ in range(48):
            self.scene.step()
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_linear_velocity(np.zeros(3))
                self._board_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        self.move(
            self.move_by_displacement(
                arm,
                z=0.12,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self.plan_success = True
        print(
            f"[make_soup] placed board at "
            f"{np.round(np.asarray(self.board.get_pose().p), 3)} arm={arm}"
        )

    def _spill_veggies_on_table(self) -> None:
        """Failure demo: tip the board over the table so produce slides off."""
        arm = self.arm
        side_sign = 1.0 if str(arm) == "right" else -1.0
        tip_max = float(np.deg2rad(55.0))

        spill_xy = np.array(
            [float(self.board_xy[0]) - side_sign * 0.02, float(self.board_xy[1]) + 0.04],
            dtype=float,
        )
        pot = np.asarray(self.pot_xy, dtype=float)
        if float(np.linalg.norm(spill_xy - pot)) < 0.18:
            spill_xy[1] = min(float(spill_xy[1]), -0.10)
        hover_z = float(self.table_top) + 0.11
        self._carry_board_level(spill_xy, hover_z)
        self._idle_steps(6)

        pour_flat = self.board.get_pose()
        ee_flat = self._ee_pose(arm)
        pivot = np.asarray(pour_flat.p, dtype=float)
        self._score_veg_spill = True
        self._release_veggies_physics(settle_on_board=True)

        tip_moves = 8

        def _ee_at(frac: float) -> list[float]:
            theta = side_sign * tip_max * frac
            ee = self._rot_about_y(ee_flat, pivot, theta)
            return [*[float(v) for v in ee.p], *[float(v) for v in ee.q]]

        for i in range(1, tip_moves + 1):
            if not self._board_held(arm):
                break
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(self.move_to_pose(arm, _ee_at(i / tip_moves)))
            self.plan_success = True
            self._idle_steps(8)
            self._check_veg_fallen()

        for _ in range(160):
            self.scene.step()
            self._check_veg_fallen()

        print(
            f"[make_soup] force_spill fallen={self._veg_fallen} "
            f"in_pot={sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
        )

    def play_once(self) -> dict[str, Any]:
        arm = self.arm
        self.plan_success = True

        if not self._grasp_board():
            self.plan_success = False
            return self.info

        # Level lift — board rides in the closed gripper under contact physics.
        for _ in range(4):
            if self._episode_failed():
                self.plan_success = False
                return self.info
            if not self._board_held(arm):
                print("[make_soup] board slipped during lift")
                self.plan_success = False
                return self.info
            self.plan_success = True
            self.move(self.close_gripper(arm, pos=0.0))
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    z=0.025,
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            for _ in range(4):
                self._update_kinematic_tasks()
                self.scene.step()
        self._flatten_board()
        self._ensure_veggies_dynamic()
        self._check_veg_fallen()
        self._check_arm_veg_contact()
        if self._episode_failed() or not self._veggies_on_board():
            if not self._veg_fallen and not self._veggies_on_board():
                self._veg_fallen = True
                self._mark_fail("vegetables dropped on the table")
            print("[make_soup] veggies fell during lift — fail")
            self.plan_success = False
            return self.info

        # Optional failure demo: tip carelessly over the table (produce spills).
        if getattr(self, "force_spill", False):
            self._spill_veggies_on_table()
            self.plan_success = False
            self.info["info"] = {
                "{A}": "chopping_board",
                "{B}": "soup_pot",
                "{C}": "cooking_range",
                "{E}": "vegetables",
                "{a}": str(arm),
            }
            return self.info

        self._pour_into_pot()
        if self._episode_failed() or not all(self._veg_in_pot(v) for v in self.veggies):
            if not all(self._veg_in_pot(v) for v in self.veggies) and not self._episode_failed():
                self._idle_steps(40)
            if self._episode_failed() or not all(self._veg_in_pot(v) for v in self.veggies):
                print(
                    f"[make_soup] pour incomplete fallen={self._veg_fallen} "
                    f"arm_touch={self._arm_veg_contact} "
                    f"in_pot={sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
                )
                self.plan_success = False
                return self.info

        # Board goes back on the table — never left on the pot.
        self._place_board_on_table()

        if self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": "chopping_board",
            "{B}": "soup_pot",
            "{C}": "cooking_range",
            "{E}": "vegetables",
            "{a}": str(getattr(self, "board_arm", arm)),
        }
        return self.info

    def check_success(self) -> bool:
        """Success = every vegetable piece is inside the pot (spill / arm touch fails).

        The burner starts on; scored criterion is produce containment only.
        """
        if not getattr(self, "_loaded", False):
            return False
        if not self.veggies:
            return False
        self._check_veg_fallen()
        self._check_arm_veg_contact()
        if self._episode_failed():
            return False
        if not all(self._veg_in_pot(v) for v in self.veggies):
            return False
        return True

    def get_score(self) -> float:
        """Fraction of vegetables currently in the pot.

        A dropped / spilled piece is simply not in the pot, so it lowers the
        fraction (e.g. 1 of 2 in the pot → 0.5) rather than zeroing the score.
        Arm-vegetable contact is an illegal manipulation and still forces 0.
        """
        if not getattr(self, "_loaded", False):
            return 0.0
        if not self.veggies:
            return 0.0
        self._check_veg_fallen()
        self._check_arm_veg_contact()
        if bool(getattr(self, "_arm_veg_contact", False)):
            return 0.0
        n = len(self.veggies)
        if n <= 0:
            return 0.0
        n_in = sum(1 for v in self.veggies if self._veg_in_pot(v))
        return float(n_in) / float(n)

    def get_obs(self) -> dict[str, Any]:
        obs = super().get_obs()
        n_in = sum(1 for v in self.veggies if self._veg_in_pot(v)) if self.veggies else 0
        obs["soup"] = {
            "stove_on": bool(self.stove_on),
            "veg_released": bool(self._veg_released),
            "veg_fallen": bool(self._veg_fallen),
            "arm_veg_contact": bool(self._arm_veg_contact),
            "fail_reason": str(getattr(self, "_fail_reason", "") or ""),
            "board_up_dot": float(self._board_up_dot()),
            "n_veg": int(len(self.veggies)),
            "n_in_pot": int(n_in),
            "all_in_pot": bool(n_in == len(self.veggies)) if self.veggies else False,
            "range_xy": list(np.asarray(getattr(self, "range_xy", (0, 0)), dtype=float)),
            "board_xy": list(np.asarray(getattr(self, "board_xy", (0, 0)), dtype=float)),
            "pot_xy": list(np.asarray(getattr(self, "pot_xy", (0, 0)), dtype=float)),
            "water_level": float(getattr(self, "water_level", 0.0)),
            "partial_score": float(self.get_score()),
        }
        return obs
