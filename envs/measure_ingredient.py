"""Push a marked jar under an oil nozzle and fill it to a target ring.

KitchenS prep-counter scene (no sink / tap / stove): silver oil dispenser, glass
jar marked with red rings at 25% / 50% / 75%, electronic scale (scene prop), and
baking props. The robot pushes the jar under the nozzle
(``catch_cup`` pillow-style contact shove),
presses the green key to latch oil ON (key turns red), then presses OFF at the
target fill. Oil that misses the jar mouth (jar not under the nozzle, or pushed
past it) puddles on the table and fails the episode. The scale is not required
for success.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.partial_score import score_descending_bands


class measure_ingredient(KitchenS_base_task):
    """Push the jar under the nozzle and fill it to a target ring."""

    # Abs error (pp) for 25/50/75% targets: [10,8.5) / [8.5,7) / [7,5) → 0.25/0.5/0.75.
    PARTIAL_ABS_BANDS = (
        (10.0, 8.5, 0.25),
        (8.5, 7.0, 0.5),
        (7.0, 5.0, 0.75),
    )
    # Abs error (pp) for 100% target: [13,12) / [12,11) / [11,10) → 0.25/0.5/0.75.
    PARTIAL_ABS_BANDS_FULL = (
        (13.0, 12.0, 0.25),
        (12.0, 11.0, 0.5),
        (11.0, 10.0, 0.75),
    )

    EGG_ORANGE = [0.95, 0.48, 0.10, 1.0]
    YUP_Q = [0.5, 0.5, 0.5, 0.5]
    BOARD_QPOS = [0.707, 0.707, 0.0, 0.0]
    # +30% vs the original 0.07 baking-board scale.
    BOARD_SCALE_DEFAULT = 0.091
    MICROWAVE_SCALE_DEFAULT = 1.3

    JAR_MODEL = "253_glass_jar"

    # Geometry (matches fill_coffee_jar jar; dispenser is a silver oil can).
    JAR_INNER_R = 0.035
    JAR_HEIGHT = 0.125
    JAR_BOTTOM_T = 0.005
    JAR_MASS = 0.22
    # Jar spawns this far toward the robot (−Y) from the fill/nozzle target.
    # Long enough for a visible contact shove; short enough that the approach
    # still fits behind the jar on the KitchenS near edge (no end teleport).
    JAR_START_GAP = 0.08
    # Nozzle must land within this radius of the jar center to fill (else spill).
    JAR_CATCH_R = 0.028

    BODY_R = 0.040
    BODY_H = 0.155
    DOME_R = 0.040
    PEDESTAL_HALF = (0.052, 0.052, 0.045)
    PLATFORM_HALF = (0.055, 0.055, 0.007)

    NOZZLE_R = 0.006
    # Push-key on the nozzle arm (fill_coffee-style press; latches on/off).
    # Key stays depressed while ON and springs back up when toggled OFF.
    # Green when up (OFF), red when depressed (ON) — same as cook_meat.
    SWITCH_BASE_HALF = (0.014, 0.014, 0.004)    # dark housing
    SWITCH_BTN_HALF = (0.010, 0.010, 0.012)     # key cap (travel ≈ half-z)
    SWITCH_COLOR_UP = [0.18, 0.78, 0.28]         # green when off / up
    SWITCH_COLOR_DOWN = [0.85, 0.10, 0.10]       # red when on / pressed
    SWITCH_RED = SWITCH_COLOR_DOWN               # legacy alias
    SWITCH_TOUCH_XY_TOL = 0.045
    SWITCH_FORCE_STIFFNESS = 800.0              # N/m spring proxy (fill_coffee)
    SWITCH_FORCE_ENGAGE_SLACK = 0.045           # m above key top where force starts
    SWITCH_ENGAGE_FORCE = 0.5                   # N; edge-trigger threshold
    SWITCH_BUTTON_VISUAL_STEP = 0.0007

    EE_TO_TCP = 0.12
    KEY_HOVER_DIS = 0.06
    KEY_PRESS_DEPTH = 0.024                     # expert depress from hover

    # catch_cup-style contact push (closed gripper shoves the jar under nozzle).
    # Gravity + table Coulomb friction brake the slide; lin damp is residual only.
    PUSH_CONTACT_GAP = 0.014
    PUSH_FINGER_DROP = 0.043
    PUSH_EDGE_MARGIN = 0.025
    PUSH_BEHIND_STANDOFF = 0.07
    PUSH_FINGER_HEIGHT_FRAC = 0.40
    PUSH_LIN_DAMP = 0.6
    PUSH_MU_STATIC = 0.95
    PUSH_MU_DYNAMIC = 0.85
    PUSH_STEP_DEFAULT = 0.040
    PUSH_PLACE_TOL = 0.025
    # Near table edge (KitchenS counter ~0.6 m deep, robot at −Y).
    TABLE_NEAR_Y = -0.34

    # Ring marks at 25% / 50% / 75%; 100% = jar rim (no extra ring).
    FILL_LEVELS = (0.25, 0.50, 0.75, 1.0)
    FILL_TOL = 0.05             # ±5% absolute fill tolerance
    # Full (100%) target cannot exceed 1.0, so success is one-sided: [90%, 100%].
    FILL_FULL_LO = 0.90
    # Slow enough that opening the switch does not already overshoot the mark;
    # oil still rises continuously while the switch stays on.
    POUR_RATE = 0.00022         # fill fraction per physics step while switch on
    # Relative half-width for sampling pour speed: rate ∼ Uniform(1±jitter)*pour_rate.
    POUR_RATE_JITTER = 0.15
    # Jar is full at 1.0; further pour (switch still ON) spills onto the table.
    OVERFLOW_LEVEL = 1.0
    # Layout randomization defaults (see task_args.measure_ingredient).
    MICROWAVE_Y_DEFAULT = 0.18  # stay at the back; only X diversifies
    # Two discrete MW poses when randomize_layout: top-left (current) or left corner.
    MICROWAVE_TOP_LEFT = (-0.32, 0.18)
    MICROWAVE_LEFT_CORNER = (-0.48, 0.22)
    # Keep MW out of the left-arm pour lane (see _sample_microwave_override).
    MICROWAVE_X_RANGE = (-0.42, -0.22)
    STATION_X_RANGE = (-0.20, -0.02)   # left-arm pour station (legacy / left side)
    STATION_X_RANGE_LEFT = (-0.22, -0.04)
    STATION_X_RANGE_RIGHT = (0.04, 0.22)
    # Dispenser Y jitter about baseline ``disp_y`` (±6 cm).
    DISP_Y_JITTER = 0.06
    # Scale (scene prop only — not part of the success path).
    SCALE_DIST_BASE = 0.25
    SCALE_DIST_JITTER = 0.10
    SCALE_NEAR_DX_RANGE = (-0.28, -0.16)
    SCALE_NEAR_DY_RANGE = (-0.12, -0.02)
    DECOR_ON_MICROWAVE_PROB = 0.35
    # Axis-aligned footprint half-sizes (m) + gap used for non-overlap layout.
    LAYOUT_MARGIN = 0.035
    DISP_HALF_XY = (0.058, 0.058)       # oil-can pedestal / body
    JAR_HALF_XY = (0.042, 0.042)
    SCALE_HALF_XY = (0.095, 0.075)
    DECOR_HALF_XY = {
        "bread": (0.12, 0.085),          # cutting board + bread
        "flour": (0.075, 0.075),
        "chips": (0.075, 0.055),
        "bowl": (0.085, 0.085),
    }
    # Spill puddle grows while oil misses the jar or the jar overfills.
    SPILL_RATE = 0.00045        # spill_amount per physics step while overflowing
    SPILL_RADIUS_MIN = 0.035    # m; first visible puddle
    SPILL_RADIUS_MAX = 0.095    # m; fully spilled puddle
    SPILL_HALF_H = 0.0018       # m; flat disk thickness on the table

    # Oil look (``oil_style`` task_arg):
    #   solid       — opaque sunflower-yellow oil (default)
    #   transparent — see-through amber (glass-jar recipe)
    OIL_STYLE_DEFAULT = "solid"
    OIL_COLOR_TRANSPARENT = [0.62, 0.58, 0.22, 0.22]
    OIL_STREAM_TRANSPARENT = [0.60, 0.56, 0.18, 0.18]
    OIL_MENISCUS_TRANSPARENT = [0.58, 0.54, 0.16, 0.28]
    # Opaque dark yellow / amber oil (reads clearly through glass).
    OIL_COLOR_SOLID = [0.62, 0.44, 0.04, 0.98]
    OIL_STREAM_SOLID = [0.60, 0.42, 0.03, 0.95]
    # Same yellow family for the table spill (slightly more opaque).
    OIL_SPILL_SOLID = [0.62, 0.44, 0.04, 0.98]
    OIL_SPILL_TRANSPARENT = [0.62, 0.58, 0.22, 0.60]
    UPRIGHT_CYL_Q = [0.70710678, 0.0, -0.70710678, 0.0]
    SILVER = [0.78, 0.80, 0.84, 1.0]
    SILVER_DARK = [0.55, 0.57, 0.60, 1.0]
    # Saturated opaque red fill mark (matches fill_coffee_jar).
    RING_RED = [1.0, 0.04, 0.02]
    RING_MESH_RADIUS = 0.0388
    RING_XY_SCALE = 1.02
    RING_Z_SCALE = 3.2
    # Cooler / less-white glass (same as fill_coffee_jar).
    GLASS = [0.72, 0.84, 0.92, 0.16]
    # Interactive viewer look (matches trap_bug plain trap): no transmission/IOR.
    # Note: the jar itself always uses transmission glass (see ``_jar_glass_material``).
    PLAIN_GLASS = [0.14, 0.26, 0.40, 0.55]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("measure_ingredient", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        # Bare baking counter: remove sink, faucet tap, and stove.
        self.clear_sink_and_range = True
        self.replace_sink_with_range = False
        self.microwave_scale_mult = float(
            self._cfg.get("microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
        )
        # Microwave pose must be chosen before super() builds the kitchen.
        self._layout_seed = int(kwags.get("seed", 0) or 0)
        self._sample_microwave_override(self._cfg, self._layout_seed)

        # Per-step state before early _update_kinematic_tasks (camera init).
        self._loaded = False
        self.tab_open = False
        self.liquid_level = 0.0
        self.overflowed = False
        self.spill_amount = 0.0
        self.opened_once = False
        self.closed_after_pour = False
        self._reset_metric_state()
        self.jar_on_scale = False
        self._episode_jar_released = False
        self.target_fill = 0.25
        self._decor_layout = {}
        self._apply_oil_style(self._parse_oil_style(self._cfg))
        self._plain_glass = bool(self._cfg.get("plain_glass", False))
        self._liquid_entity = None
        self._stream_entity = None
        self._spill_entity = None
        self._spill_radius_cached = -1.0
        self._spill_xy = None
        self._liquid_half_h_cached = -1.0
        self._switch_parts = []
        self._switch_btn = None
        self._switch_key_shapes = []
        self._switch_key_color_down = None
        self._button_home_pose = None
        self._button_visual_depth = 0.0
        self._button_target_depth = 0.0
        self._button_pressed_visual = False
        self._pressing_arm_side = ""
        self._ring_entities = []
        self._touch_latched = False
        self._ignore_tab = False
        # OFF is applied on release so oil keeps flowing until the key is up.
        self._pending_tab_off = False
        self._jar_locked = True
        self._jar_carry = False
        self._jar_carry_local = None  # TCP→jar world offset (upright hold)
        self._jar_carry_arm = None
        self._jar_carry_quat = None
        self._jar_seated_pose = None
        self._push_active = False
        self.fill_xy = None
        self.jar = None
        self.jar_visual = None
        self.scale = None
        self.table_top = 0.74

        super().setup_demo(**kwags)
        self._configure_observer_camera()

    def _layout_rng(self, salt: int = 0) -> np.random.RandomState:
        return np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + int(salt))

    def _parse_range(self, cfg, key, default):
        raw = cfg.get(key, default)
        if raw is None:
            raw = default
        return (float(raw[0]), float(raw[1]))

    def _estimate_microwave_half_xy(self, scale_mult=None):
        """World-XY half extents of the yawed microwave collision box."""
        if scale_mult is None:
            scale_mult = float(
                getattr(self, "microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
            )
        scale = 0.15 * float(scale_mult)
        # Mesh bounds before yaw; after +90° Z, (x,y) → (−y, x).
        bounds_min = np.array([-0.761475, -0.495360, -0.567261], dtype=float)
        bounds_max = np.array([0.795292, 0.435315, 0.630773], dtype=float)
        half_x = 0.5 * (bounds_max[1] - bounds_min[1]) * scale
        half_y = 0.5 * (bounds_max[0] - bounds_min[0]) * scale
        return np.array([float(half_x), float(half_y)], dtype=float)

    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin=0.0):
        """True if two axis-aligned footprints overlap (with optional gap)."""
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        h2 = np.asarray(h2, dtype=float)
        m = float(margin)
        return bool(
            abs(c1[0] - c2[0]) < (h1[0] + h2[0] + m)
            and abs(c1[1] - c2[1]) < (h1[1] + h2[1] + m)
        )

    def _footprint_clear(self, center, half, blockers, margin=None):
        """Return True if ``center/half`` does not overlap any blocker AABB."""
        if margin is None:
            margin = self.LAYOUT_MARGIN
        for b_c, b_h in blockers:
            if self._aabb_overlap(center, half, b_c, b_h, margin=margin):
                return False
        return True

    def _microwave_pose_for_layout(self, cfg):
        """Return (mw_xy, mw_half) used for collision-aware placement."""
        mw_xy = getattr(self, "microwave_xy", None)
        if mw_xy is None:
            mw_xy = getattr(self, "microwave_xy_override", None)
        if mw_xy is None:
            mw_xy = [-0.32, float(cfg.get("microwave_y", self.MICROWAVE_Y_DEFAULT))]
        mw_xy = np.asarray(mw_xy, dtype=float)
        mw_half = getattr(self, "microwave_half_xy", None)
        if mw_half is None:
            mw_half = self._estimate_microwave_half_xy()
        mw_half = np.asarray(mw_half, dtype=float)
        return mw_xy, mw_half

    def _station_clears_microwave(self, side_x, disp_y, jar_y, mw_xy, mw_half, margin=None):
        if margin is None:
            margin = self.LAYOUT_MARGIN
        disp = np.array([side_x, disp_y], dtype=float)
        jar = np.array([side_x, jar_y], dtype=float)
        if self._aabb_overlap(disp, self.DISP_HALF_XY, mw_xy, mw_half, margin):
            return False
        if self._aabb_overlap(jar, self.JAR_HALF_XY, mw_xy, mw_half, margin):
            return False
        return True

    def _nudge_x_clear_of_microwave(self, side_x, disp_y, jar_y, mw_xy, mw_half, x_lo, x_hi):
        """Push station X away from the microwave until footprints clear."""
        need = (
            float(mw_half[0])
            + max(float(self.DISP_HALF_XY[0]), float(self.JAR_HALF_XY[0]))
            + float(self.LAYOUT_MARGIN)
            + 1e-3
        )
        # Prefer the side that still lies inside [x_lo, x_hi].
        left_x = float(mw_xy[0]) - need
        right_x = float(mw_xy[0]) + need
        candidates = []
        if x_lo <= left_x <= x_hi:
            candidates.append(left_x)
        if x_lo <= right_x <= x_hi:
            candidates.append(right_x)
        # Also try clamping the preferred direction from the current sample.
        if side_x <= mw_xy[0]:
            candidates.append(float(np.clip(left_x, x_lo, x_hi)))
        else:
            candidates.append(float(np.clip(right_x, x_lo, x_hi)))
        for x in candidates:
            if self._station_clears_microwave(x, disp_y, jar_y, mw_xy, mw_half):
                return float(x)
        # Last resort: pick the in-range X maximizing |x - mw_x|.
        grid = np.linspace(x_lo, x_hi, 25)
        best = float(side_x)
        best_d = -1.0
        for x in grid:
            if not self._station_clears_microwave(x, disp_y, jar_y, mw_xy, mw_half):
                continue
            d = abs(float(x) - float(mw_xy[0]))
            if d > best_d:
                best, best_d = float(x), d
        return best

    def _sample_microwave_override(self, cfg, seed: int):
        """Microwave: top-left (current) or left corner when layout is randomized.

        Reject poses that leave no collision-free pour-station placement on
        either the left or right arm half.
        """
        mw_y = float(cfg.get("microwave_y", self.MICROWAVE_Y_DEFAULT))
        randomize = bool(cfg.get("randomize_layout", False))
        if "microwave_x" in cfg and cfg.get("microwave_x") is not None:
            self.microwave_xy_override = [float(cfg["microwave_x"]), mw_y]
            return
        if not randomize:
            self.microwave_xy_override = None
            return

        top_left = cfg.get("microwave_top_left", list(self.MICROWAVE_TOP_LEFT))
        left_corner = cfg.get(
            "microwave_left_corner", list(self.MICROWAVE_LEFT_CORNER)
        )
        pose_name = str(cfg.get("microwave_pose", "random")).lower().strip()
        rng = np.random.RandomState(int(seed) + 17)
        if pose_name in ("top_left", "top-left", "current", "default"):
            candidates = [("top_left", top_left)]
        elif pose_name in ("left_corner", "left-corner", "corner"):
            candidates = [("left_corner", left_corner)]
        else:
            order = ["top_left", "left_corner"]
            rng.shuffle(order)
            pose_map = {"top_left": top_left, "left_corner": left_corner}
            candidates = [(n, pose_map[n]) for n in order]

        base_disp_y = float(cfg.get("disp_y", 0.08))
        base_jar_y = float(cfg.get("jar_y", -0.06))
        mw_half = self._estimate_microwave_half_xy(
            cfg.get("microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
        )
        # Station may sit on either arm half — check both ranges.
        station_ranges = [
            self._parse_range(cfg, "station_x_range_left", self.STATION_X_RANGE_LEFT),
            self._parse_range(cfg, "station_x_range_right", self.STATION_X_RANGE_RIGHT),
            self._parse_range(cfg, "station_x_range", self.STATION_X_RANGE),
        ]
        for name, pose in candidates:
            mw_xy = np.array([float(pose[0]), float(pose[1])], dtype=float)
            ok = False
            for sx_lo, sx_hi in station_ranges:
                for x in np.linspace(sx_lo, sx_hi, 21):
                    if self._station_clears_microwave(
                        float(x), base_disp_y, base_jar_y, mw_xy, mw_half
                    ):
                        ok = True
                        break
                if ok:
                    break
            if ok:
                self.microwave_xy_override = [float(mw_xy[0]), float(mw_xy[1])]
                self._microwave_pose_choice = name
                return
        # Fallback: current top-left.
        self.microwave_xy_override = [float(top_left[0]), float(top_left[1])]
        self._microwave_pose_choice = "top_left"

    def _resolve_target_fill(self, cfg, rng: np.random.RandomState) -> float:
        """Parse target_fill: 0.25/0.5/0.75/1.0 or ``random``.

        ``randomize_target_fill: true`` forces a choice from ``fill_levels``
        (default ``[0.25, 0.50, 0.75, 1.0]``), same as ``target_fill: random``.
        """
        levels = cfg.get("fill_levels", self.FILL_LEVELS)
        choices = [float(x) for x in levels]
        for c in choices:
            if c not in self.FILL_LEVELS:
                raise ValueError(
                    f"fill_levels entry {c} not in {self.FILL_LEVELS}"
                )
        randomize = bool(cfg.get("randomize_target_fill", False))
        tf = cfg.get("target_fill", 0.25)
        if randomize or (isinstance(tf, str) and tf.strip().lower() == "random"):
            return float(rng.choice(choices))
        level = 0.25 if tf is None else float(tf)
        if level not in self.FILL_LEVELS:
            raise ValueError(f"target_fill must be one of {self.FILL_LEVELS} or 'random'")
        return level

    def _sample_free_xy_aabb(
        self, rng, half, blockers, x_range, y_range, tries=60, margin=None
    ):
        """Sample a table XY whose footprint clears all blocker AABBs."""
        if margin is None:
            margin = self.LAYOUT_MARGIN
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        y_lo, y_hi = float(y_range[0]), float(y_range[1])
        half = np.asarray(half, dtype=float)
        for _ in range(int(tries)):
            p = np.array(
                [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float
            )
            if self._footprint_clear(p, half, blockers, margin=margin):
                return p
        # Prefer the open right / front counter if the full box is crowded.
        safe_boxes = [
            (max(x_lo, 0.12), x_hi, y_lo, min(y_hi, 0.06)),
            (x_lo, x_hi, y_lo, y_hi),
        ]
        best, best_score = None, -1e9
        for bx0, bx1, by0, by1 in safe_boxes:
            if bx1 <= bx0 or by1 <= by0:
                continue
            for _ in range(80):
                p = np.array(
                    [rng.uniform(bx0, bx1), rng.uniform(by0, by1)], dtype=float
                )
                if self._footprint_clear(p, half, blockers, margin=margin):
                    return p
                score = min(
                    min(
                        abs(p[0] - b_c[0]) - (half[0] + b_h[0] + margin),
                        abs(p[1] - b_c[1]) - (half[1] + b_h[1] + margin),
                    )
                    for b_c, b_h in blockers
                ) if blockers else 1.0
                if score > best_score:
                    best, best_score = p, score
        if best is not None and best_score >= 0.0:
            return best
        # Last resort: park on the far-right front corner.
        return np.array([min(x_hi, 0.42), max(y_lo, -0.22)], dtype=float)

    def _resolve_layout(self, cfg):
        """Dispenser + fill target + jar-in-front spawn, sparse/on-microwave decor.

        With ``randomize_layout: true``:
          - microwave is top-left or left-corner (chosen in setup)
          - dispenser on left *or* right (clears MW), Y = baseline ± 6 cm
          - jar starts toward the robot; fill/nozzle target is under the spout
          - baking decor randomized with AABB clearance + push corridor
        """
        rng = self._layout_rng(101)
        randomize = bool(cfg.get("randomize_layout", False))
        mw_xy, mw_half = self._microwave_pose_for_layout(cfg)
        mw_top = getattr(self, "microwave_top_z", None)
        if mw_top is None:
            mw_top = self.table_top + 0.22
        mw_top = float(mw_top)
        table_z = float(self.table_top + 0.001)

        base_disp_y = float(cfg.get("disp_y", 0.08))
        # Fill target sits under the nozzle; jar starts further toward the robot.
        base_fill_y = float(cfg.get("fill_y", cfg.get("jar_y", -0.06)))
        gap = float(np.clip(base_disp_y - base_fill_y, 0.10, 0.18))
        jar_start_gap = float(cfg.get("jar_start_gap", self.JAR_START_GAP))
        disp_y_jitter = float(cfg.get("disp_y_jitter", self.DISP_Y_JITTER))

        if randomize:
            disp_y = float(
                rng.uniform(base_disp_y - disp_y_jitter, base_disp_y + disp_y_jitter)
            )
            fill_y = disp_y - gap
            jar_y = fill_y - jar_start_gap

            left_range = self._parse_range(
                cfg, "station_x_range_left", self.STATION_X_RANGE_LEFT
            )
            right_range = self._parse_range(
                cfg, "station_x_range_right", self.STATION_X_RANGE_RIGHT
            )
            side_pref = str(cfg.get("station_side", "random")).lower().strip()
            if side_pref in ("left", "l"):
                side_order = ["left"]
            elif side_pref in ("right", "r"):
                side_order = ["right"]
            else:
                side_order = ["left", "right"]
                rng.shuffle(side_order)

            side_x = None
            for side_name in side_order:
                sx_lo, sx_hi = left_range if side_name == "left" else right_range
                for _ in range(40):
                    cand = float(rng.uniform(sx_lo, sx_hi))
                    if self._station_clears_microwave(
                        cand, disp_y, jar_y, mw_xy, mw_half
                    ):
                        side_x = cand
                        break
                if side_x is not None:
                    break
            if side_x is None:
                sx_lo, sx_hi = left_range
                side_x = self._nudge_x_clear_of_microwave(
                    float(cfg.get("station_x", -0.08)),
                    disp_y,
                    jar_y,
                    mw_xy,
                    mw_half,
                    sx_lo,
                    sx_hi,
                )
        else:
            disp_y = base_disp_y
            fill_y = base_fill_y
            jar_y = fill_y - jar_start_gap
            side_x = float(cfg.get("station_x", -0.08))
            if not self._station_clears_microwave(
                side_x, disp_y, jar_y, mw_xy, mw_half
            ):
                side_x = self._nudge_x_clear_of_microwave(
                    side_x, disp_y, jar_y, mw_xy, mw_half, -0.45, 0.20
                )

        self.dispenser_xy = np.array([side_x, disp_y], dtype=float)
        self.fill_xy = np.array([side_x, fill_y], dtype=float)
        self.jar_xy = np.array([side_x, jar_y], dtype=float)
        self.arm = ArmTag("left" if side_x <= 0 else "right")

        blockers = [
            (mw_xy.copy(), mw_half.copy()),
            (self.dispenser_xy.copy(), np.asarray(self.DISP_HALF_XY, dtype=float)),
            (self.fill_xy.copy(), np.asarray(self.JAR_HALF_XY, dtype=float)),
            (self.jar_xy.copy(), np.asarray(self.JAR_HALF_XY, dtype=float)),
        ]

        # Scale stays in the scene as a prop (same placement as before); not used
        # for success. Anchor Y to the fill target (old jar under-nozzle pose).
        if randomize:
            if cfg.get("pin_scale_x") is not None and cfg.get("pin_scale_y") is not None:
                scale_xy = np.array(
                    [float(cfg["pin_scale_x"]), float(cfg["pin_scale_y"])],
                    dtype=float,
                )
            else:
                dist_base = float(cfg.get("scale_dist", self.SCALE_DIST_BASE))
                dist_jit = float(cfg.get("scale_dist_jitter", self.SCALE_DIST_JITTER))
                scale_xy = None
                side_signs = [1.0, -1.0]
                rng.shuffle(side_signs)
                for sign in side_signs:
                    for _ in range(40):
                        dist = float(
                            rng.uniform(
                                max(0.14, dist_base - dist_jit),
                                dist_base + dist_jit,
                            )
                        )
                        ang = float(rng.uniform(-0.45, 0.45))
                        dx = sign * dist * float(np.cos(ang))
                        dy = -abs(dist * float(np.sin(ang))) - float(
                            rng.uniform(0.0, 0.04)
                        )
                        cand = np.array([side_x + dx, fill_y + dy], dtype=float)
                        if side_x <= 0 and cand[0] > 0.06:
                            continue
                        if side_x > 0 and cand[0] < -0.06:
                            continue
                        cand[0] = float(np.clip(cand[0], -0.50, 0.50))
                        cand[1] = float(np.clip(cand[1], -0.26, 0.16))
                        if self._footprint_clear(cand, self.SCALE_HALF_XY, blockers):
                            scale_xy = cand
                            break
                    if scale_xy is not None:
                        break
                if scale_xy is None:
                    away = -1.0 if side_x <= float(mw_xy[0]) else 1.0
                    if side_x <= 0:
                        away = -abs(away)
                    else:
                        away = abs(away)
                    scale_xy = np.array(
                        [side_x + away * 0.22, fill_y - 0.06], dtype=float
                    )
                    if not self._footprint_clear(scale_xy, self.SCALE_HALF_XY, blockers):
                        scale_xy = np.array(
                            [side_x + away * 0.28, fill_y - 0.08], dtype=float
                        )
        else:
            scale_xy = np.array(
                [
                    float(
                        cfg.get(
                            "scale_x",
                            side_x - 0.26 if side_x <= 0 else side_x + 0.26,
                        )
                    ),
                    float(cfg.get("scale_y", -0.14)),
                ],
                dtype=float,
            )
            if not self._footprint_clear(scale_xy, self.SCALE_HALF_XY, blockers):
                away = -1.0 if side_x <= float(mw_xy[0]) else 1.0
                scale_xy = np.array([side_x + away * 0.24, fill_y - 0.06], dtype=float)
        self.scale_xy = scale_xy
        blockers.append(
            (self.scale_xy.copy(), np.asarray(self.SCALE_HALF_XY, dtype=float))
        )

        # Baking decor: free counter (AABB-clear), or on top of the microwave.
        defaults = {
            "bread": cfg.get("bread_xy", [0.28, -0.20]),
            "flour": cfg.get("flour_xy", [0.38, 0.14]),
            "chips": cfg.get("chips_xy", [0.12, -0.22]),
            "bowl": cfg.get("bowl_xy", [0.40, -0.02]),
        }
        on_mw_prob = float(
            cfg.get("decor_on_microwave_prob", self.DECOR_ON_MICROWAVE_PROB)
        )
        decor_x_range = self._parse_range(cfg, "decor_x_range", (-0.45, 0.48))
        decor_y_range = self._parse_range(cfg, "decor_y_range", (-0.26, 0.20))
        # Keep table decor off the microwave footprint and out of the pour lane.
        mw_blockers = list(blockers)
        # Push corridor: jar start → robot (−Y) and jar → fill target (+Y).
        push_corridor_c = np.array([side_x, 0.5 * (jar_y + fill_y)], dtype=float)
        push_corridor_h = np.array([0.12, 0.5 * abs(fill_y - jar_y) + 0.12], dtype=float)
        mw_blockers.append((push_corridor_c, push_corridor_h))
        on_mw_blockers = []  # footprints already placed on the microwave top

        self._decor_layout = {}
        for name, default_xy in defaults.items():
            half = self.DECOR_HALF_XY.get(name, (0.08, 0.08))
            on_mw = False
            if randomize:
                want_mw = bool(rng.rand() < on_mw_prob)
                # Skip MW top if the prop footprint cannot fit with margin.
                if want_mw and (
                    float(half[0]) > 0.70 * float(mw_half[0])
                    or float(half[1]) > 0.70 * float(mw_half[1])
                ):
                    want_mw = False
                xy = None
                if want_mw:
                    # Place on MW top with mutual clearance between decor items.
                    inset = 0.40
                    for _ in range(50):
                        cand = np.array(
                            [
                                mw_xy[0]
                                + rng.uniform(-inset * mw_half[0], inset * mw_half[0]),
                                mw_xy[1]
                                + rng.uniform(-inset * mw_half[1], inset * mw_half[1]),
                            ],
                            dtype=float,
                        )
                        if self._footprint_clear(
                            cand, half, on_mw_blockers, margin=0.02
                        ):
                            xy = cand
                            on_mw = True
                            break
                if xy is None:
                    on_mw = False
                    xy = self._sample_free_xy_aabb(
                        rng, half, mw_blockers, decor_x_range, decor_y_range
                    )
                z = (mw_top + 0.002) if on_mw else table_z
            else:
                xy = np.array([float(default_xy[0]), float(default_xy[1])], dtype=float)
                z = table_z
                # Fixed decor that would land in the MW / station gets nudged away.
                if not self._footprint_clear(xy, half, mw_blockers):
                    xy = self._sample_free_xy_aabb(
                        rng, half, mw_blockers, decor_x_range, decor_y_range
                    )
            self._decor_layout[name] = {
                "xy": xy,
                "z": float(z),
                "on_microwave": bool(on_mw),
            }
            if on_mw:
                on_mw_blockers.append((xy.copy(), np.asarray(half, dtype=float)))
            else:
                mw_blockers.append((xy.copy(), np.asarray(half, dtype=float)))

        # Hard assert for the critical failure mode reported by the user.
        if not self._station_clears_microwave(
            float(self.dispenser_xy[0]),
            float(self.dispenser_xy[1]),
            float(self.jar_xy[1]),
            mw_xy,
            mw_half,
            margin=0.0,
        ):
            print(
                f"[measure_ingredient] WARNING: station still overlaps microwave "
                f"disp={self.dispenser_xy.tolist()} mw={mw_xy.tolist()} "
                f"mw_half={mw_half.tolist()} — forcing X nudge"
            )
            side_x = self._nudge_x_clear_of_microwave(
                float(self.dispenser_xy[0]),
                float(self.dispenser_xy[1]),
                float(self.jar_xy[1]),
                mw_xy,
                mw_half,
                -0.45,
                0.25,
            )
            self.dispenser_xy[0] = side_x
            self.jar_xy[0] = side_x
            self.arm = ArmTag("left" if side_x <= 0 else "right")

        self.target_fill = self._resolve_target_fill(cfg, self._layout_rng(202))

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.05, -0.55, 1.40], dtype=np.float64)
        look_at = np.array([-0.05, -0.05, 0.92], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ------------------------------------------------------------------ materials
    def _metallic_material(self, rgb, roughness=0.22, metallic=0.95):
        rgba = list(rgb[:3]) + [1.0]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(float(roughness))
            mat.set_metallic(float(metallic))
        except Exception:
            mat.roughness = float(roughness)
            mat.metallic = float(metallic)
        return mat

    def _plain_glass_material(self):
        """Simple alpha-transparent plastic — no glass transmission/IOR (viewer-friendly)."""
        mat = sapien.render.RenderMaterial(base_color=list(self.PLAIN_GLASS))
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.55)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.55
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
        return mat

    def _glass_material(self, rgba=None, transmission=0.90):
        if bool(getattr(self, "_plain_glass", False)):
            return self._plain_glass_material()
        c = list(rgba if rgba is not None else self.GLASS)
        if len(c) == 3:
            c = c + [0.18]
        mat = sapien.render.RenderMaterial(base_color=c)
        try:
            mat.set_transmission(float(transmission))
            mat.set_transmission_roughness(0.02)
            mat.set_roughness(0.05)
            mat.set_metallic(0.0)
        except Exception:
            mat.transmission = float(transmission)
            mat.roughness = 0.05
            mat.metallic = 0.0
        try:
            mat.set_ior(1.45)
        except Exception:
            pass
        return mat

    def _opaque_material(self, rgb, alpha=1.0):
        rgba = list(rgb[:3]) + [float(alpha)]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.45)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.45
            mat.metallic = 0.0
        return mat

    def _ring_material(self):
        """Opaque saturated red for the target fill mark."""
        rgba = list(self.RING_RED[:3]) + [1.0]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.30)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.30
            mat.metallic = 0.0
        try:
            emit = [
                float(self.RING_RED[0]) * 0.35,
                float(self.RING_RED[1]) * 0.35,
                float(self.RING_RED[2]) * 0.35,
                1.0,
            ]
            mat.set_emission(emit)
        except Exception:
            try:
                mat.emission = emit
            except Exception:
                pass
        return mat

    def _parse_oil_style(self, cfg) -> str:
        """Return ``transparent`` or ``solid`` from task_args."""
        raw = cfg.get("oil_style", None)
        if raw is None and "oil_transparent" in cfg:
            raw = "transparent" if bool(cfg.get("oil_transparent")) else "solid"
        style = str(raw if raw is not None else self.OIL_STYLE_DEFAULT).strip().lower()
        aliases = {
            "transparent": "transparent",
            "clear": "transparent",
            "see_through": "transparent",
            "see-through": "transparent",
            "glass": "transparent",
            "solid": "solid",
            "opaque": "solid",
            "previous": "solid",
            "dark": "solid",
            "green": "solid",
        }
        if style not in aliases:
            raise ValueError(
                f"oil_style must be 'transparent' or 'solid' (got {raw!r})"
            )
        return aliases[style]

    def _apply_oil_style(self, style: str):
        """Set active oil colors / material mode from ``oil_style``."""
        self.oil_style = style
        self.oil_transparent = style == "transparent"
        if self.oil_transparent:
            self.oil_color = list(self.OIL_COLOR_TRANSPARENT)
            self.oil_stream_color = list(self.OIL_STREAM_TRANSPARENT)
            self.oil_meniscus = list(self.OIL_MENISCUS_TRANSPARENT)
            self.oil_spill_color = list(self.OIL_SPILL_TRANSPARENT)
        else:
            self.oil_color = list(self.OIL_COLOR_SOLID)
            self.oil_stream_color = list(self.OIL_STREAM_SOLID)
            self.oil_meniscus = None
            self.oil_spill_color = list(self.OIL_SPILL_SOLID)

    def _fluid_material(self, rgba, transmission=None):
        """Oil material: glass-clear when transparent, beer-style when solid."""
        mat = sapien.render.RenderMaterial(base_color=list(rgba))
        if getattr(self, "oil_transparent", True):
            t = 1.0 if transmission is None else float(transmission)
            try:
                mat.set_transmission(t)
                mat.set_transmission_roughness(0.0)
                mat.set_roughness(0.04)
                mat.set_metallic(0.0)
            except Exception:
                mat.roughness = 0.04
                mat.metallic = 0.0
            try:
                mat.set_ior(1.0)
            except Exception:
                pass
        else:
            # Previous dark-green look (pour_beer fluid recipe).
            try:
                mat.set_roughness(0.18)
                mat.set_metallic(0.0)
            except Exception:
                mat.roughness = 0.18
                mat.metallic = 0.0
        return mat

    def _make_oil_column(self, radius, half_h, world_xyz, rgba, name, local_z=0.0):
        """Visual-only oil cylinder via RenderShapeCylinder."""
        ent = sapien.Entity()
        ent.set_name(name)
        ent.set_pose(
            sapien.Pose([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])])
        )
        body = sapien.render.RenderBodyComponent()
        mat = self._fluid_material(rgba)
        col = sapien.render.RenderShapeCylinder(
            radius=float(radius),
            half_length=max(0.002, float(half_h)),
            material=mat,
        )
        col.set_local_pose(sapien.Pose([0.0, 0.0, float(local_z)], self.UPRIGHT_CYL_Q))
        body.attach(col)
        ent.add_component(body)
        self.scene.add_entity(ent)
        return ent

    def _add_static_box(self, pose, half_size, material=None, color=None, name="", collision=True):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            builder.add_box_collision(
                pose=sapien.Pose(),
                half_size=list(half_size),
                material=self.scene.default_physical_material,
            )
        if material is None:
            rgba = list(color if color is not None else [0.8, 0.8, 0.8, 1.0])
            if len(rgba) == 3:
                rgba = rgba + [1.0]
            material = self._opaque_material(rgba[:3], rgba[3])
        builder.add_box_visual(pose=sapien.Pose(), half_size=list(half_size), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _add_static_mesh_visual(self, filename, pose, material, name):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(
            filename=str(Path(filename).resolve()),
            material=material,
        )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)

        base_pour = float(cfg.get("pour_rate", self.POUR_RATE))
        self.pour_rate_base = base_pour
        # pour_rate_jitter: 0 → fixed; 0.15 → sample Uniform([0.85, 1.15]) * pour_rate.
        jitter = float(cfg.get("pour_rate_jitter", self.POUR_RATE_JITTER))
        self.pour_rate_jitter = max(0.0, jitter)
        if self.pour_rate_jitter > 0.0:
            scale = float(
                self._layout_rng(211).uniform(
                    1.0 - self.pour_rate_jitter, 1.0 + self.pour_rate_jitter
                )
            )
            self.pour_rate = max(1e-8, base_pour * scale)
        else:
            self.pour_rate = base_pour
        self.fill_tol = float(cfg.get("fill_tol", self.FILL_TOL))
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL))
        self.spill_rate = float(cfg.get("spill_rate", self.SPILL_RATE))
        self.force_overflow = bool(cfg.get("force_overflow", False))
        self._apply_oil_style(self._parse_oil_style(cfg))

        self.tab_open = False
        self.liquid_level = 0.0
        self.overflowed = False
        self.spill_amount = 0.0
        self.opened_once = False
        self.closed_after_pour = False
        self._reset_metric_state()
        self.jar_on_scale = False
        self._episode_jar_released = False
        self._liquid_entity = None
        self._stream_entity = None
        self._spill_entity = self._remove_entity(getattr(self, "_spill_entity", None))
        self._spill_radius_cached = -1.0
        self._spill_xy = None
        self._liquid_half_h_cached = -1.0
        self._switch_parts = []
        self._switch_btn = None
        self._switch_key_shapes = []
        self._switch_key_color_down = None
        self._button_home_pose = None
        self._button_visual_depth = 0.0
        self._button_target_depth = 0.0
        self._button_pressed_visual = False
        self._pressing_arm_side = ""
        self._ring_entities = []
        self._touch_latched = False
        self._ignore_tab = False
        self._pending_tab_off = False
        self._jar_locked = True
        self._jar_carry = False
        self._jar_carry_local = None
        self._jar_carry_arm = None
        self._jar_carry_quat = None
        self._jar_seated_pose = None

        # Jar in front of nozzle; push under spout to fill; decor free / on MW.
        self._resolve_layout(cfg)
        self.push_step = float(cfg.get("push_step", self.PUSH_STEP_DEFAULT))
        self.jar_catch_r = float(cfg.get("jar_catch_r", self.JAR_CATCH_R))
        self._push_active = False

        self._build_dispenser()
        self._build_jar()
        self._build_fill_rings()
        self._build_scale()
        self._build_baking_props()
        self._rebuild_liquid(force=True)
        self._sync_stream()
        self._sync_jar_followers()

        self.add_prohibit_area(
            sapien.Pose([*self.dispenser_xy, self.table_top + 0.1]), padding=0.08
        )
        self.add_prohibit_area(
            sapien.Pose([*self.jar_xy, self.table_top + 0.05]), padding=0.05
        )
        if getattr(self, "fill_xy", None) is not None:
            self.add_prohibit_area(
                sapien.Pose([*self.fill_xy, self.table_top + 0.05]), padding=0.04
            )
        if self.scale is not None:
            self.add_prohibit_area(self.scale, padding=0.04)

        self._loaded = True
        mw = getattr(self, "microwave_xy_override", None) or getattr(
            self, "microwave_xy", None
        )
        decor_on_mw = [
            n for n, d in (self._decor_layout or {}).items() if d.get("on_microwave")
        ]
        lo, hi = self._fill_band()
        print(
            f"[measure_ingredient] KitchenS scene={self.scene_id} "
            f"target={self.target_fill:.0%} band=[{lo:.0%},{hi:.0%}] arm={self.arm} "
            f"pour_rate={self.pour_rate:.6g} "
            f"(base={self.pour_rate_base:.6g}±{self.pour_rate_jitter:.0%}) "
            f"jar={self.jar_xy.tolist()} fill={self.fill_xy.tolist()} scale={None if self.scale_xy is None else self.scale_xy.tolist()} "
            f"mw={None if mw is None else [float(mw[0]), float(mw[1])]} "
            f"decor_on_mw={decor_on_mw} oil_style={self.oil_style}"
        )

    def _build_dispenser(self):
        """Silver metallic cylindrical oil can with dome + nozzle tab over the jar."""
        x, y = self.dispenser_xy
        z0 = self.table_top
        _, _, pedestal_hz = self.PEDESTAL_HALF
        _, _, plat_hz = self.PLATFORM_HALF
        silver = self._metallic_material(self.SILVER)
        silver_dark = self._metallic_material(self.SILVER_DARK, roughness=0.30, metallic=0.90)

        # Pedestal + platform raise the can so the nozzle clears the jar rim.
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + pedestal_hz]),
            half_size=self.PEDESTAL_HALF,
            material=silver_dark,
            name="oil_dispenser_pedestal",
        )
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + 2.0 * pedestal_hz + plat_hz]),
            half_size=self.PLATFORM_HALF,
            material=silver_dark,
            name="oil_dispenser_platform",
        )

        body_bottom = z0 + 2.0 * pedestal_hz + 2.0 * plat_hz
        body_half = self.BODY_H * 0.5
        body_z = body_bottom + body_half

        # Main cylindrical body (Sapien cylinder axis = local +X → rotate to +Z).
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=self.BODY_R,
            half_length=body_half,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=self.BODY_R,
            half_length=body_half,
            material=silver,
        )
        builder.set_initial_pose(sapien.Pose([x, y, body_z]))
        builder.build(name="oil_dispenser_body")

        # Dome cap.
        dome_z = body_bottom + self.BODY_H + self.DOME_R * 0.35
        dome_b = self.scene.create_actor_builder()
        dome_b.set_physx_body_type("static")
        dome_b.add_sphere_collision(
            pose=sapien.Pose(),
            radius=self.DOME_R,
            material=self.scene.default_physical_material,
        )
        dome_b.add_sphere_visual(pose=sapien.Pose(), radius=self.DOME_R, material=silver)
        dome_b.set_initial_pose(sapien.Pose([x, y, dome_z]))
        dome_b.build(name="oil_dispenser_dome")

        # Front glass window + olive-oil reservoir fill (shows contents).
        win_y = y - self.BODY_R + 0.004
        reservoir_rgba = (
            list(self.oil_color)
            if getattr(self, "oil_transparent", True)
            else list(self.OIL_COLOR_SOLID)
        )
        oil_fill = self._fluid_material(reservoir_rgba)
        self._add_static_box(
            pose=sapien.Pose([x, win_y - 0.001, body_z - 0.01]),
            half_size=[self.BODY_R * 0.55, 0.002, body_half * 0.70],
            material=oil_fill,
            name="oil_reservoir_fill",
            collision=False,
        )
        self._add_static_box(
            pose=sapien.Pose([x, win_y - 0.004, body_z - 0.01]),
            half_size=[self.BODY_R * 0.58, 0.0015, body_half * 0.72],
            material=self._glass_material([0.90, 0.96, 0.99, 0.20]),
            name="oil_window",
            collision=False,
        )

        # Nozzle arm from hopper front to the fixed fill target (not the jar spawn).
        fill_x, fill_y = self.fill_xy
        hopper_front_y = y - self.BODY_R
        tip_y = float(fill_y)
        nozzle_joint_z = self.table_top + self.JAR_HEIGHT + 0.070
        nozzle_outlet_z = self.table_top + self.JAR_HEIGHT + 0.035
        nozzle_y = 0.5 * (hopper_front_y + tip_y)

        # Visual-only nozzle (no collision) so the jar can be pushed underneath.
        self._add_static_box(
            pose=sapien.Pose([x, nozzle_y, nozzle_joint_z]),
            half_size=[0.007, abs(tip_y - hopper_front_y) * 0.5, 0.006],
            material=silver_dark,
            name="oil_nozzle_arm",
            collision=False,
        )
        tip_half_z = 0.5 * (nozzle_joint_z - nozzle_outlet_z)
        self._add_static_box(
            pose=sapien.Pose(
                [fill_x, tip_y, 0.5 * (nozzle_joint_z + nozzle_outlet_z)]
            ),
            half_size=[0.007, 0.007, tip_half_z],
            material=silver_dark,
            name="oil_nozzle_tip",
            collision=False,
        )
        # Nozzle opening ring.
        self._add_static_box(
            pose=sapien.Pose([fill_x, tip_y, nozzle_outlet_z]),
            half_size=[self.NOZZLE_R, self.NOZZLE_R, 0.002],
            material=self._metallic_material([0.25, 0.25, 0.27], roughness=0.35),
            name="oil_nozzle_opening",
            collision=False,
        )

        self.nozzle_outlet_xyz = np.array([fill_x, tip_y, nozzle_outlet_z], dtype=float)
        # Push-key on TOP of the nozzle arm near the tip (green up / red down).
        self.switch_base_xyz = np.array(
            [fill_x, tip_y + 0.012, nozzle_joint_z + 0.006], dtype=float
        )
        self._build_switch()

    def _switch_travel(self) -> float:
        """Max key travel (m); latched ON parks here."""
        return float(self.SWITCH_BTN_HALF[2]) * 0.90

    def _switch_button_center(self, open_: bool | None = None):
        """World XYZ of the key center (visual pose; home = UP)."""
        home = getattr(self, "_button_home_pose", None)
        if home is None:
            bx, by, bz = self.switch_base_xyz
            base_top = bz + float(self.SWITCH_BASE_HALF[2])
            half_t = float(self.SWITCH_BTN_HALF[2])
            return np.array([bx, by, base_top + half_t], dtype=float)
        if open_ is None:
            depth = float(getattr(self, "_button_visual_depth", 0.0))
        else:
            depth = self._switch_travel() if bool(open_) else 0.0
        return np.array(
            [float(home.p[0]), float(home.p[1]), float(home.p[2]) - float(depth)],
            dtype=float,
        )

    def _switch_top_z(self, open_: bool | None = None):
        """Top of the key at home (UP). Spring sensing uses the home top."""
        home = getattr(self, "_button_home_pose", None)
        if home is not None:
            return float(home.p[2]) + float(self.SWITCH_BTN_HALF[2])
        c = self._switch_button_center(False)
        return float(c[2] + float(self.SWITCH_BTN_HALF[2]))

    def _sync_switch_touch_points(self):
        home = getattr(self, "_button_home_pose", None)
        if home is not None:
            self.touch_xy = np.array([float(home.p[0]), float(home.p[1])], dtype=float)
            self.touch_top_z = float(home.p[2]) + float(self.SWITCH_BTN_HALF[2])
        else:
            c = self._switch_button_center(False)
            self.touch_xy = c[:2].copy()
            self.touch_top_z = float(c[2] + float(self.SWITCH_BTN_HALF[2]))
        self.tab_touch_xyz = np.array(
            [self.touch_xy[0], self.touch_xy[1], self.touch_top_z], dtype=float
        )

    def _clear_switch_parts(self):
        for part in list(getattr(self, "_switch_parts", []) or []):
            self._remove_entity(part)
        self._switch_parts = []
        self._switch_btn = None
        self._switch_key_shapes = []
        self._switch_key_color_down = None
        self._button_home_pose = None
        self._button_visual_depth = 0.0
        self._button_target_depth = 0.0

    def _cache_switch_key_shapes(self, btn) -> None:
        """Collect render shapes for green/red keycap recoloring."""
        shapes = []
        entity = btn.actor if hasattr(btn, "actor") else btn
        try:
            for c in entity.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    shapes = list(c.render_shapes)
                    break
        except Exception:
            shapes = []
        self._switch_key_shapes = shapes
        self._switch_key_color_down = None  # force first color sync

    def _set_switch_key_color(self, down: bool) -> None:
        """Green when up (OFF), red when depressed (ON) — cook_meat style."""
        down = bool(down)
        prev = getattr(self, "_switch_key_color_down", None)
        if prev is not None and bool(prev) == down:
            return
        self._switch_key_color_down = down
        rgb = self.SWITCH_COLOR_DOWN if down else self.SWITCH_COLOR_UP
        color = list(rgb) + [1.0]
        for shape in getattr(self, "_switch_key_shapes", []) or []:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _build_switch(self):
        """Dark housing + green push key (static collision at home; visual travels)."""
        self._clear_switch_parts()
        bx, by, bz = self.switch_base_xyz
        housing = self._metallic_material([0.18, 0.18, 0.20], roughness=0.45, metallic=0.55)
        base = self._add_static_box(
            pose=sapien.Pose([bx, by, bz + float(self.SWITCH_BASE_HALF[2])]),
            half_size=list(self.SWITCH_BASE_HALF),
            material=housing,
            name="oil_switch_base",
            collision=True,
        )
        self._switch_parts.append(base)

        base_top = bz + float(self.SWITCH_BASE_HALF[2])
        btn_z = base_top + float(self.SWITCH_BTN_HALF[2])
        home = sapien.Pose([bx, by, btn_z])
        green = self._opaque_material(self.SWITCH_COLOR_UP)
        btn = self._add_static_box(
            pose=home,
            half_size=list(self.SWITCH_BTN_HALF),
            material=green,
            name="oil_switch_button",
            collision=True,
        )
        self._switch_btn = btn
        self._switch_parts.append(btn)
        self._cache_switch_key_shapes(btn)
        self._set_switch_key_color(False)
        self._button_home_pose = home
        self._button_visual_depth = 0.0
        self._button_target_depth = 0.0
        self._sync_switch_touch_points()

    def _set_button_press_depth(self, depth: float) -> None:
        max_depth = self._switch_travel()
        self._button_target_depth = float(np.clip(depth, 0.0, max_depth))

    def _advance_button_press_visual(self) -> None:
        button = getattr(self, "_switch_btn", None)
        home = getattr(self, "_button_home_pose", None)
        if button is None or home is None:
            return
        max_depth = self._switch_travel()
        target = float(np.clip(getattr(self, "_button_target_depth", 0.0), 0.0, max_depth))
        current = float(np.clip(getattr(self, "_button_visual_depth", 0.0), 0.0, max_depth))
        step = float(self.SWITCH_BUTTON_VISUAL_STEP)
        if target > current:
            current = min(target, current + step)
        elif target < current:
            current = max(target, current - step)
        self._button_visual_depth = current
        self._button_pressed_visual = bool(current > 1e-6)
        self._set_switch_key_color(self._button_pressed_visual)
        try:
            button.set_pose(
                sapien.Pose(
                    [float(home.p[0]), float(home.p[1]), float(home.p[2] - current)],
                    list(home.q),
                )
            )
        except Exception:
            pass

    def _switch_press_signal(self):
        """Best arm TCP press candidate over the nozzle key (fill_coffee-style)."""
        if not hasattr(self, "robot"):
            return None
        touch_xy = np.asarray(getattr(self, "touch_xy", None), dtype=float)
        if touch_xy.size != 2:
            return None
        preferred = str(getattr(self, "_pressing_arm_side", ""))
        sides = [preferred] if preferred in ("left", "right") else []
        sides += [side for side in ("left", "right") if side not in sides]
        best = None
        top_z = float(getattr(self, "touch_top_z", self._switch_top_z()))
        k = float(self.SWITCH_FORCE_STIFFNESS)
        slack = float(self.SWITCH_FORCE_ENGAGE_SLACK)
        for side in sides:
            try:
                getter = (
                    self.robot.get_left_ee_pose
                    if side == "left"
                    else self.robot.get_right_ee_pose
                )
                ee = np.asarray(getter(), dtype=float)
                tcp = np.asarray(ee, dtype=float)
                tcp[2] -= float(self.EE_TO_TCP)
            except Exception:
                try:
                    getter = (
                        self.robot.get_left_tcp_pose
                        if side == "left"
                        else self.robot.get_right_tcp_pose
                    )
                    tcp = np.asarray(getter(), dtype=float)
                except Exception:
                    continue
            xy_dist = float(np.linalg.norm(tcp[:2] - touch_xy))
            if xy_dist > float(self.SWITCH_TOUCH_XY_TOL):
                continue
            force = float(k * max(0.0, top_z + slack - float(tcp[2])))
            signal = {"side": side, "tcp": tcp, "force": force}
            if best is None or signal["force"] > best["force"]:
                best = signal
        if best is not None:
            self._pressing_arm_side = best["side"]
        return best

    def _update_switch_visual_from_robot(self) -> None:
        """Press depth from gripper force; settle to latched DOWN (ON) or UP (OFF)."""
        max_depth = self._switch_travel()
        latched = max_depth if bool(getattr(self, "tab_open", False)) else 0.0
        signal = self._switch_press_signal()
        if signal is not None and float(signal["force"]) > 0.0:
            # Soft map: ~14 N ≈ full travel (same scale family as fill_coffee).
            press_depth = float(
                np.clip(float(signal["force"]) / 14.0 * max_depth, 0.0, max_depth)
            )
            self._set_button_press_depth(max(latched, press_depth))
        else:
            self._set_button_press_depth(latched)
        self._advance_button_press_visual()

    def _jar_glass_material(self, viewer_shell: bool = False):
        """Glass for the jar (same look as ``fill_coffee_jar``).

        Demo cameras use transmission glass. The interactive SAPIEN viewer does
        not composite opaque oil behind transmission materials (or a solid
        cylinder wall), so the viewer shell uses plain alpha glass — same trick
        as ``fill_coffee_jar`` / ``trap_bug``.
        """
        if viewer_shell:
            glass = sapien.render.RenderMaterial(
                base_color=[0.70, 0.82, 0.90, 0.22]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.12)
                glass.set_metallic(0.0)
            except Exception:
                glass.roughness = 0.12
                glass.metallic = 0.0
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        if bool(getattr(self, "_plain_glass", False)):
            return self._plain_glass_material()

        glass = sapien.render.RenderMaterial(base_color=[0.76, 0.88, 0.94, 0.12])
        try:
            glass.set_transmission(1.0)
            glass.set_transmission_roughness(0.0)
            glass.set_roughness(0.05)
            glass.set_metallic(0.0)
        except Exception:
            pass
        try:
            glass.set_ior(1.0)
        except Exception:
            pass
        return glass

    def _viewer_oil_material(self):
        """Fully opaque oil for viewer compositing through alpha glass walls."""
        # Match the solid oil (dark yellow) for the interactive hollow-jar viewer.
        rgba = [0.62, 0.44, 0.04, 1.0]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.22)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.22
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
        return mat

    def _build_jar_visual(self, hollow: bool = False):
        """Glass jar visual. ``hollow=True`` for SAPIEN viewer (open interior).

        Camera / expert demos keep the smooth solid transmission cylinder (looks
        correct in offline render). The interactive viewer treats that cylinder
        as an opaque volume, so viewer mode uses a thin alpha-glass shell instead.
        """
        self.jar_visual = self._remove_entity(getattr(self, "jar_visual", None))
        if self.jar is None:
            return

        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT
        bottom_t = self.JAR_BOTTOM_T
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]
        wall_h = h - bottom_t
        wall_half = wall_h * 0.5
        wall_z = bottom_t + wall_half
        glass = self._jar_glass_material(viewer_shell=bool(hollow))

        try:
            pose = self.jar.get_pose()
        except Exception:
            pose = sapien.Pose(
                [float(self.jar_xy[0]), float(self.jar_xy[1]), self.table_top + 0.001]
            )
        vis = sapien.Entity()
        vis.set_name("glass_jar_visual")
        vis.set_pose(pose)
        render_body = sapien.render.RenderBodyComponent()

        floor = sapien.render.RenderShapeCylinder(
            radius=outer_r * 0.98,
            half_length=max(0.0015, bottom_t * 0.5),
            material=glass,
        )
        floor.set_local_pose(sapien.Pose([0.0, 0.0, bottom_t * 0.5], upright_q))
        render_body.attach(floor)

        if hollow:
            # Thin faceted glass shell — empty inside so oil level is visible.
            wall_t = 0.0024
            n_seg = 36
            wall_radius = outer_r - 0.5 * wall_t
            tangent_half = wall_radius * np.tan(np.pi / n_seg) * 1.03
            for ang in np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False):
                px = float(wall_radius * np.cos(ang))
                py = float(wall_radius * np.sin(ang))
                yaw = float(ang + 0.5 * np.pi)
                q = [
                    float(np.cos(0.5 * yaw)),
                    0.0,
                    0.0,
                    float(np.sin(0.5 * yaw)),
                ]
                panel = sapien.render.RenderShapeBox(
                    [float(tangent_half), float(0.5 * wall_t), float(wall_half)],
                    glass,
                )
                panel.set_local_pose(sapien.Pose([px, py, wall_z], q))
                render_body.attach(panel)
        else:
            wall = sapien.render.RenderShapeCylinder(
                radius=outer_r,
                half_length=wall_half,
                material=glass,
            )
            wall.set_local_pose(sapien.Pose([0.0, 0.0, wall_z], upright_q))
            render_body.attach(wall)

        vis.add_component(render_body)
        self.scene.add_entity(vis)
        self.jar_visual = vis
        self._jar_visual_hollow = bool(hollow)

    def use_viewer_hollow_jar(self):
        """Swap to hollow alpha-glass shell for interactive SAPIEN viewer only."""
        self._build_jar_visual(hollow=True)
        # Rebuild oil after the shell so it composites through the alpha walls.
        self._rebuild_liquid(force=True)
        print(
            "[measure_ingredient] viewer jar: hollow alpha-glass shell "
            "(oil visible from the side)"
        )

    def _build_jar(self):
        """See-through glass jar: collision + glass visual.

        Spawns in front of the nozzle (locked); unlocked for the contact push,
        then re-locked under the nozzle while pouring.
        Interactive viewer calls ``use_viewer_hollow_jar()`` after setup.
        """
        x, y = self.jar_xy
        z0 = self.table_top + 0.001
        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT

        md_path = resolve_model_dir("253_glass_jar") / "model_data0.json"
        with open(md_path, "r") as f:
            md = json.load(f)

        # Solid cylinder collision (hollow mesh is a poor grasp target).
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, 0.0, h * 0.5], self.VERTICAL_CYL_Q),
            radius=float(outer_r),
            half_length=float(h * 0.5),
            material=self.scene.default_physical_material,
        )
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        entity = builder.build(name="glass_jar")
        try:
            entity.set_name("glass_jar")
        except Exception:
            pass

        self.jar = Actor(entity, md, mass=float(self.JAR_MASS))
        self._jar_home_pose = sapien.Pose([x, y, z0])
        self._jar_locked = True
        self._push_active = False
        self._set_jar_damping(50.0)
        # High table friction so a closed-gripper shove tracks (catch_cup pillow).
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_mass(float(self.JAR_MASS))
                    for shape in component.get_collision_shapes():
                        shape.set_physical_material(
                            self.scene.create_physical_material(
                                float(self.PUSH_MU_STATIC),
                                float(self.PUSH_MU_DYNAMIC),
                                0.0,
                            )
                        )
                except Exception:
                    pass

        self._build_jar_visual(hollow=False)

        self.jar_bottom_z = self.table_top + self.JAR_BOTTOM_T
        self.jar_fillable_h = self.JAR_HEIGHT - self.JAR_BOTTOM_T

    def _set_jar_damping(self, damping: float):
        if self.jar is None:
            return
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_linear_damping(float(damping))
                    component.set_angular_damping(float(damping))
                except Exception:
                    pass

    def _build_fill_rings(self):
        """Three thick red rings at 25% / 50% / 75% that follow the jar pose."""
        self._ring_entities = []
        ring_material = self._ring_material()
        ring_mesh = str((resolve_model_dir("253_glass_jar") / "rings" / "thin_ring.glb").resolve())
        x, y = self.jar_xy
        outer_r = float(self.JAR_INNER_R) + 0.0035
        xy = float(self.RING_XY_SCALE) * (outer_r / float(self.RING_MESH_RADIUS))
        z_sc = float(self.RING_Z_SCALE)
        scale = [xy, xy, z_sc]
        for frac in (0.25, 0.50, 0.75):
            z_local = self.JAR_BOTTOM_T + frac * self.jar_fillable_h
            builder = self.scene.create_actor_builder()
            builder.set_physx_body_type("kinematic")
            builder.add_visual_from_file(
                filename=ring_mesh,
                scale=scale,
                material=ring_material,
            )
            builder.set_initial_pose(
                sapien.Pose([x, y, self.table_top + 0.001 + z_local])
            )
            ent = builder.build(name=f"fill_ring_{int(frac * 100)}")
            self._ring_entities.append((float(frac), ent))

    def _build_scale(self):
        """Electronic kitchen scale prop (same arm side; not required for success)."""
        cfg = self._cfg
        scale_xy = getattr(self, "scale_xy", None)
        if scale_xy is None:
            side = float(self.jar_xy[0])
            scale_xy = np.array(
                [
                    float(cfg.get("scale_x", side - 0.26 if side <= 0 else side + 0.26)),
                    float(cfg.get("scale_y", -0.14)),
                ],
                dtype=float,
            )
            self.scale_xy = scale_xy
        scale_id = int(cfg.get("scale_id", 0))
        self.scale = create_actor(
            scene=self,
            pose=sapien.Pose(
                [float(scale_xy[0]), float(scale_xy[1]), self.table_top + 0.001],
                self.YUP_Q,
            ),
            modelname="072_electronicscale",
            model_id=scale_id,
            convex=True,
            is_static=True,
        )
        self.scale_id = scale_id

    def _recolor_actor(self, actor, rgba):
        if actor is None:
            return
        ent = getattr(actor, "actor", actor)
        for comp in ent.get_components():
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for s in comp.render_shapes:
                    try:
                        s.material.set_base_color(list(rgba))
                    except Exception:
                        pass

    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _make_dynamic(self, entity, mass=None, lin_damp=0.8, ang_damp=1.2):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_damping(float(lin_damp))
            rigid.set_angular_damping(float(ang_damp))
            if mass is not None:
                try:
                    entity.set_mass(float(mass))
                except Exception:
                    rigid.mass = float(mass)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        return rigid

    def _tune_actor_friction(self, entity, static_f=2.4, dynamic_f=2.0, restitution=0.0):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return
        try:
            for s in rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(float(static_f))
                m.set_dynamic_friction(float(dynamic_f))
                m.set_restitution(float(restitution))
        except Exception:
            pass

    def _settle_physics(self, steps=60):
        """Advance PhysX without recording (egg settle into the bowl)."""
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()

    def _decor_pose(self, name, default_xy, default_z=None):
        layout = (getattr(self, "_decor_layout", None) or {}).get(name)
        if layout is None:
            z = self.table_top + 0.001 if default_z is None else float(default_z)
            return (
                np.array([float(default_xy[0]), float(default_xy[1])], dtype=float),
                z,
            )
        return np.asarray(layout["xy"], dtype=float), float(layout["z"])

    def _build_baking_props(self):
        """Static baking clutter: board+bread, flour, chocolate chips, bowl+eggs."""
        cfg = self._cfg
        q = self.YUP_Q

        # Cutting board with bread on top (table or microwave top).
        bread_xy, bread_z = self._decor_pose("bread", cfg.get("bread_xy", [0.28, -0.20]))
        board_scale = float(cfg.get("board_scale_mult", self.BOARD_SCALE_DEFAULT))
        with open("assets/objects/104_board/model_data0.json", encoding="utf-8") as f:
            board_data = json.load(f)
        board_th = float(board_data["extents"][1]) * board_scale
        board_pose = sapien.Pose(
            [float(bread_xy[0]), float(bread_xy[1]), bread_z + 0.5 * board_th],
            list(self.BOARD_QPOS),
        )
        self.board = create_actor(
            scene=self,
            pose=board_pose,
            modelname="104_board",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=board_scale,
        )
        self.board.set_name("104_board")
        self.board.config = {
            "scale": [board_scale, board_scale, board_scale],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        board_top_z = bread_z + board_th

        self.bread = create_actor(
            scene=self,
            pose=sapien.Pose(
                [float(bread_xy[0]), float(bread_xy[1]), board_top_z + 0.001],
                q,
            ),
            modelname="075_bread",
            model_id=int(cfg.get("bread_id", 0)),
            convex=True,
            is_static=True,
        )

        flour_xy, flour_z = self._decor_pose("flour", cfg.get("flour_xy", [0.38, 0.14]))
        self.flour = create_actor(
            scene=self,
            pose=sapien.Pose([float(flour_xy[0]), float(flour_xy[1]), flour_z], q),
            modelname="261_flour_sack",
            model_id=0,
            convex=True,
            is_static=True,
        )

        chips_xy, chips_z = self._decor_pose("chips", cfg.get("chips_xy", [0.12, -0.22]))
        self.chocolate_chips = create_actor(
            scene=self,
            pose=sapien.Pose([float(chips_xy[0]), float(chips_xy[1]), chips_z], q),
            modelname="263_chocolate_chips_bag",
            model_id=0,
            convex=True,
            is_static=True,
        )

        bowl_xy, bowl_z = self._decor_pose("bowl", cfg.get("bowl_xy", [0.40, -0.02]))
        bowl_id = int(cfg.get("bowl_id", 1))
        bowl_scale_mult = float(cfg.get("bowl_scale_mult", 1.0))
        # Static + nonconvex keeps a hollow interior so eggs can fall inside.
        self.bowl = create_actor(
            scene=self,
            pose=sapien.Pose([float(bowl_xy[0]), float(bowl_xy[1]), bowl_z], q),
            modelname="002_bowl",
            model_id=bowl_id,
            convex=False,
            is_static=True,
            scale_mult=bowl_scale_mult,
        )
        with open(
            f"assets/objects/002_bowl/model_data{bowl_id}.json", encoding="utf-8"
        ) as f:
            bowl_data = json.load(f)
        bowl_scale = float(bowl_data["scale"][1]) * bowl_scale_mult
        bowl_h = float(bowl_data["extents"][1]) * bowl_scale
        bowl_r = 0.5 * min(
            float(bowl_data["extents"][0]), float(bowl_data["extents"][2])
        ) * bowl_scale

        # Drop full-size eggs from just above the rim; gravity seats them inside.
        # 262_egg pose origin = mesh bottom (local Y≈0).
        drop_offsets = [
            (-0.010, 0.0),
            (0.010, 0.0),
            (0.0, 0.008),
        ]
        self.eggs = []
        for i, (dx, dy) in enumerate(drop_offsets):
            drop_z = bowl_z + bowl_h + 0.012 + 0.010 * i
            egg = create_actor(
                scene=self,
                pose=sapien.Pose(
                    [
                        float(bowl_xy[0]) + dx,
                        float(bowl_xy[1]) + dy,
                        float(drop_z),
                    ],
                    q,
                ),
                modelname="262_egg",
                model_id=0,
                convex=True,
                is_static=False,
            )
            self._recolor_actor(egg, self.EGG_ORANGE)
            self._make_dynamic(egg, mass=0.05, lin_damp=1.2, ang_damp=1.8)
            self._tune_actor_friction(egg, static_f=3.0, dynamic_f=2.6, restitution=0.0)
            self.eggs.append(egg)
            self._settle_physics(60)
        self._settle_physics(100)

        # Re-drop any escapee from a short height (still PhysX, no pose snap).
        bowl_center = np.asarray([float(bowl_xy[0]), float(bowl_xy[1])], dtype=float)
        for _ in range(2):
            any_bad = False
            for egg, (dx, dy) in zip(self.eggs, drop_offsets):
                ep = np.asarray(egg.get_pose().p, dtype=float)
                xy = float(np.linalg.norm(ep[:2] - bowl_center))
                inside = (
                    xy < 0.80 * bowl_r
                    and ep[2] > bowl_z - 0.005
                    and ep[2] < bowl_z + bowl_h + 0.07
                )
                if inside:
                    continue
                any_bad = True
                retry = sapien.Pose(
                    [
                        float(bowl_xy[0]) + 0.5 * dx,
                        float(bowl_xy[1]) + 0.5 * dy,
                        float(bowl_z + bowl_h + 0.018),
                    ],
                    q,
                )
                try:
                    egg.actor.set_pose(retry)
                except Exception:
                    egg.set_pose(retry)
                rigid = self._get_rigid(egg)
                if rigid is not None:
                    try:
                        rigid.set_linear_velocity(np.zeros(3))
                        rigid.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
            if not any_bad:
                break
            self._settle_physics(80)

        for egg in self.eggs:
            rigid = self._get_rigid(egg)
            if rigid is None:
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(2.5)
                rigid.set_angular_damping(3.5)
            except Exception:
                pass

    def _sync_jar_followers(self):
        """Keep glass visual, fill rings, and oil column attached to the jar."""
        if self.jar is None:
            return
        pose = self.jar.get_pose()
        if self.jar_visual is not None:
            self.jar_visual.set_pose(pose)
        for frac, ent in getattr(self, "_ring_entities", []) or []:
            z_local = self.JAR_BOTTOM_T + frac * self.jar_fillable_h
            ent.set_pose(pose * sapien.Pose([0.0, 0.0, z_local]))
        if self._liquid_entity is not None:
            self._liquid_entity.set_pose(pose * sapien.Pose([0.0, 0.0, self.JAR_BOTTOM_T]))
        # Cache jar XY for stream / debug; liquid uses jar pose directly.
        self.jar_xy = np.asarray(pose.p[:2], dtype=float)
        self.jar_bottom_z = float(pose.p[2]) + self.JAR_BOTTOM_T

    # -------------------------------------------------- interactive capture hooks
    def interactive_record_visual_state(self) -> dict:
        """Scalars driving the render-only oil visuals (see restore below)."""
        return {
            "liquid_level": float(self.liquid_level),
            "spill_amount": float(self.spill_amount),
            "tab_open": bool(self.tab_open),
        }

    def interactive_restore_visual_state(self, state) -> None:
        """Rebuild the oil visuals for a logged frame during offscreen capture.

        The jar glass, oil column, stream, and spill puddle are render-only
        entities rebuilt by task code, so restoring actor poses alone would leave
        them frozen at the end-of-episode state. Assigning ``tab_open`` directly
        keeps ``_set_tab_open`` scoring latches out of the replay.
        """
        if not state:
            return
        self.liquid_level = float(state.get("liquid_level", self.liquid_level))
        self.spill_amount = float(state.get("spill_amount", self.spill_amount))
        self.tab_open = bool(state.get("tab_open", self.tab_open))
        self._liquid_half_h_cached = -1.0
        self._spill_radius_cached = -1.0
        self._rebuild_liquid(force=True)
        self._rebuild_spill(force=True)
        self._sync_stream()
        self._sync_jar_followers()

    # ------------------------------------------------------------------ oil visuals / dynamics
    def _set_tab_open(self, open_: bool):
        """Latch pour state; key stays down/red when ON, up/green when OFF."""
        open_ = bool(open_)
        was_open = bool(self.tab_open)
        if open_ == was_open:
            self._set_button_press_depth(self._switch_travel() if open_ else 0.0)
            self._sync_stream()
            return
        self.tab_open = open_
        if self.tab_open and not was_open:
            self.opened_once = True
            if self._metric_open_step is None:
                self._metric_open_step = self._metric_step()
        # Success / fail is scored only after the key returns to OFF.
        if was_open and not self.tab_open and self.opened_once:
            self.closed_after_pour = True
            self._latch_close_metric()
        self._set_button_press_depth(self._switch_travel() if open_ else 0.0)
        self._sync_stream()
        # Force a liquid refresh so ON/OFF is immediately visible in interactive.
        self._rebuild_liquid(force=True)

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_open_step = None    # nozzle key first turned ON
        self._metric_band_step = None    # fill first reached the lower band edge
        self._metric_close_step = None   # key turned back OFF (decisive event)
        self._metric_close_level = None  # fill frozen at that instant

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_fill_metrics(self):
        """Stamp the frame the jar first reaches the bottom of the success band."""
        try:
            if self._metric_band_step is not None:
                return
            lo, _hi = self._fill_band()
            if float(self.liquid_level) + 1e-3 >= float(lo):
                self._metric_band_step = self._metric_step()
        except Exception:
            pass

    def _latch_close_metric(self):
        """Called from _set_tab_open on the ON->OFF edge — LAST close wins.

        A jar that was under-filled and topped up is scored on the final shutoff,
        so the latch is deliberately overwritten rather than kept as a high-water mark.
        """
        try:
            self._metric_close_step = self._metric_step()
            self._metric_close_level = float(self.liquid_level)
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `shutoff_latency_steps` — steps from the fill first reaching the bottom
        of the success band until the nozzle key is turned OFF. The oil keeps running
        the whole time, so every step here eats into the overfill margin.
        extra2 `fill_error_norm` — |fill at shutoff - target_fill| / fill_tol. LOWER is
        better; <= 1.0 means the jar landed inside the success band.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_band_step, self._metric_close_step
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["shutoff_latency_steps"] = lat
        out["shutoff_latency_s"] = None if lat is None else round(float(lat) * dt, 4)
        pour = (None if (self._metric_open_step is None or a is None)
                else max(int(a) - int(self._metric_open_step), 0))
        out["pour_latency_steps"] = pour

        lvl = self._metric_close_level
        try:
            tol = max(float(getattr(self, "fill_tol", self.FILL_TOL)), 1e-6)
            out["fill_error_norm"] = (
                None if lvl is None
                else round(abs(float(lvl) - float(self.target_fill)) / tol, 4))
            out["target_fill"] = round(float(self.target_fill), 4)
        except Exception:
            out["fill_error_norm"] = None
            out["target_fill"] = None
        out["shutoff_fill"] = None if lvl is None else round(float(lvl), 4)
        try:
            out["spill_amount"] = round(float(self.spill_amount), 4)
        except Exception:
            out["spill_amount"] = None
        out["overflowed"] = bool(getattr(self, "overflowed", False))
        return out

    def _sync_stream(self):
        """Narrow oil cylinder from nozzle outlet down to the table (tab-gated)."""
        self._stream_entity = self._remove_entity(self._stream_entity)
        if not self.tab_open:
            return
        ox, oy, oz = self.nozzle_outlet_xyz
        z_lo = self.table_top + 0.001
        half_h = max(0.01, 0.5 * (oz - z_lo))
        z_c = 0.5 * (oz + z_lo)
        self._stream_entity = self._make_oil_column(
            radius=float(self.NOZZLE_R),
            half_h=half_h,
            world_xyz=[ox, oy, z_c],
            rgba=self.oil_stream_color,
            name="oil_stream",
            local_z=0.0,
        )

    def _rebuild_liquid(self, force: bool = False):
        """Rising oil column in the jar (opaque sunflower yellow or transparent)."""
        if not getattr(self, "jar_fillable_h", None):
            return
        liq_h = max(0.0, float(self.liquid_level)) * self.jar_fillable_h
        liq_half = max(0.002, 0.5 * liq_h) if self.liquid_level > 1e-4 else 0.0
        # While pouring, refresh often so the level keeps visibly rising
        # (interactive real-time steps are much slower than expert idle loops).
        interactive = bool(getattr(self, "_interactive_robot_mode", False)) or bool(
            getattr(self, "_interactive_universal_controls", False)
        )
        if bool(getattr(self, "tab_open", False)):
            min_dh = 0.00025 if interactive else 0.0008
        else:
            min_dh = 0.002
        if (
            not force
            and abs(liq_half - self._liquid_half_h_cached) < min_dh
        ):
            return
        self._liquid_half_h_cached = liq_half
        self._liquid_entity = self._remove_entity(self._liquid_entity)
        if self.liquid_level <= 1e-4:
            return

        if self.jar is not None:
            jar_pose = self.jar.get_pose()
            liquid_pose = jar_pose * sapien.Pose([0.0, 0.0, self.JAR_BOTTOM_T])
        else:
            liquid_pose = sapien.Pose(
                [float(self.jar_xy[0]), float(self.jar_xy[1]), float(self.jar_bottom_z)]
            )
        ent = sapien.Entity()
        ent.set_name("olive_oil_liquid")
        ent.set_pose(liquid_pose)
        body = sapien.render.RenderBodyComponent()

        # Viewer hollow shell: opaque oil + slightly fuller radius so the level
        # reads clearly through the alpha glass from the side.
        viewer_shell = bool(getattr(self, "_jar_visual_hollow", False))
        if viewer_shell:
            bulk_mat = self._viewer_oil_material()
            liq_r = self.JAR_INNER_R * 0.97
        else:
            bulk_mat = self._fluid_material(self.oil_color)
            liq_r = self.JAR_INNER_R * 0.90
        bulk = sapien.render.RenderShapeCylinder(
            radius=liq_r,
            half_length=liq_half,
            material=bulk_mat,
        )
        bulk.set_local_pose(sapien.Pose([0.0, 0.0, liq_half], self.UPRIGHT_CYL_Q))
        body.attach(bulk)

        # Optional meniscus only for the see-through style.
        if (
            not viewer_shell
            and self.oil_transparent
            and self.oil_meniscus is not None
        ):
            men_half = 0.0012
            men_mat = self._fluid_material(self.oil_meniscus)
            men = sapien.render.RenderShapeCylinder(
                radius=self.JAR_INNER_R * 0.92,
                half_length=men_half,
                material=men_mat,
            )
            men.set_local_pose(
                sapien.Pose(
                    [0.0, 0.0, max(men_half, 2.0 * liq_half - men_half)],
                    self.UPRIGHT_CYL_Q,
                )
            )
            body.attach(men)

        ent.add_component(body)
        self.scene.add_entity(ent)
        self._liquid_entity = ent

    def _spill_radius(self):
        """Table puddle radius from spill_amount in [0, 1]."""
        a = float(np.clip(self.spill_amount, 0.0, 1.0))
        if a <= 1e-4:
            return 0.0
        # Ease-out growth so the puddle appears quickly, then spreads.
        t = float(np.sqrt(a))
        return float(self.SPILL_RADIUS_MIN + t * (self.SPILL_RADIUS_MAX - self.SPILL_RADIUS_MIN))

    def _rebuild_spill(self, force: bool = False):
        """Yellow oil circle on the table under the jar (overflow puddle)."""
        radius = self._spill_radius()
        if not force and abs(radius - self._spill_radius_cached) < 0.0015:
            return
        self._spill_radius_cached = radius
        self._spill_entity = self._remove_entity(self._spill_entity)
        if radius <= 1e-4:
            return

        if self._spill_xy is None:
            nozzle = getattr(self, "nozzle_outlet_xyz", None)
            if nozzle is not None:
                self._spill_xy = np.array(
                    [float(nozzle[0]), float(nozzle[1])], dtype=float
                )
            elif self.jar is not None:
                jp = np.asarray(self.jar.get_pose().p, dtype=float)
                self._spill_xy = np.array([jp[0], jp[1]], dtype=float)
            else:
                fill = getattr(self, "fill_xy", None)
                self._spill_xy = np.asarray(
                    fill if fill is not None else self.jar_xy, dtype=float
                ).copy()

        sx, sy = float(self._spill_xy[0]), float(self._spill_xy[1])
        z = float(self.table_top) + 0.0022
        rgba = list(getattr(self, "oil_spill_color", self.OIL_SPILL_SOLID))
        # Flat cylinder lying on the table (same upright quat as oil columns).
        self._spill_entity = self._make_oil_column(
            radius=radius,
            half_h=self.SPILL_HALF_H,
            world_xyz=[sx, sy, z + self.SPILL_HALF_H],
            rgba=rgba,
            name="oil_spill",
            local_z=0.0,
        )

    def _jar_under_nozzle(self) -> bool:
        """True when the nozzle stream would land inside the jar mouth."""
        if self.jar is None:
            return False
        nozzle = getattr(self, "nozzle_outlet_xyz", None)
        if nozzle is None:
            fill = getattr(self, "fill_xy", None)
            if fill is None:
                return False
            nx, ny = float(fill[0]), float(fill[1])
        else:
            nx, ny = float(nozzle[0]), float(nozzle[1])
        jp = np.asarray(self.jar.get_pose().p, dtype=float)
        dist = float(np.hypot(jp[0] - nx, jp[1] - ny))
        return dist <= float(getattr(self, "jar_catch_r", self.JAR_CATCH_R))

    def _step_oil(self):
        # While ON: fill the jar only if it sits under the nozzle; otherwise the
        # stream hits the table (spill / fail). Overfilling the jar also spills.
        if self.tab_open:
            catching = self._jar_under_nozzle()
            if catching:
                next_lvl = float(self.liquid_level) + float(self.pour_rate)
                if next_lvl <= float(self.overflow_level):
                    self.liquid_level = next_lvl
                else:
                    self.liquid_level = float(self.overflow_level)
                    self.spill_amount = min(
                        1.0, float(self.spill_amount) + float(self.spill_rate)
                    )
                    if self.spill_amount > 1e-4:
                        self.overflowed = True
            else:
                # Miss: oil puddles under the nozzle outlet.
                self.spill_amount = min(
                    1.0, float(self.spill_amount) + float(self.spill_rate)
                )
                if self.spill_amount > 1e-4:
                    self.overflowed = True
                nozzle = getattr(self, "nozzle_outlet_xyz", None)
                if nozzle is not None:
                    self._spill_xy = np.array(
                        [float(nozzle[0]), float(nozzle[1])], dtype=float
                    )
            if getattr(self, "_stream_entity", None) is None:
                self._sync_stream()
        self._rebuild_liquid(force=False)
        self._rebuild_spill(force=False)
        self._track_fill_metrics()

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if getattr(self, "_jar_carry", False) and self.jar is not None:
            self._apply_jar_carry()
        elif getattr(self, "_jar_seated_pose", None) is not None and self.jar is not None:
            # Firm seat on the scale — no slow PhysX drop after release.
            try:
                self.jar.actor.set_pose(self._jar_seated_pose)
                for component in self.jar.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        elif (
            getattr(self, "_jar_locked", False)
            and not getattr(self, "_push_active", False)
            and self.jar is not None
        ):
            try:
                self.jar.actor.set_pose(self._jar_home_pose)
                for component in self.jar.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._update_switch_visual_from_robot()
        self._detect_tab_touch()
        self._step_oil()
        self._sync_jar_followers()

    def _clear_tab_collision(self):
        """Drop switch collision after the pour so the jar is reachable."""
        self._ignore_tab = True
        open_ = bool(self.tab_open)
        depth = self._switch_travel() if open_ else 0.0
        self._clear_switch_parts()
        # Keep a visual-only housing + key parked at the latched pose.
        bx, by, bz = self.switch_base_xyz
        housing = self._metallic_material([0.18, 0.18, 0.20], roughness=0.45, metallic=0.55)
        self._switch_parts.append(
            self._add_static_box(
                pose=sapien.Pose([bx, by, bz + float(self.SWITCH_BASE_HALF[2])]),
                half_size=list(self.SWITCH_BASE_HALF),
                material=housing,
                name="oil_switch_base_visual",
                collision=False,
            )
        )
        base_top = bz + float(self.SWITCH_BASE_HALF[2])
        btn_z = base_top + float(self.SWITCH_BTN_HALF[2])
        home = sapien.Pose([bx, by, btn_z])
        self._button_home_pose = home
        rgb = self.SWITCH_COLOR_DOWN if open_ else self.SWITCH_COLOR_UP
        self._switch_btn = self._add_static_box(
            pose=sapien.Pose([bx, by, btn_z - depth]),
            half_size=list(self.SWITCH_BTN_HALF),
            material=self._opaque_material(rgb),
            name="oil_switch_button_visual",
            collision=False,
        )
        self._switch_parts.append(self._switch_btn)
        self._cache_switch_key_shapes(self._switch_btn)
        self._set_switch_key_color(bool(open_))
        self._button_visual_depth = depth
        self._button_target_depth = depth
        self._button_pressed_visual = bool(open_)
        self._sync_switch_touch_points()

    def _jar_grasp_quat(self):
        """Top-down pinch (fingers around the upper jar body)."""
        return list(GRASP_DIRECTION_DIC["top_down"])

    def _jar_side_grasp_quat(self):
        """Compatibility alias — carry / place use the top-down pinch quat."""
        return self._jar_grasp_quat()

    def _tcp_pose7(self, arm_tag: ArmTag):
        if str(arm_tag) == "left":
            return np.asarray(self.robot.get_left_tcp_pose(), dtype=float)
        return np.asarray(self.robot.get_right_tcp_pose(), dtype=float)

    def _ee_pose_from_tcp(self, tcp_xyz, quat) -> list:
        """Convert a TCP target to the EE pose ``move_to_pose`` expects.

        Robot EE poses sit ``EE_TO_TCP`` behind the TCP along local +X
        (``robot._trans_endpose``).
        """
        R = np.asarray(
            sapien.Pose([0.0, 0.0, 0.0], list(quat)).to_transformation_matrix()[
                :3, :3
            ],
            dtype=float,
        )
        ee = np.asarray(tcp_xyz, dtype=float) - R @ np.array(
            [float(self.EE_TO_TCP), 0.0, 0.0], dtype=float
        )
        return [float(ee[0]), float(ee[1]), float(ee[2]), *list(quat)]

    def _seat_jar_in_gripper(self, arm_tag: ArmTag, lift: float = 0.0):
        """Snap jar axis into the TCP so the fingers visually enclose it.

        ``lift`` raises the jar into the finger pads (cm-scale) so it is not
        hanging under the palm with an air gap.
        """
        if self.jar is None:
            return
        tcp = self._tcp_pose7(arm_tag)[:3]
        jar_z = float(np.asarray(self.jar.get_pose().p, dtype=float)[2])
        jar_z = max(jar_z, float(self.table_top) + 0.001) + float(lift)
        pose = sapien.Pose([float(tcp[0]), float(tcp[1]), jar_z])
        self.jar.actor.set_pose(pose)
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_linear_velocity(np.zeros(3))
                    component.set_angular_velocity(np.zeros(3))
                except Exception:
                    pass
        self._sync_jar_followers()

    def _tcp_in_jar_grasp(self, arm_tag: ArmTag, jar_p) -> bool:
        """True if TCP is on the jar axis and mid-body (fingers can enclose it)."""
        tcp = self._tcp_pose7(arm_tag)[:3]
        dist_xy = float(np.linalg.norm(tcp[:2] - jar_p[:2]))
        z_lo = float(jar_p[2] + 0.028)
        z_hi = float(jar_p[2] + 0.90 * float(self.JAR_HEIGHT))
        ok = dist_xy <= 0.038 and z_lo <= float(tcp[2]) <= z_hi
        if not ok:
            print(
                f"[measure_ingredient] TCP not in grasp volume: "
                f"xy={dist_xy:.3f}m z={float(tcp[2]):.3f} "
                f"(need [{z_lo:.3f},{z_hi:.3f}])"
            )
        return ok

    def _grasp_jar_from_side(self, arm_tag: ArmTag) -> bool:
        """Top-down pinch via the −Y corridor, then slide clear of the spout.

        One continuous top-down wrist from approach → close → retract → place.
        Do **not** try a side ``grasp_actor`` first — a missed side attempt
        followed by a top-down insert looks like “side stick then re-grasp from
        above”. Weld only when TCP is mid-body.
        """
        if self.jar is None:
            return False
        self._ignore_tab = True
        self._touch_latched = False
        self.plan_success = True
        self._clear_tab_collision()
        self._jar_locked = False
        self._jar_seated_pose = None
        self._set_jar_damping(3.0)

        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        if not self._grasp_jar_corridor_insert(arm_tag, jar_p):
            print("[measure_ingredient] jar corridor grasp failed")
            self.plan_success = False
            return False

        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        if not self._tcp_in_jar_grasp(arm_tag, jar_p):
            self.plan_success = False
            return False

        # Commanded top-down for the whole carry (stable place IK).
        grasp_quat = self._jar_grasp_quat()
        # Lift a little into the pads so the glass is enclosed, not hanging.
        self._seat_jar_in_gripper(arm_tag, lift=0.018)
        self._start_jar_carry(arm_tag, grasp_quat=grasp_quat)

        # Slide out from under the spout first, then lift — never through it.
        self.move(
            self.move_by_displacement(
                arm_tag, y=-0.16, quat=grasp_quat, move_axis="world"
            )
        )
        if not self.plan_success:
            self.plan_success = True
        self.move(
            self.move_by_displacement(
                arm_tag, z=0.12, quat=grasp_quat, move_axis="world"
            )
        )
        if not self.plan_success:
            self.plan_success = True
        self.plan_success = True
        return True

    def _grasp_jar_corridor_insert(self, arm_tag: ArmTag, jar_p) -> bool:
        """Fallback: −Y corridor → above jar → deepen TCP into the body → close."""
        # Upper body / near-rim pinch ("from the side on the top").
        grasp_z = float(jar_p[2] + 0.092)
        hover_z = float(jar_p[2] + float(self.JAR_HEIGHT) + 0.08)
        quat = self._jar_grasp_quat()
        tcp_side = [float(jar_p[0]), float(jar_p[1] - 0.18), hover_z]
        tcp_above = [float(jar_p[0]), float(jar_p[1]), hover_z]

        self.move(self.open_gripper(arm_tag))
        if not self.plan_success:
            self.plan_success = True
        self.move(
            self.move_to_pose(arm_tag, self._ee_pose_from_tcp(tcp_side, quat))
        )
        if not self.plan_success:
            return False
        self.move(
            self.move_to_pose(arm_tag, self._ee_pose_from_tcp(tcp_above, quat))
        )
        if not self.plan_success:
            return False

        # Keep stepping down until TCP is mid-body — do not weld from hover.
        for _ in range(8):
            tcp = self._tcp_pose7(arm_tag)[:3]
            dz = float(grasp_z - tcp[2])
            if abs(dz) < 0.006:
                break
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm_tag,
                    z=float(np.clip(dz, -0.04, 0.04)),
                    quat=quat,
                    move_axis="world",
                )
            )
            if not self.plan_success:
                print("[measure_ingredient] corridor descend step failed")
                return False

        if not self._tcp_in_jar_grasp(arm_tag, jar_p):
            # Last try: absolute EE for the grasp TCP.
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm_tag,
                    self._ee_pose_from_tcp(
                        [float(jar_p[0]), float(jar_p[1]), grasp_z], quat
                    ),
                )
            )
            if not self.plan_success or not self._tcp_in_jar_grasp(arm_tag, jar_p):
                return False

        self.move(self.close_gripper(arm_tag))
        return bool(self.plan_success)

    def enable_interactive_jar_push(self) -> None:
        """Unlock the jar for contact pushing in the interactive viewer."""
        self._jar_locked = False
        self._enable_jar_push_physics()
        print(
            "[measure_ingredient] jar unlocked for pushing — "
            "shove it under the nozzle, then press the green key"
        )

    def interactive_grasp_jar(self, arm_tag: ArmTag) -> bool:
        """Interactive Space: unused (task is push + key, not grasp)."""
        print(
            "[measure_ingredient] Space grasp disabled — "
            "push the jar under the nozzle with a closed gripper"
        )
        return False

    def interactive_release_jar(self, arm_tag: ArmTag) -> bool:
        """Interactive Space: open gripper; freeze jar where it is."""
        if self.jar is None:
            return False
        self._ignore_tab = True
        self.plan_success = True
        self._push_active = False
        self._stop_jar_carry()
        pp = np.asarray(self.jar.get_pose().p, dtype=float)
        self._freeze_jar(
            sapien.Pose(
                [float(pp[0]), float(pp[1]), float(self.table_top) + 0.001],
                list(self.jar.get_pose().q),
            )
        )
        self.move(self.open_gripper(arm_tag))
        if not self.plan_success:
            self.plan_success = True
        print(
            f"[measure_ingredient] jar parked under_nozzle={self._jar_under_nozzle()} "
            f"liq={self.liquid_level:.2f}"
        )
        return True

    def _ee_pose7(self, arm_tag: ArmTag):
        if str(arm_tag) == "left":
            return np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        return np.asarray(self.robot.get_right_ee_pose(), dtype=float)

    def _start_jar_carry(self, arm_tag: ArmTag, grasp_quat=None):
        """Weld upright jar to the TCP (finger center), not the EE/palm frame.

        Mixing EE position with a commanded wrist quat produced a floating gap
        under the gripper. TCP-relative hold keeps the jar between the fingers.
        """
        tcp = self._tcp_pose7(arm_tag)[:3]
        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        quat = (
            np.asarray(grasp_quat, dtype=float)
            if grasp_quat is not None
            else np.asarray(self._ee_pose7(arm_tag)[3:7], dtype=float)
        )
        # Force axis under TCP; keep only the vertical finger→bottom offset.
        self._jar_carry_local = np.array(
            [0.0, 0.0, float(jar_p[2] - tcp[2])], dtype=float
        )
        self._jar_carry_quat = quat.copy()
        self._jar_carry_arm = ArmTag(str(arm_tag))
        self._jar_carry = True
        self._jar_locked = False
        self._apply_jar_carry()

    def _stop_jar_carry(self, place_xyz=None):
        """Release the kinematic attach. Optionally settle at ``place_xyz``."""
        self._jar_carry = False
        self._jar_carry_local = None
        self._jar_carry_quat = None
        self._jar_carry_arm = None
        if place_xyz is not None and self.jar is not None:
            self.jar.actor.set_pose(
                sapien.Pose(
                    [float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])]
                )
            )
            for component in self.jar.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    try:
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
        self._sync_jar_followers()

    def _apply_jar_carry(self):
        if self.jar is None or self._jar_carry_local is None:
            return
        arm = getattr(self, "_jar_carry_arm", None) or self.arm
        tcp = self._tcp_pose7(arm)[:3]
        off = np.asarray(self._jar_carry_local, dtype=float)
        # Always centered under the finger TCP; upright jar.
        p = np.array(
            [float(tcp[0]), float(tcp[1]), float(tcp[2] + off[2])], dtype=float
        )
        try:
            self.jar.actor.set_pose(
                sapien.Pose([float(p[0]), float(p[1]), float(p[2])])
            )
            for component in self.jar.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.set_linear_velocity(np.zeros(3))
                    component.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    def _ee_pose_for_carried_jar(self, jar_xyz):
        """EE pose (grasp quat) that puts the carried jar at ``jar_xyz``."""
        off = np.asarray(self._jar_carry_local, dtype=float)
        stored = getattr(self, "_jar_carry_quat", None)
        quat = list(stored) if stored is not None else self._jar_grasp_quat()
        dest = np.asarray(jar_xyz, dtype=float)
        tcp = np.array(
            [float(dest[0]), float(dest[1]), float(dest[2] - off[2])], dtype=float
        )
        return self._ee_pose_from_tcp(tcp, quat)

    def _carry_jar_to(self, arm_tag: ArmTag, dest_xyz, tol: float = 0.018):
        """Drive the arm (absolute pose, then axis tweaks) until jar is at dest."""
        dest = np.asarray(dest_xyz, dtype=float)
        if self.jar is None or self._jar_carry_local is None:
            return False

        self.move(self.move_to_pose(arm_tag, self._ee_pose_for_carried_jar(dest)))
        if not self.plan_success:
            self.plan_success = True

        # Axis-separated nudges until the jar is actually there (no end snap).
        for _ in range(8):
            self._apply_jar_carry()
            jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
            err = dest - jar_p
            if float(np.linalg.norm(err)) <= tol:
                return True
            moved = False
            for axis, val in (("x", err[0]), ("y", err[1]), ("z", err[2])):
                if abs(float(val)) < 0.008:
                    continue
                kw = {axis: float(val)}
                self.move(self.move_by_displacement(arm_tag, **kw))
                moved = True
                if not self.plan_success:
                    self.plan_success = True
            if not moved:
                break
        self._apply_jar_carry()
        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        return float(np.linalg.norm(dest - jar_p)) <= tol * 1.5

    def _seat_jar_on_scale(self, place_xyz):
        """Release carry and lock the jar where it already is (plate height)."""
        self._jar_carry = False
        self._jar_carry_local = None
        self._jar_carry_quat = None
        self._jar_carry_arm = None
        self._jar_locked = False
        # Use current XY (arm already delivered it); only set Z to the plate.
        if self.jar is not None:
            cur = np.asarray(self.jar.get_pose().p, dtype=float)
            target = np.asarray(place_xyz, dtype=float)
            # Tiny XY blend only if already near center — never a long snap.
            dist_xy = float(np.linalg.norm(cur[:2] - target[:2]))
            if dist_xy < 0.025:
                xy = target[:2]
            else:
                xy = cur[:2]
                print(
                    f"[measure_ingredient] seat without snap "
                    f"(jar still {dist_xy:.3f}m from center)"
                )
            pose = sapien.Pose([float(xy[0]), float(xy[1]), float(target[2])])
        else:
            pose = sapien.Pose(
                [float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])]
            )
        self._jar_seated_pose = pose
        if self.jar is None:
            return
        self.jar.actor.set_pose(pose)
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_linear_velocity(np.zeros(3))
                    component.set_angular_velocity(np.zeros(3))
                    component.set_linear_damping(50.0)
                    component.set_angular_damping(50.0)
                except Exception:
                    pass
        self._sync_jar_followers()

    def _detect_tab_touch(self):
        """Physical key: ON on press; OFF (and stream stop) on release back up."""
        if getattr(self, "_ignore_tab", False):
            self._touch_latched = False
            return
        if getattr(self, "_switch_btn", None) is None or not hasattr(self, "robot"):
            return
        signal = self._switch_press_signal()
        touching = bool(
            signal is not None
            and float(signal["force"]) > float(self.SWITCH_ENGAGE_FORCE)
        )
        if touching and not self._touch_latched:
            if not self.tab_open:
                self._set_tab_open(True)
                self._pending_tab_off = False
                print(
                    f"[measure_ingredient] key pressed → ON (held down) "
                    f"liq={self.liquid_level:.2f}"
                )
            else:
                # Second press arms OFF; oil keeps flowing until release.
                self._pending_tab_off = True
                print(
                    f"[measure_ingredient] key pressed → OFF pending "
                    f"(flow until key returns up) liq={self.liquid_level:.2f}"
                )
        if (
            getattr(self, "_pending_tab_off", False)
            and not touching
            and self._touch_latched
        ):
            self._pending_tab_off = False
            self._set_tab_open(False)
            print(
                f"[measure_ingredient] key released → OFF (up) "
                f"liq={self.liquid_level:.2f}"
            )
        self._touch_latched = touching

    def _idle_steps(self, n_steps: int, until=None):
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


    # ------------------------------------------------------------------ push jar (catch_cup style)
    def _dwell(self, steps: int) -> None:
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()

    def _push_quat(self, arm_tag):
        return list(GRASP_DIRECTION_DIC["top_down"])

    def _tcp_pos(self, arm_tag):
        pose = (
            self.robot.get_left_tcp_pose()
            if str(arm_tag) == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.array(pose[:3], dtype=float)

    def _move_tcp(self, arm_tag, xy, z, quat) -> bool:
        """Absolute TCP move via EE pose (top-down: EE sits ``EE_TO_TCP`` above TCP)."""
        self.plan_success = True
        pose = self._ee_pose_from_tcp(
            [float(xy[0]), float(xy[1]), float(z)], list(quat)
        )
        self.move(self.move_to_pose(arm_tag, pose))
        return bool(self.plan_success)

    def _jar_live_xy(self):
        return np.asarray(self.jar.get_pose().p[:2], dtype=float)

    def _measure_jar_extents(self):
        rigid = self._get_rigid(self.jar)
        if rigid is None:
            self.jar_half_xy = list(self.JAR_HALF_XY)
            self.jar_height = float(self.JAR_HEIGHT)
            return
        try:
            lo, hi = rigid.compute_global_aabb_tight()
        except Exception:
            self.jar_half_xy = list(self.JAR_HALF_XY)
            self.jar_height = float(self.JAR_HEIGHT)
            return
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        half = 0.5 * (hi - lo)
        self.jar_half_xy = [float(half[0]), float(half[1])]
        self.jar_height = float(hi[2] - lo[2])

    def _finger_drop(self, arm_tag):
        art = (
            self.robot.left_entity
            if str(arm_tag) == "left"
            else self.robot.right_entity
        )
        lows = []
        try:
            for link in art.get_links():
                name = link.get_name().lower()
                if "finger" not in name and "gripper" not in name:
                    continue
                for c in link.get_components():
                    if isinstance(c, sapien.physx.PhysxRigidBaseComponent):
                        lows.append(float(c.compute_global_aabb_tight()[0][2]))
                        break
        except Exception:
            pass
        if not lows:
            return float(self.PUSH_FINGER_DROP)
        drop = float(self._tcp_pos(arm_tag)[2]) - min(lows)
        return float(np.clip(drop, 0.0, 0.12))

    def _freeze_jar(self, pose=None) -> None:
        if getattr(self, "_push_active", False):
            return
        if pose is None and self.jar is not None:
            p = np.asarray(self.jar.get_pose().p, dtype=float)
            p[2] = float(self.table_top) + 0.001
            pose = sapien.Pose(
                [float(p[0]), float(p[1]), float(p[2])],
                list(self.jar.get_pose().q),
            )
        if pose is not None and self.jar is not None:
            self.jar.actor.set_pose(pose)
            self._jar_home_pose = pose
        rigid = self._get_rigid(self.jar)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic(True)
            except Exception:
                pass
        self._jar_locked = True
        self._push_active = False

    def _enable_jar_push_physics(self) -> None:
        """Dynamic jar resting on the table for a contact shove (catch_cup pillow).

        Z stays unlocked so gravity makes a real table normal force; Coulomb
        friction then brakes the slide. Roll/pitch stay locked so a low shove
        cannot tip the cylinder; yaw stays free.
        """
        if self.jar is None:
            return
        rigid = self._get_rigid(self.jar)
        if rigid is None:
            return
        if not getattr(self, "_push_active", False):
            # Cylinder collision bottom is at actor origin — seat flush on the
            # counter so contact (and friction) engage once gravity is on.
            p = np.asarray(self.jar.get_pose().p, dtype=float)
            p[2] = float(self.table_top)
            self.jar.actor.set_pose(
                sapien.Pose([float(p[0]), float(p[1]), float(p[2])], list(self.jar.get_pose().q))
            )
        try:
            rigid.set_kinematic(False)
            try:
                rigid.set_mass(float(self.JAR_MASS))
            except Exception:
                pass
            rigid.set_disable_gravity(False)
            # Translation free (rests on the table); no roll/pitch, yaw free.
            rigid.set_locked_motion_axes([False, False, False, True, True, False])
            rigid.set_linear_damping(float(self.PUSH_LIN_DAMP))
            rigid.set_angular_damping(2.0)
            try:
                rigid.set_max_linear_velocity(0.45)
            except Exception:
                pass
            if not getattr(self, "_push_active", False):
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            for shape in rigid.get_collision_shapes():
                shape.set_collision_groups([1, 1, 0, 0])
            rigid.wake_up()
        except Exception:
            pass
        self._tune_actor_friction(
            self.jar,
            static_f=float(self.PUSH_MU_STATIC),
            dynamic_f=float(self.PUSH_MU_DYNAMIC),
            restitution=0.0,
        )
        self._jar_locked = False

    def _push_jar_under_nozzle(self, arm_tag: ArmTag) -> bool:
        """Shove the jar under the nozzle with a closed gripper (contact only).

        Mirrors ``catch_valley_ball``: freeze for approach, unlock PhysX for the
        shove, then park wherever contact left the jar — never ``set_pose``-slide
        to the fill target.
        """
        land_xy = np.asarray(self.fill_xy, dtype=float).copy()
        self._measure_jar_extents()
        pp0 = self._jar_live_xy()
        delta = land_xy - pp0
        dist = float(np.linalg.norm(delta))
        if dist < 0.02:
            # Already seated — freeze in place (no snap to land_xy).
            self._freeze_jar()
            placed = bool(self._jar_under_nozzle())
            self.plan_success = placed
            return placed

        # Prefer +Y push (jar spawn is toward the robot); soft X so glancing
        # sideways shoves do not fling the cylinder off the counter.
        direction = delta / max(dist, 1e-6)
        if abs(float(direction[0])) < 0.45:
            direction = np.array([0.25 * float(direction[0]), float(direction[1])])
            direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        half_along = float(
            abs(direction[0]) * self.jar_half_xy[0]
            + abs(direction[1]) * self.jar_half_xy[1]
        )
        quat = self._push_quat(arm_tag)
        gap = float(self.PUSH_CONTACT_GAP)
        y_min = float(self.TABLE_NEAR_Y) + float(self.PUSH_EDGE_MARGIN)
        min_clear = half_along + gap + 0.015
        standoff = float(self.PUSH_BEHIND_STANDOFF)
        behind = pp0 - direction * (half_along + standoff)
        contact = pp0 - direction * (half_along + gap)
        # Keep the TCP behind the rear face; if the near-edge clamp would put
        # the hand inside the jar, shorten the standoff then side-offset.
        if float(behind[1]) < y_min and abs(float(direction[1])) > 1e-4:
            t_max = (float(pp0[1]) - y_min) / float(direction[1])
            t = float(np.clip(t_max, min_clear, half_along + standoff))
            behind = pp0 - direction * t
        behind[1] = max(float(behind[1]), y_min)
        if float(np.dot(pp0 - behind, direction)) < min_clear - 1e-3:
            side = np.array([-float(direction[1]), float(direction[0])], dtype=float)
            # Bias the side offset toward the arm's natural workspace.
            side_sign = -1.0 if str(arm_tag) == "left" else 1.0
            behind = pp0 - direction * min_clear + side_sign * 0.055 * side
            behind[1] = max(float(behind[1]), y_min)
        contact[1] = max(float(contact[1]), y_min)

        self.move(self.close_gripper(arm_tag=arm_tag))
        self._push_active = False
        self._freeze_jar(
            sapien.Pose(
                [float(pp0[0]), float(pp0[1]), float(self.table_top) + 0.001],
                list(self.jar.get_pose().q),
            )
        )

        drop = self._finger_drop(arm_tag)
        push_z = (
            float(self.table_top)
            + float(self.PUSH_FINGER_HEIGHT_FRAC) * float(self.jar_height)
            + drop
        )
        hover_z = float(self.table_top) + 0.18

        self._move_tcp(arm_tag, behind, hover_z, quat)
        self._move_tcp(arm_tag, behind, push_z, quat)
        tcp_low = self._tcp_pos(arm_tag).copy()
        z_hold = float(np.clip(tcp_low[2], push_z - 0.01, push_z + 0.03))

        self._enable_jar_push_physics()
        self._push_active = True
        self._dwell(2)

        self._move_tcp(arm_tag, contact, z_hold, quat)
        pp = self._jar_live_xy()
        into = pp - direction * max(half_along - 0.005, gap)
        into[1] = max(float(into[1]), y_min)
        self._move_tcp(arm_tag, into, z_hold, quat)
        print(
            f"[measure_ingredient] push approach jar={np.round(pp, 3)} "
            f"land={np.round(land_xy, 3)} behind={np.round(behind, 3)} "
            f"z_hold={z_hold:.3f}"
        )

        step = float(np.clip(getattr(self, "push_step", self.PUSH_STEP_DEFAULT), 0.02, 0.08))
        place_tol = float(self.PUSH_PLACE_TOL)
        n_chunks = int(np.ceil(dist / step)) + 40
        prev_pp = self._jar_live_xy().copy()
        n_stuck = 0
        for _ in range(max(1, n_chunks)):
            pp = self._jar_live_xy()
            err = land_xy - pp
            # Soft X corrections — glancing lateral shoves fling the cylinder.
            if abs(float(err[0])) < 0.04:
                err[0] *= 0.2
            err_n = float(np.linalg.norm(err))
            if self._jar_under_nozzle() or err_n <= place_tol:
                break
            direction = err / max(err_n, 1e-6)
            half_along = float(
                abs(direction[0]) * self.jar_half_xy[0]
                + abs(direction[1]) * self.jar_half_xy[1]
            )
            rear_now = pp - direction * (half_along + gap)
            advance = min(step, max(err_n - 0.5 * place_tol, 0.01))
            if err_n < 0.06:
                advance = min(advance, 0.5 * step)
            aim = rear_now + direction * advance
            aim[1] = max(float(aim[1]), y_min)

            moved = float(np.linalg.norm(pp - prev_pp))
            if moved < 0.004:
                n_stuck += 1
                if n_stuck >= 2:
                    z_hold = max(float(self.table_top) + drop + 0.008, z_hold - 0.008)
                    aim = pp - direction * max(half_along - 0.01, 0.0)
                    aim[1] = max(float(aim[1]), y_min)
                    n_stuck = 0
            else:
                n_stuck = 0

            self._move_tcp(arm_tag, aim, z_hold, quat)
            self._dwell(2)
            self.plan_success = True
            # Soft brake near the target so contact inertia cannot coast past it.
            if err_n < 0.08:
                rigid = self._get_rigid(self.jar)
                if rigid is not None:
                    try:
                        v = np.asarray(rigid.get_linear_velocity(), dtype=float)
                        scale = 0.20 if err_n < 0.05 else 0.40
                        rigid.set_linear_velocity(scale * v)
                        rigid.set_linear_damping(1.4)
                    except Exception:
                        pass
            prev_pp = self._jar_live_xy().copy()

        # Leave the jar where contact physics put it — no teleport / snap.
        self._push_active = False
        rigid = self._get_rigid(self.jar)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._dwell(12)
        pp = self._jar_live_xy()
        self._freeze_jar(
            sapien.Pose(
                [float(pp[0]), float(pp[1]), float(self.table_top) + 0.001],
                list(self.jar.get_pose().q),
            )
        )
        # Success only when the spout would actually land in the mouth.
        placed = bool(self._jar_under_nozzle())
        tcp = self._tcp_pos(arm_tag)
        self._move_tcp(arm_tag, tcp[:2], float(tcp[2] + 0.12), quat)
        print(
            f"[measure_ingredient] push done placed={placed} "
            f"jar {np.round(pp0, 3)}->{np.round(pp, 3)} land={np.round(land_xy, 3)} "
            f"err={float(np.linalg.norm(pp - land_xy)):.3f} "
            f"under_nozzle={self._jar_under_nozzle()}"
        )
        self.plan_success = bool(placed)
        return placed

    # ------------------------------------------------------------------ expert
    def _touch_tip_pose(self, tip_z_above_top: float):
        """Top-down EE pose with TCP ``tip_z_above_top`` above the key top."""
        tcp_z = float(self.touch_top_z) + float(tip_z_above_top)
        ee_z = tcp_z + self.EE_TO_TCP
        return [
            float(self.touch_xy[0]),
            float(self.touch_xy[1]),
            float(ee_z),
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _switch_ee_pose(self, tip_z_above: float):
        """Compatibility alias for ``_touch_tip_pose``."""
        return self._touch_tip_pose(tip_z_above)

    def _press_switch(self, arm_tag: ArmTag, want_open: bool):
        """Press the key via real TCP force — no scripted latch without contact.

        ON: depress until ``_detect_tab_touch`` engages (force > threshold), then
        release; key stays visually DOWN while ``tab_open``.
        OFF: depress for a second press edge (``_pending_tab_off``), then release
        so the detector turns oil OFF when the key returns up.
        """
        was = self.tab_open
        if want_open and was:
            return True
        if (not want_open) and (not was):
            return True

        # Contact path only — never ``_set_tab_open`` from the expert.
        self._ignore_tab = False
        self._pending_tab_off = False
        self._touch_latched = False
        self._pressing_arm_side = str(arm_tag)

        self.move(self.close_gripper(arm_tag))
        if not self.plan_success:
            return False

        high_dis = float(self.KEY_HOVER_DIS) + 0.08
        # Absolute poses (not relative displace) so a drifted post-pour wrist
        # still lands on the key XY.
        self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(high_dis)))
        if not self.plan_success:
            self.plan_success = True
        self.move(
            self.move_to_pose(arm_tag, self._touch_tip_pose(self.KEY_HOVER_DIS))
        )
        if not self.plan_success:
            print(f"[measure_ingredient] key hover failed want_open={want_open}")
            return False

        # Depress into the force-engage band: TCP below top_z + slack.
        # tip_z_above = -0.008 → ~8 mm into the key cap.
        engage_tips = (-0.004, -0.010, -0.016)

        def _force() -> float:
            sig = self._switch_press_signal()
            return float(sig["force"]) if sig is not None else 0.0

        def _depress_until(pred, label: str) -> bool:
            for tip in engage_tips:
                self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(tip)))
                if not self.plan_success:
                    self.plan_success = True
                self._idle_steps(35, until=pred)
                if pred():
                    return True
                print(
                    f"[measure_ingredient] key {label} retry tip={tip} "
                    f"force={_force():.2f}N"
                )
            return bool(pred())

        if want_open:
            if not _depress_until(lambda: bool(self.tab_open), "ON"):
                print(
                    f"[measure_ingredient] key ON failed — no contact engage "
                    f"(force={_force():.2f}N need>{self.SWITCH_ENGAGE_FORCE})"
                )
                self.plan_success = False
                return False
            # Release; latch stays ON via tab_open (visual stays down).
            self.move(
                self.move_to_pose(
                    arm_tag, self._touch_tip_pose(float(self.KEY_HOVER_DIS) + 0.04)
                )
            )
            if not self.plan_success:
                self.plan_success = True
            self._idle_steps(8)
        else:
            if not _depress_until(
                lambda: bool(self._pending_tab_off), "OFF"
            ):
                print(
                    f"[measure_ingredient] key OFF press missed contact "
                    f"(force={_force():.2f}N, tab_open={self.tab_open})"
                )
                self.plan_success = False
                return False
            self.move(
                self.move_to_pose(
                    arm_tag, self._touch_tip_pose(float(self.KEY_HOVER_DIS) + 0.04)
                )
            )
            if not self.plan_success:
                self.plan_success = True
            self._idle_steps(40, until=lambda: not bool(self.tab_open))
            if self.tab_open:
                print(
                    "[measure_ingredient] key OFF failed — still ON after release"
                )
                self.plan_success = False
                return False

        interactive = bool(getattr(self, "_interactive_robot_mode", False)) or bool(
            getattr(self, "_interactive_universal_controls", False)
        )
        # Expert: ignore ambient touches between scripted presses.
        self._ignore_tab = not interactive
        print(
            f"[measure_ingredient] press key {was}→{self.tab_open} "
            f"(want={want_open}, latched={'DOWN' if self.tab_open else 'UP'}, "
            f"contact) liq={self.liquid_level:.2f}"
        )
        return bool(self.plan_success)

    def play_once(self):
        arm = self.arm
        self.move(self.close_gripper(arm))
        if not self.plan_success:
            print("[measure_ingredient] close_gripper failed")
            return self.info

        # 1) Push jar from in-front spawn under the nozzle (contact shove).
        if not self._push_jar_under_nozzle(arm):
            print("[measure_ingredient] jar push under nozzle failed")
            self.plan_success = False
            level_pct = int(round(self.target_fill * 100))
            self.info["info"] = {
                "{A}": "olive oil dispenser",
                "{B}": f"glass jar ({level_pct}% line)",
                "{C}": "green nozzle key",
                "{a}": str(arm),
                "{L}": f"{level_pct}%",
            }
            return self.info

        # 2) Press the green key ON → stream; oil fills only if jar is under spout.
        if not self._press_switch(arm, want_open=True):
            return self.info
        if not self.tab_open:
            print("[measure_ingredient] switch failed to open")
            self.plan_success = False
            return self.info

        self._ignore_tab = True

        if getattr(self, "force_overflow", False):
            # Demo: leave ON until spill (miss or overfill).
            self.pour_rate = max(float(self.pour_rate), 0.0015)
            self.spill_rate = max(float(self.spill_rate), 0.0025)
            spill_target = float(self._cfg.get("spill_demo_amount", 0.55))
            max_wait = int(1.05 / max(1e-6, self.pour_rate)) + int(
                spill_target / max(1e-6, self.spill_rate)
            ) + 200
            self._idle_steps(
                max_wait,
                until=lambda: self.spill_amount >= spill_target,
            )
            if self.plan_success and self.tab_open:
                self._press_switch(arm, want_open=False)
            self._ignore_tab = True
            if self.plan_success:
                self.move(self.move_by_displacement(arm, z=0.08))
            self._idle_steps(90)
            self.plan_success = False
            level_pct = int(round(self.target_fill * 100))
            self.info["info"] = {
                "{A}": "olive oil dispenser",
                "{B}": f"glass jar ({level_pct}% line)",
                "{C}": "green nozzle key",
                "{a}": str(arm),
                "{L}": f"{level_pct}%",
            }
            return self.info

        # 3) Wait near the target ring, then press OFF.
        lead = float(
            self._cfg.get(
                "pour_close_lead",
                max(0.07, min(0.18, 280.0 * float(self.pour_rate) + 0.05)),
            )
        )
        if float(self.target_fill) >= 0.999:
            lead = max(lead, 0.14)
        close_start = max(0.05, float(self.target_fill) - lead)
        max_wait = int(close_start / max(1e-6, self.pour_rate)) + 160
        self._idle_steps(
            max_wait,
            until=lambda: (
                self.liquid_level >= close_start or self.overflowed
            ),
        )
        print(
            f"[measure_ingredient] after pour liq={self.liquid_level:.2f} "
            f"target={self.target_fill:.2f} overflow={self.overflowed} "
            f"spill={self.spill_amount:.2f} under={self._jar_under_nozzle()} "
            f"tab_open={self.tab_open}"
        )

        self._ignore_tab = True
        if self.plan_success:
            if not self.tab_open:
                print("[measure_ingredient] WARNING: switch already off before off-click")
            self._press_switch(arm, want_open=False)
        self._ignore_tab = True

        if self.plan_success:
            self.move(self.move_by_displacement(arm, z=0.08))

        if self.check_success():
            self.plan_success = True

        level_pct = int(round(self.target_fill * 100))
        self.info["info"] = {
            "{A}": "olive oil dispenser",
            "{B}": f"glass jar ({level_pct}% line)",
            "{C}": "green nozzle key",
            "{a}": str(arm),
            "{L}": f"{level_pct}%",
        }
        return self.info


    def _fill_band(self):
        """Success fill window ``[lo, hi]``.

        Ordinary targets use ``target ± fill_tol``. A 100% target cannot go
        above full, so the band is one-sided ``[FILL_FULL_LO, 1.0]`` (90–100%).
        """
        tol = float(getattr(self, "fill_tol", self.FILL_TOL))
        tgt = float(self.target_fill)
        lo = tgt - tol
        hi = tgt + tol
        if tgt >= 0.999:
            hi = min(hi, float(getattr(self, "overflow_level", self.OVERFLOW_LEVEL)))
            lo = float(getattr(self, "FILL_FULL_LO", 0.90))
        return float(lo), float(hi)

    def check_success(self):
        """Success only after the nozzle key is turned OFF.

        Requires: switch latched off after a pour, jar under the nozzle, fill in
        the target band, and no spill/overflow.
        """
        # Do not score while the switch is still ON — wait for OFF.
        if bool(getattr(self, "tab_open", False)):
            return False
        if not bool(getattr(self, "closed_after_pour", False)):
            return False
        if not bool(getattr(self, "opened_once", False)):
            return False
        if self.overflowed or float(getattr(self, "spill_amount", 0.0)) > 1e-4:
            return False
        if not self._jar_under_nozzle():
            return False
        lo, hi = self._fill_band()
        lvl = float(self.liquid_level)
        if lvl + 1e-3 < lo:
            return False
        if lvl - 1e-3 > hi:
            return False
        return True

    def get_score(self) -> float:
        """Partial score from absolute fill error after the key is OFF.

        Success band → 1.0. For 25/50/75% targets, abs error bands
        ``[10,8.5)%`` / ``[8.5,7)%`` / ``[7,5)%`` → 0.25 / 0.5 / 0.75.
        For 100%: ``[13,12)`` / ``[12,11)`` / ``[11,10)`` → 0.25 / 0.5 / 0.75.
        Spill / overflow / not ready → 0.
        """
        if bool(getattr(self, "tab_open", False)):
            return 0.0
        if not bool(getattr(self, "closed_after_pour", False)):
            return 0.0
        if not bool(getattr(self, "opened_once", False)):
            return 0.0
        if self.overflowed or float(getattr(self, "spill_amount", 0.0)) > 1e-4:
            return 0.0
        if not self._jar_under_nozzle():
            return 0.0
        if self.check_success():
            return 1.0
        lvl = float(self.liquid_level)
        target = float(self.target_fill)
        err_pct = abs(lvl - target) * 100.0
        bands = (
            self.PARTIAL_ABS_BANDS_FULL
            if target >= 0.999
            else self.PARTIAL_ABS_BANDS
        )
        return float(score_descending_bands(err_pct, bands))

    def get_obs(self):
        obs = super().get_obs()
        mw = getattr(self, "microwave_xy_override", None) or getattr(
            self, "microwave_xy", None
        )
        lo, hi = self._fill_band()
        obs["measure_ingredient"] = {
            "target_fill": float(self.target_fill),
            "fill_tol": float(getattr(self, "fill_tol", self.FILL_TOL)),
            "fill_lo": float(lo),
            "fill_hi": float(hi),
            "pour_rate": float(getattr(self, "pour_rate", self.POUR_RATE)),
            "pour_rate_base": float(
                getattr(self, "pour_rate_base", getattr(self, "pour_rate", self.POUR_RATE))
            ),
            "pour_rate_jitter": float(
                getattr(self, "pour_rate_jitter", self.POUR_RATE_JITTER)
            ),
            "liquid_level": float(self.liquid_level),
            "tab_open": bool(self.tab_open),
            "overflowed": bool(self.overflowed),
            "spill_amount": float(getattr(self, "spill_amount", 0.0)),
            "opened_once": bool(self.opened_once),
            "closed_after_pour": bool(self.closed_after_pour),
            "jar_under_nozzle": bool(self._jar_under_nozzle()) if self.jar is not None else False,
            "oil_style": str(getattr(self, "oil_style", self.OIL_STYLE_DEFAULT)),
            "scene_id": int(getattr(self, "scene_id", 0)),
            "station_xy": (
                None
                if getattr(self, "jar_xy", None) is None
                else [float(self.jar_xy[0]), float(self.jar_xy[1])]
            ),
            "fill_xy": (
                None
                if getattr(self, "fill_xy", None) is None
                else [float(self.fill_xy[0]), float(self.fill_xy[1])]
            ),
            "scale_xy": (
                None
                if getattr(self, "scale_xy", None) is None
                else [float(self.scale_xy[0]), float(self.scale_xy[1])]
            ),
            "microwave_xy": (
                None if mw is None else [float(mw[0]), float(mw[1])]
            ),
            "partial_score": float(self.get_score()),
        }
        return obs
