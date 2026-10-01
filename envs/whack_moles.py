from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
import sapien
import sapien.render
import numpy as np


class whack_moles(Base_Task):
    """Whack-a-mole. A fixed grass-green box with a flat top and nine circular
    holes spans both arms' reach. Hole diameter matches the mole/rabbit footprint
    so critters fit the openings. Default: 2 moles bob out of / back into fixed
    holes at randomized pop speeds (uniform in [0.6, 1.0] × POP_SPEED, with a
    smooth cosine bob). Each arm picks up a wooden mallet from a side cradle and
    carries it in a hammer pose: handle horizontal, head axis along world Z. A
    hit requires the mole's crown to be struck by the flat lower face of the head
    while above the board; handle tips, gripper pads, and side brushes do not
    count. The head is wider than a hole (0.10 m vs ~0.070 m) so it cannot plunge
    into one. Hit moles turn bright green and stay down. Each unhit mole may
    complete at most ``num_appearances`` (default 5) rise→fall cycles; missing
    all of them fails the episode.

    Task options (independent toggles in ``task_args.whack_moles``; combinable):
      - Opt 1 — rabbit distractor: ``distractor_enabled``
        Spawn ``num_distractors`` (default 1) rabbit(s) in other holes. Touching a
        rabbit turns it red and fails the episode.
        CLI: ``--task-arg distractor_enabled=true`` or legacy ``--option 1``.
      - Opt 2 — relocating moles: ``relocating_moles``
        Every time an unhit mole finishes going down, it may reappear from a
        different free hole (not occupied by another mole or rabbit). Applies to
        all moles.
        CLI: ``--task-arg relocating_moles=true`` or legacy ``--option 2``.
      - Opt 1+2: rabbit distractor(s) + relocating moles together.

    Touching the hole board with any robot link or a held cube also fails.

    difficulty=easy  — hit one mole at a time with the arm on that side.
    difficulty=hard  — hit two non-adjacent opposite-side moles with both arms at once.

    Success = every mole has been touched from above at least once, no rabbit
    distractor has been touched (Opt 1), and no mole exhausted its appearance
    budget. Board-deck contact is logged but is not part of the success criteria.
    """

    NUM_MOLES_DEFAULT = 2
    NUM_APPEARANCES_DEFAULT = 5  # max rise→fall cycles per unhit mole
    NUM_DISTRACTORS_DEFAULT = 0
    NUM_DISTRACTORS_MAX = 4       # hard cap M on rabbit distractors
    DISTRACTOR_ENABLED_DEFAULT = False   # Opt 1
    RELOCATING_MOLES_DEFAULT = False     # Opt 2
    DIFFICULTY_DEFAULT = "easy"
    # Moles bob continuously (rise → fall). A non-zero hold freezes them at the
    # crown and reads as a multi-second pause before the mallet lands — keep 0.
    POP_STEPS_DEFAULT = 0
    PRE_POP_STEPS_DEFAULT = 0
    TOUCH_TOL_DEFAULT = 0.04
    HOLE_COUNT_DEFAULT = 9
    # Circular hole diameter ≈ mole/rabbit XY footprint (mole max extent after scale).
    HOLE_SIZE_DEFAULT = 0.070
    HOLE_BAR_THICKNESS_DEFAULT = 0.014  # min rim / edge-to-edge spacing between holes
    CUBE_HOLE_SCALE_DEFAULT = 1.20  # cube side / hole_size (>1 so it can't enter a hole)

    # XY half-extents of the hole board (sized for circular openings, still in reach).
    # Z half is computed so the board sits on the table while the play surface
    # stays raised (board_z_lift).
    BOARD_HALF_XY = [0.245, 0.16]
    BOARD_TOP_HALF_Z = 0.024          # Thin deck; total board height is half the prior default.
    BOARD_COLOR = [0.10, 0.45, 0.22]  # billiard-felt green play surface
    BOARD_BODY_COLOR = [0.18, 0.09, 0.04]  # dark brown box body / walls
    # Raise the play surface above the table; the solid base fills down to the tabletop.
    BOARD_Z_LIFT_DEFAULT = 0.06
    HIDE_DEPTH = 0.100
    # At peak bob, this much of the mole body stays below the board top (m).
    # Pose uses the real mesh bottom (origin ≠ geometric center).
    MOLE_RAISE_DROP = 0.02
    MOLE_MODEL = "221_mole"       # Smackem Mole (original open-arm mesh)
    RABBIT_MODEL = "224_rabbit"    # compact loaf pose (replaces open-hand 222_rabbit)
    # Mesh is Y-up; rotate so height aligns with world Z.
    MOLE_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    MOLE_COLOR = [0.72, 0.52, 0.35]           # light brown (former rabbit color)
    MOLE_TOUCHED_COLOR = [0.20, 0.85, 0.28]  # bright green when hit
    RABBIT_COLOR = [1.0, 1.0, 1.0]           # white distractor
    RABBIT_TOUCHED_COLOR = [0.92, 0.12, 0.10]  # red on illegal touch
    HOLE_RING_WIDTH = 0.006                   # black rim around each circular hole
    HOLE_RING_HEIGHT = 0.0025
    HOLE_RING_COLOR = [0.04, 0.04, 0.04]
    CUBE_COLOR = [0.15, 0.45, 0.95]            # blue — held mallet cubes
    MOLE_SCALE_MULT = 1.80        # 1.5x prior size
    MOLE_HEIGHT = 0.1053          # world height after mole_scale_mult
    # 224_rabbit is authored near mole world height; scale_mult ≈ 1.
    RABBIT_SCALE_MULT = 1.00
    RABBIT_HEIGHT = 0.1047        # world height after scale_mult (match moles)
    # Drop the gripped cube below the finger-pad midpoint (world -Z).
    CUBE_GRASP_DROP_Z = 0.05
    # Head is a squat drum struck on its flat lower face, wider than a hole
    # (0.10 m across vs a ~0.070 m opening) so it can never drop into one.
    MALLET_HEAD_RADIUS = 0.050
    MALLET_HEAD_HALF_X = 0.030       # Half-height along the head's own axis.
    MALLET_HANDLE_RADIUS = 0.012
    # Handle stops at the head's far face instead of poking out the other side.
    MALLET_HANDLE_HALF_Y = 0.130
    MALLET_HEAD_Y = -0.105         # Head sits at the forward end of the handle.
    MALLET_HANDLE_CENTER_Y = -0.025  # Forward end passes through the head center.
    MALLET_WOOD_COLOR = [0.48, 0.25, 0.10]
    MALLET_REST_HEIGHT = 0.115
    MALLET_REST_RAIL_HALF_Y = 0.012
    # Staged/held orientation. The handle lies flat along world Y (head pointing
    # away from the robot, toward the board) while the head cylinder's axis maps
    # to world Z. That is a hammer pose: the strike drives the head's flat lower
    # face straight down onto the mole crown, not the curved side of the head.
    MALLET_LEVEL_Q = [0.0, 0.70710678, 0.0, 0.70710678]
    # Grasp the handle at the mallet origin; the head stays |MALLET_HEAD_Y| ahead.
    MALLET_GRASP_HANDLE_Y = 0.0
    # Cradle posts straddle the grasp point without fouling the fingers/head.
    MALLET_REST_POST_Y = (-0.065, 0.035)
    # Keep the horizontal handle centered vertically in the finger aperture.
    MALLET_FINGER_EE_Z = 0.075
    MALLET_TIP_GRASP_CLEARANCE = 0.045
    MALLET_PRE_GRASP_DIS = 0.120
    _MALLET_ROBOT_IGNORE_BIT = 1 << 9
    _MALLET_ROBOT_IGNORE_ID = 0xA117
    # PhysX contact groups: bit0 (1) = world/board, bit1 (2) = moles/rabbits.
    # Held mallets must keep both so the head collides with the hole grid and
    # still reports mesh contact against bobbing critters.
    _MALLET_CONTACT_GROUP = 1 | 2
    # Peak |dz/dt| upper bound (m/s). 20% below the original 0.08 so the expert
    # (and a policy) have more time to meet the crest.
    POP_SPEED = 0.064
    # Per-mole speed is randomized in [POP_SPEED * (1 - MOLE_SPEED_SPREAD), POP_SPEED].
    MOLE_SPEED_SPREAD = 0.40      # 40% below peak → lower bound
    # Begin the press early mid-rise so the descending mallet meets the crest
    # without parking above a fully raised mole first.
    PRESS_READY_FRAC = 0.18
    # Commit the stroke only once the mole is this far up its bob. Below ~0.6 the
    # crown is still inside the hole, so the head would bottom out on the board
    # instead of the mole and the hit would never register.
    PRESS_COMMIT_FRAC = 0.85
    # How far the head is driven into the crown once committed (m).
    PRESS_OVERSHOOT = 0.012
    # Head-center vs hole XY error allowed before a stroke is attempted (m).
    STRIKE_XY_TOL = 0.030
    # Approach / wait / strike cycles allowed per mole.
    STRIKE_ATTEMPTS = 4
    # Only (re)approach while the mole is this low — approaching over a crest
    # parks the mallet and reads as a multi-second freeze before the smash.
    APPROACH_MAX_FRAC = 0.35
    # While falling, allow approach up to this height (still below crest).
    APPROACH_FALLING_FRAC = 0.55
    # Abort a lingering hold as soon as we commit to a press (legacy safety).
    CUT_HOLD_ON_PRESS = True
    # Hover clearance above a fully raised mole (m); keep small for a fast jab.
    HOVER_CLEARANCE_DEFAULT = 0.018
    # Fail if a held cube underside reaches this close to the board top.
    BOARD_CUBE_CONTACT_EPS = 0.001
    # PhysX mesh contact (optional). Kinematic cube ↔ dynamic mole contacts
    # often report a small positive separation; treat <= 8 mm as touching.
    HIT_SEPARATION_MAX = 0.008
    # Cube center must be this close in XY to the critter center (m).
    HIT_XY_TOL = 0.03
    # Contact counts only on the cube underside: critter top must sit in this
    # band around the cube bottom face (m).
    HIT_BOTTOM_BAND = 0.015
    # Geometric press: cube underside within this distance of / into the crown
    # counts as a bottom-face hit even if PhysX contact was cleared by set_pose.
    HIT_GEOM_EPS = 0.006
    # Raise the whole table (and board) by this amount (m).
    TABLE_HEIGHT_BIAS_DEFAULT = 0.05

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("whack_moles", {})
        self._reset_metric_state()
        # init kinematic bookkeeping before base init (may call _update_kinematic_tasks)
        self._global_step = 0
        self.moles = []
        self._mole_rigids = []
        self._mole_shapes = []
        self._mole_state = []
        self.rabbits = []
        self._rabbit_rigids = []
        self._rabbit_shapes = []
        self._rabbit_state = []
        self.distractor_hit = False
        self.appearances_exhausted = False
        self.board_hit = False
        self.distractor_enabled = False
        self.relocating_moles = False
        self.num_appearances = self.NUM_APPEARANCES_DEFAULT
        self._suppress_board_hit = False
        self._robot_link_names = set()
        self.holes = []
        self.hammer_cubes = {}
        self._cube_comps = {}
        self._cube_weld = {}
        # Lift table (+board) 5 cm by default.
        table_lift = float(self._cfg.get(
            "table_height_bias", self.TABLE_HEIGHT_BIAS_DEFAULT))
        kwags = dict(kwags)
        kwags["table_height_bias"] = float(kwags.get("table_height_bias", 0.0)) + table_lift
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ option parsers
    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"whack_moles expected a boolean, got {value!r}")

    def _parse_distractor_enabled(self, c) -> bool:
        """Opt 1: ``distractor_enabled`` (preferred) or legacy ``option: 1``."""
        flag = c.get("distractor_enabled", None)
        legacy = c.get("option", None)
        if legacy is not None and flag is None:
            if legacy in (1, "1", "distractor_enabled", "distractor", "rabbit"):
                flag = True
            elif legacy in (2, "2", "relocating_moles", "relocating", "relocate"):
                flag = False
            else:
                raise ValueError(
                    "whack_moles option must be 1/distractor_enabled or "
                    "2/relocating_moles (or set distractor_enabled / "
                    "relocating_moles booleans)"
                )
        return self._as_bool(flag, self.DISTRACTOR_ENABLED_DEFAULT)

    def _parse_relocating_moles(self, c) -> bool:
        """Opt 2: ``relocating_moles`` (preferred) or legacy ``option: 2``."""
        flag = c.get("relocating_moles", None)
        legacy = c.get("option", None)
        if legacy is not None and flag is None:
            if legacy in (2, "2", "relocating_moles", "relocating", "relocate"):
                flag = True
            elif legacy in (1, "1", "distractor_enabled", "distractor", "rabbit"):
                flag = False
            else:
                raise ValueError(
                    "whack_moles option must be 1/distractor_enabled or "
                    "2/relocating_moles (or set distractor_enabled / "
                    "relocating_moles booleans)"
                )
        return self._as_bool(flag, self.RELOCATING_MOLES_DEFAULT)

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "distractor_enabled", False):
            parts.append("option 1")
        if getattr(self, "relocating_moles", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    # ---------------------------------------------------------------- board
    def create_board(self):
        self.hole_size = float(self._cfg.get("hole_size", self.HOLE_SIZE_DEFAULT))
        self.hole_bar_thickness = float(
            self._cfg.get("hole_bar_thickness", self.HOLE_BAR_THICKNESS_DEFAULT))
        self.hole_count = int(self._cfg.get("hole_count", self.HOLE_COUNT_DEFAULT))
        # Play-surface height above the table (= former float gap + thin deck).
        self.board_z_lift = float(
            self._cfg.get("board_z_lift", self.BOARD_Z_LIFT_DEFAULT))
        # Thicken the base so the board sits on the table with the top raised.
        top_above_table = self.board_z_lift + 2.0 * float(self.BOARD_TOP_HALF_Z)
        board_half_z = 0.5 * top_above_table
        self.BOARD_HALF = [
            float(self._cfg.get("board_half_x", self.BOARD_HALF_XY[0])),
            float(self.BOARD_HALF_XY[1]),
            float(board_half_z),
        ]
        board_cy = float(np.random.uniform(0.0, 0.05))
        self.board_center = np.array(
            [0.0, board_cy, self.table_top + board_half_z], dtype=float)
        board_color = self._cfg.get("board_color", self.BOARD_COLOR)
        board_body_color = self._cfg.get("board_body_color", self.BOARD_BODY_COLOR)
        # Deck collision thickness matches the raised play-surface band so the
        # mallet head cannot tunnel through a paper-thin plate.
        deck_thickness = 2.0 * float(self.BOARD_TOP_HALF_Z)
        self.hole_rows = int(np.floor(np.sqrt(self.hole_count)))
        self.hole_cols = int(np.ceil(self.hole_count / self.hole_rows))
        self.board = create_box_with_circular_holes(
            self.scene,
            sapien.Pose(p=self.board_center.tolist()),
            half_size=self.BOARD_HALF,
            color=list(board_color),
            body_color=list(board_body_color),
            is_static=True,
            name="hole_board",
            hole_rows=self.hole_rows,
            hole_cols=self.hole_cols,
            hole_count=self.hole_count,
            hole_size=self.hole_size,
            wall_thickness=0.02,
            top_thickness=deck_thickness,
            bar_thickness=self.hole_bar_thickness,
            hole_ring_width=float(
                self._cfg.get("hole_ring_width", self.HOLE_RING_WIDTH)),
            hole_ring_height=float(
                self._cfg.get("hole_ring_height", self.HOLE_RING_HEIGHT)),
            hole_ring_color=list(
                self._cfg.get("hole_ring_color", self.HOLE_RING_COLOR)),
        )
        self.board_top_z = float(self.board_center[2] + self.BOARD_HALF[2])

        # Prefer centers baked into the mesh actor; fall back to the same layout math.
        hole_local = None
        if hasattr(self.board, "config") and isinstance(self.board.config, dict):
            hole_local = self.board.config.get("hole_centers_local")
        self.holes = []
        self.hole_rc = []
        if hole_local:
            cols = self.hole_cols
            for i, (cx, cy) in enumerate(hole_local):
                self.holes.append(np.array(
                    [self.board_center[0] + float(cx),
                     self.board_center[1] + float(cy)], dtype=float))
                self.hole_rc.append((i // cols, i % cols))
        else:
            x_half, y_half = self.BOARD_HALF[0], self.BOARD_HALF[1]
            gap_x = (2 * x_half - self.hole_cols * self.hole_size) / (self.hole_cols + 1)
            gap_y = (2 * y_half - self.hole_rows * self.hole_size) / (self.hole_rows + 1)
            if gap_x < self.hole_bar_thickness or gap_y < self.hole_bar_thickness:
                raise ValueError("Requested hole_size is too large for the board top")
            x_centers = np.linspace(
                -x_half + gap_x + self.hole_size / 2.0,
                x_half - gap_x - self.hole_size / 2.0,
                self.hole_cols,
            )
            y_centers = np.linspace(
                -y_half + gap_y + self.hole_size / 2.0,
                y_half - gap_y - self.hole_size / 2.0,
                self.hole_rows,
            )
            # row-major: (r,c) with r along y, c along x
            for r, dy in enumerate(y_centers):
                for c, cx in enumerate(x_centers):
                    if len(self.holes) >= self.hole_count:
                        break
                    self.holes.append(np.array(
                        [self.board_center[0] + cx, self.board_center[1] + dy],
                        dtype=float))
                    self.hole_rc.append((r, c))
        self.num_holes = len(self.holes)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        self._global_step = 0
        self.moles = []
        self._mole_rigids = []
        self._mole_shapes = []
        self._mole_state = []
        self.rabbits = []
        self._rabbit_rigids = []
        self._rabbit_shapes = []
        self._rabbit_state = []
        self.distractor_hit = False
        self.appearances_exhausted = False
        self.board_hit = False
        self._robot_link_names = set()
        self.holes = []
        self.hole_rc = []
        self.touched = []
        self.schedule = []
        self._schedule_i = 0
        self.hammer_cubes = {}
        self._cube_comps = {}
        self._cube_weld = {}
        self._cubes_ready = False
        self.staged_mallets = {}
        self.mallet_rests = {}

        self.num_moles = int(self._cfg.get("num_moles", self.NUM_MOLES_DEFAULT))
        self.num_appearances = max(
            1, int(self._cfg.get("num_appearances", self.NUM_APPEARANCES_DEFAULT)))
        self.distractor_enabled = self._parse_distractor_enabled(self._cfg)
        self.relocating_moles = self._parse_relocating_moles(self._cfg)
        if self.distractor_enabled:
            # Opt 1 on: at least one rabbit; honor explicit num_distractors if larger.
            self.num_distractors = int(
                self._cfg.get("num_distractors", 1))
            self.num_distractors = max(1, self.num_distractors)
        else:
            # Opt 1 off: no rabbits (ignore stale num_distractors unless both
            # distractor_enabled and option are unset and num_distractors>0 —
            # legacy configs that only set num_distractors).
            legacy_n = self._cfg.get("num_distractors", None)
            if (
                legacy_n is not None
                and self._cfg.get("distractor_enabled", None) is None
                and self._cfg.get("option", None) is None
                and int(legacy_n) > 0
            ):
                self.distractor_enabled = True
                self.num_distractors = int(legacy_n)
            else:
                self.num_distractors = 0
        self.num_distractors = max(
            0, min(self.num_distractors, self.NUM_DISTRACTORS_MAX))
        self.difficulty = str(self._cfg.get("difficulty", self.DIFFICULTY_DEFAULT)).lower()
        if self.difficulty not in ("easy", "hard"):
            self.difficulty = self.DIFFICULTY_DEFAULT
        self.pop_steps = int(self._cfg.get("pop_steps", self.POP_STEPS_DEFAULT))
        self.pre_pop_steps = int(self._cfg.get("pre_pop_steps", self.PRE_POP_STEPS_DEFAULT))
        self.touch_tol = float(self._cfg.get("touch_tol", self.TOUCH_TOL_DEFAULT))
        self.mole_height = float(self._cfg.get("mole_height", self.MOLE_HEIGHT))
        self.rabbit_height = float(self._cfg.get("rabbit_height", self.RABBIT_HEIGHT))
        self.cube_hole_scale = float(
            self._cfg.get("cube_hole_scale", self.CUBE_HOLE_SCALE_DEFAULT))
        self.table_top = 0.74 + self.table_z_bias

        self.create_board()
        self._robot_link_names = self._collect_robot_link_names()
        # Hide yellow wrist camera-mount meshes; gray out leftover UR yellow plastic.
        self._hide_wrist_camera_mounts()
        self._gray_out_arm_yellow_parts()
        # Mallet cube: slightly larger than a hole so it cannot fall in.
        self.cube_side = float(self.hole_size) * self.cube_hole_scale
        self.cube_half = self.cube_side * 0.5
        self.cube_grasp_drop_z = float(
            self._cfg.get("cube_grasp_drop_z", self.CUBE_GRASP_DROP_Z))
        self._create_staged_mallets()
        if self.num_moles > self.num_holes:
            raise ValueError(f"num_moles ({self.num_moles}) > hole_count ({self.num_holes})")
        max_distractors = min(
            self.NUM_DISTRACTORS_MAX, self.num_holes - self.num_moles)
        if self.num_distractors > max_distractors:
            self.num_distractors = max_distractors
            if self.num_distractors < 1:
                self.distractor_enabled = False

        hole_ids = self._assign_holes(self.num_moles)
        self.mole_holes = list(hole_ids)
        self._spawn_moles()

        rabbit_holes = self._assign_holes(
            self.num_distractors, exclude=self.mole_holes)
        self.rabbit_holes = list(rabbit_holes)
        self._spawn_rabbits()

        self.touched = [False] * self.num_moles
        self._reset_metric_state()
        self.schedule = self._build_schedule()
        self._schedule_i = 0

        self.prohibited_area.append([
            self.board_center[0] - self.BOARD_HALF[0] - 0.03,
            self.board_center[1] - self.BOARD_HALF[1] - 0.03,
            self.board_center[0] + self.BOARD_HALF[0] + 0.03,
            self.board_center[1] + self.BOARD_HALF[1] + 0.03,
        ])

    def _spawn_poppable(
        self,
        hole_idx,
        name,
        modelname,
        height,
        color,
        phase,
        actors,
        rigids,
        shapes_out,
        states,
        scale_mult=1.0,
        pop_speed=None,
        raise_drop=0.0,
        mesh_z_min=None,
        mesh_z_max=None,
    ):
        """Spawn one bobbing critter (mole or rabbit) in a hole."""
        raise_drop = float(raise_drop)
        pose_p = self._critter_pose_p(
            hole_idx, raised=False, height=height,
            raise_drop=raise_drop, mesh_z_min=mesh_z_min)
        actor = create_actor(
            self.scene,
            pose=sapien.Pose(p=pose_p.tolist(), q=self.MOLE_Q),
            modelname=modelname,
            model_id=0,
            convex=True,
            is_static=False,
            scale_mult=scale_mult,
        )
        actor.set_name(name)
        actor.set_mass(0.05)
        rigid = None
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                rigid = c
                try:
                    c.set_linear_damping(50.0)
                    c.set_angular_damping(50.0)
                except Exception:
                    pass
                c.set_kinematic(False)
                c.set_disable_gravity(True)
                try:
                    for shape in c.get_collision_shapes():
                        shape.set_collision_groups([2, 2, 0, 0])
                except Exception:
                    pass
        shapes = []
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                shapes = list(c.render_shapes)
        hidden_z = float(pose_p[2])
        raised_z = float(
            self._critter_pose_p(
                hole_idx, raised=True, height=height,
                raise_drop=raise_drop, mesh_z_min=mesh_z_min)[2])
        # Cosine bob phase: 0 = hidden, pi = crown, 2pi = hidden again.
        # Stagger by phase index so neighbors are not synced.
        bob_phase = float((int(phase) % 3) * (2.0 * np.pi / 3.0))
        travel = float(raised_z - hidden_z)
        z0 = float(hidden_z + travel * 0.5 * (1.0 - np.cos(bob_phase)))
        raised0 = bool(z0 > hidden_z + 1e-4)
        speed = float(self.POP_SPEED if pop_speed is None else pop_speed)
        actors.append(actor)
        rigids.append(rigid)
        shapes_out.append(shapes)
        state = {
            "hole": int(hole_idx),
            "raised": bool(raised0),
            "touched": False,
            "appearances": 0,  # completed rise→fall cycles (moles only; rabbits unused)
            "exhausted": False,
            # Legacy fields kept for Opt-2 / debug; motion is driven by bob_phase.
            "motion": "rising" if (0.0 < bob_phase < np.pi) else "falling",
            "target_z": float(raised_z),
            "hidden_z": hidden_z,
            "raised_z": raised_z,
            "hold_left": 0,
            "height": float(height),
            "raise_drop": raise_drop,
            "pop_speed": speed,
            "bob_phase": bob_phase,
        }
        if mesh_z_min is not None:
            state["mesh_z_min"] = float(mesh_z_min)
        if mesh_z_max is not None:
            state["mesh_z_max"] = float(mesh_z_max)
        states.append(state)
        idx = len(actors) - 1
        self._set_critter_color(shapes, color)
        self._set_critter_pose(actors, rigids, states, idx, raised=raised0, z=z0)

    def _random_mole_pop_speed(self):
        """Uniform peak |dz/dt| in [POP_SPEED*(1-spread), POP_SPEED]."""
        spread = float(self._cfg.get("mole_speed_spread", self.MOLE_SPEED_SPREAD))
        spread = float(np.clip(spread, 0.0, 0.95))
        base = float(self._cfg.get("pop_speed", self.POP_SPEED))
        lo = base * (1.0 - spread)
        hi = base
        return float(np.random.uniform(lo, hi))

    def _spawn_moles(self):
        scale_mult = float(
            self._cfg.get("mole_scale_mult", self.MOLE_SCALE_MULT))
        raise_drop = float(self._cfg.get("mole_raise_drop", self.MOLE_RAISE_DROP))
        mesh_z_min, mesh_z_max = self._mole_local_z_span(scale_mult)
        for i, hole_idx in enumerate(self.mole_holes):
            # Staggered cosine phase + per-mole randomized peak speed.
            self._spawn_poppable(
                hole_idx=hole_idx,
                name=f"mole_{i}",
                modelname=self.MOLE_MODEL,
                height=self.mole_height,
                color=self.MOLE_COLOR,
                phase=i % 3,
                actors=self.moles,
                rigids=self._mole_rigids,
                shapes_out=self._mole_shapes,
                states=self._mole_state,
                scale_mult=scale_mult,
                pop_speed=self._random_mole_pop_speed(),
                raise_drop=raise_drop,
                mesh_z_min=mesh_z_min,
                mesh_z_max=mesh_z_max,
            )

    def _spawn_rabbits(self):
        scale_mult = float(
            self._cfg.get("rabbit_scale_mult", self.RABBIT_SCALE_MULT))
        for i, hole_idx in enumerate(self.rabbit_holes):
            # offset phase so rabbits are not synced with moles
            self._spawn_poppable(
                hole_idx=hole_idx,
                name=f"rabbit_{i}",
                modelname=self.RABBIT_MODEL,
                height=self.rabbit_height,
                color=self.RABBIT_COLOR,
                phase=(i + 1) % 3,
                actors=self.rabbits,
                rigids=self._rabbit_rigids,
                shapes_out=self._rabbit_shapes,
                states=self._rabbit_state,
                scale_mult=scale_mult,
            )

    def _assign_holes(self, n, exclude=None):
        """Pick n distinct holes, preferring a left/right mix for hard mode."""
        if n <= 0:
            return []
        exclude = set(exclude or [])
        left = [i for i, h in enumerate(self.holes)
                if h[0] < -0.02 and i not in exclude]
        right = [i for i, h in enumerate(self.holes)
                 if h[0] > 0.02 and i not in exclude]
        center = [i for i, h in enumerate(self.holes)
                  if abs(h[0]) <= 0.02 and i not in exclude]
        chosen = []
        # take alternately from left/right so hard-mode pairing is feasible
        pools = [list(left), list(right)]
        for p in pools:
            np.random.shuffle(p)
        while len(chosen) < n and (pools[0] or pools[1]):
            for p in pools:
                if len(chosen) >= n:
                    break
                if p:
                    chosen.append(p.pop())
        leftover = [i for i in center + left + right if i not in chosen]
        np.random.shuffle(leftover)
        while len(chosen) < n and leftover:
            chosen.append(leftover.pop())
        return chosen[:n]

    def _holes_too_close(self, h1, h2):
        r1, c1 = self.hole_rc[h1]
        r2, c2 = self.hole_rc[h2]
        # chebyshev <= 1 => adjacent (incl. diagonal) — leave gripper clearance
        return max(abs(r1 - r2), abs(c1 - c2)) <= 1

    def _build_schedule(self):
        remaining = list(range(self.num_moles))
        np.random.shuffle(remaining)
        schedule = []
        if self.difficulty == "easy":
            for i in remaining:
                schedule.append([i])
            return schedule

        # hard: pair moles on opposite sides in non-adjacent holes
        while len(remaining) >= 2:
            pair = self._pick_hard_pair(remaining)
            if pair is None:
                break
            schedule.append(pair)
            for p in pair:
                remaining.remove(p)
        for i in remaining:
            schedule.append([i])
        return schedule

    def _pick_hard_pair(self, candidates):
        left, right = [], []
        for i in candidates:
            x = float(self.holes[self.mole_holes[i]][0])
            (left if x < 0 else right).append(i)
        np.random.shuffle(left)
        np.random.shuffle(right)
        options = []
        for a in left:
            for b in right:
                if not self._holes_too_close(self.mole_holes[a], self.mole_holes[b]):
                    options.append([a, b])
        if not options:
            # fallback: any non-adjacent pair (even same side)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    a, b = candidates[i], candidates[j]
                    if not self._holes_too_close(self.mole_holes[a], self.mole_holes[b]):
                        options.append([a, b])
        if not options:
            return None
        return list(options[int(np.random.randint(0, len(options)))])

    # ---------------------------------------------------------- mole / rabbit kinematics
    def _mole_local_z_span(self, scale_mult=None):
        """Mole mesh Z range in the actor frame after ``MOLE_Q`` (Y-up → Z-up).

        ``create_actor`` does not recenter the GLB, so the origin is not the
        geometric center — raised/hidden poses must use these offsets.
        """
        import json

        if scale_mult is None:
            scale_mult = float(
                self._cfg.get("mole_scale_mult", self.MOLE_SCALE_MULT))
        path = resolve_model_dir(self.MOLE_MODEL) / "model_data0.json"
        with open(path, "r") as f:
            data = json.load(f)
        # Authored Y-up; MOLE_Q maps model Y → world Z.
        scale_y = float(data["scale"][1]) * float(scale_mult)
        cy = float(data["center"][1]) * scale_y
        half_h = 0.5 * float(data["extents"][1]) * scale_y
        return float(cy - half_h), float(cy + half_h)

    def _critter_pose_p(
        self, hole_idx, raised, height, raise_drop=0.0, mesh_z_min=None,
    ):
        h = self.holes[hole_idx]
        if raised:
            if mesh_z_min is not None:
                # Mesh bottom sits ``raise_drop`` below the deck (partially out).
                z = self.board_top_z - float(raise_drop) - float(mesh_z_min)
            else:
                z = self.board_top_z + height * 0.5 + 1e-3 - float(raise_drop)
        else:
            z = self.board_top_z - self.HIDE_DEPTH
        return np.array([h[0], h[1], z], dtype=float)

    def _mole_pose_p(self, hole_idx, raised):
        drop = self._mole_raise_drop()
        z_min, _z_max = self._mole_local_z_span()
        return self._critter_pose_p(
            hole_idx, raised, self.mole_height,
            raise_drop=drop, mesh_z_min=z_min)

    def _mole_raise_drop(self):
        return float(self._cfg.get("mole_raise_drop", self.MOLE_RAISE_DROP))

    def _occupied_holes(self, exclude_mole_idx=None):
        """Holes currently claimed by moles (except exclude) or rabbits."""
        occ = set()
        for i, st in enumerate(getattr(self, "_mole_state", []) or []):
            if exclude_mole_idx is not None and i == exclude_mole_idx:
                continue
            occ.add(int(st["hole"]))
        for st in getattr(self, "_rabbit_state", []) or []:
            occ.add(int(st["hole"]))
        return occ

    def _relocate_mole(self, idx):
        """Opt 2: move an unhit mole to a free hole after it finishes going down."""
        if not getattr(self, "relocating_moles", False):
            return
        st = self._mole_state[idx]
        if st.get("touched") or st.get("exhausted"):
            return
        old = int(st["hole"])
        occ = self._occupied_holes(exclude_mole_idx=idx)
        free = [h for h in range(self.num_holes) if h not in occ]
        if not free:
            return
        # Prefer a different hole whenever one is available.
        alts = [h for h in free if h != old]
        pool = alts if alts else free
        new_hole = int(pool[int(np.random.randint(0, len(pool)))])
        if new_hole == old:
            return
        height = float(st.get("height", self.mole_height))
        raise_drop = float(st.get("raise_drop", 0.0))
        mesh_z_min = st.get("mesh_z_min", None)
        st["hole"] = new_hole
        if getattr(self, "mole_holes", None) is not None and idx < len(self.mole_holes):
            self.mole_holes[idx] = new_hole
        hidden = self._critter_pose_p(
            new_hole, raised=False, height=height,
            raise_drop=raise_drop, mesh_z_min=mesh_z_min)
        raised = self._critter_pose_p(
            new_hole, raised=True, height=height,
            raise_drop=raise_drop, mesh_z_min=mesh_z_min)
        st["hidden_z"] = float(hidden[2])
        st["raised_z"] = float(raised[2])
        st["target_z"] = float(raised[2])

    def _mark_mole_appearances_exhausted(self, idx):
        """Pin an unhit mole down after it used all appearance cycles; fail episode."""
        st = self._mole_state[idx]
        if st.get("touched") or st.get("exhausted"):
            return
        st["exhausted"] = True
        st["motion"] = None
        st["hold_left"] = 0
        st["raised"] = False
        st["bob_phase"] = 0.0
        st["freeze_bob"] = False
        self._set_mole_pose(idx, raised=False)
        self.appearances_exhausted = True
        limit = int(getattr(self, "num_appearances", self.NUM_APPEARANCES_DEFAULT))
        self._last_fail_reason = (
            f"mole {idx} missed after {limit}/{limit} appearances")
        self.plan_success = False

    def _on_mole_gone_down(self, idx):
        """Count a completed rise→fall; relocate (Opt 2); fail if budget spent."""
        st = self._mole_state[idx]
        if st.get("touched") or st.get("exhausted"):
            return
        st["appearances"] = int(st.get("appearances", 0)) + 1
        limit = int(getattr(self, "num_appearances", self.NUM_APPEARANCES_DEFAULT))
        if st["appearances"] >= limit:
            self._mark_mole_appearances_exhausted(idx)
            return
        self._relocate_mole(idx)

    def _episode_failed(self):
        """True if a rabbit was hit or a mole exhausted its appearance budget."""
        return bool(
            getattr(self, "distractor_hit", False)
            or getattr(self, "appearances_exhausted", False)
        )

    def _set_critter_pose(self, actors, rigids, states, idx, raised=None, z=None):
        st = states[idx]
        hole = st["hole"]
        height = float(st.get("height", self.mole_height))
        raise_drop = float(st.get("raise_drop", 0.0))
        mesh_z_min = st.get("mesh_z_min", None)
        if raised is None:
            raised = st["raised"]
        p = self._critter_pose_p(
            hole, raised, height,
            raise_drop=raise_drop, mesh_z_min=mesh_z_min)
        if z is not None:
            p[2] = float(z)
        pose = sapien.Pose(p=p.tolist(), q=self.MOLE_Q)
        actors[idx].actor.set_pose(pose)
        rigid = rigids[idx]
        if rigid is not None:
            try:
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass
        st["raised"] = bool(raised)

    def _set_mole_pose(self, idx, raised=None, z=None):
        self._set_critter_pose(
            self.moles, self._mole_rigids, self._mole_state, idx,
            raised=raised, z=z)

    def _set_rabbit_pose(self, idx, raised=None, z=None):
        self._set_critter_pose(
            self.rabbits, self._rabbit_rigids, self._rabbit_state, idx,
            raised=raised, z=z)

    @staticmethod
    def _set_critter_color(shapes, rgb):
        col = list(rgb)[:3] + [1.0]
        for s in shapes:
            mats = []
            try:
                parts = list(s.get_parts())
            except Exception:
                parts = []
            if parts:
                for part in parts:
                    try:
                        mats.append(part.get_material())
                    except Exception:
                        try:
                            mats.append(part.material)
                        except Exception:
                            pass
            else:
                try:
                    mats.append(s.material)
                except Exception:
                    pass
            for mat in mats:
                if mat is None:
                    continue
                # Drop albedo texture so solid recolors (brown moles / white rabbit) stick.
                try:
                    mat.set_base_color_texture(None)
                except Exception:
                    try:
                        mat.base_color_texture = None
                    except Exception:
                        pass
                try:
                    mat.set_base_color(col)
                    mat.base_color = col
                except Exception:
                    try:
                        mat.set_base_color(col)
                    except Exception:
                        pass

    def _set_mole_color(self, idx, rgb):
        self._set_critter_color(self._mole_shapes[idx], rgb)

    def _set_rabbit_color(self, idx, rgb):
        self._set_critter_color(self._rabbit_shapes[idx], rgb)

    # ---------------------------------------------------------- hammer cubes
    @staticmethod
    def _pose7_to_mat(pose7):
        p = np.asarray(pose7[:3], dtype=float)
        q = np.asarray(pose7[3:], dtype=float)  # [w,x,y,z]
        from transforms3d.quaternions import quat2mat
        T = np.eye(4)
        T[:3, :3] = quat2mat(q)
        T[:3, 3] = p
        return T

    @staticmethod
    def _mat_to_pose(T):
        from transforms3d.quaternions import mat2quat
        q = mat2quat(T[:3, :3])  # [w,x,y,z]
        return sapien.Pose(T[:3, 3].tolist(), q.tolist())

    def _disable_link_render(self, entity, link_name):
        """Disable all render bodies on a named robot link (collision kept)."""
        try:
            link = entity.find_link_by_name(link_name)
        except Exception:
            link = None
        if link is None:
            return
        ent = link.entity if hasattr(link, "entity") else link
        for comp in ent.get_components():
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            try:
                comp.disable()
            except Exception:
                pass
            try:
                comp.visibility = 0.0
            except Exception:
                pass

    def _hide_wrist_camera_mounts(self):
        """Disable any remaining camera-mount / camera meshes on each wrist."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in ("camera_base", "camera"):
                self._disable_link_render(entity, link_name)

    def _hide_finger_pad_visuals(self):
        """Hide gray WSG finger / guide meshes that look like a held gray cube."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in (
                    "finger_left", "finger_right",
                    "gripper_left", "gripper_right"):
                self._disable_link_render(entity, link_name)

    @staticmethod
    def _is_yellowish(rgba):
        if rgba is None or len(rgba) < 3:
            return False
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        return r > 0.55 and g > 0.35 and b < 0.45 and (r + g) > (b + 0.55)

    def _gray_out_arm_yellow_parts(self):
        """Recolor yellow UR wrist/forearm plastic so it no longer looks like a cube."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        gray = [0.45, 0.45, 0.47, 1.0]
        link_names = (
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
            "forearm_link", "upper_arm_link", "ee_link",
            "wsg_50_base_link",
        )
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in link_names:
                try:
                    link = entity.find_link_by_name(link_name)
                except Exception:
                    link = None
                if link is None:
                    continue
                ent = link.entity if hasattr(link, "entity") else link
                for comp in ent.get_components():
                    if not isinstance(comp, sapien.render.RenderBodyComponent):
                        continue
                    for shape in comp.render_shapes:
                        parts = []
                        try:
                            parts = list(shape.get_parts())
                        except Exception:
                            parts = []
                        mats = []
                        if parts:
                            for part in parts:
                                try:
                                    mats.append(part.get_material())
                                except Exception:
                                    try:
                                        mats.append(part.material)
                                    except Exception:
                                        pass
                        else:
                            try:
                                mats.append(shape.get_material())
                            except Exception:
                                try:
                                    mats.append(shape.material)
                                except Exception:
                                    pass
                        for mat in mats:
                            if mat is None:
                                continue
                            try:
                                bc = list(mat.base_color)
                            except Exception:
                                try:
                                    bc = list(mat.get_base_color())
                                except Exception:
                                    bc = None
                            if not self._is_yellowish(bc):
                                continue
                            try:
                                mat.set_base_color(gray)
                                mat.base_color = gray
                                mat.set_metallic(0.1)
                                mat.set_roughness(0.6)
                                mat.set_emission([0.0, 0.0, 0.0, 1.0])
                            except Exception:
                                try:
                                    mat.set_base_color(gray)
                                except Exception:
                                    pass

    def _paint_link_cube_color(self, entity, link_name):
        """Paint a gripper link to match the held mallet cube color."""
        rgba = list(self.CUBE_COLOR)[:3] + [1.0]
        try:
            link = entity.find_link_by_name(link_name)
        except Exception:
            link = None
        if link is None:
            return
        ent = link.entity if hasattr(link, "entity") else link
        for comp in ent.get_components():
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            try:
                comp.visibility = 1.0
                if not comp.is_enabled:
                    comp.enable()
            except Exception:
                pass
            for shape in comp.render_shapes:
                mats = []
                try:
                    parts = list(shape.get_parts())
                except Exception:
                    parts = []
                if parts:
                    for part in parts:
                        try:
                            mats.append(part.get_material())
                        except Exception:
                            try:
                                mats.append(part.material)
                            except Exception:
                                pass
                else:
                    try:
                        mats.append(shape.get_material())
                    except Exception:
                        try:
                            mats.append(shape.material)
                        except Exception:
                            pass
                for mat in mats:
                    if mat is None:
                        continue
                    try:
                        mat.set_base_color(rgba)
                        mat.base_color = rgba
                        mat.set_metallic(0.0)
                        mat.set_roughness(0.35)
                        mat.set_emission([0.0, 0.0, 0.0, 1.0])
                    except Exception:
                        try:
                            mat.set_base_color(rgba)
                        except Exception:
                            pass

    def _paint_inhand_cube(self):
        """Keep the held mallet cube painted (finger pads stay gripper-gray)."""
        for cube in getattr(self, "hammer_cubes", {}).values():
            self._paint_cube(cube)

    def _paint_cube(self, cube):
        """Force the held mallet cube to CUBE_COLOR in RGB renders."""
        rgba = list(self.CUBE_COLOR)[:3] + [1.0]
        for c in cube.actor.get_components():
            if not isinstance(c, sapien.render.RenderBodyComponent):
                continue
            try:
                c.visibility = 1.0
                if not c.is_enabled:
                    c.enable()
            except Exception:
                pass
            for s in c.render_shapes:
                try:
                    mat = s.get_material()
                except Exception:
                    mat = s.material
                try:
                    mat.set_base_color(rgba)
                    mat.base_color = rgba
                    mat.set_metallic(0.0)
                    mat.set_roughness(0.35)
                    mat.set_specular(0.1)
                    mat.set_emission([0.0, 0.0, 0.0, 1.0])
                except Exception:
                    try:
                        mat.set_base_color(rgba)
                    except Exception:
                        pass

    def _finger_midpoint_world(self, arm_tag):
        """World position in the grasp aperture between the two finger pads."""
        entity = (self.robot.left_entity if str(arm_tag) == "left"
                  else self.robot.right_entity)
        fl = entity.find_link_by_name("finger_left")
        fr = entity.find_link_by_name("finger_right")
        base = entity.find_link_by_name("wsg_50_base_link")
        if fl is None or fr is None:
            return None
        p0 = np.array(fl.entity.get_pose().p, dtype=float)
        p1 = np.array(fr.entity.get_pose().p, dtype=float)
        mid = 0.5 * (p0 + p1)
        # Push from the gripper body toward the finger tips (not sideways).
        if base is not None:
            base_p = np.array(base.entity.get_pose().p, dtype=float)
            tip_dir = mid - base_p
            n = float(np.linalg.norm(tip_dir))
            if n > 1e-6:
                mid = mid + (0.035 / n) * tip_dir
        # Seat the mallet slightly below the pads so its underside hits moles.
        drop = float(getattr(self, "cube_grasp_drop_z", self.CUBE_GRASP_DROP_Z))
        mid = mid.copy()
        mid[2] -= drop
        return mid

    def _grasp_local_T_for_arm(self, arm_tag):
        """EE-local transform that seats the cube between the finger pads."""
        ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
        ee_T = self._pose7_to_mat(ee)
        mid = self._finger_midpoint_world(arm_tag)
        local = np.eye(4)
        if mid is None:
            # fallback: EE +Z points back to the wrist, plus grasp drop
            drop = float(getattr(self, "cube_grasp_drop_z", self.CUBE_GRASP_DROP_Z))
            local[2, 3] = -0.045
            local[2, 3] -= drop
            return local
        # keep EE orientation; only translate to the finger midpoint (with drop)
        ee_R_inv = ee_T[:3, :3].T
        local[:3, 3] = ee_R_inv @ (mid - ee_T[:3, 3])
        return local

    def _gripper_pos_for_cube(self):
        """Normalized gripper pos whose finger gap matches the cube width.

        WSG scale maps pos 0 → nearly closed and pos 1 → fully open. We only
        close until the pads sit against the cube faces (not fully shut).
        """
        scale = list(getattr(self.robot, "left_gripper_scale", [0.01, -0.06]))
        s0, s1 = float(scale[0]), float(scale[1])
        # Left joint ≈ -half_gap; symmetric right mimic makes gap ≈ cube_side.
        half_gap = float(self.cube_half) + 0.001  # tiny clearance so pads kiss faces
        target_joint = -half_gap
        if abs(s1 - s0) < 1e-9:
            return 0.55
        pos = (target_joint - s0) / (s1 - s0)
        return float(np.clip(pos, 0.05, 0.95))

    def _gripper_pos_for_mallet_handle(self):
        """Pinch the cylindrical handle rather than closing to the old cube width."""

        scale = list(getattr(self.robot, "left_gripper_scale", [0.01, -0.06]))
        s0, s1 = float(scale[0]), float(scale[1])
        target_joint = -(float(self.MALLET_HANDLE_RADIUS) + 0.001)
        if abs(s1 - s0) < 1e-9:
            return 0.20
        return float(np.clip((target_joint - s0) / (s1 - s0), 0.05, 0.95))

    def _finger_pad_z(self, arm_tag):
        """Return the mean world height of an arm's two finger pads."""

        entity = (self.robot.left_entity if str(arm_tag) == "left"
                  else self.robot.right_entity)
        zs = [
            float(link.entity_pose.p[2])
            for link in entity.get_links()
            if "finger" in link.get_name()
        ]
        return float(np.mean(zs)) if zs else None

    def _set_mallet_collision_groups(self, mallet, *, ignore_robot=False):
        """Enable mallet↔board and mallet↔mole contacts; optionally ignore robot."""

        rigid = self._get_rigid_dynamic_component(mallet)
        if rigid is None:
            return
        group = int(self._MALLET_CONTACT_GROUP)
        ignore_bit = int(self._MALLET_ROBOT_IGNORE_BIT) if ignore_robot else 0
        ignore_id = (int(self._MALLET_ROBOT_IGNORE_ID) & 0xFFFF) if ignore_robot else 0
        try:
            for shape in rigid.get_collision_shapes():
                shape.set_collision_groups([group, group, ignore_bit, ignore_id])
        except Exception:
            pass

    def _disable_mallet_robot_collision(self, mallet):
        """Let the fingers surround a mallet while keeping board/mole contact."""

        self._set_mallet_collision_groups(mallet, ignore_robot=True)
        ignore_bit = int(self._MALLET_ROBOT_IGNORE_BIT)
        ignore_id = int(self._MALLET_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    for shape in link.get_collision_shapes():
                        g0, g1, g2, g3 = shape.get_collision_groups()
                        shape.set_collision_groups([
                            int(g0), int(g1), int(g2) | ignore_bit,
                            (int(g3) & ~0xFFFF) | ignore_id,
                        ])
        except Exception:
            pass

    def _create_staged_mallets(self):
        """Stage level mallets in two-post cradles for an easy top grasp."""

        y = float(self.board_center[1] - self.BOARD_HALF[1] - 0.08)
        # Rest top touches the horizontal handle underside.
        rest_h = float(self.MALLET_REST_HEIGHT)
        level_q = list(self.MALLET_LEVEL_Q)
        z = float(self.table_top + rest_h + self.MALLET_HANDLE_RADIUS)
        for arm, x in (("left", -0.38), ("right", 0.38)):
            rails = []
            # Cue-style paired posts support the horizontal handle while the
            # grasp point between them stays clear for a top-down pinch.
            for post_idx, y_offset in enumerate(self.MALLET_REST_POST_Y):
                rails.append(create_box(
                    self.scene,
                    sapien.Pose([x, y + y_offset, self.table_top + 0.5 * rest_h]),
                    half_size=[0.016, 0.010, 0.5 * rest_h],
                    color=[0.32, 0.30, 0.28], is_static=True,
                    name=f"mallet_rest_{arm}_{post_idx}",
                ))
            self.mallet_rests[arm] = rails
            self.staged_mallets[arm] = self._build_mallet(
                sapien.Pose([x, y, z], level_q),
                name=f"staged_mallet_{arm}", is_static=False,
            )
            rigid = self._get_rigid_dynamic_component(self.staged_mallets[arm])
            if rigid is not None:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)

    def _build_mallet(self, pose, name, is_static=False):
        """Build a traditional T mallet: handle along Y, cylindrical head across X."""

        builder = self.scene.create_actor_builder()
        mat = self.scene.default_physical_material
        wood = sapien.render.RenderMaterial(base_color=[*self.MALLET_WOOD_COLOR, 1.0])
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS, half_length=self.MALLET_HEAD_HALF_X, material=mat,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS, half_length=self.MALLET_HEAD_HALF_X, material=wood,
        )
        handle_pose = sapien.Pose(
            [0.0, self.MALLET_HANDLE_CENTER_Y, 0.0],
            [0.70710678, 0.0, 0.0, 0.70710678],
        )
        builder.add_cylinder_collision(
            pose=handle_pose, radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y, material=mat,
        )
        builder.add_cylinder_visual(
            pose=handle_pose, radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y, material=wood,
        )
        builder.set_initial_pose(pose)
        entity = builder.build_static(name=name) if is_static else builder.build(name=name)
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [self.MALLET_HEAD_HALF_X, self.MALLET_HANDLE_HALF_Y + self.MALLET_HEAD_Y, self.MALLET_HEAD_RADIUS],
            "scale": [self.MALLET_HANDLE_RADIUS, self.MALLET_HANDLE_HALF_Y, self.MALLET_HANDLE_RADIUS],
            "target_pose": [np.eye(4).tolist()],
            # Handle-center grasp frame: the head remains beyond the fingers.
            "contact_points_pose": [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
        }
        return Actor(entity, data, mass=0.06)

    def _spawn_and_grip_cubes(self, arms=None):
        """Seat a cube between each gripper's fingers and pinch to its sides.

        The cube lives in the grasp aperture (between fingers), not as an EE tip
        attachment. Collision stays on so the oversized face blocks hole entry.
        """
        arms = tuple(arms or ("left", "right"))
        if all(arm in self.hammer_cubes for arm in arms):
            return
        arm_tags = tuple(ArmTag(arm) for arm in arms if arm not in self.hammer_cubes)
        if not arm_tags:
            return
        prev_plan = self.plan_success
        self.move(*[self.open_gripper(arm, pos=1.0) for arm in arm_tags])
        if not self.plan_success:
            self.plan_success = prev_plan

        for arm in arm_tags:
            local_T = self._grasp_local_T_for_arm(arm)
            ee = np.array(self.get_arm_pose(arm), dtype=float)
            pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
            cube = self._build_mallet(pose, name=f"hammer_mallet_{arm}")
            self._paint_cube(cube)
            rigid = None
            for c in cube.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    rigid = c
                    try:
                        c.set_linear_damping(20.0)
                        c.set_angular_damping(20.0)
                    except Exception:
                        pass
                    # Kinematic seat in the jaw aperture (between fingers).
                    # Collide with the hole board (group bit 0) and moles (bit 1).
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
                    c.set_kinematic_target(pose)
            self._disable_mallet_robot_collision(cube)
            self.hammer_cubes[str(arm)] = cube
            self._cube_comps[str(arm)] = rigid
            self._cube_weld[str(arm)] = local_T.copy()

        # Pinch to the cube sides only — do not fully close the jaws.
        grip_pos = self._gripper_pos_for_cube()
        self._cube_grip_pos = grip_pos
        prev_plan = self.plan_success
        self.move(*[self.close_gripper(arm, pos=grip_pos) for arm in arm_tags])
        if not self.plan_success:
            self.plan_success = prev_plan

        # re-seat at the pinched-jaw finger midpoint and lock the weld
        for arm_tag in arm_tags:
            arm = str(arm_tag)
            local_T = self._grasp_local_T_for_arm(ArmTag(arm))
            ee = np.array(self.get_arm_pose(ArmTag(arm)), dtype=float)
            pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
            self.hammer_cubes[arm].actor.set_pose(pose)
            rigid = self._cube_comps.get(arm)
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass
            self._cube_weld[arm] = local_T.copy()
        self._paint_inhand_cube()
        self._hide_wrist_camera_mounts()
        self._cubes_ready = bool(self.hammer_cubes)
        # Board spans both arms' reach; post-grasp cubes hang over it and below
        # the deck. Lift clear before any lateral motion or board_hit trips.
        self._lift_mallets_clear(arm_tags)

    def pickup_mallet_to_ready(self, arm_tag):
        """Open, grasp the staged handle from above, weld, then lift to ready."""

        return self.pickup_mallets((arm_tag,))

    def _staged_grasp_pose(self, arm):
        """Top-down EE pre-grasp / grasp poses for one staged mallet handle."""

        staged = self.staged_mallets.get(str(arm))
        if staged is None:
            return None, None, None
        pose = staged.get_pose()
        handle_local = np.array([0.0, float(self.MALLET_GRASP_HANDLE_Y), 0.0])
        handle_p = (np.asarray(pose.p, dtype=np.float64)
                    + pose.to_transformation_matrix()[:3, :3] @ handle_local)
        # Match the cue pickup: the planning EE stays above the shaft while the
        # fingers descend vertically around it.
        quat = list(GRASP_DIRECTION_DIC["top_down"])
        grasp_z = float(handle_p[2] + self.MALLET_FINGER_EE_Z)
        pre_pose = [
            float(handle_p[0]), float(handle_p[1]),
            grasp_z + self.MALLET_PRE_GRASP_DIS, *quat,
        ]
        grasp_pose = [float(handle_p[0]), float(handle_p[1]), grasp_z, *quat]
        return staged, pre_pose, grasp_pose

    def _weld_mallet(self, arm_tag):
        """Bind the staged mallet rigidly to the arm's current EE frame."""

        arm = str(arm_tag)
        staged = self.staged_mallets.get(arm)
        if staged is None:
            return
        ee_T = self._pose7_to_mat(np.asarray(self.get_arm_pose(arm_tag), dtype=float))
        self.hammer_cubes[arm] = staged
        self._cube_weld[arm] = np.linalg.inv(ee_T) @ staged.get_pose().to_transformation_matrix()
        rigid = self._get_rigid_dynamic_component(staged)
        self._cube_comps[arm] = rigid
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
            except Exception:
                pass
        self._cubes_ready = True

    def try_latch_staged_mallet(self, arm_tag, max_xy: float = 0.055, max_z: float = 0.07):
        """Interactive latch: weld the staged mallet if the gripper is on the handle.

        Used by teleop (G close near the cradle). Returns True when already held
        or when the latch succeeds.
        """
        arm = str(arm_tag)
        if arm in getattr(self, "hammer_cubes", {}):
            return True
        staged = getattr(self, "staged_mallets", {}).get(arm)
        if staged is None:
            return False

        pose = staged.get_pose()
        handle_local = np.array([0.0, float(self.MALLET_GRASP_HANDLE_Y), 0.0], dtype=float)
        handle_p = (
            np.asarray(pose.p, dtype=float)
            + pose.to_transformation_matrix()[:3, :3] @ handle_local
        )
        mid = self._finger_midpoint_world(arm_tag)
        if mid is None:
            mid = np.asarray(self.get_arm_pose(arm_tag), dtype=float)[:3]

        dxy = float(np.linalg.norm(mid[:2] - handle_p[:2]))
        dz = abs(float(mid[2] - handle_p[2]))
        contacted = False
        try:
            contacted = len(self.get_gripper_actor_contact_position(staged.get_name())) > 0
        except Exception:
            contacted = False
        if not contacted and (dxy > float(max_xy) or dz > float(max_z)):
            return False

        self._disable_mallet_robot_collision(staged)
        self._weld_mallet(arm_tag)
        try:
            self._paint_inhand_cube()
        except Exception:
            pass
        try:
            self._hide_wrist_camera_mounts()
        except Exception:
            pass
        return True

    def pickup_mallets(self, arm_tags=None):
        """Pick up both staged mallets at once (one scripted move per phase).

        Doing the two arms together instead of back-to-back halves the dead time
        before the first mole is played and keeps the two arms symmetric.
        """
        arm_tags = tuple(
            ArmTag(str(a)) for a in (arm_tags or (ArmTag("left"), ArmTag("right")))
            if str(a) not in self.hammer_cubes and str(a) in self.staged_mallets
        )
        if not arm_tags:
            return True

        self.plan_success = True
        poses = {}
        for arm_tag in arm_tags:
            staged, pre_pose, grasp_pose = self._staged_grasp_pose(arm_tag)
            if staged is None:
                return False
            self._disable_mallet_robot_collision(staged)
            poses[str(arm_tag)] = (pre_pose, grasp_pose)

        def both(make):
            self._move_arms([make(arm_tag) for arm_tag in arm_tags])
            return self.plan_success

        if not both(lambda a: self.open_gripper(a, pos=1.0)):
            return False
        if not both(lambda a: self.move_to_pose(a, poses[str(a)][0])):
            return False
        if not both(lambda a: self.move_to_pose(a, poses[str(a)][1])):
            return False

        # Correct small IK/model offsets so the pads actually surround the
        # elevated handle before they close, as in play_billiard's cue pickup.
        for _ in range(3):
            errors = {}
            for arm_tag in arm_tags:
                pad_z = self._finger_pad_z(arm_tag)
                if pad_z is None:
                    continue
                target = float(poses[str(arm_tag)][1][2] - self.MALLET_FINGER_EE_Z
                               + self.MALLET_TIP_GRASP_CLEARANCE)
                err = float(target - pad_z)
                if abs(err) >= 0.008:
                    errors[str(arm_tag)] = float(np.clip(err, -0.025, 0.025))
            if not errors:
                break
            self._move_arms([
                self.move_by_displacement(arm_tag=a, z=errors[str(a)], move_axis="world")
                for a in arm_tags if str(a) in errors
            ])
            if not self.plan_success:
                return False

        grip = self._gripper_pos_for_mallet_handle()
        if not both(lambda a: self.close_gripper(a, pos=grip)):
            return False
        self._dwell(15)

        # Carry the actual staged mallets; no replacement actor or magnetic snap.
        for arm_tag in arm_tags:
            self._weld_mallet(arm_tag)
        self._lift_mallets_clear(arm_tags)
        return True

    def _move_arms(self, actions):
        """Run one action per arm, together when both arms are involved."""
        actions = [a for a in actions if a is not None]
        if not actions:
            return
        if len(actions) == 1:
            self.move(actions[0])
        else:
            self.move(actions[0], actions[1])

    def _lift_mallets_clear(self, arm_tags=None):
        """Raise held mallets to hover height (escape the post-grasp low pose).

        Target EE hover from ``_hover_ee_z`` so we don't overshoot into an
        unreachable high-Z band that then breaks XY planning.
        """
        self._suppress_board_hit = True
        prev_plan = self.plan_success
        try:
            actions = []
            for arm in (arm_tags or (ArmTag("left"), ArmTag("right"))):
                hover = float(self._hover_ee_z(arm))
                cur = np.array(self.get_arm_pose(arm), dtype=float)
                dz = float(hover - cur[2])
                if abs(dz) < 0.01:
                    continue
                actions.append(self.move_by_displacement(
                    arm_tag=arm, z=dz, move_axis="world"))
            self._move_arms(actions)
        finally:
            self._suppress_board_hit = False
            if not self.plan_success:
                self.plan_success = prev_plan

    def _board_contact_z(self):
        """World Z of the play-surface top (mallet head may rest here, not below)."""
        return float(getattr(self, "board_top_z", 0.0)) + 5e-4

    def _mallet_head_geometry_from_pose(self, pose):
        """Head center / striking-face bottom for an arbitrary mallet pose."""
        mallet_T = np.asarray(pose.to_transformation_matrix(), dtype=float)
        center = (mallet_T @ np.array([0.0, self.MALLET_HEAD_Y, 0.0, 1.0]))[:3]
        axis = np.asarray(mallet_T[:3, 0], dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        r = float(self.MALLET_HEAD_RADIUS)
        half_len = float(self.MALLET_HEAD_HALF_X)
        az = float(axis[2])
        end = center - np.sign(az) * axis * half_len if abs(az) > 1e-6 else center
        bottom = end.copy()
        if abs(az) > 1e-6:
            perp = np.array([0.0, 0.0, -1.0]) + axis * az
            n = float(np.linalg.norm(perp))
            if n > 1e-9:
                bottom = end + (perp / n) * r
            else:
                bottom[2] -= r
        else:
            bottom[2] -= r
        return np.asarray(center, dtype=float), np.asarray(bottom, dtype=float)

    def _mallet_head_over_board(self, pose):
        """True if the head's XY footprint overlaps the hole-board deck."""
        if not hasattr(self, "board_center"):
            return False
        center, _bottom = self._mallet_head_geometry_from_pose(pose)
        # Conservative AABB using the head disc radius (axis near vertical).
        r = float(self.MALLET_HEAD_RADIUS)
        bx, by = float(self.board_center[0]), float(self.board_center[1])
        hx, hy = float(self.BOARD_HALF[0]), float(self.BOARD_HALF[1])
        return (
            (center[0] + r) >= (bx - hx)
            and (center[0] - r) <= (bx + hx)
            and (center[1] + r) >= (by - hy)
            and (center[1] - r) <= (by + hy)
        )

    def _clamp_mallet_pose_above_board(self, pose):
        """Raise a kinematic mallet so its head cannot tunnel into the deck.

        Welded mallets are pose-driven, so PhysX will not stop them from sinking
        through the static hole lattice — keep the striking face on/above it.
        The head is wider than a hole, so resting on the deck is always valid.
        """
        if not self._mallet_head_over_board(pose):
            return pose
        floor = self._board_contact_z()
        _center, bottom = self._mallet_head_geometry_from_pose(pose)
        min_z = float(bottom[2])
        if min_z >= floor:
            return pose
        p = np.asarray(pose.p, dtype=np.float64).copy()
        p[2] += floor - min_z
        return sapien.Pose(p.tolist(), list(pose.q))

    def mallet_resting_on_board(self, arm=None, tol=0.003):
        """True when a held mallet head is already on the play surface."""
        arms = (
            (str(arm),)
            if arm is not None
            else tuple(getattr(self, "hammer_cubes", {}) or {})
        )
        floor = self._board_contact_z()
        for a in arms:
            mallet = getattr(self, "hammer_cubes", {}).get(str(a))
            if mallet is None:
                continue
            try:
                pose = mallet.get_pose()
            except Exception:
                continue
            if not self._mallet_head_over_board(pose):
                continue
            _c, bottom = self._mallet_head_geometry_from_pose(pose)
            if float(bottom[2]) <= floor + float(tol):
                return True
        return False

    def interactive_ee_z_floor(self, side, ee_pose):
        """Lowest allowed EE Z so a welded mallet cannot enter the hole board.

        Used by ``UniversalRobotControls`` so Q/-Z teleop stops on the deck.
        """
        arm = str(side)
        local_T = getattr(self, "_cube_weld", {}).get(arm)
        if local_T is None or arm not in getattr(self, "hammer_cubes", {}):
            return None
        ee = np.asarray(ee_pose, dtype=np.float64).reshape(-1)
        if ee.size < 7:
            return None
        mallet_pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
        if not self._mallet_head_over_board(mallet_pose):
            return None
        floor = self._board_contact_z()
        _c, bottom = self._mallet_head_geometry_from_pose(mallet_pose)
        min_z = float(bottom[2])
        if min_z < floor - 1e-6:
            return float(ee[2]) + (floor - min_z)
        return None

    def _update_hammer_cubes(self):
        """Keep each cube seated in its gripper's jaw aperture (between fingers)."""
        if not getattr(self, "_cubes_ready", False):
            return
        for arm, local_T in self._cube_weld.items():
            cube = self.hammer_cubes.get(arm)
            if cube is None:
                continue
            ee = np.array(self.get_arm_pose(ArmTag(arm)), dtype=float)
            pose = self._clamp_mallet_pose_above_board(
                self._mat_to_pose(self._pose7_to_mat(ee) @ local_T))
            cube.actor.set_pose(pose)
            rigid = self._cube_comps.get(arm)
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass
            # Legacy auto-spawned tools are blue; staged wooden mallets retain
            # their wood material after pickup.
            if self._global_step % 30 == 0:
                if not cube.get_name().startswith("staged_mallet_"):
                    self._paint_cube(cube)
                self._hide_wrist_camera_mounts()

    def _advance_pop_cycle(self, actors, rigids, states, set_pose, on_gone_down=None):
        """Advance one bobbing group (moles or rabbits) for a single sim step.

        Uses a cosine bob so velocity eases to zero at the crown/bottom — no
        hard reverse and no hold plateau. ``on_gone_down(idx)`` fires each time
        a cycle wraps past the bottom (Opt 2 relocate).
        """
        dt = float(self.scene.get_timestep())
        two_pi = 2.0 * np.pi
        for idx, st in enumerate(states):
            rigid = rigids[idx]
            if rigid is None:
                continue

            # Hit / exhausted critters stay pinned down (pose only if they drifted).
            if st["touched"] or st.get("exhausted"):
                st["motion"] = None
                st["raised"] = False
                st["bob_phase"] = 0.0
                cur_z = float(rigid.entity.get_pose().p[2])
                if abs(cur_z - st["hidden_z"]) > 1e-4:
                    set_pose(idx, raised=False)
                continue

            # Hold height steady while a press is in flight so the mole cannot
            # crest (and look frozen) under a still-descending mallet.
            if st.get("freeze_bob"):
                cur_z = float(rigid.entity.get_pose().p[2])
                st["raised"] = bool(cur_z > float(st["hidden_z"]) + 1e-4)
                set_pose(idx, raised=st["raised"], z=cur_z)
                continue

            hidden = float(st["hidden_z"])
            raised = float(st["raised_z"])
            travel = max(raised - hidden, 1e-6)
            speed = float(st.get("pop_speed", self.POP_SPEED))
            # Peak |dz/dt| = travel/2 * omega = speed  →  omega = 2*speed/travel
            omega = 2.0 * speed / travel

            prev_phase = float(st.get("bob_phase", 0.0)) % two_pi
            phase = (prev_phase + omega * dt) % two_pi
            # Wrap past the bottom → finished a fall; Opt 2 may relocate.
            if on_gone_down is not None and prev_phase > phase + 1e-9:
                on_gone_down(idx)
                if st.get("touched") or st.get("exhausted"):
                    continue
                hidden = float(st["hidden_z"])
                raised = float(st["raised_z"])
                travel = max(raised - hidden, 1e-6)
                omega = 2.0 * speed / travel

            z = hidden + travel * 0.5 * (1.0 - np.cos(phase))
            rising = 0.0 < phase < np.pi
            st["bob_phase"] = float(phase)
            st["motion"] = "rising" if rising else "falling"
            st["raised"] = bool(z > hidden + 1e-4)
            st["target_z"] = raised if rising else hidden
            st["hold_left"] = 0
            set_pose(idx, raised=st["raised"], z=float(z))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._global_step = getattr(self, "_global_step", 0) + 1
        # Seat cubes at the current EE pose first so geometric bottom-face
        # checks see the true underside height this step.
        self._update_hammer_cubes()
        if not getattr(self, "_mole_state", None) and not getattr(self, "_rabbit_state", None):
            return

        # Hit/fail checks use geometry (primary) + optional PhysX contact.
        # Poll BEFORE teleporting moles: set_pose on a dynamic body clears the
        # PhysX contact manifold from the previous scene.step().
        self._poll_mole_hits()
        self._poll_rabbit_hits()
        self._poll_board_hits()

        if getattr(self, "_mole_state", None):
            self._advance_pop_cycle(
                self.moles, self._mole_rigids, self._mole_state,
                self._set_mole_pose, on_gone_down=self._on_mole_gone_down)
        if getattr(self, "_rabbit_state", None):
            self._advance_pop_cycle(
                self.rabbits, self._rabbit_rigids, self._rabbit_state,
                self._set_rabbit_pose)

        self._track_pop_metrics()

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_pop_start = {}      # mole idx -> step this appearance broke the deck
        self._metric_hit_latency = {}    # mole idx -> steps from that pop to the whack
        self._metric_hit_height = {}     # mole idx -> raise fraction at the whack

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_pop_metrics(self):
        """Stamp the start of each *current* appearance.

        Reaction time is measured against the appearance the operator is actually
        looking at, so the stamp is re-armed every time a mole ducks back under the
        deck and pops again.
        """
        try:
            for i, st in enumerate(getattr(self, "_mole_state", []) or []):
                if st.get("touched") or st.get("exhausted"):
                    continue
                up = bool(self._mole_above_surface(i))
                if up:
                    if self._metric_pop_start.get(i) is None:
                        self._metric_pop_start[i] = self._metric_step()
                else:
                    self._metric_pop_start[i] = None
        except Exception:
            pass

    def _latch_hit_metric(self, idx):
        """Called from _mark_touched, before the mole is snapped back down."""
        try:
            start = self._metric_pop_start.get(idx)
            if start is not None and idx not in self._metric_hit_latency:
                self._metric_hit_latency[idx] = max(self._metric_step() - int(start), 0)
            st = self._mole_state[idx]
            rigid = self._mole_rigids[idx]
            if rigid is not None and idx not in self._metric_hit_height:
                z = float(rigid.entity.get_pose().p[2])
                lo = float(st["hidden_z"])
                hi = float(st["raised_z"])
                span = max(hi - lo, 1e-6)
                self._metric_hit_height[idx] = float(
                    min(max((z - lo) / span, 0.0), 1.0))
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `hit_latency_steps` — mean steps from a mole breaking the deck surface
        until it is whacked, measured against the appearance that was actually hit.
        extra2 `hit_height_norm` — mean raise fraction (0 = flush with the deck,
        1 = fully crested) at the moment of the hit. HIGHER is better: striking near
        the crown means the operator caught the mole at full extension rather than
        clipping it on the way up or down.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        lats = sorted(self._metric_hit_latency.values()) if getattr(
            self, "_metric_hit_latency", None) else []
        mean_lat = (sum(lats) / len(lats)) if lats else None
        out["hit_latency_steps"] = None if mean_lat is None else round(float(mean_lat), 3)
        out["hit_latency_s"] = None if mean_lat is None else round(float(mean_lat) * dt, 4)
        out["worst_hit_latency_steps"] = lats[-1] if lats else None
        out["hits_counted"] = len(lats)

        hs = list(self._metric_hit_height.values()) if getattr(
            self, "_metric_hit_height", None) else []
        out["hit_height_norm"] = (
            round(sum(hs) / len(hs), 4) if hs else None)
        out["worst_hit_height_norm"] = round(min(hs), 4) if hs else None
        try:
            out["moles_touched"] = int(sum(1 for t in self.touched if t))
            out["moles_total"] = int(self.num_moles)
        except Exception:
            out["moles_touched"] = None
            out["moles_total"] = None
        out["distractor_hit"] = bool(getattr(self, "distractor_hit", False))
        out["board_hit"] = bool(getattr(self, "board_hit", False))
        return out

    def _dwell(self, steps):
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()

    def _sync_rise(self, idxs):
        """Align listed unhit moles onto the same rising edge (for dual-arm presses)."""
        for idx in idxs:
            st = self._mole_state[idx]
            if st["touched"] or st.get("exhausted"):
                continue
            # Start just after the bottom so the next crest is shared.
            st["bob_phase"] = 0.05
            st["motion"] = "rising"
            st["target_z"] = st["raised_z"]
            st["hold_left"] = 0
            st["raised"] = False
            self._set_mole_pose(idx, raised=False)

    def _mole_rise_frac(self, idx):
        """How far an unhit mole has risen along [hidden_z, raised_z] (0..1)."""
        st = self._mole_state[idx]
        if st.get("touched"):
            return 1.0
        if "bob_phase" in st:
            # Cosine bob: frac = 0.5*(1-cos(phase)); clamp to the rising half.
            phase = float(st["bob_phase"]) % (2.0 * np.pi)
            return float(np.clip(0.5 * (1.0 - np.cos(phase)), 0.0, 1.0))
        rigid = self._mole_rigids[idx]
        if rigid is None:
            return 1.0 if st.get("raised") else 0.0
        cur_z = float(rigid.entity.get_pose().p[2])
        hidden = float(st["hidden_z"])
        raised = float(st["raised_z"])
        travel = raised - hidden
        if travel <= 1e-6:
            return 1.0 if st.get("raised") else 0.0
        return float(np.clip((cur_z - hidden) / travel, 0.0, 1.0))

    def _mole_is_rising(self, idx):
        st = self._mole_state[idx]
        if st.get("touched") or st.get("exhausted"):
            return False
        if "bob_phase" in st:
            phase = float(st["bob_phase"]) % (2.0 * np.pi)
            return 0.0 < phase < np.pi
        return st.get("motion") == "rising"

    def _mole_ready_to_press(self, idx, frac=None):
        """True once the mole is high enough mid-rise (cosine bob, no hold)."""
        st = self._mole_state[idx]
        if st.get("touched"):
            return True
        if not self._mole_is_rising(idx):
            return False
        if frac is None:
            frac = float(self._cfg.get("press_ready_frac", self.PRESS_READY_FRAC))
        return self._mole_rise_frac(idx) >= float(frac)

    def _cut_hold_for_press(self, idxs):
        """Legacy no-op under cosine bob; clears any residual hold if present."""
        if not self._as_bool(
                self._cfg.get("cut_hold_on_press", None), self.CUT_HOLD_ON_PRESS):
            return
        for idx in idxs:
            st = self._mole_state[idx]
            if st.get("touched"):
                continue
            if st.get("motion") == "hold":
                # Nudge into the falling half of the cosine cycle.
                st["bob_phase"] = float(np.pi + 0.05)
                st["motion"] = "falling"
                st["target_z"] = st["hidden_z"]
                st["hold_left"] = 0
                st["raised"] = True

    def _wait_until_raised(self, idxs, max_steps=900, sync=False):
        """Wait until every listed (unhit) mole is high enough to press mid-rise.

        Returns as soon as moles are past ``press_ready_frac`` while still rising —
        do not wait for a top hold (that freezes them under the mallet).
        With Opt 2 (relocating_moles), moles may change holes while down — keep
        re-approaching so the mallet stays over the current hole before press.
        """
        if sync and len(idxs) > 1:
            self._sync_rise(idxs)
        last_holes = {i: int(self.mole_holes[i]) for i in idxs}
        for step in range(int(max_steps)):
            if getattr(self, "relocating_moles", False) and self.plan_success:
                for i in idxs:
                    if self.touched[i]:
                        continue
                    cur_h = int(self.mole_holes[i])
                    arm = self._arm_for_hole(cur_h)
                    hole_changed = cur_h != last_holes.get(i)
                    misaligned = (
                        getattr(self, "_cubes_ready", False)
                        and self._cube_xy_err(i, arm) > 0.05
                    )
                    # Only re-hover while the mole is still down — never after it
                    # has started rising (that would freeze the crest under the arm).
                    still_down = self._mole_rise_frac(i) < 0.2
                    if hole_changed or (misaligned and step % 40 == 0 and still_down):
                        self._approach_hole(i, arm)
                        last_holes[i] = cur_h
                        if not self.plan_success:
                            return False
            ready = all(
                self.touched[i] or self._mole_ready_to_press(i)
                for i in idxs
            )
            if ready:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()
        return False

    def _wait_until_down(self, idxs, max_steps=400):
        for _ in range(int(max_steps)):
            down = all(
                (not self._mole_state[i]["raised"])
                and self._mole_state[i].get("motion") is None
                for i in idxs
            )
            if down:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()
        return False

    # ------------------------------------------------------------- touch
    def _arm_for_hole(self, hole_idx):
        return ArmTag("right" if self.holes[hole_idx][0] > 0 else "left")

    def _mark_touched(self, idx):
        """Instant hit: recolor green and snap the mole fully down."""
        if self.touched[idx]:
            return
        self.touched[idx] = True
        self._latch_hit_metric(idx)
        st = self._mole_state[idx]
        st["touched"] = True
        st["motion"] = None
        st["hold_left"] = 0
        st["raised"] = False
        self._set_mole_color(idx, self.MOLE_TOUCHED_COLOR)
        self._set_mole_pose(idx, raised=False)

    def _critter_above_surface(self, actors, states, idx):
        """True iff the critter is not completely below the board top."""
        return self._critter_top_z(actors, states, idx) > self.board_top_z + 1e-4

    def _mole_above_surface(self, idx):
        return self._critter_above_surface(self.moles, self._mole_state, idx)

    def _critter_top_z(self, actors, states, idx):
        p = np.array(actors[idx].get_pose().p, dtype=float)
        st = states[idx]
        mesh_z_max = st.get("mesh_z_max", None)
        if mesh_z_max is not None:
            return float(p[2] + float(mesh_z_max))
        height = float(st.get("height", self.mole_height))
        return float(p[2] + height * 0.5)

    def _critter_mid_z(self, actors, states, idx):
        """Geometric mid-height of the mesh (not always the actor origin)."""
        p = np.array(actors[idx].get_pose().p, dtype=float)
        st = states[idx]
        z_min = st.get("mesh_z_min", None)
        z_max = st.get("mesh_z_max", None)
        if z_min is not None and z_max is not None:
            return float(p[2] + 0.5 * (float(z_min) + float(z_max)))
        return float(p[2])

    def _mallet_head_frame(self, mallet):
        """World head center and unit head-cylinder axis (authored along local X)."""

        mallet_T = mallet.get_pose().to_transformation_matrix()
        center = (mallet_T @ np.array([0.0, self.MALLET_HEAD_Y, 0.0, 1.0]))[:3]
        axis = np.asarray(mallet_T[:3, 0], dtype=float)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        return np.asarray(center, dtype=float), axis

    def _mallet_head_geometry(self, mallet):
        """Head center plus the lowest point of its striking surface.

        The mallet is carried with the head axis vertical, so the striking
        surface is the flat lower end face. The general form below keeps the
        value sane when the wrist is off by a few degrees.
        """

        center, axis = self._mallet_head_frame(mallet)
        r = float(self.MALLET_HEAD_RADIUS)
        half_len = float(self.MALLET_HEAD_HALF_X)
        az = float(axis[2])
        horiz = float(np.sqrt(max(1.0 - az * az, 0.0)))
        # Slide to the lower axial end, then drop by the radius perpendicular
        # to the axis (which is straight down when the head is level).
        end = center - np.sign(az) * axis * half_len if abs(az) > 1e-6 else center
        bottom = end.copy()
        bottom[2] -= r * horiz
        if abs(az) > 1e-6:
            # Perpendicular-to-axis descent also shifts XY once the head tilts.
            perp = np.array([0.0, 0.0, -1.0]) + axis * az
            n = float(np.linalg.norm(perp))
            if n > 1e-9:
                bottom = end + (perp / n) * r
        return center, bottom

    def _head_strike_z(self, mallet, xy):
        """World Z of the head's under-surface directly above ``xy``.

        Returns ``None`` when ``xy`` lies outside the head's footprint, so a
        mole that is merely beside the mallet never counts as struck.
        """

        center, axis = self._mallet_head_frame(mallet)
        r = float(self.MALLET_HEAD_RADIUS)
        half_len = float(self.MALLET_HEAD_HALF_X)
        xy = np.asarray(xy, dtype=float)[:2]
        az = float(axis[2])
        if abs(az) >= 0.5:
            # Carried pose: the head axis is (near) vertical, so the striking
            # surface is the flat lower end face. Its footprint is the face disc
            # and its height is the face plane sampled above ``xy``.
            end = center - np.sign(az) * axis * half_len
            d = xy - end[:2]
            if float(np.linalg.norm(d)) > r:
                return None
            return float(end[2] - float(np.dot(axis[:2], d)) / az)
        # Head tipped over: strike surface is the curved side of the cylinder.
        d = xy - center[:2]
        axis_xy = axis[:2]
        n = float(np.linalg.norm(axis_xy))
        u = axis_xy / n
        t = float(np.dot(d, u)) / n
        if abs(t) > half_len:
            return None
        perp = float(np.linalg.norm(d - t * axis_xy))
        if perp > r:
            return None
        return float(center[2] + t * az
                     - n * np.sqrt(max(r * r - perp * perp, 0.0)))

    def _head_half_extent_xy(self, mallet):
        """Axis-aligned XY half-extents of the head (for board-overlap tests)."""

        _center, axis = self._mallet_head_frame(mallet)
        r = float(self.MALLET_HEAD_RADIUS)
        half_len = float(self.MALLET_HEAD_HALF_X)
        return [
            abs(float(axis[i])) * half_len + r * float(np.sqrt(max(1.0 - axis[i] ** 2, 0.0)))
            for i in (0, 1)
        ]

    def _cube_bottom_contact_with_critter(self, actors, states, idx):
        """True iff the critter's crown is pressed by the mallet head underside.

        Requires all of:
          1) the critter centre lies inside the head's XY footprint,
          2) the head under-surface directly above the critter has come down
             onto / into the crown (``<= top_z + HIT_GEOM_EPS``),
          3) it has not sunk past the critter midline — that would be the head
             passing beside the body rather than striking its top.

        This is purely geometric on purpose: the head is kinematically welded to
        the gripper and ``set_pose`` clears PhysX manifolds, so contact reports
        are unreliable. PhysX contact is still accepted as a secondary signal.
        """
        if not getattr(self, "_cubes_ready", False):
            return False
        cp = np.array(actors[idx].get_pose().p, dtype=float)
        center_z = self._critter_mid_z(actors, states, idx)
        top_z = self._critter_top_z(actors, states, idx)
        sep_max = float(self._cfg.get("hit_separation_max", self.HIT_SEPARATION_MAX))
        geom_eps = float(self._cfg.get("hit_geom_eps", self.HIT_GEOM_EPS))
        actor_name = actors[idx].get_name()
        for arm, mallet in getattr(self, "hammer_cubes", {}).items():
            strike_z = self._head_strike_z(mallet, cp[:2])
            if strike_z is None:
                continue
            if strike_z < center_z:
                continue
            if strike_z <= top_z + geom_eps:
                return True
            mallet_name = mallet.get_name()
            for contact in self.scene.get_contacts():
                name0 = contact.bodies[0].entity.name
                name1 = contact.bodies[1].entity.name
                if not (
                    (name0 == actor_name and name1 == mallet_name)
                    or (name1 == actor_name and name0 == mallet_name)
                ):
                    continue
                for pt in (getattr(contact, "points", None) or []):
                    sep = getattr(pt, "separation", None)
                    if sep is not None:
                        if float(sep) <= sep_max:
                            return True
                        continue
                    imp = getattr(pt, "impulse", None)
                    if imp is not None and float(np.linalg.norm(imp)) > 1e-8:
                        return True
        return False

    def _cube_critter_hit(self, actors, states, idx):
        """Hit = the mallet head underside pressing down on the critter crown."""
        return self._cube_bottom_contact_with_critter(actors, states, idx)

    def _poll_mole_hits(self):
        """Every physics step: cube-bottom contact above surface -> green + snap down."""
        if getattr(self, "_suppress_mole_hit", False):
            return
        for idx, st in enumerate(self._mole_state):
            if st["touched"]:
                continue
            if not self._mole_above_surface(idx):
                continue
            if self._cube_critter_hit(self.moles, self._mole_state, idx):
                self._mark_touched(idx)

    def _mark_rabbit_touched(self, idx):
        """Illegal hit: recolor rabbit red, snap down, fail the episode."""
        st = self._rabbit_state[idx]
        if st["touched"]:
            return
        st["touched"] = True
        st["motion"] = None
        st["hold_left"] = 0
        st["raised"] = False
        self._set_rabbit_color(idx, self.RABBIT_TOUCHED_COLOR)
        self._set_rabbit_pose(idx, raised=False)
        self.distractor_hit = True
        # stop the expert — touching a distractor is an immediate failure
        self.plan_success = False

    def _poll_rabbit_hits(self):
        """Cube-bottom–rabbit contact above surface -> red + episode fail."""
        if not getattr(self, "_rabbit_state", None):
            return
        for idx, st in enumerate(self._rabbit_state):
            if st["touched"]:
                continue
            if not self._critter_above_surface(self.rabbits, self._rabbit_state, idx):
                continue
            if self._cube_critter_hit(self.rabbits, self._rabbit_state, idx):
                self._mark_rabbit_touched(idx)

    def _collect_robot_link_names(self):
        """Arm/wrist links that must not touch the board.

        Gripper finger pads are excluded (their collision hulls are oversized and
        falsely report board contact while the held cube is still clear). The
        cube itself is checked separately via `_cube_board_contact`.
        """
        names = set()
        robot = getattr(self, "robot", None)
        if robot is None:
            return names
        ignore = set(getattr(robot, "gripper_name", []) or [])
        ignore.update({"finger_left", "finger_right"})
        for articulation in (robot.left_entity, robot.right_entity):
            if articulation is None:
                continue
            for link in articulation.get_links():
                name = link.get_name()
                if name in ignore:
                    continue
                lname = name.lower()
                if "finger" in lname:
                    continue
                names.add(name)
        return names

    def _mark_board_hit(self):
        """Log cube–board contact. Does not abort the expert (success ignores it)."""
        if getattr(self, "board_hit", False):
            return
        self.board_hit = True

    def _cube_over_board(self, arm):
        """True if the held cube XY footprint overlaps the board top."""
        cube = getattr(self, "hammer_cubes", {}).get(str(arm))
        if cube is None or not hasattr(self, "board_center"):
            return False
        p, _axis = self._mallet_head_frame(cube)
        half_x, half_y = self._head_half_extent_xy(cube)
        bx, by = float(self.board_center[0]), float(self.board_center[1])
        hx, hy = float(self.BOARD_HALF[0]), float(self.BOARD_HALF[1])
        return (
            (p[0] + half_x) >= (bx - hx)
            and (p[0] - half_x) <= (bx + hx)
            and (p[1] + half_y) >= (by - hy)
            and (p[1] - half_y) <= (by + hy)
        )

    def _cube_board_contact(self):
        """Cube–board touch via underside depth over the board footprint.

        Held mallets also PhysX-collide with the hole lattice (contact group
        bit 0), but kinematic welding can clear manifolds — keep this geometric
        check as the authoritative board-touch signal.
        """
        if not getattr(self, "_cubes_ready", False):
            return False
        eps = float(self._cfg.get(
            "board_cube_contact_eps", self.BOARD_CUBE_CONTACT_EPS))
        for arm in getattr(self, "hammer_cubes", {}):
            if not self._cube_over_board(arm):
                continue
            if self._cube_bottom_z(ArmTag(arm)) <= self.board_top_z + eps:
                return True
        return False

    def _robot_board_contact(self):
        """True iff a non-finger robot link has PhysX contact with the board."""
        board_name = "hole_board"
        if hasattr(self, "board") and self.board is not None:
            try:
                board_name = self.board.get_name()
            except Exception:
                pass
        link_names = getattr(self, "_robot_link_names", None) or set()
        if not link_names:
            return False
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if (
                (name0 == board_name and name1 in link_names)
                or (name1 == board_name and name0 in link_names)
            ):
                return True
        return False

    def _poll_board_hits(self):
        """Held-cube contact with the hole board -> episode fail.

        Robot-link PhysX contacts against the board are not used: wrist/forearm
        collision hulls are oversized and false-trigger during a valid press
        (same reason finger pads were already excluded). The mallet cube
        underside is the authoritative board-touch signal.
        """
        if getattr(self, "board_hit", False):
            return
        if getattr(self, "_suppress_board_hit", False):
            return
        if self._cube_board_contact():
            self._mark_board_hit()

    def _cube_bottom_z(self, arm_tag):
        """World Z of the mallet head's downward striking face."""
        arm = str(arm_tag)
        if arm in getattr(self, "hammer_cubes", {}):
            _head_center, bottom = self._mallet_head_geometry(
                self.hammer_cubes[arm])
            return float(bottom[2])
        local_T = self._cube_weld.get(arm)
        if local_T is None:
            local_T = self._grasp_local_T_for_arm(arm_tag)
        ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
        cube_T = self._pose7_to_mat(ee) @ local_T
        return float(cube_T[2, 3] - self.cube_half)

    def _mallet_head_center(self, arm_tag):
        """World center of the striking head, not the handle-grasp origin."""

        mallet = self.hammer_cubes.get(str(arm_tag))
        if mallet is None:
            return None
        head_center, _bottom = self._mallet_head_geometry(mallet)
        return head_center

    def _ee_z_for_cube_bottom(self, arm_tag, cube_bottom_z):
        """EE world Z that places the held cube's underside at cube_bottom_z."""
        arm = str(arm_tag)
        if arm in getattr(self, "hammer_cubes", {}):
            ee_z = float(np.asarray(self.get_arm_pose(arm_tag), dtype=float)[2])
            current_bottom = self._cube_bottom_z(arm_tag)
            return float(ee_z + cube_bottom_z - current_bottom)
        local_T = self._cube_weld.get(arm)
        if local_T is None:
            local_T = self._grasp_local_T_for_arm(arm_tag)
        # Head underside is below the handle center by its local head thickness.
        local_z = float(local_T[2, 3])
        return float(cube_bottom_z + self.MALLET_HEAD_HALF_X - local_z)

    def _hover_ee_z(self, arm_tag):
        # Hover just above a fully raised mole so the smash is a short, fast jab
        # (large clearance made the press a slow multi-second descent over a
        # cresting mole, which read as a freeze-then-hit).
        clearance = float(self._cfg.get(
            "hover_clearance", self.HOVER_CLEARANCE_DEFAULT))
        target_bottom = float(
            self.board_top_z + self.mole_height - self._mole_raise_drop() + clearance)
        return self._ee_z_for_cube_bottom(arm_tag, target_bottom)

    def _cube_xy_err(self, idx, arm_tag):
        hole = self.holes[self.mole_holes[idx]]
        head_p = self._mallet_head_center(arm_tag)
        if head_p is None:
            return 1.0
        return float(np.linalg.norm(head_p[:2] - hole[:2]))

    def _carry_quat(self):
        """EE orientation used to carry the mallet level, head axis vertical."""
        return list(GRASP_DIRECTION_DIC["top_down"])

    def _head_bottom_drop(self, arm_tag):
        """Distance from the head center down to its striking surface (m)."""
        mallet = getattr(self, "hammer_cubes", {}).get(str(arm_tag))
        if mallet is None:
            return float(self.MALLET_HEAD_HALF_X)
        center, bottom = self._mallet_head_geometry(mallet)
        return float(center[2] - bottom[2])

    def _ee_pose_for_head(self, arm_tag, xy, head_bottom_z):
        """Absolute EE pose putting the head over ``xy`` with its underside at z.

        The mallet is rigidly welded to the EE, so the head offset is exact for
        the canonical carry orientation. Commanding an absolute pose (instead of
        chaining relative displacements) keeps the arm in one consistent posture
        and stops the wrist from drifting the mallet out of level.
        """
        local_T = self._cube_weld.get(str(arm_tag))
        if local_T is None:
            return None
        quat = self._carry_quat()
        R_ee = self._pose7_to_mat(np.array([0.0, 0.0, 0.0, *quat], dtype=float))[:3, :3]
        head_local = (local_T[:3, :3] @ np.array([0.0, self.MALLET_HEAD_Y, 0.0])
                      + local_T[:3, 3])
        offset = R_ee @ head_local
        return [
            float(xy[0] - offset[0]),
            float(xy[1] - offset[1]),
            float(head_bottom_z + self._head_bottom_drop(arm_tag) - offset[2]),
            *quat,
        ]

    def _approach_hole(self, idx, arm_tag, quick=False):
        """Park the mallet head over ``idx``'s hole at a safe hover height.

        Approach sits a few cm above jab hover so a slow XY plan cannot graze a
        cresting mole; hit checks are suppressed for the same reason.
        """
        hole = self.holes[self.mole_holes[idx]]
        clearance = float(self._cfg.get(
            "hover_clearance", self.HOVER_CLEARANCE_DEFAULT))
        # Clear the fully-raised crown while aligning.
        head_bottom = float(
            self.board_top_z
            + self.mole_height
            - self._mole_raise_drop()
            + clearance
            + 0.025
        )
        target = self._ee_pose_for_head(arm_tag, hole, head_bottom)

        self._suppress_board_hit = True
        self._suppress_mole_hit = True
        try:
            if not self.plan_success:
                self.plan_success = True
            if target is not None:
                self.move(self.move_to_pose(arm_tag, target))
            if target is None or not self.plan_success:
                # Fall back to a relative slide measured off the head itself.
                self.plan_success = True
                head_p = self._mallet_head_center(arm_tag)
                if head_p is not None:
                    self.move(self.move_by_displacement(
                        arm_tag=arm_tag,
                        x=float(hole[0] - head_p[0]),
                        y=float(hole[1] - head_p[1]),
                        z=float(head_bottom - self._cube_bottom_z(arm_tag)),
                        move_axis="world",
                    ))

            # Close the loop on the measured head center; the weld/IK can leave
            # a centimetre of residual offset.
            if getattr(self, "_cubes_ready", False):
                if not self.plan_success:
                    self.plan_success = True
                err = self._cube_xy_err(idx, arm_tag)
                if err >= (0.02 if quick else 0.012):
                    head_p = self._mallet_head_center(arm_tag)
                    self.move(self.move_by_displacement(
                        arm_tag=arm_tag,
                        x=float(hole[0] - head_p[0]),
                        y=float(hole[1] - head_p[1]),
                        z=0.0,
                        move_axis="world",
                    ))
        finally:
            self._suppress_board_hit = False
            self._suppress_mole_hit = False
        if not self.plan_success:
            self.plan_success = True

    def _press_down(self, arm_tag, depth=None, mole_idx=None):
        """Drive the mallet head's flat face down onto a raised mole crown.

        The stroke is only ever committed once the mole is near its crest (see
        ``PRESS_COMMIT_FRAC``), so aiming a centimetre into the current crown
        both lands a real contact and keeps the head clear of the board deck.
        """
        if depth is None:
            head_bottom = self._cube_bottom_z(arm_tag)
            mole_top = self.board_top_z + self.mole_height - self._mole_raise_drop()
            if mole_idx is not None and getattr(self, "_mole_state", None):
                if 0 <= int(mole_idx) < len(self._mole_state):
                    mole_top = self._critter_top_z(
                        self.moles, self._mole_state, int(mole_idx))
            overshoot = float(self._cfg.get(
                "press_overshoot", self.PRESS_OVERSHOOT))
            board_clear = float(self.board_top_z + 0.030)
            target_bottom = max(mole_top - overshoot, board_clear)
            depth = max(0.01, float(head_bottom - target_bottom))
        return self.move_by_displacement(
            arm_tag=arm_tag, z=-float(depth), move_axis="world")

    def _press_up(self, arm_tag, depth=None):
        if depth is None:
            ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
            hover_ee_z = self._hover_ee_z(arm_tag)
            depth = max(0.02, float(hover_ee_z - ee[2]))
        return self.move_by_displacement(
            arm_tag=arm_tag, z=float(depth), move_axis="world")

    # ------------------------------------------------------------- policy
    def play_once(self):
        # Pick both side-staged mallets (together) before playing.
        self.pickup_mallets()

        for group in self.schedule:
            # Soft-recover from motion-plan blips so later moles still get pressed.
            if not self.plan_success and not self._episode_failed():
                self.plan_success = True
            if self._episode_failed():
                break
            group = [i for i in group if not self.touched[i]]
            if not group:
                continue
            if len(group) == 1:
                self._play_single(group[0])
            else:
                self._play_pair(group[0], group[1])

        # Episode success is moles+no-rabbit; don't leave plan_success False from
        # a late hover blip if every mole was actually hit.
        if (
            not self._episode_failed()
            and getattr(self, "touched", None)
            and all(self.touched)
        ):
            self.plan_success = True

        arms_used = sorted({
            str(self._arm_for_hole(self.mole_holes[i])) for i in range(self.num_moles)
        })
        self.info["info"] = {
            "{A}": f"{self.MOLE_MODEL}/base0",
            "{B}": "hole_board",
            "{a}": arms_used[0] if len(arms_used) == 1 else "both arms",
            "{n}": str(self.num_moles),
            "{m}": self.difficulty,
            "{d}": str(self.num_distractors),
            "{R}": f"{self.RABBIT_MODEL}/base0",
            "{o}": self._option_label(),
        }
        return self.info

    def _wait_while(self, predicate, max_steps):
        """Step physics (and recording) until ``predicate()`` clears."""
        for _ in range(int(max_steps)):
            if not predicate():
                return True
            self._dwell(1)
        return False

    def _wait_until_approachable(self, idxs, max_steps=2500):
        """Wait until every listed mole is low enough to slide the mallet over.

        Sliding across a cresting mole would knock it sideways instead of
        striking it from above, so the mallet only travels while the moles are
        near the bottom of their bob or already on the way down.
        """
        limit = float(self._cfg.get(
            "approach_falling_frac", self.APPROACH_FALLING_FRAC))

        def blocked():
            if self._episode_failed():
                return False
            for i in idxs:
                if self.touched[i] or self._mole_state[i].get("exhausted"):
                    continue
                frac = self._mole_rise_frac(i)
                if frac <= 0.12:
                    continue
                if (not self._mole_is_rising(i)) and frac <= limit:
                    continue
                return True
            return False

        return self._wait_while(blocked, max_steps)

    def _wait_for_crest(self, idxs, max_steps=2500):
        """Kick a fresh rise if needed, then wait until the moles are near the top."""
        live = [
            i for i in idxs
            if not self.touched[i] and not self._mole_state[i].get("exhausted")
        ]
        if not live or self._episode_failed():
            return True
        # Restarting from the bottom gives both arms one shared crest in hard
        # mode, and stops an Opt-2 mole relocating between approach and strike.
        if any(self._mole_rise_frac(i) < 0.15 or not self._mole_is_rising(i)
               for i in live):
            self._sync_rise(live)
        commit = float(self._cfg.get("press_commit_frac", self.PRESS_COMMIT_FRAC))

        def not_ready():
            if self._episode_failed():
                return False
            return not all(
                self.touched[i]
                or self._mole_state[i].get("exhausted")
                or self._mole_rise_frac(i) >= commit
                for i in live)

        return self._wait_while(not_ready, max_steps)

    def _strike(self, pressers):
        """Drive the mallet head(s) onto near-crest moles, then lift clear.

        ``pressers`` is a list of ``(mole_idx, arm_tag)``, at most one per arm.
        The bob is pinned for the duration of the stroke so the mole cannot slip
        back down the hole while the head is still descending.
        """
        live = [(i, a) for i, a in pressers if not self.touched[i]]
        if not live:
            return
        for i, _arm in live:
            self._mole_state[i]["freeze_bob"] = True
        try:
            self.plan_success = True
            self._move_arms([self._press_down(a, mole_idx=i) for i, a in live])
            self._dwell(12)
            again = [(i, a) for i, a in live if not self.touched[i]]
            if again:
                self.plan_success = True
                self._move_arms(
                    [self._press_down(a, mole_idx=i) for i, a in again])
                self._dwell(10)
        finally:
            for i, _arm in live:
                self._mole_state[i]["freeze_bob"] = False
        self.plan_success = True
        self._move_arms([self._press_up(a) for _i, a in live])
        self.plan_success = True

    def _play_single(self, idx):
        """Park over the hole while the mole is down, then smash it at the crest."""
        attempts = int(self._cfg.get("strike_attempts", self.STRIKE_ATTEMPTS))
        tol = float(self._cfg.get("strike_xy_tol", self.STRIKE_XY_TOL))
        for _attempt in range(attempts):
            if (
                self.touched[idx]
                or self._episode_failed()
                or self._mole_state[idx].get("exhausted")
            ):
                return
            self.plan_success = True
            self._wait_until_approachable([idx])
            if self.touched[idx] or self._episode_failed():
                return
            arm = self._arm_for_hole(self.mole_holes[idx])
            if self._cube_xy_err(idx, arm) > 0.015:
                self._approach_hole(idx, arm, quick=True)
            if self.touched[idx] or self._episode_failed():
                return
            self._wait_for_crest([idx])
            if self.touched[idx] or self._episode_failed():
                return
            arm = self._arm_for_hole(self.mole_holes[idx])
            if self._cube_xy_err(idx, arm) > tol:
                continue  # the mole relocated (Opt 2) — realign and retry
            self._strike([(idx, arm)])

    def _play_pair(self, i, j):
        """Hard mode: both arms smash one mole each on a shared crest."""
        if self.holes[self.mole_holes[i]][0] > self.holes[self.mole_holes[j]][0]:
            i, j = j, i
        attempts = int(self._cfg.get("strike_attempts", self.STRIKE_ATTEMPTS))
        tol = float(self._cfg.get("strike_xy_tol", self.STRIKE_XY_TOL))
        for _attempt in range(attempts):
            if self._episode_failed():
                return
            live = [
                k for k in (i, j)
                if not self.touched[k] and not self._mole_state[k].get("exhausted")
            ]
            if not live:
                return
            arms = {k: self._arm_for_hole(self.mole_holes[k]) for k in live}
            if len(live) < 2 or arms[live[0]] == arms[live[1]]:
                # One mole left, or both drifted onto the same arm's half.
                for k in live:
                    self._play_single(k)
                return

            self.plan_success = True
            self._wait_until_approachable(live)
            if self._episode_failed():
                return
            for k in live:
                if self.touched[k]:
                    continue
                arm = self._arm_for_hole(self.mole_holes[k])
                if self._cube_xy_err(k, arm) > 0.015:
                    self._approach_hole(k, arm, quick=True)

            live = [
                k for k in live
                if not self.touched[k] and not self._mole_state[k].get("exhausted")
            ]
            if not live or self._episode_failed():
                return
            self._wait_for_crest(live)
            if self._episode_failed():
                return

            pressers = []
            for k in live:
                if self.touched[k]:
                    continue
                arm = self._arm_for_hole(self.mole_holes[k])
                if self._cube_xy_err(k, arm) <= tol:
                    pressers.append((k, arm))
            if len({str(a) for _k, a in pressers}) < len(pressers):
                for presser in pressers:
                    self._strike([presser])
            elif pressers:
                self._strike(pressers)

    # ------------------------------------------------------------- success
    def check_success(self):
        """Success = all moles touched; no rabbit hit; no appearance budget spent."""
        if self._episode_failed():
            return False
        if not getattr(self, "touched", None):
            return False
        return bool(all(self.touched) and len(self.touched) == self.num_moles)

    def get_obs(self):
        obs = super().get_obs()
        obs["whack_moles"] = {
            "difficulty": str(getattr(self, "difficulty", "easy")),
            "option": self._option_label(),
            "distractor_enabled": bool(getattr(self, "distractor_enabled", False)),
            "relocating_moles": bool(getattr(self, "relocating_moles", False)),
            "num_moles": int(getattr(self, "num_moles", 0)),
            "num_distractors": int(getattr(self, "num_distractors", 0)),
            "num_appearances": int(getattr(
                self, "num_appearances", self.NUM_APPEARANCES_DEFAULT)),
            "appearances": [
                int(st.get("appearances", 0))
                for st in getattr(self, "_mole_state", [])
            ],
            "touched": [bool(t) for t in getattr(self, "touched", [])],
            "distractor_hit": bool(getattr(self, "distractor_hit", False)),
            "appearances_exhausted": bool(
                getattr(self, "appearances_exhausted", False)),
            "board_hit": bool(getattr(self, "board_hit", False)),
            "raised": [bool(st.get("raised", False)) for st in getattr(self, "_mole_state", [])],
            "rabbit_raised": [
                bool(st.get("raised", False)) for st in getattr(self, "_rabbit_state", [])
            ],
            "holes": [int(h) for h in getattr(self, "mole_holes", [])],
            "rabbit_holes": [int(h) for h in getattr(self, "rabbit_holes", [])],
        }
        return obs
