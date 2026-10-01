from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d


class marble_shelf_maze(Base_Task):
    """A marble threads down a zig-zag stack of shelves hung between two parallel glass sheets.

    Two thin vertical glass panes face the robot, separated by a narrow gap. Short glass shelves
    (catch_shelf_marble / catch_cuboid window-glass look: light-blue tint + 80% transmission) are
    wedged crosswise in that gap. A marble starts at rest on the centre of the top shelf. Two
    spring push-buttons sit on the table; pressing one tilts the active shelf toward that side so
    the marble rolls off and drops onto the shelf below (or, from the bottom shelf, into the bowl).
    In the interactive viewer, hold-to-tilt / release-to-return is driven by gripper-Z (Q) on the
    keycaps (same ReactivePushButtons model as catch_marbles_trapdoors).

    Options (independent toggles; Opt 1+2 can both be on):
      - Default — after each inter-shelf drop the marble snaps to rest at the shelf centre.
      - Opt 1 — ``continuous_ball_motion``: landed marble keeps real dynamics (no centre snap).
        CLI: ``--task-arg continuous_ball_motion=true`` or ``--option 1``.
        Alias: ``natural_ball_stop=true``.
      - Opt 2 — ``oscillating_bowl_enabled``: bowl sweeps left↔right under the maze; the last
        drop must be timed so the marble falls into the moving bowl.
        CLI: ``--task-arg oscillating_bowl_enabled=true`` or ``--option 2``.

    Geometry (zig-zag offsets, per-level gaps, n_shelves) is randomized every episode. Marble
    transitions use real PhysX; shelf tilt / marble slide-to-edge / bowl oscillation are
    kinematically scripted from step counts for two-pass determinism.
    """

    N_SHELVES_DEFAULT = 4
    N_SHELVES_HARD_MAX = 6               # absolute cap -- keeps the stack within the cameras' FOV
    SHELF_LENGTH_DEFAULT = 0.18          # x-length of each shelf (10% wider each side vs 0.15)
    GLASS_GAP_DEFAULT = 0.05             # gap between the two glass sheets
    SHELF_HALF_DEPTH_DEFAULT = 0.0125    # shelf depth (y, into the glass gap)
    SHELF_HALF_THICK_DEFAULT = 0.00625   # shelf thickness (z, vertical) -- half of the previous value
    LEVEL_GAP_DEFAULT = 0.055            # vertical spacing between shelf centers (tuned so the
                                          # whole 4-shelf stack, table to top, stays inside the
                                          # head camera's vertical FOV)
    OVERLAP_MIN_DEFAULT = 0.30           # min fraction of shelf_length consecutive shelves overlap by
    OVERLAP_MAX_DEFAULT = 0.75           # max fraction of shelf_length consecutive shelves overlap by
    BOTTOM_CLEARANCE_DEFAULT = 0.15      # vertical gap from bottom shelf down to the table (raised ~10cm
                                          # so the whole stack sits higher above the table)
    LEVEL_GAP_JITTER_DEFAULT = 0.35      # +/- fractional randomization applied independently to each
                                          # inter-shelf vertical gap (renormalized so their sum still
                                          # matches the reference total stack height -- see load_actors)
    MIN_LEVEL_GAP_DEFAULT = 0.035        # floor so a jittered-down gap still lets the marble clear the fall

    BALL_RADIUS_DEFAULT = 0.012
    ROLL_OFF_SPEED_DEFAULT = 0.06        # m/s -- small horizontal speed imparted when a marble leaves a shelf edge
    CONTINUOUS_BALL_MOTION_DEFAULT = False  # Opt 1: landed marble keeps real dynamics (no centre snap)
    NATURAL_BALL_STOP_DEFAULT = False    # alias of continuous_ball_motion (legacy name)
    # Opt 1: slower release + stronger damping so the marble is less likely to roll off after landing.
    CONTINUOUS_BALL_SPEED_SCALE_DEFAULT = 0.40   # multiplies roll_off_speed when continuous_ball_motion
    CONTINUOUS_TILT_DURATION_SCALE_DEFAULT = 1.50  # slower kinematic slide-to-edge with the tilt
    CONTINUOUS_LINEAR_DAMPING = 1.20
    CONTINUOUS_ANGULAR_DAMPING = 4.0
    CONTINUOUS_STATIC_FRICTION = 1.50
    CONTINUOUS_DYNAMIC_FRICTION = 1.35

    TILT_ANGLE_DEG_DEFAULT = 45.0
    TILT_DURATION_SEC_DEFAULT = 0.5      # seconds to sweep from flat to full tilt (and to reset back) --
                                          # used as-is unless smooth_tilt is enabled (see below)
    SMOOTH_TILT_DEFAULT = False          # if True, tilt speed (not a fixed duration) drives the sweep,
                                          # so holding the button noticeably longer produces the same 45 deg max
    TILT_SPEED_DEG_PER_SEC_DEFAULT = 30.0  # deg/s used for the sweep when smooth_tilt is enabled
    TILT_CLEARANCE_DEFAULT = 0.005       # m; keep tilting shelf underside this far above the shelf/table below
    FALL_SETTLE_STEPS_DEFAULT = 160      # physics steps to let the marble fall+settle onto the next shelf
    FINAL_FALL_SETTLE_STEPS_DEFAULT = 220  # physics steps for the last drop into the bowl (falls further)

    BUTTON_X_DEFAULT = [-0.18, 0.18]     # scaled with wider shelves (±10% each side vs ±0.15)
    BUTTON_Y_DEFAULT = -0.18
    BUTTON_HALF_DEFAULT = [0.022, 0.022, 0.018]
    BUTTON_PRESS_DEPTH_DEFAULT = 0.03
    PRESS_HOLD_STEPS_DEFAULT = 20
    POST_PRESS_DWELL_DEFAULT = 10

    BOWL_ID_DEFAULT = 1
    BOWL_SCALE_MULT_DEFAULT = 0.65
    BOWL_CATCH_RADIUS_DEFAULT = 0.032     # horizontal capture tolerance around the bowl center
    BOWL_HEIGHT_DEFAULT = 0.045
    OSCILLATING_BOWL_ENABLED_DEFAULT = False  # Opt 2: bowl sweeps left↔right under the maze
    OSCILLATING_BOWL_PERIOD_DEFAULT = 3.0     # s; full left→right→left cycle
    OSCILLATING_BOWL_ALIGN_TOL_FRAC = 0.70    # fraction of bowl_catch_radius for expert timing

    # Glass shelves (exact replica of catch_shelf_marble / catch_cuboid window-glass path).
    SHELF_COLOR = [0.94, 0.97, 1.0]
    SHELF_TRANSMISSION = 0.8
    SHELF_TRANSMISSION_ROUGHNESS = 0.0
    SHELF_ROUGHNESS = 0.02
    SHELF_IOR = 1.45

    # Interactive hold-to-tilt: at least 50% slower shelf sweep than expert demos.
    INTERACTIVE_TILT_SPEED_SCALE = 0.5

    GRAVITY = 9.81

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("marble_shelf_maze", {})
        # The collector reuses this env across episodes; _init_task_env_ runs load_camera (which
        # calls _update_kinematic_tasks) BEFORE the new load_actors rebuilds the scene. Clear all
        # per-episode state here so the guard at the top of _update_kinematic_tasks blocks until
        # load_actors has run again.
        self._loaded = False
        self.shelves = []
        self.shelf_centers_x = []
        self.shelf_z = []
        self.shelf_half_len = 0.0
        self.shelf_half_thick = 0.0
        self.shelf_half_depth = 0.0
        self._shelf_cur_angle = []
        self._shelf_target_angle = []
        self.ball = None
        self._ball_rigid = None
        self.active_shelf_idx = 0
        # Completed-action partial score: retain the deepest shelf reached even
        # if the marble later falls.  This is not a distance-to-goal measure.
        self._partial_deepest_shelf = 0
        self._ball_mode = "resting"      # resting | sliding | falling | done | missed
        self._reset_metric_state()
        self._sliding_shelf_idx = -1
        self._sliding_dir = 0
        self._sliding_start_local_x = 0.0
        self._sliding_local_x = 0.0
        self._sliding_last_p = None
        self._sliding_release_angle = 0.0  # signed deg actually used for this tilt (gap-limited)
        self.bowl = None
        self.bowl_side = "left"
        self.correct_dir = []
        self.continuous_ball_motion = False
        self.natural_ball_stop = False
        self.osc_bowl_enabled = False
        self._bowl_steps = 0
        self._bowl_armed = False
        self._bowl_mid_x = 0.0
        self._bowl_amp = 0.0
        self._bowl_period = self.OSCILLATING_BOWL_PERIOD_DEFAULT
        self._bowl_base_y = 0.0
        self._bowl_base_z = 0.0
        self._bowl_q = [0.5, 0.5, 0.5, 0.5]
        self.tilt_clearance = self.TILT_CLEARANCE_DEFAULT
        self._expert_hold = None  # interactive keyboard: "left" / "right" / None
        self._hold_dir = None     # last interactive held button (None / left / right)
        self._pending_tilt_dir = None
        self._pending_active_shelf = None  # land while held → promote on release
        self._hold_tilt_shelf_idx = None   # shelf kept tilted until release after a drop
        self._button_arrows = {"left": [], "right": []}
        super()._init_task_env_(**kwags)
        # Opt 2 bowl motion from episode start (policy eval never calls play_once).
        self._bowl_armed = bool(getattr(self, "osc_bowl_enabled", False))

    # ------------------------------------------------------------------ helpers
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
        raise ValueError(f"marble_shelf_maze expected a boolean, got {value!r}")

    def _parse_continuous_ball_motion(self, c) -> bool:
        """Opt 1: marble keeps moving after landing (preferred) or legacy ``option: 1``.

        Accepts ``continuous_ball_motion`` (preferred) or the older ``natural_ball_stop`` alias.
        """
        cont = c.get("continuous_ball_motion", None)
        if cont is None:
            cont = c.get("natural_ball_stop", c.get("opt1", None))
        legacy = c.get("option", None)
        if legacy is not None and cont is None:
            if legacy in (1, "1", "continuous_ball_motion", "natural_ball_stop", "continuous"):
                cont = True
            elif legacy in (2, "2", "oscillating_bowl_enabled", "oscillating_bowl", "osc"):
                cont = False
            else:
                raise ValueError(
                    "marble_shelf_maze option must be 1/continuous_ball_motion or "
                    "2/oscillating_bowl_enabled (or set the booleans directly)"
                )
        return self._as_bool(cont, self.CONTINUOUS_BALL_MOTION_DEFAULT)

    def _parse_oscillating_bowl_enabled(self, c) -> bool:
        """Opt 2: bowl oscillates left↔right under the maze (preferred) or legacy ``option: 2``."""
        osc = c.get("oscillating_bowl_enabled", c.get("opt2", None))
        legacy = c.get("option", None)
        if legacy is not None and osc is None:
            if legacy in (2, "2", "oscillating_bowl_enabled", "oscillating_bowl", "osc"):
                osc = True
            elif legacy in (1, "1", "continuous_ball_motion", "natural_ball_stop", "continuous"):
                osc = False
            else:
                raise ValueError(
                    "marble_shelf_maze option must be 1/continuous_ball_motion or "
                    "2/oscillating_bowl_enabled (or set the booleans directly)"
                )
        return self._as_bool(osc, self.OSCILLATING_BOWL_ENABLED_DEFAULT)

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "continuous_ball_motion", False):
            parts.append("option 1")
        if getattr(self, "osc_bowl_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _use_viewer_glass(self) -> bool:
        """Interactive SAPIEN viewer cannot composite transmission glass — use plain alpha."""
        if bool(getattr(self, "_plain_glass", False)):
            return True
        cfg = getattr(self, "_cfg", {}) or {}
        if bool(cfg.get("plain_glass", False)):
            return True
        return bool(
            getattr(self, "_interactive_robot_mode", False)
            or getattr(self, "_interactive_universal_controls", False)
        )

    def _make_glass_material(self):
        """Shelf glass: transmission for demo cameras; pour_beer-style alpha for interactive."""
        if self._use_viewer_glass():
            # pour_beer-style plain alpha, but stronger light-blue + ~30% less transparent
            # than the initial 0.28 alpha (0.28 → ~0.50) so shelves read clearly in the viewer.
            glass = sapien.render.RenderMaterial(
                base_color=[0.72, 0.88, 0.98, 0.50]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.10)
                glass.set_metallic(0.0)
            except Exception:
                try:
                    glass.roughness = 0.10
                    glass.metallic = 0.0
                except Exception:
                    pass
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        # Expert / demo recording: catch_shelf_marble / catch_cuboid transmission glass.
        glass = sapien.render.RenderMaterial(base_color=[*self.SHELF_COLOR, 1.0])
        glass.set_transmission(float(self.SHELF_TRANSMISSION))
        glass.set_transmission_roughness(float(self.SHELF_TRANSMISSION_ROUGHNESS))
        glass.set_roughness(float(self.SHELF_ROUGHNESS))
        glass.set_metallic(0.0)
        try:
            glass.set_ior(float(self.SHELF_IOR))
        except Exception:
            glass.ior = float(self.SHELF_IOR)
        return glass

    def _create_glass_shelf(self, pose, half_size, is_static, name):
        """Shelf box with collision + catch_shelf_marble glass visual (moves with the entity).

        Pose z is used as-authored (already includes table_z_bias via shelf_z), matching the
        existing create_box(self.scene, ...) call sites in this task.
        """
        scene = self.scene
        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(pose)

        rigid = (
            sapien.physx.PhysxRigidDynamicComponent()
            if not is_static
            else sapien.physx.PhysxRigidStaticComponent()
        )
        # Slippery glass: a kinematic untilt must not drag the marble back up the slope.
        glass_phys = sapien.physx.PhysxMaterial(
            static_friction=0.12, dynamic_friction=0.08, restitution=0.0,
        )
        rigid.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half_size,
                material=glass_phys,
            )
        )
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(half_size, self._make_glass_material()))
        entity.add_component(rigid)
        entity.add_component(render)
        scene.add_entity(entity)

        data = {
            "center": [0, 0, 0],
            "extents": half_size,
            "scale": half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
            "contact_points_description": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "target_point_description": [],
        }
        return Actor(entity, data)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        # n_shelves can be fixed (n_shelves: k) or randomized per-episode from a range
        # (n_shelves_min/n_shelves_max) -- varying how many shelves the marble has to descend
        # before reaching the bowl changes both the length and the direction pattern of the
        # required button-press combination from one episode to the next. Capped at
        # N_SHELVES_HARD_MAX (6) so the stack always fits inside the cameras' FOV.
        n_min = c.get("n_shelves_min", None)
        n_max = c.get("n_shelves_max", None)
        if n_min is not None and n_max is not None:
            n_min = max(2, int(n_min))
            n_max = min(self.N_SHELVES_HARD_MAX, int(n_max))
            n_max = max(n_min, n_max)
            self.n_shelves = int(np.random.randint(n_min, n_max + 1))
        else:
            self.n_shelves = min(self.N_SHELVES_HARD_MAX, max(2, int(c.get("n_shelves", self.N_SHELVES_DEFAULT))))
        self.shelf_length = float(c.get("shelf_length", self.SHELF_LENGTH_DEFAULT))
        self.glass_gap = float(c.get("glass_gap", self.GLASS_GAP_DEFAULT))
        # The *nominal* level_gap sets the reference stack height (see below); the gap actually used
        # between any given pair of consecutive shelves is jittered independently per level (see the
        # level_gaps computation further down) rather than being applied uniformly.
        level_gap_cfg = float(c.get("level_gap", self.LEVEL_GAP_DEFAULT))
        bottom_clearance_cfg = float(c.get("bottom_clearance", self.BOTTOM_CLEARANCE_DEFAULT))
        stack_ref_n = int(c.get("stack_height_ref_n_shelves", self.N_SHELVES_DEFAULT))
        min_bottom_clearance = float(c.get("min_bottom_clearance", 0.04))
        total_span_ref = bottom_clearance_cfg + max(stack_ref_n - 1, 0) * level_gap_cfg

        # Overlap (as a fraction of shelf_length) between each pair of consecutive shelves is
        # randomized independently per level within [overlap_min, overlap_max]; a raw offset_min/
        # offset_max pair can still be given directly to override this if ever needed.
        self.overlap_min = float(c.get("overlap_min", self.OVERLAP_MIN_DEFAULT))
        self.overlap_max = float(c.get("overlap_max", self.OVERLAP_MAX_DEFAULT))
        default_offset_min = self.shelf_length * (1.0 - self.overlap_max)   # smallest offset = max overlap
        default_offset_max = self.shelf_length * (1.0 - self.overlap_min)   # largest offset = min overlap
        self.offset_min = float(c.get("offset_min", default_offset_min))
        self.offset_max = float(c.get("offset_max", default_offset_max))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.roll_off_speed = float(c.get("roll_off_speed", self.ROLL_OFF_SPEED_DEFAULT))
        self.continuous_ball_motion = self._parse_continuous_ball_motion(c)
        # Keep the legacy attribute name in sync so older call sites / obs still work.
        self.natural_ball_stop = bool(self.continuous_ball_motion)
        self.continuous_ball_speed_scale = float(
            c.get("continuous_ball_speed_scale", self.CONTINUOUS_BALL_SPEED_SCALE_DEFAULT)
        )
        self.continuous_tilt_duration_scale = float(
            c.get("continuous_tilt_duration_scale", self.CONTINUOUS_TILT_DURATION_SCALE_DEFAULT)
        )

        self.tilt_angle_deg = float(c.get("tilt_angle_deg", self.TILT_ANGLE_DEG_DEFAULT))
        self.tilt_clearance = float(c.get("tilt_clearance", self.TILT_CLEARANCE_DEFAULT))
        self.smooth_tilt = bool(c.get("smooth_tilt", self.SMOOTH_TILT_DEFAULT))
        self.tilt_speed_deg_per_sec = float(c.get("tilt_speed_deg_per_sec", self.TILT_SPEED_DEG_PER_SEC_DEFAULT))
        if self.smooth_tilt:
            # Speed-driven sweep: the button must stay held roughly angle/speed seconds for the
            # shelf to reach the full 45 deg (the caller already holds throughout the tilt+settle
            # dwell, so slowing the speed automatically makes the press "hold longer").
            self.tilt_duration_sec = self.tilt_angle_deg / max(self.tilt_speed_deg_per_sec, 1e-6)
        else:
            self.tilt_duration_sec = float(c.get("tilt_duration_sec", self.TILT_DURATION_SEC_DEFAULT))
        # Opt 1: slower tilt → slower scripted slide toward the edge (less likely to overshoot).
        if self.continuous_ball_motion:
            self.tilt_duration_sec *= max(float(self.continuous_tilt_duration_scale), 1e-6)
        self.fall_settle_steps = int(c.get("fall_settle_steps", self.FALL_SETTLE_STEPS_DEFAULT))
        self.final_fall_settle_steps = int(c.get("final_fall_settle_steps", self.FINAL_FALL_SETTLE_STEPS_DEFAULT))
        self._sliding_release_angle = 0.0

        self.button_x = list(c.get("button_x", self.BUTTON_X_DEFAULT))
        self.button_y = float(c.get("button_y", self.BUTTON_Y_DEFAULT))
        self.button_half = list(c.get("button_half", self.BUTTON_HALF_DEFAULT))
        self.button_press_depth = float(c.get("button_press_depth", self.BUTTON_PRESS_DEPTH_DEFAULT))
        self.press_hold_steps = int(c.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(c.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))

        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_catch_radius = float(c.get("bowl_catch_radius", self.BOWL_CATCH_RADIUS_DEFAULT))
        self.bowl_height = float(c.get("bowl_height", self.BOWL_HEIGHT_DEFAULT))
        self.osc_bowl_enabled = self._parse_oscillating_bowl_enabled(c)
        self._bowl_period = float(c.get("oscillating_bowl_period", self.OSCILLATING_BOWL_PERIOD_DEFAULT))

        self.table_z = 0.74 + self.table_z_bias
        self.maze_y = 0.0

        self.shelf_half_len = self.shelf_length / 2.0
        self.shelf_half_thick = float(c.get("shelf_half_thick", self.SHELF_HALF_THICK_DEFAULT))    # z (vertical)
        self.shelf_half_depth = float(c.get("shelf_half_depth", self.SHELF_HALF_DEPTH_DEFAULT))    # y (into the gap)

        # ---- randomize the zig-zag: alternating left/right offsets between consecutive shelves,
        # each with an independently randomized overlap fraction -- then re-center the whole stack
        # on the table's x=0 midline regardless of where the random walk happened to drift to ----
        s0 = float(np.random.choice([-1.0, 1.0]))
        signs = [s0 * ((-1.0) ** i) for i in range(self.n_shelves - 1)]
        offsets = [
            float(sign * np.random.uniform(self.offset_min, self.offset_max)) for sign in signs
        ]
        centers = [0.0]
        for off in offsets:
            centers.append(centers[-1] + off)
        mid_x = (min(centers) + max(centers)) / 2.0
        self.shelf_centers_x = [x - mid_x for x in centers]
        # the correct press direction at level i (0..n-2) is simply which way shelf i+1 sits
        self.correct_dir = ["right" if off > 0 else "left" for off in offsets]

        # ---- randomize the vertical gap between each pair of consecutive shelves independently,
        # renormalized so their sum still equals the reference total (n_shelves-1)*level_gap_cfg --
        # this keeps the top of the stack (and so the cameras' framing) the same as a uniform-gap
        # stack while still varying the individual spacings episode to episode ----
        n_gaps = max(self.n_shelves - 1, 1)
        jitter = float(c.get("level_gap_jitter", self.LEVEL_GAP_JITTER_DEFAULT))
        min_gap = float(c.get("min_level_gap", self.MIN_LEVEL_GAP_DEFAULT))
        nominal_total_gap = n_gaps * level_gap_cfg
        raw_gaps = [level_gap_cfg * float(np.random.uniform(1.0 - jitter, 1.0 + jitter)) for _ in range(n_gaps)]
        scale = nominal_total_gap / max(sum(raw_gaps), 1e-6)
        self.level_gaps = [max(g * scale, min_gap) for g in raw_gaps]
        self._level_gap_nominal = level_gap_cfg

        self.bottom_clearance = max(min_bottom_clearance, total_span_ref - sum(self.level_gaps))
        top_z = self.table_z + self.bottom_clearance + sum(self.level_gaps)
        self.shelf_z = [top_z]
        for gap in self.level_gaps:
            self.shelf_z.append(self.shelf_z[-1] - gap)

        self.shelves = []
        self._shelf_cur_angle = [0.0] * self.n_shelves
        self._shelf_target_angle = [0.0] * self.n_shelves
        for i in range(self.n_shelves):
            shelf = self._create_glass_shelf(
                pose=sapien.Pose([self.shelf_centers_x[i], self.maze_y, self.shelf_z[i]]),
                half_size=[self.shelf_half_len, self.shelf_half_depth, self.shelf_half_thick],
                is_static=False,
                name=f"maze_shelf_{i}",
            )
            rigid = self._get_rigid(shelf)
            if rigid is not None:
                try:
                    rigid.set_disable_gravity(True)
                    rigid.set_kinematic(True)
                    rigid.set_kinematic_target(
                        sapien.Pose([self.shelf_centers_x[i], self.maze_y, self.shelf_z[i]])
                    )
                except Exception:
                    pass
            self.shelves.append(shelf)

        # ---- two thin "glass" sheets flanking the shelf stack (static scenery) ----
        min_x = min(self.shelf_centers_x) - self.shelf_half_len - 0.03
        max_x = max(self.shelf_centers_x) + self.shelf_half_len + 0.03
        glass_half_w = (max_x - min_x) / 2.0
        glass_cx = (max_x + min_x) / 2.0
        glass_bottom_z = self.table_z + 0.01
        glass_top_z = top_z + self.shelf_half_thick + 0.05
        glass_half_h = (glass_top_z - glass_bottom_z) / 2.0
        glass_cz = (glass_top_z + glass_bottom_z) / 2.0
        # Demo cameras: open frames (transmission panes vanish in the raster path historically).
        # Interactive viewer: pour_beer-style plain-alpha panes so the glass walls are visible.
        glass_thick = 0.004
        near_dy = -self.glass_gap / 2.0 - glass_thick / 2.0
        far_dy = self.glass_gap / 2.0 + glass_thick / 2.0
        for side, dy in (("near", near_dy), ("far", far_dy)):
            center = [glass_cx, self.maze_y + dy, glass_cz]
            if self._use_viewer_glass():
                self._build_glass_pane(
                    center,
                    half_w=glass_half_w,
                    half_h=glass_half_h,
                    half_thick=glass_thick / 2.0,
                    side=side,
                )
            else:
                self._build_glass_frame(
                    center,
                    half_w=glass_half_w,
                    half_h=glass_half_h,
                    half_thick=glass_thick / 2.0,
                    bar_half=0.006,
                    side=side,
                )

        # ---- marble: starts frozen (kinematic, no gravity) at the centre of the top shelf ----
        ball_z0 = self.shelf_z[0] + self.shelf_half_thick + self.ball_radius + 0.001
        self.ball = create_sphere(
            self.scene,
            pose=sapien.Pose([self.shelf_centers_x[0], self.maze_y, ball_z0]),
            radius=self.ball_radius,
            color=[0.85, 0.15, 0.15],
            is_static=False,
            name="maze_marble",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            try:
                if self.continuous_ball_motion:
                    # Real dynamics from the start -- it just sits at rest under gravity/friction
                    # on the (flat) top shelf rather than being kinematically frozen.
                    self._ball_rigid.set_disable_gravity(False)
                    self._ball_rigid.set_kinematic(False)
                    self._ball_rigid.set_linear_damping(self.CONTINUOUS_LINEAR_DAMPING)
                    self._ball_rigid.set_angular_damping(self.CONTINUOUS_ANGULAR_DAMPING)
                    inelastic = sapien.physx.PhysxMaterial(
                        static_friction=self.CONTINUOUS_STATIC_FRICTION,
                        dynamic_friction=self.CONTINUOUS_DYNAMIC_FRICTION,
                        restitution=0.0,
                    )
                else:
                    self._ball_rigid.set_disable_gravity(True)
                    self._ball_rigid.set_kinematic(True)
                    inelastic = sapien.physx.PhysxMaterial(
                        static_friction=0.9, dynamic_friction=0.9, restitution=0.0,
                    )
                for shape in self._ball_rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass
        self.active_shelf_idx = 0
        self._partial_deepest_shelf = 0
        self._ball_mode = "resting"
        self._sliding_shelf_idx = -1
        self._reset_metric_state()
        self._sliding_dir = 0
        self._sliding_local_x = 0.0
        self._sliding_last_p = None
        self._pending_active_shelf = None
        self._hold_tilt_shelf_idx = None
        self._hold_dir = None

        # ---- two push buttons (spring keycaps + hollow bezel), one per side/arm ----
        self.buttons = []
        self.button_bases = []
        button_homes = []
        button_tops = []
        button_ids = []
        self._button_arrows = {"left": [], "right": []}
        for bx in self.button_x:
            bz = self.table_z + self.button_half[2]
            is_right = bx > 0
            direction = "right" if is_right else "left"
            home = sapien.Pose([bx, self.button_y, bz])
            # Hollow dark bezel like catch_marbles_trapdoors / catch_shelf_marble.
            self.button_bases.extend(
                add_key_base_border(
                    self.scene,
                    float(bx),
                    float(self.button_y),
                    float(self.table_z),
                    self.button_half,
                    name_prefix=f"shelf_button_base_{direction}",
                )
            )
            btn = create_box(
                self.scene,
                pose=home,
                half_size=list(self.button_half),
                color=[0.75, 0.20, 0.20] if bx < 0 else [0.20, 0.55, 0.75],
                is_static=True,
                name=f"shelf_button_{direction}",
            )
            self.buttons.append(btn)
            # Live world pose (table_z_bias / scene frame) for spring homes.
            world_home = btn.get_pose()
            button_homes.append(world_home)
            button_tops.append(float(world_home.p[2]) + float(self.button_half[2]))
            button_ids.append(direction)
            self.add_prohibit_area(btn, padding=0.03)
            # Decal on the button's top face: a curved arrow spelling out which way the active
            # shelf will tilt if this button is pressed (clockwise for right, counter-clockwise
            # for left -- matches _set_shelf_pose's +angle-tilts-right-edge-down convention).
            top_z = float(world_home.p[2]) + float(self.button_half[2])
            self._button_arrows[direction] = self._build_turn_arrow(
                [float(world_home.p[0]), float(world_home.p[1]), top_z],
                radius=min(self.button_half[0], self.button_half[1]) * 0.65,
                clockwise=is_right,
                color=[0.96, 0.96, 0.96],
            )
        self.left_button, self.right_button = self.buttons
        self._pending_tilt_dir = None
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=self.buttons,
            home_poses=button_homes,
            max_depth=float(self.button_half[2]),
            ids=button_ids,
        )
        self._reactive_buttons.set_tops_z(button_tops)

        # ---- bowl: Default = static under a random bottom-shelf edge; Opt 2 = kinematic
        # oscillation across the full maze x-span (left↔right) so the last drop must be timed ----
        maze_x_lo = min(self.shelf_centers_x) - self.shelf_half_len
        maze_x_hi = max(self.shelf_centers_x) + self.shelf_half_len
        self._bowl_mid_x = 0.5 * (maze_x_lo + maze_x_hi)
        self._bowl_amp = 0.5 * (maze_x_hi - maze_x_lo)
        self._bowl_base_y = self.maze_y
        self._bowl_q = [0.5, 0.5, 0.5, 0.5]
        self._bowl_base_z = self.table_z
        self._bowl_steps = 0
        self._bowl_armed = False

        bottom_center_x = self.shelf_centers_x[-1]
        if self.osc_bowl_enabled:
            # Start at a random phase so the expert can't memorize a fixed wait.
            self._bowl_phase0 = float(np.random.uniform(0.0, 2.0 * np.pi))
            bowl_x = self._bowl_mid_x + self._bowl_amp * float(np.cos(self._bowl_phase0))
            # bowl_side is filled in at the timed final drop; seed with nearest edge.
            self.bowl_side = "right" if bowl_x >= bottom_center_x else "left"
        else:
            self._bowl_phase0 = 0.0
            self.bowl_side = str(np.random.choice(["left", "right"]))
            bowl_x = bottom_center_x + (
                self.shelf_half_len if self.bowl_side == "right" else -self.shelf_half_len
            )

        bowl_pose = sapien.Pose([bowl_x, self.maze_y, self.table_z], self._bowl_q)
        self.bowl = create_actor(
            self.scene, pose=bowl_pose, modelname="002_bowl", model_id=self.bowl_id,
            convex=True, is_static=not self.osc_bowl_enabled, scale_mult=self.bowl_scale_mult,
        )
        if self.osc_bowl_enabled and self.bowl is not None:
            rigid = self._get_rigid(self.bowl)
            if rigid is not None:
                try:
                    rigid.set_disable_gravity(True)
                    rigid.set_kinematic(True)
                except Exception:
                    pass
        self.bowl_center_xy = np.array([bowl_x, self.maze_y], dtype=np.float64)

        self._loaded = True

    GLASS_COLOR = [0.80, 0.90, 0.93]

    def _build_glass_pane(self, center_xyz, half_w, half_h, half_thick, side):
        """Filled translucent pane for the interactive viewer (pour_beer plain-alpha glass)."""
        entity = sapien.Entity()
        entity.set_name(f"glass_sheet_{side}")
        entity.set_pose(sapien.Pose(list(center_xyz)))
        render = sapien.render.RenderBodyComponent()
        render.attach(
            sapien.render.RenderShapeBox(
                [float(half_w), float(half_thick), float(half_h)],
                self._make_glass_material(),
            )
        )
        entity.add_component(render)
        self.scene.add_entity(entity)

    def _build_glass_frame(self, center_xyz, half_w, half_h, half_thick, bar_half, side):
        """A visual-only (no collision) rectangular frame -- four thin edge bars, open in the
        middle -- standing in for a pane of glass. Used for both sheets (near and far) so the
        camera can see straight through the whole maze from any viewpoint, not just the one it
        happened to be tuned for."""
        cx, cy, cz = center_xyz
        bars = [
            ("top", [cx, cy, cz + half_h - bar_half], [half_w, half_thick, bar_half]),
            ("bottom", [cx, cy, cz - half_h + bar_half], [half_w, half_thick, bar_half]),
            ("left", [cx - half_w + bar_half, cy, cz], [bar_half, half_thick, half_h]),
            ("right", [cx + half_w - bar_half, cy, cz], [bar_half, half_thick, half_h]),
        ]
        for name, pos, half_size in bars:
            create_visual_box(self.scene, sapien.Pose(pos), half_size, color=self.GLASS_COLOR,
                               name=f"glass_sheet_{side}_{name}")

    def _build_turn_arrow(self, top_center_xyz, radius, clockwise, color):
        """A small curved-arrow decal lying flat on a button's top face (the x/y plane), made of
        thin visual-only box segments along an arc plus a chevron arrowhead at the leading end --
        clockwise=True draws a clockwise-reading arrow (right / blue button), False a
        counter-clockwise one (left / red button), matching which way pressing that button tilts
        the active shelf.

        The CCW arrow is an open C (≈210 deg of arc) opening left; the CW arrow on the blue
        button is that shape's exact left-right mirror (negate x), so the two tips sit opposite
        each other (≈10 o'clock vs ≈2 o'clock) as true mirror images. A near-full circle used
        to make both arrows look identical from the demo camera.

        Returns a list of ``(entity, rest_xyz)`` so decals can ride the spring keycap.
        """
        cx, cy, cz = top_center_xyz
        cz += 0.0022  # sit just above the button's top face, clear of z-fighting

        n_pts = 9
        # 210 deg of arc => 150 deg gap: open enough that the mirrored C is unmistakable.
        angles_deg = np.linspace(-105.0, 105.0, n_pts)
        pts = [np.array([radius * np.cos(np.deg2rad(a)), radius * np.sin(np.deg2rad(a))]) for a in angles_deg]
        if clockwise:
            pts = [np.array([-p[0], p[1]]) for p in pts]

        seg_half_thick = radius * 0.11
        seg_half_h = 0.0009
        parts = []

        def _place_segment(p0, p1, half_thick):
            mid = (p0 + p1) / 2.0
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-6:
                return
            heading = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            quat = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], heading)
            entity = create_visual_box(
                self.scene, sapien.Pose([cx + mid[0], cy + mid[1], cz], quat),
                half_size=[seg_len / 2.0 + seg_half_thick, half_thick, seg_half_h],
                color=color, name="turn_arrow_seg",
            )
            wp = entity.get_pose()
            parts.append((entity, [float(wp.p[0]), float(wp.p[1]), float(wp.p[2])]))

        for p0, p1 in zip(pts[:-1], pts[1:]):
            _place_segment(p0, p1, seg_half_thick)

        tangent = pts[-1] - pts[-2]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        tip = pts[-1] + tangent * radius * 0.30
        head_len = radius * 0.55
        back = tip - tangent * head_len
        for side in (1.0, -1.0):
            wing = back + normal * side * head_len * 0.62
            _place_segment(wing, tip, seg_half_thick * 1.3)
        return parts

    def _get_rigid(self, entity):
        base_entity = entity.actor if hasattr(entity, "actor") else entity
        for component in base_entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    # --------------------------------------------------------- kinematic scene motion
    def _set_shelf_pose(self, idx: int, angle_deg: float, *, initial: bool = False):
        """Rotate shelf `idx` about its own centre (its y axis) so the +x (right) end dips when
        angle_deg > 0 and the -x (left) end dips when angle_deg < 0.

        After the first pose, drive with ``set_kinematic_target`` so PhysX sees a real surface
        velocity (``set_pose`` teleports a zero-vel kinematic and drags contacting marbles).
        """
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        phi = np.deg2rad(angle_deg)      # sign convention: +angle => right edge goes down
        quat = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], phi)
        pose = sapien.Pose([cx, self.maze_y, cz], quat)
        rigid = self._get_rigid(self.shelves[idx])
        if initial or rigid is None:
            self.shelves[idx].actor.set_pose(pose)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                self.shelves[idx].actor.set_pose(pose)
        self._shelf_cur_angle[idx] = float(angle_deg)

    def _max_tilt_deg_for_shelf(self, idx: int) -> float:
        """Largest |tilt| (deg) so the dipping underside of shelf `idx` stays above the surface
        below it (next shelf's top, or the table for the bottom shelf). Caps at ``tilt_angle_deg``.

        Matches `_set_shelf_pose`'s R_y convention: dipping-edge underside world-z is
        ``cz - L*sin(|phi|) - T*cos(|phi|)``.
        """
        cfg_max = float(abs(self.tilt_angle_deg))
        if idx < self.n_shelves - 1:
            lower_top = float(self.shelf_z[idx + 1] + self.shelf_half_thick)
        else:
            lower_top = float(self.table_z)
        clearance = float(self.tilt_clearance)
        cz = float(self.shelf_z[idx])
        L = float(self.shelf_half_len)
        T = float(self.shelf_half_thick)

        def clears(phi_deg: float) -> bool:
            phi = np.deg2rad(max(0.0, float(phi_deg)))
            underside_z = cz - L * np.sin(phi) - T * np.cos(phi)
            return bool(underside_z >= lower_top + clearance)

        if clears(cfg_max):
            return cfg_max
        # Binary search the largest angle that still clears the obstacle below.
        lo, hi = 0.0, cfg_max
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if clears(mid):
                lo = mid
            else:
                hi = mid
        return float(max(lo, 1.0))  # keep a tiny tilt so the marble still rolls off

    def _ball_pose_on_shelf(self, idx: int, angle_deg: float, local_x: float):
        """World position of the marble sitting ``local_x`` along shelf ``idx``'s top surface."""
        local_z = self.shelf_half_thick + self.ball_radius
        phi = -np.deg2rad(angle_deg)
        cphi, sphi = np.cos(phi), np.sin(phi)
        wx = local_x * cphi - local_z * sphi
        wz = local_x * sphi + local_z * cphi
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        return np.array([cx + wx, self.maze_y, cz + wz], dtype=np.float64)

    def _desired_slide_local_x(self, angle_deg: float, dir_sign: float, start_local_x: float = 0.0):
        """Lock-step along-shelf coordinate for the *increasing* tilt (never used to reverse)."""
        release_abs = abs(float(getattr(self, "_sliding_release_angle", 0.0))) or abs(self.tilt_angle_deg)
        frac = float(np.clip(abs(angle_deg) / max(release_abs, 1e-6), 0.0, 1.0))
        edge_x = dir_sign * self.shelf_half_len
        return float(start_local_x + (edge_x - start_local_x) * frac)

    def _shelf_ball_local_point(self, idx: int, angle_deg: float, dir_sign: float, start_local_x: float = 0.0):
        """World position while riding shelf `idx`; along-shelf x is monotonic toward the edge."""
        local_x = float(getattr(self, "_sliding_local_x", start_local_x))
        return self._ball_pose_on_shelf(idx, angle_deg, local_x)

    def _advance_sliding_local_x(self, angle_deg: float):
        """Move the scripted marble downhill with tilt; never slide it back toward the centre."""
        dir_sign = float(self._sliding_dir)
        desired = self._desired_slide_local_x(
            angle_deg, dir_sign, float(self._sliding_start_local_x)
        )
        cur = float(getattr(self, "_sliding_local_x", self._sliding_start_local_x))
        if dir_sign >= 0.0:
            cur = max(cur, desired)
            cur = min(cur, dir_sign * self.shelf_half_len)
        else:
            cur = min(cur, desired)
            cur = max(cur, dir_sign * self.shelf_half_len)
        self._sliding_local_x = float(cur)

    def _set_ball_friction(self, static_f: float, dynamic_f: float):
        if self._ball_rigid is None:
            return
        try:
            mat = sapien.physx.PhysxMaterial(
                static_friction=float(static_f),
                dynamic_friction=float(dynamic_f),
                restitution=0.0,
            )
            for shape in self._ball_rigid.get_collision_shapes():
                shape.set_physical_material(mat)
        except Exception:
            pass

    def _unglue_sliding_ball(self, idx: int):
        """Shelf is returning before release — keep along-shelf inertia, stop parenting."""
        if self._ball_rigid is None or str(self._ball_mode) != "sliding":
            return
        angle = float(self._shelf_cur_angle[idx])
        p = self._ball_pose_on_shelf(idx, angle, float(self._sliding_local_x))
        dt = max(float(self.scene.get_timestep()), 1e-6)
        last = getattr(self, "_sliding_last_p", None)
        if last is not None:
            v = (p - np.asarray(last, dtype=np.float64)) / dt
        else:
            v = np.array([float(self._sliding_dir) * self._effective_roll_off_speed(), 0.0, 0.0])
        speed = float(np.linalg.norm(v))
        if speed > 1.5:
            v = v * (1.5 / speed)
        try:
            self.ball.set_pose(sapien.Pose(p.tolist()))
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_velocity(v.tolist())
            omega_y = float(v[0]) / max(float(self.ball_radius), 1e-6)
            self._ball_rigid.set_angular_velocity([0.0, omega_y, 0.0])
            if self.continuous_ball_motion:
                self._ball_rigid.set_linear_damping(self.CONTINUOUS_LINEAR_DAMPING)
                self._ball_rigid.set_angular_damping(self.CONTINUOUS_ANGULAR_DAMPING)
            else:
                self._ball_rigid.set_linear_damping(0.05)
                self._ball_rigid.set_angular_damping(0.5)
            # Don't let contact with the untilting shelf reverse downhill speed.
            self._set_ball_friction(0.08, 0.05)
        except Exception:
            pass
        near_edge = abs(float(self._sliding_local_x)) >= (self.shelf_half_len * 0.72)
        toward_edge = float(v[0]) * float(self._sliding_dir) > 0.03
        if near_edge or toward_edge:
            self._ball_mode = "falling"
        else:
            self._ball_mode = "resting"
            self._sliding_shelf_idx = -1
            self._sliding_dir = 0
        self._sliding_last_p = None

    def _advance_shelf_tilts(self):
        dt = float(self.scene.get_timestep())
        speed = abs(self.tilt_angle_deg) / max(self.tilt_duration_sec, 1e-6)
        if self._interactive_controls_enabled():
            speed *= float(self.INTERACTIVE_TILT_SPEED_SCALE)
        step = speed * dt
        for idx in range(self.n_shelves):
            cur = self._shelf_cur_angle[idx]
            tgt = self._shelf_target_angle[idx]
            returning = abs(tgt) + 1e-3 < abs(cur)
            if (
                returning
                and idx == self._sliding_shelf_idx
                and self._ball_mode == "sliding"
            ):
                # Unglue at the current (pre-untilt) pose so velocity is downhill, not reversed.
                self._unglue_sliding_ball(idx)
            if abs(cur - tgt) <= 1e-3:
                # Already at target: if this is the sliding shelf holding at full tilt, release.
                if (
                    idx == self._sliding_shelf_idx
                    and self._ball_mode == "sliding"
                    and abs(tgt) >= 1e-3
                    and abs(cur) >= abs(self._sliding_release_angle) - 1e-3
                ):
                    self._release_ball(idx, self._sliding_dir)
                continue
            if cur < tgt:
                cur = min(cur + step, tgt)
            else:
                cur = max(cur - step, tgt)
            self._set_shelf_pose(idx, cur)
            if idx == self._sliding_shelf_idx and self._ball_mode == "sliding":
                self._advance_sliding_local_x(cur)
                p = self._ball_pose_on_shelf(idx, cur, float(self._sliding_local_x))
                self._sliding_last_p = p.copy()
                if self._ball_rigid is not None:
                    try:
                        self._ball_rigid.set_kinematic_target(sapien.Pose(p.tolist()))
                    except Exception:
                        self.ball.set_pose(sapien.Pose(p.tolist()))
                else:
                    self.ball.set_pose(sapien.Pose(p.tolist()))
                if abs(cur) >= abs(self._sliding_release_angle) - 1e-3:
                    self._release_ball(idx, self._sliding_dir)

    def _effective_roll_off_speed(self) -> float:
        speed = abs(float(self.roll_off_speed))
        if self.continuous_ball_motion:
            speed *= max(float(self.continuous_ball_speed_scale), 0.0)
        return speed

    def _mark_ball_missed(self):
        """Table / failed landing: leave the marble where physics put it — never snap onto a shelf."""
        if str(getattr(self, "_ball_mode", "")) == "missed":
            self.plan_success = False
            self.active_shelf_idx = -1
            return
        self._ball_mode = "missed"
        self.active_shelf_idx = -1
        self._sliding_shelf_idx = -1
        self._pending_active_shelf = None
        self._hold_tilt_shelf_idx = None
        # Abort remaining expert / policy motion immediately.
        self.plan_success = False
        if self._ball_rigid is None:
            return
        try:
            # Keep real dynamics on the table so it is visibly a miss, not a rescue.
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
        except Exception:
            pass

    def _ball_left_active_shelf(self) -> bool:
        """True when a resting Opt-1 marble has clearly fallen off its current shelf.

        Requires a real drop below the shelf top (not merely sitting near the rim) so a
        near-edge landing does not false-trigger while the expert is still reacting.
        """
        # While a drop is latched pending button-release, the marble already sits on the
        # next shelf — miss checks must use that shelf, not the still-active tilted one.
        pending = getattr(self, "_pending_active_shelf", None)
        idx = int(pending) if pending is not None else int(getattr(self, "active_shelf_idx", -1))
        if idx < 0 or self.ball is None or not self.shelves:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        cx = float(self.shelf_centers_x[idx])
        top_z = float(self.shelf_z[idx] + self.shelf_half_thick)
        # Past the shelf end and already falling (center below the supported surface band).
        off_end = abs(float(p[0]) - cx) > (self.shelf_half_len + self.ball_radius)
        falling = float(p[2]) < (top_z + self.ball_radius - 0.004)
        if off_end and falling:
            return True
        # Dropped well below the shelf plane (missed contact entirely).
        if float(p[2]) < (top_z - 0.012):
            return True
        return False

    def _watch_for_ball_miss(self):
        """Fail-fast: marble on the table (or Opt-1 roll-off) ends the episode immediately.

        Called every physics tick so a slow approach cannot keep going after the marble is lost.
        """
        if not getattr(self, "_loaded", False) or self.ball is None:
            return
        mode = str(getattr(self, "_ball_mode", ""))
        if mode in ("missed", "done"):
            return
        if self._ball_in_bowl():
            return
        # Any contact with the table outside the bowl is an immediate failure.
        if self._ball_on_table():
            self._mark_ball_missed()
            return
        # Opt 1: a slow roll-off while waiting for the next press is a miss. Leftover
        # downhill speed after a cancelled tilt is a real drop — switch to falling.
        if self.continuous_ball_motion and mode == "resting" and self._ball_left_active_shelf():
            speed = 0.0
            if self._ball_rigid is not None:
                try:
                    vel = self._ball_rigid.get_linear_velocity()
                    speed = float(np.linalg.norm(vel))
                except Exception:
                    speed = 0.0
            if speed > 0.04:
                self._ball_mode = "falling"
                if int(getattr(self, "_sliding_shelf_idx", -1)) < 0:
                    self._sliding_shelf_idx = int(getattr(self, "active_shelf_idx", -1))
                return
            self._mark_ball_missed()
            return
        if (not self.continuous_ball_motion) and mode == "resting" and self._ball_left_active_shelf():
            self._ball_mode = "falling"
            if int(getattr(self, "_sliding_shelf_idx", -1)) < 0:
                self._sliding_shelf_idx = int(getattr(self, "active_shelf_idx", -1))

    def _release_ball(self, idx: int, dir_sign: float):
        if self._ball_rigid is None or self._ball_mode != "sliding":
            return
        release_angle = float(self._sliding_release_angle)
        p = self._shelf_ball_local_point(idx, release_angle, dir_sign, self._sliding_start_local_x)
        try:
            self.ball.set_pose(sapien.Pose(p.tolist()))
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_velocity([dir_sign * self._effective_roll_off_speed(), 0.0, 0.0])
            self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
            if self.continuous_ball_motion:
                self._ball_rigid.set_linear_damping(self.CONTINUOUS_LINEAR_DAMPING)
                self._ball_rigid.set_angular_damping(self.CONTINUOUS_ANGULAR_DAMPING)
            else:
                self._ball_rigid.set_linear_damping(0.05)
                self._ball_rigid.set_angular_damping(0.5)
            self._set_ball_friction(0.08, 0.05)
        except Exception:
            pass
        self._ball_mode = "falling"
        self._latch_final_release_metric(idx)
        self._sliding_last_p = None
        # Expert path eases the shelf back immediately; interactive hold-to-tilt keeps it
        # tilted until the key is released (see `_sync_hold_tilt`). The marble is already
        # dynamic here, so untilt must not kinematically parent it back up the slope.
        if not self._interactive_controls_enabled():
            self._shelf_target_angle[idx] = 0.0

    def _freeze_ball_on_shelf(self, idx: int, *, advance_active: bool = True):
        if self._ball_rigid is None:
            return
        # Never "rescue" a table miss by freezing onto a shelf.
        if self._ball_on_table():
            self._mark_ball_missed()
            return
        if self.continuous_ball_motion:
            # Leave the marble where it settled (no centre snap), but kill residual velocity so
            # it does not keep rolling off while the arm approaches the next button. Dynamics
            # remain enabled — a later shove / tilt can still move it; fail-fast still applies.
            try:
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(False)
                self._ball_rigid.set_kinematic(False)
                self._ball_rigid.set_linear_damping(self.CONTINUOUS_LINEAR_DAMPING)
                self._ball_rigid.set_angular_damping(self.CONTINUOUS_ANGULAR_DAMPING)
                self._set_ball_friction(self.CONTINUOUS_STATIC_FRICTION, self.CONTINUOUS_DYNAMIC_FRICTION)
            except Exception:
                pass
        else:
            p = np.array([self.shelf_centers_x[idx], self.maze_y,
                          self.shelf_z[idx] + self.shelf_half_thick + self.ball_radius + 0.001])
            try:
                self.ball.set_pose(sapien.Pose(p.tolist()))
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._ball_mode = "resting"
        self._partial_deepest_shelf = max(
            int(getattr(self, "_partial_deepest_shelf", 0) or 0), int(idx)
        )
        if advance_active:
            self.active_shelf_idx = idx
            self._sliding_shelf_idx = -1
            self._sliding_dir = 0
            self._pending_active_shelf = None
            self._hold_tilt_shelf_idx = None
        else:
            # Park on the next shelf but keep the fired shelf "active" until key release.
            fired = int(getattr(self, "_sliding_shelf_idx", -1))
            if fired < 0:
                fired = int(getattr(self, "active_shelf_idx", -1))
            self._pending_active_shelf = int(idx)
            self._hold_tilt_shelf_idx = fired if fired >= 0 else None
            if fired >= 0:
                self.active_shelf_idx = fired
            self._sliding_shelf_idx = -1
            # Keep `_sliding_dir` / `_sliding_release_angle` so hold can latch the same tilt.

    def _ball_on_table(self) -> bool:
        """True when the marble is resting on / near the table and not inside the bowl.

        A miss that reaches the table is a failed episode — do not rescue by snapping the ball
        onto a lower shelf.
        """
        if self.ball is None or self._ball_in_bowl():
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        return bool(float(p[2]) <= self.table_z + self.ball_radius + 0.025)

    def _bowl_xy(self):
        """Live bowl center XY (tracks Opt 2 oscillation; static otherwise)."""
        if self.bowl is None:
            return np.array(self.bowl_center_xy, dtype=np.float64)
        p = self.bowl.get_pose().p
        return np.array([float(p[0]), float(p[1])], dtype=np.float64)

    def _ball_in_bowl(self) -> bool:
        if self.ball is None:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        bowl_xy = self._bowl_xy()
        self.bowl_center_xy = bowl_xy
        horiz = float(np.linalg.norm(p[:2] - bowl_xy))
        in_z = (self.table_z - 0.01) <= p[2] <= (self.table_z + self.bowl_height)
        return bool(horiz <= (self.bowl_catch_radius + self.ball_radius) and in_z)

    def _bowl_x_at(self, future_time: float = 0.0) -> float:
        """Predicted bowl x at now + future_time under the Opt 2 cosine sweep."""
        if not getattr(self, "osc_bowl_enabled", False):
            return float(self._bowl_xy()[0])
        dt = float(self.scene.get_timestep())
        t = float(self._bowl_steps) * dt + float(future_time)
        phase = float(getattr(self, "_bowl_phase0", 0.0)) + 2.0 * np.pi * t / max(self._bowl_period, 1e-6)
        return float(self._bowl_mid_x + self._bowl_amp * np.cos(phase))

    def _animate_oscillating_bowl(self):
        if not (getattr(self, "osc_bowl_enabled", False) and self.bowl is not None and self._bowl_armed):
            return
        x = self._bowl_x_at(0.0)
        pose = sapien.Pose([x, self._bowl_base_y, self._bowl_base_z], self._bowl_q)
        try:
            self.bowl.actor.set_pose(pose)
        except Exception:
            self.bowl.set_pose(pose)
        self.bowl_center_xy = np.array([x, self._bowl_base_y], dtype=np.float64)
        self._bowl_steps += 1

    def _final_drop_lead_time(self) -> float:
        """Seconds from button-press start until the marble roughly reaches bowl height."""
        fall_h = max(float(self.shelf_z[-1] - self.table_z), 0.02)
        fall_t = float(np.sqrt(2.0 * fall_h / self.GRAVITY))
        return float(self.tilt_duration_sec) + fall_t

    def _steps_until_bowl_alignment(self, drop_x: float, lead_time: float, max_steps: int = 2500) -> int:
        """Soonest step count such that bowl_x(now + lead + steps*dt) is under drop_x."""
        dt = float(self.scene.get_timestep())
        tol = float(self.bowl_catch_radius) * float(self.OSCILLATING_BOWL_ALIGN_TOL_FRAC)
        best_steps = 0
        best_err = float("inf")
        for step in range(max(0, int(max_steps)) + 1):
            err = abs(self._bowl_x_at(lead_time + step * dt) - float(drop_x))
            if err < best_err:
                best_err = err
                best_steps = step
            if err <= tol:
                return step
        return int(best_steps)

    def _choose_final_drop_direction(self, wait: bool = True) -> str:
        """Default: static bowl_side. Opt 2: pick the bottom-shelf edge the bowl will reach soonest.

        When ``wait`` is True, also dwell until that alignment (used if the caller does not time
        the press itself). ``play_once`` passes wait=False and times inside ``_press_button``
        after the arm has already approached the button; in that case an approach-time estimate
        is folded into the side-selection lead so we don't pick a side that was only soonest
        *before* the (multi-second) approach.
        """
        if not getattr(self, "osc_bowl_enabled", False):
            return self.bowl_side
        bottom_cx = float(self.shelf_centers_x[-1])
        left_x = bottom_cx - self.shelf_half_len
        right_x = bottom_cx + self.shelf_half_len
        # ~2.5 s covers a typical single-arm button approach; only used for side selection.
        approach_lead = 0.0 if wait else 2.5
        lead = self._final_drop_lead_time() + approach_lead
        steps_l = self._steps_until_bowl_alignment(left_x, lead)
        steps_r = self._steps_until_bowl_alignment(right_x, lead)
        if steps_l <= steps_r:
            if wait:
                self._dwell(steps_l)
            self.bowl_side = "left"
            return "left"
        if wait:
            self._dwell(steps_r)
        self.bowl_side = "right"
        return "right"

    def _interactive_controls_enabled(self) -> bool:
        return bool(
            getattr(self, "_interactive_universal_controls", False)
            or getattr(self, "_interactive_robot_mode", False)
        )

    def _button_engaged(self, bank, side: str) -> bool:
        """True while a key is held past trigger (with release hysteresis)."""
        if hasattr(bank, "is_engaged"):
            return bool(bank.is_engaged(side))
        return bool(bank.is_held(side))

    def _sync_button_arrow_decals(self):
        """Keep curved-arrow decals glued to the spring keycaps."""
        bank = getattr(self, "_reactive_buttons", None)
        arrows = getattr(self, "_button_arrows", None) or {}
        if bank is None or not arrows:
            return
        for side in ("left", "right"):
            try:
                depth = float(bank.visual_depth[bank.resolve_index(side)])
            except Exception:
                depth = 0.0
            for entity, rest_xyz in arrows.get(side, []) or []:
                try:
                    pose = entity.get_pose()
                    entity.set_pose(
                        sapien.Pose(
                            [rest_xyz[0], rest_xyz[1], rest_xyz[2] - depth],
                            list(pose.q),
                        )
                    )
                except Exception:
                    pass

    def _begin_hold_tilt(self, direction: str) -> bool:
        """Start (or keep) a non-blocking tilt of the active shelf toward ``direction``."""
        if self.active_shelf_idx < 0 or not self.plan_success:
            return False
        if str(getattr(self, "_ball_mode", "")) not in ("resting", "sliding"):
            return False
        if direction not in ("left", "right"):
            return False

        idx = int(self.active_shelf_idx)
        dir_sign = 1.0 if direction == "right" else -1.0
        max_tilt = self._max_tilt_deg_for_shelf(idx)
        release_angle = dir_sign * max_tilt

        # Already sliding the same way: just keep the target latched at full tilt.
        if (
            str(self._ball_mode) == "sliding"
            and int(self._sliding_shelf_idx) == idx
            and abs(float(self._sliding_dir) - dir_sign) < 1e-6
        ):
            self._shelf_target_angle[idx] = release_angle
            return True

        cur_p = self.ball.get_pose().p
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._sliding_start_local_x = float(cur_p[0]) - self.shelf_centers_x[idx]
        self._sliding_local_x = float(self._sliding_start_local_x)
        self._sliding_last_p = None
        self._sliding_release_angle = release_angle
        self._sliding_shelf_idx = idx
        self._sliding_dir = dir_sign
        self._ball_mode = "sliding"
        self._shelf_target_angle[idx] = release_angle
        # Any previously fired shelf should ease back while this one tilts.
        for i in range(self.n_shelves):
            if i != idx:
                self._shelf_target_angle[i] = 0.0
        return True

    def _restore_ball_after_cancelled_tilt(self, idx: int):
        """Shelf returned to flat before release — park the marble and resume resting."""
        if self.ball is None or self._ball_rigid is None:
            return
        cx = float(self.shelf_centers_x[idx])
        if self.continuous_ball_motion:
            cur_p = np.array(self.ball.get_pose().p, dtype=np.float64)
            local_x = float(np.clip(cur_p[0] - cx, -self.shelf_half_len, self.shelf_half_len))
            p = np.array(
                [cx + local_x, self.maze_y,
                 self.shelf_z[idx] + self.shelf_half_thick + self.ball_radius + 0.001],
                dtype=np.float64,
            )
            try:
                self.ball.set_pose(sapien.Pose(p.tolist()))
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(False)
                self._ball_rigid.set_kinematic(False)
                self._ball_rigid.set_linear_damping(self.CONTINUOUS_LINEAR_DAMPING)
                self._ball_rigid.set_angular_damping(self.CONTINUOUS_ANGULAR_DAMPING)
            except Exception:
                pass
            self.active_shelf_idx = idx
            self._ball_mode = "resting"
            self._sliding_shelf_idx = -1
            self._sliding_dir = 0
        else:
            self._freeze_ball_on_shelf(idx)

    def _promote_pending_active_shelf(self):
        """On button release: hand control to the shelf the marble already landed on."""
        pending = getattr(self, "_pending_active_shelf", None)
        if pending is None:
            self._hold_tilt_shelf_idx = None
            return
        self.active_shelf_idx = int(pending)
        self._pending_active_shelf = None
        self._hold_tilt_shelf_idx = None
        self._sliding_dir = 0

    def _keep_hold_tilt_latched(self, held_dir):
        """While held after a drop (or mid-fall), keep the fired shelf at its release angle."""
        lock = getattr(self, "_hold_tilt_shelf_idx", None)
        if lock is None:
            lock = int(getattr(self, "_sliding_shelf_idx", -1))
        if lock is None or int(lock) < 0:
            return
        lock = int(lock)
        if held_dir in ("left", "right"):
            want = 1.0 if held_dir == "right" else -1.0
            if abs(float(getattr(self, "_sliding_dir", 0.0)) - want) < 1e-6:
                self._shelf_target_angle[lock] = float(self._sliding_release_angle)
                return
        self._shelf_target_angle[lock] = 0.0

    def _resolve_hold_tilt_ball(self):
        """Non-blocking landing / cancel / bowl resolution for interactive hold-to-tilt."""
        mode = str(getattr(self, "_ball_mode", ""))
        if mode == "sliding":
            idx = int(getattr(self, "_sliding_shelf_idx", -1))
            if (
                idx >= 0
                and abs(float(self._shelf_target_angle[idx])) < 1e-3
                and abs(float(self._shelf_cur_angle[idx])) < 1e-2
            ):
                self._unglue_sliding_ball(idx)
            return
        if mode != "falling":
            return
        if self._ball_in_bowl():
            self._ball_mode = "done"
            self.active_shelf_idx = -1
            self._pending_active_shelf = None
            self._hold_tilt_shelf_idx = None
            return
        if self._ball_on_table():
            self._mark_ball_missed()
            return
        from_idx = int(getattr(self, "_sliding_shelf_idx", -1))
        if from_idx < 0 or from_idx >= self.n_shelves - 1:
            return
        landed_idx = self._locate_landed_shelf(from_idx)
        if landed_idx is None or self._ball_on_table():
            return
        # Hold still down: park on next shelf but keep the fired shelf active until release.
        defer = getattr(self, "_hold_dir", None) in ("left", "right")
        self._freeze_ball_on_shelf(landed_idx, advance_active=not defer)

    def _sync_hold_tilt(self, held_dir):
        """Drive active-shelf target from the currently held push button (or release)."""
        mode = str(getattr(self, "_ball_mode", ""))
        idx = int(getattr(self, "active_shelf_idx", -1))
        sliding_idx = int(getattr(self, "_sliding_shelf_idx", -1))
        pending = getattr(self, "_pending_active_shelf", None)

        if mode in ("missed", "done"):
            for i in range(self.n_shelves):
                self._shelf_target_angle[i] = 0.0
            self._pending_active_shelf = None
            self._hold_tilt_shelf_idx = None
            return

        if mode == "falling":
            # Keep the fired shelf tilted while the key stays held; ease back on release.
            if sliding_idx >= 0:
                self._hold_tilt_shelf_idx = sliding_idx
            self._keep_hold_tilt_latched(held_dir)
            return

        # Released: flatten every shelf, then (if marble already landed) promote next level.
        if held_dir is None:
            for i in range(self.n_shelves):
                self._shelf_target_angle[i] = 0.0
            self._promote_pending_active_shelf()
            return

        # Still holding after a deferred landing: keep last shelf active/tilted — do not
        # start tilting the next level until the key is released.
        if pending is not None and mode == "resting":
            self._keep_hold_tilt_latched(held_dir)
            return

        if held_dir in ("left", "right") and mode in ("resting", "sliding") and idx >= 0:
            self._begin_hold_tilt(held_dir)
            return

        # Fallback: ease active / sliding shelf flat.
        if mode == "sliding" and sliding_idx >= 0:
            self._shelf_target_angle[sliding_idx] = 0.0
        elif idx >= 0:
            self._shelf_target_angle[idx] = 0.0

    def _update_reactive_buttons(self):
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        expert = getattr(self, "_expert_hold", None)
        for side in ("left", "right"):
            try:
                bank.set_forced(side, expert == side)
            except Exception:
                pass
        bank.update()
        self._sync_button_arrow_decals()

        if not self._interactive_controls_enabled():
            self._hold_dir = None
            return

        left_on = self._button_engaged(bank, "left")
        right_on = self._button_engaged(bank, "right")
        if left_on and not right_on:
            held = "left"
        elif right_on and not left_on:
            held = "right"
        else:
            held = None
        self._hold_dir = held
        self._sync_hold_tilt(held)

    def consume_pending_tilt(self) -> bool:
        """Deprecated: interactive hold-to-tilt no longer queues blocking presses."""
        self._pending_tilt_dir = None
        return False

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._update_reactive_buttons()
        self._advance_shelf_tilts()
        if self._interactive_controls_enabled():
            self._resolve_hold_tilt_ball()
        self._animate_oscillating_bowl()
        self._track_maze_metrics()
        # Catch Opt-1 roll-offs / table hits from the *previous* step so the next control
        # iteration can abort (also see take_dense_action / _dwell post-step watches).
        self._watch_for_ball_miss()

    # --------------------------------------------------------- press / dwell helpers
    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            if str(getattr(self, "_ball_mode", "")) == "missed":
                self.plan_success = False
                return
            self._update_kinematic_tasks()
            self.scene.step()
            self._watch_for_ball_miss()
            if str(getattr(self, "_ball_mode", "")) == "missed":
                self.plan_success = False
                return
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def take_dense_action(self, control_seq, save_freq=-1):
        """Same as base, but abort mid-trajectory as soon as the marble is missed (Opt 1)."""
        if str(getattr(self, "_ball_mode", "")) == "missed":
            self.plan_success = False
            return False

        class _MarbleMissAbort(Exception):
            pass

        orig_step = self.scene.step

        def _step_and_watch():
            orig_step()
            self._watch_for_ball_miss()
            if str(getattr(self, "_ball_mode", "")) == "missed":
                raise _MarbleMissAbort()

        self.scene.step = _step_and_watch
        try:
            return super().take_dense_action(control_seq, save_freq=save_freq)
        except _MarbleMissAbort:
            self.plan_success = False
            return False
        finally:
            self.scene.step = orig_step

    def take_action(self, action, action_type="qpos"):
        """Policy step: stop immediately once the marble has been missed."""
        if str(getattr(self, "_ball_mode", "")) == "missed":
            return

        class _MarbleMissAbort(Exception):
            pass

        orig_step = self.scene.step

        def _step_and_watch():
            orig_step()
            self._watch_for_ball_miss()
            if str(getattr(self, "_ball_mode", "")) == "missed":
                raise _MarbleMissAbort()

        self.scene.step = _step_and_watch
        try:
            return super().take_action(action, action_type=action_type)
        except _MarbleMissAbort:
            return
        finally:
            self.scene.step = orig_step

    def _update_transient_checks(self):
        self._watch_for_ball_miss()
        super()._update_transient_checks()

    def _tilt_active_shelf_and_wait(self, direction: str):
        """Kick off the active shelf's tilt toward `direction` (rolling the marble off it), then
        dwell fixed step counts (deterministic across the plan/render passes) for the fall+settle.

        Tilt magnitude is capped so the dipping shelf cannot penetrate the shelf (or table) below.
        If the marble misses the next shelf and reaches the table, the episode is marked missed —
        the ball is never teleported onto a lower shelf.
        """
        idx = self.active_shelf_idx
        dir_sign = 1.0 if direction == "right" else -1.0

        # Snapshot wherever the marble actually is right now (its centre, unless continuous_ball_motion
        # left it settled off-centre) and lock it kinematic -- from here the slide-to-the-edge is
        # always a deterministic, scripted sweep in lock-step with the tilt.
        cur_p = self.ball.get_pose().p
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._sliding_start_local_x = float(cur_p[0]) - self.shelf_centers_x[idx]
        self._sliding_local_x = float(self._sliding_start_local_x)
        self._sliding_last_p = None

        max_tilt = self._max_tilt_deg_for_shelf(idx)
        self._sliding_release_angle = dir_sign * max_tilt
        self._sliding_shelf_idx = idx
        self._sliding_dir = dir_sign
        self._ball_mode = "sliding"
        self._shelf_target_angle[idx] = self._sliding_release_angle

        # Dwell long enough to reach the (possibly reduced) target at the configured tilt speed.
        tilt_frac = max_tilt / max(abs(self.tilt_angle_deg), 1e-6)
        tilt_steps = int(np.ceil(self.tilt_duration_sec * tilt_frac * self.scene.get_timestep() ** -1)) + 4
        self._dwell(tilt_steps)   # sweeps the shelf to full allowed tilt and releases the marble
        is_last = idx >= self.n_shelves - 1
        if is_last:
            for i in range(max(0, int(self.final_fall_settle_steps))):
                self._update_kinematic_tasks()
                self.scene.step()
                if self.save_freq and (i % self.save_freq == 0):
                    self._take_picture()
                if self._ball_mode == "falling" and self._ball_in_bowl():
                    self._ball_mode = "done"
                    self.active_shelf_idx = -1
                    break
                if self._ball_mode == "falling" and self._ball_on_table():
                    self._mark_ball_missed()
                    break
            if self._ball_mode == "falling":
                # Timed out still falling — treat table miss / bowl miss as done.
                if self._ball_on_table() and not self._ball_in_bowl():
                    self._mark_ball_missed()
                else:
                    self._ball_mode = "done"
                    self.active_shelf_idx = -1
            return

        # Per-level vertical gaps are now randomized (see load_actors), so a gap that landed larger
        # than the nominal level_gap needs proportionally more time for the marble to actually reach
        # the shelf below -- fall time scales with sqrt(distance), so scale the dwell budget the same
        # way (only ever upward from the tuned base, never below it).
        gap = self.level_gaps[idx] if idx < len(self.level_gaps) else self._level_gap_nominal
        ratio = gap / max(self._level_gap_nominal, 1e-6)
        settle_steps = self.fall_settle_steps
        if ratio > 1.0:
            settle_steps = int(np.ceil(self.fall_settle_steps * np.sqrt(ratio))) + 10

        # Check for a landing every step. Only the immediate next shelf is a valid landing —
        # skipping ahead / rescuing onto a lower shelf looked like a teleport. Hitting the table
        # is an immediate miss.
        landed_idx = None
        for i in range(max(0, int(settle_steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._ball_mode != "falling":
                break
            if self._ball_on_table():
                self._mark_ball_missed()
                landed_idx = None
                break
            landed_idx = self._locate_landed_shelf(idx)
            if landed_idx is not None:
                break
        if landed_idx is not None and not self._ball_on_table():
            self._freeze_ball_on_shelf(landed_idx)
        elif self._ball_mode == "falling":
            # Never arrived on the next shelf within the settle budget — do not teleport; fail.
            self._mark_ball_missed()
        # If already missed (table), leave the marble where it is.

    def _press_tilt_direct(self, direction: str) -> bool:
        """Tilt the active shelf without arm motion (keyboard / sandbox)."""
        if self.active_shelf_idx < 0 or not self.plan_success:
            return False
        if str(getattr(self, "_ball_mode", "")) in ("missed", "sliding", "falling", "done"):
            return False
        if direction not in ("left", "right"):
            return False
        self._tilt_active_shelf_and_wait(direction)
        return str(getattr(self, "_ball_mode", "")) != "missed"

    def _locate_landed_shelf(self, from_idx: int):
        """After a fall from shelf `from_idx`, accept a landing only on the immediate next shelf.

        Requires the marble to be within that shelf's x-span, near its top surface, and moving
        slowly (settled contact, not a mid-air transit). Deliberately does *not* scan further
        shelves — catching the ball two levels down after a miss looked like a teleport and
        masked real failures onto the table.
        """
        if self.ball is None:
            return None
        next_idx = from_idx + 1
        if next_idx >= self.n_shelves:
            return None
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        # Already below the next shelf's top with no contact → miss in progress; don't claim a land.
        next_top = self.shelf_z[next_idx] + self.shelf_half_thick
        if float(p[2]) < next_top - 0.01:
            return None
        vz = 0.0
        vx = 0.0
        if self._ball_rigid is not None:
            try:
                vel = self._ball_rigid.get_linear_velocity()
                vx = float(vel[0])
                vz = float(vel[2])
            except Exception:
                vx, vz = 0.0, 0.0
        if abs(vz) > 0.35 or abs(vx) > 0.45:
            return None
        cx = self.shelf_centers_x[next_idx]
        top_z = next_top
        in_x = abs(p[0] - cx) <= self.shelf_half_len * 0.98
        in_z = abs(p[2] - (top_z + self.ball_radius)) <= max(0.012, 1.5 * self.ball_radius)
        if in_x and in_z:
            return next_idx
        return None

    def _press_button(self, arm_tag: ArmTag, direction: str, time_bowl: bool = False):
        if not self.plan_success or self.active_shelf_idx < 0:
            return
        if str(getattr(self, "_ball_mode", "")) == "missed":
            return
        btn = self.right_button if direction == "right" else self.left_button
        self.move(
            self.grasp_actor(
                btn, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.09,
                contact_point_id=0, gripper_pos=0.5,
            )
        )
        if not self.plan_success or str(getattr(self, "_ball_mode", "")) == "missed":
            return
        # Opt 2: approach first, then wait so tilt+fall lead time is measured from the press,
        # not from the (variable-duration) motion-plan approach.
        if time_bowl and self.osc_bowl_enabled:
            bottom_cx = float(self.shelf_centers_x[-1])
            drop_x = bottom_cx + (
                self.shelf_half_len if direction == "right" else -self.shelf_half_len
            )
            wait = self._steps_until_bowl_alignment(drop_x, self._final_drop_lead_time())
            self._dwell(wait)
            if not self.plan_success or str(getattr(self, "_ball_mode", "")) == "missed":
                return
        self.move(self.move_by_displacement(arm_tag, z=-self.button_press_depth))
        if not self.plan_success or str(getattr(self, "_ball_mode", "")) == "missed":
            return
        self._tilt_active_shelf_and_wait(direction)
        if str(getattr(self, "_ball_mode", "")) == "missed":
            return
        self._dwell(self.press_hold_steps)
        if not self.plan_success or str(getattr(self, "_ball_mode", "")) == "missed":
            return
        self.move(self.move_by_displacement(arm_tag, z=self.button_press_depth + 0.01))
        if not self.plan_success or str(getattr(self, "_ball_mode", "")) == "missed":
            return
        # Opt 1: skip the long retreat-to-origin between presses so the next button press
        # happens before the continuous marble can roll off the shelf.
        if not self.continuous_ball_motion:
            self.move(self.back_to_origin(arm_tag))
            self._dwell(self.post_press_dwell)
        else:
            self._dwell(max(2, self.post_press_dwell // 2))

    # ------------------------------------------------------------- policy
    def play_once(self):
        # Arm Opt 2 bowl motion for the whole episode (including inter-shelf presses) so the
        # phase at the final drop is consistent with the continuous oscillation the cameras see.
        self._bowl_armed = bool(self.osc_bowl_enabled)
        presses_made = []
        # Drive off self.active_shelf_idx (where the marble actually is right now) rather than a
        # loop counter. A miss (ball reaches the table, or Opt-1 roll-off / failed next-shelf
        # landing) sets active_shelf_idx to -1 and ball_mode to "missed" — no teleport onto a
        # lower shelf — and plan_success=False so the episode stops immediately.
        max_presses = self.n_shelves + 2   # generous cap so a pathological loop can't run forever
        for _ in range(max_presses):
            self._watch_for_ball_miss()
            if self.active_shelf_idx < 0 or not self.plan_success:
                break
            if self._ball_mode == "missed":
                break
            idx = self.active_shelf_idx
            is_final = idx >= self.n_shelves - 1
            if is_final:
                direction = self._choose_final_drop_direction(wait=False)
            else:
                direction = self.correct_dir[idx]
            arm_tag = ArmTag("right" if direction == "right" else "left")
            self._press_button(arm_tag, direction, time_bowl=is_final)
            presses_made.append(direction)
            if self._ball_mode == "missed" or not self.plan_success:
                break

        self.info["info"] = {
            "{A}": "marble",
            "{B}": "shelves",
            "{C}": "bowl",
            "{D}": "buttons",
            "{a}": "left arm",
            "{b}": "right arm",
            "{o}": self._option_label(),
        }
        self.info["presses_made"] = presses_made
        self.info["option_label"] = self._option_label()
        self.info["continuous_ball_motion"] = bool(self.continuous_ball_motion)
        self.info["oscillating_bowl_enabled"] = bool(self.osc_bowl_enabled)
        return self.info

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_final_ready_step = None    # marble parked on the bottom shelf
        self._metric_final_release_step = None  # bottom shelf fired
        self._metric_bowl_step = None           # marble first inside the bowl
        self._metric_final_falling = False
        self._metric_min_bowl_offset = None     # closest horizontal approach, metres

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_final_release_metric(self, idx: int):
        """Called from _release_ball when the *bottom* shelf fires."""
        try:
            if int(idx) < int(self.n_shelves) - 1:
                return
            if getattr(self, "_metric_final_release_step", None) is None:
                self._metric_final_release_step = self._metric_step()
            self._metric_final_falling = True
        except Exception:
            pass

    def _track_maze_metrics(self):
        """Per-step latches: bottom-shelf arrival, closest bowl approach, bowl entry."""
        try:
            mode = str(getattr(self, "_ball_mode", ""))
            last = int(self.n_shelves) - 1
            if (
                getattr(self, "_metric_final_ready_step", None) is None
                and mode == "resting"
                and int(getattr(self, "active_shelf_idx", -1)) == last
            ):
                self._metric_final_ready_step = self._metric_step()
            if not getattr(self, "_metric_final_falling", False) or self.ball is None:
                return
            p = np.array(self.ball.get_pose().p, dtype=np.float64)
            # Only score the descent below the shelf stack — the slide along the shelf
            # itself passes over the bowl and would fake a perfect approach.
            if float(p[2]) <= float(self.shelf_z[last]):
                horiz = float(np.linalg.norm(p[:2] - self._bowl_xy()))
                prev = getattr(self, "_metric_min_bowl_offset", None)
                if prev is None or horiz < prev:
                    self._metric_min_bowl_offset = horiz
            if getattr(self, "_metric_bowl_step", None) is None and self._ball_in_bowl():
                self._metric_bowl_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `final_drop_latency_steps` — steps from the marble coming to rest on the
        bottom shelf until that shelf is fired. Under Opt 2 the bowl sweeps left/right,
        so this window is exactly the operator's timing decision.
        extra2 `bowl_offset_norm` — closest horizontal marble-to-bowl-centre distance
        during the final descent, divided by the catch radius. LOWER is better; <= 1.0
        means it fell within the bowl mouth.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        ready = getattr(self, "_metric_final_ready_step", None)
        rel = getattr(self, "_metric_final_release_step", None)
        lat = None if (ready is None or rel is None) else max(int(rel) - int(ready), 0)
        out["final_drop_latency_steps"] = lat
        out["final_drop_latency_s"] = None if lat is None else round(lat * dt, 4)

        bowl = getattr(self, "_metric_bowl_step", None)
        fall = None if (rel is None or bowl is None) else max(int(bowl) - int(rel), 0)
        out["fall_latency_steps"] = fall
        out["fall_latency_s"] = None if fall is None else round(fall * dt, 4)

        off = getattr(self, "_metric_min_bowl_offset", None)
        try:
            denom = float(self.bowl_catch_radius) + float(self.ball_radius)
            out["bowl_offset_norm"] = None if off is None else round(float(off) / max(denom, 1e-9), 4)
        except Exception:
            out["bowl_offset_norm"] = None
        return out

    # ----------------------------------------------------------- metric/obs
    def check_success(self):
        in_bowl = self._ball_in_bowl()
        on_table = self._ball_on_table()
        missed = str(self._ball_mode) == "missed" or (on_table and not in_bowl)
        self.info["n_shelves"] = int(self.n_shelves)
        self.info["bowl_side"] = self.bowl_side
        self.info["correct_dir"] = list(self.correct_dir)
        self.info["active_shelf_idx"] = int(self.active_shelf_idx)
        self.info["ball_mode"] = str(self._ball_mode)
        self.info["ball_position"] = list(map(float, self.ball.get_pose().p)) if self.ball is not None else []
        self.info["ball_in_bowl"] = bool(in_bowl)
        self.info["ball_on_table"] = bool(on_table)
        self.info["ball_missed"] = bool(missed)
        self.info["continuous_ball_motion"] = bool(self.continuous_ball_motion)
        self.info["oscillating_bowl_enabled"] = bool(self.osc_bowl_enabled)
        self.info["option_label"] = self._option_label()
        # Success only if the marble is in the bowl; table contact without bowl = fail.
        return bool(in_bowl and not missed)

    def get_obs(self):
        obs = super().get_obs()
        obs["shelf_maze"] = {
            "n_shelves": int(self.n_shelves),
            "shelf_centers_x": list(map(float, self.shelf_centers_x)),
            "shelf_z": list(map(float, self.shelf_z)),
            "shelf_angles_deg": list(map(float, self._shelf_cur_angle)),
            "active_shelf_idx": int(self.active_shelf_idx),
            "ball_mode": str(self._ball_mode),
            "ball_position": list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0],
            "bowl_side": str(self.bowl_side),
            "bowl_position": list(map(float, self._bowl_xy())),
            "correct_dir": list(self.correct_dir),
            "ball_in_bowl": bool(self._ball_in_bowl()),
            "ball_on_table": bool(self._ball_on_table()),
            "continuous_ball_motion": bool(self.continuous_ball_motion),
            "oscillating_bowl_enabled": bool(self.osc_bowl_enabled),
            "option_label": self._option_label(),
        }
        return obs
