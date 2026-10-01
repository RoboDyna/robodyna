from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import sapien
import sapien.render
import sapien.physx
import numpy as np
import transforms3d as t3d


class hit_target(Base_Task):
    """Single-arm dynamic-intercept task (left or right).

    A round target board sways on the stick's half of the table (mirror about x=0).
    A dart ("stick") spawns on a random side of the table midline (x=0); that side's
    arm grasps it, leads the target's motion, and drives the tip into the yellow center.

    Left/right: each episode samples ``dart_side ∈ {−1, +1}`` (or config ``arm_side``).
    Left (−x) is the exact mirror of right (+x) about the table center plane x=0
    (dart pose, tip orientation, which arm acts, and the target / dynamic-blocker
    travel band). Target and moving blocker stay on the stick's half of the table
    (right → x∈[0, span], left → x∈[−span, 0]); the static blocker stays on the midline.

    Task options (independent toggles in ``task_args.hit_target``):
      - Opt 1 — static blocker: ``blocker_enabled`` (green disc in front of the target)
      - Opt 2 — dynamic blocker: ``blocker_dynamic`` (red disc that sways like the target)
      - Opt 1+2: both blockers; the dynamic (red) disc sits ``dual_blocker_gap`` in front of static

    Per episode, the gap from the target to the (first / static) blocker is sampled in
    [blocker_y_gap_min, blocker_y_gap_max] (default 5–8 cm). Opt 1+2 keeps the same
    sampled static gap and places the red disc ``dual_blocker_gap`` further in front.

    Success: tip contacts the yellow center AND the stick never hits a blocker.
    Stick–blocker contact is an immediate failure (blockers have solid collision).

    No external asset: dart, board, and blockers are SAPIEN primitives. Reserved-range
    asset ids [340,349] are documented in the NOTICE; nothing is written to assets/objects.
    """

    # ----- target geometry (flat round board standing up, facing -Y toward the robot)
    BOARD_RADIUS_DEFAULT = 0.075
    CENTER_RADIUS_DEFAULT = 0.018
    N_RINGS_DEFAULT = 4
    BOARD_THICKNESS = 0.012
    FACE_ROT_Q = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]  # cylinder axis +X -> +Y
    RING_COLORS = [
        [0.85, 0.10, 0.10],          # outer red
        [0.95, 0.95, 0.95],          # white
        [0.15, 0.35, 0.80],          # blue
        [0.90, 0.80, 0.20],          # yellow center
    ]
    RING_COLOR_NAMES = ["red", "white", "blue", "yellow"]

    # ----- blockers (Opt 1 = static green, Opt 2 = dynamic red)
    BLOCKER_ENABLED_DEFAULT = False       # Opt 1
    BLOCKER_DYNAMIC_DEFAULT = False       # Opt 2
    BLOCKER_RADIUS_DEFAULT = 0.028        # 30% smaller diameter than original 0.040
    BLOCKER_THICKNESS_DEFAULT = 0.002
    BLOCKER_Y_GAP_DEFAULT = 0.065         # nominal; per-ep sample in [min, max]
    BLOCKER_Y_GAP_MIN_DEFAULT = 0.050     # Opt1 / Opt2 / Opt1+2 static: m from target
    BLOCKER_Y_GAP_MAX_DEFAULT = 0.080
    DUAL_BLOCKER_GAP_DEFAULT = 0.010      # Opt1+2: dynamic in front of static (m)
    BLOCKER_Z_OFFSET_DEFAULT = 0.0        # coplanar with target center (same z axis)
    BLOCKER_X_OFFSET_DEFAULT = 0.0
    BLOCKER_SPEED_DEFAULT = 0.65          # match target
    STATIC_BLOCKER_COLOR = [0.15, 0.72, 0.28]   # green
    DYNAMIC_BLOCKER_COLOR = [0.85, 0.12, 0.12]  # red
    BLOCKER_CLEARANCE_Z = 0.015
    TARGET_Y_OFFSET_DEFAULT = 0.050

    # ----- dart geometry (scale=[1,1,1]; matrices are in meters; original stick size)
    DART_SHAFT_HALF = [0.045, 0.008, 0.008]   # half extents of the grip shaft (long axis = local X)
    DART_TIP_HALF = [0.012, 0.004, 0.004]     # half extents of the pointed tip
    DART_COLOR = [0.20, 0.85, 0.55]

    # ----- target motion (step-driven, applied in _update_kinematic_tasks)
    SWAY_AMP_DEFAULT = 0.243              # +0.10 m outer vs 0.143 (right→+x / left→−x)
    SWAY_PERIOD_DEFAULT = 900
    TARGET_SPEED_DEFAULT = 0.65           # base mult; per-ep ×U(1±speed_jitter)
    SPEED_JITTER_FRAC_DEFAULT = 0.20      # ±20% of base speed
    MOTION_X_MIN_DEFAULT = -0.272         # +0.10 m outer vs ±0.172
    MOTION_X_MAX_DEFAULT = 0.272
    TARGET_CENTER_X_DEFAULT = 0.0

    # ----- painted face (rings are thin visual discs stacked in front of the collider)
    RING_VISUAL_GAP = 0.0008     # per-ring y stagger, outer ring first
    RING_VISUAL_HALF = 0.001     # half thickness of one ring disc
    FACE_VISUAL_GAP = 0.001      # outer ring floats this far ahead of the collider

    # ----- contact / stick
    # The dart tip stops on the collider face, which sits ~4 mm behind the painted
    # surface, so a touch is judged against the paint and allowed this much slack
    # for IK undershoot. The weld then plants the tip flush with the paint.
    TOUCH_GAP = 0.005
    # Latching which color was struck is stricter than the stick test: a fly-over
    # that never reaches the paint must not be reported as a ring hit.
    TOUCH_LATCH_GAP = 0.002
    STICK_DIST = 0.035 / 3.0     # legacy approach band (kept for external callers)

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("hit_target", {})
        # Initialize per-step state BEFORE base setup: _init_task_env_ calls
        # _update_kinematic_tasks() during scene construction (before load_actors).
        self._step_count = 0
        self._stuck = False
        self._hit_center = False
        self._hit_blocker = False
        self._hit_color = None  # "yellow" / "blue" / "white" / "red" when tip meets the board
        self._hit_ring_index = None  # 0 = outermost ring … n-1 = bullseye
        self._hit_planar_offset = None
        self._hit_radial_offset = None
        self._reset_metric_state()
        # Partial credit for the ring stabbed: bullseye 1.0 down to 1/n on the
        # outer ring, 0.0 for a miss or a blocker strike. Success stays binary.
        self.hit_score = 0.0
        self._target_rigid = None
        self._dart_rigid = None
        self._static_blocker_rigid = None
        self._dynamic_blocker_rigid = None
        self.static_blocker = None
        self.dynamic_blocker = None
        self.blocker = None  # compat alias: frontmost blocker if any
        self.blocker_enabled = False
        self.blocker_dynamic = False
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.board_radius = float(
            cfg.get("board_radius", cfg.get("board_half_size", cfg.get("r_outer", self.BOARD_RADIUS_DEFAULT)))
        )
        self.center_radius = float(
            cfg.get("center_radius", cfg.get("center_half_size", self.CENTER_RADIUS_DEFAULT))
        )
        self.center_radius = min(self.center_radius, self.board_radius * 0.25)
        self.n_rings = max(1, int(cfg.get("n_rings", self.N_RINGS_DEFAULT)))
        self.sway_amp = float(cfg.get("sway_amp", self.SWAY_AMP_DEFAULT))
        self.sway_period = int(cfg.get("sway_period", self.SWAY_PERIOD_DEFAULT))
        self.target_speed_base = float(cfg.get("target_speed", self.TARGET_SPEED_DEFAULT))
        self.blocker_speed_base = float(cfg.get("blocker_speed", self.BLOCKER_SPEED_DEFAULT))
        self.speed_jitter_frac = abs(float(cfg.get("speed_jitter_frac", self.SPEED_JITTER_FRAC_DEFAULT)))
        self.motion_x_min = float(cfg.get("motion_x_min", self.MOTION_X_MIN_DEFAULT))
        self.motion_x_max = float(cfg.get("motion_x_max", self.MOTION_X_MAX_DEFAULT))
        if self.motion_x_min > self.motion_x_max:
            self.motion_x_min, self.motion_x_max = self.motion_x_max, self.motion_x_min
        # Nominal path center from config (overridden per-side after dart_side is chosen).
        self.target_center_x = float(cfg.get("target_center_x", self.TARGET_CENTER_X_DEFAULT))

        # Opt 1 / Opt 2 — independent toggles (both → static green + dynamic red).
        self.blocker_enabled = bool(cfg.get("blocker_enabled", self.BLOCKER_ENABLED_DEFAULT))  # Opt 1
        self.blocker_dynamic = bool(cfg.get("blocker_dynamic", self.BLOCKER_DYNAMIC_DEFAULT))  # Opt 2
        self.blocker_radius = float(cfg.get("blocker_radius", self.BLOCKER_RADIUS_DEFAULT))
        self.randomize_blocker_radius = bool(cfg.get("randomize_blocker_radius", False))
        br_jitter = float(np.clip(abs(float(cfg.get("blocker_radius_jitter", 0.10))), 0.0, 0.95))
        if self.randomize_blocker_radius and br_jitter > 0.0 and (
            self.blocker_enabled or self.blocker_dynamic
        ):
            self.blocker_radius = float(np.random.uniform(
                self.blocker_radius * (1.0 - br_jitter),
                self.blocker_radius * (1.0 + br_jitter),
            ))
        self.blocker_radius = min(self.blocker_radius, self.board_radius * 0.95)
        self.blocker_thickness = float(cfg.get("blocker_thickness", self.BLOCKER_THICKNESS_DEFAULT))
        gap_min = abs(float(cfg.get("blocker_y_gap_min", self.BLOCKER_Y_GAP_MIN_DEFAULT)))
        gap_max = abs(float(cfg.get("blocker_y_gap_max", self.BLOCKER_Y_GAP_MAX_DEFAULT)))
        if gap_min > gap_max:
            gap_min, gap_max = gap_max, gap_min
        # Per-episode distance from target to the (first / static) blocker: U[5cm, 8cm].
        # Config ``blocker_y_gap`` is only a fallback nominal if min/max are absent.
        if self.blocker_enabled or self.blocker_dynamic:
            self.blocker_y_gap = float(np.random.uniform(gap_min, gap_max))
        else:
            self.blocker_y_gap = abs(float(cfg.get("blocker_y_gap", self.BLOCKER_Y_GAP_DEFAULT)))
        self.dual_blocker_gap = abs(float(cfg.get("dual_blocker_gap", self.DUAL_BLOCKER_GAP_DEFAULT)))
        self.blocker_z_offset = float(cfg.get("blocker_z_offset", self.BLOCKER_Z_OFFSET_DEFAULT))
        self.blocker_x_offset = float(cfg.get("blocker_x_offset", self.BLOCKER_X_OFFSET_DEFAULT))
        self.target_y_offset = float(cfg.get("target_y_offset", self.TARGET_Y_OFFSET_DEFAULT))

        # ---- arm / stick side first (path is mirrored about table midline x=0).
        arm_side_cfg = cfg.get("arm_side", None)
        if arm_side_cfg is None:
            self.dart_side = float(np.random.choice([-1.0, 1.0]))
        else:
            s = str(arm_side_cfg).strip().lower()
            if s in ("left", "l", "-1"):
                self.dart_side = -1.0
            elif s in ("right", "r", "+1", "1"):
                self.dart_side = 1.0
            else:
                raise ValueError(
                    f"hit_target arm_side must be left/right/null, got {arm_side_cfg!r}"
                )

        # Per-episode speed: base ± speed_jitter_frac (default ±20%).
        # Shared scale keeps Opt2 opposite-phase stable when blocker_speed_base == target_speed_base.
        j = self.speed_jitter_frac
        speed_scale = float(np.random.uniform(1.0 - j, 1.0 + j))
        self.target_speed = float(self.target_speed_base * speed_scale)
        self.blocker_speed = float(self.blocker_speed_base * speed_scale)
        # Keep RNG stream length stable vs prior independent draws.
        _ = float(np.random.uniform(0.0, 1.0))

        # Per-episode path randomization (amplitude / period / phase / direction).
        self.sway_amp *= float(np.random.uniform(0.7, 1.0))
        self.sway_period = int(self.sway_period * float(np.random.uniform(0.85, 1.25)))
        self.sway_dir = float(np.random.choice([-1.0, 1.0]))
        self.sway_phase0 = float(np.random.uniform(0.0, 2.0 * np.pi))
        # Dynamic blocker: opposite-ish phase so |target_x − blocker_x| often exceeds the
        # disc radius — required because the shaft spans the blocker y-plane once the tip
        # is at the board, so co-phased motion would permanently occlude the bullseye.
        self.blocker_phase0 = float(
            self.sway_phase0 + np.pi + np.random.uniform(-0.35, 0.35)
        )

        # Side-local travel span: old symmetric ±sway_amp about 0 becomes [0, span] on the
        # stick's half (right) or [-span, 0] on the left — exact mirror about x=0.
        side_limit = float(
            self.motion_x_max if self.dart_side > 0 else -self.motion_x_min
        )
        side_limit = max(0.0, side_limit)
        self.side_span = float(max(0.0, min(abs(self.sway_amp), side_limit)))
        if self.blocker_enabled or self.blocker_dynamic:
            need_span = float(
                self.blocker_radius + self._shaft_clear_margin() + self.center_radius + 0.020
            )
            self.side_span = float(min(max(self.side_span, need_span), side_limit))
        # Midpoint of this side's band (for board spawn / logging); static blocker stays at 0.
        self.target_center_x = float(self.dart_side * 0.5 * self.side_span)
        self.sway_amp = float(self.side_span)  # expert prefer uses sway_amp as peak |x|

        # ---- dart: tip along +Y (toward the board); matching arm grasps.
        # Left is the exact mirror of right (x → −x).
        dart_x = self.dart_side * float(np.random.uniform(0.10, 0.22))
        dart_y = float(np.random.uniform(-0.12, -0.04))
        # Local +X shaft → world +Y so the dart sits along the y axis.
        dart_q = list(t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], np.pi / 2.0))
        dart_pose = sapien.Pose(
            [dart_x, dart_y, 0.74 + self.table_z_bias + self.DART_SHAFT_HALF[2]],
            dart_q,
        )
        self.dart = self._build_dart(dart_pose)
        self.dart.set_mass(0.02)

        # ---- target board: sways on the stick's half of the table (mirror about x=0).
        self.target_y = 0.08 + self.target_y_offset + float(np.random.uniform(-0.02, 0.02))
        self.target_z = 0.77 + self.table_z_bias + self.board_radius
        board_pose = sapien.Pose(
            [self._target_x_at(0), self.target_y, self.target_z],
            [1, 0, 0, 0],
        )
        self.target = self._build_target(board_pose)
        self._target_rigid = self._get_rigid(self.target)
        if self._target_rigid is not None:
            self._target_rigid.set_kinematic(True)

        # ---- blockers (centers share the target's z — same height as the yellow bullseye)
        self.static_blocker = None
        self.dynamic_blocker = None
        self._static_blocker_rigid = None
        self._dynamic_blocker_rigid = None
        self.blocker = None
        # Always coplanar with the board center / yellow bullseye (ignore residual offsets).
        self.blocker_z_offset = 0.0
        self.blocker_z = float(self.target_z)

        # Static (Opt 1): closer of the two when alone; behind dynamic when both.
        # Dynamic (Opt 2): alone at target_y - gap; with Opt1, 1 cm in front of static.
        if self.blocker_enabled and self.blocker_dynamic:
            self.static_blocker_y = self.target_y - self.blocker_y_gap
            self.dynamic_blocker_y = self.static_blocker_y - self.dual_blocker_gap
        elif self.blocker_enabled:
            self.static_blocker_y = self.target_y - self.blocker_y_gap
            self.dynamic_blocker_y = None
        elif self.blocker_dynamic:
            self.static_blocker_y = None
            self.dynamic_blocker_y = self.target_y - self.blocker_y_gap
        else:
            self.static_blocker_y = None
            self.dynamic_blocker_y = None

        # Compat: frontmost blocker y (used by expert clearance / obs).
        ys = [y for y in (self.static_blocker_y, self.dynamic_blocker_y) if y is not None]
        self.blocker_y = min(ys) if ys else self.target_y - self.blocker_y_gap

        if self.blocker_enabled:
            pose = sapien.Pose(
                [self._static_blocker_x(), self.static_blocker_y, self.blocker_z],
                [1, 0, 0, 0],
            )
            self.static_blocker = self._build_blocker(
                pose, color=self.STATIC_BLOCKER_COLOR, name="target_blocker_static"
            )
            self._static_blocker_rigid = self._get_rigid(self.static_blocker)
            if self._static_blocker_rigid is not None:
                self._static_blocker_rigid.set_kinematic(True)

        if self.blocker_dynamic:
            pose = sapien.Pose(
                [self._dynamic_blocker_x_at(0), self.dynamic_blocker_y, self.blocker_z],
                [1, 0, 0, 0],
            )
            self.dynamic_blocker = self._build_blocker(
                pose, color=self.DYNAMIC_BLOCKER_COLOR, name="target_blocker_dynamic"
            )
            self._dynamic_blocker_rigid = self._get_rigid(self.dynamic_blocker)
            if self._dynamic_blocker_rigid is not None:
                self._dynamic_blocker_rigid.set_kinematic(True)

        self.blocker = self.dynamic_blocker if self.dynamic_blocker is not None else self.static_blocker
        self.add_prohibit_area(self.dart, padding=0.04)

    # --------------------------------------------------------------- dart build
    def _build_dart(self, pose):
        builder = self.scene.create_actor_builder()
        sh = self.DART_SHAFT_HALF
        tp = self.DART_TIP_HALF
        tip_cx = sh[0] + tp[0]
        builder.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=sh,
                                  material=self.scene.default_physical_material)
        builder.add_box_collision(pose=sapien.Pose([tip_cx, 0, 0]), half_size=tp,
                                  material=self.scene.default_physical_material)
        shaft_mat = sapien.render.RenderMaterial(base_color=[*self.DART_COLOR, 1.0])
        tip_mat = sapien.render.RenderMaterial(base_color=[0.0, 0.0, 0.0, 1.0])
        builder.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=sh, material=shaft_mat)
        builder.add_box_visual(pose=sapien.Pose([tip_cx, 0, 0]), half_size=tp, material=tip_mat)
        builder.set_initial_pose(pose)
        entity = builder.build(name="dart")

        tip_x = sh[0] + 2 * tp[0]
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * tip_x, 2 * sh[1], 2 * sh[2]],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            # Top-down grasp frames in dart-local coords (local +X = tip / shaft).
            # With spawn yaw π/2, local +X is world +Y so fingers pinch across ±X.
            "contact_points_pose": [
                [[1, 0, 0, -0.004],
                 [0, 0, -1, 0.0],
                 [0, 1, 0, 0.0],
                 [0, 0, 0, 1]],
                [[1, 0, 0, 0.004],
                 [0, 0, -1, 0.0],
                 [0, 1, 0, 0.0],
                 [0, 0, 0, 1]],
            ],
            "functional_matrix": [
                [[1, 0, 0, tip_x],
                 [0, 1, 0, 0.0],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=0.02)

    # ------------------------------------------------------------- target build
    def _build_target(self, pose):
        builder = self.scene.create_actor_builder()
        th = self.BOARD_THICKNESS
        board_radius = self.board_radius
        face_rot = sapien.Pose([0, 0, 0], self.FACE_ROT_Q)
        builder.add_cylinder_collision(
            pose=face_rot,
            radius=board_radius,
            half_length=th,
            material=self.scene.default_physical_material,
        )
        back_mat = sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.12, 1.0])
        builder.add_cylinder_visual(
            pose=face_rot,
            radius=board_radius * 1.15,
            half_length=th,
            material=back_mat,
        )

        face_y = self._board_face_local_y()
        for k in range(self.n_rings):
            frac = (self.n_rings - k) / self.n_rings
            radius = board_radius * frac
            if k == self.n_rings - 1:
                radius = min(radius, self.center_radius if self.center_radius > 0 else radius)
            col = self.RING_COLORS[min(k, len(self.RING_COLORS) - 1)]
            yk = face_y - self.RING_VISUAL_GAP * k
            mat = sapien.render.RenderMaterial(base_color=[*col, 1.0])
            builder.add_cylinder_visual(
                pose=sapien.Pose([0, yk, 0], self.FACE_ROT_Q),
                radius=radius,
                half_length=self.RING_VISUAL_HALF,
                material=mat,
            )

        builder.set_initial_pose(pose)
        entity = builder.build(name="target_board")

        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * board_radius, 2 * th, 2 * board_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [],
            "functional_matrix": [
                [[1, 0, 0, 0.0],
                 [0, 1, 0, face_y],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=1.0)

    def _build_blocker(self, pose, color, name):
        builder = self.scene.create_actor_builder()
        th = self.blocker_thickness
        face_rot = sapien.Pose([0, 0, 0], self.FACE_ROT_Q)
        builder.add_cylinder_collision(
            pose=face_rot,
            radius=self.blocker_radius,
            half_length=th,
            material=self.scene.default_physical_material,
        )
        blocker_mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
        builder.add_cylinder_visual(
            pose=face_rot,
            radius=self.blocker_radius,
            half_length=th,
            material=blocker_mat,
        )
        builder.set_initial_pose(pose)
        entity = builder.build(name=name)
        # Solid collision: stick–blocker contact fails the episode (geom + PhysX).
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * self.blocker_radius, 2 * th, 2 * self.blocker_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [],
            "functional_matrix": [
                [[1, 0, 0, 0.0],
                 [0, 1, 0, -(th + 0.001)],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=1.0)

    # ------------------------------------------------------------- helpers
    def _get_rigid(self, actor):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _side_path_x(self, phase):
        """Map a phase angle to x on the stick's half of the table (mirror about x=0).

        Right (dart_side=+1): x ∈ [0, side_span]
        Left  (dart_side=−1): x ∈ [−side_span, 0]
        """
        s = float(self.sway_dir) * float(np.sin(phase))  # [-1, 1]
        u = 0.5 * (1.0 + s)  # [0, 1]
        return float(self.dart_side) * float(self.side_span) * u

    def _target_x_at(self, step):
        ph = self.sway_phase0 + 2.0 * np.pi * self.target_speed * step / max(1, self.sway_period)
        return self._side_path_x(ph)

    def _static_blocker_x(self):
        # Static disc stays on the table midline; only target/dynamic path is side-mirrored.
        return float(self.blocker_x_offset)

    def _dynamic_blocker_x_at(self, step):
        ph0 = float(getattr(self, "blocker_phase0", self.sway_phase0 + np.pi))
        ph = ph0 + 2.0 * np.pi * self.blocker_speed * step / max(1, self.sway_period)
        return self._side_path_x(ph) + float(self.blocker_x_offset)

    def _blocker_x_at(self, step):
        """Frontmost / primary blocker x (compat for debug / obs)."""
        if self.blocker_dynamic:
            return self._dynamic_blocker_x_at(step)
        return self._static_blocker_x()

    def _target_center_world(self):
        return np.array(self.target.get_functional_point(0, "list")[:3])

    def _board_face_local_y(self):
        """Board-local y of the outer ring plane (board front faces local −y)."""
        return float(-(float(self.BOARD_THICKNESS) + self.FACE_VISUAL_GAP))

    def _paint_front_offset(self):
        """How far the frontmost (yellow) painted surface sits ahead of the rings' origin."""
        n = max(1, int(getattr(self, "n_rings", self.N_RINGS_DEFAULT)))
        return float(self.RING_VISUAL_GAP * (n - 1) + self.RING_VISUAL_HALF)

    def _board_paint_front_y(self):
        """World y of the visible board surface the dart tip must reach."""
        return float(self._target_center_world()[1]) - self._paint_front_offset()

    def _plant_tip_y(self):
        """Tip y to command so the black tip lands on the paint (1 mm of overlap)."""
        return self._board_paint_front_y() + 0.001

    def _tip_gap_to_plant(self, tip=None) -> float:
        """How far the tip still has to travel in +Y to reach ``_plant_tip_y``."""
        if tip is None:
            tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        return float(self._plant_tip_y() - float(tip[1]))

    def _push_tip_to_plant(self, arm, max_step: float = 0.035) -> bool:
        """Drive the dart tip onto the painted face (same Y as interactive jab).

        Returns False if planning fails or a blocker is hit mid-push.
        """
        for _ in range(12):
            if self._hit_blocker or not self.plan_success:
                return False
            tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
            gap = self._tip_gap_to_plant(tip)
            if gap <= 0.004:
                return True
            step = min(float(max_step), gap)
            self.move(
                self.move_by_displacement(arm_tag=arm, y=float(step), move_axis="world")
            )
            self._check_blocker_hit()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        return bool(
            self.plan_success
            and not self._hit_blocker
            and self._tip_gap_to_plant(tip) <= 0.006
        )

    def _tip_on_board_plane(self, tip_y, gap=None):
        """True when the tip has reached the paint and has not passed out the back."""
        front = self._board_paint_front_y()
        back = float(self.target_y) + float(self.BOARD_THICKNESS)
        slack = self.TOUCH_GAP if gap is None else float(gap)
        return bool(front - slack <= float(tip_y) <= back)

    def _tip_over_board(self, tip=None):
        """True when the tip's XZ position lies within the board disc."""
        if tip is None:
            if getattr(self, "dart", None) is None:
                return False
            tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        center = self._target_center_world()
        radial = float(np.linalg.norm(np.asarray(tip)[[0, 2]] - center[[0, 2]]))
        return radial <= float(self.board_radius)

    def _any_blocker(self):
        return bool(getattr(self, "blocker_enabled", False) or getattr(self, "blocker_dynamic", False))

    def _shaft_clear_margin(self):
        """Extra XZ margin beyond disc radius for shaft/tip half-width + PhysX slack."""
        # Shaft half-width is 0.008 m; keep a modest PhysX buffer without eating the sway band.
        return 0.014

    def _blocker_clear_abs_x(self):
        """Min |x| so the shaft (same xz as tip) clears a centered static disc."""
        if not self._any_blocker():
            return 0.03
        return float(self.blocker_radius + self._shaft_clear_margin() + 0.008)

    def _safe_retreat_from_board(self, arm, side_sign, clear_need):
        """Pull tip back in front of discs without sweeping the shaft through them."""
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        out_x = side_sign * max(abs(float(tip[0])), float(clear_need) + 0.028)
        self._move_tip_x(arm, out_x, side_sign, clear_blockers=True)
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        target_y = self._standoff_y()
        while float(tip[1]) > target_y + 0.012:
            if self._hit_blocker or not self.plan_success:
                break
            if self._shaft_spans_blockers(tip[1]) and not self._xz_clear_of_blockers(
                tip[0], tip[2], step=self._step_count + 10
            ):
                out_x = side_sign * (abs(float(tip[0])) + 0.020)
                self._move_tip_x(arm, out_x, side_sign, clear_blockers=True)
                tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            step = min(0.028, float(tip[1]) - target_y)
            if step <= 0.006:
                break
            self.move(self.move_by_displacement(arm_tag=arm, y=float(-step), move_axis="world"))
            self._check_blocker_hit()
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        self._align_tip_z(arm)

    def _xz_clear_of_blockers(self, x, z, step=None, margin=None):
        """Whether a vertical shaft at (x,z) clears every blocker disc in XZ."""
        m = float(self._shaft_clear_margin() if margin is None else margin)
        need_r2 = float(self.blocker_radius + m) ** 2
        if getattr(self, "blocker_enabled", False):
            bx = self._static_blocker_x()
            if (float(x) - bx) ** 2 + (float(z) - self.blocker_z) ** 2 < need_r2:
                return False
        if getattr(self, "blocker_dynamic", False):
            s = self._step_count if step is None else int(step)
            bx = self._dynamic_blocker_x_at(s)
            if (float(x) - bx) ** 2 + (float(z) - self.blocker_z) ** 2 < need_r2:
                return False
        return True

    def _strike_clear_window(self, duration=180, margin=None):
        """True if chasing the live target center stays clear of every disc for ``duration`` steps.

        Once the tip is at the board the shaft still crosses each blocker y-plane, so the tip
        column must remain outside every disc for the whole plant window.
        """
        if not self._any_blocker():
            return True
        m = float(self._shaft_clear_margin() if margin is None else margin)
        for dt in range(0, int(duration) + 1, 4):
            s = self._step_count + dt
            tx = self._target_x_at(s)
            if not self._xz_clear_of_blockers(tx, self.blocker_z, step=s, margin=m):
                return False
        return True

    def _dynamic_clear_window(self, tip_x, tip_z, duration=150, margin=None):
        """True if the red disc stays clear of (tip_x, tip_z) for the next ``duration`` steps."""
        if not getattr(self, "blocker_dynamic", False):
            return True
        m = float(self._shaft_clear_margin() if margin is None else margin)
        need_r2 = float(self.blocker_radius + m + 0.012) ** 2
        for dt in range(0, int(duration) + 1, 4):
            bx = self._dynamic_blocker_x_at(self._step_count + dt)
            if (float(tip_x) - bx) ** 2 + (float(tip_z) - self.blocker_z) ** 2 < need_r2:
                return False
        return True

    def _dyn_threatens_tip(self, tip_x, tip_z=None, horizon=40):
        """True if the red disc is (or will soon be) too close to the tip column."""
        if not getattr(self, "blocker_dynamic", False):
            return False
        if tip_z is None:
            tip_z = self.blocker_z
        return not self._dynamic_clear_window(tip_x, tip_z, duration=horizon)

    def _point_hits_disc(self, point, bx, by, bz, radius=None, y_tol=None):
        """True if ``point`` intersects a thin disc in the XZ plane at (bx,by,bz)."""
        r = float(self.blocker_radius if radius is None else radius)
        # Tight slab: thickness + small slack (avoid false hits from distant shaft samples).
        yt = float(self.blocker_thickness + 0.006 if y_tol is None else y_tol)
        p = np.asarray(point, dtype=float)
        if abs(float(p[1]) - float(by)) > yt:
            return False
        return float((p[0] - bx) ** 2 + (p[2] - bz) ** 2) <= (r + 0.004) ** 2

    def _dart_sample_points(self):
        """Tip + shaft samples along the dart long axis (local +X → tip)."""
        if getattr(self, "dart", None) is None:
            return []
        tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        T = self.dart.get_pose().to_transformation_matrix()
        tip_dir = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        n = float(np.linalg.norm(tip_dir))
        if n < 1e-8:
            tip_dir = np.array([0.0, 1.0, 0.0])
        else:
            tip_dir = tip_dir / n
        # Full tip→butt length (original stick ≈ 0.114 m).
        length = self._dart_length()
        pts = [tip]
        for d in np.linspace(0.008, length, max(6, int(length / 0.012) + 1)):
            pts.append(tip - tip_dir * float(d))
        return pts

    def _geom_hits_blocker(self, blocker_actor, bx, by, bz):
        if blocker_actor is None:
            return False
        for p in self._dart_sample_points():
            if self._point_hits_disc(p, bx, by, bz):
                return True
        return False

    def _check_blocker_hit(self):
        """Latch failure if the stick intersects any blocker disc (geometry).

        Thin 2 mm discs are unreliable in PhysX contact queries (false positives when the
        shaft is still several cm clear), so we use full-length shaft samples instead.
        """
        if getattr(self, "_hit_blocker", False):
            return True
        if not self._any_blocker():
            return False
        if getattr(self, "dart", None) is None:
            return False

        if getattr(self, "static_blocker", None) is not None:
            bx, by, bz = self._static_blocker_x(), self.static_blocker_y, self.blocker_z
            if self._geom_hits_blocker(self.static_blocker, bx, by, bz):
                self._hit_blocker = True
                # A blocker strike is a hard failure; the ring metric must agree
                # even if the shaft clipped it after the tip planted.
                self.hit_score = 0.0
                return True

        if getattr(self, "dynamic_blocker", None) is not None:
            bx = self._dynamic_blocker_x_at(self._step_count)
            by, bz = self.dynamic_blocker_y, self.blocker_z
            if self._geom_hits_blocker(self.dynamic_blocker, bx, by, bz):
                self._hit_blocker = True
                self.hit_score = 0.0
                return True
        return False

    def _estimate_ik_lead_steps(self, distance_m, min_steps=18, max_steps=160):
        """Rough physics-step budget for a world-frame IK move of ``distance_m``.

        Calibrated from opt2 demos (~0.10 m Z lift ≈ 220 steps → ~2200 steps/m).
        Used to aim the tip at where the bullseye will be when the drop finishes.
        """
        return int(np.clip(abs(float(distance_m)) * 2200.0 + 20.0, min_steps, max_steps))

    def _xz_clear_span(self, x, z, start_step, duration, margin=None):
        """True if tip column (x,z) clears every disc from ``start_step`` for ``duration``."""
        if not self._any_blocker():
            return True
        m = float(self._shaft_clear_margin() if margin is None else margin)
        # Slightly looser than live hit checks so we don't drop into a grazing pass.
        need_r2 = float(self.blocker_radius + m + 0.012) ** 2
        z = float(z)
        x = float(x)
        for dt in range(0, int(duration) + 1, 4):
            s = int(start_step) + dt
            if getattr(self, "blocker_enabled", False):
                bx = self._static_blocker_x()
                if (x - bx) ** 2 + (z - float(self.blocker_z)) ** 2 < need_r2:
                    return False
            if getattr(self, "blocker_dynamic", False):
                bx = self._dynamic_blocker_x_at(s)
                if (x - bx) ** 2 + (z - float(self.blocker_z)) ** 2 < need_r2:
                    return False
        return True

    def _predict_side_intercept(
        self,
        side_sign,
        from_step,
        lead_min=50,
        lead_max=500,
        prefer=0.08,
        clear_need=0.0,
        clear_hold=120,
    ):
        """Pick a future step where the target is on our half (and discs stay clear)."""
        prefer = float(prefer)
        clear_need = float(clear_need)
        best = None
        for dt in range(int(lead_min), int(lead_max) + 1, 2):
            s = from_step + dt
            x = self._target_x_at(s)
            if x * side_sign < max(0.035, 0.45 * prefer):
                continue
            if clear_need > 0.0 and abs(x) < clear_need * 0.90:
                continue
            score = abs(abs(x) - prefer) + 0.00015 * dt
            if self.blocker_dynamic and clear_need > 0.0:
                # Require the tip column at the bullseye to stay clear through plant.
                if not self._xz_clear_span(x, self.blocker_z, s, int(clear_hold)):
                    continue
                # Prefer frames where the red disc is farther from the tip column.
                bx = self._dynamic_blocker_x_at(s)
                score -= 0.20 * min(0.10, abs(float(bx) - float(x)))
            if best is None or score < best[0]:
                best = (score, s, float(x))
        if best is not None:
            return best[1], best[2]
        s = from_step + 120
        x = side_sign * prefer
        return s, x

    def _predict_drop_aim(
        self,
        side_sign,
        clear_need,
        tip_x,
        tip_z,
        prefer,
        search_horizon=240,
        plant_hold=60,
    ):
        """Choose (delay, aim_x) for a high-pass drop that lands on a clear bullseye.

        ``delay`` is steps to wait before starting the Z drop; ``aim_x`` is the tip
        X that matches the bullseye at landing (now + delay + drop_lead).
        """
        drop_lead = self._estimate_ik_lead_steps(
            float(tip_z) - float(self.target_z), min_steps=28, max_steps=120
        )
        tip_x = float(tip_x)
        clear_need = float(clear_need)
        prefer = float(prefer)
        best = None
        for delay in range(0, int(search_horizon) + 1, 4):
            t0 = self._step_count + delay
            t_land = t0 + drop_lead
            aim_x = float(self._target_x_at(t_land))
            if abs(aim_x) < clear_need * 0.90 or aim_x * side_sign <= 0:
                continue
            # Must stay clear from drop start through landing + short plant dwell.
            if not self._xz_clear_span(
                aim_x, self.blocker_z, t0, drop_lead + int(plant_hold)
            ):
                continue
            # Leave time to slide tip X onto aim_x before the drop starts.
            x_lead = self._estimate_ik_lead_steps(
                abs(aim_x - tip_x), min_steps=8, max_steps=90
            )
            if delay + 4 < x_lead:
                continue
            # Prefer soon / near preferred sway amplitude / tip already close in X.
            score = (
                0.0020 * delay
                + abs(abs(aim_x) - prefer)
                + 0.35 * abs(aim_x - tip_x)
            )
            if best is None or score < best[0]:
                best = (score, delay, aim_x, drop_lead)
        if best is None:
            return None
        return {
            "delay": int(best[1]),
            "aim_x": float(best[2]),
            "drop_lead": int(best[3]),
        }

    # ---------------------------------------------------- per-step kinematic motion
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._step_count += 1

        if self._target_rigid is not None and not self._stuck:
            x = self._target_x_at(self._step_count)
            cur = self._target_rigid.entity.get_pose()
            tgt = sapien.Pose([x, self.target_y, self.target_z], cur.q)
            self._target_rigid.set_kinematic_target(tgt)

        if self._dynamic_blocker_rigid is not None and self.blocker_dynamic:
            x = self._dynamic_blocker_x_at(self._step_count)
            cur = self._dynamic_blocker_rigid.entity.get_pose()
            tgt = sapien.Pose([x, self.dynamic_blocker_y, self.blocker_z], cur.q)
            self._dynamic_blocker_rigid.set_kinematic_target(tgt)

        if self._stuck and self._dart_rigid is not None and self._target_rigid is not None:
            board_pose = self._target_rigid.entity.get_pose()
            tip_world = board_pose.to_transformation_matrix() @ np.append(self._stick_local, 1.0)
            dart_pose = sapien.Pose(tip_world[:3] - self._tip_offset_world, self._stick_dart_q)
            self._dart_rigid.set_kinematic_target(dart_pose)

        # Stick–blocker contact fails the episode (checked every physics step).
        if not self._stuck:
            self._check_blocker_hit()
        self._track_dart_metrics()

    def _ring_index_at_radius(self, radial_offset: float) -> int | None:
        """Index of the painted ring under the tip, 0 = outermost … n-1 = bullseye.

        Rings are nested discs drawn outer→inner (``RING_COLORS``) at radii
        ``board_radius * (n - k) / n``. Returns ``None`` when the tip is outside
        the board radius.
        """
        r = float(radial_offset)
        board_r = float(self.board_radius)
        if r > board_r + 1e-6:
            return None
        n = max(1, int(self.n_rings))
        index = 0
        for k in range(n):
            frac = (n - k) / n
            radius = board_r * frac
            if k == n - 1:
                radius = min(
                    radius,
                    float(self.center_radius) if self.center_radius > 0 else radius,
                )
            if r <= radius + 1e-9:
                index = k
            else:
                break
        return index

    def _ring_score(self, ring_index: int | None) -> float:
        """Partial credit for the ring that was stabbed: bullseye 1.0, outer 1/n.

        ``ring_index`` counts inward (0 = outermost), so the score runs the other
        way: ``(k + 1) / n``. With the default 4 rings that is yellow 1.00 /
        blue 0.75 / white 0.50 / red 0.25; off the board (or a blocker
        strike) scores 0.
        """
        if ring_index is None:
            return 0.0
        n = max(1, int(self.n_rings))
        k = int(np.clip(ring_index, 0, n - 1))
        return float((k + 1) / n)

    def _ring_color_at_radius(self, radial_offset: float) -> str | None:
        """Map planar tip distance from the bullseye to a painted ring color.

        Returns ``None`` when the tip is outside the board radius.
        """
        index = self._ring_index_at_radius(radial_offset)
        if index is None:
            return None
        return self.RING_COLOR_NAMES[min(index, len(self.RING_COLOR_NAMES) - 1)]

    def _record_board_hit(self) -> str | None:
        """Latch the ring color under the tip when it reaches the board face.

        Also latches ``hit_score`` — partial credit for whichever ring was
        stabbed (see ``_ring_score``). Does not change success rules: only
        yellow welds (via ``_try_form_stick``) and only yellow passes
        ``check_success``. Returns the latched color name, or ``None`` if the
        tip is not on the board.
        """
        if self._hit_blocker or getattr(self, "dart", None) is None:
            return self._hit_color
        tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        target_center = self._target_center_world()
        planar_offset = tip[[0, 2]] - target_center[[0, 2]]
        radial_offset = float(np.linalg.norm(planar_offset))
        if not self._tip_on_board_plane(tip[1], gap=self.TOUCH_LATCH_GAP):
            return self._hit_color
        ring_index = self._ring_index_at_radius(radial_offset)
        if ring_index is None:
            return self._hit_color
        color = self.RING_COLOR_NAMES[min(ring_index, len(self.RING_COLOR_NAMES) - 1)]
        self._hit_planar_offset = planar_offset.astype(float)
        self._hit_radial_offset = radial_offset
        self._hit_color = color
        self._hit_ring_index = int(ring_index)
        # Metrics: first board contact is the decisive event.
        if getattr(self, "_metric_impact_step", None) is None:
            self._metric_impact_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
            self._metric_impact_radial = float(radial_offset)
        # Keep the best ring reached: a tip that grazes an outer ring on the way
        # in must not downgrade a bullseye already scored this episode.
        self.hit_score = max(float(self.hit_score), self._ring_score(ring_index))
        return color

    def hit_result_detail(self) -> str:
        """Short end-of-episode label: which color was hit, or why it failed."""
        if self._hit_blocker:
            return "blocker hit"
        color = self._hit_color
        if color is None:
            # Last chance classification from the current tip pose.
            color = self._record_board_hit()
        if color == "yellow":
            return f"hit yellow (success, score {self.hit_score:.2f})"
        if color in ("blue", "white", "red"):
            return f"hit {color} (failure, partial score {self.hit_score:.2f})"
        if self._tip_over_board():
            return "no contact: tip stopped short of the board face"
        return "missed the board"

    def _try_form_stick(self, *, any_ring: bool = False, exact_pose: bool = False):
        """Weld the dart to the board when the tip contacts it.

        Default (expert): only yellow-center hits weld, tip planted on the paint.
        Interactive callers may pass ``any_ring=True`` / ``exact_pose=True`` so a
        tip that reaches any painted ring freezes at that exact contact pose.
        Never welds after a blocker strike. Success still requires yellow center
        (``_hit_center``).
        """
        if self._hit_blocker or self._stuck:
            return False
        color = self._record_board_hit()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        target_center = self._target_center_world()
        planar_offset = tip[[0, 2]] - target_center[[0, 2]]
        radial_offset = float(np.linalg.norm(planar_offset))
        on_center = radial_offset <= self.center_radius and color == "yellow"
        on_board = color is not None and self._tip_on_board_plane(tip[1])
        if not on_board:
            return False
        if not any_ring and not on_center:
            return False
        self._check_blocker_hit()
        if self._hit_blocker:
            return False
        self._dart_rigid = self._get_rigid(self.dart)
        if self._dart_rigid is None:
            return False
        self._dart_rigid.set_kinematic(True)
        board_pose = self._target_rigid.entity.get_pose()
        inv = np.linalg.inv(board_pose.to_transformation_matrix())
        stick_local = (inv @ np.append(tip, 1.0))[:3]
        if not exact_pose:
            # Plant the tip on the paint instead of freezing whatever approach gap
            # was left, so the black tip visibly meets the yellow center.
            stick_local[1] = self._board_face_local_y() - self._paint_front_offset() + 0.001
        self._stick_local = stick_local
        self._tip_offset_world = tip - np.array(self.dart.get_pose().p)
        self._stick_dart_q = self.dart.get_pose().q
        self._stuck = True
        self._hit_planar_offset = planar_offset.astype(float)
        self._hit_radial_offset = radial_offset
        self._hit_color = color
        if on_center:
            self._hit_center = True
            # Already latched by _record_board_hit above; keep the ladder as the
            # single source of truth so yellow == _ring_score(n-1) == 1.0.
            self.hit_score = max(float(self.hit_score), self._ring_score(self.n_rings - 1))
        return True

    def _advance(self, steps, try_stick=False):
        """Advance physics (target keeps moving), optionally attempting a center stick."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            self._check_blocker_hit()
            if try_stick and not self._stuck and not self._hit_blocker:
                self._try_form_stick()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._stuck or self._hit_blocker:
                break

    def _dwell(self, steps):
        """Advance while attempting a center stick each step."""
        self._advance(steps, try_stick=True)

    def _dbg(self, tag):
        import os
        if os.environ.get("HIT_TARGET_DEBUG") or os.environ.get("STAB_DEBUG"):
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            target_center = self._target_center_world()
            blocker_msg = ""
            if self.static_blocker is not None:
                blocker_msg += f" static={np.round(self.static_blocker.get_pose().p,3).tolist()}"
            if self.dynamic_blocker is not None:
                blocker_msg += f" dynamic={np.round(self.dynamic_blocker.get_pose().p,3).tolist()}"
            print(
                f"[HIT_TARGET] {tag}: plan_success={self.plan_success} "
                f"tip={np.round(tip,3).tolist()} center={np.round(target_center,3).tolist()} "
                f"stuck={self._stuck} center_hit={self._hit_center} step={self._step_count}"
                f"{blocker_msg}",
                flush=True,
            )

    def _dart_length(self):
        return float(2 * self.DART_SHAFT_HALF[0] + 2 * self.DART_TIP_HALF[0])

    def _high_pass_to_board(self, arm, side_sign, clear_need):
        """Lift above discs, advance to the board, then drop to bullseye height.

        Full-length stick + 5–8 cm gap means a coplanar Y push lets the moving disc
        drift into the tip before the IK finishes; flying over the discs avoids that.
        """
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        out_x = side_sign * max(abs(float(tip[0])), float(clear_need))
        self._move_tip_x(arm, out_x, side_sign, clear_blockers=True)
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        hi = float(self.blocker_z + self.blocker_radius + 0.028)
        if hi - float(tip[2]) > 0.008:
            self.move(self.move_by_displacement(arm_tag=arm, z=float(hi - tip[2]), move_axis="world"))
        if not self.plan_success or self._hit_blocker:
            return False
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        goal_y = float(self._plant_tip_y())
        for _ in range(12):
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            if float(tip[1]) >= goal_y - 0.006:
                break
            if not self.plan_success or self._hit_blocker:
                return False
            self._move_tip_x(
                arm,
                side_sign * max(abs(float(tip[0])), float(clear_need)),
                side_sign,
                clear_blockers=True,
            )
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            step = min(0.045, goal_y - float(tip[1]))
            if step <= 0.005:
                break
            self.move(self.move_by_displacement(arm_tag=arm, y=float(step), move_axis="world"))
            self._check_blocker_hit()
        if self._hit_blocker or not self.plan_success:
            return False

        # Hover high near the board; aim X at the bullseye *after* the Z-drop delay,
        # and only drop when that landing column stays clear of the red disc.
        align_tol = 1.15 * self.center_radius
        prefer = float(max(0.085, 0.95 * float(self.sway_amp)))
        for _ in range(80):
            if self._hit_blocker or not self.plan_success:
                return False
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            hi = float(self.blocker_z + self.blocker_radius + 0.028)
            if hi - float(tip[2]) > 0.010:
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, z=float(hi - tip[2]), move_axis="world"
                    )
                )
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            drop_lead = self._estimate_ik_lead_steps(
                float(tip[2]) - float(self.target_z), min_steps=30, max_steps=100
            )
            pred = self._predict_drop_aim(
                side_sign,
                clear_need,
                tip_x=float(tip[0]),
                tip_z=float(tip[2]),
                prefer=prefer,
                search_horizon=200,
                plant_hold=40,
            )
            if pred is not None:
                aim_x = float(pred["aim_x"])
                delay = int(pred["delay"])
                drop_lead = int(pred["drop_lead"])
            else:
                # Fallback: lead the live bullseye by the drop duration.
                aim_x = float(self._target_x_at(self._step_count + drop_lead))
                delay = 0
                if abs(aim_x) < float(clear_need) * 0.90 or aim_x * side_sign <= 0:
                    live = float(self._target_center_world()[0])
                    if abs(live) >= float(clear_need) * 0.90 and live * side_sign > 0:
                        aim_x = live
                    else:
                        self._advance(8, try_stick=False)
                        continue

            self._move_tip_x(arm, aim_x, side_sign, clear_blockers=True)
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])

            if delay > 14:
                self._advance(min(delay, 18), try_stick=False)
                continue

            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            # Refresh aim after lateral IK consumed steps.
            drop_lead = self._estimate_ik_lead_steps(
                float(tip[2]) - float(self.target_z), min_steps=30, max_steps=100
            )
            aim_x = float(self._target_x_at(self._step_count + drop_lead))
            if abs(aim_x) < float(clear_need) * 0.90 or aim_x * side_sign <= 0:
                self._advance(8, try_stick=False)
                continue
            self._move_tip_x(arm, aim_x, side_sign, clear_blockers=True)
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])

            aligned = abs(aim_x - float(tip[0])) <= align_tol + 0.028
            clear_ok = self._xz_clear_span(
                float(tip[0]), self.blocker_z, self._step_count, drop_lead + 45
            )
            # Also accept the old live gate so we don't stall forever.
            live_cx = float(self._target_center_world()[0])
            live_ok = (
                abs(live_cx) >= float(clear_need) * 0.90
                and live_cx * side_sign > 0
                and abs(live_cx - float(tip[0])) <= align_tol + 0.022
                and not self._dyn_threatens_tip(tip[0], self.blocker_z, horizon=50)
                and self._xz_clear_of_blockers(
                    tip[0], self.blocker_z, step=self._step_count + 20
                )
            )
            if not ((aligned and clear_ok) or live_ok):
                self._advance(8, try_stick=False)
                continue

            if self._dyn_threatens_tip(tip[0], self.blocker_z, horizon=35):
                self._advance(8, try_stick=False)
                continue

            # Final X snap to predicted landing, then drop.
            aim_x = float(self._target_x_at(self._step_count + drop_lead))
            if abs(aim_x) >= float(clear_need) * 0.90 and aim_x * side_sign > 0:
                self._move_tip_x(arm, aim_x, side_sign, clear_blockers=True)
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            cz = float(self.target_z)
            if float(tip[2]) - cz > 0.006:
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, z=float(cz - tip[2]), move_axis="world"
                    )
                )
            self._check_blocker_hit()
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            return bool(
                self.plan_success
                and not self._hit_blocker
                and self._tip_past_blockers(tip[1])
            )

        self._go_to_standoff(arm)
        self._align_tip_z(arm)
        return False

    def _align_tip_z(self, arm):
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        cz = float(self._target_center_world()[2])
        if abs(cz - tip[2]) > 0.004:
            self.move(self.move_by_displacement(arm_tag=arm, z=float(cz - tip[2]), move_axis="world"))

    def _lift_and_align_tip_to_board(self, arm, lift_z=0.10):
        """Post-grasp: lift and yaw so the tip faces the board (+Y).

        Replaces the old fixed ±π/2 yaw that assumed an X-aligned spawn. The dart
        already sits along +Y; this only corrects if the planner grasp flipped it.
        """
        tip = np.array(self.dart.get_functional_point(0, "list")[:3], dtype=float)
        body = np.array(self.dart.get_pose().p, dtype=float)
        planar = tip - body
        planar[2] = 0.0
        n = float(np.linalg.norm(planar))
        quat = None
        if n > 1e-4:
            # Angle of tip from +Y toward +X; rotate by −θ about world Z.
            theta = float(np.arctan2(planar[0], planar[1]))
            if abs(theta) > 0.05:
                cur_q = np.asarray(self.get_arm_pose(str(arm))[3:], dtype=np.float64)
                yaw = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], -theta)
                quat = list(t3d.quaternions.qmult(yaw, cur_q))
        kwargs = dict(arm_tag=arm, z=float(lift_z), move_axis="world")
        if quat is not None:
            kwargs["quat"] = quat
        self.move(self.move_by_displacement(**kwargs))

    def _strike_clearance_z(self):
        if not self._any_blocker():
            return float(self._target_center_world()[2])
        return float(self.blocker_z + self.blocker_radius + self.BLOCKER_CLEARANCE_Z)

    def _lift_over_blockers(self, arm):
        if not self._any_blocker():
            return
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        hi = self._strike_clearance_z()
        if hi - float(tip[2]) > 0.008:
            self.move(self.move_by_displacement(arm_tag=arm, z=float(hi - tip[2]), move_axis="world"))

    def _front_blocker_y(self):
        ys = [v for v in (self.static_blocker_y, self.dynamic_blocker_y) if v is not None]
        return float(min(ys)) if ys else None

    def _tip_past_blockers(self, tip_y=None):
        """True once the tip has crossed beyond the front disc face (Y progress)."""
        front = self._front_blocker_y()
        if front is None:
            return True
        if tip_y is None:
            tip_y = float(self.dart.get_functional_point(0, "list")[1])
        return float(tip_y) >= float(front) + 0.012

    def _shaft_spans_blockers(self, tip_y=None):
        """True while the stick body still intersects a blocker y-plane.

        Full-length stick tip→butt (~11.4 cm) exceeds the 5–8 cm blocker gap, so once the
        tip is near the board the shaft always spans the discs until a center stick forms.
        """
        front = self._front_blocker_y()
        if front is None:
            return False
        if tip_y is None:
            tip_y = float(self.dart.get_functional_point(0, "list")[1])
        return float(front) - 0.01 < float(tip_y) < float(front) + self._dart_length() + 0.01

    def _safe_chase_x(self, desired_x, side_sign, tip_z=None):
        """Clamp a chase x so the tip column stays outside every disc when the shaft spans."""
        x_cmd = float(desired_x)
        if tip_z is None:
            tip_z = float(self.dart.get_functional_point(0, "list")[2])
        if abs(x_cmd) < 0.025 or x_cmd * side_sign < 0.02:
            x_cmd = side_sign * max(0.06, abs(x_cmd))
        if self._any_blocker() and self._shaft_spans_blockers():
            need = self._blocker_clear_abs_x()
            if abs(x_cmd) < need or not self._xz_clear_of_blockers(
                x_cmd, tip_z, step=self._step_count + 20
            ):
                x_cmd = side_sign * max(need, abs(x_cmd))
        return float(x_cmd)

    def _standoff_y(self):
        """Pre-strike hover: short of the board; stay in front of discs until the high pass."""
        y = float(self.target_y - 0.050)
        front = self._front_blocker_y()
        if front is not None:
            y = min(y, float(front - 0.030))
        return max(y, -0.06)

    def _retreat_to_standoff(self, arm):
        # Lateral expert: never lift to clear discs on retreat (high-z IK is unreliable).
        # Stay in front of the discs at bullseye height with tip_x already clear.
        self._go_to_standoff(arm)
        self._align_tip_z(arm)

    def _go_to_standoff(self, arm):
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        desired_y = self._standoff_y()
        dy = desired_y - float(tip[1])
        if abs(dy) > 0.008:
            self.move(self.move_by_displacement(arm_tag=arm, y=float(dy), move_axis="world"))

    def _move_tip_x(self, arm, target_x, side_sign, clear_blockers=False):
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        x_cmd = float(target_x)
        if abs(x_cmd) < 0.025 or x_cmd * side_sign < 0.02:
            x_cmd = side_sign * max(0.06, abs(x_cmd))
        # Keep tip column clear whenever the shaft spans a disc (full-length stick near board).
        if clear_blockers or self._shaft_spans_blockers(tip[1]):
            if abs(float(tip[2]) - self.blocker_z) < self.blocker_radius + 0.03:
                x_cmd = self._safe_chase_x(x_cmd, side_sign, tip_z=tip[2])
        dx = x_cmd - float(tip[0])
        if abs(dx) > 0.004:
            self.move(self.move_by_displacement(arm_tag=arm, x=float(dx), move_axis="world"))

    # ------------------------------------------------------------------ policy
    def play_once(self):
        arm = ArmTag("right" if self.dart_side > 0 else "left")
        side_sign = float(self.dart_side)
        self._dbg("start")

        self.move(self.grasp_actor(
            self.dart, arm_tag=arm, pre_grasp_dis=0.08, contact_point_id=[0, 1],
        ))
        self._dbg("after grasp")

        self._lift_and_align_tip_to_board(arm, lift_z=0.10)
        self._dbg("after lift+align")

        self._align_tip_z(arm)
        self._go_to_standoff(arm)
        self._dbg("after standoff")

        # Use the wider sway: plant on the live bullseye once it is outside the discs.
        prefer = float(max(0.085, 0.95 * float(self.sway_amp)))
        align_tol = 0.90 * self.center_radius
        clear_need = float(self._blocker_clear_abs_x() + 0.008) if self._any_blocker() else 0.04
        cleared = False
        for attempt in range(20):
            if self._stuck or self._hit_blocker or not self.plan_success:
                break

            if cleared:
                # Tip near board; shaft spans discs — stick if aligned, retreat only on real threat.
                for _ in range(50):
                    if self._stuck or self._hit_blocker or not self.plan_success:
                        break
                    self._align_tip_z(arm)
                    tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                    center = self._target_center_world()
                    radial = float(np.linalg.norm(tip[[0, 2]] - center[[0, 2]]))
                    if self._tip_on_board_plane(tip[1]) and radial <= float(self.center_radius):
                        self._dwell(30)
                        continue
                    if self._dyn_threatens_tip(tip[0], tip[2], horizon=40):
                        self._safe_retreat_from_board(arm, side_sign, clear_need)
                        cleared = False
                        break
                    chase_lead = self._estimate_ik_lead_steps(
                        abs(float(center[0]) - float(tip[0])),
                        min_steps=18,
                        max_steps=55,
                    )
                    cx = float(self._target_x_at(self._step_count + chase_lead))
                    if (
                        abs(cx) >= clear_need * 0.95
                        and cx * side_sign > 0
                        and self._xz_clear_of_blockers(
                            cx, tip[2], step=self._step_count + chase_lead
                        )
                        and not self._dyn_threatens_tip(cx, tip[2], horizon=35)
                    ):
                        self._move_tip_x(arm, cx, side_sign, clear_blockers=True)
                    else:
                        self._move_tip_x(
                            arm,
                            side_sign * max(abs(float(tip[0])), clear_need),
                            side_sign,
                            clear_blockers=True,
                        )
                    self._dwell(35)
                if not cleared:
                    continue
                break

            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            intercept_step, x_lead = self._predict_side_intercept(
                side_sign,
                self._step_count,
                lead_min=20,
                lead_max=560,
                prefer=prefer,
                clear_need=clear_need if self.blocker_dynamic else 0.0,
                clear_hold=90 if self.blocker_dynamic else 0,
            )
            # Lead time ≈ lateral chase duration to the predicted intercept x.
            chase_lead = self._estimate_ik_lead_steps(
                abs(float(x_lead) - float(tip[0])), min_steps=22, max_steps=110
            )
            wait = max(0, intercept_step - self._step_count - chase_lead)
            if wait > 0:
                self._advance(min(wait, 600), try_stick=False)
            if self._hit_blocker:
                break

            for _ in range(3):
                if not self.plan_success or self._hit_blocker:
                    break
                _, x_lead = self._predict_side_intercept(
                    side_sign,
                    self._step_count,
                    lead_min=10,
                    lead_max=160,
                    prefer=prefer,
                    clear_need=clear_need if self.blocker_dynamic else 0.0,
                    clear_hold=70 if self.blocker_dynamic else 0,
                )
                self._align_tip_z(arm)
                self._move_tip_x(arm, x_lead, side_sign, clear_blockers=False)
                self._go_to_standoff(arm)
                self._advance(8, try_stick=False)

            # Aim where the bullseye will be after a short chase (not the live pose).
            chase_lead = self._estimate_ik_lead_steps(0.04, min_steps=28, max_steps=70)
            x_push = float(self._target_x_at(self._step_count + chase_lead))
            min_side = clear_need if self._any_blocker() else 0.030
            if abs(x_push) < min_side or x_push * side_sign <= 0:
                continue
            if float(self._target_x_at(self._step_count + 120)) * side_sign < min_side * 0.85:
                continue
            if self._any_blocker() and not self._strike_clear_window(duration=160):
                continue
            plant_x = side_sign * abs(x_push)
            self._align_tip_z(arm)
            self._move_tip_x(arm, plant_x, side_sign, clear_blockers=False)
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            plant_x = float(tip[0])
            if plant_x * side_sign < min_side * 0.9:
                continue

            if self._any_blocker():
                clear_x = side_sign * max(abs(plant_x), clear_need)
                if self.blocker_dynamic:
                    waited = 0
                    while waited < 800 and not self._dynamic_clear_window(
                        clear_x, self.blocker_z, duration=100
                    ):
                        if self._hit_blocker or not self.plan_success:
                            break
                        # Nudge wait using predicted next clear landing when available.
                        tip_w = np.array(self.dart.get_functional_point(0, "list")[:3])
                        pred = self._predict_drop_aim(
                            side_sign,
                            clear_need,
                            tip_x=float(tip_w[0]),
                            tip_z=float(
                                max(
                                    tip_w[2],
                                    self.blocker_z + self.blocker_radius + 0.02,
                                )
                            ),
                            prefer=prefer,
                            search_horizon=280,
                            plant_hold=45,
                        )
                        step = 12
                        if pred is not None and int(pred["delay"]) > 20:
                            step = min(48, max(12, int(pred["delay"]) // 2))
                            clear_x = side_sign * max(
                                abs(float(pred["aim_x"])), clear_need
                            )
                        self._advance(step, try_stick=False)
                        waited += step
                    if self._hit_blocker or not self.plan_success:
                        break
                    if not self._dynamic_clear_window(clear_x, self.blocker_z, duration=70):
                        continue
                if not self._strike_clear_window(duration=140):
                    continue
                # Re-check: bullseye still outside discs and near the plant column.
                chase_lead = self._estimate_ik_lead_steps(
                    0.03, min_steps=18, max_steps=50
                )
                x_now = float(self._target_x_at(self._step_count + chase_lead))
                if abs(x_now) < clear_need or x_now * side_sign <= 0:
                    continue
                if abs(x_now - clear_x) > align_tol + 0.035:
                    clear_x = side_sign * max(abs(x_now), clear_need)
                self._align_tip_z(arm)
                self._move_tip_x(arm, clear_x, side_sign, clear_blockers=True)
                tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                plant_x = float(tip[0])
                if abs(plant_x) < clear_need * 0.95:
                    continue
                if self.blocker_dynamic:
                    # Full-length stick + moving disc: fly over, hover until aligned, then drop.
                    if not self._high_pass_to_board(arm, side_sign, clear_need):
                        self._safe_retreat_from_board(arm, side_sign, clear_need)
                        continue
                else:
                    # Static disc only: lateral Y push at bullseye height (shaft stays outside x=0).
                    # Drive onto the painted face — same plant Y as interactive jab / stick check.
                    while self._tip_gap_to_plant(tip) > 0.004:
                        if self._hit_blocker or not self.plan_success:
                            break
                        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                        if abs(float(tip[0])) < clear_need * 0.95:
                            self._move_tip_x(
                                arm,
                                side_sign * (clear_need + 0.010),
                                side_sign,
                                clear_blockers=True,
                            )
                            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                            if abs(float(tip[0])) < clear_need * 0.95:
                                break
                        if not self._xz_clear_of_blockers(tip[0], tip[2], step=self._step_count + 15):
                            break
                        step_y = min(0.035, self._tip_gap_to_plant(tip))
                        if step_y <= 0.004:
                            break
                        self.move(
                            self.move_by_displacement(arm_tag=arm, y=float(step_y), move_axis="world")
                        )
                        self._check_blocker_hit()
                        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                self._check_blocker_hit()
                if self._hit_blocker:
                    break
                tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                if not self._tip_past_blockers(tip[1]):
                    self._safe_retreat_from_board(arm, side_sign, clear_need)
                    continue
                self._align_tip_z(arm)
                if not self._push_tip_to_plant(arm):
                    self._safe_retreat_from_board(arm, side_sign, clear_need)
                    continue
                cleared = True
            else:
                self._align_tip_z(arm)
                if not self._push_tip_to_plant(arm):
                    if self._hit_blocker or not self.plan_success:
                        break

            self._check_blocker_hit()
            self._dbg(f"after push attempt{attempt}")
            if self._hit_blocker:
                break

            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            plant_x = float(tip[0])
            for _ in range(40):
                if self._stuck or self._hit_blocker or not self.plan_success:
                    break
                self._align_tip_z(arm)
                if not self.plan_success:
                    break
                # Re-seat tip on the paint each dwell cycle so contact matches
                # interactive ``_plant_tip_y`` / ``_tip_on_board_plane`` rules.
                if self._tip_gap_to_plant() > 0.004:
                    if not self._push_tip_to_plant(arm):
                        break
                tip = np.array(self.dart.get_functional_point(0, "list")[:3])
                center = self._target_center_world()
                radial = float(np.linalg.norm(tip[[0, 2]] - center[[0, 2]]))
                on_yellow = (
                    self._tip_on_board_plane(tip[1])
                    and radial <= float(self.center_radius)
                )
                # If already on yellow, dwell hard for a stick — don't retreat for a soft threat.
                if on_yellow:
                    self._dwell(30)
                    continue
                # Lead the chase: aim where the bullseye will be after the next lateral IK.
                chase_lead = self._estimate_ik_lead_steps(
                    abs(float(center[0]) - float(tip[0])),
                    min_steps=18,
                    max_steps=55,
                )
                cx = float(self._target_x_at(self._step_count + chase_lead))
                if self._dyn_threatens_tip(tip[0], tip[2], horizon=40):
                    self._safe_retreat_from_board(arm, side_sign, clear_need)
                    cleared = False
                    break
                if not self._any_blocker():
                    if cx * side_sign > 0.02:
                        self._move_tip_x(arm, cx, side_sign, clear_blockers=False)
                        plant_x = float(self.dart.get_functional_point(0, "list")[0])
                elif (
                    abs(cx) >= clear_need * 0.95
                    and cx * side_sign > 0
                    and self._xz_clear_of_blockers(cx, tip[2], step=self._step_count + chase_lead)
                    and not self._dyn_threatens_tip(cx, tip[2], horizon=35)
                ):
                    self._move_tip_x(arm, cx, side_sign, clear_blockers=True)
                    plant_x = float(self.dart.get_functional_point(0, "list")[0])
                else:
                    self._move_tip_x(
                        arm,
                        side_sign * max(abs(plant_x), clear_need),
                        side_sign,
                        clear_blockers=True,
                    )
                if not self.plan_success:
                    break
                self._dwell(35)
            if self._stuck or self._hit_blocker:
                break
            if cleared:
                break
            if self.plan_success:
                self._safe_retreat_from_board(arm, side_sign, clear_need)

        self._dbg("done")
        self.info["info"] = {
            "{A}": "dart/base340",
            "{B}": "moving_target/base341",
            "{a}": str(arm),
        }
        return self.info

    # ------------------------------------------------------------------ success
    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_lift_step = None      # dart first clear of the table
        self._metric_impact_step = None    # tip first on the board face
        self._metric_impact_radial = None  # radial miss at that instant, in metres

    def _track_dart_metrics(self):
        """Latch the step the dart was first lifted clear of the table.

        This opens the aiming window: everything before it is the approach/grasp,
        which the completion time already covers. High-water mark — only the first
        lift is kept, and the query stops once latched.
        """
        if getattr(self, "_metric_lift_step", None) is not None:
            return
        try:
            dart_z = float(self.dart.get_pose().p[2])
            if dart_z < float(self.table_z_bias) + 0.85:
                return
        except Exception:
            return
        self._metric_lift_step = int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _compute_metrics(self):
        """extra1 = lift->board-impact latency, extra2 = radial miss in bullseye radii.

        ``impact_offset_norm`` is the tip's distance from the board centre at first
        contact, as a fraction of the yellow bullseye radius: 0.0 = dead centre,
        1.0 = exactly on the yellow edge, >1.0 = outside the scoring centre.
        LOWER IS BETTER. ``ring_score`` is the partial credit already tracked by
        the task (1.0 = bullseye, 1/n = outermost ring, 0.0 = miss or blocker) —
        HIGHER IS BETTER. Both are None when the board was never touched.
        """
        offset = None
        try:
            if getattr(self, "_metric_impact_radial", None) is not None:
                offset = float(self._metric_impact_radial) / max(
                    float(self.center_radius), 1e-9)
        except Exception:
            offset = None
        metrics = {
            "impact_latency_steps": None,
            "impact_latency_s": None,
            "impact_offset_norm": offset,
            "ring_score": (
                None
                if getattr(self, "_metric_impact_step", None) is None
                else float(getattr(self, "hit_score", 0.0))
            ),
        }
        start = getattr(self, "_metric_lift_step", None)
        hit = getattr(self, "_metric_impact_step", None)
        if start is not None and hit is not None and hit >= start:
            steps = int(hit - start)
            metrics["impact_latency_steps"] = steps
            try:
                metrics["impact_latency_s"] = round(
                    steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics

    def check_success(self):
        """Success: black tip welded on the yellow paint; never struck a blocker.

        Contact uses the same paint-face plane as interactive play
        (``_tip_on_board_plane`` / ``_plant_tip_y``): tip must reach the painted
        surface within the yellow radius. A tip that stops short of the paint
        does not count.
        """
        if self._hit_blocker:
            return False
        return bool(self._stuck and self._hit_center)

    def get_obs(self):
        obs = super().get_obs()
        if getattr(self, "target", None) is None or self._target_rigid is None:
            return obs
        target_center = self._target_center_world().tolist()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3]).tolist()
        planar_offset = (
            self._hit_planar_offset.tolist() if self._hit_planar_offset is not None else [-1.0, -1.0]
        )
        static_p = (
            self.static_blocker.get_pose().p.tolist()
            if self.static_blocker is not None
            else [0.0, 0.0, 0.0]
        )
        dynamic_p = (
            self.dynamic_blocker.get_pose().p.tolist()
            if self.dynamic_blocker is not None
            else [0.0, 0.0, 0.0]
        )
        obs["hit_target"] = {
            "target_center_world": target_center,
            "dart_tip_world": tip,
            "target_x": float(self._target_x_at(self._step_count)),
            "stuck": bool(self._stuck),
            "center_hit": bool(self._hit_center),
            "hit_blocker": bool(self._hit_blocker),
            "hit_color": self._hit_color,
            "hit_ring_index": (
                int(self._hit_ring_index) if self._hit_ring_index is not None else -1
            ),
            "n_rings": int(self.n_rings),
            "planar_offset": planar_offset,
            "radial_offset": float(self._hit_radial_offset) if self._hit_radial_offset is not None else -1.0,
            "hit_score": float(self.hit_score),
            "board_radius": float(self.board_radius),
            "center_radius": float(self.center_radius),
            "board_half_size": float(self.board_radius),
            "center_half_size": float(self.center_radius),
            "blocker_enabled": bool(self.blocker_enabled),
            "blocker_dynamic": bool(self.blocker_dynamic),
            "blocker_radius": float(self.blocker_radius) if self._any_blocker() else 0.0,
            "blocker_speed": float(self.blocker_speed) if self.blocker_dynamic else 0.0,
            "target_speed": float(self.target_speed),
            "dart_side": float(self.dart_side),
            "static_blocker_world": static_p,
            "dynamic_blocker_world": dynamic_p,
            "blocker_world": (
                self.blocker.get_pose().p.tolist()
                if getattr(self, "blocker", None) is not None
                else [0.0, 0.0, 0.0]
            ),
        }
        return obs
