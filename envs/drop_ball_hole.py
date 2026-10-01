from ._base_task import Base_Task
from .utils import *
import os
import tempfile
import sapien
import sapien.physx
import sapien.render
import numpy as np
import transforms3d as t3d
from shapely.geometry import Point, Polygon
from trimesh.creation import extrude_polygon


class drop_ball_hole(Base_Task):
    """Pick a ball and drop it through a rotating corner hole into the box below.

    The cap spins deterministically every physics step so plan and render passes stay in lock-step.
    A colored sphere spawns on either the left or right side of the central box. The corresponding
    arm grasps the ball, moves it to a same-side release station over the rotating platform, waits
    for the circular corner hole to pass underneath, and releases so the ball falls into the box.

    Independent platform features (both may be enabled together):
      stick_to_surface (Opt1):
        Missed drop with no XY overlap vs the real hole latches the ball onto the spinning
        platform (fail). Hole overlap keeps the ball dynamic until it settles.
      add_dummy_hole (Opt2):
        Cuts an opposite-corner decoy hole *very slightly* smaller than the ball so the ball
        cannot fall through and may get stuck seated in the opening.

    Default / Opt2 (no stick_to_surface): after the gripper releases, the ball has
    ``drop_timeout_s`` (default 2 s) to fall into the box or the episode fails.
    Ball falling off the table (onto the floor) is an immediate failure.
    """

    # ----- tunable params (CLASS DEFAULTS; overridable via task_args.drop_ball_hole) -----
    SPIN_SPEED_DEFAULT = 0.72         # cap angular speed baseline (−20% vs 0.9)
    SPIN_SPEED_JITTER_DEFAULT = 0.20  # sample speed in [1±jitter] * spin_speed
    CAP_HOLE_RADIUS_DEFAULT = 0.03    # radius of the circular opening at the cap center
    CAP_HOLE_DIAMETER_DEFAULT = 2.0 * CAP_HOLE_RADIUS_DEFAULT
    CAP_HOLE_CORNER_MARGIN_DEFAULT = 0.012  # clearance between the hole and the square edges
    CAP_HOLE_BANDS_DEFAULT = 48       # shapely circle resolution for hole cutouts (visual + collision)
    BALL_RADIUS_DEFAULT = 0.025
    BALL_SIDE_CLEARANCE_DEFAULT = 0.14
    BALL_SIDE_DEFAULT = "random"          # random | left | right — which arm operates
    BALL_Y_JITTER_DEFAULT = 0.045
    BALL_X_JITTER_DEFAULT = 0.02          # small ±x randomness on spawn (along side axis)
    BALL_COLOR_DEFAULT = [0.90, 0.10, 0.10]  # red
    TRANSPORT_CLEARANCE_Z_DEFAULT = 0.08
    RELEASE_CLEARANCE_Z_DEFAULT = 0.004
    HOLE_ALIGN_TOL_DEFAULT = 0.015
    ALIGN_SEARCH_STEPS_DEFAULT = 1200
    POST_RELEASE_STEPS_DEFAULT = 220
    # open_gripper advances ~300 physics steps, but the ball leaves the fingers
    # ~40% of the way through (contacts the plate / drops). Lead that fraction.
    RELEASE_OPEN_STEPS_DEFAULT = 300      # fallback full open duration if plan query fails
    # open_gripper ~300 steps; fingers still grip while the hole can skim under.
    # Empirically the free ball appears nearer mid-open than the early contact z.
    RELEASE_FRACTION_DEFAULT = 0.50       # fraction of open plan until ball is free
    RELEASE_FALL_LEAD_STEPS_DEFAULT = 15  # extra lead after separation to fall through
    RELEASE_HOLD_STEPS_DEFAULT = 30       # prefer hole staying under the ball this long
    # Aim slightly upstream of spin so the hole sweeps onto the ball at separation.
    RELEASE_UPSTREAM_M_DEFAULT = 0.008
    # Default / Opt2: wall-clock window after release to enter the box (Opt1 uses stick latch).
    DROP_TIMEOUT_S_DEFAULT = 2.0
    HOLE_DROP_INSET_DEFAULT = 0.006
    HOLE_XY_JITTER_DEFAULT = 0.012        # m; ±x/±y randomization for target & dummy hole centers
    CONTAINER_SHAPE_DEFAULT = "cubic"     # cubic | cylinder
    CYLINDER_DIAMETER_SCALE = 1.20
    CYLINDER_HOLE_INSET_DEFAULT = 0.025   # keep circular hole visibly inside the plate rim
    CYLINDER_HOLE_EDGE_CLEARANCE_DEFAULT = 0.008
    FORCE_PLATFORM_MISS_DEFAULT = False   # demo/eval: drop onto solid / dummy (failure demo)
    # ----- feature toggles (independent; both may be true) -----
    STICK_TO_SURFACE_DEFAULT = False      # Opt1: latch misses onto the platform
    ADD_DUMMY_HOLE_DEFAULT = False        # Opt2: cut near-ball-sized decoy hole
    # ----- Opt1 (stick_to_surface) -----
    STICK_CONTACT_STEPS_DEFAULT = 4       # on-platform frames before latch when no hole overlap
    STICK_SETTLE_STEPS_DEFAULT = 12       # near-rest frames before latch when overlapping hole rim
    STICK_VEL_TOL_DEFAULT = 0.05          # m/s; settle speed threshold
    STICK_Z_TOL_DEFAULT = 0.012           # m; max height above platform rest pose for contact
    # ----- Opt2 (add_dummy_hole) -----
    # Hole diameter ≈ frac * ball diameter; keep frac just under 1 so the ball can wedge in.
    DUMMY_HOLE_RADIUS_FRAC_DEFAULT = 0.92
    DUMMY_HOLE_RADIUS_DEFAULT = None       # absolute m; when set, overrides radius_frac
    DUMMY_HOLE_NEST_DEPTH_DEFAULT = 0.35   # fraction of ball_radius to sink when stuck in dummy hole

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("drop_ball_hole", {})
        self._cap_step = 0
        self._cap_angle = 0.0
        self._cap_tracking = False
        self.ball_in_box = False
        self.ball_released = False
        self.ball_stuck_on_platform = False
        self._drop_timed_out = False
        self._ball_fell_off_table = False
        self._steps_since_release = 0
        self._reset_metric_state()
        self.selected_arm = None
        self.bucket_floor_z = 0.0
        self.hole_orbit_radius = 0.0
        self._drop_target_xy = np.zeros(2, dtype=np.float64)
        self._release_open_steps = self.RELEASE_OPEN_STEPS_DEFAULT
        self._ball_rigid = None
        self._ball_stuck_local = None
        self._platform_contact_steps = 0
        self._platform_settle_steps = 0
        self._dummy_hole_local_xy = None
        self.dummy_hole_radius = 0.0
        # Must exist before _init_task_env_ (it calls _update_kinematic_tasks pre-load_actors).
        self.stick_to_surface = self.STICK_TO_SURFACE_DEFAULT
        self.add_dummy_hole = self.ADD_DUMMY_HOLE_DEFAULT
        self.drop_timeout_s = self.DROP_TIMEOUT_S_DEFAULT
        self.stick_contact_steps = self.STICK_CONTACT_STEPS_DEFAULT
        self.stick_settle_steps = self.STICK_SETTLE_STEPS_DEFAULT
        self.stick_vel_tol = self.STICK_VEL_TOL_DEFAULT
        self.stick_z_tol = self.STICK_Z_TOL_DEFAULT
        self.dummy_hole_radius_frac = self.DUMMY_HOLE_RADIUS_FRAC_DEFAULT
        self.dummy_hole_radius_cfg = self.DUMMY_HOLE_RADIUS_DEFAULT
        self.dummy_hole_nest_depth = self.DUMMY_HOLE_NEST_DEPTH_DEFAULT
        self.hole_xy_jitter = self.HOLE_XY_JITTER_DEFAULT
        self.spin_speed = self.SPIN_SPEED_DEFAULT
        self.spin_speed_jitter = self.SPIN_SPEED_JITTER_DEFAULT
        self.spin_speed_sampled = self.SPIN_SPEED_DEFAULT
        self.spin_omega = 0.0
        self.force_platform_miss = False
        self.ball_color = list(self.BALL_COLOR_DEFAULT)
        self.ball = None
        super()._init_task_env_(**kwags)
        # Platform spin starts here so policy eval sees it without play_once.
        self._cap_tracking = True

    @staticmethod
    def _as_bool(value, default=False):
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
        return bool(default)

    def _parse_surface_features(self):
        """Load independent Opt1/Opt2 toggles and their params."""
        cfg = self._cfg if isinstance(getattr(self, "_cfg", None), dict) else {}

        # Feature toggles (independent — both may be enabled).
        self.stick_to_surface = self._as_bool(
            cfg.get("stick_to_surface", self.STICK_TO_SURFACE_DEFAULT),
            self.STICK_TO_SURFACE_DEFAULT,
        )  # Opt1
        self.add_dummy_hole = self._as_bool(
            cfg.get("add_dummy_hole", self.ADD_DUMMY_HOLE_DEFAULT),
            self.ADD_DUMMY_HOLE_DEFAULT,
        )  # Opt2
        # Default / Opt2 post-release entry window (ignored when Opt1 stick is on).
        self.drop_timeout_s = float(cfg.get("drop_timeout_s", self.DROP_TIMEOUT_S_DEFAULT))

        # Opt1 (stick_to_surface) params
        self.stick_contact_steps = int(cfg.get("stick_contact_steps", self.STICK_CONTACT_STEPS_DEFAULT))
        self.stick_settle_steps = int(cfg.get("stick_settle_steps", self.STICK_SETTLE_STEPS_DEFAULT))
        self.stick_vel_tol = float(cfg.get("stick_vel_tol", self.STICK_VEL_TOL_DEFAULT))
        self.stick_z_tol = float(cfg.get("stick_z_tol", self.STICK_Z_TOL_DEFAULT))

        # Opt2 (add_dummy_hole) params
        self.dummy_hole_radius_frac = float(
            cfg.get("dummy_hole_radius_frac", self.DUMMY_HOLE_RADIUS_FRAC_DEFAULT)
        )
        abs_r = cfg.get("dummy_hole_radius", self.DUMMY_HOLE_RADIUS_DEFAULT)
        self.dummy_hole_radius_cfg = None if abs_r is None else float(abs_r)
        self.dummy_hole_nest_depth = float(
            cfg.get("dummy_hole_nest_depth", self.DUMMY_HOLE_NEST_DEPTH_DEFAULT)
        )

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.container_shape = str(
            cfg.get("container_shape", self.CONTAINER_SHAPE_DEFAULT)
        ).strip().lower()
        if self.container_shape == "square":
            self.container_shape = "cubic"
        if self.container_shape not in ("cubic", "cylinder"):
            self.container_shape = self.CONTAINER_SHAPE_DEFAULT
        self.spin_speed = float(cfg.get("spin_speed", self.SPIN_SPEED_DEFAULT))
        # Relative range around spin_speed, or absolute [spin_speed_min, spin_speed_max] if both set.
        self.spin_speed_jitter = float(cfg.get("spin_speed_jitter", self.SPIN_SPEED_JITTER_DEFAULT))
        self.spin_speed_min_cfg = cfg.get("spin_speed_min", None)
        self.spin_speed_max_cfg = cfg.get("spin_speed_max", None)
        hole_diameter = cfg.get("cap_hole_diameter", None)
        if hole_diameter is None:
            self.cap_hole_radius = float(cfg.get("cap_hole_radius", self.CAP_HOLE_RADIUS_DEFAULT))
        else:
            self.cap_hole_radius = 0.5 * float(hole_diameter)
        self.cap_hole_corner_margin = float(
            cfg.get("cap_hole_corner_margin", self.CAP_HOLE_CORNER_MARGIN_DEFAULT)
        )
        self.cap_hole_bands = int(cfg.get("cap_hole_bands", self.CAP_HOLE_BANDS_DEFAULT))
        self.ball_radius = float(cfg.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_side_clearance = float(cfg.get("ball_side_clearance", self.BALL_SIDE_CLEARANCE_DEFAULT))
        self.ball_side_cfg = str(cfg.get("ball_side", self.BALL_SIDE_DEFAULT)).strip().lower()
        self.ball_y_jitter = float(cfg.get("ball_y_jitter", self.BALL_Y_JITTER_DEFAULT))
        self.ball_x_jitter = float(cfg.get("ball_x_jitter", self.BALL_X_JITTER_DEFAULT))
        ball_color_cfg = cfg.get("ball_color", self.BALL_COLOR_DEFAULT)
        self.ball_color = [float(c) for c in list(ball_color_cfg)[:3]]
        self.force_platform_miss = bool(cfg.get(
            "force_platform_miss", self.FORCE_PLATFORM_MISS_DEFAULT
        ))
        self.transport_clearance_z = float(
            cfg.get("transport_clearance_z", self.TRANSPORT_CLEARANCE_Z_DEFAULT)
        )
        self.release_clearance_z = float(
            cfg.get("release_clearance_z", self.RELEASE_CLEARANCE_Z_DEFAULT)
        )
        self.hole_align_tol = float(cfg.get("hole_align_tol", self.HOLE_ALIGN_TOL_DEFAULT))
        self.align_search_steps = int(cfg.get("align_search_steps", self.ALIGN_SEARCH_STEPS_DEFAULT))
        self.post_release_steps = int(
            cfg.get("post_release_steps", cfg.get("preview_steps", self.POST_RELEASE_STEPS_DEFAULT))
        )
        self.release_open_steps = int(cfg.get("release_open_steps", self.RELEASE_OPEN_STEPS_DEFAULT))
        self.release_fraction = float(
            np.clip(cfg.get("release_fraction", self.RELEASE_FRACTION_DEFAULT), 0.15, 0.95)
        )
        self.release_fall_lead_steps = int(
            cfg.get("release_fall_lead_steps", self.RELEASE_FALL_LEAD_STEPS_DEFAULT)
        )
        self.release_hold_steps = int(
            cfg.get("release_hold_steps", self.RELEASE_HOLD_STEPS_DEFAULT)
        )
        self.release_upstream_m = float(
            cfg.get("release_upstream_m", self.RELEASE_UPSTREAM_M_DEFAULT)
        )
        self.hole_drop_inset = float(cfg.get("hole_drop_inset", self.HOLE_DROP_INSET_DEFAULT))
        self.hole_xy_jitter = float(cfg.get("hole_xy_jitter", self.HOLE_XY_JITTER_DEFAULT))
        self.cylinder_hole_inset = float(max(
            0.0, cfg.get("cylinder_hole_inset", self.CYLINDER_HOLE_INSET_DEFAULT)
        ))
        self.cylinder_hole_edge_clearance = float(max(
            0.0,
            cfg.get(
                "cylinder_hole_edge_clearance",
                self.CYLINDER_HOLE_EDGE_CLEARANCE_DEFAULT,
            ),
        ))
        self._parse_surface_features()
        self._drop_timed_out = False
        self._steps_since_release = 0
        self.ball_released = False
        self._reset_metric_state()

        # Randomized spin direction and speed sampled from a range each episode.
        self.spin_dir = float(np.random.choice([-1.0, 1.0]))
        if self.spin_speed_min_cfg is not None and self.spin_speed_max_cfg is not None:
            spd_lo = float(min(self.spin_speed_min_cfg, self.spin_speed_max_cfg))
            spd_hi = float(max(self.spin_speed_min_cfg, self.spin_speed_max_cfg))
        else:
            jitter = float(np.clip(self.spin_speed_jitter, 0.0, 0.95))
            spd_lo = self.spin_speed * (1.0 - jitter)
            spd_hi = self.spin_speed * (1.0 + jitter)
        self.spin_speed_sampled = float(np.random.uniform(spd_lo, spd_hi))
        self.spin_omega = self.spin_dir * self.spin_speed_sampled

        z0 = 0.74 + self.table_z_bias
        self.table_top_z = z0

        # ---- bucket: a wide, shallow open-top tray at table center-mid ----
        # Wide so each arm's drop spot stays over the cavity; shallow (low walls) so the tall walls
        # don't block the arm reaching over the centre (the deep-bucket version was unplannable).
        self.bucket_center = np.array([0.0, -0.02])     # x, y at center-mid
        self.bucket_half = float(cfg.get("bucket_half", 0.12))  # inner half-extent (xy)
        self.randomize_container_radius = bool(cfg.get("randomize_container_radius", False))
        cr_jitter = float(np.clip(abs(float(cfg.get("container_radius_jitter", 0.15))), 0.0, 0.95))
        if self.randomize_container_radius and cr_jitter > 0.0:
            self.bucket_half = float(np.random.uniform(
                self.bucket_half * (1.0 - cr_jitter),
                self.bucket_half * (1.0 + cr_jitter),
            ))
        self.bucket_h = 0.15                            # support box / rim height
        wall_t = 0.010
        floor_z = z0
        bc = self.bucket_center
        self.bucket_radius = self.bucket_half * (
            self.CYLINDER_DIAMETER_SCALE
            if self.container_shape == "cylinder" else 1.0
        )
        # A round tray uses a polygonal static wall so its interior is hollow.
        if self.container_shape == "cylinder":
            self.bucket_floor = create_cylinder(
                scene=self,
                pose=sapien.Pose([bc[0], bc[1], floor_z + 0.006], [1, 0, 0, 0]),
                radius=self.bucket_radius + wall_t,
                half_length=0.006,
                color=(0.55, 0.4, 0.25), name="cylinder_bucket_floor",
            )
            floor_rigid = self._get_rigid(self.bucket_floor)
            if floor_rigid is not None:
                floor_rigid.set_kinematic(True)
        else:
            self.bucket_floor = create_box(
                scene=self, pose=sapien.Pose([bc[0], bc[1], floor_z + 0.006], [1, 0, 0, 0]),
                half_size=(self.bucket_half + wall_t, self.bucket_half + wall_t, 0.006),
                color=(0.55, 0.4, 0.25), name="bucket_floor", is_static=True,
            )
        self.bucket_floor_z = floor_z + 0.012
        cz = floor_z + 0.012 + self.bucket_h / 2.0
        self.bucket_walls = []
        walls = []
        if self.container_shape == "cylinder":
            segments = 32
            ring_r = self.bucket_radius + wall_t / 2
            segment_length = 2.0 * np.pi * ring_r / segments + 0.004
            for i in range(segments):
                angle = 2.0 * np.pi * i / segments
                walls.append(((
                    bc[0] + ring_r * np.cos(angle),
                    bc[1] + ring_r * np.sin(angle),
                ), (wall_t / 2, segment_length / 2, angle)))
        else:
            walls = [
                ((bc[0], bc[1] + self.bucket_half + wall_t / 2), (self.bucket_half + wall_t, wall_t / 2, 0.0)),
                ((bc[0], bc[1] - self.bucket_half - wall_t / 2), (self.bucket_half + wall_t, wall_t / 2, 0.0)),
                ((bc[0] + self.bucket_half + wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t, 0.0)),
                ((bc[0] - self.bucket_half - wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t, 0.0)),
            ]
        for i, ((wx, wy), (hx, hy, yaw)) in enumerate(walls):
            w = create_box(
                scene=self, pose=sapien.Pose(
                    [wx, wy, cz], t3d.quaternions.axangle2quat([0, 0, 1], yaw)
                ),
                half_size=(hx, hy, self.bucket_h / 2.0),
                color=(0.6, 0.45, 0.3), name=f"bucket_wall{i}", is_static=True,
            )
            self.bucket_walls.append(w)

        self.ball_radius = float(np.clip(self.ball_radius, 0.01, 0.04))
        self.ball_side_clearance = float(max(self.ball_side_clearance, 0.01))
        self.ball_y_jitter = float(np.clip(self.ball_y_jitter, 0.0, self.bucket_half * 0.85))
        self.ball_x_jitter = float(np.clip(self.ball_x_jitter, 0.0, 0.05))
        if self.ball_side_cfg in ("left", "right"):
            self.ball_side = self.ball_side_cfg
        else:
            # random: either arm operates depending on which side the ball spawns
            self.ball_side = "left" if int(np.random.randint(0, 2)) == 0 else "right"
        side_sign = -1.0 if self.ball_side == "left" else 1.0
        # Spawn the ball outside the rotating cap footprint, not just outside the box wall,
        # so the grasp approach stays clear of the platform edge. Small ±x/±y jitter.
        platform_outer_half = self.bucket_radius + 0.012
        ball_x = float(
            bc[0]
            + side_sign * (platform_outer_half + self.ball_radius + self.ball_side_clearance)
            + np.random.uniform(-self.ball_x_jitter, self.ball_x_jitter)
        )
        ball_y = float(bc[1] + np.random.uniform(-self.ball_y_jitter, self.ball_y_jitter))
        self.ball = self._build_ball(
            pose=sapien.Pose([ball_x, ball_y, floor_z + self.ball_radius], [1, 0, 0, 0]),
            color=self.ball_color,
        )
        self._ball_rigid = self._get_rigid(self.ball.actor)

        # height of the cap plane (just above the bucket rim)
        self.cap_z = floor_z + 0.012 + self.bucket_h + 0.015
        self.cap_center = np.array([bc[0], bc[1], self.cap_z])

        # ---- rotating cap: square or circular collidable platform with a real hole ----
        self.cap_half_extent = self.bucket_half + 0.012
        self.cap_radius = self.cap_half_extent * (
            self.CYLINDER_DIAMETER_SCALE
            if self.container_shape == "cylinder" else 1.0
        )
        self.cap_hole_radius = float(np.clip(
            self.cap_hole_radius,
            0.02,
            (self.cap_radius if self.container_shape == "cylinder" else self.cap_half_extent) - 0.02,
        ))
        self.cap_hole_diameter = 2.0 * self.cap_hole_radius
        self.cap_hole_corner_margin = float(np.clip(
            self.cap_hole_corner_margin,
            0.004,
            (self.cap_radius if self.container_shape == "cylinder" else self.cap_half_extent)
            - self.cap_hole_radius - 0.002,
        ))
        self.cap_thickness = 0.006
        self._cap_base_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.hole_xy_jitter = float(np.clip(self.hole_xy_jitter, 0.0, 0.04))
        # Target (real) hole: corner (+,+) with small independent ±x/±y jitter.
        self._cap_hole_local_xy = self._randomized_hole_local_xy(
            radius=self.cap_hole_radius,
            corner_sign=(1.0, 1.0),
        )
        self.hole_orbit_radius = float(np.linalg.norm(self._cap_hole_local_xy))

        # Opt2 (add_dummy_hole): decoy hole just under ball diameter — may wedge, cannot fall through.
        self._dummy_hole_local_xy = None
        self.dummy_hole_radius = 0.0
        if self.add_dummy_hole:
            outer = self.cap_radius if self.container_shape == "cylinder" else self.cap_half_extent
            max_fit = outer - self.cap_hole_corner_margin - 0.002
            # Keep hole radius < ball radius (diameter < ball diameter) but nearly as large.
            max_r = min(self.ball_radius * 0.98, max_fit)
            if self.dummy_hole_radius_cfg is not None:
                self.dummy_hole_radius = float(np.clip(
                    float(self.dummy_hole_radius_cfg),
                    0.008,
                    max_r,
                ))
            else:
                frac = float(np.clip(self.dummy_hole_radius_frac, 0.80, 0.98))
                self.dummy_hole_radius = float(max(
                    0.008,
                    min(self.ball_radius * frac, max_r),
                ))
            # Opposite corner (-,-) with its own ±x/±y jitter.
            self._dummy_hole_local_xy = self._randomized_hole_local_xy(
                radius=self.dummy_hole_radius,
                corner_sign=(-1.0, -1.0),
            )

        self.cap_entity = self._build_cap()
        self._cap_rigid = self._get_rigid(self.cap_entity)
        if self._cap_rigid is not None:
            self._cap_rigid.set_kinematic(True)
        self._place_cap(0.0)
        radial_dir = np.array([side_sign, 0.0], dtype=np.float64)
        self._force_miss_align_dummy = False
        if self.force_platform_miss:
            if self.add_dummy_hole and self._dummy_hole_local_xy is not None:
                # Failure demo targeting Opt2 decoy hole.
                miss_r = float(np.linalg.norm(self._dummy_hole_local_xy))
                self._drop_target_xy = np.array(
                    self.cap_center[:2] + radial_dir * miss_r,
                    dtype=np.float64,
                )
                self._force_miss_align_dummy = True
            else:
                # Failure demo targeting solid platform (Opt1 stick).
                miss_r = max(0.04, min(self.hole_orbit_radius * 0.40, self.cap_half_extent * 0.45))
                self._drop_target_xy = np.array(
                    self.cap_center[:2] + radial_dir * miss_r,
                    dtype=np.float64,
                )
        else:
            self._drop_target_xy = np.array(
                self.cap_center[:2]
                + radial_dir * max(self.hole_orbit_radius - self.hole_drop_inset, 0.0),
                dtype=np.float64,
            )

        # keep clutter / spawn collisions away
        self.add_prohibit_area(self.bucket_floor, padding=0.05)
        self.add_prohibit_area(self.ball, padding=0.03)

    # ----------------------------------------------------- rotating-cap state
    def _randomized_hole_local_xy(self, radius, corner_sign):
        """Corner-biased hole center with slight ±x/±y jitter, clamped on-platform."""
        if self.container_shape == "cylinder":
            # Each opening samples an independent position near the rim, but
            # its full disk always remains inside the circular platform.
            min_orbit = radius + 0.012
            max_orbit = (
                self.cap_radius - radius - self.cap_hole_corner_margin
                - self.cylinder_hole_inset - self.cylinder_hole_edge_clearance
            )
            max_orbit = max(min_orbit, max_orbit)
            orbit = max_orbit
            # The real hole uses (+,+); the dummy uses (-,-). Preserve that
            # opposite-side relationship for the circular plate rather than
            # mapping both same-sign corners onto the same orbit angle.
            angle = np.pi / 4.0 if corner_sign[0] > 0 else -3.0 * np.pi / 4.0
            angle += float(np.random.uniform(-0.16, 0.16))
            radial_jitter = float(np.random.uniform(-self.hole_xy_jitter, self.hole_xy_jitter))
            orbit = float(np.clip(orbit + radial_jitter, min_orbit, max_orbit))
            return orbit * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        sx = 1.0 if float(corner_sign[0]) >= 0.0 else -1.0
        sy = 1.0 if float(corner_sign[1]) >= 0.0 else -1.0
        # Nominal corner placement (same as the previous fixed layout).
        base = float(self.cap_half_extent - radius - self.cap_hole_corner_margin)
        j = float(self.hole_xy_jitter)
        x = base + float(np.random.uniform(-j, j))
        y = base + float(np.random.uniform(-j, j))
        # Keep the full hole disk inside the square; stay away from the center a bit.
        max_abs = float(self.cap_half_extent - radius - 0.002)
        min_abs = float(max(radius + 0.012, base - j))
        min_abs = float(min(min_abs, max_abs))
        x = float(np.clip(x, min_abs, max_abs))
        y = float(np.clip(y, min_abs, max_abs))
        return np.array([sx * x, sy * y], dtype=np.float64)

    def _get_rigid(self, entity):
        for component in entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    def _build_ball(self, pose, color):
        entity = create_sphere(
            self,
            pose=pose,
            radius=self.ball_radius,
            color=color,
            is_static=False,
            name="sorter_ball",
        )
        z_off = 0.35 * self.ball_radius
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "extents": [2.0 * self.ball_radius, 2.0 * self.ball_radius, 2.0 * self.ball_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [
                [[0, 0, 1, 0.0], [1, 0, 0, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[1, 0, 0, 0.0], [0, 0, -1, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[0, 0, -1, 0.0], [-1, 0, 0, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
            ],
            "functional_matrix": [],
        }
        ball_actor = Actor(entity, data, mass=0.1)  # 100 g
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                inelastic = sapien.physx.PhysxMaterial(
                    static_friction=0.9,
                    dynamic_friction=0.9,
                    restitution=0.0,
                )
                for shape in rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass
        return ball_actor

    def _cap_holes(self):
        """Return list of (cx, cy, radius) cutouts in cap-local XY."""
        holes = [(
            float(self._cap_hole_local_xy[0]),
            float(self._cap_hole_local_xy[1]),
            float(self.cap_hole_radius),
        )]
        if (
            self.add_dummy_hole
            and self._dummy_hole_local_xy is not None
            and self.dummy_hole_radius > 1e-6
        ):
            holes.append((
                float(self._dummy_hole_local_xy[0]),
                float(self._dummy_hole_local_xy[1]),
                float(self.dummy_hole_radius),
            ))
        return holes

    def _cap_polygon(self):
        """Cap-top polygon in local XY with true circular hole cutouts (billiard-style)."""
        resolution = max(24, int(self.cap_hole_bands))
        if self.container_shape == "cylinder":
            poly = Point(0.0, 0.0).buffer(float(self.cap_radius), resolution=resolution)
        else:
            he = float(self.cap_half_extent)
            poly = Polygon([(-he, -he), (he, -he), (he, he), (-he, he)])
        for cx, cy, radius in self._cap_holes():
            cut = Point(float(cx), float(cy)).buffer(float(radius), resolution=resolution)
            poly = poly.difference(cut)
        if poly.is_empty:
            raise RuntimeError("cap hole cutouts removed the entire platform")
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly

    def _export_cap_mesh(self, path):
        """Extrude the holed cap polygon to a centered mesh (same as billiard lid)."""
        thickness = 2.0 * float(self.cap_thickness)
        mesh = extrude_polygon(self._cap_polygon(), height=thickness)
        mesh.apply_translation([0.0, 0.0, -0.5 * thickness])
        try:
            mesh.fix_normals()
        except Exception:
            pass
        mesh.export(path)
        return path

    def _build_cap(self):
        """Kinematic rotating plate with smooth circular hole cutouts.

        Uses a shapely disk/square minus circular buffers, extruded to a triangle
        mesh — the same approach as the billiard felt pockets — so openings look
        round instead of stair-stepped box bands.
        """
        tag = "cylinder" if self.container_shape == "cylinder" else "square"
        path = os.path.join(tempfile.gettempdir(), f"drop_ball_hole_cap_{tag}.obj")
        self._export_cap_mesh(path)

        visual_material = sapien.render.RenderMaterial(base_color=[0.45, 0.85, 0.50, 1.0])
        builder = self.scene.create_actor_builder()
        # Triangle-mesh collision is valid for kinematic (not free dynamic) bodies.
        builder.set_physx_body_type("kinematic")
        builder.add_nonconvex_collision_from_file(filename=path)
        builder.add_visual_from_file(filename=path, material=visual_material)
        builder.set_initial_pose(sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]))
        name = "cylinder_sorter_cap" if self.container_shape == "cylinder" else "sorter_cap"
        if hasattr(builder, "build_kinematic"):
            entity = builder.build_kinematic(name=name)
        else:
            entity = builder.build(name=name)
        return entity

    def _place_cap(self, angle):
        """Pose the rotating square platform with its corner pass-through hole."""
        self._cap_angle = float(angle)
        qz = t3d.quaternions.axangle2quat([0, 0, 1], angle)
        q = t3d.quaternions.qmult(qz, self._cap_base_q)
        pose = sapien.Pose(self.cap_center.tolist(), q.tolist())
        if getattr(self, "_cap_rigid", None) is not None and getattr(self, "_cap_tracking", False):
            self._cap_rigid.set_kinematic_target(pose)
        else:
            self.cap_entity.set_pose(pose)

    def _hole_world_xy_at_step(self, step, local_xy=None):
        angle = self._cap_angle_at_step(step)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ], dtype=np.float64)
        local = self._cap_hole_local_xy if local_xy is None else np.asarray(local_xy, dtype=np.float64)
        return self.cap_center[:2] + rot @ local

    def _cap_rot2(self, angle=None):
        angle = float(self._cap_angle if angle is None else angle)
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ], dtype=np.float64)

    def _world_to_cap_local_xy(self, world_xy):
        d = np.asarray(world_xy, dtype=np.float64) - self.cap_center[:2]
        return self._cap_rot2().T @ d

    def _cap_local_to_world_xy(self, local_xy):
        return self.cap_center[:2] + self._cap_rot2() @ np.asarray(local_xy, dtype=np.float64)

    def _ball_overlaps_real_hole(self, ball_xy=None):
        if ball_xy is None:
            ball_xy = np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)
        hole_xy = self.cap_center[:2] + self._cap_rot2() @ self._cap_hole_local_xy
        dist = float(np.linalg.norm(np.asarray(ball_xy, dtype=np.float64) - hole_xy))
        # No overlap <=> ball disk and hole disk are disjoint in XY.
        return bool(dist < (self.cap_hole_radius + self.ball_radius))

    def _ball_in_dummy_hole(self, ball_xy=None):
        """True when the ball center is over the decoy hole enough to wedge in it."""
        if (
            not self.add_dummy_hole
            or self._dummy_hole_local_xy is None
            or self.dummy_hole_radius <= 1e-6
            or getattr(self, "ball", None) is None
        ):
            return False
        if ball_xy is None:
            ball_xy = np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)
        hole_xy = self.cap_center[:2] + self._cap_rot2() @ self._dummy_hole_local_xy
        dist = float(np.linalg.norm(np.asarray(ball_xy, dtype=np.float64) - hole_xy))
        # Center must be inside the dummy opening (ball cannot pass; it seats in the rim).
        return bool(dist <= self.dummy_hole_radius)

    def _ball_on_platform_contact(self):
        """True when the ball is resting on the cap surface (not yet through the hole)."""
        if getattr(self, "ball", None) is None:
            return False
        p = np.asarray(self.ball.get_pose().p, dtype=np.float64)
        rest_z = float(self.cap_z + self.cap_thickness + self.ball_radius)
        # Allow a slightly lower z when the ball is nesting into the dummy hole.
        z_lo = rest_z - 0.55 * self.ball_radius
        if p[2] < z_lo:
            return False
        if p[2] > rest_z + float(self.stick_z_tol):
            return False
        local_xy = self._world_to_cap_local_xy(p[:2])
        margin = self.ball_radius * 0.35
        if self.container_shape == "cylinder":
            return bool(np.linalg.norm(local_xy) <= self.cap_radius + margin)
        return bool(abs(local_xy[0]) <= (self.cap_half_extent + margin)
                    and abs(local_xy[1]) <= (self.cap_half_extent + margin))

    def _ball_linear_speed(self):
        rigid = getattr(self, "_ball_rigid", None)
        if rigid is None:
            return 0.0
        try:
            v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
            return float(np.linalg.norm(v))
        except Exception:
            return 0.0

    def _stick_ball_to_cap(self, nest_in_dummy=False):
        """Latch the ball onto the rotating platform (missed / wedged => fail)."""
        if getattr(self, "ball", None) is None or self.ball_stuck_on_platform:
            return
        p = np.asarray(self.ball.get_pose().p, dtype=np.float64)
        local_xy = self._world_to_cap_local_xy(p[:2])
        if nest_in_dummy and self._dummy_hole_local_xy is not None:
            # Seat into the decoy opening so it reads as jammed, not floating on the rim.
            local_xy = np.asarray(self._dummy_hole_local_xy, dtype=np.float64).copy()
            nest = float(np.clip(self.dummy_hole_nest_depth, 0.0, 0.6)) * self.ball_radius
            rest_z = float(self.cap_z + self.cap_thickness + self.ball_radius - nest)
        else:
            rest_z = float(self.cap_z + self.cap_thickness + self.ball_radius)
        self._ball_stuck_local = np.array(
            [float(local_xy[0]), float(local_xy[1]), rest_z - float(self.cap_center[2])],
            dtype=np.float64,
        )
        self.ball_stuck_on_platform = True
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_linear_velocity([0, 0, 0])
                self._ball_rigid.set_angular_velocity([0, 0, 0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._update_stuck_ball()

    def _update_stuck_ball(self):
        if not self.ball_stuck_on_platform or self._ball_stuck_local is None or self.ball is None:
            return
        world_xy = self._cap_local_to_world_xy(self._ball_stuck_local[:2])
        world_z = float(self.cap_center[2] + self._ball_stuck_local[2])
        pose = sapien.Pose([float(world_xy[0]), float(world_xy[1]), world_z], [1, 0, 0, 0])
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_kinematic_target(pose)
                return
            except Exception:
                pass
        self.ball.actor.set_pose(pose)

    def _try_stick_ball_on_platform(self):
        """Latch platform misses (Opt1 stick_to_surface) and/or dummy-hole wedges (Opt2)."""
        if self.ball_stuck_on_platform or not self.ball_released or self.ball_in_box:
            return
        if getattr(self, "ball", None) is None:
            return
        if not (self.stick_to_surface or self.add_dummy_hole):
            return
        if not self._ball_on_platform_contact():
            self._platform_contact_steps = 0
            self._platform_settle_steps = 0
            return

        # Opt2 (add_dummy_hole): wedge into the decoy hole if the ball settles over it.
        if self.add_dummy_hole and self._ball_in_dummy_hole():
            self._platform_contact_steps = 0
            if self._ball_linear_speed() > float(self.stick_vel_tol):
                self._platform_settle_steps = 0
                return
            self._platform_settle_steps = int(getattr(self, "_platform_settle_steps", 0)) + 1
            if self._platform_settle_steps >= max(1, int(self.stick_settle_steps)):
                if not self._ball_in_box():
                    self._stick_ball_to_cap(nest_in_dummy=True)
            return

        if not self.stick_to_surface:
            # Opt2 only: ignore solid-platform contact; stay dynamic.
            self._platform_contact_steps = 0
            self._platform_settle_steps = 0
            return

        # Opt1 (stick_to_surface)
        overlaps = self._ball_overlaps_real_hole()
        if not overlaps:
            # Solid-platform miss: latch quickly.
            self._platform_settle_steps = 0
            self._platform_contact_steps = int(getattr(self, "_platform_contact_steps", 0)) + 1
            if self._platform_contact_steps >= max(1, int(self.stick_contact_steps)):
                self._stick_ball_to_cap()
            return

        # Overlaps the real hole: stay dynamic until settled (or falls through).
        self._platform_contact_steps = 0
        if self._ball_linear_speed() > float(self.stick_vel_tol):
            self._platform_settle_steps = 0
            return
        self._platform_settle_steps = int(getattr(self, "_platform_settle_steps", 0)) + 1
        if self._platform_settle_steps >= max(1, int(self.stick_settle_steps)):
            if not self._ball_in_box():
                self._stick_ball_to_cap()

    def _estimate_release_lead_steps(self):
        """Physics steps from commanding open_gripper until the ball meets the hole.

        ``open_gripper`` runs ~300 steps, but the ball separates from the fingers
        much earlier (~``release_fraction`` of the plan) and then needs a short
        fall/settle window while the hole stays underneath.
        """
        open_steps = None
        try:
            arm = str(getattr(self, "selected_arm", None) or getattr(self, "ball_side", "right"))
            if arm == "left":
                cur = float(self.robot.get_left_gripper_val())
                n = float(self.robot.left_plan_grippers(cur, 1.0)["num_step"])
            else:
                cur = float(self.robot.get_right_gripper_val())
                n = float(self.robot.right_plan_grippers(cur, 1.0)["num_step"])
            # ``set_gripper`` pads the plan by 50% (see Base_Task.set_gripper).
            open_steps = int(round(n * 1.5))
        except Exception:
            open_steps = None
        if open_steps is None or open_steps < 1:
            open_steps = max(1, int(getattr(self, "release_open_steps", self.RELEASE_OPEN_STEPS_DEFAULT)))
        frac = float(getattr(self, "release_fraction", self.RELEASE_FRACTION_DEFAULT))
        frac = float(np.clip(frac, 0.15, 0.95))
        fall_lead = max(0, int(getattr(self, "release_fall_lead_steps", self.RELEASE_FALL_LEAD_STEPS_DEFAULT)))
        return max(1, int(round(open_steps * frac)) + fall_lead)

    def _hole_err_at(self, step, target_xy, local_xy=None):
        hole_xy = self._hole_world_xy_at_step(int(step), local_xy=local_xy)
        return float(np.linalg.norm(hole_xy - np.asarray(target_xy, dtype=np.float64)))

    def _hole_window_score(self, center_step, target_xy, hold_steps=0, local_xy=None):
        """Worst XY error of the hole vs ``target_xy`` over a short hold window."""
        hold = max(0, int(hold_steps))
        if hold <= 0:
            return self._hole_err_at(center_step, target_xy, local_xy=local_xy)
        # Sample the approach + pass-through; worst error must stay inside the hole.
        errs = [
            self._hole_err_at(center_step + dt, target_xy, local_xy=local_xy)
            for dt in range(0, hold + 1, max(1, hold // 8))
        ]
        return float(max(errs))

    def _steps_until_hole_alignment(
        self,
        target_xy,
        max_steps=None,
        tol=None,
        lead_steps=0,
        local_xy=None,
        hold_steps=None,
    ):
        """Wait until the hole will be under ``target_xy`` after ``lead_steps``.

        When ``hold_steps`` > 0, require the hole to stay near the release point for
        that many steps after the lead (so the ball can fall through, not skim by).
        """
        target_xy = np.array(target_xy, dtype=np.float64)
        max_steps = int(self.align_search_steps if max_steps is None else max_steps)
        tol = float(self.hole_align_tol if tol is None else tol)
        lead_steps = max(0, int(lead_steps))
        if hold_steps is None:
            hold_steps = int(getattr(self, "release_hold_steps", self.RELEASE_HOLD_STEPS_DEFAULT))
        hold_steps = max(0, int(hold_steps))
        # Slightly looser on the hold window than the instantaneous center hit.
        hold_tol = tol + 0.35 * float(getattr(self, "cap_hole_radius", self.CAP_HOLE_RADIUS_DEFAULT))

        best_steps = 0
        best_score = float("inf")
        # When a decoy hole exists, prefer release windows where it is *not*
        # also under the ball (avoids Opt2 wedge / skim onto the dummy).
        avoid_dummy = (
            local_xy is None
            and bool(getattr(self, "add_dummy_hole", False))
            and getattr(self, "_dummy_hole_local_xy", None) is not None
        )
        dummy_clear = float(getattr(self, "ball_radius", 0.025)) + 0.5 * float(
            getattr(self, "dummy_hole_radius", 0.0) or 0.0
        )
        for wait_steps in range(max_steps + 1):
            t = self._cap_step + wait_steps + lead_steps
            center_err = self._hole_err_at(t, target_xy, local_xy=local_xy)
            window_err = self._hole_window_score(
                t, target_xy, hold_steps=hold_steps, local_xy=local_xy
            )
            # Prefer a tight center hit that also stays under the ball briefly.
            score = center_err + 0.35 * max(0.0, window_err - tol)
            if avoid_dummy:
                dummy_err = self._hole_err_at(
                    t, target_xy, local_xy=self._dummy_hole_local_xy
                )
                if dummy_err < dummy_clear:
                    score += (dummy_clear - dummy_err) + 0.02
            if score < best_score:
                best_score = score
                best_steps = wait_steps
            if center_err <= tol and window_err <= hold_tol:
                if (not avoid_dummy) or (
                    self._hole_err_at(t, target_xy, local_xy=self._dummy_hole_local_xy)
                    >= dummy_clear
                ):
                    return wait_steps
        return best_steps

    def _release_aim_xy(self, target_xy):
        """Shift the aim slightly upstream against spin so the hole sweeps onto the ball."""
        target_xy = np.asarray(target_xy, dtype=np.float64)
        upstream = float(getattr(self, "release_upstream_m", self.RELEASE_UPSTREAM_M_DEFAULT))
        if upstream <= 1e-6 or abs(float(getattr(self, "spin_omega", 0.0))) < 1e-9:
            return target_xy
        center = np.asarray(self.cap_center[:2], dtype=np.float64)
        radial = target_xy - center
        n = float(np.linalg.norm(radial))
        if n < 1e-6:
            return target_xy
        # Tangential unit (+CCW). Move aim upstream against the hole's travel.
        tang = np.array([-radial[1], radial[0]], dtype=np.float64) / n
        if float(self.spin_omega) < 0.0:
            tang = -tang
        return target_xy - upstream * tang

    def _steps_per_revolution(self):
        omega = abs(float(getattr(self, "spin_omega", 0.0))) + 1e-6
        return int(max(200, min(2500, round(2.0 * np.pi / (omega * 0.01)))))

    def _reactive_plate_drop(self, max_passes=2):
        """Second-chance drop: wait for the hole to pass under the ball on the plate.

        The open_gripper motion often drags the ball; the spinning plate can also
        slide it. Track *live* ball XY every step (closed-loop) until the real hole
        overlaps, then hold briefly so it can fall.

        Call this *before* ``mark_ball_released`` in Default/Opt2 so the drop-timeout
        clock does not expire while we wait for the next hole pass.
        """
        if getattr(self, "force_platform_miss", False):
            return
        rev = self._steps_per_revolution()
        hold = max(20, int(getattr(self, "release_hold_steps", 30)))
        # Looser than pre-release tol: ball may sit slightly off the geometric center.
        tol = float(self.hole_align_tol) + 0.55 * float(self.cap_hole_radius)
        for _ in range(int(max_passes)):
            if self._ball_in_box() or getattr(self, "ball_stuck_on_platform", False):
                return
            if not self._ball_on_platform_contact():
                for _ in range(40):
                    self._dwell(1)
                    if self._ball_in_box() or self._ball_on_platform_contact():
                        break
                if self._ball_in_box():
                    return
                if not self._ball_on_platform_contact():
                    return
            # Closed-loop: re-read ball XY each step (plate spin can drag it).
            saw_overlap = False
            for _ in range(rev + hold):
                self._dwell(1)
                if self._ball_in_box():
                    return
                if not self._ball_on_platform_contact():
                    if self._ball_in_box():
                        return
                    break
                ball_xy = np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)
                err = self._hole_err_at(self._cap_step, ball_xy, local_xy=None)
                if err <= tol:
                    saw_overlap = True
                    # Stay through the overlap so the ball can drop.
                    for _ in range(hold):
                        self._dwell(1)
                        if self._ball_in_box():
                            return
                        if not self._ball_on_platform_contact():
                            return
                    break
            if self._ball_in_box():
                return
            if not saw_overlap:
                return

    def _micro_correct_release_xy(self, arm_tag: ArmTag, target_xy, max_shift=0.018):
        """Nudge the grasped ball toward ``target_xy`` (keeps timing prediction honest)."""
        if getattr(self, "ball", None) is None:
            return np.asarray(target_xy, dtype=np.float64)
        cur = np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)
        delta = np.asarray(target_xy, dtype=np.float64) - cur
        dist = float(np.linalg.norm(delta))
        if dist < 2.5e-3:
            return cur
        if dist > float(max_shift):
            delta = delta * (float(max_shift) / dist)
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(delta[0]),
            y=float(delta[1]),
        ))
        return np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)

    def _wait_for_release_alignment(self, target_xy, local_xy=None, arm_tag=None):
        """Dwell until a predicted release window; optionally nudge XY onto the aim point."""
        lead = self._estimate_release_lead_steps()
        aim_xy = self._release_aim_xy(target_xy)
        wait = self._steps_until_hole_alignment(
            aim_xy, lead_steps=lead, local_xy=local_xy
        )
        self._dwell(wait)
        live_xy = np.array(
            self.ball.get_pose().p[:2] if getattr(self, "ball", None) is not None else target_xy,
            dtype=np.float64,
        )
        # Small XY nudge toward the upstream aim (skip under Opt1 stick — it
        # desynchronizes the open-lead estimate and hurts first-shot timing).
        if (
            arm_tag is not None
            and local_xy is None
            and (not bool(getattr(self, "stick_to_surface", False)))
        ):
            live_xy = self._micro_correct_release_xy(arm_tag, self._release_aim_xy(live_xy))
            lead = self._estimate_release_lead_steps()
        aim_xy = self._release_aim_xy(live_xy)
        refine = self._steps_until_hole_alignment(
            aim_xy,
            lead_steps=lead,
            local_xy=local_xy,
            max_steps=160,
            hold_steps=max(12, int(getattr(self, "release_hold_steps", 30)) // 2),
        )
        if refine > 0:
            self._dwell(refine)
        return np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)

    def _move_ball_to_height(self, arm_tag: ArmTag, target_z: float):
        if getattr(self, "ball", None) is None:
            return
        dz = float(target_z) - float(self.ball.get_pose().p[2])
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz, move_axis="world"))

    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            self.ball_in_box = self._ball_in_box()
            if (
                getattr(self, "_ball_fell_off_table", False)
                or getattr(self, "ball_stuck_on_platform", False)
            ):
                break
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _ball_in_box(self):
        """True iff the ball is resting inside the bucket cavity (below the cap).

        The only open path into the box is through a cap hole. The target hole is
        large enough for the ball; the Opt2 dummy hole is not. Stuck-on-platform
        balls are rejected separately in ``check_success``.
        """
        if getattr(self, "ball", None) is None:
            return False
        if getattr(self, "ball_stuck_on_platform", False):
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        if self.container_shape == "cylinder":
            in_x = np.linalg.norm(p[:2] - self.bucket_center) <= (self.bucket_radius - 0.15 * self.ball_radius)
            in_y = True
        else:
            in_x = abs(p[0] - self.bucket_center[0]) <= (self.bucket_half - 0.15 * self.ball_radius)
            in_y = abs(p[1] - self.bucket_center[1]) <= (self.bucket_half - 0.15 * self.ball_radius)
        above_floor = p[2] >= (self.bucket_floor_z + 0.2 * self.ball_radius)
        below_platform = p[2] <= (self.cap_z - self.cap_thickness - 0.01)
        return bool(in_x and in_y and above_floor and below_platform)

    def _ball_off_table(self):
        """True when the free ball has fallen off the tabletop onto the floor.

        In-box success and stuck-on-platform are excluded. Table footprint matches
        ``create_table`` (length=1.2, width=0.7).
        """
        if getattr(self, "ball", None) is None:
            return False
        if getattr(self, "ball_stuck_on_platform", False):
            return False
        if self._ball_in_box():
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        table_z = float(getattr(self, "table_top_z", 0.74 + float(self.table_z_bias)))
        # Clearly below the tabletop → fell to the floor / under the table.
        if float(p[2]) < table_z - 0.05:
            return True
        bias = getattr(self, "table_xy_bias", [0.0, 0.0])
        half_x, half_y = 0.60, 0.35
        margin = 0.03
        off_xy = (
            abs(float(p[0]) - float(bias[0])) > half_x + margin
            or abs(float(p[1]) - float(bias[1])) > half_y + margin
        )
        # Past the table rim and not held high (e.g. rolled/bounced off the edge).
        near_surface = float(p[2]) < table_z + float(getattr(self, "ball_radius", 0.025)) + 0.06
        return bool(off_xy and near_surface)

    def _try_latch_ball_off_table(self):
        """Latch off-table failure and mark the episode terminal for eval."""
        if getattr(self, "_ball_fell_off_table", False):
            return True
        if not self._ball_off_table():
            return False
        self._ball_fell_off_table = True
        self.ball_in_box = False
        self._last_fail_reason = "ball dropped off the table"
        self.eval_fail = True
        return True

    def _uses_drop_timeout(self):
        """Default / Opt2: require entry within ``drop_timeout_s`` after release.

        Opt1 / Opt1+2 use ``stick_to_surface`` latching instead of a wall-clock cutoff.
        """
        return not bool(getattr(self, "stick_to_surface", False))

    def _drop_timeout_steps(self):
        dt = float(self.scene.get_timestep()) if getattr(self, "scene", None) is not None else 1.0 / 250.0
        return max(1, int(round(float(self.drop_timeout_s) / max(dt, 1e-6))))

    def mark_ball_released(self):
        """Start the post-release clock (Default / Opt2 timeout) after the gripper opens."""
        self.ball_released = True
        self._steps_since_release = 0
        self._drop_timed_out = False

    def _tick_drop_timeout(self):
        """Advance Default/Opt2 release timer; latch failure after ``drop_timeout_s``."""
        if not self._uses_drop_timeout():
            return
        if not self.ball_released or self.ball_in_box or self.ball_stuck_on_platform:
            return
        if getattr(self, "_ball_fell_off_table", False):
            return
        if self._drop_timed_out:
            return
        self._steps_since_release = int(getattr(self, "_steps_since_release", 0)) + 1
        if self._steps_since_release >= self._drop_timeout_steps():
            self._drop_timed_out = True
            self._last_fail_reason = (
                f"ball not in box within {float(self.drop_timeout_s):.1f}s after release"
            )

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()
        # Guard: _init_task_env_ calls this before load_actors.
        if getattr(self, "ball", None) is None or not hasattr(self, "cap_z"):
            return
        if getattr(self, "_cap_tracking", False):
            self._cap_step += 1
            # step-driven angle => identical in plan & render passes
            angle = self.spin_omega * (self._cap_step * 0.01)
            self._place_cap(angle)
        if getattr(self, "ball_stuck_on_platform", False):
            self._update_stuck_ball()
            self.ball_in_box = False
            self._tick_drop_timeout()
            return
        if self._try_latch_ball_off_table():
            self._tick_drop_timeout()
            return
        self._try_stick_ball_on_platform()
        if getattr(self, "ball_stuck_on_platform", False):
            self.ball_in_box = False
        else:
            self.ball_in_box = self._ball_in_box()
        self._try_latch_ball_off_table()
        self._tick_drop_timeout()
        self._track_release_metric()
        self._track_in_box_metric()

    def _cap_angle_at_step(self, step):
        return self.spin_omega * (step * 0.01)

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_name = "left" if self.ball_side == "left" else "right"
        arm_tag = ArmTag(arm_name)
        self.selected_arm = arm_name
        self._cap_tracking = True
        self.move(self.grasp_actor(self.ball, arm_tag=arm_tag, pre_grasp_dis=0.08))

        transport_z = float(
            self.cap_z + self.cap_thickness + self.ball_radius + self.transport_clearance_z
        )
        release_z = float(
            self.cap_z + self.cap_thickness + self.ball_radius + self.release_clearance_z
        )
        self._move_ball_to_height(arm_tag=arm_tag, target_z=transport_z)

        ball_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        delta_xy = self._drop_target_xy - ball_xy
        if np.linalg.norm(delta_xy) > 1e-4:
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            ))

        self._move_ball_to_height(arm_tag=arm_tag, target_z=release_z)
        release_target_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        if self.force_platform_miss and getattr(self, "_force_miss_align_dummy", False):
            # Opt2 error: wait until the decoy hole is under the release point.
            self._wait_for_release_alignment(
                release_target_xy, local_xy=self._dummy_hole_local_xy, arm_tag=None
            )
        elif not self.force_platform_miss:
            # Aim where the real hole will be when the gripper finishes opening + fall.
            self._wait_for_release_alignment(
                release_target_xy, local_xy=None, arm_tag=arm_tag
            )
        else:
            # Opt1 solid-platform miss: brief dwell, no hole wait.
            self._dwell(30)
        self.move(self.open_gripper(arm_tag))
        self._dwell(25)
        # Second-chance: wait for the hole under the live on-plate ball *before*
        # arming Opt1 stick / Default-Opt2 drop-timeout. Stick only latches once
        # ``mark_ball_released`` is called, so Opt1 can still use this window.
        if not self.force_platform_miss:
            self._reactive_plate_drop(max_passes=2)
        self.mark_ball_released()
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        if self._uses_drop_timeout():
            # Wait until the ball is in the box, stuck (Opt2 dummy), or the 2 s window ends.
            max_wait = self._drop_timeout_steps() + 40
            for i in range(max_wait):
                self._update_kinematic_tasks()
                self.scene.step()
                if self.save_freq and (i % self.save_freq == 0):
                    self._take_picture()
                if (
                    self.ball_in_box
                    or self._drop_timed_out
                    or getattr(self, "ball_stuck_on_platform", False)
                    or getattr(self, "_ball_fell_off_table", False)
                ):
                    break
            if self.ball_in_box:
                self._dwell(30)
        else:
            self._dwell(self.post_release_steps)
        self._cap_tracking = False
        if (
            getattr(self, "ball_stuck_on_platform", False)
            or self._drop_timed_out
            or getattr(self, "_ball_fell_off_table", False)
        ):
            self.ball_in_box = False
        else:
            self.ball_in_box = self._ball_in_box()

        self.info["info"] = {
            "{A}": "rotating sorter platform",
            "{B}": "circular hole",
            "{C}": "bucket",
            "{D}": "sorter_ball",
            "{a}": arm_name,
        }
        return self.info

    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_release_step = None
        self._metric_in_box_step = None
        self._metric_aim_offset_norm = None
        self._metric_ball_held = False
        self._metric_free_steps = 0
        self._metric_over_cap_step = None

    # Frames of no gripper/ball contact before the drop counts as real (finger
    # contact flickers by a frame or two as the jaws part).
    RELEASE_DEBOUNCE_STEPS = 4

    def _track_release_metric(self):
        """Detect the ball actually leaving the gripper and latch the aim error.

        ``mark_ball_released`` is NOT usable as the anchor: the expert calls it
        after a second-chance alignment pass, by which time the ball may already
        be in the box. Contact loss is the true hand-off and works for teleop too.
        """
        if getattr(self, "_metric_release_step", None) is not None:
            return
        if getattr(self, "ball", None) is None:
            return
        try:
            held = len(self.get_gripper_actor_contact_position(self.ball.get_name())) > 0
        except Exception:
            return
        if held:
            self._metric_ball_held = True
            self._metric_free_steps = 0
            if getattr(self, "_metric_over_cap_step", None) is None:
                try:
                    if float(self.ball.get_pose().p[2]) >= float(
                        self.cap_z + self.cap_thickness + self.ball_radius
                    ):
                        self._metric_over_cap_step = int(
                            getattr(self, "_exp_sim_steps", 0) or 0)
                except Exception:
                    pass
            return
        if not getattr(self, "_metric_ball_held", False):
            return
        self._metric_free_steps = int(getattr(self, "_metric_free_steps", 0)) + 1
        if self._metric_free_steps < int(self.RELEASE_DEBOUNCE_STEPS):
            return
        self._latch_release_metrics()

    def _latch_release_metrics(self):
        """Record the drop instant and how well the ball was aimed at the hole.

        The aim error is captured HERE, not at the end: the cap keeps spinning, so
        the hole is somewhere else entirely by the time the episode finishes.
        Back-dated to the first contact-free frame.
        """
        if getattr(self, "_metric_release_step", None) is not None:
            return
        now = int(getattr(self, "_exp_sim_steps", 0) or 0)
        free = int(getattr(self, "_metric_free_steps", 0) or 0)
        self._metric_release_step = max(0, now - max(0, free - 1))
        try:
            ball_xy = np.asarray(self.ball.get_pose().p[:2], dtype=np.float64)
            hole_xy = self.cap_center[:2] + self._cap_rot2() @ self._cap_hole_local_xy
            dist = float(np.linalg.norm(ball_xy - hole_xy))
            self._metric_aim_offset_norm = dist / max(float(self.cap_hole_radius), 1e-9)
        except Exception:
            self._metric_aim_offset_norm = None

    def _track_in_box_metric(self):
        """Latch the first frame the ball rested inside the bucket.

        High-water mark: the ball can bounce, so only the first True is kept.
        """
        if getattr(self, "_metric_in_box_step", None) is not None:
            return
        if not bool(getattr(self, "ball_in_box", False)):
            return
        self._metric_in_box_step = int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _compute_metrics(self):
        """extra1 = hover->release latency, extra2 = aim error in hole radii.

        The decisive act is WHEN the gripper opens: the cap spins, so the drop
        must be timed to the moving hole. ``release_latency`` runs from the moment
        the held ball first cleared the cap surface (aiming window opens) to the
        instant it left the gripper — i.e. how long the operator deliberated.
        Release->in-box is reported too, but it is near-constant free fall from a
        low release height, so it does not discriminate.

        ``aim_offset_norm`` is the ball-centre to hole-centre distance at release,
        as a fraction of the cap hole radius: 0.0 = dead over the hole, 1.0 = over
        the rim, >1.0 = released over solid cap. LOWER IS BETTER. None when the
        ball was never released.
        """
        metrics = {
            "release_latency_steps": None,
            "release_latency_s": None,
            "in_box_latency_steps": None,
            "in_box_latency_s": None,
            "aim_offset_norm": (
                None
                if getattr(self, "_metric_aim_offset_norm", None) is None
                else float(self._metric_aim_offset_norm)
            ),
        }
        dt = None
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = None
        hover = getattr(self, "_metric_over_cap_step", None)
        released = getattr(self, "_metric_release_step", None)
        landed = getattr(self, "_metric_in_box_step", None)
        if hover is not None and released is not None and released >= hover:
            steps = int(released - hover)
            metrics["release_latency_steps"] = steps
            if dt is not None:
                metrics["release_latency_s"] = round(steps * dt, 6)
        if released is not None and landed is not None and landed >= released:
            steps = int(landed - released)
            metrics["in_box_latency_steps"] = steps
            if dt is not None:
                metrics["in_box_latency_s"] = round(steps * dt, 6)
        return metrics

    def check_success(self):
        """Success: ball fell through the target hole and rests inside the box."""
        self._try_latch_ball_off_table()
        stuck = bool(getattr(self, "ball_stuck_on_platform", False))
        timed_out = bool(getattr(self, "_drop_timed_out", False))
        off_table = bool(getattr(self, "_ball_fell_off_table", False))
        in_box = (not stuck) and (not off_table) and self._ball_in_box()
        self.ball_in_box = bool(in_box)
        if off_table:
            self._last_fail_reason = "ball dropped off the table"
        elif timed_out and not in_box:
            self._last_fail_reason = (
                f"ball not in box within {float(self.drop_timeout_s):.1f}s after release"
            )
        elif stuck:
            self._last_fail_reason = "ball stuck on platform"
        self.info["ball_side"] = str(getattr(self, "ball_side", "left"))
        self.info["selected_arm"] = str(getattr(self, "selected_arm", "left"))
        self.info["ball_in_box"] = bool(self.ball_in_box)
        self.info["ball_stuck_on_platform"] = stuck
        self.info["ball_off_table"] = bool(off_table)
        self.info["drop_timed_out"] = bool(timed_out and not in_box)
        self.info["drop_timeout_s"] = float(
            getattr(self, "drop_timeout_s", self.DROP_TIMEOUT_S_DEFAULT)
        )
        self.info["stick_to_surface"] = bool(
            getattr(self, "stick_to_surface", self.STICK_TO_SURFACE_DEFAULT)
        )
        self.info["add_dummy_hole"] = bool(
            getattr(self, "add_dummy_hole", self.ADD_DUMMY_HOLE_DEFAULT)
        )
        return bool(self.ball_in_box)

    # ----------------------------------------------------------------- obs
    def get_obs(self):
        obs = super().get_obs()
        obs["sorter"] = {
            "cap_angle": float(getattr(self, "_cap_angle", 0.0)),
            "spin_omega": float(getattr(self, "spin_omega", 0.0)),
            "spin_speed": float(getattr(self, "spin_speed", self.SPIN_SPEED_DEFAULT)),
            "spin_speed_sampled": float(getattr(self, "spin_speed_sampled", 0.0)),
            "spin_speed_jitter": float(getattr(self, "spin_speed_jitter", self.SPIN_SPEED_JITTER_DEFAULT)),
            "cap_hole_radius": float(getattr(self, "cap_hole_radius", self.CAP_HOLE_RADIUS_DEFAULT)),
            "cap_hole_diameter": float(getattr(self, "cap_hole_diameter", self.CAP_HOLE_DIAMETER_DEFAULT)),
            "ball_radius": float(getattr(self, "ball_radius", self.BALL_RADIUS_DEFAULT)),
            "ball_side": str(getattr(self, "ball_side", "left")),
            "ball_color": list(getattr(self, "ball_color", [0.0, 0.0, 0.0])),
            "ball_center": self.ball.get_pose().p.tolist() if getattr(self, "ball", None) is not None else [0.0, 0.0, 0.0],
            "drop_target_xy": self._drop_target_xy.tolist() if hasattr(self, "_drop_target_xy") else [0.0, 0.0],
            "ball_in_box": bool(getattr(self, "ball_in_box", False)),
            "ball_stuck_on_platform": bool(getattr(self, "ball_stuck_on_platform", False)),
            "ball_off_table": bool(getattr(self, "_ball_fell_off_table", False)),
            "drop_timed_out": bool(getattr(self, "_drop_timed_out", False)),
            "drop_timeout_s": float(getattr(self, "drop_timeout_s", self.DROP_TIMEOUT_S_DEFAULT)),
            "stick_to_surface": bool(getattr(self, "stick_to_surface", self.STICK_TO_SURFACE_DEFAULT)),
            "add_dummy_hole": bool(getattr(self, "add_dummy_hole", self.ADD_DUMMY_HOLE_DEFAULT)),
            "dummy_hole_radius": float(getattr(self, "dummy_hole_radius", 0.0)),
            "dummy_hole_radius_frac": float(getattr(self, "dummy_hole_radius_frac", self.DUMMY_HOLE_RADIUS_FRAC_DEFAULT)),
            "stick_contact_steps": int(getattr(self, "stick_contact_steps", self.STICK_CONTACT_STEPS_DEFAULT)),
            "stick_settle_steps": int(getattr(self, "stick_settle_steps", self.STICK_SETTLE_STEPS_DEFAULT)),
        }
        return obs
