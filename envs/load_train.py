from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import sapien.render
import numpy as np
import transforms3d as t3d


class load_train(Base_Task):
    """Pick up a ball in front of the robot and drop it into a passing toy train.

    A circular rail is centered on the table. A closed locomotive leads three open
    wagons around the loop (wagon count configurable; default 3). The locomotive
    starts at 12 o'clock (far side, opposite the arms) and the train moves
    continuously. The near arc passes in front of the robot. A graspable ball
    spawns randomly on the left (−x) or right (+x) near side each episode; the
    matching arm grasps it, carries it to a drop station above the near rail, times
    the release so an open wagon is underneath, and seats the ball in that wagon —
    while the train keeps moving.

    Options (independent toggles; CLI via ``--task-arg`` or legacy ``--option``):
      Default — three open wagons (all red); drop into any wagon succeeds.
      Opt 1 — ``target_wagon_mode``: pick one wagon at random as the target; that
        wagon is red and the other two are gray; success only if the ball lands in
        the red wagon.
        CLI: ``--task-arg target_wagon_mode=true`` or ``--option 1``.
      Opt 2 — ``tunnel_enabled``: matching arched tunnels on the far and near
        arcs (2–4 wagon spans; near tunnel is a 180° mirror of the far one).
        CLI: ``--task-arg tunnel_enabled=true`` or ``--option 2``.
        ``tunnel_enabled: random`` still samples with ``tunnel_prob``.
      Opt 1+2 — target red wagon plus both tunnels (both flags together).

    Train motion is step-driven in `_update_kinematic_tasks` so plan and render
    passes stay identical. Once the ball is seated in a wagon it is latched and
    rides with that car.
    """

    # ---- geometry / motion defaults (override via task_args.load_train) ----
    RAIL_CENTER_X_DEFAULT = 0.0
    RAIL_CENTER_Y_DEFAULT = 0.04
    RAIL_RADIUS_DEFAULT = 0.20
    RAIL_SEGMENTS_DEFAULT = 48
    RAIL_HALF_W_DEFAULT = 0.018
    RAIL_THICK_DEFAULT = 0.008
    RAIL_WALL_H_DEFAULT = 0.012

    N_WAGONS_DEFAULT = 3                # open cargo wagons behind the locomotive
    N_WAGONS_MIN_DEFAULT = 3
    N_WAGONS_MAX_DEFAULT = 3
    TRAIN_OMEGA_MIN_DEFAULT = 0.12      # rad / (step * 0.01); per-episode speed sampled in [min, max]
    TRAIN_OMEGA_MAX_DEFAULT = 0.22
    TRAIN_OMEGA_DEFAULT = 0.17          # used only if min/max unset
    CAR_ARC_SPACING_DEFAULT = 0.10      # meters along the rail between car centers
    CAR_HALF_LEN_DEFAULT = 0.045
    CAR_HALF_WID_DEFAULT = 0.040
    CAR_WALL_H_DEFAULT = 0.036
    CAR_FLOOR_H_DEFAULT = 0.014       # thick visible bed so wagons read as open boxes
    CAR_WALL_T_DEFAULT = 0.005
    ENGINE_CAB_H_DEFAULT = 0.024
    ENGINE_BODY_H_DEFAULT = 0.036       # closed locomotive height (no cargo opening)

    BALL_RADIUS_DEFAULT = 0.0195
    BALL_SIDE_CLEARANCE_DEFAULT = 0.11  # keep ball clear of passing wagons before the grasp
    BALL_LATERAL_DEFAULT = 0.11         # nominal |x| of ball spawn (same side as drop)
    BALL_LATERAL_MIN_DEFAULT = 0.07     # randomize |x| within the chosen arm's half
    BALL_LATERAL_MAX_DEFAULT = 0.13
    BALL_Y_JITTER_DEFAULT = 0.02        # randomize y around the near-rail spawn line
    BALL_SIDE_DEFAULT = "random"        # "random" | "left" | "right"

    TRANSPORT_CLEARANCE_Z_DEFAULT = 0.22   # carry well above loco + wagons (avoid hitting train)
    RELEASE_CLEARANCE_Z_DEFAULT = 0.20     # release from nearly the same high pose; ball falls in
    PASS_ANGLE_OFFSET_DEFAULT = 0.22    # side-biased drop so left/right arms can reach the rail
    ALIGN_TOL_DEFAULT = 0.055
    ALIGN_SEARCH_STEPS_DEFAULT = 2800
    # open_gripper alone advances ~300 physics steps; fall lead is added at runtime
    RELEASE_OPEN_STEPS_DEFAULT = 300
    POST_RELEASE_STEPS_DEFAULT = 500
    LATCH_XY_MARGIN_DEFAULT = 0.002
    LATCH_Z_MAX_DEFAULT = 0.015         # only seat after the ball is near the bed floor
    LATCH_SETTLE_STEPS_DEFAULT = 12     # consecutive near-floor frames before latching
    CATCH_RADIUS_DEFAULT = 0.055
    SIM_HZ = 250.0
    GRAVITY = 9.81
    # Max allowed miss between commanded drop target and where the ball actually ends up
    DROP_REACH_TOL_DEFAULT = 0.030

    # Opt 1 — designate one random open wagon as the drop target (red vs gray)
    TARGET_WAGON_MODE_DEFAULT = False
    WAGON_COLOR_TARGET = [0.82, 0.22, 0.18, 1.0]   # red (default / target)
    WAGON_COLOR_DISTRACTOR = [0.55, 0.55, 0.58, 1.0]  # gray (non-target under Opt 1)

    # Opt 2 — far-arc tunnel plus a 180° mirror on the near arc (each 2–4 wagons)
    TUNNEL_ENABLED_DEFAULT = False      # true | false | random (Opt 2 when true)
    TUNNEL_PROB_DEFAULT = 0.55
    TUNNEL_N_WAGONS_MIN_DEFAULT = 2
    TUNNEL_N_WAGONS_MAX_DEFAULT = 4
    TUNNEL_CENTER_JITTER_DEFAULT = 0.40  # rad L/R of +pi/2 (12 o'clock)
    TUNNEL_CLEARANCE_Z_DEFAULT = 0.10    # opening height above rail surface
    TUNNEL_WALL_T_DEFAULT = 0.010
    TUNNEL_ROOF_T_DEFAULT = 0.012
    TUNNEL_OVERHANG_DEFAULT = 0.018      # radial overhang past the rail walls

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("load_train", {})
        # Clear per-episode state BEFORE _init_task_env_: load_camera calls
        # _update_kinematic_tasks before load_actors rebuilds actors.
        self._loaded = False
        self._train_running = False
        self._train_step = 0
        self._train_angle = 0.0
        self._ball_released = False
        self._ball_latched = False
        self._latched_car_idx = None
        self._latch_local = None
        self._bed_contact_steps = 0
        self._ball_fell_off_table = False
        self.selected_arm = None
        self.ball_in_train = False
        self.target_wagon_mode = False
        self.target_wagon_idx = None
        self.tunnel_present = False
        self.tunnel_n_wagons = 0
        self.tunnel_center_angle = None
        self.tunnel_center_angle_near = None
        self.cars = []
        self._car_rigids = []
        self.ball = None
        self._ball_rigid = None
        self._reset_metrics()
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()
        # Start the train after setup so it also runs during policy eval (no play_once).
        self._train_running = True

    # ------------------------------------------------------------------ options
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
        raise ValueError(f"load_train expected a boolean, got {value!r}")

    def _parse_target_wagon_mode(self, c) -> bool:
        """Opt 1: one randomly selected target wagon (red); others gray."""
        mode = c.get(
            "target_wagon_mode",
            c.get("select_target_wagon", c.get("opt1", None)),
        )
        legacy = c.get("option", None)
        if legacy is not None and mode is None:
            if legacy in (
                1, "1", "target_wagon_mode", "select_target_wagon", "target_wagon",
            ):
                mode = True
            elif legacy in (2, "2", "tunnel_enabled", "tunnel"):
                mode = False
            else:
                raise ValueError(
                    "load_train option must be 1/target_wagon_mode or "
                    "2/tunnel_enabled (or set the named keys directly)"
                )
        return self._as_bool(mode, self.TARGET_WAGON_MODE_DEFAULT)

    def _parse_tunnel_enabled_cfg(self, c):
        """Opt 2: return raw tunnel_enabled config (bool / 'random' / str).

        Legacy ``--option 2`` forces the tunnel on. Named ``tunnel_enabled`` wins
        when set explicitly alongside ``option``.
        """
        tunnel = c.get("tunnel_enabled", c.get("opt2", None))
        legacy = c.get("option", None)
        if legacy is not None and tunnel is None:
            if legacy in (2, "2", "tunnel_enabled", "tunnel"):
                return True
            if legacy in (
                1, "1", "target_wagon_mode", "select_target_wagon", "target_wagon",
            ):
                return False
            raise ValueError(
                "load_train option must be 1/target_wagon_mode or "
                "2/tunnel_enabled (or set the named keys directly)"
            )
        if tunnel is None:
            return self.TUNNEL_ENABLED_DEFAULT
        return tunnel

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "target_wagon_mode", False):
            parts.append("option 1")
        if getattr(self, "tunnel_present", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _configure_observer_camera(self):
        """Third-person view on the circular rail (FOV zoom + high-res).

        Starts from a 2× optical zoom, then zooms out 1.5× for wider context
        (effective ~1.33× vs the default observer FOV).
        """
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        old = cams.observer_camera
        near = float(old.near) if hasattr(old, "near") else 0.1
        far = float(old.far) if hasattr(old, "far") else 100.0
        old_fovy = float(old.fovy) if hasattr(old, "fovy") else np.deg2rad(93.0)
        # 2× zoom-in, then 1.5× zoom-out → divide tan by (2/1.5).
        zoom_factor = 2.0 / 1.5
        zoom_fovy = 2.0 * float(np.arctan(np.tan(old_fovy / 2.0) / zoom_factor))
        try:
            self.scene.remove_camera(old)
        except Exception:
            pass
        # High-res third-person capture (default observer is only 320×240).
        cams.observer_camera = self.scene.add_camera(
            name="observer_camera",
            width=1920,
            height=1440,
            fovy=zoom_fovy,
            near=near,
            far=far,
        )
        camera = cams.observer_camera
        # Frame the near-rail drop from the table's upper-right corner.
        camera_pos = np.array([0.32, 0.34, 1.28], dtype=np.float64)
        look_at = np.array([0.0, -0.06, 0.86], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.rail_cx = float(cfg.get("rail_center_x", self.RAIL_CENTER_X_DEFAULT))
        self.rail_cy = float(cfg.get("rail_center_y", self.RAIL_CENTER_Y_DEFAULT))
        self.rail_radius = float(cfg.get("rail_radius", self.RAIL_RADIUS_DEFAULT))
        self.randomize_rail_radius = bool(cfg.get("randomize_rail_radius", False))
        # One-sided: current radius as lower bound, up to +rail_radius_jitter (default +10%).
        rr_jitter = float(np.clip(abs(float(cfg.get("rail_radius_jitter", 0.10))), 0.0, 0.95))
        if self.randomize_rail_radius and rr_jitter > 0.0:
            self.rail_radius = float(np.random.uniform(
                self.rail_radius, self.rail_radius * (1.0 + rr_jitter)
            ))
        self.rail_segments = int(cfg.get("rail_segments", self.RAIL_SEGMENTS_DEFAULT))
        self.rail_half_w = float(cfg.get("rail_half_w", self.RAIL_HALF_W_DEFAULT))
        self.rail_thick = float(cfg.get("rail_thick", self.RAIL_THICK_DEFAULT))
        self.rail_wall_h = float(cfg.get("rail_wall_h", self.RAIL_WALL_H_DEFAULT))

        n_wagons = cfg.get("n_wagons", None)
        if n_wagons is None:
            n_min = int(cfg.get("n_wagons_min", cfg.get("n_cars_min", self.N_WAGONS_MIN_DEFAULT)))
            n_max = int(cfg.get("n_wagons_max", cfg.get("n_cars_max", self.N_WAGONS_MAX_DEFAULT)))
            self.n_wagons = int(np.random.randint(n_min, n_max + 1))
        else:
            self.n_wagons = int(n_wagons)
        self.n_wagons = max(1, int(self.n_wagons))
        # cars[0] = closed locomotive; cars[1:] = open wagons
        self.n_cars = 1 + self.n_wagons
        self._cargo_indices = list(range(1, self.n_cars))

        # Per-episode train speed sampled in [omega_min, omega_max], then signed direction.
        omega_min = float(cfg.get("train_omega_min", self.TRAIN_OMEGA_MIN_DEFAULT))
        omega_max = float(cfg.get("train_omega_max", self.TRAIN_OMEGA_MAX_DEFAULT))
        if "train_omega" in cfg and "train_omega_min" not in cfg and "train_omega_max" not in cfg:
            # Backward-compatible: single value ±25% jitter.
            mid = float(cfg.get("train_omega", self.TRAIN_OMEGA_DEFAULT))
            omega_min, omega_max = mid * 0.75, mid * 1.25
        if omega_max < omega_min:
            omega_min, omega_max = omega_max, omega_min
        self.train_dir = float(np.random.choice([-1.0, 1.0]))
        self.train_omega = self.train_dir * float(np.random.uniform(omega_min, omega_max))

        self.car_arc_spacing = float(cfg.get("car_arc_spacing", self.CAR_ARC_SPACING_DEFAULT))
        self.car_half_len = float(cfg.get("car_half_len", self.CAR_HALF_LEN_DEFAULT))
        self.car_half_wid = float(cfg.get("car_half_wid", self.CAR_HALF_WID_DEFAULT))
        self.car_wall_h = float(cfg.get("car_wall_h", self.CAR_WALL_H_DEFAULT))
        self.car_floor_h = float(cfg.get("car_floor_h", self.CAR_FLOOR_H_DEFAULT))
        self.car_wall_t = float(cfg.get("car_wall_t", self.CAR_WALL_T_DEFAULT))
        self.engine_cab_h = float(cfg.get("engine_cab_h", self.ENGINE_CAB_H_DEFAULT))
        self.engine_body_h = float(cfg.get("engine_body_h", self.ENGINE_BODY_H_DEFAULT))

        self.ball_radius = float(cfg.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_side_clearance = float(cfg.get("ball_side_clearance", self.BALL_SIDE_CLEARANCE_DEFAULT))
        self.ball_lateral = float(cfg.get("ball_lateral", self.BALL_LATERAL_DEFAULT))
        self.ball_lateral_min = float(cfg.get("ball_lateral_min", self.BALL_LATERAL_MIN_DEFAULT))
        self.ball_lateral_max = float(cfg.get("ball_lateral_max", self.BALL_LATERAL_MAX_DEFAULT))
        if self.ball_lateral_max < self.ball_lateral_min:
            self.ball_lateral_min, self.ball_lateral_max = self.ball_lateral_max, self.ball_lateral_min
        self.ball_y_jitter = float(cfg.get("ball_y_jitter", self.BALL_Y_JITTER_DEFAULT))
        ball_side_cfg = str(cfg.get("ball_side", self.BALL_SIDE_DEFAULT)).lower()
        self.transport_clearance_z = float(
            cfg.get("transport_clearance_z", self.TRANSPORT_CLEARANCE_Z_DEFAULT)
        )
        self.release_clearance_z = float(
            cfg.get("release_clearance_z", self.RELEASE_CLEARANCE_Z_DEFAULT)
        )
        self.pass_angle_offset = float(cfg.get("pass_angle_offset", self.PASS_ANGLE_OFFSET_DEFAULT))
        self.align_tol = float(cfg.get("align_tol", self.ALIGN_TOL_DEFAULT))
        self.align_search_steps = int(cfg.get("align_search_steps", self.ALIGN_SEARCH_STEPS_DEFAULT))
        self.release_open_steps = int(cfg.get("release_open_steps", self.RELEASE_OPEN_STEPS_DEFAULT))
        self.post_release_steps = int(cfg.get("post_release_steps", self.POST_RELEASE_STEPS_DEFAULT))
        self.latch_xy_margin = float(cfg.get("latch_xy_margin", self.LATCH_XY_MARGIN_DEFAULT))
        self.latch_z_max = float(cfg.get("latch_z_max", self.LATCH_Z_MAX_DEFAULT))
        self.latch_settle_steps = int(cfg.get("latch_settle_steps", self.LATCH_SETTLE_STEPS_DEFAULT))
        self.catch_radius = float(cfg.get("catch_radius", self.CATCH_RADIUS_DEFAULT))
        self.drop_reach_tol = float(cfg.get("drop_reach_tol", self.DROP_REACH_TOL_DEFAULT))
        self._bed_contact_steps = 0
        self._reset_metrics()

        # Opt 1 — one randomly selected target wagon (red); others gray.
        self.target_wagon_mode = self._parse_target_wagon_mode(cfg)
        self.target_wagon_idx = None

        # Opt 2 — far-arc tunnel + mirrored near-arc tunnel.
        tunnel_cfg = self._parse_tunnel_enabled_cfg(cfg)
        tunnel_prob = float(cfg.get("tunnel_prob", self.TUNNEL_PROB_DEFAULT))
        if isinstance(tunnel_cfg, bool):
            self.tunnel_present = tunnel_cfg
        else:
            tunnel_s = str(tunnel_cfg).lower()
            if tunnel_s in ("true", "1", "yes", "always"):
                self.tunnel_present = True
            elif tunnel_s in ("false", "0", "no", "never"):
                self.tunnel_present = False
            else:
                # "random" (or any other string) → sample with tunnel_prob
                self.tunnel_present = bool(np.random.rand() < tunnel_prob)
        self.tunnel_n_wagons_min = int(cfg.get("tunnel_n_wagons_min", self.TUNNEL_N_WAGONS_MIN_DEFAULT))
        self.tunnel_n_wagons_max = int(cfg.get("tunnel_n_wagons_max", self.TUNNEL_N_WAGONS_MAX_DEFAULT))
        if self.tunnel_n_wagons_max < self.tunnel_n_wagons_min:
            self.tunnel_n_wagons_min, self.tunnel_n_wagons_max = (
                self.tunnel_n_wagons_max, self.tunnel_n_wagons_min
            )
        self.tunnel_n_wagons_min = max(1, self.tunnel_n_wagons_min)
        self.tunnel_n_wagons_max = max(self.tunnel_n_wagons_min, self.tunnel_n_wagons_max)
        self.tunnel_center_jitter = float(
            cfg.get("tunnel_center_jitter", self.TUNNEL_CENTER_JITTER_DEFAULT)
        )
        self.tunnel_clearance_z = float(
            cfg.get("tunnel_clearance_z", self.TUNNEL_CLEARANCE_Z_DEFAULT)
        )
        self.tunnel_wall_t = float(cfg.get("tunnel_wall_t", self.TUNNEL_WALL_T_DEFAULT))
        self.tunnel_roof_t = float(cfg.get("tunnel_roof_t", self.TUNNEL_ROOF_T_DEFAULT))
        self.tunnel_overhang = float(cfg.get("tunnel_overhang", self.TUNNEL_OVERHANG_DEFAULT))
        self.tunnel_n_wagons = 0
        self.tunnel_center_angle = None
        self.tunnel_center_angle_near = None
        self.tunnel_half_angle = 0.0

        z0 = 0.74 + self.table_z_bias
        self.table_top_z = z0
        self.rail_surface_z = z0 + self.rail_thick
        self.car_floor_z = self.rail_surface_z + 0.004
        self._dtheta_car = self.car_arc_spacing / max(self.rail_radius, 1e-6)

        # Locomotive starts at 12 o'clock (far side, opposite the robot arms at 6 o'clock).
        # Angle convention: 0=+x, +pi/2=+y (far), -pi/2=-y (near / arms).
        self._train_angle0 = 0.5 * np.pi
        self._train_angle = self._train_angle0
        self._train_step = 0
        # Per-episode left/right: ball on −x → left arm, +x → right arm.
        if ball_side_cfg in ("left", "right"):
            self.ball_side = ball_side_cfg
        else:
            # Default / "random": fair coin flip so either arm is used.
            self.ball_side = "left" if int(np.random.randint(0, 2)) == 0 else "right"
        side_sign = -1.0 if self.ball_side == "left" else 1.0
        # Pass angle: near the robot (-pi/2) biased toward the ball's side.
        # With tunnels, drop just outside the near-arc mouth so the ball is not
        # released through the roof.
        if self.tunnel_present:
            self._sample_tunnel_layout()
            mouth_clear = max(0.10, float(self.pass_angle_offset) * 0.5)
            self.pass_angle = float(
                self.tunnel_center_angle_near
                + side_sign * (self.tunnel_half_angle + mouth_clear)
            )
        else:
            self.pass_angle = -0.5 * np.pi + side_sign * self.pass_angle_offset
        self.pass_angle = float(np.arctan2(np.sin(self.pass_angle), np.cos(self.pass_angle)))
        self._drop_target_xy = self._xy_on_rail(self.pass_angle)

        # Opt 1: pick which open wagon is the drop target (cars[1..] are cargo).
        if self.target_wagon_mode and self._cargo_indices:
            self.target_wagon_idx = int(np.random.choice(self._cargo_indices))
        else:
            self.target_wagon_idx = None

        self._build_rail()
        if self.tunnel_present:
            self._build_tunnel()
        self._build_train()
        self._place_train(self._train_angle)

        # Spawn on the chosen table half so the matching arm grasps (|x| in
        # [ball_lateral_min, ball_lateral_max]); keep near the side-biased drop.
        lateral = float(np.random.uniform(self.ball_lateral_min, self.ball_lateral_max))
        drop_x = float(self._drop_target_xy[0])
        ball_x = float(np.clip(
            side_sign * lateral,
            -0.18 if side_sign < 0 else 0.04,
            -0.04 if side_sign < 0 else 0.18,
        ))
        # Nudge toward the drop station so the lateral carry stays short.
        ball_x = float(0.65 * ball_x + 0.35 * drop_x)
        ball_x = float(np.clip(
            ball_x,
            -0.18 if side_sign < 0 else 0.04,
            -0.04 if side_sign < 0 else 0.18,
        ))
        ball_y0 = float(self.rail_cy - self.rail_radius - self.ball_side_clearance)
        ball_y = float(np.clip(
            ball_y0 + np.random.uniform(-self.ball_y_jitter, self.ball_y_jitter),
            -0.28,
            -0.12,
        ))
        self.ball_color = [0.95, 0.75, 0.10]  # bright yellow — visible inside blue/red wagons
        self.ball = self._build_ball(
            pose=sapien.Pose(
                [ball_x, ball_y, z0 + self.ball_radius + 0.001],
                [1, 0, 0, 0],
            ),
            color=self.ball_color,
        )
        self._ball_rigid = self._get_rigid(self.ball)

        self.add_prohibit_area(self.ball, padding=0.04)
        self._loaded = True
        # Stay stopped through the post-spawn stability check; setup_demo / play_once
        # start motion immediately afterward so the train runs from episode start.
        self._train_running = False

    # -------------------------------------------------------------- builders
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _xy_on_rail(self, angle):
        return np.array([
            self.rail_cx + self.rail_radius * np.cos(angle),
            self.rail_cy + self.rail_radius * np.sin(angle),
        ], dtype=np.float64)

    def _yaw_quat(self, angle):
        # Car forward is along +local_x; tangent of circle is (-sin, cos).
        yaw = float(angle + 0.5 * np.pi)
        return t3d.euler.euler2quat(0.0, 0.0, yaw)

    def _build_rail(self):
        """Approximate a circular track with sleeper segments + inner/outer walls."""
        n = max(16, int(self.rail_segments))
        sleeper_half_len = 0.55 * (2.0 * np.pi * self.rail_radius / n)
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            xy = self._xy_on_rail(ang)
            q = self._yaw_quat(ang)
            create_box(
                scene=self,
                pose=sapien.Pose(
                    [xy[0], xy[1], self.table_top_z + self.rail_thick * 0.5],
                    q.tolist(),
                ),
                half_size=(sleeper_half_len, self.rail_half_w, self.rail_thick * 0.5),
                color=(0.22, 0.22, 0.24),
                name=f"rail_sleeper_{i}",
                is_static=True,
            )
            # Inner / outer guide rails (thin walls).
            for s, tag in ((-1.0, "in"), (1.0, "out")):
                radial = np.array([np.cos(ang), np.sin(ang)], dtype=np.float64)
                wxy = xy + s * (self.rail_half_w + 0.004) * radial
                create_box(
                    scene=self,
                    pose=sapien.Pose(
                        [wxy[0], wxy[1], self.table_top_z + self.rail_thick + self.rail_wall_h * 0.5],
                        q.tolist(),
                    ),
                    half_size=(sleeper_half_len * 0.95, 0.003, self.rail_wall_h * 0.5),
                    color=(0.55, 0.35, 0.18),
                    name=f"rail_wall_{tag}_{i}",
                    is_static=True,
                )
        # Center hub (visual cue that the track is circular / centered).
        create_box(
            scene=self,
            pose=sapien.Pose(
                [self.rail_cx, self.rail_cy, self.table_top_z + 0.004],
                [1, 0, 0, 0],
            ),
            half_size=(0.025, 0.025, 0.004),
            color=(0.35, 0.35, 0.38),
            name="rail_hub",
            is_static=True,
        )

    def _wrap_angle(self, ang):
        return float(np.arctan2(np.sin(ang), np.cos(ang)))

    def _sample_tunnel_layout(self):
        """Sample one 2–4 wagon span; far center near 12 o'clock, near = far + π."""
        n_w = int(np.random.randint(self.tunnel_n_wagons_min, self.tunnel_n_wagons_max + 1))
        arc_len = float(n_w) * float(self.car_arc_spacing)
        half_ang = 0.5 * arc_len / max(self.rail_radius, 1e-6)
        # Keep a margin inside the upper half [0, pi] for the far tunnel.
        margin = 0.10
        lo = half_ang + margin
        hi = np.pi - half_ang - margin
        if hi < lo:
            half_ang = max(0.05, 0.5 * (np.pi - 2.0 * margin))
            lo = half_ang + margin
            hi = np.pi - half_ang - margin
        center = 0.5 * np.pi + float(np.random.uniform(
            -self.tunnel_center_jitter, self.tunnel_center_jitter
        ))
        center = float(np.clip(center, lo, hi))
        self.tunnel_n_wagons = int(n_w)
        self.tunnel_center_angle = float(center)
        self.tunnel_center_angle_near = self._wrap_angle(center + np.pi)
        self.tunnel_half_angle = float(half_ang)

    def _build_tunnel(self):
        """Two horseshoe tunnels: far arc and a 180° mirror on the near arc."""
        if self.tunnel_center_angle is None:
            self._sample_tunnel_layout()
        n_w = int(self.tunnel_n_wagons)
        half_ang = float(self.tunnel_half_angle)
        self._spawn_tunnel_arc(self.tunnel_center_angle, half_ang, n_w, "far")
        self._spawn_tunnel_arc(self.tunnel_center_angle_near, half_ang, n_w, "near")

    def _spawn_tunnel_arc(self, center, half_ang, n_w, prefix):
        """Classic railway silhouette: vertical piers + semicircular arch."""
        open_h = max(
            self.tunnel_clearance_z,
            self.engine_body_h + self.engine_cab_h + 0.025,
        )
        z_base = self.rail_surface_z
        cavity_half_w = self.rail_half_w + 0.006 + self.tunnel_overhang
        R_clear = float(cavity_half_w)
        pier_h = max(0.012, float(open_h) - R_clear)
        spring_z = z_base + pier_h
        shell_t = max(self.tunnel_wall_t, self.tunnel_roof_t)
        R_mid = R_clear + 0.5 * shell_t
        pier_color = (0.45, 0.43, 0.39)
        arch_color = (0.36, 0.34, 0.30)
        portal_color = (0.40, 0.38, 0.34)

        n_seg = max(8, int(round(n_w * 6)))
        angs = np.linspace(center - half_ang, center + half_ang, n_seg)
        dtheta = float(angs[1] - angs[0]) if n_seg > 1 else 0.05
        seg_half_len = 0.58 * self.rail_radius * dtheta
        n_voussoir = 14
        phis = np.linspace(0.0, np.pi, n_voussoir)
        dphi = float(phis[1] - phis[0]) if n_voussoir > 1 else 0.2
        voussoir_half_arc = 0.55 * R_mid * dphi

        for i, ang in enumerate(angs):
            ang = float(ang)
            xy = self._xy_on_rail(ang)
            q_yaw = self._yaw_quat(ang)
            radial = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            tangent = np.array([-np.sin(ang), np.cos(ang), 0.0], dtype=np.float64)
            for s, tag in ((1.0, "out"), (-1.0, "in")):
                wxy = xy + s * (R_clear + 0.5 * shell_t) * radial[:2]
                create_box(
                    scene=self,
                    pose=sapien.Pose(
                        [wxy[0], wxy[1], z_base + 0.5 * pier_h],
                        q_yaw.tolist(),
                    ),
                    half_size=(seg_half_len, 0.5 * shell_t, 0.5 * pier_h),
                    color=pier_color,
                    name=f"tunnel_{prefix}_pier_{tag}_{i}",
                    is_static=True,
                )
            for j, phi in enumerate(phis):
                phi = float(phi)
                arch_out = np.array(
                    [np.cos(phi) * radial[0], np.cos(phi) * radial[1], np.sin(phi)],
                    dtype=np.float64,
                )
                pos = np.array([xy[0], xy[1], spring_z], dtype=np.float64) + R_mid * arch_out
                z_axis = arch_out / max(np.linalg.norm(arch_out), 1e-9)
                x_axis = tangent
                y_axis = np.cross(z_axis, x_axis)
                y_n = np.linalg.norm(y_axis)
                if y_n < 1e-8:
                    continue
                y_axis /= y_n
                x_axis = np.cross(y_axis, z_axis)
                rot = np.column_stack([x_axis, y_axis, z_axis])
                q = t3d.quaternions.mat2quat(rot)
                create_box(
                    scene=self,
                    pose=sapien.Pose(pos.tolist(), q.tolist()),
                    half_size=(seg_half_len, voussoir_half_arc, 0.5 * shell_t),
                    color=arch_color,
                    name=f"tunnel_{prefix}_arch_{i}_{j}",
                    is_static=True,
                )

        portal_half_len = max(0.012, 1.6 * seg_half_len)
        for k, ang in enumerate((center - half_ang, center + half_ang)):
            ang = float(ang)
            xy = self._xy_on_rail(ang)
            q_yaw = self._yaw_quat(ang)
            radial = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            tangent = np.array([-np.sin(ang), np.cos(ang), 0.0], dtype=np.float64)
            for s, tag in ((1.0, "out"), (-1.0, "in")):
                wxy = xy + s * (R_clear + 0.5 * shell_t) * radial[:2]
                create_box(
                    scene=self,
                    pose=sapien.Pose(
                        [wxy[0], wxy[1], z_base + 0.5 * pier_h],
                        q_yaw.tolist(),
                    ),
                    half_size=(portal_half_len, 0.5 * shell_t, 0.5 * pier_h),
                    color=portal_color,
                    name=f"tunnel_{prefix}_portal_pier_{tag}_{k}",
                    is_static=True,
                )
            for j, phi in enumerate(phis):
                phi = float(phi)
                arch_out = np.array(
                    [np.cos(phi) * radial[0], np.cos(phi) * radial[1], np.sin(phi)],
                    dtype=np.float64,
                )
                pos = np.array([xy[0], xy[1], spring_z], dtype=np.float64) + R_mid * arch_out
                z_axis = arch_out / max(np.linalg.norm(arch_out), 1e-9)
                x_axis = tangent
                y_axis = np.cross(z_axis, x_axis)
                y_n = np.linalg.norm(y_axis)
                if y_n < 1e-8:
                    continue
                y_axis /= y_n
                x_axis = np.cross(y_axis, z_axis)
                rot = np.column_stack([x_axis, y_axis, z_axis])
                q = t3d.quaternions.mat2quat(rot)
                create_box(
                    scene=self,
                    pose=sapien.Pose(pos.tolist(), q.tolist()),
                    half_size=(portal_half_len, voussoir_half_arc, 0.5 * shell_t),
                    color=portal_color,
                    name=f"tunnel_{prefix}_portal_arch_{k}_{j}",
                    is_static=True,
                )

    def _add_box_shape(self, builder, pose, half_size, material, visual_material):
        builder.add_box_collision(pose=pose, half_size=list(half_size), material=material)
        builder.add_box_visual(pose=pose, half_size=list(half_size), material=visual_material)

    def _build_car(self, name, is_engine=False, wall_color=None):
        """Closed locomotive (no cargo opening) or open-top wagon."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        mat = self.scene.default_physical_material
        hl, hw = self.car_half_len, self.car_half_wid
        fh = self.car_floor_h
        wh = self.car_wall_h
        wt = self.car_wall_t

        if is_engine:
            # Fully enclosed body — no opening to drop anything into.
            body_col = [0.12, 0.32, 0.62, 1.0]
            vis = sapien.render.RenderMaterial(base_color=body_col)
            body_h = self.engine_body_h
            self._add_box_shape(
                builder,
                sapien.Pose([0, 0, body_h * 0.5]),
                (hl, hw, body_h * 0.5),
                mat,
                vis,
            )
            # Cab block on top (rear of loco)
            cab_vis = sapien.render.RenderMaterial(base_color=[0.08, 0.22, 0.48, 1.0])
            cab_h = self.engine_cab_h
            self._add_box_shape(
                builder,
                sapien.Pose([-hl * 0.35, 0, body_h + cab_h * 0.5]),
                (hl * 0.45, hw * 0.85, cab_h * 0.5),
                mat,
                cab_vis,
            )
            # Chimney stub at the front
            chim_vis = sapien.render.RenderMaterial(base_color=[0.15, 0.15, 0.18, 1.0])
            self._add_box_shape(
                builder,
                sapien.Pose([hl * 0.45, 0, body_h + 0.018]),
                (0.010, 0.010, 0.018),
                mat,
                chim_vis,
            )
        else:
            # Open-top cargo wagon: thick floor + four walls = a real open box.
            body_col = list(wall_color) if wall_color is not None else list(self.WAGON_COLOR_TARGET)
            if len(body_col) == 3:
                body_col = body_col + [1.0]
            # Dark bed so the floor is obvious against the light wood track underneath.
            floor_vis = sapien.render.RenderMaterial(base_color=[0.16, 0.16, 0.18, 1.0])
            bed_vis = sapien.render.RenderMaterial(base_color=[0.28, 0.28, 0.32, 1.0])
            wall_vis = sapien.render.RenderMaterial(base_color=body_col)
            # Structural floor (collision + visual) — thick enough to read as a box bottom.
            self._add_box_shape(
                builder,
                sapien.Pose([0, 0, fh * 0.5]),
                (hl, hw, fh * 0.5),
                mat,
                floor_vis,
            )
            # Inner bed plank inset so the cavity reads as a box, not a hollow frame.
            bed_t = min(0.005, fh * 0.4)
            self._add_box_shape(
                builder,
                sapien.Pose([0, 0, fh - bed_t * 0.5 + 0.0005]),
                (max(hl - wt * 1.5, hl * 0.55), max(hw - wt * 1.5, hw * 0.55), bed_t * 0.5),
                mat,
                bed_vis,
            )
            # Walls: front/back/left/right sitting on the floor (open top).
            wall_cz = fh + wh * 0.5
            self._add_box_shape(
                builder,
                sapien.Pose([hl - wt * 0.5, 0, wall_cz]),
                (wt * 0.5, hw, wh * 0.5),
                mat,
                wall_vis,
            )
            self._add_box_shape(
                builder,
                sapien.Pose([-hl + wt * 0.5, 0, wall_cz]),
                (wt * 0.5, hw, wh * 0.5),
                mat,
                wall_vis,
            )
            self._add_box_shape(
                builder,
                sapien.Pose([0, hw - wt * 0.5, wall_cz]),
                (hl - wt, wt * 0.5, wh * 0.5),
                mat,
                wall_vis,
            )
            self._add_box_shape(
                builder,
                sapien.Pose([0, -hw + wt * 0.5, wall_cz]),
                (hl - wt, wt * 0.5, wh * 0.5),
                mat,
                wall_vis,
            )

        # Small wheel stubs (visual + light collision)
        wheel_vis = sapien.render.RenderMaterial(base_color=[0.08, 0.08, 0.08, 1.0])
        wr = 0.008
        for sx in (-0.55, 0.55):
            for sy in (-1.0, 1.0):
                self._add_box_shape(
                    builder,
                    sapien.Pose([sx * hl, sy * (hw + 0.002), wr]),
                    (wr, wr * 0.6, wr),
                    mat,
                    wheel_vis,
                )

        builder.set_initial_pose(sapien.Pose([0, 0, self.car_floor_z], [1, 0, 0, 0]))
        return builder.build(name=name)

    def _build_train(self):
        self.cars = []
        self._car_rigids = []
        for i in range(self.n_cars):
            is_engine = (i == 0)
            wall_color = None
            if not is_engine:
                if self.target_wagon_mode and self.target_wagon_idx is not None:
                    # Opt 1: red target wagon, gray distractors.
                    wall_color = (
                        self.WAGON_COLOR_TARGET
                        if i == self.target_wagon_idx
                        else self.WAGON_COLOR_DISTRACTOR
                    )
                else:
                    # Default: all open wagons red (any wagon is a valid drop).
                    wall_color = self.WAGON_COLOR_TARGET
            car = self._build_car(
                name=f"{'locomotive' if is_engine else 'wagon'}_{i}",
                is_engine=is_engine,
                wall_color=wall_color,
            )
            rigid = self._get_rigid(car)
            if rigid is not None:
                try:
                    rigid.set_disable_gravity(True)
                    rigid.set_kinematic(True)
                except Exception:
                    pass
            self.cars.append(car)
            self._car_rigids.append(rigid)

    def _build_ball(self, pose, color):
        entity = create_sphere(
            self,
            pose=pose,
            radius=self.ball_radius,
            color=color,
            is_static=False,
            name="train_ball",
        )
        z_off = 0.35 * self.ball_radius
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "extents": [2.0 * self.ball_radius] * 3,
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
        ball_actor = Actor(entity, data, mass=0.025)
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                inelastic = sapien.physx.PhysxMaterial(
                    static_friction=0.95,
                    dynamic_friction=0.95,
                    restitution=0.0,
                )
                for shape in rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass
        return ball_actor

    # ----------------------------------------------------------- train motion
    def _angle_at_step(self, step):
        return self._train_angle0 + self.train_omega * (float(step) * 0.01)

    def _car_angle_at(self, engine_angle, car_idx):
        # Wagons trail the engine opposite the travel direction.
        return float(engine_angle - float(car_idx) * self._dtheta_car * np.sign(self.train_omega or 1.0))

    def _car_pose_at(self, engine_angle, car_idx):
        ang = self._car_angle_at(engine_angle, car_idx)
        xy = self._xy_on_rail(ang)
        q = self._yaw_quat(ang)
        p = [float(xy[0]), float(xy[1]), float(self.car_floor_z)]
        return sapien.Pose(p, q.tolist()), ang

    def _place_train(self, engine_angle):
        self._train_angle = float(engine_angle)
        for i, car in enumerate(self.cars):
            pose, _ = self._car_pose_at(engine_angle, i)
            rigid = self._car_rigids[i] if i < len(self._car_rigids) else None
            if rigid is not None and self._train_running:
                try:
                    rigid.set_kinematic_target(pose)
                    continue
                except Exception:
                    pass
            car.set_pose(pose)

    def _wagon_bed_half(self):
        # Inner bed slightly smaller than outer walls.
        return (
            self.car_half_len - self.car_wall_t - self.latch_xy_margin,
            self.car_half_wid - self.car_wall_t - self.latch_xy_margin,
        )

    def _world_to_car_local(self, world_p, car_pose):
        R = car_pose.to_transformation_matrix()[:3, :3]
        return R.T @ (np.asarray(world_p, dtype=np.float64) - np.asarray(car_pose.p, dtype=np.float64))

    def _ball_in_car(self, car_idx):
        # Locomotive (idx 0) has no cargo opening — never count as loaded there.
        if self.ball is None or car_idx is None or car_idx <= 0 or car_idx >= len(self.cars):
            return False
        pose, _ = self._car_pose_at(self._train_angle, car_idx)
        try:
            pose = self.cars[car_idx].get_pose()
        except Exception:
            pass
        local = self._world_to_car_local(self.ball.get_pose().p, pose)
        hl, hw = self._wagon_bed_half()
        in_xy = abs(local[0]) <= hl and abs(local[1]) <= hw
        rest_z = self.car_floor_h + self.ball_radius
        z_lo = self.car_floor_h + 0.2 * self.ball_radius
        z_hi = rest_z + max(0.02, float(self.latch_z_max))
        in_z = z_lo <= local[2] <= z_hi
        return bool(in_xy and in_z)

    def _try_latch_ball(self):
        """Seat the ball only after it has physically landed in a wagon bed.

        Mid-air / rim-height snaps are intentionally rejected so the drop reads as a
        natural fall onto the floor rather than a teleport into the wagon.
        """
        if self._ball_latched or not self._ball_released or self.ball is None:
            return
        ball_p = np.asarray(self.ball.get_pose().p, dtype=np.float64)
        rest_z = self.car_floor_h + self.ball_radius
        # Only accept contact near the bed floor (not while the ball is still falling).
        z_lo = self.car_floor_h + 0.25 * self.ball_radius
        z_hi = rest_z + float(self.latch_z_max)
        matched = False
        best_i = None
        best_local = None
        for i in getattr(self, "_cargo_indices", list(range(1, self.n_cars))):
            car_pose = self.cars[i].get_pose()
            local = self._world_to_car_local(ball_p, car_pose)
            hl, hw = self._wagon_bed_half()
            in_xy = abs(local[0]) <= (hl + 0.006) and abs(local[1]) <= (hw + 0.006)
            in_z = z_lo <= local[2] <= z_hi
            if not (in_xy and in_z):
                continue
            matched = True
            # Keep the landed XY (small clamp only) — do not yank toward wagon center.
            local[0] = float(np.clip(local[0], -hl * 0.9, hl * 0.9))
            local[1] = float(np.clip(local[1], -hw * 0.9, hw * 0.9))
            local[2] = rest_z
            best_i = i
            best_local = local
            break
        if not matched:
            self._bed_contact_steps = 0
            return
        self._bed_contact_steps = int(getattr(self, "_bed_contact_steps", 0)) + 1
        if self._bed_contact_steps < int(self.latch_settle_steps):
            return
        self._latch_to_wagon(best_i, best_local)

    def _latch_to_wagon(self, wagon_idx, local=None):
        """Kinematically seat the ball in an open wagon; train keeps moving."""
        if wagon_idx is None or wagon_idx <= 0 or wagon_idx >= len(self.cars):
            return
        if local is None:
            local = np.array(
                [0.0, 0.0, self.car_floor_h + self.ball_radius + 0.004],
                dtype=np.float64,
            )
        self._latch_local = np.asarray(local, dtype=np.float64)
        self._latched_car_idx = int(wagon_idx)
        self._ball_latched = True
        self.ball_in_train = True
        self._latch_seat_metrics()
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_linear_velocity([0, 0, 0])
                self._ball_rigid.set_angular_velocity([0, 0, 0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._update_latched_ball()

    def _update_latched_ball(self):
        if not self._ball_latched or self._latched_car_idx is None or self.ball is None:
            return
        car = self.cars[self._latched_car_idx]
        car_pose = car.get_pose()
        R = car_pose.to_transformation_matrix()[:3, :3]
        world_p = np.asarray(car_pose.p, dtype=np.float64) + R @ self._latch_local
        pose = sapien.Pose(world_p.tolist(), car_pose.q)
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_kinematic_target(pose)
                return
            except Exception:
                pass
        self.ball.actor.set_pose(pose)

    def _steps_until_wagon_under_drop(
        self, target_xy, max_steps=None, tol=None, lead_steps=0, wagon_idx=None
    ):
        """Return (wait_steps, wagon_idx) until an OPEN wagon is under target_xy.

        If ``wagon_idx`` is set (Opt 1), only that cargo wagon is considered.
        """
        target_xy = np.asarray(target_xy, dtype=np.float64)
        max_steps = int(self.align_search_steps if max_steps is None else max_steps)
        tol = float(self.align_tol if tol is None else tol)
        lead_steps = max(0, int(lead_steps))
        cargo = getattr(self, "_cargo_indices", list(range(1, self.n_cars)))
        if wagon_idx is not None:
            cargo = [int(wagon_idx)] if int(wagon_idx) in cargo else list(cargo)
        best_steps = 0
        best_err = float("inf")
        best_wagon = cargo[0] if cargo else 1
        for wait in range(max_steps + 1):
            eng = self._angle_at_step(self._train_step + wait + lead_steps)
            for i in cargo:
                pose, _ = self._car_pose_at(eng, i)
                err = float(np.linalg.norm(np.asarray(pose.p[:2]) - target_xy))
                if err < best_err:
                    best_err = err
                    best_steps = wait
                    best_wagon = i
                if err <= tol:
                    return wait, i
        return best_steps, best_wagon

    def _ball_off_table(self):
        """True when the free ball has fallen off the tabletop onto the floor.

        Table footprint matches ``create_table`` (length=1.2, width=0.7). A ball
        still in a wagon or currently pinched by the gripper is not a failure.
        """
        if getattr(self, "ball", None) is None:
            return False
        if getattr(self, "_ball_latched", False) or self._success_in_target_wagon():
            return False
        try:
            if len(self.get_gripper_actor_contact_position(self.ball.get_name())) > 0:
                return False
        except Exception:
            pass
        p = np.asarray(self.ball.get_pose().p, dtype=np.float64)
        table_z = float(getattr(self, "table_top_z", 0.74))
        if float(p[2]) < table_z - 0.05:
            return True
        bias = getattr(self, "table_xy_bias", [0.0, 0.0])
        half_x, half_y = 0.60, 0.35
        margin = 0.03
        off_xy = (
            abs(float(p[0]) - float(bias[0])) > half_x + margin
            or abs(float(p[1]) - float(bias[1])) > half_y + margin
        )
        near_surface = float(p[2]) < table_z + float(getattr(self, "ball_radius", 0.02)) + 0.06
        return bool(off_xy and near_surface)

    def _try_latch_ball_off_table(self):
        """Latch off-table failure so interactive / eval can terminate."""
        if getattr(self, "_ball_fell_off_table", False):
            return True
        if not self._ball_off_table():
            return False
        self._ball_fell_off_table = True
        self.ball_in_train = False
        self._last_fail_reason = "ball dropped off the table"
        return True

    # -------------------------------------------------------------- kinematics
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if self._train_running:
            self._train_step += 1
            self._place_train(self._angle_at_step(self._train_step))
        if self._ball_released and not self._ball_latched:
            self._try_latch_ball()
        if self._ball_latched:
            self._update_latched_ball()
        self._try_latch_ball_off_table()
        self.ball_in_train = self._success_in_target_wagon()
        self._update_metrics()

    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _move_ball_to_height(self, arm_tag: ArmTag, target_z: float):
        if self.ball is None:
            return
        dz = float(target_z) - float(self.ball.get_pose().p[2])
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz, move_axis="world"))

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_name = "left" if self.ball_side == "left" else "right"
        arm_tag = ArmTag(arm_name)
        self.selected_arm = arm_name
        # Episode start: loco at 12 o'clock (far side), then keep moving continuously.
        self._train_angle0 = 0.5 * np.pi
        self._train_step = 0
        self._train_angle = self._train_angle0
        self._reset_metrics()
        self._place_train(self._train_angle)
        self._train_running = True
        if self.save_freq:
            self._take_picture()

        # Stay high for the whole carry + release so fingers never clip the train.
        # The ball falls from this height into a passing wagon (natural physics drop).
        release_z = float(
            self.car_floor_z + self.car_floor_h + self.car_wall_h
            + self.ball_radius + max(self.transport_clearance_z, self.release_clearance_z)
        )

        def _try_grasp(tag: ArmTag) -> bool:
            self.plan_success = True
            self.move(self.grasp_actor(self.ball, arm_tag=tag, pre_grasp_dis=0.1))
            if not self.plan_success:
                return False
            self._move_ball_to_height(arm_tag=tag, target_z=release_z)
            return float(self.ball.get_pose().p[2]) >= release_z - 0.08

        if not _try_grasp(arm_tag):
            # Left-arm sphere grasps are unreliable here; fall back to the other arm.
            try:
                self.move(self.back_to_origin(arm_tag))
            except Exception:
                pass
            alt_name = "right" if arm_name == "left" else "left"
            alt_tag = ArmTag(alt_name)
            self.plan_success = True
            if _try_grasp(alt_tag):
                arm_name, arm_tag = alt_name, alt_tag
                self.selected_arm = arm_name

        # Bail early if still not holding — avoids "random appear in wagon" teleports.
        if float(self.ball.get_pose().p[2]) < release_z - 0.08:
            self.info["info"] = self._episode_info(arm_name)
            return self.info

        # Stabilize the carry: disable gravity on the held ball until release so a
        # long wait for the train cannot drop it on the table (fall is still physics).
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_linear_velocity([0, 0, 0])
                self._ball_rigid.set_angular_velocity([0, 0, 0])
            except Exception:
                pass

        # Nudge the held ball onto the near-rail drop station (on the track path).
        reach_tol = float(getattr(self, "drop_reach_tol", self.DROP_REACH_TOL_DEFAULT))
        for _ in range(6):
            ball_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
            delta_xy = self._drop_target_xy - ball_xy
            err = float(np.linalg.norm(delta_xy))
            if err <= reach_tol:
                break
            # Abort XY push if we lost the grasp mid-carry.
            if float(self.ball.get_pose().p[2]) < release_z - 0.08:
                break
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            ))
            # Re-assert height after each XY move (planner may dip toward the train).
            self._move_ball_to_height(arm_tag=arm_tag, target_z=release_z)

        # Only release if the ball is still held over / near the rail drop.
        ball_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        held = float(self.ball.get_pose().p[2]) >= release_z - 0.08
        near_drop = float(np.linalg.norm(ball_xy - self._drop_target_xy)) <= max(reach_tol * 2.5, 0.06)
        if not (held and near_drop):
            if self._ball_rigid is not None:
                try:
                    self._ball_rigid.set_disable_gravity(False)
                except Exception:
                    pass
            self.info["info"] = self._episode_info(arm_name)
            return self.info

        timing_xy = self._drop_target_xy
        # Opt 1: wait only for the designated red target wagon.
        target_car = (
            int(self.target_wagon_idx)
            if self.target_wagon_mode and self.target_wagon_idx is not None
            else None
        )
        wait_cargo = (
            [target_car]
            if target_car is not None
            else list(self._cargo_indices)
        )

        # Lead so an open wagon is under the drop when the ball frees + lands.
        fall_h = max(
            1e-4,
            float(release_z - (self.car_floor_z + self.car_floor_h + self.ball_radius)),
        )
        fall_steps = int(round(np.sqrt(2.0 * fall_h / self.GRAVITY) * self.SIM_HZ))
        grip_free_steps = max(50, int(0.40 * self.release_open_steps))
        lead_steps = grip_free_steps + fall_steps
        wait_steps, _ = self._steps_until_wagon_under_drop(
            timing_xy,
            lead_steps=lead_steps,
            wagon_idx=target_car,
        )
        self._dwell(wait_steps)
        for _ in range(2500):
            eng = self._angle_at_step(self._train_step + lead_steps)
            dists = [
                float(np.linalg.norm(
                    np.asarray(self._car_pose_at(eng, i)[0].p[:2]) - timing_xy
                ))
                for i in wait_cargo
            ]
            if float(min(dists)) <= self.align_tol:
                break
            self._dwell(1)

        # Re-check pose after the wait; re-raise if the arm dipped.
        self._move_ball_to_height(arm_tag=arm_tag, target_z=release_z)
        ball_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        if float(np.linalg.norm(ball_xy - timing_xy)) > max(reach_tol * 2.5, 0.06):
            if self._ball_rigid is not None:
                try:
                    self._ball_rigid.set_disable_gravity(False)
                except Exception:
                    pass
            self.info["info"] = self._episode_info(arm_name)
            return self.info

        # Natural drop: open first, then enable latching so seating cannot happen
        # while the ball is still in the gripper mid-air.
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(False)
            except Exception:
                pass
        self._bed_contact_steps = 0
        self.move(self.open_gripper(arm_tag))
        self._ball_released = True
        self._latch_release_metric()
        # Retract clear of the train while the ball falls under gravity onto the bed.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="world"))
        # Densify third-view frames during the fall so the drop is visible (not 1–2 frames).
        old_save_freq = self.save_freq
        if self.save_freq:
            self.save_freq = max(1, min(int(self.save_freq), 3))
        self._dwell(max(self.post_release_steps, fall_steps + 250))
        self.save_freq = old_save_freq
        self.move(self.back_to_origin(arm_tag))
        self._dwell(80)

        self.ball_in_train = self._success_in_target_wagon()
        self.info["info"] = self._episode_info(arm_name)
        return self.info

    def _episode_info(self, arm_name: str) -> dict:
        if self.target_wagon_mode:
            target = "red wagon"
        else:
            target = "toy train"
        return {
            "{A}": "train_ball",
            "{B}": target,
            "{a}": arm_name,
            "{o}": self._option_label(),
        }

    def _success_in_target_wagon(self) -> bool:
        """True if the ball is seated in an allowed wagon (any / Opt-1 target)."""
        if self.target_wagon_mode and self.target_wagon_idx is not None:
            idx = int(self.target_wagon_idx)
            return bool(
                (self._ball_latched and self._latched_car_idx == idx)
                or self._ball_in_car(idx)
            )
        cargo = getattr(self, "_cargo_indices", list(range(1, getattr(self, "n_cars", 1))))
        return bool(
            self._ball_latched or any(self._ball_in_car(i) for i in cargo)
        )

    def check_success(self):
        self._try_latch_ball_off_table()
        off_table = bool(getattr(self, "_ball_fell_off_table", False))
        self.ball_in_train = (not off_table) and self._success_in_target_wagon()
        self._update_metrics()
        self.info["ball_side"] = str(getattr(self, "ball_side", "left"))
        self.info["selected_arm"] = str(getattr(self, "selected_arm", "left"))
        self.info["n_wagons"] = int(getattr(self, "n_wagons", max(0, getattr(self, "n_cars", 1) - 1)))
        self.info["n_cars"] = int(getattr(self, "n_cars", 0))
        self.info["ball_in_train"] = bool(self.ball_in_train)
        self.info["ball_off_table"] = bool(off_table)
        self.info["target_wagon_mode"] = bool(getattr(self, "target_wagon_mode", False))
        self.info["target_wagon_idx"] = (
            None if self.target_wagon_idx is None else int(self.target_wagon_idx)
        )
        self.info["latched_car_idx"] = (
            None if self._latched_car_idx is None else int(self._latched_car_idx)
        )
        self.info["tunnel_present"] = bool(getattr(self, "tunnel_present", False))
        self.info["tunnel_n_wagons"] = int(getattr(self, "tunnel_n_wagons", 0))
        self.info["option_label"] = self._option_label()
        if off_table:
            self._last_fail_reason = "ball dropped off the table"
        return bool(self.ball_in_train)

    def get_obs(self):
        obs = super().get_obs()
        ball_p = self.ball.get_pose().p.tolist() if self.ball is not None else [0.0, 0.0, 0.0]
        obs["train"] = {
            "train_angle": float(getattr(self, "_train_angle", 0.0)),
            "train_omega": float(getattr(self, "train_omega", 0.0)),
            "n_wagons": int(getattr(self, "n_wagons", 0)),
            "n_cars": int(getattr(self, "n_cars", 0)),
            "ball_side": str(getattr(self, "ball_side", "left")),
            "ball_center": ball_p,
            "drop_target_xy": (
                self._drop_target_xy.tolist() if hasattr(self, "_drop_target_xy") else [0.0, 0.0]
            ),
            "ball_in_train": bool(getattr(self, "ball_in_train", False)),
            "ball_off_table": bool(getattr(self, "_ball_fell_off_table", False)),
            "ball_latched": bool(getattr(self, "_ball_latched", False)),
            "target_wagon_mode": bool(getattr(self, "target_wagon_mode", False)),
            "target_wagon_idx": (
                -1 if self.target_wagon_idx is None else int(self.target_wagon_idx)
            ),
            "latched_car_idx": (
                -1 if self._latched_car_idx is None else int(self._latched_car_idx)
            ),
            "tunnel_present": bool(getattr(self, "tunnel_present", False)),
            "tunnel_n_wagons": int(getattr(self, "tunnel_n_wagons", 0)),
            "tunnel_center_angle": (
                None
                if getattr(self, "tunnel_center_angle", None) is None
                else float(self.tunnel_center_angle)
            ),
            "tunnel_center_angle_near": (
                None
                if getattr(self, "tunnel_center_angle_near", None) is None
                else float(self.tunnel_center_angle_near)
            ),
            "option_label": self._option_label(),
        }
        return obs

    # ------------------------------------------------- human-experiment metrics
    # Steps of no gripper/ball contact before the release is considered real
    # (finger contact flickers by a frame or two while the grasp settles).
    RELEASE_DEBOUNCE_STEPS = 5

    def _reset_metrics(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_release_step = None
        self._metric_seat_step = None
        self._metric_bed_offset_norm = None
        self._metric_ball_held = False
        self._metric_free_steps = 0

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_release_metric(self):
        """Explicit release marker for the scripted policy; first writer wins."""
        if getattr(self, "_metric_release_step", None) is None:
            self._metric_release_step = self._metric_step()

    def _track_release_metric(self):
        """Detect the ball leaving the gripper (teleop has no _ball_released flag).

        Only runs until the release is latched, so the contact query costs nothing
        for the rest of the episode.
        """
        if getattr(self, "_metric_release_step", None) is not None or self.ball is None:
            return
        try:
            held = len(self.get_gripper_actor_contact_position(self.ball.get_name())) > 0
        except Exception:
            return
        if held:
            self._metric_ball_held = True
            self._metric_free_steps = 0
            return
        if not getattr(self, "_metric_ball_held", False):
            return
        self._metric_free_steps = int(getattr(self, "_metric_free_steps", 0)) + 1
        if self._metric_free_steps >= int(self.RELEASE_DEBOUNCE_STEPS):
            # Back-date to the first contact-free frame.
            self._metric_release_step = max(
                0, self._metric_step() - self._metric_free_steps + 1
            )

    def _seated_wagon_idx(self):
        """Which open wagon currently holds the ball, or None."""
        if getattr(self, "_latched_car_idx", None) is not None:
            return int(self._latched_car_idx)
        if getattr(self, "target_wagon_mode", False) and self.target_wagon_idx is not None:
            idx = int(self.target_wagon_idx)
            return idx if self._ball_in_car(idx) else None
        for i in getattr(self, "_cargo_indices", []):
            if self._ball_in_car(i):
                return int(i)
        return None

    def _latch_seat_metrics(self):
        """Record when the ball first counted as loaded and how centred it landed.

        High-water mark: ``ball_in_train`` can flip back to False if the ball
        bounces out of the bed, so only the first True is kept.
        """
        if getattr(self, "_metric_seat_step", None) is not None:
            return
        try:
            if not self._success_in_target_wagon():
                return
        except Exception:
            return
        self._metric_seat_step = self._metric_step()
        try:
            idx = self._seated_wagon_idx()
            if idx is None:
                self._metric_bed_offset_norm = None
                return
            local = self._world_to_car_local(
                self.ball.get_pose().p, self.cars[idx].get_pose()
            )
            hl, hw = self._wagon_bed_half()
            self._metric_bed_offset_norm = float(max(
                abs(float(local[0])) / max(float(hl), 1e-9),
                abs(float(local[1])) / max(float(hw), 1e-9),
            ))
        except Exception:
            self._metric_bed_offset_norm = None

    def _update_metrics(self):
        try:
            self._track_release_metric()
            self._latch_seat_metrics()
        except Exception:
            pass

    def _compute_metrics(self):
        """extra1 = release->loaded latency, extra2 = landing offset in bed half-widths.

        ``bed_offset_norm`` is the ball's offset from the wagon bed centre at the
        instant it first counted as loaded, as a fraction of the bed half-extent
        (max over the along-track and across-track axes): 0.0 = dead centre,
        1.0 = touching a wagon wall. LOWER IS BETTER. None when the ball never
        landed in an accepted wagon.
        """
        metrics = {
            "load_latency_steps": None,
            "load_latency_s": None,
            "bed_offset_norm": (
                None
                if getattr(self, "_metric_bed_offset_norm", None) is None
                else float(self._metric_bed_offset_norm)
            ),
        }
        start = getattr(self, "_metric_release_step", None)
        seat = getattr(self, "_metric_seat_step", None)
        if start is not None and seat is not None and seat >= start:
            steps = int(seat - start)
            metrics["load_latency_steps"] = steps
            try:
                metrics["load_latency_s"] = round(steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics
