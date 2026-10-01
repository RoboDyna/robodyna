from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d


class catch_ramp_ball(Base_Task):
    """Catch a red ball dropped onto a long ramp that meets the back wall.

    Balls drop at the highest (+y) point of the ramp, then roll under Newton
    incline dynamics (a = g sin θ) with rail / ball–ball bounce. At the front
    edge they are released to PhysX for the free-flight catch.

    Options (independent toggles; CLI via ``--task-arg`` or legacy ``--option``):
      - Default — straight roll down the ramp.
      - Option 1 — ``wall_bounce_enabled``: lateral heading so the ball rebounds
        off a side rail. CLI: ``--task-arg wall_bounce_enabled=true``.
      - Option 2 — ``enable_distractor``: blue distractor on a separate lane;
        catching it fails. CLI: ``--task-arg enable_distractor=true``.
      Options 1+2 may be combined. Both balls sample initial speed ±20%.
    """

    SIM_HZ = 250.0
    GRAVITY = 9.81

    # Wall front face ≈ y=0.4 (wall pose y=1, half_y=0.6).
    RAMP_BACK_Y_DEFAULT = 0.395
    RAMP_FRONT_Y_DEFAULT = 0.02
    RAMP_ANGLE_DEFAULT = 0.0672  # gentler incline → slower roll
    RAMP_HALF_X = 0.20
    RAMP_HALF_Z = 0.008
    RAIL_H = 0.032
    RAIL_THICKNESS = 0.012
    LOW_EDGE_Z = 0.08

    BALL_RADIUS_DEFAULT = 0.018
    CUP_CENTER_Z = 0.043
    CUP_FWD_CLEARANCE_DEFAULT = 0.0
    RELEASE_CLEARANCE_DEFAULT = 0.055  # m past front lip ( > ball radius)
    RAMP_FRICTION_DEFAULT = 0.06       # kinetic μ; keep a > 0 on the gentle ramp
    BALL_REST_Z = 0.035
    X_SPAN_DEFAULT = 0.15
    DROP_HEIGHT_DEFAULT = 0.10
    DROP_TIME_DEFAULT = 0.50
    # Per-episode ±jitter on ramp_* + ball_radius when randomize_ramp_params.
    RANDOMIZE_RAMP_PARAMS_DEFAULT = False
    RAMP_PARAM_JITTER_DEFAULT = 0.10

    WALL_BOUNCE_ENABLED_DEFAULT = False
    ENABLE_DISTRACTOR_DEFAULT = False
    DISTRACTOR_COLOR_DEFAULT = [0.15, 0.35, 0.92]
    BALL_PATH_MODE_DEFAULT = "straight"
    WALL_ANGLE_MIN_DEFAULT = 20.0
    WALL_ANGLE_MAX_DEFAULT = 40.0

    # Initial along-ramp speed at the top after the drop (±20% sample).
    # Effective speeds are further multiplied by BALL_SPEED_SCALE (~5× slower).
    ROLL_SPEED_DEFAULT = 0.049
    ROLL_SPEED_SCALE_MIN_DEFAULT = 0.8
    ROLL_SPEED_SCALE_MAX_DEFAULT = 1.2
    BALL_SPEED_SCALE = 0.2

    FALL_TIME_DEFAULT = 0.55
    IDLE_TIME_MIN_DEFAULT = 0.3
    IDLE_TIME_MAX_DEFAULT = 0.6
    SETTLE_STEPS_DEFAULT = 90
    RIM_RADIUS_DEFAULT = 0.040

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_ramp_ball", {})
        self._loaded = False
        self._ball_phase = None
        self._reset_metric_state()
        self._distractor_phase = None
        self._cup_ready = False
        self._expert_demo = False
        self.distractor = None
        self._distractor_comp = None
        self.enable_distractor = False
        self.wall_bounce_enabled = False
        super()._init_task_env_(**kwags)
        # After settle: start here so policy eval / interactive see the roll
        # without play_once. Starting *during* _init_task_env_ would finish the
        # roll before the episode begins.
        self._start_ball_motion(expert_demo=False)

    # ---------------------------------------------------------------- helpers
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
        raise ValueError(f"catch_ramp_ball expected a boolean, got {value!r}")

    def _parse_wall_bounce_enabled(self, c) -> bool:
        wall = c.get("wall_bounce_enabled", c.get("opt1", None))
        legacy = c.get("option", None)
        if legacy is not None and wall is None:
            if legacy in (1, "1", "wall_bounce", "wall_bounce_enabled", "angled"):
                wall = True
            elif legacy in (2, "2", "enable_distractor", "distractor"):
                wall = False
            else:
                raise ValueError(
                    "catch_ramp_ball option must be 1/wall_bounce_enabled or "
                    "2/enable_distractor (or set the booleans directly)"
                )
        if wall is not None:
            return self._as_bool(wall, self.WALL_BOUNCE_ENABLED_DEFAULT)
        mode = str(c.get("ball_path_mode", self.BALL_PATH_MODE_DEFAULT)).strip().lower()
        if mode == "angled":
            return True
        if mode == "random":
            return bool(np.random.choice([False, True]))
        return bool(self.WALL_BOUNCE_ENABLED_DEFAULT)

    def _parse_enable_distractor(self, c) -> bool:
        distractor = c.get("enable_distractor", c.get("opt2", None))
        legacy = c.get("option", None)
        if legacy is not None and distractor is None:
            if legacy in (2, "2", "enable_distractor", "distractor"):
                distractor = True
            elif legacy in (1, "1", "wall_bounce", "wall_bounce_enabled", "angled"):
                distractor = False
            else:
                raise ValueError(
                    "catch_ramp_ball option must be 1/wall_bounce_enabled or "
                    "2/enable_distractor (or set the booleans directly)"
                )
        return self._as_bool(distractor, self.ENABLE_DISTRACTOR_DEFAULT)

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "wall_bounce_enabled", False):
            parts.append("option 1")
        if getattr(self, "enable_distractor", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _sample_speed(self, nominal, scale_min, scale_max):
        scale = float(np.random.uniform(min(scale_min, scale_max), max(scale_min, scale_max)))
        return float(max(1e-4, nominal * scale))

    @staticmethod
    def _sample_pm(mean, jitter):
        """Sample uniformly in ``[mean*(1-jitter), mean*(1+jitter)]``."""
        mean = float(mean)
        jitter = float(max(jitter, 0.0))
        lo, hi = mean * (1.0 - jitter), mean * (1.0 + jitter)
        if lo > hi:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-12:
            return mean
        return float(np.random.uniform(lo, hi))

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + self.table_z_bias
        self.randomize_ramp_params = bool(
            c.get("randomize_ramp_params", self.RANDOMIZE_RAMP_PARAMS_DEFAULT)
        )
        ramp_jitter = float(c.get("ramp_param_jitter", self.RAMP_PARAM_JITTER_DEFAULT))
        ramp_jitter = float(np.clip(abs(ramp_jitter), 0.0, 0.95))
        j = ramp_jitter if self.randomize_ramp_params else 0.0

        ang = self._sample_pm(c.get("ramp_angle", self.RAMP_ANGLE_DEFAULT), j)
        self.ramp_angle = ang
        self.ball_radius = self._sample_pm(
            c.get("ball_radius", self.BALL_RADIUS_DEFAULT), j
        )
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.rim_radius = float(c.get("rim_radius", self.RIM_RADIUS_DEFAULT))
        self.drop_height = float(c.get("drop_height", self.DROP_HEIGHT_DEFAULT))
        self.drop_time = max(0.05, float(c.get("drop_time", self.DROP_TIME_DEFAULT)))
        self.drop_steps = max(1, int(round(self.drop_time * self.SIM_HZ)))

        self.wall_bounce_enabled = self._parse_wall_bounce_enabled(c)
        self.enable_distractor = self._parse_enable_distractor(c)
        self.distractor_color = list(c.get("distractor_color", self.DISTRACTOR_COLOR_DEFAULT))
        self.ball_path_mode = "angled" if self.wall_bounce_enabled else "straight"

        wall_angle_min = float(c.get("angled_path_min_deg", c.get("wall_angle_min_deg", self.WALL_ANGLE_MIN_DEFAULT)))
        wall_angle_max = float(c.get("angled_path_max_deg", c.get("wall_angle_max_deg", self.WALL_ANGLE_MAX_DEFAULT)))
        x_span = float(c.get("x_span", self.X_SPAN_DEFAULT))
        self.cup_fwd_clearance = float(c.get("cup_fwd_clearance", self.CUP_FWD_CLEARANCE_DEFAULT))
        self.release_clearance = float(
            c.get("release_clearance", self.RELEASE_CLEARANCE_DEFAULT)
        )
        self.ramp_friction = self._sample_pm(
            c.get("ramp_friction", self.RAMP_FRICTION_DEFAULT), j
        )
        fall_t = float(c.get("fall_time", self.FALL_TIME_DEFAULT))
        idle_t = float(np.random.uniform(
            c.get("idle_time_min", self.IDLE_TIME_MIN_DEFAULT),
            c.get("idle_time_max", self.IDLE_TIME_MAX_DEFAULT),
        ))
        self.fall_time, self.idle_time = fall_t, idle_t
        self.fall_steps = int(round(fall_t * self.SIM_HZ))
        self.idle_steps = int(round(idle_t * self.SIM_HZ))

        # Ramp sized so the high end meets the white back wall.
        back_y = self._sample_pm(c.get("ramp_back_y", self.RAMP_BACK_Y_DEFAULT), j)
        front_y = self._sample_pm(c.get("ramp_front_y", self.RAMP_FRONT_Y_DEFAULT), j)
        if front_y >= back_y:
            front_y = float(back_y) - 0.02
        cos_a = float(np.cos(ang))
        sin_a = float(np.sin(ang))
        hy = (back_y - front_y) / max(2.0 * cos_a, 1e-6)
        cy = 0.5 * (back_y + front_y)
        hx, hz = self.RAMP_HALF_X, self.RAMP_HALF_Z
        self._hx, self._hy, self._hz = hx, hy, hz
        self.ramp_center_y = cy

        # Along-board local y: +y high/back (near wall), −y low/front.
        self._y_top = hy * 0.98
        self._y_edge = -hy * 0.95
        self._along_run = abs(self._y_top - self._y_edge)
        self.path_length = float(self._along_run / max(cos_a, 1e-6))
        # Incline acceleration along the board: a = g (sinθ − μ cosθ),
        # then scaled so balls move ~5× slower than the real incline.
        mu = float(np.clip(self.ramp_friction, 0.0, 0.95))
        a_phys = float(self.GRAVITY * (sin_a - mu * cos_a))
        if a_phys < 1e-4:
            a_phys = 1e-4
        self._accel_along = float(a_phys * (self.BALL_SPEED_SCALE ** 2))

        lane_half = min(x_span, hx - 0.5 * self.RAIL_THICKNESS - self.ball_radius - 0.001)
        lane_half = max(lane_half, self.ball_radius + 0.005)
        self._lane_min = -lane_half
        self._lane_max = lane_half
        self.min_exit_separation = 2.0 * self.ball_radius

        self._opt1_prefer_start = None
        if not self.wall_bounce_enabled:
            self.wall_angle_deg = 0.0
        else:
            wall_abs = float(np.random.uniform(min(wall_angle_min, wall_angle_max), max(wall_angle_min, wall_angle_max)))
            bounce_side = float(np.random.choice([-1.0, 1.0]))
            width = self._lane_max - self._lane_min
            if bounce_side > 0:
                self._opt1_prefer_start = -1.0
                start_guess = self._lane_min + 0.2 * width
                dist = self._lane_max - start_guess
            else:
                self._opt1_prefer_start = 1.0
                start_guess = self._lane_max - 0.2 * width
                dist = start_guess - self._lane_min
            min_angle = float(np.rad2deg(np.arctan((dist + 0.02) / max(self._along_run, 1e-6))))
            wall_abs = min(max(wall_abs, min_angle), 55.0)
            self.wall_angle_deg = float(bounce_side * wall_abs)

        nominal_speed = float(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        scale_min = float(c.get("roll_speed_scale_min", self.ROLL_SPEED_SCALE_MIN_DEFAULT))
        scale_max = float(c.get("roll_speed_scale_max", self.ROLL_SPEED_SCALE_MAX_DEFAULT))
        self.ball_speed = (
            self._sample_speed(nominal_speed, scale_min, scale_max) * self.BALL_SPEED_SCALE
        )
        self.distractor_speed = (
            self._sample_speed(nominal_speed, scale_min, scale_max) * self.BALL_SPEED_SCALE
            if self.enable_distractor else self.ball_speed
        )
        self.roll_speed_nominal = nominal_speed * self.BALL_SPEED_SCALE

        # Time to traverse the incline from rest with a = g sinθ starting at v0:
        # s = v0 t + ½ a t²  →  solve for t.
        def _roll_duration(v0):
            a = max(self._accel_along, 1e-6)
            disc = v0 * v0 + 2.0 * a * self.path_length
            return float((-v0 + np.sqrt(disc)) / a)

        self.roll_time = _roll_duration(self.ball_speed)
        self.distractor_roll_time = _roll_duration(self.distractor_speed)
        self.roll_steps = max(
            1, int(round(max(self.roll_time, self.distractor_roll_time) * self.SIM_HZ))
        )

        q = t3d.euler.euler2quat(ang, 0, 0)
        self._R = t3d.quaternions.quat2mat(q)
        # Front underside sits LOW_EDGE_Z above the table.
        cz = self.table_top + self.LOW_EDGE_Z + hy * sin_a
        self.ramp_center = np.array([0.0, cy, cz])

        self.ramp = create_box(
            self.scene, sapien.Pose(self.ramp_center.tolist(), q.tolist()),
            half_size=[hx, hy, hz], color=[0.72, 0.64, 0.52],
            is_static=True, name="ramp",
        )
        for sx in (-1.0, 1.0):
            off = np.array([sx * hx, 0.0, self.RAIL_H * 0.5 + hz])
            rail_p = self.ramp_center + self._R @ off
            create_box(
                self.scene, sapien.Pose(rail_p.tolist(), q.tolist()),
                half_size=[0.5 * self.RAIL_THICKNESS, hy, self.RAIL_H * 0.5],
                color=[0.5, 0.42, 0.32],
                is_static=True, name=f"rail_{'L' if sx < 0 else 'R'}",
            )
        for tag, ly in (("back", hy * 0.8), ("front", -hy * 0.8), ("mid", 0.0)):
            top = self.ramp_center + self._R @ np.array([0.0, ly, -hz])
            leg_h = max(0.02, float(top[2] - self.table_top))
            create_box(
                self.scene,
                sapien.Pose([0.0, float(top[1]), self.table_top + leg_h * 0.5]),
                half_size=[0.05, 0.022, leg_h * 0.5],
                color=[0.45, 0.4, 0.36],
                is_static=True, name=f"ramp_support_{tag}",
            )

        top_c = self._surface_point(0.0, self._y_top)
        edge_c = self._surface_point(0.0, self._y_edge)
        downhill = edge_c - top_c
        self._downhill = downhill / max(np.linalg.norm(downhill), 1e-6)
        self.ramp_top_y = float(top_c[1])
        self.ramp_exit_y = float(edge_c[1])

        self.ball_ball_bounces = 0
        self.drop_wall_bounces = 0
        self.exit_separation = 0.0
        self._bake_ball_lanes()

        self.ball_top = self._surface_point(self.ball_start_x, self._y_top)
        self.ball_edge = self._surface_point(self.ball_exit_x, self._y_edge)
        self.ball_drop = self.ball_top + np.array([0.0, 0.0, self.drop_height])
        if self.enable_distractor:
            self.distractor_top = self._surface_point(self.distractor_start_x, self._y_top)
            self.distractor_edge = self._surface_point(self.distractor_exit_x, self._y_edge)
            self.distractor_drop = self.distractor_top + np.array([0.0, 0.0, self.drop_height])
        else:
            self.distractor_top = self.distractor_edge = self.distractor_drop = None

        self._update_release_velocity()
        self._compute_landing()
        if self.enable_distractor:
            self._update_distractor_release_velocity()
            self._compute_distractor_landing()

        self.ball = create_sphere(
            self.scene, sapien.Pose(self.ball_drop.tolist()),
            radius=self.ball_radius, color=[0.85, 0.18, 0.18],
            is_static=False, name="ball",
        )
        self._ball_comp = self._get_rigid(self.ball)
        self._configure_ball_physics(self._ball_comp, self.ball_drop)

        self.distractor = None
        self._distractor_comp = None
        self._distractor_phase = None
        self.distractor_landing = None
        self.distractor_release_vel = None
        if self.enable_distractor:
            self._spawn_distractor()

        self.cup_id = 0
        # Spawn the cup well in front of the catch zone so the robot must
        # visibly carry it into place (not already sitting under the lip).
        cup_pose = rand_pose(
            xlim=[-0.12, 0.12], ylim=[-0.24, -0.16], zlim=[self.table_top],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
        )
        self.cup = create_actor(
            self, pose=cup_pose, modelname="021_cup", model_id=self.cup_id,
            convex=True, is_static=False,
        )
        try:
            self.cup.set_mass(0.05)
        except Exception:
            pass
        try:
            cup_mat = sapien.physx.PhysxMaterial(0.9, 0.9, 0.0)
            cc = self._cup_comp()
            if cc is not None:
                for sh in cc.get_collision_shapes():
                    sh.set_physical_material(cup_mat)
        except Exception:
            pass

        self.add_prohibit_area(self.ramp, padding=0.02)
        self.add_prohibit_area(self.cup, padding=0.05)

        self._ball_phase = "frozen"
        self._roll_i = 0
        self._reset_metric_state()
        self._loaded = True

    def _surface_point(self, local_x, local_y):
        off = np.array([float(local_x), float(local_y), self._hz + self.ball_radius])
        return self.ramp_center + self._R @ off

    def _y_from_s(self, s):
        frac = float(np.clip(s / max(self.path_length, 1e-9), 0.0, 1.0))
        return self._y_top + (self._y_edge - self._y_top) * frac

    def _get_rigid(self, actor):
        entity = actor.actor if hasattr(actor, "actor") else actor
        for component in entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    def _configure_ball_physics(self, rigid, pose):
        if rigid is None:
            return
        material = sapien.physx.PhysxMaterial(0.4, 0.35, 0.08)
        try:
            for sh in rigid.get_collision_shapes():
                sh.set_physical_material(material)
        except Exception:
            pass
        rigid.set_disable_gravity(True)
        rigid.set_kinematic(True)
        rigid.set_kinematic_target(sapien.Pose(np.asarray(pose).tolist()))

    def _sample_lane_x(self, prefer_side=None):
        if prefer_side is None:
            return float(np.random.uniform(self._lane_min, self._lane_max))
        mid = 0.5 * (self._lane_min + self._lane_max)
        if prefer_side < 0:
            return float(np.random.uniform(self._lane_min, mid))
        return float(np.random.uniform(mid, self._lane_max))

    def _sample_distractor_start_x(self, red_x):
        min_sep = self.min_exit_separation
        candidates = []
        if red_x + min_sep <= self._lane_max + 1e-9:
            candidates.append((red_x + min_sep, self._lane_max))
        if red_x - min_sep >= self._lane_min - 1e-9:
            candidates.append((self._lane_min, red_x - min_sep))
        if not candidates:
            return float(self._lane_max if abs(self._lane_max - red_x) >= abs(red_x - self._lane_min) else self._lane_min)
        low, high = candidates[int(np.random.randint(len(candidates)))]
        if high < low:
            low, high = high, low
        return float(low if high - low < 1e-6 else np.random.uniform(low, high))

    def _step_lateral_x(self, x, slope, ds):
        dy = abs(ds * np.cos(self.ramp_angle))
        x = float(x + slope * dy)
        hit = False
        for _ in range(6):
            if x < self._lane_min:
                x = self._lane_min + (self._lane_min - x)
                slope = -slope
                hit = True
            elif x > self._lane_max:
                x = self._lane_max - (x - self._lane_max)
                slope = -slope
                hit = True
            else:
                break
        return float(np.clip(x, self._lane_min, self._lane_max)), float(slope), hit

    def _resolve_ball_ball(self, s_r, x_r, slope_r, spd_r, s_d, x_d, slope_d, spd_d, x_r_prev, x_d_prev):
        min_sep = self.min_exit_separation
        if abs(x_r - x_d) >= min_sep - 1e-9 or abs(s_r - s_d) >= min_sep - 1e-9:
            return s_r, x_r, slope_r, spd_r, s_d, x_d, slope_d, spd_d, False
        closing = (x_r - x_d) * ((x_r - x_r_prev) - (x_d - x_d_prev)) < 0.0
        overlapping = abs(x_r - x_d) < min_sep
        if not (closing or overlapping):
            return s_r, x_r, slope_r, spd_r, s_d, x_d, slope_d, spd_d, False
        slope_r, slope_d = float(slope_d), float(slope_r)
        spd_r, spd_d = float(spd_d), float(spd_r)
        mid = 0.5 * (x_r + x_d)
        half = 0.5 * min_sep
        if x_r <= x_d:
            x_r, x_d = mid - half, mid + half
        else:
            x_r, x_d = mid + half, mid - half
        x_r = float(np.clip(x_r, self._lane_min, self._lane_max))
        x_d = float(np.clip(x_d, self._lane_min, self._lane_max))
        if abs(s_r - s_d) < min_sep - 1e-9:
            smid = 0.5 * (s_r + s_d)
            if s_r <= s_d:
                s_r, s_d = max(0.0, smid - half), min(self.path_length, smid + half)
            else:
                s_r, s_d = min(self.path_length, smid + half), max(0.0, smid - half)
        return s_r, x_r, slope_r, spd_r, s_d, x_d, slope_d, spd_d, True

    def _initial_roll_state(self):
        return {
            "s_r": 0.0,
            "x_r": float(self.ball_start_x),
            "slope_r": float(np.tan(np.deg2rad(self.wall_angle_deg))),
            "spd_r": float(self.ball_speed),
            "red_done": False,
            "s_d": 0.0,
            "x_d": float(self.distractor_start_x or 0.0),
            "slope_d": 0.0,
            "spd_d": float(self.distractor_speed),
            "blu_done": not self.enable_distractor,
        }

    def _integrate_roll_step(self, state, dt):
        """Advance along the incline with a = g sinθ; reflect at rails / balls."""
        rail_hits = 0
        ball_hits = 0
        a = self._accel_along

        if not state["red_done"]:
            x_r_prev = float(state["x_r"])
            state["spd_r"] = float(state["spd_r"] + a * dt)
            ds_r = float(state["spd_r"]) * dt
            state["s_r"] = min(self.path_length, state["s_r"] + ds_r)
            state["x_r"], state["slope_r"], hit_r = self._step_lateral_x(
                state["x_r"], state["slope_r"], ds_r
            )
            if hit_r:
                rail_hits += 1
            if state["s_r"] >= self.path_length - 1e-9:
                state["red_done"] = True
                state["s_r"] = self.path_length
        else:
            x_r_prev = float(state["x_r"])

        if self.enable_distractor and not state["blu_done"]:
            x_d_prev = float(state["x_d"])
            state["spd_d"] = float(state["spd_d"] + a * dt)
            ds_d = float(state["spd_d"]) * dt
            state["s_d"] = min(self.path_length, state["s_d"] + ds_d)
            state["x_d"], state["slope_d"], hit_d = self._step_lateral_x(
                state["x_d"], state["slope_d"], ds_d
            )
            if hit_d:
                rail_hits += 1
            if state["s_d"] >= self.path_length - 1e-9:
                state["blu_done"] = True
                state["s_d"] = self.path_length
        else:
            x_d_prev = float(state.get("x_d", 0.0))

        if self.enable_distractor:
            (
                state["s_r"], state["x_r"], state["slope_r"], state["spd_r"],
                state["s_d"], state["x_d"], state["slope_d"], state["spd_d"],
                bounced,
            ) = self._resolve_ball_ball(
                state["s_r"], state["x_r"], state["slope_r"], state["spd_r"],
                state["s_d"], state["x_d"], state["slope_d"], state["spd_d"],
                x_r_prev, x_d_prev,
            )
            if bounced:
                ball_hits += 1
                state["red_done"] = state["s_r"] >= self.path_length - 1e-9
                state["blu_done"] = state["s_d"] >= self.path_length - 1e-9
                if state["red_done"]:
                    state["s_r"] = self.path_length
                if state["blu_done"]:
                    state["s_d"] = self.path_length
        return rail_hits, ball_hits

    def _bake_ball_lanes(self):
        """Offline-simulate the gravity roll to get exit states for landing."""
        lateral_slope0 = float(np.tan(np.deg2rad(self.wall_angle_deg)))
        dt = 1.0 / self.SIM_HZ
        max_steps = int(round(max(self.roll_time, self.distractor_roll_time) * self.SIM_HZ * 1.5)) + 50

        best = None
        for _ in range(32):
            prefer = getattr(self, "_opt1_prefer_start", None)
            x_r0 = self._sample_lane_x(prefer_side=prefer)
            x_d0 = self._sample_distractor_start_x(x_r0) if self.enable_distractor else None
            self.ball_start_x = float(x_r0)
            self.distractor_start_x = float(x_d0) if x_d0 is not None else None
            self._slope_r0 = lateral_slope0
            self._slope_d0 = 0.0

            state = self._initial_roll_state()
            rail_hits = ball_hits = 0
            for _step in range(max_steps):
                if state["red_done"] and state["blu_done"]:
                    break
                rh, bh = self._integrate_roll_step(state, dt)
                rail_hits += rh
                ball_hits += bh

            exit_sep = (
                float(abs(state["x_r"] - state["x_d"]))
                if self.enable_distractor else self.min_exit_separation
            )
            score = exit_sep + 0.5 * float(rail_hits > 0) + 0.25 * float(ball_hits > 0)
            candidate = (
                float(x_r0),
                float(x_d0) if x_d0 is not None else None,
                float(state["x_r"]),
                float(state["x_d"]) if self.enable_distractor else None,
                float(state["slope_r"]),
                float(state["slope_d"]) if self.enable_distractor else 0.0,
                float(state["spd_r"]),
                float(state["spd_d"]) if self.enable_distractor else self.ball_speed,
                int(rail_hits), int(ball_hits), exit_sep, score,
            )
            if best is None or score > best[11]:
                best = candidate
            good_sep = (not self.enable_distractor) or exit_sep >= self.min_exit_separation - 1e-6
            good_wall = (not self.wall_bounce_enabled) or rail_hits >= 1
            if good_sep and good_wall:
                break

        (
            x_r0, x_d0, x_re, x_de, slope_re, slope_de, spd_re, spd_de,
            rail_hits, ball_hits, exit_sep, _score,
        ) = best
        self.ball_start_x = float(x_r0)
        self.ball_exit_x = float(x_re)
        self.ball_exit_slope = float(slope_re)
        self.ball_exit_speed = float(spd_re)
        self._slope_r0 = lateral_slope0
        self._slope_d0 = 0.0
        self.drop_wall_bounces = int(rail_hits)
        self.ball_ball_bounces = int(ball_hits)
        self.exit_separation = float(exit_sep)
        if self.enable_distractor:
            self.distractor_start_x = float(x_d0)
            self.distractor_exit_x = float(x_de)
            self.distractor_exit_slope = float(slope_de)
            self.distractor_exit_speed = float(spd_de)
        else:
            self.distractor_start_x = None
            self.distractor_exit_x = None
            self.distractor_exit_slope = 0.0
            self.distractor_exit_speed = self.ball_speed

    def _spawn_distractor(self):
        self.distractor = create_sphere(
            self.scene, sapien.Pose(self.distractor_drop.tolist()),
            radius=self.ball_radius, color=self.distractor_color,
            is_static=False, name="ramp_distractor_ball",
        )
        self._distractor_comp = self._get_rigid(self.distractor)
        self._configure_ball_physics(self._distractor_comp, self.distractor_drop)

    def _release_pose_and_vel(self, edge_pos, along_speed, lateral_slope):
        """World pose / velocity just past the front lip (true incline exit)."""
        along_speed = float(along_speed)
        lateral_vx = float(lateral_slope * along_speed * np.cos(self.ramp_angle))
        # Clear the ramp lip so PhysX does not tunnel into the board.
        clear = max(float(self.release_clearance), self.ball_radius + 0.02)
        pos = np.asarray(edge_pos, dtype=float) + self._downhill * clear
        vel = (
            self._downhill * along_speed
            + np.array([lateral_vx, 0.0, 0.0], dtype=float)
        )
        return pos, vel

    def _ballistic_fall_time(self, z0, vz0, z_rest):
        """Time for z0 + vz0 t − ½ g t² to reach z_rest (first positive root)."""
        dz = float(z0 - z_rest)
        if dz <= 1e-6:
            return 1e-3
        a = 0.5 * self.GRAVITY
        b = -float(vz0)
        c = -dz
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return float(np.sqrt(2.0 * dz / self.GRAVITY))
        t = (-b + np.sqrt(disc)) / (2.0 * a)
        return float(max(t, 1e-3))

    def _ballistic_landing_xy(self, pos, vel):
        z_rest = self.table_top + self.BALL_REST_Z
        t_fall = self._ballistic_fall_time(pos[2], vel[2], z_rest)
        return np.array([
            float(pos[0] + vel[0] * t_fall),
            float(pos[1] + vel[1] * t_fall),
        ]), t_fall

    def _catch_aim_xy(self, pos, vel):
        """Cup aim point: where the trajectory crosses the cup rim height.

        Aiming at the table-contact landing puts the near rim in the flight
        path, so the ball glances off instead of dropping into the vessel.
        """
        rim_z = self.table_top + 0.075
        t_rim = self._ballistic_fall_time(pos[2], vel[2], rim_z)
        return np.array([
            float(pos[0] + vel[0] * t_rim),
            float(pos[1] + vel[1] * t_rim),
        ])

    def _update_release_velocity(self):
        along_speed = float(getattr(self, "ball_exit_speed", self.ball_speed))
        slope = float(getattr(self, "ball_exit_slope", np.tan(np.deg2rad(self.wall_angle_deg))))
        pos, vel = self._release_pose_and_vel(self.ball_edge, along_speed, slope)
        self.ball_release_pos = pos
        self.release_vel = vel
        landing, t_fall = self._ballistic_landing_xy(pos, vel)
        self._t_fall = t_fall
        self.landing = landing
        self.catch_aim = self._catch_aim_xy(pos, vel)

    def _update_distractor_release_velocity(self):
        if not self.enable_distractor or self.distractor_edge is None:
            return
        along_speed = float(getattr(self, "distractor_exit_speed", self.distractor_speed))
        slope = float(getattr(self, "distractor_exit_slope", 0.0))
        pos, vel = self._release_pose_and_vel(self.distractor_edge, along_speed, slope)
        self.distractor_release_pos = pos
        self.distractor_release_vel = vel
        landing, _ = self._ballistic_landing_xy(pos, vel)
        self.distractor_landing = landing
        self.distractor_catch_aim = self._catch_aim_xy(pos, vel)

    def _compute_landing(self):
        if getattr(self, "release_vel", None) is None or getattr(self, "ball_edge", None) is None:
            return
        self._update_release_velocity()

    def _compute_distractor_landing(self):
        if not self.enable_distractor:
            return
        self._update_distractor_release_velocity()

    def _predict_landing(self):
        return np.array(self.landing), 0.0

    def _cup_comp(self):
        for cc in self.cup.actor.get_components():
            if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                return cc
        return None

    # ---------------------------------------------------------- ball motion
    def _start_ball_motion(self, expert_demo):
        if not getattr(self, "_loaded", False):
            return
        # load_actors parks the ball as "frozen". That is not a started run —
        # skip only if the drop/roll is already live (interactive calls this
        # again after setup_demo).
        if self._ball_phase in ("dropping", "rolling", "released") and not expert_demo:
            return
        self._expert_demo = bool(expert_demo)
        self._cup_ready = False
        self._reset_metric_state()
        # Metrics: the drop is what the human reacts to; the catch window opens here.
        self._metric_start_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        self._ball_phase = "dropping"
        self._drop_i = 0
        self._roll_i = 0
        self._roll_state = self._initial_roll_state()
        self.ball.set_pose(sapien.Pose(self.ball_drop.tolist()))
        if self._ball_comp is not None:
            self._ball_comp.set_disable_gravity(True)
            self._ball_comp.set_kinematic(True)
            self._ball_comp.set_linear_velocity(np.zeros(3))
            self._ball_comp.set_angular_velocity(np.zeros(3))
            self._ball_comp.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))
        if self.enable_distractor and self.distractor is not None:
            self._distractor_phase = "dropping"
            self.distractor.set_pose(sapien.Pose(self.distractor_drop.tolist()))
            if self._distractor_comp is not None:
                self._distractor_comp.set_disable_gravity(True)
                self._distractor_comp.set_kinematic(True)
                self._distractor_comp.set_linear_velocity(np.zeros(3))
                self._distractor_comp.set_angular_velocity(np.zeros(3))
                self._distractor_comp.set_kinematic_target(
                    sapien.Pose(self.distractor_drop.tolist())
                )

    def _set_kinematic_pose(self, position, roll_distance=0.0, actor=None, rigid=None):
        roll_angle = -float(roll_distance) / max(self.ball_radius, 1e-6)
        quat = t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], roll_angle)
        pose = sapien.Pose(np.asarray(position).tolist(), quat.tolist())
        target_rigid = rigid if rigid is not None else self._ball_comp
        target_actor = actor if actor is not None else self.ball
        if target_rigid is not None:
            target_rigid.set_kinematic_target(pose)
        else:
            target_actor.set_pose(pose)

    def _advance_ball(self):
        if self._ball_phase == "dropping":
            self._drop_i += 1
            frac = min(1.0, self._drop_i / float(self.drop_steps))
            vert = frac * frac
            pos = np.array([
                self.ball_top[0],
                self.ball_top[1],
                self.ball_drop[2] + (self.ball_top[2] - self.ball_drop[2]) * vert,
            ])
            self._set_kinematic_pose(pos)
            if self.enable_distractor and self._distractor_phase == "dropping":
                dpos = np.array([
                    self.distractor_top[0],
                    self.distractor_top[1],
                    self.distractor_drop[2]
                    + (self.distractor_top[2] - self.distractor_drop[2]) * vert,
                ])
                self._set_kinematic_pose(
                    dpos, actor=self.distractor, rigid=self._distractor_comp
                )
            if frac >= 1.0:
                # Always start rolling from the top after the drop — the robot
                # places the cup while the ball is already moving downhill.
                self._ball_phase = "rolling"
                if self.enable_distractor:
                    self._distractor_phase = "rolling"

        elif self._ball_phase == "rolling":
            dt = 1.0 / self.SIM_HZ
            self._roll_i += 1
            state = self._roll_state
            rh, bh = self._integrate_roll_step(state, dt)
            self.drop_wall_bounces += int(rh)
            self.ball_ball_bounces += int(bh)

            if state["red_done"] and state["blu_done"]:
                self.ball_edge = self._surface_point(state["x_r"], self._y_edge)
                self.ball_exit_x = float(state["x_r"])
                self.ball_exit_slope = float(state["slope_r"])
                self.ball_exit_speed = float(state["spd_r"])
                if self.enable_distractor:
                    self.distractor_edge = self._surface_point(state["x_d"], self._y_edge)
                    self.distractor_exit_x = float(state["x_d"])
                    self.distractor_exit_slope = float(state["slope_d"])
                    self.distractor_exit_speed = float(state["spd_d"])
                    self._update_distractor_release_velocity()
                self._update_release_velocity()
                # Always leave the lip under PhysX — never freeze waiting for the cup.
                self._release_ball()
            else:
                pos_r = self._surface_point(state["x_r"], self._y_from_s(state["s_r"]))
                self._set_kinematic_pose(pos_r, state["s_r"])
                if (
                    self.enable_distractor
                    and self._distractor_phase == "rolling"
                    and self.distractor is not None
                ):
                    pos_d = self._surface_point(state["x_d"], self._y_from_s(state["s_d"]))
                    self._set_kinematic_pose(
                        pos_d, state["s_d"],
                        actor=self.distractor, rigid=self._distractor_comp,
                    )

    def _release_ball(self):
        if self._ball_phase == "released":
            return
        self._ball_phase = "released"
        # Metrics: the ramp exit is a secondary marker (kept for reference).
        if getattr(self, "_metric_release_step", None) is None:
            self._metric_release_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        self._update_release_velocity()
        release_pos = np.asarray(
            getattr(self, "ball_release_pos", self.ball_edge), dtype=float
        )
        self.ball.set_pose(sapien.Pose(release_pos.tolist()))
        release_speed = float(getattr(self, "ball_exit_speed", self.ball_speed))
        if self._ball_comp is not None:
            self._ball_comp.set_kinematic(False)
            self._ball_comp.set_disable_gravity(False)
            self._ball_comp.set_linear_velocity(self.release_vel.tolist())
            self._ball_comp.set_angular_velocity([
                release_speed / max(self.ball_radius, 1e-6), 0.0, 0.0,
            ])
            try:
                self._ball_comp.set_linear_damping(0.08)
                self._ball_comp.set_angular_damping(0.5)
            except Exception:
                pass

        if self.enable_distractor and self.distractor is not None:
            self._distractor_phase = "released"
            self._update_distractor_release_velocity()
            d_pos = np.asarray(
                getattr(self, "distractor_release_pos", self.distractor_edge),
                dtype=float,
            )
            self.distractor.set_pose(sapien.Pose(d_pos.tolist()))
            d_speed = float(getattr(self, "distractor_exit_speed", self.distractor_speed))
            if self._distractor_comp is not None:
                self._distractor_comp.set_kinematic(False)
                self._distractor_comp.set_disable_gravity(False)
                self._distractor_comp.set_linear_velocity(
                    self.distractor_release_vel.tolist()
                )
                self._distractor_comp.set_angular_velocity([
                    d_speed / max(self.ball_radius, 1e-6), 0.0, 0.0,
                ])
                try:
                    self._distractor_comp.set_linear_damping(0.08)
                    self._distractor_comp.set_angular_damping(0.5)
                except Exception:
                    pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_ball()
        self._track_catch_metrics()

    def _dwell(self, steps):
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # ------------------------------------------------------------- policy
    def play_once(self):
        landing, _ = self._predict_landing()
        aim = np.asarray(getattr(self, "catch_aim", landing), dtype=float)
        x_land = float(aim[0])
        y_land = float(aim[1])
        arm_tag = ArmTag("right" if x_land > 0 else "left")

        # Dense RGB while collecting demos so the roll from the top is visible.
        # Keep an already-dense caller value (e.g. save_freq=5/8 for slow rolls).
        old_save_freq = self.save_freq
        if self.save_data and (self.save_freq is None or self.save_freq > 8):
            self.save_freq = 5

        # Ball drops at the top and immediately begins rolling downhill.
        # The robot's job is to fetch the cup and place it under the flight.
        self._start_ball_motion(expert_demo=True)
        # Let the drop + first part of the roll be clearly visible before grasping.
        self._dwell(self.drop_steps + int(0.45 * self.SIM_HZ))

        self.move(self.grasp_actor(self.cup, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        cup_now = np.array(self.cup.get_pose().p)
        target = np.array([x_land, y_land, self.table_top + self.CUP_CENTER_Z])
        d = target - cup_now
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, x=float(d[0]), y=float(d[1]), z=float(d[2]),
            move_axis="world",
        ))
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))

        self._cup_ready = True
        # Ball may already be in flight (or miss and fall off) — wait out the rest.
        remaining = max(0, self.roll_steps - int(getattr(self, "_roll_i", 0)))
        self._dwell(remaining + self.fall_steps + self.settle_steps)
        self.save_freq = old_save_freq

        self.info["info"] = {
            "{A}": "red ball",
            "{B}": "021_cup/base0",
            "{a}": str(arm_tag),
            "{opt}": self._option_label(),
        }
        return self.info

    # ------------------------------------------------------------- success
    def _ball_in_cup(self, ball_actor):
        if ball_actor is None:
            return False
        bp = np.array(ball_actor.get_pose().p)
        cp = np.array(self.cup.get_pose().p)
        offset = float(np.linalg.norm(bp[:2] - cp[:2]))
        return bool(
            offset < self.rim_radius
            and (self.table_top - 0.01) < bp[2] < (self.table_top + 0.12)
        )

    def _catch_state(self):
        bp = np.array(self.ball.get_pose().p)
        cp = np.array(self.cup.get_pose().p)
        offset = float(np.linalg.norm(bp[:2] - cp[:2]))
        in_vessel = bool(
            offset < self.rim_radius
            and (self.table_top - 0.01) < bp[2] < (self.table_top + 0.12)
        )
        distractor_in = bool(
            self.enable_distractor and self._ball_in_cup(self.distractor)
        )
        return offset, in_vessel, distractor_in, bp, cp

    def check_success(self):
        if getattr(self, "_ball_phase", None) != "released":
            return False
        _, in_vessel, distractor_in, _, _ = self._catch_state()
        if distractor_in:
            return False
        return bool(in_vessel)

    def get_obs(self):
        obs = super().get_obs()
        try:
            offset, in_vessel, distractor_in, bp, cp = self._catch_state()
            distractor_position = (
                list(map(float, self.distractor.get_pose().p))
                if self.enable_distractor and self.distractor is not None
                else [0.0, 0.0, 0.0]
            )
            obs["catch"] = {
                "ball_xy": [float(bp[0]), float(bp[1])],
                "cup_xy": [float(cp[0]), float(cp[1])],
                "offset": float(offset),
                "in_vessel": float(in_vessel),
                "predicted_landing": [float(self.landing[0]), float(self.landing[1])],
                "ball_speed": float(self.ball_speed),
                "distractor_speed": float(self.distractor_speed),
                "wall_bounce_enabled": float(self.wall_bounce_enabled),
                "enable_distractor": float(self.enable_distractor),
                "drop_wall_bounces": int(self.drop_wall_bounces),
                "ball_ball_bounces": int(self.ball_ball_bounces),
                "ramp_top_y": float(self.ramp_top_y),
                "ramp_exit_y": float(self.ramp_exit_y),
                "distractor_position": distractor_position,
                "distractor_in_cup": float(distractor_in),
                "option_label": self._option_label(),
                "phase": str(self._ball_phase),
            }
        except Exception:
            pass
        return obs

    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_start_step = None
        self._metric_release_step = None
        self._metric_catch_step = None
        self._metric_catch_offset_norm = None

    def _track_catch_metrics(self):
        """Latch the first frame the ball sits inside the cup, and how centred.

        High-water mark: the ball can bounce back out, so only the first True is
        kept. Runs once per step and stops querying once latched.
        """
        if getattr(self, "_metric_catch_step", None) is not None:
            return
        if getattr(self, "_ball_phase", None) != "released":
            return
        try:
            offset, in_vessel, _, _, _ = self._catch_state()
        except Exception:
            return
        if not in_vessel:
            return
        self._metric_catch_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        try:
            self._metric_catch_offset_norm = float(offset) / max(float(self.rim_radius), 1e-9)
        except Exception:
            self._metric_catch_offset_norm = None

    def _compute_metrics(self):
        """extra1 = drop->catch latency, extra2 = entry offset in cup radii.

        Latency is measured from the drop at the top of the ramp (when the human
        first sees the ball move and must get the cup under the exit), NOT from
        the ramp exit — by then a good run already has the cup in place, so that
        anchor has no spread. ``catch_offset_norm`` is the ball-to-cup horizontal
        distance at the instant the ball first counted as inside, as a fraction
        of the cup rim radius: 0.0 = dead centre, 1.0 = grazing the rim. LOWER IS
        BETTER. None when the ball never entered the cup.
        """
        metrics = {
            "catch_latency_steps": None,
            "catch_latency_s": None,
            "catch_offset_norm": (
                None
                if getattr(self, "_metric_catch_offset_norm", None) is None
                else float(self._metric_catch_offset_norm)
            ),
        }
        start = getattr(self, "_metric_start_step", None)
        caught = getattr(self, "_metric_catch_step", None)
        if start is not None and caught is not None and caught >= start:
            steps = int(caught - start)
            metrics["catch_latency_steps"] = steps
            try:
                metrics["catch_latency_s"] = round(steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics
