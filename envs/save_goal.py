from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class save_goal(Base_Task):
    """Keep a moving ball out of the goal by placing a stylized goalkeeper in time.

    A ball travels across the table toward a goal on one side. Layout can be mirrored
    across the table midline (x → −x): default puts the goal/keeper on +x (right gripper);
    ``mirrored: true`` puts them on −x (left gripper). With ``random_mirror: true`` (default)
    the side is chosen per episode when ``mirrored`` is unset.
    A red deadline line marks the last moment the robot is allowed to reposition the save_goal;
    it sits on the field side of the green placement area with a clear gap (≥1 cm by default).
    A green placement area sits directly in front of the goal. The robot must grasp the
    goalkeeper (yellow jersey / black shorts blocker), place it fully inside that green area
    while the ball is still behind the red line,
    then release it before the ball reaches the goal mouth. The goalkeeper is a solid
    blocker at whatever pose it was left in: a hit from any angle rebounds with a
    mass-aware bounce (ball 100 g, keeper 500 g). A legal save still requires the
    keeper fully inside the green area before the deadline. A soccer-style net bag
    (visual lattice) hangs behind the goal mouth when ``net_enabled`` is true.

    Task options (set in ``task_args.save_goal``; independent toggles):
      - Option 1 — field players (bounce): ``players_enabled``
        Spawn ``players_max`` static field players (same stylized figure as the
        goalkeeper, blue jersey, 30% thinner) before the red line, outside the green
        zone, and no farther from the goal than
        ``player_max_goal_dist_mult`` × green-zone depth (default 2×). The ball aims at
        one player at random, "bounces", then continues toward the goal mouth.
        CLI: ``--task-arg players_enabled=true`` or legacy ``--option 1``.
      - Option 2 — field cover: ``cover_enabled``
        A partial tunnel/cover over the center of the approach field (ball drop → red
        line). Ball enters after a ≥10 cm opening on the drop side, travels under the
        cover, then exits with a ≥10% field-length opening before the red line.
        Tunnel cavity width matches the red deadline line; side walls are solid.
        With Option 1, the cover ends before the field players.
        CLI: ``--task-arg cover_enabled=true`` or legacy ``--option 2``.
      - Mirror — ``mirrored`` / ``random_mirror``
        Flip the whole field to the left (−x) so the expert uses the left gripper.
        CLI: ``--task-arg mirrored=true``.
    """

    BALL_RADIUS_DEFAULT = 0.018
    BALL_MASS = 0.1          # kg (100 g)
    # Heavy enough to plant upright on release / resist tip from the light ball.
    KEEPER_MASS = 0.5        # kg (500 g)
    # Coefficient of restitution for the mass-aware front-face bounce handoff.
    BOUNCE_RESTITUTION = 0.85
    # PhysX ignores restitution below bounce_threshold (default 2 m/s); our shot is slow.
    BOUNCE_THRESHOLD_DEFAULT = 0.01
    # Nominal ball speed; each episode samples ±20% by default (scale 0.8–1.2).
    BALL_SPEED_DEFAULT = 0.04376  # ×0.8 vs 0.05470; episode still samples ±20%
    BALL_SPEED_SCALE_MIN_DEFAULT = 0.8
    BALL_SPEED_SCALE_MAX_DEFAULT = 1.2
    BALL_START_X_DEFAULT = 0.24
    BALL_START_Y_JITTER_DEFAULT = 0.02
    BALL_GOAL_END_X_OFFSET_DEFAULT = 0.08
    BALL_TARGET_Y_MARGIN_DEFAULT = 0.02
    BALL_ANGLE_DEG_MIN_DEFAULT = -10.0
    BALL_ANGLE_DEG_MAX_DEFAULT = 10.0

    GOAL_X_DEFAULT = 0.20
    GOAL_CENTER_Y_DEFAULT = 0.10
    GOAL_CENTER_Y_JITTER_DEFAULT = 0.0
    GOAL_HALF_W_DEFAULT = 0.11
    GOAL_POST_T_DEFAULT = 0.01
    GOAL_POST_H_DEFAULT = 0.12
    GOAL_BAR_T_DEFAULT = 0.01
    GREEN_AREA_X_LEN_DEFAULT = 0.10
    GREEN_AREA_Y_EXTRA_DEFAULT = 0.05
    # Deadline offset from goal; must be ≥ green_area_x_len + red_line_green_gap.
    RED_LINE_X_DEFAULT = 0.11
    RED_LINE_GREEN_GAP_DEFAULT = 0.01  # m; min clear gap between red line and green field edge
    RED_LINE_OUTWARD_MAX_DEFAULT = 0.05  # m; randomize further toward the field from the min gap
    # Layout mirror across y-axis (x → −x): left gripper when mirrored.
    RANDOM_MIRROR_DEFAULT = True
    MIRRORED_DEFAULT = None  # None → sample when random_mirror, else use explicit bool

    KEEPER_X_DEFAULT = 0.16
    KEEPER_SPAWN_X_DEFAULT = 0.10
    KEEPER_SPAWN_Y_DEFAULT = -0.12
    KEEPER_GOAL_CLEARANCE_DEFAULT = 0.05
    KEEPER_POSE_TOL_DEFAULT = 0.03
    # Collision half-height as a multiple of ball_radius (XY still matches the ball).
    # ~1.45 → modestly taller than the old cube so the figure reads as a keeper.
    KEEPER_HALF_Z_MULT_DEFAULT = 1.45
    KEEPER_SHIRT_COLOR = (0.95, 0.82, 0.10)   # yellow jersey
    KEEPER_SHORTS_COLOR = (0.06, 0.06, 0.07)  # black shorts
    KEEPER_SKIN_COLOR = (0.92, 0.72, 0.55)
    KEEPER_BOOT_COLOR = (0.05, 0.05, 0.05)

    BALL_SETTLE_STEPS_DEFAULT = 120

    # Option 1 — field players the ball can bounce off
    PLAYERS_ENABLED_DEFAULT = False  # Option 1 toggle
    PLAYERS_MAX_DEFAULT = 2
    # Same stylized figure as the keeper; XY footprint scaled by this (0.7 = 30% thinner).
    PLAYER_THIN_SCALE_DEFAULT = 0.70
    PLAYER_CORRIDOR_MARGIN_DEFAULT = 0.04
    PLAYER_Y_SPREAD_DEFAULT = 0.20
    # Min clear gap (edge-to-edge) between players; center spacing adds footprint widths.
    PLAYER_SEPARATION_DEFAULT = 0.15
    # Max |goal_x − player_x| as a multiple of green_area_x_len (keep players near the goal).
    PLAYER_MAX_GOAL_DIST_MULT_DEFAULT = 2.0
    # Blend of "face partner" vs "face goal" when orienting players (0=partner only, 1=goal only).
    PLAYER_GOAL_FACE_BIAS_DEFAULT = 0.22
    PLAYER_SHIRT_COLOR = (0.15, 0.40, 0.90)  # blue jersey (keeper is yellow)

    # Option 2 — partial field cover / tunnel over the approach corridor
    COVER_ENABLED_DEFAULT = False  # Option 2 toggle
    COVER_ENTRY_GAP_DEFAULT = 0.10       # m; min open gap on the ball-drop side
    COVER_EXIT_GAP_FRAC_DEFAULT = 0.10   # fraction of field length open before red line
    COVER_HALF_Y_DEFAULT = None          # None → match red-line half-width (goal_half_w + RED_LINE_Y_EXTRA)
    COVER_CLEARANCE_Z_DEFAULT = 0.12     # m; clear height under roof (fits ball + players)
    COVER_WALL_T_DEFAULT = 0.008         # m; side-wall thickness
    COVER_ROOF_T_DEFAULT = 0.006         # m; roof thickness
    COVER_LEN_MIN_DEFAULT = 0.10         # m; min cover length along travel (x)
    # Clear gap between cover exit and the nearest Opt-1 player (when both enabled).
    COVER_PLAYER_CLEARANCE_DEFAULT = 0.02
    COVER_COLOR = (0.42, 0.45, 0.50)
    # Red deadline line extends this far past each goal post (half-width = goal_half_w + extra).
    RED_LINE_Y_EXTRA_DEFAULT = 0.10

    # Soccer-style goal net (visual lattice behind the posts)
    NET_ENABLED_DEFAULT = True
    NET_DEPTH_DEFAULT = 0.09          # m; how far behind the goal mouth the net sits
    NET_CELL_DEFAULT = 0.028          # m; approx mesh spacing
    NET_STRAND_T_DEFAULT = 0.0022     # m; strand thickness
    NET_COLOR = (0.92, 0.93, 0.95)

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("save_goal", {})
        self._loaded = False
        self._ball_motion_active = False
        self._ball_step = 0
        self._ball_blocked = False
        self._ball_live = False
        self._block_was_legal = False
        self._reset_metric_state()
        self._goal_conceded = False
        self._late_failure = False
        self.goalkeeper = None
        self.ball = None
        self._ball_rigid = None
        self.goalkeeper_target_pose = None
        self.ball_start_pose = None
        self.ball_target_pose = None
        self.ball_bounce_pose = None
        self._ball_waypoints = None
        self._ball_seg_cum = None
        self._ball_path_len = 0.0
        self._players = []
        self._bounce_player_idx = -1
        self.players_enabled = False
        self.cover_enabled = False
        self._cover_parts = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_len = 0.0
        self._ball_crossed_goal = False
        self.green_area_x_min = 0.0
        self.green_area_x_max = 0.0
        self.green_area_y_min = 0.0
        self.green_area_y_max = 0.0
        self._keeper_deployed = False
        self._keeper_drop_pose = None
        super()._init_task_env_(**kwags)
        # After settle so check_stable does not see a moving ball. Policy eval
        # never calls play_once, so the shot must start here.
        self._start_shot()

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
        raise ValueError(f"save_goal expected a boolean, got {value!r}")

    def _parse_players_enabled(self, c) -> bool:
        """Option 1 toggle: ``players_enabled`` (preferred) or legacy ``option: 1``."""
        players = c.get("players_enabled", None)
        legacy = c.get("option", None)
        if legacy is not None and players is None:
            # option: 1 enables players; option: 2 is cover-only (handled separately).
            if legacy in (1, "1", "players_enabled", "players", "field_players"):
                players = True
            elif legacy in (2, "2", "cover_enabled", "cover", "field_cover"):
                players = False
            else:
                raise ValueError(
                    "save_goal option must be 1/players_enabled or 2/cover_enabled "
                    "(or set players_enabled / cover_enabled booleans)"
                )
        return self._as_bool(players, self.PLAYERS_ENABLED_DEFAULT)

    def _parse_cover_enabled(self, c) -> bool:
        """Option 2 toggle: ``cover_enabled`` (preferred) or legacy ``option: 2``."""
        cover = c.get("cover_enabled", None)
        legacy = c.get("option", None)
        if legacy is not None and cover is None:
            if legacy in (2, "2", "cover_enabled", "cover", "field_cover"):
                cover = True
            elif legacy in (1, "1", "players_enabled", "players", "field_players"):
                cover = False
            else:
                raise ValueError(
                    "save_goal option must be 1/players_enabled or 2/cover_enabled "
                    "(or set players_enabled / cover_enabled booleans)"
                )
        return self._as_bool(cover, self.COVER_ENABLED_DEFAULT)

    def _parse_mirrored(self, c) -> bool:
        """Layout mirror (x → −x): explicit ``mirrored``, else random when ``random_mirror``."""
        random_mirror = bool(c.get("random_mirror", self.RANDOM_MIRROR_DEFAULT))
        mirror_cfg = c.get("mirrored", self.MIRRORED_DEFAULT)
        if mirror_cfg is None:
            return bool(random_mirror and (np.random.rand() < 0.5))
        return self._as_bool(mirror_cfg, False)

    def _option_label(self) -> str:
        """Caption token for active options (Option 1 / Option 2)."""
        parts = []
        if getattr(self, "players_enabled", False):
            parts.append("option 1")
        if getattr(self, "cover_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def setup_scene(self, **kwargs):
        """Lower PhysX bounce_threshold so slow shot restitution can fire."""
        bounce_th = float(
            self._cfg.get("bounce_threshold", self.BOUNCE_THRESHOLD_DEFAULT)
        )
        orig_create = sapien.Engine.create_scene

        def _create_with_bounce(engine, config=None):
            if config is None:
                config = sapien.SceneConfig()
            config.bounce_threshold = bounce_th
            return orig_create(engine, config)

        sapien.Engine.create_scene = _create_with_bounce
        try:
            super().setup_scene(**kwargs)
        finally:
            sapien.Engine.create_scene = orig_create

    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _set_collision_enabled(self, entity, enabled: bool):
        """Toggle PhysX contacts. Disabled while the ball is kinematic on rails."""
        if entity is None:
            return
        groups = [1, 1, 0, 0] if enabled else [0, 0, 0, 0]
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if not isinstance(comp, sapien.physx.PhysxRigidBaseComponent):
                continue
            for shape in comp.get_collision_shapes():
                shape.set_collision_groups(list(groups))

    def _set_restitution(self, rigid, restitution: float, static_f=0.4, dynamic_f=0.3):
        if rigid is None:
            return
        try:
            mat = sapien.physx.PhysxMaterial(
                static_friction=float(static_f),
                dynamic_friction=float(dynamic_f),
                restitution=float(restitution),
            )
            for shape in rigid.get_collision_shapes():
                shape.set_physical_material(mat)
        except Exception:
            pass

    @staticmethod
    def _elastic_1d(u1: float, u2: float, m1: float, m2: float, e: float):
        """1D collision with restitution e; returns (v1, v2)."""
        m1 = max(float(m1), 1e-6)
        m2 = max(float(m2), 1e-6)
        e = float(np.clip(e, 0.0, 1.0))
        v1 = (u1 * (m1 - e * m2) + u2 * m2 * (1.0 + e)) / (m1 + m2)
        v2 = (u2 * (m2 - e * m1) + u1 * m1 * (1.0 + e)) / (m1 + m2)
        return float(v1), float(v2)

    def _keeper_xy_bounds(self):
        if getattr(self, "goalkeeper", None) is None:
            return None
        pose_m = self.goalkeeper.get_pose().to_transformation_matrix()
        local_corners = np.array([
            [-self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [-self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
        ], dtype=np.float64)
        world_corners = (pose_m @ local_corners.T).T
        return (
            float(np.min(world_corners[:, 0])),
            float(np.max(world_corners[:, 0])),
            float(np.min(world_corners[:, 1])),
            float(np.max(world_corners[:, 1])),
        )

    def _ball_path_y_at_x(self, x: float) -> float:
        """Y of the ball's final approach segment (post-bounce if Opt 1 is on) at table x."""
        if self.ball_bounce_pose is not None:
            start = np.asarray(self.ball_bounce_pose, dtype=np.float64)
        else:
            start = np.asarray(self.ball_start_pose, dtype=np.float64)
        end = np.asarray(self.ball_target_pose, dtype=np.float64)
        start_x = float(start[0])
        end_x = float(end[0])
        if abs(end_x - start_x) < 1e-8:
            return float(start[1])
        t = float((x - start_x) / (end_x - start_x))
        return float(start[1] + t * (end[1] - start[1]))

    def _set_ball_waypoints(self, waypoints):
        pts = [np.asarray(p, dtype=np.float64).copy() for p in waypoints]
        self._ball_waypoints = pts
        cum = [0.0]
        for i in range(len(pts) - 1):
            cum.append(cum[-1] + float(np.linalg.norm(pts[i + 1] - pts[i])))
        self._ball_seg_cum = cum
        self._ball_path_len = float(cum[-1])

    def _ball_pos_at_progress(self, progress: float):
        progress = float(np.clip(progress, 0.0, 1.0))
        if self._ball_waypoints is None or self._ball_path_len < 1e-8:
            return self.ball_start_pose + (self.ball_target_pose - self.ball_start_pose) * progress
        dist = progress * self._ball_path_len
        cum = self._ball_seg_cum
        pts = self._ball_waypoints
        for i in range(len(pts) - 1):
            if dist <= cum[i + 1] + 1e-9:
                seg_len = cum[i + 1] - cum[i]
                t = 0.0 if seg_len < 1e-8 else (dist - cum[i]) / seg_len
                return pts[i] + (pts[i + 1] - pts[i]) * t
        return pts[-1].copy()

    def _soccer_ball_quat_at_progress(self, progress: float):
        """Roll the textured ball along its path (no-slip) matching travel direction.

        Right-hand rule: for velocity ``v`` on the table, ``ω = (ẑ × v) / r``, so a
        ball moving +X rotates about +Y and the top surface moves with the travel.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        r = max(float(getattr(self, "ball_radius", self.BALL_RADIUS_DEFAULT)), 1e-6)
        if self._ball_waypoints is not None and self._ball_path_len >= 1e-8:
            pts = self._ball_waypoints
            cum = self._ball_seg_cum
            path_len = float(self._ball_path_len)
        else:
            p0 = np.asarray(self.ball_start_pose, dtype=np.float64)
            p1 = np.asarray(self.ball_target_pose, dtype=np.float64)
            pts = [p0, p1]
            path_len = float(np.linalg.norm(p1 - p0))
            cum = [0.0, path_len]
        dist_target = progress * path_len
        # Accumulate rotations along each segment (handles Opt-1 bounce kinks).
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        traveled = 0.0
        for i in range(len(pts) - 1):
            seg = float(cum[i + 1] - cum[i])
            if seg < 1e-12:
                continue
            take = float(min(seg, dist_target - traveled))
            if take <= 1e-12:
                if traveled >= dist_target - 1e-12:
                    break
                traveled += seg
                continue
            d = np.asarray(pts[i + 1][:2], dtype=np.float64) - np.asarray(pts[i][:2], dtype=np.float64)
            dn = float(np.linalg.norm(d))
            if dn > 1e-12:
                d /= dn
                # ω axis = ẑ × d̂ = (-dy, dx, 0)
                ax, ay = -float(d[1]), float(d[0])
                an = float(np.hypot(ax, ay))
                if an > 1e-12:
                    ax /= an
                    ay /= an
                    ang = take / r
                    hw = float(np.cos(0.5 * ang))
                    hs = float(np.sin(0.5 * ang))
                    dq_w, dq_x, dq_y, dq_z = hw, ax * hs, ay * hs, 0.0
                    # q = dq * q (apply new roll in world frame)
                    nw = dq_w * qw - dq_x * qx - dq_y * qy - dq_z * qz
                    nx = dq_w * qx + dq_x * qw + dq_y * qz - dq_z * qy
                    ny = dq_w * qy - dq_x * qz + dq_y * qw + dq_z * qx
                    nz = dq_w * qz + dq_x * qy - dq_y * qx + dq_z * qw
                    qw, qx, qy, qz = nw, nx, ny, nz
            traveled += take
            if traveled >= dist_target - 1e-12:
                break
        nrm = float(np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz))
        if nrm < 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        return [qw / nrm, qx / nrm, qy / nrm, qz / nrm]

    def _spawn_field_players(self, start_x: float, start_y: float, goal_end_x: float):
        """Opt 1: place players before the red line, near the goal (≤2× green depth)."""
        self._players = []
        self._bounce_player_idx = -1
        self.ball_bounce_pose = None
        self._players_hit = set()
        if not self.players_enabled:
            return

        # Allowed band along x (distance from goal):
        #   - before the red line and outside the green zone (near-goal bound)
        #   - no farther than player_max_goal_dist_mult × green_area_x_len (far bound)
        green_len = float(self.green_area_x_len)
        red_offset = float(abs(self.goal_x - self.red_line_x))
        min_goal_dist = float(max(green_len, red_offset) + self.player_half_xy)
        max_goal_dist = float(
            max(self.player_max_goal_dist_mult, 1.0) * green_len - self.player_half_xy
        )
        if max_goal_dist <= min_goal_dist + 1e-4:
            # Degenerate config: keep a thin band just outside the near-goal bound.
            max_goal_dist = float(min_goal_dist + max(2.0 * self.player_half_xy, 0.02))

        near_goal_x = float(self.goal_x - self.travel_dir * min_goal_dist)
        far_goal_x = float(self.goal_x - self.travel_dir * max_goal_dist)
        # Also stay clear of the ball drop.
        margin = self.player_corridor_margin + self.player_half_xy
        start_side = float(start_x) + self.travel_dir * margin
        if self.travel_dir > 0.0:
            lo = max(far_goal_x, start_side)
            hi = near_goal_x
        else:
            lo = near_goal_x
            hi = min(far_goal_x, start_side)
        if hi <= lo + 1e-4:
            return

        # Separation is a clear gap; convert to min center-to-center distance.
        min_center_dist = float(self.player_separation + 2.0 * self.player_half_xy)
        n = int(self.players_max)  # always fill the configured count when Opt 1 is on
        cy = float(self.goal_center_y)
        y_lo = float(cy - self.player_y_spread)
        y_hi = float(cy + self.player_y_spread)
        placed = []

        def _rand_x():
            return float(np.random.uniform(lo, hi))

        # Two players: one strictly above the goal centerline, one strictly below; x/y randomized.
        if n == 2 and y_hi > cy and y_lo < cy:
            center_margin = 1e-3
            above_lo = cy + center_margin
            below_hi = cy - center_margin
            for _ in range(96):
                px_a, py_a = _rand_x(), float(np.random.uniform(above_lo, y_hi))
                px_b, py_b = _rand_x(), float(np.random.uniform(y_lo, below_hi))
                if np.hypot(px_a - px_b, py_a - py_b) >= min_center_dist:
                    placed = [(px_a, py_a), (px_b, py_b)]
                    break
            else:
                # Fallback: still split across the centerline with the min gap.
                half_gap = 0.5 * min_center_dist
                placed = [
                    (_rand_x(), float(np.clip(cy + half_gap, above_lo, y_hi))),
                    (_rand_x(), float(np.clip(cy - half_gap, y_lo, below_hi))),
                ]
            # Randomize spawn order so neither side is privileged.
            if np.random.rand() < 0.5:
                placed.reverse()
        else:
            for i in range(n):
                # Alternate above / below the goal centerline when possible.
                if y_hi > cy and y_lo < cy:
                    if i % 2 == 0:
                        y_a, y_b = cy + 1e-3, y_hi
                    else:
                        y_a, y_b = y_lo, cy - 1e-3
                else:
                    y_a, y_b = y_lo, y_hi
                for _ in range(64):
                    px = _rand_x()
                    py = float(np.random.uniform(y_a, y_b))
                    if all(np.hypot(px - qx, py - qy) >= min_center_dist for qx, qy in placed):
                        placed.append((px, py))
                        break
                else:
                    side = 1.0 if (i % 2 == 0) else -1.0
                    py = float(np.clip(
                        cy + side * (0.5 * min_center_dist + 0.02 * (i // 2)),
                        y_lo,
                        y_hi,
                    ))
                    placed.append((_rand_x(), py))

        # Mesh-style feet sit at the actor origin; box figures are centered on pose.
        box_z = float(self.table_top_z + self.player_half_z)
        mid_y = float(np.mean([p[1] for p in placed])) if placed else float(self.goal_center_y)

        for i, (px, py) in enumerate(placed):
            # Face each other (toward partner) and slightly toward the goal.
            # Stylized figure faces local +X (arms along ±Y).
            if len(placed) >= 2:
                others = [p for j, p in enumerate(placed) if j != i]
                partner = min(others, key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)
                toward = np.array([partner[0] - px, partner[1] - py], dtype=np.float64)
            else:
                toward = np.array([0.0, mid_y - py], dtype=np.float64)
            if float(np.linalg.norm(toward)) < 1e-6:
                toward = np.array([0.0, 1.0], dtype=np.float64)
            toward /= float(np.linalg.norm(toward))
            bias = float(np.clip(self.player_goal_face_bias, 0.0, 1.0))
            face = (1.0 - bias) * toward + bias * np.array([self.travel_dir, 0.0], dtype=np.float64)
            if float(np.linalg.norm(face)) < 1e-6:
                face = toward
            face /= float(np.linalg.norm(face))
            # Map local +X → face: R_z(yaw) @ (1,0) = (cos yaw, sin yaw) == face
            yaw = float(np.arctan2(face[1], face[0]))
            qz = np.sin(yaw * 0.5)
            qw = np.cos(yaw * 0.5)
            player = self._build_player_figure(
                sapien.Pose([px, py, box_z], [qw, 0.0, 0.0, qz]),
                hx=float(self.player_half_xy),
                hy=float(self.player_half_xy),
                hz=float(self.player_half_z),
                shirt_color=self.PLAYER_SHIRT_COLOR,
                name=f"field_player_{i}",
                is_static=True,
            )
            self._players.append(player)
            self.add_prohibit_area(player, padding=0.02)

        self._bounce_player_idx = int(np.random.randint(0, len(self._players)))
        bx, by = placed[self._bounce_player_idx]
        # Hit point on the ball-facing side of the chosen player.
        to_player = np.array([bx - start_x, by - start_y], dtype=np.float64)
        dist_xy = float(np.linalg.norm(to_player))
        if dist_xy < 1e-6:
            to_player = np.array([self.travel_dir, 0.0], dtype=np.float64)
            dist_xy = 1.0
        inward = to_player / dist_xy
        hit_xy = np.array([bx, by], dtype=np.float64) - inward * (self.player_half_xy + self.ball_radius + 0.002)
        self.ball_bounce_pose = np.array(
            [hit_xy[0], hit_xy[1], self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        # Prefer a post-bounce end_y that still enters the goal from the bounce point.
        for _ in range(64):
            end_y = float(np.random.uniform(
                self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
                self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
            ))
            # Reject paths that clip the distractor players.
            ok = True
            for j, (px, py) in enumerate(placed):
                if j == self._bounce_player_idx:
                    continue
                if self._segment_clear_of_disk(
                    hit_xy, np.array([goal_end_x, end_y]), np.array([px, py]),
                    # Cover the square OBB corners, not just the inscribed disk.
                    self.player_half_xy * np.sqrt(2.0) + self.ball_radius + 0.01,
                ):
                    continue
                ok = False
                break
            if ok:
                self._bounce_end_y = end_y
                return
        self._bounce_end_y = float(np.clip(
            by,
            self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
            self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
        ))

    @staticmethod
    def _segment_clear_of_disk(a_xy, b_xy, c_xy, radius: float) -> bool:
        a = np.asarray(a_xy, dtype=np.float64)
        b = np.asarray(b_xy, dtype=np.float64)
        c = np.asarray(c_xy, dtype=np.float64)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(c - a)) >= radius
        t = float(np.clip(np.dot(c - a, ab) / ab2, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(c - closest)) >= radius

    def _spawn_field_cover(self, start_x: float):
        """Option 2: partial tunnel over the mid approach field (ball drop → red line).

        Leaves ≥ ``cover_entry_gap`` open on the ball-drop side and ≥
        ``cover_exit_gap_frac`` of field length open before the red line.
        Cover length is sampled uniformly in ``[cover_len_min, max_spannable]``
        (default min 10 cm; max = full mid region after those gaps).
        With Option 1, the cover also ends before the nearest field player.
        """
        self._cover_parts = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_len = 0.0
        self.cover_y_min = None
        self.cover_y_max = None
        self._cover_wall_boxes = []
        if not self.cover_enabled:
            return

        field_len = float(abs(self.red_line_x - start_x))
        if field_len < 1e-4:
            return
        entry_gap = float(max(self.cover_entry_gap, 0.0))
        exit_gap = float(max(self.cover_exit_gap_frac, 0.0) * field_len)
        # Region the cover is allowed to occupy (already excludes the required openings).
        region_start = float(start_x + self.travel_dir * entry_gap)
        region_end = float(self.red_line_x - self.travel_dir * exit_gap)

        # Option 1+2: end the cover before the nearest player along the ball path.
        player_xs = []
        for p in getattr(self, "_players", []) or []:
            try:
                player_xs.append(float(p.get_pose().p[0]))
            except Exception:
                continue
        if player_xs:
            clearance = float(
                self.cover_player_clearance
                + getattr(self, "player_half_xy", self.ball_radius * self.PLAYER_THIN_SCALE_DEFAULT)
            )
            if self.travel_dir > 0.0:
                first_player_x = float(min(player_xs))
                player_end = first_player_x - clearance
                region_end = float(min(region_end, player_end))
            else:
                first_player_x = float(max(player_xs))
                player_end = first_player_x + clearance
                region_end = float(max(region_end, player_end))

        span = float(self.travel_dir * (region_end - region_start))
        len_min = float(max(self.cover_len_min, 0.0))
        if span < len_min - 1e-9:
            # Not enough room for the min cover length after mandated gaps / players.
            return

        # Random length in [10 cm, max spannable]; then randomize placement in the mid region.
        cover_len = float(np.random.uniform(len_min, span))
        slack = float(span - cover_len)
        offset = float(np.random.uniform(0.0, slack)) if slack > 1e-9 else 0.0
        cover_a = float(region_start + self.travel_dir * offset)
        cover_b = float(cover_a + self.travel_dir * cover_len)
        x_min = float(min(cover_a, cover_b))
        x_max = float(max(cover_a, cover_b))
        self.cover_x_min = x_min
        self.cover_x_max = x_max
        self.cover_len = cover_len

        cx = 0.5 * (x_min + x_max)
        half_x = 0.5 * (x_max - x_min)
        # Cavity half-width matches the red deadline line (same y extent).
        half_y = float(self.cover_half_y)
        wall_t = float(self.cover_wall_t)
        roof_t = float(self.cover_roof_t)
        clear_z = float(self.cover_clearance_z)
        cy = float(self.goal_center_y)
        color = self.COVER_COLOR
        self.cover_y_min = float(cy - half_y)
        self.cover_y_max = float(cy + half_y)
        self._cover_wall_boxes = []

        # Side walls (leave cavity open along travel for the ball).
        wall_z = self.table_top_z + 0.5 * clear_z
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            wy = cy + sign * (half_y + 0.5 * wall_t)
            half_size = [half_x, 0.5 * wall_t, 0.5 * clear_z]
            part = create_box(
                self,
                pose=sapien.Pose([cx, wy, wall_z], [1, 0, 0, 0]),
                half_size=half_size,
                color=color,
                is_static=True,
                name=f"field_cover_wall_{tag}",
            )
            self._cover_parts.append(part)
            self._cover_wall_boxes.append(
                {
                    "center": np.array([cx, wy, wall_z], dtype=np.float64),
                    "half": np.array(half_size, dtype=np.float64),
                    "tag": tag,
                }
            )

        # Roof slab.
        roof_z = self.table_top_z + clear_z + 0.5 * roof_t
        roof = create_box(
            self,
            pose=sapien.Pose([cx, cy, roof_z], [1, 0, 0, 0]),
            half_size=[half_x, half_y + wall_t, 0.5 * roof_t],
            color=color,
            is_static=True,
            name="field_cover_roof",
        )
        self._cover_parts.append(roof)
        # Cover is a field obstacle for the ball path only; do not mark prohibit
        # areas (that blocks expert arm plans across the corridor).

    def _spawn_goal_net(self):
        """Visual soccer-style net bag behind the goal mouth (back + sides + top)."""
        self._net_parts = []
        if not getattr(self, "net_enabled", True):
            return

        depth = float(max(self.net_depth, 0.02))
        cell = float(max(self.net_cell, 0.015))
        half_t = 0.5 * float(max(self.net_strand_t, 0.001))
        cy = float(self.goal_center_y)
        yw = float(self.goal_half_w)
        h = float(self.goal_post_h)
        z0 = float(self.table_top_z)
        gx = float(self.goal_x)
        d = float(self.travel_dir)
        color = self.NET_COLOR
        back_x = float(gx + d * depth)
        mid_x = float(gx + d * (0.5 * depth))
        half_depth = 0.5 * depth

        n_y = max(3, int(round((2.0 * yw) / cell)) + 1)
        n_z = max(3, int(round(h / cell)) + 1)
        n_x = max(3, int(round(depth / cell)) + 1)
        ys = np.linspace(cy - yw, cy + yw, n_y)
        zs = np.linspace(z0 + half_t, z0 + h - half_t, n_z)
        xs = np.linspace(gx, back_x, n_x)

        def _strand(pose, half_size, name):
            part = create_visual_box(
                self,
                pose=pose,
                half_size=half_size,
                color=color,
                name=name,
            )
            self._net_parts.append(part)

        # Back panel (plane behind the mouth).
        for i, y in enumerate(ys):
            _strand(
                sapien.Pose([back_x, float(y), z0 + 0.5 * h], [1, 0, 0, 0]),
                [half_t, half_t, 0.5 * h],
                f"goal_net_back_v_{i}",
            )
        for i, z in enumerate(zs):
            _strand(
                sapien.Pose([back_x, cy, float(z)], [1, 0, 0, 0]),
                [half_t, yw, half_t],
                f"goal_net_back_h_{i}",
            )

        # Side panels (connect posts to the back corners).
        for side, y_side in (("neg", cy - yw), ("pos", cy + yw)):
            for i, x in enumerate(xs):
                _strand(
                    sapien.Pose([float(x), float(y_side), z0 + 0.5 * h], [1, 0, 0, 0]),
                    [half_t, half_t, 0.5 * h],
                    f"goal_net_side_{side}_v_{i}",
                )
            for i, z in enumerate(zs):
                _strand(
                    sapien.Pose([mid_x, float(y_side), float(z)], [1, 0, 0, 0]),
                    [half_depth, half_t, half_t],
                    f"goal_net_side_{side}_h_{i}",
                )

        # Top panel (under / along the crossbar plane, stretching back).
        top_z = float(z0 + h - half_t)
        for i, x in enumerate(xs):
            _strand(
                sapien.Pose([float(x), cy, top_z], [1, 0, 0, 0]),
                [half_t, yw, half_t],
                f"goal_net_top_y_{i}",
            )
        for i, y in enumerate(ys):
            _strand(
                sapien.Pose([mid_x, float(y), top_z], [1, 0, 0, 0]),
                [half_depth, half_t, half_t],
                f"goal_net_top_x_{i}",
            )

    def _build_player_figure(
        self,
        pose: sapien.Pose,
        *,
        hx: float,
        hy: float,
        hz: float,
        shirt_color,
        name: str = "player_figure",
        is_static: bool = False,
        mass: float = 0.0,
    ) -> Actor:
        """Stylized soccer figure: box collision + boots / shorts / shirt / head visuals."""
        from .utils.create_actor import preprocess

        scene, pose = preprocess(self, pose)
        hx = float(hx)
        hy = float(hy)
        hz = float(hz)
        H = 2.0 * hz

        # Vertical stack (bottom → top), fractions of total height.
        boot_h = 0.10 * H
        shorts_h = 0.30 * H
        shirt_h = 0.38 * H
        head_h = H - boot_h - shorts_h - shirt_h

        def _mat(rgb):
            return sapien.render.RenderMaterial(base_color=[float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0])

        def _add_box(builder, z0, z1, half_xy, color, y_off=0.0, x_off=0.0):
            half_z = 0.5 * (z1 - z0)
            cz = 0.5 * (z0 + z1)
            builder.add_box_visual(
                pose=sapien.Pose([x_off, y_off, cz]),
                half_size=[half_xy[0], half_xy[1], half_z],
                material=_mat(color),
            )

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static" if is_static else "dynamic")
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=scene.default_physical_material,
        )

        z = -hz
        # Boots
        _add_box(builder, z, z + boot_h, [hx * 0.95, hy * 0.95], self.KEEPER_BOOT_COLOR)
        z += boot_h
        # Black shorts
        _add_box(builder, z, z + shorts_h, [hx * 0.98, hy * 0.98], self.KEEPER_SHORTS_COLOR)
        z += shorts_h
        # Jersey / torso (+ arms)
        shirt_z0, shirt_z1 = z, z + shirt_h
        _add_box(builder, shirt_z0, shirt_z1, [hx * 0.92, hy * 0.72], shirt_color)
        # Arms stay inside the collision footprint (±Y) so grasp/AABB stay consistent.
        arm_half_y = hy * 0.22
        arm_out = hy - arm_half_y
        _add_box(
            builder, shirt_z0 + 0.05 * shirt_h, shirt_z1 - 0.08 * shirt_h,
            [hx * 0.50, arm_half_y], shirt_color, y_off=arm_out,
        )
        _add_box(
            builder, shirt_z0 + 0.05 * shirt_h, shirt_z1 - 0.08 * shirt_h,
            [hx * 0.50, arm_half_y], shirt_color, y_off=-arm_out,
        )
        z = shirt_z1
        # Head (box reads better than a sphere at this tiny scale)
        head_half = min(hx, hy, 0.48 * head_h) * 0.85
        _add_box(
            builder, z + 0.05 * head_h, z + head_h - 0.02 * head_h,
            [head_half, head_half], self.KEEPER_SKIN_COLOR,
        )

        builder.set_initial_pose(pose)
        if is_static:
            entity = builder.build_static(name=name)
        else:
            entity = builder.build(name=name)
        # Match create_box contact groups so ball↔figure PhysX contacts work after unlock.
        for comp in entity.get_components():
            if not isinstance(comp, sapien.physx.PhysxRigidBaseComponent):
                continue
            for shape in comp.get_collision_shapes():
                shape.set_collision_groups([1, 1, 0, 0])
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent) and mass > 0.0:
                comp.set_mass(float(mass))

        # Same contact/functional frames as create_box (scale = half_size).
        data = {
            "center": [0, 0, 0],
            "extents": [hx, hy, hz],
            "scale": [hx, hy, hz],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the figure."],
        }
        return Actor(entity, data, mass=float(mass) if mass > 0.0 else 0.01)

    def _build_goalkeeper(self, pose: sapien.Pose) -> Actor:
        """Blocker with box collision + stylized yellow-shirt / black-shorts visuals."""
        return self._build_player_figure(
            pose,
            hx=float(self.keeper_half_x),
            hy=float(self.keeper_half_y),
            hz=float(self.keeper_half_z),
            shirt_color=self.KEEPER_SHIRT_COLOR,
            name="goalkeeper",
            is_static=False,
            mass=float(getattr(self, "keeper_mass", self.KEEPER_MASS)),
        )

    def _keeper_in_zone(self):
        if getattr(self, "goalkeeper", None) is None:
            return False
        bounds = self._keeper_xy_bounds()
        if bounds is None:
            return False
        x_min, x_max, y_min, y_max = bounds
        if not self._keeper_seated_on_table():
            return False
        return bool(
            x_min >= (self.green_area_x_min - 1e-4)
            and x_max <= (self.green_area_x_max + 1e-4)
            and y_min >= (self.green_area_y_min - 1e-4)
            and y_max <= (self.green_area_y_max + 1e-4)
        )

    def _keeper_seated_on_table(self):
        """True when the keeper rests on the table (can physically block), even outside green."""
        if getattr(self, "goalkeeper", None) is None:
            return False
        # Use the lowest OBB corner so tilted poses still count as seated.
        pose_m = self.goalkeeper.get_pose().to_transformation_matrix()
        hx = float(self.keeper_half_x)
        hy = float(self.keeper_half_y)
        hz = float(self.keeper_half_z)
        local = np.array(
            [
                [-hx, -hy, -hz, 1.0],
                [-hx, -hy,  hz, 1.0],
                [-hx,  hy, -hz, 1.0],
                [-hx,  hy,  hz, 1.0],
                [ hx, -hy, -hz, 1.0],
                [ hx, -hy,  hz, 1.0],
                [ hx,  hy, -hz, 1.0],
                [ hx,  hy,  hz, 1.0],
            ],
            dtype=np.float64,
        )
        world_z = (pose_m @ local.T).T[:, 2]
        z_min = float(np.min(world_z))
        return abs(z_min - float(self.table_top_z)) <= max(0.03, 0.4 * hz)

    def _segment_hit_obb(self, pose_m, half_xyz, prev_p, next_p, inflate: float):
        """Segment vs oriented box. Returns ``(t, hit_world, n_world)`` or ``None``."""
        R = pose_m[:3, :3]
        c = pose_m[:3, 3]
        half = np.asarray(half_xyz, dtype=np.float64) + float(inflate)
        p0w = np.asarray(prev_p[:3], dtype=np.float64)
        p1w = np.asarray(next_p[:3], dtype=np.float64)
        p0 = R.T @ (p0w - c)
        p1 = R.T @ (p1w - c)
        d = p1 - p0

        if np.all(np.abs(p0) <= half + 1e-9):
            gaps = half - np.abs(p0)
            axis = int(np.argmin(gaps))
            sign = 1.0 if float(p0[axis]) >= 0.0 else -1.0
            hit_local = p0.copy()
            hit_local[axis] = sign * float(half[axis])
            n_local = np.zeros(3, dtype=np.float64)
            n_local[axis] = sign
            hit_world = c + R @ hit_local
            n_world = R @ n_local
            n_n = float(np.linalg.norm(n_world))
            if n_n > 1e-9:
                n_world = n_world / n_n
            return 0.0, hit_world, n_world

        t_enter = 0.0
        t_exit = 1.0
        hit_axis = 0
        hit_sign = 1.0
        for axis in range(3):
            lo = -float(half[axis])
            hi = float(half[axis])
            if abs(float(d[axis])) < 1e-12:
                if float(p0[axis]) < lo or float(p0[axis]) > hi:
                    return None
                continue
            inv = 1.0 / float(d[axis])
            t1 = (lo - float(p0[axis])) * inv
            t2 = (hi - float(p0[axis])) * inv
            t_near, t_far = (t1, t2) if t1 <= t2 else (t2, t1)
            if t_near > t_enter:
                t_enter = float(t_near)
                hit_axis = axis
                hit_sign = -1.0 if t1 <= t2 else 1.0
            t_exit = float(min(t_exit, t_far))
            if t_enter > t_exit + 1e-12:
                return None

        if t_enter < 0.0 or t_enter > 1.0:
            return None
        hit_local = p0 + t_enter * d
        n_local = np.zeros(3, dtype=np.float64)
        n_local[hit_axis] = hit_sign
        hit_world = c + R @ hit_local
        n_world = R @ n_local
        n_n = float(np.linalg.norm(n_world))
        if n_n > 1e-9:
            n_world = n_world / n_n
        return float(t_enter), hit_world, n_world

    def _segment_hit_keeper_obb(self, prev_p, next_p, inflate: float):
        """Segment vs oriented keeper box (current pose). Returns (t, hit_world, n_world) or None."""
        if getattr(self, "goalkeeper", None) is None:
            return None
        pose_m = self.goalkeeper.get_pose().to_transformation_matrix()
        half = np.array(
            [float(self.keeper_half_x), float(self.keeper_half_y), float(self.keeper_half_z)],
            dtype=np.float64,
        )
        return self._segment_hit_obb(pose_m, half, prev_p, next_p, inflate=inflate)

    def _try_field_player_solid_bounce(self, prev_p, next_p):
        """Solid OBB bounce off Opt-1 field players (no tunneling through them)."""
        players = getattr(self, "_players", None) or []
        if not players:
            return False
        inflate = float(self.ball_radius + 0.001)
        half = np.array(
            [
                float(self.player_half_xy),
                float(self.player_half_xy),
                float(self.player_half_z),
            ],
            dtype=np.float64,
        )
        best = None  # (t, hit_p, n_xy, idx)
        hit_set = getattr(self, "_players_hit", None)
        if hit_set is None:
            hit_set = set()
            self._players_hit = hit_set
        for i, player in enumerate(players):
            if i in hit_set:
                continue
            try:
                pose_m = player.get_pose().to_transformation_matrix()
            except Exception:
                continue
            hit = self._segment_hit_obb(pose_m, half, prev_p, next_p, inflate=inflate)
            if hit is None:
                continue
            t, hit_p, n_world = hit
            n_xy = np.array([float(n_world[0]), float(n_world[1])], dtype=np.float64)
            n_n = float(np.linalg.norm(n_xy))
            if n_n < 1e-6:
                # Mostly vertical contact — push back along approach.
                n_xy = np.array(
                    [-float(self.ball_dir[0]), -float(self.ball_dir[1])],
                    dtype=np.float64,
                )
                n_n = float(np.linalg.norm(n_xy))
                if n_n < 1e-6:
                    continue
            n_xy /= n_n
            if best is None or t < best[0]:
                best = (t, hit_p, n_xy, i)
        if best is None:
            return False
        _t, hit_p, n_xy, idx = best
        self._apply_field_player_bounce(hit_p, n_xy, idx)
        return True

    def _apply_field_player_bounce(self, hit_p, normal_xy, player_idx: int):
        """Reflect the kinematic shot off a static field player and continue toward goal."""
        e = float(self._cfg.get("bounce_restitution", self.BOUNCE_RESTITUTION))
        n = np.asarray(normal_xy[:2], dtype=np.float64)
        n_n = float(np.linalg.norm(n))
        if n_n < 1e-9:
            n = np.array([-float(self.travel_dir), 0.0], dtype=np.float64)
            n_n = 1.0
        n = n / n_n

        v = np.array(
            [
                float(self.ball_dir[0]) * float(self.ball_speed),
                float(self.ball_dir[1]) * float(self.ball_speed),
            ],
            dtype=np.float64,
        )
        vn = float(np.dot(v, n))
        if vn < 0.0:
            # Static infinite-mass wall: reflect with restitution.
            v = v - (1.0 + e) * vn * n
        else:
            v = v + max(0.02, e * float(self.ball_speed)) * n

        sep = 0.002
        p = np.asarray(hit_p, dtype=np.float64).copy()
        p[0] = float(p[0] + sep * n[0])
        p[1] = float(p[1] + sep * n[1])
        p[2] = float(self.table_top_z + self.ball_radius)

        # Prefer continuing into the goal mouth after the bounce (Opt-1 behavior).
        goal_end_x = float(self.ball_target_pose[0]) if self.ball_target_pose is not None else float(
            self.goal_x + self.travel_dir * abs(self.ball_goal_end_x_offset)
        )
        end_y = float(getattr(self, "_bounce_end_y", self.goal_center_y))
        if self.ball_target_pose is not None:
            end_y = float(self.ball_target_pose[1])
        # If the reflected velocity already aims near the mouth, follow it; else retarget.
        speed = float(np.linalg.norm(v))
        if speed > 1e-6:
            v_hat = v / speed
            # Ray from hit toward goal_end_x along reflected heading.
            if abs(float(v_hat[0])) > 1e-6 and (float(v_hat[0]) * float(self.travel_dir) > 0.0):
                t_x = (goal_end_x - float(p[0])) / float(v_hat[0])
                if t_x > 0.0:
                    pred_y = float(p[1] + t_x * v_hat[1])
                    y_lo = self.goal_center_y - self.goal_half_w + self.ball_target_y_margin
                    y_hi = self.goal_center_y + self.goal_half_w - self.ball_target_y_margin
                    if y_lo <= pred_y <= y_hi:
                        end_y = pred_y
                    else:
                        end_y = float(np.clip(pred_y, y_lo, y_hi))

        target = np.array(
            [goal_end_x, end_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        # Ensure post-bounce segment clears remaining players.
        placed = []
        for j, pl in enumerate(getattr(self, "_players", []) or []):
            if j == player_idx or j in getattr(self, "_players_hit", set()):
                continue
            try:
                pp = pl.get_pose().p
                placed.append((j, float(pp[0]), float(pp[1])))
            except Exception:
                pass
        clear_r = float(self.player_half_xy * np.sqrt(2.0) + self.ball_radius + 0.01)
        y_lo = self.goal_center_y - self.goal_half_w + self.ball_target_y_margin
        y_hi = self.goal_center_y + self.goal_half_w - self.ball_target_y_margin
        chosen_y = end_y
        for cand in [end_y] + [
            float(np.random.uniform(y_lo, y_hi)) for _ in range(24)
        ]:
            ok = True
            for _j, px, py in placed:
                if not self._segment_clear_of_disk(
                    p[:2], np.array([goal_end_x, cand]), np.array([px, py]), clear_r
                ):
                    ok = False
                    break
            if ok:
                chosen_y = float(np.clip(cand, y_lo, y_hi))
                break
        target[1] = chosen_y

        self.ball_start_pose = p.copy()
        self.ball_target_pose = target
        self.ball_bounce_pose = None
        self._bounce_end_y = float(chosen_y)
        self._set_ball_waypoints([p, target])
        ball_vec = target - p
        self.ball_dir = ball_vec / max(float(np.linalg.norm(ball_vec)), 1e-8)
        dt = float(self.scene.get_timestep())
        self.ball_total_steps = max(
            1, int(np.ceil(float(self._ball_path_len) / max(self.ball_speed * dt, 1e-8)))
        )
        self._ball_step = 0
        self._players_hit.add(int(player_idx))

        self.ball.set_pose(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        try:
            self._ball_rigid.set_kinematic_target(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        except Exception:
            pass

    def _try_cover_wall_solid_bounce(self, prev_p, next_p):
        """Solid bounce off Opt-2 tunnel side walls (no tunneling through them)."""
        walls = getattr(self, "_cover_wall_boxes", None) or []
        if not walls:
            return False
        inflate = float(self.ball_radius + 0.001)
        best = None  # (t, hit_p, n_xy)
        for wall in walls:
            pose_m = np.eye(4, dtype=np.float64)
            pose_m[:3, 3] = wall["center"]
            hit = self._segment_hit_obb(pose_m, wall["half"], prev_p, next_p, inflate=inflate)
            if hit is None:
                continue
            t, hit_p, n_world = hit
            n_xy = np.array([float(n_world[0]), float(n_world[1])], dtype=np.float64)
            n_n = float(np.linalg.norm(n_xy))
            if n_n < 1e-6:
                continue
            n_xy /= n_n
            # Only bounce when approaching the wall (inward into the solid).
            approach = np.array(
                [float(next_p[0] - prev_p[0]), float(next_p[1] - prev_p[1])],
                dtype=np.float64,
            )
            if float(np.dot(approach, n_xy)) >= 0.0:
                continue
            if best is None or t < best[0]:
                best = (t, hit_p, n_xy)
        if best is None:
            return False
        _t, hit_p, n_xy = best
        self._apply_cover_wall_bounce(hit_p, n_xy)
        return True

    def _apply_cover_wall_bounce(self, hit_p, normal_xy):
        """Reflect the kinematic shot off a tunnel wall and continue toward the goal."""
        e = float(self._cfg.get("bounce_restitution", self.BOUNCE_RESTITUTION))
        n = np.asarray(normal_xy[:2], dtype=np.float64)
        n_n = float(np.linalg.norm(n))
        if n_n < 1e-9:
            return
        n = n / n_n

        v = np.array(
            [
                float(self.ball_dir[0]) * float(self.ball_speed),
                float(self.ball_dir[1]) * float(self.ball_speed),
            ],
            dtype=np.float64,
        )
        vn = float(np.dot(v, n))
        if vn < 0.0:
            v = v - (1.0 + e) * vn * n
        else:
            v = v + max(0.02, e * float(self.ball_speed)) * n

        sep = 0.002
        p = np.asarray(hit_p, dtype=np.float64).copy()
        p[0] = float(p[0] + sep * n[0])
        p[1] = float(p[1] + sep * n[1])
        p[2] = float(self.table_top_z + self.ball_radius)

        # Keep the ball inside the tunnel cavity after a side bounce.
        if getattr(self, "cover_y_min", None) is not None and getattr(self, "cover_y_max", None) is not None:
            margin = float(self.ball_radius + 0.001)
            p[1] = float(np.clip(p[1], self.cover_y_min + margin, self.cover_y_max - margin))

        goal_end_x = float(self.ball_target_pose[0]) if self.ball_target_pose is not None else float(
            self.goal_x + self.travel_dir * abs(self.ball_goal_end_x_offset)
        )
        y_lo = self.goal_center_y - self.goal_half_w + self.ball_target_y_margin
        y_hi = self.goal_center_y + self.goal_half_w - self.ball_target_y_margin
        end_y = float(self.ball_target_pose[1]) if self.ball_target_pose is not None else float(self.goal_center_y)

        speed = float(np.linalg.norm(v))
        if speed > 1e-6 and abs(float(v[0])) > 1e-6 and (float(v[0]) * float(self.travel_dir) > 0.0):
            v_hat = v / speed
            t_x = (goal_end_x - float(p[0])) / float(v_hat[0])
            if t_x > 0.0:
                pred_y = float(p[1] + t_x * v_hat[1])
                end_y = float(np.clip(pred_y, y_lo, y_hi))
        else:
            end_y = float(np.clip(p[1], y_lo, y_hi))

        # Also keep the post-bounce aim inside the cavity while still under the cover.
        if (
            getattr(self, "cover_x_min", None) is not None
            and getattr(self, "cover_y_min", None) is not None
            and self.cover_x_min <= goal_end_x <= self.cover_x_max
        ):
            end_y = float(np.clip(end_y, self.cover_y_min + self.ball_radius, self.cover_y_max - self.ball_radius))

        target = np.array(
            [goal_end_x, end_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        self.ball_start_pose = p.copy()
        self.ball_target_pose = target
        self.ball_bounce_pose = None
        self._set_ball_waypoints([p, target])
        ball_vec = target - p
        self.ball_dir = ball_vec / max(float(np.linalg.norm(ball_vec)), 1e-8)
        dt = float(self.scene.get_timestep())
        self.ball_total_steps = max(
            1, int(np.ceil(float(self._ball_path_len) / max(self.ball_speed * dt, 1e-8)))
        )
        self._ball_step = 0
        self.ball.set_pose(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        try:
            self._ball_rigid.set_kinematic_target(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        except Exception:
            pass

    def _try_keeper_solid_bounce(self, prev_p, next_p):
        """Bounce the kinematic ball off the solid keeper OBB (any face / pose)."""
        if not self._keeper_seated_on_table():
            return False

        inflate = float(self.ball_radius + 0.001)
        hit = self._segment_hit_keeper_obb(prev_p, next_p, inflate=inflate)
        if hit is None:
            return False
        _t, hit_p, n_world = hit
        hit_p = np.asarray(hit_p, dtype=np.float64).copy()
        hit_p[2] = float(self.table_top_z + self.ball_radius)
        # Keep the contact normal in the table plane for the 2D shot.
        n_xy = np.array([float(n_world[0]), float(n_world[1])], dtype=np.float64)
        n_n = float(np.linalg.norm(n_xy))
        if n_n < 1e-6:
            # Hit mostly from above/below — push back along approach.
            n_xy = np.array(
                [-float(self.ball_dir[0]), -float(self.ball_dir[1])],
                dtype=np.float64,
            )
            n_n = float(np.linalg.norm(n_xy))
            if n_n < 1e-6:
                n_xy = np.array([-float(self.travel_dir), 0.0], dtype=np.float64)
                n_n = 1.0
        n_xy = n_xy / n_n
        self._begin_solid_bounce(hit_p, n_xy)
        return True

    def _begin_solid_bounce(self, hit_p, normal_xy):
        """Mass-aware rebound off the keeper along the contact normal (any angle)."""
        self._block_was_legal = bool(self._keeper_in_zone())
        if not self._block_was_legal:
            self._late_failure = True

        m1 = float(self.BALL_MASS)
        m2 = float(getattr(self, "keeper_mass", self.KEEPER_MASS))
        e = float(self._cfg.get("bounce_restitution", self.BOUNCE_RESTITUTION))

        n = np.asarray(normal_xy[:2], dtype=np.float64)
        n_n = float(np.linalg.norm(n))
        if n_n < 1e-9:
            n = np.array([-float(self.travel_dir), 0.0], dtype=np.float64)
            n_n = 1.0
        n = n / n_n

        # Incoming table-plane velocity along the scripted path.
        v1 = np.array(
            [
                float(self.ball_dir[0]) * float(self.ball_speed),
                float(self.ball_dir[1]) * float(self.ball_speed),
            ],
            dtype=np.float64,
        )
        # Keeper assumed at rest before impact.
        u1n = float(np.dot(v1, n))
        u2n = 0.0
        # If somehow separating already, push out gently.
        if u1n >= 0.0:
            u1n = -max(0.02, float(self.ball_speed) * 0.5)
        v1n, v2n = self._elastic_1d(u1n, u2n, m1, m2, e)
        # Tangential component unchanged (frictionless face).
        v1_t = v1 - u1n * n
        v1_after = v1_t + v1n * n
        v2_after = v2n * n

        sep = 0.0015
        p = np.asarray(hit_p, dtype=np.float64).copy()
        p[0] = float(p[0] + sep * n[0])
        p[1] = float(p[1] + sep * n[1])
        p[2] = float(self.table_top_z + self.ball_radius)
        pose = sapien.Pose(p.tolist(), [1, 0, 0, 0])
        self.ball.set_pose(pose)

        self._set_collision_enabled(self.ball, True)
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_kinematic(False)
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_linear_velocity(
                    [float(v1_after[0]), float(v1_after[1]), 0.0]
                )
                self._ball_rigid.set_angular_velocity(np.zeros(3))
                self._ball_rigid.wake_up()
            except Exception:
                pass

        # Unlock keeper with impulse along the contact normal; keep current pose.
        if getattr(self, "goalkeeper", None) is not None:
            kp = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64).copy()
            kq = list(self.goalkeeper.get_pose().q)
            # Pin feet to the table without resetting orientation.
            z_lift = float(self.table_top_z + self.keeper_half_z) - float(kp[2])
            if abs(z_lift) > 0.02:
                kp[2] = float(self.table_top_z + self.keeper_half_z)
                try:
                    self.goalkeeper.set_pose(sapien.Pose(kp.tolist(), kq))
                except Exception:
                    try:
                        self.goalkeeper.actor.set_pose(sapien.Pose(kp.tolist(), kq))
                    except Exception:
                        pass
        keeper_rigid = self._get_rigid(self.goalkeeper)
        if keeper_rigid is not None:
            try:
                keeper_rigid.set_mass(float(m2))
                keeper_rigid.set_kinematic(False)
                self._set_collision_enabled(self.goalkeeper, True)
                self._set_restitution(
                    keeper_rigid, self.BOUNCE_RESTITUTION, static_f=0.9, dynamic_f=0.7
                )
                hx, hy, hz = (
                    float(self.keeper_half_x),
                    float(self.keeper_half_y),
                    float(self.keeper_half_z),
                )
                ix = (1.0 / 12.0) * m2 * ((2 * hy) ** 2 + (2 * hz) ** 2)
                iy = (1.0 / 12.0) * m2 * ((2 * hx) ** 2 + (2 * hz) ** 2)
                iz = (1.0 / 12.0) * m2 * ((2 * hx) ** 2 + (2 * hy) ** 2)
                try:
                    keeper_rigid.set_inertia([ix, iy, iz])
                except Exception:
                    pass
                keeper_rigid.set_linear_damping(3.0)
                keeper_rigid.set_angular_damping(12.0)
                kv = np.array([float(v2_after[0]), float(v2_after[1]), 0.0], dtype=np.float64)
                keeper_rigid.set_linear_velocity(kv.tolist())
                keeper_rigid.set_angular_velocity(np.zeros(3))
                keeper_rigid.wake_up()
            except Exception:
                pass

        self._ball_blocked = True
        self._ball_live = True
        self._keeper_deployed = True

    def _wait_for_outcome(self):
        max_steps = int(self.ball_total_steps + self.ball_settle_steps)
        for i in range(max(0, max_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _place_keeper_from_top(self, arm_tag: ArmTag):
        if getattr(self, "goalkeeper", None) is None or self.goalkeeper_target_pose is None:
            return

        target_p = np.asarray(self.goalkeeper_target_pose.p, dtype=np.float64)
        frame_clearance_z = self.table_top_z + max(
            self.goal_post_h + self.keeper_half_z + 0.08,
            self.keeper_half_z + 0.18,
        )

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        lift_dz = float(frame_clearance_z - keeper_p[2])
        if lift_dz > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_dz))

        # Approach over the goal, then iteratively correct XY so the *keeper*
        # footprint lands fully inside the green zone (not just the EE on the intercept).
        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        goal_over_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.01))
        goal_over_y = float(self.goal_center_y)
        dx_goal = float(goal_over_x - keeper_p[0])
        dy_goal = float(goal_over_y - keeper_p[1])
        if abs(dx_goal) > 1e-4 or abs(dy_goal) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx_goal, y=dy_goal))

        for _ in range(4):
            keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
            dx = float(target_p[0] - keeper_p[0])
            dy = float(target_p[1] - keeper_p[1])
            if abs(dx) < 0.003 and abs(dy) < 0.003:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        place_dz = float(target_p[2] - keeper_p[2])
        if abs(place_dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=place_dz))

        # Final XY polish at table height (grasp offset often shows up here).
        for _ in range(2):
            keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
            dx = float(target_p[0] - keeper_p[0])
            dy = float(target_p[1] - keeper_p[1])
            if abs(dx) < 0.002 and abs(dy) < 0.002:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))

        self.move(self.open_gripper(arm_tag))
        for _ in range(8):
            self._update_kinematic_tasks()
            self.scene.step()
        # Seat where released — stay dynamic so a save bounce can transfer mass.
        self._seat_keeper_dynamic()
        self._keeper_deployed = True
        self._latch_keeper_deploy_metric()

    def _retreat_arm_home(self, arm_tag: ArmTag, lift_z: float = 0.08):
        """After releasing the keeper, clear the goal then return the arm home.

        Forces ``plan_success`` so a failed place step cannot skip the retreat.
        A failed home plan does not invalidate an already-deployed save.
        """
        deployed = bool(getattr(self, "_keeper_deployed", False))
        self.plan_success = True
        if lift_z and abs(float(lift_z)) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=float(lift_z), move_axis="arm"))
            self.plan_success = True
        self.move(self.back_to_origin(arm_tag))
        if deployed:
            self.plan_success = True

    def _seat_keeper_dynamic(self):
        """Pin the keeper on the table at its current pose, ready for a solid bounce.

        Held kinematic until impact so it cannot tip/slide before the shot arrives.
        Orientation from placement is preserved — the OBB bounce uses the live pose.
        """
        if getattr(self, "goalkeeper", None) is None:
            return
        pose = self.goalkeeper.get_pose()
        p = np.asarray(pose.p, dtype=np.float64).copy()
        p[2] = float(self.table_top_z + self.keeper_half_z)
        # Preserve placement orientation (solid bounce works for any pose).
        seat_pose = sapien.Pose(p.tolist(), list(pose.q))
        try:
            self.goalkeeper.set_pose(seat_pose)
        except Exception:
            try:
                self.goalkeeper.actor.set_pose(seat_pose)
            except Exception:
                seat_pose = pose
        self._keeper_drop_pose = seat_pose
        rigid = self._get_rigid(self.goalkeeper)
        if rigid is None:
            return
        try:
            m = float(getattr(self, "keeper_mass", self.KEEPER_MASS))
            rigid.set_mass(m)
            # Box inertia about center (uniform density) — stable when unlocked.
            hx, hy, hz = float(self.keeper_half_x), float(self.keeper_half_y), float(self.keeper_half_z)
            ix = (1.0 / 12.0) * m * ((2 * hy) ** 2 + (2 * hz) ** 2)
            iy = (1.0 / 12.0) * m * ((2 * hx) ** 2 + (2 * hz) ** 2)
            iz = (1.0 / 12.0) * m * ((2 * hx) ** 2 + (2 * hy) ** 2)
            try:
                rigid.set_inertia([ix, iy, iz])
            except Exception:
                pass
            rigid.set_linear_damping(3.0)
            rigid.set_angular_damping(12.0)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            self._set_restitution(rigid, self.BOUNCE_RESTITUTION, static_f=0.9, dynamic_f=0.7)
            self._set_collision_enabled(self.goalkeeper, True)
            # Freeze until the solid hit unlocks for mass transfer.
            rigid.set_kinematic(True)
        except Exception:
            pass

    def _hold_keeper_upright(self):
        """Stabilize the deployed keeper on the table without resetting its orientation."""
        if getattr(self, "goalkeeper", None) is None:
            return
        if getattr(self, "_ball_blocked", False) or getattr(self, "_ball_live", False):
            return
        if not getattr(self, "_keeper_deployed", False):
            return
        pose = self.goalkeeper.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        target_z = float(self.table_top_z + self.keeper_half_z)
        q = list(pose.q)
        if abs(float(p[2]) - target_z) > 2e-3:
            seat = sapien.Pose([float(p[0]), float(p[1]), target_z], q)
            try:
                self.goalkeeper.set_pose(seat)
            except Exception:
                try:
                    self.goalkeeper.actor.set_pose(seat)
                except Exception:
                    return
        rigid = self._get_rigid(self.goalkeeper)
        if rigid is None:
            return
        try:
            if not rigid.get_kinematic():
                rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    def _freeze_keeper_in_place(self):
        """Back-compat alias: seat dynamically (mass must stay in play for bounce)."""
        self._seat_keeper_dynamic()

    def _hold_keeper_kinematic(self):
        """Back-compat: re-seat dynamically instead of locking kinematic."""
        self._seat_keeper_dynamic()

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.table_top_z = 0.74 + self.table_z_bias

        self.goal_x_abs = float(c.get("goal_x", c.get("goal_y", self.GOAL_X_DEFAULT)))
        self.goal_center_y = float(c.get("goal_center_y", self.GOAL_CENTER_Y_DEFAULT))
        goal_center_y_jitter = float(c.get("goal_center_y_jitter", self.GOAL_CENTER_Y_JITTER_DEFAULT))
        self.goal_center_y = float(np.random.uniform(
            self.goal_center_y - goal_center_y_jitter,
            self.goal_center_y + goal_center_y_jitter,
        ))
        self.goal_half_w = float(c.get("goal_half_w", self.GOAL_HALF_W_DEFAULT))
        self.goal_post_t = float(c.get("goal_post_t", self.GOAL_POST_T_DEFAULT))
        self.goal_post_h = float(c.get("goal_post_h", self.GOAL_POST_H_DEFAULT))
        self.goal_bar_t = float(c.get("goal_bar_t", self.GOAL_BAR_T_DEFAULT))
        self.green_area_x_len = float(c.get("green_area_x_len", self.GREEN_AREA_X_LEN_DEFAULT))
        self.green_area_y_extra = float(c.get("green_area_y_extra", self.GREEN_AREA_Y_EXTRA_DEFAULT))
        self.red_line_green_gap = abs(
            float(c.get("red_line_green_gap", self.RED_LINE_GREEN_GAP_DEFAULT))
        )
        # Min deadline offset: fully before the green zone + required clear gap.
        min_red_offset = float(self.green_area_x_len + self.red_line_green_gap)
        self.red_line_goal_offset = float(
            c.get("red_line_x", c.get("red_line_y", self.RED_LINE_X_DEFAULT))
        )
        # When on: sample the red line on the field side of green (never into/onto green).
        # Legacy ``red_line_inward_max`` is treated as outward range toward the field.
        self.randomize_red_line = bool(c.get("randomize_red_line", False))
        outward_max = abs(
            float(
                c.get(
                    "red_line_outward_max",
                    c.get("red_line_inward_max", self.RED_LINE_OUTWARD_MAX_DEFAULT),
                )
            )
        )
        if self.randomize_red_line:
            lo = min_red_offset
            hi = min_red_offset + outward_max
            self.red_line_goal_offset = float(np.random.uniform(lo, hi))
        else:
            self.red_line_goal_offset = float(max(abs(self.red_line_goal_offset), min_red_offset))

        self.keeper_x_abs = float(c.get("keeper_x", c.get("keeper_y", self.KEEPER_X_DEFAULT)))
        self.keeper_pose_tol = float(c.get("keeper_pose_tol", self.KEEPER_POSE_TOL_DEFAULT))
        self.keeper_spawn_x = float(c.get("keeper_spawn_x", self.KEEPER_SPAWN_X_DEFAULT))
        self.keeper_spawn_y = float(c.get("keeper_spawn_y", self.KEEPER_SPAWN_Y_DEFAULT))
        self.keeper_goal_clearance = float(c.get("keeper_goal_clearance", self.KEEPER_GOAL_CLEARANCE_DEFAULT))
        self.keeper_mass = float(c.get("keeper_mass", self.KEEPER_MASS))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        # Nominal speed with per-episode ±20% sampling (override via ball_speed_scale_*).
        # Legacy absolute ball_speed_min/max still accepted if provided.
        self.ball_speed_default = float(c.get("ball_speed", self.BALL_SPEED_DEFAULT))
        self.ball_speed_scale_min = float(
            c.get("ball_speed_scale_min", self.BALL_SPEED_SCALE_MIN_DEFAULT)
        )
        self.ball_speed_scale_max = float(
            c.get("ball_speed_scale_max", self.BALL_SPEED_SCALE_MAX_DEFAULT)
        )
        if "ball_speed_min" in c or "ball_speed_max" in c:
            lo = float(c.get(
                "ball_speed_min",
                self.ball_speed_default * self.ball_speed_scale_min,
            ))
            hi = float(c.get(
                "ball_speed_max",
                self.ball_speed_default * self.ball_speed_scale_max,
            ))
            self.ball_speed_min = float(min(lo, hi))
            self.ball_speed_max = float(max(lo, hi))
        else:
            self.ball_speed_min = float(
                self.ball_speed_default * min(self.ball_speed_scale_min, self.ball_speed_scale_max)
            )
            self.ball_speed_max = float(
                self.ball_speed_default * max(self.ball_speed_scale_min, self.ball_speed_scale_max)
            )
        self.ball_start_x_abs = float(c.get("ball_start_x", self.BALL_START_X_DEFAULT))
        self.ball_start_y_jitter = float(c.get("ball_start_y_jitter", c.get("ball_start_x_jitter", self.BALL_START_Y_JITTER_DEFAULT)))
        self.ball_goal_end_x_offset = float(c.get("ball_goal_end_x_offset", c.get("ball_goal_end_y_offset", self.BALL_GOAL_END_X_OFFSET_DEFAULT)))
        self.ball_target_y_margin = float(c.get("ball_target_y_margin", c.get("ball_target_x_margin", self.BALL_TARGET_Y_MARGIN_DEFAULT)))
        self.ball_angle_deg_min = float(c.get("ball_angle_deg_min", self.BALL_ANGLE_DEG_MIN_DEFAULT))
        self.ball_angle_deg_max = float(c.get("ball_angle_deg_max", self.BALL_ANGLE_DEG_MAX_DEFAULT))
        self.ball_settle_steps = int(c.get("ball_settle_steps", self.BALL_SETTLE_STEPS_DEFAULT))

        # Option 1 — field players (also accepts legacy ``option: 1`` / --option 1).
        self.players_enabled = self._parse_players_enabled(c)
        self.players_max = max(1, int(c.get("players_max", self.PLAYERS_MAX_DEFAULT)))
        self.player_thin_scale = float(c.get("player_thin_scale", self.PLAYER_THIN_SCALE_DEFAULT))
        self.player_corridor_margin = float(
            c.get("player_corridor_margin", self.PLAYER_CORRIDOR_MARGIN_DEFAULT)
        )
        self.player_y_spread = float(c.get("player_y_spread", self.PLAYER_Y_SPREAD_DEFAULT))
        self.player_separation = float(c.get("player_separation", self.PLAYER_SEPARATION_DEFAULT))
        self.player_max_goal_dist_mult = float(
            c.get("player_max_goal_dist_mult", self.PLAYER_MAX_GOAL_DIST_MULT_DEFAULT)
        )
        self.player_goal_face_bias = float(
            c.get("player_goal_face_bias", self.PLAYER_GOAL_FACE_BIAS_DEFAULT)
        )
        # Footprint / height filled after keeper sizing below (same figure, thinner XY).
        self.player_half_xy = None
        self.player_half_z = None
        self._players = []
        self._bounce_player_idx = -1
        self.ball_bounce_pose = None
        self._bounce_end_y = None
        self._players_hit = set()
        # Stash raw overrides for after keeper_half_* are known.
        self._player_half_xy_override = c.get("player_half_xy", None)
        self._player_half_z_override = c.get("player_half_z", None)

        # Option 2 — field cover (also accepts legacy ``option: 2`` / --option 2).
        self.cover_enabled = self._parse_cover_enabled(c)
        self.cover_entry_gap = float(c.get("cover_entry_gap", self.COVER_ENTRY_GAP_DEFAULT))
        self.cover_exit_gap_frac = float(
            c.get("cover_exit_gap_frac", self.COVER_EXIT_GAP_FRAC_DEFAULT)
        )
        self.red_line_y_extra = float(c.get("red_line_y_extra", self.RED_LINE_Y_EXTRA_DEFAULT))
        self.red_line_half_y = float(self.goal_half_w + self.red_line_y_extra)
        # Tunnel cavity matches the red line width unless cover_half_y is set explicitly.
        if "cover_half_y" in c:
            self.cover_half_y = float(c["cover_half_y"])
        elif self.COVER_HALF_Y_DEFAULT is not None:
            self.cover_half_y = float(self.COVER_HALF_Y_DEFAULT)
        else:
            self.cover_half_y = float(self.red_line_half_y)
        self.cover_clearance_z = float(
            c.get("cover_clearance_z", self.COVER_CLEARANCE_Z_DEFAULT)
        )
        self.cover_wall_t = float(c.get("cover_wall_t", self.COVER_WALL_T_DEFAULT))
        self.cover_roof_t = float(c.get("cover_roof_t", self.COVER_ROOF_T_DEFAULT))
        self.cover_len_min = float(c.get("cover_len_min", self.COVER_LEN_MIN_DEFAULT))
        self.cover_player_clearance = float(
            c.get("cover_player_clearance", self.COVER_PLAYER_CLEARANCE_DEFAULT)
        )
        self._cover_parts = []
        self._cover_wall_boxes = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_y_min = None
        self.cover_y_max = None
        self.cover_len = 0.0

        # Soccer-style visual net behind the goal mouth.
        self.net_enabled = self._as_bool(c.get("net_enabled", None), self.NET_ENABLED_DEFAULT)
        self.net_depth = float(c.get("net_depth", self.NET_DEPTH_DEFAULT))
        self.net_cell = float(c.get("net_cell", self.NET_CELL_DEFAULT))
        self.net_strand_t = float(c.get("net_strand_t", self.NET_STRAND_T_DEFAULT))
        self._net_parts = []

        # Mirror flips the whole field across the table midline (goal/ball/players/cover).
        # mirrored → travel_dir −1 → goal on −x → left gripper; else +x → right gripper.
        self.mirrored = self._parse_mirrored(c)
        self.travel_dir = -1.0 if self.mirrored else 1.0
        self.goal_x = float(self.travel_dir * abs(self.goal_x_abs))
        # Clamp again in case green length / gap changed after offset sampling.
        min_red_offset = float(self.green_area_x_len + self.red_line_green_gap)
        self.red_line_goal_offset = float(max(abs(self.red_line_goal_offset), min_red_offset))
        self.red_line_x = float(self.goal_x - self.travel_dir * self.red_line_goal_offset)
        self.ball_speed = float(np.random.uniform(self.ball_speed_min, self.ball_speed_max))
        self.keeper_half_x = self.ball_radius
        self.keeper_half_y = self.ball_radius
        keeper_z_mult = float(c.get("keeper_half_z_mult", self.KEEPER_HALF_Z_MULT_DEFAULT))
        if "keeper_half_z" in c:
            self.keeper_half_z = float(c["keeper_half_z"])
        else:
            self.keeper_half_z = float(self.ball_radius * keeper_z_mult)
        # Field players: same figure as keeper, blue shirt, thinner XY footprint.
        thin = float(np.clip(self.player_thin_scale, 0.2, 1.0))
        if self._player_half_xy_override is not None:
            self.player_half_xy = float(self._player_half_xy_override)
        else:
            self.player_half_xy = float(self.ball_radius * thin)
        if self._player_half_z_override is not None:
            self.player_half_z = float(self._player_half_z_override)
        else:
            self.player_half_z = float(self.keeper_half_z)
        green_half_x = 0.5 * self.green_area_x_len
        green_half_y = self.goal_half_w + 0.5 * self.green_area_y_extra
        self.green_area_center_x = float(self.goal_x - self.travel_dir * green_half_x)
        self.green_area_center_y = float(self.goal_center_y)
        self.green_area_x_min = float(min(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_x_max = float(max(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_y_min = float(self.goal_center_y - green_half_y)
        self.green_area_y_max = float(self.goal_center_y + green_half_y)

        goal_end_x = float(self.goal_x + self.travel_dir * abs(self.ball_goal_end_x_offset))
        dt = float(self.scene.get_timestep())
        start_x = float(-self.travel_dir * abs(self.ball_start_x_abs))
        start_y = float(np.random.uniform(
            self.goal_center_y - self.ball_start_y_jitter,
            self.goal_center_y + self.ball_start_y_jitter,
        ))

        # Spawn Opt-1 players before choosing the post-bounce goal aim.
        self._spawn_field_players(start_x, start_y, goal_end_x)
        # Option 2 cover over the mid field (after start/red-line are known).
        self._spawn_field_cover(start_x)

        if self.ball_bounce_pose is not None and self._bounce_end_y is not None:
            end_y = float(self._bounce_end_y)
        else:
            for _ in range(64):
                launch_angle_deg = float(np.random.uniform(self.ball_angle_deg_min, self.ball_angle_deg_max))
                angle_rad = np.deg2rad(launch_angle_deg)
                end_y = float(start_y + np.tan(angle_rad) * (goal_end_x - start_x))
                if abs(end_y - self.goal_center_y) <= (self.goal_half_w - self.ball_target_y_margin):
                    break
            else:
                end_y = float(np.clip(
                    end_y,
                    self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
                    self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
                ))

        self.launch_angle_deg = float(np.degrees(np.arctan2(end_y - start_y, goal_end_x - start_x)))
        self.ball_start_pose = np.array(
            [start_x, start_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        self.ball_target_pose = np.array(
            [goal_end_x, end_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        if self.ball_bounce_pose is not None:
            self._set_ball_waypoints(
                [self.ball_start_pose, self.ball_bounce_pose, self.ball_target_pose]
            )
            ball_vec = self.ball_target_pose - self.ball_bounce_pose
        else:
            self._set_ball_waypoints([self.ball_start_pose, self.ball_target_pose])
            ball_vec = self.ball_target_pose - self.ball_start_pose
        ball_dist = float(self._ball_path_len)
        self.ball_dir = ball_vec / max(float(np.linalg.norm(ball_vec)), 1e-8)
        self.ball_total_steps = max(1, int(np.ceil(ball_dist / max(self.ball_speed * dt, 1e-8))))

        keeper_x_min = float(self.green_area_x_min + self.keeper_half_x)
        keeper_x_max = float(self.green_area_x_max - self.keeper_half_x)
        self.goal_intersection_y = self._ball_path_y_at_x(self.goal_x)
        preferred_keeper_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.002))
        self.keeper_x = float(np.clip(preferred_keeper_x, keeper_x_min, keeper_x_max))
        keeper_y = float(self._ball_path_y_at_x(self.keeper_x))
        keeper_y = float(np.clip(
            keeper_y,
            self.green_area_y_min + self.keeper_half_y,
            self.green_area_y_max - self.keeper_half_y,
        ))
        self.goalkeeper_target_pose = sapien.Pose(
            [self.keeper_x, keeper_y, self.table_top_z + self.keeper_half_z],
            [1, 0, 0, 0],
        )

        goal_color = (0.92, 0.92, 0.94)
        post_half = [self.goal_post_t * 0.5, self.goal_post_t * 0.5, self.goal_post_h * 0.5]
        self.goal_left_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y - self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_left_post",
        )
        self.goal_right_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y + self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_right_post",
        )
        self.goal_bar = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y, self.table_top_z + self.goal_post_h - self.goal_bar_t * 0.5], [1, 0, 0, 0]),
            half_size=[self.goal_post_t * 0.5, self.goal_half_w + self.goal_post_t, self.goal_bar_t * 0.5],
            color=goal_color,
            is_static=True,
            name="goal_bar",
        )
        self._spawn_goal_net()
        create_visual_box(
            self,
            pose=sapien.Pose([self.red_line_x, self.goal_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[0.002, self.red_line_half_y, 0.001],
            color=(0.95, 0.12, 0.12),
            name="red_line",
        )
        create_visual_box(
            self,
            pose=sapien.Pose([self.green_area_center_x, self.green_area_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[green_half_x, green_half_y, 0.001],
            color=(0.18, 0.72, 0.25),
            name="goal_green_area",
        )

        keeper_x0 = float(self.goal_x)
        goal_lower_y = float(self.goal_center_y - self.goal_half_w)
        keeper_y0 = float(goal_lower_y - self.keeper_goal_clearance - self.keeper_half_y)
        self.goalkeeper = self._build_goalkeeper(
            sapien.Pose([keeper_x0, keeper_y0, self.table_top_z + self.keeper_half_z], [1, 0, 0, 0]),
        )
        self.goalkeeper.set_mass(float(self.keeper_mass))
        keeper_rigid = self._get_rigid(self.goalkeeper)
        if keeper_rigid is not None:
            try:
                keeper_rigid.set_mass(float(self.keeper_mass))
                keeper_rigid.set_linear_damping(3.0)
                keeper_rigid.set_angular_damping(12.0)
                self._set_restitution(
                    keeper_rigid, self.BOUNCE_RESTITUTION, static_f=0.9, dynamic_f=0.7
                )
                self._set_collision_enabled(self.goalkeeper, True)
            except Exception:
                pass

        self.ball = create_sphere(
            self.scene,
            pose=sapien.Pose(self.ball_start_pose.tolist(), [1, 0, 0, 0]),
            radius=self.ball_radius,
            color=(0.95, 0.95, 0.95),
            is_static=False,
            name="goal_ball",
            texture_id="soccer_ball",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            try:
                mass = float(self.BALL_MASS)
                self._ball_rigid.set_mass(mass)
                inertia = 0.4 * mass * (self.ball_radius ** 2)
                self._ball_rigid.set_inertia([inertia, inertia, inertia])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
                self._ball_rigid.set_linear_damping(0.05)
                self._ball_rigid.set_angular_damping(0.05)
                self._ball_rigid.set_linear_velocity(np.zeros(3))
                self._ball_rigid.set_angular_velocity(np.zeros(3))
                self._set_restitution(
                    self._ball_rigid, self.BOUNCE_RESTITUTION, static_f=0.2, dynamic_f=0.15
                )
            except Exception:
                pass
        # Kinematic approach must not PhysX-shove the keeper; contacts enable on bounce.
        self._set_collision_enabled(self.ball, False)

        self.add_prohibit_area(self.goalkeeper, padding=0.02)
        self._loaded = True

    def _begin_mass_bounce(self, hit_p):
        """Back-compat: front-normal solid bounce along −travel_dir."""
        n = np.array([-float(self.travel_dir), 0.0], dtype=np.float64)
        self._begin_solid_bounce(hit_p, n)

    def _update_live_ball(self):
        """After a save bounce: follow PhysX, pin to the table, watch for a late goal."""
        if self._ball_rigid is None or getattr(self, "ball", None) is None:
            self._ball_motion_active = False
            return
        try:
            p = np.asarray(self.ball.get_pose().p, dtype=np.float64).copy()
            v = np.asarray(self._ball_rigid.get_linear_velocity(), dtype=np.float64).copy()
        except Exception:
            self._ball_motion_active = False
            return

        z = float(self.table_top_z + self.ball_radius)
        if abs(float(p[2]) - z) > 1e-3 or abs(float(v[2])) > 1e-3:
            p[2] = z
            v[2] = 0.0
            pose = sapien.Pose(p.tolist(), list(self.ball.get_pose().q))
            try:
                self.ball.set_pose(pose)
                self._ball_rigid.set_linear_velocity(v.tolist())
            except Exception:
                pass

        if self.travel_dir * p[0] >= self.travel_dir * self.goal_x:
            self._ball_crossed_goal = True
        if (
            (not self._goal_conceded)
            and self.travel_dir * p[0] >= self.travel_dir * self.goal_x
            and abs(float(p[1] - self.goal_center_y)) <= self.goal_half_w
        ):
            # A rebound that somehow still enters the mouth counts as conceded.
            self._goal_conceded = True
            self._ball_blocked = False

        speed = float(np.linalg.norm(v[:2]))
        self._ball_step += 1
        if speed < 0.004 or self._ball_step > int(self.ball_total_steps + self.ball_settle_steps):
            self._ball_motion_active = False
            try:
                self._ball_rigid.set_linear_velocity(np.zeros(3))
                self._ball_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    # ------------------------------------------------------------- motion / checks
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._track_save_metrics()
        if not getattr(self, "_ball_motion_active", False):
            return
        if getattr(self, "_ball_live", False):
            self._update_live_ball()
            return
        if self._ball_blocked:
            return
        if self._ball_rigid is None or getattr(self, "ball", None) is None:
            return

        # Keep the deployed keeper planted while the shot approaches (pose preserved).
        self._hold_keeper_upright()

        self._ball_step += 1
        progress = min(1.0, self._ball_step / float(self.ball_total_steps))
        prev_progress = min(1.0, (self._ball_step - 1) / float(self.ball_total_steps))
        prev_p = self._ball_pos_at_progress(prev_progress)
        next_p = self._ball_pos_at_progress(progress)

        # Green-zone placement gates late-failure / success, not physical blocking.
        keeper_ok = bool(self._keeper_in_zone())
        if (not self._late_failure) and (self.travel_dir * next_p[0] >= self.travel_dir * self.red_line_x) and (not keeper_ok):
            self._late_failure = True

        # Field players are solid OBBs — bounce instead of tunneling.
        if self._try_field_player_solid_bounce(prev_p, next_p):
            return

        # Tunnel side walls are solid — bounce instead of tunneling.
        if self._try_cover_wall_solid_bounce(prev_p, next_p):
            return

        # Solid OBB at the live pose: any face / angle bounce (legal if in green zone).
        if self._try_keeper_solid_bounce(prev_p, next_p):
            return

        if self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x:
            self._ball_crossed_goal = True
        if (
            (not self._goal_conceded)
            and self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x
            and abs(float(next_p[1] - self.goal_center_y)) <= self.goal_half_w
        ):
            self._goal_conceded = True

        pose = sapien.Pose(next_p.tolist(), self._soccer_ball_quat_at_progress(progress))
        self.ball.set_pose(pose)
        try:
            self._ball_rigid.set_kinematic_target(pose)
        except Exception:
            pass
        if progress >= 1.0:
            self._ball_motion_active = False

    def _start_shot(self):
        """Begin the kinematic shot. Idempotent after setup_demo."""
        if getattr(self, "_ball_motion_active", False):
            return
        self._ball_step = 0
        self._ball_blocked = False
        self._ball_live = False
        self._block_was_legal = False
        self._reset_metric_state()
        self._goal_conceded = False
        self._late_failure = False
        self._ball_crossed_goal = False
        self._keeper_deployed = False
        self._keeper_drop_pose = None
        self._players_hit = set()
        self._ball_motion_active = True
        if getattr(self, "ball", None) is not None:
            self._set_collision_enabled(self.ball, False)

    # ----------------------------------------------------------------- policy
    def play_once(self):
        self._start_shot()

        arm_tag = ArmTag("left" if self.mirrored else "right")
        grasp_contact_id = [0, 1, 2, 3]

        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.goalkeeper,
                arm_tag=arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self._place_keeper_from_top(arm_tag)
        # After the keeper is dropped, always return the arm to its origin pose.
        self._retreat_arm_home(arm_tag, lift_z=0.08)
        # Keep the keeper dynamic (mass-aware bounce); only re-seat pose/damping.
        self._seat_keeper_dynamic()

        self._wait_for_outcome()

        self.info["info"] = {
            "{A}": "goalkeeper",
            "{B}": "goal_frame",
            "{a}": str(arm_tag),
        }
        # Option captions use {o}; attach when Option 1 and/or Option 2 is on.
        if self.players_enabled or self.cover_enabled:
            self.info["info"]["{o}"] = self._option_label()
        return self.info

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_shot_step = None      # ball started travelling
        self._metric_deploy_step = None    # keeper let go on the table
        self._metric_impact_step = None    # ball hit the keeper (or crossed the line)
        self._metric_intercept_err = None  # keeper-to-shot-line y gap at deploy, metres

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _shot_y_at_goal_line(self) -> float:
        """Where the scripted shot path crosses the goal line, in world y."""
        pts = getattr(self, "_ball_waypoints", None)
        if pts is None or len(pts) < 2:
            a = np.asarray(self.ball_start_pose, dtype=np.float64)
            b = np.asarray(self.ball_target_pose, dtype=np.float64)
            pts = [a, b]
        pts = [np.asarray(q, dtype=np.float64) for q in pts]
        gx = float(self.goal_x)
        d = float(self.travel_dir)
        for i in range(len(pts) - 1):
            x0, x1 = float(pts[i][0]), float(pts[i + 1][0])
            if (d * x0) <= (d * gx) <= (d * x1) and abs(x1 - x0) > 1e-9:
                t = (gx - x0) / (x1 - x0)
                return float(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
        return float(pts[-1][1])

    def _latch_keeper_deploy_metric(self):
        """Called the moment the keeper is released onto the table.

        Latches the aim error there and then: after this the keeper is dynamic and
        the bounce moves it, so its final pose no longer reflects the placement.
        """
        try:
            if self._metric_deploy_step is not None:
                return
            self._metric_deploy_step = self._metric_step()
            kp = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
            self._metric_intercept_err = abs(float(kp[1]) - self._shot_y_at_goal_line())
        except Exception:
            self._metric_intercept_err = None

    def _track_save_metrics(self):
        """Per-step latches: shot start, and the resolving impact / line crossing."""
        try:
            if (self._metric_shot_step is None
                    and getattr(self, "_ball_motion_active", False)
                    and int(getattr(self, "_ball_step", 0)) > 0):
                self._metric_shot_step = self._metric_step()
            if self._metric_impact_step is None and (
                getattr(self, "_ball_blocked", False)
                or getattr(self, "_ball_crossed_goal", False)
            ):
                self._metric_impact_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `deploy_latency_steps` — steps from the shot starting to roll until the
        keeper is released on the table. The shot is already in flight, so this is the
        operator's read-and-commit time; it must beat the ball to the line.
        extra2 `intercept_error_norm` — |keeper y - shot y at the goal line| at the
        instant of release, divided by the keeper half-width. LOWER is better; <= 1.0
        means the block body actually covers the shot line.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a = getattr(self, "_metric_shot_step", None)
        b = getattr(self, "_metric_deploy_step", None)
        lat = None if (a is None or b is None) else int(b) - int(a)
        out["deploy_latency_steps"] = lat
        out["deploy_latency_s"] = None if lat is None else round(lat * dt, 4)

        c = getattr(self, "_metric_impact_step", None)
        margin = None if (b is None or c is None) else int(c) - int(b)
        out["deploy_margin_steps"] = margin  # >0 = keeper was down before the ball arrived
        out["deploy_margin_s"] = None if margin is None else round(margin * dt, 4)

        err = getattr(self, "_metric_intercept_err", None)
        try:
            denom = max(float(self.keeper_half_y), 1e-9)
            out["intercept_error_norm"] = None if err is None else round(float(err) / denom, 4)
            out["intercept_error_m"] = None if err is None else round(float(err), 5)
        except Exception:
            out["intercept_error_norm"] = None
            out["intercept_error_m"] = None
        return out

    # ---------------------------------------------------------------- success / obs
    def check_success(self):
        keeper_ok = self._keeper_in_zone()
        block_legal = bool(getattr(self, "_block_was_legal", False))
        success = bool(
            keeper_ok
            and block_legal
            and self._ball_blocked
            and (not self._late_failure)
            and (not self._goal_conceded)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        self.info["save_goal"] = {
            "keeper_in_zone": bool(keeper_ok),
            "block_was_legal": block_legal,
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
            "players_enabled": bool(getattr(self, "players_enabled", False)),
            "n_players": int(len(getattr(self, "_players", []) or [])),
            "bounce_player_idx": int(getattr(self, "_bounce_player_idx", -1)),
            "cover_enabled": bool(getattr(self, "cover_enabled", False)),
            "mirrored": bool(getattr(self, "mirrored", False)),
            "net_enabled": bool(getattr(self, "net_enabled", False)),
            "n_net_parts": int(len(getattr(self, "_net_parts", []) or [])),
            "cover_x_min": (
                float(self.cover_x_min) if getattr(self, "cover_x_min", None) is not None else None
            ),
            "cover_x_max": (
                float(self.cover_x_max) if getattr(self, "cover_x_max", None) is not None else None
            ),
        }
        return success

    def get_obs(self):
        obs = super().get_obs()
        player_positions = []
        for p in getattr(self, "_players", []) or []:
            try:
                player_positions.append(p.get_pose().p.tolist())
            except Exception:
                player_positions.append([0.0, 0.0, 0.0])
        obs["save_goal"] = {
            "ball_pos": self.ball.get_pose().p.tolist() if getattr(self, "ball", None) is not None else [0.0, 0.0, 0.0],
            "keeper_pos": self.goalkeeper.get_pose().p.tolist() if getattr(self, "goalkeeper", None) is not None else [0.0, 0.0, 0.0],
            "keeper_target": self.goalkeeper_target_pose.p.tolist() if self.goalkeeper_target_pose is not None else [0.0, 0.0, 0.0],
            "keeper_in_zone": bool(self._keeper_in_zone()),
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
            "ball_speed": float(getattr(self, "ball_speed", 0.0)),
            "ball_speed_default": float(getattr(self, "ball_speed_default", 0.0)),
            "launch_angle_deg": float(getattr(self, "launch_angle_deg", 0.0)),
            "red_line_x": float(getattr(self, "red_line_x", 0.0)),
            "goal_x": float(getattr(self, "goal_x", 0.0)),
            "goal_center_y": float(getattr(self, "goal_center_y", 0.0)),
            "goal_intersection_y": float(getattr(self, "goal_intersection_y", 0.0)),
            "green_area_center": [float(getattr(self, "green_area_center_x", 0.0)), float(getattr(self, "green_area_center_y", 0.0))],
            "green_area_bounds": [
                float(getattr(self, "green_area_x_min", 0.0)),
                float(getattr(self, "green_area_x_max", 0.0)),
                float(getattr(self, "green_area_y_min", 0.0)),
                float(getattr(self, "green_area_y_max", 0.0)),
            ],
            "travel_dir": float(getattr(self, "travel_dir", 0.0)),
            "mirrored": bool(getattr(self, "mirrored", False)),
            "net_enabled": bool(getattr(self, "net_enabled", False)),
            "n_net_parts": int(len(getattr(self, "_net_parts", []) or [])),
            "players_enabled": bool(getattr(self, "players_enabled", False)),
            "n_players": int(len(getattr(self, "_players", []) or [])),
            "bounce_player_idx": int(getattr(self, "_bounce_player_idx", -1)),
            "player_positions": player_positions,
            "ball_bounce_pos": (
                self.ball_bounce_pose.tolist()
                if getattr(self, "ball_bounce_pose", None) is not None
                else [0.0, 0.0, 0.0]
            ),
            "cover_enabled": bool(getattr(self, "cover_enabled", False)),
            "cover_x_range": [
                float(self.cover_x_min) if getattr(self, "cover_x_min", None) is not None else 0.0,
                float(self.cover_x_max) if getattr(self, "cover_x_max", None) is not None else 0.0,
            ],
        }
        return obs
