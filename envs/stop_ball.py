from ._office_base_task import Office_base_task
from .utils import *
from ._GLOBAL_CONFIGS import *
import json
import sapien
import sapien.physx
import numpy as np
from pathlib import Path
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult, quat2mat


class stop_ball(Office_base_task):
    """Stop a table-tennis ball rolling off the table after it falls from a shelf.

    Scene: full-width wall shelf with plant + sparse décor (globe / trophy /
    rest / clock). A ``027_table-tennis`` ball starts moving immediately on an
    angled path that may exit the front or a side edge; the matching arm reacts.
    Table décor is kept off the predicted roll corridor so only the arm can stop
    it. Success requires an arm hit that keeps the ball on the table — a touch
    that then deflects off any edge or drops below the tabletop fails. Settling
    under the shelf / at the back of the table is fine. The ball is sized to the
    open WSG finger gap plus 0.5 cm so an open gripper placed in front of it can
    catch it.
    """

    BALL_IDS = [0]  # orange table-tennis (id 1 is hard to see in head cam)
    # Asset has no authored scale; 0.02 × mean(extents) ≈ 4.36 cm diameter.
    BALL_BASE_SCALE = 0.02
    # Fully-open WSG 50-110 inner gap (URDF ±0.055 m).
    WSG_OPEN_FINGER_GAP = 0.110
    BALL_CATCH_MARGIN = 0.005  # +0.5 cm so the ball cannot pass through open pads
    # 11.5 cm diameter / 4.36 cm base ≈ 2.64×
    BALL_SCALE_MULT_DEFAULT = 2.64
    # 100% larger than the previous 0.50 scale.
    TROPHY_SCALE_MULT = 1.00
    # Prior screen was 1.5×; 30% smaller → 1.05× (~41 cm wide).
    SCREEN_SCALE_MULT = 1.05

    TISSUE_IDS = list(range(7))
    GLOBE_IDS = [2, 3]
    TROPHY_IDS = [3, 4, 0]  # prefer narrower desk-trophy variants
    REST_IDS = list(range(4))
    ALARM_IDS = list(range(6))
    SCREEN_IDS = list(range(4))
    SPEAKER_IDS = list(range(6))
    FAN_IDS = list(range(7))
    BOOK_IDS = [0, 1]
    SEAL_IDS = [0, 1, 2, 3, 4, 6]
    MILKTEA_IDS = [0, 1, 2, 4, 5, 6]

    ROLL_SPEED_DEFAULT = 0.15
    ROLL_SPEED_JITTER_FRAC = 0.30  # sample table-roll speed in mean ± this fraction
    FALL_SPEED_XY_DEFAULT = 0.07
    SHELF_ROLL_DURATION_DEFAULT = 1.00
    GRAVITY = 9.81

    TABLE_EDGE_Y_DEFAULT = -0.30
    TABLE_X_EDGE = 0.52          # |x| past this = fallen off a side
    # Table width 0.7 m, +Y toward the wall; under-shelf settles are still on-table.
    TABLE_BACK_Y_DEFAULT = 0.35
    # Heading from −Y (toward robot): 0 = straight front; ± → angled / side exit.
    ROLL_ANGLE_MAX_DEFAULT = 0.70   # rad (~40°)
    SIDE_EXIT_ANGLE_MIN = 0.55      # rad; steeper headings tend to exit a side
    INTERCEPT_FRAC_DEFAULT = 0.55
    CATCH_SHELF_CLEARANCE_DEFAULT = 0.12
    BLOCK_X_MAX = 0.34  # furthest |x| the blocking hand can hold on the table
    BLOCK_Z_ABOVE_DEFAULT = 0.04
    UPRIGHT_HOLD_DEFAULT = 0.0  # reactive: ball starts moving immediately

    # Scripted roll ends this far from the hand; PhysX resolves the impact.
    HANDOFF_DISTANCE_DEFAULT = 0.10
    # The ball collides as a convex hull, so it slides instead of rolling.
    # Near-zero Coulomb friction emulates rolling resistance; damping is what
    # bleeds off the post-contact motion.
    LIVE_LINEAR_DAMPING_DEFAULT = 0.12
    LIVE_ANGULAR_DAMPING_DEFAULT = 0.3
    LIVE_FRICTION_DEFAULT = 0.006
    LIVE_RESTITUTION_DEFAULT = 0.12
    BALL_MASS = 0.05
    MIN_COUPLE_GAP = 0.055  # m; closer than this the ball is inside the gripper
    SETTLE_SPEED_DEFAULT = 0.025
    # ~0.3 s at 250 Hz. Must be long enough that a ball reversing off the
    # blocker (which passes through near-zero speed) cannot latch "stopped"
    # mid-bounce and report success while it is still moving.
    SETTLE_HOLD_STEPS_DEFAULT = 75
    # Above this the ball is unambiguously moving again, so a latched "stopped"
    # is revoked rather than reported as a stop.
    RESUME_SPEED_MULT = 3.0
    MAX_LIVE_STEPS_DEFAULT = 520
    # Confirmation window once the ball has cleared the physical lip; short,
    # because clearing the lip already means it is going over.
    OFF_TABLE_HOLD_STEPS_DEFAULT = 8
    # Depth below the tabletop that counts as a real vertical drop.
    FALL_Z_DROP_DEFAULT = 0.03

    BALL_X_ABS_MIN = 0.14
    BALL_X_ABS_MAX = 0.30

    # Physical tabletop size used to build the table; the fall test uses these
    # (not the conservative scoring lines) so "fell" means it really left the top.
    TABLE_LENGTH = 1.2   # x
    TABLE_WIDTH = 0.7    # y

    SHELF_WIDTH = 1.20
    SHELF_DEPTH = 0.18
    SHELF_THICK = 0.02
    SHELF_Y = 0.26
    SHELF_Z_ABOVE_TABLE = 0.30

    BALL_ROBOT_IGNORE_BIT = 1 << 21
    BALL_ROBOT_IGNORE_ID = 0x0B41

    PROP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    # +90° about world Z: asset face (−Z) points toward the robot (−Y).
    # (User-facing “−90°” from a side-on default; with PROP_UPRIGHT this is +90°.)
    FACE_ROBOT_Q = np.asarray(
        qmult(euler2quat(0.0, 0.0, np.pi / 2.0, axes="sxyz"), PROP_UPRIGHT_Q),
        dtype=np.float64,
    )
    # Monitor + book: same upright bases, yawed +180° about world Z.
    FACE_AWAY_Q = np.asarray(
        qmult(euler2quat(0.0, 0.0, np.pi, axes="sxyz"), FACE_ROBOT_Q),
        dtype=np.float64,
    )
    BOOK_Q = np.asarray(
        qmult(euler2quat(0.0, 0.0, np.pi, axes="sxyz"), PROP_UPRIGHT_Q),
        dtype=np.float64,
    )
    BALL_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("stop_ball", {})
        self._loaded = False
        self.ball = None
        self._ball_rigid = None
        self._traj = []
        self._traj_step = 0
        # parked | rolling | live | stopped | fallen
        self._ball_state = "parked"
        self._live_steps = 0
        self._settle_steps = 0
        self._off_table_steps = 0
        self._armed = False
        self._arm_in_place = False
        self._handoff_idx = 0
        self._traj_dt = 0.004
        self._fell_off = False
        self._stopped = False
        self._arm_contacted = False
        self._stopper_contacted = False
        self._stopper_placed = False
        self._reset_metric_state()
        self._table_start_idx = 0
        self._roll_start_idx = 0
        self._intercept_idx = 0
        self._fall_off_idx = 0
        self._land_idx = 0
        self._intercept_pos = np.zeros(3)
        self._roll_dir = np.array([0.0, -1.0], dtype=np.float64)
        self._roll_angle = 0.0
        self._exit_edge = "front"  # front | left | right
        self._ball_centre_offset = np.zeros(3, dtype=np.float64)
        self.arm_side = "right"
        self.decor = []
        self._occ_shelf = []
        self._occ_table = []
        self._robot_groups_backup = None
        super().setup_demo(**kwags)
        self._configure_observer_camera()
        # Ball rolls from episode start so policy eval does not wait for play_once.
        self._release_ball()

    # --------------------------------------------------------------- scene
    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Office table + wall with a full-width single shelf."""
        self.arr_v = 1
        self.table_xy_bias = list(table_xy_bias)
        table_height = float(self.office_info["table_height"])
        self.table_z_bias = 0.0

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory = Path("assets/background_texture") / texture_type
            count = len([p for p in directory.iterdir() if p.is_file()])
            wall_id, table_id, floor_id = np.random.randint(0, count, size=3)
            self.wall_texture = f"{texture_type}/{wall_id}"
            self.table_texture = f"{texture_type}/{table_id}"
            self.floor_texture = f"{texture_type}/{floor_id}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture = self.table_texture = self.floor_texture = None

        self.floor_parts = []
        for i, pos in enumerate(([1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0])):
            floor = create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
                is_static=True,
            )
            self.floor_parts.append(floor)

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )
        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=self.TABLE_LENGTH,
            width=self.TABLE_WIDTH,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=self.table_texture,
        )

        shelf_z = table_height + self.SHELF_Z_ABOVE_TABLE
        shelf_y = self.SHELF_Y
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        half_z = 0.5 * self.SHELF_THICK
        self.shelf = create_box(
            self.scene,
            sapien.Pose(p=[0.0, shelf_y, shelf_z]),
            half_size=[half_x, half_y, half_z],
            color=(0.55, 0.42, 0.30),
            name="single_wall_shelf",
            is_static=True,
        )
        for bx in (-0.45, 0.0, 0.45):
            create_box(
                self.scene,
                sapien.Pose(p=[bx, shelf_y + half_y - 0.01, shelf_z - 0.06]),
                half_size=[0.012, 0.01, 0.06],
                color=(0.35, 0.35, 0.35),
                name="shelf_bracket",
                is_static=True,
            )

        self.office_info["furn_x_v"]["shelf"] = [0.0, 0.0, 0.0]
        self.office_info["shelf_area"] = [self.SHELF_WIDTH, self.SHELF_DEPTH]
        self.office_info["shelf_heights"] = [shelf_z + half_z]
        self.office_info["shelf_lims"] = [
            -half_x, shelf_y - half_y, half_x, shelf_y + half_y,
        ]
        self.shelf_lims = list(self.office_info["shelf_lims"])
        self.prohibited_area.append([
            self.shelf_lims[0] - 0.02,
            self.shelf_lims[1] - 0.02,
            self.shelf_lims[2] + 0.02,
            self.shelf_lims[3] + 0.02,
        ])
        self.cabinet = None
        self.file_holder = None
        self.wooden_box = None

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.50, 0.55, 1.55], dtype=np.float64)
        look_at = np.array([0.0, 0.05, 0.95], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ------------------------------------------------------------------ helpers
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _make_kinematic(self, entity):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
        except Exception:
            pass
        return rigid

    def _set_entity_pose(self, entity, pose):
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                if rigid.kinematic:
                    rigid.set_kinematic_target(pose)
                    return
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    def _ball_quat(self):
        return self.BALL_Q.copy()

    def _roll_axis(self):
        """Spin axis for rolling along the current heading: z x heading."""
        axis = np.array([-self._roll_dir[1], self._roll_dir[0], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(axis))
        return axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _ball_roll_quat(self, distance):
        """Orientation after rolling `distance` metres along the heading."""
        if abs(float(distance)) < 1e-9:
            return self.BALL_Q.copy()
        angle = float(distance) / max(self.ball_radius, 1e-4)
        q_spin = axangle2quat(self._roll_axis(), angle)
        return np.asarray(qmult(q_spin, self.BALL_Q), dtype=np.float64)

    @staticmethod
    def _resolve_scale(modelname, model_id, scale_mult=1.0, fallback=0.05):
        path = resolve_model_dir(modelname) / f"model_data{int(model_id)}.json"
        data = json.loads(path.read_text())
        base = data.get("scale")
        if not base:
            base = [float(fallback)] * 3
        return [float(s) * float(scale_mult) for s in base]

    @classmethod
    def _ball_radius(cls, scale_mult=2.64):
        path = Path("assets/objects/027_table-tennis/model_data0.json")
        data = json.loads(path.read_text())
        extents = np.asarray(data["extents"], dtype=np.float64)
        diam = float(np.mean(extents) * cls.BALL_BASE_SCALE * float(scale_mult))
        return 0.5 * diam

    def _footprint_ok(self, occ, cx, cy, hx, hy, pad=0.01):
        box = (cx - hx - pad, cy - hy - pad, cx + hx + pad, cy + hy + pad)
        for x0, y0, x1, y1 in occ:
            if not (box[2] <= x0 or box[0] >= x1 or box[3] <= y0 or box[1] >= y1):
                return False
        return True

    def _reserve(self, occ, cx, cy, hx, hy, pad=0.01):
        occ.append((
            cx - hx - pad, cy - hy - pad, cx + hx + pad, cy + hy + pad,
        ))

    def _spawn_static(
        self, modelname, model_id, p, hx, hy, scale_mult=1.0, fallback=0.05,
        surface="table", quat=None, pad=0.02, force=False,
    ):
        cx, cy = float(p[0]), float(p[1])
        occ = self._occ_shelf if surface == "shelf" else self._occ_table
        if not force and not self._footprint_ok(occ, cx, cy, hx, hy, pad=pad):
            return None
        if quat is None:
            q = self.PROP_UPRIGHT_Q.tolist()
        else:
            q = np.asarray(quat, dtype=np.float64).tolist()
        pose = sapien.Pose(np.asarray(p, dtype=np.float64).tolist(), q)
        scale = self._resolve_scale(modelname, model_id, scale_mult, fallback)
        actor = self._create_scaled_static_object(
            modelname, int(model_id), pose, list(scale), collision=False,
        )
        self._reserve(occ, cx, cy, hx, hy, pad=pad)
        self.decor.append(actor)
        return actor

    def _measure_shelf_plate(self):
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        shelf_y = self.SHELF_Y
        z_top = self.table_top + self.SHELF_Z_ABOVE_TABLE + 0.5 * self.SHELF_THICK
        self.shelf_plate_z = float(z_top)
        self.shelf_plate_xlim = (-half_x, half_x)
        self.shelf_plate_ylim = (shelf_y - half_y, shelf_y + half_y)
        self.shelf_lims = [
            self.shelf_plate_xlim[0], self.shelf_plate_ylim[0],
            self.shelf_plate_xlim[1], self.shelf_plate_ylim[1],
        ]

    # --------------------------------------------------------------- trajectory
    def _sample_roll_heading(self):
        """Sample a table-roll heading from −Y (0) toward ±side exits."""
        amax = float(getattr(self, "roll_angle_max", self.ROLL_ANGLE_MAX_DEFAULT))
        # Mix mild front angles with steeper side-exit angles.
        if np.random.rand() < 0.45:
            lo = float(self.SIDE_EXIT_ANGLE_MIN)
            ang = float(np.random.uniform(lo, amax)) * float(np.random.choice([-1.0, 1.0]))
        else:
            ang = float(np.random.uniform(-amax, amax))
        # Unit direction: angle measured from −Y toward +X.
        direction = np.array([np.sin(ang), -np.cos(ang)], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        return float(ang), direction

    @staticmethod
    def _heading_dir(ang):
        return np.array([np.sin(ang), -np.cos(ang)], dtype=np.float64)

    def _feasible_angle(self, ang, x0, y0, y_lip, t_fall):
        """Shrink a heading until the arm can meet the roll before it exits.

        A steep heading crosses the arm's reach limit in x while the ball is
        still tucked under the shelf, where the hand cannot go, so the ball
        would leave by a side edge untouched.
        """
        y_hand = float(self.shelf_front_y) - float(self.catch_shelf_clearance)
        speed_xy = abs(float(self.fall_speed_xy))
        reach = float(self.BLOCK_X_MAX) - 0.03
        for k in range(9):
            a = float(ang) * (1.0 - 0.12 * k)
            vx, vy = self._heading_dir(a)
            shelf_dx = float(np.clip(0.35 * vx * abs(y0 - y_lip), -0.08, 0.08))
            x_drop = float(np.clip(x0 + shelf_dx, -0.50, 0.50))
            x_land = x_drop + vx * speed_xy * t_fall
            y_land = y_lip + vy * speed_xy * t_fall
            dy = y_land - y_hand
            if dy <= 0.0:
                return a
            x_at_hand = x_land + (vx / max(abs(vy), 1e-6)) * dy
            if abs(x_at_hand) <= reach:
                return a
        return 0.0

    def _build_trajectory(self):
        """Shelf roll → free-fall → angled table roll toward a table edge."""
        dt = float(self.scene.get_timestep())
        self._traj_dt = dt
        g = self.GRAVITY
        r = self.ball_radius
        v = self.roll_speed
        traj = []
        traveled = 0.0

        def append(x, y, z, roll_s=0.0):
            q = self._ball_roll_quat(roll_s)
            traj.append((
                np.array([x, y, z], dtype=np.float64),
                np.asarray(q, dtype=np.float64),
            ))

        x0 = float(self.ball_start[0])
        y0 = float(self.ball_start[1])
        z_shelf = float(self.shelf_z_surf + r)
        y_shelf_edge = float(self.shelf_front_y)
        z_table = self.table_top + r
        x_edge = float(self.table_x_edge)
        y_edge = float(self.table_edge_y)

        ang, direction = self._sample_roll_heading()
        # Bias heading toward the ball's half so the matching arm can reach.
        if abs(ang) > 0.15 and np.sign(ang) != np.sign(x0) and abs(x0) > 0.05:
            ang = -ang
        t_fall_est = float(np.sqrt(
            2.0 * max(1e-4, (self.shelf_z_surf + r) - (self.table_top + r)) / g,
        ))
        ang = self._feasible_angle(ang, x0, y0, y_shelf_edge, t_fall_est)
        direction = self._heading_dir(ang)
        self._roll_angle = float(ang)
        self._roll_dir = direction.copy()
        vx, vy = float(direction[0]), float(direction[1])

        # Shelf: roll to front lip with a little lateral drift matching heading.
        shelf_t = max(0.20, float(self.shelf_roll_duration))
        n_shelf = max(1, int(round(shelf_t / dt)))
        # Lateral drift on shelf stays modest so the ball still leaves the front lip.
        shelf_dx = float(np.clip(0.35 * vx * abs(y0 - y_shelf_edge), -0.08, 0.08))
        x_drop = float(np.clip(x0 + shelf_dx, -0.50, 0.50))
        for i in range(1, n_shelf + 1):
            frac = i / n_shelf
            s = frac * frac * (3.0 - 2.0 * frac)
            x = x0 + s * (x_drop - x0)
            y = y0 + s * (y_shelf_edge - y0)
            traveled += abs(y_shelf_edge - y0) / n_shelf
            append(x, y, z_shelf, roll_s=traveled)
        self._drop_idx = len(traj) - 1

        # Free-fall with the same horizontal heading.
        speed_xy = abs(float(self.fall_speed_xy))
        dz = max(1e-4, z_shelf - z_table)
        t_fall = float(np.sqrt(2.0 * dz / g))
        n_fall = max(1, int(round(t_fall / dt)))
        x_land = x_drop + vx * speed_xy * t_fall
        y_land = y_shelf_edge + vy * speed_xy * t_fall
        for i in range(1, n_fall + 1):
            t = (i / n_fall) * t_fall
            x = x_drop + vx * speed_xy * t
            y = y_shelf_edge + vy * speed_xy * t
            z = z_shelf - 0.5 * g * t * t
            traveled += speed_xy * (t_fall / n_fall)
            append(x, y, max(z, z_table), roll_s=traveled)
        append(x_land, y_land, z_table, roll_s=traveled)
        self._land_y = float(y_land)
        self._land_idx = len(traj) - 1

        # Table roll along heading until a table edge is crossed.
        table_start_idx = len(traj)
        self._roll_start_idx = int(table_start_idx)
        x, y = float(x_land), float(y_land)
        max_table_steps = max(1, int(round(1.8 / max(v, 1e-4) / dt)))
        exit_edge = "front"
        for _ in range(max_table_steps):
            x += vx * v * dt
            y += vy * v * dt
            traveled += v * dt
            append(x, y, z_table, roll_s=traveled)
            if y <= y_edge:
                exit_edge = "front"
                break
            if x <= -x_edge:
                exit_edge = "left"
                break
            if x >= x_edge:
                exit_edge = "right"
                break
        self._exit_edge = exit_edge
        n_table = max(1, len(traj) - table_start_idx)
        intercept_i = table_start_idx + int(round(
            self.intercept_frac * max(0, n_table - 1),
        ))
        intercept_i = int(np.clip(intercept_i, table_start_idx, len(traj) - 1))
        # Snap to the nearest point the hand can actually occupy: inside the
        # arm's x reach, clear of the shelf overhang, short of the front lip.
        y_hand = float(self.shelf_front_y) - float(self.catch_shelf_clearance)
        feasible = [
            j for j in range(table_start_idx, min(len(traj), table_start_idx + n_table))
            if abs(float(traj[j][0][0])) <= self.BLOCK_X_MAX
            and y_edge + 0.06 <= float(traj[j][0][1]) <= y_hand
        ]
        if feasible:
            intercept_i = min(feasible, key=lambda j: abs(j - intercept_i))
        self._table_start_idx = int(table_start_idx)
        self._intercept_idx = int(intercept_i)
        self._intercept_pos = traj[self._intercept_idx][0].copy()

        # Continue past the edge so a miss is visible.
        n_off = max(1, int(round(0.55 / dt)))
        x_e, y_e = float(traj[-1][0][0]), float(traj[-1][0][1])
        for i in range(1, n_off + 1):
            t = i * dt
            x = x_e + vx * speed_xy * t
            y = y_e + vy * speed_xy * t
            z = z_table - 0.5 * g * t * t
            traveled += speed_xy * dt
            append(x, y, z, roll_s=traveled)

        self._traj = traj
        self._fall_off_idx = table_start_idx + n_table
        self._update_handoff_idx()

    def _update_handoff_idx(self):
        """Trajectory index where PhysX takes over, a lead distance before the hand."""
        dt = float(getattr(self, "_traj_dt", 0.004))
        lead = int(self.handoff_distance / max(self.roll_speed * dt, 1e-6))
        self._handoff_idx = max(
            int(self._land_idx) + 1, int(self._intercept_idx) - lead,
        )

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        base_speed = float(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        jitter = float(c.get("roll_speed_jitter_frac", self.ROLL_SPEED_JITTER_FRAC))
        jitter = float(np.clip(jitter, 0.0, 0.95))
        self.roll_speed_mean = float(base_speed)
        self.roll_speed = float(
            np.random.uniform(base_speed * (1.0 - jitter), base_speed * (1.0 + jitter))
        )
        self.fall_speed_xy = float(c.get("fall_speed_xy", self.FALL_SPEED_XY_DEFAULT))
        self.shelf_roll_duration = float(c.get(
            "shelf_roll_duration", self.SHELF_ROLL_DURATION_DEFAULT,
        ))
        self.table_edge_y = float(c.get("table_edge_y", self.TABLE_EDGE_Y_DEFAULT))
        self.table_x_edge = float(c.get("table_x_edge", self.TABLE_X_EDGE))
        self.table_back_y = float(c.get("table_back_y", self.TABLE_BACK_Y_DEFAULT))
        self.roll_angle_max = float(c.get("roll_angle_max", self.ROLL_ANGLE_MAX_DEFAULT))
        self.intercept_frac = float(c.get("intercept_frac", self.INTERCEPT_FRAC_DEFAULT))
        self.catch_shelf_clearance = float(c.get(
            "catch_shelf_clearance", self.CATCH_SHELF_CLEARANCE_DEFAULT,
        ))
        self.block_z_above = float(c.get("block_z_above", self.BLOCK_Z_ABOVE_DEFAULT))
        self.upright_hold = float(c.get("upright_hold", self.UPRIGHT_HOLD_DEFAULT))
        self.handoff_distance = float(c.get(
            "handoff_distance", self.HANDOFF_DISTANCE_DEFAULT,
        ))
        self.live_linear_damping = float(c.get(
            "live_linear_damping", self.LIVE_LINEAR_DAMPING_DEFAULT,
        ))
        self.live_angular_damping = float(c.get(
            "live_angular_damping", self.LIVE_ANGULAR_DAMPING_DEFAULT,
        ))
        self.live_friction = float(c.get("live_friction", self.LIVE_FRICTION_DEFAULT))
        self.live_restitution = float(c.get(
            "live_restitution", self.LIVE_RESTITUTION_DEFAULT,
        ))
        self.settle_speed = float(c.get("settle_speed", self.SETTLE_SPEED_DEFAULT))
        self.settle_hold_steps = int(c.get(
            "settle_hold_steps", self.SETTLE_HOLD_STEPS_DEFAULT,
        ))
        self.max_live_steps = int(c.get("max_live_steps", self.MAX_LIVE_STEPS_DEFAULT))
        self.off_table_hold_steps = int(c.get(
            "off_table_hold_steps", self.OFF_TABLE_HOLD_STEPS_DEFAULT,
        ))
        self.fall_z_drop = float(c.get("fall_z_drop", self.FALL_Z_DROP_DEFAULT))
        self.ball_scale_mult = float(c.get(
            "ball_scale_mult", self.BALL_SCALE_MULT_DEFAULT,
        ))
        self.trophy_scale_mult = float(c.get("trophy_scale_mult", self.TROPHY_SCALE_MULT))
        self.screen_scale_mult = float(c.get("screen_scale_mult", self.SCREEN_SCALE_MULT))

        ball_ids = c.get("ball_ids", self.BALL_IDS)
        try:
            ball_ids = [int(x) for x in ball_ids]
        except Exception:
            ball_ids = list(self.BALL_IDS)
        if "ball_id" in c:
            self.ball_id = int(c["ball_id"])
        else:
            self.ball_id = int(np.random.choice(ball_ids or self.BALL_IDS))

        self.ball_radius = float(c.get(
            "ball_radius", self._ball_radius(self.ball_scale_mult),
        ))

        self.table_top = 0.74 + float(self.table_z_bias)
        self._measure_shelf_plate()
        self.shelf_front_y = float(self.shelf_plate_ylim[0])
        self.shelf_back_y = float(self.shelf_plate_ylim[1])
        self.shelf_z_surf = float(self.shelf_plate_z)

        # Spawn on a random shelf half; arm is chosen after the angled traj is known.
        arm_cfg = str(c.get("arm_side", "random")).lower()
        spawn_side = arm_cfg if arm_cfg in ("left", "right") else (
            "left" if int(np.random.randint(0, 2)) == 0 else "right"
        )
        sign = -1.0 if spawn_side == "left" else 1.0

        x0, x1 = self.shelf_plate_xlim
        y0, y1 = self.shelf_plate_ylim
        ball_x = float(np.clip(
            sign * np.random.uniform(self.BALL_X_ABS_MIN, self.BALL_X_ABS_MAX),
            x0 + 0.06,
            x1 - 0.06,
        ))
        # Prefer mid/back of shelf so there is a clear front opening to the lip.
        ball_y = float(np.random.uniform(
            y0 + 0.40 * self.SHELF_DEPTH,
            y1 - 0.02,
        ))
        ball_z = self.shelf_z_surf + self.ball_radius
        self.ball_start = np.array([ball_x, ball_y, ball_z], dtype=np.float64)

        self.decor = []
        self._occ_shelf = []
        self._occ_table = []
        # Ball slot + front corridor on shelf (opening toward the lip).
        self._reserve(self._occ_shelf, ball_x, ball_y, 0.05, 0.05, pad=0.02)
        mid_y = 0.5 * (ball_y + self.shelf_front_y)
        self._reserve(
            self._occ_shelf, ball_x, mid_y,
            0.08, 0.5 * abs(ball_y - self.shelf_front_y) + 0.02, pad=0.02,
        )

        self.ball, self.ball_radius = self._create_ball_actor(
            self.ball_start, self._ball_quat(),
            self.ball_id, self.ball_scale_mult,
        )
        self._ball_rigid = self._make_kinematic(self.ball)
        self._decouple_ball_from_robot()
        self.ball_radius, self._ball_centre_offset = self._measure_ball_hull(
            self.ball_radius,
        )
        self._make_ball_uniform()
        self.ball_start[2] = self.shelf_z_surf + self.ball_radius
        self._set_entity_pose(
            self.ball, self._ball_pose_for(self.ball_start, self._ball_quat()),
        )

        # Build angled traj first, then pick the arm from the intercept x.
        self._build_trajectory()
        ix = float(self._intercept_pos[0])
        if arm_cfg in ("left", "right"):
            self.arm_side = arm_cfg
        else:
            self.arm_side = "left" if ix < 0.0 else "right"
        sign = -1.0 if self.arm_side == "left" else 1.0

        # Reserve an open corridor along the predicted table path (angled).
        self._reserve_roll_corridor()
        self._load_decorations(ball_x, sign)

        self._traj_step = 0
        self._ball_state = "parked"
        self._fell_off = False
        self._stopped = False
        self._arm_contacted = False
        self._stopper_contacted = False
        self._stopper_placed = False
        self._reset_metric_state()
        self._settle_steps = 0
        self._off_table_steps = 0
        self._live_steps = 0
        self._armed = False
        self._arm_in_place = False
        self._loaded = True

    def _reserve_roll_corridor(self):
        """Keep décor completely off the predicted table roll path.

        Dense, wide AABBs so props cannot sit on the lane (visually or as a
        physical blocker if collision is ever enabled).
        """
        if not self._traj:
            return
        i0 = int(self._land_idx)
        i1 = max(int(self._fall_off_idx), i0 + 1)
        n = max(i1 - i0, 1)
        step = max(1, n // 28)
        for i in range(i0, i1 + 1, step):
            p = self._traj[int(i)][0]
            self._reserve(
                self._occ_table, float(p[0]), float(p[1]),
                0.11, 0.11, pad=0.045,
            )
        # Extra clearance around the intended intercept (arm owns this space).
        ip = self._intercept_pos
        self._reserve(
            self._occ_table, float(ip[0]), float(ip[1]),
            0.16, 0.16, pad=0.05,
        )

    def _try_place_candidates(self, candidates, surface, n_attempts=10):
        """Place décor from a shuffled candidate list; skip overlaps."""
        order = list(candidates)
        np.random.shuffle(order)
        for item in order:
            (modelname, ids, hx, hy, scale_mult, fallback, pos_fn,
             quat, pad) = item
            for _ in range(int(n_attempts)):
                mid = int(np.random.choice(list(ids)))
                p = pos_fn()
                if self._spawn_static(
                    modelname, mid, p, hx=hx, hy=hy,
                    scale_mult=scale_mult, fallback=fallback, surface=surface,
                    quat=quat, pad=pad,
                ) is not None:
                    break

    def _load_decorations(self, ball_x, sign):
        z_shelf = self.shelf_z_surf
        y0, y1 = self.shelf_plate_ylim
        x0, x1 = self.shelf_plate_xlim
        face = self.FACE_ROBOT_Q

        # Plant on the far opposite end from the ball.
        plant_x = float(np.clip(-sign * 0.52, x0 + 0.07, x1 - 0.07))
        plant_y = float(0.5 * (y0 + y1))
        self._spawn_static(
            "120_plant", 0,
            [plant_x, plant_y, z_shelf + 0.01],
            hx=0.07, hy=0.07, scale_mult=0.48, surface="shelf", pad=0.03,
        )

        # Discrete x-slots away from ball corridor + plant; stagger front/back.
        def free_slots(min_gap=0.18):
            xs = [-0.45, -0.25, 0.0, 0.25, 0.45]
            return [x for x in xs
                    if abs(x - ball_x) >= min_gap and abs(x - plant_x) >= min_gap]

        slots = free_slots()
        np.random.shuffle(slots)

        def pop_pose(y_bias, prefer_far=True):
            if not slots:
                return None
            # Prefer slots farthest from the ball so décor spreads out.
            if prefer_far:
                slots.sort(key=lambda x: -abs(x - ball_x))
            sx = float(slots.pop(0))
            if y_bias == "front":
                sy = float(y0 + 0.30 * self.SHELF_DEPTH)
            else:
                sy = float(y1 - 0.30 * self.SHELF_DEPTH)
            return [sx, sy, z_shelf + 0.01]

        # Desk-sized trophy, then clock (faces robot), globe, rest.
        for modelname, ids, hx, hy, sm, fb, ybias, quat, pad in [
            ("090_trophy", self.TROPHY_IDS, 0.08, 0.08, self.trophy_scale_mult, 0.05,
             "back", None, 0.02),
            ("046_alarm-clock", self.ALARM_IDS, 0.07, 0.055, 0.60, 0.05, "front", face, 0.02),
            ("089_globe", self.GLOBE_IDS, 0.08, 0.08, 0.50, 0.05, "front", None, 0.02),
            ("094_rest", self.REST_IDS, 0.08, 0.04, 0.65, 0.05, "back", None, 0.02),
        ]:
            p = pop_pose(ybias)
            if p is None:
                continue
            mid = int(ids[0]) if modelname == "090_trophy" else int(np.random.choice(list(ids)))
            if self._spawn_static(
                modelname, mid, p,
                hx=hx, hy=hy, scale_mult=sm, fallback=fb, surface="shelf",
                quat=quat, pad=pad,
            ) is None:
                # Last-chance place without footprint veto (still skip ball slot).
                self._spawn_static(
                    modelname, mid, p,
                    hx=hx * 0.7, hy=hy * 0.7, scale_mult=sm, fallback=fb,
                    surface="shelf", quat=quat, pad=0.005, force=True,
                )

        # ---- table décor: never force onto the reserved roll corridor ----
        def place_table(modelname, ids, preferred, hx, hy, sm, fb=0.05,
                        quat=None, pad=0.02, n_try=24, jitter=(0.10, 0.08)):
            z = float(preferred[2])
            for _ in range(int(n_try)):
                pj = [
                    float(np.clip(
                        preferred[0] + np.random.uniform(-jitter[0], jitter[0]),
                        -0.52, 0.52,
                    )),
                    float(np.clip(
                        preferred[1] + np.random.uniform(-jitter[1], jitter[1]),
                        -0.28, 0.16,
                    )),
                    z,
                ]
                if self._spawn_static(
                    modelname, int(np.random.choice(list(ids))), pj,
                    hx=hx, hy=hy, scale_mult=sm, fallback=fb, surface="table",
                    quat=quat, pad=pad,
                ) is not None:
                    return True
            return False  # skip rather than block the ball lane

        opp = -float(sign)
        place_table(
            "023_tissue-box", self.TISSUE_IDS,
            [opp * 0.46, -0.25, self.table_top + 0.005],
            0.055, 0.045, 0.80, pad=0.02,
        )
        for modelname, ids, hx, hy, sm, fb, p, quat in [
            ("099_fan", self.FAN_IDS, 0.06, 0.06, 0.85, 0.05,
             [opp * 0.48, -0.05, self.table_top + 0.005], None),
            ("043_book", self.BOOK_IDS, 0.08, 0.06, 0.75, 0.05,
             [opp * 0.28, -0.25, self.table_top + 0.005], self.BOOK_Q),
            ("100_seal", self.SEAL_IDS, 0.045, 0.045, 0.90, 0.05,
             [sign * 0.46, -0.22, self.table_top + 0.005], None),
            ("101_milk-tea", self.MILKTEA_IDS, 0.05, 0.05, 0.70, 0.05,
             [opp * 0.46, 0.05, self.table_top + 0.005], None),
        ]:
            place_table(modelname, ids, p, hx, hy, sm, fb=fb, quat=quat, pad=0.02)

        # Screen + speaker under shelf, pushed off the roll lane (no force).
        under_y = 0.14
        screen_x = float(np.clip(opp * 0.40, -0.48, 0.48))
        place_table(
            "097_screen", self.SCREEN_IDS,
            [screen_x, under_y, self.table_top + 0.01],
            0.11, 0.05, self.screen_scale_mult, quat=self.FACE_AWAY_Q, pad=0.02,
            n_try=30, jitter=(0.18, 0.04),
        )
        place_table(
            "098_speaker", self.SPEAKER_IDS,
            [float(np.clip(screen_x - sign * 0.26, -0.50, 0.50)),
             under_y, self.table_top + 0.01],
            0.055, 0.055, 1.0, fb=0.06, quat=face, pad=0.02,
            n_try=30, jitter=(0.18, 0.04),
        )

    def _decouple_ball_from_robot(self):
        """Ignore robot↔ball contacts during the scripted roll (hand is waiting)."""
        if self._ball_rigid is None:
            return
        bit = int(self.BALL_ROBOT_IGNORE_BIT)
        ident = int(self.BALL_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for shape in self._ball_rigid.get_collision_shapes():
                g0, g1, _, _ = shape.get_collision_groups()
                shape.set_collision_groups([int(g0), int(g1), bit, ident])
            backup = []
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    for shape in link.get_collision_shapes():
                        g = list(shape.get_collision_groups())
                        backup.append((shape, g[:]))
                        shape.set_collision_groups([
                            int(g[0]), int(g[1]), int(g[2]) | bit,
                            (int(g[3]) & ~0xFFFF) | ident,
                        ])
            self._robot_groups_backup = backup
        except Exception:
            self._robot_groups_backup = None

    def _recouple_ball_to_robot(self):
        """Restore contacts so the gripper can physically pinch the ball."""
        if self._ball_rigid is not None:
            try:
                for shape in self._ball_rigid.get_collision_shapes():
                    shape.set_collision_groups([1, 1, 0, 0])
            except Exception:
                pass
        if self._robot_groups_backup:
            try:
                for shape, g in self._robot_groups_backup:
                    shape.set_collision_groups([int(x) for x in g])
            except Exception:
                pass
            self._robot_groups_backup = None

    def _measure_ball_hull(self, fallback_r):
        """Radius and origin→centre offset of the real hull.

        The asset's origin is not its centre, so spinning it about the origin
        would swing the ball around instead of rolling it. Measured with the
        ball at identity orientation, hence the offset is already local.
        """
        r = float(fallback_r)
        offset = np.zeros(3, dtype=np.float64)
        if self._ball_rigid is None:
            return r, offset
        try:
            aabb = np.asarray(
                self._ball_rigid.compute_global_aabb_tight(), dtype=np.float64,
            )
            extent = np.asarray(aabb[1]) - np.asarray(aabb[0])
            if float(np.min(extent)) > 1e-4:
                r = float(np.mean(extent)) * 0.5
                centre = 0.5 * (aabb[0] + aabb[1])
                offset = centre - np.array(self.ball.get_pose().p, dtype=np.float64)
        except Exception:
            pass
        return r, offset

    def _create_ball_actor(self, centre, quat, model_id, scale_mult):
        """Table-tennis ball with an analytic sphere collider.

        The shipped convex hull is faceted, so it tumbles and stalls instead of
        rolling, and its origin sits ~3 cm off the geometry. Only the visual
        mesh is reused; the visual is recentred on the actor origin.
        """
        modeldir = Path("assets/objects/027_table-tennis")
        visual = modeldir / "visual" / f"base{int(model_id)}.glb"
        if not visual.exists():
            visual = modeldir / f"base{int(model_id)}.glb"
        data = json.loads((modeldir / f"model_data{int(model_id)}.json").read_text())
        scale = float(self.BALL_BASE_SCALE) * float(scale_mult)
        radius = 0.5 * float(np.mean(data["extents"])) * scale
        mesh_centre = np.asarray(data["center"], dtype=np.float64) * scale

        material = sapien.physx.PhysxMaterial(
            static_friction=self.live_friction * 1.5,
            dynamic_friction=self.live_friction,
            restitution=self.live_restitution,
        )
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_sphere_collision(radius=radius, material=material)
        builder.add_visual_from_file(
            filename=str(visual),
            scale=[scale] * 3,
            pose=sapien.Pose((-mesh_centre).tolist()),
        )
        builder.set_initial_pose(sapien.Pose(
            np.asarray(centre, dtype=np.float64).tolist(),
            np.asarray(quat, dtype=np.float64).tolist(),
        ))
        entity = builder.build(name="table_tennis_ball")
        return Actor(entity, None, mass=self.BALL_MASS), radius

    def _make_ball_uniform(self):
        """Put the centre of mass on the hull centre with a solid-sphere inertia.

        The asset's origin is offset from its geometry, so the default mass
        properties make the ball rock like a weeble instead of rolling.
        """
        if self._ball_rigid is None:
            return
        m = float(self.BALL_MASS)
        inertia = 0.4 * m * self.ball_radius ** 2
        try:
            self._ball_rigid.auto_compute_mass = False
        except Exception:
            pass
        try:
            self._ball_rigid.set_mass(m)
            self._ball_rigid.set_cmass_local_pose(
                sapien.Pose(self._ball_centre_offset.tolist()),
            )
            self._ball_rigid.set_inertia([inertia] * 3)
        except Exception:
            pass

    def _hull_offset_world(self, quat):
        q = np.asarray(quat, dtype=np.float64)
        return quat2mat(q) @ self._ball_centre_offset

    def _ball_pose_for(self, centre, quat):
        """Pose that puts the ball's *centre* on `centre` at orientation `quat`."""
        p = np.asarray(centre, dtype=np.float64) - self._hull_offset_world(quat)
        return sapien.Pose(p.tolist(), np.asarray(quat, dtype=np.float64).tolist())

    def _ball_centre(self):
        pose = self.ball.get_pose()
        return np.array(pose.p, dtype=np.float64) + self._hull_offset_world(pose.q)

    def _tune_ball_friction(self, static_f=None, dynamic_f=None, restitution=None):
        """Glide like a rolling ball, absorb the gripper hit without rattling."""
        if self._ball_rigid is None:
            return
        dynamic_f = self.live_friction if dynamic_f is None else dynamic_f
        static_f = dynamic_f * 1.5 if static_f is None else static_f
        restitution = self.live_restitution if restitution is None else restitution
        try:
            for s in self._ball_rigid.get_collision_shapes():
                m = getattr(s, "physical_material", None)
                if m is None:
                    continue
                m.set_static_friction(float(static_f))
                m.set_dynamic_friction(float(dynamic_f))
                m.set_restitution(float(restitution))
        except Exception:
            pass

    # ----------------------------------------------------------- kinematics
    def _release_ball(self):
        if self._ball_state != "parked":
            return
        self._ball_state = "rolling"
        if self._metric_release_step is None:
            self._metric_release_step = self._metric_step()
        self._traj_step = 0
        self._armed = True

    def _is_interactive(self):
        """True for viewer teleop (robot or keyboard), not scripted collect."""
        return bool(getattr(self, "_interactive_session", False)) or bool(
            getattr(self, "_interactive_robot_mode", False)
        )

    def _near_any_arm(self, dist=0.09):
        """True if the ball centre is within `dist` (xy) of either TCP."""
        for side in ("left", "right"):
            try:
                if float(np.linalg.norm(self._ball_to_tcp(side)[:2])) <= float(dist):
                    return True
            except Exception:
                continue
        return False

    def _try_recouple_ball(self):
        """Restore robot↔ball contacts.

        Expert: wait until the hand is clear of the sphere so the approach
        sweep cannot swat the ball. Interactive: always restore — the gap gate
        was leaving contacts off while the user held the hand on the path, so
        the (still decoupled) ball tunneled straight through the arm.
        """
        if self._ball_rigid is None or self._robot_groups_backup is None:
            return
        if self._is_interactive():
            self._recouple_ball_to_robot()
            return
        try:
            gap = float(np.linalg.norm(self._ball_to_tcp(self.arm_side)[:2]))
        except Exception:
            return
        if gap >= self.MIN_COUPLE_GAP:
            self._recouple_ball_to_robot()

    def _go_live(self):
        """Hand the rolling ball to PhysX just before it reaches the gripper."""
        if self._ball_state != "rolling" or self._ball_rigid is None:
            return
        # Expert: contacts stay off until the hand has stopped and is clear,
        # otherwise the approach sweep swats the ball. Interactive: always
        # enable contacts so teleop can physically block the ball.
        if self._is_interactive():
            self._recouple_ball_to_robot()
        elif self._arm_in_place:
            self._try_recouple_ball()
        v = float(self.roll_speed)
        direction = np.array(
            [float(self._roll_dir[0]), float(self._roll_dir[1]), 0.0], dtype=np.float64,
        )
        spin_axis = self._roll_axis()
        try:
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_damping(self.live_linear_damping)
            self._ball_rigid.set_angular_damping(self.live_angular_damping)
            self._ball_rigid.set_linear_velocity(direction * v)
            self._ball_rigid.set_angular_velocity(
                spin_axis * (v / max(self.ball_radius, 1e-4)),
            )
        except Exception:
            pass
        self._tune_ball_friction()
        self._ball_state = "live"
        self._live_steps = 0
        self._latch_table_entry()

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_release_step = None  # ball let go at the top of the shelf
        self._metric_live_step = None    # ball landed on the table, PhysX took over
        self._metric_entry_margin = None # distance to the exit edge at that landing
        self._metric_contact_step = None # arm first touched the ball
        self._metric_contact_margin = None
        self._metric_stop_step = None    # ball confirmed stopped

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _edge_margin(self, p):
        """Distance from the ball centre to the edge it is heading for, in metres."""
        edge = str(getattr(self, "_exit_edge", "front"))
        if edge == "left":
            return float(p[0]) + float(self.table_x_edge)
        if edge == "right":
            return float(self.table_x_edge) - float(p[0])
        return float(p[1]) - float(self.table_edge_y)

    def _latch_table_entry(self):
        """Called when the ball goes live on the table — start of the reaction window."""
        try:
            if self._metric_live_step is None:
                self._metric_live_step = self._metric_step()
                self._metric_entry_margin = self._edge_margin(self._ball_centre())
        except Exception:
            pass

    def _latch_contact_metric(self):
        """Called the instant the arm first touches the ball, before it deflects."""
        try:
            if self._metric_contact_step is None:
                self._metric_contact_step = self._metric_step()
                self._metric_contact_margin = self._edge_margin(self._ball_centre())
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `intercept_latency_steps` — steps from the ball being let go at the
        top of the shelf until the arm first touches it: the full reaction window a
        human sees. `table_intercept_latency_steps` is the same clock started when
        the ball actually lands on the table and PhysX takes over.
        extra2 `edge_margin_norm` — table left between the ball and the edge it was
        heading for at the moment of contact, over the margin it had on landing.
        HIGHER is better: 1.0 = intercepted the instant it landed, 0 = caught it right
        at the lip. `stop_margin_norm` is the same measure at the final resting pose.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_release_step, self._metric_contact_step
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["intercept_latency_steps"] = lat
        out["intercept_latency_s"] = None if lat is None else round(float(lat) * dt, 4)
        live = self._metric_live_step
        tbl = None if (live is None or b is None) else max(int(b) - int(live), 0)
        out["table_intercept_latency_steps"] = tbl
        c = self._metric_stop_step
        settle = None if (b is None or c is None) else max(int(c) - int(b), 0)
        out["settle_latency_steps"] = settle

        d0 = self._metric_entry_margin
        d1 = self._metric_contact_margin
        ok = d0 is not None and float(d0) > 1e-6
        out["edge_margin_norm"] = (
            None if (not ok or d1 is None) else round(float(d1) / float(d0), 4))
        out["edge_margin_m"] = None if d1 is None else round(float(d1), 4)
        try:
            rest = self._edge_margin(self._ball_centre())
            out["stop_margin_norm"] = (
                None if not ok else round(float(rest) / float(d0), 4))
        except Exception:
            out["stop_margin_norm"] = None
        out["exit_edge"] = str(getattr(self, "_exit_edge", "front"))
        out["arm_contacted"] = bool(getattr(self, "_arm_contacted", False))
        out["fell_off"] = bool(getattr(self, "_fell_off", False))
        return out

    def _table_lip(self):
        """(half_x, front_y) of the physical tabletop, including table bias."""
        bias = list(getattr(self, "table_xy_bias", [0.0, 0.0]) or [0.0, 0.0])
        half_x = 0.5 * float(self.TABLE_LENGTH) + abs(float(bias[0]))
        front_y = -0.5 * float(self.TABLE_WIDTH) + float(bias[1])
        return half_x, front_y

    def _ball_dropped(self, p):
        """True only for a real vertical drop below the tabletop."""
        return bool(float(p[2]) < self.table_top - float(self.fall_z_drop))

    def _ball_past_edge_now(self, p):
        """Ball centre has cleared the physical lip, so it must go over."""
        half_x, front_y = self._table_lip()
        return bool(float(p[1]) <= front_y or abs(float(p[0])) >= half_x)

    def _ball_off_table(self, p):
        return bool(self._ball_dropped(p) or self._ball_past_edge_now(p))

    def _confirm_fall(self, p):
        """Latch a fall only for a real drop or a *sustained* off-table exit.

        A single bounce past the edge line (e.g. off the bridge / gripper) that
        the ball then recovers from must not end the episode as a failure.
        """
        if self._ball_dropped(p):
            self._off_table_steps = 0
            return True
        if self._ball_past_edge_now(p):
            self._off_table_steps += 1
            return self._off_table_steps >= int(self.off_table_hold_steps)
        self._off_table_steps = 0
        return False

    def _ball_speed(self):
        if self._ball_rigid is None:
            return 0.0
        try:
            return float(np.linalg.norm(
                np.array(self._ball_rigid.get_linear_velocity(), dtype=np.float64),
            ))
        except Exception:
            return 0.0

    def _boost_ball_along_path(self):
        """Re-inject roll velocity so a missed ball keeps going off the table.

        PhysX damping / friction can stall a live ball that never hit the arm;
        without this boost it would rest on the table and look like a stop.
        """
        if self._ball_rigid is None:
            return
        direction = np.array(
            [float(self._roll_dir[0]), float(self._roll_dir[1]), 0.0],
            dtype=np.float64,
        )
        n = float(np.linalg.norm(direction[:2]))
        if n < 1e-6:
            direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        else:
            direction[:2] /= n
        v = float(self.roll_speed)
        spin_axis = self._roll_axis()
        try:
            self._ball_rigid.set_linear_velocity(direction * v)
            self._ball_rigid.set_angular_velocity(
                spin_axis * (v / max(self.ball_radius, 1e-4)),
            )
        except Exception:
            pass

    def _poll_arm_contact(self):
        """Record whether the ball has touched any robot link (PhysX contacts)."""
        if self._arm_contacted or self.ball is None:
            return
        try:
            ball_name = self.ball.get_name()
            robot_links = set()
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    robot_links.add(link.get_name())
            for contact in self.scene.get_contacts():
                n0 = contact.bodies[0].entity.name
                n1 = contact.bodies[1].entity.name
                if (n0 == ball_name and n1 in robot_links) or (
                    n1 == ball_name and n0 in robot_links
                ):
                    self._arm_contacted = True
                    self._latch_contact_metric()
                    return
        except Exception:
            pass
        # Fallback: gripper almost touching the ball centre counts as a hit.
        # Used when contact reporting is sparse for the sphere collider.
        try:
            gap = float(np.linalg.norm(self._ball_to_tcp(self.arm_side)[:3]))
            if gap <= (self.ball_radius + 0.028):
                self._arm_contacted = True
                self._latch_contact_metric()
        except Exception:
            pass

    def _advance_ball(self):
        if not self._loaded or self.ball is None:
            return
        if self._ball_state == "stopped":
            # A settled ball can still be nudged off the table afterwards.
            if self._confirm_fall(self._ball_centre()):
                self._fell_off = True
                self._stopped = False
                self._ball_state = "fallen"
            elif self._ball_speed() > self.settle_speed * float(self.RESUME_SPEED_MULT):
                # Clearly rolling again — it was not a stop after all.
                self._stopped = False
                self._settle_steps = 0
                self._ball_state = "live"
            return
        if self._ball_state == "live":
            # PhysX owns the ball now — watch contacts and whether it stays up.
            self._live_steps += 1
            if self._is_interactive() and self._robot_groups_backup is not None:
                self._recouple_ball_to_robot()
            self._poll_arm_contact()
            blocker_contacted = bool(self._arm_contacted or self._stopper_contacted)
            p = self._ball_centre()
            if self._confirm_fall(p):
                self._fell_off = True
                self._ball_state = "fallen"
            elif self._ball_speed() <= self.settle_speed:
                # Robot mode uses an arm; keyboard mode uses the placed bridge.
                # A miss must keep rolling off. A ball already hanging past the
                # edge must never settle: the fall has to win that race, or the
                # episode latches "stopped" out of bounds and never terminates.
                if blocker_contacted and not self._ball_past_edge_now(p):
                    self._settle_steps += 1
                    if self._settle_steps >= self.settle_hold_steps:
                        self._ball_state = "stopped"
                        self._stopped = True
                        if self._metric_stop_step is None:
                            self._metric_stop_step = self._metric_step()
                else:
                    self._settle_steps = 0
                    # Never boost a contacted ball or one overlapping a hand —
                    # that would drive it through the blocker.
                    if (
                        not blocker_contacted
                        and not self._near_any_arm(dist=max(0.09, float(self.ball_radius) + 0.04))
                    ):
                        self._boost_ball_along_path()
            elif (
                not blocker_contacted
                and not self._near_any_arm(dist=max(0.09, float(self.ball_radius) + 0.04))
                and self._ball_speed() < 0.55 * float(self.roll_speed)
            ):
                # Miss far from either hand: top up speed so damping cannot stall it.
                self._boost_ball_along_path()
            else:
                self._settle_steps = 0
            return
        if self._ball_state != "rolling":
            return
        if self._traj_step >= len(self._traj):
            p = self._ball_centre()
            if self._ball_off_table(p):
                self._ball_state = "fallen"
                self._fell_off = True
            elif self._is_interactive() or self._armed:
                self._go_live()
            return
        interactive = self._is_interactive()
        if self._armed and self._traj_step > int(self._land_idx):
            if interactive:
                self._go_live()
                return
            if self._arm_in_place:
                near = float(np.linalg.norm(self._ball_to_tcp(self.arm_side)[:2]))
                if near <= self.handoff_distance or self._traj_step >= self._handoff_idx:
                    self._go_live()
                    return

        pos, quat = self._traj[self._traj_step]
        self._set_entity_pose(self.ball, self._ball_pose_for(pos, quat))
        if self._traj_step >= self._fall_off_idx and not self._stopped:
            off_xy = (
                float(pos[1]) <= self.table_edge_y + 0.01
                or abs(float(pos[0])) >= self.table_x_edge - 0.01
            )
            if off_xy or pos[2] < self.table_top - 0.05:
                self._fell_off = True
                self._ball_state = "fallen"
        self._traj_step += 1

    def _tcp_pose(self, arm_tag):
        pose = (
            self.robot.get_left_tcp_pose() if arm_tag == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.array(pose, dtype=np.float64)

    def _tcp_pos(self, arm_tag):
        return self._tcp_pose(arm_tag)[:3]

    def _ball_to_tcp(self, arm_tag):
        return self._ball_centre() - self._tcp_pos(arm_tag)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_ball()

    def check_stable(self):
        if self.ball is not None and self._ball_state == "parked":
            self._set_entity_pose(
                self.ball, self._ball_pose_for(self.ball_start, self._ball_quat()),
            )
        return super().check_stable()

    def _dwell(self, steps):
        if not hasattr(self, "_pic_counter"):
            self._pic_counter = 0
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            self._pic_counter += 1
            if self.save_freq and (self._pic_counter % self.save_freq == 0):
                self._take_picture()

    # ------------------------------------------------------------- policy
    def _block_tcp(self):
        tcp = self._intercept_pos.copy()
        y_max = min(
            float(self.shelf_front_y) - float(self.catch_shelf_clearance),
            float(self._intercept_pos[1]),
        )
        tcp[1] = min(float(tcp[1]), y_max)
        tcp[1] = max(float(tcp[1]), float(self.table_edge_y) + 0.06)
        # Keep the block pose on the operating arm's reachable half.
        if self.arm_side == "right":
            tcp[0] = float(np.clip(tcp[0], 0.05, self.BLOCK_X_MAX))
        else:
            tcp[0] = float(np.clip(tcp[0], -self.BLOCK_X_MAX, -0.05))
        # Open-finger catch at ball-centre height so pads meet the equator.
        tcp[2] = self.table_top + max(self.ball_radius + 0.01, self.block_z_above)
        return tcp

    def _block_quat(self, arm_tag):
        key = "down_left" if str(arm_tag) == "right" else "down_right"
        return list(GRASP_DIRECTION_DIC[key])

    def _approach_block(self, arm_tag):
        target = self._block_tcp()
        quat = self._block_quat(arm_tag)
        return self.move_to_pose(
            arm_tag, [float(target[0]), float(target[1]), float(target[2])] + quat,
        )

    def _wait_until_handoff(self, max_steps=2500):
        """Run the scripted roll out; _advance_ball flips it to dynamic on time."""
        for _ in range(int(max_steps)):
            if self._ball_state != "rolling" or self._fell_off:
                return self._ball_state == "live"
            if self._traj_step >= self._fall_off_idx:
                return False
            self._dwell(1)
        return False

    def _align_hand_to_line(self, arm_tag):
        """Slide the hand sideways onto the roll line at its own depth.

        The IK rarely lands the hand on the planned intercept, and a few
        centimetres of lateral error is enough for the ball to roll past it.
        Runs before contacts are restored, so the ball cannot be swatted.
        """
        if not self._traj:
            return
        tcp = self._tcp_pos(arm_tag)
        lo = max(int(self._land_idx), int(self._traj_step))
        hi = min(int(self._fall_off_idx), len(self._traj) - 1)
        if lo >= hi:
            return
        j = min(range(lo, hi), key=lambda i: abs(self._traj[i][0][1] - tcp[1]))
        limit = ((0.05, self.BLOCK_X_MAX) if str(arm_tag) == "right"
                 else (-self.BLOCK_X_MAX, -0.05))
        x_line = float(np.clip(self._traj[j][0][0], *limit))
        dx = float(np.clip(x_line - float(tcp[0]), -0.12, 0.12))
        if abs(dx) < 0.012:
            return
        self.move(self.move_by_displacement(arm_tag, x=dx))

    def _settle_live_ball(self):
        """Resolve the PhysX phase: arm stop on the table, or miss → fall off.

        Without an arm hit the ball is kept on its exit heading until it leaves
        the table. Budget expiry never counts as a stop for a miss; after a hit
        we wait for a real settle (or a later fall) instead of freezing early.
        """
        for i in range(max(int(self.max_live_steps), 900)):
            if self._ball_state != "live":
                break
            if not self._arm_contacted and self._ball_speed() < 0.6 * float(self.roll_speed):
                self._boost_ball_along_path()
            self._dwell(1)
        if self._ball_state != "live":
            return
        if not self._arm_contacted:
            # Miss: keep driving the exit heading until the ball leaves the table.
            for _ in range(1200):
                if self._ball_state != "live" or self._fell_off:
                    break
                if self._ball_speed() < 0.8 * float(self.roll_speed):
                    self._boost_ball_along_path()
                self._dwell(1)
            return
        # Hit but still moving: give it time to settle on the table or fall off.
        for _ in range(900):
            if self._ball_state != "live":
                break
            self._dwell(1)
        if (
            self._ball_state == "live"
            and not self._ball_off_table(self._ball_centre())
            and self._ball_speed() <= max(self.settle_speed * 4.0, 0.08)
        ):
            self._ball_state = "stopped"
            self._stopped = True

    def play_once(self):
        arm_tag = ArmTag(self.arm_side)
        self._pic_counter = 0

        # Reactive: ball starts rolling immediately; arm approaches while it moves.
        self._release_ball()
        # Open fingers: ball diameter = WSG gap + 0.5 cm, so the pads catch it.
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self._approach_block(arm_tag))
        self._align_hand_to_line(arm_tag)
        self._arm_in_place = True

        if self._wait_until_handoff():
            self._settle_live_ball()
            self._dwell(15)
            self.plan_success = True
        elif self._ball_state == "rolling":
            # No live handoff (e.g. arm never arrived): finish the scripted path
            # so the ball falls off the table instead of freezing mid-roll.
            while self._traj_step < len(self._traj):
                self._dwell(1)
                if self._fell_off or self._ball_state == "fallen":
                    break
        elif self._ball_state == "live":
            self._settle_live_ball()

        self.info["info"] = {
            "{A}": f"027_table-tennis/base{self.ball_id}",
            "{B}": "single_wall_shelf",
            "{a}": str(arm_tag),
            "{angle}": f"{float(self._roll_angle):.2f}",
            "{edge}": str(self._exit_edge),
        }
        return self.info

    def check_success(self):
        """Success = a valid blocker stopped the ball and it stayed on the table.

        Merely touching the robot arm or keyboard bridge is not enough: a
        deflection that then falls off any edge or drops below the tabletop
        fails. Settling under the shelf / at the back is success. Settling
        without contact with the active control mode's blocker still fails.
        """
        if self._fell_off or self._ball_state == "fallen":
            return False
        blocker_contacted = bool(self._arm_contacted or self._stopper_contacted)
        if not blocker_contacted or self.ball is None:
            return False
        if not self._stopped:
            return False
        p = self._ball_centre()
        on_table = (
            abs(p[2] - (self.table_top + self.ball_radius)) < 0.05
            and self.table_top - 0.01 <= p[2] <= self.table_top + 2.0 * self.ball_radius + 0.04
        )
        # Same lip the fall test uses, so a ball resting near the edge is either
        # a success or a fall — never stuck between the two with no verdict.
        half_x, front_y = self._table_lip()
        in_xy = (
            abs(p[0]) < half_x
            and front_y < p[1] <= self.table_back_y + 0.02
        )
        # Settle-hold already happened (`_stopped`); residual crawl after an
        # angled deflection is fine so long as the ball stays on the table.
        return bool(on_table and in_xy)

    def get_score(self) -> float:
        """Partial score for stop_ball.

        Stopped on table after arm/bridge contact → 1.
        Contacted the gripper (or keyboard bridge) but still fell off → 0.5.
        No contact → 0. No other partial scores.
        """
        if self.ball is None:
            return 0.0
        if self.check_success():
            return 1.0
        contacted = bool(
            getattr(self, "_arm_contacted", False)
            or getattr(self, "_stopper_contacted", False)
        )
        if not contacted:
            return 0.0
        fell = bool(getattr(self, "_fell_off", False)) or self._ball_state == "fallen"
        if fell or self._ball_off_table(self._ball_centre()):
            return 0.5
        return 0.0

    def get_obs(self):
        obs = super().get_obs()
        obs["stop_ball"] = {
            "ball_state": str(self._ball_state),
            "fell_off": bool(self._fell_off),
            "stopped": bool(self._stopped),
            "arm_contacted": bool(self._arm_contacted),
            "stopper_placed": bool(self._stopper_placed),
            "stopper_contacted": bool(self._stopper_contacted),
            "traj_step": int(self._traj_step),
            "intercept_idx": int(self._intercept_idx),
            "arm_side": str(self.arm_side),
            "roll_angle": float(getattr(self, "_roll_angle", 0.0)),
            "exit_edge": str(getattr(self, "_exit_edge", "front")),
            "roll_speed": float(getattr(self, "roll_speed", 0.0)),
            "roll_speed_mean": float(getattr(self, "roll_speed_mean", 0.0)),
            "ball_speed": float(self._ball_speed()),
            "partial_score": float(self.get_score()),
        }
        return obs
