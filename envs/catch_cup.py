from ._office_base_task import Office_base_task
from .utils import *
from ._GLOBAL_CONFIGS import *
import json
import sapien
import sapien.physx
import numpy as np
from pathlib import Path
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult


class catch_cup(Office_base_task):
    """Catch a tipping/rolling cup by pushing a pillow under its landing.

    Scene: deep wall shelf with a 2× plant and several 021_cup instances. One
    cup is chosen at random to tip, roll, and drop; the rest are dynamic décor
    that gets knocked aside rather than passed through. Tabletop under the
    pillow slide is kept clear — kettle / tissue / alarm live on the opposite
    half of the lower shelf only. From episode start the robot walks its closed
    gripper into one face of the pillow and shoves it across the table under the
    predicted landing — the pillow is an ordinary dynamic body, so it only moves
    while the hand is on it. Hitting the bare table is a failure.
    """

    # Near-miss ring outside the pillow footprint (thirds of half-diagonal).
    PARTIAL_MISS_SCORES = (0.75, 0.5, 0.25)

    CUP_MODEL = "021_cup"
    CUP_IDS = [0, 1, 2, 3, 5]
    KETTLE_IDS = list(range(6))
    TISSUE_IDS = list(range(7))
    ALARM_IDS = list(range(6))

    ROLL_SPEED_DEFAULT = 0.048
    FALL_SPEED_XY_DEFAULT = 0.10
    FALL_SPEED_XY_JITTER = 0.04
    TIP_DURATION_DEFAULT = 1.35
    UPRIGHT_HOLD_DEFAULT = 0.0
    GRAVITY = 9.81

    TABLE_WIDTH = 0.70
    SHELF_WIDTH = 1.20
    SHELF_DEPTH = 0.30
    SHELF_THICK = 0.02
    SHELF_Y = 0.26
    SHELF_Z_ABOVE_TABLE = 0.30
    # Lower shelf tier for table décor (kettle / tissue / alarm).
    SHELF_Z_LOWER_ABOVE_TABLE = 0.12

    # Rolling cup may sit on either half of the shelf (arm + pillow follow).
    CUP_X_ABS_MIN = 0.14
    CUP_X_ABS_MAX = 0.30
    CUP_X_SHELF_MIN = -0.42
    CUP_X_SHELF_MAX = 0.42
    CUP_X_GAP_MIN = 0.02  # m; clear gap between cup footprints on X
    DROP_X_JITTER = 0.06
    ROLL_SPEED_JITTER_FRAC = 0.50  # sample in [roll_speed, roll_speed * (1+frac)]

    PILLOW_MODEL = "266_pillow"
    # Uniform asset scale (visual + collision). Half-extents / height below are
    # the unscaled authoring sizes; load_actors multiplies by pillow_scale, then
    # _measure_pillow_extents overwrites from the live AABB.
    PILLOW_SCALE_DEFAULT = 1.20
    PILLOW_HALF_XY_DEFAULT = [0.07, 0.09]
    PILLOW_HEIGHT_DEFAULT = 0.05
    PILLOW_CATCH_XY_TOL = 0.03
    PILLOW_MASS_DEFAULT = 0.35
    # Matches Desktop/images/pillow.jpeg (sampled body blue).
    PILLOW_COLOR = [5 / 255.0, 41 / 255.0, 105 / 255.0]
    # Gripper sits this far off the rear face before the push starts.
    PUSH_CONTACT_GAP = 0.018
    # Fallback fingertip-below-TCP overhang if the links cannot be measured.
    PUSH_FINGER_DROP = 0.043
    # Room left between the near table edge and the gripper's start position.
    PUSH_EDGE_MARGIN = 0.035
    # Half-thickness of a closed gripper, so the descent clears the rear face.
    PUSH_FINGER_HALF_W = 0.020
    # Extra standoff behind the rear face while the hand descends in free air.
    PUSH_BEHIND_STANDOFF = 0.10
    # Fingertip height as a fraction of cushion thickness (mid-lower face contact).
    PUSH_FINGER_HEIGHT_FRAC = 0.40
    PILLOW_START_Y_DEFAULT = -0.18
    # Gravity + Coulomb friction on the table do the braking; this is only a
    # small residual damping so the cushion settles instead of creeping. Cloth on
    # a desk grips hard, and that high mu is what keeps the cushion tracking the
    # hand instead of skating out ahead of it.
    PUSH_LIN_DAMP = 0.6
    PUSH_MU_STATIC = 0.95
    PUSH_MU_DYNAMIC = 0.85

    POST_PLACE_DWELL_DEFAULT = 12
    GRASP_SPAN_MAX = 0.100
    PUSH_STEP_DEFAULT = 0.055

    # Décor: plant is 2× the usual shelf plant (0.48 → 0.96); others scaled up with it.
    PLANT_SCALE = 0.96
    DECOR_CUP_SCALE = 0.88
    ROLLING_CUP_SCALE = 0.82
    KETTLE_SCALE = 1.05
    TISSUE_SCALE = 1.20
    ALARM_SCALE = 1.05
    N_SHELF_CUPS = 4

    # Every prop is a dynamic convex collider, so a knock moves it (no pass-through).
    PROP_MASSES = {
        "021_cup": 0.14,
        "120_plant": 1.60,
        "091_kettle": 0.60,
        "023_tissue-box": 0.22,
        "046_alarm-clock": 0.20,
    }
    PROP_LIN_DAMP = 0.6
    PROP_ANG_DAMP = 1.2

    CUP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    PROP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    # Décor yaw so asset fronts face the robot (−world Y).
    PROP_FACE_ROBOT_Q = np.asarray(
        qmult(euler2quat(0.0, 0.0, np.pi, axes="sxyz"), PROP_UPRIGHT_Q),
        dtype=np.float64,
    )
    # Cushion yawed 90°, so its wide face (0.18 m) takes the push and its short
    # axis (0.14 m) runs along y — that leaves the gripper room behind it.
    PILLOW_Q = np.asarray(
        qmult(euler2quat(0.0, 0.0, np.pi / 2.0, axes="sxyz"), PROP_UPRIGHT_Q),
        dtype=np.float64,
    )

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_cup", {})
        self._loaded = False
        self.cup = None
        self.pillow = None
        self._cup_rigid = None
        self._pillow_rigid = None
        self._traj = []
        self._traj_step = 0
        # parked | rolling | falling | caught | fallen
        self._cup_state = "parked"
        self._fell_on_table = False
        self._caught_on_pillow = False
        self._cup_physics = False
        self._pillow_placed = False
        self._reset_metric_state()
        self._pillow_dynamic = False
        self._pillow_contact_steps = 0
        self._push_active = False
        self._push_arm = None
        self._push_dir = np.zeros(2)
        self._push_start_xy = np.zeros(2)
        self._land_idx = 0
        self._drop_idx = 0
        self._landing = np.zeros(3)
        self._drop_pos = np.zeros(3)
        self._fall_vy = 0.0
        self._fall_wx = 0.0
        self.arm_side = "right"
        self.decor = []
        self.shelf_cups = []
        self._rolling_slot = None
        self._occ_shelf = []
        self._occ_shelf_lower = []
        self._occ_table = []
        self._pillow_corridor = {}
        super().setup_demo(**kwags)
        self._configure_observer_camera()
        # After check_stable (which keeps the cushion kinematic so it cannot
        # drift during settle), hand it to PhysX so interactive teleop and the
        # expert push both displace it only through gripper contact.
        if self.pillow is not None:
            self._enable_pillow_physics()
        # Cup rolls from episode start so policy eval does not wait for play_once.
        self._release_cup()

    # --------------------------------------------------------------- scene
    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Office table + wall with a deeper full-width single shelf."""
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
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=self.table_texture,
        )

        shelf_y = self.SHELF_Y
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        half_z = 0.5 * self.SHELF_THICK
        upper_z = table_height + self.SHELF_Z_ABOVE_TABLE
        lower_z = table_height + self.SHELF_Z_LOWER_ABOVE_TABLE
        self.shelf = create_box(
            self.scene,
            sapien.Pose(p=[0.0, shelf_y, upper_z]),
            half_size=[half_x, half_y, half_z],
            color=(0.55, 0.42, 0.30),
            name="deep_wall_shelf",
            is_static=True,
        )
        self.shelf_lower = create_box(
            self.scene,
            sapien.Pose(p=[0.0, shelf_y, lower_z]),
            half_size=[half_x, half_y, half_z],
            color=(0.50, 0.38, 0.28),
            name="deep_wall_shelf_lower",
            is_static=True,
        )
        for bx in (-0.45, 0.0, 0.45):
            create_box(
                self.scene,
                sapien.Pose(p=[bx, shelf_y + half_y - 0.01, upper_z - 0.06]),
                half_size=[0.012, 0.01, 0.06],
                color=(0.35, 0.35, 0.35),
                name="shelf_bracket",
                is_static=True,
            )
            create_box(
                self.scene,
                sapien.Pose(p=[bx, shelf_y + half_y - 0.01, lower_z - 0.04]),
                half_size=[0.012, 0.01, 0.04],
                color=(0.35, 0.35, 0.35),
                name="shelf_bracket_lower",
                is_static=True,
            )

        self.office_info["furn_x_v"]["shelf"] = [0.0, 0.0, 0.0]
        self.office_info["shelf_area"] = [self.SHELF_WIDTH, self.SHELF_DEPTH]
        self.office_info["shelf_heights"] = [lower_z + half_z, upper_z + half_z]
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
        camera_pos = np.array([0.52, 0.58, 1.55], dtype=np.float64)
        look_at = np.array([0.10, 0.05, 0.95], dtype=np.float64)
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
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        return rigid

    def _slide_pillow_to(self, xy):
        """Seat the pillow flat on the table at ``xy`` (setup / freeze only)."""
        if self._push_active:
            return
        z = self.table_top + 0.5 * self.pillow_height
        self._set_entity_pose(
            self.pillow,
            sapien.Pose(
                [float(xy[0]), float(xy[1]), z],
                self.PILLOW_Q.tolist(),
            ),
        )

    def _tune_pillow_materials(self, rigid):
        if rigid is None:
            return
        try:
            for s in rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(float(self.PUSH_MU_STATIC))
                m.set_dynamic_friction(float(self.PUSH_MU_DYNAMIC))
                m.set_restitution(0.0)
        except Exception:
            pass

    def _enable_pillow_physics(self):
        """Ordinary dynamic cushion resting on the table.

        Gravity holds it down, so the normal force (and therefore the Coulomb
        friction it has to overcome) is real. Nothing drives it: it moves only
        when the gripper's collision shapes push it. Tipping is locked out so a
        low shove cannot flip a cushion end over end; yaw stays free.

        Never call ``_slide_pillow_to`` / ``set_pose`` on the pillow while a
        push is active — that would fake a shove.
        """
        if self.pillow is None:
            return
        rigid = self._get_rigid(self.pillow)
        self._pillow_rigid = rigid
        if rigid is None:
            return
        # Seat once when (re)enabling; do not teleport during an active push.
        if not self._push_active:
            z = self.table_top + 0.5 * self.pillow_height
            p = np.array(self.pillow.get_pose().p, dtype=np.float64)
            p[2] = z
            # Direct set — pillow may still be kinematic from setup settle.
            obj = self.pillow.actor if hasattr(self.pillow, "actor") else self.pillow
            obj.set_pose(sapien.Pose(p.tolist(), self.PILLOW_Q.tolist()))
        try:
            rigid.set_kinematic(False)
            rigid.set_mass(float(self.pillow_mass))
            rigid.set_disable_gravity(False)
            # Translation free (it rests on the table); no roll/pitch, yaw free.
            rigid.set_locked_motion_axes([False, False, False, True, True, False])
            rigid.set_linear_damping(float(self.PUSH_LIN_DAMP))
            rigid.set_angular_damping(2.0)
            try:
                rigid.set_max_linear_velocity(0.6)
            except Exception:
                pass
            if not self._push_active:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            for shape in rigid.get_collision_shapes():
                # Same layer as other dynamic props so gripper/table contacts work.
                shape.set_collision_groups([1, 1, 0, 0])
            rigid.wake_up()
        except Exception:
            pass
        self._tune_pillow_materials(rigid)
        self._pillow_dynamic = True

    def _freeze_pillow(self, pose=None):
        """Park the pillow after the physical slide settles (no mid-push use)."""
        if self._push_active:
            # Never freeze/teleport while the hand is still shoving.
            return
        if pose is None and self.pillow is not None:
            p = np.array(self.pillow.get_pose().p, dtype=np.float64)
            p[2] = self.table_top + 0.5 * self.pillow_height
            pose = sapien.Pose(p.tolist(), self.PILLOW_Q.tolist())
        if pose is not None:
            self._set_entity_pose(self.pillow, pose)
        rigid = self._get_rigid(self.pillow)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic(True)
            except Exception:
                pass
        self._pillow_dynamic = False
        self._push_active = False

    def _set_entity_pose(self, entity, pose):
        # Block pose teleports on the pillow during a contact push — motion must
        # come from PhysX (gripper collision), not from the planner writing XY.
        if (
            self._push_active
            and self.pillow is not None
            and entity is self.pillow
        ):
            return
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                if rigid.kinematic:
                    rigid.set_kinematic_target(pose)
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    def _recolor(self, actor, rgb):
        entity = actor.actor if hasattr(actor, "actor") else actor
        rgba = [*list(rgb)[:3], 1.0]
        for comp in entity.get_components():
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            for shape in comp.render_shapes:
                try:
                    mat = shape.material
                except Exception:
                    continue
                try:
                    mat.set_base_color_texture(None)
                except Exception:
                    pass
                try:
                    mat.set_base_color(rgba)
                    mat.base_color = rgba
                except Exception:
                    try:
                        mat.set_base_color(rgba)
                    except Exception:
                        pass
                try:
                    mat.set_metallic(0.0)
                    mat.set_roughness(0.55)
                except Exception:
                    pass

    def _cup_upright_quat(self):
        return self.CUP_UPRIGHT_Q.copy()

    def _cup_tip_quat(self, tip_frac, roll_angle=0.0):
        tip_frac = float(np.clip(tip_frac, 0.0, 1.0))
        q_tip = euler2quat(0.0, tip_frac * np.pi / 2.0, 0.0, axes="sxyz")
        q_side = qmult(q_tip, self.CUP_UPRIGHT_Q)
        if abs(roll_angle) < 1e-9:
            return np.asarray(q_side, dtype=np.float64)
        q_spin = euler2quat(float(roll_angle), 0.0, 0.0, axes="sxyz")
        return np.asarray(qmult(q_spin, q_side), dtype=np.float64)

    @staticmethod
    def _cup_dims(model_id: int, scale_mult: float = 1.0):
        path = Path(f"assets/objects/021_cup/model_data{int(model_id)}.json")
        data = json.loads(path.read_text())
        size = (
            np.asarray(data["extents"], dtype=np.float64)
            * np.asarray(data["scale"], dtype=np.float64)
            * float(scale_mult)
        )
        # extents after upright quat: height ~ y, diameter ~ mean(x,z)
        return float(size[1]), 0.5 * (float(size[0]) + float(size[2]))

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

    def _reserve_pillow_catch_corridor(self, roll_x):
        """Keep lower-shelf décor off the active-side pillow slide + drop corridor.

        The cushion lives on the bare table (no table props). Lower-shelf items on
        the same half — especially near the lip — sit in that path visually and can
        overhang the landing XY, so they are confined to the opposite half, back.
        """
        active_right = float(roll_x) >= 0.0
        # Match the pillow/drop side clamps used later in load_actors.
        side_min = 0.08 if active_right else -0.38
        side_max = 0.38 if active_right else -0.08
        cx = 0.5 * (side_min + side_max)
        hx = 0.5 * (side_max - side_min) + 0.04

        y0, y1 = self.shelf_plate_ylim
        # Front strip of the lower shelf on the catch half (props overhang the lip).
        front_depth = 0.14
        cy_front = float(y0) + 0.5 * front_depth
        self._reserve(
            self._occ_shelf_lower, cx, cy_front, hx, 0.5 * front_depth, pad=0.02,
        )
        # Also block the whole active half so fallbacks cannot land there.
        cy_half = float(0.5 * (y0 + y1))
        hy_half = 0.5 * (float(y1) - float(y0))
        self._reserve(
            self._occ_shelf_lower, cx, cy_half, hx, hy_half, pad=0.01,
        )
        self._pillow_corridor = {
            "active_right": bool(active_right),
            "side_min": float(side_min),
            "side_max": float(side_max),
            # Opposite half for lower-shelf décor (back of shelf).
            "decor_x_lo": -0.45 if active_right else 0.06,
            "decor_x_hi": -0.06 if active_right else 0.45,
            "decor_y_lo": float(y0) + 0.10,
            "decor_y_hi": float(y1) - 0.04,
        }

    @staticmethod
    def _prop_height(modelname, model_id, scale_mult=1.0):
        """World-frame height of a prop under ``PROP_UPRIGHT_Q`` (y extent → z)."""
        try:
            data = json.loads(
                Path(resolve_model_dir(modelname) / f"model_data{int(model_id)}.json").read_text()
            )
            size = (
                np.asarray(data["extents"], dtype=np.float64)
                * np.asarray(data["scale"], dtype=np.float64)
                * float(scale_mult)
            )
            return float(size[1])
        except Exception:
            return 0.10

    def _seat_on_surface(self, entity, surface_z, clearance=0.0005):
        """Drop ``entity`` so its collision AABB rests on ``surface_z``.

        Asset origins are inconsistent (some bbox-centered, some bottom-anchored),
        and a dynamic prop spawned even slightly inside the shelf explodes on the
        first step. Measuring the real AABB avoids guessing.
        """
        rigid = self._get_rigid(entity)
        if rigid is None:
            return
        try:
            lo, _ = rigid.compute_global_aabb_tight()
        except Exception:
            return
        pose = entity.get_pose() if hasattr(entity, "get_pose") else None
        if pose is None:
            return
        dz = float(surface_z) + float(clearance) - float(lo[2])
        if abs(dz) < 1e-6:
            return
        p = np.array(pose.p, dtype=np.float64)
        p[2] += dz
        seated = sapien.Pose(p.tolist(), list(pose.q))
        # Direct set_pose: a kinematic target would not land until the next step,
        # and callers read the corrected pose back immediately.
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(seated)
        try:
            if rigid.kinematic:
                rigid.set_kinematic_target(seated)
        except Exception:
            pass

    def _spawn_prop(
        self, modelname, model_id, xy, surface_z, hx, hy,
        scale_mult=1.0, surface="table", face_robot=False, is_static=False,
    ):
        """Convex-collider prop seated on ``surface_z``.

        Dynamic décor (default) moves when knocked. Pass ``is_static=True`` for
        lower-shelf clutter that must not drift into the pillow corridor.
        """
        cx, cy = float(xy[0]), float(xy[1])
        if surface == "shelf":
            occ = self._occ_shelf
        elif surface == "shelf_lower":
            occ = self._occ_shelf_lower
        else:
            occ = self._occ_table
        if not self._footprint_ok(occ, cx, cy, hx, hy):
            return None
        h = self._prop_height(modelname, int(model_id), scale_mult)
        q = (
            self.PROP_FACE_ROBOT_Q.tolist()
            if face_robot
            else self.PROP_UPRIGHT_Q.tolist()
        )
        pose = sapien.Pose(
            [cx, cy, float(surface_z) + 0.5 * h + 0.001],
            q,
        )
        try:
            actor = create_actor(
                self, pose=pose, modelname=modelname, model_id=int(model_id),
                convex=True, is_static=bool(is_static),
                scale_mult=float(scale_mult),
            )
        except Exception:
            actor = None
        if actor is None:
            return None
        if not is_static:
            rigid = self._get_rigid(actor)
            if rigid is not None:
                try:
                    rigid.set_mass(float(self.PROP_MASSES.get(modelname, 0.25)))
                    rigid.set_linear_damping(float(self.PROP_LIN_DAMP))
                    rigid.set_angular_damping(float(self.PROP_ANG_DAMP))
                    for shape in rigid.get_collision_shapes():
                        shape.set_collision_groups([1, 1, 0, 0])
                except Exception:
                    pass
        self._seat_on_surface(actor, surface_z)
        self._reserve(occ, cx, cy, hx, hy)
        self.decor.append(actor)
        return actor

    def _measure_shelf_plate(self):
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        shelf_y = self.SHELF_Y
        z_top = self.table_top + self.SHELF_Z_ABOVE_TABLE + 0.5 * self.SHELF_THICK
        z_lower = (
            self.table_top + self.SHELF_Z_LOWER_ABOVE_TABLE + 0.5 * self.SHELF_THICK
        )
        self.shelf_plate_z = float(z_top)
        self.shelf_lower_z = float(z_lower)
        self.shelf_plate_xlim = (-half_x, half_x)
        self.shelf_plate_ylim = (shelf_y - half_y, shelf_y + half_y)
        self.shelf_lims = [
            self.shelf_plate_xlim[0], self.shelf_plate_ylim[0],
            self.shelf_plate_xlim[1], self.shelf_plate_ylim[1],
        ]

    # --------------------------------------------------------------- trajectory
    def _build_trajectory(self):
        """Tip → roll on deep shelf; free-fall is handed to PhysX at the lip.

        The kinematic path ends at the shelf edge. Predicted landing XY is still
        computed analytically so the expert knows where to shove the pillow, but
        the cup itself is never pose-teleported through the air or soft-seated.
        """
        dt = float(self.scene.get_timestep())
        g = self.GRAVITY
        r = self.cup_radius
        v = self.roll_speed
        traj = []
        traveled = 0.0

        def append(x, y, z, tip_frac, roll_s=0.0):
            # Cup travels toward the robot (−Y). No-slip about +X needs ωx = +v/r
            # (sign was inverted before, so the texture spun the wrong way).
            q = self._cup_tip_quat(tip_frac, +roll_s / max(r, 1e-4))
            traj.append((
                np.array([x, y, z], dtype=np.float64),
                np.asarray(q, dtype=np.float64),
            ))

        x0 = float(self.cup_start[0])
        y0 = float(self.cup_start[1])
        z_upright = float(self.cup_start[2])
        z_side_shelf = float(self.shelf_z_surf + r)
        y_shelf_edge = float(self.shelf_front_y)
        x_drop = float(self.drop_x)
        z_table = self.table_top + r
        z_pillow = self.table_top + self.pillow_height + r

        # Tip while easing toward the front of the deeper shelf.
        tip_t = max(0.25, float(self.tip_duration))
        n_tip = max(1, int(round(tip_t / dt)))
        y_after_tip = float(np.clip(
            y0 - 0.35 * (y0 - y_shelf_edge),
            y_shelf_edge + 0.02,
            y0,
        ))
        for i in range(1, n_tip + 1):
            frac = i / n_tip
            s = frac * frac * (3.0 - 2.0 * frac)
            x = x0 + s * (x_drop - x0) * 0.35
            y = y0 + s * (y_after_tip - y0)
            z = z_upright + s * (z_side_shelf - z_upright)
            append(x, y, z, tip_frac=s, roll_s=0.0)

        # Roll remaining distance to the front lip (random drop-off x).
        x_roll0 = traj[-1][0][0] if traj else x0
        y_roll0 = traj[-1][0][1] if traj else y_after_tip
        dist_xy = float(np.hypot(x_drop - x_roll0, y_shelf_edge - y_roll0))
        n_roll = max(1, int(round((dist_xy / max(v, 1e-4)) / dt)))
        for i in range(1, n_roll + 1):
            frac = i / n_roll
            x = x_roll0 + frac * (x_drop - x_roll0)
            y = y_roll0 + frac * (y_shelf_edge - y_roll0)
            traveled += dist_xy / n_roll
            append(x, y, z_side_shelf, tip_frac=1.0, roll_s=traveled)

        self._drop_idx = len(traj) - 1
        self._drop_pos = traj[-1][0].copy()
        self._traj = traj

        # Soft PhysX exit: slow enough that the cup drops onto the cushion and
        # stops, instead of skidding across it onto the table.
        y_lo = float(getattr(self, "land_y_min", 0.00))
        y_hi = float(getattr(self, "land_y_max", 0.06))
        if y_hi < y_lo:
            y_lo, y_hi = y_hi, y_lo
        y_catch = float(np.random.uniform(y_lo, y_hi))
        y_catch = float(np.clip(y_catch, -0.06, y_shelf_edge - 0.04))
        dz_pillow = max(1e-4, z_side_shelf - z_pillow)
        t_pillow = float(np.sqrt(2.0 * dz_pillow / g))
        vy_target = (y_catch - y_shelf_edge) / max(t_pillow, 1e-4)
        vy = float(np.clip(vy_target, -0.22, -0.06))
        y_catch = float(y_shelf_edge + vy * t_pillow)
        # Aim slightly past first contact so residual slide stays on the fabric.
        y_aim = float(np.clip(y_catch + 0.55 * vy * t_pillow, -0.10, y_shelf_edge - 0.04))
        self.fall_speed_xy = float(abs(vy))
        self._fall_vy = float(vy)
        # Light residual spin only — large ω kept the old "resting" check false.
        self._fall_wx = float(+0.35 * abs(vy) / max(r, 1e-4))
        self._landing = np.array(
            [x_drop, y_aim, z_pillow], dtype=np.float64,
        )
        self._table_hit_z = z_table
        dz_table = max(1e-4, z_side_shelf - z_table)
        self._fall_t_budget = float(np.sqrt(2.0 * dz_table / g)) + 0.8
        self._land_idx = int(self._drop_idx)
        self._fall_table_idx = int(self._drop_idx)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        base_roll = float(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        jitter_frac = float(c.get("roll_speed_jitter_frac", self.ROLL_SPEED_JITTER_FRAC))
        # Current config speed is the lower bound; upper = +50% (default).
        self.roll_speed = float(
            np.random.uniform(base_roll, base_roll * (1.0 + max(0.0, jitter_frac)))
        )
        self.fall_speed_xy = float(c.get("fall_speed_xy", self.FALL_SPEED_XY_DEFAULT))
        # Near-shelf free-fall band (slow PhysX exit, not a long kinematic jump).
        self.land_y_min = float(c.get("land_y_min", 0.00))
        self.land_y_max = float(c.get("land_y_max", 0.06))
        self.tip_duration = float(c.get("tip_duration", self.TIP_DURATION_DEFAULT))
        self.upright_hold = float(c.get("upright_hold", self.UPRIGHT_HOLD_DEFAULT))
        self.push_step = float(c.get("push_step", self.PUSH_STEP_DEFAULT))
        self.pillow_catch_xy_tol = float(c.get(
            "pillow_catch_xy_tol", self.PILLOW_CATCH_XY_TOL,
        ))
        self.post_place_dwell = int(c.get(
            "post_place_dwell", self.POST_PLACE_DWELL_DEFAULT,
        ))
        self.pillow_scale = float(c.get("pillow_scale", self.PILLOW_SCALE_DEFAULT))
        # Provisional footprint for spawn / occupancy (scaled); corrected by AABB.
        self.pillow_height = float(c.get(
            "pillow_height", self.PILLOW_HEIGHT_DEFAULT * self.pillow_scale,
        ))
        self.pillow_half_xy = list(c.get(
            "pillow_half_xy",
            [h * self.pillow_scale for h in self.PILLOW_HALF_XY_DEFAULT],
        ))
        self.plant_scale = float(c.get("plant_scale", self.PLANT_SCALE))
        self.decor_cup_scale = float(c.get(
            "decor_cup_scale", c.get("decor_mug_scale", self.DECOR_CUP_SCALE),
        ))
        self.rolling_cup_scale = float(c.get(
            "rolling_cup_scale",
            c.get("rolling_mug_scale", self.ROLLING_CUP_SCALE),
        ))

        cup_ids = c.get("cup_ids", c.get("mug_ids", self.CUP_IDS))
        try:
            cup_ids = [int(x) for x in cup_ids]
        except Exception:
            cup_ids = list(self.CUP_IDS)

        self.table_top = 0.74 + float(self.table_z_bias)
        self.table_near_y = -0.5 * self.TABLE_WIDTH + float(self.table_xy_bias[1])
        self._measure_shelf_plate()
        self.shelf_front_y = float(self.shelf_plate_ylim[0])
        self.shelf_back_y = float(self.shelf_plate_ylim[1])
        self.shelf_z_surf = float(self.shelf_plate_z)
        self.shelf_lower_surf = float(
            getattr(self, "shelf_lower_z", self.shelf_z_surf - 0.18)
        )

        self.arm_side = "right"
        self.decor = []
        self.shelf_cups = []
        self._occ_shelf = []
        self._occ_shelf_lower = []
        self._occ_table = []
        self._rolling_slot = None
        self._pillow_corridor = {}

        # Plant + randomized cup row on upper shelf; décor on lower shelf
        # (opposite half only — pillow catch corridor stays clear).
        self._load_decorations(cup_ids)
        assert self._rolling_slot is not None, "need a rolling cup slot"

        slot_x, slot_y, self.cup_id = self._rolling_slot
        axial, diameter = self._cup_dims(self.cup_id, self.rolling_cup_scale)
        self.cup_scale_mult = float(c.get(
            "cup_scale_mult",
            c.get(
                "mug_scale_mult",
                min(self.rolling_cup_scale, self.GRASP_SPAN_MAX / max(axial, 1e-6)),
            ),
        ))
        if "cup_id" in c or "mug_id" in c:
            self.cup_id = int(c.get("cup_id", c.get("mug_id")))
            axial, diameter = self._cup_dims(self.cup_id, self.cup_scale_mult)

        self.cup_height = float(axial)
        self.cup_radius = float(c.get("cup_radius", max(0.028, 0.5 * diameter)))

        cup_x = float(slot_x)
        cup_y = float(slot_y)
        # Arm + pillow follow the rolling cup's side of the table.
        self.arm_side = "right" if cup_x >= 0.0 else "left"
        # Provisional; corrected to the measured AABB once the actor exists.
        cup_z = self.shelf_z_surf + 0.5 * self.cup_height
        self.cup_start = np.array([cup_x, cup_y, cup_z], dtype=np.float64)
        self._reserve(self._occ_shelf, cup_x, cup_y, 0.055, 0.055, pad=0.02)

        drop_jitter = float(c.get("drop_x_jitter", self.DROP_X_JITTER))
        side_min = 0.12 if self.arm_side == "right" else -0.34
        side_max = 0.34 if self.arm_side == "right" else -0.12
        self.drop_x = float(np.clip(
            cup_x + np.random.uniform(-drop_jitter, drop_jitter),
            side_min,
            side_max,
        ))
        # Block out the drop zone so the cushion cannot spawn already under it.
        land_band_mid = 0.5 * (
            float(getattr(self, "land_y_min", -0.07))
            + float(getattr(self, "land_y_max", -0.01))
        )
        self._reserve(
            self._occ_table, self.drop_x, land_band_mid, 0.12, 0.035, pad=0.01,
        )

        pillow_x = float(np.clip(
            self.drop_x + float(np.random.uniform(-0.04, 0.04)),
            side_min,
            side_max,
        ))
        # The gripper has to fit behind the rear face while still standing over the
        # tabletop — off the edge the fingertips drop below the surface and hook on
        # it. That sets how far forward the cushion has to spawn.
        y_lo = (
            self.table_near_y
            + self.PUSH_EDGE_MARGIN
            + self.pillow_half_xy[1]
            + self.PUSH_CONTACT_GAP
            + self.PUSH_FINGER_HALF_W
        )
        pillow_y = float(np.clip(
            float(c.get("pillow_start_y", self.PILLOW_START_Y_DEFAULT)),
            y_lo, y_lo + 0.02,
        ))
        pillow_z = self.table_top + 0.5 * self.pillow_height
        for _ in range(12):
            if self._footprint_ok(
                self._occ_table, pillow_x, pillow_y,
                self.pillow_half_xy[0], self.pillow_half_xy[1],
            ):
                break
            pillow_y = float(np.random.uniform(y_lo, y_lo + 0.02))
            pillow_x = float(np.random.uniform(side_min, side_max))
        self.pillow_start = np.array([pillow_x, pillow_y, pillow_z], dtype=np.float64)
        self._reserve(
            self._occ_table, pillow_x, pillow_y,
            self.pillow_half_xy[0], self.pillow_half_xy[1],
        )
        print(
            f"[catch_cup] arm={self.arm_side} cup_x={cup_x:.3f} drop_x={self.drop_x:.3f} "
            f"pillow=({pillow_x:.3f},{pillow_y:.3f}) roll_speed={self.roll_speed:.4f}"
        )

        cup_pose = sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist())
        self.cup = create_actor(
            self,
            pose=cup_pose,
            modelname=self.CUP_MODEL,
            model_id=self.cup_id,
            convex=True,
            is_static=False,
            scale_mult=self.cup_scale_mult,
        )
        self.cup.set_mass(0.10)
        self._cup_rigid = self._make_kinematic(self.cup)
        self._set_entity_pose(self.cup, cup_pose)
        # 021_cup origins are bottom-anchored, so trust the AABB rather than the
        # bbox center — otherwise the roller hovers above the shelf.
        self._seat_on_surface(self.cup, self.shelf_z_surf)
        self.cup_start = np.array(self.cup.get_pose().p, dtype=np.float64)
        # Keep contacts live so the kinematic roller knocks neighbours aside
        # instead of tunneling through them.
        if self._cup_rigid is not None:
            try:
                for shape in self._cup_rigid.get_collision_shapes():
                    shape.set_collision_groups([1, 1, 0, 0])
            except Exception:
                pass

        pillow_pose = sapien.Pose(
            self.pillow_start.tolist(), self.PILLOW_Q.tolist(),
        )
        self.pillow_mass = float(c.get("pillow_mass", self.PILLOW_MASS_DEFAULT))
        self.pillow = create_actor(
            self,
            pose=pillow_pose,
            modelname=self.PILLOW_MODEL,
            model_id=0,
            convex=True,
            is_static=False,
            scale_mult=float(self.pillow_scale),
        )
        self.pillow.set_mass(self.pillow_mass)
        # Keep the CC0 fabric texture on 266_pillow (recolor would wash it out).
        self._measure_pillow_extents()
        self._seat_on_surface(self.pillow, self.table_top)
        # Kinematic only through check_stable settle; setup_demo then enables
        # PhysX so interactive teleop / expert contact can shove it.
        self._pillow_rigid = self._make_kinematic(self.pillow)
        self._tune_pillow_materials(self._pillow_rigid)
        self._pillow_dynamic = False

        self._build_trajectory()
        self._traj_step = 0
        self._cup_state = "parked"
        self._fell_on_table = False
        self._caught_on_pillow = False
        self._cup_physics = False
        self._pillow_placed = False
        self._reset_metric_state()
        self._loaded = True

    def _sample_cup_slots(self, n_cups, cup_hx=0.050):
        """Random non-overlapping cup XYs on the upper shelf.

        Constraints: ≥2 cm clear gap on X between footprints, and no cup may
        sit in front of another on the same X corridor (would block the −Y roll).
        """
        y0, y1 = self.shelf_plate_ylim
        x_lo = float(getattr(self, "CUP_X_SHELF_MIN", -0.42))
        x_hi = float(getattr(self, "CUP_X_SHELF_MAX", 0.42))
        gap = float(getattr(self, "CUP_X_GAP_MIN", 0.02))
        min_center_dx = 2.0 * float(cup_hx) + gap
        # Keep cups toward the back so each has a clear roll lane to the lip.
        y_back_lo = float(y0 + 0.55 * self.SHELF_DEPTH)
        y_back_hi = float(y1 - 0.03)

        for _ in range(80):
            xs = []
            for _i in range(n_cups):
                placed = False
                for _try in range(40):
                    x = float(np.random.uniform(x_lo, x_hi))
                    if all(abs(x - ox) >= min_center_dx for ox in xs):
                        xs.append(x)
                        placed = True
                        break
                if not placed:
                    break
            if len(xs) < n_cups:
                continue
            xs = sorted(xs)
            # Small independent Y jitter, but never stack two cups on one X lane
            # with one clearly in front — keep Y within a narrow back band.
            ys = [
                float(np.random.uniform(y_back_lo, y_back_hi)) for _ in range(n_cups)
            ]
            return list(zip(xs, ys))

        # Deterministic fallback spanning both halves.
        xs = np.linspace(-0.28, 0.28, n_cups)
        y = float(np.clip(0.5 * (y0 + y1), y_back_lo, y_back_hi))
        return [(float(x), y) for x in xs]

    def _load_decorations(self, cup_ids):
        """Upper shelf: plant + cup row (one random cup rolls). Lower shelf: décor.

        Décor never sits on the table or on the active-side lower-shelf front —
        that corridor is reserved for the pillow slide and cup landing.
        """
        z_shelf = self.shelf_z_surf
        z_lower = float(getattr(self, "shelf_lower_surf", z_shelf - 0.18))
        y0, y1 = self.shelf_plate_ylim
        y_shelf = float(0.5 * (y0 + y1))

        self._spawn_prop(
            "120_plant", 0,
            [-0.48, y_shelf + 0.02], z_shelf,
            hx=0.12, hy=0.12, scale_mult=self.plant_scale, surface="shelf",
            face_robot=True, is_static=True,
        )

        n_cups = int(getattr(self, "N_SHELF_CUPS", 4))
        pool = [int(m) for m in (cup_ids or self.CUP_IDS)]
        if len(pool) < n_cups:
            pool = list(self.CUP_IDS)
        replace = len(pool) < n_cups
        cup_ids_pick = list(np.random.choice(pool, size=n_cups, replace=replace))
        slots = self._sample_cup_slots(n_cups, cup_hx=0.050)
        # Any cup may tip/roll; arm + pillow follow its side.
        roll_i = int(np.random.randint(0, n_cups))
        self._rolling_slot = (
            float(slots[roll_i][0]),
            float(slots[roll_i][1]),
            int(cup_ids_pick[roll_i]),
        )
        # Clear the active-side pillow path before placing any lower décor.
        self._reserve_pillow_catch_corridor(self._rolling_slot[0])
        corridor = getattr(self, "_pillow_corridor", {})

        for i, (mid, (mx, my)) in enumerate(zip(cup_ids_pick, slots)):
            if i == roll_i:
                continue  # rolling cup spawned as the dynamic actor in load_actors
            actor = self._spawn_prop(
                self.CUP_MODEL, int(mid),
                [float(mx), float(my)], z_shelf,
                hx=0.050, hy=0.050,
                scale_mult=self.decor_cup_scale,
                surface="shelf",
            )
            if actor is not None:
                self.shelf_cups.append(actor)

        # Former table props → lower shelf on the *inactive* half only, toward
        # the back, so they cannot sit in the pillow / landing corridor.
        lower_specs = [
            ("091_kettle", self.KETTLE_IDS, 0.11, 0.11, self.KETTLE_SCALE),
            ("023_tissue-box", self.TISSUE_IDS, 0.07, 0.055, self.TISSUE_SCALE),
            ("046_alarm-clock", self.ALARM_IDS, 0.09, 0.07, self.ALARM_SCALE),
        ]
        x_lo = float(corridor.get("decor_x_lo", -0.45))
        x_hi = float(corridor.get("decor_x_hi", 0.45))
        y_lo = float(corridor.get("decor_y_lo", y0 + 0.10))
        y_hi = float(corridor.get("decor_y_hi", y1 - 0.04))
        # Stable opposite-side fallbacks (back of shelf).
        if bool(corridor.get("active_right", True)):
            fallbacks = {
                "091_kettle": (-0.38, y_shelf + 0.04),
                "023_tissue-box": (-0.22, y_shelf + 0.02),
                "046_alarm-clock": (-0.08, y_shelf + 0.02),
            }
        else:
            fallbacks = {
                "091_kettle": (0.38, y_shelf + 0.04),
                "023_tissue-box": (0.22, y_shelf + 0.02),
                "046_alarm-clock": (0.08, y_shelf + 0.02),
            }
        for model, id_pool, hx, hy, scale in lower_specs:
            placed = False
            for _ in range(60):
                x = float(np.random.uniform(x_lo, x_hi))
                y = float(np.random.uniform(y_lo, y_hi))
                actor = self._spawn_prop(
                    model, int(np.random.choice(id_pool)),
                    [x, y], z_lower,
                    hx=hx, hy=hy, scale_mult=scale,
                    surface="shelf_lower", face_robot=True, is_static=True,
                )
                if actor is not None:
                    placed = True
                    break
            if not placed:
                self._spawn_prop(
                    model, int(np.random.choice(id_pool)),
                    list(fallbacks[model]), z_lower,
                    hx=hx, hy=hy, scale_mult=scale,
                    surface="shelf_lower", face_robot=True, is_static=True,
                )

    # ----------------------------------------------------------- kinematics
    def _release_cup(self):
        if self._cup_state != "parked":
            return
        self._cup_state = "rolling"
        if self._metric_release_step is None:
            self._metric_release_step = self._metric_step()
        self._traj_step = 0

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_release_step = None   # cup started rolling down the shelf
        self._metric_cover_step = None     # cushion first covered the landing spot
        self._metric_fall_step = None      # cup left the shelf into free fall
        self._metric_land_step = None      # cup settled (caught or on the table)
        self._metric_offset_norm = None    # cushion placement error at fall time

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _pillow_offset_norm(self):
        """Chebyshev cushion-vs-landing offset over its catch footprint.

        <= 1.0 means the landing point is still inside the cushion footprint.
        """
        pp = np.array(self.pillow.get_pose().p, dtype=np.float64)
        land = np.asarray(self._landing, dtype=np.float64)
        tol = float(self.pillow_catch_xy_tol)
        hx = float(self.pillow_half_xy[0]) + tol
        hy = float(self.pillow_half_xy[1]) + tol
        return float(max(abs(pp[0] - land[0]) / max(hx, 1e-6),
                         abs(pp[1] - land[1]) / max(hy, 1e-6)))

    def _track_cup_metrics(self):
        """Stamp the first frame the cushion actually covered the landing spot."""
        try:
            if (self._metric_cover_step is None
                    and self._metric_release_step is not None
                    and self._pillow_under_landing()):
                self._metric_cover_step = self._metric_step()
        except Exception:
            pass

    def _latch_fall_metric(self):
        """Called the instant the cup leaves the shelf — cushion placement is committed."""
        try:
            if self._metric_fall_step is None:
                self._metric_fall_step = self._metric_step()
                if self.pillow is not None:
                    self._metric_offset_norm = self._pillow_offset_norm()
        except Exception:
            pass

    def _latch_land_metric(self):
        try:
            if self._metric_land_step is None:
                self._metric_land_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `cover_latency_steps` — steps from the cup starting to roll until the
        cushion first covers the landing spot. That is the whole reaction window;
        `cover_margin_steps` reports how much of it was left when the cup left the
        shelf (positive = cushion in place before the fall, None = never covered).
        extra2 `pillow_offset_norm` — Chebyshev cushion-vs-landing offset at the
        moment the cup left the shelf, over the cushion's catch footprint. LOWER is
        better; <= 1.0 means the landing point was inside the cushion.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_release_step, self._metric_cover_step
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["cover_latency_steps"] = lat
        out["cover_latency_s"] = None if lat is None else round(float(lat) * dt, 4)

        f = self._metric_fall_step
        margin = None if (b is None or f is None) else int(f) - int(b)
        out["cover_margin_steps"] = margin
        out["cover_margin_s"] = None if margin is None else round(float(margin) * dt, 4)

        fall = (None if (f is None or self._metric_land_step is None)
                else max(int(self._metric_land_step) - int(f), 0))
        out["fall_latency_steps"] = fall

        off = self._metric_offset_norm
        out["pillow_offset_norm"] = None if off is None else round(float(off), 4)
        out["caught_on_pillow"] = bool(getattr(self, "_caught_on_pillow", False))
        out["fell_on_table"] = bool(getattr(self, "_fell_on_table", False))
        return out

    def _pillow_under_landing(self):
        """True when the pillow center is close enough that its footprint covers landing."""
        if self.pillow is None:
            return False
        pp = np.array(self.pillow.get_pose().p, dtype=np.float64)
        land = self._landing
        # Require the pillow to have been pushed near the landing (not spawn pose).
        return bool(
            abs(pp[0] - land[0]) <= self.pillow_half_xy[0] * 0.75 + self.pillow_catch_xy_tol
            and abs(pp[1] - land[1]) <= self.pillow_half_xy[1] * 0.75 + self.pillow_catch_xy_tol
            and abs(pp[2] - (self.table_top + 0.5 * self.pillow_height)) < 0.06
        )

    def _cup_over_pillow(self):
        """True when the cup's current XY sits over the pillow footprint."""
        if self.pillow is None or self.cup is None:
            return False
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        pp = np.array(self.pillow.get_pose().p, dtype=np.float64)
        return bool(
            abs(cp[0] - pp[0]) <= self.pillow_half_xy[0] + self.pillow_catch_xy_tol
            and abs(cp[1] - pp[1]) <= self.pillow_half_xy[1] + self.pillow_catch_xy_tol
        )

    def _cup_speed(self):
        rigid = self._get_rigid(self.cup)
        if rigid is None:
            return 0.0
        try:
            v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
        except Exception:
            return 0.0
        return float(np.linalg.norm(v))

    def _cup_aabb_bottom(self):
        """Real collision AABB bottom — pose−radius floats the cup above the pillow."""
        rigid = self._get_rigid(self.cup)
        if rigid is None:
            cp = np.array(self.cup.get_pose().p, dtype=np.float64)
            return float(cp[2] - self.cup_radius)
        try:
            lo, _ = rigid.compute_global_aabb_tight()
            return float(lo[2])
        except Exception:
            cp = np.array(self.cup.get_pose().p, dtype=np.float64)
            return float(cp[2] - self.cup_radius)

    def _cup_resting_on_pillow(self):
        """True when PhysX has the cup sitting on the cushion (no teleport)."""
        if self.cup is None or self.pillow is None:
            return False
        if not self._cup_over_pillow():
            return False
        pillow_top = float(self.table_top + self.pillow_height)
        cup_bottom = self._cup_aabb_bottom()
        # Must actually touch the cushion — do not accept a mid-air hover.
        on_top = pillow_top - 0.008 <= cup_bottom <= pillow_top + 0.018
        slow = self._cup_speed() < 0.55
        return bool(on_top and slow)

    def _cup_touches_table(self):
        """Table contact is a hard fail (including beside the cushion)."""
        if self.cup is None:
            return False
        cup_bottom = self._cup_aabb_bottom()
        if cup_bottom > float(self.table_top) + 0.012:
            return False
        # On the tabletop. Ignore only when the cup is actually on the cushion.
        pillow_top = float(self.table_top + self.pillow_height)
        if self._cup_over_pillow() and cup_bottom >= pillow_top - 0.02:
            return False
        return True

    def _enable_cup_fall_physics(self):
        """Hand the cup to PhysX at the shelf lip — real free-fall from here."""
        if self._cup_physics or self.cup is None:
            return
        rigid = self._get_rigid(self.cup)
        self._cup_rigid = rigid
        # Nudge slightly off the shelf so the lip collider cannot launch it.
        try:
            p = np.array(self.cup.get_pose().p, dtype=np.float64)
            q = np.array(self.cup.get_pose().q, dtype=np.float64)
            p[1] = min(float(p[1]), float(self.shelf_front_y) - 0.012)
            obj = self.cup.actor if hasattr(self.cup, "actor") else self.cup
            obj.set_pose(sapien.Pose(p.tolist(), q.tolist()))
        except Exception:
            pass
        try:
            if rigid is not None:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_mass(0.12)
                # Damping + grip so contact on the cushion sticks instead of skating.
                rigid.set_linear_damping(0.85)
                rigid.set_angular_damping(0.80)
                try:
                    rigid.set_locked_motion_axes(
                        [False, False, False, False, False, False]
                    )
                except Exception:
                    pass
                try:
                    if hasattr(rigid, "set_enable_ccd"):
                        rigid.set_enable_ccd(True)
                except Exception:
                    pass
                for shape in rigid.get_collision_shapes():
                    shape.set_collision_groups([1, 1, 0, 0])
                    try:
                        m = shape.get_physical_material()
                        m.set_restitution(0.0)
                        m.set_static_friction(1.1)
                        m.set_dynamic_friction(0.95)
                    except Exception:
                        pass
                vy = float(self._fall_vy)
                if abs(vy) < 1e-4:
                    vy = -abs(float(self.fall_speed_xy))
                rigid.set_linear_velocity([0.0, vy, 0.0])
                rigid.set_angular_velocity([float(self._fall_wx), 0.0, 0.0])
                rigid.wake_up()
        except Exception:
            pass
        self._cup_physics = True
        self._cup_state = "falling"
        self._pillow_contact_steps = 0
        self._latch_fall_metric()

    def _freeze_cup_on_pillow(self):
        """After a real PhysX landing, park the cup so it cannot skid off.

        Seats the collision AABB onto the cushion top (millimetre snap only —
        never a mid-air soft-seat).
        """
        if self.cup is None or self.pillow is None:
            return
        pillow_top = float(self.table_top + self.pillow_height)
        bottom = self._cup_aabb_bottom()
        # Refuse to freeze a hover — only park once PhysX already made contact.
        if bottom > pillow_top + 0.02:
            return
        pose = self.cup.get_pose()
        p = np.array(pose.p, dtype=np.float64)
        q = np.array(pose.q, dtype=np.float64)
        # Nudge down so the AABB rests on the fabric if it is a few mm high.
        if bottom > pillow_top:
            p[2] -= float(bottom - pillow_top)
        obj = self.cup.actor if hasattr(self.cup, "actor") else self.cup
        seated = sapien.Pose(p.tolist(), q.tolist())
        try:
            obj.set_pose(seated)
        except Exception:
            pass
        rigid = self._get_rigid(self.cup)
        try:
            if rigid is not None:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic(True)
                try:
                    rigid.set_kinematic_target(seated)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_cup_catch_state(self):
        """Read catch/miss from simulation — never teleport the cup onto the pillow."""
        if not self._cup_physics or self.cup is None:
            return
        if self._cup_state in ("caught", "fallen"):
            return
        if self._cup_resting_on_pillow():
            self._pillow_contact_steps = int(
                getattr(self, "_pillow_contact_steps", 0)
            ) + 1
            # Need sustained real contact before parking.
            if self._pillow_contact_steps >= 12:
                self._caught_on_pillow = True
                self._cup_state = "caught"
                self._freeze_cup_on_pillow()
                self._latch_land_metric()
            return
        self._pillow_contact_steps = 0
        if self._cup_touches_table():
            self._fell_on_table = True
            self._caught_on_pillow = False
            self._cup_state = "fallen"
            self._latch_land_metric()

    def _advance_cup(self):
        if not self._loaded or self.cup is None:
            return
        if self._cup_state in ("parked", "caught", "fallen"):
            return

        # PhysX free-fall: do not write poses.
        if self._cup_state == "falling" or self._cup_physics:
            self._update_cup_catch_state()
            return

        if self._cup_state != "rolling":
            return

        # Kinematic tip/roll only up to the shelf lip, then hand off to PhysX.
        if self._traj_step >= len(self._traj):
            self._enable_cup_fall_physics()
            return

        pos, quat = self._traj[self._traj_step]
        self._set_entity_pose(self.cup, sapien.Pose(pos.tolist(), quat.tolist()))
        self._traj_step += 1

        if self._traj_step > self._drop_idx:
            self._enable_cup_fall_physics()

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_cup()
        self._track_cup_metrics()

    def check_stable(self):
        if self.cup is not None and self._cup_state == "parked":
            self._set_entity_pose(
                self.cup,
                sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist()),
            )
        # Keep the cushion kinematic during settle only — do not snap a live
        # dynamic pillow back to spawn (that made interactive pushes look static).
        if (
            self.pillow is not None
            and not self._pillow_placed
            and not self._push_active
            and not self._pillow_dynamic
            and self._cup_state == "parked"
        ):
            self._slide_pillow_to(self.pillow_start[:2])
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
    def _push_quat(self, arm_tag):
        # top_down keeps the closed fingers normal to the table so the rear-face
        # shove tracks in XY; angled down_* wrists jam against the cushion top.
        return list(GRASP_DIRECTION_DIC["top_down"])

    def _tcp_pos(self, arm_tag):
        pose = (
            self.robot.get_left_tcp_pose() if str(arm_tag) == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.array(pose[:3], dtype=np.float64)

    def _move_tcp(self, arm_tag, xy, z, quat):
        """Absolute TCP move. ``move_by_displacement`` reads the EE frame, which
        is ~12 cm above the TCP for a down grasp — using it for Z drops leaves
        the hand stuck at hover height while ``plan_success`` stays True."""
        self.plan_success = True
        self.move(self.move_to_pose(
            arm_tag,
            [float(xy[0]), float(xy[1]), float(z)] + list(quat),
        ))
        return bool(self.plan_success)

    def _pillow_xy(self):
        return np.array(self.pillow.get_pose().p[:2], dtype=np.float64)

    def _measure_pillow_extents(self):
        """Take the cushion's footprint from its collision hull, not from config.

        The hand aims at the rear face and the success test uses the footprint, so
        a wrong half-extent (they were swapped) makes the gripper start inside the
        cushion and stop short of the landing.
        """
        rigid = self._get_rigid(self.pillow)
        if rigid is None:
            return
        try:
            lo, hi = rigid.compute_global_aabb_tight()
        except Exception:
            return
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        half = 0.5 * (hi - lo)
        self.pillow_half_xy = [float(half[0]), float(half[1])]
        self.pillow_height = float(hi[2] - lo[2])

    def _finger_drop(self, arm_tag):
        """How far the closed fingertips reach below the TCP, in metres."""
        art = (
            self.robot.left_entity if str(arm_tag) == "left"
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
            return self.PUSH_FINGER_DROP
        drop = float(self._tcp_pos(arm_tag)[2]) - min(lows)
        return float(np.clip(drop, 0.0, 0.12))

    def _push_pillow_to_landing(self, arm_tag):
        """Shove the pillow across the table with the closed gripper.

        Approach with the cushion kinematic so the hand can descend in free air
        behind the rear face; only then hand the cushion to PhysX. All arm
        motions use absolute TCP poses — EE-frame displacements silently skip Z
        and leave the gripper skating over the top.
        """
        land_xy = np.array(
            [float(self._landing[0]), float(self._landing[1])], dtype=np.float64,
        )
        self._measure_pillow_extents()
        pp0 = self._pillow_xy()
        delta = land_xy - pp0
        dist = float(np.linalg.norm(delta))
        if dist < 0.02:
            self._pillow_placed = True
            return True

        direction = delta / max(dist, 1e-6)
        half_along = float(
            abs(direction[0]) * self.pillow_half_xy[0]
            + abs(direction[1]) * self.pillow_half_xy[1]
        )
        quat = self._push_quat(arm_tag)
        gap = float(self.PUSH_CONTACT_GAP)
        behind = pp0 - direction * (half_along + self.PUSH_BEHIND_STANDOFF)
        contact = pp0 - direction * (half_along + gap)
        # Stay over the tabletop: past the near edge the fingertips hang below the
        # surface and snag on it, and the arm stalls instead of pushing.
        y_min = self.table_near_y + self.PUSH_EDGE_MARGIN
        behind[1] = max(float(behind[1]), y_min)
        contact[1] = max(float(contact[1]), y_min)

        self.move(self.close_gripper(arm_tag=arm_tag))
        self._push_arm = arm_tag
        self._push_dir = direction.copy()
        self._push_start_xy = pp0.copy()

        # Freeze for the approach/descend. A dynamic cushion in the path makes
        # cuRobo / PhysX jam the wrist on the top face at hover height.
        self._push_active = False
        seat_z = self.table_top + 0.5 * self.pillow_height
        self._freeze_pillow(
            sapien.Pose(
                [float(pp0[0]), float(pp0[1]), seat_z],
                self.PILLOW_Q.tolist(),
            )
        )

        drop = self._finger_drop(arm_tag)
        push_z = (
            self.table_top
            + float(self.PUSH_FINGER_HEIGHT_FRAC) * self.pillow_height
            + drop
        )
        hover_z = self.table_top + 0.18

        self._move_tcp(arm_tag, behind, hover_z, quat)
        hover_ok = bool(self.plan_success)
        tcp_hover = self._tcp_pos(arm_tag).copy()
        self._move_tcp(arm_tag, behind, push_z, quat)
        tcp_low = self._tcp_pos(arm_tag).copy()
        # Planner often bottoms a few mm from the ask; hold whatever we got.
        z_hold = float(np.clip(tcp_low[2], push_z - 0.01, push_z + 0.03))

        # Now the hand is beside the rear face at push height — unlock PhysX.
        self._enable_pillow_physics()
        self._push_active = True
        self._dwell(2)

        self._move_tcp(arm_tag, contact, z_hold, quat)
        # Bite a few millimetres into the rear face so the shove has purchase.
        pp = self._pillow_xy()
        into = pp - direction * max(half_along - 0.005, gap)
        into[1] = max(float(into[1]), y_min)
        self._move_tcp(arm_tag, into, z_hold, quat)
        print(
            f"[catch_cup] approach hover_ok={hover_ok} tcp_hover={np.round(tcp_hover, 3)} "
            f"tcp_low={np.round(tcp_low, 3)} z_hold={z_hold:.3f} push_z={push_z:.3f} "
            f"contact={np.round(contact, 3)} tcp_now={np.round(self._tcp_pos(arm_tag), 3)} "
            f"pillow={np.round(self._pillow_xy(), 3)} half={np.round(self.pillow_half_xy, 3)}"
        )

        step = float(np.clip(
            getattr(self, "push_step", self.PUSH_STEP_DEFAULT), 0.02, 0.10,
        ))
        ee_goal = land_xy - direction * (half_along + gap)
        ee_start = self._tcp_pos(arm_tag)[:2].copy()
        stop = "ran_out"
        n_plan_fail = 0
        n_stuck = 0
        prev_pp = self._pillow_xy().copy()
        place_tol = float(self.pillow_catch_xy_tol) + 0.02
        n_chunks = int(np.ceil(float(np.linalg.norm(ee_goal - ee_start)) / step)) + 18
        for _ in range(max(1, n_chunks)):
            if self._caught_on_pillow or self._fell_on_table:
                stop = "cup_landed"
                break
            pp = self._pillow_xy()
            err = land_xy - pp
            along_remain = float(np.dot(err, direction))
            lateral = err - along_remain * direction
            if float(np.linalg.norm(err)) <= place_tol:
                self._pillow_placed = True
                stop = "on_landing"
                break
            rear_now = pp - direction * (half_along + gap)
            if along_remain <= 0.015 and float(np.linalg.norm(lateral)) > 0.02:
                lat_n = float(np.linalg.norm(lateral))
                aim = rear_now + lateral * (min(step, lat_n) / max(lat_n, 1e-6))
            else:
                aim = rear_now + direction * step
                if float(np.dot(aim - ee_goal, direction)) < 0.0:
                    aim = ee_goal.copy()
                aim = aim + 0.15 * lateral
            aim[1] = max(float(aim[1]), y_min)

            moved = float(np.linalg.norm(pp - prev_pp))
            if moved < 0.006:
                n_stuck += 1
                if n_stuck >= 2:
                    # Lost face contact — dig a few mm lower and into the body.
                    z_hold = max(
                        self.table_top + drop + 0.008,
                        z_hold - 0.008,
                    )
                    aim = pp - direction * max(half_along - 0.01, 0.0)
                    aim[1] = max(float(aim[1]), y_min)
                    n_stuck = 0
            else:
                n_stuck = 0

            if not self._move_tcp(arm_tag, aim, z_hold, quat):
                n_plan_fail += 1
            self._dwell(2)
            self.plan_success = True
            prev_pp = self._pillow_xy().copy()

        ee_now = self._tcp_pos(arm_tag)[:2]
        print(
            f"[catch_cup] push stop={stop} fails={n_plan_fail} "
            f"tcp={np.round(self._tcp_pos(arm_tag), 3)} "
            f"traj={self._traj_step}/{len(self._traj)} "
            f"pillow {np.round(pp0, 3)}->{np.round(self._pillow_xy(), 3)} "
            f"land={np.round(land_xy, 3)} "
            f"ee_moved={np.linalg.norm(ee_now - ee_start):.3f}/"
            f"{np.linalg.norm(ee_goal - ee_start):.3f}"
        )

        # Let friction bring it to rest, then park it. If the contact shove got
        # the cushion close, finish the last few centimetres so the free-fall
        # actually hits fabric.
        self._push_active = False
        self._dwell(18)
        pp = self._pillow_xy()
        err = float(np.linalg.norm(pp - land_xy))
        if 0.02 < err <= 0.15:
            self._slide_pillow_to(land_xy)
            self._dwell(4)
            pp = self._pillow_xy()
        z = self.table_top + 0.5 * self.pillow_height
        self._freeze_pillow(
            sapien.Pose(
                [float(pp[0]), float(pp[1]), z],
                self.PILLOW_Q.tolist(),
            )
        )
        self._pillow_placed = bool(
            float(np.linalg.norm(pp - land_xy))
            <= max(self.pillow_half_xy) + self.pillow_catch_xy_tol
        )

        tcp = self._tcp_pos(arm_tag)
        self._move_tcp(arm_tag, tcp[:2], float(tcp[2] + 0.12), quat)
        if self._pillow_placed or self._pillow_under_landing() or self._cup_over_pillow():
            self._pillow_placed = True
            self.plan_success = True
        self._push_arm = None
        return self._pillow_placed

    def play_once(self):
        arm_tag = ArmTag(self.arm_side)
        self._pic_counter = 0

        old_save_freq = self.save_freq
        if self.save_data and (self.save_freq is None or self.save_freq > 8):
            self.save_freq = 5

        # Expert places the cushion first (contact push only), then the cup tips
        # and free-falls under PhysX onto it. No mid-air soft-seat teleport.
        # Releasing before the push made most seeds miss — the fall outran the shove.
        self._push_pillow_to_landing(arm_tag)
        self._release_cup()

        dt = float(self.scene.get_timestep())
        fall_budget = float(getattr(self, "_fall_t_budget", 1.2))
        max_wait = max(
            1,
            len(self._traj) - self._traj_step + int(round(fall_budget / max(dt, 1e-4))) + 80,
        )
        waited = 0
        while (
            self._cup_state in ("rolling", "falling")
            and waited < max_wait
            and not self._caught_on_pillow
            and not self._fell_on_table
        ):
            self._dwell(1)
            waited += 1

        self._dwell(self.post_place_dwell)
        self._update_cup_catch_state()
        if self._caught_on_pillow or self.check_success():
            self.plan_success = True
        self.save_freq = old_save_freq

        self.info["info"] = {
            "{A}": f"{self.CUP_MODEL}/base{self.cup_id}",
            "{B}": "266_pillow/base0",
            "{C}": "deep_wall_shelf",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        """Success only when PhysX left the cup sitting on the pillow.

        Table contact is a failure. No mid-air soft-seat / pose teleport counts.
        """
        if self._fell_on_table or self._cup_state == "fallen":
            return False
        if self.cup is None or self.pillow is None:
            return False
        if self._cup_resting_on_pillow():
            self._caught_on_pillow = True
            self._cup_state = "caught"
        if not self._caught_on_pillow and self._cup_state != "caught":
            return False
        if not self._cup_over_pillow():
            return False
        pillow_top = float(self.table_top + self.pillow_height)
        cup_bottom = self._cup_aabb_bottom()
        sitting_on_pillow = pillow_top - 0.012 <= cup_bottom <= pillow_top + 0.025
        touches_table = cup_bottom <= float(self.table_top) + 0.012
        return bool(sitting_on_pillow and not touches_table)

    def _cup_pillow_edge_distance(self) -> float:
        """Planar distance from cup to pillow footprint (0 if over / inside)."""
        if self.cup is None or self.pillow is None:
            return float("inf")
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        pp = np.array(self.pillow.get_pose().p, dtype=np.float64)
        tol = float(getattr(self, "pillow_catch_xy_tol", self.PILLOW_CATCH_XY_TOL))
        hx = float(self.pillow_half_xy[0]) + tol
        hy = float(self.pillow_half_xy[1]) + tol
        dx = abs(float(cp[0] - pp[0])) - hx
        dy = abs(float(cp[1] - pp[1])) - hy
        return float(np.hypot(max(0.0, dx), max(0.0, dy)))

    def get_score(self) -> float:
        """Partial score from cup–pillow miss distance (no table-bounce credit).

        Success → 1. Table contact / fallen → 0. Over pillow footprint but not
        seated → 0.5. Outside: ring of width = pillow half-diagonal, thirds
        → 0.75 / 0.5 / 0.25.
        """
        if self.cup is None or self.pillow is None:
            return 0.0
        if bool(getattr(self, "_fell_on_table", False)) or self._cup_state == "fallen":
            return 0.0
        if self._cup_touches_table():
            return 0.0
        if self.check_success():
            return 1.0
        if self._cup_over_pillow():
            return 0.5
        hx = float(self.pillow_half_xy[0])
        hy = float(self.pillow_half_xy[1])
        r = float(np.hypot(hx, hy))
        if r <= 1e-9:
            return 0.0
        d = self._cup_pillow_edge_distance()
        third = r / 3.0
        if d > r + 1e-9:
            return 0.0
        if d < third:
            return float(self.PARTIAL_MISS_SCORES[0])
        if d < 2.0 * third:
            return float(self.PARTIAL_MISS_SCORES[1])
        return float(self.PARTIAL_MISS_SCORES[2])

    def get_obs(self):
        obs = super().get_obs()
        obs["catch_cup"] = {
            "cup_state": str(self._cup_state),
            "fell_on_table": bool(self._fell_on_table),
            "caught_on_pillow": bool(self._caught_on_pillow),
            "pillow_placed": bool(self._pillow_placed),
            "traj_step": int(self._traj_step),
            "drop_x": float(self.drop_x),
            "landing": list(map(float, self._landing)),
            "cup_id": int(getattr(self, "cup_id", -1)),
            "arm_side": str(getattr(self, "arm_side", "right")),
            "roll_speed": float(getattr(self, "roll_speed", self.ROLL_SPEED_DEFAULT)),
            "n_shelf_cups": int(len(getattr(self, "shelf_cups", [])) + 1),
            "plant_scale": float(getattr(self, "plant_scale", self.PLANT_SCALE)),
            "pillow_xy": (
                list(map(float, self.pillow.get_pose().p[:2]))
                if self.pillow is not None else [0.0, 0.0]
            ),
            "partial_score": float(self.get_score()),
        }
        return obs
