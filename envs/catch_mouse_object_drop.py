"""Catch a shelf object shoved off by a scurrying mouse into a cushioned basket.

Scene: two-tier full-width wall shelf. Upper: random cups / wineglasses / beer
and wine bottles (mouse-width X gaps, no Y-lane blocking) plus plant/mug décor.
Lower: tissue (moved off the table) and a couple of random office props. The
robot grasps a custom open basket at the table midline (x≈0) and pick-and-places
it under the shelf lip. A kinematic mouse then runs a pre-planned, obstacle-free
route, turns in behind exactly one target, and physically shoves it forward
until it goes over the front edge. If a shove leaves the object on the shelf,
the mouse backs up and pushes again (up to a few retries). Mouse speed is
sampled from mean×0.7 to mean×1.56 (slow −30%, fast side +20% above the old
+30% cap).

Everything after the basket is placed is simulated, not animated: shelf objects
are free dynamic bodies the mouse can shove, the fall and the landing come out
of the solver, and success is read off the resting pose. Nothing is teleported
into place. The mouse starts scurrying at episode start while the arm places
the basket and does not pause for the catcher — placement is a race against
the knock.
"""

from ._office_base_task import Office_base_task
from .utils import *
from ._GLOBAL_CONFIGS import *
import json
import sapien
from collections import deque
import sapien.physx
import numpy as np
from pathlib import Path
from transforms3d.euler import euler2quat


class catch_mouse_object_drop(Office_base_task):
    """Mouse knocks one fragile shelf object; catch it in a pillow-lined basket."""

    # Near-miss ring outside the basket footprint (thirds of half-diagonal).
    PARTIAL_MISS_SCORES = (0.75, 0.5, 0.25)

    # Skip base6: silver metallic mug (wrong look / mesh orientation for this task).
    CUP_IDS = [0, 1, 2, 5, 3]
    WINEGLASS_IDS = [1, 2, 4]  # skip oversized base0 / tall base3
    BASKET_IDS = list(range(5))
    TISSUE_IDS = list(range(7))

    # Meshes are bottom-origin (local Y-up). Pose z = shelf surface, not half-height.
    # (modelname, id_pool or None→0, scale_mult, rolls, label)
    # upright_q: cups use RoboTwin [0.5]*4; glass/bottles match make_soup / pour_beer.
    DECOR_UPRIGHT_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
    OBJECT_CATALOG = [
        # (model, id_pool, scale_mult, rolls, label, upright_q_name)
        # Realistic sizes: cup ~7 cm, wineglass ~19 cm, beer ~23 cm, wine ~24 cm
        ("021_cup", CUP_IDS, 0.80, True, "cup", "twin"),
        ("088_wineglass", WINEGLASS_IDS, 0.75, False, "wineglass", "decor"),
        # Bottle smults are capped at spawn to fit inside the catch basket.
        ("255_beer_bottle", None, 0.85, True, "beer_bottle", "decor"),
        ("265_wine_bottle", None, 0.70, True, "wine_bottle", "decor"),
    ]

    PLANT_SCALE = 0.90          # doubled vs prior 0.45
    MUG_SCALE = 0.715           # prior 0.55 * 1.30
    N_OBJECTS_DEFAULT = 4
    MOUSE_SPEED_DEFAULT = 0.05  # 50% of prior 0.10 m/s mean
    MOUSE_SPEED_JITTER_FRAC = 0.30  # slow side: mean × (1 − frac)
    # Fast-side cap is (1 + frac) raised by this factor (e.g. 1.3 → 1.56).
    MOUSE_SPEED_FAST_SIDE_MULT = 1.20
    MOUSE_START_GAP_MIN = 0.15
    MOUSE_HALF_XY = 0.020       # mouse body clearance for route planning
    MOUSE_STANDOFF = 0.012      # gap behind the target before the shove begins
    MOUSE_GRID_RES = 0.01       # route search resolution on the shelf plate
    # If a shove leaves the target on the shelf, back up and push again.
    MOUSE_SHOVE_RETRIES = 4     # total shove attempts (1 first + up to 3 retries)
    MOUSE_SHOVE_OBSERVE_STEPS = 45  # wait after path end before deciding to retry
    MOUSE_RETRY_SPEED_CAP = 0.09  # soft retries — hard re-hits launch bottles
    MOUSE_RETRY_SPEED_MULT = 1.15
    # Kinematic mouse penetration can invent huge impulses; clamp while seated.
    SHELF_MAX_LINEAR_SPEED = 0.30   # m/s; tip/slide, not ballistic launch
    SHELF_MAX_ANGULAR_SPEED = 5.0   # rad/s
    FALL_MAX_LINEAR_SPEED = 2.5     # m/s once off the lip (gravity ok)
    # Masses: bottles were ~0.12 kg and flew when the mouse drove through them.
    MASS_BY_LABEL = {
        "cup": 0.16,
        "wineglass": 0.12,
        "beer_bottle": 0.38,
        "wine_bottle": 0.42,
    }
    LIN_DAMP_BY_LABEL = {
        "cup": 0.70,
        "wineglass": 0.85,
        "beer_bottle": 1.10,
        "wine_bottle": 1.20,
    }
    ANG_DAMP_BY_LABEL = {
        "cup": 0.80,
        "wineglass": 1.00,
        "beer_bottle": 1.40,
        "wine_bottle": 1.50,
    }
    # Clear gap between fragile footprints so the mouse can scurry between them.
    OBJ_X_GAP_MIN = 0.06
    SETTLE_STEPS = 420          # sim steps allowed for the object to come to rest
    # A bounce off the basket rim passes through near-zero speed at its apex, so
    # an instantaneous speed test reads it as a catch. Require the object to rest
    # on the cushion at this speed for this many consecutive steps instead.
    SETTLE_SPEED = 0.12
    SETTLE_DWELL_STEPS = 30
    # Objects bigger than the basket settle across the rim instead of dropping
    # in; require most of the footprint over the mouth rather than containment.
    CATCH_FOOTPRINT_FRAC = 0.5
    # A rim bounce passes through near-zero speed at its apex, so an
    # instantaneous speed test reads it as a catch. Require the object to be
    # at rest on the cushion for a continuous dwell instead.
    SETTLE_SPEED = 0.12         # |v| + 0.05|w| below this counts as at rest
    SETTLE_DWELL_STEPS = 30     # consecutive at-rest steps before a catch counts

    SHELF_WIDTH = 1.20
    SHELF_DEPTH = 0.30
    SHELF_THICK = 0.02
    SHELF_Y = 0.26
    SHELF_Z_ABOVE_TABLE = 0.30
    # Lower shelf tier for former table décor (tissue + random office props).
    # Shallower / set back toward the wall so the catch zone under the upper
    # lip stays clear for the basket + handle.
    SHELF_Z_LOWER_ABOVE_TABLE = 0.12
    SHELF_LOWER_DEPTH = 0.20
    SHELF_LOWER_Y = 0.31

    # Near-edge vs deep bands along shelf depth (y). Keep clear of the lip/back.
    EDGE_Y_FRAC = (0.10, 0.24)   # fraction of depth from front lip
    # Deep band stops well short of the back edge: the mouse has to fit behind
    # the target to shove it forward.
    DEEP_Y_FRAC = (0.38, 0.55)
    # Usable x span for fragile objects (mouse runway reserved outside this).
    SHELF_OBJ_X_LIM = (-0.36, 0.40)
    SHELF_OBJ_Z_EPS = 0.002      # sit bottoms just above the shelf plate

    # Lower-shelf office décor (tissue moved off the table + extras).
    KETTLE_IDS = list(range(6))
    ALARM_IDS = list(range(6))
    BOOK_IDS = [0, 1]
    SEAL_IDS = [0, 1, 2, 3, 4, 6]
    TISSUE_SCALE = 1.10
    KETTLE_SCALE = 0.95
    ALARM_SCALE = 0.95
    BOOK_SCALE = 0.75
    SEAL_SCALE = 0.90
    # Always place tissue; then sample this many extras from LOWER_EXTRA_CATALOG.
    N_LOWER_EXTRA_DEFAULT = 2
    LOWER_EXTRA_CATALOG = [
        # (model, id_pool, hx, hy, scale)
        ("091_kettle", KETTLE_IDS, 0.10, 0.10, KETTLE_SCALE),
        ("046_alarm-clock", ALARM_IDS, 0.08, 0.06, ALARM_SCALE),
        ("043_book", BOOK_IDS, 0.08, 0.06, BOOK_SCALE),
        ("100_seal", SEAL_IDS, 0.045, 0.045, SEAL_SCALE),
    ]

    BASKET_MODEL = "cushion_basket"  # procedural open basket (not 076_breadbasket)
    # The cushion is a part of the basket actor, not a separate body: a loose
    # pillow either interpenetrates the basket or has to be teleported along
    # with it, and both show up as objects popping between frames.
    CUSHION_HALF_Z = 0.013
    CUSHION_INSET = 0.006
    # Realistic catch basket ~23×19×7 cm (outer), center-origin Z-up.
    BASKET_HALF_XY_DEFAULT = [0.115, 0.095]
    BASKET_HEIGHT_DEFAULT = 0.070
    BASKET_WALL = 0.008
    # Grab handle over the robot-facing wall: a bar on two end posts. The bar is
    # raised so both fingers descend through free air before closing on it.
    BASKET_HANDLE_BAR_HALF = [0.034, 0.0065, 0.0065]
    BASKET_HANDLE_POST_HALF = [0.006, 0.0065, 0.01175]
    BASKET_HANDLE_CLEAR = 0.030   # bar center above the rim top
    # Contact frame sits below the bar center: the TCP settles ~1 cm high, so
    # this is what actually puts the fingers around the bar.
    BASKET_GRASP_Z_OFFSET = -0.010
    # Midline spawn so either arm can reach; y ∈ [current, current+10 cm].
    BASKET_START_Y_DEFAULT = -0.28
    BASKET_START_Y_SPAN = 0.10
    CATCH_XY_TOL = 0.035
    POST_PLACE_DWELL_DEFAULT = 14

    UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    PROP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    BASKET_UPRIGHT_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    MOUSE_COLOR = [0.45, 0.38, 0.32]
    MOUSE_EAR_COLOR = [0.72, 0.55, 0.55]
    PILLOW_COLOR = [0.92, 0.78, 0.72]
    BASKET_COLOR = [0.62, 0.42, 0.28]  # woven-wood tone
    BASKET_HANDLE_COLOR = [0.50, 0.34, 0.22]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_mouse_object_drop", {})
        self._loaded = False
        self.shelf_objects = []
        self.target = None
        self.target_idx = -1
        self.basket = None
        self.pillow = None
        self.mouse_parts = []
        self._target_rigid = None
        self._target_live = False
        self._obj_state = "parked"  # parked | falling | caught | fallen
        self._mouse_state = "idle"  # idle | running | done
        self._fell_on_table = False
        self._caught = False
        self._settle_steps = 0
        self._basket_placed = False
        self._holding_basket = False
        self._reset_metric_state()
        self._mouse_path = []
        self._mouse_cum = [0.0]
        self._mouse_path_len = 0.0
        self._mouse_shove_s = 0.0
        self._mouse_s = 0.0
        self._mouse_z = 0.0
        self._mouse_heading = 0.0
        self._allow_shove = True
        self._mouse_shove_attempts = 0
        self._shove_observe = 0
        self._mouse_obstacles = []
        self._decor_obstacles = []
        self._landing = np.zeros(3)
        self.arm_side = "right"
        self.target_upright_q = self.UPRIGHT_Q.copy()
        self.decor = []
        self._occ_shelf = []
        self._occ_shelf_lower = []
        self._occ_table = []
        super().setup_demo(**kwags)
        self._configure_observer_camera()
        # Mouse + shelf objects go live after settle (policy eval never calls play_once).
        self._allow_shove = True
        self._activate_target()
        self._release_mouse()

    # --------------------------------------------------------------- scene
    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Office table + wall with a deeper full-width two-tier shelf."""
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
        lower_y = float(self.SHELF_LOWER_Y)
        lower_half_y = 0.5 * float(self.SHELF_LOWER_DEPTH)
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
            sapien.Pose(p=[0.0, lower_y, lower_z]),
            half_size=[half_x, lower_half_y, half_z],
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
                sapien.Pose(p=[bx, lower_y + lower_half_y - 0.01, lower_z - 0.04]),
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
        # Bird's-eye that frames shelf + catch zone + both arms.
        camera_pos = np.array([0.05, -0.05, 1.85], dtype=np.float64)
        look_at = np.array([0.05, 0.12, 0.90], dtype=np.float64)
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

    def _disable_collision(self, entity):
        """Make an actor purely visual.

        The cushion is kinematic and rides inside the basket, so any residual
        overlap turns into solver impulses that pin (or eject) the basket the
        gripper is trying to lift.
        """
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if not isinstance(comp, sapien.physx.PhysxRigidBaseComponent):
                continue
            for shape in comp.get_collision_shapes():
                shape.set_collision_groups([0, 0, 0, 0])

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

    @staticmethod
    def _model_size(modelname, model_id, scale_mult=1.0):
        path = resolve_model_dir(modelname) / f"model_data{int(model_id)}.json"
        data = json.loads(path.read_text())
        size = (
            np.asarray(data["extents"], dtype=np.float64)
            * np.asarray(data["scale"], dtype=np.float64)
            * float(scale_mult)
        )
        # Mesh Y is typically the upright axis after PROP_UPRIGHT_Q.
        height = float(size[1])
        radius = 0.5 * float(0.5 * (size[0] + size[2]))
        half_x = 0.5 * float(size[0])
        half_y = 0.5 * float(size[2])  # footprint along shelf depth after upright
        return height, max(0.02, radius), half_x, half_y

    def _basket_inner_half_xy(self) -> tuple[float, float]:
        """Cushion-pad half-size (same clearance as the catch volume)."""
        hx, hy = [float(v) for v in getattr(self, "basket_half_xy", self.BASKET_HALF_XY_DEFAULT)]
        inset = float(self.BASKET_WALL) + float(self.CUSHION_INSET)
        return max(0.02, hx - inset), max(0.02, hy - inset)

    def _scale_object_to_basket(
        self, modelname, model_id, scale_mult: float, label: str,
    ) -> float:
        """Cap bottle scale so the mesh fits inside the catch basket when tipped.

        Cups / glasses are left alone. Bottles are limited by the cushion pad:
        length ≤ long inner side, diameter ≤ short inner side (with margin).
        """
        smult = float(scale_mult)
        if "bottle" not in str(label):
            return smult
        h1, r1, _, _ = self._model_size(modelname, int(model_id), 1.0)
        diam1 = 2.0 * float(r1)
        ihx, ihy = self._basket_inner_half_xy()
        margin = 0.90
        max_len = 2.0 * max(ihx, ihy) * margin
        max_diam = 2.0 * min(ihx, ihy) * margin
        if h1 > 1e-6:
            smult = min(smult, max_len / h1)
        if diam1 > 1e-6:
            smult = min(smult, max_diam / diam1)
        return float(max(0.25, smult))

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

    def _spawn_static_shelf(
        self, modelname, model_id, p, hx, hy, scale_mult=1.0, surface="shelf",
    ):
        """Static décor seated on upper/lower shelf (bottom-origin, non-overlapping)."""
        cx, cy = float(p[0]), float(p[1])
        occ = self._occ_shelf_lower if surface == "shelf_lower" else self._occ_shelf
        if not self._footprint_ok(occ, cx, cy, hx, hy, pad=0.015):
            return None
        pose = sapien.Pose(
            [cx, cy, float(p[2])], self.PROP_UPRIGHT_Q.tolist(),
        )
        try:
            data = json.loads(
                (resolve_model_dir(modelname) / f"model_data{int(model_id)}.json").read_text()
            )
            base = data.get("scale") or [1.0, 1.0, 1.0]
            scale = [float(s) * float(scale_mult) for s in base]
        except Exception:
            scale = [float(scale_mult)] * 3
        actor = self._create_scaled_static_object(
            modelname, int(model_id), pose, list(scale), collision=False,
        )
        self._reserve(occ, cx, cy, hx, hy, pad=0.015)
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

    def _x_gap_ok(self, placed, cx, hx, gap):
        """True if ``cx`` leaves a clear mouse-width gap from every placed object."""
        for ox, _oy, ohx, _ohy in placed:
            if abs(cx - ox) < float(ohx) + float(hx) + float(gap):
                return False
        return True

    def _y_lane_clear(self, placed, cx, cy, hx, hy):
        """Reject stacking two objects on the same X corridor (Y blocking)."""
        for ox, oy, ohx, ohy in placed:
            # Overlapping X corridors → must not sit one in front of the other.
            if abs(cx - ox) < float(ohx) + float(hx) + 0.01:
                if abs(cy - oy) > 0.5 * (float(ohy) + float(hy)):
                    return False
        return True

    def _lower_shelf_ylim(self):
        """World-Y bounds of the set-back lower plate (not the upper plate)."""
        cy = float(self.SHELF_LOWER_Y)
        hy = 0.5 * float(self.SHELF_LOWER_DEPTH)
        return cy - hy, cy + hy

    def _load_lower_shelf_decor(self, z_seat, y0, y1):
        """Tissue (off the table) + a couple of random non-overlapping office props."""
        catalog = list(self.LOWER_EXTRA_CATALOG)
        n_extra = int(self._cfg.get("n_lower_extra", self.N_LOWER_EXTRA_DEFAULT))
        n_extra = int(np.clip(n_extra, 1, len(catalog)))
        order = list(np.random.permutation(len(catalog)))
        extras = [catalog[int(i)] for i in order[:n_extra]]
        specs = [
            ("023_tissue-box", self.TISSUE_IDS, 0.07, 0.055, self.TISSUE_SCALE),
            *extras,
        ]
        # Keep décor on the set-back lower plate, not the upper-shelf Y span.
        ly0, ly1 = self._lower_shelf_ylim()
        y_mid = float(0.5 * (ly0 + ly1))
        for model, id_pool, hx, hy, scale in specs:
            placed = False
            for _ in range(60):
                x = float(np.random.uniform(-0.48, 0.48))
                y = float(np.random.uniform(ly0 + 0.03, ly1 - 0.03))
                actor = self._spawn_static_shelf(
                    model, int(np.random.choice(list(id_pool))),
                    [x, y, z_seat],
                    hx=hx, hy=hy, scale_mult=scale, surface="shelf_lower",
                )
                if actor is not None:
                    placed = True
                    break
            if not placed:
                # Spread fallbacks across the lower plate.
                for fx in (-0.38, 0.0, 0.38, -0.20, 0.20):
                    actor = self._spawn_static_shelf(
                        model, int(np.random.choice(list(id_pool))),
                        [float(fx), y_mid, z_seat],
                        hx=hx, hy=hy, scale_mult=scale, surface="shelf_lower",
                    )
                    if actor is not None:
                        break

    def _upright_q(self, name="twin"):
        if name == "decor":
            return self.DECOR_UPRIGHT_Q.copy()
        return self.UPRIGHT_Q.copy()

    # --------------------------------------------------------------- mouse
    def _spawn_mouse(self, x, y, z, heading):
        """Procedural kinematic rodent (body + head + ears + tail).

        Part offsets are stored nose-along-local-+X and rotated by the heading
        every step, so the mouse turns to face where it is going and shoves the
        target nose-first instead of sliding sideways into it.
        """
        self.mouse_parts = []
        body = create_box(
            self.scene,
            sapien.Pose(p=[x, y, z]),
            half_size=[0.028, 0.016, 0.014],
            color=self.MOUSE_COLOR,
            is_static=False,
            name="mouse_body",
        )
        try:
            body.set_mass(0.03)
        except Exception:
            pass
        self._make_kinematic(body)
        self.mouse_parts.append(("body", body, np.array([0.0, 0.0, 0.0])))

        head = create_sphere(
            self.scene,
            sapien.Pose(p=[x + 0.032, y, z + 0.004]),
            radius=0.013,
            color=self.MOUSE_COLOR,
            is_static=False,
            name="mouse_head",
        )
        self._make_kinematic(head)
        # Visual only — multi-part kinematic contact stacked impulses and launched
        # tall bottles. Body alone does the shove.
        self._disable_collision(head)
        self.mouse_parts.append(("head", head, np.array([0.032, 0.0, 0.004])))

        for side, name in ((1.0, "mouse_ear_l"), (-1.0, "mouse_ear_r")):
            ear = create_sphere(
                self.scene,
                sapien.Pose(p=[x + 0.018, y + side * 0.012, z + 0.016]),
                radius=0.007,
                color=self.MOUSE_EAR_COLOR,
                is_static=False,
                name=name,
            )
            self._make_kinematic(ear)
            self._disable_collision(ear)
            self.mouse_parts.append(
                (name, ear, np.array([0.018, side * 0.012, 0.016]))
            )

        tail = create_box(
            self.scene,
            sapien.Pose(p=[x - 0.042, y, z]),
            half_size=[0.018, 0.004, 0.004],
            color=[0.55, 0.40, 0.40],
            is_static=False,
            name="mouse_tail",
        )
        self._make_kinematic(tail)
        self._disable_collision(tail)
        self.mouse_parts.append(("tail", tail, np.array([-0.042, 0.0, 0.0])))
        self.mouse = body
        self.mouse_pos = np.array([x, y, z], dtype=np.float64)
        self._set_mouse_pose([x, y, z], heading=heading)

    def _set_mouse_pose(self, xyz, heading=None):
        self.mouse_pos = np.asarray(xyz, dtype=np.float64).copy()
        if heading is None:
            heading = float(getattr(self, "_mouse_heading", 0.0))
        self._mouse_heading = float(heading)
        self.mouse_facing = float(np.cos(heading))
        c, s = float(np.cos(heading)), float(np.sin(heading))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        quat = euler2quat(0.0, 0.0, float(heading), axes="sxyz")
        for _name, part, offset in self.mouse_parts:
            off = np.asarray(offset, dtype=np.float64).copy()
            off[:2] = rot @ off[:2]
            p = self.mouse_pos + off
            self._set_entity_pose(part, sapien.Pose(p.tolist(), list(quat)))

    # ------------------------------------------------------------ mouse route
    def _blocked(self, x, y):
        """True if a mouse-sized footprint at (x, y) overlaps décor/other items."""
        mh = self.MOUSE_HALF_XY
        for ox, oy, hx, hy in self._mouse_obstacles:
            if abs(x - ox) <= hx + mh and abs(y - oy) <= hy + mh:
                return True
        return False

    def _segment_clear(self, p0, p1):
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(dist / 0.01) + 1)
        for i in range(n + 1):
            q = p0 + (p1 - p0) * (i / n)
            if self._blocked(float(q[0]), float(q[1])):
                return False
        return True

    @staticmethod
    def _chaikin(points, iterations=2):
        """Corner-cutting so the scurry reads as a curve, not a set of turns."""
        pts = [np.asarray(p, dtype=np.float64) for p in points]
        for _ in range(int(iterations)):
            if len(pts) < 3:
                break
            out = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                out.append(0.75 * a + 0.25 * b)
                out.append(0.25 * a + 0.75 * b)
            out.append(pts[-1])
            pts = out
        return pts

    def _mouse_standoff(self, ty, thy):
        """Point directly behind the target where the shove starts."""
        y_hi = float(self.shelf_plate_ylim[1] - self.MOUSE_HALF_XY - 0.010)
        return float(min(ty + thy + self.MOUSE_HALF_XY + self.MOUSE_STANDOFF, y_hi))

    def _plan_mouse_route(self, start_x, tx, ty, thx, thy):
        """Obstacle-free route ending in a straight shove toward the shelf lip.

        The mouse has to end up *behind* the target and drive forward, otherwise
        it would knock the object further onto the shelf instead of over the
        front edge. A grid search finds the way around décor — a straight
        corridor often does not exist once the plant is in the way.
        """
        y0, y1 = self.shelf_plate_ylim
        x0, x1 = self.shelf_plate_xlim
        mh = self.MOUSE_HALF_XY
        res = float(self.MOUSE_GRID_RES)

        gx = np.arange(x0 + mh, x1 - mh + 1e-9, res)
        gy = np.arange(y0 + mh + 0.005, y1 - mh - 0.005 + 1e-9, res)
        free = np.ones((len(gx), len(gy)), dtype=bool)
        for i, xv in enumerate(gx):
            for j, yv in enumerate(gy):
                if self._blocked(float(xv), float(yv)):
                    free[i, j] = False

        def nearest_free(x, y):
            i = int(np.clip(np.searchsorted(gx, x), 0, len(gx) - 1))
            j = int(np.clip(np.searchsorted(gy, y), 0, len(gy) - 1))
            if free[i, j]:
                return i, j
            best, best_d = None, 1e9
            for a in range(len(gx)):
                for b in range(len(gy)):
                    if not free[a, b]:
                        continue
                    d = (a - i) ** 2 + (b - j) ** 2
                    if d < best_d:
                        best, best_d = (a, b), d
            return best

        push_y0 = self._mouse_standoff(ty, thy)
        push_y1 = float(self.shelf_front_y + mh - 0.010)

        src = nearest_free(start_x, y0 + 0.05)
        dst = nearest_free(tx, push_y0)
        if src is None or dst is None:
            return [(start_x, push_y0), (tx, push_y0)], push_y1, False

        prev = -np.ones((len(gx), len(gy), 2), dtype=np.int32)
        seen = np.zeros_like(free)
        seen[src] = True
        queue = deque([src])
        while queue:
            ci, cj = queue.popleft()
            if (ci, cj) == dst:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < len(gx) and 0 <= nj < len(gy)):
                    continue
                if seen[ni, nj] or not free[ni, nj]:
                    continue
                if di and dj and not (free[ci + di, cj] and free[ci, cj + dj]):
                    continue  # no corner cutting through a gap
                seen[ni, nj] = True
                prev[ni, nj] = (ci, cj)
                queue.append((ni, nj))

        if not seen[dst]:
            return [(start_x, push_y0), (tx, push_y0)], push_y1, False

        cells = []
        cur = dst
        while cur != src:
            cells.append(cur)
            pi, pj = prev[cur[0], cur[1]]
            cur = (int(pi), int(pj))
        cells.append(src)
        cells.reverse()
        pts = [np.array([gx[i], gy[j]], dtype=np.float64) for i, j in cells]

        # Collapse the grid staircase into a few long legs.
        route = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._segment_clear(pts[i], pts[j]):
                j -= 1
            route.append(pts[j])
            i = j
        route[-1] = np.array([tx, push_y0], dtype=np.float64)
        return route, push_y1, True

    def _set_mouse_path(self, route, push_y1, z):
        """Arc-length path: approach legs, then a dead-straight shove to the lip.

        Chaikin is only applied when every new corner stays clear of décor —
        otherwise a smoothed bend can cut through the plant.
        """
        raw = [np.asarray(p, dtype=np.float64) for p in route]
        smoothed = self._chaikin(raw, iterations=2)
        if all(not self._blocked(float(p[0]), float(p[1])) for p in smoothed):
            pts = smoothed
        else:
            pts = raw
        pts.append(np.array([pts[-1][0], push_y1], dtype=np.float64))

        cleaned = [pts[0]]
        for q in pts[1:]:
            if float(np.linalg.norm(q - cleaned[-1])) > 1e-4:
                cleaned.append(q)
        self._mouse_path = cleaned
        seg = [0.0]
        for a, b in zip(cleaned[:-1], cleaned[1:]):
            seg.append(seg[-1] + float(np.linalg.norm(b - a)))
        self._mouse_cum = seg
        self._mouse_path_len = float(seg[-1])
        # Path length at the start of the final shove leg (debug / retries).
        self._mouse_shove_s = float(seg[-2]) if len(seg) >= 2 else 0.0
        self._mouse_s = 0.0
        self._mouse_z = float(z)

    def _mouse_point_at(self, s):
        """Arc-length lookup → constant speed, no per-step jumps."""
        pts = self._mouse_path
        cum = self._mouse_cum
        s = float(np.clip(s, 0.0, self._mouse_path_len))
        i = int(np.searchsorted(cum, s, side="right")) - 1
        i = int(np.clip(i, 0, len(pts) - 2))
        span = max(cum[i + 1] - cum[i], 1e-9)
        f = (s - cum[i]) / span
        p = pts[i] + (pts[i + 1] - pts[i]) * f
        d = pts[i + 1] - pts[i]
        heading = float(np.arctan2(d[1], d[0]))
        return p, heading

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        base_speed = float(c.get("mouse_speed", self.MOUSE_SPEED_DEFAULT))
        jitter = float(c.get("mouse_speed_jitter_frac", self.MOUSE_SPEED_JITTER_FRAC))
        jitter = float(np.clip(jitter, 0.0, 0.95))
        fast_mult = float(c.get(
            "mouse_speed_fast_side_mult", self.MOUSE_SPEED_FAST_SIDE_MULT,
        ))
        self.mouse_speed_mean = float(base_speed)
        speed_lo = base_speed * (1.0 - jitter)
        speed_hi = base_speed * (1.0 + jitter) * max(1.0, fast_mult)
        self.mouse_speed = float(np.random.uniform(speed_lo, speed_hi))
        self.mouse_start_gap_min = float(
            c.get("mouse_start_gap_min", self.MOUSE_START_GAP_MIN)
        )
        self.catch_xy_tol = float(c.get("catch_xy_tol", self.CATCH_XY_TOL))
        self.post_place_dwell = int(c.get(
            "post_place_dwell", self.POST_PLACE_DWELL_DEFAULT,
        ))
        self.basket_half_xy = list(c.get("basket_half_xy", self.BASKET_HALF_XY_DEFAULT))
        self.basket_height = float(c.get("basket_height", self.BASKET_HEIGHT_DEFAULT))
        n_objects = int(c.get("n_objects", self.N_OBJECTS_DEFAULT))
        n_objects = int(np.clip(n_objects, 3, 5))
        obj_gap = float(c.get("obj_x_gap_min", self.OBJ_X_GAP_MIN))

        self.table_top = 0.74 + float(self.table_z_bias)
        self._measure_shelf_plate()
        self.shelf_front_y = float(self.shelf_plate_ylim[0])
        self.shelf_back_y = float(self.shelf_plate_ylim[1])
        self.shelf_z_surf = float(self.shelf_plate_z)
        self.shelf_lower_surf = float(
            getattr(self, "shelf_lower_z", self.shelf_z_surf - 0.18)
        )
        depth = self.SHELF_DEPTH
        y0, y1 = self.shelf_plate_ylim
        x0, x1 = self.shelf_plate_xlim

        self.decor = []
        self._occ_shelf = []
        self._occ_shelf_lower = []
        self._occ_table = []
        self.shelf_objects = []
        self._decor_obstacles = []  # (cx, cy, hx, hy) for mouse avoidance
        z_seat = self.shelf_z_surf + self.SHELF_OBJ_Z_EPS
        z_lower = self.shelf_lower_surf + self.SHELF_OBJ_Z_EPS
        y_back = float(y0 + 0.72 * depth)
        y_mid = float(y0 + 0.52 * depth)

        # Upper-shelf décor first (randomized plant / mug); tissue lives below.
        plant_x = float(np.random.uniform(-0.48, -0.34))
        plant = self._spawn_static_shelf(
            "120_plant", 0,
            [plant_x, y_back, z_seat],
            hx=0.09, hy=0.09, scale_mult=self.PLANT_SCALE,
        )
        if plant is not None:
            self._decor_obstacles.append((plant_x, y_back, 0.09, 0.09))
        else:
            for px in (-0.42, -0.34, -0.28):
                plant = self._spawn_static_shelf(
                    "120_plant", 0,
                    [px, y_back, z_seat],
                    hx=0.085, hy=0.085, scale_mult=self.PLANT_SCALE,
                )
                if plant is not None:
                    self._decor_obstacles.append((px, y_back, 0.085, 0.085))
                    break
        if np.random.rand() < 0.85:
            for _ in range(30):
                mug_x = float(np.random.uniform(-0.28, 0.40))
                mug_y = float(y0 + np.random.uniform(0.58, 0.78) * depth)
                mug = self._spawn_static_shelf(
                    "039_mug", int(np.random.randint(0, 13)),
                    [mug_x, mug_y, z_seat],
                    hx=0.055, hy=0.055, scale_mult=self.MUG_SCALE,
                )
                if mug is not None:
                    self._decor_obstacles.append((mug_x, mug_y, 0.055, 0.055))
                    break

        # Former table props → lower shelf (tissue + random office extras).
        self._load_lower_shelf_decor(z_lower, y0, y1)

        # Narrow mouse corridors at the far ends (mid-depth only — décor can sit behind).
        end_pad = 0.07
        self._reserve(
            self._occ_shelf, x0 + end_pad, y_mid, end_pad, 0.06, pad=0.015,
        )
        self._reserve(
            self._occ_shelf, x1 - end_pad, y_mid, end_pad, 0.06, pad=0.015,
        )

        # Prefer distinct fragile types (cup / wineglass / beer / wine).
        catalog = list(self.OBJECT_CATALOG)
        order = np.random.permutation(len(catalog))
        choices = [catalog[int(i)] for i in order[: min(n_objects, len(catalog))]]
        while len(choices) < n_objects:
            choices.append(catalog[int(np.random.randint(0, len(catalog)))])

        # Mix near-edge / deep; types and exact XY are randomized per seed.
        n_edge = max(1, n_objects // 2)
        modes = (["edge"] * n_edge) + (["deep"] * (n_objects - n_edge))
        np.random.shuffle(modes)
        type_mode = list(zip(choices, modes))
        np.random.shuffle(type_mode)

        x_lo, x_hi = self.SHELF_OBJ_X_LIM
        x_lo = max(x_lo, -0.30)
        placed_objs = []  # (cx, cy, hx, hy) for gap / Y-lane checks

        for (model, id_pool, smult, rolls, label, uq_name), mode in type_mode:
            model_id = (
                int(np.random.choice(id_pool)) if id_pool is not None else 0
            )
            smult = self._scale_object_to_basket(model, model_id, smult, label)
            height, radius, hx, hy = self._model_size(model, model_id, smult)
            hx = max(hx, 0.028) + 0.012
            hy = max(hy, 0.028) + 0.012
            uq = self._upright_q(uq_name)

            placed = False
            cx = cy = None
            for _try in range(80):
                cx = float(np.random.uniform(x_lo, x_hi))
                if mode == "edge":
                    f0, f1 = self.EDGE_Y_FRAC
                else:
                    f0, f1 = self.DEEP_Y_FRAC
                cy = float(y0 + np.random.uniform(f0, f1) * depth)
                if not self._x_gap_ok(placed_objs, cx, hx, obj_gap):
                    continue
                if not self._y_lane_clear(placed_objs, cx, cy, hx, hy):
                    continue
                if self._footprint_ok(self._occ_shelf, cx, cy, hx, hy, pad=0.015):
                    placed = True
                    break
            if not placed:
                # Fallback: linspace-ish slot with the gap constraint.
                for t in np.linspace(0.0, 1.0, n_objects + 2)[1:-1]:
                    cx = float(x_lo + t * (x_hi - x_lo))
                    f0, f1 = self.EDGE_Y_FRAC if mode == "edge" else self.DEEP_Y_FRAC
                    cy = float(y0 + 0.5 * (f0 + f1) * depth)
                    if (
                        self._x_gap_ok(placed_objs, cx, hx, obj_gap * 0.7)
                        and self._y_lane_clear(placed_objs, cx, cy, hx, hy)
                        and self._footprint_ok(
                            self._occ_shelf, cx, cy, hx, hy, pad=0.01,
                        )
                    ):
                        placed = True
                        break
            if not placed:
                continue

            # Bottom of mesh sits on the shelf plate (meshes are Y-up, bottom-origin).
            cz = z_seat
            pose = sapien.Pose([cx, cy, cz], uq.tolist())
            actor = create_actor(
                self,
                pose=pose,
                modelname=model,
                model_id=model_id,
                convex=True,
                is_static=False,
                scale_mult=float(smult),
            )
            if actor is None:
                continue
            try:
                actor.set_mass(float(self.MASS_BY_LABEL.get(label, 0.16)))
            except Exception:
                pass
            self._make_kinematic(actor)
            self._set_entity_pose(actor, pose)
            self._reserve(self._occ_shelf, cx, cy, hx, hy, pad=0.015)
            placed_objs.append((cx, cy, hx, hy))
            entry = {
                "actor": actor,
                "model": model,
                "model_id": int(model_id),
                "label": label,
                "rolls": bool(rolls),
                "mode": mode,
                "upright_q": uq,
                "start": np.array([cx, cy, cz], dtype=np.float64),
                "height": float(height),
                "radius": float(radius),
                "hx": float(hx),
                "hy": float(hy),
            }
            self.shelf_objects.append(entry)

        if len(self.shelf_objects) < 2:
            # Hard fallback: one cup near edge on the right.
            model, id_pool, smult, rolls, label, uq_name = self.OBJECT_CATALOG[0]
            model_id = int(np.random.choice(id_pool))
            height, radius, hx, hy = self._model_size(model, model_id, smult)
            uq = self._upright_q(uq_name)
            cx, cy = 0.22, float(y0 + 0.18 * depth)
            cz = z_seat
            pose = sapien.Pose([cx, cy, cz], uq.tolist())
            actor = create_actor(
                self, pose=pose, modelname=model, model_id=model_id,
                convex=True, is_static=False, scale_mult=float(smult),
            )
            actor.set_mass(float(self.MASS_BY_LABEL.get(label, 0.16)))
            self._make_kinematic(actor)
            self.shelf_objects = [{
                "actor": actor, "model": model, "model_id": model_id,
                "label": label, "rolls": rolls, "mode": "edge",
                "upright_q": uq,
                "start": np.array([cx, cy, cz]), "height": height,
                "radius": radius, "hx": hx, "hy": hy,
            }]

        # Choose the knock target uniformly among objects the mouse can line up
        # behind (either half of the shelf). Preferring the right side made
        # left-side falls almost never appear.
        reachable = [
            i for i, e in enumerate(self.shelf_objects)
            if self._mouse_standoff(float(e["start"][1]), float(e["hy"]))
            >= float(e["start"][1]) + float(e["hy"]) + self.MOUSE_HALF_XY
        ]
        if not reachable:
            reachable = list(range(len(self.shelf_objects)))
        self.target_idx = int(reachable[int(np.random.randint(0, len(reachable)))])
        self.target_info = self.shelf_objects[self.target_idx]
        self.target = self.target_info["actor"]
        self.target_start = self.target_info["start"].copy()
        self.target_radius = float(self.target_info["radius"])
        self.target_rolls = bool(self.target_info["rolls"])
        self.target_mode = str(self.target_info["mode"])
        self.target_label = str(self.target_info["label"])
        self.target_upright_q = np.asarray(
            self.target_info.get("upright_q", self.UPRIGHT_Q), dtype=np.float64,
        )
        # Landing = straight down off the shelf lip. The mouse shoves the object
        # forward at a known speed, so the drop is short and predictable; the
        # basket is placed there and physics does the rest (no scripted fall).
        tx = float(self.target_start[0])
        ty = float(self.target_start[1])
        self.drop_x = tx

        # An object shoved over the lip topples and drops almost straight down,
        # so the basket's back edge is tucked just under the shelf rather than
        # led out in front of it.
        land_y = float(self.shelf_front_y + 0.035 - self.basket_half_xy[1])
        self._landing = np.array(
            [tx, land_y, self.table_top + self.basket_height], dtype=np.float64,
        )

        # Clear landing strip under the shelf front so nothing else spawns there.
        self._reserve(
            self._occ_table, tx, land_y,
            self.basket_half_xy[0] + 0.02, self.basket_half_xy[1] + 0.02,
            pad=0.02,
        )

        # Obstacles = every non-target shelf actor + décor (plant/mug).
        self._mouse_obstacles = list(self._decor_obstacles)
        for i, e in enumerate(self.shelf_objects):
            if i == self.target_idx:
                continue
            self._mouse_obstacles.append(
                (float(e["start"][0]), float(e["start"][1]),
                 float(e["hx"]), float(e["hy"]))
            )

        # Mouse start side is randomized; both ends are tried if the preferred
        # side has no clear route around décor.
        z_mouse = self.shelf_z_surf + 0.016
        ends = [("left", float(x0 + 0.06)), ("right", float(x1 - 0.06))]
        if float(np.random.rand()) < 0.5:
            ends.reverse()
        # Prefer a start that is not already on top of the target.
        ends.sort(key=lambda e: abs(tx - e[1]) < self.mouse_start_gap_min)

        route = push_y1 = None
        viable = []
        for end_name, mx in ends:
            r, py, found = self._plan_mouse_route(
                mx, tx, ty,
                float(self.target_info["hx"]), float(self.target_info["hy"]),
            )
            if found:
                viable.append((end_name, r, py))
        if viable:
            end_name, route, push_y1 = viable[int(np.random.randint(0, len(viable)))]
            self.mouse_end = end_name
        else:
            # No verified clear route — fall back to the preferred end anyway.
            end_name, mx = ends[0]
            route, push_y1, _ = self._plan_mouse_route(
                mx, tx, ty,
                float(self.target_info["hx"]), float(self.target_info["hy"]),
            )
            self.mouse_end = end_name
        self._set_mouse_path(route, push_y1, z_mouse)
        p0, h0 = self._mouse_point_at(0.0)
        self._spawn_mouse(float(p0[0]), float(p0[1]), z_mouse, h0)

        # Basket on the table midline (x≈0) so either arm can reach it; the
        # placing arm still follows the knock-target side for the landing.
        self.basket_id = 0
        basket_y_lo = float(c.get("basket_start_y", self.BASKET_START_Y_DEFAULT))
        basket_y_span = float(c.get("basket_start_y_span", self.BASKET_START_Y_SPAN))
        self.arm_side = "right" if float(self.drop_x) >= 0.0 else "left"
        basket_x, basket_y = self._sample_basket_start(
            y_lo=basket_y_lo,
            y_hi=basket_y_lo + basket_y_span,
        )
        hz = 0.5 * float(self.basket_height)
        self.basket_hz = hz
        self.basket_start = np.array(
            [basket_x, basket_y, self.table_top + hz], dtype=np.float64,
        )
        self._reserve(
            self._occ_table, basket_x, basket_y,
            self.basket_half_xy[0], self.basket_half_xy[1],
        )

        basket_pose = sapien.Pose(
            self.basket_start.tolist(), self.BASKET_UPRIGHT_Q.tolist(),
        )
        self.basket = self._create_cushion_basket(basket_pose)
        self.add_prohibit_area(self.basket, padding=0.04)

        self.pillow = None  # cushion is a part of the basket actor now
        self._obj_state = "parked"
        self._mouse_state = "idle"
        self._fell_on_table = False
        self._caught = False
        self._settle_steps = 0
        self._basket_placed = False
        self._holding_basket = False
        self._reset_metric_state()
        self._target_live = False
        self._allow_shove = True
        self._mouse_shove_attempts = 0
        self._shove_observe = 0
        self._loaded = True

    def _create_cushion_basket(self, pose: sapien.Pose) -> Actor:
        """Open wooden basket carrying a raised grab handle on the robot side.

        ``076_breadbasket`` ships with an empty ``contact_points_pose`` list, so
        there is no way to grasp it; this builds a procedural Z-up catcher
        (~20×15×7 cm) plus a handle bar held clear of the rim by two end posts.
        The clearance matters: a rim-flush lip leaves the inner finger nothing to
        close against, so the gripper shuts on air.
        """
        hx, hy = [float(v) for v in self.basket_half_xy]
        hz = float(self.basket_hz)
        wt = float(self.BASKET_WALL)
        bhx, bhy, bhz = [float(v) for v in self.BASKET_HANDLE_BAR_HALF]
        phx, phy, phz = [float(v) for v in self.BASKET_HANDLE_POST_HALF]

        # Handle sits directly over the −Y (robot-facing) wall so both fingers
        # descend through open air on either side of the bar.
        handle_y = -(hy - wt * 0.5)
        bar_z = hz + float(self.BASKET_HANDLE_CLEAR)
        post_x = bhx - phx
        post_z = 0.5 * (hz + (bar_z - bhz))
        self._basket_handle_local = np.array([0.0, handle_y, bar_z], dtype=np.float64)

        floor_hz = wt * 0.5
        side_hz = hz - floor_hz
        side_z = -hz + floor_hz + side_hz
        body_parts = [
            # Floor
            (sapien.Pose([0.0, 0.0, -hz + floor_hz]), [hx, hy, floor_hz]),
            # +X / −X walls
            (sapien.Pose([hx - wt * 0.5, 0.0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([-hx + wt * 0.5, 0.0, side_z]), [wt * 0.5, hy, side_hz]),
            # +Y / −Y walls
            (sapien.Pose([0.0, hy - wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
            (sapien.Pose([0.0, -hy + wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
        ]
        # Cushion pad built into the basket: the object lands on this.
        chz = float(self.CUSHION_HALF_Z)
        inset = float(self.CUSHION_INSET)
        cushion_z = -hz + wt + chz
        self._cushion_top_local = float(cushion_z + chz)
        self._cushion_half_xy = [hx - wt - inset, hy - wt - inset]
        cushion_parts = [
            (sapien.Pose([0.0, 0.0, cushion_z]),
             [hx - wt - inset, hy - wt - inset, chz]),
        ]
        handle_parts = [
            (sapien.Pose([post_x, handle_y, post_z]), [phx, phy, phz]),
            (sapien.Pose([-post_x, handle_y, post_z]), [phx, phy, phz]),
            (sapien.Pose([0.0, handle_y, bar_z]), [bhx, bhy, bhz]),
        ]
        parts = body_parts + cushion_parts + handle_parts

        wood = sapien.render.RenderMaterial(base_color=[*self.BASKET_COLOR, 1.0])
        wood.roughness = 0.78
        wood.metallic = 0.0
        handle_mat = sapien.render.RenderMaterial(
            base_color=[*self.BASKET_HANDLE_COLOR, 1.0]
        )
        handle_mat.roughness = 0.72

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        phys = self.scene.create_physical_material(0.95, 0.90, 0.0)
        for local_pose, half in parts:
            builder.add_box_collision(
                pose=local_pose, half_size=list(half), material=phys,
            )
        builder.set_initial_pose(pose)
        entity = builder.build(name="cushion_basket")

        cushion_mat = sapien.render.RenderMaterial(
            base_color=[*self.PILLOW_COLOR, 1.0]
        )
        cushion_mat.roughness = 0.95

        n_body = len(body_parts)
        n_cushion = len(cushion_parts)
        render_body = sapien.render.RenderBodyComponent()
        for i, (local_pose, half) in enumerate(parts):
            if i < n_body:
                mat = wood
            elif i < n_body + n_cushion:
                mat = cushion_mat
            else:
                mat = handle_mat
            shape = sapien.render.RenderShapeBox(list(half), mat)
            shape.set_local_pose(local_pose)
            render_body.attach(shape)
        entity.add_component(render_body)

        # Only the two frames whose fingers close along ±Y are authored: an
        # X-closing approach would straddle the bar lengthwise and hit the posts.
        gy = float(handle_y)
        gz = float(bar_z + self.BASKET_GRASP_Z_OFFSET)
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [hx * 2.0, hy * 2.0, hz * 2.0],
            "scale": [1.0, 1.0, 1.0],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[1, 0, 0, 0.0], [0, 0, -1, gy], [0, 1, 0, gz], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, gy], [0, 1, 0, gz], [0, 0, 0, 1]],
            ],
            "contact_points_group": [[0, 1]],
            "contact_points_mask": [True],
            "contact_points_description": ["Front grab handle, grasp from above."],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }
        basket = Actor(entity, data, mass=0.22)
        for comp in basket.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    comp.set_linear_damping(4.0)
                    comp.set_angular_damping(8.0)
                except Exception:
                    pass
                break
        return basket

    def _sample_basket_start(self, y_lo, y_hi):
        """Spawn the basket on the table midline so both arms can reach it.

        X stays near 0; Y is randomized between the prior near-robot pose and
        10 cm farther toward the shelf. The placing arm is chosen from
        ``drop_x`` separately.
        """
        hx = float(self.basket_half_xy[0])
        hy = float(self.basket_half_xy[1])
        lo = float(min(y_lo, y_hi))
        hi = float(max(y_lo, y_hi))
        for _ in range(40):
            bx = float(np.random.uniform(-0.03, 0.03))
            by = float(np.random.uniform(lo, hi))
            if self._footprint_ok(self._occ_table, bx, by, hx, hy):
                return bx, by
        return 0.0, float(0.5 * (lo + hi))

    # ----------------------------------------------------------- kinematics
    def _release_actor_to_physics(
        self, actor, *, mass=None, lin_damp=0.70, ang_damp=0.80,
    ):
        """Make a shelf actor a free dynamic body the mouse can shove."""
        rigid = self._get_rigid(actor)
        if rigid is None:
            return
        try:
            if mass is not None:
                rigid.set_mass(float(mass))
        except Exception:
            pass
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            # Near-zero damping made kinematic mouse contacts launch bottles.
            rigid.set_linear_damping(float(lin_damp))
            rigid.set_angular_damping(float(ang_damp))
        except Exception:
            pass
        # Do not set_max_linear_velocity here — that would also throttle freefall.
        # Shelf launch is clamped per-step in _tame_shelf_velocities.
        try:
            for shape in rigid.get_collision_shapes():
                m = shape.get_physical_material()
                m.set_restitution(0.0)
                m.set_static_friction(1.05)
                m.set_dynamic_friction(0.90)
        except Exception:
            pass

    def _object_mass_damping(self, label: str):
        lab = str(label)
        mass = float(self.MASS_BY_LABEL.get(lab, 0.16))
        lin = float(self.LIN_DAMP_BY_LABEL.get(lab, 0.70))
        ang = float(self.ANG_DAMP_BY_LABEL.get(lab, 0.80))
        return mass, lin, ang

    def _activate_target(self):
        """Hand every shelf object over to physics just before the mouse sets off.

        All cups/bottles start kinematic so they cannot drift during settle, then
        are released together. Non-targets stay free so a scurrying mouse that
        brushes them can push them aside instead of freezing against a static
        wall. Only the chosen target's fall is scored.
        """
        if self._target_live:
            return
        for entry in self.shelf_objects:
            label = str(entry.get("label", ""))
            mass, lin, ang = self._object_mass_damping(label)
            self._release_actor_to_physics(
                entry["actor"], mass=mass, lin_damp=lin, ang_damp=ang,
            )
        self._target_live = True

    def _tame_shelf_velocities(self):
        """Clamp runaway speeds from kinematic mouse penetration.

        While an object still sits on the shelf plate, keep tip/slide speeds
        near the mouse's push rate. Once it has left the lip, allow a normal fall
        but still reject absurd launch velocities.
        """
        if not self._target_live:
            return
        shelf_z = float(self.shelf_z_surf)
        for entry in self.shelf_objects:
            actor = entry.get("actor")
            rigid = self._get_rigid(actor)
            if rigid is None:
                continue
            try:
                lo, _ = rigid.compute_global_aabb_tight()
                bottom = float(lo[2])
            except Exception:
                p = np.asarray(
                    (actor.actor if hasattr(actor, "actor") else actor).get_pose().p,
                    dtype=np.float64,
                )
                bottom = float(p[2]) - float(entry.get("radius", 0.03))
            on_shelf = bottom >= shelf_z - 0.025
            max_v = (
                float(self.SHELF_MAX_LINEAR_SPEED) if on_shelf
                else float(self.FALL_MAX_LINEAR_SPEED)
            )
            max_w = float(self.SHELF_MAX_ANGULAR_SPEED) if on_shelf else 12.0
            try:
                v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
                speed = float(np.linalg.norm(v))
                if speed > max_v and speed > 1e-8:
                    rigid.set_linear_velocity((v * (max_v / speed)).tolist())
                w = np.asarray(rigid.get_angular_velocity(), dtype=np.float64)
                w_n = float(np.linalg.norm(w))
                if w_n > max_w and w_n > 1e-8:
                    rigid.set_angular_velocity((w * (max_w / w_n)).tolist())
            except Exception:
                pass

    def _release_mouse(self):
        if self._mouse_state != "idle":
            return
        self._mouse_state = "running"
        if self._metric_start_step is None:
            self._metric_start_step = self._metric_step()

    def _advance_mouse(self):
        if self._mouse_state != "running":
            return
        dt = float(self.scene.get_timestep())
        nxt = self._mouse_s + float(self.mouse_speed) * dt
        # Continuous run: do not pause at the stand-off for basket placement.
        self._mouse_s = nxt
        if self._mouse_s >= self._mouse_path_len:
            self._mouse_s = self._mouse_path_len
            if self._mouse_state != "done":
                self._mouse_shove_attempts = int(
                    getattr(self, "_mouse_shove_attempts", 0)
                ) + 1
                self._shove_observe = 0
            self._mouse_state = "done"
        p, heading = self._mouse_point_at(self._mouse_s)
        self._set_mouse_pose(
            [float(p[0]), float(p[1]), self._mouse_z], heading=heading,
        )

    def _target_still_on_shelf(self) -> bool:
        """True when the knock target never left the shelf plate."""
        if self.target is None:
            return False
        if self._caught or self._fell_on_table:
            return False
        if self._obj_state in ("falling", "caught", "fallen"):
            return False
        bottom = self._target_aabb_bottom()
        if bottom < float(self.shelf_z_surf) - 0.025:
            return False
        p = np.asarray(self.target.get_pose().p, dtype=np.float64)
        # Already past the front lip — treat as falling / off-shelf.
        if float(p[1]) < float(self.shelf_front_y) - 0.015:
            return False
        return True

    def _retry_mouse_shove(self) -> bool:
        """Back up behind the live target pose and shove toward the lip again."""
        if self.target is None:
            return False
        attempts = int(getattr(self, "_mouse_shove_attempts", 0))
        max_tries = int(
            getattr(self, "_cfg", {}).get(
                "mouse_shove_retries", self.MOUSE_SHOVE_RETRIES
            )
        )
        if attempts >= max_tries:
            return False

        p = np.asarray(self.target.get_pose().p, dtype=np.float64)
        tx, ty = float(p[0]), float(p[1])
        hy = float(self.target_info.get("hy", self.target_radius))
        mh = float(self.MOUSE_HALF_XY)
        y_hi = float(self.shelf_plate_ylim[1] - mh - 0.010)
        behind_y = float(
            min(ty + hy + mh + float(self.MOUSE_STANDOFF) + 0.020, y_hi)
        )
        # Nudge past the lip, but not deep into free space (that launches bottles).
        push_y1 = float(self.shelf_front_y - 0.012)
        mx = float(self.mouse_pos[0])
        my = float(self.mouse_pos[1])
        route = [
            np.array([mx, my], dtype=np.float64),
            np.array([mx, behind_y], dtype=np.float64),
            np.array([tx, behind_y], dtype=np.float64),
        ]
        self._set_mouse_path(route, push_y1, float(self._mouse_z))
        self._allow_shove = True
        self._mouse_state = "running"
        self._shove_observe = 0
        # Mildly firmer retry — large speed jumps eject tall bottles.
        base = float(getattr(self, "mouse_speed_mean", self.mouse_speed))
        cap = float(getattr(self, "MOUSE_RETRY_SPEED_CAP", 0.09))
        mult = float(getattr(self, "MOUSE_RETRY_SPEED_MULT", 1.15))
        self.mouse_speed = float(min(cap, max(float(self.mouse_speed), base) * mult))
        # Kill residual object velocity so the next contact is a clean push.
        rigid = self._get_rigid(self.target)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        return True

    def _maybe_retry_mouse_shove(self) -> None:
        """After a finished shove, retry if the object is still on the shelf."""
        if self._mouse_state != "done":
            return
        if self._caught or self._fell_on_table:
            return
        if not self._target_still_on_shelf():
            return
        observe = int(
            getattr(self, "_cfg", {}).get(
                "mouse_shove_observe_steps", self.MOUSE_SHOVE_OBSERVE_STEPS
            )
        )
        self._shove_observe = int(getattr(self, "_shove_observe", 0)) + 1
        if self._shove_observe < max(1, observe):
            return
        self._shove_observe = 0
        self._retry_mouse_shove()

    def _cushion_top_z(self):
        if self.basket is None:
            return self.table_top
        bz = float(self.basket.get_pose().p[2])
        return bz + float(getattr(self, "_cushion_top_local", 0.0))

    def _target_aabb_bottom(self):
        """Real collision AABB bottom — pose.z alone misses tipped landings."""
        if self.target is None:
            return float(self.table_top + 1.0)
        rigid = self._get_rigid(self.target)
        if rigid is None:
            p = np.array(self.target.get_pose().p, dtype=np.float64)
            return float(p[2] - self.target_radius)
        try:
            lo, _ = rigid.compute_global_aabb_tight()
            return float(lo[2])
        except Exception:
            p = np.array(self.target.get_pose().p, dtype=np.float64)
            return float(p[2] - self.target_radius)

    def _target_footprint_overlap_frac(self) -> float:
        """Fraction of the target's XY footprint that lies over the basket mouth."""
        if self.target is None or self.basket is None:
            return 0.0
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        half = getattr(self, "_cushion_half_xy", self.basket_half_xy)
        tol = float(getattr(self, "catch_xy_tol", self.CATCH_XY_TOL))
        bx0, bx1 = float(bp[0]) - (half[0] + tol), float(bp[0]) + (half[0] + tol)
        by0, by1 = float(bp[1]) - (half[1] + tol), float(bp[1]) + (half[1] + tol)
        rigid = self._get_rigid(self.target)
        ox0 = ox1 = oy0 = oy1 = None
        if rigid is not None:
            try:
                lo, hi = rigid.compute_global_aabb_tight()
                ox0, ox1 = float(lo[0]), float(hi[0])
                oy0, oy1 = float(lo[1]), float(hi[1])
            except Exception:
                ox0 = None
        if ox0 is None:
            p = np.array(self.target.get_pose().p, dtype=np.float64)
            r = float(self.target_radius)
            ox0, ox1 = float(p[0]) - r, float(p[0]) + r
            oy0, oy1 = float(p[1]) - r, float(p[1]) + r
        area = max(1e-9, (ox1 - ox0) * (oy1 - oy0))
        ix = max(0.0, min(ox1, bx1) - max(ox0, bx0))
        iy = max(0.0, min(oy1, by1) - max(oy0, by0))
        return float(ix * iy / area)

    def _target_supported_in_basket(self) -> bool:
        """Target is resting in / on the basket rather than on the tabletop.

        Objects wider or taller than the basket cannot drop inside; they settle
        across the rim. Those count as caught as long as most of the footprint
        is over the mouth and nothing rests on the tabletop.
        """
        if self.target is None or self.basket is None:
            return False
        bottom = self._target_aabb_bottom()
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        rim_top = float(bp[2]) + float(self.basket_hz)
        # On the tabletop, or perched above the basket (décor / mid-air).
        if bottom <= float(self.table_top) + 0.015:
            return False
        if bottom > rim_top + 0.03:
            return False
        return self._target_footprint_overlap_frac() >= self.CATCH_FOOTPRINT_FRAC

    def _target_in_basket(self, *, require_settled=False):
        """True when the target's footprint sits over the basket mouth.

        ``require_settled`` gates success: the object must be *supported* by the
        basket (on the cushion, or across the rim when it is too big to drop in)
        and have stayed at rest for ``SETTLE_DWELL_STEPS`` consecutive steps.
        Being airborne inside the basket volume is not a catch — a bounce off the
        rim dips below any instantaneous speed threshold at its apex. Table miss
        checks use ``require_settled=False`` so a bounce inside the basket is not
        misread as a table hit.
        """
        if self.target is None or self.basket is None:
            return False
        supported = self._target_supported_in_basket()
        if require_settled:
            if not supported:
                return False
            if self._target_speed() >= self.SETTLE_SPEED:
                return False
            return int(getattr(self, "_settle_steps", 0)) >= self.SETTLE_DWELL_STEPS
        if supported:
            return True
        p = np.array(self.target.get_pose().p, dtype=np.float64)
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        half = getattr(self, "_cushion_half_xy", self.basket_half_xy)
        inside_xy = (
            abs(p[0] - bp[0]) <= half[0] + self.catch_xy_tol
            and abs(p[1] - bp[1]) <= half[1] + self.catch_xy_tol
        )
        if not inside_xy:
            return False
        # Airborne but clearly inside / above the basket volume (bounce in flight).
        bottom = self._target_aabb_bottom()
        return bool(
            bottom > float(self.table_top) + 0.018
            and bottom < float(bp[2]) + float(self.basket_hz) + 0.06
        )

    def _update_settle_dwell(self) -> None:
        """Count consecutive steps the target has been at rest in the basket."""
        if (
            self._target_supported_in_basket()
            and self._target_speed() < self.SETTLE_SPEED
        ):
            self._settle_steps = int(getattr(self, "_settle_steps", 0)) + 1
        else:
            self._settle_steps = 0

    def _target_speed(self):
        rigid = self._get_rigid(self.target)
        if rigid is None:
            return 0.0
        try:
            v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
            w = np.asarray(rigid.get_angular_velocity(), dtype=np.float64)
        except Exception:
            return 0.0
        return float(np.linalg.norm(v) + 0.05 * np.linalg.norm(w))

    def _object_touches_table(self):
        """True once any part of the target is on/through the table top.

        Basket-cushion contact is not a table hit. Used as a hard failure: once
        set, success is impossible. AABB bottom (not pose.z) so tipped bottles
        still register when they land on the tabletop.
        """
        if self.target is None:
            return False
        # Still on / above the shelf — not a table miss.
        bottom = self._target_aabb_bottom()
        if bottom > float(self.shelf_z_surf) - 0.02:
            return False
        if self._target_in_basket(require_settled=False):
            return False
        return bool(bottom <= float(self.table_top) + 0.012)

    def _update_catch_state(self):
        """Read the outcome off the simulation — nothing is teleported."""
        if not self._target_live or self.target is None or self._fell_on_table:
            return
        bottom = self._target_aabb_bottom()
        if bottom < self.shelf_z_surf - 0.03 and self._obj_state == "parked":
            self._obj_state = "falling"
            self._latch_fall_metric()

        # Table contact outside the basket is an immediate permanent fail.
        if self._object_touches_table():
            self._fell_on_table = True
            self._caught = False
            self._obj_state = "fallen"
            self._settle_steps = 0
            self._latch_land_metric()
            return

        self._update_settle_dwell()
        if self._target_in_basket(require_settled=True):
            self._caught = True
            self._obj_state = "caught"
            self._latch_land_metric()

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_start_step = None   # mouse set loose toward the shelf
        self._metric_cover_step = None   # basket first seated under the landing spot
        self._metric_fall_step = None    # object knocked off the shelf
        self._metric_land_step = None    # object settled (in the basket or on the table)
        self._metric_offset_norm = None  # basket placement error at knock-off time

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _basket_offset_norm(self):
        """Chebyshev basket-vs-landing offset over the basket's catch footprint.

        <= 1.0 means the landing point is still inside the basket mouth.
        """
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        land = np.asarray(self._landing, dtype=np.float64)
        tol = float(self.catch_xy_tol)
        hx = float(self.basket_half_xy[0]) + tol
        hy = float(self.basket_half_xy[1]) + tol
        return float(max(abs(bp[0] - land[0]) / max(hx, 1e-6),
                         abs(bp[1] - land[1]) / max(hy, 1e-6)))

    def _latch_fall_metric(self):
        """Called when the object leaves the shelf — basket placement is committed."""
        try:
            if self._metric_fall_step is None:
                self._metric_fall_step = self._metric_step()
                if self.basket is not None:
                    self._metric_offset_norm = self._basket_offset_norm()
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

        extra1 `cover_latency_steps` — steps from the mouse being set loose until the
        basket is first seated under the landing spot. `cover_margin_steps` reports
        how much slack was left when the object was actually knocked off (positive =
        basket in place first; None = never covered before the fall).
        extra2 `basket_offset_norm` — Chebyshev basket-vs-landing offset at the moment
        the object left the shelf, over the basket's catch footprint. LOWER is better;
        <= 1.0 means the landing point was inside the basket mouth.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_start_step, self._metric_cover_step
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
        out["basket_offset_norm"] = None if off is None else round(float(off), 4)
        out["caught"] = bool(getattr(self, "_caught", False))
        out["fell_on_table"] = bool(getattr(self, "_fell_on_table", False))
        return out

    def _basket_under_landing(self):
        if self.basket is None:
            return False
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        land = self._landing
        hz = float(getattr(self, "basket_hz", 0.5 * self.basket_height))
        seated = abs(float(bp[2]) - (float(self.table_top) + hz)) <= 0.025
        return bool(
            seated
            and abs(bp[0] - land[0]) <= self.basket_half_xy[0] + self.catch_xy_tol
            and abs(bp[1] - land[1]) <= self.basket_half_xy[1] + self.catch_xy_tol
        )

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if self.basket is not None and self._basket_under_landing():
            self._basket_placed = True
            if self._metric_cover_step is None:
                self._metric_cover_step = self._metric_step()
        self._advance_mouse()
        self._tame_shelf_velocities()
        self._maybe_retry_mouse_shove()
        self._update_catch_state()
        # Pin non-targets only while the shelf is still kinematic. After
        # _activate_target they are free dynamic bodies and must not be teleported.
        if not self._target_live:
            for i, entry in enumerate(self.shelf_objects):
                if i == self.target_idx:
                    continue
                uq = entry.get("upright_q", self.UPRIGHT_Q)
                self._set_entity_pose(
                    entry["actor"],
                    sapien.Pose(entry["start"].tolist(), np.asarray(uq).tolist()),
                )

    def check_stable(self):
        # Only the still-kinematic target is held; once physics owns it, and
        # once the basket is in the gripper, nothing here repositions anything.
        if not self._target_live and self.target is not None:
            self._set_entity_pose(
                self.target,
                sapien.Pose(
                    self.target_start.tolist(),
                    np.asarray(self.target_upright_q).tolist(),
                ),
            )
        if self._mouse_state == "idle" and self.mouse_parts:
            self._set_mouse_pose(self.mouse_pos)
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
    def _tcp_pos(self, arm_tag):
        pose = (
            self.robot.get_left_tcp_pose() if str(arm_tag) == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.array(pose[:3], dtype=np.float64)

    def interactive_grasp_basket(self, arm_tag) -> bool:
        """Interactive Space: grasp the basket handle with unconstrained moves.

        ``grasp_actor`` inserts a straight-line constrained descent that the
        planner cannot solve onto this handle, so the arm stalls at pre-grasp
        and the gripper shuts on air (first Space press fails / flashes red).
        Matching the expert path — two plain ``move_to_pose`` steps — makes the
        first press succeed.
        """
        if self.basket is None:
            return False
        self.plan_success = True
        pre_pose, grasp_pose = self.choose_grasp_pose(
            self.basket,
            arm_tag=arm_tag,
            pre_dis=0.10,
            target_dis=0.0,
            contact_point_id=[0, 1],
        )
        if pre_pose is None or grasp_pose is None:
            self.plan_success = False
            return False

        self.plan_success = True
        self.move(self.open_gripper(arm_tag))
        self.plan_success = True
        moved = self.move(self.move_to_pose(arm_tag, pre_pose))
        if moved is False or not bool(self.plan_success):
            return False
        self.plan_success = True
        moved = self.move(self.move_to_pose(arm_tag, grasp_pose))
        if moved is False or not bool(self.plan_success):
            return False

        self.plan_success = True
        moved = self.move(self.close_gripper(arm_tag, pos=0.0))
        if moved is False or not bool(self.plan_success):
            return False
        self._holding_basket = True
        print(f"[catch_mouse_object_drop] basket handle grasped ({arm_tag})")
        return True

    def interactive_release_basket(self, arm_tag) -> bool:
        """Interactive Space: open gripper and drop the basket in place."""
        self.plan_success = True
        moved = self.move(self.open_gripper(arm_tag))
        ok = moved is not False and bool(self.plan_success)
        if ok:
            self._holding_basket = False
            print("[catch_mouse_object_drop] basket released")
        return ok

    def _pick_place_basket_to_landing(self, arm_tag):
        """Grasp the basket's handle and set it down under the landing.

        The descent is issued as two plain moves rather than ``grasp_actor``:
        its second action is a straight-line constrained move, which the planner
        cannot solve down onto the handle, so the arm would stall at pre-grasp
        and shut on air. Carry displacements are measured off the basket itself
        so the grasp offset does not bias where it lands. The basket is never
        repositioned by hand — it is carried and released by the gripper.
        """
        land_xy = np.array(
            [float(self._landing[0]), float(self._landing[1])], dtype=np.float64,
        )
        hz = float(getattr(self, "basket_hz", 0.5 * self.basket_height))
        place_z = self.table_top + hz

        if not self.interactive_grasp_basket(arm_tag):
            raise UnStableError("catch_mouse_object_drop: basket handle grasp failed")

        # Lift clear of the table.
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.15, move_axis="world",
        ))

        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        if bp[2] - place_z < 0.05:
            raise UnStableError("catch_mouse_object_drop: basket handle grasp slipped")

        # Carry over the landing (displacements measured on the basket pose).
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(land_xy[0] - bp[0]),
            y=float(land_xy[1] - bp[1]),
            move_axis="world",
        ))

        # Lower until the basket is just above the table, then let it go.
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(land_xy[0] - bp[0]),
            y=float(land_xy[1] - bp[1]),
            z=float(place_z + 0.004 - bp[2]),
            move_axis="world",
        ))
        # Second micro-lower if the long carry from x≈0 left the basket high.
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        if float(bp[2]) > place_z + 0.015:
            self.plan_success = True
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(land_xy[0] - bp[0]),
                y=float(land_xy[1] - bp[1]),
                z=float(place_z + 0.003 - bp[2]),
                move_axis="world",
            ))

        self.plan_success = True
        self.move(self.open_gripper(arm_tag=arm_tag))
        self._holding_basket = False
        self._dwell(16)  # let the basket settle on the table under gravity

        self._basket_placed = bool(self._basket_under_landing())

        # Retreat so the arm is clear of the catch zone.
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="world"))
        self.plan_success = True
        self.move(self.back_to_origin(arm_tag=arm_tag))
        if self._basket_placed:
            self.plan_success = True
        return self._basket_placed

    def play_once(self):
        arm_tag = ArmTag(self.arm_side)
        self._pic_counter = 0

        old_save_freq = self.save_freq
        if self.save_data and (self.save_freq is None or self.save_freq > 8):
            self.save_freq = 5

        # Mouse scurries from the first frame while the arm places the basket.
        # No stand-off wait: the knock is a race against placement timing.
        self._allow_shove = True
        self._activate_target()
        self._release_mouse()
        self._pick_place_basket_to_landing(arm_tag)

        dt = max(float(self.scene.get_timestep()), 1e-4)
        remain = max(0.0, self._mouse_path_len - float(self._mouse_s))
        run_steps = int(remain / max(self.mouse_speed, 1e-4) / dt) + 80
        # Budget for back-up / re-shove passes if the first contact stalls.
        retry_budget = int(self.MOUSE_SHOVE_RETRIES) * (
            int(220 / max(self.mouse_speed, 1e-4) / dt) + int(self.MOUSE_SHOVE_OBSERVE_STEPS)
        )
        max_wait = run_steps + retry_budget
        waited = 0
        while waited < max_wait:
            if self._caught or self._fell_on_table:
                break
            if self._mouse_state == "done" and not self._target_still_on_shelf():
                break
            if (
                self._mouse_state == "done"
                and int(getattr(self, "_mouse_shove_attempts", 0))
                >= int(self.MOUSE_SHOVE_RETRIES)
            ):
                break
            if self._mouse_state not in ("running", "done"):
                break
            self._dwell(1)
            waited += 1

        # Let the knocked object fall and come to rest — no scripted landing.
        settle = 0
        while (
            settle < self.SETTLE_STEPS
            and not self._caught
            and not self._fell_on_table
        ):
            self._dwell(1)
            settle += 1

        self._dwell(self.post_place_dwell)
        if self._caught:
            self.plan_success = True
        self.save_freq = old_save_freq

        info = self.target_info
        self.info["info"] = {
            "{A}": f"{info['model']}/base{info['model_id']}",
            "{B}": self.BASKET_MODEL,
            "{C}": "cushion",
            "{D}": "mouse",
            "{E}": "deep_wall_shelf",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # Pass only if the object rests in the basket and never touched the table.
        if self._fell_on_table or self._obj_state == "fallen":
            return False
        if self.target is None or self.basket is None:
            return False
        # Live re-check so interactive sessions terminate even if a prior frame
        # missed the contact latch.
        if self._object_touches_table():
            self._fell_on_table = True
            self._caught = False
            self._obj_state = "fallen"
            return False
        if self._target_in_basket(require_settled=True):
            self._caught = True
            self._obj_state = "caught"
            return True
        return False

    def _target_basket_edge_distance(self) -> float:
        """Planar gap between the target's footprint and the basket mouth.

        Zero when the footprints overlap or touch, so an object that came to
        rest against the basket reads as the closest possible miss.
        """
        if self.target is None or self.basket is None:
            return float("inf")
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        half = getattr(self, "_cushion_half_xy", self.basket_half_xy)
        tol = float(getattr(self, "catch_xy_tol", self.CATCH_XY_TOL))
        bx0, bx1 = float(bp[0]) - (half[0] + tol), float(bp[0]) + (half[0] + tol)
        by0, by1 = float(bp[1]) - (half[1] + tol), float(bp[1]) + (half[1] + tol)
        rigid = self._get_rigid(self.target)
        ox0 = None
        if rigid is not None:
            try:
                lo, hi = rigid.compute_global_aabb_tight()
                ox0, ox1 = float(lo[0]), float(hi[0])
                oy0, oy1 = float(lo[1]), float(hi[1])
            except Exception:
                ox0 = None
        if ox0 is None:
            p = np.array(self.target.get_pose().p, dtype=np.float64)
            r = float(self.target_radius)
            ox0, ox1 = float(p[0]) - r, float(p[0]) + r
            oy0, oy1 = float(p[1]) - r, float(p[1]) + r
        dx = max(0.0, bx0 - ox1, ox0 - bx1)
        dy = max(0.0, by0 - oy1, oy0 - by1)
        return float(np.hypot(dx, dy))

    def get_score(self) -> float:
        """Partial score from how close the object ended up to the basket.

        Settled in / across the basket → 1.0. Over the basket but still moving
        → 0.5. Otherwise the object missed and is down on the table, which is
        exactly what the near-miss ring measures: the planar gap between its
        footprint and the basket mouth, in thirds of the mouth half-diagonal
        → 0.75 / 0.5 / 0.25, and 0 past that ring.
        """
        if self.target is None or self.basket is None:
            return 0.0
        # Never knocked off the shelf — there is no catch attempt to score.
        if self._target_still_on_shelf():
            return 0.0
        if self.check_success():
            return 1.0
        fell = bool(getattr(self, "_fell_on_table", False)) or self._obj_state == "fallen"
        if not fell and self._target_in_basket(require_settled=False):
            return 0.5
        half = getattr(self, "_cushion_half_xy", self.basket_half_xy)
        r = float(np.hypot(float(half[0]), float(half[1])))
        if r <= 1e-9:
            return 0.0
        d = self._target_basket_edge_distance()
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
        obs["catch_mouse_object_drop"] = {
            "obj_state": str(self._obj_state),
            "mouse_state": str(self._mouse_state),
            "mouse_shove_attempts": int(getattr(self, "_mouse_shove_attempts", 0)),
            "fell_on_table": bool(self._fell_on_table),
            "caught": bool(self._caught),
            "basket_placed": bool(self._basket_placed),
            "target_label": str(getattr(self, "target_label", "")),
            "target_mode": str(getattr(self, "target_mode", "")),
            "target_rolls": bool(getattr(self, "target_rolls", False)),
            "mouse_end": str(getattr(self, "mouse_end", "")),
            "mouse_s": float(getattr(self, "_mouse_s", 0.0)),
            "mouse_speed": float(getattr(self, "mouse_speed", 0.0)),
            "mouse_speed_mean": float(getattr(self, "mouse_speed_mean", 0.0)),
            "drop_x": float(getattr(self, "drop_x", 0.0)),
            "landing": list(map(float, self._landing)),
            "n_shelf_objects": int(len(self.shelf_objects)),
            "partial_score": float(self.get_score()),
        }
        return obs
