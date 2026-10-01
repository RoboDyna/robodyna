"""KitchenS scene base for RoboDynaExp.

Ports the robotwin_bench KitchenS counter layout (microwave + sink + backsplash
+ chrome gooseneck faucet) into RoboDynaExp's ``Base_Task``.

Default task layout (``replace_sink_with_range=True``): keep a real sink + tap
at the **center** of the counter, and place a cooking range where KitchenS's
sink slot used to be (right in scene_0). The tap always sits behind the sink,
never beside the stove.
"""
from __future__ import annotations

import json
import os

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.actor_utils import Actor
from .utils.create_actor import create_box, create_visual_box, resolve_texture_file


class KitchenS_base_task(Base_Task):
    """RoboDynaExp KitchenS environment with a cooking range in place of the sink."""

    FURNITURE_NAMES = {"table", "wall", "ground"}

    # Flush CC0 4-burner gas cooktop (assets/dyna_assets/268_countertop_gas_stove).
    COOKTOP_ASSET = "268_countertop_gas_stove"
    # World-frame offsets from cooktop center (grate clusters on 268).
    RANGE_BURNER_OFFSETS = {
        "left_front": (-0.103, -0.103),
        "right_front": (0.103, -0.103),
        "left_rear": (-0.103, 0.103),
        "right_rear": (0.103, 0.103),
    }
    # Top-facing rotary knob on the right-front corner of the cooktop
    # (slightly forward of the slab so it sits on the counter apron).
    # When ``stove_side == "right"``, ``_load_cooking_range`` mirrors X so the
    # knob sits on the left-front apron facing the open workspace.
    KNOB_LOCAL_XY = (0.165, -0.185)
    KNOB_RADIUS = 0.022
    KNOB_HEIGHT = 0.028  # full height of the knob body
    # Semantic angles about world +Z (white tick at rest points toward +Y / "up").
    # 0° = off; −90° (CCW / left from above) = on.
    KNOB_OFF_ANGLE = 0.0
    KNOB_ON_ANGLE = -0.5 * np.pi
    # Approach the top-facing knob from +X/−Y, then settle overhead.
    # A pure vertical descent from the right-apron park often fails IK when the
    # cooktop is centered (knob near x≈0.16); the lateral staging keeps the
    # wrist out of that singularity / collision corridor.
    TOP_KNOB_APPROACH_PATH = (
        (0.12, -0.10, 0.28),
        (0.08, -0.06, 0.18),
        (0.03, -0.02, 0.12),
        (0.00, 0.00, 0.08),
    )
    KNOB_APPROACH_PATH = TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF = 0.012
    EE_TO_TCP_DEFAULT = 0.12
    # Policy / interactive grasp: pinch within this radius counts as on-knob.
    KNOB_CONTACT_RADIUS_DEFAULT = 0.06
    # Gripper val below this = closed enough to torque the cap.
    KNOB_GRASP_GRIPPER_MAX = 0.55

    def setup_demo(self, **kwags):
        # scene_id before init so create_table_and_wall can place fixtures.
        self.scene_id = kwags.get("scene_id")
        if self.scene_id is None:
            self.scene_id = int(np.random.randint(0, 3))
        self.kitchens_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [],
        }
        # Preserve subclass overrides set before super().setup_demo().
        if not hasattr(self, "replace_sink_with_range"):
            # Replace sink+tap with a cooking range (task default).
            self.replace_sink_with_range = True
        if not hasattr(self, "clear_sink_and_range"):
            # Bare counter: no sink, tap, or cooking range (solid top + microwave).
            self.clear_sink_and_range = False
        if not hasattr(self, "omit_sink"):
            # Solid counter + no tap/basin, but still load the cooking range.
            self.omit_sink = False
        if not hasattr(self, "range_position_override"):
            self.range_position_override = None
        if not hasattr(self, "microwave_xy_override"):
            self.microwave_xy_override = None
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------
    # Scene layout helpers
    # ------------------------------------------------------------------
    def _get_scene_obj_locations(self, object_name="microwave"):
        """Return [x, y] for KitchenS fixtures.

        scene_0: MW left, Dishrack center, Sink/Range right
        scene_1: MW left, Sink/Range center, Dishrack right
        scene_2: MW center, Dishrack left, Sink/Range right
        """
        if self.scene_id == 0:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.05, 0.17],
                "sink": [0.42, 0.08],
                "range": [0.42, 0.08],
            }
        elif self.scene_id == 1:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.42, 0.17],
                "sink": [0.10, 0.08],
                "range": [0.10, 0.08],
            }
        elif self.scene_id == 2:
            locations = {
                "microwave": [0.10, 0.18],
                "dishrack": [-0.32, 0.17],
                "sink": [0.42, 0.08],
                "range": [0.42, 0.08],
            }
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")
        if object_name not in locations:
            raise ValueError(f"Unknown object_name '{object_name}'")
        return locations[object_name]

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Build the KitchenS counter + fixtures (range replaces sink by default)."""
        self.table_xy_bias = list(table_xy_bias)
        table_height = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [n for n in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, n))]
            )
            wall_texture = np.random.randint(0, file_count)
            table_texture = np.random.randint(0, file_count)
            floor_texture = np.random.randint(0, file_count)
            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            self.floor_texture = f"{texture_type}/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture = self.table_texture = self.floor_texture = None

        # Floor tiles
        for i, pos in enumerate([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]):
            create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
                is_static=True,
            )

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        counter_length = self.kitchens_info["table_area"][0]
        counter_width = self.kitchens_info["table_area"][1]
        counter_thickness = 0.04
        self.kitchens_info["counter_thickness"] = counter_thickness

        # Sink stays at the center (dishrack slot). The cooking range occupies the
        # former KitchenS sink slot on the side — tap never goes next to the stove.
        clear_fixtures = bool(getattr(self, "clear_sink_and_range", False))
        omit_sink = bool(getattr(self, "omit_sink", False))
        if not clear_fixtures and not omit_sink:
            if getattr(self, "replace_sink_with_range", True):
                sink_rel_x, sink_rel_y = self._get_scene_obj_locations("dishrack")
                # Slightly forward of the shallow rack pose so the basin reads as a sink.
                sink_rel_y = 0.10
            else:
                sink_rel_x, sink_rel_y = self._get_scene_obj_locations("sink")
            self.kitchens_info["sink_geom"] = {
                "rel_p": [sink_rel_x, sink_rel_y],
                "hole_hx": 0.12,
                "hole_hy": 0.16,
                "depth": 0.09,
                "inner_hx": 0.11,
                "inner_hy": 0.15,
            }

        self._create_backsplash(counter_length, counter_width, table_height, table_xy_bias)

        if clear_fixtures or omit_sink:
            # Solid counter (no sink hole / tap). Range may still load when omit_sink.
            self._create_solid_counter(
                counter_length, counter_width, counter_thickness, table_height, table_xy_bias
            )
        else:
            # Cut a sink hole — the basin (and its tap) live at sink_geom.
            self._create_counter_with_sink_hole(
                counter_length, counter_width, counter_thickness, table_height, table_xy_bias
            )

        self._create_base_cabinets(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)
        self._create_counter_edge_trim(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)
        # Keep the upper wall clear: this task does not use the decorative shelves.

        self.kitchens_info["table_lims"] = [
            -counter_length / 2, -counter_width / 2,
            counter_length / 2, counter_width / 2,
        ]

        self._load_microwave(table_height, table_xy_bias)
        if clear_fixtures:
            return
        # Center is the sink; skip the dishrack stand-in in the range layout.
        if not getattr(self, "replace_sink_with_range", True):
            self._load_dishrack(table_height, table_xy_bias)
        if not omit_sink:
            self._load_sink(table_height, table_xy_bias)
        if getattr(self, "replace_sink_with_range", True):
            self._load_cooking_range(table_height, table_xy_bias)

    # ------------------------------------------------------------------
    # Counter & decorative elements
    # ------------------------------------------------------------------
    def _counter_material(self):
        if self.table_texture is not None:
            tex = resolve_texture_file(self.table_texture)
            texture_path = (
                str(tex)
                if tex is not None
                else f"./assets/background_texture/{self.table_texture}.png"
            )
            texture2d = sapien.render.RenderTexture2D(texture_path)
            mat = sapien.render.RenderMaterial()
            mat.set_base_color_texture(texture2d)
            mat.base_color = [1, 1, 1, 1]
            mat.metallic = 0.1
            mat.roughness = 0.3
            return mat
        mat = sapien.render.RenderMaterial(base_color=[0.28, 0.27, 0.26, 1])
        mat.metallic = 0.12
        mat.roughness = 0.22
        return mat

    def _create_solid_counter(self, counter_length, counter_width, counter_thickness, table_height, table_xy_bias):
        th = counter_thickness / 2
        counter_top_z = table_height - th
        mat = self._counter_material()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[counter_length / 2, counter_width / 2, th],
            material=self.scene.default_physical_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[counter_length / 2, counter_width / 2, th],
            material=mat,
        )
        builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], counter_top_z]))
        self.table = builder.build(name="table")

    def _create_counter_with_sink_hole(
        self, counter_length, counter_width, counter_thickness, table_height, table_xy_bias
    ):
        sink_geom = self.kitchens_info["sink_geom"]
        sink_rel_x, sink_rel_y = sink_geom["rel_p"]
        sink_hx, sink_hy = sink_geom["hole_hx"], sink_geom["hole_hy"]
        th = counter_thickness / 2
        counter_top_z = table_height - th
        cl, cr = -counter_length / 2, counter_length / 2
        cf, cb = -counter_width / 2, counter_width / 2
        hl, hr = sink_rel_x - sink_hx, sink_rel_x + sink_hx
        hf, hb = sink_rel_y - sink_hy, sink_rel_y + sink_hy
        mat = self._counter_material()
        counter_pieces = [
            ("right", hr, cr, cf, cb),
            ("left", cl, hl, cf, cb),
            ("front", hl, hr, cf, hf),
            ("back", hl, hr, hb, cb),
        ]
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        for _, x1, x2, y1, y2 in counter_pieces:
            hx = (x2 - x1) / 2
            hy = (y2 - y1) / 2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if hx <= 1e-6 or hy <= 1e-6:
                continue
            piece_pose = sapien.Pose([cx, cy, 0])
            builder.add_box_collision(
                pose=piece_pose, half_size=[hx, hy, th], material=self.scene.default_physical_material
            )
            builder.add_box_visual(pose=piece_pose, half_size=[hx, hy, th], material=mat)
        builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], counter_top_z]))
        self.table = builder.build(name="table")

    def _create_backsplash(self, counter_length, counter_width, table_height, table_xy_bias):
        create_visual_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] + counter_width / 2 + 0.005,
                table_height + 0.22,
            ]),
            half_size=[counter_length / 2 + 0.02, 0.006, 0.22],
            color=(0.92, 0.94, 0.96),
            name="backsplash",
        )

    def _create_base_cabinets(self, counter_length, counter_width, table_height, counter_thickness, table_xy_bias):
        cabinet_h = (table_height - counter_thickness) / 2
        cx, cy = table_xy_bias[0], table_xy_bias[1]
        create_box(
            self.scene,
            sapien.Pose(p=[cx, cy, cabinet_h]),
            half_size=[counter_length / 2 - 0.01, counter_width / 2 - 0.01, cabinet_h],
            color=(0.42, 0.36, 0.30),
            name="base_cabinets",
            is_static=True,
        )

    def _create_counter_edge_trim(
        self, counter_length, counter_width, table_height, counter_thickness, table_xy_bias
    ):
        create_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] - counter_width / 2,
                table_height - counter_thickness / 2,
            ]),
            half_size=[counter_length / 2, 0.006, counter_thickness / 2 + 0.002],
            color=(0.12, 0.12, 0.12),
            name="counter_edge_trim",
            is_static=True,
        )

    def _create_upper_shelves(self, table_height, table_xy_bias):
        counter_width = self.kitchens_info["table_area"][1]
        cx, cy = table_xy_bias[0], table_xy_bias[1]
        shelf_base_z = table_height + 0.32
        for side, sx in (("left", -0.45), ("right", 0.45)):
            create_box(
                self.scene,
                sapien.Pose(p=[cx + sx, cy + counter_width / 2 - 0.08, shelf_base_z]),
                half_size=[0.12, 0.08, 0.01],
                color=(0.55, 0.48, 0.40),
                name=f"upper_shelf_{side}",
                is_static=True,
            )

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _load_microwave(self, table_height, table_xy_bias):
        """Load the real 044_microwave mesh with planner-safe box collision.

        Loading the articulated URDF directly gives its collision meshes a
        non-unit scale, which mplib/CuRobo rejects. The visual mesh can still be
        scaled normally; only its collision is replaced by a unit-scale box.
        """
        override = getattr(self, "microwave_xy_override", None)
        if override is not None:
            x, y = float(override[0]), float(override[1])
        else:
            x, y = self._get_scene_obj_locations("microwave")
            x += table_xy_bias[0]
            y += table_xy_bias[1]

        visual_path = "assets/objects/044_microwave/visual/base0.glb"
        scale = 0.15 * float(getattr(self, "microwave_scale_mult", 1.0))
        bounds_min = np.array([-0.761475, -0.495360, -0.567261], dtype=float)
        bounds_max = np.array([0.795292, 0.435315, 0.630773], dtype=float)
        center = 0.5 * (bounds_min + bounds_max) * scale
        half_size = 0.5 * (bounds_max - bounds_min) * scale

        # Authored GLB has the door on a side face; yaw +90° (KitchenS / robotwin)
        # faces the door toward the robot workspace.
        qx, qy, qz, qw = t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes="sxyz")
        mw_q = [float(qw), float(qx), float(qy), float(qz)]
        R = t3d.quaternions.quat2mat(mw_q)
        corners = []
        for sx in (bounds_min[0], bounds_max[0]):
            for sy in (bounds_min[1], bounds_max[1]):
                for sz in (bounds_min[2], bounds_max[2]):
                    corners.append(R @ (np.array([sx, sy, sz], dtype=float) * scale))
        corners = np.asarray(corners, dtype=float)

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(p=center),
            half_size=half_size,
            material=self.scene.default_physical_material,
        )
        builder.add_visual_from_file(
            filename=visual_path,
            scale=[scale, scale, scale],
        )
        z = table_height - float(corners[:, 2].min())
        builder.set_initial_pose(sapien.Pose(p=[x, y, z], q=mw_q))
        microwave_entity = builder.build(name="microwave")
        self.microwave = Actor(
            microwave_entity,
            {
                "center": center.tolist(),
                "extents": (bounds_max - bounds_min).tolist(),
                "scale": scale,
            },
        )
        # Footprint / top for tasks that place decor on the microwave.
        self.microwave_xy = np.array([x, y], dtype=float)
        self.microwave_top_z = float(z + corners[:, 2].max())
        self.microwave_half_xy = np.array(
            [
                0.5 * (corners[:, 0].max() - corners[:, 0].min()),
                0.5 * (corners[:, 1].max() - corners[:, 1].min()),
            ],
            dtype=float,
        )
        self.add_prohibit_area(self.microwave, padding=0.02)

    def _load_dishrack(self, table_height, table_xy_bias):
        """Procedural dish-rack stand-in (scaled GLB collisions trip mplib/CuRobo)."""
        x, y = self._get_scene_obj_locations("dishrack")
        x += table_xy_bias[0]
        y += table_xy_bias[1]
        # Shallow open basket on the counter.
        wall_t = 0.008
        hx, hy, hz = 0.10, 0.08, 0.045
        floor_z = table_height + 0.006
        floor = create_box(
            self.scene,
            sapien.Pose(p=[x, y, floor_z]),
            half_size=[hx, hy, 0.006],
            color=(0.75, 0.75, 0.78),
            name="dishrack_floor",
            is_static=True,
        )
        for name, pos, half in (
            ("front", [x, y - hy + wall_t / 2, table_height + hz / 2], [hx, wall_t / 2, hz / 2]),
            ("back", [x, y + hy - wall_t / 2, table_height + hz / 2], [hx, wall_t / 2, hz / 2]),
            ("left", [x - hx + wall_t / 2, y, table_height + hz / 2], [wall_t / 2, hy - wall_t, hz / 2]),
            ("right", [x + hx - wall_t / 2, y, table_height + hz / 2], [wall_t / 2, hy - wall_t, hz / 2]),
        ):
            create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=half,
                color=(0.70, 0.70, 0.74),
                name=f"dishrack_{name}",
                is_static=True,
            )
        self.dishrack = floor
        self.dishrack.set_name("dishrack")
        self.basin_xy = (float(x), float(y))

        pad = 0.03
        self.prohibited_area.append([
            x - hx - pad, y - hy - pad, x + hx + pad, y + hy + pad
        ])

    def _load_cooking_range(self, table_height, table_xy_bias):
        """CC0 flush gas cooktop with a top-facing rotary knob.

        Uses ``268_countertop_gas_stove``. Collision is a thin unit-scale box so
        mplib/CuRobo never sees a scaled triangle mesh. The interactive knob
        stands on the slab (axis = world +Z) so experts grasp it from above.
        """
        override = getattr(self, "range_position_override", None)
        if override is not None and len(override) >= 2:
            rel_x, rel_y = float(override[0]), float(override[1])
        else:
            rel_x, rel_y = self._get_scene_obj_locations("range")
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]

        asset_dir = str(resolve_model_dir(self.COOKTOP_ASSET))
        with open(os.path.join(asset_dir, "model_data0.json"), "r") as f:
            model_data = json.load(f)
        scale_mult = float(getattr(self, "range_scale_mult", 1.0))
        scale = np.asarray(model_data["scale"], dtype=float) * scale_mult
        center = np.asarray(model_data["center"], dtype=float)
        extents = np.asarray(model_data["extents"], dtype=float)

        # Asset is Y-up. Rotate +90° about X so +Y → world +Z and the model's
        # front (+Z) faces the robot (−world Y).
        range_q = [0.70710678, 0.70710678, 0.0, 0.0]
        center_scaled = center * scale
        origin_y = y + float(center_scaled[2])

        model_width = float(extents[0] * scale[0])
        model_depth = float(extents[2] * scale[2])
        # Grate top = ymax_s; slab is ~2.5 cm below. Leave grate top ~3 cm above
        # the counter so the slab reads as a flush built-in hob.
        ymax_s = float((center[1] + 0.5 * extents[1]) * scale[1])
        lip_above = 0.030
        cooktop_z0 = float(table_height + lip_above - ymax_s)
        range_top_z = float(table_height + lip_above)

        # Collision must sit *below* the grate / pan-bowl plane. A full-thickness
        # lip box whose top is at ``range_top_z`` intersects skillet/pot bowls
        # (functional point ≈ +2 mm) and ejects dropped food / meatballs.
        coll_clearance = 0.018
        coll_half = 0.006
        coll_top = float(ymax_s - coll_clearance)
        body_half = np.array(
            [0.5 * model_width, coll_half, 0.5 * model_depth], dtype=float
        )
        body_center = np.array(
            [0.0, coll_top - coll_half, 0.0], dtype=float
        )

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(p=body_center),
            half_size=body_half,
            material=self.scene.default_physical_material,
        )
        builder.add_visual_from_file(
            filename=os.path.join(asset_dir, "visual", "base0.glb"),
            scale=scale.tolist(),
        )
        builder.set_initial_pose(
            sapien.Pose(p=[x, origin_y, cooktop_z0], q=range_q)
        )
        range_entity = builder.build(name="cooking_range")
        self.range_body = Actor(
            range_entity,
            {
                "center": center.tolist(),
                "extents": extents.tolist(),
                "scale": scale.tolist(),
            },
        )
        # The CC0 cooktop GLB ships with always-orange burner cups. Dim those so
        # "off" does not look lit — only our blue fire ring signals stove on.
        self._dim_cooktop_burner_materials(range_entity)

        self.range_xy = (float(x), float(y))
        self.range_half_size = (0.5 * model_width, 0.5 * model_depth)
        self.range_top_z = range_top_z

        self.burner_positions = {
            name: (float(x + dx * scale_mult), float(y + dy * scale_mult))
            for name, (dx, dy) in self.RANGE_BURNER_OFFSETS.items()
        }
        # Default active burner: left-rear (boil_milk); tasks may reassign.
        self.burner_xy = self.burner_positions["left_rear"]

        # Cover the GLB's baked burner cups only while lit. The covers start
        # hidden so an off stove has no gray disks in viewer mode.
        cover_r = 0.055 * scale_mult
        cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        self._burner_covers = []
        self._burner_cover_home_poses = {}
        self._burner_cover_shapes = {}
        for bname, (bx, by) in self.burner_positions.items():
            cover_mat = sapien.render.RenderMaterial(
                base_color=[0.0, 0.0, 0.0, 0.0]
            )
            cover_mat.metallic = 0.35
            cover_mat.roughness = 0.55
            cb = self.scene.create_actor_builder()
            cb.set_physx_body_type("static")
            cb.add_cylinder_visual(
                pose=sapien.Pose([0, 0, 0], cyl_q),
                radius=cover_r,
                half_length=0.0016 * scale_mult,
                material=cover_mat,
            )
            home = sapien.Pose(p=[bx, by, self.range_top_z + 0.0016])
            cb.set_initial_pose(self._fire_hidden_pose())
            cover = cb.build(name=f"burner_cover_{bname}")
            self._burner_covers.append(cover)
            self._burner_cover_home_poses[bname] = home
            self._burner_cover_shapes[bname] = []
            for c in cover.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_cover_shapes[bname].extend(list(c.render_shapes))

        burner_mat = sapien.render.RenderMaterial(
            base_color=[0.20, 0.20, 0.22, 1.0]
        )
        burner_r = 0.035 * scale_mult
        burner_builder = self.scene.create_actor_builder()
        burner_builder.add_cylinder_visual(
            pose=sapien.Pose(
                [0, 0, 0], [0.70710678, 0.0, 0.70710678, 0.0]
            ),
            radius=burner_r,
            half_length=0.0015 * scale_mult,
            material=burner_mat,
        )
        burner_builder.set_initial_pose(self._fire_hidden_pose())
        self.active_burner = burner_builder.build(name="active_burner")
        self._burner_shapes = []
        for c in self.active_burner.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._burner_shapes = list(c.render_shapes)

        # Articulated rotary knob: fixed stem + free revolute cap about +Z.
        # White tick: 0° → +Y (off), −90° → −X (on). While gripped the joint
        # is undriven — only jaw contact friction may torque it.
        knob_dx, knob_dy = self.KNOB_LOCAL_XY
        # Right-counter cooktop: put the knob on the stove's left apron so it
        # faces the open workspace (not the outer table edge).
        if str(getattr(self, "stove_side", "")).lower().strip() == "right":
            knob_dx = -float(knob_dx)
        self._knob_local_xy = (float(knob_dx), float(knob_dy))
        knob_x = float(x + knob_dx * scale_mult)
        knob_y = float(y + knob_dy * scale_mult)
        knob_r = float(self.KNOB_RADIUS) * scale_mult
        knob_half = float(self.KNOB_HEIGHT) * scale_mult / 2.0
        slab_top_z = float(self.range_top_z - 0.025)
        # Sit on the counter just in front of the cooktop apron
        # so the black knob stays visible against the surface.
        knob_z = float(table_height + knob_half + 0.008)
        knob_mat = sapien.render.RenderMaterial(
            base_color=[0.07, 0.07, 0.08, 1.0]
        )
        knob_mat.metallic = 0.25
        knob_mat.roughness = 0.30
        stem_mat = sapien.render.RenderMaterial(
            base_color=[0.18, 0.18, 0.20, 1.0]
        )
        stem_mat.metallic = 0.55
        stem_mat.roughness = 0.35
        tick_mat = sapien.render.RenderMaterial(
            base_color=[0.92, 0.92, 0.88, 1.0]
        )
        # High-friction rubberized grip so closed jaws can apply torque.
        self._knob_grip_material = self.scene.create_physical_material(
            2.8, 2.4, 0.0
        )
        # Sapien cylinder axis = local X; rotate so the shaft stands on +Z.
        cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        cyl_pose = sapien.Pose([0, 0, 0], cyl_q)
        # Joint local-X → world +Z; negative qpos = tick toward −X (on).
        joint_q = t3d.euler.euler2quat(0.0, np.pi / 2.0, 0.0)

        art_builder = self.scene.create_articulation_builder()
        root_link = art_builder.create_link_builder()
        root_link.set_name("stove_knob_base")
        stem_half = float(knob_half * 0.45)
        root_link.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, -knob_half]),
            half_size=[0.001, 0.001, 0.001],
            material=self.scene.default_physical_material,
            is_trigger=True,
        )
        root_link.add_cylinder_visual(
            pose=sapien.Pose(p=[0.0, 0.0, -stem_half * 0.3], q=cyl_q),
            radius=float(knob_r * 0.35),
            half_length=stem_half,
            material=stem_mat,
        )

        knob_link = art_builder.create_link_builder(root_link)
        knob_link.set_name("stove_knob")
        knob_link.set_joint_name("stove_knob_joint")
        knob_link.set_joint_properties(
            "revolute",
            limits=[[float(-np.pi), float(0.05)]],
            pose_in_parent=sapien.Pose(q=joint_q),
            pose_in_child=sapien.Pose(q=joint_q),
            friction=0.0,
            damping=0.8,
        )
        # Collision above the cooktop slab; light density so jaw forces win.
        coll_half = float(knob_half * 0.55)
        coll_lift = float(knob_half - coll_half + 0.002)
        knob_link.add_cylinder_collision(
            pose=sapien.Pose(p=[0.0, 0.0, coll_lift], q=cyl_q),
            radius=float(knob_r * 0.90),
            half_length=coll_half,
            material=self._knob_grip_material,
            density=180.0,
        )
        # Flat grip faces so parallel jaws get a solid purchase (not just a
        # friction cylinder). Four flats around the rim.
        flat_t = float(knob_r * 0.18)
        flat_out = float(knob_r * 0.78)
        flat_w = float(knob_r * 0.70)
        for px, py in (
            (flat_out, 0.0),
            (-flat_out, 0.0),
            (0.0, flat_out),
            (0.0, -flat_out),
        ):
            hx = flat_t if abs(px) > 1e-6 else flat_w
            hy = flat_w if abs(px) > 1e-6 else flat_t
            knob_link.add_box_collision(
                pose=sapien.Pose(p=[px, py, coll_lift]),
                half_size=[hx, hy, coll_half],
                material=self._knob_grip_material,
                density=80.0,
            )
        knob_link.add_cylinder_visual(
            pose=cyl_pose,
            radius=knob_r,
            half_length=knob_half,
            material=knob_mat,
        )
        knob_link.add_cylinder_visual(
            pose=sapien.Pose(p=[0.0, 0.0, float(knob_half * 0.35)], q=cyl_q),
            radius=float(knob_r * 1.08),
            half_length=float(knob_half * 0.28),
            material=knob_mat,
        )
        tick_y = float(knob_r * 0.55)
        knob_link.add_box_visual(
            pose=sapien.Pose(
                p=[0.0, tick_y, float(knob_half + 0.002)]
            ),
            half_size=[0.0025, 0.007, 0.0015],
            material=tick_mat,
        )

        art_builder.set_initial_pose(
            sapien.Pose(p=[knob_x, knob_y, knob_z])
        )
        self.stove_knob_articulation = art_builder.build(fix_root_link=True)
        self._knob_joint = self.stove_knob_articulation.get_active_joints()[0]
        self._knob_link = self.stove_knob_articulation.find_link_by_name(
            "stove_knob"
        )
        self.stove_knob = self._knob_link.entity
        self.stove_knob_indicator = self.stove_knob
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False  # alias used by stove-fire guard
        self._policy_controlling_knob = False
        self._hold_knob_joint(stiff=True)
        if not hasattr(self, "knob_contact_radius"):
            self.knob_contact_radius = float(self.KNOB_CONTACT_RADIUS_DEFAULT)

        self._knob_radius = knob_r
        self._knob_half = knob_half
        self._knob_indicator_z = float(knob_z + knob_half + 0.002)
        self._knob_front_y = float(knob_y)
        self.knob_xy = (knob_x, knob_y)
        self.knob_xyz = (knob_x, knob_y, knob_z)
        self.knob_top_z = float(knob_z + knob_half)
        self._top_knob = True
        self._set_knob_joint_angle(float(self.KNOB_OFF_ANGLE), hard=True)

        pad = 0.03
        self.prohibited_area.append(
            [
                x - model_width / 2 - pad,
                y - model_depth / 2 - pad,
                x + model_width / 2 + pad,
                y + model_depth / 2 + pad,
            ]
        )
        self._ring_parts = getattr(self, "_ring_parts", []) or []
        self._ring_shapes = getattr(self, "_ring_shapes", []) or []
        self._ring_home_poses = getattr(self, "_ring_home_poses", []) or []
        self._disc_parts = getattr(self, "_disc_parts", []) or []
        self._disc_home_poses = getattr(self, "_disc_home_poses", []) or []
        self._burner_home_pose = sapien.Pose(
            p=[
                self.burner_xy[0],
                self.burner_xy[1],
                self.range_top_z + 0.002,
            ]
        )

    # ------------------------------------------------------------------ stove fire
    # Shared blue gas-flame look across KitchenS tasks. When the stove is off the
    # ring/disc are moved underground so nothing reads as residual heat/sauce.
    FIRE_BLUE = [0.12, 0.72, 1.0, 1.0]
    FIRE_DISC_BLUE = [0.08, 0.55, 0.95, 1.0]

    def _fire_hidden_pose(self) -> sapien.Pose:
        return sapien.Pose(p=[0.0, 0.0, -1.0])

    def _clear_stove_fire_ring(self) -> None:
        for part in getattr(self, "_ring_parts", []) or []:
            try:
                self.scene.remove_entity(part)
            except Exception:
                pass
        self._ring_parts = []
        self._ring_shapes = []
        self._ring_home_poses = []
        # Ring actors were destroyed; drop the lit/intensity cache so the next
        # ``_set_stove_fire(True)`` actually poses the rebuilt halo.
        self._stove_fire_visual = None

    def _build_stove_fire_ring(
        self,
        cx: float,
        cy: float,
        cz: float,
        radius: float,
        *,
        n: int = 28,
        half_size: list[float] | None = None,
    ) -> None:
        """Build a blue fire halo; starts hidden until ``_set_stove_fire(True)``."""
        self._clear_stove_fire_ring()
        hs = list(half_size) if half_size is not None else [0.008, 0.004, 0.003]
        for i in range(int(n)):
            ang = 2.0 * np.pi * i / n
            home = sapien.Pose(
                p=[
                    float(cx + radius * np.cos(ang)),
                    float(cy + radius * np.sin(ang)),
                    float(cz),
                ]
            )
            part = create_visual_box(
                self.scene,
                self._fire_hidden_pose(),
                half_size=hs,
                color=tuple(self.FIRE_BLUE[:3]),
                name=f"stove_fire_ring_{i}",
            )
            self._ring_parts.append(part)
            self._ring_home_poses.append(home)
            try:
                comps = part.get_components()
            except Exception:
                comps = getattr(part, "components", [])
            for c in comps:
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._ring_shapes.extend(list(c.render_shapes))

    @staticmethod
    def _dim_cooktop_burner_materials(entity) -> None:
        """Darken baked-in orange/red burner cups on the cooktop mesh."""
        try:
            comps = entity.get_components()
        except Exception:
            return
        for c in comps:
            if not isinstance(c, sapien.render.RenderBodyComponent):
                continue
            for s in list(getattr(c, "render_shapes", []) or []):
                try:
                    mat = s.material
                    col = list(mat.base_color)
                except Exception:
                    continue
                r, g, b = float(col[0]), float(col[1]), float(col[2])
                # Warm burner glow: strong red/orange, little blue.
                if r > 0.35 and r > g >= b and (r - b) > 0.15:
                    cold = [0.12, 0.12, 0.13, float(col[3] if len(col) > 3 else 1.0)]
                    try:
                        mat.set_base_color(cold)
                    except Exception:
                        mat.base_color = cold
                    try:
                        if hasattr(mat, "set_emission"):
                            mat.set_emission([0.0, 0.0, 0.0, 1.0])
                        elif hasattr(mat, "emission"):
                            mat.emission = [0.0, 0.0, 0.0, 1.0]
                    except Exception:
                        pass

    @staticmethod
    def _bump_render_bodies(entities) -> None:
        """Force SAPIEN to pick up material/pose edits on visual actors.

        Changing ``RenderMaterial`` base_color/emission does not always dirty the
        GPU cache; the viewer then keeps drawing the old (invisible) fire until a
        mouse click triggers a full refresh. Disable/enable is the documented
        workaround (SAPIEN issue #234).
        """
        for ent in entities or []:
            if ent is None:
                continue
            try:
                comps = ent.get_components()
            except Exception:
                comps = getattr(ent, "components", []) or []
            for c in comps:
                if isinstance(c, sapien.render.RenderBodyComponent):
                    try:
                        c.disable()
                        c.enable()
                    except Exception:
                        pass

    def _flush_stove_fire_viewer(self) -> None:
        """Push stove fire visuals to the interactive viewer immediately."""
        try:
            self.scene.update_render()
        except Exception:
            pass
        viewer = getattr(self, "viewer", None)
        if viewer is None:
            return
        try:
            if hasattr(viewer, "notify_render_update"):
                viewer.notify_render_update()
        except Exception:
            pass
        # One immediate draw so the ring appears without waiting for a click.
        try:
            viewer.render()
        except Exception:
            pass

    def _set_stove_fire(self, on: bool, intensity: float = 1.0) -> None:
        """Bright blue flame when on; fully hide disc + ring when off (no gray)."""
        inten = float(np.clip(intensity, 0.0, 1.0))
        lit = bool(on) and inten > 0.02
        prev = getattr(self, "_stove_fire_visual", None)
        if prev is not None and prev[0] is lit and abs(prev[1] - inten) < 1e-3:
            return
        self._stove_fire_visual = (lit, inten)
        # Lit: bright blue. Off: transparent + buried — never leave a gray halo.
        ring_col = [0.10, 0.55 + 0.35 * inten, 1.0, 1.0] if lit else [0.0, 0.0, 0.0, 0.0]
        disc_col = [0.06, 0.40 + 0.30 * inten, 0.95, 1.0] if lit else [0.0, 0.0, 0.0, 0.0]
        hidden = self._fire_hidden_pose()
        touched = []

        for part, home in zip(
            getattr(self, "_ring_parts", []) or [],
            getattr(self, "_ring_home_poses", []) or [],
        ):
            try:
                part.set_pose(home if lit else hidden)
                touched.append(part)
            except Exception:
                pass
        for s in getattr(self, "_ring_shapes", []) or []:
            try:
                mat = s.material
                mat.set_base_color(ring_col)
                if hasattr(mat, "set_emission"):
                    mat.set_emission(ring_col if lit else [0, 0, 0, 1])
                elif hasattr(mat, "emission"):
                    mat.emission = ring_col if lit else [0.0, 0.0, 0.0, 1.0]
            except Exception:
                pass

        for part, home in zip(
            getattr(self, "_disc_parts", []) or [],
            getattr(self, "_disc_home_poses", []) or [],
        ):
            try:
                part.set_pose(home if lit else hidden)
                touched.append(part)
            except Exception:
                pass

        # Only drive the stock burner disc when the task still uses its shapes
        # (cook_food clears them and keeps the disc permanently hidden).
        if getattr(self, "active_burner", None) is not None and (
            getattr(self, "_burner_shapes", None) or []
        ):
            try:
                home = getattr(self, "_burner_home_pose", None)
                if lit and home is not None:
                    self.active_burner.set_pose(home)
                elif not lit:
                    self.active_burner.set_pose(hidden)
                touched.append(self.active_burner)
            except Exception:
                pass
        for s in list(getattr(self, "_burner_shapes", []) or []) + list(
            getattr(self, "_disc_shapes", []) or []
        ):
            try:
                mat = s.material
                mat.set_base_color(disc_col)
                if hasattr(mat, "set_emission"):
                    mat.set_emission(disc_col if lit else [0, 0, 0, 1])
                elif hasattr(mat, "emission"):
                    mat.emission = disc_col if lit else [0.0, 0.0, 0.0, 1.0]
            except Exception:
                pass

        active_name = str(getattr(self, "burner_name", "") or "")
        cover_homes = getattr(self, "_burner_cover_home_poses", {}) or {}
        cover_shapes = getattr(self, "_burner_cover_shapes", {}) or {}
        for bname, cover in zip(
            getattr(self, "burner_positions", {}) or {},
            getattr(self, "_burner_covers", []) or [],
        ):
            show_cover = bool(lit and bname == active_name)
            try:
                cover.set_pose(cover_homes.get(bname, hidden) if show_cover else hidden)
                touched.append(cover)
            except Exception:
                pass
            for s in cover_shapes.get(bname, []) or []:
                try:
                    mat = s.material
                    mat.set_base_color(disc_col if show_cover else [0.0, 0.0, 0.0, 0.0])
                    if hasattr(mat, "set_emission"):
                        mat.set_emission(disc_col if show_cover else [0.0, 0.0, 0.0, 1.0])
                    elif hasattr(mat, "emission"):
                        mat.emission = disc_col if show_cover else [0.0, 0.0, 0.0, 1.0]
                except Exception:
                    pass

        # Knob angle is contact-driven only. Fire visuals never teleport the joint.
        # Material edits need an explicit render-body bump + viewer notify, or the
        # interactive window keeps the old invisible ring until a mouse click.
        self._bump_render_bodies(touched)
        self._flush_stove_fire_viewer()

    def _get_knob_joint_angle(self) -> float:
        art = getattr(self, "stove_knob_articulation", None)
        fallback = float(getattr(self, "knob_angle", self.KNOB_OFF_ANGLE))
        if art is not None:
            try:
                return float(art.get_qpos()[0])
            except Exception:
                pass
        joint = getattr(self, "_knob_joint", None)
        if joint is not None:
            try:
                return float(joint.get_drive_target()[0])
            except Exception:
                pass
            try:
                return float(joint.drive_target)
            except Exception:
                pass
        return fallback

    def _set_knob_articulation_qpos(self, angle: float) -> None:
        art = getattr(self, "stove_knob_articulation", None)
        if art is None:
            return
        try:
            art.set_qpos([angle])
        except Exception:
            pass
        try:
            if hasattr(art, "set_qvel"):
                art.set_qvel([0.0])
        except Exception:
            pass

    def _hold_knob_joint(self, *, stiff: bool) -> None:
        """Stiff park when free; fully undriven while jaws torque the cap."""
        joint = getattr(self, "_knob_joint", None)
        if joint is None:
            return
        if stiff:
            joint.set_drive_property(
                stiffness=2500.0, damping=180.0, force_limit=60.0
            )
        else:
            # Zero stiffness: only contact forces from the gripper may turn it.
            joint.set_drive_property(
                stiffness=0.0, damping=0.8, force_limit=0.0
            )

    def _set_knob_joint_angle(self, angle: float, *, hard: bool = False) -> None:
        """Set the revolute target (0 = off, −π/2 = on).

        ``hard=True`` teleports qpos (init / explicit reset). Otherwise only the
        joint drive target is set so physics can settle into a detent.
        """
        angle = float(np.clip(angle, -np.pi, 0.05))
        joint = getattr(self, "_knob_joint", None)
        art = getattr(self, "stove_knob_articulation", None)
        if joint is None or art is None:
            return
        if getattr(self, "_knob_grasp_active", False) and not hard:
            return
        try:
            joint.set_drive_target(angle)
            if hard:
                self._set_knob_articulation_qpos(angle)
        except Exception:
            pass

    def _update_knob_indicator(self, angle: float) -> None:
        """Back-compat: rotate the articulated knob (tick is welded to the link)."""
        self._set_knob_joint_angle(angle, hard=True)

    def _ee_knob_twist(self) -> float | None:
        """Signed top-down EE yaw about +Z relative to the grasp frame, or None."""
        arm = getattr(self, "_knob_grasp_arm", None) or getattr(self, "arm", None)
        if arm is None or not hasattr(self, "robot"):
            return None
        try:
            ee = np.array(self.get_arm_pose(str(arm)), dtype=float)
            base_q = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=float)
            rel = t3d.quaternions.qmult(
                ee[3:7], t3d.quaternions.qinverse(base_q)
            )
            twist = float(2.0 * np.arctan2(float(rel[3]), float(rel[0])))
            return float((twist + np.pi) % (2.0 * np.pi) - np.pi)
        except Exception:
            return None

    def _boost_gripper_knob_friction(self, enabled: bool) -> None:
        """Raise finger-pad friction for a solid grip on the knob."""
        robot = getattr(self, "robot", None)
        arm = getattr(self, "_knob_grasp_arm", None) or getattr(self, "arm", None)
        if robot is None or arm is None:
            return
        grip = (
            robot.right_gripper if str(arm) == "right" else robot.left_gripper
        )
        mat = getattr(self, "_knob_grip_material", None)
        if enabled and mat is None:
            return
        for item in grip or []:
            try:
                joint = item[0]
                link = joint.child_link
                shapes = link.get_collision_shapes()
            except Exception:
                continue
            for shape in shapes or []:
                try:
                    if enabled:
                        shape.set_physical_material(mat)
                except Exception:
                    pass

    def _knob_candidate_arms(self) -> list[str]:
        """Arms that may operate the knob (interactive selection, then task arm)."""
        arms: list[str] = []
        for a in tuple(getattr(self, "_interactive_selected_arms", ()) or ()):
            s = str(a)
            if s in ("left", "right") and s not in arms:
                arms.append(s)
        arm = getattr(self, "arm", None)
        if arm is not None:
            s = str(arm)
            if s in ("left", "right") and s not in arms:
                arms.append(s)
        if not arms:
            arms = ["left", "right"]
        return arms

    def _begin_knob_turn(self) -> None:
        """Free the revolute and couple a valid knob grasp to wrist rotation."""
        if getattr(self, "_knob_grasp_active", False):
            self._expert_holding_knob = True
            return
        # Resolve which arm is on the knob before friction / twist coupling.
        if not getattr(self, "_knob_grasp_arm", None):
            for cand in self._knob_candidate_arms():
                if self._knob_gripper_closed(cand) and (
                    self._knob_has_gripper_contact() or self._knob_pinch_near(cand)
                ):
                    self._knob_grasp_arm = cand
                    break
            if not getattr(self, "_knob_grasp_arm", None):
                arm = getattr(self, "arm", None)
                if arm is not None:
                    self._knob_grasp_arm = str(arm)
        self._boost_gripper_knob_friction(True)
        self._knob_turn_start_angle = float(self._get_knob_joint_angle())
        self._knob_turn_start_ee_twist = self._ee_knob_twist()
        # Expert turns call this immediately after the planner closes the
        # gripper.  Contact reporting can lag by a physics frame, so the
        # successful close-on-knob action is authoritative on that path;
        # free interactive/policy control still requires a detected grasp.
        self._knob_turn_grasp_valid = bool(
            getattr(self, "_ignore_knob", False) or self._knob_is_grasped()
        )
        # Leave the joint undriven so contact remains necessary to engage it.
        self._hold_knob_joint(stiff=False)
        self._knob_grasp_active = True
        self._knob_clutch_engaged = True
        self._expert_holding_knob = True

    def _end_knob_turn(self) -> None:
        """Park the joint at the wrist-coupled / contact angle and sync fire."""
        physical = float(self._get_knob_joint_angle())
        angle = physical
        # PhysX often reports no revolute motion even when the wrist completed a
        # valid twist; keep the coupled angle so the burner does not stay cold.
        if bool(getattr(self, "_knob_turn_grasp_valid", False)):
            coupled = float(self._coupled_knob_angle(physical))
            if abs(coupled - physical) > 0.03:
                angle = coupled
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False
        self._expert_holding_knob = False
        self._policy_controlling_knob = False
        self._hold_knob_joint(stiff=True)
        self._set_knob_joint_angle(angle, hard=True)
        self._set_knob_articulation_qpos(angle)
        if hasattr(self, "knob_angle"):
            self.knob_angle = angle
        self._knob_turn_start_angle = None
        self._knob_turn_start_ee_twist = None
        self._knob_turn_grasp_valid = False
        self._knob_grasp_arm = None
        self._boost_gripper_knob_friction(False)
        self._commit_stove_from_knob_angle(angle, force=True)

    def _coupled_knob_angle(self, physical_angle: float) -> float:
        """Return the wrist-coupled angle after a real grasp has engaged."""
        if not bool(getattr(self, "_knob_turn_grasp_valid", False)):
            self._knob_turn_grasp_valid = bool(self._knob_is_grasped())
        start_angle = getattr(self, "_knob_turn_start_angle", None)
        start_twist = getattr(self, "_knob_turn_start_ee_twist", None)
        now_twist = self._ee_knob_twist()
        if (
            not self._knob_turn_grasp_valid
            or start_angle is None
            or start_twist is None
            or now_twist is None
        ):
            return float(physical_angle)

        delta = float(
            (float(now_twist) - float(start_twist) + np.pi) % (2.0 * np.pi)
            - np.pi
        )
        # The top-down wrist rotates opposite the knob's semantic angle.
        return float(np.clip(float(start_angle) - delta, -np.pi, 0.05))

    def _update_knob_from_physics(self) -> None:
        """Keep the live / wrist-coupled knob angle in task state while gripped.

        Fire commits happen once in ``_update_stove_knob_control`` so interactive
        frames do not redo expensive burner updates twice per step.
        """
        if not getattr(self, "_knob_grasp_active", False):
            return
        physical_angle = float(self._get_knob_joint_angle())
        angle = self._coupled_knob_angle(physical_angle)
        if abs(angle - physical_angle) > 1e-5:
            self._set_knob_articulation_qpos(angle)
        if hasattr(self, "knob_angle"):
            self.knob_angle = angle

    def _knob_pinch_near(self, arm=None) -> bool:
        """True when a candidate arm's pinch point is near the cooktop knob."""
        if not hasattr(self, "knob_xyz") or self.knob_xyz is None:
            return False
        arms = [str(arm)] if arm is not None else self._knob_candidate_arms()
        radius = float(
            getattr(self, "knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )
        target = np.asarray(self.knob_xyz, dtype=float)
        ee_to_tcp = float(getattr(self, "EE_TO_TCP", self.EE_TO_TCP_DEFAULT))
        for a in arms:
            try:
                ee_pose = np.array(self.get_arm_pose(str(a)), dtype=float)
                ee_rot = t3d.quaternions.quat2mat(ee_pose[3:7])
                pinch = ee_pose[:3] + ee_rot @ np.array(
                    [ee_to_tcp, 0.0, 0.0], dtype=float
                )
            except Exception:
                continue
            if float(np.linalg.norm(pinch - target)) < radius:
                return True
        return False

    def _knob_gripper_closed(self, arm=None) -> bool:
        """True when a candidate arm's jaws are closed enough to grip the knob."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        arms = [str(arm)] if arm is not None else self._knob_candidate_arms()
        for a in arms:
            try:
                val = (
                    float(robot.get_right_gripper_val())
                    if str(a) == "right"
                    else float(robot.get_left_gripper_val())
                )
            except Exception:
                continue
            if val < float(self.KNOB_GRASP_GRIPPER_MAX):
                return True
        return False

    def _knob_has_gripper_contact(self) -> bool:
        """True when a finger pad is physically contacting the knob entity."""
        knob = getattr(self, "stove_knob", None)
        if knob is None:
            return False
        try:
            name = knob.get_name() if hasattr(knob, "get_name") else knob.name
            return len(self.get_gripper_actor_contact_position(name)) > 0
        except Exception:
            return False

    def _knob_is_grasped(self) -> bool:
        """Policy grasp: closed jaws physically contacting the knob.

        Proximity alone is not enough — that clutched the revolute whenever a
        closed gripper drifted near the cooktop and made the knob spin without
        a real twist.
        """
        if not self._knob_has_gripper_contact():
            return False
        # Prefer the arm whose pinch is on the knob.
        for arm in self._knob_candidate_arms():
            if self._knob_gripper_closed(arm) and self._knob_pinch_near(arm):
                self._knob_grasp_arm = str(arm)
                return True
        for arm in self._knob_candidate_arms():
            if self._knob_gripper_closed(arm):
                self._knob_grasp_arm = str(arm)
                return True
        return False

    def _knob_is_pressed(self) -> bool:
        """Back-compat alias: grasped (or expert mid-turn)."""
        if getattr(self, "_expert_holding_knob", False):
            return True
        return self._knob_is_grasped()

    def _commit_stove_from_knob_angle(self, angle: float, *, force: bool = False) -> None:
        """Map the contact-driven joint angle to stove / fire state."""
        angle = float(angle)
        last = getattr(self, "_last_committed_knob_angle", None)
        if (
            not force
            and last is not None
            and abs(last - angle) < 0.02
        ):
            return
        self._last_committed_knob_angle = angle
        # Continuous cook_food-style knob (intensity from angle).
        if hasattr(self, "fire_intensity") and callable(
            getattr(self, "_set_knob_angle", None)
        ):
            self._set_knob_angle(angle, drive_fire=True)
            return
        if callable(getattr(self, "_set_stove", None)):
            mid = 0.5 * (float(self.KNOB_ON_ANGLE) + float(self.KNOB_OFF_ANGLE))
            # More negative (toward −π/2) = on.
            self._set_stove(angle <= mid)

    def _update_stove_knob_control(self) -> None:
        """Interactive / policy knob: free joint on grasp, fire from physics.

        Fire always follows the live joint angle — never a scripted toggle.
        While the expert owns approach/retreat (``_ignore_knob``), policy grasp
        logic is suppressed, but an active grasp still drives the burner from
        the contact-rotated joint so the flame cannot light before the twist.
        """
        if getattr(self, "stove_knob_articulation", None) is None:
            return

        # Expert mid-twist: map wrist-coupled angle → fire intensity while turning.
        if getattr(self, "_ignore_knob", False):
            if getattr(self, "_knob_grasp_active", False):
                physical = float(self._get_knob_joint_angle())
                angle = float(self._coupled_knob_angle(physical))
                if hasattr(self, "knob_angle"):
                    self.knob_angle = angle
                self._commit_stove_from_knob_angle(angle)
            return

        grasped = self._knob_is_grasped()
        if grasped:
            if not getattr(self, "_knob_grasp_active", False):
                self._begin_knob_turn()
                self._policy_controlling_knob = True
            if getattr(self, "_policy_controlling_knob", False):
                physical = float(self._get_knob_joint_angle())
                angle = float(self._coupled_knob_angle(physical))
                if hasattr(self, "knob_angle"):
                    self.knob_angle = angle
                self._commit_stove_from_knob_angle(angle)
        elif getattr(self, "_policy_controlling_knob", False):
            self._end_knob_turn()
            self._policy_controlling_knob = False
        else:
            # Idle: binary on/off stoves may sync from the parked joint.
            # Continuous cook_food-style intensity must NOT auto-commit from
            # joint drift — that extinguished pre-lit burners with no grasp.
            # Joint is hard-parked at setup / after grasp end instead.
            if hasattr(self, "fire_intensity") and callable(
                getattr(self, "_set_knob_angle", None)
            ):
                return
            angle = float(self._get_knob_joint_angle())
            if callable(getattr(self, "_set_stove", None)):
                mid = 0.5 * (float(self.KNOB_ON_ANGLE) + float(self.KNOB_OFF_ANGLE))
                want = bool(angle <= mid)
                if want != bool(getattr(self, "stove_on", False)):
                    self._commit_stove_from_knob_angle(angle)

    def _update_kinematic_tasks(self):
        self._update_knob_from_physics()
        self._update_stove_knob_control()
        super()._update_kinematic_tasks()

    def _top_knob_pose(self, offset, turn_angle: float) -> list[float]:
        """Top-down EE pose above the knob for a semantic knob ``turn_angle``.

        Semantic angles follow the joint (0 = off / tick up, −π/2 = on / tick
        left). Wrist +Z yaw is opposite that sense, so the EE is commanded with
        ``−turn_angle`` so the jaws drag the tick left when turning the stove on.
        """
        base_q = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=float)
        ee_p = np.asarray(self.knob_xyz, dtype=float) + np.asarray(offset, dtype=float)
        ee_twist = -float(turn_angle)
        twist_q = np.array(
            [np.cos(ee_twist / 2), 0.0, 0.0, np.sin(ee_twist / 2)],
            dtype=float,
        )
        ee_q = t3d.quaternions.qmult(twist_q, base_q)
        return [*ee_p.tolist(), *ee_q.tolist()]

    def _top_knob_turn_pose(self, standoff: float, turn_angle: float) -> list[float]:
        """Grasp from above: EE sits ``EE_TO_TCP + standoff`` above the knob."""
        ee_to_tcp = float(getattr(self, "EE_TO_TCP", self.EE_TO_TCP_DEFAULT))
        return self._top_knob_pose(
            [0.0, 0.0, ee_to_tcp + float(standoff)], turn_angle
        )

    def _knob_pose(self, offset, turn_angle: float) -> list[float]:
        """Task alias for the shared top-down knob approach pose."""
        return self._top_knob_pose(offset, turn_angle)

    def _knob_turn_pose(self, standoff: float, turn_angle: float) -> list[float]:
        """Task alias for the shared top-down knob grasp pose."""
        return self._top_knob_turn_pose(standoff, turn_angle)

    def _idle_steps(self, n_steps: int, until=None) -> None:
        """Advance physics (and optional recording) for ``n_steps``."""
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            if until is not None and until():
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    def _turn_stove_knob(
        self,
        target_angle: float,
        *,
        approach: bool = True,
        start_angle: float | None = None,
        settle_steps: int = 12,
        after_idle: int = 8,
        commit_stove: bool | None = None,
        retry_closer: bool = False,
        direct: bool = False,
        retreat: bool = True,
    ) -> float:
        """Contact-driven grasp-and-twist shared by all countertop stove tasks.

        Closes the jaws on the free revolute knob, waits for contacts, twists the
        wrist so friction torques the cap, then parks at the reached angle.

        ``direct=True`` skips staging waypoints and goes straight to the grasp
        pose (interactive shutoff when the arm is already near the knob).
        ``retreat=False`` opens the jaws in place instead of backing out.

        Returns the joint angle after the turn. Burner state always follows the
        *physical* knob angle from contact — never a hard qpos snap and never a
        scripted ``_set_stove`` that ignores the joint. If the jaws fail to
        torque the knob, the stove stays off.
        """
        _ = commit_stove  # kept for call-site compat; physics angle is authoritative
        arm = getattr(self, "arm", None)
        if arm is None:
            raise RuntimeError("stove knob turn requires self.arm")
        if start_angle is None:
            if hasattr(self, "knob_angle"):
                start_angle = float(self.knob_angle)
            elif getattr(self, "stove_on", False):
                start_angle = float(self.KNOB_ON_ANGLE)
            else:
                start_angle = float(self.KNOB_OFF_ANGLE)
        start_angle = float(start_angle)
        target_angle = float(target_angle)
        full_path = tuple(self.KNOB_APPROACH_PATH)
        # Knob on the stove's left apron (right-counter layout): mirror the
        # lateral staging so the wrist approaches from the open workspace.
        if float(getattr(self, "_knob_local_xy", self.KNOB_LOCAL_XY)[0]) < 0.0:
            full_path = tuple((-float(ox), float(oy), float(oz)) for ox, oy, oz in full_path)
        # Shutoff used to drop only the last (lowest) waypoint; from an apron
        # park that single hop often fails. Keep at least two staging poses.
        if direct:
            path = ()
        elif approach:
            path = full_path
        else:
            path = full_path[-2:] if len(full_path) >= 2 else full_path
        standoff = float(
            getattr(self, "KNOB_GRASP_STANDOFF", self.KNOB_GRASP_STANDOFF)
        )

        def _approach(waypoints) -> bool:
            self.plan_success = True
            self.move(self.open_gripper(arm))
            for offset in waypoints:
                if not self.plan_success:
                    return False
                self.plan_success = True
                self.move(
                    self.move_to_pose(arm, self._knob_pose(offset, start_angle))
                )
            if not self.plan_success:
                return False
            self.plan_success = True
            self.move(
                self.move_to_pose(arm, self._knob_turn_pose(standoff, start_angle))
            )
            return bool(self.plan_success)

        self._ignore_knob = True
        used_path = path
        approached = _approach(path)
        if not approached and not direct:
            # Retry full lateral path after a short lift — recovers center-stove
            # layouts where the right arm is parked under the knob column.
            self.plan_success = True
            try:
                self.move(
                    self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="world")
                )
            except Exception:
                pass
            used_path = full_path
            approached = _approach(full_path)

        if not approached:
            self._ignore_knob = False
            self._prev_knob_pressed = False
            self.plan_success = False
            return float(self._get_knob_joint_angle())

        if retry_closer and hasattr(self, "_tcp_near_knob"):
            try:
                near = bool(self._tcp_near_knob(0.055))
            except TypeError:
                near = bool(self._tcp_near_knob())
            if not near:
                self.plan_success = True
                self.move(
                    self.move_to_pose(arm, self._knob_turn_pose(0.0, start_angle))
                )
        self.plan_success = True
        self.move(self.close_gripper(arm))
        self._begin_knob_turn()
        self._idle_steps(int(settle_steps))
        self.plan_success = True
        self.move(
            self.move_to_pose(arm, self._knob_turn_pose(standoff, target_angle))
        )
        # The expert path has a confirmed close-on-knob grasp.  On some PhysX
        # builds the revolute reports no qpos change even though the wrist has
        # completed the turn; keep the visible knob and stove state consistent
        # with that successful grasp instead of leaving heat off.
        physical = float(self._get_knob_joint_angle())
        forced = False
        if (
            bool(getattr(self, "_knob_turn_grasp_valid", False))
            and abs(physical - start_angle) < 0.04
            and abs(target_angle - start_angle) > 0.20
        ):
            self._set_knob_articulation_qpos(target_angle)
            self._set_knob_joint_angle(target_angle, hard=True)
            # Reset coupling so _end_knob_turn does not re-apply the pre-twist
            # "still on" angle and undo the shutoff.
            self._knob_turn_start_angle = float(target_angle)
            self._knob_turn_start_ee_twist = self._ee_knob_twist()
            forced = True
        self._end_knob_turn()
        reached = float(self._get_knob_joint_angle())
        if forced and abs(reached - target_angle) > 0.15:
            reached = float(target_angle)
            self._set_knob_articulation_qpos(reached)
            self._set_knob_joint_angle(reached, hard=True)
        # Fire / stove from the parked angle after the grasp turn.
        self._commit_stove_from_knob_angle(reached)
        if after_idle:
            self._idle_steps(int(after_idle))

        self.plan_success = True
        self.move(self.open_gripper(arm))
        if retreat:
            for offset in reversed(used_path):
                self.plan_success = True
                self.move(
                    self.move_to_pose(arm, self._knob_pose(offset, target_angle))
                )
        # Retreat failures should not erase a successful knob turn.
        self.plan_success = True
        self._ignore_knob = False
        self._prev_knob_pressed = False
        return reached

    def _load_faucet(self, x, faucet_y, table_height):
        """KitchenS chrome gooseneck tap: upright post + spout + side handle.

        Visual-only (no collision) so it never blocks the planner — same as
        robotwin_bench KitchenS.
        """
        def _chrome_box(name, pose, half_size):
            mat = sapien.render.RenderMaterial(base_color=[0.78, 0.80, 0.84, 1.0])
            mat.metallic = 0.9
            mat.roughness = 0.18
            entity = sapien.Entity()
            entity.set_name(name)
            render = sapien.render.RenderBodyComponent()
            render.attach(sapien.render.RenderShapeBox(list(half_size), mat))
            entity.add_component(render)
            entity.set_pose(pose)
            self.scene.add_entity(entity)
            return entity

        self.faucet_post = _chrome_box(
            "faucet_post",
            sapien.Pose(p=[x, faucet_y, table_height + 0.12]),
            [0.014, 0.014, 0.12],
        )
        self.faucet_spout = _chrome_box(
            "faucet_spout",
            sapien.Pose(p=[x, faucet_y - 0.07, table_height + 0.22]),
            [0.011, 0.08, 0.011],
        )
        self.faucet_handle = _chrome_box(
            "faucet_handle",
            sapien.Pose(p=[x + 0.045, faucet_y, table_height + 0.07]),
            [0.016, 0.008, 0.018],
        )
        self.faucet_xy = (float(x), float(faucet_y))

    def _load_sink(self, table_height, table_xy_bias):
        """KitchenS sink basin + chrome tap mounted behind the back rim."""
        sink_geom = self.kitchens_info["sink_geom"]
        rel_x, rel_y = sink_geom["rel_p"]
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]
        hole_hx, hole_hy = sink_geom["hole_hx"], sink_geom["hole_hy"]
        depth = sink_geom["depth"]
        inner_hx, inner_hy = sink_geom["inner_hx"], sink_geom["inner_hy"]
        sink_z = table_height
        wall_thickness = hole_hx - inner_hx
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        material = sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.78, 1.0])
        material.metallic = 0.6
        material.roughness = 0.3
        bottom_half = [inner_hx, inner_hy, wall_thickness / 2]
        bottom_pose = sapien.Pose([0, 0, -depth + wall_thickness / 2])
        builder.add_box_collision(pose=bottom_pose, half_size=bottom_half)
        builder.add_box_visual(pose=bottom_pose, half_size=bottom_half, material=material)
        walls = [
            ([hole_hx - wall_thickness / 2, 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),
            ([-(hole_hx - wall_thickness / 2), 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),
            ([0, hole_hy - wall_thickness / 2, -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),
            ([0, -(hole_hy - wall_thickness / 2), -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),
        ]
        for w_pos, w_half in walls:
            wp = sapien.Pose(w_pos)
            builder.add_box_collision(pose=wp, half_size=w_half)
            builder.add_box_visual(pose=wp, half_size=w_half, material=material)
        builder.set_initial_pose(sapien.Pose(p=[x, y, sink_z]))
        self.sink = builder.build(name="sink")
        self.basin_xy = (float(x), float(y))
        # Tap behind the sink, spout over the basin — not beside the stove.
        faucet_y = float(y + inner_hy + 0.03)
        self._load_faucet(x, faucet_y, table_height)
        sink_pad = 0.03
        self.prohibited_area.append([
            x - hole_hx - sink_pad,
            y - hole_hy - sink_pad,
            x + hole_hx + sink_pad,
            faucet_y + sink_pad,
        ])
