"""Full Robotwin office scene port for RoboDynaExp dynamic tasks.

Reproduces the basic office furniture layout from
``robotwin_bench/benchmark/bench_envs/office/_office_base_task.py`` while
retaining RoboDynaExp's dynamic-task hooks and collector.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from ._base_task import Base_Task
from .utils import *


class Office_base_task(Base_Task):
    """RoboDynaExp task base with the complete Robotwin office furniture."""

    FURNITURE_NAMES = {
        "table",
        "wall",
        "ground",
        "121_wall-shelf",
        "036_cabinet",
        "042_wooden_box",
        "122_file-holder",
    }

    def setup_demo(self, **kwags):
        # The Robotwin office uses a fixed 0.74 m table height.
        kwags.setdefault("domain_randomization", {})["random_table_height"] = 0
        # Layout variant is drawn in create_table_and_wall, after the episode
        # seed is applied, so a given seed always builds the same office.
        self.arr_v = 0
        self.office_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [-0.6, -0.35, 0.6, 0.35],
            "shelf_heights": [0.9, 1.127],
            "shelf_area": [0.62, 0.26],
            "shelf_lims": [],
            "shelf_padding": 0.09,
            "file_holder_area": [0.22, 0.16],
            "file_holder_lims": [],
            "file_holder_heights": [0.82, 0.942],
            "drawer_height": 0.76,
            "furn_x_v": {
                "shelf": [-0.24, 0.0, 0.24],
                "cabinet": [0.23, 0.48, -0.48],
                "file_holder": [0.48, -0.48, -0.23],
            },
        }
        super()._init_task_env_(**kwags)

    def _create_bench_glb(self, model_name, pose, scale, mass=0.1):
        """Load one of the office-only GLBs under assets/dyna_assets."""
        model_dir = resolve_model_dir(model_name)
        glb = model_dir / "base.glb"
        if not glb.exists():
            candidates = sorted(model_dir.glob("*.glb"))
            if not candidates:
                raise FileNotFoundError(f"No GLB found in {model_dir}")
            glb = candidates[0]
        scale = [float(scale)] * 3 if isinstance(scale, (int, float)) else list(scale)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_nonconvex_collision_from_file(filename=str(glb), scale=scale)
        builder.add_visual_from_file(filename=str(glb), scale=scale)
        builder.set_initial_pose(pose)
        actor = builder.build(name=model_name)
        return actor

    def _create_scaled_static_object(self, model_name, model_id, pose, scale, collision=True):
        """Load a regular object with an explicit non-uniform scale.

        ``042_wooden_box/model_data0.json`` has no authored scale, so this
        direct builder preserves the exact office-base scale.
        """
        model_dir = resolve_model_dir(model_name)
        collision_path = model_dir / "collision" / f"base{model_id}.glb"
        visual = model_dir / "visual" / f"base{model_id}.glb"
        if not visual.exists():
            visual = model_dir / f"base{model_id}.glb"
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision and collision_path.exists():
            builder.add_nonconvex_collision_from_file(
                filename=str(collision_path), scale=scale,
            )
        builder.add_visual_from_file(filename=str(visual), scale=scale)
        builder.set_initial_pose(pose)
        return builder.build(name=model_name)

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Build the complete basic office scene with Robotwin dimensions."""
        # Subclasses may restrict layouts via `_office_arr_choices` (e.g. skip the
        # centered-shelf arrangement when an arm needs a clear side lane).
        choices = getattr(self, "_office_arr_choices", [0, 1, 2])
        self.arr_v = int(np.random.choice(list(choices)))
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

        # Same four floor tiles as the Robotwin office base.
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

        depth = 0.28
        shelf_x = self.office_info["furn_x_v"]["shelf"][self.arr_v]
        self.shelf = self._create_bench_glb(
            "121_wall-shelf",
            sapien.Pose(
                p=[shelf_x, depth, table_height + 0.27],
                q=[0.7071, 0.7071, 0.0, 0.0],
            ),
            scale=[1.7, 0.86, 1.8],
            mass=2.0,
        )
        sx, sy = self.office_info["shelf_area"]
        self.office_info["shelf_lims"] = [
            shelf_x - sx / 2,
            depth - sy / 2,
            shelf_x + sx / 2,
            depth + sy / 2,
        ]
        self.shelf_lims = list(self.office_info["shelf_lims"])
        self.prohibited_area.append([
            self.shelf_lims[0] - 0.03,
            self.shelf_lims[1] - 0.02,
            self.shelf_lims[2] + 0.03,
            self.shelf_lims[3] + 0.02,
        ])

        cabinet_x = self.office_info["furn_x_v"]["cabinet"][self.arr_v]
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            modelname="036_cabinet",
            modelid=46653,
            pose=sapien.Pose(
                p=[cabinet_x, depth, table_height],
                q=[0.7071, 0.0, 0.0, 0.7071],
            ),
            fix_root_link=True,
        )
        self.cabinet.set_mass(0.5)
        # Exact cabinet footprint reservation from its office location.
        self.prohibited_area.append([
            cabinet_x - 0.13, depth - 0.14,
            cabinet_x + 0.13, depth + 0.14,
        ])

        holder_x = self.office_info["furn_x_v"]["file_holder"][self.arr_v]
        holder_q = euler2quat(-np.pi / 2, 0, 0, axes="sxyz")
        self.wooden_box = self._create_scaled_static_object(
            "042_wooden_box",
            0,
            sapien.Pose(
                p=[holder_x, depth - 0.06, 0.813],
                q=holder_q,
            ),
            scale=[0.09, 0.07, 0.1],
        )
        self.file_holder = self._create_bench_glb(
            "122_file-holder",
            sapien.Pose(
                p=[holder_x, depth - 0.05, table_height + 0.075],
                q=[0.7071, 0.7071, 0.0, 0.0],
            ),
            scale=[0.38, 0.7, 0.4],
            mass=0.1,
        )
        fx, fy = self.office_info["file_holder_area"]
        self.office_info["file_holder_lims"] = [
            holder_x - fx / 2,
            depth - 0.05 - fy / 2,
            holder_x + fx / 2,
            depth - 0.05 + fy / 2,
        ]
        self.prohibited_area.append([
            self.office_info["file_holder_lims"][0] - 0.01,
            self.office_info["file_holder_lims"][1],
            self.office_info["file_holder_lims"][2] + 0.01,
            self.office_info["file_holder_lims"][3],
        ])
