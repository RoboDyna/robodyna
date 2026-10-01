"""Trap a bug that scurries out from under the office wall-shelf.

Scene mirrors the basic office bench layout (`121_wall-shelf` at the back of
the table, from robotwin_bench office). A cockroach, spider, or ant emerges
from under the bookshelf and runs left or right. The robot must place a hollow
~50%-transparent glass trap box over the still-moving bug (never paused for the
place). On success the bug stops under the trap; trap collision stays on.
The trap is always kinematic (welded while carried, never unlocked; mass
0.5 kg vs a 4 g bug so a landed trap cannot be shoved). Evaluation arms when
the gripper opens or the trap leaves the hand; the landing pose is frozen
as-is (no reseat). The bug only stays out for ``walk_time`` seconds, then
retreats under the shelf (task fails). The bug's heading always matches its
travel direction.
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.physx
import sapien.render
from transforms3d.euler import euler2quat

from ._office_base_task import Office_base_task
from .utils import *
from .utils.actor_utils import Actor
from ._GLOBAL_CONFIGS import *


class trap_bug(Office_base_task):
    BUG_SPEED_MIN = 0.07
    BUG_SPEED_MAX = 0.14
    TRAP_HALF = [0.045, 0.045, 0.028]
    TRAP_WALL = 0.004                 # visual wall thickness
    # Collision walls thicker than the glass visual so WSG fingers meet solid
    # outer faces instead of tunneling into the hollow cavity.
    TRAP_WALL_COLLISION = 0.012
    TRAP_MASS = 0.5              # 500 g box; 4 g bug cannot shove it once landed
    # WSG gap ≈ 2 * |joint|; joint spans ~0.055 m → ~0.11 m fully open.
    # Outer box width is 2*hx ≈ 0.09 m → hold near gripper_val ≈ 0.78.
    TRAP_HOLD_GRIPPER = 0.78
    TRAP_LATCH_GVAL_MAX = 0.90   # latch while closing around the outer walls
    TRAP_RELEASE_GVAL_MIN = 0.94 # release only when nearly fully open
    # Trap is released this far above its seated height, so the fingers clear
    # the rim instead of scraping the table.
    RELEASE_CLEARANCE = 0.05
    HOVER_LIFT = 0.10            # extra height while carrying to the intercept
    INTERCEPT_AHEAD = 0.14       # how far down the run line the trap is parked
    WAIT_STEPS = 400             # per attempt, waiting for the bug to arrive
    # Fraction of the gripper-open action that elapses before the box lets go
    # (measured on the aloha gripper: release at ~170 of 300 steps).
    RELEASE_FRACTION = 0.57
    RELEASE_LAG_STEPS = 170      # fallback when the gripper plan is unavailable
    EMERGE_STEPS = 40
    # Seconds the bug stays out (emerge + scuttle) before retreating to hide.
    WALK_TIME = 15.0
    # Max heading slew rate (rad/s) so the bug turns into its travel direction
    # instead of snapping when velocity wobbles or walls bounce it.
    TURN_RATE = 10.0
    # Species catalog. Flat-on-table uses local Y → world Z.
    # `forward_y` is which way the mesh head points at yaw=0 (+1 → +Y, -1 → -Y).
    # All current bug meshes face -Y at yaw=0 after flattening, so travel yaw
    # is arctan2(vx, forward_y * vy) with forward_y=-1 (head always leads).
    BUG_SPECIES = {
        "cockroach": {
            "model": "200_cockroach",
            "label": "cockroach",
            "half_h": 0.003,
            "forward_y": -1.0,
            "mass": 0.004,  # 4 g
        },
        "spider": {
            "model": "201_spider",
            "label": "spider",
            "half_h": 0.008,
            "forward_y": -1.0,
            "mass": 0.004,  # 4 g
        },
        "ant": {
            "model": "202_ant",
            "label": "ant",
            "half_h": 0.004,
            "forward_y": -1.0,
            "mass": 0.004,  # 4 g
        },
    }
    DEFAULT_BUG_TYPES = ("cockroach", "spider", "ant")

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("trap_bug", {})
        # Init before _init_task_env_ — load_camera may call _update_kinematic_tasks.
        self._loaded = False
        self.bug = None
        self.trap = None
        self._bug_rigid = None
        self._bug_moving = False
        self._bug_vel = np.zeros(3, dtype=np.float64)
        self._bug_phase = 0.0
        self._bug_mode = "emerge"
        self._bug_yaw = 0.0
        self._bug_half_h = 0.003
        self._bug_forward_y = -1.0
        self._bug_walk_elapsed = 0.0
        self._bug_escaped = False
        self.walk_time = float(self.WALK_TIME)
        self.bug_type = "cockroach"
        self.bug_model = "200_cockroach"
        self.run_sign = 1
        self.arm_side = "right"
        self._trap_anchored = False
        self._trap_released = False  # gripper opened / trap left the hand
        self._partial_score = 0.0  # latched at trap landing (bug keeps running after)
        self._reset_metric_state()
        self._trap_falling = False
        self._trap_welded = False
        self._trap_weld_offset = None
        self._trap_weld_arm = None
        self._trap_rigid = None
        self._trap_anchor_pose = None
        self._trap_collision_groups_backup = None
        self._bug_captured = False
        self._sim_steps = 0
        # Bias the office shelf to the requested run side (skip centered shelf so
        # left/right lanes stay unambiguous). Trap always spawns on that same side.
        run_side = str(self._cfg.get("run_side", "random")).lower().strip()
        if run_side == "left":
            self._office_arr_choices = [0]
        elif run_side == "right":
            self._office_arr_choices = [2]
        else:
            self._office_arr_choices = [0, 2]
        super().setup_demo(**kwags)
        self._configure_observer_camera()
        # After settle so the bug is not moving during check_stable.
        self._start_bug()

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.42, 0.55, 1.50], dtype=np.float64)
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
                rigid.set_kinematic_target(pose)
                return
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    # -------------------------------------------------------------- glass trap
    def _make_glass_material(self):
        """Nearly clear glass (high transmission, low alpha) so the trapped bug stays visible."""
        glass = sapien.render.RenderMaterial(base_color=[0.88, 0.95, 0.98, 0.12])
        glass.set_transmission(0.92)
        glass.set_transmission_roughness(0.01)
        glass.set_roughness(0.03)
        glass.set_metallic(0.0)
        try:
            glass.set_ior(1.45)
        except Exception:
            glass.ior = 1.45
        return glass

    def _make_plain_trap_material(self):
        """Simple alpha-transparent plastic — no glass transmission/IOR (viewer-friendly)."""
        # Darker blue-gray + higher alpha so the box reads clearly on the white table.
        mat = sapien.render.RenderMaterial(base_color=[0.18, 0.32, 0.48, 0.55])
        mat.set_transmission(0.0)
        try:
            mat.set_transmission_roughness(1.0)
        except Exception:
            pass
        mat.set_roughness(0.55)
        mat.set_metallic(0.0)
        try:
            mat.set_ior(1.0)
        except Exception:
            mat.ior = 1.0
        return mat

    def _create_glass_trap(self, pose: sapien.Pose) -> Actor:
        """Hollow open-bottom square box (reverse box) with glass or plain visuals."""
        hx, hy, hz = [float(v) for v in self.trap_half]
        wt_vis = float(self.trap_wall)
        wt_col = float(getattr(self, "trap_wall_collision", self.TRAP_WALL_COLLISION))
        wt_col = max(wt_col, wt_vis)
        scene = self.scene
        builder = scene.create_actor_builder()
        # Always kinematic: policies cannot call unlock, and the bug must not shove it.
        builder.set_physx_body_type("kinematic")

        def _parts(wt: float):
            top_hz = wt / 2.0
            side_hz = hz - top_hz
            side_z = -hz + side_hz
            return [
                (sapien.Pose([0, 0, hz - top_hz]), [hx, hy, top_hz]),
                (sapien.Pose([hx - wt / 2, 0, side_z]), [wt / 2, hy, side_hz]),
                (sapien.Pose([-hx + wt / 2, 0, side_z]), [wt / 2, hy, side_hz]),
                (sapien.Pose([0, hy - wt / 2, side_z]), [hx - wt, wt / 2, side_hz]),
                (sapien.Pose([0, -hy + wt / 2, side_z]), [hx - wt, wt / 2, side_hz]),
            ]

        # Grippy, dead-bounce glass so the released box lands where it was
        # dropped instead of skating off the bug.
        material = sapien.physx.PhysxMaterial(
            static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        for local_pose, half in _parts(wt_col):
            builder.add_box_collision(
                pose=local_pose,
                half_size=half,
                material=material,
            )

        builder.set_initial_pose(pose)
        entity = builder.build(name="glass_trap")

        # Interactive / viewer: plain alpha transparency. Expert demos: glass.
        if bool(getattr(self, "_plain_trap", False)):
            visual = self._make_plain_trap_material()
        else:
            visual = self._make_glass_material()
        render_body = sapien.render.RenderBodyComponent()
        for local_pose, half in _parts(wt_vis):
            shape = sapien.render.RenderShapeBox(half, visual)
            shape.set_local_pose(local_pose)
            render_body.attach(shape)
        entity.add_component(render_body)

        data = {
            "center": [0, 0, 0],
            "extents": [hx * 2, hy * 2, hz * 2],
            "scale": [1, 1, 1],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            # Top-lid center, four yaw approaches from above. Grasp pinches the
            # *outer* walls (gripper held near TRAP_HOLD_GRIPPER), not a full close
            # into the hollow cavity.
            "contact_points_pose": [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -hz],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "contact_points_description": ["Top lid center, pinch outer walls from above."],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": ["Bottom rim of the open trap."],
        }
        return Actor(entity, data)

    def _yaw_quat_flat(self, yaw: float) -> np.ndarray:
        """Flat on table (mesh local Y → world Z) + heading about world Z."""
        return np.asarray(euler2quat(np.pi / 2, 0, float(yaw), axes="sxyz"), dtype=np.float64)

    def _heading_yaw(self, vx: float, vy: float) -> float:
        """Yaw that points the bug's head along velocity (world XY).

        At yaw=0 the mesh head faces ``(0, forward_y)``. Rotating by yaw
        about world Z, the matching travel heading is
        ``arctan2(vx, forward_y * vy)``.
        """
        if abs(vx) + abs(vy) < 1e-6:
            vx, vy = float(self.run_sign), 0.0
        return float(np.arctan2(vx, self._bug_forward_y * vy))

    def _select_bug_species(self):
        """Pick bug species from ``bug_type`` / ``species`` or randomize.

        Config:
          - ``bug_type`` / ``species``: ``cockroach`` | ``spider`` | ``ant`` | ``random``
          - ``bug_types``: pool used when random (default all three)
        """
        c = self._cfg
        forced = c.get("bug_type", c.get("species", "random"))
        choices = list(c.get("bug_types", self.DEFAULT_BUG_TYPES))
        key = str(forced).lower().strip() if forced is not None else "random"
        if key in ("", "random", "none", "any"):
            key = str(np.random.choice(choices)).lower().strip()
        if key not in self.BUG_SPECIES:
            raise ValueError(
                f"unknown trap_bug species: {forced!r} "
                f"(expected one of {list(self.BUG_SPECIES)} or 'random')"
            )
        spec = self.BUG_SPECIES[key]
        self.bug_type = key
        self.bug_model = spec["model"]
        self._bug_half_h = float(spec["half_h"])
        self._bug_forward_y = float(np.sign(spec.get("forward_y", -1.0)) or -1.0)
        self._bug_mass = float(spec["mass"])
        return spec

    def _resolve_run_side(self, shelf_x: float) -> int:
        """+1 = bug/trap on right, −1 = left. Honors ``run_side`` config."""
        run_side = str(self._cfg.get("run_side", "random")).lower().strip()
        if run_side == "left":
            return -1
        if run_side == "right":
            return 1
        # random / unset: follow shelf offset, or coin-flip if shelf is centered.
        if abs(shelf_x) > 1e-6:
            return int(np.sign(shelf_x))
        return int(np.random.choice([-1, 1]))

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.trap_half = list(c.get("trap_half", self.TRAP_HALF))
        self.trap_wall = float(c.get("trap_wall", self.TRAP_WALL))
        self.trap_wall_collision = float(c.get(
            "trap_wall_collision", self.TRAP_WALL_COLLISION))
        # plain_trap: opaque-ish alpha box (no glass transmission) for interactive viewers.
        self._plain_trap = bool(c.get("plain_trap", False))
        self.trap_hold_gripper = float(c.get(
            "trap_hold_gripper", self.TRAP_HOLD_GRIPPER))
        self.trap_latch_gval_max = float(c.get(
            "trap_latch_gval_max", self.TRAP_LATCH_GVAL_MAX))
        self.trap_release_gval_min = float(c.get(
            "trap_release_gval_min", self.TRAP_RELEASE_GVAL_MIN))
        self.release_clearance = float(
            c.get("release_clearance", self.RELEASE_CLEARANCE))
        self.emerge_steps = int(c.get("emerge_steps", self.EMERGE_STEPS))
        self.walk_time = float(c.get("walk_time", self.WALK_TIME))
        if self.walk_time <= 0.0:
            raise ValueError("trap_bug walk_time must be > 0")
        # Accept legacy roach_speed_* keys from earlier configs.
        speed_min = float(c.get("bug_speed_min", c.get("roach_speed_min", self.BUG_SPEED_MIN)))
        speed_max = float(c.get("bug_speed_max", c.get("roach_speed_max", self.BUG_SPEED_MAX)))
        if speed_min <= 0 or speed_max < speed_min:
            raise ValueError("trap_bug bug_speed_min/max invalid")

        self._select_bug_species()

        self.table_top = float(self.office_info["table_height"])
        shelf_x = float(self.office_info["furn_x_v"]["shelf"][self.arr_v])
        shelf_front_y = float(self.shelf_lims[1])

        # Bug runs left or right; trap is placed on that same side so the arm
        # can pick it up and drop it over the bug (reachability is side-bound).
        self.run_sign = self._resolve_run_side(shelf_x)
        self.arm_side = "right" if self.run_sign > 0 else "left"

        # Spawn under the bookshelf front lip, head already aimed along the run.
        spawn_x = float(np.clip(
            shelf_x + self.run_sign * np.random.uniform(0.02, 0.10),
            -0.28, 0.28,
        ))
        spawn_y = float(shelf_front_y + 0.01)
        spawn_z = self.table_top + self._bug_half_h
        self._bug_start = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)
        # Initially facing out from under the shelf (emerge direction = -Y).
        self._bug_yaw = self._heading_yaw(0.0, -1.0)
        bug_q = self._yaw_quat_flat(self._bug_yaw)

        self.bug = create_actor(
            self,
            pose=sapien.Pose(p=self._bug_start.tolist(), q=bug_q.tolist()),
            modelname=self.bug_model,
            model_id=0,
            convex=True,
            is_static=False,
        )
        self.bug.set_mass(self._bug_mass)
        self._bug_rigid = self._make_kinematic(self.bug)
        self._set_entity_pose(
            self.bug, sapien.Pose(self._bug_start.tolist(), bug_q.tolist()))

        self._base_speed = float(np.random.uniform(speed_min, speed_max))
        self._bug_phase = 0.0
        self._bug_moving = False
        self._bug_mode = "emerge"
        self._bug_walk_elapsed = 0.0
        self._bug_escaped = False
        self._emerge_target_y = float(np.clip(
            spawn_y - np.random.uniform(0.10, 0.16),
            -0.22, 0.12,
        ))
        # Stay on the arm's half after emerging so place stays reachable.
        if self.run_sign > 0:
            xmin, xmax = 0.02, 0.32
        else:
            xmin, xmax = -0.32, -0.02
        self._table_bounds = (xmin, xmax, -0.25, min(shelf_front_y - 0.02, 0.12))

        # Trap on the same side the bug will run toward (pickable by that arm).
        trap_x = float(np.clip(
            self.run_sign * np.random.uniform(0.16, 0.28),
            -0.30, 0.30,
        ))
        trap_y = float(np.random.uniform(-0.18, -0.05))
        trap_z = self.table_top + self.trap_half[2]
        self.trap = self._create_glass_trap(
            sapien.Pose(p=[trap_x, trap_y, trap_z], q=[1, 0, 0, 0]))
        print(
            f"[trap_bug] bug={self.bug_type} run_side="
            f"{'right' if self.run_sign > 0 else 'left'} "
            f"arm={self.arm_side} trap=({trap_x:.2f},{trap_y:.2f})"
        )
        self.trap_mass = float(c.get("trap_mass", self.TRAP_MASS))
        try:
            self.trap.set_mass(self.trap_mass)
        except Exception:
            pass
        self._trap_rigid = self._get_rigid(self.trap)
        if self._trap_rigid is not None:
            try:
                self._trap_rigid.set_kinematic(True)
                self._trap_rigid.set_kinematic_target(self.trap.get_pose())
            except Exception:
                pass
        self._trap_anchored = False
        self._trap_released = False
        self._partial_score = 0.0
        self._trap_falling = False
        self._trap_welded = False
        self._trap_weld_offset = None
        self._trap_weld_arm = None
        self._trap_anchor_pose = None
        self._reset_metric_state()
        self._trap_collision_groups_backup = None

        self.add_prohibit_area(self.trap, padding=0.04)
        try:
            self.prohibited_area.append([
                self._table_bounds[0] - 0.02, self._emerge_target_y - 0.08,
                self._table_bounds[1] + 0.02, spawn_y + 0.04,
            ])
        except Exception:
            pass

        self._loaded = True

    # ---------------------------------------------------------- bug motion
    def _trap_overlaps_bug_height(self) -> bool:
        """True when the hollow trap walls can touch the bug on the tabletop."""
        if self.trap is None:
            return False
        trap_z = float(self.trap.get_pose().p[2])
        hz = float(self.trap_half[2])
        bug_z = float(self.table_top + self._bug_half_h)
        # Wall bottoms are at ~trap_z - hz; collide once they reach the bug.
        return (trap_z - hz) <= (bug_z + 0.012) and (trap_z + hz) >= (bug_z - 0.005)

    def _resolve_trap_collision(self, cur, new_p, vx, vy):
        """Keep the kinematic bug from tunneling through the hollow trap walls.

        Both actors are kinematic, so PhysX will not separate them — we resolve
        XY against the trap's wall boxes in the trap local frame whenever the
        trap is low enough to overlap the bug. The open cavity stays free so a
        drop-over capture still works; only the walls block.
        """
        if self.trap is None or self._bug_captured or not self._trap_overlaps_bug_height():
            return new_p, vx, vy

        pose = self.trap.get_pose()
        T = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
        R = T[:3, :3]
        origin = T[:3, 3]

        def to_local(pw):
            return R.T @ (np.asarray(pw, dtype=np.float64)[:3] - origin)

        cur_l = to_local(cur)
        new_l = to_local(new_p)
        cx, cy = float(cur_l[0]), float(cur_l[1])
        nx, ny = float(new_l[0]), float(new_l[1])

        wall = float(self.trap_wall)
        # Inflate walls slightly so the bug body cannot clip through thin glass.
        r = 0.008
        ox = float(self.trap_half[0])
        oy = float(self.trap_half[1])
        ix = max(ox - wall - r, 0.004)
        iy = max(oy - wall - r, 0.004)

        def in_cavity(x, y):
            return abs(x) < ix and abs(y) < iy

        def in_outer(x, y):
            return abs(x) < ox and abs(y) < oy

        def in_wall(x, y):
            return in_outer(x, y) and not in_cavity(x, y)

        if not in_wall(nx, ny):
            return new_p, vx, vy

        v_loc = R.T @ np.array([vx, vy, 0.0], dtype=np.float64)
        lvx, lvy = float(v_loc[0]), float(v_loc[1])

        if in_cavity(cx, cy):
            # Hit an inner wall from inside — bounce back into the cavity.
            if abs(nx) / max(ix, 1e-6) >= abs(ny) / max(iy, 1e-6):
                side = float(np.sign(nx if nx != 0 else (cx if cx != 0 else 1.0)))
                nx = side * (ix - 1e-4)
                lvx = -abs(lvx) * side
            else:
                side = float(np.sign(ny if ny != 0 else (cy if cy != 0 else 1.0)))
                ny = side * (iy - 1e-4)
                lvy = -abs(lvy) * side
        else:
            # Hit an outer wall from outside — stay outside; bounce along outward normal.
            if abs(nx) / max(ox, 1e-6) >= abs(ny) / max(oy, 1e-6):
                side = float(np.sign(nx if nx != 0 else (cx if cx != 0 else 1.0)))
                nx = side * ox
                lvx = abs(lvx) * side
            else:
                side = float(np.sign(ny if ny != 0 else (cy if cy != 0 else 1.0)))
                ny = side * oy
                lvy = abs(lvy) * side

        v_world = R @ np.array([lvx, lvy, 0.0], dtype=np.float64)
        vx, vy = float(v_world[0]), float(v_world[1])
        if abs(vx) > 1e-6:
            self.run_sign = int(np.sign(vx))

        world = R @ np.array([nx, ny, float(new_l[2])], dtype=np.float64) + origin
        new_p = np.array([float(world[0]), float(world[1]), float(new_p[2])],
                         dtype=np.float64)
        return new_p, vx, vy

    def _ee_pose(self, arm) -> sapien.Pose:
        p = self.get_arm_pose(ArmTag(str(arm)))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _trap_grasp_point(self) -> np.ndarray:
        """World position of the top-lid center (intended grasp)."""
        pose = self.trap.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        R = pose.to_transformation_matrix()[:3, :3]
        top = R @ np.array([0.0, 0.0, float(self.trap_half[2])], dtype=np.float64)
        return p + top

    def _arm_tcp_pos(self, side: str):
        """Gripper-center (TCP) world position; EE as fallback."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return None
        for name in (f"get_{side}_tcp_pose", f"get_{side}_ee_pose"):
            fn = getattr(robot, name, None)
            if not callable(fn):
                continue
            try:
                return np.asarray(fn()[:3], dtype=np.float64)
            except Exception:
                continue
        return None

    def _arm_gripper_val(self, side: str) -> float:
        robot = getattr(self, "robot", None)
        if robot is None:
            return 1.0
        fn = (
            robot.get_left_gripper_val
            if side == "left"
            else robot.get_right_gripper_val
        )
        try:
            return float(fn())
        except Exception:
            return 1.0

    def _gripper_contacts_trap(self, side: str | None = None) -> bool:
        """True when a gripper (optionally ``side``) is contacting the trap."""
        if self.trap is None:
            return False
        try:
            pts = self.get_gripper_actor_contact_position(self.trap.get_name())
        except Exception:
            return False
        if not pts:
            return False
        if side is None:
            return True
        tcp = self._arm_tcp_pos(side)
        if tcp is None:
            return True
        # Contact points near this arm's TCP count as that gripper.
        for pt in pts:
            if float(np.linalg.norm(np.asarray(pt[:3], dtype=np.float64) - tcp)) < 0.12:
                return True
        return False

    def _arm_can_latch_trap(self, side: str) -> tuple[bool, float, float]:
        """Whether ``side`` is closed enough around the trap's outer walls.

        Returns ``(ok, dist_to_top, gripper_val)``. Uses TCP (not EE): the EE
        frame sits ~12 cm behind the fingers, so EE→center never met the old
        10 cm weld threshold when the jaws were actually on the lid.

        The hollow box is ~9 cm wide — a correct outer-wall pinch sits near
        ``TRAP_HOLD_GRIPPER`` (~0.78), not a full close. Requiring gval<0.55
        forced the fingers through the empty cavity.
        """
        gval = self._arm_gripper_val(side)
        tcp = self._arm_tcp_pos(side)
        if tcp is None:
            return False, 1e9, gval
        top = self._trap_grasp_point()
        delta = tcp - top
        dist = float(np.linalg.norm(delta))
        dxy = float(np.linalg.norm(delta[:2]))
        dz = float(abs(delta[2]))
        contacted = self._gripper_contacts_trap(side)
        # Latch window: fingers over the lid (or PhysX contact) while closing
        # around the outer walls — not fully open, not forced fully closed.
        near = contacted or (dxy < 0.07 and dz < 0.09) or dist < 0.10
        latch_max = float(getattr(self, "trap_latch_gval_max", self.TRAP_LATCH_GVAL_MAX))
        ok = near and gval < latch_max
        return ok, dist, gval

    def _set_trap_pose(self, pose: sapien.Pose) -> None:
        """Drive the always-kinematic trap to ``pose`` (no dynamic unlock)."""
        if self.trap is None:
            return
        rigid = self._trap_rigid or self._get_rigid(self.trap)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
            except Exception:
                pass
            self._trap_rigid = rigid
        obj = self.trap.actor if hasattr(self.trap, "actor") else self.trap
        try:
            obj.set_pose(pose)
        except Exception:
            pass

    def _closest_arm_to_trap(self):
        """Nearest arm by TCP→top distance (legacy helper for weld fallback)."""
        if self.trap is None or getattr(self, "robot", None) is None:
            return None, 1e9, 1.0
        best = (None, 1e9, 1.0)
        for side in ("left", "right"):
            tcp = self._arm_tcp_pos(side)
            if tcp is None:
                continue
            d = float(np.linalg.norm(tcp - self._trap_grasp_point()))
            g = self._arm_gripper_val(side)
            if d < best[1]:
                best = (side, d, g)
        return best

    def _set_trap_physx_collisions_enabled(self, enabled: bool) -> None:
        """Enable/disable trap PhysX shapes.

        While welded the hollow box sits in the fingers and on the table; those
        contacts shove the dynamic arm so teleop feels stuck (often can't move
        sideways). Bug↔trap interaction stays geometric (both kinematic), so
        PhysX shapes can stay off while carried/falling and return on land.
        """
        if self.trap is None:
            return
        rigid = self._trap_rigid or self._get_rigid(self.trap)
        if rigid is None:
            return
        try:
            shapes = list(rigid.get_collision_shapes())
        except Exception:
            return
        if not shapes:
            return
        if not enabled:
            if self._trap_collision_groups_backup is None:
                backup = []
                for shape in shapes:
                    try:
                        backup.append(list(shape.get_collision_groups()))
                    except Exception:
                        backup.append([1, 1, 0, 0])
                self._trap_collision_groups_backup = backup
            for shape in shapes:
                try:
                    shape.set_collision_groups([0, 0, 0, 0])
                except Exception:
                    pass
            return
        backup = self._trap_collision_groups_backup
        for i, shape in enumerate(shapes):
            groups = backup[i] if backup and i < len(backup) else [1, 1, 0, 0]
            try:
                shape.set_collision_groups(list(groups))
            except Exception:
                pass
        self._trap_collision_groups_backup = None

    def weld_trap_to_gripper(self, arm=None) -> bool:
        """Attach the kinematic trap to an EE (explicit or nearest latchable)."""
        if self.trap is None or self._trap_anchored or self._trap_falling:
            return False
        if arm is None:
            # Prefer an arm that is actually closed and near the lid.
            for side in ("left", "right"):
                ok, _, _ = self._arm_can_latch_trap(side)
                if ok:
                    arm = side
                    break
            if arm is None:
                side, dist, gval = self._closest_arm_to_trap()
                latch_max = float(getattr(
                    self, "trap_latch_gval_max", self.TRAP_LATCH_GVAL_MAX))
                if side is None or dist > 0.12 or gval > latch_max:
                    return False
                arm = side
        arm = ArmTag(str(arm))
        self._trap_weld_arm = arm
        self._trap_weld_offset = self._ee_pose(arm).inv() * self.trap.get_pose()
        self._trap_welded = True
        # Drop PhysX contacts so the carried box cannot pin the arm against the
        # table / fingers while teleoping.
        self._set_trap_physx_collisions_enabled(False)
        self._sync_welded_trap()
        return True

    def _sync_welded_trap(self) -> None:
        if not self._trap_welded or self._trap_weld_offset is None or self._trap_weld_arm is None:
            return
        self._set_trap_pose(self._ee_pose(self._trap_weld_arm) * self._trap_weld_offset)

    def release_trap(self) -> None:
        """Gripper opened / trap left the hand → arm evaluation and start the drop."""
        if self.trap is None or self._trap_released:
            # Still allow a second call to begin falling if somehow stuck welded.
            if self._trap_welded:
                self._trap_welded = False
                self._trap_weld_offset = None
                self._trap_weld_arm = None
            return
        self._trap_welded = False
        self._trap_weld_offset = None
        self._trap_weld_arm = None
        self._trap_released = True
        if self._metric_release_step is None:
            self._metric_release_step = self._metric_step()
        if not self._trap_anchored:
            self._trap_falling = True

    def _freeze_trap_as_landed(self) -> None:
        """Freeze the trap at its current pose — do not reseat / reorient it."""
        if self.trap is None or self._trap_anchored:
            return
        pose = self.trap.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        # Only prevent going through the tabletop; keep XY and orientation.
        min_z = float(self.table_top + self.trap_half[2])
        if float(p[2]) < min_z:
            p[2] = min_z
            pose = sapien.Pose(p.tolist(), list(pose.q))
        self._set_trap_pose(pose)
        self._trap_anchor_pose = pose
        self._trap_anchored = True
        self._trap_falling = False
        self._trap_released = True
        self._latch_land_metric(p)
        # Restore PhysX shapes now that the box is seated (bug resolve is geometric).
        self._set_trap_physx_collisions_enabled(True)
        # Capture verdict at land time.
        if self.bug is not None and self._bug_rigid is not None:
            rp = np.asarray(self.bug.get_pose().p, dtype=np.float64)
            if (abs(rp[0] - p[0]) < self.trap_half[0]
                    and abs(rp[1] - p[1]) < self.trap_half[1]):
                self._bug_captured = True
                self._stop_bug()
        self._partial_score = self._score_at_landing()
        print(
            f"[trap_bug] trap landed: bug {self._bug_trap_edge_distance() * 100:.1f} cm "
            f"from footprint (ring {min(self.trap_half[0], self.trap_half[1]) * 100:.1f} cm) "
            f"score={self._partial_score:.2f}"
        )

    # Back-compat name used by interactive helper / older callers.
    def _anchor_trap(self) -> None:
        self._freeze_trap_as_landed()

    def _step_trap_fall(self) -> None:
        if not self._trap_falling or self.trap is None or self._trap_anchored:
            return
        dt = float(self.scene.get_timestep())
        pose = self.trap.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        # Kinematic drop (trap stays kinematic; no dynamic unlock).
        p[2] -= 1.8 * dt
        land_z = float(self.table_top + self.trap_half[2])
        if float(p[2]) <= land_z + 0.008:
            self._set_trap_pose(sapien.Pose(p.tolist(), list(pose.q)))
            self._freeze_trap_as_landed()
            return
        self._set_trap_pose(sapien.Pose(p.tolist(), list(pose.q)))

    def _maybe_auto_weld_or_release(self) -> None:
        """Policy-safe attach/detach: no unlock API; detect grasp / open / slip."""
        if self.trap is None or self._trap_anchored or self._trap_falling:
            return
        if self._trap_welded:
            # Sync first — EE motion between steps would otherwise look like a slip.
            self._sync_welded_trap()
            arm = str(self._trap_weld_arm) if self._trap_weld_arm is not None else None
            if arm is None:
                self.release_trap()
                return
            # Only the welded arm's gripper controls release (not the other hand).
            # Outer-wall hold sits near ~0.78; only release when nearly fully open.
            release_min = float(getattr(
                self, "trap_release_gval_min", self.TRAP_RELEASE_GVAL_MIN))
            if self._arm_gripper_val(arm) > release_min:
                self.release_trap()
            return
        # Prefer the interactively highlighted arm(s), then the other.
        sides = ["left", "right"]
        selected = tuple(getattr(self, "_interactive_selected_arms", ()) or ())
        if selected:
            sides = [s for s in selected if s in ("left", "right")]
            sides += [s for s in ("left", "right") if s not in sides]
        for side in sides:
            ok, _, _ = self._arm_can_latch_trap(side)
            if ok:
                if self.weld_trap_to_gripper(side):
                    print(f"[trap_bug] trap latched to {side} gripper")
                return

    def _stop_bug(self):
        """Halt scuttle once trapped — keeps PhysX trap collision intact."""
        self._bug_moving = False
        self._bug_vel = np.zeros(3, dtype=np.float64)
        if self.bug is None or self._bug_rigid is None:
            return
        pose = self.bug.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        p[2] = float(self.table_top + self._bug_half_h)
        try:
            self._bug_rigid.set_kinematic_target(
                sapien.Pose(p.tolist(), list(pose.q)))
        except Exception:
            pass

    def _park_bug_hidden(self) -> None:
        """Park under the shelf — appearance window over; episode fails."""
        self._bug_mode = "hidden"
        self._bug_moving = False
        self._bug_escaped = True
        if self._metric_escape_step is None:
            self._metric_escape_step = self._metric_step()
        self._bug_vel = np.zeros(3, dtype=np.float64)
        if self.bug is None or self._bug_rigid is None:
            return
        # Face back into the shelf (+Y) while tucked under the lip.
        self._bug_yaw = self._heading_yaw(0.0, 1.0)
        q = self._yaw_quat_flat(self._bug_yaw)
        hide_p = np.asarray(self._bug_start, dtype=np.float64).copy()
        hide_p[2] = float(self.table_top + self._bug_half_h)
        try:
            self._bug_rigid.set_kinematic_target(
                sapien.Pose(hide_p.tolist(), q.tolist()))
        except Exception:
            pass
        try:
            self._set_entity_pose(
                self.bug, sapien.Pose(hide_p.tolist(), q.tolist()))
        except Exception:
            pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._sim_steps += 1
        self._maybe_auto_weld_or_release()
        self._step_trap_fall()
        self._track_trap_metrics()
        # Keep a landed trap pinned at the exact landing pose.
        if self._trap_anchored and self._trap_anchor_pose is not None:
            self._set_trap_pose(self._trap_anchor_pose)
        elif self._trap_welded:
            self._sync_welded_trap()
        # Interactive / late cover: stop as soon as success holds.
        if (
            self._bug_moving
            and self._trap_anchored
            and not self._bug_captured
            and self.check_success()
        ):
            self._bug_captured = True
            self._stop_bug()
        if not self._bug_moving or self._bug_rigid is None:
            return

        dt = float(self.scene.get_timestep())
        self._bug_phase += dt
        pose = self._bug_rigid.entity.get_pose()
        p = np.array(pose.p, dtype=np.float64)

        # Bursty scuttle + light lateral wobble (realistic, never paused for the place).
        pulse = 0.55 + 0.45 * abs(np.sin(self._bug_phase * 10.0))
        speed = self._base_speed * pulse
        wobble = 0.015 * np.sin(self._bug_phase * 14.0)

        # Timed appearance: after walk_time, retreat under the shelf (fail).
        if (
            not self._bug_captured
            and self._bug_mode in ("emerge", "run")
        ):
            self._bug_walk_elapsed += dt
            if self._bug_walk_elapsed >= float(self.walk_time):
                self._bug_mode = "hide"

        xmin, xmax, ymin, ymax = self._table_bounds

        if self._bug_mode == "emerge":
            vy = -speed
            vx = wobble * 0.5 * self.run_sign
            if p[1] <= self._emerge_target_y:
                self._bug_mode = "run"
        elif self._bug_mode == "hide":
            # Steer home under the shelf (spawn is outside the run bounds).
            home = np.asarray(self._bug_start, dtype=np.float64)
            delta = home[:2] - p[:2]
            dist = float(np.linalg.norm(delta))
            if dist < 0.025:
                self._park_bug_hidden()
                return
            direction = delta / max(dist, 1e-6)
            vx = float(direction[0] * speed)
            vy = float(direction[1] * speed)
        else:
            vx = self.run_sign * speed
            vy = -0.15 * speed + wobble

        if self._bug_mode != "hide":
            if p[0] + vx * dt < xmin or p[0] + vx * dt > xmax:
                self.run_sign *= -1
                vx = self.run_sign * abs(vx)
            if p[1] + vy * dt < ymin:
                vy = abs(speed) * 0.25
            if p[1] + vy * dt > ymax:
                vy = -abs(speed) * 0.3

        new_p = p + np.array([vx, vy, 0.0], dtype=np.float64) * dt
        # Always collide with low trap walls (including while fleeing to hide).
        new_p, vx, vy = self._resolve_trap_collision(p, new_p, vx, vy)
        if self._bug_mode != "hide":
            new_p[0] = float(np.clip(new_p[0], xmin, xmax))
            new_p[1] = float(np.clip(new_p[1], ymin, ymax))
        else:
            # Allow crossing the shelf lip to reach the hide pose.
            home = np.asarray(self._bug_start, dtype=np.float64)
            new_p[0] = float(np.clip(new_p[0], min(xmin, home[0]) - 0.02,
                                     max(xmax, home[0]) + 0.02))
            new_p[1] = float(np.clip(new_p[1], ymin, max(ymax, home[1]) + 0.02))
        new_p[2] = float(self.table_top + self._bug_half_h)

        self._bug_vel = np.array([vx, vy, 0.0], dtype=np.float64)
        # Slew the head toward the travel direction (never leave the body
        # sideways relative to velocity).
        target_yaw = self._heading_yaw(vx, vy)
        dyaw = (target_yaw - self._bug_yaw + np.pi) % (2.0 * np.pi) - np.pi
        max_turn = float(self.TURN_RATE) * dt
        self._bug_yaw = float(self._bug_yaw + float(np.clip(dyaw, -max_turn, max_turn)))
        q = self._yaw_quat_flat(self._bug_yaw)
        self._bug_rigid.set_kinematic_target(
            sapien.Pose(p=new_p.tolist(), q=q.tolist()))

    def check_stable(self):
        """Keep the kinematic bug parked under the shelf while the scene settles."""
        if self.bug is not None and not self._bug_moving:
            self._bug_yaw = self._heading_yaw(0.0, -1.0)
            q = self._yaw_quat_flat(self._bug_yaw)
            self._set_entity_pose(
                self.bug, sapien.Pose(self._bug_start.tolist(), q.tolist()))
        return super().check_stable()

    def _dwell(self, steps):
        # Counter is episode-global so single-step waits do not capture a frame
        # per physics tick.
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._sim_steps % self.save_freq == 0):
                self._take_picture()

    def _start_bug(self):
        if getattr(self, "_bug_moving", False):
            return
        self._bug_moving = True
        self._bug_walk_elapsed = 0.0
        self._bug_escaped = False
        if self._metric_bug_start_step is None:
            self._metric_bug_start_step = self._metric_step()
        if self._bug_mode in ("hidden", "hide"):
            self._bug_mode = "emerge"

    # ------------------------------------------------------------- policy
    def _intercept_point(self, arm_tag) -> np.ndarray:
        """A spot on the bug's run line where the trap waits for it.

        The lane (y) is where the bug will have drifted to by the time it
        reaches the intercept, not where it is now — it sidles forward the
        whole way there.
        """
        rp = np.array(self.bug.get_pose().p, dtype=np.float64)
        xmin, xmax, ymin, ymax = self._table_bounds
        speed = max(self._base_speed * 0.836, 1e-3)
        ahead = rp[0] + self.run_sign * self.INTERCEPT_AHEAD
        if not (xmin + 0.04 <= ahead <= xmax - 0.04):
            # Too close to a wall to get there first; the bug bounces back.
            ahead = rp[0] - self.run_sign * self.INTERCEPT_AHEAD
        x = float(np.clip(ahead, xmin + 0.04, xmax - 0.04))
        t_cross = abs(x - rp[0]) / speed + self._release_delay(arm_tag)
        y = float(np.clip(rp[1] - 0.15 * speed * t_cross,
                          ymin + 0.03, min(ymax, 0.05)))
        return np.array([x, y], dtype=np.float64)

    def _fall_time(self) -> float:
        return float(np.sqrt(max(2.0 * self.release_clearance / 9.81, 0.0)))

    def _release_delay(self, arm_tag) -> float:
        """Seconds from commanding the gripper open until the trap lands.

        The gripper open is a dense action several hundred sim steps long and
        the box only drops partway through it, so the bug covers real ground
        between the command and the catch.
        """
        steps = float(self.RELEASE_LAG_STEPS)
        try:
            plan_fn = (self.robot.left_plan_grippers if arm_tag == "left"
                       else self.robot.right_plan_grippers)
            cur = (self.robot.get_left_gripper_val() if arm_tag == "left"
                   else self.robot.get_right_gripper_val())
            n = float(plan_fn(cur, 1.0)["num_step"]) * 1.5  # set_gripper pads by 50%
            steps = n * self.RELEASE_FRACTION
        except Exception:
            pass
        return steps * float(self.scene.get_timestep()) + self._fall_time()

    def _predict_arrival(self, t: float) -> np.ndarray:
        """Where the bug will be in ``t`` seconds, reflecting off the bounds.

        Uses the mean scuttle speed rather than the instantaneous burst so a
        long lead is not thrown off by the gait pulse.
        """
        rp = np.array(self.bug.get_pose().p, dtype=np.float64)
        xmin, xmax, ymin, ymax = self._table_bounds
        speed = self._base_speed * 0.836  # mean of the 0.55..1.0 pulse
        x = rp[0] + self.run_sign * speed * t
        span = xmax - xmin
        if span > 1e-6:
            u = (x - xmin) % (2.0 * span)
            x = xmin + (u if u <= span else 2.0 * span - u)
        y = float(np.clip(rp[1] - 0.15 * speed * t, ymin, ymax))
        return np.array([x, y], dtype=np.float64)

    def _wait_for_bug_under(self, arm_tag, x_tol=0.012, y_tol=0.024) -> bool:
        """Hold the trap still and let the bug scuttle under it.

        The bug is never paused; we step the sim until the spot it will
        occupy once the released trap lands is centred under the trap. Gives
        up early once the bug has sidled out of the lane, so the caller can
        re-aim instead of burning the whole window.
        """
        for _ in range(int(self.WAIT_STEPS)):
            if self._bug_escaped or self._bug_mode in ("hide", "hidden"):
                return False
            tp = np.array(self.trap.get_pose().p, dtype=np.float64)
            arrival = self._predict_arrival(self._release_delay(arm_tag))
            if abs(arrival[1] - tp[1]) >= y_tol:
                return False
            if abs(arrival[0] - tp[0]) < x_tol:
                return True
            self._dwell(1)
        return False

    def _reaim_lane(self, arm_tag) -> None:
        """Nudge the held trap back onto the bug's lane (y drifts slowly)."""
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        target = self._intercept_point(arm_tag)
        dy = float(np.clip(target[1] - tp[1], -0.08, 0.08))
        dx = float(np.clip(target[0] - tp[0], -0.08, 0.08))
        if abs(dy) < 0.004 and abs(dx) < 0.02:
            return
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, x=dx, y=dy, move_axis="world"))

    def play_once(self):
        # Start scurrying immediately — never pause for the place.
        self._start_bug()
        self._dwell(self.emerge_steps)

        arm_tag = ArmTag(self.arm_side)

        # Trap stays kinematic always — carry via EE weld (no unlock/lock).
        # Pinch the outer walls (hold width), do not crush through the cavity.
        hold = float(getattr(self, "trap_hold_gripper", self.TRAP_HOLD_GRIPPER))
        self.move(self.grasp_actor(
            self.trap, arm_tag=arm_tag, pre_grasp_dis=0.08,
            gripper_pos=hold, contact_point_id=[0, 1, 2, 3]))
        self.weld_trap_to_gripper(arm_tag)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))

        # Carry the trap high over an intercept spot on the bug's run line.
        # Displacements are measured on the trap itself, so the grasp offset
        # does not bias the alignment.
        target = self._intercept_point(arm_tag)
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        hover_z = self.table_top + self.trap_half[2] + self.release_clearance + self.HOVER_LIFT
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(target[0] - tp[0]),
            y=float(target[1] - tp[1]),
            z=float(hover_z - tp[2]),
            move_axis="world",
        ))

        # Drop from a raised hover: high enough for the fingers to clear the rim.
        drop_z = self.table_top + self.trap_half[2] + self.release_clearance
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=float(drop_z - tp[2]), move_axis="world"))

        # Poised over the lane: wait for the bug, re-aiming if it drifts.
        for _ in range(8):
            if self._wait_for_bug_under(arm_tag):
                break
            self._reaim_lane(arm_tag)

        self.move(self.open_gripper(arm_tag=arm_tag))
        # Gripper open / slip arms evaluation; kinematic fall freezes the landing pose as-is.
        self.release_trap()
        for _ in range(80):
            if self._trap_anchored:
                break
            self._dwell(1)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="world"))

        # Hold on success (bug stopped under the trap) or a miss.
        self._dwell(60)

        self.info["info"] = {
            "{A}": f"{self.bug_model}/base0",
            "{S}": self.bug_type,
            "{B}": "glass_trap",
            "{C}": "121_wall-shelf",
            "{a}": str(arm_tag),
            "{d}": "left" if self.run_sign < 0 else "right",
        }
        return self.info

    # ------------------------------------------------------------- success
    def _bug_under_trap(self) -> bool:
        if self.trap is None or self.bug is None:
            return False
        trap_p = np.array(self.trap.get_pose().p, dtype=np.float64)
        bug_p = np.array(self.bug.get_pose().p, dtype=np.float64)
        return bool(
            abs(bug_p[0] - trap_p[0]) < self.trap_half[0] * 0.95
            and abs(bug_p[1] - trap_p[1]) < self.trap_half[1] * 0.95
        )

    def _bug_trap_edge_distance(self) -> float:
        """Planar distance from bug to the trap footprint (0 if under / inside)."""
        if self.trap is None or self.bug is None:
            return float("inf")
        trap_p = np.array(self.trap.get_pose().p, dtype=np.float64)
        bug_p = np.array(self.bug.get_pose().p, dtype=np.float64)
        hx = float(self.trap_half[0]) * 0.95
        hy = float(self.trap_half[1]) * 0.95
        dx = abs(float(bug_p[0] - trap_p[0])) - hx
        dy = abs(float(bug_p[1] - trap_p[1])) - hy
        return float(np.hypot(max(0.0, dx), max(0.0, dy)))
    # --------------------------------------------------- experiment metrics
    def _reset_metric_state(self) -> None:
        """Per-episode metric latches. Called at every reset site."""
        self._metric_bug_start_step = None
        self._metric_release_step = None
        self._metric_land_step = None
        self._metric_escape_step = None
        self._metric_land_offset = None
        self._metric_land_covered = None
        self._metric_best_offset = None

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _trap_offset_norm(self, trap_p) -> float:
        """Bug centre vs trap footprint, normalised by the success half-extents.

        <= 1.0 means the bug is under the trap by the same test ``check_success``
        uses (``trap_half * 0.95``), so the number is directly readable as
        "how close to a clean capture".
        """
        bp = np.asarray(self.bug.get_pose().p, dtype=np.float64)
        tp = np.asarray(trap_p, dtype=np.float64)
        hx = float(self.trap_half[0]) * 0.95
        hy = float(self.trap_half[1]) * 0.95
        return float(max(abs(bp[0] - tp[0]) / max(hx, 1e-6),
                         abs(bp[1] - tp[1]) / max(hy, 1e-6)))

    def _latch_land_metric(self, trap_p) -> None:
        """The trap has come to rest — the verdict is fixed at this instant.

        Read the offset here, not at the end: a captured bug is stopped and a
        missed one keeps running, so a late recompute would flatter the capture
        and punish the miss for motion that happened after the outcome.
        """
        try:
            if self._metric_land_step is not None:
                return
            self._metric_land_step = self._metric_step()
            if self.bug is None or self._bug_rigid is None:
                return
            off = self._trap_offset_norm(trap_p)
            self._metric_land_offset = off
            self._metric_land_covered = bool(off <= 1.0)
        except Exception:
            pass

    def _track_trap_metrics(self) -> None:
        """Closest the bug ever came to being under the (still airborne) trap.

        A high-water mark of the *best* alignment while the trap was still in
        play: it separates "aimed badly" from "aimed right but dropped late".
        """
        try:
            if self._metric_land_step is not None:
                return
            if self.trap is None or self.bug is None or self._bug_rigid is None:
                return
            if not self._bug_moving:
                return
            off = self._trap_offset_norm(self.trap.get_pose().p)
            if (self._metric_best_offset is None
                    or off < self._metric_best_offset):
                self._metric_best_offset = off
        except Exception:
            pass

    def _compute_metrics(self) -> dict:
        """Human-experiment metrics.

        extra1 ``cover_latency_steps`` — bug starts running -> trap lands on the
        table. That is the decisive event: once the trap is anchored the verdict
        is already decided, so this is the whole reaction window.

        extra2 ``bug_offset_norm`` — bug centre vs trap footprint at landing,
        normalised by ``trap_half * 0.95`` (the success half-extents).
        Dimensionless, LOWER IS BETTER; <= 1.0 == covered, 0.0 == dead centre.
        ``None`` when the trap never landed.
        """
        out = {}
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = 0.004
        start = self._metric_bug_start_step
        land = self._metric_land_step
        rel = self._metric_release_step

        lat = None if (land is None or start is None) else int(land - start)
        out["cover_latency_steps"] = lat
        out["cover_latency_s"] = None if lat is None else round(lat * dt, 4)
        out["release_latency_steps"] = (
            None if (rel is None or start is None) else int(rel - start))
        out["fall_latency_steps"] = (
            None if (land is None or rel is None) else int(land - rel))
        esc = self._metric_escape_step
        out["escape_margin_steps"] = (
            None if (esc is None or land is None) else int(esc - land))

        off = self._metric_land_offset
        out["bug_offset_norm"] = None if off is None else round(float(off), 4)
        best = self._metric_best_offset
        out["best_offset_norm"] = None if best is None else round(float(best), 4)
        out["covered_at_landing"] = self._metric_land_covered
        out["bug_escaped"] = bool(getattr(self, "_bug_escaped", False))
        out["trap_landed"] = bool(getattr(self, "_trap_anchored", False))
        try:
            out["walk_time_s"] = round(float(self.walk_time), 4)
        except Exception:
            out["walk_time_s"] = None
        return out

    def check_success(self):
        """Success = released trap has landed, with the bug under its footprint.

        Evaluation arms when the gripper opens or the trap leaves the hand.
        Hovering while still held must not count. Landing pose is left as-is
        (no reseat / upright teleport).
        """
        if self.trap is None or self.bug is None:
            return False
        if bool(getattr(self, "_bug_escaped", False)):
            return False
        if not bool(getattr(self, "_trap_released", False)):
            return False
        if not bool(getattr(self, "_trap_anchored", False)):
            return False
        return bool(self._bug_under_trap())

    def _score_at_landing(self) -> float:
        """Score the bug's proximity to the trap at the moment the trap lands.

        Full success (bug under the footprint) → 1.0. Otherwise a ring one trap
        half-width wide extends out from the footprint edge, split into equal
        thirds: nearest → 0.75, middle → 0.5, outer → 0.25; beyond the ring → 0.
        """
        if self.trap is None or self.bug is None:
            return 0.0
        if bool(getattr(self, "_bug_escaped", False)):
            return 0.0
        if self._bug_under_trap():
            return 1.0
        r = float(min(self.trap_half[0], self.trap_half[1]))
        if r <= 1e-9:
            return 0.0
        d = self._bug_trap_edge_distance()
        third = r / 3.0
        if d > r + 1e-9:
            return 0.0
        if d < third:
            return 0.75
        if d < 2.0 * third:
            return 0.5
        return 0.25

    def get_score(self) -> float:
        """Latched partial score from the trap's landing frame.

        The bug keeps scuttling (and retreats under the shelf when its walk
        window expires), so a live distance read at episode end would almost
        always fall outside the ring. The score is therefore frozen at the
        landing frame — the same instant ``check_success`` evaluation arms.
        """
        if self.check_success():
            return 1.0
        if not bool(getattr(self, "_trap_anchored", False)):
            return 0.0
        return float(getattr(self, "_partial_score", 0.0))

    def get_obs(self):
        obs = super().get_obs()
        bug_p = self.bug.get_pose().p if self.bug is not None else [0, 0, 0]
        trap_p = self.trap.get_pose().p if self.trap is not None else [0, 0, 0]
        obs["trap_bug"] = {
            "bug_type": str(self.bug_type),
            "bug_model": str(self.bug_model),
            "bug_pos": [float(x) for x in bug_p],
            "trap_pos": [float(x) for x in trap_p],
            "bug_vel": [float(x) for x in self._bug_vel],
            "bug_yaw": float(self._bug_yaw),
            "run_dir": "left" if self.run_sign < 0 else "right",
            "bug_moving": bool(self._bug_moving),
            "bug_mode": str(self._bug_mode),
            "walk_time": float(self.walk_time),
            "walk_elapsed": float(self._bug_walk_elapsed),
            "bug_escaped": bool(self._bug_escaped),
            "partial_score": float(self.get_score()),
        }
        return obs
