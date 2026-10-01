from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
import os
import sapien
import sapien.render
import sapien.physx
import numpy as np
import transforms3d as t3d


class pick_ripe_apple(Base_Task):
    """Pick a ripening apple from a tree into a basket.

    A decorative tree sits at the table center (x=0, toward the back). By default a single
    **good** apple hangs (green → red → black). Opt 1 adds a second **spoiled** apple
    (green → yellow → black) that must not be picked. The arm matching the good side waits
    for red, pinches that apple (detaches from the tree, freezes color, held by gripper friction),
    clears laterally, and drops it into a breadbasket. Opt 2 makes the basket oscillate
    left↔right under the branches (default: static).

    Metric: ripeness_score = clamp(1 - |r_grasp - red_window| / 0.5, 0, 1), latched at detach of
    the good apple; episode succeeds only if the good apple was picked inside the ripeness
    window (|r_grasp - red_window| <= red_tolerance), it ends up in the basket, and the
    spoiled apple (if any) does not. Dropping the picked good apple on the table (missed
    basket) is a failure and ends the episode.

    ========================================================================
    FROZEN SETUP STRATEGY — DO NOT CHANGE (user-locked)
    ========================================================================
    Tree:     tree_x=0, tree_y≈0.05 (toward back wall); tall trunk; staggered
              high/low branches; visual foliage only; trunk+branch collision.
    Stem:     black thin stem under the fruit branch; same hang_x as the apple.
    Apple:    APPLE_HANG_Q = identity [1,0,0,0] (visual +Z top into the stem);
              APPLE_DROP_BELOW_BRANCH so ~½ black stem shows; Y offset so mesh
              bbox center sits on branch/stem Y (APPLE_MESH_CENTER_Y * frac).
    Hang X:   outer bound = branch tip; sample Uniform(0, APPLE_X_JITTER=2cm)
              inward toward trunk/bark for BOTH stem and apple together:
                hang_x = tip - apple_side * U(0, jitter)
    Basket:   076_breadbasket in front; BASKET_Q = packing upright then +90° Z;
              sample model_id ∈ {0,1,2,3,4}; hollow (nonconvex) collision
              (base3/base4 use rebuilt open-cup collision); original appearance.
    Do NOT revisit Rx±90° hang quats, tip-only X, or dual boards/bowls.
    ========================================================================
    FROZEN CONTROL / GRASPING — DO NOT CHANGE (user-locked)
    ========================================================================
    Grasp:    front approach via _try_front_grasp; GRASP_DIRECTION_DIC front /
              arm-specific fallbacks (_front_grasp_quat_keys). Horizontal pinch
              from table front (−Y), not top-down.
    Quats:    right prefers front_left (+ TCP err reject GRASP_TCP_ERR_MAX);
              left uses front (then front_right, front_left).
    Hold:     detach from stem on grasp; physical friction / contact hold only
              (no EE weld or attach). Gravity on after jaw settle.
    Clear:    lateral ±X by apple_side (CLEAR_LATERAL), then Z lift
              (CLEAR_LIFT_Z) before transport — not straight up through branch.
    Drop:     open gripper over basket; natural gravity fall (no place/weld).
    Do NOT revert to weld/attach or straight-up-through-branch lift.
    ========================================================================
    """

    # ---- task params (class defaults; override via task_args.pick_ripe_apple) ----
    RIPEN_STEPS_DEFAULT = 3600        # +20% vs 3000 → another 20% slower ripen
    # Mean target ripeness ~0.8; ±0.02 band.
    RED_WINDOW_MIN_DEFAULT = 0.78
    RED_WINDOW_MAX_DEFAULT = 0.82
    RED_TOLERANCE_DEFAULT = 0.12
    GRASP_LEAD_STEPS = 1008           # scaled with ripen_steps (was 840 @ 3000)
    # Table y spans ~[-0.35, +0.35]; the back wall is at y=1. Default tree sits near the back edge.
    TREE_X_DEFAULT = 0.0
    TREE_Y_DEFAULT = 0.05             # toward back wall (wall at y=1; table ~±0.35)
    BASKET_Y_DEFAULT = -0.20          # in front of the tree — clearance so the grasp misses the rim
    BASKET_X_DEFAULT = 0.0
    BASKET_SCALE_DEFAULT = 1.0
    # Opt 1 — second spoiled apple (yellow→black); default OFF (single good apple).
    TWO_APPLES_ENABLED_DEFAULT = False
    # Opt 2 — oscillating basket between branch-tip X bounds (default OFF / static).
    BASKET_MOVE_ENABLED_DEFAULT = False
    BASKET_SPEED_MIN_DEFAULT = 0.08   # m/s; per-ep Uniform[min, max]
    BASKET_SPEED_MAX_DEFAULT = 0.16
    DROP_PREDICT_TIME = 1.2           # s; expert aims ahead of moving basket
    DROP_ALIGN_TOL = 0.035            # m; wait until basket under held apple before open
    DROP_ALIGN_MAX_STEPS = 1000
    APPLE_HANG_DX_DEFAULT = 0.18      # |x| of hanging apple at branch tip (outer bound)
    APPLE_SCALE_DEFAULT = 0.78
    # ~real medium apple; mesh@density=1000 is ~0.21 kg — keep mass near that range.
    APPLE_MASS = 0.30
    # Branch heights (m above table). One side is high, the other low (randomized).
    BRANCH_Z_HIGH = 0.25              # was 0.15; +10 cm
    BRANCH_Z_LOW = 0.20               # was 0.10; +10 cm
    TRUNK_HEIGHT = 0.50               # was 0.30; +20 cm
    # --- FROZEN hang geometry (see class docstring) — do not retune without user ask ---
    # Visual top (texture pole / local +Z) faces up under the branch stem (identity quat).
    # Mesh origin is the local-Y tip; bbox center is ~0.0374·scale along +Y — offset hang
    # so the sphere center (not the tip) sits on the branch/stem Y.
    # Partial Y offset: full model_data center overshoots; ~half centers the branch on the fruit.
    APPLE_MESH_CENTER_Y = 0.0374355
    APPLE_Y_CENTER_FRAC = 0.45
    APPLE_DROP_BELOW_BRANCH = 0.042   # lower so ~½ of the black stem shows below the branch
    STEM_HALF_THICK = 0.002           # was 0.004; diameter halved → 4 mm
    # Hang X ∈ [branch_tip toward bark by APPLE_X_JITTER, branch_tip]; same for stem + apple.
    APPLE_X_JITTER = 0.02            # m; max inward offset from branch tip toward trunk
    APPLE_HANG_Q = [1.0, 0.0, 0.0, 0.0]
    # Packing upright [0.5,0.5,0.5,0.5] then +90° about world Z.
    BASKET_Q = [0.0, 0.0, 0.70710678, 0.70710678]
    BASKET_MODEL = "076_breadbasket"
    BASKET_IDS = [0, 1, 2, 3, 4]
    COLOR_STOPS = [
        (0.0, [0.20, 0.62, 0.18]),     # unripe: green
        (0.8, [0.92, 0.10, 0.08]),     # ripe: vivid red (aligned with red_window mean)
        (1.0, [0.05, 0.04, 0.03]),     # overripe: near-black
    ]
    # Spoiled / distractor apple — never enters a red pick window as "good".
    SPOILED_COLOR_STOPS = [
        (0.0, [0.20, 0.62, 0.18]),     # unripe: green
        (0.8, [0.92, 0.78, 0.08]),     # yellowing
        (1.0, [0.05, 0.04, 0.03]),     # rotten: near-black
    ]
    TRUNK_COLOR = [0.45, 0.28, 0.12]
    LEAF_COLOR = [0.22, 0.58, 0.20]
    BRANCH_COLOR = [0.40, 0.24, 0.10]
    STEM_COLOR = [0.0, 0.0, 0.0]      # black
    # Cylinder local axis is +X; this quat maps it onto world +Z (vertical trunk).
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("pick_ripe_apple", {})
        self._reset_metric_state()
        super()._init_task_env_(**kwags)
        # Start ripening only AFTER setup settle (~2000 steps in _init_task_env_).
        self._ripen_started = True

    def _parse_side_cfg(self, side_cfg):
        s = str(side_cfg).lower()
        if s in ("left", "l", "-1"):
            return -1.0
        if s in ("right", "r", "+1", "1"):
            return 1.0
        return float(np.random.choice([-1.0, 1.0]))

    def _apply_apple_mass_properties(self, rigid, mass=None):
        """Set mass *and* matching inertia/COM.

        ``Actor`` / ``set_mass`` alone leave the default ~10 g inertia tensor, so a
        "heavy" apple still spins and skates like a light one. Match play_billiard /
        save_goal: solid-sphere I = 2/5 m R^2 about the collision hull centre.
        """
        if rigid is None:
            return
        m = float(self.APPLE_MASS if mass is None else mass)
        try:
            shapes = list(rigid.get_collision_shapes())
            verts = np.asarray(shapes[0].get_vertices(), dtype=np.float64)
            center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
            rms_r = float(np.sqrt(np.mean(np.sum((verts - center) ** 2, axis=1))))
            inertia = 0.4 * m * (rms_r ** 2)
            rigid.set_mass(m)
            rigid.set_cmass_local_pose(sapien.Pose(center.tolist()))
            rigid.set_inertia([inertia, inertia, inertia])
        except Exception:
            try:
                rigid.set_mass(m)
            except Exception:
                pass

    def _configure_hanging_apple(self, apple):
        """Apply frozen hang physics: kinematic, no gravity, high jaw friction."""
        rigid = next(
            (c for c in apple.actor.get_components()
             if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)), None)
        self._apply_apple_mass_properties(rigid)
        if rigid is not None:
            try:
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
                rigid.set_linear_damping(5.0)
                rigid.set_angular_damping(20.0)
                for s in rigid.get_collision_shapes():
                    m = s.get_physical_material()
                    # High friction so a closed parallel-jaw pinch can carry the apple.
                    m.set_static_friction(6.0)
                    m.set_dynamic_friction(6.0)
                    m.set_restitution(0.0)
            except Exception:
                pass
        shapes = []
        for c in apple.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                shapes = list(c.render_shapes)
        return rigid, shapes

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.ripen_steps = int(cfg.get("ripen_steps", self.RIPEN_STEPS_DEFAULT))
        self.red_window = float(np.random.uniform(
            cfg.get("red_window_min", self.RED_WINDOW_MIN_DEFAULT),
            cfg.get("red_window_max", self.RED_WINDOW_MAX_DEFAULT)))
        self.red_tol = float(cfg.get("red_tolerance", self.RED_TOLERANCE_DEFAULT))

        self.tree_x = float(cfg.get("tree_x", self.TREE_X_DEFAULT))
        self.tree_y = float(cfg.get("tree_y", self.TREE_Y_DEFAULT))
        self.basket_x = float(cfg.get("basket_x", self.BASKET_X_DEFAULT))
        self.basket_y = float(cfg.get("basket_y", self.BASKET_Y_DEFAULT))
        self.basket_scale = float(cfg.get("basket_scale", self.BASKET_SCALE_DEFAULT))
        hang_dx = float(cfg.get("apple_hang_dx", self.APPLE_HANG_DX_DEFAULT))
        self.hang_dx = hang_dx
        apple_sm = float(os.environ.get(
            "PRA_APPLE_SCALE", str(cfg.get("apple_scale_mult", self.APPLE_SCALE_DEFAULT))))

        # good_side selects the red-path apple; apple_side kept as alias for configs/CLI.
        side_cfg = cfg.get("good_side", cfg.get("apple_side", "random"))
        self.apple_side = self._parse_side_cfg(side_cfg)   # good apple side (−1 left, +1 right)
        self.good_side = self.apple_side
        self.spoiled_side = -self.apple_side

        # Opt 1: second spoiled apple (aliases: two_apples_enabled / spoiled_apple_enabled).
        two_raw = cfg.get(
            "two_apples_enabled",
            cfg.get("spoiled_apple_enabled", self.TWO_APPLES_ENABLED_DEFAULT),
        )
        self.two_apples_enabled = bool(two_raw)
        self.spoiled_apple_enabled = self.two_apples_enabled

        # Opt 2: basket ping-pongs between branch-tip X extents.
        self.basket_move_enabled = bool(cfg.get(
            "basket_move_enabled", self.BASKET_MOVE_ENABLED_DEFAULT))
        speed_lo = float(cfg.get("basket_speed_min", self.BASKET_SPEED_MIN_DEFAULT))
        speed_hi = float(cfg.get("basket_speed_max", self.BASKET_SPEED_MAX_DEFAULT))
        if speed_hi < speed_lo:
            speed_lo, speed_hi = speed_hi, speed_lo
        self.basket_speed = float(np.random.uniform(speed_lo, speed_hi))
        self._basket_move_dir = float(np.random.choice([-1.0, 1.0]))
        # Bounds = branch tips (same X extents used for hanging apples).
        self.basket_x_min = float(self.tree_x - hang_dx)
        self.basket_x_max = float(self.tree_x + hang_dx)

        z0 = 0.74 + self.table_z_bias
        self._z0 = z0

        # Staggered branch heights: randomly pick which side is the higher branch.
        z_high = float(cfg.get("branch_z_high", self.BRANCH_Z_HIGH))
        z_low = float(cfg.get("branch_z_low", self.BRANCH_Z_LOW))
        self.high_branch_side = self._parse_side_cfg(
            cfg.get("high_branch_side", "random"))
        self.branch_z = {
            -1.0: z_high if self.high_branch_side < 0 else z_low,
            +1.0: z_high if self.high_branch_side > 0 else z_low,
        }
        apple_drop = float(cfg.get("apple_drop_below_branch", self.APPLE_DROP_BELOW_BRANCH))
        trunk_h = float(cfg.get("trunk_height", self.TRUNK_HEIGHT))
        x_jit = float(cfg.get("apple_x_jitter", self.APPLE_X_JITTER))
        # Identity mesh extends along +Y from the pose origin — pull hang_y forward so
        # the bbox center lands on the branch/stem axis (tree_y).
        apple_scale = 0.8748 * apple_sm
        hang_y = self.tree_y - (
            self.APPLE_Y_CENTER_FRAC * self.APPLE_MESH_CENTER_Y * apple_scale)

        # Hang sides: always good; spoiled only when Opt 1 is on.
        hang_sides = [self.apple_side]
        if self.two_apples_enabled:
            hang_sides.append(self.spoiled_side)

        # Frozen hang X per spawned side: tip − side * U(0, jitter).
        self.hang_x_by_side = {}
        self._hang_poses = {}
        hang_x_locals = {}
        for side in hang_sides:
            tip = self.tree_x + side * hang_dx
            hx = float(tip - side * np.random.uniform(0.0, x_jit))
            self.hang_x_by_side[side] = hx
            hang_x_locals[side] = hx - self.tree_x
            self._hang_poses[side] = sapien.Pose(
                [hx, hang_y, z0 + self.branch_z[side] - apple_drop],
                list(self.APPLE_HANG_Q),
            )
        # Compat aliases for frozen grasp helpers (good apple only).
        self.hang_x = self.hang_x_by_side[self.apple_side]
        self._hang_pose = self._hang_poses[self.apple_side]

        # ---- tree: stems only under spawned apples (frozen hang coupling) ----
        self.tree = self._build_tree(
            sapien.Pose([self.tree_x, self.tree_y, z0], [1, 0, 0, 0]),
            trunk_h=trunk_h,
            branch_z_left=self.branch_z[-1.0],
            branch_z_right=self.branch_z[+1.0],
            hang_dx=hang_dx,
            apple_drop=apple_drop,
            hang_x_locals=hang_x_locals,
        )

        # ---- hanging apple(s) (kinematic while attached) ----
        self.apples = {}
        self._apple_rigids = {}
        self._apple_shapes_by_side = {}
        for side in hang_sides:
            tag = "left" if side < 0 else "right"
            apple = create_actor(
                self, pose=self._hang_poses[side],
                modelname="220_apple_plain", model_id=0, convex=True,
                is_static=False, scale_mult=apple_sm,
            )
            apple.set_name(f"220_apple_plain_{tag}")
            rigid, shapes = self._configure_hanging_apple(apple)
            self.apples[side] = apple
            self._apple_rigids[side] = rigid
            self._apple_shapes_by_side[side] = shapes

        # Frozen grasp / detach / success aliases → good apple only.
        self.apple = self.apples[self.apple_side]
        self._apple_rigid = self._apple_rigids[self.apple_side]
        self._apple_shapes = self._apple_shapes_by_side[self.apple_side]
        if self.two_apples_enabled:
            self.spoiled_apple = self.apples[self.spoiled_side]
            self._spoiled_rigid = self._apple_rigids[self.spoiled_side]
            self._spoiled_shapes = self._apple_shapes_by_side[self.spoiled_side]
        else:
            self.spoiled_apple = None
            self._spoiled_rigid = None
            self._spoiled_shapes = []

        # ---- receptacle: 076_breadbasket, random visual id ----
        # Start at center; Opt 2 oscillates between branch-tip X bounds via set_pose
        # when static, or kinematic target when dynamic.
        self.basket_id = int(np.random.choice(self.BASKET_IDS))
        self.basket = create_actor(
            self,
            pose=sapien.Pose(
                [self.basket_x, self.basket_y, z0],
                list(self.BASKET_Q),
            ),
            modelname=self.BASKET_MODEL,
            model_id=self.basket_id,
            # Triangle mesh: a convex hull caps the opening so the apple
            # hits an invisible lid and bounces out.
            convex=False,
            is_static=True,
            scale_mult=self.basket_scale,
        )
        self._basket_rigid = next(
            (c for c in self.basket.actor.get_components()
             if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)), None)
        if self.basket_move_enabled and self._basket_rigid is not None:
            try:
                self._basket_rigid.set_kinematic(True)
                self._basket_rigid.set_disable_gravity(True)
            except Exception:
                pass
        self.basket_color = None
        self.basket_texture = None
        self.basket_base_z = float(self.basket.get_pose().p[2])
        bcfg = getattr(self.basket, "config", None) or {}
        extents = bcfg.get("extents", [0.0, 0.7, 0.0])
        scale = bcfg.get("scale", [self.basket_scale] * 3)
        basket_height = float(extents[1]) * float(scale[1])
        if basket_height <= 0.0:
            basket_height = 0.07 * self.basket_scale
        self.basket_top_z = self.basket_base_z + basket_height
        self.basket_center = np.array(
            [self.basket_x, self.basket_y], dtype=np.float64)
        self.basket_half_xy = self._basket_mouth_half_xy(extents, scale)

        for apple in self.apples.values():
            self.add_prohibit_area(apple, padding=0.03)
        self.add_prohibit_area(self.basket, padding=0.05)

        # ripeness / attach state (good = red path; spoiled = yellow path if Opt 1)
        self.ripeness = 0.0
        self.spoiled_ripeness = 0.0
        self._ripen_started = False
        self._reset_metric_state()
        self._apple_attached = True          # good apple still hanging
        self._spoiled_attached = bool(self.two_apples_enabled)
        self.r_grasp = None                  # latched at good-apple detach
        self._set_apple_color(self.ripeness)
        if self.two_apples_enabled:
            self._set_spoiled_color(self.spoiled_ripeness)

    def _build_tree(self, pose, trunk_h, branch_z_left, branch_z_right,
                    hang_dx, apple_drop, hang_x_locals):
        """Simple concept-sketch tree: tall trunk, top-only canopy, two staggered branches.

        Trunk and branches have collision matching their visuals so the gripper
        cannot pass through the tree body. Canopy foliage and stems stay
        visual-only (do not block the front grasp approach).
        ``hang_x_locals`` maps side (−1 left / +1 right) → tree-local hang X for each stem.
        """
        builder = self.scene.create_actor_builder()
        phys = self.scene.default_physical_material
        trunk_mat = sapien.render.RenderMaterial(base_color=[*self.TRUNK_COLOR, 1.0])
        branch_mat = sapien.render.RenderMaterial(base_color=[*self.BRANCH_COLOR, 1.0])
        leaf_mat = sapien.render.RenderMaterial(base_color=[*self.LEAF_COLOR, 1.0])
        stem_mat = sapien.render.RenderMaterial(base_color=[*self.STEM_COLOR, 1.0])

        # ---- tall thin trunk (collision matches visual radius) ----
        trunk_r = 0.014
        trunk_pose = sapien.Pose([0, 0, trunk_h / 2], self.VERTICAL_CYL_Q)
        builder.add_cylinder_collision(
            pose=trunk_pose, radius=trunk_r, half_length=trunk_h / 2, material=phys)
        builder.add_cylinder_visual(
            pose=trunk_pose, radius=trunk_r, half_length=trunk_h / 2, material=trunk_mat)

        # ---- two long horizontal branches at staggered heights ----
        # Tip ends exactly above the apple stem (not past it), so the branch doesn't
        # visually skewer the fruit in top-down views.
        branch_len = float(hang_dx)
        branch_half = [branch_len / 2, 0.008, 0.008]
        branch_z = {-1.0: float(branch_z_left), +1.0: float(branch_z_right)}
        for sign, bz in branch_z.items():
            bp = sapien.Pose([sign * branch_len / 2, 0.0, bz], [1, 0, 0, 0])
            builder.add_box_collision(
                pose=bp, half_size=branch_half, material=phys)
            builder.add_box_visual(pose=bp, half_size=branch_half, material=branch_mat)

        # ---- stem under EACH apple (frozen hang: same X as that side's apple) ----
        if hang_x_locals is None or not hang_x_locals:
            raise ValueError("hang_x_locals is required so stem X matches each apple")
        st = self.STEM_HALF_THICK
        for side, hang_x_local in hang_x_locals.items():
            bz = branch_z[float(side)]
            stem_top = bz - 0.008
            # Full stem would reach into the apple; keep top on branch, shorten 50% from bottom.
            stem_bot_full = bz - float(apple_drop) - 0.018
            stem_len = 0.5 * abs(stem_top - stem_bot_full)
            stem_bot = stem_top - stem_len
            stem_half_h = max(1e-4, stem_len / 2)
            stem_z = (stem_top + stem_bot) / 2
            stem_pose = sapien.Pose([float(hang_x_local), 0.0, stem_z], [1, 0, 0, 0])
            builder.add_box_visual(
                pose=stem_pose, half_size=[st, st, stem_half_h], material=stem_mat)

        # ---- canopy ONLY at the top (cloud-like blob, concept sketch) ----
        canopy = [
            ([0.00, 0.00, trunk_h + 0.01], [0.055, 0.050, 0.035]),
            ([0.04, 0.01, trunk_h + 0.00], [0.040, 0.038, 0.030]),
            ([-0.04, -0.01, trunk_h + 0.00], [0.040, 0.038, 0.030]),
            ([0.00, 0.03, trunk_h - 0.01], [0.038, 0.035, 0.028]),
            ([0.00, -0.03, trunk_h - 0.01], [0.038, 0.035, 0.028]),
            ([0.00, 0.00, trunk_h + 0.045], [0.042, 0.040, 0.028]),
        ]
        for center, half in canopy:
            builder.add_box_visual(
                pose=sapien.Pose(center, [1, 0, 0, 0]), half_size=half, material=leaf_mat)

        builder.set_initial_pose(pose)
        entity = builder.build_static(name="apple_tree")
        # Match create_box groups so robot/gripper PhysX contacts the trunk & branches.
        for comp in entity.get_components():
            if not isinstance(comp, sapien.physx.PhysxRigidBaseComponent):
                continue
            for shape in comp.get_collision_shapes():
                shape.set_collision_groups([1, 1, 0, 0])
        data = {
            "center": [0, 0, 0],
            "extents": [2 * branch_len, 0.12, trunk_h + 0.10],
            "scale": [1, 1, 1],
            "transform_matrix": np.eye(4).tolist(),
            "contact_points_pose": [],
            "functional_matrix": [],
            "contact_points_description": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "target_point_description": [],
        }
        return Actor(entity, data)

    # ----------------------------------------------------------- ripeness
    def _color_for(self, r, stops=None):
        d = float(np.clip(r, 0.0, 1.0))
        stops = stops if stops is not None else self.COLOR_STOPS
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        return list(rgb) + [1.0]

    def _set_shapes_color(self, shapes, r, stops=None):
        col = self._color_for(r, stops=stops)
        for s in shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _set_apple_color(self, r):
        """Recolor the good (red-path) apple."""
        self._set_shapes_color(self._apple_shapes, r, stops=self.COLOR_STOPS)

    def _set_spoiled_color(self, r):
        """Recolor the spoiled (yellow-path) apple."""
        self._set_shapes_color(
            getattr(self, "_spoiled_shapes", []), r, stops=self.SPOILED_COLOR_STOPS)

    # --------------------------------------------------------- grasp / detach
    def _tcp_pos(self, arm):
        p = (self.robot.get_left_tcp_pose() if str(arm) == "left"
             else self.robot.get_right_tcp_pose())
        return np.array(p[:3], dtype=float)

    def _enable_apple_dynamic(self, gravity=False, rigid=None):
        """Make an apple a free rigid body (contact/friction grasp, no EE weld)."""
        body = self._apple_rigid if rigid is None else rigid
        if body is None:
            return
        try:
            body.set_kinematic(False)
            body.set_disable_gravity(not bool(gravity))
            body.set_linear_velocity(np.zeros(3))
            body.set_angular_velocity(np.zeros(3))
            # Light damping so mass/inertia dominate (heavy viscous damping made all
            # masses feel identical — terminal speed ≈ F/c).
            if gravity:
                body.set_linear_damping(0.6)
                body.set_angular_damping(2.5)
            else:
                body.set_linear_damping(0.4)
                body.set_angular_damping(1.5)
        except Exception:
            pass

    def _detach_apple(self, side=None):
        """Break the tree attachment and free that side's apple for a physical hold.

        Default ``side`` is the good (red-path) apple. Detaching the spoiled
        apple clears ``_spoiled_attached`` only — it does not latch ``r_grasp``.
        """
        side = float(self.apple_side if side is None else side)
        rigid = self._apple_rigids.get(side)
        if abs(side - float(self.apple_side)) < 0.5:
            if self._apple_attached:
                self._apple_attached = False
                if self.r_grasp is None:
                    self.r_grasp = float(self.ripeness)
                self._latch_detach_metric()
                self._set_apple_color(self.ripeness)   # freeze visual at current ripeness
        else:
            if getattr(self, "_spoiled_attached", False):
                self._spoiled_attached = False
                self._set_spoiled_color(self.spoiled_ripeness)
        # Float in place until the jaws finish closing, then gravity comes back on.
        self._enable_apple_dynamic(gravity=False, rigid=rigid)

    def _enable_held_apple_gravity(self, rigid=None):
        """Turn gravity on so the closed gripper must hold the apple by contact."""
        body = self._apple_rigid if rigid is None else rigid
        if body is None:
            return
        try:
            body.set_kinematic(False)
            body.set_disable_gravity(False)
            # Mild settle damping; restitution=0 + proper inertia do the real work.
            body.set_linear_damping(0.8)
            body.set_angular_damping(3.0)
        except Exception:
            pass

    def _dampen_apple_for_basket_drop(self, rigid=None):
        """Zero restitution + mild damping at release so the apple seats in the basket."""
        body = self._apple_rigid if rigid is None else rigid
        if body is None:
            return
        try:
            body.set_linear_damping(1.0)
            body.set_angular_damping(4.0)
            for s in body.get_collision_shapes():
                m = s.get_physical_material()
                m.set_restitution(0.0)
                m.set_static_friction(6.0)
                m.set_dynamic_friction(6.0)
        except Exception:
            pass

    def _apple_resting_in_basket(self, apple_p, basket_x):
        """True if the good apple is seated in the basket (center below the rim)."""
        return self._pose_in_basket(
            apple_p, np.array([float(basket_x), float(self.basket_y)], dtype=np.float64))

    def _advance_basket_motion(self):
        """Opt 2: ping-pong basket X between branch-tip bounds (never pauses)."""
        if (not getattr(self, "basket_move_enabled", False)
                or getattr(self, "basket", None) is None):
            return
        pose = self.basket.get_pose()
        prev_x = float(pose.p[0])
        dt = float(self.scene.get_timestep())
        speed = abs(float(self.basket_speed))
        next_x = float(prev_x + self._basket_move_dir * speed * dt)
        lo = float(self.basket_x_min)
        hi = float(self.basket_x_max)
        if next_x > hi:
            next_x = hi
            self._basket_move_dir = -1.0
        elif next_x < lo:
            next_x = lo
            self._basket_move_dir = 1.0
        dx = next_x - prev_x
        new_pose = sapien.Pose(
            [next_x, float(self.basket_y), float(pose.p[2])], list(pose.q))
        try:
            # Basket is static+nonconvex (hollow); Opt 2 teleports via set_pose.
            if self._basket_rigid is not None:
                self._basket_rigid.set_kinematic_target(new_pose)
            self.basket.actor.set_pose(new_pose)
        except Exception:
            try:
                self.basket.actor.set_pose(new_pose)
            except Exception:
                pass
        self.basket_center = np.array([next_x, float(self.basket_y)], dtype=np.float64)
        # Carry a landed good apple with the moving basket (set_pose would leave it behind).
        if (abs(dx) > 1e-9
                and not getattr(self, "_apple_attached", True)
                and getattr(self, "apple", None) is not None):
            ap = self.apple.get_pose()
            if self._apple_resting_in_basket(self._apple_eval_p(self.apple), next_x - dx):
                carried = sapien.Pose(
                    [float(ap.p[0]) + dx, float(ap.p[1]), float(ap.p[2])],
                    list(ap.q))
                try:
                    self.apple.actor.set_pose(carried)
                    if self._apple_rigid is not None:
                        self._apple_rigid.set_linear_velocity(
                            np.array([dx / max(dt, 1e-6), 0.0, 0.0]))
                        self._apple_rigid.set_angular_velocity(np.zeros(3))
                except Exception:
                    pass

    def _predict_basket_x(self, future_time: float) -> float:
        """Predict basket X after ``future_time`` seconds of ping-pong motion."""
        pose = self.basket.get_pose()
        x_now = float(pose.p[0])
        if (not getattr(self, "basket_move_enabled", False)) or future_time <= 0.0:
            return x_now
        speed = abs(float(self.basket_speed))
        if speed <= 1e-8:
            return x_now
        lo, hi = float(self.basket_x_min), float(self.basket_x_max)
        if hi <= lo:
            return float(np.clip(x_now, lo, hi))
        x = float(np.clip(x_now, lo, hi))
        direction = 1.0 if self._basket_move_dir >= 0 else -1.0
        remaining = float(future_time)
        for _ in range(32):
            if remaining <= 0.0:
                break
            boundary = hi if direction > 0 else lo
            dist = abs(boundary - x)
            t_to = dist / speed if dist > 1e-9 else 0.0
            if remaining <= t_to:
                x += direction * speed * remaining
                return float(np.clip(x, lo, hi))
            x = boundary
            remaining -= t_to
            direction *= -1.0
        return float(np.clip(x, lo, hi))

    def _wait_basket_align(self, hold_x, tol=None, max_steps=None):
        """Wait until moving basket X passes under the held apple (basket keeps moving)."""
        tol = self.DROP_ALIGN_TOL if tol is None else float(tol)
        max_steps = self.DROP_ALIGN_MAX_STEPS if max_steps is None else int(max_steps)
        if not getattr(self, "basket_move_enabled", False):
            return True
        for j in range(max_steps):
            bx = float(self.basket.get_pose().p[0])
            if abs(bx - float(hold_x)) <= tol:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (j % self.save_freq == 0):
                self._take_picture()
        return False

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not hasattr(self, "apple"):
            return
        # Opt 2: advance basket every physics step (same path as ripeness).
        self._advance_basket_motion()
        # Keep each attached apple pinned to its frozen hang pose.
        if getattr(self, "_apple_attached", False) and self._apple_rigid is not None:
            try:
                self._apple_rigid.set_kinematic_target(self._hang_pose)
            except Exception:
                pass
        if (getattr(self, "_spoiled_attached", False)
                and getattr(self, "_spoiled_rigid", None) is not None):
            try:
                self._spoiled_rigid.set_kinematic_target(
                    self._hang_poses[self.spoiled_side])
            except Exception:
                pass
        if not getattr(self, "_ripen_started", False):
            return
        step = 1.0 / max(1, self.ripen_steps)
        # Good apple: green → red → black while still on the tree.
        if getattr(self, "_apple_attached", False):
            self.ripeness = min(1.0, self.ripeness + step)
            self._set_apple_color(self.ripeness)
        # Spoiled apple: green → yellow → black; expert never grasps it.
        if getattr(self, "_spoiled_attached", False):
            self.spoiled_ripeness = min(1.0, self.spoiled_ripeness + step)
            self._set_spoiled_color(self.spoiled_ripeness)
        self._track_apple_metrics()

    def _ripen_until(self, target):
        max_steps = int(self.ripen_steps) + 600
        for j in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (j % self.save_freq == 0):
                self._take_picture()
            if (not self._apple_attached) or self.ripeness >= target:
                break

    # -------------------------------------------------------- front grasp
    # --- FROZEN control / grasping (see class docstring) — do not retune ---
    # Approach from table front (-Y → +Y): EE +X along world +Y, fingers open
    # along ±X (see GRASP_DIRECTION_DIC["front"]). Matches the concept sketch —
    # horizontal pinch at apple height, not a top-down contact-point grasp.
    FRONT_PRE_GRASP_DIS = 0.10
    FRONT_GRASP_DIS = 0.0
    # TCP sits this far behind the pinch along EE -X (same convention as get_grasp_pose).
    EE_TO_PINCH = 0.12
    # Reject IK "successes" that land the TCP far from the apple center. Right-arm
    # pure ``front`` often reports Success with ~15° down-pitch and closes under.
    GRASP_TCP_ERR_MAX = 0.025
    # After detach: slide outward away from the trunk/branch, then lift.
    CLEAR_LATERAL = 0.08
    CLEAR_LIFT_Z = 0.10
    DROP_SETTLE_STEPS = 200
    GRASP_SETTLE_STEPS = 25

    def _apple_grasp_center(self, apple=None):
        """World-space geometric center of a hanging apple (scaled mesh center)."""
        fruit = self.apple if apple is None else apple
        pose = fruit.get_pose()
        scale = np.asarray(fruit.config["scale"], dtype=np.float64)
        center_local = np.asarray(fruit.config["center"], dtype=np.float64) * scale
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        return np.asarray(pose.p, dtype=np.float64) + R @ center_local

    def _apple_eval_p(self, apple=None):
        """Mesh-center pose used for in-basket / on-table tests (not the actor origin)."""
        fruit = self.apple if apple is None else apple
        if fruit is None:
            return np.zeros(3, dtype=np.float64)
        try:
            return np.asarray(self._apple_grasp_center(fruit), dtype=np.float64)
        except Exception:
            return np.array(fruit.get_pose().p, dtype=np.float64)

    def _basket_mouth_half_xy(self, extents, scale):
        """World-XY half-extents of the basket mouth after ``BASKET_Q``."""
        ext = np.asarray(extents, dtype=np.float64)
        sc = np.asarray(scale, dtype=np.float64)
        hx = 0.5 * float(ext[0]) * float(sc[0])
        hy = float(ext[1]) * float(sc[1])
        hz = 0.5 * float(ext[2]) * float(sc[2])
        if hx <= 1e-6 or hz <= 1e-6:
            return (0.078, 0.111)
        q = np.asarray(self.basket.get_pose().q, dtype=np.float64)
        R = t3d.quaternions.quat2mat(q)
        corners = np.array(
            [[sx * hx, sy * hy, sz * hz]
             for sx in (-1.0, 1.0) for sy in (0.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
        world = (R @ corners.T).T
        half_x = 0.5 * float(world[:, 0].max() - world[:, 0].min())
        half_y = 0.5 * float(world[:, 1].max() - world[:, 1].min())
        if half_x <= 1e-6 or half_y <= 1e-6:
            return (0.078, 0.111)
        return (half_x, half_y)

    def _basket_xy_now(self):
        bp = np.array(self.basket.get_pose().p, dtype=np.float64)
        return np.array([float(bp[0]), float(bp[1])], dtype=np.float64)

    def _xy_in_basket_mouth(self, xy, basket_xy=None, margin=0.0):
        """True when XY sits inside the rectangular basket mouth."""
        c = (self._basket_xy_now() if basket_xy is None
             else np.asarray(basket_xy, dtype=np.float64)[:2])
        half_x, half_y = getattr(self, "basket_half_xy", (0.078, 0.111))
        d = np.abs(np.asarray(xy, dtype=np.float64)[:2] - c)
        return bool(
            d[0] <= max(0.0, float(half_x) - float(margin))
            and d[1] <= max(0.0, float(half_y) - float(margin))
        )

    def _ripeness_fail_reason(self):
        return (
            f"picked outside ripeness window (r_grasp="
            f"{-1.0 if self.r_grasp is None else float(self.r_grasp):.3f}, "
            f"window={self.red_window:.3f}±"
            f"{float(getattr(self, 'red_tol', self.RED_TOLERANCE_DEFAULT)):.3f})"
        )

    def _front_grasp_quat_keys(self, arm_tag):
        """Quat order for a horizontal front pinch.

        Right arm: prefer ``front_left`` (embodiment perfect direction). Pure
        ``front`` often false-succeeds with ~15° down-pitch; we try it last and
        reject by TCP error so a bad pose does not strand the arm.
        """
        if str(arm_tag) == "left":
            return ("front", "front_right", "front_left")
        return ("front_left", "front_right", "front")

    def _front_pre_distances(self, arm_tag):
        """Pre-grasp standoffs to try per arm."""
        if str(arm_tag) == "right":
            return (0.10, 0.08, 0.12)
        return (0.10, 0.08)

    def _ee_pose_front(self, center, quat, pre_dis):
        """EE pose whose pinch point is at ``center``, approaching along EE +X."""
        R = t3d.quaternions.quat2mat(np.asarray(quat, dtype=np.float64))
        p = np.asarray(center, dtype=np.float64) + R @ np.array(
            [-(self.EE_TO_PINCH + float(pre_dis)), 0.0, 0.0], dtype=np.float64)
        return list(p) + list(quat)

    def _try_front_grasp(
        self, arm_tag, pre_grasp_dis=None, grasp_dis=None, gripper_pos=0.0, side=None,
    ):
        """Reach from the front (-Y), pinch the apple, close the gripper.

        Tries arm-preferred front quats and a few pre-distances. Rejects plans
        whose achieved TCP is far from the apple center (right-arm ``front``
        false-success). Detaches the apple to a dynamic body just before close
        so the jaws hold it by contact/friction — no EE weld.

        ``side`` selects which hanging apple (−1 left / +1 right); default is
        the good (red-path) apple. Grasp geometry / clear strategy is unchanged.
        """
        side = float(self.apple_side if side is None else side)
        apple = self.apples.get(side)
        rigid = self._apple_rigids.get(side)
        if apple is None or rigid is None:
            self.plan_success = False
            return False

        g_dis = self.FRONT_GRASP_DIS if grasp_dis is None else float(grasp_dis)
        center = self._apple_grasp_center(apple)
        pre_list = (
            (float(pre_grasp_dis),)
            if pre_grasp_dis is not None
            else self._front_pre_distances(arm_tag)
        )

        for key in self._front_grasp_quat_keys(arm_tag):
            quat = list(GRASP_DIRECTION_DIC[key])
            for pre_dis in pre_list:
                pre_pose = self._ee_pose_front(center, quat, pre_dis)
                grasp_pose = self._ee_pose_front(center, quat, g_dis)
                if os.environ.get("PRA_DEBUG"):
                    print(f"[PRA] try front quat={key} pre={pre_dis:.2f} "
                          f"side={side:+.0f} center={np.round(center, 3)} "
                          f"pre_y={pre_pose[1]:.3f} grasp_y={grasp_pose[1]:.3f}",
                          flush=True)

                self.plan_success = True
                self.move(self.move_to_pose(arm_tag, pre_pose))
                if not self.plan_success:
                    if os.environ.get("PRA_DEBUG"):
                        print(f"[PRA] front pre fail quat={key} "
                              f"fail={getattr(self, '_last_plan_fail', None)}",
                              flush=True)
                    continue

                if abs(pre_dis - g_dis) > 1e-6:
                    self.move(self.move_to_pose(arm_tag, grasp_pose))
                    if not self.plan_success:
                        if os.environ.get("PRA_DEBUG"):
                            print(f"[PRA] front grasp fail quat={key} "
                                  f"fail={getattr(self, '_last_plan_fail', None)}",
                                  flush=True)
                        continue

                tcp = self._tcp_pos(arm_tag)
                err = float(np.linalg.norm(tcp - center))
                if os.environ.get("PRA_DEBUG"):
                    print(f"[PRA] tcp_err={err:.4f} TCP={np.round(tcp, 3)} "
                          f"C={np.round(center, 3)}", flush=True)
                if err > self.GRASP_TCP_ERR_MAX:
                    if os.environ.get("PRA_DEBUG"):
                        print(f"[PRA] reject quat={key} pre={pre_dis:.2f} "
                              f"(tcp too far from apple center)", flush=True)
                    continue

                # Free apple from the stem so fingers can pinch a dynamic body.
                self._detach_apple(side=side)
                self.move(self.close_gripper(arm_tag, pos=gripper_pos))
                if not self.plan_success:
                    # Apple is already free; do not hunt another IK with it floating.
                    self.plan_success = False
                    return False
                # Let contacts form while still floating, then enable gravity so the
                # closed jaws hold via friction (no EE weld).
                for j in range(int(self.GRASP_SETTLE_STEPS)):
                    self._update_kinematic_tasks()
                    self.scene.step()
                    if self.save_freq and (j % self.save_freq == 0):
                        self._take_picture()
                self._enable_held_apple_gravity(rigid=rigid)
                for j in range(10):
                    self._update_kinematic_tasks()
                    self.scene.step()
                return True

        self.plan_success = False
        return False

    # ------------------------------------------------------------- policy
    def play_once(self):
        # Expert only uses the arm on the good (red-path) side; never approaches spoiled.
        arm = ArmTag("left" if self.apple_side < 0 else "right")
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] good_side={self.apple_side:+.0f} spoiled={self.spoiled_side:+.0f} "
                  f"arm={arm} high_branch={self.high_branch_side:+.0f} "
                  f"branch_z={self.branch_z[self.apple_side]:.3f} "
                  f"ripen_steps={self.ripen_steps} red_window={self.red_window:.3f}",
                  flush=True)

        # OBSERVE: wait until almost red, leading by the grasp+lift duration.
        target = max(0.18, self.red_window - self.GRASP_LEAD_STEPS / max(1, self.ripen_steps))
        self._ripen_until(target)
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] trigger_ripeness={self.ripeness:.3f} "
                  f"spoiled_r={self.spoiled_ripeness:.3f} target={target:.3f}",
                  flush=True)

        # ACT: approach good apple from the front (-Y), pinch (physical hold), clear, drop.
        self.move(self.open_gripper(arm))
        self.plan_success = True
        if not self._try_front_grasp(
                arm, grasp_dis=self.FRONT_GRASP_DIS, gripper_pos=0.0):
            if os.environ.get("PRA_DEBUG"):
                print(f"[PRA] after grasp: plan={self.plan_success} "
                      f"fail={getattr(self, '_last_plan_fail', None)} "
                      f"apple={np.array(self.apple.get_pose().p)}", flush=True)
            self.info["info"] = {
                "{A}": "220_apple_plain/base0",
                "{B}": f"{self.BASKET_MODEL}/base{self.basket_id}",
                "{a}": str(arm),
            }
            return self.info
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] after grasp: plan={self.plan_success} "
                  f"apple={np.array(self.apple.get_pose().p)}", flush=True)

        # Clear the branch/trunk: slide further outward (±X), then lift.
        clear_x = float(self.CLEAR_LATERAL * self.apple_side)  # left:-X, right:+X
        self.move(self.move_by_displacement(
            arm_tag=arm, x=clear_x, move_axis="world"))
        self.move(self.move_by_displacement(
            arm_tag=arm, z=self.CLEAR_LIFT_Z, move_axis="world"))
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] after clear/lift: ripeness={self.ripeness:.3f} "
                  f"r_grasp={self.r_grasp} attached={self._apple_attached} "
                  f"plan={self.plan_success} clear_x={clear_x:+.3f} "
                  f"basket_move={self.basket_move_enabled} "
                  f"basket_speed={self.basket_speed:.3f}", flush=True)

        # Transport: aim at predicted basket X (Opt 2). Basket never pauses.
        ap = np.array(self.apple.get_pose().p)
        bp = np.array(self.basket.get_pose().p)
        pred_x = self._predict_basket_x(self.DROP_PREDICT_TIME)
        hover_z = max(self.basket_top_z + 0.05, float(ap[2]))
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] drop aim: basket_now={bp[0]:+.3f} pred_x={pred_x:+.3f} "
                  f"apple={np.round(ap, 3)}", flush=True)
        self.move(self.move_by_displacement(
            arm_tag=arm,
            x=float(pred_x - ap[0]),
            y=float(self.basket_y - ap[1]),
            z=float(hover_z - ap[2]),
            move_axis="world",
        ))
        placed = self.plan_success
        # Wait for the still-moving basket to pass under the held apple, then release.
        hold_x = float(self.apple.get_pose().p[0])
        aligned = self._wait_basket_align(hold_x)
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] align={aligned} hold_x={hold_x:+.3f} "
                  f"basket_x={float(self.basket.get_pose().p[0]):+.3f} "
                  f"(basket keeps moving through drop)", flush=True)
        # Open while basket is still oscillating — natural gravity drop / settle.
        self._dampen_apple_for_basket_drop()
        self.move(self.open_gripper(arm))
        for j in range(int(self.DROP_SETTLE_STEPS)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (j % self.save_freq == 0):
                self._take_picture()
        self.move(self.move_by_displacement(arm_tag=arm, z=0.08, move_axis="arm"))
        self.plan_success = placed

        self.info["info"] = {
            "{A}": "220_apple_plain/base0",
            "{B}": f"{self.BASKET_MODEL}/base{self.basket_id}",
            "{a}": str(arm),
        }
        return self.info

    # --------------------------------------------------------- ripeness score
    def _ripeness_score(self):
        if self.r_grasp is None:
            return 0.0
        return float(np.clip(1.0 - abs(self.r_grasp - self.red_window) / 0.5, 0.0, 1.0))

    def _grasp_in_red_window(self):
        """Whether the good apple was detached while inside the ripeness window."""
        if self.r_grasp is None:
            return False
        tol = float(getattr(self, "red_tol", self.RED_TOLERANCE_DEFAULT))
        return bool(abs(float(self.r_grasp) - float(self.red_window)) <= tol)

    # ------------------------------------------------------------- success
    def _pose_in_basket(self, apple_p, basket_xy):
        """True when the apple mesh center sits in the basket mouth.

        The breadbasket is a shallow rectangle; the actor origin is the mesh tip,
        so callers should pass ``_apple_eval_p`` (bbox center). A seated fruit's
        center is near the rim — do not require it to sit a full centimetre below
        ``basket_top_z``. Lip bounces are rejected by the mouth margin, not Z.
        """
        ap = np.asarray(apple_p, dtype=np.float64)
        in_xy = self._xy_in_basket_mouth(ap[:2], basket_xy, margin=0.005)
        above_floor = float(ap[2]) > (float(self.basket_base_z) - 0.02)
        below_rim = float(ap[2]) < (float(self.basket_top_z) + 0.03)
        return bool(in_xy and above_floor and below_rim)

    def _actor_speed(self, actor) -> float:
        if actor is None:
            return 0.0
        try:
            body = next(
                (c for c in actor.actor.get_components()
                 if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)), None)
            if body is None:
                return 0.0
            return float(np.linalg.norm(body.get_linear_velocity()))
        except Exception:
            return 0.0

    def _apple_held_by_gripper(self, apple) -> bool:
        """True while either gripper still contacts ``apple``."""
        if apple is None:
            return False
        try:
            return len(self.get_gripper_actor_contact_position(apple.get_name())) > 0
        except Exception:
            return False

    def _good_apple_dropped_on_table(self) -> bool:
        """Picked good apple released and settled on the table (missed the basket).

        Used to terminate interactive / eval episodes as FAILURE once the fruit
        is no longer held and rests outside the basket near table height.
        """
        if self.r_grasp is None or getattr(self, "_apple_attached", True):
            return False
        if getattr(self, "apple", None) is None:
            return False
        if self._apple_held_by_gripper(self.apple):
            return False
        ap = self._apple_eval_p(self.apple)
        basket_xy = self._basket_xy_now()
        # Over the mouth (falling in / seated near the rim) is not a table miss.
        if self._xy_in_basket_mouth(ap[:2], basket_xy) or self._pose_in_basket(ap, basket_xy):
            return False
        table_z = float(getattr(self, "_z0", 0.74 + float(self.table_z_bias)))
        z = float(ap[2])
        if z > float(self.basket_top_z) + 0.08:
            return False
        if z < table_z - 0.08:
            return True
        return bool(z <= table_z + 0.10)

    # ------------------------------------------------- experiment metrics
    # Frames of no gripper/apple contact before the release counts as real (finger
    # contact flickers by a frame or two as the jaws part).
    RELEASE_DEBOUNCE_STEPS = 4

    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_window_open_step = None   # ripeness first entered the red window
        self._metric_detach_step = None        # good apple pulled off the tree
        self._metric_basket_step = None        # good apple first inside the basket
        self._metric_release_offset = None     # basket-centre offset at let-go, normalised
        self._metric_apple_free = False
        self._metric_held_seen = False         # jaws have actually contacted the apple
        self._metric_free_frames = 0
        self._metric_last_held_xy = None

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_detach_metric(self):
        """Called from _detach_apple the first time the GOOD apple comes off."""
        try:
            if self._metric_detach_step is None:
                self._metric_detach_step = self._metric_step()
            self._metric_apple_free = True
        except Exception:
            pass

    def _track_apple_metrics(self):
        """Per-step latches: window opening, basket entry, closest release offset."""
        try:
            if self._metric_window_open_step is None and getattr(self, "_apple_attached", False):
                tol = float(getattr(self, "red_tol", self.RED_TOLERANCE_DEFAULT))
                if float(self.ripeness) >= float(self.red_window) - tol:
                    self._metric_window_open_step = self._metric_step()
            if not getattr(self, "_metric_apple_free", False):
                return
            bp = np.array(self.basket.get_pose().p, dtype=np.float64)
            ap = self._apple_eval_p(self.apple)
            # Latch the aim the moment the jaws let go — after that the apple is
            # just falling and its XY no longer reflects the operator's choice.
            # Gate on having actually been held (detach precedes jaw contact by a
            # few frames) and debounce, since finger contact flickers as they part.
            held = self._apple_held_by_gripper(self.apple)
            if held:
                self._metric_held_seen = True
                self._metric_free_frames = 0
                self._metric_last_held_xy = (float(ap[0]), float(ap[1]))
            elif self._metric_held_seen:
                self._metric_free_frames += 1
            if (self._metric_release_offset is None
                    and self._metric_held_seen
                    and self._metric_free_frames >= self.RELEASE_DEBOUNCE_STEPS):
                rx, ry = self._metric_last_held_xy or (float(ap[0]), float(ap[1]))
                self._metric_release_offset = float(
                    np.linalg.norm(np.array([rx, ry], dtype=np.float64) - bp[:2])) / 0.09
            if self._metric_basket_step is None and self._pose_in_basket(ap, bp[:2]):
                self._metric_basket_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `pick_latency_steps` — steps from the ripeness window opening until the
        apple is pulled off the tree. Signed: negative means it was picked early, before
        the window opened. This is the whole task's timing decision.
        extra2 `ripeness_error_norm` — |r_grasp - red_window| / red_tol at the instant of
        detach. LOWER is better; <= 1.0 is inside the accepted window.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        w = getattr(self, "_metric_window_open_step", None)
        d = getattr(self, "_metric_detach_step", None)
        lat = None if (w is None or d is None) else int(d) - int(w)
        out["pick_latency_steps"] = lat
        out["pick_latency_s"] = None if lat is None else round(lat * dt, 4)

        b = getattr(self, "_metric_basket_step", None)
        drop = None if (d is None or b is None) else max(int(b) - int(d), 0)
        out["basket_latency_steps"] = drop
        out["basket_latency_s"] = None if drop is None else round(drop * dt, 4)

        try:
            r = getattr(self, "r_grasp", None)
            tol = float(getattr(self, "red_tol", self.RED_TOLERANCE_DEFAULT))
            out["ripeness_error_norm"] = (
                None if r is None
                else round(abs(float(r) - float(self.red_window)) / max(tol, 1e-9), 4)
            )
            out["r_grasp"] = None if r is None else round(float(r), 4)
        except Exception:
            out["ripeness_error_norm"] = None
            out["r_grasp"] = None

        off = getattr(self, "_metric_release_offset", None)
        out["drop_offset_norm"] = None if off is None else round(float(off), 4)
        return out

    def check_success(self):
        # Success = good apple picked inside the ripeness window AND in the basket
        # AND spoiled (if any) not in basket.
        basket_xy = self._basket_xy_now()
        self.basket_center = basket_xy

        ap = self._apple_eval_p(self.apple)
        over_basket = self._xy_in_basket_mouth(ap[:2], basket_xy)
        settled = (
            self._pose_in_basket(ap, basket_xy)
            and (not self._apple_held_by_gripper(self.apple))
            and (self._actor_speed(self.apple) < 0.12)
        )
        good_in = bool(settled and (self.r_grasp is not None))
        ripeness_ok = self._grasp_in_red_window()
        on_table = bool(self._good_apple_dropped_on_table())

        spoiled = getattr(self, "spoiled_apple", None)
        spoiled_in = False
        if spoiled is not None:
            spoiled_in = bool(self._pose_in_basket(
                self._apple_eval_p(spoiled), basket_xy))

        success = bool(good_in and ripeness_ok and not spoiled_in)
        if not success:
            # An in/over-basket drop must not be labeled "on table" — that hid
            # ripeness misses and false-negatived seated fruit (seed 10000).
            if not ripeness_ok and self.r_grasp is not None and (good_in or over_basket):
                self._last_fail_reason = self._ripeness_fail_reason()
            elif on_table:
                self._last_fail_reason = "good apple dropped on table (missed basket)"
            elif spoiled_in:
                self._last_fail_reason = "spoiled apple in basket"
            elif not good_in:
                self._last_fail_reason = "good apple not in basket"

        ripe = self._ripeness_score()
        self.info["ripeness_score"] = ripe
        self.info["r_grasp"] = float(self.r_grasp) if self.r_grasp is not None else -1.0
        self.info["final_score"] = ripe if success else 0.0
        self.info["in_basket"] = bool(good_in)
        self.info["on_table"] = bool(on_table)
        self.info["ripeness_ok"] = bool(ripeness_ok)
        self.info["red_window"] = float(self.red_window)
        self.info["red_tolerance"] = float(
            getattr(self, "red_tol", self.RED_TOLERANCE_DEFAULT))
        self.info["spoiled_in_basket"] = bool(spoiled_in)
        self.info["apple_side"] = float(self.apple_side)
        self.info["good_side"] = float(self.apple_side)
        self.info["spoiled_side"] = float(getattr(self, "spoiled_side", 0.0))
        self.info["basket_speed"] = float(getattr(self, "basket_speed", 0.0))
        self.info["basket_move_enabled"] = bool(getattr(self, "basket_move_enabled", False))
        self.info["basket_id"] = int(getattr(self, "basket_id", self.BASKET_IDS[0]))
        self.info["basket_model"] = str(getattr(self, "BASKET_MODEL", "076_breadbasket"))
        self.info["basket_color"] = list(getattr(self, "basket_color", []) or [])
        self.info["basket_texture"] = getattr(self, "basket_texture", None)
        self.info["two_apples_enabled"] = bool(getattr(self, "two_apples_enabled", False))
        self.info["spoiled_apple_enabled"] = bool(getattr(self, "spoiled_apple_enabled", False))
        self.info["n_apples"] = int(len(getattr(self, "apples", {}) or {}))
        return success

    def get_obs(self):
        obs = super().get_obs()
        bp = (np.array(self.basket.get_pose().p)
              if getattr(self, "basket", None) is not None else np.zeros(3))
        obs["ripening"] = {
            "ripeness": float(getattr(self, "ripeness", 0.0)),
            "spoiled_ripeness": float(getattr(self, "spoiled_ripeness", 0.0)),
            "r_grasp": float(self.r_grasp) if getattr(self, "r_grasp", None) is not None else -1.0,
            "red_window": float(getattr(self, "red_window", 0.5)),
            "red_tolerance": float(
                getattr(self, "red_tol", self.RED_TOLERANCE_DEFAULT)),
            "ripeness_ok": bool(self._grasp_in_red_window()),
            "attached": bool(getattr(self, "_apple_attached", False)),
            "spoiled_attached": bool(getattr(self, "_spoiled_attached", False)),
            "apple_side": float(getattr(self, "apple_side", 0.0)),
            "good_side": float(getattr(self, "good_side", getattr(self, "apple_side", 0.0))),
            "spoiled_side": float(getattr(self, "spoiled_side", 0.0)),
            "basket_x": float(bp[0]),
            "basket_speed": float(getattr(self, "basket_speed", 0.0)),
            "basket_move_enabled": bool(getattr(self, "basket_move_enabled", False)),
            "basket_dir": float(getattr(self, "_basket_move_dir", 0.0)),
            "basket_id": int(getattr(self, "basket_id", self.BASKET_IDS[0])),
            "basket_model": str(getattr(self, "BASKET_MODEL", "076_breadbasket")),
            "basket_color": list(getattr(self, "basket_color", []) or []),
            "basket_texture": getattr(self, "basket_texture", None),
            "two_apples_enabled": bool(getattr(self, "two_apples_enabled", False)),
            "n_apples": int(len(getattr(self, "apples", {}) or {})),
        }
        return obs
