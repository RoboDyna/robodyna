from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import os
import tempfile
import time
import sapien
import sapien.render
import sapien.physx
import numpy as np
import transforms3d as t3d
from shapely.geometry import Point, Polygon
from trimesh.creation import extrude_polygon


class play_billiard(Base_Task):
    """Single-arm billiards: strike the red primary ball into a pocket with a cue stick.

    Table is centered at (x=0, y=0.1). The primary ball spawns on a random side; the
    matching arm (left/right) grasps a cue placed on that same side. Success = primary
    ball falls through an allowed pocket into the hollow interior. Automatic failure if:
    robot-link contact with the primary ball, cue contact with any non-target ball, or
    any non-target ball falls into a pocket. Only the blue cue tip may contact the
    primary ball, and only for a single hit (shaft/butt never collide with balls;
    tip collision disables after).

    Options (independent toggles; CLI via ``--task-arg`` or legacy ``--option``):
      Default — only the red target ball; success in any of the 6 pockets.
      Opt 1 — ``specific_hole``: episode picks one top pocket (left / middle / right)
        and draws an arrow pointing at it; success only in that pocket.
        CLI: ``--task-arg specific_hole=true`` or ``--option 1``.
      Opt 2 — ``enable_distractors``: spawn up to 2 colored balls on the table;
        cue must not touch them and they must not be pocketed; primary may still
        sink in any hole.
        CLI: ``--task-arg enable_distractors=true`` or ``--option 2``.
      Opt 1+2 — episode picks one top pocket (arrow) and distractors occupy a
        different top pocket. The nominated hole and the shot into it stay clear.
        CLI: both ``--task-arg`` flags together.

    Legacy aliases still work: ``pocket_mode=top_three`` → Opt 1;
    ``blocker_mode=path_to_top|unused_tops|block_one`` → Opt 2.

    Built entirely from SAPIEN primitives (reserved ids [350,359] in the NOTICE below).
    """

    # ----- table (centered; doubled playing surface)
    TABLE_CX_DEFAULT = 0.0
    TABLE_CY_DEFAULT = 0.1
    TABLE_HALF_LEN_DEFAULT = 0.22   # along +X (long side); full length 0.44 m
    TABLE_HALF_WID_DEFAULT = 0.14   # along +Y (short side); full width 0.28 m
    BASE_HEIGHT = 0.10              # 10 cm hollow wood box under the felt
    WALL_T = 0.012                  # wall thickness of the hollow box
    FLOOR_HALF_Z = 0.005            # thin inner floor so pocketed balls rest inside
    LID_HALF_Z = 0.003              # thin green lid (with real pocket openings)
    # Invisible collision slab under the felt (top flush with felt_top). Thick
    # enough that a held/kinematic cue cannot tunnel into the hollow box; same
    # pocket cutouts so balls still fall through.
    FELT_COLLISION_HALF_Z = 0.016
    RAIL_HALF_H = 0.002             # very low cushion so the cue clears it
    RAIL_HALF_T = 0.006
    # Pocket diameter = 1.2 × ball diameter ⇒ radius = 1.2 × ball radius.
    POCKET_DIAMETER_SCALE = 1.2
    BASE_COLOR = [0.42, 0.26, 0.12]
    FELT_COLOR = [0.10, 0.45, 0.22]
    RAIL_COLOR = [0.35, 0.18, 0.08]
    POCKET_COLOR = [0.04, 0.04, 0.04]
    # Cylinder default axis is +X; this quat aligns it with +Z (vertical wells).
    Z_CYL_Q = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], np.pi / 2).tolist()

    # Pocket index layout from _build_table:
    # 0 near_left, 1 top_left, 2 near_right, 3 top_right, 4 near_middle, 5 top_middle
    POCKET_NAMES = [
        "near_left", "top_left", "near_right", "top_right", "near_middle", "top_middle",
    ]
    TOP_POCKET_IDS = (1, 5, 3)  # left, middle, right along the far (+Y) edge

    # ----- balls
    BALL_RADIUS_DEFAULT = 0.012
    NUM_EXTRA_BALLS_DEFAULT = 0
    SPECIFIC_HOLE_DEFAULT = False       # Opt 1
    ENABLE_DISTRACTORS_DEFAULT = False  # Opt 2
    PRIMARY_COLOR = [0.90, 0.12, 0.12]
    EXTRA_COLORS = [
        [0.15, 0.35, 0.90],  # blue
        [0.95, 0.75, 0.10],  # yellow
        [0.55, 0.20, 0.75],  # purple
    ]
    BALL_MASS = 0.04
    # Target-hole arrow (Opt 1): bright marker pointing at the goal pocket.
    ARROW_COLOR = [0.98, 0.85, 0.10]
    # PhysX only applies restitution when |v_rel| > bounce_threshold (engine
    # default is 2 m/s). Cue strike is ~0.55 m/s, so we must lower it or
    # ball–ball hits stay inelastic (both balls "stick" and slide together).
    BOUNCE_THRESHOLD_DEFAULT = 0.15
    BALL_RESTITUTION = 0.92
    BALL_FRICTION = 0.05
    BALL_LINEAR_DAMPING = 0.04
    BALL_ANGULAR_DAMPING = 0.08
    BALL_SLEEP_THRESHOLD = 0.001

    # ----- cue rod (local +X = tip direction; cylinder + spherical ends)
    CUE_RADIUS = 0.0072  # 1.2× prior 6 mm shaft for a more graspable stick
    CUE_HALF_LEN = 0.180            # 3x the original 12 cm cue length
    CUE_COLOR = [0.72, 0.52, 0.28]
    CUE_TIP_COLOR = [0.05, 0.35, 1.0]  # vivid blue tip (was off-white)
    CUE_MASS = 0.05
    # Grasp must seat the shaft between the finger pads. Do NOT teleport-lift the
    # cue into the gripper after close — that looks like a magnet attach.
    CUE_WELD_LIFT_Z = 0.0
    # WSG50 pads sit only ~3 cm below the planning EE (not the 12 cm TCP offset
    # assumed by get_grasp_pose). Top-down IK also cannot reach table height, so
    # the cue rests on a short stand and we target finger-pad height explicitly.
    CUE_STAND_HEIGHT = 0.084         # 30% lower than the original 12 cm cradle
    # Planning-EE Z − shaft Z. Pads sit ≈3 cm below EE; +3 cm extra puts the
    # shaft between the fingertips instead of deep in the jaw.
    CUE_FINGER_EE_Z = 0.060
    CUE_TIP_GRASP_CLEARANCE = 0.030  # target pad-link Z − shaft Z
    # After lift, deepen weld slightly so tip↔felt strikes stay reachable.
    CUE_STRIKE_HANG = 0.085
    CUE_PRE_GRASP_DIS = 0.12
    CUE_GRASP_OPEN = 0.85
    CUE_GRASP_CLOSE = 0.0
    # Hold close to the butt end, leaving the long tip section free for aiming.
    CUE_HANDLE_GRASP_FROM_BUTT = 0.030

    # ----- strike / settle
    APPROACH_GAP = 0.050
    STRIKE_PUSH = 0.055
    # Tip hover clearance above the felt so the shaft clears the 10 cm box + rails.
    HOVER_CLEARANCE = 0.10
    SETTLE_STEPS_DEFAULT = 500
    STRIKE_IMPULSE = 0.62
    # Tip apex → ball-center distance at surface touch ≈ ball_r + 2*cue_r
    # (apex sits one cue radius past the tip-sphere center). Extra slack for
    # alignment error; PhysX contact is still required to fire the impulse.
    STRIKE_CONTACT_GAP = 0.008
    # Collision ignore pair: cue ↔ robot links share this g2/g3 so they do not
    # collide after weld, while cue tip ↔ ball (default g3) still generates contacts.
    _CUE_ROBOT_IGNORE_BIT = 1 << 8
    _CUE_ROBOT_IGNORE_ID = 0xB111
    # Unique g2 bits per ball. PhysX skips a pair when (g2_A & g2_B) and g3
    # match — a shared bit made every ball ignore every other ball, so the
    # red primary tunneled through distractors. Shaft / spent tip OR the
    # full mask and still ignore every ball.
    _CUE_SHAFT_BALL_IGNORE_BITS = (1 << 9, 1 << 10, 1 << 11)
    _CUE_SHAFT_BALL_IGNORE_MASK = (1 << 9) | (1 << 10) | (1 << 11)
    POCKET_SINK_Z = 0.012

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("play_billiard", {})
        # Invalidate per-step state before _init_task_env_ (it may call
        # _update_kinematic_tasks during camera load, before load_actors).
        self._loaded = False
        # False through check_stable so spawn contacts cannot strike or sink.
        self._billiard_live = False
        self._cue_welded = False
        self._primary_pocketed = False
        self._primary_pocket_id = None
        self._distractor_pocketed = False
        self._pocketed_extra_ids = set()
        self._pocketed_extra_entities = set()
        self._robot_ball_contact = False
        self._cue_distractor_contact = False
        self._strike_armed = False
        self._strike_done = False
        self._cue_tip_hit_allowed = True
        self._reset_metric_state()
        self._cue_tip_shapes = []
        self._cue_shaft_shapes = []
        self.cue = None
        self.primary_ball = None
        self.extra_balls = []
        self._extra_rigids = []
        self._primary_rigid = None
        self._pocket_centers = []
        self._arm_side = "right"
        self._target_arrow_parts = []
        self._blocked_pocket_id = None
        super()._init_task_env_(**kwags)
        # After settle: balls rest, cue is on the stand. Scoring starts only
        # from a real tip hit / Space — never from leftover spawn contacts.
        self._billiard_live = True

    def setup_scene(self, **kwargs):
        """Create the scene with a low bounce threshold so ball–ball hits rebound.

        SAPIEN/PhysX defaults ``bounce_threshold=2`` m/s, which silently disables
        restitution for our cue-strike speeds. Patch ``Engine.create_scene`` for
        this call only so Base_Task's scene setup picks up the billiard values.
        """
        bounce_th = float(
            self._cfg.get("bounce_threshold", self.BOUNCE_THRESHOLD_DEFAULT)
        )
        orig_create = sapien.Engine.create_scene

        def _create_with_bounce(engine, config=None):
            if config is None:
                config = sapien.SceneConfig()
            config.bounce_threshold = bounce_th
            # Fast primary after the strike can otherwise tunnel through a resting ball.
            try:
                config.enable_ccd = True
            except Exception:
                pass
            return orig_create(engine, config)

        sapien.Engine.create_scene = _create_with_bounce
        try:
            super().setup_scene(**kwargs)
        finally:
            sapien.Engine.create_scene = orig_create

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
        raise ValueError(f"play_billiard expected a boolean, got {value!r}")

    def _parse_specific_hole(self, c) -> bool:
        """Opt 1: specific target pocket (+ arrow)."""
        specific = c.get("specific_hole", c.get("opt1", None))
        legacy = c.get("option", None)
        if legacy is not None and specific is None:
            if legacy in (1, "1", "specific_hole", "specific", "top_three"):
                specific = True
            elif legacy in (2, "2", "enable_distractors", "enable_distractor", "distractors"):
                specific = False
            else:
                raise ValueError(
                    "play_billiard option must be 1/specific_hole or "
                    "2/enable_distractors (or set the named booleans directly)"
                )
        if specific is not None:
            return self._as_bool(specific, self.SPECIFIC_HOLE_DEFAULT)
        # Legacy pocket_mode
        mode = str(c.get("pocket_mode", "any")).strip().lower()
        return mode in ("top_three", "specific", "specific_hole")

    def _parse_enable_distractors(self, c) -> bool:
        """Opt 2: up to two colored distractor balls on the table."""
        distractors = c.get(
            "enable_distractors",
            c.get("enable_distractor", c.get("opt2", None)),
        )
        legacy = c.get("option", None)
        if legacy is not None and distractors is None:
            if legacy in (2, "2", "enable_distractors", "enable_distractor", "distractors"):
                distractors = True
            elif legacy in (1, "1", "specific_hole", "specific", "top_three"):
                distractors = False
            else:
                raise ValueError(
                    "play_billiard option must be 1/specific_hole or "
                    "2/enable_distractors (or set the named booleans directly)"
                )
        if distractors is not None:
            return self._as_bool(distractors, self.ENABLE_DISTRACTORS_DEFAULT)
        # Legacy blocker_mode (num_extra_balls is now only a count when Opt2 is on)
        blocker = str(c.get("blocker_mode", "none")).strip().lower()
        return blocker in ("path_to_top", "unused_tops", "block_one")

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.table_cx = float(c.get("table_cx", self.TABLE_CX_DEFAULT))
        self.table_cy = float(c.get("table_cy", self.TABLE_CY_DEFAULT))
        self.table_half_len = float(c.get("table_half_len", self.TABLE_HALF_LEN_DEFAULT))
        self.table_half_wid = float(c.get("table_half_wid", self.TABLE_HALF_WID_DEFAULT))
        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        # Pocket diameter = 1.2 × ball diameter (override with pocket_radius if given).
        default_pr = self.POCKET_DIAMETER_SCALE * self.ball_radius
        self.pocket_radius = float(c.get("pocket_radius", default_pr))
        self.specific_hole = self._parse_specific_hole(c)
        self.enable_distractors = self._parse_enable_distractors(c)
        # Compat aliases used by aim / success / obs.
        self.pocket_mode = "top_three" if self.specific_hole else "any"
        n_cfg = int(c.get("num_extra_balls", self.NUM_EXTRA_BALLS_DEFAULT))
        if self.enable_distractors:
            # Up to two balls: keep an explicit 1–2, else sample.
            if n_cfg <= 0:
                self.num_extra_balls = int(np.random.randint(1, 3))
            else:
                self.num_extra_balls = int(np.clip(n_cfg, 1, 2))
            # Prefer unused_tops when Opt1+2 (block one hole); else path obstacles.
            raw_blocker = str(c.get("blocker_mode", "")).strip().lower()
            if raw_blocker in ("none", "path_to_top", "unused_tops", "block_one"):
                self.blocker_mode = raw_blocker if raw_blocker != "none" else (
                    "block_one" if self.specific_hole else "path_to_top"
                )
            else:
                self.blocker_mode = "block_one" if self.specific_hole else "path_to_top"
        else:
            self.num_extra_balls = 0
            self.blocker_mode = "none"
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.strike_impulse = float(c.get("strike_impulse", self.STRIKE_IMPULSE))

        self.z0 = 0.74 + self.table_z_bias
        self.base_top = self.z0 + self.BASE_HEIGHT
        self.felt_top = self.base_top + 2.0 * self.LID_HALF_Z
        self.floor_top = self.z0 + 2.0 * self.FLOOR_HALF_Z
        self.ball_z = self.felt_top + self.ball_radius + 0.001
        # Workspace table surface for interactive Q floor (fingers ≥ this height).
        self.table_top = float(self.z0)

        self._primary_pocketed = False
        self._primary_pocket_id = None
        self._distractor_pocketed = False
        self._robot_ball_contact = False
        self._cue_distractor_contact = False
        self._cue_welded = False
        self._strike_armed = False
        self._strike_done = False
        self._cue_tip_hit_allowed = True
        self._reset_metric_state()
        self._cue_tip_shapes = []
        self._cue_shaft_shapes = []
        self._ball_ignore_bit_i = 0
        self._aim_dir = np.array([1.0, 0.0], dtype=np.float64)
        self._target_pocket = None
        self._target_pocket_id = None
        self._target_pocket_name = "any"
        self._allowed_pocket_ids = list(range(6))
        self._target_arrow_parts = []
        self._blocked_pocket_id = None
        self._pocketed_extra_ids = set()
        self._pocketed_extra_entities = set()

        self._build_table()
        self._spawn_balls()
        self._spawn_cue()
        self._robot_link_names = self._collect_robot_link_names()
        self._loaded = True

    # ------------------------------------------------------------- table build
    def _add_pocket_floor_well_visual(self, px, py, index):
        """Dark disc on the inner floor, visible through the lid cutout (no collision)."""
        well_mat = sapien.render.RenderMaterial(base_color=[*self.POCKET_COLOR, 1.0])
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.Z_CYL_Q),
            radius=self.pocket_radius * 0.95,
            half_length=0.0008,
            material=well_mat,
        )
        builder.set_initial_pose(
            sapien.Pose([px, py, self.floor_top + 0.001], [1, 0, 0, 0])
        )
        builder.build_static(name=f"pocket_floor_well_{index}")

    def _felt_polygon_with_pockets(self, cx, cy, hl, hw):
        """Table-top polygon in lid-local XY with circular pocket holes cut out."""
        pr = float(self.pocket_radius)
        poly = Polygon([(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)])
        for p in self._pocket_centers:
            local = Point(float(p[0] - cx), float(p[1] - cy))
            poly = poly.difference(local.buffer(pr, resolution=24))
        if poly.is_empty:
            raise RuntimeError("pocket cutouts removed the entire felt lid")
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly

    def _export_extruded_lid_mesh(self, poly, thickness, path):
        mesh = extrude_polygon(poly, height=float(thickness))
        mesh.apply_translation([0.0, 0.0, -0.5 * float(thickness)])
        try:
            mesh.fix_normals()
        except Exception:
            pass
        mesh.export(path)
        return path

    def _build_felt_lid_with_holes(self, cx, cy, hl, hw):
        """Green felt lid visual + thicker static collision (same pocket openings).

        The held cue is kinematic/welded, so thin triangle meshes alone are easy
        to tunnel through. A thicker collision slab (top flush with ``felt_top``)
        gives the table a solid playing surface while preserving pocket holes.
        """
        poly = self._felt_polygon_with_pockets(cx, cy, hl, hw)
        tmp = tempfile.gettempdir()

        # Thin green visual (matches prior look / felt_top height).
        vis_thickness = 2.0 * self.LID_HALF_Z
        vis_path = os.path.join(tmp, "play_billiard_felt_lid_vis.obj")
        self._export_extruded_lid_mesh(poly, vis_thickness, vis_path)
        felt_mat = sapien.render.RenderMaterial(base_color=[*self.FELT_COLOR, 1.0])
        vis_builder = self.scene.create_actor_builder()
        vis_builder.set_physx_body_type("static")
        vis_builder.add_visual_from_file(filename=vis_path, material=felt_mat)
        # Keep a thin collision on the visual too (balls rest right at felt_top).
        vis_builder.add_nonconvex_collision_from_file(filename=vis_path)
        vis_builder.set_initial_pose(
            sapien.Pose([cx, cy, self._lid_z], [1, 0, 0, 0])
        )
        vis_builder.build_static(name="felt_lid")

        # Thicker invisible collision slab: top face at felt_top, extends downward
        # into the hollow box so the cue cannot sink inside the table.
        coll_half = float(self.FELT_COLLISION_HALF_Z)
        coll_thickness = 2.0 * coll_half
        coll_path = os.path.join(tmp, "play_billiard_felt_lid_coll.obj")
        self._export_extruded_lid_mesh(poly, coll_thickness, coll_path)
        coll_z = float(self.felt_top - coll_half)
        coll_builder = self.scene.create_actor_builder()
        coll_builder.set_physx_body_type("static")
        coll_builder.add_nonconvex_collision_from_file(filename=coll_path)
        coll_builder.set_initial_pose(
            sapien.Pose([cx, cy, coll_z], [1, 0, 0, 0])
        )
        coll_builder.build_static(name="felt_lid_collision")

    def _build_hollow_box(self, cx, cy, hl, hw):
        """10 cm hollow wood box: floor + four walls (empty interior)."""
        wt = self.WALL_T
        wall_hz = 0.5 * self.BASE_HEIGHT
        wall_z = self.z0 + wall_hz
        floor_z = self.z0 + self.FLOOR_HALF_Z
        inner_hl = hl - wt
        inner_hw = hw - wt

        # Inner floor — pocketed balls land here.
        create_box(
            self,
            pose=sapien.Pose([cx, cy, floor_z], [1, 0, 0, 0]),
            half_size=[inner_hl, inner_hw, self.FLOOR_HALF_Z],
            color=tuple(self.BASE_COLOR),
            is_static=True,
            name="billiard_floor",
        )
        # Long walls (±Y).
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            create_box(
                self,
                pose=sapien.Pose(
                    [cx, cy + sign * (hw - 0.5 * wt), wall_z], [1, 0, 0, 0]
                ),
                half_size=[hl, 0.5 * wt, wall_hz],
                color=tuple(self.BASE_COLOR),
                is_static=True,
                name=f"billiard_wall_long_{tag}",
            )
        # Short walls (±X), inset so corners do not double-stack.
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            create_box(
                self,
                pose=sapien.Pose(
                    [cx + sign * (hl - 0.5 * wt), cy, wall_z], [1, 0, 0, 0]
                ),
                half_size=[0.5 * wt, hw - wt, wall_hz],
                color=tuple(self.BASE_COLOR),
                is_static=True,
                name=f"billiard_wall_short_{tag}",
            )

    def _build_table(self):
        cx, cy = self.table_cx, self.table_cy
        hl, hw = self.table_half_len, self.table_half_wid
        pr = self.pocket_radius
        # Inset so each pocket is a full circle inside the green lid.
        rim = 0.008
        pin = pr + rim
        self._lid_z = self.base_top + self.LID_HALF_Z

        self._pocket_centers = [
            np.array([cx - hl + pin, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx - hl + pin, cy + hw - pin, self.felt_top], dtype=np.float64),
            np.array([cx + hl - pin, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx + hl - pin, cy + hw - pin, self.felt_top], dtype=np.float64),
            np.array([cx, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx, cy + hw - pin, self.felt_top], dtype=np.float64),
        ]

        # 1) Hollow 10 cm wood box (empty cavity under the lid).
        self._build_hollow_box(cx, cy, hl, hw)

        # 2) Green felt lid with genuine circular cutouts (balls fall through).
        self._build_felt_lid_with_holes(cx, cy, hl, hw)
        # Dark well pads on the inner floor — seen through the holes, not lid paint.
        for i, p in enumerate(self._pocket_centers):
            self._add_pocket_floor_well_visual(float(p[0]), float(p[1]), i)

        # 3) Very low continuous perimeter rail (cue clears it).  The genuine
        # pocket cutouts are inset in the felt lid, so unlike split rail pieces
        # this leaves no distracting gap on the table's outside edge.
        rail_z = self.felt_top + self.RAIL_HALF_H
        for sign_y, tag in ((-1.0, "neg"), (1.0, "pos")):
            y = cy + sign_y * (hw + self.RAIL_HALF_T)
            create_box(
                self,
                pose=sapien.Pose([cx, y, rail_z], [1, 0, 0, 0]),
                half_size=[hl, self.RAIL_HALF_T, self.RAIL_HALF_H],
                color=tuple(self.RAIL_COLOR),
                is_static=True,
                name=f"rail_long_{tag}",
            )
        for sign_x, tag in ((-1.0, "neg"), (1.0, "pos")):
            x = cx + sign_x * (hl + self.RAIL_HALF_T)
            create_box(
                self,
                pose=sapien.Pose([x, cy, rail_z], [1, 0, 0, 0]),
                half_size=[self.RAIL_HALF_T, hw, self.RAIL_HALF_H],
                color=tuple(self.RAIL_COLOR),
                is_static=True,
                name=f"rail_short_{tag}",
            )

        self.prohibited_area.append([
            cx - hl - 0.04,
            cy - hw - 0.04,
            cx + hl + 0.04,
            cy + hw + 0.04,
        ])

    # ------------------------------------------------------------- balls
    def _table_xy_bounds(self):
        margin = self.pocket_radius + self.ball_radius + 0.015
        x_lo = self.table_cx - self.table_half_len + margin
        x_hi = self.table_cx + self.table_half_len - margin
        y_lo = self.table_cy - self.table_half_wid + margin
        y_hi = self.table_cy + self.table_half_wid - margin
        return x_lo, x_hi, y_lo, y_hi

    def _set_target_pocket(self, pid):
        """Lock success / language / aim to a single pocket id."""
        self._target_pocket_id = int(pid)
        self._target_pocket = self._pocket_centers[self._target_pocket_id]
        self._target_pocket_name = self.POCKET_NAMES[self._target_pocket_id]
        self._allowed_pocket_ids = [self._target_pocket_id]

    def _select_goal_pocket(self):
        """Choose the episode goal pocket (and allowed success set).

        Default / Opt 2 only → any of 6 pockets.
        Opt 1 / Opt 1+2 → one of the three far (+Y) pockets. Distractor
        placement (Opt 1+2) happens later and must not occupy this hole.
        """
        self._blocked_pocket_id = None
        if not self.specific_hole:
            self._target_pocket_id = None
            self._target_pocket = None
            self._target_pocket_name = "any"
            self._allowed_pocket_ids = list(range(6))
            return

        forced = str(self._cfg.get("target_pocket", "")).lower()
        name_to_id = {
            "top_left": 1, "top_middle": 5, "top_mid": 5, "top_right": 3,
            "near_left": 0, "near_middle": 4, "near_mid": 4, "near_right": 2,
        }

        if forced in name_to_id and name_to_id[forced] in self.TOP_POCKET_IDS:
            pid = name_to_id[forced]
        else:
            pid = int(np.random.choice(self.TOP_POCKET_IDS))
        self._set_target_pocket(pid)

    def _draw_target_arrow(self, pocket_id):
        """Flat yellow arrow on the felt pointing at the given pocket (Opt 1)."""
        pxy = np.asarray(self._pocket_centers[int(pocket_id)][:2], dtype=np.float64)
        center = np.array([self.table_cx, self.table_cy], dtype=np.float64)
        toward = pxy - center
        n = float(np.linalg.norm(toward))
        if n < 1e-6:
            toward = np.array([0.0, 1.0], dtype=np.float64)
        else:
            toward = toward / n
        # Tip just inboard of the pocket rim; shaft extends toward table center.
        tip_xy = pxy - toward * (self.pocket_radius + 0.012)
        yaw = float(np.arctan2(toward[1], toward[0]))
        z = float(self.felt_top + 0.0015)
        # Local +X = pointing direction (toward the hole).
        parts = [
            ("shaft", [-0.018, 0.0], 0.0, [0.016, 0.0035, 0.0012]),
            ("head_upper", [0.006, 0.007], 0.70, [0.012, 0.0032, 0.0012]),
            ("head_lower", [0.006, -0.007], -0.70, [0.012, 0.0032, 0.0012]),
        ]
        c, s = np.cos(yaw), np.sin(yaw)
        self._target_arrow_parts = []
        for name, (lx, ly), local_yaw, half_size in parts:
            x = float(tip_xy[0] + c * lx - s * ly)
            y = float(tip_xy[1] + s * lx + c * ly)
            part_yaw = yaw + local_yaw
            q = [np.cos(part_yaw / 2.0), 0.0, 0.0, np.sin(part_yaw / 2.0)]
            arrow = create_visual_box(
                self,
                sapien.Pose([x, y, z], q),
                half_size=half_size,
                color=self.ARROW_COLOR,
                name=f"target_pocket_arrow_{name}",
            )
            self._target_arrow_parts.append(arrow)

    def _spawn_balls(self):
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()

        # Random left/right side for primary ball (arm must match that side).
        self._arm_side = "left" if np.random.rand() < 0.5 else "right"
        # Keep primary clearly on one half for reachability.
        if self._arm_side == "right":
            px = float(np.random.uniform(max(0.04, x_lo), min(0.16, x_hi)))
        else:
            px = float(np.random.uniform(max(x_lo, -0.16), min(-0.04, x_hi)))
        py = float(np.clip(self.table_cy + np.random.uniform(-0.05, 0.05), y_lo, y_hi))
        self.primary_ball = self._make_ball(
            [px, py, self.ball_z], self.PRIMARY_COLOR, "primary_ball"
        )
        self._primary_rigid = self._get_rigid_entity(self.primary_ball)

        # Goal pocket before blockers so Opt1+2 knows which hole to block / leave open.
        self._select_goal_pocket()
        if self.specific_hole and self._target_pocket_id is not None:
            self._draw_target_arrow(self._target_pocket_id)

        self.extra_balls = []
        self._extra_rigids = []
        occupied = [np.array([px, py], dtype=np.float64)]
        if not self.enable_distractors or self.num_extra_balls <= 0:
            return

        n = self.num_extra_balls
        if self.blocker_mode in ("block_one", "unused_tops") or (
            self.specific_hole and self.enable_distractors
        ):
            # Opt 1+2: occupy one non-target top pocket. Never sit in the
            # nominated hole or on the shot from the primary into it.
            positions = self._place_opt12_distractors(n, occupied)
            for i, pos in enumerate(positions):
                color = self.EXTRA_COLORS[i % len(self.EXTRA_COLORS)]
                ball = self._make_ball(
                    [pos[0], pos[1], self.ball_z], color, f"extra_ball_{i}"
                )
                self.extra_balls.append(ball)
                self._extra_rigids.append(self._get_rigid_entity(ball))
                occupied.append(pos)
        else:
            # Opt 2 only: place up to 2 balls between primary and a random top hole.
            top_ids = list(self.TOP_POCKET_IDS)
            np.random.shuffle(top_ids)
            path_pocket = self._pocket_centers[top_ids[0]]
            for i in range(n):
                pos = self._blocker_on_path(
                    occupied[0], path_pocket[:2], occupied, frac=0.35 + 0.2 * i
                )
                if pos is None:
                    pos = self._blocker_near_pocket(top_ids[0], occupied)
                if pos is None:
                    continue
                color = self.EXTRA_COLORS[i % len(self.EXTRA_COLORS)]
                ball = self._make_ball(
                    [pos[0], pos[1], self.ball_z], color, f"extra_ball_{i}"
                )
                self.extra_balls.append(ball)
                self._extra_rigids.append(self._get_rigid_entity(ball))
                occupied.append(pos)

    def _target_shot_clearance(self):
        """Center-to-segment gap that keeps a distractor off the nominated shot."""
        return 3.0 * self.ball_radius

    def _target_mouth_clearance(self):
        """Keep distractors out of the nominated pocket mouth."""
        return float(self.pocket_radius + 4.0 * self.ball_radius)

    def _obstructs_target_hole(self, xy, start_xy=None):
        """True if ``xy`` sits in the target mouth or on the primary→target shot."""
        if self._target_pocket_id is None:
            return False
        pxy = np.asarray(self._pocket_centers[self._target_pocket_id][:2], dtype=np.float64)
        cand = np.asarray(xy, dtype=np.float64)
        if float(np.linalg.norm(cand - pxy)) < self._target_mouth_clearance():
            return True
        if start_xy is None:
            if self.primary_ball is None:
                return False
            start_xy = self._ball_xy(self.primary_ball)
        return not self._line_clear(
            np.asarray(start_xy, dtype=np.float64),
            pxy,
            [cand],
            self._target_shot_clearance(),
        )

    def _place_opt12_distractors(self, n, occupied):
        """Park extras at one non-target top pocket with the selected hole open.

        Prefer the farthest other top pocket (opposite corner when the target
        is a corner) so the middle pocket does not sit on a cross-table shot.
        """
        occupied = [np.asarray(o, dtype=np.float64) for o in occupied]
        target = self._target_pocket_id
        if target is None:
            candidates = list(self.TOP_POCKET_IDS)
        else:
            candidates = [i for i in self.TOP_POCKET_IDS if i != int(target)]
        txy = (
            np.asarray(self._pocket_centers[int(target)][:2], dtype=np.float64)
            if target is not None else np.array([self.table_cx, self.table_cy])
        )
        scored = []
        for i in candidates:
            dist = float(np.linalg.norm(self._pocket_centers[i][:2] - txy))
            scored.append((-dist, float(np.random.random()), i))
        scored.sort()
        candidates = [i for _, _, i in scored]

        best = []
        best_id = None
        for blocked in candidates:
            trial = []
            occ = list(occupied)
            for _ in range(n):
                pos = self._blocker_near_pocket(
                    blocked, occ, keep_target_clear=True
                )
                if pos is None:
                    break
                trial.append(pos)
                occ.append(pos)
            if len(trial) > len(best):
                best = trial
                best_id = blocked
            if len(trial) >= n:
                break

        self._blocked_pocket_id = best_id
        return best

    def _blocker_near_pocket(self, pocket_id, occupied, min_sep=None, keep_target_clear=False):
        """Place a blocker just inboard of a pocket mouth."""
        if min_sep is None:
            min_sep = 3.0 * self.ball_radius
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()
        pxy = self._pocket_centers[pocket_id][:2]
        center = np.array([self.table_cx, self.table_cy], dtype=np.float64)
        inward = center - pxy
        n = float(np.linalg.norm(inward))
        if n < 1e-6:
            return None
        inward /= n
        for dist in (0.045, 0.055, 0.035, 0.065, 0.080):
            for jitter in range(16):
                lat = np.array([-inward[1], inward[0]]) * np.random.uniform(-0.02, 0.02)
                cand = pxy + inward * dist + lat
                cand[0] = float(np.clip(cand[0], x_lo, x_hi))
                cand[1] = float(np.clip(cand[1], y_lo, y_hi))
                if any(np.linalg.norm(cand - o) < min_sep for o in occupied):
                    continue
                if keep_target_clear and self._obstructs_target_hole(cand):
                    continue
                return cand
        return None

    def _blocker_on_path(self, start_xy, end_xy, occupied, frac=0.45, min_sep=None):
        if min_sep is None:
            min_sep = 3.0 * self.ball_radius
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()
        d = end_xy - start_xy
        if float(np.linalg.norm(d)) < 1e-6:
            return None
        for _ in range(20):
            f = float(np.clip(frac + np.random.uniform(-0.08, 0.08), 0.25, 0.75))
            perp = np.array([-d[1], d[0]], dtype=np.float64)
            pn = float(np.linalg.norm(perp))
            if pn > 1e-6:
                perp = perp / pn * np.random.uniform(-0.015, 0.015)
            else:
                perp = 0.0
            cand = start_xy + d * f + perp
            cand[0] = float(np.clip(cand[0], x_lo, x_hi))
            cand[1] = float(np.clip(cand[1], y_lo, y_hi))
            if all(np.linalg.norm(cand - o) >= min_sep for o in occupied):
                return cand
        return None

    def _make_ball(self, pos, color, name):
        ball = create_sphere(
            self.scene,
            pose=sapien.Pose(pos, [1, 0, 0, 0]),
            radius=self.ball_radius,
            color=tuple(color),
            is_static=False,
            name=name,
        )
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                # Keep full dynamic simulation (distractors must roll/bounce too).
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                mass = float(self.BALL_MASS)
                rigid.set_mass(mass)
                # set_mass disables auto inertia; use solid-sphere I = 2/5 m r^2.
                inertia = 0.4 * mass * (self.ball_radius ** 2)
                rigid.set_inertia([inertia, inertia, inertia])
                rigid.set_linear_damping(self.BALL_LINEAR_DAMPING)
                rigid.set_angular_damping(self.BALL_ANGULAR_DAMPING)
                rigid.set_sleep_threshold(self.BALL_SLEEP_THRESHOLD)
                try:
                    rigid.set_solver_position_iterations(8)
                    rigid.set_solver_velocity_iterations(4)
                except Exception:
                    pass
                try:
                    if hasattr(rigid, "set_enable_ccd"):
                        rigid.set_enable_ccd(True)
                except Exception:
                    pass
            except Exception:
                pass
            # Bouncy, low-friction surfaces so ball–ball impacts transfer momentum.
            try:
                mat = sapien.physx.PhysxMaterial(
                    static_friction=self.BALL_FRICTION,
                    dynamic_friction=self.BALL_FRICTION,
                    restitution=self.BALL_RESTITUTION,
                )
                for shape in rigid.get_collision_shapes():
                    shape.set_physical_material(mat)
            except Exception:
                pass
            bits = self._CUE_SHAFT_BALL_IGNORE_BITS
            i = int(getattr(self, "_ball_ignore_bit_i", 0))
            bit = bits[min(i, len(bits) - 1)]
            self._ball_ignore_bit_i = i + 1
            self._tag_ball_ignore_shaft(rigid, bit)
            try:
                rigid.wake_up()
            except Exception:
                pass
        return ball

    def _iter_live_ball_rigids(self, include_kinematic=False):
        """Yield (entity, rigid) for non-pocketed balls still on the felt."""
        if self.primary_ball is not None and self._primary_rigid is not None:
            if not self._primary_pocketed:
                yield self.primary_ball, self._primary_rigid
        pocketed = getattr(self, "_pocketed_extra_entities", set())
        for ball, rigid in zip(self.extra_balls, getattr(self, "_extra_rigids", [])):
            if ball is None or rigid is None:
                continue
            if ball in pocketed:
                continue
            if not include_kinematic:
                try:
                    if rigid.get_kinematic():
                        continue
                except Exception:
                    pass
            yield ball, rigid

    def _wake_all_balls(self):
        """Wake resting distractors so a moving primary can transfer momentum."""
        for _, rigid in self._iter_live_ball_rigids(include_kinematic=True):
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.wake_up()
            except Exception:
                pass

    def _ensure_balls_dynamic(self):
        """Keep non-pocketed balls fully dynamic so hits knock them around."""
        for _, rigid in self._iter_live_ball_rigids(include_kinematic=True):
            try:
                if rigid.get_kinematic():
                    rigid.set_kinematic(False)
                if rigid.get_disable_gravity():
                    rigid.set_disable_gravity(False)
            except Exception:
                pass

    # ------------------------------------------------------------- cue
    def _spawn_cue_stand(self, cue_x, cue_y):
        """Short wood cradle so the WSG can top-down grasp without hitting the table."""
        stand_h = float(self.CUE_STAND_HEIGHT)
        half_h = 0.5 * stand_h
        z = float(self.z0 + half_h)
        wood = (0.42, 0.28, 0.16)
        # Two posts under the shaft (along +Y); mid-span clear for handle grasp.
        for i, dy in enumerate((-0.052, 0.052)):
            create_box(
                self,
                pose=sapien.Pose([float(cue_x), float(cue_y + dy), z], [1, 0, 0, 0]),
                half_size=[0.014, 0.010, half_h],
                color=wood,
                is_static=True,
                name=f"cue_stand_{i}",
            )

    def _spawn_cue(self):
        """Lie the cue on a side stand, shaft along +Y (same side as arm)."""
        side_clear = 0.09
        if self._arm_side == "right":
            cue_x = float(self.table_cx + self.table_half_len + side_clear)
        else:
            cue_x = float(self.table_cx - self.table_half_len - side_clear)
        cue_y = float(self.table_cy + np.random.uniform(-0.04, 0.04))
        self._spawn_cue_stand(cue_x, cue_y)
        # Resting on the stand top (reachable by WSG finger pads).
        cue_z = float(self.z0 + self.CUE_STAND_HEIGHT + self.CUE_RADIUS)
        # Local +X (tip) → world +Y so the shaft runs along the table short side.
        along_y_q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], np.pi / 2).tolist()
        pose = sapien.Pose([cue_x, cue_y, cue_z], along_y_q)
        self.cue = self._build_cue(pose)
        self.cue.set_mass(self.CUE_MASS)
        self._cue_rigid = self._get_rigid(self.cue)
        self._cache_cue_collision_shapes()
        self._apply_cue_collision_groups(
            ignore_robot=False, tip_hits_balls=True
        )
        self.add_prohibit_area(self.cue, padding=0.04)

    def _build_cue(self, pose):
        """Round cue rod: cylinder shaft with spherical tip and butt.

        Shaft collision is shortened so only the blue tip sphere can touch balls;
        the cylinder + butt still collide with the table / cradle.
        """
        builder = self.scene.create_actor_builder()
        r = self.CUE_RADIUS
        half = self.CUE_HALF_LEN
        mat = self.scene.default_physical_material
        # Stop the shaft capsule before the tip sphere so shaft/ball ignore is
        # meaningful near the tip (otherwise the cylinder end still pokes balls).
        shaft_half = max(1e-4, float(half) - float(r))
        builder.add_cylinder_collision(radius=r, half_length=shaft_half, material=mat)
        builder.add_sphere_collision(pose=sapien.Pose([half, 0, 0]), radius=r, material=mat)
        builder.add_sphere_collision(pose=sapien.Pose([-half, 0, 0]), radius=r, material=mat)

        shaft_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_COLOR, 1.0])
        tip_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_TIP_COLOR, 1.0])
        tip_mat.roughness = 0.45
        tip_mat.metallic = 0.0
        tip_mat.emission = [0.02, 0.08, 0.22, 1.0]  # slight glow; keep tip diameter readable
        butt_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_COLOR, 1.0])
        # Tip sphere radius matches shaft (CUE_RADIUS) — flush diameter with the rod.
        builder.add_cylinder_visual(radius=r, half_length=half, material=shaft_mat)
        builder.add_sphere_visual(pose=sapien.Pose([half, 0, 0]), radius=r, material=tip_mat)
        builder.add_sphere_visual(pose=sapien.Pose([-half, 0, 0]), radius=r, material=butt_mat)
        builder.set_initial_pose(pose)
        entity = builder.build(name="cue_stick")

        tip_x = half + r  # apex of the tip sphere
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * tip_x, 2 * r, 2 * r],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            # Top-down frames at shaft center (local +X = tip). Z overridden in pick.
            "contact_points_pose": [
                [
                    [1, 0, 0, 0.0],
                    [0, 0, -1, 0.0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0.0],
                    [0, 0, -1, 0.0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
            ],
            "functional_matrix": [
                [
                    [1, 0, 0, tip_x],
                    [0, 1, 0, 0.0],
                    [0, 0, 1, 0.0],
                    [0, 0, 0, 1],
                ],
            ],
        }
        return Actor(entity, data, mass=self.CUE_MASS)

    # ------------------------------------------------------------- helpers
    def _get_rigid(self, actor):
        ent = actor.actor if isinstance(actor, Actor) else actor
        for c in ent.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _get_rigid_entity(self, entity):
        for c in entity.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _collect_robot_link_names(self):
        names = set()
        for articulation in (self.robot.left_entity, self.robot.right_entity):
            if articulation is None:
                continue
            for link in articulation.get_links():
                names.add(link.get_name())
        return names

    def _pose_to_mat(self, pose7):
        T = np.eye(4)
        T[:3, :3] = t3d.quaternions.quat2mat(np.asarray(pose7[3:], dtype=np.float64))
        T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
        return T

    def _mat_to_pose(self, T):
        q = t3d.quaternions.mat2quat(T[:3, :3])
        return sapien.Pose(T[:3, 3].tolist(), q.tolist())

    def _ball_xy(self, ball):
        return np.asarray(ball.get_pose().p[:2], dtype=np.float64)

    def _line_clear(self, start_xy, end_xy, blockers, clearance):
        """True if no blocker center is within `clearance` of the segment start→end."""
        d = end_xy - start_xy
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return True
        u = d / length
        for b in blockers:
            v = b - start_xy
            t = float(np.clip(np.dot(v, u), 0.0, length))
            closest = start_xy + t * u
            if np.linalg.norm(b - closest) < clearance:
                return False
        return True

    def _behind_in_workspace(self, behind):
        """Approach point must stay on the active arm's half of the table."""
        if behind[1] < -0.14 or behind[1] > 0.26:
            return False
        if self._arm_side == "right":
            return 0.02 <= behind[0] <= 0.32
        return -0.32 <= behind[0] <= -0.02

    def _choose_pocket(self):
        """Pick an aim pocket for the expert (respects goal mode + clear path)."""
        ball_xy = self._ball_xy(self.primary_ball)
        blockers = [self._ball_xy(b) for b in self.extra_balls]
        clearance = 2.2 * self.ball_radius

        # Prefer the episode goal when Opt 1 (specific hole); else any allowed.
        if self.specific_hole and self._target_pocket_id is not None:
            aim_ids = [self._target_pocket_id]
        else:
            aim_ids = list(self._allowed_pocket_ids)

        def collect(ids, require_plus_y=True):
            out = []
            for pid in ids:
                pocket = self._pocket_centers[pid]
                pxy = pocket[:2]
                aim = pxy - ball_xy
                dist = float(np.linalg.norm(aim))
                if dist < self.pocket_radius + self.ball_radius:
                    continue
                behind = ball_xy - (aim / dist) * self.APPROACH_GAP
                if not self._behind_in_workspace(behind):
                    continue
                if require_plus_y and aim[1] < -0.005:
                    continue
                clear = self._line_clear(ball_xy, pxy, blockers, clearance)
                score = dist + (0.0 if clear else 0.55)
                score -= 0.20 * max(0.0, float(aim[1]))
                # Prefer the three far (+Y) pockets — more reliable expert sinks.
                if pid in self.TOP_POCKET_IDS:
                    score -= 0.08
                out.append((score, clear, pocket, pid))
            return out

        candidates = collect(aim_ids, require_plus_y=True)
        if not candidates:
            candidates = collect(aim_ids, require_plus_y=False)
        if not candidates and not self.specific_hole:
            candidates = collect(list(range(6)), require_plus_y=False)
        if not candidates:
            # Absolute fallback: farthest +Y pocket on this side.
            pid = max(
                (1, 5) if self._arm_side == "left" else (3, 5),
                key=lambda i: float(self._pocket_centers[i][1]),
            )
            return self._pocket_centers[pid], pid
        clear_ones = [c for c in candidates if c[1]]
        pool = clear_ones if clear_ones else candidates
        pool.sort(key=lambda x: x[0])
        return pool[0][2], pool[0][3]

    # ---------------------------------------------------- weld / kinematics
    def _cache_cue_collision_shapes(self):
        """Split cue collision shapes into blue tip vs shaft/butt."""
        self._cue_tip_shapes = []
        self._cue_shaft_shapes = []
        rigid = self._get_rigid(self.cue)
        if rigid is None:
            return
        tip_x = float(self.CUE_HALF_LEN)
        try:
            for shape in rigid.get_collision_shapes():
                x = float(shape.get_local_pose().p[0])
                if abs(x - tip_x) < 1e-3:
                    self._cue_tip_shapes.append(shape)
                else:
                    self._cue_shaft_shapes.append(shape)
        except Exception:
            pass

    def _tag_ball_ignore_shaft(self, rigid, bit=None):
        """Each ball ignores cue shaft/butt (and a spent tip) via its own g2 bit."""
        if rigid is None:
            return
        if bit is None:
            bit = int(self._CUE_SHAFT_BALL_IGNORE_BITS[0])
        iid = int(self._CUE_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for shape in rigid.get_collision_shapes():
                g0, g1, g2, g3 = shape.get_collision_groups()
                shape.set_collision_groups([
                    int(g0),
                    int(g1),
                    int(g2) | int(bit),
                    (int(g3) & ~0xFFFF) | iid,
                ])
        except Exception:
            pass

    def _apply_cue_collision_groups(self, *, ignore_robot: bool, tip_hits_balls: bool):
        """Tip may hit balls once; shaft/butt never do. Optional cue↔robot ignore."""
        if not self._cue_tip_shapes and not self._cue_shaft_shapes:
            self._cache_cue_collision_shapes()
        robot_bit = int(self._CUE_ROBOT_IGNORE_BIT) if ignore_robot else 0
        shaft_ball_bit = int(self._CUE_SHAFT_BALL_IGNORE_MASK)
        iid = int(self._CUE_ROBOT_IGNORE_ID) & 0xFFFF
        tip_g2 = robot_bit if tip_hits_balls else (robot_bit | shaft_ball_bit)
        shaft_g2 = robot_bit | shaft_ball_bit
        # g3 must match balls whenever shaft_ball_bit is set on either side.
        tip_g3 = iid if tip_g2 else 0
        shaft_g3 = iid
        try:
            for shape in self._cue_tip_shapes:
                shape.set_collision_groups([1, 1, int(tip_g2), int(tip_g3)])
            for shape in self._cue_shaft_shapes:
                shape.set_collision_groups([1, 1, int(shaft_g2), int(shaft_g3)])
        except Exception:
            pass

    def _disable_cue_tip_ball_collision(self):
        """After the first tip hit, never contact balls again with this cue."""
        self._cue_tip_hit_allowed = False
        self._apply_cue_collision_groups(
            ignore_robot=bool(self._cue_welded),
            tip_hits_balls=False,
        )

    def _disable_cue_robot_collision(self):
        """Ignore cue↔robot collisions after weld; keep tip↔ball until first hit.

        Previously set cue groups to [0,0,0,0], which disabled *all* cue
        collisions (including the ball), so strikes never registered real
        contact and the impulse had to fire on tip proximity alone.
        """
        self._apply_cue_collision_groups(
            ignore_robot=True,
            tip_hits_balls=bool(self._cue_tip_hit_allowed),
        )
        ignore_bit = int(self._CUE_ROBOT_IGNORE_BIT)
        ignore_id = int(self._CUE_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    for shape in link.get_collision_shapes():
                        g0, g1, g2, g3 = shape.get_collision_groups()
                        shape.set_collision_groups([
                            int(g0),
                            int(g1),
                            int(g2) | ignore_bit,
                            (int(g3) & ~0xFFFF) | ignore_id,
                        ])
        except Exception:
            pass

    def _weld_cue_to_ee(self, arm_tag, lift_z=None):
        """Lock cue to EE using the post-grasp relative pose (no teleport seat).

        Records the EE↔cue transform after the fingers have closed on the shaft.
        Optional ``lift_z`` is kept for debugging only; production uses 0 so the
        stick does not jump into the gripper.
        """
        if lift_z is None:
            lift_z = float(self.CUE_WELD_LIFT_Z)
        ee = np.asarray(self.get_arm_pose(str(arm_tag)), dtype=np.float64)
        ee_T = self._pose_to_mat(ee)
        cue_T = self.cue.get_pose().to_transformation_matrix()
        self._cue_ee_T = np.linalg.inv(ee_T) @ cue_T
        if abs(lift_z) > 1e-6:
            lift_world = np.array([0.0, 0.0, float(lift_z)], dtype=np.float64)
            self._cue_ee_T[:3, 3] += ee_T[:3, :3].T @ lift_world
        self._cue_arm = str(arm_tag)
        seated = self._clamp_cue_pose_above_felt(self._mat_to_pose(ee_T @ self._cue_ee_T))
        self.cue.actor.set_pose(seated)
        rigid = self._get_rigid(self.cue)
        if rigid is not None:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(seated)
        self._disable_cue_robot_collision()
        self._cue_welded = True
        # While welded, gripper pads can ghost through the cue; do not fail on
        # those contacts until after the tip has spent its one hit.
        self._strike_armed = True

    def _finger_pad_z(self, arm):
        """Mean world-Z of the WSG finger links (approx. pad height)."""
        ent = self.robot.left_entity if str(arm) == "left" else self.robot.right_entity
        zs = [
            float(link.entity_pose.p[2])
            for link in ent.get_links()
            if "finger" in link.get_name()
        ]
        return float(np.mean(zs)) if zs else None

    def _pick_up_cue(self, arm):
        """Open → descend onto the cue handle → close → weld."""
        cue_p = np.asarray(self.cue.get_pose().p, dtype=np.float64)
        _pre, grasp = self.choose_grasp_pose(
            self.cue,
            arm_tag=arm,
            pre_dis=float(self.CUE_PRE_GRASP_DIS),
            target_dis=0.0,
            contact_point_id=0,
        )
        if grasp is None:
            return False
        # ``choose_grasp_pose`` provides a valid top-down orientation.  Position
        # that grasp at the butt/handle instead of the cue centre so the long
        # tip section remains free for manual aiming and striking.
        cue_R = t3d.quaternions.quat2mat(np.asarray(self.cue.get_pose().q, dtype=np.float64))
        handle_local = np.array([
            -self.CUE_HALF_LEN + self.CUE_HANDLE_GRASP_FROM_BUTT,
            0.0,
            0.0,
        ])
        handle_p = cue_p + cue_R @ handle_local
        grasp_z = float(cue_p[2] + self.CUE_FINGER_EE_Z)
        pre_z = float(grasp_z + self.CUE_PRE_GRASP_DIS)
        quat = list(grasp[3:7])
        pre_pose = [float(handle_p[0]), float(handle_p[1]), pre_z, *quat]
        grasp_pose = [float(handle_p[0]), float(handle_p[1]), grasp_z, *quat]

        self._disable_cue_robot_collision()

        self.move(self.open_gripper(arm, pos=float(self.CUE_GRASP_OPEN)))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm, pre_pose))
        if not self.plan_success:
            return False

        self.move(self.move_to_pose(arm, grasp_pose))
        if not self.plan_success:
            return False

        # Keep pad links ~3 cm above the shaft (fingertip pinch).
        target_pad_z = float(cue_p[2] + self.CUE_TIP_GRASP_CLEARANCE)
        pad_z = self._finger_pad_z(arm)
        if pad_z is not None:
            for _ in range(4):
                err = float(target_pad_z - pad_z)
                if abs(err) < 0.008:
                    break
                step = float(np.clip(err, -0.025, 0.025))
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, z=step, move_axis="world"
                    )
                )
                if not self.plan_success:
                    self.plan_success = True
                    break
                pad_z = self._finger_pad_z(arm)
                if pad_z is None:
                    break

        self.move(self.close_gripper(arm, pos=float(self.CUE_GRASP_CLOSE)))
        if not self.plan_success:
            return False

        self._dwell(20)
        self._weld_cue_to_ee(arm, lift_z=0.0)
        return True

    def _seat_cue_for_strike(self, arm):
        """Deepen the post-grasp weld so tip↔felt strikes stay reachable."""
        hang = float(self.CUE_STRIKE_HANG)
        if hang < 1e-4 or not self._cue_welded:
            return
        self._cue_ee_T = np.array(self._cue_ee_T, dtype=np.float64, copy=True)
        self._cue_ee_T[0, 3] += hang
        self._update_welded_cue()
        self._dwell(8)

    def _contact_is_touching(self, contact):
        """True if a PhysX contact pair has near-zero separation or impulse."""
        points = getattr(contact, "points", None) or []
        if not points:
            return True
        for pt in points:
            sep = float(getattr(pt, "separation", 0.0))
            impulse = np.asarray(
                getattr(pt, "impulse", [0, 0, 0]), dtype=np.float64
            )
            if sep <= 1e-3 or float(np.linalg.norm(impulse)) > 1e-8:
                return True
        return False

    def _contact_pair_names(self, contact):
        """Actor names for a PhysX contact pair; None if a body was removed."""
        names = []
        bodies = getattr(contact, "bodies", None) or ()
        for i in range(2):
            body = bodies[i] if i < len(bodies) else None
            ent = getattr(body, "entity", None) if body is not None else None
            names.append(None if ent is None else getattr(ent, "name", None))
        return names[0], names[1]

    def _cue_ball_contacting(self):
        """True only when PhysX reports touching tip↔primary-ball contact.

        Shaft/butt shapes ignore balls via collision groups, so any cue↔ball
        contact here is from the blue tip sphere. Accepts a contact pair if any
        point is penetrating/near-touching (separation <= 1 mm) or carries a
        non-zero impulse.
        """
        if self.cue is None or self.primary_ball is None:
            return False
        if not getattr(self, "_cue_tip_hit_allowed", True):
            return False
        cue_name = self.cue.get_name()
        ball_name = self.primary_ball.get_name()
        try:
            for contact in self.scene.get_contacts():
                n0, n1 = self._contact_pair_names(contact)
                if n0 is None or n1 is None:
                    continue
                if not (
                    (n0 == cue_name and n1 == ball_name)
                    or (n1 == cue_name and n0 == ball_name)
                ):
                    continue
                if self._contact_is_touching(contact):
                    return True
        except Exception:
            pass
        return False

    def _check_cue_distractor_contact(self):
        """Fail the episode if the cue touches any non-target ball."""
        if (
            getattr(self, "_cue_distractor_contact", False)
            or not self._loaded
            or self.cue is None
            or not self.extra_balls
        ):
            return
        cue_name = self.cue.get_name()
        extra_names = {
            b.get_name() for b in self.extra_balls if b is not None
        }
        if not extra_names:
            return
        try:
            for contact in self.scene.get_contacts():
                n0, n1 = self._contact_pair_names(contact)
                if n0 is None or n1 is None:
                    continue
                if not (
                    (n0 == cue_name and n1 in extra_names)
                    or (n1 == cue_name and n0 in extra_names)
                ):
                    continue
                if self._contact_is_touching(contact):
                    self._cue_distractor_contact = True
                    return
        except Exception:
            pass

    def _cue_min_z(self, pose):
        """Lowest world-Z of the cue capsule (shaft + tip/butt spheres)."""
        p = np.asarray(pose.p, dtype=np.float64)
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        axis_z = float(R[2, 0])  # local +X tip axis, world Z component
        half = float(self.CUE_HALF_LEN)
        r = float(self.CUE_RADIUS)
        # Segment extent along Z, then radial extent of the capsule.
        seg_min = float(p[2]) - half * abs(axis_z)
        radial = r * float(np.sqrt(max(0.0, 1.0 - axis_z * axis_z)))
        return seg_min - radial

    def _felt_contact_z(self):
        """World-Z floor for the cue capsule resting on the playing surface."""
        return float(self.felt_top) + 5e-4

    def _clamp_cue_pose_above_felt(self, pose):
        """Raise ``pose`` so the cue capsule cannot enter the hollow table.

        Welded cues are kinematic and pose-driven, so PhysX will not stop them
        from tunneling a thin lid — this keeps the stick on the playing surface.
        """
        floor = self._felt_contact_z()
        min_z = self._cue_min_z(pose)
        if min_z >= floor:
            return pose
        p = np.asarray(pose.p, dtype=np.float64).copy()
        p[2] += floor - min_z
        return sapien.Pose(p.tolist(), list(pose.q))

    def cue_resting_on_felt(self, tol=0.003):
        """True when the welded cue capsule is already on the playing surface."""
        if not getattr(self, "_cue_welded", False) or self.cue is None:
            return False
        try:
            min_z = self._cue_min_z(self.cue.get_pose())
        except Exception:
            return False
        return bool(min_z <= self._felt_contact_z() + float(tol))

    def interactive_support_z(self, side, ee_pose):
        """Workspace table top — fingertips must stay at or above this."""
        return float(getattr(self, "table_top", getattr(self, "z0", 0.74)))

    def _finger_extent_below_ee(self, side, ee_z) -> float:
        """How far below the EE the lowest finger geometry sits (meters)."""
        ent = (
            self.robot.left_entity if str(side) == "left" else self.robot.right_entity
        )
        lo = None
        if ent is not None:
            for link in ent.get_links():
                name = str(link.get_name() or "").lower()
                if "finger" not in name and "gripper" not in name:
                    continue
                try:
                    aabb = link.compute_global_aabb_tight()
                    z = float(aabb[0][2])
                except Exception:
                    try:
                        z = float(link.entity_pose.p[2])
                    except Exception:
                        continue
                lo = z if lo is None else min(lo, z)
        if lo is None:
            return 0.18
        return max(float(ee_z) - float(lo), 0.05)

    def interactive_ee_z_floor(self, side, ee_pose):
        """Lowest allowed EE Z so fingertips stay at/above the workspace table.

        Also raises the floor further when a welded cue would enter the felt.
        """
        ee = np.asarray(ee_pose, dtype=np.float64).reshape(-1)
        if ee.size < 3:
            return None
        table_z = float(getattr(self, "table_top", getattr(self, "z0", 0.74)))
        finger_below = self._finger_extent_below_ee(side, float(ee[2]))
        # +4 cm vs fingertip-at-table so the gripper clears the workspace surface.
        floor = table_z + finger_below + 0.008 + 0.04

        # Welded-cue felt contact (if any) can only raise this floor.
        if (
            getattr(self, "_cue_welded", False)
            and str(side) == str(getattr(self, "_cue_arm", ""))
            and getattr(self, "_cue_ee_T", None) is not None
            and ee.size >= 7
        ):
            felt = self._felt_contact_z()
            cue_pose = self._mat_to_pose(self._pose_to_mat(ee) @ self._cue_ee_T)
            min_z = self._cue_min_z(cue_pose)
            if min_z < felt - 1e-6:
                floor = max(floor, float(ee[2]) + (felt - min_z))
        return float(floor)

    def _update_welded_cue(self):
        if not self._cue_welded:
            return
        ee = np.asarray(self.get_arm_pose(self._cue_arm), dtype=np.float64)
        cue_T = self._pose_to_mat(ee) @ self._cue_ee_T
        pose = self._clamp_cue_pose_above_felt(self._mat_to_pose(cue_T))
        self.cue.actor.set_pose(pose)
        rigid = self._get_rigid(self.cue)
        if rigid is not None:
            rigid.set_kinematic_target(pose)

    def _check_robot_ball_contact(self):
        if self._robot_ball_contact or not self._loaded:
            return
        if self.primary_ball is None:
            return
        # Keyboard mode strips the arms; leftover contacts have no entity.
        if getattr(self, "_interactive_robot_mode", True) is False:
            return
        # While the cue is welded (or after the tip has spent its hit), WSG pads
        # can ghost through the stick and spuriously touch the ball — ignore.
        if self._cue_welded or self._strike_armed or self._strike_done:
            return
        ball_name = self.primary_ball.get_name()
        try:
            for contact in self.scene.get_contacts():
                name0, name1 = self._contact_pair_names(contact)
                if name0 is None or name1 is None:
                    continue
                if (
                    (name0 == ball_name and name1 in self._robot_link_names)
                    or (name1 == ball_name and name0 in self._robot_link_names)
                ):
                    self._robot_ball_contact = True
                    return
        except Exception:
            pass

    def _stick_tip_dir_xy(self):
        """Unit XY direction the cue tip is pointing (stick orientation, not pocket aim)."""
        tip = self._tip_xyz()
        cue_p = np.asarray(self.cue.get_pose().p, dtype=np.float64)
        d = tip[:2] - cue_p[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-4:
            return np.array(self._aim_dir, dtype=np.float64)
        return d / n

    def _try_apply_strike_impulse(self):
        """One-shot tip hit: impart velocity on first blue-tip↔ball contact.

        Shaft/butt never collide with balls. After this fires, tip↔ball collision
        is disabled so the stick cannot push or re-hit. Used by the expert push
        and by interactive/manual tip contact (no Space strike).
        """
        if (
            self._strike_done
            or self._primary_pocketed
            or not getattr(self, "_cue_tip_hit_allowed", True)
        ):
            return
        # Keyboard Space applies the hit; do not auto-strike from tip proximity.
        if getattr(self, "_interactive_robot_mode", True) is False:
            return
        tip = self._tip_xyz()
        ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
        dist = float(np.linalg.norm(tip - ball_p))
        # Apex-to-center envelope for a tip-sphere touch (see STRIKE_CONTACT_GAP).
        contact_thresh = (
            self.ball_radius + 2.0 * self.CUE_RADIUS + self.STRIKE_CONTACT_GAP
        )
        if dist > contact_thresh:
            return
        # Hard gate: PhysX must report actual touching tip contact (no proximity kick).
        if not self._cue_ball_contacting():
            return
        rigid = self._primary_rigid
        if rigid is None:
            return
        # Prefer stick pointing dir; fall back to aim if the tip frame is near-degenerate.
        stick = self._stick_tip_dir_xy()
        if float(np.dot(stick, self._aim_dir)) < 0.25:
            stick = np.array(self._aim_dir, dtype=np.float64)
        direction = np.array([stick[0], stick[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            return
        direction /= n
        self._latch_strike_metrics(direction)
        try:
            rigid.set_linear_velocity(direction * self.strike_impulse)
            rigid.set_angular_velocity(np.zeros(3))
            rigid.wake_up()
        except Exception:
            pass
        # Distractors start asleep on the felt; wake them so the impact can bounce.
        self._wake_all_balls()
        self._strike_done = True
        self._strike_armed = False
        self._disable_cue_tip_ball_collision()
        if os.environ.get("PLAY_BILLIARD_DEBUG"):
            print(
                f"[PLAY_BILLIARD] strike_impulse: tip_dist={dist:.4f} "
                f"dir={np.round(direction, 3).tolist()} "
                f"impulse={self.strike_impulse}",
                flush=True,
            )

    def _ball_inside_hollow(self, p):
        """True once the ball has fallen through a pocket into the hollow box."""
        cx, cy = self.table_cx, self.table_cy
        hl = self.table_half_len - self.WALL_T
        hw = self.table_half_wid - self.WALL_T
        in_xy = abs(float(p[0]) - cx) <= hl and abs(float(p[1]) - cy) <= hw
        # Below the lid underside ⇒ inside the cavity.
        return in_xy and float(p[2]) < self.base_top - 0.008

    def _nearest_pocket_id(self, xy):
        best_i, best_d = None, 1e9
        for i, pocket in enumerate(self._pocket_centers):
            d = float(np.linalg.norm(xy - pocket[:2]))
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def _mark_distractor_pocketed(self, ball):
        """Any distractor in a pocket fails the episode."""
        self._distractor_pocketed = True
        try:
            idx = self.extra_balls.index(ball)
            self._pocketed_extra_ids.add(idx)
        except ValueError:
            pass

    def _check_and_sink_pockets(self):
        # Always scan primary + distractors (do not stop after primary sinks).
        balls = [(self.primary_ball, True)] + [(b, False) for b in self.extra_balls]
        for ball, is_primary in balls:
            if ball is None:
                continue
            if (not is_primary) and ball in getattr(self, "_pocketed_extra_entities", set()):
                continue
            p = np.asarray(ball.get_pose().p, dtype=np.float64)
            # Natural fall through a real pocket opening into the hollow interior.
            if self._ball_inside_hollow(p):
                if is_primary:
                    if not self._primary_pocketed:
                        pid, _ = self._nearest_pocket_id(p[:2])
                        self._primary_pocket_id = pid
                        self._primary_pocketed = True
                else:
                    self._mark_distractor_pocketed(ball)
                    if not hasattr(self, "_pocketed_extra_entities"):
                        self._pocketed_extra_entities = set()
                    self._pocketed_extra_entities.add(ball)
                self._rest_ball_on_floor(ball, p)
                continue
            for i, pocket in enumerate(self._pocket_centers):
                if np.linalg.norm(p[:2] - pocket[:2]) <= self.pocket_radius:
                    # Nudge through the opening so physics can finish the drop.
                    if float(p[2]) > self.base_top - 0.002:
                        self._drop_ball_through_pocket(ball, pocket)
                    inside = self._ball_inside_hollow(
                        np.asarray(ball.get_pose().p, dtype=np.float64)
                    )
                    if inside:
                        if is_primary and not self._primary_pocketed:
                            self._primary_pocket_id = i
                            self._primary_pocketed = True
                        elif not is_primary:
                            self._mark_distractor_pocketed(ball)
                            if not hasattr(self, "_pocketed_extra_entities"):
                                self._pocketed_extra_entities = set()
                            self._pocketed_extra_entities.add(ball)
                    break

    def _drop_ball_through_pocket(self, ball, pocket):
        """Release the ball into the hollow cavity under the pocket."""
        drop_pos = [
            float(pocket[0]),
            float(pocket[1]),
            float(self.base_top - self.ball_radius - 0.005),
        ]
        pose = sapien.Pose(drop_pos, [1, 0, 0, 0])
        ball.set_pose(pose)
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity([0.0, 0.0, -0.4])
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _rest_ball_on_floor(self, ball, p):
        """Settle a pocketed ball on the inner floor (keep it visible inside)."""
        rest_z = self.floor_top + self.ball_radius + 0.001
        pose = sapien.Pose([float(p[0]), float(p[1]), rest_z], [1, 0, 0, 0])
        ball.set_pose(pose)
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._update_welded_cue()
        self._track_billiard_metrics()
        if not getattr(self, "_billiard_live", False):
            self._ensure_balls_dynamic()
            return
        if getattr(self, "_interactive_robot_mode", True) is False:
            self._check_cue_distractor_contact()
            self._ensure_balls_dynamic()
            self._check_and_sink_pockets()
            return
        self._check_robot_ball_contact()
        self._check_cue_distractor_contact()
        self._try_apply_strike_impulse()
        self._ensure_balls_dynamic()
        self._check_and_sink_pockets()

    def _dwell(self, steps):
        # After the primary sinks, keep simulating briefly so a trailing
        # distractor that also falls is still detected as a failure.
        post_primary_steps = 80
        post = None
        pace_realtime = self._interactive_viewer_realtime_pace()
        t_start = time.perf_counter()
        last_render_t = t_start - 1.0
        n = max(0, int(steps))
        for i in range(n):
            self._update_kinematic_tasks()
            self.scene.step()
            if pace_realtime:
                last_render_t = self._pace_interactive_control_step(
                    i,
                    t_start,
                    last_render_t,
                    force_render=(i + 1 == n),
                )
            if self.save_freq and i % self.save_freq == 0:
                self._take_picture()
            if (
                self._robot_ball_contact
                or self._distractor_pocketed
                or getattr(self, "_cue_distractor_contact", False)
            ):
                break
            if self._primary_pocketed:
                if post is None:
                    post = 0
                else:
                    post += 1
                    if post >= post_primary_steps:
                        break
        if pace_realtime:
            self._interactive_pacer_resync = True

    def _dbg(self, tag):
        if os.environ.get("PLAY_BILLIARD_DEBUG"):
            tip = np.asarray(self.cue.get_functional_point(0, "list")[:3])
            ball = np.asarray(self.primary_ball.get_pose().p)
            dist = float(np.linalg.norm(tip - ball))
            touching = bool(self._cue_ball_contacting()) if self._cue_welded else False
            print(
                f"[PLAY_BILLIARD] {tag}: plan={self.plan_success} "
                f"tip={np.round(tip, 3).tolist()} ball={np.round(ball, 3).tolist()} "
                f"tip_dist={dist:.4f} cue_contact={touching} "
                f"strike_done={self._strike_done} pocketed={self._primary_pocketed} "
                f"robot_touch={self._robot_ball_contact}",
                flush=True,
            )

    def _tip_xyz(self):
        return np.asarray(self.cue.get_functional_point(0, "list")[:3], dtype=np.float64)

    def _move_tip_z_to(self, arm, target_z, max_step=0.08):
        """Raise/lower the tip in world Z (chunked) so it clears the hollow table."""
        for _ in range(4):
            tip = self._tip_xyz()
            dz = float(target_z - tip[2])
            if abs(dz) < 0.008:
                return True
            step = float(np.clip(dz, -max_step, max_step))
            self.move(
                self.move_by_displacement(arm_tag=arm, z=step, move_axis="world")
            )
            if not self.plan_success:
                return False
        return abs(float(target_z - self._tip_xyz()[2])) < 0.015

    # ------------------------------------------------------------------ policy
    def play_once(self):
        arm = ArmTag(self._arm_side)
        hover_z = float(self.felt_top + self.HOVER_CLEARANCE)

        # 1) Proper pick: open → descend onto shaft → close → weld (no teleport).
        if not self._pick_up_cue(arm):
            self.info["info"] = {
                "{A}": "cue stick",
                "{B}": "red ball",
                "{a}": str(arm),
                "{C}": self._target_pocket_name.replace("_", " "),
            }
            return self.info
        self._dbg("after_grasp")

        # Retreat slightly in -Y then lift high BEFORE any approach.
        # Lift is a real arm motion that carries the already-grasped cue.
        table_near_y = self.table_cy - self.table_half_wid
        tip = self._tip_xyz()
        if tip[1] > table_near_y - 0.04:
            self.move(
                self.move_by_displacement(
                    arm_tag=arm,
                    y=float((table_near_y - 0.06) - tip[1]),
                    move_axis="world",
                )
            )
        self._move_tip_z_to(arm, hover_z, max_step=0.09)
        # Deepen hold for strike reachability (jaws already closed around shaft).
        self._seat_cue_for_strike(arm)
        # Tip drops with the reseat — climb back to hover for the table approach.
        self._move_tip_z_to(arm, hover_z, max_step=0.09)
        self._dbg("after_lift")

        # 2) Choose aim pocket and point the stick at it.
        pocket, aim_pid = self._choose_pocket()
        # Keep episode goal for Opt 1; for any-mode record what we aimed at.
        if not self.specific_hole:
            self._target_pocket = pocket
            self._target_pocket_id = aim_pid
            self._target_pocket_name = self.POCKET_NAMES[aim_pid]
        ball_xy = self._ball_xy(self.primary_ball)
        aim = pocket[:2] - ball_xy
        aim_n = float(np.linalg.norm(aim))
        if aim_n < 1e-6:
            aim = np.array([0.0, 1.0])
            aim_n = 1.0
        self._aim_dir = aim / aim_n

        tip = self._tip_xyz()
        cue_p = np.asarray(self.cue.get_pose().p, dtype=np.float64)
        tip_dir_xy = tip[:2] - cue_p[:2]
        if float(np.linalg.norm(tip_dir_xy)) < 1e-4:
            tip_ang = 0.0
        else:
            tip_ang = float(np.arctan2(tip_dir_xy[1], tip_dir_xy[0]))
        aim_ang = float(np.arctan2(self._aim_dir[1], self._aim_dir[0]))
        yaw_delta = (aim_ang - tip_ang + np.pi) % (2.0 * np.pi) - np.pi
        yaw_delta = float(np.clip(yaw_delta, -np.deg2rad(120), np.deg2rad(120)))
        cur_q = np.array(self.get_arm_pose(str(arm))[3:], dtype=np.float64)
        yaw_q = t3d.quaternions.axangle2quat([0, 0, 1], yaw_delta)
        new_q = t3d.quaternions.qmult(yaw_q, cur_q)
        tip_z = float(self._tip_xyz()[2])
        self.move(
            self.move_by_displacement(
                arm_tag=arm,
                z=float(hover_z - tip_z),
                quat=list(new_q),
                move_axis="world",
            )
        )
        self._dbg("after_yaw")
        if self.plan_success:
            self._move_tip_z_to(arm, hover_z)

        # 3) Tip behind the ball along the aim line (hover → XY → lower).
        behind = ball_xy - self._aim_dir * self.APPROACH_GAP
        safe_y = float(table_near_y - 0.05)
        waypoints = [
            np.array([float(behind[0]), safe_y], dtype=np.float64),
            np.array(
                [float(behind[0]), float(min(behind[1], table_near_y + 0.01))],
                dtype=np.float64,
            ),
            behind,
        ]
        for wp in waypoints:
            for _ in range(8):
                tip = self._tip_xyz()
                if tip[2] < hover_z - 0.015:
                    if not self._move_tip_z_to(arm, hover_z):
                        break
                    tip = self._tip_xyz()
                dx = float(wp[0] - tip[0])
                dy = float(wp[1] - tip[1])
                if abs(dx) < 0.010 and abs(dy) < 0.010:
                    break
                step = 0.055
                scale = min(1.0, step / max(np.hypot(dx, dy), 1e-6))
                dz_keep = float(max(0.0, hover_z - tip[2]))
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm,
                        x=dx * scale,
                        y=dy * scale,
                        z=dz_keep,
                        move_axis="world",
                    )
                )
                if not self.plan_success:
                    break
            if not self.plan_success:
                break
        self._dbg("after_align_xy")

        # Refresh ball pose, then lower and home tip onto the contact point.
        ball_xy = self._ball_xy(self.primary_ball)
        # Leave a small pre-contact gap; tip-sphere touch ≈ ball_r + 2*cue_r in apex dist.
        contact_gap = self.ball_radius + 2.0 * self.CUE_RADIUS + 0.003
        contact_xy = ball_xy - self._aim_dir * contact_gap
        behind = ball_xy - self._aim_dir * self.APPROACH_GAP
        strike_z = float(self.ball_z)

        def _home_tip_xy(target_xy, tol=0.008, hops=5):
            for _ in range(hops):
                tip = self._tip_xyz()
                dx = float(target_xy[0] - tip[0])
                dy = float(target_xy[1] - tip[1])
                if abs(dx) < tol and abs(dy) < tol:
                    return True
                step = 0.04
                scale = min(1.0, step / max(np.hypot(dx, dy), 1e-6))
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, x=dx * scale, y=dy * scale, move_axis="world"
                    )
                )
                if not self.plan_success:
                    return False
            tip = self._tip_xyz()
            return abs(float(target_xy[0] - tip[0])) < 0.015 and abs(
                float(target_xy[1] - tip[1])
            ) < 0.015

        if self.plan_success:
            _home_tip_xy(behind, tol=0.010)
        if self.plan_success:
            self._move_tip_z_to(arm, strike_z, max_step=0.06)
        # Arm strike phase: tip near the ball; ignore gripper ghost contacts.
        self._strike_armed = True
        self._strike_done = False
        if self.plan_success:
            _home_tip_xy(contact_xy, tol=0.006, hops=6)
        if self.plan_success:
            # Re-seat Z after XY hops (planner often lifts the tip).
            self._move_tip_z_to(arm, strike_z, max_step=0.04)
        # Final close: drive tip straight at the ball until near contact.
        if self.plan_success:
            contact_thresh = (
                self.ball_radius + 2.0 * self.CUE_RADIUS + self.STRIKE_CONTACT_GAP
            )
            for _ in range(8):
                tip = self._tip_xyz()
                ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
                dist = float(np.linalg.norm(tip - ball_p))
                if dist <= contact_thresh * 0.95:
                    break
                delta = ball_p - tip
                # Prefer closing in XY; keep Z near strike height.
                step = min(0.02, 0.55 * dist)
                nxy = float(np.linalg.norm(delta[:2]))
                if nxy < 1e-6:
                    break
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm,
                        x=float(delta[0] / nxy * step),
                        y=float(delta[1] / nxy * step),
                        z=float(np.clip(strike_z - tip[2], -0.02, 0.02)),
                        move_axis="world",
                    )
                )
                if not self.plan_success:
                    self.plan_success = True
                    break
                self._dwell(6)
        self._dbg("after_align_z")

        # Strike: push through the ball along aim. Impulse only after tip–ball contact.
        through_xy = ball_xy + self._aim_dir * self.STRIKE_PUSH
        for _ in range(10):
            if self._strike_done or self._primary_pocketed or not self.plan_success:
                break
            tip = self._tip_xyz()
            # Keep tip at ball height so the blue tip can actually touch.
            dz = float(strike_z - tip[2])
            if abs(dz) > 0.008:
                self._move_tip_z_to(arm, strike_z, max_step=0.03)
                tip = self._tip_xyz()
            ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
            # Correct lateral error, then advance toward/through the ball.
            to_ball = ball_p[:2] - tip[:2]
            along = float(np.dot(to_ball, self._aim_dir))
            lateral = to_ball - self._aim_dir * along
            # If already past the ball center, keep driving toward through_xy.
            advance = self._aim_dir * (0.022 if along > -0.005 else 0.015)
            target = tip[:2] + lateral * 0.85 + advance
            # Clamp progress so we don't overshoot the through point badly.
            to_through = through_xy - tip[:2]
            if float(np.linalg.norm(to_through)) < 0.008:
                break
            self.move(
                self.move_by_displacement(
                    arm_tag=arm,
                    x=float(target[0] - tip[0]),
                    y=float(target[1] - tip[1]),
                    move_axis="world",
                )
            )
            self._dwell(10)
        self._dbg("after_strike")
        retract = -self._aim_dir * 0.04
        tip = self._tip_xyz()
        self.move(
            self.move_by_displacement(
                arm_tag=arm,
                x=float(retract[0]),
                y=float(retract[1]),
                z=float(max(0.04, hover_z - tip[2])),
                move_axis="world",
            )
        )
        self._dbg("after_retract")

        self._dwell(self.settle_steps)
        self._dbg("after_settle")

        self.info["info"] = {
            "{A}": "cue stick",
            "{B}": "red ball",
            "{a}": str(arm),
            "{C}": self._target_pocket_name.replace("_", " "),
        }
        return self.info

    # ------------------------------------------------------------------ success
    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_armed_step = None    # tip in place, strike phase armed
        self._metric_strike_step = None   # impulse actually applied
        self._metric_sink_step = None     # primary fell into a pocket
        self._metric_aim_error_deg = None # stick-vs-ideal angle at the strike
        self._metric_strike_ball_xy = None

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_strike_metrics(self, direction_xy):
        """Called from _try_apply_strike_impulse with the direction actually used.

        The aim error is captured here because the ball leaves immediately after —
        the post-shot geometry says nothing about what the operator aimed at.
        """
        try:
            if self._metric_strike_step is not None:
                return
            self._metric_strike_step = self._metric_step()
            ball = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)[:2]
            self._metric_strike_ball_xy = [float(ball[0]), float(ball[1])]
            ids = list(getattr(self, "_allowed_pocket_ids", []) or [])
            d = np.asarray(direction_xy, dtype=np.float64)[:2]
            dn = float(np.linalg.norm(d))
            if not ids or dn < 1e-9:
                self._metric_aim_error_deg = None
                return
            d = d / dn
            best = None
            for pid in ids:
                ideal = np.asarray(self._pocket_centers[int(pid)], dtype=np.float64)[:2] - ball
                inorm = float(np.linalg.norm(ideal))
                if inorm < 1e-9:
                    continue
                cos = float(np.clip(np.dot(d, ideal / inorm), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cos)))
                if best is None or ang < best:
                    best = ang
            self._metric_aim_error_deg = best
        except Exception:
            self._metric_aim_error_deg = None

    # Primary-ball speed (m/s) that counts as "the shot has been struck".
    STRIKE_SPEED_THRESHOLD = 0.05

    def _track_billiard_metrics(self):
        """Per-step latches: strike arming, the shot itself, and the primary sinking.

        The strike is detected from the primary ball's own velocity rather than from
        ``_try_apply_strike_impulse`` — that scripted path only runs in interactive
        mode, while the expert drives the cue into the ball under real physics.
        """
        try:
            if self._metric_armed_step is None and getattr(self, "_strike_armed", False):
                self._metric_armed_step = self._metric_step()
            if self._metric_strike_step is None and self._primary_rigid is not None:
                v = np.asarray(self._primary_rigid.get_linear_velocity(), dtype=np.float64)
                if float(np.linalg.norm(v[:2])) >= self.STRIKE_SPEED_THRESHOLD:
                    self._latch_strike_metrics(v[:2])
            if self._metric_sink_step is None and getattr(self, "_primary_pocketed", False):
                self._metric_sink_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `strike_latency_steps` — steps from the cue tip reaching the strike pose
        (aim armed) until the impulse is applied. This is the aiming / commit window.
        extra2 `aim_error_norm` — angle between the stick direction at the strike and the
        straight line from the primary ball to the nearest ALLOWED pocket, divided by the
        angular half-width that pocket subtends from the ball at that moment. LOWER is
        better; <= 1.0 means the shot line was inside the pocket mouth.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a = getattr(self, "_metric_armed_step", None)
        b = getattr(self, "_metric_strike_step", None)
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["strike_latency_steps"] = lat
        out["strike_latency_s"] = None if lat is None else round(lat * dt, 4)

        c = getattr(self, "_metric_sink_step", None)
        roll = None if (b is None or c is None) else max(int(c) - int(b), 0)
        out["sink_latency_steps"] = roll
        out["sink_latency_s"] = None if roll is None else round(roll * dt, 4)

        ang = getattr(self, "_metric_aim_error_deg", None)
        out["aim_error_deg"] = None if ang is None else round(float(ang), 4)
        try:
            # Angular half-width of the pocket mouth as seen from the ball AT STRIKE TIME
            # (the ball has moved since, so the live pose would give the wrong scale).
            ball = getattr(self, "_metric_strike_ball_xy", None)
            ids = list(getattr(self, "_allowed_pocket_ids", []) or [])
            tol = None
            if ball is not None and ids:
                b_xy = np.asarray(ball, dtype=np.float64)
                dists = [
                    float(np.linalg.norm(
                        np.asarray(self._pocket_centers[int(pid)], dtype=np.float64)[:2] - b_xy))
                    for pid in ids
                ]
                d0 = min(dists) if dists else None
                if d0 is not None and d0 > 1e-6:
                    tol = float(np.degrees(np.arctan2(float(self.pocket_radius), d0)))
            out["aim_tolerance_deg"] = None if tol is None else round(tol, 4)
            out["aim_error_norm"] = (
                None if (ang is None or not tol) else round(float(ang) / max(tol, 1e-9), 4)
            )
        except Exception:
            out["aim_tolerance_deg"] = None
            out["aim_error_norm"] = None
        return out

    def check_success(self):
        """Success = primary in an allowed pocket with no foul.

        Default / Opt 2: any of the 6 pockets is allowed.
        Opt 1 / Opt 1+2: only the nominated target pocket is allowed.
        Fouls (automatic failure):
          - robot link touches the primary ball
          - cue stick touches any non-target ball
          - any non-target ball falls into any pocket
        """
        if self._robot_ball_contact:
            return False
        if getattr(self, "_cue_distractor_contact", False):
            return False
        if getattr(self, "_distractor_pocketed", False):
            return False
        # Re-scan distractors in case success is checked without a sink step.
        for ball in self.extra_balls:
            if ball is None:
                continue
            p = np.asarray(ball.get_pose().p, dtype=np.float64)
            if self._ball_inside_hollow(p):
                self._distractor_pocketed = True
                return False
        if not self._primary_pocketed:
            if self.primary_ball is None:
                return False
            p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
            if not self._ball_inside_hollow(p):
                return False
            pid, _ = self._nearest_pocket_id(p[:2])
            self._primary_pocket_id = pid
            self._primary_pocketed = True
        # Must land in an allowed pocket for this episode's mode.
        if self._primary_pocket_id is None:
            return False
        return int(self._primary_pocket_id) in set(self._allowed_pocket_ids)

    def get_obs(self):
        obs = super().get_obs()
        if not getattr(self, "_loaded", False) or self.primary_ball is None:
            return obs
        ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
        tip = (
            np.asarray(self.cue.get_functional_point(0, "list")[:3], dtype=np.float64)
            if self.cue is not None
            else np.zeros(3)
        )
        pocket = (
            self._target_pocket.tolist()
            if self._target_pocket is not None
            else [0.0, 0.0, 0.0]
        )
        obs["play_billiard"] = {
            "primary_ball": ball_p.tolist(),
            "cue_tip": tip.tolist(),
            "target_pocket": pocket,
            "target_pocket_name": self._target_pocket_name,
            "pocket_mode": self.pocket_mode,
            "specific_hole": float(self.specific_hole),
            "enable_distractors": float(self.enable_distractors),
            "blocker_mode": self.blocker_mode,
            "blocked_pocket_id": float(
                -1 if self._blocked_pocket_id is None else self._blocked_pocket_id
            ),
            "arm_side": self._arm_side,
            "aim_dir": self._aim_dir.tolist(),
            "primary_pocketed": float(self._primary_pocketed),
            "distractor_pocketed": float(getattr(self, "_distractor_pocketed", False)),
            "robot_ball_contact": float(self._robot_ball_contact),
            "cue_distractor_contact": float(
                getattr(self, "_cue_distractor_contact", False)
            ),
            "num_extra_balls": float(len(self.extra_balls)),
            "strike_done": float(self._strike_done),
        }
        return obs


# NOTICE: Primitive-only task. Reserved object-id range [350, 359] documents the
# logical identities used in language templates (cue_stick/base350, primary_ball/base351,
# billiard_table/base352). Nothing is written under assets/objects/.
