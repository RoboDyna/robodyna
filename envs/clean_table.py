"""Clean a coffee spill before it reaches a laptop.

Setup (dual UR5e / ur5-wsg):
  - A coffee mug tips over toward the laptop (fall along the spill direction).
  - A real ``015_laptop`` sits left **or** right of the mug (seed-randomized)
    at a fixed ~25 cm X gap; tip/spill always run mug → laptop.
  - Spill starts at the mug *rim* on the table, then spreads toward the laptop
    with a seed-randomized irregular lobed / teardrop shape (±30% speed).
  - A yellow sponge on the mug side (offset away from the laptop); the matching
    arm **dabs** each stain (lift → lower onto the table → lift), avoiding the
    mug. Mug is dynamic so collisions from the arm/sponge move it instead of
    ghosting through.
  - Background décor randomized in the upper 30% of the table Y span (no overlap):
    plant, office file holder, pen cup, and alarm clock.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.action import Action
from .utils.create_actor import create_actor, create_box, create_sapien_urdf_obj
from .utils.partial_score import score_half_open_intervals


class clean_table(Base_Task):
    """Tip a coffee mug, then wipe the irregular spill before it hits a laptop."""

    # clean_frac [lo, hi) → partial score (laptop hit / tiny spill → 0).
    PARTIAL_CLEAN_BANDS = (
        (0.85, 1.0, 0.75),
        (0.60, 0.85, 0.5),
        (0.35, 0.60, 0.25),
    )
    PARTIAL_MIN_SPILL = 0.20

    MUG_MODEL = "039_mug"
    MUG_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]
    PROP_UPRIGHT_Q = [0.5, 0.5, 0.5, 0.5]

    # Latte-like coffee (matches reference spill photo).
    COFFEE_COLOR = [0.55, 0.38, 0.22, 0.95]
    SPILL_COLOR = [0.52, 0.34, 0.18, 0.92]
    SPONGE_COLOR = (0.95, 0.82, 0.22)

    SPILL_STEPS_DEFAULT = 2500  # ~3x faster than the previous 7500 default
    REACH_LAPTOP_LEVEL_DEFAULT = 0.88
    CLEAN_TOL_DEFAULT = 0.08  # fraction of spawned spots that may remain dirty
    POST_TIP_WAIT_DEFAULT = 60
    EXPERT_WIPE_LEVEL_DEFAULT = 0.20
    # Spill amount right after tip — high enough that the whole initial
    # multi-spot puddle is visible (no "wait by the mug for the next lobe").
    INITIAL_PUDDLE_LEVEL_DEFAULT = 0.22
    # Extra XY margin (m) beyond the pad half-extents when testing spot overlap.
    SPOT_CONTACT_EXTRA = 0.004
    # Pad bottom must be within this of the table top to count as wiping.
    SPONGE_TABLE_CONTACT_Z = 0.012
    # Consecutive contact steps required before a single spot clears.
    SPOT_CLEAR_DWELL = 6
    # Keep sponge this far above the table while translating between dabs.
    WIPE_SAFE_Z = 0.11
    # Don't dab within this margin of the (live) mug footprint.
    MUG_AVOID_MARGIN = 0.025

    SPONGE_HALF = (0.038, 0.028, 0.014)
    # Small grasp cube on top of the pad (meters, half-extents) ~16×12×40 mm.
    # Tall enough that a top-down pinch seats on the cube, not the pad face.
    SPONGE_HANDLE_HALF = (0.008, 0.006, 0.020)
    SPONGE_HANDLE_COLOR = (0.55, 0.42, 0.22)
    # Fully open for approach (narrow openings leave a bad WSG wrist that
    # teleports on the next move). Close to ~handle width (~16 mm ≈ 0.25).
    SPONGE_GRASP_OPEN = 1.0
    SPONGE_GRASP_CLOSE = 0.25
    EE_TO_TCP = 0.12
    GRASP_TCP_TOL = 0.04
    MUG_SCALE_MULT = 0.60  # stock mug is large; shrink for graspable station size
    MUG_MASS = 0.18

    # Table is 1.2 × 0.7 m (see Base_Task.create_table_and_wall).
    TABLE_HALF_X = 0.55
    TABLE_HALF_Y = 0.32
    # Center-to-center X gap (m), randomized per seed in this band.
    MUG_LAPTOP_GAP = (0.30, 0.40)
    MUG_Y_RANGE = (-0.10, 0.50)
    # Décor lives in the far / "upper" 30% of the table Y span.
    DECOR_Y_FRAC = 0.30
    SPILL_SPEED_JITTER = 0.30  # ±30% of configured spill rate

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("clean_table", {}))

        # Per-step state must exist before early _update_kinematic_tasks calls.
        self._loaded = False
        self.cup_tipped = False
        self.spill_active = False
        self.spill_amount = 0.0
        self.max_spill_amount = 0.0
        self.spill_cleaned = 0.0
        self.laptop_reached = False
        self._partial_score = 0.0  # latched at the laptop-contact frame
        self.cleaned_ok = False
        self._reset_metric_state()
        self._spill_frozen = False
        self._coffee_entity = None
        self._spill_entity = None
        self._spill_visual_cached = None
        self._expert_wiping = False
        self._wipe_armed = False
        self._wipe_target = None
        self._wipe_contact_steps = 0
        self._wipe_contact_map = {}
        self._sponge_welded = False
        self._sponge_weld_offset = None
        self._sponge_hold_quat = None
        self._mug_rigid = None
        self._sponge_rigid = None
        self._spill_spots: list[dict] = []
        self.cup = None
        self.mug = None
        self.laptop = None
        self.sponge = None
        self.arm = ArmTag("right")
        self.table_top = 0.74
        self._spill_seed = 0
        self._lobe_phase = 0.0
        self._spill_speed_mult = 1.0
        self._spill_pattern = {}
        self.laptop_side = -1.0
        self.mug_side = 1.0
        self.wipe_safe_z = self.WIPE_SAFE_Z
        self.mug_avoid_margin = self.MUG_AVOID_MARGIN

        super()._init_task_env_(**kwags)
        # Mug tips after settle so policy eval does not wait for play_once.
        self._animate_tip()

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = 0.74 + float(self.table_z_bias)

        self.reach_laptop_level = float(
            cfg.get("reach_laptop_level", self.REACH_LAPTOP_LEVEL_DEFAULT)
        )
        self.clean_tol = float(cfg.get("clean_tol", self.CLEAN_TOL_DEFAULT))
        self.post_tip_wait = int(cfg.get("post_tip_wait", self.POST_TIP_WAIT_DEFAULT))
        self.expert_wipe_level = float(
            cfg.get("expert_wipe_level", self.EXPERT_WIPE_LEVEL_DEFAULT)
        )
        self.initial_puddle_level = float(
            cfg.get("initial_puddle_level", self.INITIAL_PUDDLE_LEVEL_DEFAULT)
        )
        self.spot_contact_extra = float(
            cfg.get("spot_contact_extra", self.SPOT_CONTACT_EXTRA)
        )
        self.spot_clear_dwell = int(cfg.get("spot_clear_dwell", self.SPOT_CLEAR_DWELL))
        self.wipe_safe_z = float(cfg.get("wipe_safe_z", self.WIPE_SAFE_Z))
        self.mug_avoid_margin = float(cfg.get("mug_avoid_margin", self.MUG_AVOID_MARGIN))

        self.spill_amount = 0.0
        self.max_spill_amount = 0.0
        self.spill_cleaned = 0.0
        self.cup_tipped = False
        self.spill_active = False
        self._spill_frozen = False
        self.laptop_reached = False
        self._partial_score = 0.0
        self.cleaned_ok = False
        self._reset_metric_state()
        self._coffee_entity = self._remove_entity(getattr(self, "_coffee_entity", None))
        self._spill_entity = self._remove_entity(getattr(self, "_spill_entity", None))
        self._spill_visual_cached = None
        self._spill_spots = []
        self._expert_wiping = False
        self._wipe_armed = False
        self._wipe_target = None
        self._wipe_contact_steps = 0
        self._wipe_contact_map = {}
        self._sponge_welded = False
        self._sponge_weld_offset = None
        self._sponge_hold_quat = None
        self._mug_rigid = None

        self._sample_layout_and_spill()

        self._spawn_mug()
        self._spawn_coffee_in_mug()
        self._spawn_laptop()
        self._spawn_sponge()
        self._spawn_decorations()

        # Rim contact on the table after tip — origin of the spill (NOT under the body).
        rim_along = float(getattr(self, "mug_half_height", 0.04)) + 0.008
        self.spill_origin = self.mug_xy + self.spill_dir * rim_along
        # Path ends at the laptop near edge (do NOT floor at 0.14 — that overshoots
        # into the chassis on short gaps and used to hide geometric contact).
        laptop_half = float(getattr(self, "laptop_half_along", 0.10))
        laptop_near = self.laptop_xy - self.spill_dir * laptop_half
        raw_path = float(np.dot(laptop_near - self.spill_origin, self.spill_dir))
        self.spill_path_len = max(0.04, raw_path)
        self.laptop_near_xy = laptop_near

        self.add_prohibit_area(self.mug, padding=0.04)
        self.add_prohibit_area(self.sponge, padding=0.03)
        self.cup = self.mug  # keep tip/wipe helpers that still say "cup"
        self._build_spill_spots()
        self._loaded = True
        print(
            f"[clean_table] arm={self.arm} laptop_side="
            f"{'right' if self.laptop_side > 0 else 'left'} "
            f"mug={self.mug_xy} laptop={self.laptop_xy} sponge={self.sponge_xy} "
            f"dir={self.spill_dir} path={self.spill_path_len:.3f}m "
            f"spill_steps={self.spill_steps} speed_mult={self._spill_speed_mult:.2f} "
            f"spots={len(self._spill_spots)} seed={self._spill_seed}"
        )

    def _sample_layout_and_spill(self):
        """Seed-randomized laptop/mug/sponge poses, spill speed, and pattern seed.

        Base layout (matches the original working scene): laptop on one side of
        the mug with a fixed ~25 cm X gap. Mirroring puts the laptop on the
        opposite side of the mug. Tip/spill always run from mug → laptop.
        """
        cfg = self._cfg
        randomize = bool(cfg.get("randomize_layout", True))

        base_laptop_y = float(cfg.get("laptop_y", -0.02))
        gap_lo, gap_hi = self.MUG_LAPTOP_GAP
        gap_lo = float(cfg.get("mug_laptop_gap_min", gap_lo))
        gap_hi = float(cfg.get("mug_laptop_gap_max", gap_hi))
        mug_y_lo, mug_y_hi = self.MUG_Y_RANGE
        mug_y_lo = float(cfg.get("mug_y_min", mug_y_lo))
        mug_y_hi = float(cfg.get("mug_y_max", mug_y_hi))

        # Laptop side relative to mug: left => laptop.x < mug.x (original).
        side_cfg = cfg.get("laptop_side", None)
        if side_cfg is None:
            mug_side_cfg = cfg.get("mug_side", "random")
            if mug_side_cfg == "left":
                side_cfg = "right"
            elif mug_side_cfg == "right":
                side_cfg = "left"
            else:
                side_cfg = "random"

        if side_cfg in ("left", "right") and (
            (not randomize) or bool(cfg.get("force_side", False))
        ):
            laptop_side = -1.0 if side_cfg == "left" else 1.0
        else:
            laptop_side = 1.0 if (np.random.rand() > 0.5) else -1.0

        # Per-seed center-to-center gap (default 20–40 cm).
        gap = float(np.random.uniform(gap_lo, gap_hi)) if randomize else float(gap_lo)
        gap = max(0.30, min(0.40, gap))

        if randomize:
            # Pair straddles the midline: mug on one half, laptop on the other,
            # exactly ``gap`` apart along X (original working geometry).
            half = 0.5 * gap
            mid = float(np.random.uniform(-0.15, 0.15) * half)
            # laptop_side < 0 ⇒ laptop left of mug (original).
            mug_x = mid - laptop_side * half
            laptop_x = mid + laptop_side * half
            # ±20% rigid shift of the whole pair (gap unchanged).
            x_jit = float(np.random.uniform(-0.20, 0.20)) * half
            mug_x -= x_jit
            laptop_x -= x_jit
            mug_x = laptop_x - laptop_side * gap

            decor_y_lo = self.TABLE_HALF_Y - float(self.DECOR_Y_FRAC) * (
                2.0 * self.TABLE_HALF_Y
            )
            mug_y_hi_eff = min(mug_y_hi, decor_y_lo - 0.04, self.TABLE_HALF_Y - 0.04)
            mug_y_lo_eff = max(mug_y_lo, -self.TABLE_HALF_Y + 0.04)
            if mug_y_lo_eff > mug_y_hi_eff:
                mug_y_lo_eff, mug_y_hi_eff = -0.10, 0.08
            mug_y = float(np.random.uniform(mug_y_lo_eff, mug_y_hi_eff))
            y_jit = float(np.random.uniform(0.70, 1.30))  # ±30%
            laptop_y = float(base_laptop_y * y_jit)
            laptop_y = float(np.clip(laptop_y, mug_y - 0.06, mug_y + 0.06))
            laptop_y = float(
                np.clip(laptop_y, -self.TABLE_HALF_Y + 0.04, self.TABLE_HALF_Y - 0.04)
            )
            mug_x = float(
                np.clip(mug_x, -self.TABLE_HALF_X + 0.05, self.TABLE_HALF_X - 0.05)
            )
            laptop_x = float(
                np.clip(laptop_x, -self.TABLE_HALF_X + 0.05, self.TABLE_HALF_X - 0.05)
            )
            # If clipping broke the gap, restore it from the mug.
            if abs(laptop_x - mug_x) < gap - 1e-4:
                laptop_x = mug_x + laptop_side * gap
                laptop_x = float(
                    np.clip(
                        laptop_x, -self.TABLE_HALF_X + 0.05, self.TABLE_HALF_X - 0.05
                    )
                )
                mug_x = laptop_x - laptop_side * gap
                mug_x = float(
                    np.clip(mug_x, -self.TABLE_HALF_X + 0.05, self.TABLE_HALF_X - 0.05)
                )
        else:
            # Deterministic: opposite halves with configured gap.
            half = 0.5 * gap
            mug_x = (-laptop_side) * half
            laptop_x = laptop_side * half
            mug_y = float(cfg.get("mug_y", -0.02))
            laptop_y = base_laptop_y

        mug_side = 1.0 if mug_x >= 0.0 else -1.0
        # Sponge on the mug half, toward the robot, offset away from the laptop.
        sponge_dx = float(cfg.get("sponge_dx", 0.12))
        sponge_x = mug_x - laptop_side * sponge_dx
        sponge_y = float(cfg.get("sponge_y", -0.18))
        if randomize:
            sponge_y = float(
                np.clip(
                    sponge_y + np.random.uniform(-0.04, 0.04),
                    -self.TABLE_HALF_Y + 0.05,
                    min(mug_y, 0.0) - 0.06,
                )
            )
            sponge_x = float(
                np.clip(sponge_x, -self.TABLE_HALF_X + 0.05, self.TABLE_HALF_X - 0.05)
            )

        self.laptop_side = float(laptop_side)
        self.mug_side = float(mug_side)
        self.arm = ArmTag("right" if mug_side >= 0 else "left")
        self.cup_xy = np.array([mug_x, mug_y], dtype=float)
        self.mug_xy = self.cup_xy.copy()
        self.laptop_xy = np.array([laptop_x, laptop_y], dtype=float)
        self.sponge_xy = np.array([sponge_x, sponge_y], dtype=float)
        self._layout_gap = float(abs(laptop_x - mug_x))

        # Spill / tip direction: always mug → laptop.
        delta = self.laptop_xy - self.mug_xy
        n = float(np.linalg.norm(delta))
        self.spill_dir = (
            (delta / n) if n > 1e-6 else np.array([float(laptop_side), 0.0])
        )
        self.spill_lat = np.array([-self.spill_dir[1], self.spill_dir[0]], dtype=float)

        # Spill speed ±30% of configured rate (lower spill_steps → faster).
        base_steps = int(cfg.get("spill_steps", self.SPILL_STEPS_DEFAULT))
        speed_jit = float(cfg.get("spill_speed_jitter", self.SPILL_SPEED_JITTER))
        if randomize and speed_jit > 0:
            self._spill_speed_mult = float(
                np.random.uniform(1.0 - speed_jit, 1.0 + speed_jit)
            )
            self.spill_steps = max(1, int(round(base_steps / self._spill_speed_mult)))
        else:
            self._spill_speed_mult = 1.0
            self.spill_steps = base_steps

        self._spill_seed = int(cfg.get("spill_seed", np.random.randint(0, 10_000)))
        self._lobe_phase = float((self._spill_seed % 97) / 97.0 * 2.0 * np.pi)
        # Extra pattern knobs used by _build_spill_spots (organic stream variety).
        rng = np.random.RandomState(self._spill_seed)
        self._spill_pattern = {
            "wobble1": float(rng.uniform(0.008, 0.018)),
            "wobble2": float(rng.uniform(0.004, 0.012)),
            "freq1": float(rng.uniform(1.6, 2.8)),
            "freq2": float(rng.uniform(3.5, 6.5)),
            "width_scale": float(rng.uniform(0.85, 1.20)),
            "side_bias": float(rng.uniform(-0.25, 0.25)),
            "n_main": int(rng.randint(6, 10)),
            "puddle_spread": float(rng.uniform(0.85, 1.20)),
        }

    def _spawn_mug(self):
        self.mug_id = int(self._cfg.get("mug_id", 0))
        scale_mult = float(self._cfg.get("mug_scale_mult", self.MUG_SCALE_MULT))
        pose = sapien.Pose(
            [float(self.mug_xy[0]), float(self.mug_xy[1]), self.table_top + 0.001],
            self.MUG_UPRIGHT_Q,
        )
        # Dynamic body so arm/sponge collisions can knock it; held kinematic
        # until the tip animation finishes.
        self.mug = create_actor(
            self,
            pose=pose,
            modelname=self.MUG_MODEL,
            model_id=self.mug_id,
            convex=True,
            is_static=False,
            scale_mult=scale_mult,
        )
        self.mug.set_name("coffee_mug")
        self._mug_upright_pose = pose
        self._mug_rigid = self._get_rigid(self.mug)
        self._make_kinematic(self.mug, mass=float(self._cfg.get("mug_mass", self.MUG_MASS)))

        cfg = getattr(self.mug, "config", {}) or {}
        ext = np.array(cfg.get("extents", [1.0, 1.0, 1.0]), dtype=float)
        sc = cfg.get("scale", [1, 1, 1])
        sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        world = ext * sc
        # Local Y is up when upright.
        self.mug_height = float(world[1])
        self.mug_half_height = 0.5 * self.mug_height
        # Outer body radius (bbox includes the handle).
        self.mug_radius = 0.42 * float(max(world[0], world[2]))
        # Cup-head opening (not the handle). Pose origin is the cup bottom
        # (functional point 1); AABB includes the handle so inner radius and
        # cavity offset are fit to the circular body (~mesh-measured).
        wmax = float(max(world[0], world[2]))
        self.mug_inner_r = 0.26 * wmax
        self._mug_center_offset = np.zeros(3, dtype=float)
        self._mug_cavity_offset = np.zeros(3, dtype=float)
        # Nudge coffee into the cup head, opposite the handle functional point.
        try:
            fmat = cfg.get("functional_matrix") or []
            if fmat:
                h = np.array(fmat[0], dtype=float)[:3, 3] * sc
                R = t3d.quaternions.quat2mat(
                    np.asarray(self.MUG_UPRIGHT_Q, dtype=float)
                )
                h_xy = (R @ h)[:2]
                n = float(np.linalg.norm(h_xy))
                if n > 1e-6:
                    # Away from handle into the cup head (~0.12 × AABB width).
                    away = -h_xy / n
                    self._mug_cavity_offset[:2] = away * (0.12 * wmax)
        except Exception:
            pass
        self.cup_height = self.mug_height
        self.cup_radius = self.mug_radius
        self.cup_inner_r = self.mug_inner_r

    def _spawn_coffee_in_mug(self):
        """Coffee column filling the cup-head opening (~40% full).

        PartNet ``039_mug`` pose origin is the cup bottom; the handle sticks out
        sideways. Place an opaque column on the cavity axis so the surface
        matches the inner circle of the mug.
        """
        self._coffee_entity = self._remove_entity(self._coffee_entity)
        off = np.asarray(
            getattr(self, "_mug_cavity_offset", np.zeros(3)), dtype=float
        )
        inner_r = float(max(0.008, self.mug_inner_r))
        fill_frac = float(self._cfg.get("coffee_fill_frac", 0.40))
        fill_frac = float(np.clip(fill_frac, 0.15, 0.92))
        pose_p = np.asarray(self._mug_upright_pose.p, dtype=float)
        cx = float(pose_p[0] + off[0])
        cy = float(pose_p[1] + off[1])
        z_bottom = float(self.table_top) + 0.08 * float(self.mug_height)
        z_top = float(self.table_top) + fill_frac * float(self.mug_height)
        half_h = max(0.0025, 0.5 * (z_top - z_bottom))
        z_c = z_bottom + half_h
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=list(self.COFFEE_COLOR))
        try:
            mat.set_roughness(0.35)
            mat.set_metallic(0.0)
            mat.set_transmission(0.0)
        except Exception:
            mat.roughness = 0.35
            mat.metallic = 0.0
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=inner_r,
            half_length=half_h,
            material=mat,
        )
        builder.set_initial_pose(sapien.Pose(p=[cx, cy, z_c]))
        self._coffee_entity = builder.build(name="coffee_column")

    def _spawn_laptop(self):
        """Real PartNet ``015_laptop`` articulation, root fixed on the table."""
        lid = int(self._cfg.get("laptop_id", 0))
        lx, ly = float(self.laptop_xy[0]), float(self.laptop_xy[1])
        # PartNet default sits edge-on (screen normal ±X). +90° about Z faces keyboard
        # / screen toward the robot workspace (−Y), independent of mug side.
        yaw = float(self._cfg.get("laptop_yaw", 0.5 * np.pi))
        q = t3d.euler.euler2quat(0.0, 0.0, yaw, axes="sxyz")
        pose = sapien.Pose([lx, ly, self.table_top + 0.002], q.tolist())
        self.laptop = create_sapien_urdf_obj(
            self,
            pose=pose,
            modelname="015_laptop",
            modelid=lid,
            fix_root_link=True,
        )
        self.laptop_id = lid
        # Table footprint half-extents (m). PartNet extents are mesh-local —
        # multiply by authored scale (≈0.15). After laptop_yaw ≈ +90° Z,
        # model Y → world ±X (spill axis) and model X → world ±Y.
        self.laptop_half_along = 0.10
        self.laptop_half_lat = 0.08
        try:
            cfg = getattr(self.laptop, "config", {}) or {}
            ext = np.array(cfg.get("extents", [0.2, 0.15, 0.2]), dtype=float)
            sc = cfg.get("scale", 0.15)
            sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
            world = ext * sc
            self.laptop_half_along = 0.45 * float(max(world[0], world[1]))
            self.laptop_half_lat = 0.45 * float(min(world[0], world[1]))
        except Exception:
            pass

    def _spawn_sponge(self):
        """Yellow pad + small top handle so the gripper can lift cleanly."""
        hx, hy, hz = [float(v) for v in self.SPONGE_HALF]
        hhx, hhy, hhz = [float(v) for v in self.SPONGE_HANDLE_HALF]
        handle_z = hz + hhz  # handle sits flush on the pad top

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        pad_mat = sapien.render.RenderMaterial(
            base_color=[*self.SPONGE_COLOR, 1.0]
        )
        handle_mat = sapien.render.RenderMaterial(
            base_color=[*self.SPONGE_HANDLE_COLOR, 1.0]
        )
        try:
            pad_mat.set_roughness(0.85)
            handle_mat.set_roughness(0.55)
        except Exception:
            pass

        # Pad (entity origin = pad center). Visual + collision for table rest;
        # pad collision is disabled while pinching the handle so fingers cannot
        # stop on the wide yellow body.
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=self.scene.default_physical_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=pad_mat,
        )
        # Handle stub on top — the only shape the gripper should pinch.
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, handle_z]),
            half_size=[hhx, hhy, hhz],
            material=self.scene.default_physical_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, handle_z]),
            half_size=[hhx, hhy, hhz],
            material=handle_mat,
        )

        pose = sapien.Pose(
            [
                float(self.sponge_xy[0]),
                float(self.sponge_xy[1]),
                self.table_top + hz + 0.001,
            ],
            [1, 0, 0, 0],
        )
        builder.set_initial_pose(pose)
        entity = builder.build(name="sponge")

        # Grasp frames on the small top handle (meters; scale=1).
        # Top-down contact frames so the gripper pinches the stub, not the pad.
        top_z = float(handle_z)
        data = {
            "center": [0, 0, 0],
            "extents": [hx * 2, hy * 2, (hz + 2 * hhz) * 2],
            "scale": [1.0, 1.0, 1.0],
            "contact_points_pose": [
                # Four yaw variants, all centered on the handle top.
                [[0, 0, 1, 0.0], [1, 0, 0, 0.0], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[1, 0, 0, 0.0], [0, 0, -1, 0.0], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, 0.0], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[0, 0, -1, 0.0], [-1, 0, 0, 0.0], [0, 1, 0, top_z], [0, 0, 0, 1]],
            ],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }
        self.sponge = Actor(entity, data, mass=0.045)
        self._sponge_handle_z = handle_z
        self._sponge_handle_half = (hhx, hhy, hhz)
        self._sponge_rigid = None
        self._sponge_pad_shape = None
        self._sponge_handle_shape = None
        for c in self.sponge.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(4.0)
                    c.set_angular_damping(4.0)
                except Exception:
                    pass
                self._sponge_rigid = c
                try:
                    shapes = list(c.get_collision_shapes())
                    # Spawn order: pad then handle.
                    if len(shapes) >= 1:
                        self._sponge_pad_shape = shapes[0]
                    if len(shapes) >= 2:
                        self._sponge_handle_shape = shapes[1]
                    # SAPIEN box shapes default to contype/conaffinity 0 (no
                    # contacts). Turn both on so the pad rests on the table and
                    # the handle can actually stop the fingers.
                    for shape in shapes:
                        self._set_shape_collision_enabled(shape, True)
                except Exception:
                    pass
                break
        self._sponge_spawn_pose = pose

    def _create_bench_glb(self, model_name: str, pose: sapien.Pose, scale, mass=0.1):
        """Static décor from ``assets/dyna_assets`` (office file holder, etc.)."""
        model_dir = resolve_model_dir(model_name)
        glb = model_dir / "base.glb"
        if not glb.exists():
            candidates = sorted(model_dir.glob("*.glb"))
            if not candidates:
                raise FileNotFoundError(f"No GLB found in {model_dir}")
            glb = candidates[0]
        sc = [float(scale)] * 3 if isinstance(scale, (int, float)) else list(scale)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_nonconvex_collision_from_file(filename=str(glb), scale=sc)
        builder.add_visual_from_file(filename=str(glb), scale=sc)
        builder.set_initial_pose(pose)
        actor = builder.build(name=model_name)
        try:
            for c in actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidStaticComponent):
                    break
            actor.set_name(model_name)
        except Exception:
            pass
        return actor

    def _spawn_static_prop(
        self,
        modelname: str,
        model_id: int,
        xy,
        *,
        scale_mult: float = 1.0,
        quat=None,
        z_lift: float = 0.001,
        pad: float = 0.03,
        name: str | None = None,
    ):
        """Static table décor seated on the tabletop."""
        q = list(quat) if quat is not None else list(self.PROP_UPRIGHT_Q)
        pose = sapien.Pose(
            [float(xy[0]), float(xy[1]), self.table_top + float(z_lift)],
            q,
        )
        actor = create_actor(
            self,
            pose=pose,
            modelname=modelname,
            model_id=int(model_id),
            convex=True,
            is_static=True,
            scale_mult=float(scale_mult),
        )
        if actor is None:
            return None
        if name:
            actor.set_name(name)
        try:
            self.add_prohibit_area(actor, padding=float(pad))
        except Exception:
            pass
        self.decor.append(actor)
        return actor

    def _decor_y_band(self):
        """Upper (far) 30% of the table Y span, in world coords."""
        y_max = self.TABLE_HALF_Y
        y_min = -self.TABLE_HALF_Y
        span = y_max - y_min
        lo = y_max - float(self.DECOR_Y_FRAC) * span
        return float(lo), float(y_max - 0.02)

    def _occupied_rects(self):
        """Axis-aligned keep-out boxes for mug / laptop / sponge / spill corridor."""
        rects = []
        for xy, hx, hy in (
            (self.mug_xy, 0.07, 0.07),
            (self.laptop_xy, 0.14, 0.12),
            (self.sponge_xy, 0.07, 0.07),
        ):
            rects.append(
                [
                    float(xy[0] - hx),
                    float(xy[1] - hy),
                    float(xy[0] + hx),
                    float(xy[1] + hy),
                ]
            )
        # Spill corridor from mug toward laptop.
        mid = 0.5 * (self.mug_xy + self.laptop_xy)
        along = max(0.10, float(np.linalg.norm(self.laptop_xy - self.mug_xy)) * 0.55)
        lat = 0.08
        # Approximate AABB of the corridor (axis-aligned; slightly oversized).
        rects.append(
            [
                float(min(self.mug_xy[0], self.laptop_xy[0]) - 0.06),
                float(min(self.mug_xy[1], self.laptop_xy[1]) - lat),
                float(max(self.mug_xy[0], self.laptop_xy[0]) + 0.06),
                float(max(self.mug_xy[1], self.laptop_xy[1]) + lat),
            ]
        )
        # Keep mid used so linters don't complain if along is unused later.
        _ = (mid, along)
        return rects

    @staticmethod
    def _rects_overlap(a, b, pad: float = 0.0) -> bool:
        return not (
            a[2] + pad <= b[0]
            or b[2] + pad <= a[0]
            or a[3] + pad <= b[1]
            or b[3] + pad <= a[1]
        )

    def _sample_decor_xy(self, half_xy, occupied, rng, max_tries: int = 80):
        """Sample a non-overlapping XY in the upper-Y décor band."""
        y_lo, y_hi = self._decor_y_band()
        hx, hy = float(half_xy[0]), float(half_xy[1])
        x_lo, x_hi = -self.TABLE_HALF_X + hx + 0.02, self.TABLE_HALF_X - hx - 0.02
        y_lo = max(y_lo, -self.TABLE_HALF_Y + hy + 0.02)
        y_hi = min(y_hi, self.TABLE_HALF_Y - hy - 0.02)
        if y_lo > y_hi or x_lo > x_hi:
            return None
        for _ in range(int(max_tries)):
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            rect = [x - hx, y - hy, x + hx, y + hy]
            if any(self._rects_overlap(rect, o, pad=0.015) for o in occupied):
                continue
            return np.array([x, y], dtype=float), rect
        return None

    def _spawn_decorations(self):
        """Plant, file holder, pen cup, and clock in the upper 30% Y band (no overlap)."""
        cfg = self._cfg
        self.decor = []
        randomize = bool(cfg.get("randomize_layout", True))
        rng = np.random.RandomState(
            int(cfg.get("decor_seed", self._spill_seed + 17))
        )
        occupied = self._occupied_rects()
        y_lo, y_hi = self._decor_y_band()
        back_y = float(cfg.get("decor_y", 0.5 * (y_lo + y_hi)))

        def _place_or_default(default_xy, half_xy, fallback_x: float | None = None):
            if not randomize:
                return np.array(default_xy, dtype=float), [
                    default_xy[0] - half_xy[0],
                    default_xy[1] - half_xy[1],
                    default_xy[0] + half_xy[0],
                    default_xy[1] + half_xy[1],
                ]
            hit = self._sample_decor_xy(half_xy, occupied, rng)
            if hit is not None:
                return hit
            # Fallback: try a few distinct slots in the décor band.
            candidates = []
            if fallback_x is not None:
                candidates.append(float(fallback_x))
            candidates.extend(
                [
                    -0.45 * self.laptop_side,
                    0.45 * self.laptop_side,
                    -0.30,
                    0.30,
                    0.0,
                    -0.48,
                    0.48,
                ]
            )
            for x0 in candidates:
                x = float(
                    np.clip(
                        x0,
                        -self.TABLE_HALF_X + half_xy[0],
                        self.TABLE_HALF_X - half_xy[0],
                    )
                )
                for y in (back_y, y_lo + 0.02, y_hi - 0.02, 0.5 * (y_lo + y_hi)):
                    y = float(np.clip(y, y_lo, y_hi))
                    rect = [
                        x - half_xy[0],
                        y - half_xy[1],
                        x + half_xy[0],
                        y + half_xy[1],
                    ]
                    if any(self._rects_overlap(rect, o, pad=0.015) for o in occupied):
                        continue
                    return np.array([x, y], dtype=float), rect
            # Last resort: accept default even if tight.
            x, y = float(default_xy[0]), float(default_xy[1])
            rect = [x - half_xy[0], y - half_xy[1], x + half_xy[0], y + half_xy[1]]
            return np.array([x, y], dtype=float), rect

        # Plant (pot).
        plant_half = (0.07, 0.07)
        plant_xy, plant_rect = _place_or_default(
            [float(cfg.get("plant_x", -0.44)), back_y],
            plant_half,
            fallback_x=-0.44,
        )
        occupied.append(plant_rect)
        plant_id = int(cfg.get("plant_id", 0))
        plant_scale = float(cfg.get("plant_scale", 0.55))
        self.plant = self._spawn_static_prop(
            "120_plant",
            plant_id,
            plant_xy,
            scale_mult=plant_scale,
            pad=0.04,
            name="120_plant",
        )

        # Office file holder.
        file_half = (0.12, 0.09)
        file_xy, file_rect = _place_or_default(
            [
                float(cfg.get("file_x", 0.02)),
                float(cfg.get("file_y", back_y + 0.01)),
            ],
            file_half,
            fallback_x=0.02,
        )
        occupied.append(file_rect)
        file_scale = cfg.get("file_scale", [0.38, 0.70, 0.40])
        file_q = cfg.get("file_quat", [0.7071, 0.7071, 0.0, 0.0])
        try:
            self.file_holder = self._create_bench_glb(
                "122_file-holder",
                sapien.Pose(
                    p=[float(file_xy[0]), float(file_xy[1]), self.table_top + 0.075],
                    q=list(file_q),
                ),
                scale=file_scale,
                mass=0.1,
            )
            self.decor.append(self.file_holder)
            self.prohibited_area.append(
                [
                    float(file_xy[0]) - 0.12,
                    float(file_xy[1]) - 0.09,
                    float(file_xy[0]) + 0.12,
                    float(file_xy[1]) + 0.09,
                ]
            )
        except Exception as e:
            print(f"[clean_table] file holder spawn failed: {e}")
            self.file_holder = None

        # Pen holder / pen cup.
        pen_half = (0.05, 0.05)
        pen_xy, pen_rect = _place_or_default(
            [float(cfg.get("pencup_x", 0.40)), back_y - 0.02],
            pen_half,
            fallback_x=0.40,
        )
        occupied.append(pen_rect)
        pen_id = int(cfg.get("pencup_id", 0))
        pen_scale = float(cfg.get("pencup_scale", 1.0))
        self.pencup = self._spawn_static_prop(
            "059_pencup",
            pen_id,
            pen_xy,
            scale_mult=pen_scale,
            pad=0.03,
            name="059_pencup",
        )

        # Alarm clock facing the robot (−Y).
        clock_half = (0.05, 0.05)
        clock_xy, clock_rect = _place_or_default(
            [float(cfg.get("clock_x", -0.18)), back_y - 0.04],
            clock_half,
            fallback_x=-0.18,
        )
        occupied.append(clock_rect)
        clock_id = int(cfg.get("clock_id", 0))
        clock_scale = float(cfg.get("clock_scale", 0.60))
        face_robot_q = t3d.quaternions.qmult(
            t3d.euler.euler2quat(0.0, 0.0, 0.5 * np.pi, axes="sxyz"),
            np.asarray(self.PROP_UPRIGHT_Q, dtype=float),
        )
        self.clock = self._spawn_static_prop(
            "046_alarm-clock",
            clock_id,
            clock_xy,
            scale_mult=clock_scale,
            quat=face_robot_q.tolist(),
            pad=0.03,
            name="046_alarm-clock",
        )
        print(
            f"[clean_table] décor band y=[{y_lo:.2f},{y_hi:.2f}] "
            f"plant=({plant_xy[0]:.2f},{plant_xy[1]:.2f}) "
            f"file=({file_xy[0]:.2f},{file_xy[1]:.2f}) "
            f"pencup=({pen_xy[0]:.2f},{pen_xy[1]:.2f}) "
            f"clock=({clock_xy[0]:.2f},{clock_xy[1]:.2f})"
        )

    # ------------------------------------------------------------------ spill visuals
    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    def _build_spill_spots(self):
        """Precompute overlapping spill lobes (one visual puddle, many clearables).

        Layout:
          1) Under-mug seep (visual only).
          2) Initial puddle — overlapping small lobes that form one larger stain
             past the rim (all spawn early). A single sponge press may clear
             several at once; that is intended.
          3) Growth stream toward the laptop — only spawns while dirt remains;
             growth freezes the moment the table is fully clean.

        Stream wobble / width / lobe count are seed-randomized so each episode
        spreads with a different organic pattern toward the laptop.
        """
        origin = self.spill_origin
        d = self.spill_dir
        lat = self.spill_lat
        path = float(self.spill_path_len)
        phase = self._lobe_phase
        pat = getattr(self, "_spill_pattern", None) or {}
        wobble1 = float(pat.get("wobble1", 0.012))
        wobble2 = float(pat.get("wobble2", 0.008))
        freq1 = float(pat.get("freq1", 2.1))
        freq2 = float(pat.get("freq2", 5.0))
        width_scale = float(pat.get("width_scale", 1.0))
        side_bias = float(pat.get("side_bias", 0.0))
        n_main = int(pat.get("n_main", 7))
        puddle_spread = float(pat.get("puddle_spread", 1.0))
        spots: list[dict] = []

        # Under-mug seep (visual only — dabbing there would hit the mug).
        spots.append(
            {
                "pos": origin - d * 0.025,
                "radius": 0.022,
                "spawn": 0.04,
                "cleaned": False,
                "along": -0.025,
                "under_mug": True,
            }
        )

        # Initial puddle: organic overlapping lobes (NOT a grid). Centers sit
        # well inside each other's radii so they read as one coffee stain.
        puddle_base = 0.030
        # (along, lat, radius) — irregular teardrop / amoeba cluster.
        puddle_lobes = (
            (0.000, 0.000, 0.028),
            (0.012, 0.016, 0.024),
            (0.014, -0.014, 0.025),
            (0.028, 0.004, 0.026),
            (0.022, 0.022, 0.022),
            (0.024, -0.020, 0.023),
            (0.040, -0.006, 0.024),
            (0.038, 0.018, 0.021),
            (0.036, -0.018, 0.021),
            (0.052, 0.002, 0.020),
            (0.048, 0.014, 0.018),
            (0.050, -0.012, 0.019),
        )
        for i, (along_off, lat_off, rad) in enumerate(puddle_lobes):
            along = puddle_base + along_off * puddle_spread
            # Small organic jitter — keep overlap, avoid lattice look.
            ja = 0.004 * np.sin(phase + 1.3 * i)
            jl = 0.005 * np.cos(phase * 0.7 + 0.9 * i) + side_bias * 0.01
            lat_scaled = lat_off * puddle_spread
            spots.append(
                {
                    "pos": origin + d * (along + ja) + lat * (lat_scaled + jl),
                    "radius": float(rad * (0.92 + 0.08 * width_scale)),
                    "spawn": float(0.05 + 0.008 * i),
                    "cleaned": False,
                    "along": float(along + ja),
                    "initial_puddle": True,
                }
            )

        # Growth stream past the puddle toward the laptop (only while dirty).
        stream_start = puddle_base + 0.055
        stream_len = max(0.05, path - stream_start)
        for i in range(n_main):
            t = (i + 0.35) / n_main
            along = stream_start + t * stream_len
            wobble = wobble1 * np.sin(freq1 * np.pi * t + phase)
            wobble += wobble2 * np.sin(freq2 * np.pi * t + 0.6 * phase)
            wobble += side_bias * 0.012 * t
            width = (0.018 + 0.016 * (np.sin(np.pi * min(t, 0.95)) ** 0.55)) * width_scale
            spots.append(
                {
                    "pos": origin + d * along + lat * wobble,
                    "radius": float(width),
                    "spawn": float(0.40 + 0.55 * t),
                    "cleaned": False,
                    "along": float(along),
                }
            )
            if 0.15 < t < 0.85:
                side = (0.018 + 0.012 * t) * (1.0 if i % 2 == 0 else -0.9)
                side += side_bias * 0.01
                spots.append(
                    {
                        "pos": origin + d * along + lat * (wobble + side),
                        "radius": float(width * 0.70),
                        "spawn": float(0.45 + 0.50 * t),
                        "cleaned": False,
                        "along": float(along),
                    }
                )

        # Leading tip.
        spots.append(
            {
                "pos": origin + d * path + lat * (0.008 * np.sin(phase) + side_bias * 0.006),
                "radius": 0.016 * width_scale,
                "spawn": 0.96,
                "cleaned": False,
                "along": path,
            }
        )
        self._spill_spots = spots

    def _table_fully_clean(self) -> bool:
        """True when every spawned wipeable lobe has been cleared."""
        wipeable = [s for s in self._active_spots() if not s.get("under_mug")]
        return bool(wipeable) and not self._dirty_spots()

    def _active_spots(self) -> list[dict]:
        """Spots that have spawned at the current spill_amount."""
        amt = float(self.spill_amount)
        return [s for s in self._spill_spots if amt >= float(s["spawn"])]

    def _dirty_spots(self) -> list[dict]:
        """Wipeable dirty lobes (excludes under-mug seep)."""
        return [
            s
            for s in self._active_spots()
            if (not s["cleaned"]) and (not s.get("under_mug"))
        ]

    def _dirty_visual_spots(self) -> list[dict]:
        """All uncleared active lobes, including under-mug (for rendering)."""
        return [s for s in self._active_spots() if not s["cleaned"]]

    def _spill_clean_frac(self) -> float:
        active = [
            s for s in self._active_spots() if not s.get("under_mug")
        ]
        if not active:
            return 0.0
        cleaned = sum(1 for s in active if s["cleaned"])
        return float(cleaned) / float(len(active))

    def _spill_dirty_frac(self) -> float:
        return 1.0 - self._spill_clean_frac()

    def _spill_center_xy(self) -> np.ndarray:
        dirty = self._dirty_spots()
        if not dirty:
            active = self._active_spots()
            if not active:
                return self.spill_origin.copy()
            pts = np.stack([s["pos"] for s in active], axis=0)
            return pts.mean(axis=0)
        # Prefer the nearest dirty spot to the sponge if available.
        try:
            sp = self._sponge_pad_world()[:2]
            return min(dirty, key=lambda s: float(np.linalg.norm(s["pos"] - sp)))["pos"].copy()
        except Exception:
            pts = np.stack([s["pos"] for s in dirty], axis=0)
            return pts.mean(axis=0)

    def _spill_front_along(self) -> float:
        dirty = self._dirty_spots()
        if not dirty:
            return 0.0
        return float(max(s["along"] + s["radius"] for s in dirty))

    def _spot_hits_laptop(self, spot: dict) -> bool:
        """True if a spill lobe disk intersects the laptop table footprint."""
        if self.laptop_xy is None:
            return False
        hx = float(getattr(self, "laptop_half_along", 0.10))
        hy = float(getattr(self, "laptop_half_lat", 0.08))
        c = np.asarray(self.laptop_xy, dtype=float)
        p = np.asarray(spot["pos"], dtype=float)
        r = float(spot["radius"])
        # Circle vs AABB: distance from center to box, then compare to radius.
        dx = max(abs(float(p[0] - c[0])) - hx, 0.0)
        dy = max(abs(float(p[1] - c[1])) - hy, 0.0)
        return (dx * dx + dy * dy) <= (r * r)

    def _latch_laptop_fail_score(self) -> None:
        """Freeze wipe progress as the partial score at the laptop-fail frame.

        The spill only stops growing once the table is fully clean, so a live
        read after the failure would keep sliding as new lobes spawn. Laptop
        contact remains a failure — this only preserves how much was cleaned.
        """
        frac = float(self._spill_clean_frac())
        self._partial_score = float(
            score_half_open_intervals(frac, self.PARTIAL_CLEAN_BANDS)
        )

    def _mark_laptop_reached_if_contact(self) -> bool:
        """Fail as soon as any active (spawned) wipeable lobe touches the laptop.

        Checked on dirty *and* already-cleaned active lobes: wiping coffee off
        the chassis after contact still counts as failure.
        """
        if self.laptop_reached:
            return True
        for s in self._active_spots():
            if s.get("under_mug"):
                continue
            if self._spot_hits_laptop(s):
                self._latch_laptop_fail_score()
                self.laptop_reached = True
                print(
                    f"[clean_table] FAIL spill contacted laptop "
                    f"along={float(s.get('along', 0.0)):.3f} "
                    f"r={float(s['radius']):.3f} "
                    f"amt={self.spill_amount:.2f} "
                    f"cleaned={bool(s.get('cleaned'))} "
                    f"clean_frac={self._spill_clean_frac():.2f} "
                    f"score={self._partial_score:.2f}"
                )
                return True
        return False

    def _spill_radius(self) -> float:
        dirty = self._dirty_spots()
        if not dirty:
            return 0.03
        return float(max(s["radius"] for s in dirty))

    def _spill_visual_key(self):
        """Cache key: which spots are visible (spawned and not cleaned)."""
        amt = float(self.spill_amount)
        return tuple(
            (i, bool(s["cleaned"]))
            for i, s in enumerate(self._spill_spots)
            if amt >= float(s["spawn"])
        )

    def _rebuild_spill(self, force: bool = False):
        key = self._spill_visual_key()
        cached = getattr(self, "_spill_visual_cached", None)
        if not force and key == cached and self._spill_entity is not None:
            return
        if not force and key == cached and not key:
            return
        self._spill_visual_cached = key

        dirty = self._dirty_visual_spots()
        self._spill_entity = self._remove_entity(self._spill_entity)
        if not dirty:
            return

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=list(self.SPILL_COLOR))
        try:
            mat.set_roughness(0.55)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.55

        for s in dirty:
            local = s["pos"] - self.spill_origin
            builder.add_cylinder_visual(
                pose=sapien.Pose(
                    [float(local[0]), float(local[1]), 0.0], self.VERTICAL_CYL_Q
                ),
                radius=max(0.008, float(s["radius"])),
                half_length=0.0016,
                material=mat,
            )
        z = self.table_top + 0.0022
        builder.set_initial_pose(
            sapien.Pose(
                p=[float(self.spill_origin[0]), float(self.spill_origin[1]), z]
            )
        )
        self._spill_entity = builder.build(name="coffee_spill")

    def _tip_axis(self) -> np.ndarray:
        """World axis to tip the mug so its opening falls toward the laptop."""
        # cross(+Z, spill_dir) → tip axis in the table plane.
        axis = np.array([-self.spill_dir[1], self.spill_dir[0], 0.0], dtype=float)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis /= n
        return axis

    def _set_mug_pose(self, pose: sapien.Pose) -> None:
        self.mug.actor.set_pose(pose)
        if self._mug_rigid is not None:
            try:
                if self._mug_rigid.kinematic:
                    self._mug_rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _animate_tip(self, n_steps: int = 24):
        """Tip the mug onto its side with the opening toward the laptop.

        ``tip_axis = Z × spill_dir`` already flips when the laptop mirrors, so
        a fixed +90° swings the mouth onto spill_dir for both left and right
        laptop placements (same as the original working left-laptop scene).
        """
        if bool(getattr(self, "cup_tipped", False)):
            return
        upright = np.asarray(self.MUG_UPRIGHT_Q, dtype=float)
        axis = self._tip_axis()
        tip_ang = 0.5 * np.pi
        tipped_q = t3d.quaternions.qmult(
            t3d.quaternions.axangle2quat(axis, tip_ang), upright
        )
        z0 = self.table_top + 0.001
        z1 = self.table_top + float(self.mug_radius) + 0.002
        p0 = self.mug_xy.copy()
        # Center sits behind the rim along −spill_dir once opening rests on spill_origin.
        p1 = self.spill_origin - self.spill_dir * float(self.mug_half_height)

        self._make_kinematic(self.mug)
        for i in range(1, int(n_steps) + 1):
            t = i / float(n_steps)
            ang = t * tip_ang
            q = t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat(axis, ang), upright
            )
            p = p0 + t * (p1 - p0)
            z = z0 + t * (z1 - z0)
            self._set_mug_pose(
                sapien.Pose([float(p[0]), float(p[1]), z], q.tolist())
            )
            if t > 0.55 and self._coffee_entity is not None:
                self._coffee_entity = self._remove_entity(self._coffee_entity)
            if t > 0.65 and not self.spill_active:
                self.cup_tipped = True
                self.spill_active = True
                self._latch_spill_start()
                # Reveal the full initial multi-spot puddle immediately.
                puddle = float(
                    getattr(self, "initial_puddle_level", self.INITIAL_PUDDLE_LEVEL_DEFAULT)
                )
                self.spill_amount = max(self.spill_amount, 0.5 * puddle)
                self.max_spill_amount = max(self.max_spill_amount, self.spill_amount)
                self._rebuild_spill(force=True)
            self._idle_steps(1)

        self._set_mug_pose(
            sapien.Pose([float(p1[0]), float(p1[1]), z1], tipped_q.tolist())
        )
        self._coffee_entity = self._remove_entity(self._coffee_entity)
        self.cup_tipped = True
        self.spill_active = True
        self._latch_spill_start()
        puddle = float(
            getattr(self, "initial_puddle_level", self.INITIAL_PUDDLE_LEVEL_DEFAULT)
        )
        self.spill_amount = max(self.spill_amount, puddle)
        self.max_spill_amount = max(self.max_spill_amount, self.spill_amount)
        self._rebuild_spill(force=True)
        # Release to real dynamics so arm/sponge hits can shove the mug.
        self._enable_mug_physics()
        print("[clean_table] tip animation done — mug fell toward laptop / spill")

    # ------------------------------------------------------------------ dynamics
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _make_kinematic(self, entity, mass=None):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            if mass is not None:
                rigid.set_mass(float(mass))
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        return rigid

    def _enable_mug_physics(self):
        """Dynamic mug on the table — moves when the arm or sponge collides."""
        rigid = self._get_rigid(self.mug)
        self._mug_rigid = rigid
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            # Heavier + damped so light dabs don't drag it across the spill,
            # but a solid arm hit still displaces it.
            rigid.set_mass(float(self._cfg.get("mug_mass", max(self.MUG_MASS, 0.32))))
            rigid.set_linear_damping(2.8)
            rigid.set_angular_damping(3.0)
            try:
                rigid.set_max_linear_velocity(0.8)
            except Exception:
                pass
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            for shape in rigid.get_collision_shapes():
                try:
                    m = shape.get_physical_material()
                    m.set_static_friction(0.75)
                    m.set_dynamic_friction(0.60)
                    m.set_restitution(0.0)
                except Exception:
                    pass
        except Exception:
            pass
        # Brief settle so the tipped pose stops bouncing before wiping.
        self._idle_steps(12)

    def _mug_xy_now(self) -> np.ndarray:
        try:
            return np.asarray(self.mug.get_pose().p[:2], dtype=float)
        except Exception:
            return self.mug_xy.copy()

    def _spot_clear_of_mug(self, spot: dict) -> bool:
        """True if dabbing this spot won't drive the sponge into the mug."""
        mug_xy = self._mug_xy_now()
        avoid_r = (
            float(self.mug_radius)
            + float(self.SPONGE_HALF[0])
            + float(self.mug_avoid_margin)
        )
        dist = float(np.linalg.norm(np.asarray(spot["pos"], dtype=float) - mug_xy))
        return dist >= avoid_r

    def _dab_target_xy(self, spot: dict) -> np.ndarray:
        """XY to dab: push contact slightly away from the mug if the lobe is near."""
        target = np.asarray(spot["pos"], dtype=float).copy()
        mug_xy = self._mug_xy_now()
        delta = target - mug_xy
        dist = float(np.linalg.norm(delta))
        min_r = (
            float(self.mug_radius)
            + float(self.SPONGE_HALF[0])
            + float(self.mug_avoid_margin)
        )
        if dist < min_r and dist > 1e-6:
            # Prefer pushing along the spill (away from mug body / toward laptop).
            push = self.spill_dir * (min_r - dist + 0.01)
            target = target + push
        elif dist < min_r:
            target = target + self.spill_dir * min_r
        return target

    def _tcp_pos(self, arm: ArmTag) -> np.ndarray:
        p = (
            self.robot.get_right_tcp_pose()
            if str(arm) == "right"
            else self.robot.get_left_tcp_pose()
        )
        return np.asarray(p[:3], dtype=float)

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _set_sponge_pose(self, pose: sapien.Pose) -> None:
        """Stamp sponge pose on the entity + kinematic target (make_soup style)."""
        try:
            self.sponge.actor.set_pose(pose)
        except Exception:
            pass
        if self._sponge_rigid is not None:
            try:
                self._sponge_rigid.set_disable_gravity(True)
                if not self._sponge_rigid.kinematic:
                    self._sponge_rigid.set_kinematic(True)
                self._sponge_rigid.set_linear_velocity(np.zeros(3))
                self._sponge_rigid.set_angular_velocity(np.zeros(3))
                self._sponge_rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _freeze_sponge(self, pose: sapien.Pose | None = None) -> None:
        if self._sponge_rigid is not None:
            try:
                self._sponge_rigid.set_disable_gravity(True)
                self._sponge_rigid.set_kinematic(True)
                self._sponge_rigid.set_linear_velocity(np.zeros(3))
                self._sponge_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        if pose is not None:
            self._set_sponge_pose(pose)

    def _sponge_handle_world(self) -> np.ndarray:
        """World position of the small top handle (grasp target)."""
        sp = self.sponge.get_pose()
        p = np.asarray(sp.p, dtype=float)
        R = t3d.quaternions.quat2mat(np.asarray(sp.q, dtype=float))
        hz = float(getattr(self, "_sponge_handle_z", self.SPONGE_HALF[2] + self.SPONGE_HANDLE_HALF[2]))
        return p + R @ np.array([0.0, 0.0, hz], dtype=float)

    def _hold_ee_quat(self) -> list[float]:
        """Wrist orientation used for grasp and every dab (pad faces table)."""
        if self._sponge_hold_quat is not None:
            return list(self._sponge_hold_quat)
        return [float(v) for v in GRASP_DIRECTION_DIC["top_down"]]

    def _ee_pose_at_tcp(self, tcp_xyz) -> list[float]:
        """EE target for a top-down TCP location (ur5-wsg: EE sits above TCP)."""
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]) + float(self.EE_TO_TCP),
            *self._hold_ee_quat(),
        ]

    def _move_tcp_abs(self, tcp_xyz) -> bool:
        """Absolute top-down move; keeps the sponge pad parallel to the table."""
        self.plan_success = True
        self.move(self.move_to_pose(self.arm, self._ee_pose_at_tcp(tcp_xyz)))
        ok = bool(self.plan_success)
        if not ok:
            self.plan_success = True
        self._sync_welded_sponge()
        return ok

    def _weld_sponge_to_ee(self, arm: ArmTag) -> bool:
        tcp = self._tcp_pos(arm)
        handle = self._sponge_handle_world()
        dist = float(np.linalg.norm(tcp - handle))
        if dist > 0.12:
            print(f"[clean_table] refuse sponge weld — tcp-handle={dist:.3f}")
            return False
        self._freeze_sponge()
        self._sponge_weld_offset = self._ee_pose(arm).inv() * self.sponge.get_pose()
        self._sponge_welded = True
        print(f"[clean_table] sponge welded via handle (tcp-handle={dist:.3f})")
        return True

    def _set_shape_collision_enabled(self, shape, enabled: bool) -> None:
        if shape is None:
            return
        try:
            # PhysX contact filter: [1,1,0,0] still collides with the WSG fingers
            # on this build — only an all-zero mask lets the pinch close through
            # the yellow pad. Re-enable with the default [1,1,1,1] mask.
            if enabled:
                shape.set_collision_groups([1, 1, 1, 1])
            else:
                shape.set_collision_groups([0, 0, 0, 0])
        except Exception:
            pass

    def _set_pad_collision_enabled(self, enabled: bool) -> None:
        """Toggle only the wide yellow pad — leave the handle pinchable."""
        self._set_shape_collision_enabled(
            getattr(self, "_sponge_pad_shape", None), enabled
        )

    def _set_sponge_collision_enabled(self, enabled: bool) -> None:
        """While held, ignore collisions so PhysX cannot yank the pad off the EE."""
        if self._sponge_rigid is None:
            return
        try:
            for shape in self._sponge_rigid.get_collision_shapes():
                self._set_shape_collision_enabled(shape, enabled)
        except Exception:
            pass

    def _seat_sponge_in_hand(self) -> None:
        """Pad-flat sponge seated under the current TCP (handle in the fingers)."""
        tcp = self._tcp_pos(self.arm)
        hz = float(
            getattr(
                self,
                "_sponge_handle_z",
                self.SPONGE_HALF[2] + self.SPONGE_HANDLE_HALF[2],
            )
        )
        # Pad center is directly below the gripped handle / TCP.
        sponge_p = np.array(
            [float(tcp[0]), float(tcp[1]), float(tcp[2]) - hz], dtype=float
        )
        sponge_p[2] = max(float(sponge_p[2]), self._contact_z())
        upright = sapien.Pose(sponge_p.tolist(), [1, 0, 0, 0])
        self._set_sponge_collision_enabled(False)
        self._freeze_sponge(upright)
        self._sponge_weld_offset = self._ee_pose(self.arm).inv() * upright
        self._sponge_welded = True
        self._sync_welded_sponge()

    def _adopt_top_down_hold(self) -> None:
        """Pad flat under the TCP — same seating used to pick and to dab.

        Do not re-issue ``close_gripper`` here: a second close on an already
        pinched WSG flings the arm to a wild IK pose. The grasp path already
        closed on the handle; just seat the pad under the current TCP.
        """
        self._sponge_hold_quat = [float(v) for v in GRASP_DIRECTION_DIC["top_down"]]
        self._seat_sponge_in_hand()
        print("[clean_table] sponge seated on handle for vertical dabs")

    def _sponge_pad_world(self) -> np.ndarray:
        """World pad-center of the yellow sponge entity (source of wipe contact)."""
        return np.asarray(self.sponge.get_pose().p, dtype=float)

    def _sync_welded_sponge(self) -> None:
        """Keep the real yellow pad glued under the TCP while held."""
        if not self._sponge_welded:
            return
        tcp = self._tcp_pos(self.arm)
        hz = float(
            getattr(
                self,
                "_sponge_handle_z",
                self.SPONGE_HALF[2] + self.SPONGE_HANDLE_HALF[2],
            )
        )
        p = np.array(
            [float(tcp[0]), float(tcp[1]), float(tcp[2]) - hz], dtype=float
        )
        # Keep the pad above the table when the TCP dips, but never invent a
        # table-height pad while the arm is still high (that cleared stains from
        # mid-air proximity).
        p[2] = max(float(p[2]), self.table_top + float(self.SPONGE_HALF[2]))
        flat = sapien.Pose(p.tolist(), [1, 0, 0, 0])
        self._set_sponge_pose(flat)
        self._sponge_weld_offset = self._ee_pose(self.arm).inv() * flat

    def _contact_z(self) -> float:
        return self.table_top + float(self.SPONGE_HALF[2]) + 0.002

    def _safe_z(self) -> float:
        return self.table_top + float(self.wipe_safe_z)

    def _handle_tcp_z(self, sponge_z: float) -> float:
        """TCP height when the handle is gripped and pad center is at sponge_z."""
        return float(sponge_z) + float(
            getattr(self, "_sponge_handle_z", self.SPONGE_HALF[2] + self.SPONGE_HANDLE_HALF[2])
        )

    def _drive_tcp_toward(self, target, max_steps: int = 16, step: float = 0.04) -> None:
        """Nudge the arm so TCP approaches ``target`` (world XYZ), top-down."""
        for _ in range(int(max_steps)):
            tcp = self._tcp_pos(self.arm)
            d = np.asarray(target, dtype=float) - tcp
            if float(np.linalg.norm(d)) < 0.015:
                return
            nxt = tcp + np.clip(d, -step, step)
            if not self._move_tcp_abs(nxt):
                self._move_ok(
                    dx=float(np.clip(d[0], -step, step)),
                    dy=float(np.clip(d[1], -step, step)),
                    dz=float(np.clip(d[2], -step, step)),
                    quat=self._hold_ee_quat(),
                )

    def _top_down_pose(self, tcp_xyz) -> list[float]:
        """EE pose with TCP at ``tcp_xyz`` and wrist locked top-down."""
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]) + float(self.EE_TO_TCP),
            *[float(v) for v in GRASP_DIRECTION_DIC["top_down"]],
        ]

    def _grasp_sponge(self) -> bool:
        """Pick the small top cube — same pattern as ``make_soup._grasp_board``.

        Order matters (this is what was wrong before):
          1) OPEN gripper
          2) hover straight ABOVE the cube (no side / outside approach)
          3) lower onto the cube
          4) CLOSE only then
          5) seat + lift
        """
        arm = self.arm
        spawn = self._sponge_spawn_pose
        self._freeze_sponge(spawn)
        self._sponge_hold_quat = [float(v) for v in GRASP_DIRECTION_DIC["top_down"]]

        open_pos = float(self._cfg.get("sponge_grasp_open", self.SPONGE_GRASP_OPEN))
        close_pos = float(
            self._cfg.get("sponge_grasp_close", self.SPONGE_GRASP_CLOSE)
        )
        tcp_tol = float(self._cfg.get("grasp_tcp_tol", self.GRASP_TCP_TOL))

        # Pad must use the all-zero collision mask ([1,1,0,0] still blocks the
        # WSG). Keep the handle collidable so fingers stop on the cube.
        self._set_pad_collision_enabled(False)
        self._set_shape_collision_enabled(
            getattr(self, "_sponge_handle_shape", None), True
        )

        # 1) Open first — never approach the cube with a closed gripper.
        self.plan_success = True
        self.move(self.open_gripper(arm, pos=open_pos))
        self._idle_steps(2)
        self._freeze_sponge(spawn)

        handle = self._sponge_handle_world()

        # 2) Hover straight above the cube (make_soup style).
        hover = False
        for z_off, dx, dy in (
            (0.14, 0.0, 0.0),
            (0.18, 0.0, 0.0),
            (0.14, -0.02, 0.02),
            (0.22, 0.0, 0.0),
        ):
            target = np.array(
                [handle[0] + dx, handle[1] + dy, handle[2] + z_off], dtype=float
            )
            self.plan_success = True
            self._freeze_sponge(spawn)
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(target))])
            )
            if self.plan_success:
                hover = True
                print(f"[clean_table] handle hover ok z_off={z_off}")
                break
        if not hover:
            print("[clean_table] handle hover unreachable")
            self._set_sponge_collision_enabled(True)
            self._set_pad_collision_enabled(True)
            return False

        # 3) Descend onto the cube with relative world steps (make_soup style).
        handle = self._sponge_handle_world()
        goal = np.array(
            [handle[0], handle[1], float(handle[2]) + 0.012], dtype=float
        )
        top_down = list(GRASP_DIRECTION_DIC["top_down"])
        for _ in range(6):
            self._freeze_sponge(spawn)
            delta = goal - self._tcp_pos(arm)
            if float(np.linalg.norm(delta)) < 0.010:
                break
            step = np.clip(delta, -0.05, 0.05)
            before = self._tcp_pos(arm)
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(step[0]),
                    y=float(step[1]),
                    z=float(step[2]),
                    quat=top_down,
                    move_axis="world",
                )
            )
            after = self._tcp_pos(arm)
            if (not self.plan_success) or float(np.linalg.norm(after - before)) > 0.20:
                print("[clean_table] handle descent failed")
                self.plan_success = True
                self._set_sponge_collision_enabled(True)
                return False

        # 4) Close only once the TCP is on the cube.
        dist = float(np.linalg.norm(self._tcp_pos(arm) - self._sponge_handle_world()))
        if dist > tcp_tol * 2.5:
            print(f"[clean_table] refuse close — TCP far from handle ({dist:.3f} m)")
            self._set_sponge_collision_enabled(True)
            return False
        self.plan_success = True
        self.move(self.close_gripper(arm, pos=close_pos))
        self._idle_steps(6)

        dist = float(np.linalg.norm(self._tcp_pos(arm) - self._sponge_handle_world()))
        if dist > tcp_tol * 2.5:
            print(f"[clean_table] refuse weld — TCP far from handle ({dist:.3f} m)")
            self._set_sponge_collision_enabled(True)
            return False

        # 5) Seat pad under TCP (disables all sponge contacts while held).
        self._seat_sponge_in_hand()
        self._lift_sponge()
        ok = bool(self._sponge_welded)
        if not ok:
            self._set_sponge_collision_enabled(True)
        else:
            print(f"[clean_table] grasped handle cube tcp-dist={dist:.3f} arm={arm}")
        return ok

    def _sponge_on_table(self, sp: np.ndarray | None = None) -> bool:
        """True when the yellow pad bottom is pressing the tabletop."""
        if sp is None:
            sp = self._sponge_pad_world()
        pad_bottom = float(sp[2]) - float(self.SPONGE_HALF[2])
        return pad_bottom <= self.table_top + float(
            getattr(self, "SPONGE_TABLE_CONTACT_Z", 0.012)
        )

    def _spot_touching(self, spot: dict, sp: np.ndarray) -> bool:
        """True when the sponge pad footprint overlaps this spot on the table."""
        if not self._sponge_on_table(sp):
            return False
        # Axis-aligned pad box vs circular stain.
        hx = float(self.SPONGE_HALF[0]) + float(self.spot_contact_extra)
        hy = float(self.SPONGE_HALF[1]) + float(self.spot_contact_extra)
        r = float(spot["radius"])
        dx = abs(float(sp[0]) - float(spot["pos"][0]))
        dy = abs(float(sp[1]) - float(spot["pos"][1]))
        return dx <= (hx + r) and dy <= (hy + r)

    def _clear_spot(self, spot: dict) -> None:
        """Mark one spill lobe cleaned and refresh the visible puddle."""
        if spot.get("cleaned"):
            return
        spot["cleaned"] = True
        self._latch_wipe_metric()
        active = [s for s in self._active_spots() if not s.get("under_mug")]
        self.spill_cleaned = (
            float(sum(1 for s in active if s["cleaned"])) / max(1, len(active))
        )
        print(
            f"[clean_table] cleared spot along={spot['along']:.3f} "
            f"r={spot['radius']:.3f} dirty_left={len(self._dirty_spots())}"
        )
        # Fully clean → stop growth immediately (no new lobes after a wipe-out).
        if self._table_fully_clean():
            self._freeze_spill_growth("table clean")
            self.cleaned_ok = True
        self._rebuild_spill(force=True)

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_spill_step = None    # mug tipped, spill went live
        self._metric_first_wipe = None    # first lobe cleared
        self._metric_clear_step = None    # table first fully clean
        self._metric_worst_front = 0.0    # closest the spill front ever got to the laptop
        self._metric_wipes = 0

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_spill_start(self):
        try:
            if self._metric_spill_step is None:
                self._metric_spill_step = self._metric_step()
        except Exception:
            pass

    def _latch_wipe_metric(self):
        """Called from _clear_spot, before the clean/freeze bookkeeping runs."""
        try:
            self._metric_wipes += 1
            if self._metric_first_wipe is None:
                self._metric_first_wipe = self._metric_step()
        except Exception:
            pass

    def _track_spill_metrics(self):
        """High-water mark on the spill front, and the first fully-clean frame.

        ``spill_cleaned`` is recomputed against the *currently active* lobes, so it
        drops back as new lobes spawn — the front high-water mark is the value that
        survives.
        """
        try:
            if not getattr(self, "spill_active", False):
                return
            front = float(self._spill_front_along())
            frac = front / max(1e-6, float(self.spill_path_len))
            if frac > self._metric_worst_front:
                self._metric_worst_front = float(frac)
            if (self._metric_clear_step is None
                    and self._metric_wipes > 0
                    and self._table_fully_clean()):
                self._metric_clear_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `first_wipe_latency_steps` — steps from the mug tipping until the first
        spill lobe is dabbed. The spill keeps spreading toward the laptop the whole
        time, so this is the reaction window that actually matters.
        `clear_latency_steps` reports how long the full clean-up took.
        extra2 `spill_front_norm` — the closest the dirty spill front ever got to the
        laptop, as a fraction of the mug->laptop path. LOWER is better; >= 1.0 means it
        arrived (`laptop_reached`), and the task's own failure threshold is
        ``reach_laptop_level``.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a = self._metric_spill_step
        b = self._metric_first_wipe
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["first_wipe_latency_steps"] = lat
        out["first_wipe_latency_s"] = None if lat is None else round(float(lat) * dt, 4)

        c = self._metric_clear_step
        clr = None if (a is None or c is None) else max(int(c) - int(a), 0)
        out["clear_latency_steps"] = clr
        out["clear_latency_s"] = None if clr is None else round(float(clr) * dt, 4)
        out["wipes_counted"] = int(self._metric_wipes)

        out["spill_front_norm"] = round(float(self._metric_worst_front), 4)
        try:
            out["reach_laptop_level"] = round(float(self.reach_laptop_level), 4)
            out["max_spill_amount"] = round(float(self.max_spill_amount), 4)
        except Exception:
            out["reach_laptop_level"] = None
            out["max_spill_amount"] = None
        out["laptop_reached"] = bool(getattr(self, "laptop_reached", False))
        return out

    def _freeze_spill_growth(self, reason: str = "") -> None:
        """Once the table is clean, no new lobes may spawn."""
        if self._spill_frozen:
            return
        self._spill_frozen = True
        print(
            f"[clean_table] spill growth frozen"
            + (f" ({reason})" if reason else "")
            + f" amt={self.spill_amount:.2f}"
        )

    def _try_clear_spots_under_sponge(self) -> int:
        """Clear dirty spots the yellow pad is pressing on the table.

        Runs for expert and interactive play — no ``_expert_wiping`` gate. Contact
        uses the sponge entity pose (pad on table + XY overlap), not arm proximity.
        """
        if not self.spill_active or self.laptop_reached:
            self._wipe_contact_map = {}
            self._wipe_contact_steps = 0
            return 0
        try:
            sp = self._sponge_pad_world()
        except Exception:
            self._wipe_contact_map = {}
            self._wipe_contact_steps = 0
            return 0

        if not self._sponge_on_table(sp):
            self._wipe_contact_map = {}
            self._wipe_contact_steps = 0
            return 0

        # Expert dabs may arm a single target; otherwise wipe every overlapping lobe.
        dirty = self._dirty_spots()
        if self._wipe_armed and self._wipe_target is not None:
            target = self._wipe_target
            candidates = (
                [target]
                if (not target.get("cleaned")) and target in dirty
                else []
            )
            # Also clear any other lobes the pad covers in the same press.
            for s in dirty:
                if s is not target and self._spot_touching(s, sp):
                    candidates.append(s)
        else:
            candidates = [s for s in dirty if self._spot_touching(s, sp)]

        if not candidates:
            self._wipe_contact_map = {}
            self._wipe_contact_steps = 0
            return 0

        dwell = max(1, int(self.spot_clear_dwell))
        if not getattr(self, "_expert_wiping", False):
            # Interactive teleop: clear promptly once the pad is on the stain.
            dwell = max(1, min(dwell, 3))

        if not isinstance(getattr(self, "_wipe_contact_map", None), dict):
            self._wipe_contact_map = {}

        active_ids = {id(s) for s in candidates}
        for sid in list(self._wipe_contact_map.keys()):
            if sid not in active_ids:
                del self._wipe_contact_map[sid]

        cleared = 0
        for spot in candidates:
            if not self._spot_touching(spot, sp):
                continue
            sid = id(spot)
            self._wipe_contact_map[sid] = int(self._wipe_contact_map.get(sid, 0)) + 1
            if self._wipe_contact_map[sid] < dwell:
                continue
            self._clear_spot(spot)
            self._wipe_contact_map.pop(sid, None)
            cleared += 1
        self._wipe_contact_steps = cleared
        return cleared

    def _step_spill(self):
        if not self.spill_active or self.laptop_reached:
            return
        prev_key = getattr(self, "_spill_visual_cached", None)

        # Geometric contact BEFORE wipe: clearing a lobe on the chassis must
        # not erase the failure (coffee already reached the laptop).
        self._mark_laptop_reached_if_contact()
        if self.laptop_reached:
            self._rebuild_spill(force=self._spill_visual_key() != prev_key)
            return

        # Clear first, then decide freeze/grow. Growing before clear let new
        # lobes spawn on the step after the table looked empty.
        self._try_clear_spots_under_sponge()

        if not self._spill_frozen:
            if self._table_fully_clean():
                # No dirt left → never spawn another lobe.
                self._freeze_spill_growth("table clean")
                self.cleaned_ok = True
            else:
                # Dirt remains → keep spreading (slower while the expert dabs).
                rate = 1.0 / max(1, self.spill_steps)
                if getattr(self, "_expert_wiping", False):
                    rate *= 0.35
                self.spill_amount = min(1.0, self.spill_amount + rate)
                self.max_spill_amount = max(self.max_spill_amount, self.spill_amount)

        # New lobes may spawn into the laptop; also keep the old fractional
        # front check as a backup for long-gap stream tips.
        self._mark_laptop_reached_if_contact()
        if (not self.laptop_reached) and (not self._spill_frozen):
            front = self._spill_front_along()
            front_frac = front / max(1e-6, float(self.spill_path_len))
            if (
                self.spill_amount >= self.reach_laptop_level
                and front_frac >= self.reach_laptop_level
                and self._dirty_spots()
            ):
                self._latch_laptop_fail_score()
                self.laptop_reached = True
                print(
                    f"[clean_table] FAIL spill reached laptop "
                    f"front={front:.3f} path={self.spill_path_len:.3f} "
                    f"amt={self.spill_amount:.2f} dirty={len(self._dirty_spots())} "
                    f"clean_frac={self._spill_clean_frac():.2f} "
                    f"score={self._partial_score:.2f}"
                )

        # Rebuild when new spots spawn (growth) even if nothing was cleared.
        if self._spill_visual_key() != prev_key:
            self._rebuild_spill(force=True)
        else:
            self._rebuild_spill(force=False)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._sync_welded_sponge()
        self._step_spill()
        self._track_spill_metrics()

    def _update_render(self):
        # Always seat the pad under the TCP before cameras sample the scene.
        self._sync_welded_sponge()
        return super()._update_render()

    def _take_picture(self):
        self._sync_welded_sponge()
        return super()._take_picture()

    def _idle_steps(self, n_steps: int, until=None):
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            if until is not None and until():
                break
            self._update_kinematic_tasks()
            self.scene.step()
            # Re-stamp after the physics step so the pad cannot drift off the EE.
            self._sync_welded_sponge()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    # ------------------------------------------------------------------ expert motion
    def _move_ok(self, dx=0.0, dy=0.0, dz=0.0, quat=None) -> bool:
        # Prefer keeping the current wrist (quat=None). Forcing top_down after a
        # narrow pinch often teleports the WSG; only override when caller asks.
        before = self._tcp_pos(self.arm)
        self.plan_success = True
        self.move(
            self.move_by_displacement(
                self.arm,
                x=float(dx),
                y=float(dy),
                z=float(dz),
                quat=quat,
                move_axis="world",
            )
        )
        ok = bool(self.plan_success)
        after = self._tcp_pos(self.arm)
        jump = float(np.linalg.norm(after - before))
        # Reject teleports from bad wrist configs so wipe can try another step.
        if ok and jump > 0.25:
            ok = False
        if not ok:
            self.plan_success = True
            return False
        self._sync_welded_sponge()
        return ok

    def _next_dirty_spot(self) -> dict | None:
        """Next wipeable spot: rim→laptop."""
        dirty = self._dirty_spots()
        if not dirty:
            return None
        return min(dirty, key=lambda s: (float(s["along"]), float(s["spawn"])))

    def _lift_sponge(self) -> None:
        """Raise sponge straight up (top-down wrist locked)."""
        target_z = self._safe_z()
        for _ in range(8):
            sp = self._sponge_pad_world()
            dz = target_z - float(sp[2])
            if abs(dz) < 0.015:
                return
            self._move_ok(dz=float(np.clip(dz, -0.07, 0.07)))

    def _translate_above(self, target_xy, max_steps: int = 18) -> None:
        """XY translate at safe height; wrist stays top-down (no sponge twist)."""
        target_z = self._safe_z()
        for _ in range(int(max_steps)):
            sp = self._sponge_pad_world()
            d = np.array(
                [target_xy[0] - sp[0], target_xy[1] - sp[1], target_z - sp[2]],
                dtype=float,
            )
            if float(np.linalg.norm(d[:2])) < 0.014 and abs(d[2]) < 0.015:
                return
            self._move_ok(
                dx=float(np.clip(d[0], -0.055, 0.055)),
                dy=float(np.clip(d[1], -0.055, 0.055)),
                dz=float(np.clip(d[2], -0.05, 0.05)),
            )

    def _lower_onto_table(self, max_steps: int = 12) -> None:
        """Vertical Z-only press — pad stays parallel to the table."""
        target_z = self._contact_z()
        for _ in range(int(max_steps)):
            sp = self._sponge_pad_world()
            dz = target_z - float(sp[2])
            if abs(dz) < 0.008:
                break
            self._move_ok(dz=float(np.clip(dz, -0.04, 0.04)))
        self._idle_steps(2)

    def _dab_spot(self, spot: dict) -> bool:
        """Lift → hover → vertical dab → lift. Same top-down posture as grasp.

        Wrist orientation is locked; only XYZ changes. No sponge flip / twist.
        """
        self._wipe_target = spot
        self._wipe_armed = False
        self._wipe_contact_steps = 0
        target = self._dab_target_xy(spot)

        self._lift_sponge()
        self._translate_above(target)

        # If still far from the dab XY, try a lateral offset around the mug.
        sp = self._sponge_pad_world()
        if float(np.linalg.norm(sp[:2] - target)) > 0.04:
            lat = self.spill_lat
            for sign in (1.0, -1.0, 1.5, -1.5):
                alt = target + lat * (0.035 * sign)
                self._translate_above(alt, max_steps=12)
                sp = self._sponge_pad_world()
                if float(np.linalg.norm(sp[:2] - alt)) < 0.03:
                    target = alt
                    break

        # Pure vertical touch — pad perpendicular to the table.
        self._lower_onto_table()

        self._wipe_armed = True
        dwell_need = max(1, int(self.spot_clear_dwell))
        for _ in range(dwell_need + 6):
            if spot["cleaned"] or self.laptop_reached:
                break
            self._idle_steps(1)

        # Clear every dirty lobe currently under the pad (one vertical press).
        if not self.laptop_reached:
            sp = self._sponge_pad_world()
            if self._sponge_on_table(sp):
                for s in list(self._dirty_spots()):
                    if self._spot_touching(s, sp):
                        self._clear_spot(s)

        self._wipe_armed = False
        self._wipe_target = None
        self._lift_sponge()
        return bool(spot["cleaned"])

    def _wipe_spill(self):
        """Dab each lobe rim→laptop: lift, touch table, lift. Avoid the mug."""
        self._expert_wiping = True
        self._wipe_armed = False
        self._wipe_target = None
        self._wipe_contact_steps = 0
        self._last_failed_along = None
        # Under-mug seep is unreachable without hitting the cup — clear for success
        # after it has been visible during the spill growth phase.
        for s in self._spill_spots:
            if s.get("under_mug") and not s["cleaned"]:
                s["cleaned"] = True
        self._rebuild_spill(force=True)
        # Re-seat only — a second close_gripper flings the WSG arm.
        self._seat_sponge_in_hand()
        self._lift_sponge()

        visits = 0
        max_visits = 36
        while visits < max_visits and not self.laptop_reached and not self.cleaned_ok:
            dirty = sorted(
                self._dirty_spots(),
                key=lambda s: (float(s["along"]), float(s["spawn"])),
            )
            if not dirty:
                # Table is clean — freeze growth so no new spots appear.
                self._freeze_spill_growth("wipe done")
                self.cleaned_ok = True
                break

            # After a failed dab, advance along the spill instead of stalling.
            spot = dirty[0]
            if self._last_failed_along is not None:
                farther = [s for s in dirty if float(s["along"]) > self._last_failed_along + 1e-6]
                if farther:
                    spot = farther[0]
                elif len(dirty) > 1:
                    spot = dirty[1]

            before = len(self._dirty_spots())
            cleared = self._dab_spot(spot)
            after = len(self._dirty_spots())
            visits += 1
            print(
                f"[clean_table] dab visit={visits} along={spot['along']:.3f} "
                f"dirty {before}->{after} cleared={cleared} "
                f"amt={self.spill_amount:.2f}"
            )
            if cleared:
                self._last_failed_along = None
                self._idle_steps(3)
            else:
                self._last_failed_along = float(spot["along"])
            if self._table_fully_clean():
                self._freeze_spill_growth("wipe done")
                self.cleaned_ok = True
                break

        self._wipe_armed = False
        self._wipe_target = None
        self._expert_wiping = False
        self._lift_sponge()

    def play_once(self):
        arm = self.arm
        self.plan_success = True

        # 1) Mug tips and stains immediately — do not wait for the sponge pick.
        self._idle_steps(4)
        self._animate_tip(n_steps=24)
        # Puddle is already active; short settle while spill keeps spreading.
        self._idle_steps(
            max(int(self.post_tip_wait), 12),
            until=lambda: (
                len(self._dirty_spots()) >= 3
                or self.spill_amount >= max(
                    float(self.expert_wipe_level),
                    float(self.initial_puddle_level),
                )
                or self.laptop_reached
            ),
        )

        if self.laptop_reached:
            self.plan_success = False
            self.info["info"] = self._info_dict(arm)
            return self.info

        # 2) Grab the sponge while the spill is already on the table.
        if not self._grasp_sponge():
            self.plan_success = False
            self.info["info"] = self._info_dict(arm)
            return self.info

        self.plan_success = True
        # Hold the grasp orientation — do not let the wrist twist before wiping.
        self.move(
            self.move_by_displacement(
                arm, z=0.06, move_axis="world", quat=self._hold_ee_quat()
            )
        )
        self._sync_welded_sponge()

        if self.laptop_reached:
            self.plan_success = False
            self.info["info"] = self._info_dict(arm)
            return self.info

        # 3) Wipe the active puddle (growth continues only while dirt remains).
        self._wipe_spill()

        if self.laptop_reached:
            self.plan_success = False
        elif self.check_success():
            self.plan_success = True
        else:
            self._wipe_spill()
            self.plan_success = bool(self.check_success()) and (not self.laptop_reached)

        self.info["info"] = self._info_dict(arm)
        print(
            f"[clean_table] done success={self.plan_success} "
            f"amt={self.spill_amount:.2f} clean_frac={self._spill_clean_frac():.2f} "
            f"dirty={len(self._dirty_spots())} reached={self.laptop_reached}"
        )
        return self.info

    def _info_dict(self, arm: ArmTag) -> dict:
        return {
            "{A}": f"{self.MUG_MODEL}/base{self.mug_id}",
            "{B}": f"015_laptop/{self.laptop_id}",
            "{C}": "sponge",
            "{a}": str(arm),
        }

    def check_success(self):
        if self.laptop_reached:
            return False
        if not self.cup_tipped:
            return False
        active = [s for s in self._active_spots() if not s.get("under_mug")]
        if len(active) < 3:
            return False
        # All currently spawned wipeable spots must be dabbed.
        if self._dirty_spots():
            return False
        # Must have let the spill grow enough before claiming success.
        if self.max_spill_amount < 0.20:
            return False
        return True

    def get_score(self) -> float:
        """Partial score from wipe progress.

        Full success → 1. Otherwise ``clean_frac`` bands ``[0.85,1)`` /
        ``[0.60,0.85)`` / ``[0.35,0.60)`` → 0.75 / 0.5 / 0.25. Reaching the
        laptop is still a failure, but keeps the credit earned up to that
        frame: the spill only stops growing once the table is fully clean, so
        without a latch the bands would be unreachable in a played episode.
        """
        if not bool(getattr(self, "cup_tipped", False)):
            return 0.0
        if float(getattr(self, "max_spill_amount", 0.0)) < float(self.PARTIAL_MIN_SPILL):
            return 0.0
        if bool(getattr(self, "laptop_reached", False)):
            return float(getattr(self, "_partial_score", 0.0))
        if self.check_success():
            return 1.0
        frac = float(self._spill_clean_frac())
        return float(score_half_open_intervals(frac, self.PARTIAL_CLEAN_BANDS))

    def get_obs(self):
        obs = super().get_obs()
        obs["coffee_spill"] = {
            "spill_amount": float(self.spill_amount),
            "clean_frac": float(self._spill_clean_frac()),
            "dirty_spots": int(len(self._dirty_spots())),
            "active_spots": int(len(self._active_spots())),
            "spill_front": float(self._spill_front_along()) if self.spill_active else 0.0,
            "cup_tipped": bool(self.cup_tipped),
            "laptop_reached": bool(self.laptop_reached),
            "cleaned_ok": bool(self.cleaned_ok),
            "mug_side": float(self.mug_side),
            "laptop_side": float(getattr(self, "laptop_side", 0.0)),
            "spill_speed_mult": float(getattr(self, "_spill_speed_mult", 1.0)),
            "partial_score": float(self.get_score()),
        }
        return obs
