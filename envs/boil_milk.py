"""Boil milk on a KitchenS cooking range up to a marked red target ring.

KitchenS scene with microwave + cooking range (no sink/tap). A stainless
saucepan of milk sits on a burner; a milk carton and mug rest on the open
counter. The robot turns the stove on; white milk rises while boiling. Shut
the stove off once the milk reaches the red ring inside the pot.
Success = milk reached the ring, then stove off before spill.
Failure = spill over the rim, or shutoff before the milk hits the ring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_box, UnStableError
from .utils.partial_score import score_descending_bands


class boil_milk(KitchenS_base_task):
    """Turn the range on, let milk rise to the red ring, then shut it off."""

    # Deficit (percentage points below target) → partial score.
    # Authoring bands [10,7.5) / [7.5,5) / [5,0) → 0.25 / 0.5 / 0.75.
    PARTIAL_BELOW_BANDS = (
        (10.0, 7.5, 0.25),
        (7.5, 5.0, 0.5),
        (5.0, 0.0, 0.75),
    )

    BOIL_STEPS_DEFAULT = 6000         # steps for liquid_level 0→1 while stove is on
    SETTLE_STEPS_DEFAULT = 1200       # steps for liquid_level → baseline while stove is off
    BASELINE_LEVEL_DEFAULT = 0.35     # resting fill fraction of pot height
    # Red ring fill mark: 100 = pot rim, 0 = pot floor (see ``target_ring``).
    TARGET_RING_DEFAULT = 80
    # Start the shutoff reach this far below the ring (milk keeps rising en route).
    SHUTOFF_LEAD_DEFAULT = 0.08
    OVERFLOW_LEVEL_DEFAULT = 0.98     # rim / spill threshold (failure if reached while on)
    KNOB_CONTACT_RADIUS_DEFAULT = 0.06   # pinch-to-knob distance that counts as held
    EE_TO_TCP = 0.12                  # EE frame → TCP offset (cook_meat / dispense_gummy)
    # Y-up meshes stood upright: target world heights for realistic counter props.
    MUG_TARGET_HEIGHT = 0.0735        # prior 10.5 cm, then −30%
    MILK_TARGET_HEIGHT = 0.200        # ~20 cm 1 L carton
    # Straight-down approach onto the top-facing cooktop knob.
    KNOB_APPROACH_PATH = KitchenS_base_task.TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF = 0.012
    MILK_COLOR = [0.96, 0.96, 0.93, 0.92]
    TARGET_RING_COLOR = (0.92, 0.08, 0.08)

    # Default cooktop / microwave anchors when stove_side randomizes L/R.
    RANGE_Y = 0.14
    RANGE_X_RIGHT = 0.28
    RANGE_X_LEFT = -0.28
    MICROWAVE_Y = 0.18
    MICROWAVE_X_RIGHT = 0.32
    MICROWAVE_X_LEFT = -0.32
    BURNER_NAMES = ("left_front", "right_front", "left_rear", "right_rear")

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("boil_milk", {})
        # Prefer scene_0; actual L/R is driven by stove_side + overrides below.
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        # Bare counter beside the range — no sink basin / chrome tap.
        self.omit_sink = True
        self.clear_sink_and_range = False
        # Stove was 1.5× default, then cut 30% → 1.05×. Microwave +30%.
        self.range_scale_mult = float(self._cfg.get("range_scale_mult", 1.05))
        self.microwave_scale_mult = float(self._cfg.get("microwave_scale_mult", 1.3))

        seed = int(kwags.get("seed", 0) or 0)
        self._layout_seed = seed
        rng = np.random.RandomState(seed + 17)
        self.stove_side, self.range_position_override, self.microwave_xy_override = (
            self._sample_stove_microwave_layout(self._cfg, rng)
        )

        # Per-step state must exist before any early _update_kinematic_tasks call.
        self.stove_on = False
        self.liquid_level = float(self._cfg.get("baseline_level", self.BASELINE_LEVEL_DEFAULT))
        self.baseline_level = float(self.liquid_level)
        self.max_liquid_level = float(self.liquid_level)
        self.overflowed = False
        self.reached_target = False
        self.turned_on_once = False
        self.turned_off_after_boil = False
        self._partial_score = 0.0  # latched at the stove-off transition
        self._reset_metric_state()
        self._liquid_entity = None
        self._spill_entity = None
        self._target_ring_parts = []
        self._spill_amount = 0.0
        self._burner_shapes = []
        self._knob_press_latched = False
        self._prev_knob_pressed = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self.force_overflow = bool(self._cfg.get("force_overflow", False))
        # Placeholder; load_actors samples the ring with ±jitter from the seed.
        self.target_ring = float(
            np.clip(self._cfg.get("target_ring", self.TARGET_RING_DEFAULT), 0.0, 100.0)
        )
        self.target_level = self.target_ring / 100.0
        self.pot_burner = "left_rear"

        super().setup_demo(**kwags)

    def _sample_stove_microwave_layout(self, cfg, rng: np.random.RandomState):
        """Place stove left or right; microwave always on the opposite side."""
        side = str(cfg.get("stove_side", "random")).lower().strip()
        if side not in ("left", "right"):
            side = str(rng.choice(["left", "right"]))
        range_y = float(cfg.get("range_xy", [self.RANGE_X_RIGHT, self.RANGE_Y])[1])
        mw_y = float(cfg.get("microwave_y", self.MICROWAVE_Y))
        if side == "left":
            range_xy = [
                float(cfg.get("range_x_left", self.RANGE_X_LEFT)),
                range_y,
            ]
            mw_xy = [
                float(cfg.get("microwave_x_right", self.MICROWAVE_X_RIGHT)),
                mw_y,
            ]
        else:
            range_xy = [
                float(cfg.get("range_x_right",
                              cfg.get("range_xy", [self.RANGE_X_RIGHT, self.RANGE_Y])[0])),
                range_y,
            ]
            mw_xy = [
                float(cfg.get("microwave_x_left", self.MICROWAVE_X_LEFT)),
                mw_y,
            ]
        return side, range_xy, mw_xy

    def _select_pot_burner(self, cfg, rng: np.random.RandomState) -> str:
        burners = getattr(self, "burner_positions", None) or {}
        names = [n for n in self.BURNER_NAMES if n in burners] or list(burners.keys())
        if not names:
            return "left_rear"
        choice = cfg.get("pot_burner", "random")
        if isinstance(choice, str) and choice.lower().strip() not in ("", "random"):
            name = choice.lower().strip()
            if name in burners:
                return name
        return str(rng.choice(names))

    def _sample_target_ring(self, cfg, rng: np.random.RandomState) -> float:
        """Configured fill mark with multiplicative ±``target_ring_jitter`` (default 10%)."""
        base = float(cfg.get("target_ring", self.TARGET_RING_DEFAULT))
        jitter = float(cfg.get("target_ring_jitter", 0.10))
        if jitter > 0.0:
            scale = 1.0 + float(rng.uniform(-abs(jitter), abs(jitter)))
            ring = base * scale
        else:
            ring = base
        return float(np.clip(ring, 0.0, 100.0))

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.boil_steps = int(cfg.get("boil_steps", self.BOIL_STEPS_DEFAULT))
        self.randomize_boil_steps = bool(cfg.get("randomize_boil_steps", False))
        bs_jitter = float(np.clip(abs(float(cfg.get("boil_steps_jitter", 0.20))), 0.0, 0.95))
        rng = np.random.RandomState(int(getattr(self, "_layout_seed", 0) or 0) + 41)
        if self.randomize_boil_steps and bs_jitter > 0.0:
            scale = 1.0 + float(rng.uniform(-bs_jitter, bs_jitter))
            self.boil_steps = max(1, int(round(self.boil_steps * scale)))
        self.settle_steps = int(cfg.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.baseline_level = float(cfg.get("baseline_level", self.BASELINE_LEVEL_DEFAULT))
        self.target_ring = self._sample_target_ring(cfg, rng)
        self.target_level = self.target_ring / 100.0
        lead = float(cfg.get("shutoff_lead", self.SHUTOFF_LEAD_DEFAULT))
        # Prefer explicit expert_shutoff_level if set; else ring minus lead.
        if "expert_shutoff_level" in cfg:
            self.expert_shutoff_level = float(cfg["expert_shutoff_level"])
        else:
            self.expert_shutoff_level = max(
                self.baseline_level + 0.02, self.target_level - lead
            )
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL_DEFAULT))
        self.knob_contact_radius = float(
            cfg.get("knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )

        self.liquid_level = self.baseline_level
        self.max_liquid_level = self.baseline_level
        self.stove_on = False
        self.overflowed = False
        self.reached_target = False
        self.turned_on_once = False
        self.turned_off_after_boil = False
        self._partial_score = 0.0
        self._reset_metric_state()
        self._knob_press_latched = False
        self._prev_knob_pressed = False
        self._expert_holding_knob = False
        self._ignore_knob = False
        self.force_overflow = bool(cfg.get("force_overflow", False))
        self._spill_amount = 0.0
        if getattr(self, "_spill_entity", None) is not None:
            try:
                self.scene.remove_entity(self._spill_entity)
            except Exception:
                pass
            self._spill_entity = None
        self._clear_target_ring()

        if not hasattr(self, "burner_xy"):
            raise UnStableError("cooking range missing — KitchenS base did not load a range")

        self.pot_burner = self._select_pot_burner(cfg, rng)
        if self.pot_burner in getattr(self, "burner_positions", {}):
            self.burner_xy = self.burner_positions[self.pot_burner]
        # KitchenS fire cover / lit-burner logic keys off ``burner_name``.
        self.burner_name = str(self.pot_burner)

        # Stainless saucepan (tea-pan style): thin hollow cylinder + dual-rod
        # handle with black grips — matches Desktop/images/tea_pan.jpeg.
        # Procedural so mplib/CuRobo never see a non-unit-scale mesh.
        self.pot_id = int(cfg.get("pot_id", 0))
        # Defaults: prior saucepan size then −30%.
        pot_r = float(cfg.get("pot_radius", 0.054))
        pot_h = float(cfg.get("pot_height", 0.0735))
        # Thin wall so the lip does not read as a second inner layer from above.
        wall = float(cfg.get("pot_wall", 0.0022))
        pot_z0 = float(self.range_top_z)
        cx, cy = float(self.burner_xy[0]), float(self.burner_xy[1])
        # Keep the stock burner disc under the pot (make_soup fire effect): the
        # disc glows blue with the ring, so the flame reads as a real burner.
        self._burner_shapes = []
        if getattr(self, "active_burner", None) is not None:
            self._burner_home_pose = sapien.Pose(
                p=[cx, cy, float(self.range_top_z) + 0.0015]
            )
            try:
                self.active_burner.set_pose(self._burner_home_pose)
            except Exception:
                pass
            for c in self.active_burner.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_shapes = list(c.render_shapes)
        self._disc_parts = []
        self._disc_shapes = []
        self._disc_home_poses = []
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        metal = sapien.render.RenderMaterial(base_color=[0.74, 0.75, 0.77, 1.0])
        metal.metallic = 0.92
        metal.roughness = 0.16

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        # Floor (inset so its rim is not visible inside the wall lip).
        floor_r = pot_r - wall
        floor_h = 0.002
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, floor_h / 2], vertical_q),
            radius=floor_r, half_length=floor_h / 2,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, floor_h / 2], vertical_q),
            radius=floor_r, half_length=floor_h / 2, material=metal,
        )

        # Faceted hollow wall (open top so the milk column stays visible).
        wall_segments = 32
        wall_radius = pot_r - wall / 2
        tangent_half = wall_radius * np.tan(np.pi / wall_segments) * 1.02
        for ang in np.linspace(0, 2 * np.pi, wall_segments, endpoint=False):
            px = wall_radius * np.cos(ang)
            py = wall_radius * np.sin(ang)
            tangent_angle = ang + np.pi / 2
            q = [
                float(np.cos(tangent_angle / 2)),
                0.0,
                0.0,
                float(np.sin(tangent_angle / 2)),
            ]
            wall_pose = sapien.Pose([px, py, pot_h / 2], q)
            builder.add_box_collision(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=self.scene.default_physical_material,
            )
            builder.add_box_visual(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=metal,
            )

        builder.set_initial_pose(sapien.Pose(p=[cx, cy, pot_z0]))
        self.pot = builder.build(name="tea_pot")

        # Dual-rod handle toward −X. Black sleeve per rod; rods stop short of the
        # tip so nothing protrudes / no hanging loop at the far end.
        handle_len = float(cfg.get("handle_length", 0.091))
        handle_z = pot_z0 + pot_h * 0.62
        rod_r = 0.0028
        rod_sep = 0.014
        tip_overhang = 0.004  # black grip extends past metal rods
        rod_x0 = cx - pot_r + 0.003
        tip_x = cx - pot_r - handle_len
        rod_x1 = tip_x + tip_overhang
        rod_mid_x = 0.5 * (rod_x0 + rod_x1)
        rod_half = 0.5 * (rod_x0 - rod_x1)
        # Grip covers most of each rod and ends at tip_x (clean tip, no cap piece).
        grip_x0 = rod_x0 - 0.012  # leave a short metal stub at the pot
        grip_x1 = tip_x
        grip_mid = 0.5 * (grip_x0 + grip_x1)
        grip_half = 0.5 * (grip_x0 - grip_x1)
        for side, tag in ((-1, "a"), (1, "b")):
            y = cy + side * rod_sep / 2
            create_box(
                self.scene,
                sapien.Pose(p=[rod_mid_x, y, handle_z]),
                half_size=[rod_half, rod_r, rod_r],
                color=(0.70, 0.71, 0.73),
                name=f"tea_pot_rod_{tag}",
                is_static=True,
            )
            create_box(
                self.scene,
                sapien.Pose(p=[grip_mid, y, handle_z]),
                half_size=[grip_half, rod_r + 0.0022, rod_r + 0.0022],
                color=(0.05, 0.05, 0.06),
                name=f"tea_pot_grip_{tag}",
                is_static=True,
            )

        # Milk fills to the inner wall — no silver floor ring inside.
        self.pot_inner_radius = pot_r - wall
        self.pot_inner_height = pot_h - floor_h
        self.pot_bottom_z = pot_z0 + floor_h
        self.pot_xy = (cx, cy)
        self.pot_radius = pot_r
        self.pot_height = pot_h

        # Blue fire halo around the pot base on the selected burner (not a fixed
        # left_rear default) — disc + ring + cover all follow ``burner_xy``.
        self._clear_stove_fire_ring()
        self._stove_fire_visual = None  # force cover/disc refresh on next set
        self._build_stove_fire_ring(
            cx,
            cy,
            float(self.range_top_z) + 0.0015,
            float(pot_r + 0.009),
            n=28,
            half_size=[0.007, 0.0035, 0.002],
        )
        self._rebuild_liquid(force=True)
        self._build_target_ring()
        self._set_stove_fire(False)

        # Milk carton + mug on the open counter between microwave and stove.
        self._load_milk_box(cfg)
        self._load_mug(cfg)

        # Arm that can reach the knob (left stove → left arm, right → right).
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        print(
            f"[boil_milk] stove_side={getattr(self, 'stove_side', '?')} "
            f"range={np.round(self.range_xy, 3).tolist()} "
            f"mw={None if self.microwave_xy is None else np.round(self.microwave_xy, 3).tolist()} "
            f"burner={self.pot_burner} arm={self.arm} "
            f"knob_x={float(self.knob_xy[0]):.3f} "
            f"target_ring={self.target_ring:.1f}% "
            f"shutoff={self.expert_shutoff_level:.3f}"
        )

    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin: float = 0.0) -> bool:
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        h2 = np.asarray(h2, dtype=float)
        m = float(margin)
        return bool(
            abs(float(c1[0] - c2[0])) < (float(h1[0]) + float(h2[0]) + m)
            and abs(float(c1[1] - c2[1])) < (float(h1[1]) + float(h2[1]) + m)
        )

    def _fixture_blockers(self):
        """Microwave + cooktop footprints for decor non-overlap."""
        blockers = []
        if getattr(self, "range_xy", None) is not None:
            blockers.append((
                np.asarray(self.range_xy, dtype=float),
                np.asarray(getattr(self, "range_half_size", (0.14, 0.16)), dtype=float),
            ))
        if getattr(self, "microwave_xy", None) is not None and getattr(
            self, "microwave_half_xy", None
        ) is not None:
            blockers.append((
                np.asarray(self.microwave_xy, dtype=float),
                np.asarray(self.microwave_half_xy, dtype=float),
            ))
        return blockers

    def _counter_apron_bounds(self):
        """Open counter between microwave and stove (adapts to either L/R layout)."""
        blockers = self._fixture_blockers()
        if len(blockers) >= 2:
            (a_c, a_h), (b_c, b_h) = blockers[0], blockers[1]
            # Order by X so the gap between fixtures is the free apron.
            if float(a_c[0]) > float(b_c[0]):
                a_c, a_h, b_c, b_h = b_c, b_h, a_c, a_h
            x_lo = float(a_c[0] + a_h[0] + 0.04)
            x_hi = float(b_c[0] - b_h[0] - 0.04)
        else:
            hx = getattr(self, "range_half_size", (0.14, 0.16))[0]
            rx = float(self.range_xy[0]) if hasattr(self, "range_xy") else 0.28
            if rx >= 0:
                x_lo, x_hi = -0.08, float(rx - hx - 0.08)
            else:
                x_lo, x_hi = float(rx + hx + 0.08), 0.08
        y_lo, y_hi = -0.18, 0.12
        if x_hi <= x_lo + 0.05:
            # Fallback mid-apron if fixtures sit too close.
            x_lo, x_hi = -0.10, 0.10
        return x_lo, x_hi, y_lo, y_hi

    def _sample_free_prop_xy(
        self,
        half_xy,
        blockers,
        rng: np.random.RandomState,
        margin: float = 0.025,
        fallback=None,
    ):
        """Sample a non-overlapping XY in the open apron."""
        x_lo, x_hi, y_lo, y_hi = self._counter_apron_bounds()
        hx, hy = float(half_xy[0]), float(half_xy[1])
        for _ in range(80):
            x = float(rng.uniform(x_lo + hx, x_hi - hx)) if x_hi - x_lo > 2 * hx else 0.5 * (x_lo + x_hi)
            y = float(rng.uniform(y_lo + hy, y_hi - hy)) if y_hi - y_lo > 2 * hy else 0.5 * (y_lo + y_hi)
            cand = np.array([x, y], dtype=float)
            ok = True
            for b_c, b_h in blockers:
                if self._aabb_overlap(cand, (hx, hy), b_c, b_h, margin):
                    ok = False
                    break
            if ok:
                return (float(cand[0]), float(cand[1]))
        if fallback is not None:
            return fallback
        return (float(0.5 * (x_lo + x_hi)), float(0.5 * (y_lo + y_hi)))

    @staticmethod
    def _model_data(modelname: str, model_id: int) -> dict:
        path = resolve_model_dir(modelname) / f"model_data{model_id}.json"
        with open(path) as f:
            return json.load(f)

    @classmethod
    def _yup_authored_height(cls, modelname: str, model_id: int) -> float:
        """World height (Y) of a Y-up mesh at ``scale_mult=1``."""
        data = cls._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        ext = data["extents"]
        return float(sc[1]) * float(ext[1])

    @classmethod
    def _yup_authored_half_xy(cls, modelname: str, model_id: int) -> tuple[float, float]:
        """Half footprint (X, Z) of an upright Y-up mesh at ``scale_mult=1``."""
        data = cls._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        ext = data["extents"]
        return 0.5 * float(sc[0]) * float(ext[0]), 0.5 * float(sc[2]) * float(ext[2])

    def _scale_for_target_height(
        self, modelname: str, model_id: int, target_h: float, override=None
    ) -> float:
        """``scale_mult`` so the upright mesh matches ``target_h`` (meters)."""
        if override is not None:
            return float(override)
        authored = self._yup_authored_height(modelname, model_id)
        if authored <= 1e-6:
            return 1.0
        return float(target_h) / authored

    def _place_upright_prop(
        self,
        modelname: str,
        model_id: int,
        xy: tuple[float, float],
        scale: float,
        yaw: float,
        name: str,
    ):
        upright = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        q = qmult(euler2quat(0.0, 0.0, yaw, axes="sxyz"), upright)
        z = 0.74 + float(self.table_z_bias) + 0.001
        pose = sapien.Pose([float(xy[0]), float(xy[1]), z], q.tolist())
        actor = create_actor(
            self,
            pose=pose,
            modelname=modelname,
            model_id=model_id,
            convex=True,
            is_static=True,
            scale_mult=scale,
        )
        if actor is not None:
            actor.set_name(name)
            if getattr(actor, "config", None) is None:
                hx, hz = self._yup_authored_half_xy(modelname, model_id)
                actor.config = {
                    "scale": [scale, scale, scale],
                    "center": [0.0, 0.0, 0.0],
                    "extents": [2.0 * hx, 0.20, 2.0 * hz],
                }
            try:
                self.add_prohibit_area(actor, padding=0.02)
            except Exception:
                pad = 0.06
                self.prohibited_area.append(
                    [xy[0] - pad, xy[1] - pad, xy[0] + pad, xy[1] + pad]
                )
        return actor

    def _apron_prop_slots(self, milk_half, mug_half, gap: float, rng=None):
        """Sample two non-overlapping apron poses clear of MW / stove / each other.

        Returns ``(milk_xy, mug_xy)`` visual-center positions with ≥ ``gap`` clearance
        between body radii. Mug uses a cavity-focused radius (handle excluded), matching
        ``clean_table``'s ``0.42 * width`` heuristic for ``039_mug``.
        """
        if rng is None:
            rng = np.random.RandomState(
                int(getattr(self, "_layout_seed", 0) or 0) + 91
            )
        milk_hx, milk_hz = float(milk_half[0]), float(milk_half[1])
        mug_hx, mug_hz = float(mug_half[0]), float(mug_half[1])
        milk_half_xy = (milk_hx, milk_hz)
        # 039_mug extents include the handle; body/cavity is much smaller.
        mug_body = (
            0.42 * float(max(2.0 * mug_hx, 2.0 * mug_hz)) * 0.5,
            0.42 * float(max(2.0 * mug_hx, 2.0 * mug_hz)) * 0.5,
        )
        # Inflate halves by gap/2 so the sampled centers keep an air gap.
        milk_sample = (milk_hx + 0.5 * gap, milk_hz + 0.5 * gap)
        mug_sample = (mug_body[0] + 0.5 * gap, mug_body[1] + 0.5 * gap)

        blockers = self._fixture_blockers()
        x_lo, x_hi, y_lo, y_hi = self._counter_apron_bounds()
        milk_fb = (float(0.5 * (x_lo + x_hi)), float(y_hi - milk_hz - 0.01))
        mug_fb = (float(0.5 * (x_lo + x_hi)), float(y_lo + mug_body[1] + 0.01))

        milk_xy = self._sample_free_prop_xy(
            milk_sample, blockers, rng, margin=0.02, fallback=milk_fb
        )
        blockers = blockers + [(np.asarray(milk_xy, dtype=float), np.asarray(milk_half_xy))]
        mug_xy = self._sample_free_prop_xy(
            mug_sample, blockers, rng, margin=0.02, fallback=mug_fb
        )
        return milk_xy, mug_xy

    def _yup_visual_center_offset(self, modelname: str, model_id: int, scale: float, q):
        """World offset from actor pose → mesh geometric center (upright Y-up props)."""
        data = self._model_data(modelname, model_id)
        center = np.array(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        # ``scale`` arg is scale_mult; authored scale already in model_data.
        # create_actor applies authored*scale_mult; center metadata is in mesh units
        # and is multiplied by the final world scale (authored * mult).
        final_sc = float(sc[0]) * float(scale)
        R = t3d.quaternions.quat2mat(np.asarray(q, dtype=float))
        return (R @ (center * final_sc)).astype(float)

    def _load_milk_box(self, cfg):
        """Place a random ``038_milk-box`` on the apron (paired slots with mug)."""
        rng = np.random.RandomState(int(getattr(self, "_layout_seed", 0) or 0) + 91)
        n_variants = 4
        mid = int(cfg.get("milk_box_id", -1))
        if mid < 0 or mid >= n_variants:
            mid = int(rng.randint(0, n_variants))
        self.milk_box_id = mid
        target_h = float(cfg.get("milk_box_height", self.MILK_TARGET_HEIGHT))
        scale = self._scale_for_target_height(
            "038_milk-box", mid, target_h, cfg.get("milk_box_scale")
        )
        self.milk_box_scale = scale
        hx, hz = self._yup_authored_half_xy("038_milk-box", mid)
        self._milk_half_xy = (hx * scale, hz * scale)

        # Mug sizing needed for joint slot placement (same defaults as _load_mug).
        mug_mid = int(cfg.get("mug_id", 0))
        if mug_mid < 0 or mug_mid >= 10:
            mug_mid = int(rng.randint(0, 10))
        mug_target_h = float(cfg.get("mug_height", self.MUG_TARGET_HEIGHT))
        mug_scale = self._scale_for_target_height(
            "039_mug", mug_mid, mug_target_h, cfg.get("mug_scale")
        )
        mhx, mhz = self._yup_authored_half_xy("039_mug", mug_mid)
        mug_half = (mhx * mug_scale, mhz * mug_scale)
        gap = float(cfg.get("mug_gap", 0.03))
        milk_xy, mug_xy = self._apron_prop_slots(
            self._milk_half_xy, mug_half, gap, rng=rng
        )
        self._planned_mug_xy = mug_xy
        self._planned_mug_scale = mug_scale
        self._planned_mug_id = mug_mid

        yaw = float(rng.uniform(-0.6, 0.6))
        try:
            self.milk_box = self._place_upright_prop(
                "038_milk-box", mid, milk_xy, scale, yaw, f"038_milk-box/base{mid}"
            )
        except Exception as e:
            print(f"[boil_milk] failed to load milk box: {e}")
            self.milk_box = None
        self.milk_box_xy = milk_xy

    def _load_mug(self, cfg):
        """Place an upright ``039_mug`` in the pre-planned non-overlapping apron slot."""
        mid = int(cfg.get("mug_id", getattr(self, "_planned_mug_id", 0)))
        if mid < 0 or mid >= 10:
            mid = int(getattr(self, "_planned_mug_id", 0))
        self.mug_id = mid
        target_h = float(cfg.get("mug_height", self.MUG_TARGET_HEIGHT))
        scale = float(
            getattr(self, "_planned_mug_scale", None)
            or self._scale_for_target_height(
                "039_mug", mid, target_h, cfg.get("mug_scale")
            )
        )
        self.mug_scale = scale
        ux, uy = getattr(self, "_planned_mug_xy", None) or (
            self.milk_box_xy[0],
            self.milk_box_xy[1] - 0.14,
        )

        rng = np.random.RandomState(int(getattr(self, "_layout_seed", 0) or 0) + 97)
        yaw = float(rng.uniform(-0.5, 0.5))
        upright = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        q = qmult(euler2quat(0.0, 0.0, yaw, axes="sxyz"), upright)
        # Compensate 039_mug origin ≠ cavity center so the cup body lands on the slot.
        try:
            off = self._yup_visual_center_offset("039_mug", mid, scale, q)
            pose_xy = (float(ux - off[0]), float(uy - off[1]))
        except Exception:
            pose_xy = (float(ux), float(uy))
        try:
            self.mug = self._place_upright_prop(
                "039_mug", mid, pose_xy, scale, yaw, f"039_mug/base{mid}"
            )
        except Exception as e:
            print(f"[boil_milk] failed to load mug: {e}")
            self.mug = None
        self.mug_xy = (float(ux), float(uy))

    # ---------------------------------------------------------------- target ring
    # Same thin torus mesh as measure_ingredient / fill_coffee_jar (native R≈0.0388).
    _RING_MESH = resolve_model_dir("253_glass_jar") / "rings" / "thin_ring.glb"
    _RING_MESH_RADIUS = 0.0388

    def _clear_target_ring(self):
        for part in getattr(self, "_target_ring_parts", []) or []:
            try:
                self.scene.remove_entity(part)
            except Exception:
                pass
        self._target_ring_parts = []

    def _build_target_ring(self):
        """Single red circle inside the pot (same thin_ring mesh as measure_ingredient).

        ``target_ring`` is 0–100: 100 = pot rim, 0 = pot floor.
        """
        self._clear_target_ring()
        if not getattr(self, "pot_inner_height", None):
            return
        frac = float(np.clip(getattr(self, "target_level", 0.8), 0.0, 1.0))
        cx, cy = float(self.pot_xy[0]), float(self.pot_xy[1])
        z = float(self.pot_bottom_z) + frac * float(self.pot_inner_height)
        # Just inside the inner wall so the mark stays visible above the milk.
        target_r = float(self.pot_inner_radius) * 0.96
        scale = float(target_r / self._RING_MESH_RADIUS)
        rgba = list(self.TARGET_RING_COLOR[:3]) + [0.92]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.45)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.45
            mat.metallic = 0.0
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(
            filename=str(self._RING_MESH.resolve()),
            scale=[scale, scale, scale],
            material=mat,
        )
        builder.set_initial_pose(sapien.Pose([cx, cy, z]))
        ent = builder.build(name="milk_target_ring")
        self._target_ring_parts.append(ent)

    # ---------------------------------------------------------------- liquid / stove visuals
    def _liquid_half_height(self) -> float:
        return max(0.004, 0.5 * self.liquid_level * self.pot_inner_height)

    def _rebuild_liquid(self, force: bool = False):
        """Recreate the translucent liquid column to match ``liquid_level``."""
        half_h = self._liquid_half_height()
        # Skip tiny updates to keep planning pass cheap.
        if (
            not force
            and self._liquid_entity is not None
            and abs(half_h - getattr(self, "_liquid_half_h_cached", -1.0)) < 0.0015
        ):
            return
        self._liquid_half_h_cached = half_h

        if self._liquid_entity is not None:
            try:
                self.scene.remove_entity(self._liquid_entity)
            except Exception:
                pass
            self._liquid_entity = None

        # Visual-only static cylinder (no collision). Sapien cylinder axis = local +X;
        # VERTICAL_CYL_Q rotates it to world +Z (same as pick_ripe_apple trunk).
        radius = self.pot_inner_radius
        color = list(self.MILK_COLOR)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=color)
        mat.metallic = 0.0
        mat.roughness = 0.35
        vertical_cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]  # 90° about Y
        local_pose = sapien.Pose([0, 0, 0], vertical_cyl_q)
        builder.add_cylinder_visual(
            pose=local_pose, radius=radius, half_length=half_h, material=mat
        )
        z = self.pot_bottom_z + half_h
        builder.set_initial_pose(sapien.Pose(p=[self.pot_xy[0], self.pot_xy[1], z]))
        self._liquid_entity = builder.build(name="pot_liquid")

    def _set_burner_glow(self, on: bool):
        """Blue fire ring only; solid burner disc stays hidden after load_actors."""
        self._set_stove_fire(bool(on), intensity=1.0 if on else 0.0)

    def _set_stove(self, on: bool):
        on = bool(on)
        if on == self.stove_on:
            return
        self.stove_on = on
        if on:
            if self._metric_on_step is None:
                self._metric_on_step = self._metric_step()
            self.turned_on_once = True
        elif self.turned_on_once and not self.overflowed:
            # Count as a boil shutoff only if milk already hit the red ring.
            if self.reached_target or self.max_liquid_level >= self.target_level - 1e-3:
                self.turned_off_after_boil = True
            # Latch the partial score at the shutoff frame — milk settles back
            # toward baseline afterward, so a live read would understate it.
            self._partial_score = self._score_at_shutoff()
            self._latch_shutoff_metric()
            peak = float(getattr(self, "max_liquid_level", 0.0))
            target = float(getattr(self, "target_level", 0.0))
            print(
                f"[boil_milk] stove off: peak={peak * 100:.1f}% "
                f"target={target * 100:.1f}% deficit={(target - peak) * 100:.1f}pp "
                f"score={self._partial_score:.2f}"
            )
        self._set_burner_glow(on)

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_on_step = None        # burner first lit
        self._metric_target_step = None    # milk first reached the red ring
        self._metric_off_step = None       # burner switched off (decisive event)
        self._metric_off_level = None      # liquid level at that instant

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_shutoff_metric(self):
        """Called from _set_stove on the on->off edge, before the level settles back."""
        try:
            if self._metric_off_step is None:
                self._metric_off_step = self._metric_step()
                self._metric_off_level = float(self.liquid_level)
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `shutoff_latency_steps` — steps from the milk first touching the red
        ring until the burner is switched off. This is the whole reaction window: the
        milk keeps rising the entire time, so every step here eats overflow headroom.
        extra2 `level_error_norm` — |level at shutoff - target| / (overflow - target).
        LOWER is better; 0 = cut the flame exactly on the ring, 1 = only shut off at
        the boil-over point.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_target_step, self._metric_off_step
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["shutoff_latency_steps"] = lat
        out["shutoff_latency_s"] = None if lat is None else round(float(lat) * dt, 4)
        heat = (None if (self._metric_on_step is None or self._metric_target_step is None)
                else max(int(self._metric_target_step) - int(self._metric_on_step), 0))
        out["heat_latency_steps"] = heat

        lvl = self._metric_off_level
        try:
            span = float(self.overflow_level) - float(self.target_level)
            out["level_error_norm"] = (
                None if (lvl is None or span <= 1e-9)
                else round(abs(float(lvl) - float(self.target_level)) / span, 4))
        except Exception:
            out["level_error_norm"] = None
        out["shutoff_level"] = None if lvl is None else round(float(lvl), 4)
        try:
            out["target_level"] = round(float(self.target_level), 4)
            out["max_liquid_level"] = round(float(self.max_liquid_level), 4)
        except Exception:
            out["target_level"] = None
            out["max_liquid_level"] = None
        out["overflowed"] = bool(getattr(self, "overflowed", False))
        return out

    def _spawn_spill_puddle(self, scale: float = 1.0):
        """Compact white milk puddle under the pot (visual only).

        Spill outer radius is slightly larger than the pot footprint so the
        spill reads as a visible white puddle around the base.
        ``scale`` is kept for call-site compatibility; size is fixed.
        """
        del scale  # fixed size — no grow-animation
        if self._spill_entity is not None:
            try:
                self.scene.remove_entity(self._spill_entity)
            except Exception:
                pass
            self._spill_entity = None

        cx, cy = float(self.pot_xy[0]), float(self.pot_xy[1])
        z = float(self.range_top_z) + 0.003
        pot_r = float(self.pot_radius)
        spill_r = 1.12 * pot_r
        milk = sapien.render.RenderMaterial(base_color=[0.97, 0.97, 0.95, 0.97])
        milk.metallic = 0.0
        milk.roughness = 0.65
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], vertical_q),
            radius=float(spill_r),
            half_length=0.0010,
            material=milk,
        )
        builder.set_initial_pose(sapien.Pose(p=[cx, cy, z]))
        self._spill_entity = builder.build(name="milk_spill")

    def _trigger_overflow(self):
        """Rim reached while boiling: spill, kill flame, reset pot level, fail."""
        if self.overflowed:
            return
        self.overflowed = True
        self._spill_amount = 1.0
        # Kill the flame without counting as a successful shutoff.
        self.stove_on = False
        self._set_burner_glow(False)
        self._spawn_spill_puddle()
        # Milk drops back to the resting level after the spill.
        self.liquid_level = float(self.baseline_level)
        self._rebuild_liquid(force=True)

    # ---------------------------------------------------------------- per-step dynamics
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        # Guard: _update_kinematic_tasks runs during camera init BEFORE load_actors.
        if not getattr(self, "pot_inner_height", None):
            return
        if not hasattr(self, "liquid_level"):
            return

        # Knob / fire: KitchenS_base_task._update_stove_knob_control — burner
        # follows the physical knob angle (including during the expert twist).

        # Boiling continues for every sim step while the burner is on — including
        # the shutoff reach. Rising stops only when the knob angle turns the
        # stove off (or overflow commits once the hand leaves the knob).
        if self.overflowed:
            self.liquid_level = float(self.baseline_level)
        elif self.stove_on:
            self.liquid_level = min(
                1.0, self.liquid_level + 1.0 / max(1, self.boil_steps)
            )
            self.max_liquid_level = max(self.max_liquid_level, self.liquid_level)
            if self.liquid_level >= self.target_level - 1e-3:
                self.reached_target = True
                if self._metric_target_step is None:
                    self._metric_target_step = self._metric_step()
            # Do not kill the flame mid-knob-turn: the expert is still reaching
            # to shut off. Overflow commits only once the hand is free again.
            if (
                self.liquid_level >= self.overflow_level
                and not getattr(self, "_ignore_knob", False)
            ):
                self._trigger_overflow()
        else:
            # Settle back toward the baseline once the stove is off.
            if self.liquid_level > self.baseline_level:
                self.liquid_level = max(
                    self.baseline_level,
                    self.liquid_level - 1.0 / max(1, self.settle_steps),
                )

        self._rebuild_liquid(force=False)

    def _idle_steps(self, n_steps: int, until=None):
        """Advance sim (and record frames) without arm motion — NOT delay().

        ``until`` is polled each step and stops the wait early. It must be passed
        here rather than wrapping ``_idle_steps(1)`` so the save_freq throttle
        keeps counting across the whole wait.
        """
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

    def _idle_until_level(self, level: float, max_steps: int = 4000):
        """Watch the pot until the liquid reaches ``level`` (or it stops rising)."""
        # Replay pass follows recorded joints; a long level-wait stalls rendering.
        if not getattr(self, "need_plan", True):
            self._idle_steps(30)
            self.liquid_level = max(float(self.liquid_level), float(level))
            return
        self._idle_steps(
            max_steps,
            until=lambda: self.liquid_level >= float(level) or self.overflowed,
        )

    # ---------------------------------------------------------------- expert motion
    def _turn_knob(self, want_on: bool):
        """Contact-driven cooktop knob twist; fire follows the knob angle only."""
        self._turn_stove_knob(
            self.KNOB_ON_ANGLE if bool(want_on) else self.KNOB_OFF_ANGLE,
            start_angle=(
                self.KNOB_ON_ANGLE if self.stove_on else self.KNOB_OFF_ANGLE
            ),
        )
        # If the milk crested the rim during the reach and the stove is STILL on
        # (missed shutoff), commit the overflow now that the hand is clear.
        if (
            self.stove_on
            and not self.overflowed
            and self.liquid_level >= self.overflow_level
        ):
            self._trigger_overflow()

    def play_once(self):
        arm = self.arm

        # 1) Turn the stove ON — the milk starts rising on this twist.
        self._turn_knob(want_on=True)

        if self.force_overflow:
            # Failure demo: leave the burner on until the milk hits the rim,
            # then hold so the white cooktop spill is visible.
            self._idle_until_level(self.overflow_level, max_steps=8000)
            if not self.overflowed:
                self._trigger_overflow()
            self._idle_steps(120)  # let the puddle finish spreading
        else:
            # 2) Watch until milk is high enough that the shutoff reach still
            #    finishes under the rim. Rising continues through the whole reach.
            self._idle_until_level(self.expert_shutoff_level)

            # 3) Grasp+twist OFF — boiling stops only on this twist, not before.
            self._turn_knob(want_on=False)

            # 4) Let the liquid settle back toward baseline (visible in the video).
            settle = max(0.0, self.liquid_level - self.baseline_level)
            steps_settle = int(round(settle * self.settle_steps))
            self._idle_steps(max(1, min(steps_settle, self.settle_steps)))

        self.info["info"] = {
            "{A}": "tea_pot",
            "{D}": (
                f"038_milk-box/base{self.milk_box_id}"
                if getattr(self, "milk_box", None) is not None
                else "038_milk-box"
            ),
            "{E}": (
                f"039_mug/base{self.mug_id}"
                if getattr(self, "mug", None) is not None
                else "039_mug"
            ),
            "{B}": "cooking_range",
            "{C}": "stove_knob",
            "{a}": str(arm),
            "{side}": str(getattr(self, "stove_side", "right")),
            "{burner}": str(getattr(self, "pot_burner", "left_rear")),
        }
        return self.info

    def check_success(self):
        """Success: milk reached the red ring, then stove off before spill."""
        if self.overflowed:
            return False
        if not self.turned_on_once:
            return False
        # Fail if shutoff happened without the milk ever hitting the mark.
        if not self.reached_target and self.max_liquid_level < self.target_level - 1e-3:
            return False
        if not self.turned_off_after_boil:
            return False
        if self.stove_on:
            return False
        if self.liquid_level > self.overflow_level - 0.05:
            return False
        return True

    def _score_at_shutoff(self) -> float:
        """Deficit-band score from the peak milk level at the shutoff frame.

        Overflow → 0. Under-target shutoff uses deficit bands ``[10,7.5)%`` /
        ``[7.5,5)%`` / ``[5,0)%`` below target → 0.25 / 0.5 / 0.75. At/above the
        target the deficit is <= 0, so the bands return their default (0); the
        full-success case is handled in ``get_score``.
        """
        if bool(getattr(self, "overflowed", False)):
            return 0.0
        peak = float(getattr(self, "max_liquid_level", 0.0))
        target = float(getattr(self, "target_level", 0.0))
        deficit_pct = (target - peak) * 100.0
        if deficit_pct <= 0.0:
            return 0.0
        return float(score_descending_bands(deficit_pct, self.PARTIAL_BELOW_BANDS))

    def get_score(self) -> float:
        """Partial score from peak milk level vs the red-ring target.

        Full success → 1. Otherwise the score is the value latched the instant
        the stove turned off (milk settles back afterward, so a live read would
        understate the peak). Overflow or never-shut-off → 0.
        """
        if self.check_success():
            return 1.0
        if bool(getattr(self, "overflowed", False)):
            return 0.0
        if bool(getattr(self, "stove_on", False)):
            return 0.0
        return float(getattr(self, "_partial_score", 0.0))

    def get_obs(self):
        obs = super().get_obs()
        obs["boiling"] = {
            "stove_on": bool(self.stove_on),
            "liquid_level": float(self.liquid_level),
            "max_liquid_level": float(self.max_liquid_level),
            "overflowed": bool(self.overflowed),
            "reached_target": bool(self.reached_target),
            "target_ring": float(self.target_ring),
            "target_level": float(self.target_level),
            "baseline_level": float(self.baseline_level),
            "spill_amount": float(getattr(self, "_spill_amount", 0.0)),
            "stove_side": str(getattr(self, "stove_side", "right")),
            "pot_burner": str(getattr(self, "pot_burner", "left_rear")),
            "range_xy": list(np.asarray(getattr(self, "range_xy", (0, 0)), dtype=float)),
            "partial_score": float(self.get_score()),
        }
        return obs
