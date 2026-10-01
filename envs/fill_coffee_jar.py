"""Fill a marked glass jar with beans from a glass-box dispenser (KitchenS).

Inherits ``KitchenS_base_task`` (cooking range on a solid kitchen counter; no
microwave / sink / tap). The dispenser is a raised clear glass hopper packed
with real bean meshes. Pressing the button on top opens a nozzle above the jar
and releases beans into a glass jar marked with red ring lines at 25% / 50% /
75% (rim = full). Target fill is randomized (15%–80% in 5% steps by default).

Counter decor: random ``113_coffee-box``, ``038_milk-box``, and ``039_mug``.
A random ``009_kettle`` sits on one of the stove burners.

Dispense amount is gated by **press force** on the button (four thresholds).
Each press–release cycle dispenses once for the achieved force level (up to the
final / hardest level if pushed further); holding continuously does not keep
filling — release and press again for another shot. Success / failure is scored
only after the button has been idle for ``IDLE_SCORE_SEC`` (default 3 s).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import sapien.render
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_box
from .utils.partial_score import score_descending_bands


class fill_coffee_jar(KitchenS_base_task):
    """Press the dispenser button to fill a marked glass jar to a target level.

    Task options (``task_args.fill_coffee_jar``):
      - ``target_fill``: fraction in {0.15, 0.20, …, 0.80} | ``random`` (default)
      - ``fill_tol``: success band half-width (default 0.05 = ±5%)
      - ``idle_score_sec``: seconds without a press before scoring (default 3)
      - ``randomize_layout``: seed-randomized non-overlapping station pose
      - ``force_thresholds``: 4 increasing force cutoffs (N)
      - ``beans_per_force_level``: beans dispensed at each force level
      - ``beans_full``: bean count that packs to the rim / 100%
      - ``scene_id``: 0 | 1 | 2 (KitchenS fixture layout)

    Peak button press force maps to four dispense levels; fill rises with beans.
    One press–release = one dispense (continuous hold does not re-fire). Within a
    press the level follows how hard the button is pushed, up to the final level.
    After ``idle_score_sec`` without a press following any completed press
    (timer resets on every press), success if fill ∈ [target − tol, target + tol],
    else failure.
    """

    # Abs error (pp) from target → partial: [10,8.5) / [8.5,7) / [7,5) → 0.25/0.5/0.75.
    # Under- and over-fill both count; success band is ±5% so partial starts past 5 pp.
    PARTIAL_ABS_BANDS = (
        (10.0, 8.5, 0.25),
        (8.5, 7.0, 0.5),
        (7.0, 5.0, 0.75),
    )

    BEAN_MODEL = "252_coffee_bean"
    JAR_MODEL = "253_glass_jar"
    # Max beans that may be dispensed (enough to pass the 25% ring densely).
    BEANS_FULL = 160
    # Four force thresholds (N) → beans per press level (light → hard).
    # Bean counts are spaced so each level raises fill by ~0.625%/3.125%/
    # 6.25%/9.375% of the jar (beans_full=160 → 1/5/10/15 beans).
    FORCE_THRESHOLDS = (3.0, 6.0, 10.0, 14.0)
    BEANS_PER_FORCE_LEVEL = (1, 5, 10, 15)
    # Spring proxy (N/m): F = k * virtual key compression.  Compression starts
    # only inside the blue key's travel range, so dispense is tied to pressure
    # against the button rather than to a keyboard/event trigger.
    FORCE_STIFFNESS = 800.0
    FORCE_ENGAGE_SLACK = 0.05
    # Expert press depths from hover for force levels 1..4.
    PRESS_DEPTHS = (0.020, 0.030, 0.044, 0.057)
    PRESS_SAMPLE_S = 0.40  # hold time to sample peak force
    # Target fill: 15%–80% in 5% steps (default episode samples randomly).
    FILL_LEVELS = tuple(round(0.15 + 0.05 * i, 2) for i in range(14))  # 0.15..0.80
    FILL_TOL = 0.05  # success band: target_fill ± fill_tol
    IDLE_SCORE_SEC = 3.0  # score only after this long without a button press
    # Station footprints for non-overlap layout (half-extents, meters).
    DISP_HALF_XY = (0.065, 0.065)
    JAR_HALF_XY = (0.045, 0.045)
    LAYOUT_MARGIN = 0.025
    # Dense mound packing (used for freeze + bean-need estimates).
    _BEAN_R = 0.0055
    _BEAN_H = 0.0065
    _PILE_R_SCALE = 0.72

    # Glass-box dispenser (inspired by reference photo — tall clear column on a base).
    BOX_HALF = (0.035, 0.035, 0.063)       # 30% shorter glass hopper
    PEDESTAL_HALF = (0.052, 0.052, 0.035)  # 30% lower pedestal
    PLATFORM_HALF = (0.058, 0.058, 0.008)  # platform between pedestal and hopper
    # Red push button on top of the glass lid (press target).
    BTN_BASE_HALF = (0.016, 0.016, 0.003)
    BTN_HALF = (0.011, 0.011, 0.010)
    BTN_BASE_COLOR = (0.12, 0.12, 0.14)
    BTN_COLOR = (0.08, 0.36, 0.95)
    BTN_TOUCH_XY_TOL = 0.05   # m; fingertip XY tolerance around button center
    # Teleop Q over the key: fraction of default Z_SPEED (finer force control).
    DISPENSE_Z_DOWN_SCALE = 0.22
    BEAN_FILL_FRAC = 0.65                  # visual fill inside the glass box
    EE_TO_TCP = 0.12
    KEY_HOVER_DIS = 0.06
    KEY_PRESS_DEPTH = 0.057  # default = hardest force level
    BUTTON_VISUAL_STEP = 0.0007
    SETTLE_STEPS = 80

    JAR_INNER_R = 0.035
    JAR_HEIGHT = 0.125
    JAR_BOTTOM_T = 0.005

    GLASS = [0.72, 0.84, 0.92, 0.16]
    # Interactive viewer look (matches trap_bug plain trap): no transmission/IOR.
    PLAIN_GLASS = [0.14, 0.26, 0.40, 0.55]
    BEAN_BROWN = [0.30, 0.14, 0.05]
    # Fill marks: saturated opaque red (not washed-out translucent).
    RING_RED = [1.0, 0.04, 0.02]
    # thin_ring.glb native radius ≈ jar outer; Z scale thickens the band.
    RING_MESH_RADIUS = 0.0388
    RING_XY_SCALE = 1.02
    RING_Z_SCALE = 3.2

    # Counter / stove decor (static props; not task goals).
    COFFEE_BOX_MODEL = "113_coffee-box"
    MILK_BOX_MODEL = "038_milk-box"
    MUG_MODEL = "039_mug"
    KETTLE_MODEL = "009_kettle"
    UPRIGHT_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
    MILK_TARGET_HEIGHT = 0.18
    MUG_TARGET_HEIGHT = 0.075
    COFFEE_BOX_SCALE = 1.15
    KETTLE_SCALE = 1.0
    # Authored mesh meta for 009_kettle (visual GLBs exist; model_data may be absent).
    KETTLE_MODEL_DATA = {
        0: {
            "center": [0.033335, 0.3552585, 0.0680835],
            "extents": [1.378758, 1.356831, 1.535575],
            "scale": [0.1, 0.1, 0.1],
        },
        1: {
            "center": [0.0000505, 0.2696635, 0.1514755],
            "extents": [0.932073, 1.451321, 1.315845],
            "scale": [0.1, 0.1, 0.1],
        },
        2: {
            "center": [0.004627, 0.2282145, 0.152863],
            "extents": [1.186794, 1.415909, 1.565162],
            "scale": [0.1, 0.1, 0.1],
        },
    }

    # Stove L/R anchors (microwave omitted in this task).
    RANGE_Y = 0.14
    RANGE_X_RIGHT = 0.28
    RANGE_X_LEFT = -0.28
    # Tight station band so the top-button press still reaches force thresholds.
    # Wider |x| / far −y poses plan the reach but often register near-zero force.
    STATION_X_LEFT = (-0.105, -0.07)
    STATION_X_RIGHT = (0.055, 0.085)
    # Dispenser Y jitter relative to the baseline ``disp_y`` (meters).
    DISP_Y_UP = 0.015   # +1.5 cm
    DISP_Y_DOWN = 0.02  # −2 cm
    JAR_Y_MIN = -0.17

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("fill_coffee_jar", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self._layout_seed = int(kwags.get("seed", 0) or 0)
        self.replace_sink_with_range = True
        self.omit_sink = True  # solid counter; no sink basin or faucet tap
        self._ensure_kettle_assets()

        rng = np.random.RandomState(self._layout_seed + 17)
        self.stove_side, self.range_position_override = self._sample_stove_side(
            self._cfg, rng
        )

        self._loaded = False
        self.beans = []
        self.beans_in_jar = 0
        self.press_count = 0
        self.target_fill = 0.25
        self.dispenser_touch_surface = None
        self.jar = None
        self.jar_visual = None
        self.fill_visual = None
        self._jar_visual_hollow = False
        self.decor_coffee_box = None
        self.decor_milk_box = None
        self.decor_mug = None
        self.decor_kettle = None
        self.kettle_burner = None
        self._touch_latched = False
        self._dispensing = False
        self._press_active = False
        self._awaiting_release = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0
        self._press_dispense_level = 0
        self._reset_metric_state()
        self._button_visual_depth = 0.0
        self._button_target_depth = 0.0
        self._press_idle_s = 0.0
        self.table_top = 0.74
        self._plain_glass = bool(self._cfg.get("plain_glass", False))

        super().setup_demo(**kwags)
        self._configure_observer_camera()

    def _sample_stove_side(self, cfg, rng: np.random.RandomState):
        """Place the cooktop on the left or right half of the counter."""
        side = str(cfg.get("stove_side", "random")).lower().strip()
        if side not in ("left", "right"):
            side = str(rng.choice(["left", "right"]))
        range_y = float(cfg.get("range_xy", [self.RANGE_X_RIGHT, self.RANGE_Y])[1])
        if side == "left":
            range_xy = [float(cfg.get("range_x_left", self.RANGE_X_LEFT)), range_y]
        else:
            range_xy = [
                float(
                    cfg.get(
                        "range_x_right",
                        cfg.get("range_xy", [self.RANGE_X_RIGHT, self.RANGE_Y])[0],
                    )
                ),
                range_y,
            ]
        return side, range_xy

    def _load_microwave(self, table_height, table_xy_bias):
        """Leave the left/back counter open — no microwave in this task."""
        self.microwave = None
        self.microwave_xy = None
        self.microwave_half_xy = None
        self.microwave_top_z = None
        return

    @classmethod
    def _ensure_kettle_assets(cls):
        """Ensure ``009_kettle`` has create_actor-compatible model_data + collision."""
        root = resolve_model_dir(cls.KETTLE_MODEL)
        visual = root / "visual"
        if not visual.exists():
            return
        collision = root / "collision"
        collision.mkdir(exist_ok=True)
        for mid, meta in cls.KETTLE_MODEL_DATA.items():
            vfile = visual / f"base{mid}.glb"
            cfile = collision / f"base{mid}.glb"
            if vfile.exists() and not cfile.exists():
                try:
                    cfile.symlink_to(vfile.resolve())
                except OSError:
                    # Fall back to using the visual path name if symlink fails.
                    pass
            mpath = root / f"model_data{mid}.json"
            if not mpath.exists():
                data = {
                    "center": list(meta["center"]),
                    "extents": list(meta["extents"]),
                    "scale": list(meta["scale"]),
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
                mpath.write_text(json.dumps(data, indent=4) + "\n")

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.05, -0.55, 1.40], dtype=np.float64)
        look_at = np.array([-0.05, -0.05, 0.92], dtype=np.float64)
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
    def _plain_glass_material(self):
        """Simple alpha-transparent plastic — no glass transmission/IOR (viewer-friendly)."""
        mat = sapien.render.RenderMaterial(base_color=list(self.PLAIN_GLASS))
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.55)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.55
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            try:
                mat.ior = 1.0
            except Exception:
                pass
        return mat

    def _glass_material(self, rgba=None, transmission=0.90):
        """Nearly-clear glass matching trap_bug / reference acrylic look."""
        if bool(getattr(self, "_plain_glass", False)):
            return self._plain_glass_material()
        c = list(rgba if rgba is not None else self.GLASS)
        if len(c) == 3:
            c = c + [0.18]
        mat = sapien.render.RenderMaterial(base_color=c)
        try:
            mat.set_transmission(float(transmission))
            mat.set_transmission_roughness(0.02)
            mat.set_roughness(0.05)
            mat.set_metallic(0.0)
        except Exception:
            mat.transmission = float(transmission)
            mat.roughness = 0.05
            mat.metallic = 0.0
        try:
            mat.set_ior(1.45)
        except Exception:
            try:
                mat.ior = 1.45
            except Exception:
                pass
        return mat

    def _opaque_material(self, rgb, alpha=1.0):
        rgba = list(rgb[:3]) + [float(alpha)]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.45)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.45
            mat.metallic = 0.0
        return mat

    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    def _jar_glass_material(self, viewer_shell: bool = False):
        """Glass for the jar.

        Demo cameras use transmission glass. The interactive SAPIEN viewer does
        not composite opaque fill behind transmission materials (or a solid
        cylinder wall), so the viewer shell uses plain alpha glass — same trick
        as ``measure_ingredient`` / ``trap_bug``.
        """
        if viewer_shell:
            glass = sapien.render.RenderMaterial(
                base_color=[0.70, 0.82, 0.90, 0.22]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.12)
                glass.set_metallic(0.0)
            except Exception:
                glass.roughness = 0.12
                glass.metallic = 0.0
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        if bool(getattr(self, "_plain_glass", False)):
            return self._plain_glass_material()

        glass = sapien.render.RenderMaterial(base_color=[0.76, 0.88, 0.94, 0.12])
        try:
            glass.set_transmission(1.0)
            glass.set_transmission_roughness(0.0)
            glass.set_roughness(0.05)
            glass.set_metallic(0.0)
        except Exception:
            pass
        try:
            glass.set_ior(1.0)
        except Exception:
            pass
        return glass

    def _viewer_coffee_material(self):
        """Fully opaque coffee column for viewer compositing through alpha glass."""
        mat = sapien.render.RenderMaterial(
            base_color=[0.22, 0.10, 0.04, 1.0]
        )
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.85)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.85
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
        return mat

    def _add_static_box(self, pose, half_size, material=None, color=None, name="", collision=True):
        """Build a static box; supports translucent glass via RenderMaterial."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            builder.add_box_collision(
                pose=sapien.Pose(),
                half_size=list(half_size),
                material=self.scene.default_physical_material,
            )
        if material is None:
            rgba = list(color if color is not None else [0.8, 0.8, 0.8, 1.0])
            if len(rgba) == 3:
                rgba = rgba + [1.0]
            material = self._opaque_material(rgba[:3], rgba[3])
        builder.add_box_visual(pose=sapien.Pose(), half_size=list(half_size), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _build_hollow_box(self, pose, half_size, wall, material, name, open_top=False, collision=True):
        """Hollow rectangular shell (4 walls + optional lid + bottom) — true glass box."""
        hx, hy, hz = [float(v) for v in half_size]
        wt = float(wall)
        top_hz = wt * 0.5
        side_hz = hz - (0.0 if open_top else top_hz)
        # Keep a floor so beans / fill sit inside.
        floor_hz = wt * 0.5
        side_z = -hz + floor_hz + side_hz
        parts = [
            (sapien.Pose([0, 0, -hz + floor_hz]), [hx, hy, floor_hz]),  # bottom
            (sapien.Pose([hx - wt * 0.5, 0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([-hx + wt * 0.5, 0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([0, hy - wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
            (sapien.Pose([0, -hy + wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
        ]
        if not open_top:
            parts.append((sapien.Pose([0, 0, hz - top_hz]), [hx, hy, top_hz]))

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            for local_pose, half in parts:
                builder.add_box_collision(
                    pose=local_pose,
                    half_size=list(half),
                    material=self.scene.default_physical_material,
                )
        for local_pose, half in parts:
            builder.add_box_visual(pose=local_pose, half_size=list(half), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _add_static_mesh_visual(self, filename, pose, material, name, scale=None):
        """Add a smooth mesh visual while forcing the intended render material."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        kwargs = {
            "filename": str(Path(filename).resolve()),
            "material": material,
        }
        if scale is not None:
            sc = list(scale)
            if len(sc) == 1:
                sc = [float(sc[0])] * 3
            kwargs["scale"] = [float(v) for v in sc[:3]]
        builder.add_visual_from_file(**kwargs)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    # ------------------------------------------------------------------ layout
    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin=0.0):
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        h2 = np.asarray(h2, dtype=float)
        m = float(margin)
        return bool(
            abs(c1[0] - c2[0]) < (h1[0] + h2[0] + m)
            and abs(c1[1] - c2[1]) < (h1[1] + h2[1] + m)
        )

    def _layout_blockers(self):
        """Kitchen fixture footprints the station must not overlap."""
        blockers = []
        # Microwave is omitted for this task; only the cooktop blocks the station.
        range_xy = getattr(self, "range_xy", None)
        range_half = getattr(self, "range_half_size", None)
        if range_xy is not None and range_half is not None:
            blockers.append(
                (np.asarray(range_xy, dtype=float), np.asarray(range_half, dtype=float))
            )
        return blockers

    @staticmethod
    def _model_data(modelname: str, model_id: int) -> dict:
        path = resolve_model_dir(modelname) / f"model_data{model_id}.json"
        with open(path) as f:
            return json.load(f)

    @classmethod
    def _available_model_ids(cls, modelname: str) -> list[int]:
        root = resolve_model_dir(modelname)
        ids = []
        for p in root.glob("model_data*.json"):
            try:
                mid = int(p.stem.replace("model_data", ""))
            except ValueError:
                continue
            vis = root / "visual" / f"base{mid}.glb"
            col = root / "collision" / f"base{mid}.glb"
            if vis.exists() or col.exists():
                ids.append(mid)
        return sorted(ids)

    @classmethod
    def _yup_authored_height(cls, modelname: str, model_id: int) -> float:
        data = cls._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(sc, (int, float)):
            sc = [float(sc)] * 3
        return float(sc[1]) * float(data["extents"][1])

    @classmethod
    def _yup_authored_half_xy(cls, modelname: str, model_id: int) -> tuple[float, float]:
        data = cls._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(sc, (int, float)):
            sc = [float(sc)] * 3
        ext = data["extents"]
        return 0.5 * float(sc[0]) * float(ext[0]), 0.5 * float(sc[2]) * float(ext[2])

    def _scale_for_target_height(self, modelname: str, model_id: int, target_h: float) -> float:
        authored = self._yup_authored_height(modelname, model_id)
        if authored <= 1e-6:
            return 1.0
        return float(target_h) / authored

    def _yup_pose_z(self, modelname: str, model_id: int, scale_mult: float, surface_z: float) -> float:
        """Pose Z so an upright Y-up mesh rests on ``surface_z``."""
        data = self._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(sc, (int, float)):
            sc = [float(sc)] * 3
        final_sy = float(sc[1]) * float(scale_mult)
        cy = float(data.get("center", [0.0, 0.0, 0.0])[1])
        ey = float(data["extents"][1])
        bottom_local_y = cy - 0.5 * ey
        return float(surface_z - bottom_local_y * final_sy)

    def _place_upright_prop(
        self,
        modelname: str,
        model_id: int,
        xy: tuple[float, float],
        scale_mult: float,
        yaw: float,
        name: str,
        surface_z: float | None = None,
    ):
        z_surf = float(self.table_top if surface_z is None else surface_z)
        z = self._yup_pose_z(modelname, model_id, scale_mult, z_surf)
        q = qmult(euler2quat(0.0, 0.0, float(yaw), axes="sxyz"), self.UPRIGHT_Q)
        pose = sapien.Pose([float(xy[0]), float(xy[1]), float(z)], q.tolist())
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
            print(f"[fill_coffee_jar] failed to spawn {name} ({modelname}/base{model_id})")
            return None
        try:
            actor.set_name(name)
        except Exception:
            pass
        try:
            self.add_prohibit_area(actor, padding=0.02)
        except Exception:
            pad = 0.05
            self.prohibited_area.append(
                [xy[0] - pad, xy[1] - pad, xy[0] + pad, xy[1] + pad]
            )
        return actor

    def _decor_clear(
        self,
        xy: np.ndarray,
        half_xy: tuple[float, float],
        blockers: list,
        margin: float = 0.02,
    ) -> bool:
        for b_c, b_h in blockers:
            if self._aabb_overlap(xy, half_xy, b_c, b_h, margin):
                return False
        return True

    def _decor_free_bounds(self):
        """Open counter half opposite the stove, away from the jar approach corridor."""
        rx = float(getattr(self, "range_xy", (self.RANGE_X_RIGHT, self.RANGE_Y))[0])
        rhx = float(getattr(self, "range_half_size", (0.14, 0.16))[0])
        # Prefer the free half (same side as the dispenser).
        free_sign = -1.0 if rx >= 0.0 else 1.0
        if free_sign < 0:
            x_lo, x_hi = -0.42, min(-0.02, float(rx - rhx - 0.06))
        else:
            x_lo, x_hi = max(0.02, float(rx + rhx + 0.06)), 0.42
        # Keep back / mid counter; jar corridor occupies the near-robot apron.
        y_lo, y_hi = -0.02, 0.22
        if x_hi <= x_lo + 0.06:
            x_lo, x_hi = (free_sign * 0.34, free_sign * 0.12) if free_sign < 0 else (
                free_sign * 0.12,
                free_sign * 0.34,
            )
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
        return float(x_lo), float(x_hi), float(y_lo), float(y_hi)

    def _jar_corridor_blocker(self):
        """Keep decor out of the robot approach lane in front of the jar."""
        jx, jy = float(self.jar_xy[0]), float(self.jar_xy[1])
        # Corridor from jar toward the robot (−Y), wide enough for the arm.
        half_x = max(float(self.JAR_HALF_XY[0]) + 0.06, 0.10)
        half_y = 0.14
        cy = jy - half_y
        return (
            np.array([jx, cy], dtype=float),
            np.array([half_x, half_y], dtype=float),
        )

    def _spawn_counter_decor(self, cfg, rng: np.random.RandomState):
        """Coffee box + milk carton + mug on open space (clear of stove / station / corridor)."""
        coffee_ids = self._available_model_ids(self.COFFEE_BOX_MODEL) or [0]
        milk_ids = self._available_model_ids(self.MILK_BOX_MODEL) or [0]
        mug_ids = [i for i in self._available_model_ids(self.MUG_MODEL) if i <= 8] or [0]

        coffee_id = int(cfg.get("coffee_box_id", -1))
        if coffee_id < 0 or coffee_id not in coffee_ids:
            coffee_id = int(rng.choice(coffee_ids))
        milk_id = int(cfg.get("milk_box_id", -1))
        if milk_id < 0 or milk_id not in milk_ids:
            milk_id = int(rng.choice(milk_ids))
        mug_id = int(cfg.get("mug_id", -1))
        if mug_id < 0 or mug_id not in mug_ids:
            mug_id = int(rng.choice(mug_ids))

        coffee_scale = float(cfg.get("coffee_box_scale", self.COFFEE_BOX_SCALE))
        milk_h = float(cfg.get("milk_box_height", self.MILK_TARGET_HEIGHT))
        mug_h = float(cfg.get("mug_height", self.MUG_TARGET_HEIGHT))
        milk_scale = self._scale_for_target_height(self.MILK_BOX_MODEL, milk_id, milk_h)
        mug_scale = self._scale_for_target_height(self.MUG_MODEL, mug_id, mug_h)

        chx, chz = self._yup_authored_half_xy(self.COFFEE_BOX_MODEL, coffee_id)
        mhx, mhz = self._yup_authored_half_xy(self.MILK_BOX_MODEL, milk_id)
        uhx, uhz = self._yup_authored_half_xy(self.MUG_MODEL, mug_id)
        coffee_half = (chx * coffee_scale, chz * coffee_scale)
        milk_half = (mhx * milk_scale, mhz * milk_scale)
        # Mug extents include the handle; body footprint is smaller.
        mug_half = (0.42 * uhx * mug_scale, 0.42 * uhz * mug_scale)

        blockers = list(self._layout_blockers())
        blockers.append((self.dispenser_xy, self.DISP_HALF_XY))
        blockers.append((self.jar_xy, self.JAR_HALF_XY))
        blockers.append(self._jar_corridor_blocker())

        x_lo, x_hi, y_lo, y_hi = self._decor_free_bounds()
        free_sign = -1.0 if float(self.dispenser_xy[0]) <= 0.0 else 1.0
        specs = [
            ("coffee", coffee_half, coffee_scale, self.COFFEE_BOX_MODEL, coffee_id, "decor_coffee_box"),
            ("milk", milk_half, milk_scale, self.MILK_BOX_MODEL, milk_id, "decor_milk_box"),
            ("mug", mug_half, mug_scale, self.MUG_MODEL, mug_id, "decor_mug"),
        ]
        placed = {}
        for key, half, scale, model, mid, name in specs:
            pose_xy = None
            for _ in range(80):
                xy = np.array(
                    [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float
                )
                if self._decor_clear(xy, half, blockers, margin=0.025):
                    pose_xy = xy
                    break
            if pose_xy is None:
                # Deterministic fallback behind the station on the free half.
                fallback = {
                    "coffee": np.array([free_sign * 0.30, 0.14]),
                    "milk": np.array([free_sign * 0.22, 0.14]),
                    "mug": np.array([free_sign * 0.26, 0.04]),
                }
                pose_xy = fallback[key]
            yaw = float(rng.uniform(-0.6, 0.6))
            actor = self._place_upright_prop(
                model, mid, (float(pose_xy[0]), float(pose_xy[1])), scale, yaw, name
            )
            setattr(self, name, actor)
            placed[key] = {
                "id": int(mid),
                "xy": [float(pose_xy[0]), float(pose_xy[1])],
                "scale": float(scale),
            }
            blockers.append((pose_xy, half))
        self._decor_layout = placed
        self.coffee_box_id = coffee_id
        self.milk_box_id = milk_id
        self.mug_id = mug_id

    def _spawn_kettle_on_burner(self, cfg, rng: np.random.RandomState):
        """Place a random ``009_kettle`` on a random cooktop burner."""
        burners = getattr(self, "burner_positions", None) or {}
        if not burners:
            print("[fill_coffee_jar] no burners — skipping kettle decor")
            return
        kettle_ids = self._available_model_ids(self.KETTLE_MODEL) or list(
            self.KETTLE_MODEL_DATA.keys()
        )
        kettle_id = int(cfg.get("kettle_id", -1))
        if kettle_id < 0 or kettle_id not in kettle_ids:
            kettle_id = int(rng.choice(list(kettle_ids)))
        burner_name = cfg.get("kettle_burner", "random")
        names = list(burners.keys())
        if not isinstance(burner_name, str) or burner_name.lower() == "random":
            burner_name = str(rng.choice(names))
        elif burner_name not in burners:
            burner_name = str(rng.choice(names))
        bx, by = burners[burner_name]
        scale = float(cfg.get("kettle_scale", self.KETTLE_SCALE))
        yaw = float(rng.uniform(-np.pi, np.pi))
        surface_z = float(getattr(self, "range_top_z", self.table_top) + 0.002)
        self.decor_kettle = self._place_upright_prop(
            self.KETTLE_MODEL,
            kettle_id,
            (float(bx), float(by)),
            scale,
            yaw,
            "decor_kettle",
            surface_z=surface_z,
        )
        self.kettle_id = int(kettle_id)
        self.kettle_burner = str(burner_name)
        self.kettle_xy = (float(bx), float(by))

    def _station_clear(self, side_x, disp_y, jar_y, blockers, margin=None):
        if margin is None:
            margin = self.LAYOUT_MARGIN
        disp = np.array([side_x, disp_y], dtype=float)
        jar = np.array([side_x, jar_y], dtype=float)
        # Dispenser ↔ jar must not overlap each other.
        if self._aabb_overlap(disp, self.DISP_HALF_XY, jar, self.JAR_HALF_XY, margin):
            return False
        for b_c, b_h in blockers:
            if self._aabb_overlap(disp, self.DISP_HALF_XY, b_c, b_h, margin):
                return False
            if self._aabb_overlap(jar, self.JAR_HALF_XY, b_c, b_h, margin):
                return False
        return True

    def _resolve_target_fill(self, cfg, rng: np.random.RandomState) -> float:
        tf = cfg.get("target_fill", "random")
        if tf is None or (isinstance(tf, str) and tf.lower() == "random"):
            return float(rng.choice(self.FILL_LEVELS))
        val = round(float(tf), 2)
        # Accept exact fill levels, or values that land on the 5% grid in range.
        if val in self.FILL_LEVELS:
            return float(val)
        if 0.15 - 1e-9 <= val <= 0.80 + 1e-9 and abs(val * 20.0 - round(val * 20.0)) < 1e-6:
            return float(round(val, 2))
        raise ValueError(
            f"target_fill must be one of {list(self.FILL_LEVELS)} or 'random'"
        )

    def _sample_station_layout(self, cfg, rng: np.random.RandomState):
        """Place dispenser+jar on the free half opposite the stove.

        Keeps a shared X (nozzle alignment) with the jar in front (−y) of the
        dispenser so the jar stays under the nozzle. Pose jitter is kept inside
        a tight press-reliable band (far stations often register ~0 N on the button).
        """
        randomize = bool(cfg.get("randomize_layout", True))
        blockers = self._layout_blockers()
        rx = float(getattr(self, "range_xy", (self.RANGE_X_RIGHT, self.RANGE_Y))[0])
        free_sign = -1.0 if rx >= 0.0 else 1.0
        # Baseline station on the free half (left when stove is right, and vice versa).
        base_x = float(cfg.get("station_x", free_sign * 0.08))
        if np.sign(base_x) != 0 and np.sign(base_x) != free_sign:
            base_x = float(free_sign * abs(base_x))
        base_disp_y = float(cfg.get("disp_y", -0.02))
        base_jar_y = float(cfg.get("jar_y", -0.16))
        y_lo = base_disp_y - float(cfg.get("disp_y_down", self.DISP_Y_DOWN))
        y_hi = base_disp_y + float(cfg.get("disp_y_up", self.DISP_Y_UP))
        jar_y_min = float(cfg.get("jar_y_min", self.JAR_Y_MIN))
        if free_sign < 0:
            x_lo, x_hi = (
                float(cfg.get("station_x_left_lo", self.STATION_X_LEFT[0])),
                float(cfg.get("station_x_left_hi", self.STATION_X_LEFT[1])),
            )
        else:
            x_lo, x_hi = (
                float(cfg.get("station_x_right_lo", self.STATION_X_RIGHT[0])),
                float(cfg.get("station_x_right_hi", self.STATION_X_RIGHT[1])),
            )

        if not randomize:
            side_x = float(np.clip(base_x, x_lo, x_hi))
            disp_y = float(np.clip(base_disp_y, y_lo, y_hi))
            gap = float(np.clip(base_disp_y - base_jar_y, 0.125, 0.145))
            jar_y = max(disp_y - gap, jar_y_min)
            if not self._station_clear(side_x, disp_y, jar_y, blockers):
                for x in np.linspace(side_x, 0.5 * (x_lo + x_hi), 20):
                    if self._station_clear(float(x), disp_y, jar_y, blockers):
                        side_x = float(x)
                        break
            return side_x, disp_y, jar_y

        for _ in range(100):
            side_x = float(rng.uniform(x_lo, x_hi))
            disp_y = float(rng.uniform(y_lo, y_hi))
            # Keep jar under the nozzle: same X, modest gap ahead of the dispenser.
            gap = float(rng.uniform(0.125, 0.145))
            jar_y = disp_y - gap
            if jar_y < jar_y_min:
                continue
            if self._station_clear(side_x, disp_y, jar_y, blockers):
                return side_x, disp_y, jar_y

        # Deterministic fallback on the free half (center of the press-safe band).
        gap = float(np.clip(base_disp_y - base_jar_y, 0.125, 0.145))
        disp_y = float(np.clip(base_disp_y, y_lo, y_hi))
        return (
            float(np.clip(base_x, x_lo, x_hi)),
            disp_y,
            max(disp_y - gap, jar_y_min),
        )

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)
        seed = int(getattr(self, "_layout_seed", 0) or 0)
        # Separate streams so target choice doesn't collide with layout jitter.
        rng_target = np.random.RandomState(seed + 202)
        rng_layout = np.random.RandomState(seed + 101)

        self.target_fill = self._resolve_target_fill(cfg, rng_target)
        self.beans_full = int(cfg.get("beans_full", self.BEANS_FULL))
        thr = cfg.get("force_thresholds", self.FORCE_THRESHOLDS)
        beans_lv = cfg.get("beans_per_force_level", self.BEANS_PER_FORCE_LEVEL)
        self.force_thresholds = tuple(float(x) for x in thr)
        beans_base = [int(x) for x in beans_lv]
        self.randomize_beans_per_force_level = bool(
            cfg.get("randomize_beans_per_force_level", False)
        )
        beans_jitter = float(
            np.clip(abs(float(cfg.get("beans_per_force_level_jitter", 0.10))), 0.0, 0.95)
        )
        if self.randomize_beans_per_force_level and beans_jitter > 0.0:
            rng_beans = np.random.RandomState(seed + 303)
            jittered = []
            for n in beans_base:
                scale = 1.0 + float(rng_beans.uniform(-beans_jitter, beans_jitter))
                jittered.append(max(1, int(round(n * scale))))
            # Keep non-decreasing so harder presses never dispense fewer beans.
            for i in range(1, len(jittered)):
                jittered[i] = max(jittered[i], jittered[i - 1])
            beans_base = jittered
        self.beans_per_force_level = tuple(beans_base)
        if len(self.force_thresholds) != 4 or len(self.beans_per_force_level) != 4:
            raise ValueError("force_thresholds and beans_per_force_level need length 4")
        if any(
            self.force_thresholds[i] >= self.force_thresholds[i + 1]
            for i in range(3)
        ):
            raise ValueError("force_thresholds must be strictly increasing")
        self.force_stiffness = float(cfg.get("force_stiffness", self.FORCE_STIFFNESS))
        self.force_engage_slack = float(
            cfg.get("force_engage_slack", self.FORCE_ENGAGE_SLACK)
        )
        depths = cfg.get("press_depths", self.PRESS_DEPTHS)
        self.press_depths = tuple(float(x) for x in depths)
        if len(self.press_depths) != 4:
            raise ValueError("press_depths needs length 4")
        self.press_sample_s = float(cfg.get("press_sample_s", self.PRESS_SAMPLE_S))
        # Convenience aliases used by fill estimates / expert loop.
        self.beans_per_press_min = int(self.beans_per_force_level[0])
        self.beans_per_press_max = int(self.beans_per_force_level[-1])
        self.fill_tol = float(cfg.get("fill_tol", self.FILL_TOL))
        self.idle_score_sec = float(cfg.get("idle_score_sec", self.IDLE_SCORE_SEC))
        self.beans = []
        self.beans_in_jar = 0
        self.press_count = 0
        self._touch_latched = False
        self._reset_metric_state()
        self._dispensing = False
        self._press_active = False
        self._awaiting_release = False
        self._press_idle_s = 0.0
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0
        self._press_dispense_level = 0

        side_x, disp_y, jar_y = self._sample_station_layout(cfg, rng_layout)
        self.dispenser_xy = np.array([side_x, disp_y], dtype=float)
        self.jar_xy = np.array([side_x, jar_y], dtype=float)
        self.layout_ok = bool(
            self._station_clear(side_x, disp_y, jar_y, self._layout_blockers())
        )

        self._build_dispenser()
        self._build_jar()
        self._build_fill_rings()

        self.add_prohibit_area(sapien.Pose([*self.dispenser_xy, self.table_top + 0.1]), padding=0.08)
        self.add_prohibit_area(sapien.Pose([*self.jar_xy, self.table_top + 0.05]), padding=0.05)

        rng_decor = np.random.RandomState(seed + 303)
        self._spawn_counter_decor(cfg, rng_decor)
        self._spawn_kettle_on_burner(cfg, rng_decor)

        self._loaded = True
        levels = ", ".join(
            f"≥{t:.0f}N→{n}" for t, n in zip(self.force_thresholds, self.beans_per_force_level)
        )
        print(
            f"[fill_coffee_jar] KitchenS scene={self.scene_id} seed={seed} "
            f"stove_side={getattr(self, 'stove_side', '?')} "
            f"range={np.round(getattr(self, 'range_xy', (0, 0)), 3).tolist()} "
            f"target={self.target_fill:.0%}±{self.fill_tol:.0%} "
            f"disp={self.dispenser_xy.tolist()} jar={self.jar_xy.tolist()} "
            f"layout_ok={self.layout_ok} "
            f"coffee={getattr(self, 'coffee_box_id', '?')} "
            f"milk={getattr(self, 'milk_box_id', '?')} "
            f"mug={getattr(self, 'mug_id', '?')} "
            f"kettle={getattr(self, 'kettle_id', '?')}@{getattr(self, 'kettle_burner', '?')} "
            f"(~{self._beans_needed()}/{self.beans_full} beans; force {levels})"
        )

    def _build_dispenser(self):
        """Raised glass bean hopper; red top button opens a nozzle over the jar."""
        x, y = self.dispenser_xy
        z0 = self.table_top
        bx, by, bz = self.BOX_HALF
        px, py, pz = self.PLATFORM_HALF
        _, _, pedestal_hz = self.PEDESTAL_HALF
        wall = 0.0035
        glass = self._glass_material()

        # A simple solid pedestal raises the hopper high enough for the jar and nozzle.
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + pedestal_hz]),
            half_size=self.PEDESTAL_HALF,
            color=[0.13, 0.13, 0.15, 1.0],
            name="dispenser_pedestal",
        )
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + 2.0 * pedestal_hz + pz]),
            half_size=self.PLATFORM_HALF,
            color=[0.10, 0.10, 0.12, 1.0],
            name="dispenser_platform",
        )

        # Tall hollow clear hopper above the pedestal.
        hopper_bottom_z = z0 + 2.0 * pedestal_hz + 2.0 * pz
        box_z = hopper_bottom_z + bz
        self._build_hollow_box(
            pose=sapien.Pose([x, y, box_z]),
            half_size=self.BOX_HALF,
            wall=wall,
            material=glass,
            name="dispenser_glass_box",
            open_top=True,
            collision=True,
        )
        # Clear lid (visual/structural only) — press target is the button above.
        lid_hz = 0.003
        lid_z = box_z + bz + lid_hz
        self._add_static_box(
            pose=sapien.Pose([x, y, lid_z]),
            half_size=[bx * 1.01, by * 1.01, lid_hz],
            material=self._glass_material([0.74, 0.86, 0.93, 0.24]),
            name="dispenser_lid",
            collision=True,
        )
        lid_top = lid_z + lid_hz
        # Hollow dark bezel + blue push button centered on the lid.
        bhx, bhy, bhz = self.BTN_HALF
        btn_z = lid_top + bhz
        add_key_base_border(
            self,
            float(x),
            float(y),
            float(lid_top),
            self.BTN_HALF,
            color=list(self.BTN_BASE_COLOR),
            name_prefix="dispenser_button_base",
        )
        self.dispenser_touch_surface = self._add_static_box(
            pose=sapien.Pose([x, y, btn_z]),
            half_size=[bhx, bhy, bhz],
            color=[*self.BTN_COLOR, 1.0],
            name="dispenser_push_button",
            collision=True,
        )
        self._button_home_pose = sapien.Pose([x, y, btn_z])
        self._button_pressed_pose = sapien.Pose([x, y, btn_z - bhz])
        self._button_pressed_visual = False

        # One packed mesh containing many individual coffee beans (not a solid block).
        self._add_static_mesh_visual(
            filename=resolve_model_dir("252_coffee_bean") / "reservoir_fill.glb",
            pose=sapien.Pose([x, y, box_z]),
            material=self._opaque_material(self.BEAN_BROWN),
            name="dispenser_reservoir_beans",
        )

        # Nozzle ends short of jar center so the fill column stays visible.
        nozzle_joint_z = self.table_top + self.JAR_HEIGHT + 0.049
        nozzle_outlet_z = self.table_top + self.JAR_HEIGHT + 0.0245
        hopper_front_y = y - by
        jar_x, jar_y = self.jar_xy
        tip_y = jar_y + 0.018
        nozzle_y = 0.5 * (hopper_front_y + tip_y)
        self._add_static_box(
            pose=sapien.Pose([x, nozzle_y, nozzle_joint_z]),
            half_size=[0.006, abs(tip_y - hopper_front_y) * 0.5, 0.005],
            color=[0.42, 0.44, 0.47, 1.0],
            name="dispenser_nozzle_arm",
        )
        self._add_static_box(
            pose=sapien.Pose(
                [jar_x, tip_y, 0.5 * (nozzle_joint_z + nozzle_outlet_z)]
            ),
            half_size=[
                0.006,
                0.006,
                0.5 * (nozzle_joint_z - nozzle_outlet_z),
            ],
            color=[0.36, 0.38, 0.41, 1.0],
            name="dispenser_nozzle_tip",
        )
        self._add_static_box(
            pose=sapien.Pose([jar_x, tip_y, nozzle_outlet_z]),
            half_size=[0.007, 0.007, 0.002],
            color=[0.10, 0.08, 0.06, 1.0],
            name="dispenser_nozzle_opening",
            collision=False,
        )
        self.nozzle_outlet_xyz = np.array(
            [jar_x, jar_y, nozzle_outlet_z], dtype=float
        )

        self.touch_xy = np.asarray([x, y], dtype=float)
        self.touch_top_z = float(btn_z + bhz)
        # Same floor math as ``ReactivePushButtons.min_ee_z_over_key``: EE Z at
        # the tip height that reaches force level 4 (``force_full``).
        max_force = float(self.FORCE_THRESHOLDS[-1])
        stiffness = max(float(self.FORCE_STIFFNESS), 1e-6)
        slack = float(self.FORCE_ENGAGE_SLACK)
        tip_l4 = float(self.touch_top_z) + slack - (max_force / stiffness)
        self._dispenser_ee_z_floor = float(
            tip_l4 - 0.003 + float(self.EE_TO_TCP)
        )

    def _set_button_press_depth(self, depth: float) -> None:
        max_depth = float(getattr(self, "BTN_HALF", (0.0, 0.0, 0.0))[2])
        self._button_target_depth = float(np.clip(depth, 0.0, max_depth))

    def _advance_button_press_visual(self) -> None:
        button = getattr(self, "dispenser_touch_surface", None)
        home = getattr(self, "_button_home_pose", None)
        if button is None or home is None:
            return
        max_depth = float(getattr(self, "BTN_HALF", (0.0, 0.0, 0.0))[2])
        target = float(np.clip(getattr(self, "_button_target_depth", 0.0), 0.0, max_depth))
        current = float(np.clip(getattr(self, "_button_visual_depth", 0.0), 0.0, max_depth))
        step = float(getattr(self, "BUTTON_VISUAL_STEP", 0.0007))
        if target > current:
            current = min(target, current + step)
        elif target < current:
            current = max(target, current - step)
        self._button_visual_depth = current
        self._button_pressed_visual = bool(current > 1e-6)
        try:
            button.set_pose(
                sapien.Pose(
                    [float(home.p[0]), float(home.p[1]), float(home.p[2] - current)],
                    list(home.q),
                )
            )
        except Exception:
            pass

    def _set_button_pressed_visual(self, pressed: bool) -> None:
        """Compatibility helper; robot paths use pressure-derived target depth."""
        max_depth = float(getattr(self, "BTN_HALF", (0.0, 0.0, 0.0))[2])
        self._set_button_press_depth(max_depth if bool(pressed) else 0.0)

    def _button_press_signal(self):
        """Return the best current button press candidate from either arm."""
        if not hasattr(self, "robot"):
            return None
        touch_xy = np.asarray(getattr(self, "touch_xy", None), dtype=float)
        if touch_xy.size != 2:
            return None
        preferred = str(getattr(self, "_pressing_arm_side", ""))
        sides = [preferred] if preferred in ("left", "right") else []
        sides += [side for side in ("left", "right") if side not in sides]
        best = None
        for side in sides:
            try:
                getter = (
                    self.robot.get_left_ee_pose
                    if side == "left"
                    else self.robot.get_right_ee_pose
                )
                ee = np.asarray(getter(), dtype=float)
                # Same virtual tip as ReactivePushButtons / cook_meat keys.
                tcp = np.asarray(ee, dtype=float)
                tcp[2] -= float(self.EE_TO_TCP)
            except Exception:
                try:
                    getter = (
                        self.robot.get_left_tcp_pose
                        if side == "left"
                        else self.robot.get_right_tcp_pose
                    )
                    tcp = np.asarray(getter(), dtype=float)
                except Exception:
                    continue
            xy_dist = float(np.linalg.norm(tcp[:2] - touch_xy))
            if xy_dist > float(self.BTN_TOUCH_XY_TOL):
                continue
            contact_force = float(self._lid_contact_force())
            spring_force = float(
                self.force_stiffness
                * max(0.0, float(self.touch_top_z + self.force_engage_slack - tcp[2]))
            )
            force = max(spring_force, contact_force)
            signal = {"side": side, "tcp": tcp, "force": force}
            if best is None or signal["force"] > best["force"]:
                best = signal
        if best is not None:
            self._pressing_arm_side = best["side"]
        return best

    def _over_dispenser_button(self, side, pose) -> bool:
        """True when EE or TCP XY is over the dispenser push button."""
        touch_xy = getattr(self, "touch_xy", None)
        if touch_xy is None:
            return False
        target = np.asarray(touch_xy, dtype=float)
        tol = max(
            float(self.BTN_TOUCH_XY_TOL) * 1.35,
            float(self.BTN_HALF[0]) + 0.025,
            float(self.BTN_HALF[1]) + 0.025,
        )
        samples = []
        p = np.asarray(pose, dtype=float).reshape(-1) if pose is not None else None
        if p is not None and p.size >= 2:
            samples.append(p[:2].astype(float))
        robot = getattr(self, "robot", None)
        if robot is not None:
            for name in (
                f"get_{side}_tcp_pose",
                f"get_{side}_ee_pose",
            ):
                fn = getattr(robot, name, None)
                if not callable(fn):
                    continue
                try:
                    samples.append(np.asarray(fn()[:2], dtype=float))
                except Exception:
                    continue
        for xy in samples:
            if float(np.linalg.norm(xy - target)) <= tol:
                return True
        return False

    def interactive_ee_z_floor(self, side, pose):
        """EE floor over the key at force level 4 (reactive-button style).

        ``UniversalRobotControls`` *replaces* the table+finger band with this
        value while over the key — same as ``ReactivePushButtons.min_ee_z_over_key``.
        That lets Q reach level 4, then blocks further descent.
        """
        if not self._over_dispenser_button(side, pose):
            return None
        cached = getattr(self, "_dispenser_ee_z_floor", None)
        if cached is not None:
            return float(cached)
        touch_top = getattr(self, "touch_top_z", None)
        if touch_top is None:
            return None
        max_force = float(self.force_thresholds[-1])
        stiffness = max(float(self.force_stiffness), 1e-6)
        slack = float(self.force_engage_slack)
        tip_l4 = float(touch_top) + slack - (max_force / stiffness)
        return float(tip_l4 - 0.003 + float(self.EE_TO_TCP))

    def interactive_teleop_z_speed_scale(self, side, pose, z_delta: float):
        """Slow Q (lowering) over the dispenser for finer press-force control."""
        if float(z_delta) >= 0.0:
            return None
        if not self._over_dispenser_button(side, pose):
            return None
        return float(self.DISPENSE_Z_DOWN_SCALE)

    def _update_button_pressed_visual_from_robot(self) -> None:
        signal = self._button_press_signal()
        if signal is None:
            self._set_button_press_depth(0.0)
        else:
            max_depth = float(getattr(self, "BTN_HALF", (0.0, 0.0, 0.0))[2])
            target = float(
                np.clip(
                    float(signal["force"])
                    / max(float(self.force_thresholds[-1]), 1e-6)
                    * max_depth,
                    0.0,
                    max_depth,
                )
            )
            self._set_button_press_depth(target)
        self._advance_button_press_visual()

    def _build_jar_visual(self, hollow: bool = False):
        """Glass jar visual. ``hollow=True`` for SAPIEN viewer (open interior).

        Camera / expert demos keep the smooth solid transmission cylinder (looks
        correct in offline render). The interactive viewer treats that cylinder
        as an opaque volume, so viewer mode uses a thin alpha-glass shell instead.
        """
        self.jar_visual = self._remove_entity(getattr(self, "jar_visual", None))
        if self.jar is None:
            return

        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT
        bottom_t = self.JAR_BOTTOM_T
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]
        wall_h = h - bottom_t
        wall_half = wall_h * 0.5
        wall_z = bottom_t + wall_half
        glass = self._jar_glass_material(viewer_shell=bool(hollow))

        try:
            pose = self.jar.get_pose()
        except Exception:
            pose = sapien.Pose(
                [float(self.jar_xy[0]), float(self.jar_xy[1]), self.table_top + 0.001]
            )
        vis = sapien.Entity()
        vis.set_name("glass_jar_visual")
        vis.set_pose(pose)
        render_body = sapien.render.RenderBodyComponent()

        floor = sapien.render.RenderShapeCylinder(
            radius=outer_r * 0.98,
            half_length=max(0.0015, bottom_t * 0.5),
            material=glass,
        )
        floor.set_local_pose(sapien.Pose([0.0, 0.0, bottom_t * 0.5], upright_q))
        render_body.attach(floor)

        if hollow:
            # Thin faceted glass shell — empty inside so the coffee column reads.
            wall_t = 0.0024
            n_seg = 36
            wall_radius = outer_r - 0.5 * wall_t
            tangent_half = wall_radius * np.tan(np.pi / n_seg) * 1.03
            for ang in np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False):
                px = float(wall_radius * np.cos(ang))
                py = float(wall_radius * np.sin(ang))
                yaw = float(ang + 0.5 * np.pi)
                q = [
                    float(np.cos(0.5 * yaw)),
                    0.0,
                    0.0,
                    float(np.sin(0.5 * yaw)),
                ]
                panel = sapien.render.RenderShapeBox(
                    [float(tangent_half), float(0.5 * wall_t), float(wall_half)],
                    glass,
                )
                panel.set_local_pose(sapien.Pose([px, py, wall_z], q))
                render_body.attach(panel)
        else:
            wall = sapien.render.RenderShapeCylinder(
                radius=outer_r,
                half_length=wall_half,
                material=glass,
            )
            wall.set_local_pose(sapien.Pose([0.0, 0.0, wall_z], upright_q))
            render_body.attach(wall)

        vis.add_component(render_body)
        self.scene.add_entity(vis)
        self.jar_visual = vis
        self._jar_visual_hollow = bool(hollow)

    def use_viewer_hollow_jar(self):
        """Swap to hollow alpha-glass shell for interactive SAPIEN viewer only."""
        self._build_jar_visual(hollow=True)
        # Rebuild fill after the shell so it composites through the alpha walls.
        self._sync_fill_visual()
        print(
            "[fill_coffee_jar] viewer jar: hollow alpha-glass shell "
            "(coffee fill column visible from the side)"
        )

    def _build_jar(self):
        """Clear glass cylinder (original jar design) — no handle/spout.

        Smooth see-through cylinder via ``RenderShapeCylinder`` (IOR=1) + thin
        floor disk. Collision from the hollow jar mesh (no GLB visual).
        Default visual is the smooth transmission cylinder (demo cameras).
        Interactive viewer calls ``use_viewer_hollow_jar()`` after setup.
        """
        x, y = self.jar_xy
        z0 = self.table_top + 0.001

        col_path = (resolve_model_dir(self.JAR_MODEL) / "collision" / "base0.glb").resolve()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_nonconvex_collision_from_file(filename=str(col_path), scale=[1, 1, 1])
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        self.jar = builder.build(name="glass_jar")
        try:
            self.jar.set_name("glass_jar")
        except Exception:
            pass

        self.jar_bottom_z = self.table_top + self.JAR_BOTTOM_T
        self.jar_fillable_h = self.JAR_HEIGHT - self.JAR_BOTTOM_T
        self._build_jar_visual(hollow=False)

    def _ring_material(self):
        """Opaque saturated red for fill marks (readable through the jar)."""
        rgba = list(self.RING_RED[:3]) + [1.0]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.30)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.30
            mat.metallic = 0.0
        # Mild emission so the bands stay sharp under glass/transmission.
        try:
            emit = [
                float(self.RING_RED[0]) * 0.35,
                float(self.RING_RED[1]) * 0.35,
                float(self.RING_RED[2]) * 0.35,
                1.0,
            ]
            mat.set_emission(emit)
        except Exception:
            try:
                mat.emission = emit
            except Exception:
                pass
        return mat

    def _build_fill_rings(self):
        """Three thick red rings at 25% / 50% / 75% of the jar fill height."""
        x, y = self.jar_xy
        ring_material = self._ring_material()
        ring_mesh = resolve_model_dir(self.JAR_MODEL) / "rings" / "thin_ring.glb"
        outer_r = float(self.JAR_INNER_R) + 0.0035
        xy = float(self.RING_XY_SCALE) * (outer_r / float(self.RING_MESH_RADIUS))
        z_sc = float(self.RING_Z_SCALE)
        scale = [xy, xy, z_sc]
        for frac in (0.25, 0.50, 0.75):
            z = self.jar_bottom_z + frac * self.jar_fillable_h
            self._add_static_mesh_visual(
                filename=ring_mesh,
                pose=sapien.Pose([x, y, z]),
                material=ring_material,
                name=f"fill_ring_{int(frac * 100)}",
                scale=scale,
            )

    # ------------------------------------------------------------------ dispense / fill
    def _beans_per_layer(self) -> int:
        pile_r = self.JAR_INNER_R * self._PILE_R_SCALE
        max_ring = max(1, int((pile_r - self._BEAN_R) / (2.0 * self._BEAN_R * 0.95)))
        n = 1
        for ring in range(1, max_ring + 1):
            n += max(6, int(round(2.0 * math.pi * ring)))
        return n

    def _beans_needed(self) -> int:
        """Beans required to reach the target fill fraction of the jar."""
        return int(math.ceil(float(self.target_fill) * float(self.beans_full)))

    def _beans_in_jar_list(self):
        x, y = self.jar_xy
        r = self.JAR_INNER_R + 0.008
        z_lo = self.jar_bottom_z - 0.005
        z_hi = self.table_top + self.JAR_HEIGHT + 0.02
        out = []
        for b in self.beans:
            p = np.asarray(b.get_pose().p, dtype=float)
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= r * r and z_lo <= p[2] <= z_hi:
                out.append(b)
        return out

    def _count_beans_in_jar(self) -> int:
        return len(self._beans_in_jar_list())

    def _effective_bean_count(self) -> int:
        """Beans that count toward fill (in-jar + still streaming this press)."""
        n = int(getattr(self, "beans_in_jar", 0) or 0)
        if getattr(self, "_press_active", False):
            n += int(getattr(self, "_press_spawned", 0) or 0)
        return int(min(self.beans_full, max(0, n)))

    def _current_fill(self) -> float:
        """Fill fraction vs red-ring scale — proportional to beans dispensed.

        More coffee out → higher level. Uses bean count (not packed max-Z) so
        small force-level differences still move the visible column.
        """
        return float(self._effective_bean_count()) / float(max(1, self.beans_full))

    def _pile_height(self) -> float:
        """Coffee column height above the jar floor (meters), from fill fraction."""
        return float(self._current_fill() * self.jar_fillable_h)

    def _spawn_bean(self, pose: sapien.Pose):
        bean = create_actor(
            self,
            pose=pose,
            modelname=self.BEAN_MODEL,
            model_id=0,
            convex=True,
            is_static=False,
        )
        bean.set_mass(0.003)
        bean.set_name(f"coffee_bean_{len(self.beans)}")
        # Force dark-brown tint and damp so beans settle into a pile.
        try:
            for c in bean.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            s.material.set_base_color([*self.BEAN_BROWN, 1.0])
                        except Exception:
                            pass
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    c.set_linear_damping(1.8)
                    c.set_angular_damping(2.0)
                    try:
                        c.set_linear_velocity([0.0, 0.0, -0.35])
                        c.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
        except Exception:
            pass
        self.beans.append(bean)
        return bean

    def _freeze_beans(self, beans):
        """Freeze beans into a mound whose top matches the bean-count fill height."""
        x, y = self.jar_xy
        pile_r = self.JAR_INNER_R * self._PILE_R_SCALE
        inside = []
        for bean in beans:
            p = np.asarray(bean.get_pose().p, dtype=float)
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= (self.JAR_INNER_R - 0.002) ** 2:
                inside.append(bean)
        if not inside:
            return

        bean_r = self._BEAN_R
        bean_h = self._BEAN_H
        # Target top of the pile from dispensed count (same scale as fill column).
        fill = min(1.0, float(len(inside)) / float(max(1, self.beans_full)))
        target_top = self.jar_bottom_z + fill * self.jar_fillable_h
        max_ring = max(1, int((pile_r - bean_r) / (2.0 * bean_r * 0.95)))
        positions = []
        layer = 0
        while len(positions) < len(inside):
            z = self.jar_bottom_z + bean_h * 0.5 + layer * bean_h * 0.92
            # Cap so we never pack above the rim.
            if z > self.jar_bottom_z + self.jar_fillable_h - bean_h * 0.2:
                z = self.jar_bottom_z + self.jar_fillable_h - bean_h * 0.2
            slots = [(0.0, 0.0)]
            for ring in range(1, max_ring + 1):
                n_ring = max(6, int(round(2.0 * math.pi * ring)))
                rad = ring * (2.0 * bean_r * 0.95)
                if rad > pile_r - bean_r:
                    break
                for k in range(n_ring):
                    ang = 2.0 * math.pi * k / n_ring + (0.15 * layer)
                    slots.append((rad * math.cos(ang), rad * math.sin(ang)))
            for sx, sy in slots:
                if len(positions) >= len(inside):
                    break
                jx = float(np.random.uniform(-0.0008, 0.0008))
                jy = float(np.random.uniform(-0.0008, 0.0008))
                positions.append((x + sx + jx, y + sy + jy, z))
            layer += 1
            if layer > 60:
                break

        # Stretch / compress layers so the mound top tracks fill fraction.
        if positions:
            z0 = self.jar_bottom_z + bean_h * 0.5
            z_max = max(pz for _, _, pz in positions)
            span = max(1e-6, z_max - z0)
            scale = max(0.15, (target_top - z0) / span)
            positions = [
                (px, py, z0 + (pz - z0) * scale) for px, py, pz in positions
            ]

        for bean, (px, py, pz) in zip(inside, positions):
            yaw = float(np.random.uniform(0, 2 * np.pi))
            qx, qy, qz, qw = t3d.euler.euler2quat(0.25, 0.15, yaw)
            rest = sapien.Pose([px, py, pz], [qw, qx, qy, qz])
            # Actor wrapper has no set_pose — must move the underlying entity
            # or the render mesh stays on the jar floor while physx reports high.
            try:
                bean.actor.set_pose(rest)
            except Exception:
                pass
            try:
                for component in bean.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
                        component.set_kinematic(True)
                        component.set_kinematic_target(rest)
            except Exception:
                pass

    def _sync_fill_visual(self):
        """Opaque coffee column proportional to beans dispensed (vs red rings).

        Top-down cameras flatten a real bean stack into a small floor sprinkle, so
        the fill body is what makes force-level differences readable. Interactive
        viewer uses a fuller opaque column so the level reads through the hollow
        alpha-glass shell (same approach as measure_ingredient oil).
        """
        self.fill_visual = self._remove_entity(getattr(self, "fill_visual", None))

        h = float(self._pile_height())
        if h < 0.003:
            return

        x, y = self.jar_xy
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]
        half = h * 0.5
        viewer_shell = bool(getattr(self, "_jar_visual_hollow", False))
        if viewer_shell:
            mat = self._viewer_coffee_material()
            fill_r = self.JAR_INNER_R * 0.97
        else:
            mat = sapien.render.RenderMaterial(base_color=[0.22, 0.10, 0.04, 0.92])
            try:
                mat.set_transmission(0.0)
                mat.set_roughness(0.85)
                mat.set_metallic(0.0)
            except Exception:
                pass
            fill_r = self.JAR_INNER_R * 0.90

        ent = sapien.Entity()
        ent.set_name("coffee_fill_visual")
        ent.set_pose(sapien.Pose([x, y, self.jar_bottom_z]))
        body = sapien.render.RenderBodyComponent()
        col = sapien.render.RenderShapeCylinder(
            radius=fill_r,
            half_length=half,
            material=mat,
        )
        col.set_local_pose(sapien.Pose([0.0, 0.0, half], upright_q))
        body.attach(col)
        ent.add_component(body)
        self.scene.add_entity(ent)
        self.fill_visual = ent

    def _sim_dt(self) -> float:
        try:
            return float(self.scene.get_timestep())
        except Exception:
            return 1.0 / 250.0

    def _lid_contact_force(self) -> float:
        """PhysX contact force (N) between gripper links and the push button."""
        if not hasattr(self, "robot"):
            return 0.0
        dt = self._sim_dt()
        btn = "dispenser_push_button"
        grip = set(getattr(self.robot, "gripper_name", []) or [])
        imp = 0.0
        try:
            for contact in self.scene.get_contacts():
                n0 = contact.bodies[0].entity.name
                n1 = contact.bodies[1].entity.name
                if btn not in (n0, n1):
                    continue
                other = n1 if n0 == btn else n0
                if grip and other not in grip:
                    # Still accept left-arm / finger-like links.
                    o = other.lower()
                    if not (
                        o.startswith("left")
                        or o.startswith("right")
                        or "finger" in o
                        or "pad" in o
                        or "hand" in o
                        or "gripper" in o
                    ):
                        continue
                for pt in getattr(contact, "points", None) or []:
                    v = np.asarray(getattr(pt, "impulse", [0.0, 0.0, 0.0]), dtype=float)
                    imp += float(np.linalg.norm(v))
        except Exception:
            return 0.0
        return float(imp / max(dt, 1e-6))

    def _lid_spring_force(self) -> float:
        """Spring proxy (N) from virtual key compression.

        Position-controlled presses often yield sparse contact impulses against a
        static button; this proxy makes pressure a deterministic function of
        how far the gripper pushes into the blue key's travel range.
        """
        signal = self._button_press_signal()
        if signal is None:
            return 0.0
        tip_z = float(signal["tcp"][2])
        engage_z = float(self.touch_top_z + self.force_engage_slack)
        pen = max(0.0, engage_z - tip_z)
        return float(self.force_stiffness * pen)

    def _lid_press_force(self) -> float:
        """Effective button press force (N) used for the 4 dispense thresholds.

        Position-controlled presses into a static button produce huge, noisy PhysX
        impulse spikes (often >100 N) that would collapse every press to the
        hardest level. The spring force is a stable, depth-correlated signal:
        more key compression means more pressure and therefore more beans.
        Contact impulse is still available via ``_lid_contact_force`` for debug.
        """
        return float(self._lid_spring_force())

    def _force_level(self, force_n: float) -> int:
        """Map force (N) → level in {0,1,2,3,4} (0 = below first threshold)."""
        level = 0
        for i, thr in enumerate(self.force_thresholds):
            if float(force_n) >= float(thr):
                level = i + 1
        return int(level)

    def _beans_for_level(self, level: int) -> int:
        """Bean count for dispense level in {1,2,3,4} (0 → none)."""
        level = int(level)
        if level <= 0:
            return 0
        level = min(level, len(self.beans_per_force_level))
        return int(self.beans_per_force_level[level - 1])

    def _beans_for_force(self, force_n: float) -> int:
        """Map force (N) → bean count via the four thresholds."""
        return self._beans_for_level(self._force_level(force_n))

    def _spawn_one_dispensed_bean(self):
        """Drop a single bean from the nozzle into the jar."""
        ang = float(np.random.uniform(0, 2 * np.pi))
        rad = float(np.random.uniform(0.0, 0.014))
        ox = rad * math.cos(ang)
        oy = rad * math.sin(ang)
        yaw = float(np.random.uniform(0, 2 * np.pi))
        qx, qy, qz, qw = t3d.euler.euler2quat(0.4, 0.2, yaw)
        pose = sapien.Pose(
            [
                self.nozzle_outlet_xyz[0] + ox,
                self.nozzle_outlet_xyz[1] + oy,
                self.nozzle_outlet_xyz[2],
            ],
            [qw, qx, qy, qz],
        )
        self._spawn_bean(pose)

    def _start_press(self, *, require_release: bool = True):
        if self._press_active or self.beans_in_jar >= self.beans_full:
            return
        # After a dispense, the button must be released before another shot.
        if require_release and bool(getattr(self, "_awaiting_release", False)):
            return
        self._press_active = True
        self._dispensing = True
        self._awaiting_release = False
        if self._metric_first_press is None:
            self._metric_first_press = self._metric_step()
        # Each new press restarts the post-release idle scoring window.
        self._press_idle_s = 0.0
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0
        self._press_dispense_level = 0

    def _update_dispense_level(self, force_n: float) -> int:
        """Track the in-press dispense level from current force (up to level 4).

        Soft press → that level's beans; push harder in the same press and the
        target rises to the achieved force level (max final level). Continuous
        hold still only dispenses once — release is required to re-arm.
        """
        raw = int(self._force_level(force_n))
        if raw > int(getattr(self, "_press_dispense_level", 0) or 0):
            self._press_dispense_level = raw
        self._press_force_level = int(getattr(self, "_press_dispense_level", 0) or 0)
        return int(self._press_force_level)

    def _tick_press(self):
        """While the button is held, stream beans for the current dispense level."""
        if not self._press_active:
            return
        if self.beans_in_jar + self._press_spawned >= self.beans_full:
            return
        self._press_steps += 1
        self._press_hold_s = self._press_steps * self._sim_dt()
        force_n = self._lid_press_force()
        if force_n > self._press_peak_force:
            self._press_peak_force = force_n
        level = self._update_dispense_level(self._press_peak_force)
        target = self._beans_for_level(level)
        target = min(target, self.beans_full - self.beans_in_jar)
        while self._press_spawned < target:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            # Raise the fill column as beans stream out (force → amount → level).
            if self._press_spawned == target or self._press_spawned % 3 == 0:
                self._sync_fill_visual()
            # Small settle between beans so the cascade stays visible.
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

    def _end_press(self):
        """Finalize a press: top up to the dispense level, settle, freeze.

        Arms ``_awaiting_release`` so a continuous hold cannot start another
        dispense until the gripper leaves the button.
        """
        if not self._press_active:
            return
        # One last pressure sample in case the peak arrived on the release frame.
        force_n = self._lid_press_force()
        if force_n > self._press_peak_force:
            self._press_peak_force = force_n
        level = self._update_dispense_level(self._press_peak_force)
        # Ignore ghost touches during approach (below first threshold → no beans).
        if level <= 0 and int(self._press_spawned) <= 0:
            self._press_active = False
            self._dispensing = False
            self._press_steps = 0
            self._press_spawned = 0
            self._press_hold_s = 0.0
            self._press_peak_force = 0.0
            self._press_force_level = 0
            self._press_dispense_level = 0
            return
        want = self._beans_for_level(level)
        want = min(want, self.beans_full - self.beans_in_jar)
        while self._press_spawned < want:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            if self._press_spawned == want or self._press_spawned % 3 == 0:
                self._sync_fill_visual()
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

        self.press_count += 1
        spawned = int(self._press_spawned)
        peak = float(self._press_peak_force)
        # Close the press session before settle so touch-detect cannot re-enter.
        self._press_active = False
        # Must lift off before another press can arm (blocks continuous fill).
        self._awaiting_release = True
        # Idle window starts only after lift-off; keep it zero through settle.
        self._press_idle_s = 0.0
        self._dwell(self.SETTLE_STEPS)
        self._freeze_beans(self.beans)
        self._dwell(6)
        self.beans_in_jar = self._count_beans_in_jar()
        self._sync_fill_visual()
        fill = self._current_fill()
        self._latch_press_metric(fill)
        print(
            f"[fill_coffee_jar] force={peak:.1f}N level={level}/4 → {spawned} beans "
            f"fill={fill:.0%} "
            f"(thresholds {list(self.force_thresholds)} → {list(self.beans_per_force_level)})"
        )
        self._dispensing = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0
        self._press_dispense_level = 0

    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            if not getattr(self, "_dispensing", False):
                self._update_kinematic_tasks()
            else:
                super()._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _touch_tip_pose(self, tip_z_above_top: float):
        tcp_z = self.touch_top_z + tip_z_above_top
        ee_z = tcp_z + self.EE_TO_TCP
        return [float(self.touch_xy[0]), float(self.touch_xy[1]), ee_z, *GRASP_DIRECTION_DIC["top_down"]]

    def _press_dispenser(self, arm_tag: ArmTag, force_level: int = 4):
        """Press the top button to the requested force level (1–4) and dispense."""
        self._pressing_arm_side = str(arm_tag)
        level = int(np.clip(force_level, 1, 4))
        depth = float(self.press_depths[level - 1])
        # Press with the WSG fingers closed so the gripper behaves like one
        # rigid button-pushing tool rather than straddling the dispenser top.
        self.move(self.close_gripper(arm_tag))
        if not self.plan_success:
            print("[fill_coffee_jar] could not close gripper for dispenser press")
            return False
        # Two-stage approach: high waypoint, then drop to hover (more reliable IK).
        high_dis = self.KEY_HOVER_DIS + 0.08
        self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(high_dis)))
        if not self.plan_success:
            # Fallback: direct hover.
            self.plan_success = True
            self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(self.KEY_HOVER_DIS)))
        else:
            self.move(self.move_by_displacement(arm_tag, z=-(high_dis - self.KEY_HOVER_DIS)))
        if not self.plan_success:
            print(f"[fill_coffee_jar] hover failed to {self._touch_tip_pose(self.KEY_HOVER_DIS)}")
            return False
        self.move(self.move_by_displacement(arm_tag, z=-depth))
        if not self.plan_success:
            return False

        # Sample peak force while pressed (expert path; does not rely on the
        # per-step touch detector, which can miss during move()).
        self._awaiting_release = False
        self._start_press(require_release=False)
        hold_steps = max(1, int(round(self.press_sample_s / self._sim_dt())))
        for _ in range(hold_steps):
            measured = self._lid_press_force()
            if measured > self._press_peak_force:
                self._press_peak_force = measured
            # Expert requests an explicit level — allow that level in one shot.
            self._press_dispense_level = max(
                int(getattr(self, "_press_dispense_level", 0) or 0), int(level)
            )
            self._press_force_level = int(self._press_dispense_level)
            self._tick_press()
            if not self._press_active:
                break
            super()._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._press_steps % max(1, int(self.save_freq)) == 0):
                self._take_picture()
        self._end_press()
        self._awaiting_release = False

        self.move(self.move_by_displacement(arm_tag, z=depth))
        self._dwell(8)
        return True

    def _detect_lid_touch(self):
        if self.dispenser_touch_surface is None or not hasattr(self, "robot"):
            return
        signal = self._button_press_signal()
        force_n = float(self._lid_press_force()) if signal is not None else 0.0
        touching = bool(signal is not None and force_n > 0.5)
        # After a completed dispense, ignore further contact until a real release.
        # Idle scoring starts only once the gripper has lifted off.
        if bool(getattr(self, "_awaiting_release", False)):
            if not touching:
                self._awaiting_release = False
                self._touch_latched = False
                self._update_press_idle(touching=False)
            else:
                self._touch_latched = True
                self._press_idle_s = 0.0
            return
        if touching and not self._touch_latched:
            self._start_press()
        if touching and self._press_active:
            self._tick_press()
            # Finish once the hardest allowed level has been held briefly.
            if (
                int(getattr(self, "_press_dispense_level", 0) or 0) >= 4
                and self._press_hold_s >= self.press_sample_s
            ):
                self._end_press()
        if (not touching) and self._touch_latched and self._press_active:
            self._end_press()
        self._touch_latched = touching
        self._update_press_idle(touching=bool(touching))

    def _update_press_idle(self, *, touching: bool) -> None:
        """Post-press idle timer for success/failure scoring.

        - Before the first real press: do not count (never score).
        - On every press / while held / while settling / until lift-off: reset to 0.
        - After release: accumulate; at ``idle_score_sec`` evaluate fill vs target.
        """
        busy = bool(
            touching
            or getattr(self, "_press_active", False)
            or getattr(self, "_dispensing", False)
            or getattr(self, "_awaiting_release", False)
        )
        if busy:
            self._press_idle_s = 0.0
            return
        if int(getattr(self, "press_count", 0) or 0) <= 0:
            self._press_idle_s = 0.0
            return
        self._press_idle_s = float(getattr(self, "_press_idle_s", 0.0)) + float(
            self._sim_dt()
        )

    def _fill_ready_to_score(self) -> bool:
        """True after a real press, then ``idle_score_sec`` with the button released."""
        if int(getattr(self, "press_count", 0) or 0) <= 0:
            return False
        if bool(getattr(self, "_press_active", False)):
            return False
        if bool(getattr(self, "_awaiting_release", False)):
            return False
        need = float(getattr(self, "idle_score_sec", self.IDLE_SCORE_SEC))
        return float(getattr(self, "_press_idle_s", 0.0)) + 1e-9 >= need

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._update_button_pressed_visual_from_robot()
        # While actively holding we still need touch edge detection for release
        # / max-duration stop; `_tick_press` is driven from `_detect_lid_touch`.
        if getattr(self, "_dispensing", False) and not self._press_active:
            # Settle after a shot — idle window stays at 0 until lift-off.
            self._update_press_idle(touching=False)
            return
        self._detect_lid_touch()

    # ------------------------------------------------------------------ expert / success
    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_first_press = None   # button first held down
        self._metric_last_press = None    # last dispense finished (fill committed)
        self._metric_final_fill = None    # fill fraction right after that dispense

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_press_metric(self, fill):
        """Called at the end of every dispense — LAST press wins.

        Each shot changes the fill, so the decisive commit is the final one; an
        earlier under-dose that was topped up is not the value being scored.
        """
        try:
            self._metric_last_press = self._metric_step()
            self._metric_final_fill = float(fill)
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `fill_latency_steps` — steps from the first button press until the last
        dispense finished: the whole dosing window, including every top-up shot.
        extra2 `fill_error_norm` — |final fill - target_fill| / fill_tol. LOWER is
        better; <= 1.0 means the jar landed inside the success band.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a, b = self._metric_first_press, self._metric_last_press
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["fill_latency_steps"] = lat
        out["fill_latency_s"] = None if lat is None else round(float(lat) * dt, 4)
        out["first_press_step"] = None if a is None else int(a)
        try:
            out["press_count"] = int(self.press_count)
        except Exception:
            out["press_count"] = None

        f = self._metric_final_fill
        try:
            tol = max(float(self.fill_tol), 1e-6)
            out["fill_error_norm"] = (
                None if f is None
                else round(abs(float(f) - float(self.target_fill)) / tol, 4))
            out["target_fill"] = round(float(self.target_fill), 4)
        except Exception:
            out["fill_error_norm"] = None
            out["target_fill"] = None
        out["final_fill"] = None if f is None else round(float(f), 4)
        return out

    def _fill_band(self):
        lo = float(self.target_fill) - float(self.fill_tol)
        hi = float(self.target_fill) + float(self.fill_tol)
        return max(0.0, lo), min(1.0, hi)

    def _force_level_for_remaining(self, fill: float) -> int:
        """Pick the smallest force level that enters the success band without overshoot."""
        _, hi = self._fill_band()
        lo, _ = self._fill_band()
        if fill + 1e-6 >= lo:
            return 1
        need_frac = max(0.0, float(self.target_fill) - float(fill))
        need_beans = int(math.ceil(need_frac * float(self.beans_full)))
        # Prefer a level that lands inside [lo, hi].
        best = 4
        for i, n in enumerate(self.beans_per_force_level):
            pred = fill + float(n) / float(self.beans_full)
            if pred + 1e-6 >= lo and pred - 1e-6 <= hi:
                return i + 1
            if n >= need_beans:
                best = i + 1
                break
        return int(best)

    def play_once(self):
        # Station lives on the free half opposite the stove; pick that arm.
        arm = ArmTag("left" if float(self.dispenser_xy[0]) <= 0.0 else "right")
        self.move(self.close_gripper(arm))
        if not self.plan_success:
            print("[fill_coffee_jar] close_gripper failed")
            return self.info

        needed = self._beans_needed()
        lo, hi = self._fill_band()
        max_presses = int(math.ceil(needed / max(1, self.beans_per_press_min))) + 4
        for i in range(max_presses):
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            if fill + 1e-6 >= lo:
                break
            if not self.plan_success:
                print(f"[fill_coffee_jar] plan failed before press {i}")
                break
            level = self._force_level_for_remaining(fill)
            # Fresh plan flag each press (a prior miss shouldn't abort the episode).
            self.plan_success = True
            ok = self._press_dispenser(arm, force_level=level)
            if not ok:
                self.plan_success = True
                ok = self._press_dispenser(arm, force_level=level)
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            print(
                f"[fill_coffee_jar] press={i} force_lvl={level} ok={ok} "
                f"beans={self.beans_in_jar}/{needed} fill={fill:.0%} "
                f"(band {lo:.0%}–{hi:.0%}) plan={self.plan_success}"
            )

        # Target band reached → withdraw, then idle long enough to score.
        if self.plan_success:
            self.move(self.move_by_displacement(arm, z=0.08))
        idle_need = float(getattr(self, "idle_score_sec", self.IDLE_SCORE_SEC))
        idle_steps = max(1, int(round(idle_need / max(self._sim_dt(), 1e-6))))
        self._dwell(idle_steps)

        if self.check_success():
            self.plan_success = True

        level_pct = int(round(self.target_fill * 100))
        self.info["info"] = {
            "{A}": "coffee dispenser",
            "{B}": f"glass jar ({level_pct}% line)",
            "{C}": "252_coffee_bean/base0",
            "{a}": str(arm),
            "{L}": f"{level_pct}%",
            "{side}": str(getattr(self, "stove_side", "right")),
            "{burner}": str(getattr(self, "kettle_burner", "")),
        }
        return self.info

    def check_success(self):
        """Success after idle: fill inside target_fill ± fill_tol (default ±5%)."""
        if not getattr(self, "layout_ok", True):
            return False
        if not self._fill_ready_to_score():
            return False
        self.beans_in_jar = self._count_beans_in_jar()
        fill = self._current_fill()
        lo, hi = self._fill_band()
        return bool(self.beans_in_jar > 0 and lo - 1e-3 <= fill <= hi + 1e-3)

    def get_score(self) -> float:
        """Partial score from |fill − target| after idle (under and over both count).

        Success band (±``fill_tol``, default ±5%) → 1.0. Outside it, abs error
        bands ``[10,8.5)%`` / ``[8.5,7)%`` / ``[7,5)%`` → 0.25 / 0.5 / 0.75.
        Not ready to score / empty jar / error ≥ 10 pp → 0.
        """
        if not getattr(self, "layout_ok", True):
            return 0.0
        if not self._fill_ready_to_score():
            return 0.0
        self.beans_in_jar = self._count_beans_in_jar()
        if int(self.beans_in_jar) <= 0:
            return 0.0
        if self.check_success():
            return 1.0
        fill = float(self._current_fill())
        target = float(self.target_fill)
        err_pct = abs(fill - target) * 100.0
        return float(score_descending_bands(err_pct, self.PARTIAL_ABS_BANDS))

    def get_obs(self):
        obs = super().get_obs()
        lo, hi = self._fill_band()
        obs["coffee_jar"] = {
            "target_fill": float(self.target_fill),
            "fill_tol": float(self.fill_tol),
            "idle_score_sec": float(getattr(self, "idle_score_sec", self.IDLE_SCORE_SEC)),
            "press_idle_s": float(getattr(self, "_press_idle_s", 0.0)),
            "ready_to_score": bool(self._fill_ready_to_score()),
            "fill_lo": float(lo),
            "fill_hi": float(hi),
            "fill": float(self._current_fill()),
            "pile_height": float(self._pile_height()),
            "beans_in_jar": int(self.beans_in_jar),
            "beans_full": int(self.beans_full),
            "force_thresholds": [float(x) for x in self.force_thresholds],
            "beans_per_force_level": [int(x) for x in self.beans_per_force_level],
            "lid_force": float(self._lid_press_force()) if self._loaded else 0.0,
            "press_count": int(self.press_count),
            "dispenser_xy": [float(x) for x in self.dispenser_xy],
            "jar_xy": [float(x) for x in self.jar_xy],
            "layout_ok": bool(getattr(self, "layout_ok", True)),
            "scene_id": int(self.scene_id),
            "coffee_box_id": int(getattr(self, "coffee_box_id", -1)),
            "milk_box_id": int(getattr(self, "milk_box_id", -1)),
            "mug_id": int(getattr(self, "mug_id", -1)),
            "kettle_id": int(getattr(self, "kettle_id", -1)),
            "kettle_burner": str(getattr(self, "kettle_burner", "")),
            "stove_side": str(getattr(self, "stove_side", "right")),
            "range_xy": list(np.asarray(getattr(self, "range_xy", (0, 0)), dtype=float)),
            "partial_score": float(self.get_score()),
        }
        return obs
