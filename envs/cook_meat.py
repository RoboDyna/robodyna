"""Cook-meat task with contact-gated doneness and collision-aware placement.

Author: Rui Heng Yang
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Sequence, TypeAlias

import numpy as np
import sapien
import sapien.render
import trimesh

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils.action import ArmTag
from .utils.create_actor import UnStableError, create_actor, create_box, resolve_model_dir
from .utils.key_symbol import attach_key_symbol, sync_key_symbol
from .utils.rand_create_actor import rand_pose
from .utils.reactive_button import ReactivePushButtons

import transforms3d as t3d


AABB: TypeAlias = tuple[float, float, float, float]


class cook_meat(Base_Task):
    """Cook a steak into a configured doneness range via a cook key.

    Default: place steak on the pan, press the key to latch cooking ON (key
    stays down), press again while ON to latch OFF (cooking stops on that
    press; stove does not turn off by itself). Success is doneness-in-range
    at shutoff — steak may stay on the pan.

    Options (``task_args.cook_meat``; independent toggles):
      Opt 1 — hold cook  →  ``cook_button_enabled`` (**default: false**)
          Hold the key to cook while the steak is on the pan (springs up on
          release). Success/failure is evaluated on the **first release** after
          cooking started (not while the key is still held). CLI:
          ``--task-arg cook_button_enabled=true`` / ``--option 1``.
      Opt 2 — dual setup  →  ``dual_setup_enabled`` (**default: false**)
          Mirror a second station with ≥10 cm clearance; both arms cook.
          Either steak past the upper doneness bound fails the episode.
          CLI: ``--task-arg dual_setup_enabled=true`` or ``--option 2``.
      Opt 1+2 — dual stations with hold-to-cook keys. Same overcook fail.

    ``max_episode_steps`` (default 15000) caps eval ``step_lim`` and collection
    length for every scenario (default / Opt1 / Opt2 / Opt1+2).

    Keycap is green when up and red when depressed (latched ON, or held in
    Opt 1 / Opt 1+2).
    """

    COOK_STEPS_DEFAULT: ClassVar[int] = 686  # −20% speed vs prior 549
    COOK_SPEED_JITTER_DEFAULT: ClassVar[float] = 0.20  # per-ep cook_steps ~ U(nom×(1±j))
    TARGET_DONENESS_DEFAULT: ClassVar[float] = 0.5
    MAX_EPISODE_STEPS_DEFAULT: ClassVar[int] = 15000  # eval / collection episode cutoff
    COOK_BUTTON_ENABLED_DEFAULT: ClassVar[bool] = False  # false=latch; true=hold (Opt 1)
    DUAL_SETUP_ENABLED_DEFAULT: ClassVar[bool] = False  # Opt 2
    # Opt 1: require this many consecutive unengaged steps before first-release
    # shutoff. Prevents tip/spring flicker from ending the episode mid-press.
    HOLD_RELEASE_CONFIRM_STEPS: ClassVar[int] = 12
    TARGET_DONENESS_RANGE_DEFAULT: ClassVar[tuple[float, float]] = (0.40, 0.60)
    TARGET_DONENESS_RANGE_JITTER_DEFAULT: ClassVar[float] = 0.0
    # Legacy fallback when target_doneness_range is absent.
    COOK_DONENESS_TOL_DEFAULT: ClassVar[float] = 0.08
    # Colored keycap + thin black base (marble / punch_dual_holes styling).
    KEY_HALF: ClassVar[tuple[float, float, float]] = (0.020, 0.020, 0.014)
    KEY_BASE_HALF: ClassVar[tuple[float, float, float]] = (0.032, 0.032, 0.005)
    KEY_BASE_COLOR: ClassVar[tuple[float, float, float]] = (0.08, 0.08, 0.08)
    KEY_COLOR_UP: ClassVar[tuple[float, float, float]] = (0.18, 0.78, 0.28)  # green
    KEY_COLOR_DOWN: ClassVar[tuple[float, float, float]] = (0.85, 0.10, 0.10)  # red
    KEY_COLOR: ClassVar[tuple[float, float, float]] = (0.18, 0.78, 0.28)  # alias = up
    KEY_OFFSET_X_DEFAULT: ClassVar[float] = 0.14   # m; |lateral| offset from pan center
    KEY_Y_BIAS_DEFAULT: ClassVar[float] = -0.04    # m; toward robot (−y) from pan center
    KEY_PRESS_DEPTH_DEFAULT: ClassVar[float] = 0.055
    KEY_PRESS_XY_DEFAULT: ClassVar[float] = 0.06
    # EE frame sits ~EE_TO_TCP above the TCP; a true key contact leaves ee_z ≈
    # key_top_z + EE_TO_TCP. CuRobo often stops a few cm short of the mesh, so
    # allow ~key_top_z + 0.20 (matches dispense_gummy 0.17 with margin).
    KEY_PRESS_DZ_DEFAULT: ClassVar[float] = 0.20
    KEY_HOVER_DIS_DEFAULT: ClassVar[float] = 0.06  # m above key top before press
    EE_TO_TCP: ClassVar[float] = 0.12  # EE frame → TCP offset (dispense_gummy convention)
    # Skillet mesh AABB includes a long handle; board only needs the bowl clear.
    PAN_BOWL_HALF: ClassVar[float] = 0.095
    # Key must clear this radius from pan center (bowl + margin so it never sits on the rim).
    KEY_PAN_CLEAR_HALF: ClassVar[float] = 0.12
    # XY clearance between pan / board / key footprints within one station.
    PROP_CLEARANCE: ClassVar[float] = 0.03
    # Minimum gap between left and right pan bowls (Opt 2 / Opt 1+2).
    DUAL_STATION_CLEARANCE: ClassVar[float] = 0.10
    # Dual pan |x|; bowls need ≥ DUAL_STATION_CLEARANCE, so |x| ≥ bowl_half + 0.05.
    DUAL_PAN_NOM_X: ClassVar[float] = 0.15
    # Default (right-side) stable centers; Opt 2 / left mirror by negating X.
    PAN_NOM_X: ClassVar[float] = 0.10
    PAN_NOM_Y: ClassVar[float] = -0.10
    # Board sits mostly +y of the pan (clear of bowl, in head view).
    BOARD_OFF_X: ClassVar[float] = 0.02
    BOARD_OFF_Y: ClassVar[float] = 0.22
    # Small diversity only — keep the layout near the stable nominals.
    PLACE_JITTER_XY_DEFAULT: ClassVar[float] = 0.01   # ±1 cm
    PLACE_YAW_LIM_DEFAULT: ClassVar[float] = float(np.pi / 18)  # ±10° about +Z
    KEY_MIN_GAP: ClassVar[float] = 0.04  # min XY gap key ↔ pan/board
    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    BOARD_BASE_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    # cooking gradient: raw red -> MEDIUM warm red-brown (at the 0.5 target) -> well-done dark brown.
    # 0.5 is deliberately a clear MEDIUM (still reddish), not full brown, so picking at the 0.5
    # target reads visually as "medium", not "fully cooked".
    COLOR_STOPS: ClassVar[list[tuple[float, list[float]]]] = [
        (0.0, [1.00, 0.12, 0.09]),    # raw: vivid saturated red
        (0.5, [0.66, 0.30, 0.14]),    # medium: warm red-brown (clearly transitional)
        (1.0, [0.16, 0.08, 0.04]),    # well done: dark brown
    ]

    def setup_demo(self, **kwargs: Any) -> None:
        """Initialize one episode from the shared task configuration."""
        self._cook_cfg = dict(kwargs.get("task_args", {}).get("cook_meat", {}))
        self._apply_legacy_option()
        self._ep_seed = int(kwargs.get("seed", 0))
        self._reset_metric_state()
        super()._init_task_env_(**kwargs)
        # Base eval may overwrite step_lim from yaml ``eval_step_limit`` after
        # load_actors; re-apply the task_args cutoff for every scenario.
        self._apply_max_episode_steps()

    def _apply_max_episode_steps(self) -> None:
        """Apply ``max_episode_steps`` to eval ``step_lim`` and save cutoff."""
        limit = int(
            getattr(self, "max_episode_steps", self.MAX_EPISODE_STEPS_DEFAULT)
        )
        limit = max(1, limit)
        self.max_episode_steps = limit
        self.step_lim = limit
        self._max_episode_steps = limit
        self._episode_timed_out = False

    def _apply_legacy_option(self) -> None:
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / cook_button / button → Opt 1 cook_button_enabled=true (hold-to-cook)
        2 / dual / dual_setup → Opt 2 dual_setup_enabled=true
        """
        legacy = self._cook_cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "cook_button_enabled",
            "1": "cook_button_enabled",
            "cook_button": "cook_button_enabled",
            "cook_button_enabled": "cook_button_enabled",
            "button": "cook_button_enabled",
            2: "dual_setup_enabled",
            "2": "dual_setup_enabled",
            "dual": "dual_setup_enabled",
            "dual_setup": "dual_setup_enabled",
            "dual_setup_enabled": "dual_setup_enabled",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        if key == "cook_button_enabled":
            self._cook_cfg["cook_button_enabled"] = True
        elif key == "dual_setup_enabled":
            self._cook_cfg["dual_setup_enabled"] = True
        else:
            raise ValueError(
                "cook_meat option must be 1/cook_button_enabled or "
                "2/dual_setup_enabled (or set those keys directly)"
            )

    def _option_label(self) -> str:
        parts: list[str] = []
        if getattr(self, "cook_button_enabled", False):
            parts.append("option 1")
        if getattr(self, "dual_setup_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "default"

    @property
    def use_cook_button(self) -> bool:
        """Cook keys are always present (default latch or Opt 1 hold)."""
        return True

    @property
    def use_hold_cook(self) -> bool:
        """Opt 1 / Opt 1+2: hold key to cook (vs default latch ON/OFF)."""
        return bool(getattr(self, "cook_button_enabled", False))

    @staticmethod
    def _union_aabb(aabbs: Sequence[AABB]) -> AABB:
        """Axis-aligned union of one or more XY AABBs."""
        return (
            min(a[0] for a in aabbs),
            min(a[1] for a in aabbs),
            max(a[2] for a in aabbs),
            max(a[3] for a in aabbs),
        )

    @staticmethod
    def _expand_aabb(aabb: AABB, margin: float) -> AABB:
        """Expand an XY AABB by ``margin`` on every side."""
        m = float(margin)
        return (aabb[0] - m, aabb[1] - m, aabb[2] + m, aabb[3] + m)

    # ------------------------------------------------------- footprint checks
    # rand_pose()/create_actor() do NOT check for overlap between objects, and add_prohibit_area()
    # only feeds the eval-time trajectory-extension check (_base_task.py) -- there is no built-in
    # guard against two hand-placed static props (e.g. the pan and the board) landing on top of
    # each other. Rather than approximate each object's footprint from model_data's "extents"
    # field, use the REAL collision mesh: script/create_object_data.py:275 computes "extents" via
    # trimesh's `bounding_box_oriented` -- a MINIMUM-VOLUME oriented box that can be tilted relative
    # to the mesh's own local axes (this is pronounced for the skillet, an irregular shape with a
    # handle: extents-based math overestimates its world footprint by ~35% in x, ~10% in y versus
    # the actual mesh, measured directly). Loading the real mesh and rotating/scaling its actual
    # vertices gives the true footprint regardless of how any given asset's OMBB happens to be
    # oriented, and needs no per-asset special-casing.
    _MESH_CACHE: ClassVar[dict[str, np.ndarray]] = {}

    @staticmethod
    def _applied_scale(modelname: str, model_id: int, scale_mult: float) -> float:
        """Return the scale applied to raw collision-mesh vertices."""
        # The scale create_actor() actually bakes into the mesh is model_data["scale"] * scale_mult
        # (create_actor.py:542,552,564). model_data's raw collision-glb vertices are in UN-scaled
        # mesh units, so the footprint helpers below MUST be fed this same product -- feeding only
        # scale_mult inflates the footprint by 1/authored_scale (e.g. for small authored scales).
        # Assets with no "scale" key (e.g. 104_board) get authored scale 1.0, matching create_actor's
        # (1,1,1) default for that case.
        try:
            model_data_path = str(resolve_model_dir(modelname) / f"model_data{model_id}.json")
            with open(model_data_path, encoding="utf-8") as model_data_file:
                model_data = json.load(model_data_file)
            authored = float(model_data["scale"][0])
        except (KeyError, TypeError, FileNotFoundError, ValueError):
            authored = 1.0
        return authored * float(scale_mult)

    def _require_actor(self, actor: Any, modelname: str):
        """Raise if ``create_actor`` failed to load a mesh (None has no ``set_name``)."""
        if actor is None:
            raise UnStableError(
                f"cook_meat: missing asset {modelname} "
                "(install with bash script/_download_assets.sh)"
            )
        return actor

    @classmethod
    def _mesh_vertices(cls, collision_path: str) -> np.ndarray:
        """Load and cache vertices from a collision mesh."""
        if collision_path not in cls._MESH_CACHE:
            mesh = trimesh.load(collision_path, force="mesh")
            cls._MESH_CACHE[collision_path] = np.asarray(mesh.vertices, dtype=float)
        return cls._MESH_CACHE[collision_path]

    @classmethod
    def _mesh_aabb_xy(
        cls,
        collision_path: str,
        pose: sapien.Pose,
        scale: float,
        padding: float = 0.0,
    ) -> AABB:
        """Return the padded world-space XY bounds of a collision mesh."""
        vertices = cls._mesh_vertices(collision_path) * scale
        transform = pose.to_transformation_matrix()
        world = (transform[:3, :3] @ vertices.T).T + np.asarray(pose.p)
        return (
            float(world[:, 0].min()) - padding, float(world[:, 1].min()) - padding,
            float(world[:, 0].max()) + padding, float(world[:, 1].max()) + padding,
        )

    @classmethod
    def _mesh_world_z_extent(
        cls,
        collision_path: str,
        pose: sapien.Pose,
        scale: float,
    ) -> tuple[float, float]:
        """Return the transformed mesh's minimum and maximum world Z."""
        vertices = cls._mesh_vertices(collision_path) * scale
        transform = pose.to_transformation_matrix()
        world = (transform[:3, :3] @ vertices.T).T + np.asarray(pose.p)
        return float(world[:, 2].min()), float(world[:, 2].max())

    @classmethod
    def _bowl_center_xy(
        cls,
        collision_path: str,
        pose: sapien.Pose,
        scale: float,
        floor_band: float = 0.010,
    ) -> tuple[float, float]:
        """Return the world XY center of a pan's bowl floor.

        ``get_functional_point(0)`` sits up to ~3.5 cm off the bowl center (the
        offset differs per ``base*`` mesh), so a steak dropped there can end up
        against the bowl wall, where the fingers hit the rim on the way down and
        the steak can no longer be picked up. The floor band of the collision
        mesh gives the real center: it is per-mesh, follows the sampled yaw, and
        mirrors automatically with the pan pose.
        """
        vertices = cls._mesh_vertices(collision_path) * scale
        transform = pose.to_transformation_matrix()
        world = (transform[:3, :3] @ vertices.T).T + np.asarray(pose.p)
        floor = world[world[:, 2] < world[:, 2].min() + float(floor_band)]
        if len(floor) == 0:
            return float(pose.p[0]), float(pose.p[1])
        return float(floor[:, 0].mean()), float(floor[:, 1].mean())

    @classmethod
    def _footprint_offsets(
        cls,
        collision_path: str,
        qpos: Sequence[float],
        scale: float,
    ) -> AABB:
        """Return rotation-and-scale-adjusted XY bounds before translation."""
        # Fixed world-XY footprint box (vmin_x, vmin_y, vmax_x, vmax_y) of the mesh at this
        # rotation+scale BEFORE translation. Translating a rigid footprint merely shifts its AABB, so
        # AABB(x, y) = (vmin_x + x, vmin_y + y, vmax_x + x, vmax_y + y). Computing this once lets the
        # grid fallback below offset it per cell instead of re-transforming the mesh every candidate.
        vertices = cls._mesh_vertices(collision_path) * scale
        rotation = sapien.Pose([0.0, 0.0, 0.0], qpos).to_transformation_matrix()[:3, :3]
        rotated = (rotation @ vertices.T).T
        return (
            float(rotated[:, 0].min()),
            float(rotated[:, 1].min()),
            float(rotated[:, 0].max()),
            float(rotated[:, 1].max()),
        )

    @staticmethod
    def _aabb_gap(a: AABB, b: AABB) -> float:
        """Return the maximum signed gap between two XY AABBs.

        A positive result means that at least one axis separates the boxes;
        zero means that their boundaries touch; and a negative result means
        that they overlap on both axes.
        """
        gx = max(b[0] - a[2], a[0] - b[2])
        gy = max(b[1] - a[3], a[1] - b[3])
        return max(gx, gy)

    # ------------------------------------------------------ head-camera view
    # The front video the episodes are rendered from is "head_camera" (ur5-wsg static_camera_list,
    # assets/embodiments/ur5-wsg/config.yml): a D435 (task_config/_camera_config.yml -> fovy 37
    # deg, 320x240) at [0.0,-0.50,2.0] looking forward [0,0.45,-1.0] with left [-1,0,0]. So the
    # visible patch of the table is a trapezoid; an object spawned too far to the side / too near
    # the front edge would render partly out of frame. These constants + the projection below let
    # us REJECT any spawn whose footprint leaves the image.
    # (random_head_camera_dis is 0 in both demo_dynamic and debug_dynamic, so the pose is exact.)
    _CAM_POS: ClassVar[np.ndarray] = np.array([0.0, -0.50, 2.0])
    _CAM_FWD: ClassVar[np.ndarray] = np.array([0.0, 0.45, -1.0])
    _CAM_LEFT: ClassVar[np.ndarray] = np.array([-1.0, 0.0, 0.0])
    _CAM_FOVY_DEG: ClassVar[float] = 37.0
    _CAM_W: ClassVar[int] = 320
    _CAM_H: ClassVar[int] = 240

    @classmethod
    def _project_to_head_cam(
        cls,
        points_world: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world-space points into the fixed head camera."""
        # World (N,3) -> (u, v, depth). Mirrors camera.py:101-120: camera-local axes are the columns
        # [forward, left, up] with up = forward x left, so p_cam = R^T (p - cam_pos) gives
        # x=depth-along-view, y=left, z=up; pinhole with square pixels from fovy. Validated to <1px
        # against the known visible band (near edge y~-0.247 -> v~H, far edge y~0.433 -> v~0).
        f = cls._CAM_FWD / np.linalg.norm(cls._CAM_FWD)
        l = cls._CAM_LEFT / np.linalg.norm(cls._CAM_LEFT)
        up = np.cross(f, l)
        displacement = np.asarray(points_world, dtype=float) - cls._CAM_POS
        depth = displacement @ f
        y_cam = displacement @ l
        z_cam = displacement @ up
        focal_length = (cls._CAM_H / 2.0) / np.tan(
            np.deg2rad(cls._CAM_FOVY_DEG) / 2.0
        )
        center_x, center_y = cls._CAM_W / 2.0, cls._CAM_H / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = center_x - focal_length * (y_cam / depth)
            v = center_y - focal_length * (z_cam / depth)
        return u, v, depth

    @classmethod
    def _footprint_in_head_view(
        cls,
        aabb_xy: AABB,
        z: float,
        margin_px: int = 4,
    ) -> bool:
        """Return whether all footprint corners lie inside the head-camera view."""
        # True iff all four footprint corners (at table-top height z) project strictly inside the
        # image with a small pixel margin -- i.e. the object renders fully within the front video.
        xmin, ymin, xmax, ymax = aabb_xy
        corners = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
        u, v, depth = cls._project_to_head_cam(corners)
        if np.any(depth <= 0):
            return False
        return bool(
            np.all(u >= margin_px)
            and np.all(u <= cls._CAM_W - margin_px)
            and np.all(v >= margin_px)
            and np.all(v <= cls._CAM_H - margin_px)
        )

    def _sample_clear_pose(
        self,
        xlim: Sequence[float],
        ylim: Sequence[float],
        qpos: Sequence[float],
        collision_path: str,
        scale: float,
        avoid_aabbs: Sequence[AABB],
        padding: float,
        view_z: float,
        tries: int = 400,
        grid_step: float = 0.005,
    ) -> tuple[sapien.Pose | None, AABB | None]:
        """Sample a pose whose XY footprint is clear and inside the camera frame.

        Visibility is approximated by projecting the four AABB corners at
        ``view_z``. A failed search returns ``(None, None)`` so the caller can
        reject the episode seed without expanding the configured spawn band.
        """
        def is_valid(aabb: AABB) -> bool:
            clearance = min(
                (self._aabb_gap(aabb, other) for other in avoid_aabbs),
                default=float("inf"),
            )
            return clearance > 0 and self._footprint_in_head_view(aabb, view_z)

        # Phase 1: random rejection sampling -- gives per-seed spatial diversity when a spot exists.
        for _ in range(tries):
            candidate = rand_pose(xlim=xlim, ylim=ylim, qpos=qpos)
            candidate_aabb = self._mesh_aabb_xy(
                collision_path, candidate, scale, padding=padding
            )
            if is_valid(candidate_aabb):
                return candidate, candidate_aabb

        # Phase 2: scan a 5-mm grid to reduce misses from random rejection sampling. The footprint
        # shape is translation-invariant, so offset one precomputed box per grid cell. Select a
        # valid cell at random to preserve layout diversity.
        offsets = self._footprint_offsets(collision_path, qpos, scale)

        def aabb_at(x: float, y: float) -> AABB:
            return (
                offsets[0] + x - padding,
                offsets[1] + y - padding,
                offsets[2] + x + padding,
                offsets[3] + y + padding,
            )
        xlo, xhi = min(xlim), max(xlim)
        ylo, yhi = min(ylim), max(ylim)
        valid_cells: list[tuple[float, float, AABB]] = []
        nx = max(2, int((xhi - xlo) / grid_step) + 1)
        ny = max(2, int((yhi - ylo) / grid_step) + 1)
        for gx in np.linspace(xlo, xhi, nx):
            for gy in np.linspace(ylo, yhi, ny):
                candidate_aabb = aabb_at(float(gx), float(gy))
                if is_valid(candidate_aabb):
                    valid_cells.append((float(gx), float(gy), candidate_aabb))
        if valid_cells:
            grid_x, grid_y, candidate_aabb = valid_cells[
                np.random.randint(len(valid_cells))
            ]
            return sapien.Pose([grid_x, grid_y, 0.741], qpos), candidate_aabb
        return None, None

    @classmethod
    def _sample_cook_steps(cls, config: dict[str, Any]) -> int:
        """Sample per-episode cook duration around a nominal rate ± jitter.

        Default setup: ``cook_steps`` (nominal steps for doneness 0→1) times
        ``U(1 ± cook_speed_jitter)`` with jitter defaulting to 20%.

        Legacy ``cook_steps_min`` / ``cook_steps_max`` (without ``cook_steps``)
        still sample uniformly in that inclusive range.
        """
        jitter = float(config.get("cook_speed_jitter", cls.COOK_SPEED_JITTER_DEFAULT))
        jitter = float(np.clip(abs(jitter), 0.0, 0.95))

        if "cook_steps" in config:
            nom = float(config["cook_steps"])
        elif "cook_steps_min" in config or "cook_steps_max" in config:
            lo = float(config.get("cook_steps_min", config.get("cook_steps_max", cls.COOK_STEPS_DEFAULT)))
            hi = float(config.get("cook_steps_max", lo))
            if hi < lo:
                lo, hi = hi, lo
            return max(1, int(round(float(np.random.uniform(lo, hi)))))
        else:
            nom = float(cls.COOK_STEPS_DEFAULT)

        nom = max(1.0, nom)
        lo = nom * (1.0 - jitter)
        hi = nom * (1.0 + jitter)
        return max(1, int(round(float(np.random.uniform(lo, hi)))))


    @staticmethod
    def _compose_yaw_qpos(base_qpos: Sequence[float], yaw: float) -> list[float]:
        """Lay flat with ``base_qpos``, then rotate about world +Z (table surface).

        World-Z yaw must pre-multiply the base orientation. Post-multiplying applies
        yaw in the asset's local frame and tips pans/boards off the table.
        """
        yaw_q = t3d.euler.euler2quat(0.0, 0.0, float(yaw))
        return list(t3d.quaternions.qmult(yaw_q, list(base_qpos)))

    def _aabb_clear(
        self,
        aabb: AABB,
        avoid_aabbs: Sequence[AABB],
        view_z: float,
        min_gap: float = 0.0,
    ) -> bool:
        """True when ``aabb`` clears all avoid boxes and stays in head view."""
        gap = min(
            (self._aabb_gap(aabb, other) for other in avoid_aabbs),
            default=float("inf"),
        )
        return gap > float(min_gap) and self._footprint_in_head_view(aabb, view_z)

    def _sample_mirrored_prop_pose(
        self,
        *,
        side: float,
        nom_x: float,
        nom_y: float,
        base_qpos: Sequence[float],
        collision_path: str,
        scale: float,
        avoid_aabbs: Sequence[AABB],
        padding: float,
        view_z: float,
        jitter_xy: float,
        yaw_lim: float,
        tries: int = 80,
    ) -> tuple[sapien.Pose | None, AABB | None, list[float] | None]:
        """Place a prop from a right-side nominal, mirrored when ``side < 0``.

        Applies small XY jitter and a random Z yaw, rejecting overlaps.
        """
        for _ in range(tries):
            jx = float(np.random.uniform(-jitter_xy, jitter_xy))
            jy = float(np.random.uniform(-jitter_xy, jitter_xy))
            yaw = float(np.random.uniform(-yaw_lim, yaw_lim))
            qpos = self._compose_yaw_qpos(base_qpos, yaw)
            x = float(side) * (float(nom_x) + jx)
            y = float(nom_y) + jy
            pose = sapien.Pose([x, y, 0.741], qpos)
            aabb = self._mesh_aabb_xy(collision_path, pose, scale, padding=padding)
            if self._aabb_clear(aabb, avoid_aabbs, view_z, min_gap=0.0):
                return pose, aabb, qpos
        return None, None, None

    # ---------------------------------------------------------------- actors
    def _spawn_station(
        self,
        side: float,
        tag: str,
        avoid_aabbs: list[AABB],
        bz: float,
    ) -> tuple[dict[str, Any], list[AABB]]:
        """Spawn one cook station (pan, cook key, board, steak) on ``side``.

        Layout is authored for the right side and mirrored for the left (Opt 2).
        Pan and board each get small XY jitter and a random Z yaw. Cook keys sit
        on the outer lateral side, clear of the pan bowl and board.
        """
        prop_pad = float(self.PROP_CLEARANCE)
        jitter_xy = float(getattr(self, "place_jitter_xy", self.PLACE_JITTER_XY_DEFAULT))
        yaw_lim = float(getattr(self, "place_yaw_lim", self.PLACE_YAW_LIM_DEFAULT))
        skillet_id = int(np.random.choice([0, 1, 2, 3]))
        skillet_path = f"assets/objects/106_skillet/collision/base{skillet_id}.glb"
        skillet_scale = self._applied_scale("106_skillet", skillet_id, self.pan_scale)

        # Dual: fixed |x| so bowls keep ≥10 cm; single uses the tighter right/left nominal.
        pan_nom_x = self.DUAL_PAN_NOM_X if self.dual_setup_enabled else self.PAN_NOM_X
        skillet_pose = skillet_aabb = skillet_qpos = None
        # Sample pan with bowl-vs-avoid clearance (full skillet mesh AABB includes the
        # handle and was blocking the mirrored dual station).
        for _ in range(120):
            jx = float(np.random.uniform(-jitter_xy, jitter_xy))
            jy = float(np.random.uniform(-jitter_xy, jitter_xy))
            yaw = float(np.random.uniform(-yaw_lim, yaw_lim))
            qpos = self._compose_yaw_qpos(self.SKILLET_BASE_QPOS, yaw)
            x = float(side) * (float(pan_nom_x) + jx)
            y = float(self.PAN_NOM_Y) + jy
            pose = sapien.Pose([x, y, 0.741], qpos)
            mesh_aabb = self._mesh_aabb_xy(
                skillet_path, pose, skillet_scale, padding=prop_pad
            )
            br = float(self.PAN_BOWL_HALF)
            bowl = (x - br, y - br, x + br, y + br)
            # Dual sibling avoid is already expand(bowl, DUAL_CLEARANCE) — compare the
            # raw bowl (do not also apply PROP_CLEARANCE or the 2nd pan never fits).
            if self.dual_setup_enabled:
                gap = min(
                    (self._aabb_gap(bowl, other) for other in avoid_aabbs),
                    default=float("inf"),
                )
            else:
                bowl_pad = self._expand_aabb(bowl, prop_pad)
                gap = min(
                    (self._aabb_gap(bowl_pad, other) for other in avoid_aabbs),
                    default=float("inf"),
                )
            # View-gate the real bowl (or full mesh); padded bowl often clips the
            # head-camera frame at dual-station |x| even when the pan is visible.
            in_view = self._footprint_in_head_view(
                mesh_aabb, bz
            ) or self._footprint_in_head_view(bowl, bz)
            if gap > 0.0 and in_view:
                skillet_pose, skillet_aabb, skillet_qpos = pose, mesh_aabb, qpos
                break
        if skillet_pose is None or skillet_aabb is None or skillet_qpos is None:
            raise UnStableError(
                f"cook_meat: pan not placeable ({tag}, seed {self._ep_seed}) -- skip"
            )
        skillet = self._require_actor(
            create_actor(
                self,
                pose=skillet_pose,
                modelname="106_skillet",
                model_id=skillet_id,
                convex=True,
                is_static=True,
                scale_mult=self.pan_scale,
            ),
            "106_skillet",
        )
        skillet_name = f"106_skillet_{tag}"
        skillet.set_name(skillet_name)
        # Pan is static, so its bowl center never moves after spawn.
        bowl_center_xy = self._bowl_center_xy(skillet_path, skillet_pose, skillet_scale)
        # Mesh AABB includes the handle; board/key only need the bowl footprint clear.
        px, py = float(skillet_pose.p[0]), float(skillet_pose.p[1])
        br = float(self.PAN_BOWL_HALF)
        bowl_aabb: AABB = (px - br, py - br, px + br, py + br)
        avoid = list(avoid_aabbs) + [self._expand_aabb(bowl_aabb, prop_pad)]

        # Cutting board: nominal offset from pan (outer ±x, +y), then jitter + yaw.
        board_scale_mult = self.board_scale_mult
        board_applied_scale = self._applied_scale("104_board", 0, board_scale_mult)
        board_path = "assets/objects/104_board/collision/base0.glb"
        board_nom_x = abs(px) + self.BOARD_OFF_X
        board_nom_y = py + self.BOARD_OFF_Y
        board_pose, board_aabb, board_qpos = self._sample_mirrored_prop_pose(
            side=side,
            nom_x=board_nom_x,
            nom_y=board_nom_y,
            base_qpos=self.BOARD_BASE_QPOS,
            collision_path=board_path,
            scale=board_applied_scale,
            avoid_aabbs=avoid,
            padding=prop_pad,
            view_z=bz,
            jitter_xy=jitter_xy,
            yaw_lim=yaw_lim,
        )
        if board_pose is None or board_aabb is None or board_qpos is None:
            # Fallback: absolute right-side board nominal (still mirrored by side).
            board_pose, board_aabb, board_qpos = self._sample_mirrored_prop_pose(
                side=side,
                nom_x=pan_nom_x + self.BOARD_OFF_X,
                nom_y=self.PAN_NOM_Y + self.BOARD_OFF_Y,
                base_qpos=self.BOARD_BASE_QPOS,
                collision_path=board_path,
                scale=board_applied_scale,
                avoid_aabbs=avoid,
                padding=prop_pad,
                view_z=bz,
                jitter_xy=jitter_xy,
                yaw_lim=yaw_lim,
                tries=120,
            )
        if board_pose is None or board_aabb is None or board_qpos is None:
            raise UnStableError(
                f"cook_meat: board not placeable ({tag}, seed {self._ep_seed}) -- skip"
            )
        board_xy = board_pose.p[:2]
        probe_pose = sapien.Pose([0.0, 0.0, 0.0], board_qpos)
        board_z_min, board_z_max = self._mesh_world_z_extent(
            board_path, probe_pose, board_applied_scale
        )
        board_th = board_z_max - board_z_min
        board_spawn_z = bz - board_z_min
        board_pose = sapien.Pose(
            [float(board_xy[0]), float(board_xy[1]), board_spawn_z], board_qpos
        )
        board_aabb = self._mesh_aabb_xy(
            board_path, board_pose, board_applied_scale, padding=prop_pad
        )
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        board = self._require_actor(
            create_actor(
                self,
                pose=board_pose,
                modelname="104_board",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=board_scale_mult,
            ),
            "104_board",
        )
        board.set_name(f"104_board_{tag}")
        board.config = {
            "scale": [board_scale_mult, board_scale_mult, board_scale_mult],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        board_top = bz + board_th
        avoid = list(avoid) + [self._expand_aabb(board_aabb, prop_pad)]

        steak_pose = rand_pose(
            xlim=[float(board_xy[0]), float(board_xy[0])],
            ylim=[float(board_xy[1]), float(board_xy[1])],
            zlim=[board_top],
            qpos=self._compose_yaw_qpos(
                self.BOARD_BASE_QPOS, float(np.random.uniform(-yaw_lim, yaw_lim))
            ),
            rotate_rand=False,
        )
        steak = self._require_actor(
            create_actor(
                self,
                pose=steak_pose,
                modelname="200_steak",
                model_id=0,
                convex=True,
                is_static=False,
                scale_mult=(1.0, self.steak_thick, 1.0),
            ),
            "200_steak",
        )
        steak_name = f"200_steak_{tag}"
        steak.set_name(steak_name)
        steak.set_mass(0.05)

        # Cook key on the outer lateral side of the pan, biased toward the robot (−y)
        # so it stays clear of the board (+y) and does not overlap the bowl.
        cook_key = None
        cook_key_base = None
        key_home_pose = None
        key_xy = None
        key_top_z = None
        key_aabb: AABB | None = None
        pan_xy = np.asarray(skillet_pose.p[:2], dtype=np.float64)
        hx, hy, _ = self.KEY_BASE_HALF
        clear_r = float(self.KEY_PAN_CLEAR_HALF)
        key_gap = float(self.KEY_MIN_GAP)
        lat_sign = -1.0 if side < 0 else 1.0
        lat0 = max(clear_r + hx + key_gap, abs(self.key_offset_x))
        # Clear the bowl rim; board already sits in ``avoid``.
        key_avoid = list(avoid) + [
            (
                float(pan_xy[0] - clear_r),
                float(pan_xy[1] - clear_r),
                float(pan_xy[0] + clear_r),
                float(pan_xy[1] + clear_r),
            )
        ]
        found = False
        key_x = key_y = 0.0
        # Search outward (±x) and toward the robot (−y). Do not require head-cam
        # containment for the key — left-side outer slots often leave the frame
        # and were rejecting every dual+button seed.
        dx_vals = list(np.linspace(lat0, lat0 + 0.18, 12))
        dy_vals = list(np.linspace(float(self.key_y_bias), float(self.key_y_bias) - 0.16, 10))
        # Also try near pan y if the board blocks the −y corridor.
        dy_vals += list(np.linspace(0.02, -0.06, 5))
        for require_view in (True, False):
            for dx in dx_vals:
                for dy in dy_vals:
                    kx = float(pan_xy[0] + lat_sign * dx)
                    ky = float(pan_xy[1] + dy)
                    # Keep key on the table footprint.
                    if abs(kx) > 0.42 or ky < -0.28 or ky > 0.22:
                        continue
                    cand: AABB = (kx - hx, ky - hy, kx + hx, ky + hy)
                    gap = min(
                        (self._aabb_gap(cand, other) for other in key_avoid),
                        default=float("inf"),
                    )
                    if gap <= key_gap:
                        continue
                    if require_view and not self._footprint_in_head_view(cand, bz):
                        continue
                    key_x, key_y = kx, ky
                    key_aabb = cand
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if not found or key_aabb is None:
            raise UnStableError(
                f"cook_meat: cook key not placeable ({tag}, seed {self._ep_seed}) -- skip"
            )
        base_hz = float(self.KEY_BASE_HALF[2])
        cap_hz = float(self.KEY_HALF[2])
        base_z = bz + base_hz
        cap_z = bz + 2.0 * base_hz + cap_hz
        cook_key_base = create_box(
            self,
            pose=sapien.Pose([key_x, key_y, base_z], [1, 0, 0, 0]),
            half_size=list(self.KEY_BASE_HALF),
            color=list(self.KEY_BASE_COLOR),
            name=f"cook_key_base_{tag}",
            is_static=True,
        )
        key_home = sapien.Pose([key_x, key_y, cap_z], [1, 0, 0, 0])
        cook_key = create_box(
            self,
            pose=key_home,
            half_size=list(self.KEY_HALF),
            color=list(self.KEY_COLOR_UP),
            name=f"cook_key_{tag}",
            is_static=True,
        )
        key_xy = (key_x, key_y)
        key_top_z = float(bz + 2.0 * base_hz + 2.0 * cap_hz)
        key_home_pose = key_home
        symbol_kind = "push" if self.use_hold_cook else "on_off"
        symbol_parts, symbol_locals = attach_key_symbol(
            self, cook_key, self.KEY_HALF, symbol_kind, f"cook_key_symbol_{tag}"
        )
        avoid.append(self._expand_aabb(key_aabb, key_gap))
        self.add_prohibit_area(cook_key_base, padding=0.02)
        self.add_prohibit_area(cook_key, padding=0.02)

        self.add_prohibit_area(skillet, padding=0.05)
        self.add_prohibit_area(board, padding=0.03)
        self.add_prohibit_area(steak, padding=0.03)

        steak_shapes: list[Any] = []
        for c in steak.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                steak_shapes = list(c.render_shapes)
        key_shapes: list[Any] = []
        key_entity = cook_key.actor if hasattr(cook_key, "actor") else cook_key
        for c in key_entity.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                key_shapes = list(c.render_shapes)

        station = {
            "tag": tag,
            "side": float(side),
            "arm": ArmTag("right" if side > 0 else "left"),
            "skillet": skillet,
            "skillet_id": skillet_id,
            "skillet_name": skillet_name,
            "bowl_center_xy": bowl_center_xy,
            "board": board,
            "board_xy": (float(board_xy[0]), float(board_xy[1])),
            "board_top": float(board_top),
            "steak": steak,
            "steak_name": steak_name,
            "cook_key": cook_key,
            "cook_key_base": cook_key_base,
            "key_xy": key_xy,
            "key_top_z": key_top_z,
            "key_home_pose": key_home_pose if cook_key is not None else None,
            "key_symbol_parts": symbol_parts,
            "key_symbol_locals": symbol_locals,
            "key_shapes": key_shapes,
            "_key_color_down": None,  # force first color sync to green
            "steak_shapes": steak_shapes,
            "doneness": 0.0,
            "max_doneness": 0.0,
            "grasp_doneness": None,  # doneness when cooking stopped
            "cooking_active": False,
            "awaiting_return_grasp": False,
            "cook_phase_done": False,
            "cook_on": False,  # latch mode: key latched ON
            "_hold_cooked": False,  # Opt 1: True after at least one hold-cook tick
            "_hold_release_steps": 0,  # Opt 1: consecutive unengaged steps
            "_pending_off": False,
            "_touch_latched": False,
            "_ignore_key": False,  # expert sets latch explicitly
            "_expert_key_held": False,  # expert depress / hold visual
        }
        self._set_station_meat_color(station, 0.0)
        # Sibling dual station only needs bowl↔bowl ≥10 cm. Board/key sit outward
        # (+y / outer ±x); folding them into the mid-line keep-out made the 2nd pan
        # unplaceable once XY jitter shrank.
        keep_parts = [bowl_aabb, board_aabb]
        if key_aabb is not None:
            keep_parts.append(key_aabb)
        station_keepout = self._union_aabb(keep_parts)
        out_avoid = list(avoid_aabbs)
        if self.dual_setup_enabled:
            out_avoid.append(
                self._expand_aabb(bowl_aabb, self.DUAL_STATION_CLEARANCE)
            )
        else:
            out_avoid.append(station_keepout)
        return station, out_avoid

    def load_actors(self) -> None:
        """Create the randomized task layout and reset cooking state."""
        config = self._cook_cfg
        self.cook_steps = self._sample_cook_steps(config)
        configured_range = config.get("target_doneness_range")
        if configured_range is not None:
            if not isinstance(configured_range, (list, tuple)) or len(configured_range) != 2:
                raise ValueError("cook_meat.target_doneness_range must be [minimum, maximum]")
            range_min, range_max = map(float, configured_range)
        else:
            default_min, default_max = self.TARGET_DONENESS_RANGE_DEFAULT
            range_min = float(config.get("target_doneness_min", default_min))
            range_max = float(config.get("target_doneness_max", default_max))
        if not 0.0 <= range_min <= range_max <= 1.0:
            raise ValueError("cook_meat target doneness range must satisfy 0 <= min <= max <= 1")
        range_jitter = float(config.get(
            "target_doneness_range_jitter",
            self.TARGET_DONENESS_RANGE_JITTER_DEFAULT,
        ))
        if range_jitter < 0.0:
            raise ValueError("cook_meat.target_doneness_range_jitter must be non-negative")
        # Shift the complete interval together, preserving its width and keeping
        # both endpoints inside [0, 1].
        shift_min = max(-range_jitter, -range_min)
        shift_max = min(range_jitter, 1.0 - range_max)
        range_shift = float(np.random.uniform(shift_min, shift_max))
        self.target_doneness_base_range = (range_min, range_max)
        self.target_doneness_range_jitter = range_jitter
        self.target_doneness_range_shift = range_shift
        self.target_doneness_range = (
            range_min + range_shift,
            range_max + range_shift,
        )
        # The expert stops at the center; success accepts the full configured interval.
        self.target_doneness = sum(self.target_doneness_range) / 2.0
        self.cook_doneness_tol = float(
            config.get("cook_doneness_tol", self.COOK_DONENESS_TOL_DEFAULT)
        )
        # Opt 1: hold-to-cook. Default / Opt 2 alone: latch ON/OFF. Keys always spawn.
        self.cook_button_enabled = bool(
            config.get("cook_button_enabled", self.COOK_BUTTON_ENABLED_DEFAULT)
        )
        self.dual_setup_enabled = bool(
            config.get("dual_setup_enabled", self.DUAL_SETUP_ENABLED_DEFAULT)
        )
        self.max_episode_steps = max(
            1,
            int(config.get("max_episode_steps", self.MAX_EPISODE_STEPS_DEFAULT)),
        )
        # Provisional; setup_demo re-applies after base eval step_lim load.
        self._max_episode_steps = int(self.max_episode_steps)
        self._episode_timed_out = False
        self.key_offset_x = float(config.get("key_offset_x", self.KEY_OFFSET_X_DEFAULT))
        self.key_y_bias = float(config.get("key_y_bias", self.KEY_Y_BIAS_DEFAULT))
        self.key_press_depth = float(
            config.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT)
        )
        self.key_press_xy = float(config.get("key_press_xy", self.KEY_PRESS_XY_DEFAULT))
        self.key_press_dz = float(config.get("key_press_dz", self.KEY_PRESS_DZ_DEFAULT))
        self.key_hover_dis = float(
            config.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT)
        )
        self.pan_scale = float(config.get("pan_scale", 1.0))
        self.place_dx = float(config.get("place_dx", 0.0))
        self.place_dy = float(config.get("place_dy", 0.0))
        self.board_scale_mult = float(config.get("board_scale_mult", 0.07))
        self.steak_thick = float(config.get("steak_thick", 1.6))
        self.place_jitter_xy = float(
            config.get("place_jitter_xy", self.PLACE_JITTER_XY_DEFAULT)
        )
        self.place_yaw_lim = float(
            config.get("place_yaw_lim", self.PLACE_YAW_LIM_DEFAULT)
        )
        self.board_th = 0.0

        bz = 0.74 + self.table_z_bias
        avoid: list[AABB] = []
        self.stations: list[dict[str, Any]] = []

        if self.dual_setup_enabled:
            # Both sides: left then right so arms each own a full mirrored station.
            for side, tag in ((-1.0, "left"), (1.0, "right")):
                st, avoid = self._spawn_station(side, tag, avoid, bz)
                self.stations.append(st)
            self._side = 0.0
        else:
            side = -1.0 if (self._ep_seed % 2 == 0) else 1.0
            self._side = side
            tag = "left" if side < 0 else "right"
            st, avoid = self._spawn_station(side, tag, avoid, bz)
            self.stations.append(st)

        # Primary aliases for single-station helpers / dynamic path / older tests.
        primary = self.stations[0]
        self.skillet = primary["skillet"]
        self.skillet_id = primary["skillet_id"]
        self.board = primary["board"]
        self.steak = primary["steak"]
        self.cook_key = primary["cook_key"]
        self.cook_key_base = primary["cook_key_base"]
        self._key_xy = primary["key_xy"]
        self._key_top_z = primary["key_top_z"]
        self._steak_shapes = primary["steak_shapes"]
        self.doneness = primary["doneness"]
        self.max_doneness = primary["max_doneness"]
        self._grasp_doneness = primary["grasp_doneness"]
        self._reset_metric_state()
        self._cooking_active = False
        # Legacy alias used by older unit tests / success helpers.
        self.plate = primary["board"]
        self._init_reactive_cook_keys()

    def _init_reactive_cook_keys(self) -> None:
        """Spring cook keys with measure_ingredient-style ON/OFF latch visuals."""
        self._reactive_buttons = None
        actors, homes, tops, ids = [], [], [], []
        for st in self.stations:
            key = st.get("cook_key")
            home = st.get("key_home_pose")
            if key is None or home is None:
                continue
            actors.append(key)
            homes.append(home)
            tops.append(float(st["key_top_z"]))
            ids.append(str(st["tag"]))
        if not actors:
            return
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=actors,
            home_poses=homes,
            max_depth=float(self.KEY_HALF[2]),
            ids=ids,
            xy_tol=float(self.key_press_xy),
        )
        self._reactive_buttons.set_tops_z(tops)

    # -------------------------------------------------------- cooking state
    def _set_station_meat_color(self, station: dict[str, Any], doneness: float) -> None:
        """Update one station's steak render shapes for the supplied doneness."""
        d = float(np.clip(doneness, 0.0, 1.0))
        stops = self.COLOR_STOPS
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        color = list(rgb) + [1.0]
        for render_shape in station.get("steak_shapes", []):
            try:
                render_shape.material.set_base_color(color)
            except Exception:
                pass

    def _set_meat_color(self, doneness: float) -> None:
        """Update the primary steak color (single-station / test helper)."""
        if getattr(self, "stations", None):
            self._set_station_meat_color(self.stations[0], doneness)
            self.stations[0]["doneness"] = float(doneness)
            return
        # Fallback for unit tests that never call load_actors.
        st = {"steak_shapes": getattr(self, "_steak_shapes", [])}
        self._set_station_meat_color(st, doneness)

    def _steak_on_pan_station(self, station: dict[str, Any]) -> bool:
        """Return whether this station's steak contacts its skillet."""
        try:
            return bool(
                self.check_actors_contact(station["steak_name"], station["skillet_name"])
            )
        except Exception:
            return False

    def _steak_on_pan(self) -> bool:
        """Primary-station contact helper."""
        if getattr(self, "stations", None):
            return self._steak_on_pan_station(self.stations[0])
        try:
            return bool(self.check_actors_contact("200_steak", "106_skillet"))
        except Exception:
            return False

    def _steak_held(self, station: dict[str, Any]) -> bool:
        """True when a gripper finger is in contact with this station's steak."""
        try:
            contacts = self.get_gripper_actor_contact_position(station["steak_name"])
            return len(contacts) > 0
        except Exception:
            return False

    def _latch_grasp_doneness(self, station: dict[str, Any], *, force: bool = False) -> None:
        """Freeze this station's cook quality (legacy helper / interactive board snap).

        Preferred path is ``_set_station_cook_on(st, False)``, which latches
        doneness when the key toggles OFF. This remains for interactive board
        snaps and older call sites.
        """
        if station.get("grasp_doneness") is not None and not force:
            return
        if not force and not self._steak_held(station):
            return
        station["grasp_doneness"] = float(station["doneness"])
        station["cooking_active"] = False
        station["awaiting_return_grasp"] = False
        station["cook_phase_done"] = True
        station["cook_on"] = False
        station["_pending_off"] = False

    def _set_station_cook_on(self, station: dict[str, Any], on: bool) -> None:
        """Latch cook key ON (cooking) or OFF (freeze doneness at shutoff)."""
        on = bool(on)
        was = bool(station.get("cook_on", False))
        station["cook_on"] = on
        station["_pending_off"] = False
        if on:
            station["grasp_doneness"] = None
            station["cook_phase_done"] = False
            station["cooking_active"] = False
            return
        if was or station.get("grasp_doneness") is None:
            # Shutoff freezes the cook score used by check_success.
            station["grasp_doneness"] = float(station["doneness"])
        station["cook_phase_done"] = True
        station["cooking_active"] = False

    def _set_station_key_color(self, station: dict[str, Any], down: bool) -> None:
        """Green when up, red when depressed."""
        down = bool(down)
        prev = station.get("_key_color_down")
        if prev is not None and bool(prev) == down:
            return
        station["_key_color_down"] = down
        rgb = self.KEY_COLOR_DOWN if down else self.KEY_COLOR_UP
        color = list(rgb) + [1.0]
        for shape in station.get("key_shapes", []) or []:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _key_visually_down(self, station: dict[str, Any]) -> bool:
        """True when the keycap is depressed (latched, held, or springing down)."""
        if bool(station.get("_expert_key_held")):
            return True
        if not self.use_hold_cook and bool(station.get("cook_on")):
            return True
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return False
        try:
            idx = bank.resolve_index(str(station["tag"]))
            return float(bank.visual_depth[idx]) > 1e-4
        except Exception:
            return False

    def _button_is_pressed_station(self, station: dict[str, Any]) -> bool:
        """True when this station's key is actively cooking (latched ON or held).

        Hold mode (Opt 1) uses press-session hysteresis (``is_engaged``): once
        the keycap crosses the trigger it stays engaged until it fully springs
        back, so brief tip wobble above the trigger does not count as a release.
        """
        if station.get("cook_key") is None:
            return False
        if bool(station.get("_expert_key_held")):
            return True
        if self.use_hold_cook:
            bank = getattr(self, "_reactive_buttons", None)
            if bank is None:
                return False
            tag = str(station["tag"])
            try:
                if hasattr(bank, "is_engaged"):
                    if bool(bank.is_engaged(tag)):
                        return True
                elif bool(bank.is_held(tag)):
                    return True
            except Exception:
                pass
            # Tip still driving the spring (pre-trigger or on the edge).
            try:
                return bool(self._key_tip_pressing(station))
            except Exception:
                return False
        return bool(station.get("cook_on"))

    def _button_is_pressed(self) -> bool:
        """True if any station's cook key is cooking (obs / tests)."""
        for st in getattr(self, "stations", []) or []:
            if self._button_is_pressed_station(st):
                return True
        return False

    def _key_tip_pressing(self, station: dict[str, Any]) -> bool:
        """True when a gripper tip is pressing this station's key (force proxy)."""
        if bool(getattr(self, "_interactive_arms_removed", False)):
            return False
        if not bool(getattr(self, "_interactive_robot_mode", True)):
            return False
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return False
        try:
            idx = bank.resolve_index(str(station["tag"]))
        except Exception:
            return False
        tip = None
        for side in bank._sides_for_button(idx):
            candidate = bank._tip_xyz(side)
            if candidate is None:
                continue
            home_xy = np.asarray(bank.home_poses[idx].p[:2], dtype=float)
            if float(np.linalg.norm(candidate[:2] - home_xy)) > float(bank.xy_tol):
                continue
            tip = candidate
            break
        if tip is None:
            return False
        top_z = float(bank.tops_z[idx])
        force = float(bank.force_stiffness) * max(
            0.0, top_z + float(bank.force_engage_slack) - float(tip[2])
        )
        engage = float(bank.force_full) * (
            float(bank.trigger_depth) / max(float(bank.max_depth), 1e-6)
        )
        return force >= engage

    def _advance_station_cook(self, station: dict[str, Any]) -> None:
        """Advance one station's doneness by one cook tick and recolor."""
        station["doneness"] = min(
            1.0, float(station["doneness"]) + 1.0 / max(1, self.cook_steps)
        )
        station["max_doneness"] = max(
            float(station["max_doneness"]), float(station["doneness"])
        )
        self._set_station_meat_color(station, station["doneness"])

    def _update_reactive_cook_keys(self) -> None:
        """Animate keys; latch ON/OFF (default) or spring hold (Opt 1).

        Latch: press while OFF → ON (stays down, cooking runs); press while ON →
        OFF immediately (cooking stops). Stove never turns off by itself.
        Hold (Opt 1): key springs with the gripper; no latch state.
        Keycap turns red while depressed and green when up.
        """
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        stations = getattr(self, "stations", None) or []
        hold = self.use_hold_cook
        for st in stations:
            tag = str(st.get("tag", ""))
            if not tag:
                continue
            if hold:
                # Spring-only; expert hold forces the key down while cooking.
                bank.set_forced(tag, bool(st.get("_expert_key_held")))
            else:
                forced = bool(st.get("cook_on")) or bool(st.get("_expert_key_held"))
                bank.set_forced(tag, forced)

        triggered = set(bank.update())
        for st in stations:
            sync_key_symbol(
                st.get("key_symbol_parts"),
                st.get("key_symbol_locals"),
                st.get("cook_key"),
            )

        if not hold:
            for st in stations:
                tag = str(st.get("tag", ""))
                if not tag or st.get("_ignore_key"):
                    continue
                touching = self._key_tip_pressing(st)
                if not st.get("cook_on"):
                    if tag in triggered:
                        self._set_station_cook_on(st, True)
                        # Consume this press so release is not a fresh edge.
                        st["_touch_latched"] = True
                    else:
                        st["_touch_latched"] = touching
                    continue

                # Rising tip edge while ON → OFF now; cooking stops with cook_on.
                if touching and not st.get("_touch_latched"):
                    self._set_station_cook_on(st, False)
                st["_touch_latched"] = touching

        for st in stations:
            self._set_station_key_color(st, self._key_visually_down(st))

    def _update_kinematic_tasks(self) -> None:
        """Advance base dynamics and per-station cooking state by one step."""
        super()._update_kinematic_tasks()
        stations = getattr(self, "stations", None) or []
        if not stations:
            return
        self._update_reactive_cook_keys()
        hold = self.use_hold_cook
        for st in stations:
            on_pan = self._steak_on_pan_station(st)
            if hold:
                if st.get("cook_phase_done"):
                    continue
                pressed = self._button_is_pressed_station(st)
                cooked = bool(st.get("_hold_cooked")) or float(
                    st.get("max_doneness", 0.0)
                ) > 0.0
                if pressed:
                    st["_hold_release_steps"] = 0
                    if on_pan:
                        self._advance_station_cook(st)
                        st["grasp_doneness"] = None
                        st["_hold_cooked"] = True
                elif cooked:
                    # Debounce unengaged frames so tip/spring flicker mid-press
                    # cannot freeze the shutoff score early.
                    steps = int(st.get("_hold_release_steps", 0)) + 1
                    st["_hold_release_steps"] = steps
                    need = int(
                        getattr(
                            self,
                            "HOLD_RELEASE_CONFIRM_STEPS",
                            self.HOLD_RELEASE_CONFIRM_STEPS,
                        )
                    )
                    if steps >= max(1, need):
                        st["grasp_doneness"] = float(st["doneness"])
                        st["cook_phase_done"] = True
            else:
                # Latch: cook only while cook_on (OFF press clears it immediately).
                if st.get("cook_on") and on_pan:
                    self._advance_station_cook(st)
        # Keep primary aliases in sync for obs / success helpers.
        primary = stations[0]
        self.doneness = float(primary["doneness"])
        self.max_doneness = float(primary["max_doneness"])
        self._grasp_doneness = primary["grasp_doneness"]
        self._track_cook_metrics(stations)

    def _cook_idle(self) -> None:
        """Step until every station reaches target doneness (or dwell limit)."""
        max_steps = int(round(self.target_doneness * self.cook_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if all(
                float(st["doneness"]) >= self.target_doneness for st in self.stations
            ):
                break

    def _move_pair(self, left_action, right_action=None) -> None:
        """Move one or both arms (``right_action`` optional for single-station)."""
        if right_action is None:
            self.move(left_action)
        else:
            self.move(left_action, right_action)

    def _cook_key_tip_pose(
        self, station: dict[str, Any], tip_z_above_top: float
    ) -> list[float]:
        """Top-down EE pose above a station's cook key (dispense_gummy-style)."""
        key_xy = station.get("key_xy")
        key_top_z = station.get("key_top_z")
        if key_xy is None or key_top_z is None:
            raise RuntimeError("cook_meat: cook key pose missing while button mode on")
        tcp_z = float(key_top_z) + float(tip_z_above_top)
        ee_z = tcp_z + float(self.EE_TO_TCP)
        return [
            float(key_xy[0]),
            float(key_xy[1]),
            ee_z,
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _safe_grasp_actor(self, actor: Any, arm_tag: ArmTag, **kwargs: Any):
        """``grasp_actor`` that skips the seed when no contact pose exists."""
        pre_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=float(kwargs.get("pre_grasp_dis", 0.1)),
            target_dis=float(kwargs.get("grasp_dis", 0.0)),
            contact_point_id=kwargs.get("contact_point_id"),
        )
        if pre_pose is None or grasp_pose is None:
            raise UnStableError(
                f"cook_meat: no grasp pose (seed {self._ep_seed}) -- skip"
            )
        return self.grasp_actor(actor, arm_tag=arm_tag, **kwargs)

    def _press_cook_keys(self, want_on: bool) -> None:
        """Press each station's cook key to latch ON or OFF.

        ON: cooking starts at depress; key stays down; gripper stays closed for
        the cook wait so OFF can press immediately.
        OFF: cooking stops at depress (``cook_on`` clears); then gripper opens.
        Uses absolute ``move_to_pose`` hover/press targets.
        """
        want_on = bool(want_on)
        for st in self.stations:
            if st["cook_key"] is None:
                raise RuntimeError("cook_meat: cook_key missing")
            st["_ignore_key"] = True
            st["_pending_off"] = False

        hover = float(getattr(self, "key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        # Target TCP near the key top; EE frame is EE_TO_TCP above TCP.
        press_above = max(0.0, hover - float(self.key_press_depth))

        def _press_pair(left_st, right_st=None) -> None:
            if right_st is None:
                arm = left_st["arm"]
                if want_on:
                    self.move(self.close_gripper(arm))
                    self.move(
                        self.move_to_pose(arm, self._cook_key_tip_pose(left_st, hover))
                    )
                # OFF: already hovering with gripper closed after the cook wait —
                # go straight to depress so cooking freezes at the press.
                self.move(
                    self.move_to_pose(arm, self._cook_key_tip_pose(left_st, press_above))
                )
                left_st["_expert_key_held"] = True
                # Latch ON/OFF at depress (cooking freezes only when OFF is pressed).
                self._set_station_cook_on(left_st, want_on)
                # Brief dwell at bottom so the depress is visible in demos.
                for _ in range(4):
                    self._update_kinematic_tasks()
                    self.scene.step()
                self.move(self.move_to_pose(arm, self._cook_key_tip_pose(left_st, hover)))
                left_st["_expert_key_held"] = False
                # Keep gripper closed after ON so OFF can press immediately.
                if not want_on:
                    self.move(self.open_gripper(arm))
                return

            la, ra = left_st["arm"], right_st["arm"]
            if want_on:
                self.move(self.close_gripper(la), self.close_gripper(ra))
                self.move(
                    self.move_to_pose(la, self._cook_key_tip_pose(left_st, hover)),
                    self.move_to_pose(ra, self._cook_key_tip_pose(right_st, hover)),
                )
            self.move(
                self.move_to_pose(la, self._cook_key_tip_pose(left_st, press_above)),
                self.move_to_pose(ra, self._cook_key_tip_pose(right_st, press_above)),
            )
            for st in (left_st, right_st):
                st["_expert_key_held"] = True
                self._set_station_cook_on(st, want_on)
            for _ in range(4):
                self._update_kinematic_tasks()
                self.scene.step()
            self.move(
                self.move_to_pose(la, self._cook_key_tip_pose(left_st, hover)),
                self.move_to_pose(ra, self._cook_key_tip_pose(right_st, hover)),
            )
            for st in (left_st, right_st):
                st["_expert_key_held"] = False
            if not want_on:
                self.move(self.open_gripper(la), self.open_gripper(ra))

        if len(self.stations) == 1:
            _press_pair(self.stations[0])
        else:
            left = next(st for st in self.stations if st["arm"] == "left")
            right = next(st for st in self.stations if st["arm"] == "right")
            _press_pair(left, right)

        for st in self.stations:
            # Keep ignore during scripted expert; interactive clears on load.
            interactive = bool(getattr(self, "_interactive_robot_mode", False)) or bool(
                getattr(self, "_interactive_universal_controls", False)
            )
            st["_ignore_key"] = not interactive
            st["_touch_latched"] = False
            st["_pending_off"] = False
            self._set_station_key_color(st, self._key_visually_down(st))

    def _hold_cook_buttons_until_done(self) -> None:
        """Opt 1 / Opt 1+2: press and hold each cook key until target doneness."""
        for st in self.stations:
            if st["cook_key"] is None:
                raise RuntimeError("cook_meat: cook_key missing")
            st["_ignore_key"] = True

        hover = float(getattr(self, "key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        press_above = max(0.0, hover - float(self.key_press_depth))

        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            self.move(self.close_gripper(arm))
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, hover)))
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, press_above)))
            st["_expert_key_held"] = True
            self._cook_idle()
            st["cook_phase_done"] = True
            st["grasp_doneness"] = float(st["doneness"])
            st["_expert_key_held"] = False
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, hover)))
            self.move(self.open_gripper(arm))
        else:
            left = next(st for st in self.stations if st["arm"] == "left")
            right = next(st for st in self.stations if st["arm"] == "right")
            la, ra = left["arm"], right["arm"]
            self.move(self.close_gripper(la), self.close_gripper(ra))
            self.move(
                self.move_to_pose(la, self._cook_key_tip_pose(left, hover)),
                self.move_to_pose(ra, self._cook_key_tip_pose(right, hover)),
            )
            self.move(
                self.move_to_pose(la, self._cook_key_tip_pose(left, press_above)),
                self.move_to_pose(ra, self._cook_key_tip_pose(right, press_above)),
            )
            for st in self.stations:
                st["_expert_key_held"] = True
            self._cook_idle()
            for st in self.stations:
                st["cook_phase_done"] = True
                st["grasp_doneness"] = float(st["doneness"])
                st["_expert_key_held"] = False
            self.move(
                self.move_to_pose(la, self._cook_key_tip_pose(left, hover)),
                self.move_to_pose(ra, self._cook_key_tip_pose(right, hover)),
            )
            self.move(self.open_gripper(la), self.open_gripper(ra))

        for st in self.stations:
            interactive = bool(getattr(self, "_interactive_robot_mode", False)) or bool(
                getattr(self, "_interactive_universal_controls", False)
            )
            st["_ignore_key"] = not interactive
            self._set_station_key_color(st, False)

    def _press_cook_buttons(self) -> None:
        """Cook via hold (Opt 1) or latch ON → wait → OFF (default / Opt 2)."""
        if self.use_hold_cook:
            self._hold_cook_buttons_until_done()
            return
        self._press_cook_keys(want_on=True)
        # Stay above the keys while cooking — only a small lift, no home retract
        # (long travel back to origin overshoots the doneness window before OFF).
        lift = 0.04
        if len(self.stations) == 1:
            self.move(
                self.move_by_displacement(
                    arm_tag=self.stations[0]["arm"], z=lift, move_axis="world"
                )
            )
        else:
            left = next(st for st in self.stations if st["arm"] == "left")
            right = next(st for st in self.stations if st["arm"] == "right")
            self.move(
                self.move_by_displacement(arm_tag=left["arm"], z=lift, move_axis="world"),
                self.move_by_displacement(arm_tag=right["arm"], z=lift, move_axis="world"),
            )
        self._cook_idle()
        self._press_cook_keys(want_on=False)

    # ------------------------------------------------------------- policy
    def _dbg(self, tag: str) -> None:
        """Print opt-in planner diagnostics for task tuning."""
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_meat] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self) -> dict[str, Any]:
        """Run the expert trajectory for one episode.

        The steak is always a stationary target (options only add a second
        station). Never use the base-task moving-target workflow even when the
        shared config sets ``use_dynamic: true``.
        """
        return self._play_once_static()

    def _pan_place_target(self, station: dict[str, Any]) -> list[float]:
        """Return the drop pose (XYZ + quat) that centers a steak in this pan.

        XY comes from the bowl-floor center so the steak keeps the same finger
        clearance to the rim on every mesh, yaw and side; Z stays the functional
        point's height. ``place_dx`` is mirrored on the left station so one
        config value nudges both stations the same way relative to the pan.
        """
        target = list(station["skillet"].get_functional_point(0))
        bowl_x, bowl_y = station.get(
            "bowl_center_xy", (target[0], target[1])
        )
        side = 1.0 if float(station.get("side", 1.0)) > 0 else -1.0
        target[0] = float(bowl_x) + side * self.place_dx
        target[1] = float(bowl_y) + self.place_dy
        return target

    def _place_steaks_on_pans(self) -> None:
        """Place each held steak into its skillet bowl (parallel when dual)."""
        def _place_one(st: dict[str, Any]) -> None:
            arm = st["arm"]
            pan_target = self._pan_place_target(st)
            self.move(
                self.place_actor(
                    st["steak"],
                    target_pose=pan_target,
                    arm_tag=arm,
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.02,
                    is_open=True,
                )
            )
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))

        if len(self.stations) == 1:
            st = self.stations[0]
            _place_one(st)
            if not self._steak_on_pan_station(st):
                # One retry — first place often drops short after yawed board grasp.
                _place_one(st)
            if not self._steak_on_pan_station(st):
                raise UnStableError(
                    f"cook_meat: steak not on pan after place (seed {self._ep_seed}) -- skip"
                )
            return

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        actions = []
        for st in (left, right):
            pan_target = self._pan_place_target(st)
            actions.append(
                self.place_actor(
                    st["steak"],
                    target_pose=pan_target,
                    arm_tag=st["arm"],
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.02,
                    is_open=True,
                )
            )
        self.move(actions[0], actions[1])
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.10, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.10, move_axis="arm"),
        )
        missing = [st for st in (left, right) if not self._steak_on_pan_station(st)]
        for st in missing:
            _place_one(st)
        if any(not self._steak_on_pan_station(st) for st in (left, right)):
            raise UnStableError(
                f"cook_meat: steak not on pan after place (seed {self._ep_seed}) -- skip"
            )

    def _return_steaks_to_boards(self) -> None:
        """Grasp cooked steaks off pans and set them back on their cutting boards.

        Dual: both arms grasp together and place together so one steak does not
        keep cooking while the other is returned.
        """
        def _board_target(st: dict[str, Any]) -> list[float]:
            # Live board pose (more reliable than cached XY after any sim drift).
            p = st["board"].get_pose().p
            return [float(p[0]), float(p[1]), float(st["board_top"]) + 0.03]

        def _d_board(st: dict[str, Any]) -> float:
            sxy = np.asarray(st["steak"].get_pose().p[:2], dtype=float)
            bxy = np.asarray(st["board"].get_pose().p[:2], dtype=float)
            return float(np.linalg.norm(sxy - bxy))

        def _latch_after_grasp(st: dict[str, Any]) -> None:
            self._latch_grasp_doneness(st, force=True)

        def _place_on_board(st: dict[str, Any]) -> None:
            arm = st["arm"]
            place_kwargs = dict(
                arm_tag=arm,
                constrain="free",
                pre_dis=0.10,
                dis=0.015,
                is_open=True,
            )
            self.move(self.place_actor(st["steak"], target_pose=_board_target(st), **place_kwargs))
            # Retries: left-arm places often undershoot (~0.17–0.21 m from board).
            for dis in (0.01, 0.0):
                if _d_board(st) < 0.12:
                    break
                self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.08))
                self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
                self.move(
                    self.place_actor(
                        st["steak"],
                        target_pose=_board_target(st),
                        arm_tag=arm,
                        constrain="free",
                        pre_dis=0.12,
                        dis=dis,
                        is_open=True,
                    )
                )

        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            st["awaiting_return_grasp"] = True
            self.move(self.open_gripper(arm))
            self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.1))
            _latch_after_grasp(st)
            self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
            _place_on_board(st)
            self.move(self.move_by_displacement(arm_tag=arm, z=0.08))
            return

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        for st in (left, right):
            st["awaiting_return_grasp"] = True
        self.move(self.open_gripper(left["arm"]), self.open_gripper(right["arm"]))
        self.move(
            self._safe_grasp_actor(left["steak"], arm_tag=left["arm"], pre_grasp_dis=0.1),
            self._safe_grasp_actor(right["steak"], arm_tag=right["arm"], pre_grasp_dis=0.1),
        )
        _latch_after_grasp(left)
        _latch_after_grasp(right)
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.12, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.12, move_axis="arm"),
        )
        # Place sequentially so one arm's retry cannot collide with the other.
        for st in (left, right):
            _place_on_board(st)
            self.move(self.move_by_displacement(arm_tag=st["arm"], z=0.08))

    def _pan_cook_table(self) -> None:
        """Place steak(s) on pan(s), then cook via hold or latch key."""
        self._place_steaks_on_pans()
        self._dbg("place_in_pan")
        self._press_cook_buttons()
        self._dbg("cook_done")

    def _task_info(self, arm_tag: ArmTag | None = None) -> dict[str, str]:
        """Build the language-template substitutions recorded for this episode."""
        primary = self.stations[0] if getattr(self, "stations", None) else None
        skillet_id = (
            primary["skillet_id"]
            if primary is not None
            else getattr(self, "skillet_id", 0)
        )
        if arm_tag is None:
            if self.dual_setup_enabled:
                arm_str = "both arms"
            elif primary is not None:
                arm_str = str(primary["arm"])
            else:
                arm_str = "left"
        else:
            arm_str = str(arm_tag)
        info = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{skillet_id}",
            "{C}": "104_board/base0",
            "{E}": "cook_key",
            "{a}": arm_str,
            "{o}": self._option_label(),
        }
        if self.dual_setup_enabled:
            info["{n}"] = "2"
        return info

    def _play_once_static(self) -> dict[str, Any]:
        """Execute the expert trajectory for stationary steak(s)."""
        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=arm, z=0.1, move_axis="arm"))
            self._pan_cook_table()
            self.info["info"] = self._task_info(arm)
            return self.info

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        self.move(
            self._safe_grasp_actor(left["steak"], arm_tag=left["arm"], pre_grasp_dis=0.1),
            self._safe_grasp_actor(right["steak"], arm_tag=right["arm"], pre_grasp_dis=0.1),
        )
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.1, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.1, move_axis="arm"),
        )
        self._pan_cook_table()
        self.info["info"] = self._task_info()
        return self.info

    def get_dynamic_motion_config(self) -> dict[str, Any] | None:
        """Cook-meat never uses the shared moving-target workflow."""
        return None

    # ------------------------------------------------------------- success
    def _doneness_range_bounds(self) -> tuple[float, float]:
        """Inclusive [low, high] success band for doneness."""
        target_range = getattr(self, "target_doneness_range", None)
        if target_range is not None:
            return float(target_range[0]), float(target_range[1])
        tol = float(getattr(self, "cook_doneness_tol", self.COOK_DONENESS_TOL_DEFAULT))
        t = float(self.target_doneness)
        return t - tol, t + tol

    def _doneness_in_target_range(self, doneness: float) -> bool:
        """Return whether doneness is inside the configured inclusive success range."""
        low, high = self._doneness_range_bounds()
        return low <= float(doneness) <= high

    def _station_overcooked(self, station: dict[str, Any]) -> bool:
        """True once this steak has gone past the upper doneness bound."""
        _, high = self._doneness_range_bounds()
        vals = [
            float(station.get("doneness", 0.0)),
            float(station.get("max_doneness", 0.0)),
        ]
        g = station.get("grasp_doneness")
        if g is not None:
            vals.append(float(g))
        return any(v > high for v in vals)

    def any_station_overcooked(self) -> bool:
        """True if any steak (dual or single) has exceeded the success band."""
        stations = getattr(self, "stations", None)
        if not stations:
            _, high = self._doneness_range_bounds()
            d = float(getattr(self, "doneness", 0.0))
            g = getattr(self, "_grasp_doneness", None)
            return bool(d > high or (g is not None and float(g) > high))
        return any(self._station_overcooked(st) for st in stations)

    def _station_success(self, station: dict[str, Any]) -> bool:
        """Return whether one steak was shut off inside the doneness range.

        Cook quality uses ``grasp_doneness`` (value when cooking stopped). Hold
        mode (Opt 1): that freeze happens on the first key release after cooking
        started — do not evaluate while the key is still held. The steak may
        remain on the pan — board return is not required. Dual episodes call
        this once per steak, and both must pass.
        """
        if self._station_overcooked(station):
            return False
        if station.get("grasp_doneness") is None:
            return False
        # Cooking must be stopped: latch OFF, or hold key released / phase done.
        if self.use_hold_cook:
            if self._button_is_pressed_station(station):
                return False
            if not bool(station.get("cook_phase_done")):
                return False
        elif bool(station.get("cook_on")):
            return False
        return bool(self._doneness_in_target_range(float(station["grasp_doneness"])))

    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self) -> None:
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_band_open_step = None   # every steak first inside the band
        self._metric_shutoff_step = None     # every steak's cooking stopped
        self._metric_doneness_error = None   # worst |grasp - centre| / half-width

    def _track_cook_metrics(self, stations: list[dict[str, Any]]) -> None:
        """Latch when the success band opened and when the last steak was shut off.

        High-water marks: doneness keeps rising past the band and a station can
        be re-cooked, so only the first crossing / first full shutoff is kept.
        """
        try:
            low, high = self._doneness_range_bounds()
            if getattr(self, "_metric_band_open_step", None) is None and all(
                float(st.get("doneness", 0.0)) >= low for st in stations
            ):
                self._metric_band_open_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
            if getattr(self, "_metric_shutoff_step", None) is not None:
                return
            grasps = [st.get("grasp_doneness") for st in stations]
            if any(g is None for g in grasps):
                return
            self._metric_shutoff_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
            centre = 0.5 * (low + high)
            half = max(0.5 * (high - low), 1e-9)
            self._metric_doneness_error = float(max(
                abs(float(g) - centre) / half for g in grasps
            ))
        except Exception:
            pass

    def _compute_metrics(self) -> dict[str, Any]:
        """extra1 = band-open->shutoff latency, extra2 = doneness error in half-bands.

        ``doneness_error_norm`` is |grasp_doneness - band centre| divided by the
        band half-width, worst over stations: 0.0 = the ideal doneness, 1.0 =
        exactly on a band edge, >1.0 = under/overcooked. LOWER IS BETTER. None
        when cooking was never shut off.
        """
        metrics: dict[str, Any] = {
            "shutoff_latency_steps": None,
            "shutoff_latency_s": None,
            "doneness_error_norm": (
                None
                if getattr(self, "_metric_doneness_error", None) is None
                else float(self._metric_doneness_error)
            ),
        }
        start = getattr(self, "_metric_band_open_step", None)
        off = getattr(self, "_metric_shutoff_step", None)
        if start is not None and off is not None and off >= start:
            steps = int(off - start)
            metrics["shutoff_latency_steps"] = steps
            try:
                metrics["shutoff_latency_s"] = round(
                    steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics

    def check_success(self) -> bool:
        """Return whether every steak was cooked into the target doneness range.

        Key must be up (latch OFF, or hold mode's first release) with
        ``grasp_doneness`` in ``target_doneness_range``. Dual: both steaks must
        pass. Any steak past the upper bound is an immediate fail (do not wait
        for the other station to shut off). Hold mode never succeeds while a
        key is still held, but overcooking still fails.
        """
        stations = getattr(self, "stations", None)
        if self.any_station_overcooked():
            return False
        if not stations:
            # Unit-test path that builds a bare task without load_actors.
            if self._grasp_doneness is None:
                return False
            return bool(self._doneness_in_target_range(float(self._grasp_doneness)))
        if bool(getattr(self, "dual_setup_enabled", False)) and len(stations) != 2:
            return False
        return all(self._station_success(st) for st in stations)

    def get_obs(self) -> dict[str, Any]:
        """Extend the base observation with per-frame cooking measurements."""
        obs = super().get_obs()
        stations = getattr(self, "stations", None) or []
        obs["cooking"] = {
            "doneness": float(getattr(self, "doneness", 0.0)),
            "target_doneness": float(
                getattr(self, "target_doneness", self.TARGET_DONENESS_DEFAULT)
            ),
            "target_doneness_range": list(
                getattr(self, "target_doneness_range", self.TARGET_DONENESS_RANGE_DEFAULT)
            ),
            "target_doneness_range_shift": float(
                getattr(self, "target_doneness_range_shift", 0.0)
            ),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
            "cook_button_enabled": bool(getattr(self, "cook_button_enabled", False)),
            "dual_setup_enabled": bool(getattr(self, "dual_setup_enabled", False)),
            "use_cook_button": True,
            "use_hold_cook": bool(self.use_hold_cook),
            "button_pressed": bool(self._button_is_pressed()),
            "cook_on": [bool(st.get("cook_on")) for st in stations],
            "n_stations": int(len(stations)),
            "station_doneness": [float(st["doneness"]) for st in stations],
        }
        try:
            if stations:
                sxy = np.array(stations[0]["steak"].get_pose().p)[:2]
                pxy = np.array(stations[0]["skillet"].get_functional_point(0))[:2]
                bxy = np.array(stations[0].get("board_xy", stations[0]["board"].get_pose().p[:2]))
                obs["cooking"]["steak_xy"] = [float(sxy[0]), float(sxy[1])]
                obs["cooking"]["pan_xy"] = [float(pxy[0]), float(pxy[1])]
                obs["cooking"]["place_offset"] = [
                    float(sxy[0] - pxy[0]),
                    float(sxy[1] - pxy[1]),
                ]
                obs["cooking"]["board_xy"] = [float(bxy[0]), float(bxy[1])]
        except Exception:
            pass
        return obs
