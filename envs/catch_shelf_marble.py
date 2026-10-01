from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.physx
import numpy as np
import os

_CSM_DEBUG = os.environ.get("CSM_DEBUG", "0") == "1"


class catch_shelf_marble(Base_Task):
    """A belt-mounted bowl catches a marble that cascades down a stack of tilted, interleaved shelves.

    Between `n_shelves_min` and `n_shelves_max` (default 4-7) short shelves hang above a belt,
    stacked top to bottom, each shifted left/right of the one above it (overlapping by at most half
    a shelf length) so the layout zig-zags like a bagatelle board; each shelf's tilt magnitude
    (`tilt_min_deg`-`tilt_max_deg`, default 15-45) is randomized independently every episode, and
    the *direction* each shelf leans is also drawn independently (not forced to alternate), so the
    cascade's net horizontal drift -- and therefore which part of the belt the marble ultimately
    drops onto -- varies far more from episode to episode than a strict left-right-left zig-zag
    would allow. The number of shelves is randomized per episode too (unless `n_shelves` is set
    explicitly), and the vertical spacing between shelves auto-scales with the shelf count (see
    `stack_height`) so the overall stack height stays roughly constant regardless of how many
    shelves are in play. A marble starts at rest on the centre of the top shelf and slides/falls
    down through the whole stack onto the belt below.

    Unlike a button-hop belt, the bowl here moves CONTINUOUSLY: a left action key (pressed by the
    left arm) slides the bowl left at a constant speed for as long as it is held, and a right
    action key (pressed by the right arm) slides it right; releasing either key freezes the bowl on
    the spot. The robot must hold the correct key long enough to place the bowl under the marble's
    predicted landing point on the belt before it arrives.

    The full descent (kinematic slide along each shelf following the instantaneous downhill
    ``sign(phi)`` — so an oscillating shelf correctly reverses the marble when its tip reverses —
    then a short parabolic free-fall onto the next shelf or, from the bottom shelf, onto the
    belt) is precomputed analytically in `load_actors` from the randomized geometry, so both
    collector passes replay the identical marble path and the same `target_catch_x` the expert
    policy aims for.

    Options (independent toggles; CLI via ``--task-arg`` or legacy ``--option``):
      - Default — marble pauses on the top shelf until a bowl key press edge (gripper or
        keyboard). Holding slides the bowl; the marble is released once on that first press.
      - Option 1 — ``reactive_marble``: marble starts moving/falling from episode start
        (released at ``play_once``, not on key-press). Uses ``reactive_roll_speed`` so the arm
        can still catch up. CLI: ``--task-arg reactive_marble=true`` or ``--option 1``.
      - Option 2 — ``oscillating_shelf_enabled``: one random non-top shelf sweeps ``-x..+x``.
        CLI: ``--task-arg oscillating_shelf_enabled=true`` or ``--option 2``.
      Options 1 and 2 may be combined.

    Other task_args knobs:
    - `n_shelves_min`/`n_shelves_max`: per-episode randomized shelf count range. Defaults 4-7. Set
      `n_shelves` explicitly to pin a fixed count instead (skips the randomization).
    - `stack_height`: total vertical span (m) from the bottom shelf to the top shelf; `level_gap`
      (the per-level spacing) is derived as `stack_height / (n_shelves - 1)` so taller stacks
      (more shelves) don't grow the scene's overall height. Set `level_gap` explicitly to pin a
      fixed per-level spacing instead (skips the derivation).
    - `max_stack_span` (default 0.65m): soft cap on cascade centre-to-centre width. The
      hard bound is the button window: shelves and belt must stay inside
      ``[key_x_left - layout_pad_from_key, key_x_right + layout_pad_from_key]`` (default
      ±20 cm past the green/blue keys) and are recentered on table ``x=0``. The layout may
      be shorter than that window; it is never longer.
    - `layout_pad_from_key` (default 0.20m): how far past each bowl key the shelf/belt
      arrangement may extend along x (+5 cm/side vs the prior 0.15m pad).
    - `tilt_min_deg`/`tilt_max_deg`: the randomized range (degrees) each shelf's tilt magnitude is
      drawn from independently every episode (direction is separately, independently randomized per
      shelf and tied to that shelf's own zig-zag offset; see `load_actors`). Defaults 15-45.
    - Opt 1 ``reactive_marble``: release at ``play_once`` start; effective slide speed becomes
      ``reactive_roll_speed`` (free-fall legs between shelves are unaffected).
    - Opt 2 ``oscillating_shelf_enabled``: one *non-top* shelf (indices ``1..n_shelves-1``, or
      pinned via ``oscillating_shelf_index``; index ``0`` rejected) sweeps
      ``x * cos(2*pi*t/oscillating_shelf_period)`` for the whole episode, including while the
      marble is parked. Descent plan is recomputed at release against the live osc phase so both
      collector passes agree on ``target_catch_x``.
    """

    N_SHELVES_MIN_DEFAULT = 4
    N_SHELVES_MAX_DEFAULT = 7
    SHELF_LENGTH_DEFAULT = 0.20
    SHELF_DEPTH_DEFAULT = 0.10
    SHELF_THICK_DEFAULT = 0.016
    LEVEL_GAP_DEFAULT = 0.13              # per-level spacing when n_shelves == 4 (the old fixed count)
    STACK_HEIGHT_DEFAULT = LEVEL_GAP_DEFAULT * (N_SHELVES_MIN_DEFAULT - 1)  # 0.39; kept constant
                                           # across n_shelves so a 7-shelf stack isn't much taller
                                           # than a 4-shelf one -- level_gap is derived from this
    OFFSET_MIN_FRAC_DEFAULT = 0.55        # min |offset| as a fraction of shelf_length -> overlap <= 45%
    OFFSET_MAX_FRAC_DEFAULT = 0.92        # max |offset| as a fraction of shelf_length -> overlap >= 8%
    TILT_MIN_DEG_DEFAULT = 15.0
    TILT_MAX_DEG_DEFAULT = 45.0
    BOTTOM_CLEARANCE_DEFAULT = 0.22       # belt surface up to the bottom shelf's underside
    STACK_SHIFT_RANGE_DEFAULT = 0.0       # no random x-shift; cascade is centered on table x=0
    MAX_STACK_SPAN_DEFAULT = 0.65         # soft cap on (max - min) shelf-centre x; further limited by
                                           # the green/blue key window (+ layout_pad_from_key)
    LAYOUT_PAD_FROM_KEY_DEFAULT = 0.20    # m; belt/shelves may extend this far past each bowl key (+5 cm/side)

    OSCILLATING_SHELF_ENABLED_DEFAULT = False
    OSCILLATING_SHELF_PERIOD_DEFAULT = 3.0    # s, full -x -> +x -> -x cycle (runs even while marble parked)
    # Heuristic lead (sim steps) from "decide which key" to "key actually pressed / marble released"
    # in non-reactive mode -- used only to pick a provisional `target_catch_x` for arm selection;
    # the true plan is recomputed at `_release_marble` against the live osc phase.
    OSC_KEY_APPROACH_LEAD_STEPS_DEFAULT = 280
    # Flat-shelf deadband for scripted slide: |phi| below this → marble holds (no preferred downhill).
    SLIDE_FLAT_PHI_EPS = 1e-4             # rad (~0.006 deg); osc tip passes through this near cos=0
    # If the oscillating tip reverses after the marble has already passed this fraction of
    # half-length toward an edge, commit to falling off that edge (momentum / lip) instead of
    # reversing and rocking forever around centre.
    SLIDE_LIP_COMMIT_FRAC = 0.45

    BALL_RADIUS_DEFAULT = 0.014
    ROLL_SPEED_DEFAULT = 0.175            # m/s, constant scripted speed for both slide and fall legs
    MAX_FALL_STEPS_DEFAULT = 500          # safety cap (per leg) for the offline descent-plan search
    GRAVITY = 9.81

    REACTIVE_MARBLE_DEFAULT = False       # if True: release at play_once start, not on key-press
    REACTIVE_ROLL_SPEED_DEFAULT = 0.045   # m/s; slower slide speed used only when reactive_marble
                                           # is on, so the descent lasts long enough for the arm's
                                           # fixed reach/press sequence to still catch up in time

    BOWL_ID_DEFAULT = 1
    BOWL_RADIUS_DEFAULT = 0.05
    BOWL_SCALE_MULT_DEFAULT = 0.8
    BOWL_CATCH_XY_TOL_DEFAULT = 0.035
    BOWL_SPEED_DEFAULT = 0.18             # m/s, continuous bowl speed while a key is held
    BELT_THICKNESS_DEFAULT = 0.015
    BELT_MARGIN_DEFAULT = 0.10

    KEY_HALF_DEFAULT = [0.028, 0.028, 0.016]
    KEY_HOVER_DIS_DEFAULT = 0.06
    KEY_PRESS_DEPTH_DEFAULT = 0.055
    KEY_PRESS_XY_DEFAULT = 0.045
    KEY_PRESS_DZ_DEFAULT = 0.17
    # Max keycap travel as a fraction of key half-height (keeps the top above the base rim).
    KEY_TRAVEL_FRAC_DEFAULT = 0.85
    KEY_SPRING_STEP_DEFAULT = 0.0007
    # Thin hollow bezel under each key (add_key_base_border).
    KEY_BASE_WALL_T = 0.006
    KEY_BASE_HALF_Z = 0.004
    KEY_BASE_MARGIN = 0.008
    KEY_X_LEFT_DEFAULT = -0.26
    KEY_X_RIGHT_DEFAULT = 0.26
    KEY_Y_DEFAULT = -0.13
    EE_TO_TCP = 0.12

    PRESS_LOOP_TOL_DEFAULT = 0.006
    PRESS_LOOP_MAX_STEPS_DEFAULT = 500
    POST_CATCH_DWELL_DEFAULT = 20

    # Glass shelves (catch_cuboid window-glass path): very light blue tint + 80% transmission.
    SHELF_COLOR = [0.94, 0.97, 1.0]
    SHELF_TRANSMISSION = 0.8
    SHELF_TRANSMISSION_ROUGHNESS = 0.0
    SHELF_ROUGHNESS = 0.02
    SHELF_IOR = 1.45
    BELT_COLOR = [0.10, 0.10, 0.12]
    KEY_BASE_COLOR = [0.28, 0.28, 0.31]
    LEFT_KEY_COLOR = [0.20, 0.70, 0.35]
    RIGHT_KEY_COLOR = [0.18, 0.48, 0.82]
    MARBLE_COLOR = [0.85, 0.15, 0.15]
    BOWL_COLOR = [0.95, 0.82, 0.12]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_shelf_marble", {})
        # The collector reuses this env across episodes; _init_task_env_ runs load_camera (which
        # calls _update_kinematic_tasks) BEFORE the new load_actors rebuilds the scene, so every
        # per-episode state variable is cleared here and the _loaded guard blocks stale updates.
        self._loaded = False
        self.shelves = []
        self.shelf_centers_x = []
        self.shelf_z = []
        self.shelf_dir = []
        self.shelf_angle_deg = []
        self.shelf_half_len = 0.0
        self.shelf_half_thick = 0.0
        self.ball = None
        self.bowl = None
        self.bowl_q = [0.5, 0.5, 0.5, 0.5]
        self.belt = None
        self.descent_legs = []
        self.total_marble_steps = 0
        self.target_catch_x = 0.0
        self._marble_state = "parked"     # parked -> descending -> resolved
        self._marble_result = None        # None | "caught" | "missed"
        self._leg_idx = 0
        self._leg_step = 0
        self.osc_enabled = False
        self.osc_shelf_idx = -1
        self._osc_steps = 0
        self._osc_armed = False  # flipped True at play_once; keeps check_stable from seeing motion
        self._reset_metric_state()
        self.key_xy = {}
        self.key_rest_xyz = {}
        self.key_arrows = {}
        self.keys = {}
        self._key_home = {}
        self._key_pressed = {"left": False, "right": False}
        self._key_depression = {"left": 0.0, "right": 0.0}
        self._reactive_buttons = None
        # Interactive / scripted latch: None | "left" | "right" (ORed into EE detect).
        self._expert_hold = None
        self._bowl_force_stop = False
        self._bowl_drive_clamp = None
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()
        # After settle: osc must not run during check_stable. Policy eval never
        # calls play_once. Default marble still waits for a physical key press.
        self._osc_armed = True
        if bool(getattr(self, "reactive_marble", False)):
            self._release_marble()

    def _configure_observer_camera(self):
        """Frame the whole belt + shelf stack from the table's upper-right corner (third-person
        overview), mirroring `dispense_gummy`'s convention. The shelf stack here is taller than
        that fixture, so the camera sits further back and higher.

        Recreates the shared `observer_camera` (default 320×240 / 93° fovy) at 1280×960 with a
        true 2× optical zoom (fovy halved via `2*atan(tan(fovy/2)/2)`), so demo recordings stay
        sharp and fill the frame with the fixture instead of empty table/background."""
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        old = cams.observer_camera
        near = float(old.near) if hasattr(old, "near") else 0.1
        far = float(old.far) if hasattr(old, "far") else 100.0
        old_fovy = float(old.fovy) if hasattr(old, "fovy") else np.deg2rad(93.0)
        # 2× optical zoom: new_fovy = 2 * atan(tan(old_fovy/2) / 2)
        zoom_fovy = 2.0 * float(np.arctan(np.tan(old_fovy / 2.0) / 2.0))
        try:
            self.scene.remove_camera(old)
        except Exception:
            pass
        cams.observer_camera = self.scene.add_camera(
            name="observer_camera",
            width=1280,
            height=960,
            fovy=zoom_fovy,
            near=near,
            far=far,
        )
        camera = cams.observer_camera
        camera_pos = np.array([0.38, 0.52, 1.45], dtype=np.float64)
        look_at = np.array([0.0, -0.05, 0.95], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = np.stack([forward, left, up], axis=1)
        camera_matrix[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(camera_matrix))

    # ------------------------------------------------------------------ helpers
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
        raise ValueError(f"catch_shelf_marble expected a boolean, got {value!r}")

    def _parse_reactive_marble(self, c) -> bool:
        """Opt 1: marble starts moving at episode start (preferred) or legacy ``option: 1``."""
        reactive = c.get("reactive_marble", c.get("opt1", None))
        legacy = c.get("option", None)
        if legacy is not None and reactive is None:
            if legacy in (1, "1", "reactive_marble", "reactive"):
                reactive = True
            elif legacy in (2, "2", "oscillating_shelf_enabled", "oscillating_shelf", "osc"):
                reactive = False
            else:
                raise ValueError(
                    "catch_shelf_marble option must be 1/reactive_marble or "
                    "2/oscillating_shelf_enabled (or set the booleans directly)"
                )
        return self._as_bool(reactive, self.REACTIVE_MARBLE_DEFAULT)

    def _parse_oscillating_shelf_enabled(self, c) -> bool:
        """Opt 2: one shelf oscillates (preferred) or legacy ``option: 2``."""
        osc = c.get("oscillating_shelf_enabled", c.get("opt2", None))
        legacy = c.get("option", None)
        if legacy is not None and osc is None:
            if legacy in (2, "2", "oscillating_shelf_enabled", "oscillating_shelf", "osc"):
                osc = True
            elif legacy in (1, "1", "reactive_marble", "reactive"):
                osc = False
            else:
                raise ValueError(
                    "catch_shelf_marble option must be 1/reactive_marble or "
                    "2/oscillating_shelf_enabled (or set the booleans directly)"
                )
        return self._as_bool(osc, self.OSCILLATING_SHELF_ENABLED_DEFAULT)

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "reactive_marble", False):
            parts.append("option 1")
        if getattr(self, "osc_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _use_viewer_glass(self) -> bool:
        """Interactive SAPIEN viewer cannot composite transmission glass — use plain alpha."""
        if bool(getattr(self, "_plain_glass", False)):
            return True
        if bool(self._cfg.get("plain_glass", False)):
            return True
        return bool(
            getattr(self, "_interactive_robot_mode", False)
            or getattr(self, "_interactive_universal_controls", False)
        )

    def _make_glass_material(self):
        """Shelf glass: transmission for demo cameras; pour_beer-style alpha for interactive."""
        if self._use_viewer_glass():
            # pour_beer-style plain alpha, but stronger light-blue + ~30% less transparent
            # than the initial 0.28 alpha (0.28 → ~0.50) so shelves read clearly in the viewer.
            glass = sapien.render.RenderMaterial(
                base_color=[0.72, 0.88, 0.98, 0.50]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.10)
                glass.set_metallic(0.0)
            except Exception:
                try:
                    glass.roughness = 0.10
                    glass.metallic = 0.0
                except Exception:
                    pass
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        # Expert / demo recording: catch_cuboid-style light-blue + 80% transmission.
        glass = sapien.render.RenderMaterial(base_color=[*self.SHELF_COLOR, 1.0])
        glass.set_transmission(float(self.SHELF_TRANSMISSION))
        glass.set_transmission_roughness(float(self.SHELF_TRANSMISSION_ROUGHNESS))
        glass.set_roughness(float(self.SHELF_ROUGHNESS))
        glass.set_metallic(0.0)
        try:
            glass.set_ior(float(self.SHELF_IOR))
        except Exception:
            glass.ior = float(self.SHELF_IOR)
        return glass

    def _create_glass_shelf(self, pose, half_size, is_static, name):
        """Shelf box with collision + catch_cuboid-style glass visual (moves with the entity)."""
        # Mirror create_entity_box, but attach a transmission glass material instead of opaque paint.
        scene = self.scene
        entity = sapien.Entity()
        entity.set_name(name)
        # Respect table_z_bias the same way create_box(self, ...) does via preprocess.
        z_bias = float(getattr(self, "table_z_bias", 0.0) or 0.0)
        posed = sapien.Pose(
            [float(pose.p[0]), float(pose.p[1]), float(pose.p[2]) + z_bias],
            pose.q,
        )
        entity.set_pose(posed)

        rigid = (
            sapien.physx.PhysxRigidDynamicComponent()
            if not is_static
            else sapien.physx.PhysxRigidStaticComponent()
        )
        rigid.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half_size,
                material=scene.default_physical_material,
            )
        )
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(half_size, self._make_glass_material()))
        entity.add_component(rigid)
        entity.add_component(render)
        scene.add_entity(entity)

        # Same Actor metadata create_box attaches, so helpers that expect an Actor keep working.
        data = {
            "center": [0, 0, 0],
            "extents": half_size,
            "scale": half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
            "contact_points_description": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "target_point_description": [],
        }
        return Actor(entity, data)

    def _recolor(self, actor, rgb):
        """Solid-color override. Clears base-color textures (002_bowl ships textured)."""
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
                # Drop metallic/speckled look so the solid yellow reads cleanly.
                try:
                    mat.set_metallic(0.0)
                    mat.set_roughness(0.35)
                except Exception:
                    pass

    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
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

    # -------------------------------------------------------- shelf geometry
    def _shelf_phi(self, idx, osc_steps=None):
        """`phi` fed into both the shelf's rendered/collision quaternion (`load_actors`) and the
        `_shelf_local_to_world`/`_shelf_surface_z_at_local` family below. No sign flip here: with
        `quat = [cos(phi/2), 0, sin(phi/2), 0]` (a standard +Y-axis rotation) and those helpers'
        `R_y(phi)` convention, `phi = +deg2rad(angle_deg)` is exactly what makes a positive
        `shelf_angle_deg` tip the +local_x (right) edge down, matching the convention the
        offset/tilt-direction coupling in `load_actors` (`shelf_angle_deg[i] = shelf_dir[i] *
        magnitude`) relies on.

        For the one shelf picked as `osc_shelf_idx` (never the top shelf; see
        `oscillating_shelf_enabled`), the angle is modulated by
        `cos(2*pi * osc_steps*dt / oscillating_shelf_period)` against the episode-absolute
        `_osc_steps` clock (defaults to live `self._osc_steps` when `osc_steps` is None). All
        other shelves ignore the clock entirely."""
        base_deg = self.shelf_angle_deg[idx]
        if self.osc_enabled and idx == self.osc_shelf_idx:
            if osc_steps is None:
                osc_steps = self._osc_steps
            dt = float(self.scene.get_timestep())
            t = float(osc_steps) * dt
            base_deg = base_deg * float(np.cos(2.0 * np.pi * t / self.osc_period))
        return np.deg2rad(base_deg)

    def _shelf_local_to_world(self, idx, local_x, osc_steps=None):
        """World point on shelf `idx`'s tilted top surface, `local_x` measured along the shelf's own
        long axis from its centre (matches the +x=>right-edge-down convention used throughout:
        positive angle tips the +local_x edge down). For the oscillating shelf, `osc_steps`
        selects which point in its sweep to evaluate (see `_shelf_phi`); for every other shelf the
        surface is fixed and this argument has no effect.

        Must exactly match the rotation actually applied to the shelf's box actor -- the quaternion
        built from `phi` in `load_actors` (`[cos(phi/2), 0, sin(phi/2), 0]`) is a standard +Y-axis
        rotation, i.e. `world = R_y(phi) @ local` with `R_y(phi) = [[cphi,0,sphi],[0,1,0],
        [-sphi,0,cphi]]`. (A previous version of this formula used the mirror-image rotation
        `R_y(-phi)`, which put the marble's scripted position off the real tilted surface by an
        amount that grew with `|local_x|` -- visually, the marble drifted through the shelf mesh
        instead of riding its top face.)"""
        phi = self._shelf_phi(idx, osc_steps=osc_steps)
        cphi, sphi = np.cos(phi), np.sin(phi)
        local_z = self.shelf_half_thick + self.ball_radius
        wx = local_x * cphi + local_z * sphi
        wz = -local_x * sphi + local_z * cphi
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        return np.array([cx + wx, self.belt_y, cz + wz], dtype=np.float64)

    def _shelf_world_x_to_local(self, idx, world_x, osc_steps=None):
        """Inverse of `_shelf_local_to_world`'s x-mapping (same `R_y(phi)` convention, same
        `osc_steps` semantics)."""
        phi = self._shelf_phi(idx, osc_steps=osc_steps)
        cphi, sphi = np.cos(phi), np.sin(phi)
        if abs(cphi) < 1e-6:
            return None
        local_z = self.shelf_half_thick + self.ball_radius
        cx = self.shelf_centers_x[idx]
        return (world_x - cx - local_z * sphi) / cphi

    def _shelf_surface_z_at_local(self, idx, local_x, osc_steps=None):
        """Same `R_y(phi)` convention (and `osc_steps` semantics) as
        `_shelf_local_to_world`'s z-mapping."""
        phi = self._shelf_phi(idx, osc_steps=osc_steps)
        cphi, sphi = np.cos(phi), np.sin(phi)
        local_z = self.shelf_half_thick + self.ball_radius
        cz = self.shelf_z[idx]
        return cz - local_x * sphi + local_z * cphi

    # -------------------------------------------------- offline descent plan
    def _downhill_sign_from_phi(self, phi):
        """+1 when +local_x is downhill, -1 when -local_x is downhill, 0 when effectively flat.

        Matches `_shelf_local_to_world`: positive `phi` tips the +local_x edge down."""
        if abs(float(phi)) < self.SLIDE_FLAT_PHI_EPS:
            return 0.0
        return 1.0 if float(phi) > 0.0 else -1.0

    def _simulate_slide_locals(self, shelf_idx, start_local, origin_osc_steps, cum_steps):
        """Step `local_x` along shelf `shelf_idx` following the *instantaneous* downhill.

        For static shelves this is a constant-speed slide to `sign(phi)*half_len`. For the
        oscillating shelf, `phi` flips with `cos(ωt)`, so the marble waits when flat and
        reverses when the tip reverses — instead of blindly chasing the load-time `shelf_dir`
        edge (which looked like sliding uphill / a random direction).

        Tip-reverse near a lip (`SLIDE_LIP_COMMIT_FRAC`): once the marble is past that fraction
        of half-length toward an edge, a slope flip commits the exit on that edge rather than
        rocking forever around centre (travel time to the lip can exceed a quarter osc period).

        Returns one `local_x` per sim step (length >= 1). Clocking matches playback:
        step `k` (1-based) uses `origin_osc_steps + cum_steps + k`, same as live `_osc_steps`
        after `_update_kinematic_tasks` increments then calls `_advance_marble`."""
        dt = float(self.scene.get_timestep())
        step_dist = max(float(self.roll_speed), 1e-4) * dt
        half = float(self.shelf_half_len)
        local = float(np.clip(start_local, -half, half))
        locals_out = []
        prev_downhill = 0.0
        lip = float(self.SLIDE_LIP_COMMIT_FRAC) * half

        # Static shelves finish in ~half_len/roll_speed; osc shelves may reverse for a few periods.
        one_way = int(round((2.0 * half / max(float(self.roll_speed), 1e-4)) / dt)) + 2
        if self.osc_enabled and shelf_idx == self.osc_shelf_idx:
            period_steps = max(1, int(round(float(self.osc_period) / max(dt, 1e-6))))
            max_steps = max(one_way, 4 * period_steps)
        else:
            max_steps = max(1, one_way)

        for k in range(1, max_steps + 1):
            osc_at = int(origin_osc_steps) + int(cum_steps) + k
            phi = self._shelf_phi(shelf_idx, osc_steps=osc_at)
            downhill = self._downhill_sign_from_phi(phi)

            # Lip commit: tip reversed after we were already heading toward / past mid-lip.
            if (
                prev_downhill != 0.0
                and downhill != 0.0
                and downhill != prev_downhill
                and abs(local) >= lip
                and (1.0 if local > 0.0 else -1.0) == prev_downhill
            ):
                local = float(prev_downhill * half)
                locals_out.append(local)
                break

            if downhill != 0.0:
                edge = downhill * half
                delta = edge - local
                if abs(delta) <= step_dist:
                    local = edge
                else:
                    local = local + step_dist * (1.0 if delta > 0.0 else -1.0)
                prev_downhill = downhill
            # else: flat — hold position; keep prev_downhill so a flip after a flat still commits
            local = float(np.clip(local, -half, half))
            locals_out.append(local)
            if abs(local) >= half - 1e-9:
                break
        else:
            # Safety: still on the shelf after max_steps — exit toward the nearer / last downhill edge.
            if abs(local) < 1e-9:
                fallback = prev_downhill if prev_downhill != 0.0 else float(self.shelf_dir[shelf_idx])
                local = float(fallback * half)
            else:
                local = float((1.0 if local > 0.0 else -1.0) * half)
            locals_out.append(local)

        if not locals_out:
            locals_out = [local]
        return locals_out

    def _compute_descent_plan(self, origin_osc_steps=0):
        """Analytically precompute the marble's entire path (a list of deterministic kinematic
        "legs": slide-along-a-shelf, then parabolic-free-fall-to-the-next-shelf-or-the-belt) from
        the randomized shelf geometry. This is what makes the two collector passes replay
        bit-identically and gives the expert policy a `target_catch_x` to aim the bowl at.

        `origin_osc_steps` is the episode-absolute `_osc_steps` value at the instant of marble
        release; `cum_steps` then counts ticks since that release. Every `_shelf_*` lookup below
        therefore sees the oscillating shelf (if any) at precisely the angle it will actually be
        at when the marble gets there -- matching live playback, which keys off `_osc_steps`.

        Slide direction follows `sign(phi)` at each tick (see `_simulate_slide_locals`), so an
        oscillating shelf's tip reverse correctly sends the marble downhill instead of toward the
        fixed load-time `shelf_dir` edge."""
        dt = float(self.scene.get_timestep())
        g = self.GRAVITY
        legs = []
        cur_shelf = 0
        cur_local = 0.0
        cum_steps = 0
        x = z = 0.0
        for _ in range(self.n_shelves + 2):
            locals_traj = self._simulate_slide_locals(
                cur_shelf, cur_local, origin_osc_steps, cum_steps
            )
            slide_steps = len(locals_traj)
            edge_local = float(locals_traj[-1])
            legs.append({
                "type": "slide",
                "shelf": cur_shelf,
                "start_local": float(cur_local),
                "end_local": edge_local,
                "locals": [float(v) for v in locals_traj],
                "steps": int(slide_steps),
            })
            cum_steps += slide_steps

            edge_pos = self._shelf_local_to_world(
                cur_shelf, edge_local, osc_steps=origin_osc_steps + cum_steps
            )
            # Leave in the direction of the edge we exited (or live downhill if we timed out mid-shelf).
            if abs(edge_local) >= self.shelf_half_len - 1e-6:
                exit_sign = 1.0 if edge_local > 0.0 else -1.0
            else:
                phi_exit = self._shelf_phi(
                    cur_shelf, osc_steps=origin_osc_steps + cum_steps
                )
                exit_sign = self._downhill_sign_from_phi(phi_exit)
                if exit_sign == 0.0:
                    exit_sign = float(self.shelf_dir[cur_shelf])
            vx = exit_sign * self.roll_speed
            landed_shelf, landed_local = None, None
            k = 0
            for k in range(1, self.max_fall_steps + 1):
                t = k * dt
                x = float(edge_pos[0] + vx * t)
                z = float(edge_pos[2] - 0.5 * g * t * t)
                if z <= self.belt_surface_z + self.ball_radius:
                    z = self.belt_surface_z + self.ball_radius
                    break
                hit = False
                for j in range(cur_shelf + 1, self.n_shelves):
                    osc_at = origin_osc_steps + cum_steps + k
                    lx = self._shelf_world_x_to_local(j, x, osc_steps=osc_at)
                    if lx is None or abs(lx) > self.shelf_half_len:
                        continue
                    surf_z = self._shelf_surface_z_at_local(j, lx, osc_steps=osc_at)
                    if z <= surf_z:
                        landed_shelf, landed_local = j, float(lx)
                        hit = True
                        break
                if hit:
                    break
            legs.append({
                "type": "fall",
                "start_pos": (float(edge_pos[0]), float(edge_pos[1]), float(edge_pos[2])),
                "vx": float(vx),
                "steps": int(k),
            })
            cum_steps += k

            if landed_shelf is None:
                self.target_catch_x = float(x)
                break
            cur_shelf, cur_local = landed_shelf, landed_local
        else:
            self.target_catch_x = float(x)

        self.descent_legs = legs
        self.total_marble_steps = int(sum(leg["steps"] for leg in legs))

    def _apply_target_catch_bounds(self):
        """Clip `target_catch_x` into the bowl's travel range and grow `press_loop_max_steps` if the
        new target is farther than the budget sized at load time."""
        self.target_catch_x = float(np.clip(self.target_catch_x, self.bowl_x_min, self.bowl_x_max))
        bowl_x0 = float(self.bowl.get_pose().p[0]) if self.bowl is not None else 0.5 * (
            self.bowl_x_min + self.bowl_x_max
        )
        dt = float(self.scene.get_timestep())
        needed_steps = int(np.ceil(abs(self.target_catch_x - bowl_x0) / max(self.bowl_speed, 1e-4) / dt)) + 80
        self.press_loop_max_steps = max(self.press_loop_max_steps, needed_steps)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.n_shelves_min = int(c.get("n_shelves_min", self.N_SHELVES_MIN_DEFAULT))
        self.n_shelves_max = int(c.get("n_shelves_max", self.N_SHELVES_MAX_DEFAULT))
        if "n_shelves" in c:
            self.n_shelves = int(c["n_shelves"])
        else:
            self.n_shelves = int(np.random.randint(self.n_shelves_min, self.n_shelves_max + 1))
        self.randomize_shelf_params = bool(c.get("randomize_shelf_params", False))
        shelf_jitter = float(np.clip(abs(float(c.get("shelf_param_jitter", 0.10))), 0.0, 0.95))
        j = shelf_jitter if self.randomize_shelf_params else 0.0

        def _pm(mean):
            mean = float(mean)
            if j <= 0.0:
                return mean
            return float(np.random.uniform(mean * (1.0 - j), mean * (1.0 + j)))

        self.shelf_length = _pm(c.get("shelf_length", self.SHELF_LENGTH_DEFAULT))
        self.shelf_depth = float(c.get("shelf_depth", self.SHELF_DEPTH_DEFAULT))
        self.shelf_thick = float(c.get("shelf_thick", self.SHELF_THICK_DEFAULT))
        self.stack_height = float(c.get("stack_height", self.STACK_HEIGHT_DEFAULT))
        if "level_gap" in c:
            self.level_gap = float(c["level_gap"])
        else:
            self.level_gap = self.stack_height / max(1, self.n_shelves - 1)
        self.offset_min_frac = float(c.get("offset_min_frac", self.OFFSET_MIN_FRAC_DEFAULT))
        self.offset_max_frac = float(c.get("offset_max_frac", self.OFFSET_MAX_FRAC_DEFAULT))
        self.tilt_min_deg = float(c.get("tilt_min_deg", self.TILT_MIN_DEG_DEFAULT))
        self.tilt_max_deg = float(c.get("tilt_max_deg", self.TILT_MAX_DEG_DEFAULT))
        self.bottom_clearance = float(c.get("bottom_clearance", self.BOTTOM_CLEARANCE_DEFAULT))
        self.stack_shift_range = float(c.get("stack_shift_range", self.STACK_SHIFT_RANGE_DEFAULT))
        self.max_stack_span = float(c.get("max_stack_span", self.MAX_STACK_SPAN_DEFAULT))
        self.layout_pad_from_key = float(
            c.get("layout_pad_from_key", self.LAYOUT_PAD_FROM_KEY_DEFAULT)
        )

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.roll_speed = _pm(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        self.max_fall_steps = int(c.get("max_fall_steps", self.MAX_FALL_STEPS_DEFAULT))

        self.reactive_marble = self._parse_reactive_marble(c)
        self.reactive_roll_speed = _pm(
            c.get("reactive_roll_speed", self.REACTIVE_ROLL_SPEED_DEFAULT)
        )
        if self.reactive_marble:
            # The marble starts falling immediately (see `play_once`), well before the arm's fixed
            # reach/press sequence finishes -- slow the slide legs down so the descent has a chance
            # of still being catchable. The gravity-timed free-fall legs are untouched.
            self.roll_speed = self.reactive_roll_speed

        self.osc_enabled = self._parse_oscillating_shelf_enabled(c)
        self.osc_period = float(c.get("oscillating_shelf_period", self.OSCILLATING_SHELF_PERIOD_DEFAULT))
        self.osc_key_approach_lead_steps = int(
            c.get("osc_key_approach_lead_steps", self.OSC_KEY_APPROACH_LEAD_STEPS_DEFAULT)
        )
        # Never oscillate the top shelf (idx 0) -- the marble parks / launches from there.
        if self.osc_enabled and self.n_shelves >= 2:
            osc_idx_cfg = c.get("oscillating_shelf_index", None)
            if osc_idx_cfg is not None:
                idx = int(osc_idx_cfg) % self.n_shelves
                if idx == 0:
                    idx = 1
                self.osc_shelf_idx = idx
            else:
                self.osc_shelf_idx = int(np.random.randint(1, self.n_shelves))
        else:
            if self.osc_enabled and self.n_shelves < 2:
                self.osc_enabled = False
            self.osc_shelf_idx = -1
        self._osc_steps = 0
        self._osc_armed = False

        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_radius = float(c.get("bowl_radius", self.BOWL_RADIUS_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_catch_xy_tol = float(c.get("bowl_catch_xy_tol", self.BOWL_CATCH_XY_TOL_DEFAULT))
        self.bowl_speed = float(c.get("bowl_speed", self.BOWL_SPEED_DEFAULT))
        self.belt_thickness = float(c.get("belt_thickness", self.BELT_THICKNESS_DEFAULT))
        self.belt_margin = float(c.get("belt_margin", self.BELT_MARGIN_DEFAULT))

        self.key_half = list(c.get("key_half", self.KEY_HALF_DEFAULT))
        self.key_hover_dis = float(c.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        self.key_press_depth = float(c.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT))
        self.key_press_xy = float(c.get("key_press_xy", self.KEY_PRESS_XY_DEFAULT))
        self.key_press_dz = float(c.get("key_press_dz", self.KEY_PRESS_DZ_DEFAULT))
        travel_frac = float(c.get("key_travel_frac", self.KEY_TRAVEL_FRAC_DEFAULT))
        # Prefer explicit key_travel when set; otherwise scale with keycap height.
        if c.get("key_travel", None) is not None:
            self.key_travel = float(c.get("key_travel"))
        else:
            self.key_travel = float(self.key_half[2]) * float(np.clip(travel_frac, 0.2, 1.0))
        self.key_spring_step = float(c.get("key_spring_step", self.KEY_SPRING_STEP_DEFAULT))
        self.key_x_left = float(c.get("key_x_left", self.KEY_X_LEFT_DEFAULT))
        self.key_x_right = float(c.get("key_x_right", self.KEY_X_RIGHT_DEFAULT))
        self.key_y = float(c.get("key_y", self.KEY_Y_DEFAULT))

        # Hard x-window for shelves + belt: green key − pad … blue key + pad, centered on table.
        self.layout_x_min = float(self.key_x_left) - float(self.layout_pad_from_key)
        self.layout_x_max = float(self.key_x_right) + float(self.layout_pad_from_key)
        if self.layout_x_max <= self.layout_x_min:
            raise ValueError(
                f"catch_shelf_marble layout window empty: "
                f"[{self.layout_x_min:.3f}, {self.layout_x_max:.3f}]"
            )

        self.press_loop_tol = float(c.get("press_loop_tol", self.PRESS_LOOP_TOL_DEFAULT))
        self.press_loop_max_steps = int(c.get("press_loop_max_steps", self.PRESS_LOOP_MAX_STEPS_DEFAULT))
        self.post_catch_dwell = int(c.get("post_catch_dwell", self.POST_CATCH_DWELL_DEFAULT))

        self.table_top = 0.74 + self.table_z_bias
        self.belt_y = 0.0
        self.belt_surface_z = self.table_top + self.belt_thickness

        self.shelf_half_len = self.shelf_length / 2.0
        self.shelf_half_depth = self.shelf_depth / 2.0
        self.shelf_half_thick = self.shelf_thick / 2.0

        # Cap centre-span so shelves + belt_margin fit inside the key window (may be smaller).
        allowed_width = float(self.layout_x_max - self.layout_x_min)
        fit_centers_span = allowed_width - self.shelf_length - 2.0 * self.belt_margin
        self.max_stack_span = float(min(self.max_stack_span, max(0.05, fit_centers_span)))

        # ---- randomize the zig-zag positions: each consecutive shelf-to-shelf offset direction is
        # drawn independently (not forced to alternate left-right-left), so the cascade's net drift
        # varies a lot more episode to episode -- sometimes several shelves in a row drift the same
        # way (large net displacement, marble drops near one end of the belt), sometimes they zig-zag
        # back and forth (small net displacement, marble drops near the middle). Magnitude is always
        # in [offset_min_frac, offset_max_frac] * shelf_length so consecutive shelves overlap by
        # somewhere in (0%, 50%] regardless of direction; only the *sign* combination is resampled
        # (never the magnitudes) if it would push past `max_stack_span` / the key window. The whole
        # cascade is then recentered on table x=0. ----
        centers = [0.0]
        offsets = []
        for _attempt in range(200):
            offset_signs = [float(np.random.choice([-1.0, 1.0])) for _ in range(self.n_shelves - 1)]
            offsets = [
                float(sign * np.random.uniform(
                    self.offset_min_frac * self.shelf_length, self.offset_max_frac * self.shelf_length
                ))
                for sign in offset_signs
            ]
            centers = [0.0]
            for off in offsets:
                centers.append(centers[-1] + off)
            if (max(centers) - min(centers)) > self.max_stack_span:
                continue
            mid = 0.5 * (min(centers) + max(centers))
            # Optional micro-shift (default 0); reject if it would leave the key window.
            shift = float(np.random.uniform(-self.stack_shift_range, self.stack_shift_range))
            centered = [c - mid + shift for c in centers]
            smin = min(centered) - self.shelf_half_len - self.belt_margin
            smax = max(centered) + self.shelf_half_len + self.belt_margin
            if smin >= self.layout_x_min - 1e-9 and smax <= self.layout_x_max + 1e-9:
                centers = centered
                break
        else:
            # Last attempt: force-center without shift and clamp span (should already fit).
            mid = 0.5 * (min(centers) + max(centers))
            centers = [c - mid for c in centers]
        self.shelf_centers_x = list(centers)

        # ---- tilt: direction for shelves 0..N-2 is tied to the offset toward the shelf below it
        # (so the marble's downhill edge lines up with the shelf it must land on); the bottom
        # shelf's direction (which side of the belt the marble finally exits toward) is
        # independent/random. Magnitude (steepness) is randomized per shelf, 15-45 degrees. ----
        self.shelf_dir = [1.0 if off > 0 else -1.0 for off in offsets]
        self.shelf_dir.append(float(np.random.choice([-1.0, 1.0])))
        self.shelf_angle_deg = [
            self.shelf_dir[i] * float(np.random.uniform(self.tilt_min_deg, self.tilt_max_deg))
            for i in range(self.n_shelves)
        ]

        bottom_z = self.belt_surface_z + self.bottom_clearance
        top_z = bottom_z + (self.n_shelves - 1) * self.level_gap
        self.shelf_z = [top_z - i * self.level_gap for i in range(self.n_shelves)]

        # ---- precompute the marble's full path (needs belt_surface_z + shelf geometry, not the
        # belt's x-extent) so target_catch_x is known before the belt/bowl bounds are sized.
        # With an oscillating shelf the landing x depends on release phase -- sample a few phases
        # so the belt spans every reachable landing, then keep the phase-0 plan as the provisional
        # target (replaced at `_release_marble` with the live-phase plan). ----
        dt = float(self.scene.get_timestep())
        landing_xs = []
        if self.osc_enabled:
            period_steps = max(1, int(round(self.osc_period / max(dt, 1e-6))))
            for frac in np.linspace(0.0, 1.0, 8, endpoint=False):
                self._compute_descent_plan(origin_osc_steps=int(round(frac * period_steps)))
                landing_xs.append(float(self.target_catch_x))
        self._compute_descent_plan(origin_osc_steps=0)
        landing_xs.append(float(self.target_catch_x))

        shelf_min_x = min(self.shelf_centers_x) - self.shelf_half_len
        shelf_max_x = max(self.shelf_centers_x) + self.shelf_half_len
        land_min_x = min(landing_xs)
        land_max_x = max(landing_xs)
        # Size belt to content, then clamp into the key window (may be shorter; never longer).
        self.belt_x_min = max(
            self.layout_x_min,
            min(shelf_min_x, land_min_x) - self.belt_margin,
        )
        self.belt_x_max = min(
            self.layout_x_max,
            max(shelf_max_x, land_max_x) + self.belt_margin,
        )
        if self.belt_x_max <= self.belt_x_min + 2.0 * (self.bowl_radius + 0.01):
            # Degenerate clamp — fall back to the full allowed window centered on 0.
            self.belt_x_min = float(self.layout_x_min)
            self.belt_x_max = float(self.layout_x_max)
        self.bowl_x_min = self.belt_x_min + self.bowl_radius + 0.01
        self.bowl_x_max = self.belt_x_max - self.bowl_radius - 0.01
        self.target_catch_x = float(np.clip(self.target_catch_x, self.bowl_x_min, self.bowl_x_max))

        # ---- the belt width (and therefore how far the bowl may need to slide) now varies with the
        # randomized cascade -- resize the key-hold step budget to the farthest landing this episode
        # can produce so a wide-belt / late-phase episode can't spuriously time out mid-slide. ----
        bowl_x0 = 0.5 * (self.bowl_x_min + self.bowl_x_max)
        farthest = max(abs(x - bowl_x0) for x in landing_xs)
        needed_steps = int(np.ceil(farthest / max(self.bowl_speed, 1e-4) / dt)) + 80
        self.press_loop_max_steps = max(self.press_loop_max_steps, needed_steps)

        # ---- belt (static) ----
        belt_center_x = 0.5 * (self.belt_x_min + self.belt_x_max)
        belt_half_x = 0.5 * (self.belt_x_max - self.belt_x_min)
        self.belt = create_box(
            self,
            pose=sapien.Pose([belt_center_x, self.belt_y, self.table_top + 0.5 * self.belt_thickness]),
            half_size=[belt_half_x, self.shelf_half_depth + 0.02, 0.5 * self.belt_thickness],
            color=self.BELT_COLOR,
            is_static=True,
            name="marble_belt",
        )
        self.add_prohibit_area(self.belt, padding=0.03)

        # ---- shelves: static + fixed-tilt, except the one oscillating shelf (if enabled), which is
        # built kinematic instead so `_animate_oscillating_shelf` can re-pose it every step.
        # Visual: catch_cuboid-style light-blue glass (80% transmission). ----
        self.shelves = []
        for i in range(self.n_shelves):
            phi = self._shelf_phi(i, osc_steps=0)  # rest pose at episode start (_osc_steps == 0)
            quat = [np.cos(phi / 2.0), 0.0, np.sin(phi / 2.0), 0.0]
            is_osc = self.osc_enabled and i == self.osc_shelf_idx
            shelf = self._create_glass_shelf(
                pose=sapien.Pose([self.shelf_centers_x[i], self.belt_y, self.shelf_z[i]], quat),
                half_size=[self.shelf_half_len, self.shelf_half_depth, self.shelf_half_thick],
                is_static=not is_osc,
                name=f"catch_shelf_{i}",
            )
            if is_osc:
                self._make_kinematic(shelf)
            self.shelves.append(shelf)

        # ---- marble: parked (kinematic) at the centre of the top shelf until play_once releases it ----
        ball_pose = self._shelf_local_to_world(0, 0.0)
        self.ball = create_sphere(
            self,
            pose=sapien.Pose(ball_pose.tolist()),
            radius=self.ball_radius,
            color=self.MARBLE_COLOR,
            is_static=False,
            name="catch_marble",
        )
        self._make_kinematic(self.ball)
        self._marble_state = "parked"
        self._marble_result = None
        self._leg_idx = 0
        self._leg_step = 0
        self._reset_metric_state()

        # ---- bowl: kinematic, rides the belt, starts at the belt's horizontal centre ----
        self.bowl_x_start = 0.5 * (self.bowl_x_min + self.bowl_x_max)
        bowl_pose = sapien.Pose([self.bowl_x_start, self.belt_y, self.belt_surface_z], self.bowl_q)
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
            scale_mult=self.bowl_scale_mult,
        )
        self.bowl.set_mass(0.06)
        self._recolor(self.bowl, self.BOWL_COLOR)
        self._make_kinematic(self.bowl)
        self.add_prohibit_area(self.bowl, padding=0.05)
        if _CSM_DEBUG:
            print(
                f"[CSM] load_actors done: n_shelves={self.n_shelves} level_gap={self.level_gap:.4f} "
                f"osc_enabled={self.osc_enabled} osc_shelf_idx={self.osc_shelf_idx} "
                f"bowl_x_start={self.bowl_x_start:.4f} "
                f"bowl_x_min={self.bowl_x_min:.4f} bowl_x_max={self.bowl_x_max:.4f} "
                f"bowl_pose_p={self.bowl.get_pose().p.round(4)} "
                f"target_catch_x={self.target_catch_x:.4f}",
                flush=True,
            )

        # ---- two action keys: left (pressed by the left arm) slides the bowl left while held,
        # right (pressed by the right arm) slides it right while held ----
        self.key_xy = {
            "left": (self.key_x_left, self.key_y),
            "right": (self.key_x_right, self.key_y),
        }
        # Key sits on the table inside a hollow bezel; travel is a fraction of key height so the
        # colored top stays above the rim at full press (never disappears into a solid base cube).
        self.key_top_z = self.table_top + 2.0 * float(self.key_half[2])
        key_colors = {"left": self.LEFT_KEY_COLOR, "right": self.RIGHT_KEY_COLOR}
        for side, (kx, ky) in self.key_xy.items():
            self._add_key_base_border(side, kx, ky)
            self.key_rest_xyz[side] = [kx, ky, self.table_top + float(self.key_half[2])]
            home = sapien.Pose(self.key_rest_xyz[side])
            self.keys[side] = create_box(
                self,
                pose=home,
                half_size=self.key_half,
                color=key_colors[side],
                is_static=True,
                name=f"action_key_{side}",
            )
            world_home = self.keys[side].get_pose()
            self._key_home[side] = world_home
            # Pass table-frame Z; create_visual_box adds table_z_bias itself.
            self.key_arrows[side] = self._draw_arrow(
                side, kx, ky, self.table_top + 2.0 * float(self.key_half[2]) + 0.0015
            )
            self.add_prohibit_area(self.keys[side], padding=0.04)

        max_depth = float(min(self.key_travel, float(self.key_half[2])))
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[self.keys[s] for s in ("left", "right")],
            home_poses=[self._key_home[s] for s in ("left", "right")],
            max_depth=max_depth,
            ids=["left", "right"],
            xy_tol=float(self.key_press_xy),
            visual_step=float(self.key_spring_step),
        )
        self._reactive_buttons.set_tops_z([
            float(self._key_home[s].p[2]) + float(self.key_half[2]) for s in ("left", "right")
        ])
        self.key_top_z = float(self._key_home["left"].p[2]) + float(self.key_half[2])
        self._loaded = True

    def _add_key_base_border(self, side, kx, ky):
        """Hollow dark bezel around the keycap (four walls, open center)."""
        add_key_base_border(
            self,
            float(kx),
            float(ky),
            float(self.table_top),
            self.key_half,
            wall_t=float(self.KEY_BASE_WALL_T),
            half_z=float(self.KEY_BASE_HALF_Z),
            margin=float(self.KEY_BASE_MARGIN),
            color=list(self.KEY_BASE_COLOR),
            name_prefix=f"action_key_base_{side}",
        )

    def _draw_arrow(self, side, key_x, key_y, z):
        # Author one left-pointing arrow, then rigidly rotate it 180 degrees for the right key, so
        # the two icons are exact mirror opposites.
        rotation = 0.0 if side == "left" else np.pi
        color = [0.95, 0.95, 0.95]
        parts = [
            ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
            ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
            ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ]
        c, s = np.cos(rotation), np.sin(rotation)
        arrows = []
        for name, (local_x, local_y), local_yaw, half_size in parts:
            x = key_x + c * local_x - s * local_y
            y = key_y + s * local_x + c * local_y
            yaw = local_yaw + rotation
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            arrow = create_visual_box(
                self,
                sapien.Pose([x, y, z], q),
                half_size=half_size,
                color=color,
                name=f"{side}_key_arrow_{name}",
            )
            # Snapshot world pose after create_visual_box (table_z_bias applied).
            wp = arrow.get_pose()
            arrows.append((arrow, [float(wp.p[0]), float(wp.p[1]), float(wp.p[2])]))
        return arrows

    # --------------------------------------------------------------- marble
    def _release_marble(self):
        if self._marble_state != "parked":
            return
        self._marble_state = "descending"
        # Metrics: the drop opens the window to get the bowl under the marble.
        if getattr(self, "_metric_release_step", None) is None:
            self._metric_release_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        self._marble_result = None
        self._leg_idx = 0
        self._leg_step = 0
        # Oscillation may have been running the whole time the marble was parked -- replan against
        # the live phase so `target_catch_x` / descent legs match what the shelf will actually do.
        if self.osc_enabled:
            self._compute_descent_plan(origin_osc_steps=self._osc_steps)
            self._apply_target_catch_bounds()
            if self._bowl_drive_clamp is not None:
                sign = self._bowl_drive_clamp[0]
                self._bowl_drive_clamp = (sign, self.target_catch_x)

    def _advance_marble(self):
        if self.ball is None or self._marble_state != "descending":
            return
        legs = self.descent_legs
        if self._leg_idx >= len(legs):
            self._marble_state = "landed"
            return
        leg = legs[self._leg_idx]
        self._leg_step += 1
        step = self._leg_step
        steps_total = max(1, leg["steps"])

        if leg["type"] == "slide":
            traj = leg.get("locals")
            if traj:
                local_x = float(traj[min(step, len(traj)) - 1])
            else:
                # Legacy lerp fallback (pre-trajectory plans).
                frac = min(1.0, step / steps_total)
                local_x = leg["start_local"] + frac * (leg["end_local"] - leg["start_local"])
            # Live `_osc_steps` matches `_compute_descent_plan(origin_osc_steps=...)`'s absolute
            # clock, so the oscillating shelf (if this is that shelf) is at the planned angle.
            # `locals` already followed sign(phi) at each tick, so this stays downhill.
            pos = self._shelf_local_to_world(leg["shelf"], local_x, osc_steps=self._osc_steps)
        else:
            dt = float(self.scene.get_timestep())
            t = min(step, steps_total) * dt
            sx, sy, sz = leg["start_pos"]
            x = sx + leg["vx"] * t
            z = max(sz - 0.5 * self.GRAVITY * t * t, self.belt_surface_z + self.ball_radius)
            pos = np.array([x, sy, z], dtype=np.float64)

        self._set_entity_pose(self.ball, sapien.Pose(pos.tolist()))

        if step >= steps_total:
            self._leg_idx += 1
            self._leg_step = 0
            if self._leg_idx >= len(legs):
                self._marble_state = "landed"

    def _resolve_marble(self):
        """Judge catch vs. miss from the bowl's position. Deliberately *not* called the instant
        the marble finishes its (precomputed, fixed-duration) descent -- `_advance_marble` only
        transitions to `"landed"` at that point, and `play_once` calls this afterwards, once the
        arm's own action sequence has actually finished. In `reactive_marble` mode especially, the
        marble's short scripted fall can finish well before the arm's fixed
        close_gripper/reach/press/hold sequence does (the bowl is still correctly *en route* to
        `target_catch_x` at that instant, just not there yet) -- resolving on the spot would judge
        a still-converging approach as a miss."""
        self._marble_state = "resolved"
        bowl_x = float(self.bowl.get_pose().p[0])
        # Metrics: this offset is exactly what decides caught vs. missed.
        try:
            self._metric_resolve_offset_norm = abs(
                bowl_x - float(self.target_catch_x)
            ) / max(float(self.bowl_catch_xy_tol), 1e-9)
        except Exception:
            self._metric_resolve_offset_norm = None
        if abs(bowl_x - self.target_catch_x) <= self.bowl_catch_xy_tol:
            self._marble_result = "caught"
        else:
            self._marble_result = "missed"
            self._set_entity_pose(
                self.ball,
                sapien.Pose([self.target_catch_x, self.belt_y, self.belt_surface_z + self.ball_radius]),
            )

    def _ride_marble_in_bowl(self):
        if self.ball is None or self.bowl is None:
            return
        bp = self.bowl.get_pose().p
        self._set_entity_pose(self.ball, sapien.Pose([bp[0], bp[1], bp[2] + self.ball_radius + 0.01]))

    # ------------------------------------------------------------------ keys
    def _detect_action_keys(self):
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        expert = getattr(self, "_expert_hold", None)
        for side in ("left", "right"):
            bank.set_forced(side, expert == side)
        triggered = bank.update()
        for side in ("left", "right"):
            pressed = bool(bank.is_held(side))
            if _CSM_DEBUG and pressed != self._key_pressed.get(side, False):
                print(f"[CSM] {side} key pressed={pressed}", flush=True)
            self._key_pressed[side] = pressed
            self._key_depression[side] = float(bank.visual_depth[bank.resolve_index(side)])
        # Default mode: marble stays parked until a bowl key is pressed.  Reactive mode releases
        # at play_once start instead.  Press edge (not hold) starts the descent — works for both
        # gripper teleop and keyboard latch (``_expert_hold``).
        if (
            not bool(getattr(self, "reactive_marble", False))
            and self._marble_state == "parked"
            and triggered
        ):
            if _CSM_DEBUG:
                print(f"[CSM] key press edge {triggered} → release marble", flush=True)
            self._release_marble()

    def _animate_keys(self):
        # Keycaps are posed by ReactivePushButtons; keep arrow decals in sync.
        for side in ("left", "right"):
            depth = float(self._key_depression.get(side, 0.0))
            for arrow, rest_xyz in self.key_arrows.get(side, []):
                pose = arrow.get_pose()
                self._set_entity_pose(
                    arrow,
                    sapien.Pose(
                        [rest_xyz[0], rest_xyz[1], rest_xyz[2] - depth], pose.q
                    ),
                )

    # ------------------------------------------------------------------ bowl
    def _advance_bowl(self):
        if self.bowl is None:
            return
        if self._bowl_force_stop:
            return
        dt = float(self.scene.get_timestep())
        left_p = self._key_pressed.get("left", False)
        right_p = self._key_pressed.get("right", False)
        dx = 0.0
        if left_p and not right_p:
            dx = -self.bowl_speed * dt
        elif right_p and not left_p:
            dx = self.bowl_speed * dt
        if dx == 0.0:
            return
        cur_x = float(self.bowl.get_pose().p[0])
        new_x = cur_x + dx
        # Clamp against an in-progress `_hold_key_to_target` goal so a single fixed-size step can't
        # carry the bowl past the point the scripted policy is aiming for -- otherwise the last step
        # before the loop's tolerance check trips can overshoot by up to one step's worth of travel.
        clamp = self._bowl_drive_clamp
        if clamp is not None:
            clamp_sign, clamp_x = clamp
            new_x = min(new_x, clamp_x) if clamp_sign > 0 else max(new_x, clamp_x)
        new_x = float(np.clip(new_x, self.bowl_x_min, self.bowl_x_max))
        self._set_entity_pose(self.bowl, sapien.Pose([new_x, self.belt_y, self.belt_surface_z], self.bowl_q))

    # ---------------------------------------------------------- scene motion
    def _animate_oscillating_shelf(self):
        """Smoothly re-pose the one oscillating (non-top) shelf every step -- including while the
        marble is still frozen/parked on the top shelf. See `_shelf_phi` for the `cos(...)` sweep;
        `_compute_descent_plan` / `_advance_marble` stay in lockstep via the shared `_osc_steps`
        clock."""
        if not (self.osc_enabled and self.shelves and 0 <= self.osc_shelf_idx < len(self.shelves)):
            return
        idx = self.osc_shelf_idx
        phi = self._shelf_phi(idx, osc_steps=self._osc_steps)
        quat = [np.cos(phi / 2.0), 0.0, np.sin(phi / 2.0), 0.0]
        pos = [self.shelf_centers_x[idx], self.belt_y, self.shelf_z[idx]]
        self._set_entity_pose(self.shelves[idx], sapien.Pose(pos, quat))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._detect_action_keys()
        self._animate_keys()
        self._advance_bowl()
        # Osc clock starts at play_once (`_osc_armed`), not during check_stable settle -- otherwise
        # the moving shelf fails the pose-stability test. Once armed it advances whether or not the
        # marble has been released, so the shelf keeps sweeping while the ball is still frozen.
        if self._osc_armed:
            self._osc_steps += 1
            self._animate_oscillating_shelf()
        if self._marble_state == "descending":
            self._advance_marble()
        elif self._marble_state == "resolved" and self._marble_result == "caught":
            self._ride_marble_in_bowl()
        self._track_bowl_in_position_metric()

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # ------------------------------------------------------------------ keys / policy helpers
    def _key_tip_pose(self, side, tip_z_above_top):
        kx, ky = self.key_xy[side]
        tcp_z = self.key_top_z + tip_z_above_top
        return [kx, ky, tcp_z + self.EE_TO_TCP, *GRASP_DIRECTION_DIC["top_down"]]

    def _hold_key_to_target(self, side, target_x, max_steps=None, on_pressed=None):
        """Press and hold action key `side` (continuously sliding the bowl) until the bowl's x
        reaches `target_x`, then release. Uses live position feedback (not a fixed step count) so
        the stop point is accurate regardless of IK/timing jitter during the approach.

        `on_pressed`, if given, fires right after the key is physically pressed down (before the
        monitoring loop starts) -- this is used to release the marble at that moment, so the
        marble's (fixed-duration) descent overlaps with the window where the bowl can actually
        move, instead of ticking away during the arm's earlier close-gripper/approach motion."""
        if max_steps is None:
            max_steps = self.press_loop_max_steps
        arm = ArmTag(side)
        # Set the drive clamp *before* any arm motion: `_key_pressed[side]` can flip True mid-way
        # through the press-down displacement below (as soon as the EE crosses the depth
        # threshold), which already drives `_advance_bowl` via `move()`'s own per-step
        # `_update_kinematic_tasks` calls -- so the clamp must be active for that motion too, not
        # just for the explicit monitoring loop that follows.
        self._bowl_force_stop = False
        sign = 1.0 if side == "right" else -1.0
        self._bowl_drive_clamp = (sign, target_x)
        self.move(self.move_to_pose(arm, self._key_tip_pose(side, self.key_hover_dis)))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return
        self.move(self.move_by_displacement(arm, z=-self.key_press_depth))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return
        if on_pressed is not None:
            on_pressed()

        steps = 0
        while steps < max_steps:
            cur_x = float(self.bowl.get_pose().p[0])
            if sign * (target_x - cur_x) <= self.press_loop_tol:
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (steps % self.save_freq == 0):
                self._take_picture()
            steps += 1
        self._bowl_drive_clamp = None

        # Lock the bowl in place *before* retracting: the arm's retract motion (move_by_displacement)
        # is IK/timing-driven and its exact duration jitters, which previously let the EE linger over
        # the key's press-depth zone for a variable number of extra steps -- adding an uncontrolled,
        # non-deterministic amount of extra bowl travel on top of the precisely-monitored loop above.
        # Forcing the stop here makes the bowl's final position bounded purely by `press_loop_tol`.
        self._bowl_force_stop = True
        self.move(self.move_by_displacement(arm, z=self.key_press_depth))
        self._dwell(6)

    # ------------------------------------------------------------------ policy
    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        # Start the oscillating shelf as soon as the episode action begins -- including the
        # close_gripper / approach window where the marble is still frozen on the top shelf.
        self._osc_armed = True
        self._reset_metric_state()

        if _CSM_DEBUG:
            print(f"[CSM] play_once start: bowl_pose_p={self.bowl.get_pose().p.round(4)}", flush=True)

        if self.reactive_marble:
            # Drop the marble before the arm does anything at all -- every subsequent move() call
            # steps `_update_kinematic_tasks` (and therefore `_advance_marble`) internally, so the
            # descent keeps running concurrently with close_gripper/reach/press below instead of
            # waiting for them.
            self._release_marble()

        self.move(self.close_gripper(left))
        self.move(self.close_gripper(right))

        if _CSM_DEBUG:
            print(
                f"[CSM] after close_gripper: bowl_pose_p={self.bowl.get_pose().p.round(4)} "
                f"plan_success={self.plan_success} total_marble_steps={self.total_marble_steps}",
                flush=True,
            )

        # With an oscillating shelf the landing depends on the release-time phase. In non-reactive
        # mode release is still ~reach+press away -- provisionally replan with that lead so the
        # chosen key side matches the eventual target; `_release_marble` then replans exactly.
        if self.osc_enabled and not self.reactive_marble:
            self._compute_descent_plan(
                origin_osc_steps=self._osc_steps + self.osc_key_approach_lead_steps
            )
            self._apply_target_catch_bounds()

        bowl_x0 = float(self.bowl.get_pose().p[0])
        dx = self.target_catch_x - bowl_x0
        held_side = None
        if abs(dx) > self.press_loop_tol:
            held_side = "right" if dx > 0 else "left"
            if _CSM_DEBUG:
                print(
                    f"[CSM] holding {held_side}: bowl_x0={bowl_x0:.4f} dx={dx:.4f} "
                    f"target_catch_x={self.target_catch_x:.4f} osc_shelf_idx={self.osc_shelf_idx}",
                    flush=True,
                )
            # In the default (non-reactive) mode, release the marble only once the key is actually
            # pressed down, so its fixed-duration fall overlaps with the window where the bowl can
            # move (not the earlier close-gripper/approach steps, which don't move the bowl at
            # all). In reactive mode the marble is already falling (released at the top of
            # play_once above), so there's nothing to trigger here.
            on_pressed = None if self.reactive_marble else self._release_marble
            self._hold_key_to_target(held_side, self.target_catch_x, on_pressed=on_pressed)
            if _CSM_DEBUG:
                print(
                    f"[CSM] after hold: bowl_x={float(self.bowl.get_pose().p[0]):.4f}",
                    flush=True,
                )
        elif not self.reactive_marble:
            self._release_marble()

        # Let the marble finish its full (precomputed) descent regardless of how long the key
        # hold above took; this loop is a pure step-and-wait, no arm motion.
        max_wait = self.total_marble_steps + 60
        waited = 0
        while self._marble_state == "descending" and waited < max_wait:
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (waited % self.save_freq == 0):
                self._take_picture()
            waited += 1

        # Judge catch vs. miss only now, using the bowl's actual (by now fully settled) position --
        # not at whatever earlier instant the marble's fixed-duration descent happened to finish.
        self._resolve_marble()
        self._dwell(self.post_catch_dwell)

        self.info["info"] = {
            "{A}": "marble",
            "{B}": "tilted shelves",
            "{C}": f"002_bowl/base{self.bowl_id}",
            "{D}": f"{held_side or 'either'} action key",
            "{a}": f"{held_side or 'left'} arm",
        }
        return self.info

    # ------------------------------------------------------------------ success
    # ------------------------------------------------- human-experiment metrics
    def _reset_metric_state(self):
        """Clear the per-episode metric latches (see _compute_metrics)."""
        self._metric_release_step = None
        self._metric_in_position_step = None
        self._metric_resolve_offset_norm = None

    def _track_bowl_in_position_metric(self):
        """Latch the first frame the bowl reached the marble's landing column.

        High-water mark: the bowl can drift back out (the belt keeps moving), so
        only the first arrival is kept. Stops querying once latched.
        """
        if getattr(self, "_metric_in_position_step", None) is not None:
            return
        if getattr(self, "_metric_release_step", None) is None:
            return
        try:
            bowl_x = float(self.bowl.get_pose().p[0])
            if abs(bowl_x - float(self.target_catch_x)) > float(self.bowl_catch_xy_tol):
                return
        except Exception:
            return
        self._metric_in_position_step = int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _compute_metrics(self):
        """extra1 = drop->bowl-in-position latency, extra2 = final aim error in tolerances.

        ``catch_offset_norm`` is |bowl_x - target_catch_x| at the instant the catch
        was judged, as a fraction of ``bowl_catch_xy_tol``: 0.0 = perfectly under
        the marble, 1.0 = exactly on the pass/fail boundary, >1.0 = a miss.
        LOWER IS BETTER. None when the catch was never judged.
        """
        metrics = {
            "in_position_latency_steps": None,
            "in_position_latency_s": None,
            "catch_offset_norm": (
                None
                if getattr(self, "_metric_resolve_offset_norm", None) is None
                else float(self._metric_resolve_offset_norm)
            ),
        }
        start = getattr(self, "_metric_release_step", None)
        arrived = getattr(self, "_metric_in_position_step", None)
        if start is not None and arrived is not None and arrived >= start:
            steps = int(arrived - start)
            metrics["in_position_latency_steps"] = steps
            try:
                metrics["in_position_latency_s"] = round(
                    steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics

    def check_success(self):
        self.info["target_catch_x"] = float(self.target_catch_x)
        self.info["marble_result"] = str(self._marble_result)
        self.info["marble_position"] = list(map(float, self.ball.get_pose().p)) if self.ball is not None else []
        return bool(self._marble_state == "resolved" and self._marble_result == "caught")

    def get_obs(self):
        obs = super().get_obs()
        obs["catch_shelf_marble"] = {
            "n_shelves": int(self.n_shelves),
            "shelf_centers_x": list(map(float, self.shelf_centers_x)),
            "shelf_z": list(map(float, self.shelf_z)),
            "shelf_angles_deg": list(map(float, self.shelf_angle_deg)),
            "reactive_marble": bool(self.reactive_marble),
            "oscillating_shelf_enabled": bool(self.osc_enabled),
            "oscillating_shelf_idx": int(self.osc_shelf_idx) if self.osc_enabled else -1,
            "option_label": self._option_label(),
            "target_catch_x": float(self.target_catch_x),
            "bowl_x": float(self.bowl.get_pose().p[0]) if self.bowl is not None else 0.0,
            "marble_state": str(self._marble_state),
            "marble_result": str(self._marble_result),
            "marble_position": (
                list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0]
            ),
        }
        return obs
