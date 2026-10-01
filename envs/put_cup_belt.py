from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class put_cup_belt(Base_Task):
    """Single-arm place-between-yellow-sticks task, with optional blue curtains.

    Main goal: place the cup into the moving gap between the two yellow sticks on the belt.
    The active arm starts with a cup in the near zone; the yellow sticks translate laterally on a
    fixed belt plate (the plate itself does not move) and keep sliding through deposit. The
    policy waits until the yellow-stick gap is reachable, tracks it, deposits the cup between
    the sticks, then retracts.

    Layout is randomly mirrored across the y-axis (x → −x) each episode: right-arm default
    on +x, left-arm mirror on −x. Cup XY pose (and a small yaw) is jittered every episode.

    Config options (task_args.put_cup_belt):
      blue_curtains_enabled / opt1 — spawn blue curtain strips in front of the belt
          (collidable; touching a strip fails the attempt). When enabled, the policy must
          also wait for a curtain gap over the yellow-stick corridor. Strip group x is
          sampled along the belt so the outer panel edges stay on the plate (flush at
          the extremes). Strips are 1.6× cup-tall (20% shorter than the previous 2×
          panels) and twice as wide as the original panels, with a 1 cm clear gap
          between them. Strip half-size is sampled each episode as
          nominal × U(1±curtain_size_jitter) (default ±20%).
      blue_curtain_dynamic_enabled / opt2 — if true without opt1, the blue curtains
          sway as a group. If both opt1 and opt2 are on, the center strip stays
          static and one side strip (left or right, sampled each episode) sways.
      belt_speed / belt_speed_jitter — nominal yellow-stick speed; each episode samples
          U((1−j)·nom, (1+j)·nom) with j default 0.20.

    Belt / curtain kinematics are pure functions of an internal step counter so the two
    collector passes (plan-only and render) stay deterministic.

    Metric: placement_score = clamp(1 - center_offset(cup, yellow-gap)/tol, 0, 1); forced
    to 0 if the cup ever collided with a blue curtain.
    """

    # ---- tunable params (CLASS DEFAULTS; overridable via task_args.put_cup_belt) ----
    CURTAIN_FREQ_DEFAULT = 0.8          # curtain oscillation: cycles per (sim-second-equivalent)
    CURTAIN_AMP_DEFAULT = 0.05          # lateral sway amplitude of the curtain group (m)
    CURTAIN_AMP_OPT2_MULT = 6.0         # opt2 group sway: 3× the previous 2× (0.10 m) stroke
    CURTAIN_PHASE_DEFAULT = 0.0         # initial phase offset (rad)
    BELT_SPEED_DEFAULT = 0.08           # nominal belt/yellow-stick speed; each ep samples ±jitter
    BELT_SPEED_JITTER_DEFAULT = 0.20    # fraction; speed ~ U((1-j)*nom, (1+j)*nom)
    CURTAIN_SIZE_JITTER_DEFAULT = 0.20  # fraction; strip half-size ~ U((1-j), (1+j)) * nominal
    BELT_RANGE_DEFAULT = 0.08           # yellow sticks travel +/- this in x before reversing (m)
    SLOT_START_DEFAULT = 0.0            # initial slot x offset within belt range (m)
    PLACEMENT_TOL_DEFAULT = 0.05        # tolerance for placement_score (m)
    GAP_TOL_DEFAULT = 0.06              # how close (in x) the gap center must be to the corridor for "open"
    REACH_TOL_DEFAULT = 0.05            # how close the slot x must be to the corridor
    # Yellow-stick center spacing (clear gap = spacing - 2*half_x); 2cm tighter than 0.12.
    MOVING_COMPONENT_SPACING_DEFAULT = 0.10
    BLUE_CURTAINS_ENABLED_DEFAULT = False   # default: no curtains; opt1 enables
    BLUE_CURTAIN_DYNAMIC_ENABLED_DEFAULT = False  # opt2: swaying curtains (implies opt1)
    LIFT_OFF_ONLY_AFTER_PLACE_DEFAULT = False
    POST_PLACE_LIFT_Z_DEFAULT = 0.10
    RANDOM_MIRROR_DEFAULT = True        # randomly flip layout to left-arm side
    # Cup |x| spawn band (m). Kept inward of the old 0.065–0.115 range so the cup
    # stays framed in the head camera.
    CUP_X_ABS_MIN = 0.070
    CUP_X_ABS_MAX = 0.095

    REACH_STEPS_EST = 90                # est. physics steps for the forward reach to complete
    N_STRIPS = 3                        # curtain vertical strips
    STRIP_GAP_IDX = 1                   # which inter-strip gap is "the gap" (center)
    CURTAIN_Y = 0.03                     # mid-zone y of the curtain plane
    BELT_CENTER_Y_OFFSET = 0.10         # belt centerline y (keep front edge behind curtains)
    BELT_Z_OFF = 0.01                   # belt platform half-height sits this far above table
    DT = 1.0 / 250.0                    # matches default scene timestep
    BELT_PLATE_HALF_Y = 0.048           # slightly thinner so belt can sit closer for left-arm reach
    BELT_PLATE_EDGE_MARGIN = 0.0        # outer peg faces flush with plate edges at travel extremes
    # Extra length on the placement (lean) side, as a fraction of the symmetric half-x.
    BELT_LEAN_EXTEND_FRAC = 0.30
    MOVING_COMPONENT_HALF_X = 0.008     # with 0.10 spacing → ~8.4cm clear gap (fits cup)
    MOVING_COMPONENT_HALF_Z = 0.018
    # Strip xy; height is 2× cup × 0.80 after spawn (20% shorter than the 2×-cup panels).
    # Width is 2× the original 1.2 cm half-x. Clear gap between panels is 1 cm.
    STRIP_HALF_XY = (0.024, 0.004)
    STRIP_HEIGHT_SCALE = 0.80
    STRIP_CLEAR_GAP = 0.01
    STRIP_SPACING = 2.0 * STRIP_HALF_XY[0] + STRIP_CLEAR_GAP  # 0.058 m at nominal width
    LIFT_CLEARANCE = 0.06               # lift = cup_height + this so the cup clears cup-tall curtains
    # Curtain hit uses geometric AABB plus PhysX box collision on the panels.
    CURTAIN_CONTACT_PENETRATION_X = 0.010   # m; must overlap this much in X
    CURTAIN_CONTACT_PENETRATION_Y = 0.018   # m; approach axis — was the loose false-positive
    CURTAIN_CONTACT_PENETRATION_Z = 0.008   # m
    BLUE_CUP_RGBA = [0.10, 0.35, 0.95, 1.0]
    CUP_UPRIGHT_QPOS = [0.707, 0.707, 0.0, 0.0]

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags not stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("put_cup_belt", {})
        # CRITICAL: the env instance is REUSED across episodes. _init_task_env_ resets the scene
        # and calls _update_kinematic_tasks (via load_camera) BEFORE our load_actors rebuilds the
        # bodies. Invalidate stale kinematic-body handles now so the per-step hook is a no-op until
        # load_actors recreates them -- otherwise set_kinematic_target on a freed body segfaults.
        self.curtain_strips = []
        self.curtain_single_side_motion = False
        self.moving_strip_idx = -1
        self.belt_pegs = []
        self.belt_plate = None
        self._attempt_active = False
        self._reset_metric_state()
        if hasattr(self, "_kin_step"):
            del self._kin_step
        super()._init_task_env_(**kwags)

    # --------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.curtain_freq = float(c.get("curtain_freq", self.CURTAIN_FREQ_DEFAULT))
        self.curtain_amp = float(c.get("curtain_amp", self.CURTAIN_AMP_DEFAULT))
        belt_speed_nom = float(c.get("belt_speed", self.BELT_SPEED_DEFAULT))
        speed_jitter = float(c.get("belt_speed_jitter", self.BELT_SPEED_JITTER_DEFAULT))
        speed_jitter = float(np.clip(speed_jitter, 0.0, 0.95))
        # Per-episode speed sample around the nominal (±jitter, default ±20%).
        self.belt_speed = float(
            np.random.uniform(belt_speed_nom * (1.0 - speed_jitter),
                              belt_speed_nom * (1.0 + speed_jitter))
        )
        self.belt_range = float(c.get("belt_range", self.BELT_RANGE_DEFAULT))
        self.placement_tol = float(c.get("placement_tol", self.PLACEMENT_TOL_DEFAULT))
        self.gap_tol = float(c.get("gap_tol", self.GAP_TOL_DEFAULT))
        self.reach_tol = float(c.get("reach_tol", self.REACH_TOL_DEFAULT))
        self.slot_spacing = float(c.get("moving_component_spacing",
                                        c.get("slot_spacing", self.MOVING_COMPONENT_SPACING_DEFAULT)))
        self.randomize_slot_spacing = bool(c.get("randomize_slot_spacing", False))
        spacing_jitter = float(c.get("moving_component_spacing_jitter",
                                     c.get("slot_spacing_jitter", 0.15)))
        spacing_jitter = float(np.clip(abs(spacing_jitter), 0.0, 0.95))
        if self.randomize_slot_spacing and spacing_jitter > 0.0:
            self.slot_spacing = float(np.random.uniform(
                self.slot_spacing * (1.0 - spacing_jitter),
                self.slot_spacing * (1.0 + spacing_jitter),
            ))
        # opt1: spawn blue curtains; opt2: make them sway (opt2 implies curtains present)
        opt1 = bool(
            c.get(
                "blue_curtains_enabled",
                c.get("opt1", c.get("curtain_enabled", self.BLUE_CURTAINS_ENABLED_DEFAULT)),
            )
        )
        opt2 = bool(
            c.get(
                "blue_curtain_dynamic_enabled",
                c.get("opt2", c.get("blue_curtain_dynamic", self.BLUE_CURTAIN_DYNAMIC_ENABLED_DEFAULT)),
            )
        )
        self.blue_curtain_dynamic_enabled = bool(opt2)
        self.blue_curtains_enabled = bool(opt1 or opt2)
        # opt1+2: center strip fixed; one of the two side strips sways.
        # opt2 alone: the whole curtain group sways together, with 3× the prior
        # (already 2×) lateral stroke — 6× nominal amp, then clamped to the plate.
        self.curtain_single_side_motion = bool(opt1 and opt2)
        if self.blue_curtain_dynamic_enabled and not self.curtain_single_side_motion:
            opt2_mult = float(c.get("curtain_amp_opt2_mult", self.CURTAIN_AMP_OPT2_MULT))
            self.curtain_amp = float(self.curtain_amp * max(1.0, opt2_mult))
        self.lift_off_only_after_place = bool(
            c.get("lift_off_only_after_place", self.LIFT_OFF_ONLY_AFTER_PLACE_DEFAULT)
        )
        self.post_place_lift_z = float(c.get("post_place_lift_z", self.POST_PLACE_LIFT_Z_DEFAULT))
        random_mirror = bool(c.get("random_mirror", self.RANDOM_MIRROR_DEFAULT))
        mirror_cfg = c.get("mirrored", None)
        if mirror_cfg is None:
            self.mirrored = bool(random_mirror and (np.random.rand() < 0.5))
        else:
            self.mirrored = bool(mirror_cfg)
        self.side = -1.0 if self.mirrored else 1.0

        # seed-randomized curtain phase / belt start / slot start
        self.curtain_phase = float(c.get("curtain_phase",
                                         np.random.uniform(0, 2 * np.pi)))
        slot_start_cfg = c.get("slot_start", None)
        self.belt_dir = float(np.random.choice([-1.0, 1.0]))

        # internal step counter drives BOTH dynamics (two-pass determinism)
        self._kin_step = 0
        self._curtain_hit = False
        self._attempt_active = False
        self._reset_metric_state()
        self._deposit_step = None
        self._slot_x_at_deposit = None
        self._belt_frozen_offset = None  # unused: yellow sticks never halt during place

        z0 = 0.74 + self.table_z_bias
        self.belt_y = float(c.get("belt_y", self.table_xy_bias[1] + self.BELT_CENTER_Y_OFFSET))
        self.curtain_y = float(c.get("curtain_y", self.CURTAIN_Y))

        # Fixed belt plate: sized so yellow sticks stay on the surface at both ±belt_range extremes.
        self.belt_motion_limit = float(max(0.0, self.belt_range))
        self.moving_component_half_size = (
            self.MOVING_COMPONENT_HALF_X,
            self.BELT_PLATE_HALF_Y * 0.85,
            self.MOVING_COMPONENT_HALF_Z,
        )
        # Outer peg at travel extreme: |x - belt_center| = spacing + limit + peg_half.
        plate_half_x = float(
            self.slot_spacing
            + self.belt_motion_limit
            + self.moving_component_half_size[0]
            + self.BELT_PLATE_EDGE_MARGIN
        )
        # Symmetric half-x first; lean-side extension is applied after corridor_x is known.
        self.belt_plate_half_size = (plate_half_x, self.BELT_PLATE_HALF_Y, 0.004)
        self._belt_plate_half_x_sym = plate_half_x
        self.belt_travel_min = -float(self.belt_motion_limit)
        self.belt_travel_max = float(self.belt_motion_limit)
        # slot_start is sampled after travel limits match the (possibly lean-extended) plate.

        # ----- the cup: active side near zone (right +x / left −x when mirrored) -----
        self.cup_id = 0
        # Slight pose jitter: keep |x| in the head-camera frame and reachable for both
        # arms; keep y clear of curtains at rest. Small yaw only (upright cup).
        x_lo = float(c.get("cup_x_abs_min", self.CUP_X_ABS_MIN))
        x_hi = float(c.get("cup_x_abs_max", self.CUP_X_ABS_MAX))
        if x_hi < x_lo:
            x_lo, x_hi = x_hi, x_lo
        self.cup_x = float(self.side * np.random.uniform(x_lo, x_hi))
        self.cup_y = float(np.random.uniform(-0.055, -0.028))
        cup_pose = rand_pose(
            xlim=[self.cup_x, self.cup_x], ylim=[self.cup_y, self.cup_y], zlim=[z0],
            qpos=self.CUP_UPRIGHT_QPOS, rotate_rand=True, rotate_lim=[0.0, 0.0, 0.18],
        )
        self.cup = create_actor(
            self, pose=cup_pose, modelname="021_cup",
            model_id=self.cup_id, convex=True, is_static=False,
        )
        self.cup.set_mass(0.15)
        self._set_cup_color(self.BLUE_CUP_RGBA)
        for comp in self.cup.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    comp.set_linear_damping(3.0)
                    comp.set_angular_damping(8.0)
                except Exception:
                    pass
        self.cup_height = float(self._actor_world_size(self.cup)[2])
        self.lift_z = float(self.cup_height + self.LIFT_CLEARANCE)

        # Corridor = cup column. Belt + curtains share it so the yellow gap and curtain opening
        # pass through a single reachable reach line.
        self.corridor_x = float(self.side * np.clip(abs(self.cup_x), x_lo, x_hi))
        self.curtain_center_x = self.corridor_x
        self.belt_center_x = float(c.get("belt_center_x", self.corridor_x))
        # Keep the inboard plate edge; add 30% of the symmetric half-x on the lean side
        # (right-arm / +x or left-arm / −x) so the belt is longer on the side it sits on.
        lean_extra = float(c.get("belt_lean_extend_frac", self.BELT_LEAN_EXTEND_FRAC))
        lean_extra = max(0.0, lean_extra) * float(self._belt_plate_half_x_sym)
        self.belt_plate_half_size = (
            float(self._belt_plate_half_x_sym) + 0.5 * lean_extra,
            self.BELT_PLATE_HALF_Y,
            0.004,
        )
        self._belt_plate_base_x = float(self.belt_center_x) + float(self.side) * 0.5 * lean_extra

        # ----- optional blue curtains (nominal height ~ cup; xy/z scaled ±size_jitter) -----
        size_jitter = float(c.get("curtain_size_jitter", self.CURTAIN_SIZE_JITTER_DEFAULT))
        size_jitter = float(np.clip(size_jitter, 0.0, 0.95))
        self.curtain_size_scale = float(
            np.random.uniform(1.0 - size_jitter, 1.0 + size_jitter)
        )
        s = float(self.curtain_size_scale)
        height_scale = float(c.get("curtain_height_scale", self.STRIP_HEIGHT_SCALE))
        strip_half_z = max(self.cup_height, 1e-3) * s * height_scale
        self.strip_half = (
            float(self.STRIP_HALF_XY[0]) * s,
            float(self.STRIP_HALF_XY[1]) * s,
            float(strip_half_z),
        )
        # Keep a 1 cm clear gap even when strip width is jittered this episode.
        clear_gap = float(c.get("curtain_clear_gap", self.STRIP_CLEAR_GAP))
        self.strip_spacing = float(2.0 * self.strip_half[0] + max(0.0, clear_gap))
        if self.blue_curtains_enabled:
            self.lift_z = float(max(self.lift_z, 2.0 * self.strip_half[2] + 0.03))
        self._strip_base_x = []
        self.curtain_strips = []
        n = self.N_STRIPS
        strip_offsets = [(i - (n - 1) / 2.0) * self.strip_spacing for i in range(n)]
        self.curtain_center_idx = int((n - 1) // 2)
        side_idxs = [i for i in range(n) if i != self.curtain_center_idx]
        moving_cfg = c.get("moving_curtain_idx", c.get("moving_curtain_side", None))
        if moving_cfg is None:
            self.moving_strip_idx = int(np.random.choice(side_idxs)) if side_idxs else 0
        else:
            if isinstance(moving_cfg, str):
                key = moving_cfg.strip().lower()
                if key in ("left", "l", "0"):
                    self.moving_strip_idx = int(side_idxs[0]) if side_idxs else 0
                elif key in ("right", "r"):
                    self.moving_strip_idx = int(side_idxs[-1]) if side_idxs else 0
                else:
                    self.moving_strip_idx = int(side_idxs[0]) if side_idxs else 0
            else:
                idx = int(moving_cfg)
                self.moving_strip_idx = idx if idx in side_idxs else int(side_idxs[0] if side_idxs else 0)
        curtain_half_span_x = (
            (max(abs(off) for off in strip_offsets) + self.strip_half[0]) if strip_offsets else 0.0
        )
        # Rest pose: slide the group along the belt; at the extremes the outer edge
        # is flush with the plate. Sway is then limited so it cannot run off the plate.
        plate_left = float(self._belt_plate_base_x - self.belt_plate_half_size[0])
        plate_right = float(self._belt_plate_base_x + self.belt_plate_half_size[0])
        x_lo = float(plate_left + curtain_half_span_x)
        x_hi = float(plate_right - curtain_half_span_x)
        cx_cfg = c.get("curtain_center_x", None)
        if x_hi < x_lo:
            self.curtain_center_x = float(0.5 * (plate_left + plate_right))
        elif cx_cfg is None:
            self.curtain_center_x = float(np.random.uniform(x_lo, x_hi))
        else:
            self.curtain_center_x = float(np.clip(float(cx_cfg), x_lo, x_hi))
        left_clear = max(0.0, (self.curtain_center_x - curtain_half_span_x) - plate_left)
        right_clear = max(0.0, plate_right - (self.curtain_center_x + curtain_half_span_x))
        group_limit = max(
            0.0,
            min(self.curtain_amp, left_clear, right_clear),
        )
        # Single-side sway must not drive a side strip through the static center strip.
        inward_clear = max(
            0.0,
            float(self.strip_spacing) - 2.0 * float(self.strip_half[0]) - 0.004,
        )
        if self.curtain_single_side_motion:
            self.curtain_motion_limit = max(0.0, min(self.curtain_amp, inward_clear))
        else:
            self.curtain_motion_limit = group_limit
        if self.blue_curtains_enabled:
            # Park curtains just in front of the belt so the near-zone grasp is clear.
            belt_front_y = self.belt_y - self.belt_plate_half_size[1] - self.strip_half[1] - 0.012
            self.curtain_y = float(belt_front_y)
            for i, off in enumerate(strip_offsets):
                bx = self.curtain_center_x + off
                self._strip_base_x.append(bx)
                strip = create_box(
                    scene=self,
                    pose=sapien.Pose([bx, self.curtain_y, z0 + self.strip_half[2]],
                                     [1, 0, 0, 0]),
                    half_size=self.strip_half,
                    color=(0.20, 0.45, 0.75),
                    name=f"curtain_strip_{i}",
                    is_static=False,
                )
                for comp in strip.actor.get_components():
                    if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                        comp.set_kinematic(True)
                self.curtain_strips.append(strip)

        # ----- belt + yellow sticks (plate FIXED; only yellow sticks slide ±belt_range) -----
        self.slot_z = z0 + self.BELT_Z_OFF
        self.n_slots = 3
        self.empty_slot_idx = 1
        self.belt_plate = create_box(
            scene=self,
            pose=sapien.Pose(
                [self._belt_plate_base_x, self.belt_y, z0 + self.belt_plate_half_size[2]],
                [1, 0, 0, 0],
            ),
            half_size=self.belt_plate_half_size,
            color=(0.35, 0.35, 0.40),
            name="belt_plate",
            is_static=False,
        )
        for comp in self.belt_plate.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_kinematic(True)

        self.belt_pegs = []
        self._peg_base_x = []
        for i in range(self.n_slots):
            if i == self.empty_slot_idx:
                continue
            bx = self.belt_center_x + (i - (self.n_slots - 1) / 2.0) * self.slot_spacing
            peg = create_box(
                scene=self,
                pose=sapien.Pose(
                    [bx, self.belt_y, z0 + self.BELT_Z_OFF + self.moving_component_half_size[2]],
                    [1, 0, 0, 0],
                ),
                half_size=self.moving_component_half_size,
                color=(0.8, 0.7, 0.3),
                name=f"belt_peg_{i}",
                is_static=False,
            )
            for comp in peg.actor.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_kinematic(True)
            self.belt_pegs.append(peg)
            self._peg_base_x.append(bx)

        self._slot_base_x = (
            self.belt_center_x
            + (self.empty_slot_idx - (self.n_slots - 1) / 2.0) * self.slot_spacing
        )
        # Travel so the left peg's left face and the right peg's right face meet the
        # plate edges (asymmetric when the belt is lean-extended).
        peg_hx = float(self.moving_component_half_size[0])
        plate_left = float(self._belt_plate_base_x - self.belt_plate_half_size[0])
        plate_right = float(self._belt_plate_base_x + self.belt_plate_half_size[0])
        left_rest = float(min(self._peg_base_x)) if self._peg_base_x else self.belt_center_x
        right_rest = float(max(self._peg_base_x)) if self._peg_base_x else self.belt_center_x
        self.belt_travel_min = float(plate_left - (left_rest - peg_hx))
        self.belt_travel_max = float(plate_right - (right_rest + peg_hx))
        if self.belt_travel_max < self.belt_travel_min:
            self.belt_travel_min, self.belt_travel_max = self.belt_travel_max, self.belt_travel_min
        self.belt_motion_limit = float(
            max(abs(self.belt_travel_min), abs(self.belt_travel_max))
        )
        if slot_start_cfg is None:
            self.slot_start = float(np.random.uniform(self.belt_travel_min, self.belt_travel_max))
        else:
            self.slot_start = float(np.clip(
                float(slot_start_cfg), self.belt_travel_min, self.belt_travel_max
            ))

        self.add_prohibit_area(self.cup, padding=0.05)
        # Do not add curtain strips as prohibit areas: they sit on the reach corridor and
        # block left-arm grasp/lift IK. Physical contact is still checked for success.
        # Soft padding only: a large belt prohibit blocks the place approach (esp. left arm).
        self.add_prohibit_area(self.belt_plate, padding=0.01)

        self._apply_kinematics(initial=True)

    # ------------------------------------------------------- dynamics model
    def _t(self):
        """Internal time (seconds-equivalent) from the step counter."""
        return self._kin_step * self.DT

    def _curtain_offset(self, t=None):
        if not getattr(self, "blue_curtain_dynamic_enabled", False):
            return 0.0
        if t is None:
            t = self._t()
        amp = float(getattr(self, "curtain_motion_limit", self.curtain_amp))
        if amp <= 1e-6:
            return 0.0
        return amp * np.sin(2 * np.pi * self.curtain_freq * t + self.curtain_phase)

    def _strip_offset(self, strip_idx, t=None):
        """Lateral offset for one curtain strip at time t."""
        if not getattr(self, "blue_curtain_dynamic_enabled", False):
            return 0.0
        if getattr(self, "curtain_single_side_motion", False):
            if int(strip_idx) != int(getattr(self, "moving_strip_idx", -1)):
                return 0.0
        return self._curtain_offset(t)

    def _belt_offset(self, t=None):
        """Triangle-wave travel of the yellow sticks; extremes flush with plate edges."""
        if t is None:
            t = self._t()
        lo = float(getattr(self, "belt_travel_min", -getattr(self, "belt_motion_limit", 0.0)))
        hi = float(getattr(self, "belt_travel_max", getattr(self, "belt_motion_limit", 0.0)))
        span = float(hi - lo)
        if span <= 1e-6:
            return float(np.clip(self.slot_start, lo, hi))
        phase = (self.belt_dir * self.belt_speed * t + self.slot_start)
        m = (phase - lo) % (2.0 * span)
        if m <= span:
            tri = lo + m
        else:
            tri = hi - (m - span)
        return float(np.clip(tri, lo, hi))

    def slot_x(self, t=None):
        return self._slot_base_x + self._belt_offset(t)

    def gap_center_x(self, t=None):
        """World x of the curtain gap used for timing.

        Group sway (opt2): the STRIP_GAP_IDX-th inter-strip gap, translated with the group.
        Single-side sway (opt1+2): the gap between the static center strip and the moving side.
        """
        if not self.blue_curtains_enabled or len(self._strip_base_x) < 2:
            return float(self.curtain_center_x)
        if getattr(self, "curtain_single_side_motion", False):
            i0 = int(getattr(self, "curtain_center_idx", self.STRIP_GAP_IDX))
            i1 = int(getattr(self, "moving_strip_idx", i0 + 1))
            if i0 > i1:
                i0, i1 = i1, i0
            if i1 >= len(self._strip_base_x):
                return float(self.curtain_center_x)
            rest = 0.5 * (self._strip_base_x[i0] + self._strip_base_x[i1])
            return rest + 0.5 * self._curtain_offset(t)
        if len(self._strip_base_x) <= self.STRIP_GAP_IDX + 1:
            return float(self.curtain_center_x)
        i = self.STRIP_GAP_IDX
        rest = 0.5 * (self._strip_base_x[i] + self._strip_base_x[i + 1])
        return rest + self._curtain_offset(t)

    def _set_body(self, actor, target_pose, initial=False):
        """Drive a kinematic body to target_pose. On the first call also hard-set the
        entity pose; thereafter use set_kinematic_target so PhysX interpolates each step."""
        if initial:
            actor.actor.set_pose(target_pose)
        for comp in actor.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_kinematic_target(target_pose)

    def _set_cup_color(self, rgba):
        if getattr(self, "cup", None) is None:
            return
        for comp in self.cup.actor.get_components():
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for shape in comp.render_shapes:
                    try:
                        shape.material.set_base_color(rgba)
                    except Exception:
                        pass

    def _actor_world_size(self, actor):
        if not hasattr(actor, "config") or actor.config is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        scale = actor.config.get("scale", [1.0, 1.0, 1.0])
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * 3
        extents = np.array(actor.config.get("extents", [0.0, 0.0, 0.0]), dtype=np.float64)
        local_half = 0.5 * extents * np.array(scale, dtype=np.float64)
        rot = actor.get_pose().to_transformation_matrix()[:3, :3]
        world_half = np.abs(rot) @ local_half
        return 2.0 * world_half

    def _apply_kinematics(self, initial=False):
        """Push step-driven targets onto kinematic bodies.

        Belt plate stays fixed. Only yellow sticks (and optional swaying curtains) translate.
        opt1+2 moves a single side strip; opt2 moves the whole curtain group.
        """
        if self.blue_curtains_enabled and self.blue_curtain_dynamic_enabled:
            for i, strip in enumerate(self.curtain_strips):
                p = strip.get_pose()
                co = self._strip_offset(i)
                self._set_body(strip,
                               sapien.Pose([self._strip_base_x[i] + co, p.p[1], p.p[2]], p.q),
                               initial=initial)
        if getattr(self, "belt_plate", None) is not None and initial:
            p = self.belt_plate.get_pose()
            self._set_body(
                self.belt_plate,
                sapien.Pose([self._belt_plate_base_x, p.p[1], p.p[2]], p.q),
                initial=True,
            )
        bo = self._belt_offset()
        for j, peg in enumerate(self.belt_pegs):
            p = peg.get_pose()
            self._set_body(peg,
                           sapien.Pose([self._peg_base_x[j] + bo, p.p[1], p.p[2]], p.q),
                           initial=initial)

    def _check_curtain_contact(self):
        if self._curtain_hit or not self._attempt_active or not self.blue_curtains_enabled:
            return
        cup_p = np.array(self.cup.get_pose().p, dtype=float)
        # Curtains are a front gate: ignore once past / when flying over strip tops.
        if cup_p[1] > float(self.curtain_y) + 0.02:
            return
        strip_top = float(0.74 + self.table_z_bias + 2.0 * self.strip_half[2])
        cup_bottom = float(cup_p[2] - 0.5 * self.cup_height)
        if cup_bottom > strip_top - 0.015:
            return
        # Geometric AABB vs cup cylinder (panels also have PhysX collision on).
        # Require penetration so bbox-inflated near-misses do not count as contact.
        cup_r = 0.5 * float(self._actor_world_size(self.cup)[0])
        cup_hz = 0.5 * float(self.cup_height)
        pen_x = float(self.CURTAIN_CONTACT_PENETRATION_X)
        pen_y = float(self.CURTAIN_CONTACT_PENETRATION_Y)
        pen_z = float(self.CURTAIN_CONTACT_PENETRATION_Z)
        for i, strip in enumerate(self.curtain_strips):
            sp = np.array(strip.get_pose().p, dtype=float)
            sh = np.array(self.strip_half, dtype=float)
            lim_x = max(0.003, cup_r + sh[0] - pen_x)
            lim_y = max(0.002, cup_r + sh[1] - pen_y)
            lim_z = max(0.003, cup_hz + sh[2] - pen_z)
            if (abs(cup_p[0] - sp[0]) < lim_x
                    and abs(cup_p[1] - sp[1]) < lim_y
                    and abs(cup_p[2] - sp[2]) < lim_z):
                self._curtain_hit = True
                import os
                if os.environ.get("CCS_DEBUG"):
                    print(f"[CCS] CURTAIN HIT strip {i} at step {self._kin_step} "
                          f"cup_z={cup_p[2]:.3f} cup_y={cup_p[1]:.3f}", flush=True)
                return

    def _update_kinematic_tasks(self):
        # base hook first (drives any built-in DOMINO motion), then our step-driven dynamics
        super()._update_kinematic_tasks()
        # this hook can fire (via base scene setup) before load_actors builds our scene
        if not hasattr(self, "_kin_step") or getattr(self, "belt_plate", None) is None:
            return
        self._kin_step += 1
        self._apply_kinematics()
        self._check_curtain_contact()
        self._track_cup_metrics()

    # ------------------------------------------------------------- helpers
    def _dwell(self, steps):
        """Advance both dynamics `steps` physics steps while recording frames."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _corridor_x(self):
        return float(getattr(self, "corridor_x", self.curtain_center_x))

    def _window_open(self, t=None):
        """True when the yellow gap is at the reach corridor (and curtain gap if swaying)."""
        corridor = self._corridor_x()
        slot = self.slot_x(t)
        reach_ok = abs(slot - corridor) < self.reach_tol
        if not self.blue_curtains_enabled:
            return reach_ok
        if not self.blue_curtain_dynamic_enabled:
            # Static curtains: gap sits near the corridor by construction; only timing the belt.
            return reach_ok
        gap = self.gap_center_x(t)
        gap_ok = abs(gap - corridor) < self.gap_tol
        return gap_ok and reach_ok

    def _steps_until_window(self, lead_steps=0, max_steps=2000):
        """Analytically scan future step counts for the first open window lead_steps ahead."""
        for s in range(int(max_steps)):
            t = (self._kin_step + s + lead_steps) * self.DT
            if self._window_open(t=t):
                return s
        return 0

    def _wait_for_window(self, lead_steps=0, max_steps=2000):
        """Dwell until the yellow(+curtain) window is open lead_steps in the future."""
        n = self._steps_until_window(lead_steps=lead_steps, max_steps=max_steps)
        self._dwell(n)
        return self._window_open(t=(self._kin_step + lead_steps) * self.DT)

    def _find_cup_grasp(self, arm_tag: ArmTag):
        # Left-arm grasps: contact 2 — other IDs leave a wrist pose
        # that cannot finish the forward belt reach.
        if arm_tag == "left":
            candidates = [
                (2, 0.10),
                (2, 0.08),
                (3, 0.10),
                (0, 0.10),
                (None, 0.10),
            ]
        else:
            candidates = [
                (0, 0.10),
                (1, 0.10),
                (0, 0.08),
                (1, 0.08),
                (None, 0.10),
                (2, 0.10),
                (3, 0.10),
            ]
        for contact_point_id, pre_grasp_dis in candidates:
            pre_g, g = self.choose_grasp_pose(
                self.cup,
                arm_tag=arm_tag,
                pre_dis=pre_grasp_dis,
                target_dis=0.0,
                contact_point_id=contact_point_id,
            )
            if pre_g is not None and g is not None:
                return contact_point_id, pre_grasp_dis
        return None, None

    def _nudge_x_to_slot(self, arm_tag: ArmTag, max_step=0.06, lead_steps=0):
        """World-x correction toward the (optionally predicted) yellow gap."""
        if lead_steps:
            target_x = float(self.slot_x(t=(self._kin_step + int(lead_steps)) * self.DT))
        else:
            target_x = float(self.slot_x())
        dx = float(target_x - self.cup.get_pose().p[0])
        if abs(dx) < 1e-3:
            return
        dx = float(np.clip(dx, -max_step, max_step))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        import os
        if os.environ.get("CCS_DEBUG"):
            cup_p = np.round(np.array(self.cup.get_pose().p), 3).tolist()
            print(
                f"[CCS] {tag}: plan={self.plan_success} hit={self._curtain_hit} "
                f"mirrored={getattr(self, 'mirrored', False)} "
                f"cup={cup_p} slot_x={self.slot_x():.3f} corridor={self._corridor_x():.3f} "
                f"belt_speed={self.belt_speed:.3f} curtain_scale={getattr(self, 'curtain_size_scale', 1.0):.2f}",
                flush=True,
            )

    def play_once(self):
        arm_tag = ArmTag("left" if self.mirrored else "right")

        # 1) Grasp the cup and lift clear of the (cup-tall) blue curtains.
        grasp_contact_id, pre_grasp_dis = self._find_cup_grasp(arm_tag)
        if pre_grasp_dis is None:
            self.plan_success = False
            self.info["info"] = {"{A}": f"021_cup/base{self.cup_id}", "{a}": str(arm_tag)}
            return self.info
        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag=arm_tag,
                pre_grasp_dis=pre_grasp_dis,
                gripper_pos=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        self._dbg("after_grasp")
        if not self.plan_success:
            self.info["info"] = {"{A}": f"021_cup/base{self.cup_id}", "{a}": str(arm_tag)}
            return self.info
        self._dwell(6)
        # World-frame lift. With 1 cm slits the cup cannot thread the curtains, so
        # raise until the cup bottom is above the strip tops before any +Y.
        half_lift = 0.5 * self.lift_z
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=half_lift))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.lift_z - half_lift))
        table_z = float(0.74 + self.table_z_bias)
        min_clear_z = float(table_z + self.cup_height + 0.02)
        over_top_z = min_clear_z
        if self.blue_curtains_enabled:
            strip_top = float(table_z + 2.0 * self.strip_half[2])
            over_top_z = float(strip_top + 0.5 * self.cup_height + 0.02)
            min_clear_z = float(max(min_clear_z, over_top_z))
        for _ in range(5):
            cup_z = float(self.cup.get_pose().p[2])
            if cup_z >= min_clear_z - 0.005:
                break
            dz = float(min(0.045, min_clear_z - cup_z))
            if dz <= 1e-3:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz))
        cup_z = float(self.cup.get_pose().p[2])
        if cup_z < table_z + self.cup_height + 0.02:
            # Grasp likely never attached; abort rather than plow through curtains at table height.
            self.plan_success = False
            self._dbg("after_lift_fail")
            self.info["info"] = {"{A}": f"021_cup/base{self.cup_id}", "{a}": str(arm_tag)}
            return self.info
        self._dbg("after_lift")

        # Align to the shared cup/belt/curtain corridor before the timed approach.
        align_dx = float(self._corridor_x() - self.cup.get_pose().p[0])
        if abs(align_dx) > 1e-3:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=align_dx))
            self._dbg("after_xalign")
        self._attempt_active = True

        # 2) Observe dynamics, then wait until the yellow gap (and swaying curtain gap) will
        #    sit on the corridor when the forward reach finishes.
        self._dwell(10)
        self._wait_for_window(lead_steps=self.REACH_STEPS_EST, max_steps=2500)
        self._reach_step = self._kin_step

        # 3) Reach to the belt row in short Y steps, holding height in the same displacement.
        #    Recenter X toward the corridor between steps (separate move) so a failed X nudge
        #    does not block Y progress. Belt never freezes.
        z_keep = float(max(
            self.cup.get_pose().p[2],
            over_top_z,
            self.slot_z + self.cup_height + 0.05,
        ))
        place_y = float(self.belt_y)
        for _ in range(10):
            if not self.plan_success:
                break
            cup_p = np.array(self.cup.get_pose().p, dtype=float)
            dy = float(place_y - cup_p[1])
            if abs(dy) < 0.012:
                break
            step_y = float(np.clip(dy, -0.03, 0.03))
            step_z = float(max(0.0, z_keep - cup_p[2]))
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=step_y, z=step_z))
            if not self.plan_success:
                break
            # Soft corridor recenter (ignore failure by not coupling into the Y command).
            dx = float(self._corridor_x() - self.cup.get_pose().p[0])
            if abs(dx) > 0.015:
                saved = self.plan_success
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm_tag, x=float(np.clip(dx, -0.02, 0.02))
                    )
                )
                if not self.plan_success and saved:
                    # X nudge failed; keep going with Y from the last good state.
                    self.plan_success = True
        self._dbg("after_reach")
        if not self.plan_success:
            self.info["info"] = {
                "{A}": f"021_cup/base{self.cup_id}",
                "{a}": str(arm_tag),
                "{flip}": "mirrored" if self.mirrored else "default",
            }
            return self.info

        # Hover until the predicted gap (short lead) is under the gripper, tracking in X.
        for _ in range(120):
            pred = float(self.slot_x(t=(self._kin_step + 30) * self.DT))
            err = abs(pred - float(self.cup.get_pose().p[0]))
            if err < self.reach_tol * 0.6:
                break
            self._dwell(3)
        self._nudge_x_to_slot(arm_tag, max_step=0.06, lead_steps=35)
        self._dbg("after_hover_align")

        # Staged descent while tracking the still-moving yellow gap (aim ahead of the motion).
        cup_z = float(self.cup.get_pose().p[2])
        target_z = float(self.slot_z) + 0.008
        remaining = cup_z - target_z
        if remaining > 0.02:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.45 * remaining))
            self._nudge_x_to_slot(arm_tag, max_step=0.05, lead_steps=35)
            self._dbg("after_lower_mid")
            cup_z = float(self.cup.get_pose().p[2])
            remaining = cup_z - target_z
        if remaining > 0.005:
            self._nudge_x_to_slot(arm_tag, max_step=0.045, lead_steps=28)
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-remaining))
            self._nudge_x_to_slot(arm_tag, max_step=0.035, lead_steps=15)
            self._dbg("after_lower")

        self._deposit_step = self._kin_step
        self._slot_x_at_deposit = self.slot_x()
        self.move(self.open_gripper(arm_tag))

        # Settle while yellow sticks keep sliding on the fixed plate.
        self._dwell(40)

        # 4) Lift straight up in world frame (arm-axis retract can drag a still-contacting cup).
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.post_place_lift_z))
        # Retract IK may fail after a good place; don't discard a seated cup for that.
        if not self.plan_success and self._deposit_step is not None:
            self.plan_success = True
        self._dbg("after_retract")
        if not self.lift_off_only_after_place:
            self.move(self.back_to_origin(arm_tag))
            if not self.plan_success and self._deposit_step is not None:
                self.plan_success = True

        self.info["info"] = {
            "{A}": f"021_cup/base{self.cup_id}",
            "{a}": str(arm_tag),
            "{flip}": "mirrored" if self.mirrored else "default",
        }
        return self.info

    def play_once_bk(self):
        return self.play_once()

    # ------------------------------------------------------------- metric
    def _cup_between_yellow_tools(self):
        """True if the cup center lies strictly between the two yellow sticks in x."""
        cup_x = float(self.cup.get_functional_point(0, "pose").p[0])
        peg_xs = [float(p.get_pose().p[0]) for p in getattr(self, "belt_pegs", [])]
        if len(peg_xs) < 2:
            return abs(cup_x - self.slot_x()) < self.placement_tol
        lo, hi = min(peg_xs), max(peg_xs)
        # Stay clear of the peg bodies themselves (half-width + small margin).
        margin = float(self.moving_component_half_size[0]) + 0.005
        return bool((lo + margin) < cup_x < (hi - margin))

    def _cup_held(self):
        """True while the cup is still attached to the hand (gripper contact).

        Gripper-open is not required for release: loss of finger contact (slip-off)
        counts as detached. Interactive play uses G to open/close; kinematic
        teleop holds are no longer used.
        """
        if getattr(self, "cup", None) is None:
            return False
        for comp in self.cup.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    if bool(comp.kinematic):
                        return True
                except Exception:
                    pass
                break
        try:
            return len(self.get_gripper_actor_contact_position(self.cup.get_name())) > 0
        except Exception:
            return False

    def _center_offset(self):
        """Distance from cup to the yellow gap, with a looser y band on the belt plate."""
        cup_p = np.array(self.cup.get_functional_point(0, "pose").p[:2])
        dx = float(cup_p[0] - self.slot_x())
        # Accept any y on the plate (plus small margin); score is driven by gap alignment in x.
        half_y = float(self.belt_plate_half_size[1]) + 0.02
        dy = float(cup_p[1] - self.belt_y)
        dy = 0.0 if abs(dy) <= half_y else (abs(dy) - half_y)
        return float(np.hypot(dx, dy))

    def placement_score(self):
        if self._curtain_hit:
            return 0.0
        if not self._cup_between_yellow_tools():
            return 0.0
        off = self._center_offset()
        return float(np.clip(1.0 - off / max(self.placement_tol, 1e-6), 0.0, 1.0))

    # ------------------------------------------------- experiment metrics
    # Frames of no gripper/cup contact before the release counts as real (finger
    # contact flickers by a frame or two as the jaws part).
    RELEASE_DEBOUNCE_STEPS = 4

    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_grasp_step = None       # cup first in hand
        self._metric_window_step = None      # first open window while holding the cup
        self._metric_release_step = None     # cup let go
        self._metric_release_offset = None   # gap offset at let-go, normalised
        self._metric_held_seen = False
        self._metric_free_frames = 0
        self._metric_last_held_offset = None

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_cup_metrics(self):
        """Per-step latches: grasp, window opening, release + the offset at release."""
        try:
            held = bool(self._cup_held())
            if held:
                if self._metric_grasp_step is None:
                    self._metric_grasp_step = self._metric_step()
                self._metric_held_seen = True
                self._metric_free_frames = 0
                self._metric_last_held_offset = float(self._center_offset())
                if self._metric_window_step is None and self._window_open():
                    self._metric_window_step = self._metric_step()
                # Picked up again after a fumble — the earlier let-go was not the
                # decisive one, so re-arm and let the FINAL release win.
                self._metric_release_step = None
                self._metric_release_offset = None
                return
            if not self._metric_held_seen or self._metric_release_step is not None:
                return
            self._metric_free_frames += 1
            if self._metric_free_frames >= self.RELEASE_DEBOUNCE_STEPS:
                # Back-date to the first contact-free frame — that is when the cup
                # actually left the hand, before it drifted.
                self._metric_release_step = self._metric_step() - self._metric_free_frames + 1
                off = self._metric_last_held_offset
                self._metric_release_offset = (
                    None if off is None
                    else float(off) / max(float(self.placement_tol), 1e-9)
                )
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `release_latency_steps` — steps from the yellow gap first arriving at the
        reach corridor (with the cup already in hand) until the cup is let go. This is
        the operator's commit window against the moving belt / swaying curtain.
        extra2 `gap_offset_norm` — cup-to-gap-centre distance at the instant of release,
        divided by the placement tolerance. LOWER is better; <= 1.0 lands in the slot.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        w = getattr(self, "_metric_window_step", None)
        r = getattr(self, "_metric_release_step", None)
        lat = None if (w is None or r is None) else int(r) - int(w)
        out["release_latency_steps"] = lat
        out["release_latency_s"] = None if lat is None else round(lat * dt, 4)

        g = getattr(self, "_metric_grasp_step", None)
        carry = None if (g is None or r is None) else max(int(r) - int(g), 0)
        out["carry_latency_steps"] = carry

        off = getattr(self, "_metric_release_offset", None)
        out["gap_offset_norm"] = None if off is None else round(float(off), 4)
        try:
            out["placement_score"] = round(float(self.placement_score()), 4)
        except Exception:
            out["placement_score"] = None
        out["curtain_hit"] = bool(getattr(self, "_curtain_hit", False))
        return out

    # ------------------------------------------------------------- success
    def check_success(self):
        """Success = cup detached from the hand and seated between the yellow tools.

        Detachment is finger-contact / teleop-hold loss (slip-off counts); the gripper
        does not need to open. Landing elsewhere after detach is failure.
        """
        if self._curtain_hit:
            return False
        held = bool(self._cup_held())
        cup_p = self.cup.get_functional_point(0, "pose").p
        z_ok = (self.slot_z - 0.04) < float(cup_p[2]) < (self.slot_z + 0.12)
        # On / above the belt plate in y.
        half_y = float(self.belt_plate_half_size[1]) + 0.03
        y_ok = abs(float(cup_p[1]) - self.belt_y) <= half_y
        between = self._cup_between_yellow_tools()
        import os
        if os.environ.get("CCS_DEBUG"):
            print(
                f"[CCS] check: cup_xyz={np.round(np.array(cup_p),3).tolist()} "
                f"slot_x={self.slot_x():.3f} between={between} y_ok={y_ok} z_ok={z_ok} "
                f"held={held} hit={self._curtain_hit} "
                f"score={self.placement_score():.2f}",
                flush=True,
            )
        # Still pinched / teleop-held: do not score a place yet.
        if held:
            return False
        return bool(z_ok and y_ok and between and (not self._curtain_hit))

    # ------------------------------------------------------------- obs
    def get_obs(self):
        obs = super().get_obs()
        obs["curtain_slot"] = {
            "curtain_phase": float(2 * np.pi * self.curtain_freq * self._t() + self.curtain_phase),
            "curtain_offset": float(self._curtain_offset()),
            "gap_center_x": float(self.gap_center_x()),
            "slot_x": float(self.slot_x()),
            "belt_motion_limit": float(getattr(self, "belt_motion_limit", 0.0)),
            "belt_speed": float(self.belt_speed),
            "curtain_size_scale": float(getattr(self, "curtain_size_scale", 1.0)),
            "window_open": bool(self._window_open()),
            "curtain_hit": bool(self._curtain_hit),
            "placement_score": float(self.placement_score()),
            "blue_curtains_enabled": bool(self.blue_curtains_enabled),
            "blue_curtain_dynamic_enabled": bool(getattr(self, "blue_curtain_dynamic_enabled", False)),
            "curtain_single_side_motion": bool(getattr(self, "curtain_single_side_motion", False)),
            "moving_strip_idx": int(getattr(self, "moving_strip_idx", -1)),
            "moving_component_spacing": float(self.slot_spacing),
            "lift_off_only_after_place": bool(self.lift_off_only_after_place),
            "mirrored": bool(getattr(self, "mirrored", False)),
        }
        return obs
