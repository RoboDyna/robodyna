from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
import transforms3d as t3d


class sort_apples_belt(Base_Task):
    """Sort a fixed number (4) of red/green apples streaming down a conveyor into the
    color-matched basket by HOLDING the matching side button to aim a pivoting diverter plank.

    Apples spawn at the far end with spacing and ride the belt as a continuous STREAM (2-3 on the
    belt at once; a new one spawns once the previous has travelled far enough) at a steady,
    per-episode-random speed. Apples SIT on the belt (no rolling) and NEVER freeze in place —
    near the fork they become dynamic and keep being pushed into the plank. When the plank is
    diverted left/right, apples already at the fork leave with that divert (they hit the plank
    and slide along its edge into the basket). Two color-coded baskets sit at the fork. Buttons
    are hold-to-actuate; releasing returns the plank to rest:

      - Hold LEFT  → diagonal divert toward the left basket
      - Hold RIGHT → diagonal divert toward the right basket
      - Hold BOTH  (Opt 2) → halves part about their centers (dump into the garbage bin)
      - Release    → plank returns to the default horizontal (x-axis) pose

    Task options (``task_args.sort_apples_belt``; independent toggles):
      - Default — alternating red/green: ``color_mode: alternating``
      - Option 1 — independent random red/green: ``color_mode: random``
          CLI: ``--task-arg color_mode=random`` or legacy ``--option 1``.
      - Option 2 — rotten apple + dump hatch: ``rotten_prob`` > 0 (default 0.3 via ``--option 2``)
          Exactly one random apple is always brown (rotten). Independently, with probability
          ``rotten_prob``, a second apple is also marked rotten. Rotten fruit use the same
          belt→PhysX divert dynamics as colored fruit (plank collision, left/right slide).
          A garbage bin sits at the lowest-y end of the belt; holding BOTH buttons opens the
          dump passage so rotten fruit can ride off into the bin (success). Landing in a
          color basket is failure. Releasing closes the plank back to horizontal.
          CLI: ``--task-arg rotten_prob=0.3`` or legacy ``--option 2``.

    The diverter / hatch moves ONLY while a gripper PHYSICALLY holds a side button (EE-to-button
    proximity, detected every step in `_update_kinematic_tasks`) — routing is driven by the actual
    press, so the task is policy-evaluable. The stream + diverter run autonomously (started in
    `setup_demo`), so the scene is live during a policy rollout, not only during the expert's
    `play_once`.

    Metric: sorting_accuracy = correct / n_apples (success = ALL apples correctly sorted), scored
    from where each apple PHYSICALLY settles, plus macro-F1 over present classes. Opt 2: a rotten
    apple that settles in either color basket is an automatic failure (must go to the dump/bin).
    """

    # ---- class-default task params (override via task_args.sort_apples_belt) ----
    N_APPLES_MIN_DEFAULT = 4
    N_APPLES_MAX_DEFAULT = 4              # fixed episode size (all options)
    # Upper belt speed; lower = 30% below upper. Sample U[lower, upper] per ep.
    BELT_SPEED_MAX_DEFAULT = 0.0016    # m per advance tick (−20% vs 0.0020)
    BELT_SPEED_MIN_DEFAULT = BELT_SPEED_MAX_DEFAULT * 0.7  # 0.00112; 30% below upper
    ADVANCE_EVERY_DEFAULT = 4          # physics steps between belt advances
    GATE_DEFAULT_LEFT_DEFAULT = True   # gate direction before the first press
    OBSERVE_DIST = 0.10
    GATE_TILT = 0.425                  # blade yaw (rad); half of prior 0.85
    DUMP_OPEN_ANGLE = np.pi / 2        # each half yaws ±90° about its own center to open the middle
    SPAWN_GAP = 0.36                   # (+20% vs 0.30); more time between successive apples
    ACCUM_SPACING = 0.045              # center-to-center gap when apples queue at the plank
    DECISION_DY = 0.15
    DUMP_DECISION_EXTRA = 0.32         # start dual press early; hatch opens only on contact
    PRESS_XY = 0.06
    PRESS_DZ = 0.18                    # proximity for "near button" (side divert)
    # EE frame sits above the TCP; CuRobo also undershoots deep −Z presses. 12 cm
    # still requires a clear downward press (hover is ~20 cm) without being so tight
    # that a successful contact never registers.
    PRESS_DZ_ACTIVE = 0.12
    BUTTON_DX = 0.25
    BUTTON_Y_DEFAULT = -0.16
    COLOR_MODE_DEFAULT = "alternating"  # Default; Opt 1 = "random"
    ROTTEN_PROB_DEFAULT = 0.0           # Default off; Opt 2 sets 0.3 (extra rotten chance)
    ROTTEN_PROB_WHEN_OPT2 = 0.3         # P(second rotten); first is always present when enabled

    RED = [0.85, 0.10, 0.10]
    GREEN = [0.15, 0.70, 0.18]
    BROWN = [0.42, 0.26, 0.10]          # rotten apple
    PLANK_COLOR = [0.92, 0.88, 0.78]    # white / light wood

    # color ids: 0=red, 1=green, 2=rotten
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_ROTTEN = 2

    # belt geometry (table-frame); belt runs along y, fork near the robot (low y).
    # BELT_Y_FAR must leave enough run for SPAWN_GAP before the queue, otherwise the next
    # apple never spawns while one is waiting at the plank (no accumulation).
    BELT_X_DEFAULT = 0.0
    BELT_HALF_W = 0.10
    BELT_Y_FAR = 0.42
    BELT_Y_FORK = -0.03                 # was -0.06; plank +3 cm toward +Y (upstream)
    BELT_Y_END = -0.12                  # physical near end of the belt slab (past the fork)
    # Dump release must be PAST the slab so PhysX can fall (cannot fall through the belt).
    BELT_Y_DUMP = -0.16
    BELT_TOP_DZ = 0.025
    BELT_LIFT = 0.135
    APPLE_SCALE = 0.5
    BASKET_SCALE = 0.65                 # keep plasticbox lip below belt so fruit can fall in
    BASKET_MODEL = "062_plasticbox"
    BASKET_INSTANCE_IDS = (7, 8, 9, 10)
    # Upright open-top plasticbox (same as place_cans_plasticbox).
    BASKET_Q = [0.5, 0.5, 0.5, 0.5]
    APPLE_R = 0.017
    PHYSICS_RELEASE_DY = 0.055         # front of the queue, just upstream of the plank
    BASKET_CATCH_R = 0.11
    GARBAGE_CATCH_R = 0.16
    # Apple must clear this |x| (past the belt rim) before a basket catch can fire —
    # prevents early snap while still sliding along the plank (v22).
    BASKET_CATCH_MIN_ABS_X = 0.115
    # Max height above table origin for a basket catch. Must stay below belt_surf
    # (~_z0+0.135) so tip/mid-air freezes never return; high enough for plasticbox rim.
    BASKET_CATCH_MAX_DZ = 0.090
    N_SLATS = 4
    # 011_dustbin default scale 0.2 → ~0.74 m tall; scale_mult is 1.5× the prior 0.16.
    DUSTBIN_SCALE_MULT = 0.24
    # Fallback rim height above pose origin if model_data is missing (~actual ~0.176).
    DUSTBIN_TARGET_H = 0.15
    # Rim/top clearance under `_belt_surf` so the bin does not poke through the belt.
    DUSTBIN_RIM_BELOW_BELT = 0.02
    # Dustbin pose Y offset below BELT_Y_END so the belt-facing rim sits flush with
    # the belt exit (computed from 011_dustbin extents at DUSTBIN_SCALE_MULT).
    # Fallback used only if model_data is missing.
    DUSTBIN_DY = 0.052
    # Episode budget (sim steps via _step_ctr / _step_record); 15000 @ 250 Hz ≈ 60 s.
    MAX_EPISODE_STEPS = 15000
    # Cap per button-hold wait so a stuck apple can reseat/retry instead of burning
    # the whole episode budget waiting for a deposit that never comes.
    HOLD_WAIT_MAX = 900

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("sort_apples_belt", {}))
        self._apply_legacy_option()
        # Ensure stream is OFF during base settle / gripper open (re-used task instances
        # may still have _stream_active=True from a prior episode).
        self._stream_active = False
        self._reset_metric_state()
        super()._init_task_env_(**kwags)
        # activate the stream AFTER setup (the base settles the scene during _init_task_env_ with the
        # stream OFF, so apples don't deposit before the episode). Set here -- NOT in play_once -- so
        # the belt/apples also run during a policy-evaluation rollout (where play_once is not called).
        self._stream_active = True

    def _apply_legacy_option(self):
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / random / color_mode → Opt 1 color_mode=random
        2 / rotten / dump → Opt 2 rotten_prob=0.3 (always ≥1 rotten; P=0.3 for a second)
        """
        legacy = self._cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "color_mode_random",
            2: "rotten",
            "1": "color_mode_random",
            "2": "rotten",
            "random": "color_mode_random",
            "color_mode": "color_mode_random",
            "color_mode_random": "color_mode_random",
            "rotten": "rotten",
            "dump": "rotten",
            "rotten_prob": "rotten",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        if key == "color_mode_random":
            if "color_mode" not in self._cfg:
                self._cfg["color_mode"] = "random"
        elif key == "rotten":
            if "rotten_prob" not in self._cfg:
                self._cfg["rotten_prob"] = self.ROTTEN_PROB_WHEN_OPT2
        else:
            raise ValueError(
                "sort_apples_belt option must be 1/color_mode=random or "
                "2/rotten_prob (or set color_mode / rotten_prob directly)"
            )

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "color_mode", self.COLOR_MODE_DEFAULT) == "random":
            parts.append("option 1")
        if float(getattr(self, "rotten_prob", 0.0)) > 0.0:
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        n_min = int(cfg.get("n_apples_min", self.N_APPLES_MIN_DEFAULT))
        n_max = int(cfg.get("n_apples_max", self.N_APPLES_MAX_DEFAULT))
        self.n_apples = int(np.random.randint(n_min, n_max + 1))
        # Per-episode speed: uniform in [0.7×upper, upper] (lower is 30% below current/upper speed).
        s_max = float(cfg.get("belt_speed_max", self.BELT_SPEED_MAX_DEFAULT))
        s_min = float(cfg.get("belt_speed_min", s_max * 0.7))
        self.belt_speed = float(np.random.uniform(s_min, s_max)) * float(cfg.get("belt_speed_scale", 1.0))
        self.SPAWN_GAP = float(cfg.get("spawn_gap", self.SPAWN_GAP))
        self.advance_every = int(cfg.get("advance_every", self.ADVANCE_EVERY_DEFAULT))
        self.gate_default_left = bool(cfg.get("gate_default_left", self.GATE_DEFAULT_LEFT_DEFAULT))
        # Gate / divert-plank hinge center along belt Y (class default BELT_Y_FORK).
        self.BELT_Y_FORK = float(cfg.get(
            "gate_center_y", cfg.get("belt_y_fork", self.BELT_Y_FORK)))

        self.color_mode = str(cfg.get("color_mode", self.COLOR_MODE_DEFAULT)).lower()
        if self.color_mode not in ("alternating", "random"):
            self.color_mode = self.COLOR_MODE_DEFAULT
        self.rotten_prob = float(cfg.get("rotten_prob", self.ROTTEN_PROB_DEFAULT))
        self.rotten_prob = float(np.clip(self.rotten_prob, 0.0, 1.0))
        self.rotten_enabled = self.rotten_prob > 0.0

        self._z0 = 0.74 + self.table_z_bias
        belt_top = self._z0 + self.BELT_LIFT
        self._belt_surf = belt_top

        # ---- randomization: apple colors + which side is red vs green ----
        self.apple_colors = self._sample_apple_colors(self.n_apples)
        self.green_on_left = bool(np.random.rand() < 0.5)

        self._basket_x = {"left": -0.18, "right": 0.18}
        # Plastic boxes stay at prior Y (old fork -0.01 + 0.05); do not follow plank +3 cm.
        self._basket_y = -0.02
        if self.green_on_left:
            self._color_side = {self.COLOR_GREEN: "left", self.COLOR_RED: "right"}
        else:
            self._color_side = {self.COLOR_GREEN: "right", self.COLOR_RED: "left"}
        self._color_side[self.COLOR_ROTTEN] = "dump"
        self._side_color = {v: k for k, v in self._color_side.items() if v in ("left", "right")}

        # ---- belt: a long static slab along y, surface at belt_top (elevated on legs) ----
        # Opt 2 extends the slab past the fork so a dump apple can ride off the near end into
        # the garbage bin (bin itself sits JUST past the slab — not under it).
        belt_near = self.BELT_Y_END if self.rotten_enabled else self.BELT_Y_FORK
        belt_len = self.BELT_Y_FAR - belt_near + 0.04
        belt_cy = (self.BELT_Y_FAR + belt_near) / 2.0
        belt_slab_cz = belt_top - self.BELT_TOP_DZ / 2.0
        self._belt_near = float(belt_near)
        self.belt = create_box(
            self.scene,
            sapien.Pose([self.BELT_X_DEFAULT, belt_cy, belt_slab_cz]),
            half_size=[self.BELT_HALF_W, belt_len / 2.0, self.BELT_TOP_DZ / 2.0],
            color=[0.20, 0.20, 0.22], is_static=True, name="belt",
        )
        leg_top = belt_top - self.BELT_TOP_DZ
        leg_h = leg_top - self._z0
        for lx in (-self.BELT_HALF_W * 0.45, self.BELT_HALF_W * 0.45):
            for ly in (belt_cy - belt_len / 2.0 + 0.04, belt_cy + belt_len / 2.0 - 0.04):
                create_box(
                    self.scene, sapien.Pose([lx, ly, self._z0 + leg_h / 2.0]),
                    half_size=[0.012, 0.012, leg_h / 2.0], color=[0.3, 0.3, 0.33],
                    is_static=True, name="belt_leg",
                )

        # ---- moving belt slats ----
        self._slat_spacing = belt_len / self.N_SLATS
        self._slat_near = belt_cy - belt_len / 2.0
        self.slats = []
        self._slat_ys = []
        for k in range(self.N_SLATS):
            sy = self._slat_near + k * self._slat_spacing
            sl = create_box(
                self.scene, sapien.Pose([self.BELT_X_DEFAULT, sy, belt_top + 0.001]),
                half_size=[self.BELT_HALF_W * 0.92, 0.006, 0.003],
                color=[0.10, 0.10, 0.12], is_static=True, name=f"slat_{k}",
            )
            self.slats.append(sl)
            self._slat_ys.append(sy)

        # ---- two-piece white wooden diverter plank ----
        # Default rest: both halves HORIZONTAL along the x-axis. Hold left/right to divert
        # diagonally; hold both (Opt 2) to part about each half's center (dump). Release → rest.
        self.gate_left = self.gate_default_left
        self._dump_open = False
        self._gate_mode = "rest"          # rest | divert | dump
        self._dump_anim_phase = "rest"    # rest | divert | part
        self._blade_half_len = 0.075
        self._blade_half_w = 0.007      # half of prior 0.014 (plank thickness)
        self._blade_half_h = 0.028
        # Bottom of plank flush with belt top so apples cannot slip under.
        self._plank_cz = belt_top + self._blade_half_h
        self.gate_pieces = {}
        self._gate_comps = {}
        plank_mat = sapien.physx.PhysxMaterial(
            static_friction=0.35, dynamic_friction=0.28, restitution=0.0)
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            # Kinematic (not static): pose updates via set_kinematic_target so PhysX
            # keeps continuous collision with dynamic apples (static+set_pose tunnels).
            piece = create_box(
                self.scene,
                sapien.Pose([sgn * self._blade_half_len, self.BELT_Y_FORK, self._plank_cz]),
                half_size=[self._blade_half_len, self._blade_half_w, self._blade_half_h],
                color=self.PLANK_COLOR, is_static=False, name=f"diverter_plank_{side}",
            )
            comp = None
            for cc in piece.actor.get_components():
                if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                    comp = cc
                    break
            if comp is not None:
                try:
                    comp.set_mass(1.0)
                except Exception:
                    pass
                comp.set_kinematic(True)
                for sh in comp.get_collision_shapes():
                    try:
                        sh.set_physical_material(plank_mat)
                    except Exception:
                        pass
            self.gate_pieces[side] = piece
            self._gate_comps[side] = comp
        # Start at horizontal rest (x-axis).
        self._gate_yaw = 0.0
        self._gate_yaw_target = 0.0
        self._gate_yaw_left = 0.0
        self._gate_yaw_right = 0.0
        self._gate_yaw_left_target = 0.0
        self._gate_yaw_right_target = 0.0
        self._gate_rate = 0.035
        self._dump_gate_rate = 0.08
        self._apply_gate_pose()

        # ---- TWO side buttons ----
        btn_y = float(cfg.get("button_y", self.BUTTON_Y_DEFAULT))
        self._button_xy = {"left": (-self.BUTTON_DX, btn_y), "right": (self.BUTTON_DX, btn_y)}
        btn_half = [0.022, 0.022, 0.015]
        self._button_top_z = self._z0 + 2.0 * float(btn_half[2])
        self.buttons = {}
        self._button_home = {}
        self._button_held = {"left": False, "right": False}
        self._reactive_buttons = None
        for side, (bx, by) in self._button_xy.items():
            add_key_base_border(
                self.scene,
                float(bx),
                float(by),
                float(self._z0),
                btn_half,
                color=[0.10, 0.10, 0.12],
                name_prefix=f"button_base_{side}",
            )
            btn_color = self.RED if self._side_color[side] == self.COLOR_RED else self.GREEN
            home = sapien.Pose([bx, by, self._z0 + float(btn_half[2])])
            self._button_home[side] = home
            self.buttons[side] = create_box(
                self.scene, home,
                half_size=list(btn_half), color=btn_color,
                is_static=True, name=f"button_{side}",
            )
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[self.buttons[s] for s in ("left", "right")],
            home_poses=[self._button_home[s] for s in ("left", "right")],
            max_depth=float(btn_half[2]),
            ids=["left", "right"],
            xy_tol=float(self.PRESS_XY),
            # Only engage once the tip is near the keycap (not ~5 cm above).
            force_engage_slack=0.012,
            trigger_depth_frac=0.45,
            # WSG fingers sit ~16 cm below EE (AABB); default 0.12 lets Q
            # drive fingertips through these short table buttons.
            ee_to_tcp=0.16,
        )
        self._reactive_buttons.set_tops_z([self._button_top_z, self._button_top_z])

        # ---- receptacles at the fork (062_plasticbox) ----
        self.basket_id = int(np.random.choice(self.BASKET_INSTANCE_IDS))
        self.baskets = {}
        self._ref_apples = []
        for side in ("left", "right"):
            bx = self._basket_x[side]
            basket = create_actor(
                self, pose=sapien.Pose([bx, self._basket_y, self._z0],
                                       list(self.BASKET_Q)),
                modelname=self.BASKET_MODEL, model_id=self.basket_id,
                convex=True, is_static=True, scale_mult=self.BASKET_SCALE,
            )
            self.baskets[side] = basket
            color = self.RED if self._side_color[side] == self.COLOR_RED else self.GREEN
            for k in range(2):
                ra = create_actor(
                    self, pose=sapien.Pose([bx + (k - 0.5) * 0.03, self._basket_y,
                                            self._z0 + 0.03]),
                    modelname="220_apple_plain", model_id=0, convex=True, is_static=True,
                    scale_mult=self.APPLE_SCALE,
                )
                self._recolor(ra, color)
                self._ref_apples.append(ra)

        # ---- Opt 2: 011_dustbin — belt-facing rim flush with BELT_Y_END, top under belt ----
        self.garbage_bin = None
        self.garbage_bin_id = 0
        self._dustbin_scale_mult = float(cfg.get("dustbin_scale_mult", self.DUSTBIN_SCALE_MULT))
        # Base asset orient [0.5]*4, then +90° about world Z (matches spawn pose below).
        q_base = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        q_yaw = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], np.pi / 2)
        q_bin = t3d.quaternions.qmult(q_yaw, q_base)
        # Rim (higher-Y edge toward the belt) at BELT_Y_END → center downstream by rim offset.
        # Rim Z above pose origin used so the top sits under `_belt_surf`.
        rim_dy = float(self.DUSTBIN_DY)
        rim_dz = float(self.DUSTBIN_TARGET_H)
        try:
            import json as _json
            from pathlib import Path as _Path
            md_path = (_Path(__file__).resolve().parents[1]
                       / "assets" / "objects" / "011_dustbin" / "model_data0.json")
            md = _json.loads(md_path.read_text())
            sm = float(self._dustbin_scale_mult)
            applied = np.asarray(md["scale"], dtype=float) * sm
            ext = np.asarray(md["extents"], dtype=float) * applied
            center = np.asarray(md["center"], dtype=float) * applied
            R = t3d.quaternions.quat2mat(q_bin)
            hs = ext * 0.5
            corners = np.array(
                [[sx, sy, sz]
                 for sx in (-hs[0], hs[0])
                 for sy in (-hs[1], hs[1])
                 for sz in (-hs[2], hs[2])],
                dtype=float,
            ) + center
            world = (R @ corners.T).T
            rim_dy = float(world.max(axis=0)[1])  # upper rim Y relative to pose origin
            rim_dz = float(world.max(axis=0)[2])  # top Z relative to pose origin
        except Exception:
            pass
        self._dustbin_rim_dy = rim_dy
        self._dustbin_rim_dz = rim_dz
        self._garbage_xy = (self.BELT_X_DEFAULT, float(self.BELT_Y_END) - rim_dy)
        # Pose Z: rim/top = pose_z + rim_dz = `_belt_surf` - clearance (below belt surface).
        below = float(cfg.get("dustbin_rim_below_belt", self.DUSTBIN_RIM_BELOW_BELT))
        self._garbage_z = float(self._belt_surf) - rim_dz - below
        self._dustbin_rim_z = self._garbage_z + rim_dz
        if self.rotten_enabled:
            gx, gy = self._garbage_xy
            sm = float(self._dustbin_scale_mult)
            self.garbage_bin = create_actor(
                self,
                pose=sapien.Pose([gx, gy, self._garbage_z], q_bin.tolist()),
                modelname="011_dustbin",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=sm,
            )
            try:
                self.add_prohibit_area(self.garbage_bin, padding=0.04)
            except Exception:
                self.prohibited_area.append([gx - 0.10, gy - 0.10, gx + 0.10, gy + 0.10])

        # ---- incoming apples ----
        self.apples = []
        self.apple_shapes = []
        self._stage_pose = sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FAR + 0.30,
                                        self._z0 + 0.30])
        self._apple_comps = []
        inelastic = sapien.physx.PhysxMaterial(
            static_friction=0.22, dynamic_friction=0.18, restitution=0.0)
        for i in range(self.n_apples):
            ap = create_actor(
                self, pose=sapien.Pose([self._stage_pose.p[0],
                                        self._stage_pose.p[1] + i * 0.02,
                                        self._stage_pose.p[2]]),
                modelname="220_apple_plain", model_id=0, convex=True, is_static=False,
                scale_mult=self.APPLE_SCALE,
            )
            self._recolor(ap, self._rgb_for_color(self.apple_colors[i]))
            comp = None
            for cc in ap.actor.get_components():
                if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                    comp = cc
            if comp is not None:
                for sh in comp.get_collision_shapes():
                    sh.set_physical_material(inelastic)
                try:
                    if hasattr(comp, "set_enable_ccd"):
                        comp.set_enable_ccd(True)
                except Exception:
                    pass
                comp.set_disable_gravity(True)
                comp.set_kinematic(True)
            self.apples.append(ap)
            self._apple_comps.append(comp)

        # ---- streaming / routing / scoring state ----
        self.cur_idx = 0
        self._step_ctr = 0
        self._timed_out = False
        self._apple_y = [None] * self.n_apples
        self._apple_mode = ["belt"] * self.n_apples   # belt | physics | done
        self._spawned = 0
        self._deposited = [False] * self.n_apples
        self._decided = [False] * self.n_apples
        self._stream_active = False
        self._routed = [None] * self.n_apples
        self.delivered = [None] * self.n_apples
        self._reset_metric_state()
        self.results = [None] * self.n_apples
        self.press_count = 0
        self.dump_press_count = 0
        self._pic_ctr = 0
        self._awaiting_dump_press = False
        self._expert_hold = None  # None | "left" | "right" | "dump" — forces mode while expert holds
        self._divert_batch = []   # apple indices released together by the current divert hold
        # Belt linear speed (m/s) used when releasing apples into dynamics.
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = 1.0 / 250.0
        self._belt_vel_mps = float(self.belt_speed) / max(1, self.advance_every) / max(dt, 1e-6)
        self._belt_vel_mps = float(np.clip(self._belt_vel_mps, 0.02, 0.35))

        self.add_prohibit_area(self.belt, padding=0.04)
        for b in self.buttons.values():
            self.add_prohibit_area(b, padding=0.05)
        for b in self.baskets.values():
            self.add_prohibit_area(b, padding=0.04)

    def _sample_apple_colors(self, n):
        """Sample red/green by color_mode; Opt 2 always marks ≥1 rotten (+ optional second)."""
        if self.color_mode == "random":
            colors = [int(np.random.rand() < 0.5) for _ in range(n)]
        else:
            first = int(np.random.rand() < 0.5)
            colors = [first if (i % 2 == 0) else (1 - first) for i in range(n)]

        self.has_rotten = False
        self.rotten_idx = None
        self.rotten_indices = []
        # Opt 2 (rotten_prob > 0): always one rotten; with P=rotten_prob add a second.
        if self.rotten_enabled and n >= 1:
            idx = int(np.random.randint(0, n))
            colors[idx] = self.COLOR_ROTTEN
            self.has_rotten = True
            self.rotten_idx = idx
            self.rotten_indices = [idx]
            if n >= 2 and float(np.random.rand()) < self.rotten_prob:
                others = [i for i in range(n) if i != idx]
                idx2 = int(others[int(np.random.randint(0, len(others)))])
                colors[idx2] = self.COLOR_ROTTEN
                self.rotten_indices.append(idx2)
        return colors

    def _rgb_for_color(self, c):
        if c == self.COLOR_RED:
            return self.RED
        if c == self.COLOR_GREEN:
            return self.GREEN
        return self.BROWN

    # --------------------------------------------------------------- rendering
    def _shapes_of(self, actor):
        out = []
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                out = list(c.render_shapes)
        return out

    def _recolor(self, actor, rgb):
        for s in self._shapes_of(actor):
            try:
                s.material.set_base_color(list(rgb) + [1.0])
            except Exception:
                pass

    def _piece_pose(self, side, yaw):
        """World pose of a half-plank.

        Rest / dump: each half rotates about its OWN center (horizontal bar that can
        open in the middle). Divert: halves share a center hinge (diagonal blade).
        """
        sgn = -1.0 if side == "left" else 1.0
        cz = self._plank_cz
        own_center = self._gate_mode != "divert"
        if own_center:
            cx = sgn * self._blade_half_len
            cy = self.BELT_Y_FORK
        else:
            cx = sgn * self._blade_half_len * np.cos(yaw)
            cy = self.BELT_Y_FORK + sgn * self._blade_half_len * np.sin(yaw)
        q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        return sapien.Pose([cx, cy, cz], q)

    def _apply_gate_pose(self):
        if not getattr(self, "gate_pieces", None):
            return
        for side in ("left", "right"):
            pose = self._piece_pose(side, getattr(self, f"_gate_yaw_{side}"))
            comp = getattr(self, "_gate_comps", {}).get(side)
            if comp is not None:
                try:
                    comp.set_kinematic_target(pose)
                except Exception:
                    self.gate_pieces[side].actor.set_pose(pose)
            else:
                self.gate_pieces[side].actor.set_pose(pose)

    def _set_gate_target(self):
        """Update half-plank yaw targets from the current hold mode."""
        mode = getattr(self, "_gate_mode", "rest")
        if mode == "dump" or self._dump_open:
            self._gate_mode = "dump"
            self._dump_anim_phase = "part"
            self._gate_yaw_left_target = -self.DUMP_OPEN_ANGLE   # CW
            self._gate_yaw_right_target = self.DUMP_OPEN_ANGLE   # CCW
            self._gate_yaw_target = 0.0
        elif mode == "divert":
            self._dump_anim_phase = "divert"
            base = self.GATE_TILT if self.gate_left else -self.GATE_TILT
            self._gate_yaw_target = base
            self._gate_yaw_left_target = base
            self._gate_yaw_right_target = base
        else:
            # Released → default horizontal along the x-axis.
            self._gate_mode = "rest"
            self._dump_anim_phase = "rest"
            self._dump_open = False
            self._gate_yaw_target = 0.0
            self._gate_yaw_left_target = 0.0
            self._gate_yaw_right_target = 0.0

    def _dump_gap_open(self):
        """True once each half has swung far enough to leave a center passage."""
        thr = 0.65 * self.DUMP_OPEN_ANGLE
        return (self._dump_open
                and abs(self._gate_yaw_left) >= thr
                and abs(self._gate_yaw_right) >= thr)

    def _animate_gate(self):
        if not hasattr(self, "_gate_yaw_left"):
            return
        rate = self._dump_gate_rate if self._gate_mode == "dump" else self._gate_rate
        moved = False
        for attr, tgt_attr in (
            ("_gate_yaw_left", "_gate_yaw_left_target"),
            ("_gate_yaw_right", "_gate_yaw_right_target"),
        ):
            cur = getattr(self, attr)
            tgt = getattr(self, tgt_attr)
            d = tgt - cur
            if abs(d) < 1e-6:
                continue
            step = rate if d > 0 else -rate
            setattr(self, attr, tgt if abs(step) >= abs(d) else cur + step)
            moved = True
        self._gate_yaw = 0.5 * (self._gate_yaw_left + self._gate_yaw_right)
        self._apply_gate_pose()

    # --------------------------------------------------------------- belt sim
    def _hide(self, actor, idx_offset):
        actor.actor.set_pose(sapien.Pose([self._stage_pose.p[0] + idx_offset,
                                          self._stage_pose.p[1] + 1.0,
                                          self._stage_pose.p[2] - 1.0]))

    def _spawn(self, idx):
        self._apple_y[idx] = self.BELT_Y_FAR
        self._apple_mode[idx] = "belt"
        self._latch_spawn_metric(idx)
        comp = self._apple_comps[idx]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
            # Sit upright on the belt (no rolling).
            comp.set_kinematic_target(
                sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FAR,
                             self._belt_surf + self.APPLE_R]))

    def _soften_apple(self, comp):
        """Kill residual motion so settled apples don't bounce."""
        try:
            comp.set_linear_damping(4.0)
            comp.set_angular_damping(4.0)
            comp.set_linear_velocity([0.0, 0.0, 0.0])
            comp.set_angular_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

    def _belt_vel(self):
        v = float(getattr(self, "_belt_vel_mps", 0.08))
        return max(0.03, v)

    def _release_to_physics(self, idx, side=None):
        """Hand apple to PhysX so it collides with the plank and slides into a basket.

        Belts carry apples kinematically (sitting, not rolling). Near the fork they become
        dynamic with belt-direction velocity only — no sideways bias. The static diagonal
        plank provides the deflection; apples slide along its edge into the basket.

        Rotten fruit use this same path for left/right divert contact. ``side="dump"`` is
        only applied while the dump hatch is open (otherwise fall back to the current
        divert side so the apple still collides with the plank like a colored apple).
        """
        if side is None:
            if self._gate_mode == "dump" or self._dump_open:
                side = "dump"
            else:
                side = "left" if self.gate_left else "right"
        if side == "dump" and not self._dump_gap_open():
            side = "left" if self.gate_left else "right"
        if self._apple_mode[idx] == "physics":
            # Allow re-route (e.g. dump hatch just opened on an already-dynamic rotten).
            self._routed[idx] = side
            return
        self._routed[idx] = side
        self._apple_mode[idx] = "physics"
        self._latch_commit_metric(idx, side)
        comp = self._apple_comps[idx]
        if comp is None:
            return
        y = float(self._apple_y[idx]) if self._apple_y[idx] is not None else self.BELT_Y_FORK + 0.05
        # Keep upright; approach the plank along the belt centerline.
        self.apples[idx].actor.set_pose(sapien.Pose(
            [self.BELT_X_DEFAULT, y, self._belt_surf + self.APPLE_R]))
        comp.set_kinematic(False)
        comp.set_disable_gravity(False)
        try:
            comp.set_linear_damping(0.35)
            comp.set_angular_damping(2.5)  # damp spin so they slide rather than tumble
        except Exception:
            pass
        vb = -self._belt_vel()
        comp.set_linear_velocity([0.0, vb, 0.0])
        comp.set_angular_velocity([0.0, 0.0, 0.0])

    def _release_dump_fall(self, idx):
        """Drop off the belt past the near end into the dustbin.

        Place the apple clear of the belt collider — PhysX cannot fall through the slab.
        """
        self._routed[idx] = "dump"
        self._apple_mode[idx] = "physics"
        self._latch_commit_metric(idx, "dump")
        comp = self._apple_comps[idx]
        if comp is None:
            self._deposited[idx] = True
            return
        gx, gy = float(self._garbage_xy[0]), float(self._garbage_xy[1])
        # Past the physical belt end (clear of the slab), then into the bin.
        y_cur = float(self._apple_y[idx]) if self._apple_y[idx] is not None else self.BELT_Y_DUMP
        y = min(y_cur, float(self.BELT_Y_END) - 0.04, float(self.BELT_Y_DUMP))
        z = self._belt_surf + self.APPLE_R
        self.apples[idx].actor.set_pose(sapien.Pose([self.BELT_X_DEFAULT, y, z]))
        self._apple_y[idx] = y
        comp.set_kinematic(False)
        comp.set_disable_gravity(False)
        try:
            comp.set_linear_damping(0.4)
            comp.set_angular_damping(1.0)
        except Exception:
            pass
        # Carry off the end toward the bin (and down).
        comp.set_linear_velocity([0.0, min(-0.18, (gy - y) * 2.5), -0.45])
        comp.set_angular_velocity([0.0, 0.0, 0.0])

    def _freeze_deposit(self, idx, side):
        """Lock a settled apple in place (no bounce) once it reaches a receptacle.

        Freeze at the *current* pose — never teleport/lerp into the basket.
        """
        self._routed[idx] = side
        self._deposited[idx] = True
        self._apple_mode[idx] = "done"
        comp = self._apple_comps[idx]
        if comp is not None:
            p = np.array(self.apples[idx].get_pose().p, dtype=float)
            self.apples[idx].actor.set_pose(sapien.Pose(p.tolist()))
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
            self._soften_apple(comp)
        # Dump hatch closes when the operator RELEASES the buttons (not on deposit).

    def _try_catch(self, idx):
        """Return receptacle side if the dynamic apple has arrived, else None.

        Basket catches only after the apple has left the belt/plank support and dropped
        into the receptacle — never freeze mid-air at tip / belt height (v22 + fall fix).
        """
        p = np.array(self.apples[idx].get_pose().p, dtype=float)
        if self.rotten_enabled:
            gx, gy = self._garbage_xy
            # Never freeze dump fruit on/near the belt — only once it's dropped into the bin.
            still_on_belt = (abs(p[0]) < self.BELT_HALF_W * 1.05
                             and p[1] > self.BELT_Y_END - 0.01
                             and p[2] > self._belt_surf - 0.02)
            if (not still_on_belt
                    and np.hypot(p[0] - gx, p[1] - gy) < self.GARBAGE_CATCH_R
                    and p[1] <= float(gy) + 0.06
                    and p[2] < float(self._dustbin_rim_z) - 0.03
                    and (self._routed[idx] == "dump"
                         or self.apple_colors[idx] == self.COLOR_ROTTEN)):
                return "dump"
        # Still on / over the elevated belt footprint → keep sliding, do not catch.
        on_belt = (abs(p[0]) < self.BELT_HALF_W * 1.02
                   and p[1] > self._belt_near - 0.02
                   and p[2] > self._belt_surf - 0.01)
        if on_belt:
            return None
        # Must have cleared the belt rim before a basket deposit counts.
        if abs(p[0]) < self.BASKET_CATCH_MIN_ABS_X:
            return None
        # Off-belt fall fix: freeze only once clearly down in the receptacle —
        # never at plank-tip / belt height (that was the mid-air freeze).
        if p[2] > self._z0 + self.BASKET_CATCH_MAX_DZ:
            return None
        for side in ("left", "right"):
            if np.hypot(p[0] - self._basket_x[side], p[1] - self._basket_y) < self.BASKET_CATCH_R:
                if self._routed[idx] in ("left", "right") and self._routed[idx] != side:
                    continue
                return side
        return None

    def _reseat_on_belt(self, idx):
        """Return a still-on-belt physics apple to kinematic waiting (gate closed / retry)."""
        p = np.array(self.apples[idx].get_pose().p, dtype=float)
        y = max(float(p[1]), self.BELT_Y_FORK + self.PHYSICS_RELEASE_DY)
        self._apple_y[idx] = y
        self._apple_mode[idx] = "belt"
        self._routed[idx] = None
        comp = self._apple_comps[idx]
        if comp is None:
            return
        pose = sapien.Pose(
            [self.BELT_X_DEFAULT, y, self._belt_surf + self.APPLE_R])
        self.apples[idx].actor.set_pose(pose)
        comp.set_kinematic(True)
        comp.set_disable_gravity(True)
        self._soften_apple(comp)
        try:
            comp.set_kinematic_target(pose)
        except Exception:
            pass

    def _step_physics_apples(self):
        """Belt push into the plank + catch detection; lateral divert from PhysX contact.

        Keep a -Y belt feed so fruit press into the diverted plank; lateral motion comes
        from collision (not a free along-blade velocity assist). A light outward nudge
        applies only after |x| shows the plank already deflected the apple. No kinematic
        tip→box slide / teleport. Off-belt: gravity finishes the drop.
        """
        for i in range(self._spawned):
            if self._deposited[i] or self._apple_mode[i] != "physics":
                continue
            p = np.array(self.apples[i].get_pose().p, dtype=float)
            self._apple_y[i] = float(p[1])
            caught = self._try_catch(i)
            if caught is not None:
                self._freeze_deposit(i, caught)
                continue
            comp = self._apple_comps[i]
            if comp is None:
                continue
            over_belt = (abs(p[0]) < self.BELT_HALF_W * 1.15
                         and p[1] > self._belt_near - 0.04
                         and p[2] < self._belt_surf + self.APPLE_R + 0.06)
            # Rotten / dump: only while the hatch is open (or already past the belt end).
            # If the hatch closed while still at the plank, fall back to normal divert
            # collision so rotten fruit behave like colored fruit.
            if self._routed[i] == "dump":
                past_end = float(p[1]) <= float(self.BELT_Y_END) - 0.01
                if (not self._dump_gap_open()) and (not past_end):
                    self._routed[i] = "left" if self.gate_left else "right"
                else:
                    gx, gy = float(self._garbage_xy[0]), float(self._garbage_xy[1])
                    vb = -self._belt_vel()
                    try:
                        comp.set_angular_velocity([0.0, 0.0, 0.0])
                    except Exception:
                        pass
                    # Still over the slab: keep riding off the end (do not try to fall through mesh).
                    if over_belt and p[1] > self.BELT_Y_END - 0.01:
                        if p[2] < self._belt_surf + self.APPLE_R - 0.002:
                            comp.set_linear_velocity([0.0, vb, 0.03])
                        else:
                            try:
                                vz_cur = float(np.array(comp.get_linear_velocity(), dtype=float)[2])
                            except Exception:
                                vz_cur = 0.0
                            comp.set_linear_velocity([0.0, vb, min(0.0, vz_cur)])
                    else:
                        # Clear of the belt — fall into the bin.
                        try:
                            v = np.array(comp.get_linear_velocity(), dtype=float)
                        except Exception:
                            v = np.zeros(3)
                        vy = min(float(v[1]), (gy - p[1]) * 2.5, -0.12)
                        vz = min(float(v[2]), -0.35)
                        comp.set_linear_velocity([0.0, vy, vz])
                    continue
            # Keep feeding belt velocity — never teleport toward a basket.
            if over_belt and self._routed[i] != "dump":
                try:
                    v = np.array(comp.get_linear_velocity(), dtype=float)
                except Exception:
                    v = np.zeros(3)
                vb = -self._belt_vel()
                # Pure -Y belt feed into the plank face (do not inject free lateral vx).
                vy = min(float(v[1]), vb)
                vx = float(v[0])
                vz = float(v[2])
                diverted = self._divert_tilted()
                if diverted and self._routed[i] in (None, "left", "right"):
                    self._routed[i] = "left" if self.gate_left else "right"
                # Contact-gated only: |x| > ~8 mm means PhysX already deflected the
                # apple off the centerline. Then a light outward nudge keeps the slide
                # alive — never steer along the blade before contact (that left a gap).
                if abs(p[0]) > 0.008:
                    sgn = -1.0 if (self._routed[i] == "left" or p[0] < 0) else 1.0
                    slide = 0.14 if diverted else 0.10
                    vx = float(np.clip(vx + sgn * slide * 0.08, -0.28, 0.28))
                    if abs(vx) < 0.04:
                        vx = sgn * 0.06
                    vy = min(vy, vb * 0.85)
                # Near the belt rim: stop upward belt-sit bias and start the drop so
                # fruit is not held at tip height by over_belt vz lift.
                near_rim = abs(p[0]) > self.BELT_HALF_W * 0.82
                if near_rim:
                    if self._routed[i] in ("left", "right"):
                        sgn_r = -1.0 if self._routed[i] == "left" else 1.0
                        vx = sgn_r * max(abs(vx), 0.18)
                        vx = float(np.clip(vx, -0.34, 0.34))
                    vz = min(vz, -0.28)
                elif p[2] < self._belt_surf + self.APPLE_R - 0.002:
                    vz = max(vz, 0.02)
                elif p[2] > self._belt_surf + self.APPLE_R + 0.018:
                    vz = min(vz, -0.04)
                try:
                    comp.set_angular_velocity([0.0, 0.0, 0.0])
                except Exception:
                    pass
                comp.set_linear_velocity([vx, vy, vz])
            elif (not over_belt and self._routed[i] in ("left", "right")
                  and abs(p[0]) >= 0.08):
                # Off / near the belt rim: gravity finishes the drop. Keep residual outward +
                # downward speed so PhysX tip contact cannot pin fruit mid-air (no
                # kinematic slide / teleport).
                try:
                    v = np.array(comp.get_linear_velocity(), dtype=float)
                    sgn = -1.0 if self._routed[i] == "left" else 1.0
                    vx = float(v[0])
                    vx = sgn * max(abs(vx), 0.16)
                    vx = float(np.clip(vx, -0.35, 0.35))
                    vy = float(np.clip(v[1], -0.25, 0.15))
                    vz = float(v[2])
                    if p[2] > self._z0 + self.BASKET_CATCH_MAX_DZ:
                        vz = min(vz, -0.50)
                    vz = float(np.clip(vz, -0.65, 0.05))
                    try:
                        comp.set_angular_velocity([0.0, 0.0, 0.0])
                    except Exception:
                        pass
                    comp.set_linear_velocity([vx, vy, vz])
                except Exception:
                    pass
            # Only force-deposit if already near a receptacle (no routed-only teleport).
            if p[2] < self._z0 - 0.08 or abs(p[0]) > 0.55 or p[1] < self.BELT_Y_END - 0.25:
                side = self._try_catch(i)
                if side is not None:
                    self._freeze_deposit(i, side)

    def _advance_slats(self):
        if not getattr(self, "slats", None):
            return
        span = self.N_SLATS * self._slat_spacing
        for k in range(self.N_SLATS):
            y = self._slat_ys[k] - self.belt_speed
            if y < self._slat_near:
                y += span
            self._slat_ys[k] = y
            self.slats[k].actor.set_pose(sapien.Pose([self.BELT_X_DEFAULT, y, self._belt_surf + 0.001]))

    def _accum_front_y(self):
        """Y of the front queue slot (just upstream of the plank)."""
        return self.BELT_Y_FORK + self.PHYSICS_RELEASE_DY

    def _queue_indices(self):
        """Belt-riding undeposited apples, frontmost (lowest y) first."""
        idxs = []
        for i in range(self._spawned):
            if (self._apple_y[i] is None or self._deposited[i]
                    or self._apple_mode[i] != "belt"):
                continue
            idxs.append(i)
        idxs.sort(key=lambda i: float(self._apple_y[i]))
        return idxs

    def _plank_queue(self):
        """Apples at/near the plank (belt or still-on-belt physics), frontmost first."""
        front_y = self._accum_front_y()
        limit = front_y + 5 * self.ACCUM_SPACING
        idxs = []
        for i in range(self._spawned):
            if self._apple_y[i] is None or self._deposited[i]:
                continue
            if self._apple_mode[i] == "done":
                continue
            if self._apple_y[i] > limit:
                continue
            if self._apple_mode[i] == "physics":
                p = np.array(self.apples[i].get_pose().p, dtype=float)
                if abs(p[0]) > self.BELT_HALF_W * 0.95:
                    continue
            idxs.append(i)
        idxs.sort(key=lambda i: float(self._apple_y[i]))
        return idxs

    def _divert_tilted(self):
        return (self._gate_mode == "divert"
                and abs(self._gate_yaw_left_target) > 0.2
                and abs(self._gate_yaw_left - self._gate_yaw_left_target) < 0.12)

    def _advance_stream(self):
        if self._spawned < self.n_apples:
            prev = self._spawned - 1
            # Spawn once the previous apple has cleared SPAWN_GAP below the spawn line.
            if self._spawned == 0 or (self._apple_y[prev] is not None
                                      and self._apple_y[prev] <= self.BELT_Y_FAR - self.SPAWN_GAP):
                self._spawn(self._spawned)
                self._spawned += 1
        front = None
        for i in range(self._spawned):
            if self._apple_y[i] is None or self._deposited[i]:
                continue
            if front is None:
                front = i
            # physics apples are not belt-kinematic.
            if self._apple_mode[i] in ("physics", "done"):
                continue
            self._apple_y[i] -= self.belt_speed

        release_y = self._accum_front_y()
        tilted = self._divert_tilted()
        want = "left" if self.gate_left else "right"

        for i in range(self._spawned):
            if self._apple_y[i] is None or self._deposited[i]:
                continue
            if self._apple_mode[i] in ("physics", "done"):
                continue
            y = float(self._apple_y[i])
            # Near the fork: become dynamic and collide with the plank — identical for
            # colored and rotten fruit. Dump routing is only used while the hatch is open;
            # otherwise rotten use the current divert side so they deflect left/right.
            if y <= release_y:
                is_rotten = self.apple_colors[i] == self.COLOR_ROTTEN
                if is_rotten and self._dump_gap_open():
                    if y <= float(self.BELT_Y_END) - 0.03:
                        self._release_dump_fall(i)
                    else:
                        self._release_to_physics(i, side="dump")
                    continue
                tgt = self._target_side_for_apple(i)
                if is_rotten:
                    tgt = want  # plank contact / divert; scoring still requires dump
                if tilted and tgt in ("left", "right") and tgt != want:
                    self._apple_y[i] = max(y, release_y)
                    continue
                self._release_to_physics(i, side=want if tilted else tgt)
                if tilted:
                    cur = list(getattr(self, "_divert_batch", []) or [])
                    if i not in cur:
                        cur.append(i)
                    self._divert_batch = cur

        if tilted:
            # Stamp divert side onto matching apples already at the plank.
            for i in self._plank_queue():
                if self._deposited[i]:
                    continue
                tgt = self._target_side_for_apple(i)
                if self.apple_colors[i] == self.COLOR_ROTTEN:
                    tgt = want
                if tgt != want:
                    continue
                if self._apple_mode[i] == "belt":
                    self._release_to_physics(i, side=want)
                else:
                    self._routed[i] = want
                cur = list(getattr(self, "_divert_batch", []) or [])
                if i not in cur:
                    cur.append(i)
                self._divert_batch = cur

        # Dump hatch just opened: re-route in-flight rotten still over the belt center.
        if self._dump_gap_open():
            for i in range(self._spawned):
                if (self._deposited[i]
                        or self.apple_colors[i] != self.COLOR_ROTTEN
                        or self._apple_mode[i] != "physics"):
                    continue
                p = np.array(self.apples[i].get_pose().p, dtype=float)
                if abs(p[0]) < self.BELT_HALF_W * 0.95 and p[1] > self.BELT_Y_END - 0.02:
                    self._routed[i] = "dump"

        for i in self._queue_indices():
            if self._apple_mode[i] in ("physics", "done"):
                continue
            comp = self._apple_comps[i]
            if comp is None:
                continue
            try:
                if hasattr(comp, "get_kinematic") and not comp.get_kinematic():
                    continue
            except Exception:
                pass
            try:
                comp.set_kinematic_target(
                    sapien.Pose([self.BELT_X_DEFAULT, self._apple_y[i],
                                 self._belt_surf + self.APPLE_R]))
            except Exception:
                pass
        if front is not None:
            self.cur_idx = front

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not hasattr(self, "_step_ctr"):
            return
        self._step_ctr += 1
        self._animate_gate()
        if self._stream_active:
            self._detect_button_press()
            self._step_physics_apples()
            if self._step_ctr % max(1, self.advance_every) == 0:
                self._advance_slats()
                self._advance_stream()

    def _pending_dump_apple(self):
        """Index of an in-flight rotten apple that still needs the dump hatch, else None."""
        for i in range(self.n_apples):
            if (self.apple_colors[i] == self.COLOR_ROTTEN
                    and self._apple_y[i] is not None
                    and not self._deposited[i]):
                return i
        return None

    def _detect_button_press(self):
        """Hold-to-actuate button → plank coupling (continuous, not click/toggle).

        - Hold LEFT only  → divert left (diagonal)
        - Hold RIGHT only → divert right
        - Hold BOTH (Opt 2) → dump hatch open (halves parted)
        - Release all     → plank returns to default horizontal (x-axis)

        Expert hold (`_expert_hold`) reinforces the commanded mode while the
        scripted policy is actively pressing; it must be cleared *before* the
        arms lift. Release (no expert latch, no press) always returns the
        plank to horizontal — never a free-running open/tilt animation.
        """
        bank = getattr(self, "_reactive_buttons", None)
        expert = getattr(self, "_expert_hold", None)
        if bank is not None:
            if expert == "dump":
                bank.set_forced("left", True)
                bank.set_forced("right", True)
            elif expert in ("left", "right"):
                bank.set_forced(expert, True)
                bank.set_forced("left" if expert == "right" else "right", False)
            else:
                bank.set_forced("left", False)
                bank.set_forced("right", False)
            bank.update()
            pressed = {
                "left": bool(bank.is_held("left")),
                "right": bool(bank.is_held("right")),
            }
        else:
            pressed = {"left": False, "right": False}

        awaiting = getattr(self, "_awaiting_dump_press", False)

        if expert == "dump" or (expert is None and pressed["left"] and pressed["right"]
                                and self.rotten_enabled and not awaiting):
            if not self._dump_open:
                self.dump_press_count += 1
                self.press_count += 1
            self._dump_open = True
            self._gate_mode = "dump"
            self._set_gate_target()
        elif expert == "left" or (expert is None and pressed["left"] and not pressed["right"]
                                  and not awaiting):
            if self._gate_mode != "divert" or not self.gate_left or self._dump_open:
                if expert is None:
                    self.press_count += 1
            self._dump_open = False
            self.gate_left = True
            self._gate_mode = "divert"
            self._set_gate_target()
        elif expert == "right" or (expert is None and pressed["right"] and not pressed["left"]
                                   and not awaiting):
            if self._gate_mode != "divert" or self.gate_left or self._dump_open:
                if expert is None:
                    self.press_count += 1
            self._dump_open = False
            self.gate_left = False
            self._gate_mode = "divert"
            self._set_gate_target()
        elif expert is None and not awaiting and not pressed["left"] and not pressed["right"]:
            if self._gate_mode != "rest" or self._dump_open:
                self._dump_open = False
                self._gate_mode = "rest"
                self._set_gate_target()

        self._button_held = dict(pressed)
        self._button_pressed = dict(pressed)

    # --------------------------------------------------------------- helpers
    def _budget_left(self):
        return max(0, int(self.MAX_EPISODE_STEPS) - int(getattr(self, "_step_ctr", 0)))

    def _budget_exhausted(self):
        return int(getattr(self, "_step_ctr", 0)) >= int(self.MAX_EPISODE_STEPS)

    def _apple_on_belt(self, idx):
        """True while an in-flight apple is still on / over the conveyor footprint.

        Deposited fruit and fruit that have left the belt (basket, dump, or past the
        near end) are off-belt. Uses the same over_belt test as `_step_physics_apples`.
        """
        if idx >= int(getattr(self, "_spawned", 0)):
            return False
        if self._deposited[idx] or self._apple_mode[idx] == "done":
            return False
        if self._apple_y[idx] is None:
            return False
        if self._apple_mode[idx] == "belt":
            return True
        p = np.array(self.apples[idx].get_pose().p, dtype=float)
        over_belt = (abs(p[0]) < self.BELT_HALF_W * 1.15
                     and p[1] > self._belt_near - 0.04
                     and p[2] < self._belt_surf + self.APPLE_R + 0.06)
        return bool(over_belt)

    def _belt_stream_cleared(self):
        """True once every apple has spawned and been deposited.

        Do not treat mid-flight physics apples (off the belt footprint but not yet
        caught in a basket/bin) as cleared — that ended episodes before the catch.
        """
        if int(getattr(self, "_spawned", 0)) < int(getattr(self, "n_apples", 0)):
            return False
        return all(bool(self._deposited[i]) for i in range(self.n_apples))

    def _step_record(self):
        if self._budget_exhausted():
            self._timed_out = True
            return
        self._update_kinematic_tasks()
        self.scene.step()
        if self.save_freq and (self._pic_ctr % self.save_freq == 0):
            self._take_picture()
        self._pic_ctr += 1
        if self._budget_exhausted():
            self._timed_out = True

    def _freeze_arm_drives(self, arms=("left", "right")):
        """Re-assert current arm drive targets so a hold pose does not drift."""
        zero_v = None
        for tag in arms:
            try:
                q = (self.robot.get_left_arm_jointState()
                     if tag == "left" else self.robot.get_right_arm_jointState())
                pos = list(q[:-1])
                if zero_v is None:
                    zero_v = [0.0] * len(pos)
                self.robot.set_arm_joints(pos, zero_v, tag)
            except Exception:
                pass

    def _step_record_holding(self, arms=("left", "right")):
        """Physics step while keeping the expert EE pose on the button(s)."""
        self._freeze_arm_drives(arms)
        self._step_record()

    def _release_gate_to_rest(self):
        """Drop expert latch and command plank horizontal before arms leave buttons."""
        self._expert_hold = None
        self._dump_open = False
        self._gate_mode = "rest"
        self._set_gate_target()

    def _press_depth_from_ee(self, arm_tag):
        """Lower EE from the current hover onto the button (within PRESS_DZ_ACTIVE).

        ``grasp_actor(..., grasp_dis≈0.09)`` parks ~20 cm above the keycap; a fixed
        −7 cm tap never reaches the button, so the plank was driven only by the
        expert latch (looks like an automatic open after a tap).
        """
        get_ee = (self.robot.get_left_ee_pose if str(arm_tag) == "left"
                  else self.robot.get_right_ee_pose)
        try:
            ee_z = float(get_ee()[2])
        except Exception:
            ee_z = float(self._button_top_z) + 0.20
        # Aim at the keycap; IK typically stops a few cm short, which is still
        # inside PRESS_DZ_ACTIVE after the deeper press.
        target_z = float(self._button_top_z)
        return float(np.clip(ee_z - target_z, 0.08, 0.28))

    def _hold_both_buttons_until(self, apple_idx):
        """Hold BOTH buttons until the dump apple is deposited, then release → rest.

        Plank angle follows button/expert-hold state only. Rest targets are set
        *before* the arms lift so the hatch cannot keep opening after release.
        """
        if self._budget_exhausted():
            self._timed_out = True
            self.plan_success = False
            return
        left = ArmTag("left")
        right = ArmTag("right")
        self._awaiting_dump_press = True
        self.move(
            self.grasp_actor(self.buttons["left"], arm_tag=left, pre_grasp_dis=0.09,
                             grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0),
            self.grasp_actor(self.buttons["right"], arm_tag=right, pre_grasp_dis=0.09,
                             grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0),
        )
        if not self.plan_success or self._budget_exhausted():
            self._awaiting_dump_press = False
            self._release_gate_to_rest()
            self.plan_success = False
            return
        dz_l = self._press_depth_from_ee(left)
        dz_r = self._press_depth_from_ee(right)
        self.move(
            self.move_by_displacement(left, z=-dz_l),
            self.move_by_displacement(right, z=-dz_r),
        )
        if not self.plan_success or self._budget_exhausted():
            self._awaiting_dump_press = False
            self._release_gate_to_rest()
            self.plan_success = False
            return
        self._awaiting_dump_press = False
        self._expert_hold = "dump"
        if not self._dump_open:
            self.dump_press_count += 1
            self.press_count += 1
        self._dump_open = True
        self._gate_mode = "dump"
        self._set_gate_target()
        # Stay pressed until the hatch has parted AND the dump apple is done.
        # Releasing on deposit alone (before the plank finishes opening) looks like
        # a tap followed by an automatic hatch animation.
        gap_seen = False
        for _ in range(min(self.HOLD_WAIT_MAX, self._budget_left())):
            self._step_record_holding(("left", "right"))
            if self._dump_gap_open():
                gap_seen = True
            if self._belt_stream_cleared() or self._budget_exhausted():
                break
            if gap_seen and (
                    self._deposited[apple_idx] or not self._apple_on_belt(apple_idx)):
                break
        # Rest first (suppress proximity re-latch), then lift — never animate dump
        # after the arms leave the buttons.
        self._awaiting_dump_press = True
        self._release_gate_to_rest()
        lift = 0.5 * (dz_l + dz_r)
        if not self._budget_exhausted() and self.plan_success:
            self.move(
                self.move_by_displacement(left, z=lift),
                self.move_by_displacement(right, z=lift),
            )
            for _ in range(min(20, self._budget_left())):
                self._step_record()
        self._awaiting_dump_press = False
        self._release_gate_to_rest()
        self.plan_success = bool(self._deposited[apple_idx]) and not self._timed_out

    def _hold_side_button_until(self, side, apple_idx, batch=None):
        """Hold one side button; every apple queued at the plank leaves with this divert."""
        if self._budget_exhausted():
            self._timed_out = True
            self.plan_success = False
            return
        arm_tag = ArmTag("left" if side == "left" else "right")
        hold_arm = "left" if side == "left" else "right"
        self.move(self.grasp_actor(self.buttons[side], arm_tag=arm_tag, pre_grasp_dis=0.08,
                                   grasp_dis=0.08, contact_point_id=0, gripper_pos=0.0))
        if not self.plan_success or self._budget_exhausted():
            self._release_gate_to_rest()
            self.plan_success = False
            return
        dz = self._press_depth_from_ee(arm_tag)
        self.move(self.move_by_displacement(arm_tag, z=-dz))
        if not self.plan_success or self._budget_exhausted():
            self._release_gate_to_rest()
            self.plan_success = False
            return
        want_left = (side == "left")
        self._expert_hold = "left" if want_left else "right"
        self._dump_open = False
        self.gate_left = want_left
        self._gate_mode = "divert"
        self.press_count += 1
        queued = list(batch) if batch is not None else self._plank_queue()
        if apple_idx not in queued:
            queued = [apple_idx] + list(queued)
        self._divert_batch = list(queued)
        for i in queued:
            self._decided[i] = True
        self._set_gate_target()
        for _ in range(min(self.HOLD_WAIT_MAX, self._budget_left())):
            self._step_record_holding((hold_arm,))
            if self._budget_exhausted() or self._belt_stream_cleared():
                break
            # Release once every batch apple is deposited or has left the belt
            # footprint (mid-air fall finishes under play_once settle / wait).
            if self._divert_batch and all(
                    self._deposited[i] or not self._apple_on_belt(i)
                    for i in self._divert_batch):
                break
            if not self._divert_batch and (
                    self._deposited[apple_idx] or not self._apple_on_belt(apple_idx)):
                break
        self._divert_batch = []
        self._release_gate_to_rest()
        if not self._budget_exhausted() and self.plan_success:
            self.move(self.move_by_displacement(arm_tag, z=dz))
            for _ in range(min(16, self._budget_left())):
                self._step_record()
        self._release_gate_to_rest()
        self.plan_success = bool(self._deposited[apple_idx]) and not self._timed_out

    def _front_undeposited(self):
        """Frontmost undeposited in-flight apple (whether or not already decided)."""
        for i in range(self._spawned):
            if self._apple_y[i] is None or self._deposited[i]:
                continue
            return i
        return None

    def _target_side_for_apple(self, idx):
        return self._color_side[self.apple_colors[idx]]

    # --------------------------------------------------------------- policy
    def play_once(self):
        self._stream_active = True
        self._step_ctr = 0
        self._timed_out = False
        self._pic_ctr = 0
        max_steps = int(self.MAX_EPISODE_STEPS)
        guard = 0
        retries = [0] * self.n_apples
        while guard < max_steps and not self._budget_exhausted():
            guard += 1
            # End once every apple is deposited (not merely off the belt footprint).
            if self._belt_stream_cleared():
                break
            fi = self._front_undeposited()
            if (fi is not None and self._apple_y[fi] is not None and not self._deposited[fi]):
                # Already released to PhysX with a route — wait for catch; do not
                # re-press (that would steal the divert for the next color).
                if (self._apple_mode[fi] == "physics"
                        and self._routed[fi] is not None
                        and not self._apple_on_belt(fi)):
                    self._step_record()
                    continue
                queued = self._plank_queue()
                # Opt 2: if any rotten is already in the plank queue, dump first (hatch lets
                # only brown fruit through; fresh fruit keep packing). Then divert.
                rotten_q = [i for i in queued
                            if self.apple_colors[i] == self.COLOR_ROTTEN and not self._deposited[i]]
                if rotten_q and self.rotten_enabled:
                    ri = rotten_q[0]
                    self._decided[ri] = True
                    self._hold_both_buttons_until(ri)
                    if self._budget_exhausted():
                        break
                    if not self._deposited[ri] and retries[ri] < 2:
                        retries[ri] += 1
                        continue
                    continue

                side = self._target_side_for_apple(fi)
                lead = (self.DECISION_DY + self.DUMP_DECISION_EXTRA) if side == "dump" else self.DECISION_DY
                # Act once the front apple has joined the plank queue.
                if self._apple_y[fi] <= self._accum_front_y() + lead:
                    queued = self._plank_queue()
                    # Only divert apples that match this hold — wait out conflicting colors.
                    matching = [i for i in queued
                                if self._target_side_for_apple(i) == side]
                    conflict = [i for i in queued
                                if (self._target_side_for_apple(i) != side
                                    and self.apple_colors[i] != self.COLOR_ROTTEN)]
                    if side in ("left", "right") and conflict and not matching:
                        self._step_record()
                        continue
                    self._decided[fi] = True
                    if side == "dump":
                        self._hold_both_buttons_until(fi)
                        if self._budget_exhausted():
                            break
                        if self._deposited[fi]:
                            continue
                        if retries[fi] < 2:
                            retries[fi] += 1
                            continue
                        self._reseat_on_belt(fi)
                        self._apple_y[fi] = self._accum_front_y() + 0.04
                        self._decided[fi] = False
                        retries[fi] = 0
                        self._step_record()
                        continue
                    # Matching colors at the plank leave with this divert.
                    batch = matching if matching else [fi]
                    self._hold_side_button_until(side, fi, batch=batch)
                    if self._budget_exhausted():
                        break
                    # Reseat any trailing apple that landed on the wrong side.
                    for i in list(queued):
                        if not self._deposited[i]:
                            continue
                        want = self._target_side_for_apple(i)
                        landed = self._settled_side(i)
                        if (want in ("left", "right") and landed is not None
                                and landed != want and retries[i] < 2):
                            retries[i] += 1
                            self._deposited[i] = False
                            self._reseat_on_belt(i)
                    if self._deposited[fi]:
                        landed = self._settled_side(fi)
                        if landed is not None and landed != side and retries[fi] < 2:
                            retries[fi] += 1
                            self._deposited[fi] = False
                            self._reseat_on_belt(fi)
                            continue
                    elif retries[fi] < 2:
                        retries[fi] += 1
                        continue
                    else:
                        self._reseat_on_belt(fi)
                        self._apple_y[fi] = self._accum_front_y() + 0.04
                        self._decided[fi] = False
                        retries[fi] = 0
                        self._step_record()
                        continue
                    continue
            self._step_record()
        if self._budget_exhausted():
            self._timed_out = True
            self.plan_success = False
        self._expert_hold = None
        self._divert_batch = []
        self._dump_open = False
        self._gate_mode = "rest"
        self._set_gate_target()
        # Settle remaining mid-flight fruit into baskets/bin (budget permitting).
        settle = min(120 if self.rotten_enabled else 80, self._budget_left())
        for _ in range(settle):
            self._step_record()
            if all(self._deposited[i] for i in range(self.n_apples)):
                break
        # Final plan flag from deposits (per-hold overwrites can leave a stale False).
        if self._timed_out or self._budget_exhausted():
            self._timed_out = True
            self.plan_success = False
        else:
            self.plan_success = all(bool(self._deposited[i]) for i in range(self.n_apples))

        self.info["info"] = {
            "{A}": "220_apple_plain/base0",
            "{B}": f"{self.BASKET_MODEL}/base{self.basket_id}",
            "{a}": "both",
            "{o}": self._option_label(),
        }
        if self.garbage_bin is not None:
            self.info["info"]["{C}"] = "011_dustbin/base0"
        return self.info

    # --------------------------------------------------------------- success
    def _pose_in_basket(self, p):
        """Return 'left'/'right' if pose is clearly in a basket, else None."""
        if abs(p[0]) < self.BASKET_CATCH_MIN_ABS_X * 0.9:
            return None
        # Must have dropped into the box volume (not tip / belt height).
        if p[2] > self._z0 + self.BASKET_CATCH_MAX_DZ:
            return None
        for side in ("left", "right"):
            if np.hypot(p[0] - self._basket_x[side], p[1] - self._basket_y) < self.BASKET_CATCH_R:
                return side
        return None

    def _settled_side(self, idx):
        p = np.array(self.apples[idx].get_pose().p, dtype=float)
        catch_z = self._z0 + self.BASKET_CATCH_MAX_DZ
        # Opt 2: rotten in a basket always counts as that basket — never as dump —
        # even if it was freeze-routed toward the bin.
        if self.rotten_enabled and self.apple_colors[idx] == self.COLOR_ROTTEN:
            basket = self._pose_in_basket(p)
            if basket is not None:
                return basket
            if (self._deposited[idx] and self._routed[idx] in ("left", "right")
                    and abs(p[0]) >= self.BASKET_CATCH_MIN_ABS_X * 0.9
                    and p[2] < catch_z):
                return self._routed[idx]
            gx, gy = self._garbage_xy
            if (np.hypot(p[0] - gx, p[1] - gy) < self.GARBAGE_CATCH_R
                    and p[1] <= float(gy) + 0.06
                    and p[2] < float(self._dustbin_rim_z)):
                return "dump"
            if self._deposited[idx] and self._routed[idx] == "dump":
                return "dump"
            return None
        # Once freeze-deposited in the box / bin, trust the catch side.
        if (self._deposited[idx] and self._routed[idx] is not None
                and (self._routed[idx] == "dump"
                     or (abs(p[0]) >= self.BASKET_CATCH_MIN_ABS_X * 0.9
                         and p[2] < catch_z))):
            return self._routed[idx]
        if self.rotten_enabled:
            gx, gy = self._garbage_xy
            if (np.hypot(p[0] - gx, p[1] - gy) < self.GARBAGE_CATCH_R
                    and p[1] <= float(gy) + 0.06
                    and p[2] < float(self._dustbin_rim_z)
                    and self._routed[idx] == "dump"):
                return "dump"
        # Still on / above the elevated belt → not settled in a receptacle.
        if p[2] > self._belt_surf - 0.01 and abs(p[0]) < self.BELT_HALF_W * 0.9:
            return None
        if p[2] > self._z0 + 0.16:
            return None
        basket = self._pose_in_basket(p)
        if basket is not None:
            return basket
        return None

    def _eval_landings(self):
        for idx in range(self.n_apples):
            # Always score rotten (basket = hard fail); skip unrouted fresh still in flight.
            if (self._routed[idx] is None
                    and self.apple_colors[idx] != self.COLOR_ROTTEN):
                continue
            side = self._settled_side(idx)
            self.delivered[idx] = side
            want = self._target_side_for_apple(idx)
            if self.apple_colors[idx] == self.COLOR_ROTTEN:
                # Rotten succeeds only in the dump; left/right basket is always wrong.
                self.results[idx] = (side == "dump")
            else:
                self.results[idx] = (side is not None and side == want)

    def _rotten_in_basket(self):
        """True if any rotten apple settled in the left or right basket."""
        if not self.rotten_enabled:
            return False
        for i in range(self.n_apples):
            if self.apple_colors[i] != self.COLOR_ROTTEN:
                continue
            side = self.delivered[i]
            if side is None:
                side = self._settled_side(i)
            if side in ("left", "right"):
                return True
        return False

    def _macro_f1(self):
        labels = [self.COLOR_RED, self.COLOR_GREEN]
        if any(c == self.COLOR_ROTTEN for c in self.apple_colors):
            labels.append(self.COLOR_ROTTEN)
        f1s = []
        for c in labels:
            tp = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] == c and self.results[i])
            fp = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] != c and self.delivered[i] is not None
                     and self._color_side.get(c) == self.delivered[i])
            fn = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] == c and not self.results[i])
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        return float(np.mean(f1s)) if f1s else 0.0

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        n = int(getattr(self, "n_apples", 0) or 0)
        self._metric_spawn_step = [None] * n
        self._metric_commit_step = [None] * n   # apple handed to physics past the fork
        self._metric_commit_correct = [None] * n
        self._metric_commit_margin = [None] * n # belt distance still to run at commit, metres

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_spawn_metric(self, idx):
        try:
            if self._metric_spawn_step[idx] is None:
                self._metric_spawn_step[idx] = self._metric_step()
        except Exception:
            pass

    def _latch_commit_metric(self, idx, side):
        """Called from the release paths — the gate side is final from here.

        Records whether the committed route matched the apple's true class at the
        instant of commit; the later landing can still be spoiled by physics, but
        the operator's decision is fixed right here.
        """
        try:
            if self._metric_commit_step[idx] is not None:
                return
            self._metric_commit_step[idx] = self._metric_step()
            want = self._target_side_for_apple(idx)
            if self.apple_colors[idx] == self.COLOR_ROTTEN:
                want = "dump"
            self._metric_commit_correct[idx] = bool(side is not None and side == want)
            y = self._apple_y[idx]
            self._metric_commit_margin[idx] = (
                None if y is None else float(y) - float(self.BELT_Y_END)
            )
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `route_latency_steps` — mean steps from an apple appearing at the far end
        of the belt until it is committed past the diverter (gate side final). The belt
        never stops, so this is the classify-and-act window.
        extra2 `route_accuracy` — fraction of committed apples whose gate side matched
        their true class AT THE MOMENT OF COMMIT. HIGHER is better. Reported alongside
        the task's own end-of-episode ``sorting_accuracy`` / ``macro_f1``.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        lats = []
        try:
            for i in range(int(self.n_apples)):
                a = self._metric_spawn_step[i]
                b = self._metric_commit_step[i]
                if a is not None and b is not None:
                    lats.append(max(int(b) - int(a), 0))
        except Exception:
            lats = []
        mean_lat = (sum(lats) / len(lats)) if lats else None
        out["route_latency_steps"] = None if mean_lat is None else round(float(mean_lat), 3)
        out["route_latency_s"] = None if mean_lat is None else round(float(mean_lat) * dt, 4)
        out["first_route_latency_steps"] = min(lats) if lats else None
        out["routes_counted"] = len(lats)

        try:
            oks = [c for c in self._metric_commit_correct if c is not None]
        except Exception:
            oks = []
        out["route_accuracy"] = (
            round(sum(1 for c in oks if c) / len(oks), 4) if oks else None
        )
        try:
            out["sorting_accuracy"] = round(float(self.sorting_accuracy), 4)
            out["macro_f1"] = round(float(self.macro_f1), 4)
        except Exception:
            out["sorting_accuracy"] = None
            out["macro_f1"] = None
        return out

    def check_success(self):
        self._eval_landings()
        correct = sum(1 for r in self.results if r)
        self.sorting_accuracy = correct / float(self.n_apples)
        self.macro_f1 = self._macro_f1()
        # Episode step budget exceeded → failure.
        if (
            getattr(self, "_timed_out", False)
            or bool(getattr(self, "_episode_timed_out", False))
            or self._budget_exhausted()
        ):
            self._timed_out = True
            return False
        # Opt 2 hard rule: rotten apple in either basket → episode failure.
        if self._rotten_in_basket():
            return False
        fed = all(r is not None for r in self._routed)
        return bool(fed and correct == self.n_apples)

    def get_obs(self):
        obs = super().get_obs()
        if hasattr(self, "_routed"):
            self._eval_landings()
        correct = sum(1 for r in self.results if r)
        obs["sorting"] = {
            "gate_left": bool(getattr(self, "gate_left", True)),
            "dump_open": bool(getattr(self, "_dump_open", False)),
            "cur_idx": int(getattr(self, "cur_idx", 0)),
            "press_count": int(getattr(self, "press_count", 0)),
            "dump_press_count": int(getattr(self, "dump_press_count", 0)),
            "delivered_count": int(sum(1 for d in self.delivered if d is not None)),
            "correct": int(correct),
            "sorting_accuracy": float(correct / float(self.n_apples)),
            "macro_f1": float(self._macro_f1()),
            "green_on_left": bool(getattr(self, "green_on_left", True)),
            "n_apples": int(getattr(self, "n_apples", 0)),
            "belt_speed": float(getattr(self, "belt_speed", 0.0)),
            "color_mode": str(getattr(self, "color_mode", self.COLOR_MODE_DEFAULT)),
            "rotten_prob": float(getattr(self, "rotten_prob", 0.0)),
            "has_rotten": bool(getattr(self, "has_rotten", False)),
            "n_rotten": int(len(getattr(self, "rotten_indices", []) or [])),
            "option": self._option_label(),
        }
        return obs
