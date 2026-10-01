from ._base_task import Base_Task
from .utils import *
from .utils.action import Action, ArmTag
from ._GLOBAL_CONFIGS import *
import os
import sapien
import sapien.render
import numpy as np


class punch_dual_holes(Base_Task):
    """Dual hole-punch on two independent belts, serviced simultaneously by both arms.

    Two parallel belts (LEFT belt entirely in the left zone, RIGHT belt entirely in the
    right zone) each carry a row of square punch cards toward a fixed gantry-mounted
    punch head. Matching indexed squares share the same layout, so they reach the stamp
    line together. Each arm presses only its side's button; the gantry punch heads
    descend on their own — they are not held by the grippers.

    Options (task_args.punch_dual_holes) — each is an independent toggle:
      Opt 1 — missing tile     → missing_tile_mode
          false / "none"          = every slot has a tile (default); both arms always
                                    punch together at each stop
          true / "single_random"  = one random (side, index) slot is empty, so the
                                    robot alternates between punching ONE side and
                                    punching BOTH sides at the same time
          CLI: --task-arg missing_tile_mode=true   or   --option 1
      Opt 2 — belt motion      → belt_continous_motion  (alias: belt_continuous_motion)
          false / "discrete" / "stop_per_tile" = DEFAULT: tiles advance then STOP under
              the stamps so both arms can press together. Hold duration is
              tile_pause_s (default 2.0 s) scaled by U(1±tile_pause_jitter)
              (default ±40%); unpressed ready tiles are marked missed when the
              pause expires, then the belts resume.
          true / "continuous" = belts never stop; press while the tile overlaps the stamp.
              Continuous belt speed is belt_speed × belt_continuous_speed_scale (default 1.6×).
          CLI: --task-arg belt_continous_motion=true   or   --option 2

    A press only counts as a stamp when at least ``STAMP_OVERLAP_MIN`` (30%) of the
    punch-head footprint lands on the card; anything less marks that card missed.

    Tile spacing is always variable for now (random gaps in [square_gap_min,
    square_gap_max]); equal_tile_spacing is not exposed as an option.

    Scoring: a press records |page_center_x - target_x| for the page under that head;
    per belt we take mean clamp(1 - offset/tol, 0, 1) over successfully punched pages.
    """

    # ---- belt geometry (per side; mirrored across x=0) ---------------------
    N_PAGES_DEFAULT = 4                 # moving squares per belt
    PUNCH_TOL_DEFAULT = 0.035           # offset tolerance (m) for full score
    BELT_HALF = (0.13, 0.05, 0.008)     # legacy reference half-extents (x,y,z)
    PAGE_HALF = (0.022, 0.022, 0.004)   # square punch-card half-extents
    # Colored keycap (pressed) + thin black base, matching marble-task key styling.
    BUTTON_HALF = (0.020, 0.020, 0.014)
    KEY_BASE_HALF = (0.032, 0.032, 0.005)
    KEY_BASE_COLOR = (0.08, 0.08, 0.08)
    LEFT_BUTTON_COLOR = (0.20, 0.70, 0.35)
    RIGHT_BUTTON_COLOR = (0.18, 0.48, 0.82)
    PUNCH_HALF = (0.026, 0.026, 0.065)  # gantry punch-head half-extents (+30% vs 0.02/0.05)
    PUNCH_SCALE_MULT = 1.3              # visual/collision scale on 100_seal
    # Fraction of the punch-head footprint that must land on a card. MIN is what counts
    # as stamped; READY is the margin the expert waits for before committing a tap.
    STAMP_OVERLAP_MIN = 0.3
    STAMP_OVERLAP_READY = 0.4
    PUNCH_MODEL = "100_seal"
    PUNCH_MODEL_IDS = (2, 3, 4)
    PUNCH_Q = [0.5, 0.5, 0.5, 0.5]
    BELT_INWARD_REACH_DEFAULT = 0.13
    BELT_OUTWARD_SCALE_DEFAULT = 2.0
    SQUARE_START_MARGIN_DEFAULT = 0.035
    SQUARE_GAP_DEFAULT = 0.030
    SQUARE_GAP_MIN_DEFAULT = 0.020
    SQUARE_GAP_MAX_DEFAULT = 0.060
    BELT_EDGE_MARGIN_DEFAULT = 0.025
    SQUARE_PLACEMENT_MODE_DEFAULT = "variable"  # always variable for now (not an option)
    MISSING_TILE_MODE_DEFAULT = "none"
    BELT_CONTINOUS_MOTION_DEFAULT = False   # Opt 2 default: discrete stop-per-tile
    BELT_CONTINUOUS_SPEED_SCALE_DEFAULT = 1.6  # Opt 2: multiply belt_speed when continuous
    BELT_SPEED_MIN_DEFAULT = 0.00112
    BELT_SPEED_MAX_DEFAULT = 0.00196

    # belt center y (toward the robot's working area) and surface z above table
    BELT_Y = -0.05
    SURF_DZ = 0.016                     # page sits this high above belt-slab center top
    PUNCH_ANIM_STEPS = 48            # down+up stroke length (physics steps); keep visible on demos
    PUNCH_DROP_M = 0.06              # how far the gantry head descends on a press
    TILE_PAUSE_S_DEFAULT = 2.0          # discrete: nominal hold under stamp (seconds)
    TILE_PAUSE_JITTER_DEFAULT = 0.40    # discrete: ± fraction on tile_pause_s per episode
    # Wall-clock hold once the belts arm: tiles sit still for this long before the
    # first advance, so a human (or the expert) can line the stamps up. The arms are
    # NOT held — only the belt waits. Applies to both discrete and continuous modes.
    BELT_START_FREEZE_S_DEFAULT = 4.0
    PAGE_EXIT_MARGIN = 0.002
    HIDE_Z = -10.0
    PUNCH_REST_Z_EXTRA = 0.03
    # Stamp image: create_box texture_id → assets/dyna_textures/<id>.png
    # (cube-UV atlas with the robot head in the +Z/-Z tiles). Sized to sit on
    # the card face (slightly inset from PAGE_HALF).
    PUNCH_MARK_TEXTURE_ID = "custom/robot_head_punch"
    STAMP_HALF = (0.020, 0.020, 0.0015)  # ~paper face, slightly inset
    STAMP_Z_EPS = 0.003

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's task_args block
        self._cfg = kwags.get("task_args", {}).get("punch_dual_holes", {})
        self._apply_legacy_option()
        self._reset_metric_state()
        super()._init_task_env_(**kwags)
        # Belts run from episode start (policy eval never calls play_once).
        self._belt_active = True
        self._belt_running = True

    def _apply_legacy_option(self):
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / missing_tile / missing_tile_mode → Opt 1 missing_tile_mode=true
        2 / continuous / belt_continous_motion → Opt 2 continuous belt
        """
        legacy = self._cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "missing_tile_mode",
            2: "belt_continous_motion",
            "1": "missing_tile_mode",
            "2": "belt_continous_motion",
            "missing_tile": "missing_tile_mode",
            "missing_tile_mode": "missing_tile_mode",
            "single_random": "missing_tile_mode",
            "continuous": "belt_continous_motion",
            "belt_continous_motion": "belt_continous_motion",
            "belt_continuous_motion": "belt_continous_motion",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        if key is None:
            raise ValueError(
                "punch_dual_holes option must be 1/missing_tile_mode or "
                "2/belt_continous_motion (or set the named booleans directly)"
            )
        if key == "missing_tile_mode" and "missing_tile_mode" not in self._cfg:
            self._cfg["missing_tile_mode"] = True
        elif key == "belt_continous_motion" and "belt_continous_motion" not in self._cfg \
                and "belt_continuous_motion" not in self._cfg:
            self._cfg["belt_continous_motion"] = True

    def _normalize_tile_spacing_mode(self):
        # Not an option: always variable spacing for now.
        return "variable"

    def _normalize_missing_tile_mode(self):
        raw_mode = self._cfg.get("missing_tile_mode", self.MISSING_TILE_MODE_DEFAULT)
        if isinstance(raw_mode, bool):
            return "single_random" if raw_mode else "none"
        if isinstance(raw_mode, (int, float)):
            return "single_random" if int(raw_mode) != 0 else "none"
        mode = str(raw_mode).strip().lower()
        if mode in ("single_random", "single", "one_missing", "random_one", "enabled", "on", "true"):
            return "single_random"
        return "none"

    def _normalize_belt_continous_motion(self):
        # Opt 3 — prefer the historical (misspelled) key; accept corrected alias too.
        raw_mode = self._cfg.get(
            "belt_continous_motion",
            self._cfg.get(
                "belt_continuous_motion",
                self._cfg.get("belt_motion_mode", self.BELT_CONTINOUS_MOTION_DEFAULT),
            ),
        )
        if isinstance(raw_mode, bool):
            return raw_mode
        if isinstance(raw_mode, (int, float)):
            return int(raw_mode) != 0
        mode = str(raw_mode).strip().lower()
        if mode in ("continuous", "moving", "run", "stream", "always_on", "true", "on", "yes"):
            return True
        if mode in (
            "discrete",
            "stop_per_tile",
            "stepwise",
            "paused",
            "stop",
            "false",
            "off",
            "no",
        ):
            return False
        return bool(self.BELT_CONTINOUS_MOTION_DEFAULT)

    def _get_tile_pause_steps(self):
        """Discrete-mode max hold under the stamp, in physics steps.

        Prefer tile_pause_s (seconds, default 2.0). Explicit tile_pause_steps still
        wins when provided without tile_pause_s, for older configs. When using
        seconds, apply per-episode ``tile_pause_jitter`` (default ±40%).
        """
        has_steps = "tile_pause_steps" in self._cfg
        has_secs = "tile_pause_s" in self._cfg or "tile_pause_sec" in self._cfg
        if has_steps and not has_secs:
            return max(1, int(self._cfg.get("tile_pause_steps")))
        pause_s = float(
            self._cfg.get(
                "tile_pause_s",
                self._cfg.get("tile_pause_sec", self.TILE_PAUSE_S_DEFAULT),
            )
        )
        pause_s = max(0.0, pause_s)
        jitter = float(
            self._cfg.get("tile_pause_jitter", self.TILE_PAUSE_JITTER_DEFAULT)
        )
        jitter = float(np.clip(jitter, 0.0, 0.95))
        if jitter > 0.0 and pause_s > 0.0:
            scale = float(np.random.uniform(1.0 - jitter, 1.0 + jitter))
            pause_s *= scale
        dt = float(self.scene.get_timestep()) if hasattr(self, "scene") else (1.0 / 250.0)
        return max(1, int(round(pause_s / max(dt, 1e-8))))

    def _sample_start_margin(self):
        base_margin = max(
            float(self._cfg.get("square_start_margin", self.SQUARE_START_MARGIN_DEFAULT)),
            self.PAGE_HALF[0] + 0.004,
        )
        return base_margin

    def _sample_square_layout(self, placement_mode, start_margin):
        start_margin = max(float(start_margin), self.PAGE_HALF[0] + 0.004)
        fixed_gap = max(0.0, float(self._cfg.get("square_gap", self.SQUARE_GAP_DEFAULT)))
        gap_min = max(0.0, float(self._cfg.get("square_gap_min", self.SQUARE_GAP_MIN_DEFAULT)))
        gap_max = max(gap_min, float(self._cfg.get("square_gap_max", self.SQUARE_GAP_MAX_DEFAULT)))

        offsets = []
        gaps = []
        offset = start_margin
        center_step_base = 2.0 * self.PAGE_HALF[0]
        for k in range(self.n_pages):
            offsets.append(float(offset))
            if k == self.n_pages - 1:
                break
            gap = fixed_gap if placement_mode == "equal" else float(np.random.uniform(gap_min, gap_max))
            gap = max(0.0, float(gap))
            gaps.append(gap)
            offset += center_step_base + gap
        return offsets, gaps

    def _build_square_layouts(self):
        self.tile_spacing_mode = self._normalize_tile_spacing_mode()
        self.square_offsets = {}
        self.square_gaps = {}
        start_margin = self._sample_start_margin()
        offsets, gaps = self._sample_square_layout(
            self.tile_spacing_mode,
            start_margin=start_margin,
        )
        for side in ("left", "right"):
            self.square_offsets[side] = list(offsets)
            self.square_gaps[side] = list(gaps)

    def _sample_missing_tile(self):
        self.missing_tile_mode = self._normalize_missing_tile_mode()
        self.missing_tile_side = None
        self.missing_tile_index = None
        self.page_missing = {side: [False] * self.n_pages for side in ("left", "right")}
        if self.missing_tile_mode != "single_random" or self.n_pages <= 0:
            return
        self.missing_tile_side = str(np.random.choice(["left", "right"]))
        self.missing_tile_index = int(np.random.randint(0, self.n_pages))
        self.page_missing[self.missing_tile_side][self.missing_tile_index] = True

    def _build_arrival_schedule(self, side, speed):
        speed = max(float(speed), 1e-8)
        arrival_eff_steps = []
        aligned_offsets = []
        prev_eff = None
        prev_offset = None
        for offset in self.square_offsets[side]:
            if prev_eff is None:
                eff = max(0, int(round(offset / speed)))
            else:
                gap_eff = max(1, int(round((offset - prev_offset) / speed)))
                eff = prev_eff + gap_eff
            arrival_eff_steps.append(eff)
            aligned_offsets.append(float(eff * speed))
            prev_eff = eff
            prev_offset = offset
        return arrival_eff_steps, aligned_offsets

    def _belt_reaches_for_side(self, side):
        inward_reach = float(
            self._cfg.get("belt_inward_reach", self.BELT_INWARD_REACH_DEFAULT)
        )
        outward_reach_cfg = self._cfg.get("belt_outward_reach", None)
        if outward_reach_cfg is None:
            outward_reach = inward_reach * float(
                self._cfg.get("belt_outward_scale", self.BELT_OUTWARD_SCALE_DEFAULT)
            )
        else:
            outward_reach = float(outward_reach_cfg)
        edge_margin = float(self._cfg.get("belt_edge_margin", self.BELT_EDGE_MARGIN_DEFAULT))
        required_offsets = getattr(self, "aligned_square_offsets", {}).get(
            side,
            self.square_offsets[side],
        )
        required_outward = required_offsets[-1] + self.PAGE_HALF[0] + edge_margin
        outward_reach = max(outward_reach, required_outward)
        return float(inward_reach), float(outward_reach)

    def _sample_belt_timing(self):
        belt_speed_cfg = self._cfg.get("belt_speed", None)
        if belt_speed_cfg is not None:
            speed = max(1e-8, float(belt_speed_cfg))
        else:
            speed_min = float(self._cfg.get("belt_speed_min", self.BELT_SPEED_MIN_DEFAULT))
            speed_max = float(self._cfg.get("belt_speed_max", self.BELT_SPEED_MAX_DEFAULT))
            speed_lo = min(speed_min, speed_max)
            speed_hi = max(speed_min, speed_max)
            speed = float(np.random.uniform(speed_lo, speed_hi))
        # Opt 2 continuous runs at a higher surface speed (default 1.6×) so tiles don't crawl.
        continuous = bool(getattr(self, "belt_continous_motion", False))
        if continuous:
            scale = float(
                self._cfg.get(
                    "belt_continuous_speed_scale",
                    self.BELT_CONTINUOUS_SPEED_SCALE_DEFAULT,
                )
            )
            speed = max(1e-8, speed * max(0.0, scale))
        # Discrete: random start delay. Continuous: no phase hold — tiles must move
        # immediately once the belt is active (no "paused at the start" look).
        phase = 0 if continuous else int(np.random.randint(0, 60))
        return speed, phase

    # ------------------------------------------------------------ actors
    def load_actors(self):
        self.n_pages = max(1, int(self._cfg.get("n_pages", self.N_PAGES_DEFAULT)))
        self.punch_tol = float(self._cfg.get("punch_tol", self.PUNCH_TOL_DEFAULT))
        self._build_square_layouts()
        self._sample_missing_tile()

        z0 = 0.74 + self.table_z_bias       # table top surface z

        # Side sign: left=-1, right=+1. Belt timing and square layouts are shared, so
        # matching indexed squares stay synchronized between the two sides.
        self._sides = {}
        self.belt = {}
        self.punch_head = {}
        self.button = {}
        self.button_base = {}
        self._button_home = {}
        self._button_top_z = {}
        self._reactive_buttons = None
        self.pages = {}           # side -> list[Actor]
        self.page_stamped_actors = {}  # side -> list[Actor|None] prebuilt stamped twins
        self.page_stamps = {}     # side -> list[bool]  (True = swapped to stamped twin)
        self.page_target_x = {}   # side -> list[float]  (target punch x for each page)
        self.page_start_x = {}    # side -> list[float]
        self.page_punched = {}    # side -> list[bool]
        self.page_offset = {}     # side -> list[float]  (recorded offset at punch)
        self.belt_speed = {}      # side -> px/step
        self.belt_phase = {}      # side -> int start-delay steps
        self._punch_x = {}        # side -> punch gantry x (world)
        self._punch_rest_z = {}   # side -> punch head rest z
        self._punch_y = {}        # side -> punch/belt y
        self._punch_press = {}    # side -> remaining descend frames (visual)
        self._pending_mark = {}   # side -> page idx waiting for bottom-of-stroke reveal
        self.belt_inner_reach = {}
        self.belt_outward_reach = {}
        self.belt_bounds_x = {}
        self.belt_inner_edge_x = {}
        self.page_arrival_eff = {}
        self.aligned_square_offsets = {}
        self.page_hidden = {}
        self.page_missed = {}
        self.belt_continous_motion = self._normalize_belt_continous_motion()
        self.tile_pause_steps = self._get_tile_pause_steps()
        self.tile_pause_s = float(self.tile_pause_steps) * float(self.scene.get_timestep())
        belt_freeze_s = max(0.0, float(self._cfg.get(
            "belt_start_freeze_s", self.BELT_START_FREEZE_S_DEFAULT)))
        self.belt_start_freeze_steps = max(0, int(round(
            belt_freeze_s / max(float(self.scene.get_timestep()), 1e-8))))
        self._belt_freeze_i = 0
        self.invalid_empty_press = False
        self.invalid_empty_press_count = 0
        self.invalid_empty_press_sides = []
        self.punch_model_id = int(np.random.choice(self.PUNCH_MODEL_IDS))
        shared_belt_timing = self._sample_belt_timing()

        for sign, side in ((-1.0, "left"), (1.0, "right")):
            punch_x = sign * 0.20
            speed, phase = shared_belt_timing
            self.belt_speed[side] = speed
            self.belt_phase[side] = phase
            arrival_eff_steps, aligned_offsets = self._build_arrival_schedule(side, speed)
            self.page_arrival_eff[side] = arrival_eff_steps
            self.aligned_square_offsets[side] = aligned_offsets
            inward_reach, outward_reach = self._belt_reaches_for_side(side)
            inner_edge_x = punch_x - sign * inward_reach
            outer_edge_x = punch_x + sign * outward_reach
            belt_cx = 0.5 * (inner_edge_x + outer_edge_x)
            belt_half_x = 0.5 * abs(outer_edge_x - inner_edge_x)
            self.belt_inner_reach[side] = inward_reach
            self.belt_outward_reach[side] = outward_reach
            self.belt_bounds_x[side] = (float(min(inner_edge_x, outer_edge_x)), float(max(inner_edge_x, outer_edge_x)))
            self.belt_inner_edge_x[side] = float(inner_edge_x)
            self._punch_y[side] = self.BELT_Y

            # belt slab (static scenery)
            belt = create_box(
                self,
                pose=sapien.Pose([belt_cx, self.BELT_Y, z0 + self.BELT_HALF[2]], [1, 0, 0, 0]),
                half_size=(belt_half_x, self.BELT_HALF[1], self.BELT_HALF[2]),
                color=(0.15, 0.15, 0.18),
                name=f"belt_{side}",
                is_static=True,
            )
            self.belt[side] = belt
            belt_top_z = z0 + 2 * self.BELT_HALF[2]

            # punch head: fixed gantry actuator near the inner working area, while the longer
            # outward belt segment holds the queued cards before they travel inward to the head.
            self._punch_x[side] = punch_x
            punch_rest_z = belt_top_z + self.PAGE_HALF[2] * 2 + self.PUNCH_HALF[2] + self.PUNCH_REST_Z_EXTRA
            self._punch_rest_z[side] = punch_rest_z
            head = create_actor(
                self,
                pose=sapien.Pose([punch_x, self.BELT_Y, punch_rest_z], self.PUNCH_Q),
                modelname=self.PUNCH_MODEL,
                model_id=self.punch_model_id,
                convex=True,
                is_static=False,
                scale_mult=self.PUNCH_SCALE_MULT,
            )
            self._make_kinematic(head)
            head.set_name(f"punch_{side}")
            self.punch_head[side] = head
            self._punch_press[side] = 0

            # action keys: colored keycap in a hollow bezel (shelf-marble style), in the
            # arm's near reach (outer, in front of the belt).
            button_x = sign * 0.26
            button_y = self.BELT_Y + 0.13
            cap_hz = float(self.BUTTON_HALF[2])
            cap_z = z0 + cap_hz
            button_base = add_key_base_border(
                self,
                float(button_x),
                float(button_y),
                float(z0),
                self.BUTTON_HALF,
                color=list(self.KEY_BASE_COLOR),
                name_prefix=f"button_base_{side}",
            )
            button_home = sapien.Pose([button_x, button_y, cap_z], [1, 0, 0, 0])
            button = create_box(
                self,
                pose=button_home,
                half_size=self.BUTTON_HALF,
                color=self.RIGHT_BUTTON_COLOR if sign > 0 else self.LEFT_BUTTON_COLOR,
                name=f"button_{side}",
                is_static=True,
            )
            self.button_base[side] = button_base
            self.button[side] = button
            self._button_home[side] = button_home
            self._button_top_z[side] = float(cap_z + cap_hz)

            # cards queue on the OUTER side of the punch head and march inward toward it. Page 0
            # is nearest the head (arrives first); each higher-index page uses the configured
            # placement interval pattern, so they reach the head in index order k = 0, 1, 2, ...
            pages = []
            stamped_pages = []
            tx_list = []
            sx_list = []
            for k, offset in enumerate(self.aligned_square_offsets[side]):
                sx = punch_x + sign * offset                          # further out = arrives later
                # per-page randomized target offset relative to punch center
                toff = float(np.random.uniform(-0.012, 0.012))
                tx = punch_x + toff
                missing = bool(self.page_missing[side][k])
                page_z = self.HIDE_Z if missing else belt_top_z + self.PAGE_HALF[2]
                page_q = [1, 0, 0, 0]
                page = create_box(
                    self,
                    pose=sapien.Pose([sx, self.BELT_Y, page_z], page_q),
                    half_size=self.PAGE_HALF,
                    color=(0.93, 0.93, 0.88),
                    name=f"page_{side}_{k}",
                    is_static=False,   # kinematic: scripted via set_pose, never falls
                )
                self._make_kinematic(page)
                pages.append(page)
                # Pre-build textured stamp overlay off-stage (revealed on punch).
                stamped = None
                if not missing:
                    stamped = create_box(
                        self,
                        pose=sapien.Pose([sx, self.BELT_Y, self.HIDE_Z], page_q),
                        half_size=self.STAMP_HALF,
                        color=(1.0, 1.0, 1.0),
                        name=f"page_{side}_{k}_stamped",
                        is_static=False,
                        texture_id=self.PUNCH_MARK_TEXTURE_ID,
                    )
                    self._make_kinematic(stamped)
                stamped_pages.append(stamped)
                tx_list.append(tx)
                sx_list.append(sx)

            self.pages[side] = pages
            self.page_stamped_actors[side] = stamped_pages
            self.page_stamps[side] = [False] * self.n_pages  # True once swapped to stamped twin
            self.page_target_x[side] = tx_list
            self.page_start_x[side] = sx_list
            self.page_punched[side] = [bool(v) for v in self.page_missing[side]]
            self.page_missed[side] = [False] * self.n_pages
            self.page_offset[side] = [None] * self.n_pages
            self.page_hidden[side] = [bool(v) for v in self.page_missing[side]]
            self._sides[side] = sign

            # reserve space so clutter / randomizers stay clear
            self.add_prohibit_area(belt, padding=0.02)
            for wall in button_base:
                self.add_prohibit_area(wall, padding=0.03)
            self.add_prohibit_area(button, padding=0.03)

        # belt simulation clock (shared step counter; each belt reads its own phase/speed)
        self._belt_step = 0
        self._belt_active = False
        self._reset_metric_state()
        self._belt_freeze_i = 0      # start-of-episode hold, counted once _belt_active
        self._belt_running = False   # only True inside the explicit dwell loops
        # Persistent frame counter so short idles (e.g. idle(1)) don't dump a picture
        # every physics step — that makes continuous motion look frozen at playback fps.
        self._belt_pic_ctr = 0
        # which page index (if any) is currently under each punch head
        self._under_head = {"left": None, "right": None}
        self._init_reactive_buttons()

    def _init_reactive_buttons(self):
        sides = [s for s in ("left", "right") if s in self.button]
        if not sides:
            self._reactive_buttons = None
            return
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[self.button[s] for s in sides],
            home_poses=[self._button_home[s] for s in sides],
            max_depth=float(self.BUTTON_HALF[2]),
            ids=sides,
            # Snappier keycap + deeper trigger so the stamp edge lands when the
            # gripper has actually pressed, not on a light approach.
            visual_step=0.002,
            trigger_depth_frac=0.55,
        )
        self._reactive_buttons.set_tops_z([self._button_top_z[s] for s in sides])

    def _update_reactive_buttons(self):
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        # Keycap spring + punch fire share one path for expert demos and teleop:
        # button press edge always drives the gantry stamp stroke, then scoring.
        triggered = bank.update()
        for side in triggered:
            self._on_button_pressed(side)

    def _on_button_pressed(self, side):
        """Button press edge: stamp head goes down, then register the punch result."""
        if not getattr(self, "_punch_press", None):
            return
        self._trigger_punch_head(side)
        self._fire_punch(side)

    # --------------------------------------------------- belt kinematics
    @staticmethod
    def _make_kinematic(actor):
        """Turn a dynamic rigid box into a kinematic body: it holds its pose under gravity
        and can be repositioned every step with set_pose (used for the moving pages and the
        descending gantry punch heads)."""
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_kinematic(True)
                except Exception:
                    pass

    def _page_x_at(self, side, k, step):
        """World x of page k of `side` at the given belt step. Pages enter from the OUTER end,
        march toward the punch head (belt center), pass under it, and continue off the inner
        end -- so once punched a page leaves the head zone and the next page becomes current."""
        sign = self._sides[side]
        eff = max(0, step - self.belt_phase[side])
        return self.page_start_x[side][k] - sign * self.belt_speed[side] * eff

    def _stamp_pose_for_page(self, page_pose):
        p = page_pose.p
        return sapien.Pose(
            [
                float(p[0]),
                float(p[1]),
                float(p[2]) + self.PAGE_HALF[2] + self.STAMP_HALF[2] + self.STAMP_Z_EPS,
            ],
            page_pose.q,
        )

    def _set_page_pose(self, side, k, x):
        p = self.pages[side][k]
        cur = p.get_pose()
        pose = sapien.Pose([x, cur.p[1], cur.p[2]], cur.q)
        p.actor.set_pose(pose)
        # Keep revealed stamp overlays glued to their page.
        if self.page_stamps[side][k]:
            stamp = self.page_stamped_actors[side][k]
            if stamp is not None:
                stamp.actor.set_pose(self._stamp_pose_for_page(pose))

    def _hide_page(self, side, k):
        if self.page_hidden[side][k]:
            return
        p = self.pages[side][k].get_pose()
        hide = sapien.Pose([p.p[0], p.p[1], self.HIDE_Z], p.q)
        self.pages[side][k].actor.set_pose(hide)
        stamp = self.page_stamped_actors[side][k]
        if stamp is not None and self.page_stamps[side][k]:
            stamp.actor.set_pose(self._stamp_pose_for_page(hide))
        self.page_hidden[side][k] = True

    def _move_with_belt_motion(self, action1, action2=None, advance_belts=False):
        prev = self._belt_running
        self._belt_running = bool(advance_belts)
        try:
            if action2 is not None:
                self.move(action1, action2)
            else:
                self.move(action1)
        finally:
            self._belt_running = prev

    def _page_has_exited_belt(self, side, x):
        sign = self._sides[side]
        inner_edge_x = self.belt_inner_edge_x[side]
        return sign * (x - inner_edge_x) <= -(self.PAGE_HALF[0] + self.PAGE_EXIT_MARGIN)

    def _refresh_pages_at_current_step(self):
        for side in ("left", "right"):
            best_k, best_d = None, 1e9
            for k in range(self.n_pages):
                if self.page_hidden[side][k]:
                    continue
                x = self._page_x_at(side, k, self._belt_step)
                # Keep punched (stamped) cards on the belt so the robot-head mark
                # stays visible; only hide once they are well past the inner edge.
                if self.page_punched[side][k] and self._page_has_exited_belt(side, x):
                    # Delay hide: travel an extra card-width past the exit line.
                    sign = self._sides[side]
                    extra = 2.0 * self.PAGE_HALF[0]
                    if sign * (x - self.belt_inner_edge_x[side]) <= -(self.PAGE_HALF[0] + self.PAGE_EXIT_MARGIN + extra):
                        self._hide_page(side, k)
                    else:
                        self._set_page_pose(side, k, x)
                    continue
                self._set_page_pose(side, k, x)
                if self.page_punched[side][k]:
                    continue
                d = abs(x - self._punch_x[side])
                if d < best_d:
                    best_d, best_k = d, k
            self._under_head[side] = best_k if best_d < 0.05 else None

    def _update_punch_heads(self):
        """Advance gantry punch-head strokes. Runs every physics step while animating."""
        press = getattr(self, "_punch_press", None)
        if not press:
            return
        pending = getattr(self, "_pending_mark", None)
        if pending is None:
            self._pending_mark = {}
            pending = self._pending_mark
        for side in ("left", "right"):
            if side not in press or press[side] <= 0:
                continue
            press[side] -= 1
            frac = 1.0 - press[side] / max(1, self.PUNCH_ANIM_STEPS)
            # Triangle: 0 → 1 → 0 over the stroke; bottom of travel at frac=0.5.
            tri = 1.0 - abs(2 * frac - 1.0)
            drop = tri * float(self.PUNCH_DROP_M)
            h = self.punch_head[side]
            hp = h.get_pose()
            h.actor.set_pose(
                sapien.Pose(
                    [hp.p[0], hp.p[1], self._punch_rest_z[side] - drop],
                    hp.q,
                )
            )
            # Reveal the card mark once the head reaches the bottom of the stroke.
            if frac >= 0.45 and side in pending:
                k = pending.pop(side)
                self._apply_punch_mark(side, k)

    def _align_page_under_punch(self, side, k):
        """Snap page `k` so its center is exactly under the punch at the current held step."""
        x = self._page_x_at(side, k, self._belt_step)
        dx = self._punch_x[side] - x
        if abs(dx) > 0.0:
            self.page_start_x[side][k] += dx
            x = self._page_x_at(side, k, self._belt_step)
        self._set_page_pose(side, k, x)
        self._under_head[side] = k

    def _update_kinematic_tasks(self):
        # base hook (drives any DOMINO dynamic objects); runs EVERY physics step
        super()._update_kinematic_tasks()
        self._update_reactive_buttons()
        # Stamp strokes must advance even when belts are idle (discrete press windows).
        self._update_punch_heads()
        self._track_punch_ready()
        if not getattr(self, "_belt_active", False):
            return
        # Start-of-episode hold: tiles stay put for belt_start_freeze_s so the stamps
        # can be lined up first. Arms move freely throughout; only _belt_step is held,
        # so every downstream arrival/alignment calculation just shifts with it.
        freeze_total = int(getattr(self, "belt_start_freeze_steps", 0))
        if int(getattr(self, "_belt_freeze_i", 0)) < freeze_total:
            self._belt_freeze_i = int(getattr(self, "_belt_freeze_i", 0)) + 1
            self._refresh_pages_at_current_step()
            return
        # In continuous mode the belt advances on every physics step once active, independent
        # of stamping logic. In stepwise mode it only advances when explicit belt motion is
        # requested via _belt_running.
        belt_advancing = bool(getattr(self, "belt_continous_motion", False)) or bool(
            getattr(self, "_belt_running", False)
        )
        if not belt_advancing:
            # still refresh page poses so they render at their held position, but don't advance
            self._refresh_pages_at_current_step()
            return
        self._belt_step += 1
        self._refresh_pages_at_current_step()

    def _trigger_punch_head(self, side):
        self._punch_press[side] = self.PUNCH_ANIM_STEPS

    def _mark_invalid_empty_press(self, side):
        if self.missing_tile_mode == "none":
            return
        self.invalid_empty_press = True
        self.invalid_empty_press_count += 1
        self.invalid_empty_press_sides.append(str(side))
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] INVALID_EMPTY_PRESS side={side} step={self._belt_step}", flush=True)

    def _mark_missed_page(self, side, k):
        if self.page_punched[side][k]:
            return
        self.page_missed[side][k] = True
        self.page_punched[side][k] = True
        # No robot-head mark on misses — only successful stamp contact shows it.
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] MISS {side} page{k} step={self._belt_step}", flush=True)

    def _fire_punch(self, side, k=Ellipsis):
        """Register a punch on `side` at the current belt step: punch whichever page is
        under the head, recording its x-offset from the punch center.

        The gantry head descend is started by ``_on_button_pressed`` / ``_trigger_punch_head``
        so a press always shows the stroke even when this scores a miss or no-ops.
        """
        if k is Ellipsis:
            k = self._under_head[side]
        if k is not None and self.page_punched[side][k]:
            return
        # Fallback path (no prior button edge): still run the stroke.
        if self._punch_press.get(side, 0) <= 0:
            self._trigger_punch_head(side)
        if k is None:
            self._mark_invalid_empty_press(side)
            return
        if not self._page_satisfies_stamp_criterion(side, k):
            self._mark_missed_page(side, k)
            return
        x = self._page_x_at(side, k, self._belt_step)
        off = abs(x - self.page_target_x[side][k])
        self.page_punched[side][k] = True
        self.page_offset[side][k] = float(off)
        self._latch_punch_metric(side, k)
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] FIRE {side} page{k} off={off:.4f} step={self._belt_step}", flush=True)
        # Defer the card mark until the head reaches the bottom of its stroke.
        self._pending_mark[side] = k

    def _apply_punch_mark(self, side, k):
        """Reveal the prebuilt textured stamp overlay on top of page (side, k)."""
        if self.page_stamps[side][k]:
            return
        stamped = self.page_stamped_actors[side][k]
        if stamped is None:
            return
        page_pose = self.pages[side][k].get_pose()
        stamped.actor.set_pose(self._stamp_pose_for_page(page_pose))
        self.page_stamps[side][k] = True
        if os.environ.get("DHP_DEBUG"):
            p = stamped.get_pose().p
            print(
                f"[dhp] STAMP {side} page{k} pose=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})",
                flush=True,
            )

    def _belt_idle(self, steps, advance_belts=True):
        """Dwell for `steps` physics steps while optionally advancing the belts."""
        prev = self._belt_running
        self._belt_running = bool(advance_belts)
        try:
            for _ in range(max(0, int(steps))):
                self._update_kinematic_tasks()
                self.scene.step()
                self._belt_pic_ctr = int(getattr(self, "_belt_pic_ctr", 0)) + 1
                if self.save_freq and (self._belt_pic_ctr % self.save_freq == 0):
                    self._take_picture()
        finally:
            self._belt_running = prev

    # -------------------------------------------------------- press poses
    def _button_press_actions(self, side, descend):
        """Build a (arm_tag, [Action]) that moves the side's gripper straight down by
        `descend` onto its button (relative move along the world -z)."""
        arm = ArmTag(side)
        return self.move_by_displacement(arm_tag=arm, z=-descend)

    def _button_release_actions(self, side, ascend):
        arm = ArmTag(side)
        return self.move_by_displacement(arm_tag=arm, z=ascend)

    def _hover_button(self, side):
        """Move the side's gripper to hover just above its button (top-down)."""
        arm = ArmTag(side)
        # grasp_actor on the button's top-down contact point positions the (closed) gripper
        # right above the button without committing the press.
        return self.grasp_actor(self.button[side], arm_tag=arm, pre_grasp_dis=0.09,
                                grasp_dis=0.09, contact_point_id=0)

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] {tag}: plan={self.plan_success} step={self._belt_step} "
                  f"L_pun={self.page_punched['left']} R_pun={self.page_punched['right']} "
                  f"under={self._under_head}", flush=True)

    def _page_arrival_step(self, side, k):
        """The belt step at which page k of `side` is centered on its punch head."""
        return int(self.page_arrival_eff[side][k]) + self.belt_phase[side]

    def _next_unpunched_page(self, side):
        for k in range(self.n_pages):
            if not self.page_punched[side][k]:
                return k
        return None

    def _page_is_under_stamp(self, side, k):
        x = self._page_x_at(side, k, self._belt_step)
        tol = max(1e-6, 0.25 * self.belt_speed[side])
        return abs(x - self._punch_x[side]) <= tol

    @staticmethod
    def _interval_overlap_length(center_a, half_a, center_b, half_b):
        lo = max(center_a - half_a, center_b - half_b)
        hi = min(center_a + half_a, center_b + half_b)
        return max(0.0, hi - lo)

    def _stamp_overlap_ratio(self, side, k):
        """Fraction of the punch-head footprint that lies on page ``k``.

        Normalized by the head area (not the card). With the enlarged stamp the head is
        larger than the card, so the ratio caps below 1 even when fully covering the page.
        """
        page_x = self._page_x_at(side, k, self._belt_step)
        punch_x = self._punch_x[side]
        x_overlap = self._interval_overlap_length(
            page_x,
            self.PAGE_HALF[0],
            punch_x,
            self.PUNCH_HALF[0],
        )
        y_overlap = self._interval_overlap_length(
            self.BELT_Y,
            self.PAGE_HALF[1],
            self._punch_y[side],
            self.PUNCH_HALF[1],
        )
        punch_area = (2.0 * self.PUNCH_HALF[0]) * (2.0 * self.PUNCH_HALF[1])
        if punch_area <= 1e-8:
            return 0.0
        return float((x_overlap * y_overlap) / punch_area)

    def _page_satisfies_stamp_criterion(self, side, k):
        """Whether a press landing now counts as a stamp on page ``k``."""
        return self._stamp_overlap_ratio(side, k) >= self.STAMP_OVERLAP_MIN

    def _page_ready_for_press(self, side, k):
        """Whether the expert should commit a tap on page ``k`` at this step.

        Stricter than the credit threshold so scripted demos still punch near the card
        center: on a discrete stop the page is snapped under the head, and in continuous
        motion we wait for overlap margin above ``STAMP_OVERLAP_MIN``.
        """
        if self.belt_continous_motion:
            return self._stamp_overlap_ratio(side, k) >= self.STAMP_OVERLAP_READY
        return self._page_is_under_stamp(side, k)

    def _ready_pages_at_current_step(self):
        ready_by_side = {}
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._page_ready_for_press(side, k):
                ready_by_side[side] = k
                self._under_head[side] = k
        return ready_by_side

    def _final_belt_runout_steps(self):
        travel_steps = []
        for side in ("left", "right"):
            speed = max(self.belt_speed[side], 1e-8)
            travel = self.belt_inner_reach[side] + self.PAGE_HALF[0] + self.PAGE_EXIT_MARGIN
            travel_steps.append(int(np.ceil(travel / speed)))
        return max(40, max(travel_steps) + 8)

    def _pages_arriving_at_step(self, step_target):
        pages_by_side = {}
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._page_arrival_step(side, k) == step_target:
                pages_by_side[side] = k
        return pages_by_side

    def _mark_missed_ready_pages(self, ready_by_side):
        for side, k in ready_by_side.items():
            self._mark_missed_page(side, k)

    def _mark_overdue_pages(self):
        marked_any = False
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._belt_step > self._page_arrival_step(side, k) and not self._page_satisfies_stamp_criterion(side, k):
                self._mark_missed_page(side, k)
                marked_any = True
        return marked_any

    def _build_press_plan(self, ready_by_side, descend=0.05):
        pressed_sides = [side for side in ("left", "right") if side in ready_by_side]
        if not pressed_sides:
            return [], None, None
        if len(pressed_sides) == 2:
            return (
                pressed_sides,
                self._button_press_actions("left", descend),
                self._button_press_actions("right", descend),
            )
        return pressed_sides, self._button_press_actions(pressed_sides[0], descend), None

    def _estimate_press_duration(self, ready_by_side, descend=0.05):
        pressed_sides, action1, action2 = self._build_press_plan(ready_by_side, descend=descend)
        if not pressed_sides:
            return 0, pressed_sides, action1, action2
        duration = self.calculate_move_duration(action1, action2)
        return int(duration), pressed_sides, action1, action2

    def _press_ready_sides(self, ready_by_side, descend=0.05, advance_belts=False):
        pressed_sides, action1, action2 = self._build_press_plan(ready_by_side, descend=descend)
        if not pressed_sides:
            return []
        # 1) Land the grippers on the keys first — do not force keycaps yet, or the
        #    stamp stroke would start while the arms are still descending.
        self._move_with_belt_motion(action1, action2, advance_belts=advance_belts)
        # 2) Now depress the keys; ReactivePushButtons press edge drives the gantry
        #    stamp down (same path as teleop).
        bank = getattr(self, "_reactive_buttons", None)
        if bank is not None:
            for side in pressed_sides:
                # Clear any latch from a light tip graze during the descend so the
                # forced press below still emits a fresh edge after contact.
                try:
                    idx = bank.resolve_index(side)
                    bank._latched[idx] = False
                except Exception:
                    pass
                bank.set_forced(side, True)
            steps = max(
                8,
                int(bank.trigger_depth / max(bank.visual_step, 1e-6)) + 2,
            )
            self._belt_idle(steps, advance_belts=advance_belts)
            for side in pressed_sides:
                bank.set_forced(side, False)
        # 3) Safety net if the press edge never fired.
        for side in pressed_sides:
            k = ready_by_side[side]
            if not self.page_punched[side][k]:
                self._on_button_pressed(side)
        return pressed_sides

    def _continuous_press_descend(self):
        """Shorter tap in continuous mode so press+release doesn't eat a full inter-tile gap."""
        return float(self._cfg.get("continuous_press_descend", 0.028))

    def _eta_to_page(self, side, k):
        return int(self._page_arrival_step(side, k) - self._belt_step)

    def _next_stamp_targets(self):
        """Next unpunched page per side (may be empty if that side is done)."""
        targets = {}
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is not None:
                targets[side] = k
        return targets

    def _run_continuous_belt_policy(self):
        """Constant-pace belt: never stop; quick on-the-fly taps with lead time.

        Frame recording uses a persistent `_belt_pic_ctr` (see `_belt_idle`), so
        per-step cruise / lead waits stay at real-time playback speed. The old
        idle(1) loop reset `i % save_freq` every call and dumped one video frame
        per physics step, which made Opt 2 look like discrete stop/go.
        """
        descend = self._continuous_press_descend()
        ascend = descend
        probe = self._next_stamp_targets()
        if len(probe) >= 1:
            press_dur, _, _, _ = self._estimate_press_duration(probe, descend=descend)
        else:
            press_dur = 20
        # Begin the tap this many steps before arrival so the fire lands near center.
        lead = max(8, int(press_dur))

        while not all(all(self.page_punched[s]) for s in ("left", "right")):
            targets = self._next_stamp_targets()
            if not targets:
                break

            etas = {side: self._eta_to_page(side, k) for side, k in targets.items()}
            min_eta = min(etas.values())

            ready_now = self._ready_pages_at_current_step()
            if ready_now:
                pressed = self._press_ready_sides(
                    ready_now, descend=descend, advance_belts=True
                )
                self._dbg(f"cont press@step={self._belt_step} sides={pressed}")
                self._release_pressed_sides(pressed, ascend=ascend, advance_belts=True)
                self._mark_overdue_pages()
                continue

            if min_eta > lead:
                # Far from the next stamp — cruise in one-frame chunks (still continuous).
                chunk = max(1, int(self.save_freq or 15))
                self._belt_idle(min(min_eta - lead, chunk), advance_belts=True)
                continue

            # Lead window: fine-step until overlap, then quick-tap. Pictures stay at save_freq.
            wait_guard = 0
            max_wait = max(lead * 3, 60)
            while wait_guard < max_wait:
                ready_now = self._ready_pages_at_current_step()
                if ready_now:
                    break
                if self._mark_overdue_pages():
                    break
                self._belt_idle(1, advance_belts=True)
                wait_guard += 1
            else:
                self._mark_overdue_pages()
                continue

            ready_now = self._ready_pages_at_current_step()
            if not ready_now:
                continue
            pressed = self._press_ready_sides(
                ready_now, descend=descend, advance_belts=True
            )
            self._dbg(f"cont lead-press@step={self._belt_step} sides={pressed}")
            self._release_pressed_sides(pressed, ascend=ascend, advance_belts=True)

    def _release_pressed_sides(self, pressed_sides, ascend=0.05, advance_belts=False):
        if not pressed_sides:
            return
        if len(pressed_sides) == 2:
            self._move_with_belt_motion(
                self._button_release_actions("left", ascend),
                self._button_release_actions("right", ascend),
                advance_belts=advance_belts,
            )
        else:
            self._move_with_belt_motion(
                self._button_release_actions(pressed_sides[0], ascend),
                advance_belts=advance_belts,
            )

    def _handle_discrete_stamp_stop(self, ready_by_side, descend=0.05, ascend=0.05):
        """Stop ready tiles under the stamps for up to `tile_pause_s` (default 2.0 s).

        Belts stay frozen for the whole stop. The expert presses at the start of the
        window (both arms together when both sides are ready), then the stop dwells
        for any remaining pause budget before the belts resume. Ready tiles still
        unpunched after the press are marked missed.
        """
        pause_budget = max(1, int(self.tile_pause_steps))
        press_dur, _, _, _ = self._estimate_press_duration(ready_by_side, descend=descend)
        # Release mirrors the press displacement; treat costs as equal.
        estimated_action_steps = max(0, int(press_dur) * 2)

        pressed_sides = self._press_ready_sides(
            ready_by_side,
            descend=descend,
            advance_belts=False,
        )
        self._dbg(f"after press @step={self._belt_step} sides={pressed_sides}")
        self._release_pressed_sides(pressed_sides, ascend=ascend, advance_belts=False)

        unfinished = {
            side: k
            for side, k in ready_by_side.items()
            if not self.page_punched[side][k]
        }
        if unfinished:
            self._mark_missed_ready_pages(unfinished)

        remaining = max(0, pause_budget - estimated_action_steps)
        if remaining > 0:
            self._belt_idle(remaining, advance_belts=False)

    def play_once(self):
        # 1) Bring both grippers to hover over their buttons simultaneously (truly dual-arm).
        #    Belts stay off during the approach so the arrival schedule still lines up;
        #    continuous motion starts immediately afterward (phase=0, no start delay).
        continuous_motion = self.belt_continous_motion
        self._move_with_belt_motion(
            self._hover_button("left"),
            self._hover_button("right"),
            advance_belts=False,
        )
        self._dbg("after hover")

        # start the belts running
        self._belt_active = True

        # 2) Discrete (default): advance to the next stamp stop, HOLD under the heads for
        #    up to tile_pause_s so both arms can press together, then resume. Continuous:
        #    belt runs at a constant pace every physics step; arms do quick on-the-fly taps.
        if continuous_motion:
            self._run_continuous_belt_policy()
        else:
            while not all(all(self.page_punched[s]) for s in ("left", "right")):
                next_steps = []
                for side in ("left", "right"):
                    k = self._next_unpunched_page(side)
                    if k is not None:
                        next_steps.append(self._page_arrival_step(side, k))
                if not next_steps:
                    break
                step_target = min(next_steps)
                self._run_belts_to(step_target)
                ready_by_side = self._ready_pages_at_current_step()
                if not ready_by_side:
                    continue
                # Snap ready tiles exactly under the stamp so both arms share one stop.
                for side, k in ready_by_side.items():
                    self._align_page_under_punch(side, k)
                self._handle_discrete_stamp_stop(ready_by_side)

        # let stamped squares clear the inner belt edge and disappear before ending the episode
        self._belt_idle(self._final_belt_runout_steps(), advance_belts=True)
        self._belt_active = False

        self.info["info"] = {"{a}": "left", "{b}": "right"}
        return self.info

    def _run_belts_to(self, step_target, max_extra=400):
        """Advance the belts until self._belt_step reaches step_target, recording frames."""
        prev = self._belt_running
        self._belt_running = True
        try:
            guard = 0
            # The start-of-episode hold burns wall-clock steps without moving _belt_step,
            # so the guard has to cover it or the first call bails out before the target.
            remaining_freeze = max(0, int(getattr(self, "belt_start_freeze_steps", 0))
                                   - int(getattr(self, "_belt_freeze_i", 0)))
            guard_limit = step_target + max_extra + remaining_freeze
            while self._belt_step < step_target and guard < guard_limit:
                self._update_kinematic_tasks()
                self.scene.step()
                self._belt_pic_ctr = int(getattr(self, "_belt_pic_ctr", 0)) + 1
                if self.save_freq and (self._belt_pic_ctr % self.save_freq == 0):
                    self._take_picture()
                guard += 1
        finally:
            self._belt_running = prev

    # ------------------------------------------------------------- scoring
    def _side_score(self, side):
        offs = [o for o in self.page_offset[side] if o is not None]
        if not offs:
            return 0.0
        return float(np.mean([np.clip(1.0 - o / self.punch_tol, 0.0, 1.0) for o in offs]))

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        n = int(getattr(self, "n_pages", 0) or 0)
        self._metric_ready_step = {"left": [None] * n, "right": [None] * n}
        self._metric_punch_step = {"left": [None] * n, "right": [None] * n}

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_punch_ready(self):
        """Latch the first step each unpunched tile satisfies the stamp criterion.

        This is the leading edge of the operator's window: the tile is credit-eligible
        under the head from here until the belt carries it past.
        """
        try:
            for side in ("left", "right"):
                slots = self._metric_ready_step.get(side)
                if slots is None:
                    continue
                for k in range(int(self.n_pages)):
                    if slots[k] is not None or self.page_hidden[side][k]:
                        continue
                    if self.page_punched[side][k] or self.page_missing[side][k]:
                        continue
                    if self._page_satisfies_stamp_criterion(side, k):
                        slots[k] = self._metric_step()
        except Exception:
            pass

    def _latch_punch_metric(self, side, k):
        """Called from _fire_punch the moment a tile is credited."""
        try:
            slots = self._metric_punch_step.get(side)
            if slots is not None and slots[k] is None:
                slots[k] = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `punch_latency_steps` — mean steps from a tile first becoming
        credit-eligible under the stamp head until the punch actually fires, averaged
        over the tiles punched. The belt keeps moving, so this is the reaction window.
        extra2 `punch_offset_norm` — mean recorded punch offset divided by ``punch_tol``.
        LOWER is better; <= 1.0 scores, 0.0 is dead-centre.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        lats = []
        try:
            for side in ("left", "right"):
                for k in range(int(self.n_pages)):
                    a = self._metric_ready_step[side][k]
                    b = self._metric_punch_step[side][k]
                    if a is not None and b is not None:
                        lats.append(max(int(b) - int(a), 0))
        except Exception:
            lats = []
        mean_lat = (sum(lats) / len(lats)) if lats else None
        out["punch_latency_steps"] = None if mean_lat is None else round(float(mean_lat), 3)
        out["punch_latency_s"] = None if mean_lat is None else round(float(mean_lat) * dt, 4)
        out["worst_punch_latency_steps"] = max(lats) if lats else None
        out["punches_counted"] = len(lats)

        offs = []
        try:
            for side in ("left", "right"):
                offs.extend([float(o) for o in self.page_offset[side] if o is not None])
        except Exception:
            offs = []
        try:
            tol = max(float(self.punch_tol), 1e-9)
            out["punch_offset_norm"] = round(sum(offs) / len(offs) / tol, 4) if offs else None
            out["worst_punch_offset_norm"] = round(max(offs) / tol, 4) if offs else None
        except Exception:
            out["punch_offset_norm"] = None
            out["worst_punch_offset_norm"] = None
        try:
            out["punch_score_mean"] = round(float(self.punch_score_mean), 4)
        except Exception:
            out["punch_score_mean"] = None
        return out

    def check_success(self):
        """Success iff every present tile was punched (missing slots are skipped).

        A present tile counts as punched only when it has a recorded punch offset and
        was not marked missed. Empty-slot presses (Opt 1 missing tile) are tracked in
        obs for debugging but do **not** fail the episode.
        """
        all_present_punched = True
        for side in ("left", "right"):
            for k in range(self.n_pages):
                if self.page_missing[side][k]:
                    continue
                if self.page_missed[side][k] or self.page_offset[side][k] is None:
                    all_present_punched = False
                    break
            if not all_present_punched:
                break
        self.punch_score_L = self._side_score("left")
        self.punch_score_R = self._side_score("right")
        self.punch_score_mean = 0.5 * (self.punch_score_L + self.punch_score_R)
        return bool(all_present_punched)

    # record per-belt punch state into the trajectory (per frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["hole_punch"] = {
            "left_punched": [bool(b) for b in self.page_punched["left"]],
            "right_punched": [bool(b) for b in self.page_punched["right"]],
            "left_offsets": [None if o is None else float(o) for o in self.page_offset["left"]],
            "right_offsets": [None if o is None else float(o) for o in self.page_offset["right"]],
            "equal_tile_spacing": self.tile_spacing_mode == "equal",
            "tile_spacing_mode": self.tile_spacing_mode,
            "placement_mode": self.tile_spacing_mode,
            "square_placement_mode": self.tile_spacing_mode,
            "missing_tile_mode": self.missing_tile_mode,
            "missing_tile_side": self.missing_tile_side,
            "missing_tile_index": self.missing_tile_index,
            "belt_continous_motion": bool(self.belt_continous_motion),
            "belt_continuous_motion": bool(self.belt_continous_motion),
            "left_missing_tiles": [bool(v) for v in self.page_missing["left"]],
            "right_missing_tiles": [bool(v) for v in self.page_missing["right"]],
            "tile_pause_steps": int(self.tile_pause_steps),
            "tile_pause_s": float(getattr(self, "tile_pause_s", self.tile_pause_steps * self.scene.get_timestep())),
            "left_missed_tiles": [bool(v) for v in self.page_missed["left"]],
            "right_missed_tiles": [bool(v) for v in self.page_missed["right"]],
            "invalid_empty_press": bool(self.invalid_empty_press),
            "invalid_empty_press_count": int(self.invalid_empty_press_count),
            "invalid_empty_press_sides": list(self.invalid_empty_press_sides),
            "square_gaps": [float(g) for g in self.square_gaps["left"]],
            "left_square_offsets": [float(o) for o in self.square_offsets["left"]],
            "right_square_offsets": [float(o) for o in self.square_offsets["right"]],
            "left_square_gaps": [float(g) for g in self.square_gaps["left"]],
            "right_square_gaps": [float(g) for g in self.square_gaps["right"]],
            "left_belt_outward_reach": float(self.belt_outward_reach["left"]),
            "right_belt_outward_reach": float(self.belt_outward_reach["right"]),
            "belt_speed_left": float(self.belt_speed["left"]),
            "belt_speed_right": float(self.belt_speed["right"]),
            "punch_score_L": float(self._side_score("left")),
            "punch_score_R": float(self._side_score("right")),
            "punch_tol": float(self.punch_tol),
        }
        return obs
