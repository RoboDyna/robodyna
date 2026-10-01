from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import numpy as np
import os


class pack_fruits(Base_Task):
    """Pack red/green apples from moving belts into color-matched breadbaskets.

    Two conveyor slabs sit with a gap centered on the table and run toward the
    robot (-y). Scenario options (independent; Opt 1+2 combines both):

    - **Default**: one color (red *or* green), one basket on the matching side,
      **3–5** apples, **one colored apple at a time**.
    - **Opt 1** (``two_colors_enabled``): **two baskets** and **both colors**
      (red + green). Red apples ride the **left** belt into the left basket,
      green the **right** belt into the right basket — each arm stays on its
      side. Each color independently has **2–3** apples (counts need not
      match). Still only one colored apple at a time (no dual grasp).
    - **Opt 2** (``distractor_enabled``): same as default, plus ≥1 black
      distractor apple (must not end in a basket); 30% chance of a second black apple.
      Black fruit is graspable like the colored apples (pinch / carry / drop) but
      packing one into a basket fails. It may ride with a colored apple (same
      belt behind it, or the other belt).
    - **Opt 1+2**: two colors, dedicated belts (2–3 each color), ≥1 black
      distractor (either belt); one colored apple at a time (black may
      co-appear).

    Success requires every colored apple to rest in its color-matched basket
    (red → left basket, green → right), and no black distractor in any basket.
    Default / Opt 2 still spawn the
    single color on either belt; two-color scenarios pin each color to its
    basket-side belt so the matching arm never reaches across.

    Belt speed is sampled each episode as nominal × U(1 ± belt_speed_jitter)
    (default ±20%), independently per belt.
    """

    N_ITEMS_MIN_DEFAULT = 3          # single-color episode count
    N_ITEMS_MAX_DEFAULT = 5
    N_PER_COLOR_MIN_DEFAULT = 2      # two-color: each color independently
    N_PER_COLOR_MAX_DEFAULT = 3
    BELT_GAP_DEFAULT = 0.10
    BELT_SPEED_DEFAULT = 0.0008       # m advanced per belt tick (slow enough to pick)
    BELT_SPEED_JITTER_DEFAULT = 0.20  # fraction; speed ~ U((1-j)*nom, (1+j)*nom)
    ADVANCE_EVERY_DEFAULT = 3         # physics steps between belt ticks
    SPAWN_GAP_DEFAULT = 0.16          # y-gap between consecutive spawns on a belt
    SPAWN_DELAY_S_DEFAULT = 2.0       # unused (kept for config compat); spawn waits on drop/despawn
    # legacy pair-wave knobs (kept for config compat; colored fruit is always solo)
    PAIR_STAGGER_ENABLED_DEFAULT = False
    PAIR_STAGGER_Y_DEFAULT = None
    SINGLE_WAVE_ANY_BELT_DEFAULT = False

    # same belt slab dimensions as control_quality
    BELT_HALF_LEN = 0.30
    BELT_HALF_WID = 0.07
    BELT_THICK = 0.012
    BELT_Y = 0.0
    BELT_Y_FAR = 0.26                 # spawn y (far end)
    BELT_Y_NEAR = -0.26               # leave / wrap y
    PICK_Y = 0.24                     # begin moving the arm into place (fruit keeps rolling)
    PICK_Y_END = -0.16                # give up past this (still moving; never park)
    PICK_STATION_Y = 0.02             # hover / grab y; fruit rolls through here
    # pinch only once the gripper has approached within ~2 cm
    ATTACH_XY = 0.02
    ATTACH_Z_MAX = 0.055              # TCP may sit slightly above the fruit
    GRASP_SETTLE_STEPS = 25           # contacts form before gravity (pick_ripe_apple)
    HIDE_Z = -10.0

    FRUIT_MODEL = "035_apple"
    FRUIT_SCALE = 0.80                # larger -> easier grasp
    # 035_apple model_data0.json (authored scale already applied by create_actor)
    _APPLE_AUTHOR_SCALE = 0.7
    _APPLE_CENTER_Y = 0.03814048367178239
    _APPLE_EXTENT_Y = 0.0919138697184135
    FRUIT_R = 0.026                   # approx half-extent at FRUIT_SCALE
    FRUIT_MASS = 0.12                 # ~small apple; too-light shells explode on contact
    # Cap PhysX depenetration so overlapping fruit in the basket cannot launch.
    # orientation that exposes a top-down contact frame (see pick_ripe_apple);
    # rotates local +y -> world +z, so mesh center offset affects ride height
    FRUIT_Q = [0.707, 0.707, 0.0, 0.0]
    BASKET_SCALE = 1.15                # bigger opening — easier drop-in target
    # fallback mouth half-extents; the real ones come from the basket mesh in
    # load_actors. The basket is clearly rectangular (~16 x 22 cm), so a single
    # radius would call a fruit resting on the table beside it "in the basket".
    BASKET_HALF_XY = (0.078, 0.111)
    BASKET_Y = 0.0                    # table midline (toward the belts / "higher")
    BASKET_X = 0.34                   # nudged out so the bigger basket clears the belts
    # after grasp: raise the gripper this far along world +Z (in place),
    # then slide horizontally over the basket at that height, then release.
    PICK_LIFT = 0.10

    N_SLATS = 5
    APPLE_COLOR = [0.85, 0.12, 0.10]   # red
    GREEN_COLOR = [0.12, 0.62, 0.18]   # green
    BELT_COLOR = [0.18, 0.18, 0.20]
    SLAT_COLOR = [0.10, 0.10, 0.12]

    # ---- distractor fruit (Opt 2; spawn-side only; never packed/counted) ----
    DISTRACTOR_ENABLED_DEFAULT = False
    TWO_COLORS_ENABLED_DEFAULT = False
    # episode-level: always ≥1 black when enabled; this is P(second black)
    DISTRACTOR_EXTRA_PROB_DEFAULT = 0.30
    DISTRACTOR_COLOR_DEFAULT = [0.05, 0.05, 0.05]  # black; distinct from red/green
    # min center-to-center Y gap from any active same-belt real fruit, as a
    # multiple of fruit diameter (2*FRUIT_R) — "at least twice the fruit's size"
    DISTRACTOR_MIN_GAP_MULT_DEFAULT = 2.0

    # fruit type -> owning side / arm / basket (red left, green right)
    TYPE_SIDE = {"apple": "left", "green": "right"}
    TYPE_RGB = None  # filled in __init__ path via class attrs below

    @classmethod
    def _type_rgb(cls, ftype):
        return cls.APPLE_COLOR if ftype == "apple" else cls.GREEN_COLOR

    def _apply_fruit_mass_properties(self, rigid, mass=None):
        """Set mass *and* matching inertia/COM.

        ``Actor.set_mass`` alone leaves the default ~10 g inertia tensor, so a
        dropped apple still spins and contact-explodes like a light shell —
        often jumping out of the basket. Match pick_ripe_apple / play_billiard:
        solid-sphere I = 2/5 m R^2 about the collision hull centre.
        """
        if rigid is None:
            return
        m = float(self.FRUIT_MASS if mass is None else mass)
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
                r = float(getattr(self, "fruit_r", self.FRUIT_R))
                inertia = 0.4 * m * (r ** 2)
                rigid.set_inertia([inertia, inertia, inertia])
            except Exception:
                pass

    def _stabilize_fruit_rigid(self, rigid):
        """Keep stacked apples from contact-exploding (still fully dynamic)."""
        if rigid is None:
            return
        try:
            rigid.set_linear_damping(5.0)
            rigid.set_angular_damping(20.0)
            rigid.set_sleep_threshold(0.08)
            rigid.set_max_depenetration_velocity(0.4)
            rigid.set_max_linear_velocity(1.5)
            rigid.set_max_angular_velocity(12.0)
            rigid.set_solver_position_iterations(8)
            rigid.set_solver_velocity_iterations(4)
            for s in rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(4.0)
                m.set_dynamic_friction(4.0)
                m.set_restitution(0.0)
        except Exception:
            pass

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("pack_fruits", {}) or {})
        # Resolve scenario from (in order): explicit kwarg, top-level config
        # key written by the GUI temp yml, CLI/env from the launcher.
        scenario = str(
            kwags.get("scenario")
            or kwags.get("interactive_scenario")
            or os.environ.get("ROBODYNA_SCENARIO", "")
            or ""
        ).strip().lower()
        scenario_overrides = {
            "default": {"two_colors_enabled": False, "distractor_enabled": False},
            "opt1": {"two_colors_enabled": True, "distractor_enabled": False},
            "opt2": {"two_colors_enabled": False, "distractor_enabled": True},
            "opt1+2": {"two_colors_enabled": True, "distractor_enabled": True},
        }
        if scenario in scenario_overrides:
            self._cfg.update(scenario_overrides[scenario])
        self._interactive_scenario = scenario or None
        # guards: _update_kinematic_tasks runs before load_actors finishes
        self._belt_ready = False
        self._belt_running = False
        self._grasping_dist_slots = set()
        self.n_distractor_slots = 0
        self._reset_metric_state()
        super()._init_task_env_(**kwags)
        # Stream fruit from episode start (policy eval never calls play_once).
        if getattr(self, "_belt_ready", False):
            self._belt_running = True

    # --------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        # Opt 1: dual red+green with both baskets; Opt 2: black distractors
        self.two_colors_enabled = bool(
            cfg.get("two_colors_enabled", cfg.get("opt1", self.TWO_COLORS_ENABLED_DEFAULT))
        )
        _dist = cfg.get("distractor_enabled", cfg.get("opt2", self.DISTRACTOR_ENABLED_DEFAULT))
        self.distractor_enabled = bool(_dist)
        self.distractor_extra_prob = float(
            cfg.get("distractor_extra_prob",
                    cfg.get("distractor_prob", self.DISTRACTOR_EXTRA_PROB_DEFAULT))
        )
        self.distractor_color = list(cfg.get("distractor_color", self.DISTRACTOR_COLOR_DEFAULT))[:3]
        self.distractor_min_gap_mult = float(cfg.get("distractor_min_gap_mult", self.DISTRACTOR_MIN_GAP_MULT_DEFAULT))

        n_min = int(cfg.get("n_items_min", self.N_ITEMS_MIN_DEFAULT))
        n_max = int(cfg.get("n_items_max", self.N_ITEMS_MAX_DEFAULT))
        if n_max < n_min:
            n_min, n_max = n_max, n_min
        per_min = int(cfg.get("n_per_color_min", self.N_PER_COLOR_MIN_DEFAULT))
        per_max = int(cfg.get("n_per_color_max", self.N_PER_COLOR_MAX_DEFAULT))
        if per_max < per_min:
            per_min, per_max = per_max, per_min

        # Count / color mix
        if self.two_colors_enabled:
            # Each color independently U{per_min..per_max} (default 2–3), unless fixed.
            if "n_apple" in cfg or "n_green" in cfg or "n_orange" in cfg:
                n_apple = int(cfg.get("n_apple", per_min))
                n_green = int(cfg.get("n_green", cfg.get("n_orange", per_min)))
            elif "n_per_color" in cfg:
                n_per = int(cfg["n_per_color"])
                n_apple = n_green = n_per
            else:
                n_apple = int(np.random.randint(per_min, per_max + 1))
                n_green = int(np.random.randint(per_min, per_max + 1))
            n_apple = max(1, n_apple)
            n_green = max(1, n_green)
            types = (["apple"] * n_apple) + (["green"] * n_green)
            np.random.shuffle(types)
            self.active_colors = ["apple", "green"]
            self.n_items = int(n_apple + n_green)
        else:
            # Single color: U{n_min..n_max} (default 3–5).
            if "n_items" in cfg:
                self.n_items = max(1, int(cfg["n_items"]))
            elif "n_apple" in cfg or "n_green" in cfg or "n_orange" in cfg:
                n_a = int(cfg.get("n_apple", 0))
                n_g = int(cfg.get("n_green", cfg.get("n_orange", 0)))
                total = n_a + n_g
                self.n_items = total if total > 0 else int(np.random.randint(n_min, n_max + 1))
            else:
                self.n_items = int(np.random.randint(n_min, n_max + 1))
            self.n_items = max(1, int(self.n_items))
            color = str(np.random.choice(["apple", "green"]))
            force_color = cfg.get("fruit_color", cfg.get("single_color", None))
            if force_color is not None:
                fc = str(force_color).lower().strip()
                if fc in ("red", "apple"):
                    color = "apple"
                elif fc in ("green", "orange"):
                    color = "green"
            types = [color] * self.n_items
            self.active_colors = [color]

        self.item_types = [str(t) for t in types]
        self.n_apple = int(sum(1 for t in self.item_types if t == "apple"))
        self.n_green = int(sum(1 for t in self.item_types if t == "green"))
        self.n_orange = self.n_green  # legacy alias for observers / metrics

        print(
            f"[pack_fruits] scenario={getattr(self, '_interactive_scenario', None)!r} "
            f"two_colors={self.two_colors_enabled} distractor={self.distractor_enabled} "
            f"colors={self.active_colors} n_items={self.n_items} "
            f"(red={self.n_apple}, green={self.n_green})",
            flush=True,
        )

        # colored fruit always appears one-at-a-time (no dual grasp)
        self.spawn_mode = "single"
        self.pick_lift = float(cfg.get("pick_lift", self.PICK_LIFT))
        self.spawn_delay_s = float(cfg.get("spawn_delay_s", self.SPAWN_DELAY_S_DEFAULT))
        self.pair_stagger_enabled = bool(cfg.get("pair_stagger_enabled", self.PAIR_STAGGER_ENABLED_DEFAULT))
        _stagger_raw = cfg.get("pair_stagger_y", self.PAIR_STAGGER_Y_DEFAULT)
        self.pair_stagger_y_max = None if _stagger_raw is None else float(_stagger_raw)
        self._pair_stagger_y = 0.0
        self.single_wave_any_belt = bool(cfg.get("single_wave_any_belt", self.SINGLE_WAVE_ANY_BELT_DEFAULT))

        self.belt_gap = float(cfg.get("belt_gap", self.BELT_GAP_DEFAULT))
        # shared default speed; optional per-side overrides (belt_speed_left / belt_speed_right)
        default_speed = float(cfg.get("belt_speed", self.BELT_SPEED_DEFAULT))
        speed_jitter = float(cfg.get("belt_speed_jitter", self.BELT_SPEED_JITTER_DEFAULT))
        speed_jitter = float(np.clip(speed_jitter, 0.0, 0.95))
        lo, hi = 1.0 - speed_jitter, 1.0 + speed_jitter
        # Per-episode sample around each belt's nominal (±jitter, default ±20%).
        self.belt_speed = {
            "left": float(cfg.get("belt_speed_left", default_speed)) * float(np.random.uniform(lo, hi)),
            "right": float(cfg.get("belt_speed_right", default_speed)) * float(np.random.uniform(lo, hi)),
        }
        self.advance_every = int(cfg.get("advance_every", self.ADVANCE_EVERY_DEFAULT))
        self.spawn_gap = float(cfg.get("spawn_gap", self.SPAWN_GAP_DEFAULT))
        self.randomize_spawn_gap = bool(cfg.get("randomize_spawn_gap", False))
        sg_jitter = float(np.clip(abs(float(cfg.get("spawn_gap_jitter", 0.20))), 0.0, 0.95))
        if self.randomize_spawn_gap and sg_jitter > 0.0:
            self.spawn_gap = float(np.random.uniform(
                self.spawn_gap * (1.0 - sg_jitter),
                self.spawn_gap * (1.0 + sg_jitter),
            ))
        self.pick_y = float(cfg.get("pick_y", self.PICK_Y))
        self.pick_y_end = float(cfg.get("pick_y_end", self.PICK_Y_END))
        self.pick_station_y = float(cfg.get("pick_station_y", self.PICK_STATION_Y))
        self.basket_y = float(cfg.get("basket_y", cfg.get("box_y", self.BASKET_Y)))
        self.basket_scale = float(cfg.get("basket_scale", self.BASKET_SCALE))
        self.fruit_scale = float(cfg.get("fruit_scale", self.FRUIT_SCALE))
        # half-size along the axis that becomes world-z under FRUIT_Q
        s = self._APPLE_AUTHOR_SCALE * self.fruit_scale
        self.fruit_r = 0.5 * self._APPLE_EXTENT_Y * s

        z0 = 0.74 + self.table_z_bias
        self.table_top = z0
        self.belt_top_z = z0 + 2.0 * self.BELT_THICK
        # seat mesh bottom on the belt (not actor origin + sphere radius)
        z_bottom_from_pose = self._APPLE_CENTER_Y * s - 0.5 * self._APPLE_EXTENT_Y * s
        self._fruit_ride_z = self.belt_top_z - z_bottom_from_pose + 0.002

        half_gap = 0.5 * self.belt_gap
        self.belt_cx = {
            "left":  -(half_gap + self.BELT_HALF_WID),
            "right": +(half_gap + self.BELT_HALF_WID),
        }

        # ---- two belt slabs + moving slats ----
        self.belts = {}
        self.slats = {"left": [], "right": []}
        self._slat_ys = {"left": [], "right": []}
        self._slat_spacing = (2.0 * self.BELT_HALF_LEN) / self.N_SLATS
        self._slat_near = self.BELT_Y - self.BELT_HALF_LEN
        for side, cx in self.belt_cx.items():
            belt = create_box(
                scene=self,
                pose=sapien.Pose([cx, self.BELT_Y, z0 + self.BELT_THICK], [1, 0, 0, 0]),
                half_size=[self.BELT_HALF_WID, self.BELT_HALF_LEN, self.BELT_THICK],
                color=self.BELT_COLOR,
                name=f"belt_{side}",
                is_static=True,
            )
            self.belts[side] = belt
            self.add_prohibit_area(belt, padding=0.02)
            for k in range(self.N_SLATS):
                sy = self._slat_near + k * self._slat_spacing
                sl = create_box(
                    scene=self,
                    pose=sapien.Pose([cx, sy, self.belt_top_z + 0.001], [1, 0, 0, 0]),
                    half_size=[self.BELT_HALF_WID * 0.92, 0.006, 0.003],
                    color=self.SLAT_COLOR,
                    name=f"slat_{side}_{k}",
                    is_static=True,
                )
                self.slats[side].append(sl)
                self._slat_ys[side].append(sy)
                # Slats are visual-only: kinematic set_pose would fling any
                # dynamic apple resting on the belt. The static belt slab carries
                # contacts; the conveyor can keep moving while grasping.
                try:
                    for comp in sl.actor.get_components():
                        if not hasattr(comp, "get_collision_shapes"):
                            continue
                        for shape in comp.get_collision_shapes():
                            shape.set_collision_groups([0, 0, 0, 0])
                except Exception:
                    pass

        # ---- breadbaskets: only for active colors (red→left, green→right) ----
        self.basket_id = int(np.random.choice([0, 1, 2, 3, 4]))
        self.basket_x = float(cfg.get("basket_x", self.BASKET_X))
        all_centers = {
            "apple":  np.array([-self.basket_x, self.basket_y], dtype=np.float64),
            "green":  np.array([+self.basket_x, self.basket_y], dtype=np.float64),
        }
        self.basket_centers = {c: all_centers[c] for c in self.active_colors}
        self.baskets = {}
        self.basket_base_z = {}
        self.basket_top_z = {}
        self.basket_half_xy = {}
        for ftype, center in self.basket_centers.items():
            basket = create_actor(
                self,
                pose=sapien.Pose(
                    [float(center[0]), float(center[1]), z0],
                    [0.5, 0.5, 0.5, 0.5],
                ),
                modelname="076_breadbasket",
                model_id=self.basket_id,
                # Triangle mesh: a convex hull fills the bowl and squeezes
                # stacked apples until they explode out of the basket.
                convex=False,
                is_static=True,
                scale_mult=self.basket_scale,
            )
            # tint basket rim toward fruit color for readability
            tint = self._type_rgb(ftype)
            self._recolor(basket, [0.55 * tint[0] + 0.35,
                                   0.55 * tint[1] + 0.30,
                                   0.55 * tint[2] + 0.20])
            self.baskets[ftype] = basket
            self.basket_base_z[ftype] = float(basket.get_pose().p[2])
            # the [0.5,0.5,0.5,0.5] pose orientation maps local +y (the
            # mesh's authored "up"/height axis, origin at the base) onto
            # world +z, so world rim height above the origin is simply
            # extents[1] * scale[1] from the (already scale_mult-applied)
            # model config. Used so the carry/hover target clears the
            # actual basket wall instead of the old fixed guess, which sat
            # below the rim and caused the gripper to clip the basket.
            bcfg = getattr(basket, "config", None) or {}
            extents = bcfg.get("extents", [0.0, 0.7, 0.0])
            scale = bcfg.get("scale", [self.basket_scale] * 3)
            basket_height = float(extents[1]) * float(scale[1])
            if basket_height <= 0.0:
                basket_height = 0.07 * self.basket_scale
            self.basket_top_z[ftype] = self.basket_base_z[ftype] + basket_height
            # the same orientation cycles local x -> world y and local z ->
            # world x, so the mouth footprint is rectangular in world XY
            half_x = 0.5 * float(extents[2]) * float(scale[2])
            half_y = 0.5 * float(extents[0]) * float(scale[0])
            if half_x <= 0.0 or half_y <= 0.0:
                half_x, half_y = self.BASKET_HALF_XY
            self.basket_half_xy[ftype] = (half_x, half_y)
            self.add_prohibit_area(basket, padding=0.04)

        print(
            f"[pack_fruits] spawned {len(self.baskets)} basket(s): "
            f"{ {k: np.round(v, 3).tolist() for k, v in self.basket_centers.items()} }",
            flush=True,
        )

        # Two-color (Opt 1 / 1+2): each color rides the belt on its basket
        # side (red→left, green→right) so that arm never reaches across.
        # Single-color (default / Opt 2): apples may appear on either belt;
        # the color-matched arm still does the pick. Black distractors are
        # scheduled independently and may use either belt in every scenario.
        if self.two_colors_enabled:
            self.item_sides = [
                self.TYPE_SIDE.get(t, "left") for t in self.item_types
            ]
        else:
            self.item_sides = [
                str(np.random.choice(["left", "right"])) for _ in self.item_types
            ]
        self.item_arms = [
            self.TYPE_SIDE.get(t, s) for t, s in zip(self.item_types, self.item_sides)
        ]
        print(
            f"[pack_fruits] belts={self.item_sides} arms={self.item_arms}",
            flush=True,
        )
        self._pack_fail_counts = [0] * self.n_items
        self._pack_fail_limit = 3

        # stage all fruits off-table; they appear gradually on the belts
        self.items = []
        self._item_comps = []
        self._item_y = [None] * self.n_items       # None = not yet on belt / packed
        self._item_x = [0.0] * self.n_items        # lateral pose on the belt (not forced to center)
        self._item_roll = [0.0] * self.n_items
        self._spawned_mask = [False] * self.n_items
        self._spawned = 0
        self._reset_metric_state()
        self._packed = [False] * self.n_items
        self._missed = [False] * self.n_items  # rode off belt end without pack_fruits
        # gripper has reached the drop pose above this fruit's basket
        self._over_basket = [False] * self.n_items
        self._place_counts = {c: 0 for c in self.active_colors}
        self._welded = [False] * self.n_items
        self._weld_offset = [None] * self.n_items
        self._weld_arm = [None] * self.n_items
        # wave-partner tracking (legacy pair path; unused for colored solo spawn)
        self._pair_partner = [None] * self.n_items
        self._grasping_idxs = set()  # fruits mid-intercept; stay on the moving stream
        # nestable hold for the pick→above-basket→drop→return cycle; blocks new spawns
        self._spawn_hold_depth = 0
        self._stage_pose = sapien.Pose([0.0, 1.2, z0 + 0.4], [1, 0, 0, 0])

        for i, ftype in enumerate(self.item_types):
            rgb = self._type_rgb(ftype)
            fruit = create_actor(
                self,
                pose=sapien.Pose(
                    [self._stage_pose.p[0] + 0.03 * i,
                     self._stage_pose.p[1],
                     self._stage_pose.p[2]],
                    self.FRUIT_Q,
                ),
                modelname=self.FRUIT_MODEL,
                model_id=0,
                convex=True,
                is_static=False,
                scale_mult=self.fruit_scale,
            )
            self._recolor(fruit, rgb)
            try:
                fruit.actor.set_name(f"pack_fruit_{i}")
            except Exception:
                pass
            comp = None
            for c in fruit.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    comp = c
                    self._apply_fruit_mass_properties(c)
                    self._stabilize_fruit_rigid(c)
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
            self.items.append(fruit)
            self._item_comps.append(comp)

        # ---- distractor fruits: episode plans ≥1 (+ optional 2nd) when Opt 2
        # is on. Scheduled to co-appear with specific colored waves (same belt
        # behind the colored apple, or the other belt). Graspable like colored
        # fruit (physical pinch); packing one into a basket fails the episode.
        if self.distractor_enabled:
            n_dist = 1 + (1 if bool(np.random.rand() < self.distractor_extra_prob) else 0)
        else:
            n_dist = 0
        self.n_distractor_plan = int(n_dist)
        self.n_distractor_slots = int(n_dist)
        # schedule: which colored-wave index each distractor joins, and side
        wave_order = list(range(self.n_items))
        np.random.shuffle(wave_order)
        self._distractor_schedule = []
        for s in range(self.n_distractor_slots):
            wave = int(wave_order[s % len(wave_order)])
            # prefer distinct waves when two distractors
            if s > 0 and self.n_items > 1:
                wave = int(wave_order[s % len(wave_order)])
            side = str(np.random.choice(["left", "right"]))
            self._distractor_schedule.append({
                "wave": wave,
                "side": side,
                "spawned": False,
            })
        self.distractors = []
        self._distractor_comps = []
        self._distractor_y = [None] * self.n_distractor_slots
        self._distractor_roll = [0.0] * self.n_distractor_slots
        self._distractor_side = [None] * self.n_distractor_slots
        self._grasping_dist_slots = set()  # black apples mid-pinch / carry
        for s in range(self.n_distractor_slots):
            distractor = create_actor(
                self,
                pose=sapien.Pose(
                    [self._stage_pose.p[0] + 0.03 * s + 0.5,
                     self._stage_pose.p[1],
                     self._stage_pose.p[2]],
                    self.FRUIT_Q,
                ),
                modelname=self.FRUIT_MODEL,
                model_id=0,
                convex=True,
                is_static=False,
                scale_mult=self.fruit_scale,
            )
            self._recolor(distractor, self.distractor_color)
            try:
                distractor.actor.set_name(f"pack_distractor_{s}")
            except Exception:
                pass
            d_comp = None
            for c in distractor.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    d_comp = c
                    self._apply_fruit_mass_properties(c)
                    self._stabilize_fruit_rigid(c)
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
            self.distractors.append(distractor)
            self._distractor_comps.append(d_comp)

        self._step_ctr = 0
        self._pic_ctr = 0
        self._belt_ready = True
        self._belt_running = False

    # ------------------------------------------------------------- rendering
    def _recolor(self, actor, rgb):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([*list(rgb)[:3], 1.0])
                    except Exception:
                        pass

    # ------------------------------------------------------------- belt sim
    def _hide(self, idx):
        self.items[idx].actor.set_pose(sapien.Pose(
            [self._stage_pose.p[0] + 0.05 * idx,
             self._stage_pose.p[1] + 0.5,
             self.HIDE_Z],
            [1, 0, 0, 0],
        ))
        self._item_y[idx] = None

    def _spawn(self, idx, y_offset=0.0):
        """Place fruit idx onto the far end of its matching belt.

        ``y_offset`` (<=0) gives a pair-wave partner a small head start —
        it spawns that much closer to the pick station (instead of both
        fruits starting at exactly ``BELT_Y_FAR``) so the pair doesn't ride
        in perfect lockstep (see ``_spawn_wave_pair`` / ``_pair_stagger_y``).
        Only ever negative so spawns stay within the belt's physical
        length (never past ``BELT_Y_FAR``).
        """
        side = self.item_sides[idx]
        y0 = self.BELT_Y_FAR + float(y_offset)
        self._item_y[idx] = y0
        # Anywhere across the belt width — not forced to the centerline.
        half = max(0.01, float(self.BELT_HALF_WID) - float(self.fruit_r) * 0.9)
        x0 = float(self.belt_cx[side]) + float(np.random.uniform(-half, half))
        self._item_x[idx] = x0
        self._item_roll[idx] = 0.0
        self._over_basket[idx] = False
        self._latch_spawn_metric(idx)
        comp = self._item_comps[idx]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
        self._set_fruit_pose(
            idx, x0, y0, self._fruit_ride_z
        )

    def _spawn_wave_pair(self):
        """Spawn one fruit per belt side together as a pair wave.

        When ``pair_stagger_enabled`` is set, one of the two fruits (picked
        at random) gets a head start toward the pick station. The gap is
        sampled per wave from U(0, max), where max defaults to one fruit
        diameter (``2 * fruit_r``) so the pair may start aligned or offset
        by up to one fruit. ``_wait_pair_at_station`` widens its arrival
        window by the same sampled amount.
        """
        picked = []
        for i in range(self.n_items):
            if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                continue
            side = self.item_sides[i]
            if any(self.item_sides[j] == side for j in picked):
                continue
            picked.append(i)
            if len(picked) >= 2:
                break
        if self.pair_stagger_enabled:
            stagger_max = (
                self.pair_stagger_y_max
                if self.pair_stagger_y_max is not None
                else (2.0 * float(self.fruit_r))
            )
            stagger = float(np.random.uniform(0.0, max(0.0, stagger_max)))
        else:
            stagger = 0.0
        self._pair_stagger_y = stagger
        leading_k = 1 if bool(np.random.rand() < 0.5) else 0
        for k, i in enumerate(picked):
            y_offset = -stagger if k == leading_k else 0.0
            self._spawn(i, y_offset=y_offset)
            self._spawned_mask[i] = True
        if len(picked) == 2:
            self._pair_partner[picked[0]] = picked[1]
            self._pair_partner[picked[1]] = picked[0]
        self._spawned = int(sum(self._spawned_mask))

    def _active_pair_partner(self, idx):
        """Return idx's still-outstanding wave-partner, or None.

        Lets the caller keep routing a staggered pair through ``_pack_pair``
        (which tolerates a bounded arrival gap via ``_wait_pair_at_station``)
        even when the head-started fruit enters the ready window well
        before its partner does.
        """
        j = self._pair_partner[idx]
        if j is None:
            return None
        if self._packed[j] or self._missed[j] or j in self._grasping_idxs:
            return None
        if not self._spawned_mask[j] or self._item_y[j] is None:
            return None
        return j

    def _clamp_fruit_belt_x(self, side, x):
        """Keep a fruit X within its belt slab (any lateral position, not just center)."""
        cx = float(self.belt_cx[side])
        half = max(0.01, float(self.BELT_HALF_WID) - float(self.fruit_r) * 0.85)
        return float(np.clip(float(x), cx - half, cx + half))

    def _set_fruit_pose(self, idx, x, y, z, roll=0.0):
        # keep the authored grasp orientation (FRUIT_Q); optional spin about world -y
        # is composed on top so the contact frame stays top-down-approachable
        base_q = np.array(self.FRUIT_Q, dtype=np.float64)
        if abs(roll) > 1e-6:
            # spin about world x (belt travel axis) while preserving top-down graspability poorly;
            # use identity spin — visual motion comes from translation + slats
            q = base_q
        else:
            q = base_q
        pose = sapien.Pose([float(x), float(y), float(z)], q.tolist())
        # always write the entity pose so grasp/contact queries see the true location
        self.items[idx].actor.set_pose(pose)
        if idx < len(getattr(self, "_item_x", [])):
            self._item_x[idx] = float(x)
        comp = self._item_comps[idx]
        if comp is not None:
            try:
                comp.set_kinematic_target(pose)
            except Exception:
                pass

    def _advance_slats(self):
        span = self.N_SLATS * self._slat_spacing
        for side, cx in self.belt_cx.items():
            speed = self.belt_speed[side]
            for k in range(self.N_SLATS):
                y = self._slat_ys[side][k] - speed
                if y < self._slat_near:
                    y += span
                self._slat_ys[side][k] = y
                self.slats[side][k].actor.set_pose(
                    sapien.Pose([cx, y, self.belt_top_z + 0.001], [1, 0, 0, 0])
                )

    def _fruit_blocks_spawn(self, idx):
        """True while this fruit must delay the next spawn wave.

        Blocks from appearance until it is packed into a basket or has left the
        belt. Carried / welded fruit always blocks (not packed yet).
        """
        if not self._spawned_mask[idx]:
            return False
        if self._packed[idx] or self._missed[idx]:
            return False
        return True

    def _can_spawn_next(self):
        """Next wave only when no fruit is outstanding and no pack cycle is running."""
        if int(getattr(self, "_spawn_hold_depth", 0)) > 0:
            return False
        if self._has_active_fruit():
            return False
        return True

    def _has_active_fruit(self):
        return any(self._fruit_blocks_spawn(i) for i in range(self.n_items))

    def _begin_spawn_hold(self):
        self._spawn_hold_depth = int(getattr(self, "_spawn_hold_depth", 0)) + 1

    def _end_spawn_hold(self):
        self._spawn_hold_depth = max(0, int(getattr(self, "_spawn_hold_depth", 0)) - 1)

    # -------------------------------------------------------- distractors
    def _hide_distractor(self, slot):
        """Off-table park (mirrors ``_hide``, but for a distractor slot)."""
        self._grasping_dist_slots.discard(slot)
        self.distractors[slot].actor.set_pose(sapien.Pose(
            [self._stage_pose.p[0] + 0.05 * slot + 0.5,
             self._stage_pose.p[1] + 0.5,
             self.HIDE_Z],
            [1, 0, 0, 0],
        ))
        self._distractor_y[slot] = None
        self._distractor_side[slot] = None

    def _set_distractor_pose(self, slot, x, y, z):
        q = np.array(self.FRUIT_Q, dtype=np.float64)
        pose = sapien.Pose([float(x), float(y), float(z)], q.tolist())
        self.distractors[slot].actor.set_pose(pose)
        comp = self._distractor_comps[slot]
        if comp is not None:
            try:
                comp.set_kinematic_target(pose)
            except Exception:
                pass

    def _spawn_distractor(self, slot=None, side=None, prefer_behind_colored=True):
        """Put one distractor on a belt.

        ``side`` may be forced (Opt 2 schedule). When a colored fruit is
        already on that belt, place the black apple behind it (farther +y)
        with ``distractor_min_gap_mult`` × diameter clearance so they ride
        together without overlapping. Independent of colored-fruit spawn-gating
        / success bookkeeping; a currently pinched distractor is not respawned.
        """
        if slot is None:
            grasping = getattr(self, "_grasping_dist_slots", set()) or set()
            for s in range(self.n_distractor_slots):
                if self._distractor_y[s] is not None or s in grasping:
                    continue
                try:
                    z = float(self.distractors[s].get_pose().p[2])
                except Exception:
                    z = self.HIDE_Z
                if z > 0.0:
                    continue  # on table / in a basket — do not recycle the slot
                slot = s
                break
        if slot is None:
            return  # every slot busy; skip
        if self._distractor_y[slot] is not None:
            return
        if slot in getattr(self, "_grasping_dist_slots", set()):
            return  # currently pinched / carried — do not yank back onto the belt

        if side is None:
            side = str(np.random.choice(["left", "right"]))
        side = str(side)
        min_gap = float(self.distractor_min_gap_mult) * (2.0 * self.fruit_r)
        active_ys = [
            self._item_y[i] for i in range(self.n_items)
            if self.item_sides[i] == side and self._item_y[i] is not None
        ]
        if active_ys and prefer_behind_colored:
            # behind = farther from robot = higher y
            closest_to_far = max(active_ys)
            candidate = closest_to_far + min_gap
            if candidate <= self.BELT_Y_FAR + 1e-9:
                y0 = candidate
            else:
                # not enough room behind — put in front with gap instead
                y0 = closest_to_far - min_gap
        elif active_ys:
            closest_to_far = max(active_ys)
            candidate = closest_to_far + min_gap
            y0 = candidate if candidate <= self.BELT_Y_FAR + 1e-9 else closest_to_far - min_gap
        else:
            y0 = self.BELT_Y_FAR - float(np.random.uniform(0.0, 0.03))

        self._distractor_side[slot] = side
        self._distractor_y[slot] = y0
        self._distractor_roll[slot] = 0.0
        comp = self._distractor_comps[slot]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
        self._set_distractor_pose(slot, self.belt_cx[side], y0, self._fruit_ride_z)
        if bool(os.environ.get("PACKING_DEBUG")):
            gap_note = (f"gap_to_nearest={min(abs(y0 - y) for y in active_ys):.4f}"
                        if active_ys else "no active real fruit on belt")
            print(f"[pack_fruits]  distractor_{slot} spawn side={side} y0={y0:.4f} "
                  f"min_gap_req={min_gap:.4f} {gap_note}", flush=True)

    def _spawn_scheduled_distractors(self, wave_idx):
        """Spawn any black apples scheduled to co-appear with this colored wave."""
        if not getattr(self, "distractor_enabled", False):
            return
        for slot, sched in enumerate(self._distractor_schedule):
            if sched.get("spawned"):
                continue
            if int(sched["wave"]) != int(wave_idx):
                continue
            self._spawn_distractor(slot=slot, side=sched.get("side"))
            sched["spawned"] = True

    def _maybe_spawn_distractor(self):
        """Legacy no-op hook (distractors are wave-scheduled)."""
        return

    def _advance_distractors(self):
        """Belt-ride for black apples still on a conveyor.

        Pinched / carried distractors leave the stream (``_distractor_y`` is
        None, or the slot is in ``_grasping_dist_slots``) and are not
        despawned. Unpicked fruit still rides off the near end and hides.
        """
        grasping = getattr(self, "_grasping_dist_slots", set()) or set()
        for s in range(self.n_distractor_slots):
            if self._distractor_y[s] is None or s in grasping:
                continue
            side = self._distractor_side[s]
            speed = self.belt_speed[side]
            self._distractor_y[s] -= speed
            if self._distractor_y[s] < self.BELT_Y_NEAR:
                if bool(os.environ.get("PACKING_DEBUG")):
                    print(f"[pack_fruits]  distractor_{s} left belt — despawn", flush=True)
                self._hide_distractor(s)
                continue
            self._distractor_roll[s] += speed / max(self.fruit_r, 1e-4)
            x = self.belt_cx[side]
            self._set_distractor_pose(s, x, self._distractor_y[s], self._fruit_ride_z)

    def _despawn_off_belt(self, idx):
        """Fruit reached the near end without a pick — hide it and free the wave."""
        import os
        if self._packed[idx] or self._missed[idx]:
            return
        self._missed[idx] = True
        self._over_basket[idx] = False
        self._item_y[idx] = None
        self._welded[idx] = False
        self._grasping_idxs.discard(idx)
        self._hide(idx)
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[pack_fruits]  {self.item_types[idx]}_{idx} left belt — despawn",
                  flush=True)

    def _maybe_spawn(self):
        """Spawn the next colored apple only after the previous one is clear.

        Always one colored fruit at a time. Black distractors scheduled for
        this wave may co-appear (same belt behind, or the other belt).
        """
        import os
        if self._spawned >= self.n_items:
            return
        if not self._can_spawn_next():
            return

        for i in range(self.n_items):
            if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                continue
            wave_idx = int(self._spawned)
            self._spawn(i)
            self._spawned_mask[i] = True
            self._spawned = int(sum(self._spawned_mask))
            self._spawn_scheduled_distractors(wave_idx)
            if bool(os.environ.get("PACKING_DEBUG")):
                print(f"[pack_fruits]  spawn wave={wave_idx} "
                      f"{self.item_types[i]}_{i} side={self.item_sides[i]}",
                      flush=True)
            break

    def _advance_stream(self):
        self._resolve_freed_orphans()
        self._maybe_spawn()

        for i in range(self.n_items):
            if (not self._spawned_mask[i] or self._packed[i] or self._missed[i]
                    or self._welded[i] or self._item_y[i] is None):
                continue
            side = self.item_sides[i]
            speed = self.belt_speed[side]
            # continuous motion — never park / pause on the belt
            self._item_y[i] -= speed
            if self._item_y[i] < self.BELT_Y_NEAR and i not in self._grasping_idxs:
                # leave the belt (do not wrap / reappear)
                self._despawn_off_belt(i)
                continue
            self._item_roll[i] += speed / max(self.fruit_r, 1e-4)
            x = self._item_x[i] if self._item_x[i] is not None else self.belt_cx[side]
            self._set_fruit_pose(
                i, float(x), self._item_y[i], self._fruit_ride_z,
                roll=self._item_roll[i],
            )

        self._advance_distractors()

    def _ee_pos(self, arm):
        p = (self.robot.get_left_ee_pose() if arm == "left"
             else self.robot.get_right_ee_pose())
        return np.array(p[:3], dtype=float)

    def _ee_pose_full(self, arm):
        """Full 6-DOF planning-EE pose (position + orientation), as a sapien.Pose."""
        p = (self.robot.get_left_ee_pose() if arm == "left"
             else self.robot.get_right_ee_pose())
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _tcp_pos(self, arm):
        """Gripper-center pose (fingertips), not the retracted planning EE frame."""
        p = (self.robot.get_left_tcp_pose() if arm == "left"
             else self.robot.get_right_tcp_pose())
        return np.array(p[:3], dtype=float)

    def _mark_fruit_held(self, idx, arm):
        """Bookkeeping for a friction-held fruit (no EE weld / pose glue)."""
        arm_name = "left" if str(arm) == "left" else "right"
        self._weld_arm[idx] = arm_name
        self._weld_offset[idx] = None
        self._item_y[idx] = None
        self._welded[idx] = True
        self._set_fruit_collision_enabled(idx, True)

    def _update_welded_fruits(self):
        """No-op: fruit is held by jaw friction, not glued to the EE."""
        return

    def _release_fruit(self, idx):
        """Clear held bookkeeping; fruit drops under gravity when jaws open."""
        if bool(os.environ.get("PACKING_DEBUG")):
            p = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[pack_fruits]  RELEASE fruit_{idx} step={self._step_ctr} p={p.round(4)}", flush=True)
        self._latch_release_offset(idx)
        self._welded[idx] = False
        self._weld_arm[idx] = None
        self._weld_offset[idx] = None
        self._set_fruit_collision_enabled(idx, True)
        self._enable_fruit_gravity(idx)
        self._calm_fruit(idx, damping=(4.0, 16.0))
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_max_depenetration_velocity(0.4)
                rigid.set_max_linear_velocity(1.5)
            except Exception:
                pass

    def _fruit_over_belt(self, idx, margin=0.03):
        """Return belt side if fruit XY sits on a conveyor slab, else None."""
        p = np.array(self.items[idx].get_pose().p, dtype=float)
        if p[1] < self.BELT_Y_NEAR - margin or p[1] > self.BELT_Y_FAR + margin:
            return None
        best_side, best_dx = None, 1e9
        for side, cx in self.belt_cx.items():
            dx = abs(float(p[0]) - float(cx))
            if dx <= self.BELT_HALF_WID + margin and dx < best_dx:
                best_side, best_dx = side, dx
        return best_side

    def _set_fruit_collision_enabled(self, idx, enabled: bool):
        """Toggle PhysX contacts on a colored apple."""
        rigid = self._item_comps[idx] if idx < len(self._item_comps) else None
        if rigid is None:
            return
        groups = [1, 1, 0, 0] if enabled else [0, 0, 0, 0]
        try:
            for shape in rigid.get_collision_shapes():
                shape.set_collision_groups(list(groups))
        except Exception:
            pass

    def _calm_fruit(self, idx, damping=(0.8, 4.0)):
        """Zero velocities and set damping so a pinch/drop is not a throw."""
        rigid = self._item_comps[idx] if idx < len(self._item_comps) else None
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_linear_damping(float(damping[0]))
            rigid.set_angular_damping(float(damping[1]))
        except Exception:
            pass

    def _free_fruit_for_physical_grasp(self, idx):
        """Leave the belt at the current pose as a dynamic body (no teleport).

        Interactive hold is friction/contact only — same idea as pick_ripe_apple.
        Gravity stays off until the jaws finish closing. Pose is unchanged.
        """
        self._item_y[idx] = None
        self._welded[idx] = False
        self._weld_arm[idx] = None
        self._weld_offset[idx] = None
        self._over_basket[idx] = False
        self._set_fruit_collision_enabled(idx, True)
        rigid = self._item_comps[idx]
        if rigid is None:
            return
        try:
            pose = self.items[idx].get_pose()
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(True)
            self.items[idx].actor.set_pose(pose)
            self._calm_fruit(idx, damping=(1.2, 6.0))
        except Exception:
            pass

    def _enable_fruit_gravity(self, idx):
        """Turn gravity on so the closed gripper must hold the apple by contact."""
        rigid = self._item_comps[idx]
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            self._calm_fruit(idx, damping=(0.8, 4.0))
        except Exception:
            pass

    def _fruit_held_by_gripper(self, idx) -> bool:
        """True while a gripper still contacts this colored apple."""
        if idx < 0 or idx >= len(self.items):
            return False
        try:
            name = self.items[idx].get_name()
            return len(self.get_gripper_actor_contact_position(name)) > 0
        except Exception:
            return False

    def _set_distractor_collision_enabled(self, slot, enabled: bool):
        """Toggle PhysX contacts on a black apple."""
        comps = getattr(self, "_distractor_comps", [])
        rigid = comps[slot] if slot < len(comps) else None
        if rigid is None:
            return
        groups = [1, 1, 0, 0] if enabled else [0, 0, 0, 0]
        try:
            for shape in rigid.get_collision_shapes():
                shape.set_collision_groups(list(groups))
        except Exception:
            pass

    def _calm_distractor(self, slot, damping=(0.8, 4.0)):
        """Zero velocities and set damping so a pinch/drop is not a throw."""
        comps = getattr(self, "_distractor_comps", [])
        rigid = comps[slot] if slot < len(comps) else None
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_linear_damping(float(damping[0]))
            rigid.set_angular_damping(float(damping[1]))
        except Exception:
            pass

    def _free_distractor_for_physical_grasp(self, slot):
        """Leave the belt at the current pose as a dynamic body (no teleport).

        Same physical pinch as colored fruit: gravity stays off until the
        jaws finish closing. Pose is unchanged.
        """
        self._distractor_y[slot] = None
        self._set_distractor_collision_enabled(slot, True)
        rigid = self._distractor_comps[slot] if slot < len(self._distractor_comps) else None
        if rigid is None:
            return
        try:
            pose = self.distractors[slot].get_pose()
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(True)
            self.distractors[slot].actor.set_pose(pose)
            self._calm_distractor(slot, damping=(1.2, 6.0))
        except Exception:
            pass

    def _enable_distractor_gravity(self, slot):
        """Turn gravity on so the closed gripper must hold the apple by contact."""
        rigid = self._distractor_comps[slot] if slot < len(self._distractor_comps) else None
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            self._calm_distractor(slot, damping=(0.8, 4.0))
        except Exception:
            pass

    def _distractor_held_by_gripper(self, slot) -> bool:
        """True while a gripper still contacts this black apple."""
        dists = getattr(self, "distractors", []) or []
        if slot < 0 or slot >= len(dists):
            return False
        try:
            name = dists[slot].get_name()
            return len(self.get_gripper_actor_contact_position(name)) > 0
        except Exception:
            return False

    def _distractor_over_belt(self, slot, margin=0.03):
        """Return belt side if this black apple's XY sits on a conveyor, else None."""
        dists = getattr(self, "distractors", []) or []
        if slot < 0 or slot >= len(dists):
            return None
        p = np.array(dists[slot].get_pose().p, dtype=float)
        if p[1] < self.BELT_Y_NEAR - margin or p[1] > self.BELT_Y_FAR + margin:
            return None
        best_side, best_dx = None, 1e9
        for side, cx in self.belt_cx.items():
            dx = abs(float(p[0]) - float(cx))
            if dx <= self.BELT_HALF_WID + margin and dx < best_dx:
                best_side, best_dx = side, dx
        return best_side

    def _reseat_distractor_on_belt(self, slot, side=None, y=None, x=None):
        """Put a free black apple back on the kinematic belt stream."""
        p = np.array(self.distractors[slot].get_pose().p, dtype=float)
        if side is None:
            side = self._distractor_over_belt(slot) or self._distractor_side[slot]
        side = str(side or "left")
        if side not in self.belt_cx:
            side = "left"
        if y is None:
            y = float(np.clip(p[1], self.BELT_Y_NEAR + 0.02, self.BELT_Y_FAR))
        if x is None:
            x = float(p[0])
        x = self._clamp_fruit_belt_x(side, x)
        self._distractor_side[slot] = side
        self._distractor_y[slot] = float(y)
        self._distractor_roll[slot] = 0.0
        self._grasping_dist_slots.discard(slot)
        self._set_distractor_collision_enabled(slot, True)
        rigid = self._distractor_comps[slot] if slot < len(self._distractor_comps) else None
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(5.0)
                rigid.set_angular_damping(20.0)
            except Exception:
                pass
        self._set_distractor_pose(slot, x, self._distractor_y[slot], self._fruit_ride_z)
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[pack_fruits]  reseat distractor_{slot} on {side} belt "
                  f"x={x:.3f} y={y:.3f}", flush=True)
        return True

    def _mark_distractor_table_rest(self, slot):
        """Leave a dropped black apple on the table (no belt reset)."""
        self._distractor_y[slot] = None
        self._grasping_dist_slots.discard(slot)
        self._set_distractor_collision_enabled(slot, True)
        rigid = self._distractor_comps[slot] if slot < len(self._distractor_comps) else None
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                self._calm_distractor(slot, damping=(1.5, 8.0))
            except Exception:
                pass

    def interactive_ee_z_ceiling(self, side, pose):
        """Allow raising above the home EE while carrying a freed apple.

        UniversalRobotControls otherwise caps Q/E at the captured origin height.
        """
        side = "left" if str(side) == "left" else "right"
        holding = False
        for i in range(int(getattr(self, "n_items", 0))):
            if self._packed[i] or self._missed[i]:
                continue
            if self._item_y[i] is not None:
                continue
            if self._welded[i] and str(self._weld_arm[i]) == side:
                holding = True
                break
            if i in getattr(self, "_grasping_idxs", set()):
                try:
                    tcp = self._tcp_pos(side)
                    fp = np.array(self.items[i].get_pose().p, dtype=float)
                    if float(np.linalg.norm(fp[:2] - tcp[:2])) < 0.14:
                        holding = True
                        break
                except Exception:
                    holding = True
                    break
        if not holding:
            for s in getattr(self, "_grasping_dist_slots", set()) or ():
                try:
                    tcp = self._tcp_pos(side)
                    fp = np.array(self.distractors[s].get_pose().p, dtype=float)
                    if float(np.linalg.norm(fp[:2] - tcp[:2])) < 0.14:
                        holding = True
                        break
                except Exception:
                    holding = True
                    break
        if not holding:
            return None
        controls = getattr(self, "_interactive_robot_controls", None)
        home = None
        if controls is not None:
            home = getattr(controls, "_origin_pose", {}).get(side)
        home_z = float(home[2]) if home is not None else float(pose[2])
        rim = 0.0
        if getattr(self, "basket_top_z", None):
            rim = max(float(z) for z in self.basket_top_z.values())
        return max(home_z + 0.08, rim + 0.18, float(pose[2]) + 0.12)

    def _reseat_on_belt(self, idx, side=None, y=None, x=None, keep_grasping=False):
        """Put a free fruit back on the kinematic belt stream (same ride physics).

        Keeps the fruit's current lateral X when possible — not snapped to the
        belt centerline.

        ``keep_grasping``: leave ``idx`` in ``_grasping_idxs`` (failed mid-pack
        pinch) so orphan cleanup does not yoink the apple to the far end.
        """
        if side is None:
            if self.two_colors_enabled:
                side = self.TYPE_SIDE.get(
                    self.item_types[idx], self.item_sides[idx]
                )
            else:
                side = self._fruit_over_belt(idx) or self.item_sides[idx]
        side = str(side)
        if side not in self.belt_cx:
            side = "left"
        p = np.array(self.items[idx].get_pose().p, dtype=float)
        if y is None:
            y = float(np.clip(p[1], self.BELT_Y_NEAR + 0.02, self.BELT_Y_FAR))
        if x is None:
            x = float(p[0])
        x = self._clamp_fruit_belt_x(side, x)
        self.item_sides[idx] = side
        self.item_arms[idx] = self.TYPE_SIDE.get(self.item_types[idx], side)
        self._item_y[idx] = float(y)
        self._item_x[idx] = float(x)
        self._item_roll[idx] = 0.0
        self._welded[idx] = False
        self._weld_arm[idx] = None
        self._over_basket[idx] = False
        self._packed[idx] = False
        self._missed[idx] = False
        if not keep_grasping:
            self._grasping_idxs.discard(idx)
        self._set_fruit_collision_enabled(idx, True)
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(5.0)
                rigid.set_angular_damping(20.0)
            except Exception:
                pass
        self._set_fruit_pose(idx, x, self._item_y[idx], self._fruit_ride_z)
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[pack_fruits]  reseat fruit_{idx} on {side} belt "
                  f"x={x:.3f} y={y:.3f}", flush=True)
        return True

    def _mark_table_rest(self, idx):
        """Leave a dropped apple on the table (resolved miss — no belt reset)."""
        self._missed[idx] = True
        self._packed[idx] = False
        self._over_basket[idx] = False
        self._item_y[idx] = None
        self._welded[idx] = False
        self._weld_arm[idx] = None
        self._grasping_idxs.discard(idx)
        self._set_fruit_collision_enabled(idx, True)
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                self._calm_fruit(idx, damping=(1.5, 8.0))
            except Exception:
                pass

    def _resolve_freed_orphans(self):
        """Reseat freed fruits that are not held/packed/missed (unblocks spawn)."""
        held = set(getattr(self, "_grasping_idxs", set()) or ())
        for i in range(int(self.n_items)):
            if not self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                continue
            if self._welded[i] or self._item_y[i] is not None:
                continue
            if i in held:
                continue
            # Limbo: freed, not tracked by interactive hold — put back on the belt.
            self._reseat_on_belt(i, side=self.item_sides[i], y=float(self.BELT_Y_FAR))
            if bool(os.environ.get("PACKING_DEBUG")):
                print(f"[pack_fruits]  orphan fruit_{i} reseated (spawn unblock)", flush=True)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_belt_ready", False):
            return
        # held fruit rides jaw friction (no EE weld / pose glue)
        self._update_welded_fruits()
        if bool(os.environ.get("PACKING_DEBUG")):
            self._accumulate_basket_contacts()
        if not getattr(self, "_belt_running", False):
            return
        self._step_ctr += 1
        if self._step_ctr % max(1, self.advance_every) == 0:
            self._advance_slats()
            self._advance_stream()

    def _belt_dwell(self, steps):
        """Advance belts + stream for `steps` while recording frames. Belt stays on."""
        self._belt_running = True
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % self.save_freq == 0):
                self._take_picture()
            self._pic_ctr += 1

    # ------------------------------------------------------------- pack_fruits
    def _xy_inside_basket(self, xy, ftype, margin=0.0):
        """True when a world XY sits inside a basket's rectangular mouth."""
        c = self.basket_centers[ftype]
        half_x, half_y = self.basket_half_xy.get(ftype, self.BASKET_HALF_XY)
        d = np.abs(np.asarray(xy, dtype=np.float64)[:2] - c)
        return bool(d[0] <= max(0.0, half_x - margin)
                    and d[1] <= max(0.0, half_y - margin))

    def _basket_for_target(self, target_xy):
        """Basket a drop target belongs to (may not be the fruit's own color)."""
        t = np.asarray(target_xy, dtype=np.float64)[:2]
        return min(
            self.basket_centers,
            key=lambda k: float(np.linalg.norm(t - self.basket_centers[k])),
        )

    def _pose_in_basket(self, p, ftype):
        """True when world pose ``p`` sits in ``ftype``'s basket volume."""
        in_xy = self._xy_inside_basket(p[:2], ftype)
        above = p[2] >= (self.basket_base_z[ftype] - 0.02)
        stack_h = 0.18 + 2.2 * float(self.fruit_r) * max(1, int(self.n_items) - 1)
        below = p[2] <= (self.basket_base_z[ftype] + stack_h)
        return bool(in_xy and above and below)

    def _fruit_in_basket(self, idx):
        ftype = self.item_types[idx]
        p = np.array(self.items[idx].get_pose().p, dtype=np.float64)
        return self._pose_in_basket(p, ftype)

    def _distractor_in_any_basket(self, slot):
        """True when a visible black apple rests in any spawned basket."""
        if slot < 0 or slot >= len(getattr(self, "distractors", []) or []):
            return False
        p = np.array(self.distractors[slot].get_pose().p, dtype=np.float64)
        if float(p[2]) < 0.0:
            return False
        for ftype in getattr(self, "baskets", {}) or {}:
            if self._pose_in_basket(p, ftype):
                return True
        return False

    def _mark_packed(self, idx, freeze=False):
        """Resolve a fruit. Keep it dynamic in the basket so later apples can
        stack on contact instead of exploding off a kinematic pin.
        """
        ftype = self.item_types[idx]
        if not self._packed[idx]:
            self._place_counts[ftype] = self._place_counts[ftype] + 1
        self._latch_pack_metric(idx)
        self._packed[idx] = True
        self._missed[idx] = False
        self._over_basket[idx] = True
        self._item_y[idx] = None
        self._welded[idx] = False
        self._weld_arm[idx] = None
        self._grasping_idxs.discard(idx)
        self._set_fruit_collision_enabled(idx, True)
        rigid = self._item_comps[idx]
        if rigid is None:
            return
        try:
            if freeze:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                self._calm_fruit(idx, damping=(5.0, 20.0))
            else:
                # Keep fruit↔fruit / fruit↔basket contacts so apples stack.
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                self._calm_fruit(idx, damping=(8.0, 24.0))
                try:
                    rigid.set_max_depenetration_velocity(0.4)
                    rigid.set_max_linear_velocity(1.5)
                except Exception:
                    pass
        except Exception:
            pass

    def _restore_fruit_stream_pose(self, idx):
        """Put actor back on the live belt pose after a temporary planning teleport."""
        if self._item_y[idx] is None:
            return
        side = self.item_sides[idx]
        x = self._item_x[idx] if self._item_x[idx] is not None else self.belt_cx[side]
        self._set_fruit_pose(
            idx, float(x), self._item_y[idx], self._fruit_ride_z,
            roll=self._item_roll[idx],
        )

    def _basket_target_xy(self, idx, slot_offset=0, basket=None):
        """Drop pose in a basket; ``basket`` overrides the fruit's own color.

        Slots are spread over the mouth footprint rather than a fixed pattern:
        the basket is narrow along world X, so a fixed 2.8 cm offset there put
        the fruit against the wall with no room for the slide's residual error.
        """
        ftype = basket or self.item_types[idx]
        slot = self._place_counts[ftype] + int(slot_offset)
        c = self.basket_centers[ftype]
        half_x, half_y = self.basket_half_xy.get(ftype, self.BASKET_HALF_XY)
        sx = max(0.0, 0.45 * (half_x - self.fruit_r))
        sy = max(0.0, 0.45 * (half_y - self.fruit_r))
        offsets = [
            (0.0, 0.0), (sx, sy), (-sx, sy), (sx, -sy), (-sx, -sy),
        ]
        ox, oy = offsets[slot % len(offsets)]
        return c + np.array([ox, oy], dtype=float)

    def _plan_station_pre(self, idx, arm):
        side = self.item_sides[idx]
        cx = self._item_x[idx]
        if cx is None:
            cx = self.belt_cx[side]
        # Hover slightly upstream so the TCP meets the apple instead of sitting
        # downstream of it (right-arm Y lag was ~3 cm with station at pick_y).
        hover_y = float(self.pick_station_y) + 0.03
        self._set_fruit_pose(idx, float(cx), hover_y, self._fruit_ride_z)
        try:
            # hover close enough that a short descend can reach the attach distance
            pre_pose, _ = self.choose_grasp_pose(
                self.items[idx], arm_tag=arm, pre_dis=0.05, target_dis=0.0,
            )
        finally:
            self._restore_fruit_stream_pose(idx)
        return pre_pose

    def _intercept_lead_y(self, side, reach_steps=0, close_steps=16):
        """How far upstream (−y) the fruit travels during approach + close.

        Belt advances ``belt_speed`` every ``advance_every`` physics steps.
        With the arm already at the ready hover, ``reach_steps`` should be ~0 —
        oversized lead parked the TCP downstream and the pinch started late.
        """
        speed = max(float(self.belt_speed.get(side, self.BELT_SPEED_DEFAULT)), 1e-6)
        every = max(1, int(self.advance_every))
        travel = speed * float(int(reach_steps) + int(close_steps)) / float(every)
        return float(np.clip(travel, 0.012, 0.05))

    def _wait_fruit_at_station(self, idx, side):
        """Dwell until the fruit reaches the ready TCP — then jaws can shut."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        speed = max(self.belt_speed[side], 1e-6)
        arrive_lead = self._intercept_lead_y(side, reach_steps=12, close_steps=16)
        ftype = self.item_types[idx]
        arm_name = self.TYPE_SIDE.get(ftype, side)
        max_wait = int((self.BELT_Y_FAR - self.BELT_Y_NEAR) / speed) + 80
        for _ in range(max_wait):
            y = self._item_y[idx]
            if y is None:
                return False
            if y < self.pick_y_end:
                if dbg:
                    print(f"[pack_fruits]  fruit passed station y={y:.3f}", flush=True)
                return False
            tcp_y = float(self._tcp_pos(arm_name)[1])
            # Trigger off the real TCP, not the nominal station y.
            if y <= tcp_y + arrive_lead:
                return True
            self._belt_dwell(max(1, self.advance_every))
        return False

    def _wait_pair_at_station(self, idx_l, idx_r):
        """Dwell until both fruits near the pick station together.

        Widens its arrival window by the last sampled ``_pair_stagger_y`` when
        the pair-gap option is on, so a staggered pair (one fruit deliberately
        trailing the other on the belt — see ``_spawn_wave_pair``) is still
        treated as "arrived together" instead of forcing extra dwell time that
        could carry the lead fruit past ``pick_y_end``.
        """
        speed = max(min(self.belt_speed.values()), 1e-6)
        stagger = self._pair_stagger_y if self.pair_stagger_enabled else 0.0
        arrive_lead = self._intercept_lead_y(
            "left" if self.belt_speed["left"] <= self.belt_speed["right"] else "right"
        ) + stagger
        max_wait = int((self.BELT_Y_FAR - self.BELT_Y_NEAR) / speed) + 100
        for _ in range(max_wait):
            yl, yr = self._item_y[idx_l], self._item_y[idx_r]
            if yl is None or yr is None:
                return False
            if yl < self.pick_y_end or yr < self.pick_y_end:
                return False
            if yl <= self.pick_station_y + arrive_lead and yr <= self.pick_station_y + arrive_lead:
                return True
            self._belt_dwell(max(1, self.advance_every))
        return False

    def _plan_final_grasp(self, idx, arm, lead_y=0.02):
        """Grasp pose at the fruit, with a small downstream lead for belt motion.

        ``lead_y`` shifts the planning pose toward the robot (−y) so the TCP
        arrives where the still-moving apple will be — fruit is never paused.
        """
        if self._item_y[idx] is None:
            return None
        side = self.item_sides[idx]
        x = self._item_x[idx]
        if x is None:
            x = self.belt_cx[side]
        y = float(self._item_y[idx]) - float(lead_y)
        roll = float(self._item_roll[idx])
        self._set_fruit_pose(idx, float(x), y, self._fruit_ride_z, roll=roll)
        try:
            _, grasp_pose = self.choose_grasp_pose(
                self.items[idx], arm_tag=arm, pre_dis=0.05, target_dis=0.0,
            )
        except Exception:
            grasp_pose = None
        finally:
            self._restore_fruit_stream_pose(idx)
        return grasp_pose

    def _tcp_near_fruit(self, idx, arm, xy_tol=None, z_max=None):
        """True if the gripper has approached close enough to pinch."""
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
        dz = float(tcp[2] - fp[2])
        tol = float(self.ATTACH_XY if xy_tol is None else xy_tol)
        z_hi = float(self.ATTACH_Z_MAX if z_max is None else z_max)
        return xy <= tol and -0.02 <= dz <= z_hi

    def _fruit_gripper_dist(self, idx, arm):
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
        dz = float(tcp[2] - fp[2])
        return xy, dz

    def _settle_grasp_contacts(self, steps=None):
        n = int(self.GRASP_SETTLE_STEPS if steps is None else steps)
        for j in range(n):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _set_gripper_fast(self, arm, target_pos, n_steps=12):
        """Drive gripper joints over a few physics steps (no 200-step Action)."""
        arm_name = "left" if str(arm) == "left" else "right"
        now = float(
            self.robot.get_left_gripper_val() if arm_name == "left"
            else self.robot.get_right_gripper_val()
        )
        n = max(4, int(n_steps))
        target = float(target_pos)
        vals = np.linspace(now, target, n, dtype=float)
        per_step = (target - now) / float(n)
        for i in range(n):
            self.robot.set_gripper(vals[i], arm_name, per_step)
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _close_gripper_tracking_fruit(self, idx, arm, target_pos=0.0, n_steps=16):
        """Close the gripper while TCP tracks / centers on the still-moving apple.

        The apple stays on the kinematic stream the whole time (no pause).
        Close is short (default 16) so jaws shut before the fruit slides through —
        the default Action close is 200 steps and was the main timing lag.

        Targets the TCP (fingertips) with closed-loop XY/Z correction and a small
        −y lead for remaining close steps.
        """
        arm_name = "left" if str(arm) == "left" else "right"
        side = self.item_sides[idx]
        speed = max(float(self.belt_speed.get(side, self.BELT_SPEED_DEFAULT)), 0.0)
        every = max(1, int(self.advance_every))
        now = float(
            self.robot.get_left_gripper_val() if arm_name == "left"
            else self.robot.get_right_gripper_val()
        )
        n = max(8, int(n_steps))
        target = float(target_pos)
        vals = np.linspace(now, target, n, dtype=float)
        per_step = (target - now) / float(n)
        grip = {"num_step": n, "per_step": per_step, "result": vals}

        ee0 = np.array(
            self.robot.get_left_ee_pose() if arm_name == "left"
            else self.robot.get_right_ee_pose(),
            dtype=float,
        )
        tcp0 = self._tcp_pos(arm_name)
        fp0 = np.array(self.items[idx].get_pose().p, dtype=float)
        z_off0 = float(tcp0[2] - fp0[2])
        z_off1 = 0.018
        ee_q = ee0[3:7].copy()
        joints = (self.robot.left_arm_joints if arm_name == "left"
                  else self.robot.right_arm_joints)

        for i in range(n):
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            on_belt = (self._item_y[idx] is not None
                       and self._item_y[idx] >= self.pick_y_end)
            if on_belt or self._item_y[idx] is None:
                ee = np.array(
                    self.robot.get_left_ee_pose() if arm_name == "left"
                    else self.robot.get_right_ee_pose(),
                    dtype=float,
                )
                tcp = self._tcp_pos(arm_name)
                ee_from_tcp = ee[:3] - tcp
                t = float(i + 1) / float(n)
                t_z = min(1.0, t * 1.8)
                err_xy = tcp[:2] - fp[:2]
                err_z = float(tcp[2] - fp[2])
                z_off = float(z_off0 + t_z * (z_off1 - z_off0))
                z_cmd = fp[2] + z_off - 0.5 * max(0.0, err_z - z_off)
                lead_y = 0.0
                if on_belt:
                    remain = float(n - i)
                    lead_y = speed * remain / float(every) + 0.018 * (1.0 - t)
                tcp_target = np.array(
                    [
                        fp[0] - 0.7 * err_xy[0],
                        fp[1] - 0.7 * err_xy[1] - lead_y,
                        z_cmd,
                    ],
                    dtype=float,
                )
                target_ee = np.concatenate([tcp_target + ee_from_tcp, ee_q])
                q_goal = self._ik_arm_joints_for_ee(arm, target_ee)
                if q_goal is not None:
                    q_now = np.array(
                        [float(j.get_drive_target()[0]) for j in joints], dtype=float
                    )
                    self.robot.set_arm_joints(
                        q_goal, (q_goal - q_now).astype(float), arm_name
                    )
            self.robot.set_gripper(grip["result"][i], arm_name, grip["per_step"])
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _close_grippers_tracking_pair(self, idx_l, arm_l, idx_r, arm_r, target_pos=0.0, n_steps=50):
        """Close both grippers while each TCP tracks its still-moving apple."""
        def _grip(arm_name):
            now = float(
                self.robot.get_left_gripper_val() if arm_name == "left"
                else self.robot.get_right_gripper_val()
            )
            n = max(8, int(n_steps))
            target = float(target_pos)
            return {
                "num_step": n,
                "per_step": (target - now) / float(n),
                "result": np.linspace(now, target, n, dtype=float),
            }

        grip_l = _grip("left")
        grip_r = _grip("right")

        def _track_state(idx, arm_name):
            ee = np.array(
                self.robot.get_left_ee_pose() if arm_name == "left"
                else self.robot.get_right_ee_pose(),
                dtype=float,
            )
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            return {
                "off_xy": ee[:2] - fp[:2],
                "ee_z": float(ee[2]),
                "ee_q": ee[3:7].copy(),
                "joints": (self.robot.left_arm_joints if arm_name == "left"
                           else self.robot.right_arm_joints),
            }

        st_l = _track_state(idx_l, "left")
        st_r = _track_state(idx_r, "right")
        n = max(int(grip_l["num_step"]), int(grip_r["num_step"]))

        def _step_track(idx, arm_name, st):
            if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
                return
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            target_ee = np.concatenate([
                np.array([fp[0] + st["off_xy"][0], fp[1] + st["off_xy"][1], st["ee_z"]],
                         dtype=float),
                st["ee_q"],
            ])
            q_goal = self._ik_arm_joints_for_ee(arm_name, target_ee)
            if q_goal is None:
                return
            q_now = np.array(
                [float(j.get_drive_target()[0]) for j in st["joints"]], dtype=float
            )
            self.robot.set_arm_joints(q_goal, (q_goal - q_now).astype(float), arm_name)

        for i in range(n):
            _step_track(idx_l, "left", st_l)
            _step_track(idx_r, "right", st_r)
            if i < grip_l["num_step"]:
                self.robot.set_gripper(grip_l["result"][i], "left", grip_l["per_step"])
            if i < grip_r["num_step"]:
                self.robot.set_gripper(grip_r["result"][i], "right", grip_r["per_step"])
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _pinch_fruit(self, idx, arm, n_steps=16):
        """Pinch a still-moving belt apple; hold by jaw friction (no weld / no pause).

        The apple keeps riding the stream while the jaws close and the TCP
        tracks it. Only after a confirmed pinch does it leave the belt as a
        dynamic body — same idea as interactive_pack_fruits.

        Sequence: brief TCP track with jaws still pre-shaped (meet the apple
        using known belt speed), then a short snap close — avoids the planner's
        200-step close lag.
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            return False

        arm_name = "left" if str(arm) == "left" else "right"
        pre_pos = float(
            self.robot.get_left_gripper_val() if arm_name == "left"
            else self.robot.get_right_gripper_val()
        )
        # Meet the fruit first (no jaw motion), then snap shut.
        self._close_gripper_tracking_fruit(
            idx, arm, target_pos=pre_pos, n_steps=20
        )
        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            return False
        self._close_gripper_tracking_fruit(
            idx, arm, target_pos=0.0, n_steps=int(n_steps)
        )

        xy, dz = self._fruit_gripper_dist(idx, arm)
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        # Fruit still upstream of TCP → jaws closed early; keep on belt.
        upstream = float(fp[1] - tcp[1])
        near = (self._item_y[idx] is not None
                and xy <= 0.025 and -0.02 <= dz <= 0.085
                and upstream <= 0.015)
        contact = self._fruit_held_by_gripper(idx)
        if not (contact and near):
            if dbg:
                print(f"[pack_fruits]  pinch miss (still on belt) "
                      f"xy={xy:.3f} dz={dz:.3f} up={upstream:.3f} "
                      f"contact={contact}", flush=True)
            return False

        # Leave the stream at the current pose; settle contacts, then gravity.
        self._free_fruit_for_physical_grasp(idx)
        self._settle_grasp_contacts(20)
        # Extra squeeze after freeing so jaw contacts form before gravity.
        self._close_gripper_tracking_fruit(idx, arm, target_pos=0.0, n_steps=8)
        self._enable_fruit_gravity(idx)
        self._settle_grasp_contacts(20)
        held = self._fruit_held_by_gripper(idx) and self._fruit_near_gripper(
            idx, arm, xy_tol=0.035, z_lo=-0.02, z_hi=0.07
        )
        if held:
            self._mark_fruit_held(idx, arm)
        else:
            # Do not leave a freed dynamic apple sitting off-stream for retries.
            self._reseat_on_belt(idx, keep_grasping=True)
            self._set_gripper_fast(arm, 0.35, n_steps=8)
        if dbg:
            tcp = self._tcp_pos(arm_name)
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[pack_fruits]  pinch {self.item_types[idx]}_{idx} at "
                  f"{arm_name} tcp={np.round(tcp, 3)} fruit={np.round(fp, 3)} "
                  f"held={held} contact={self._fruit_held_by_gripper(idx)} "
                  f"xy={xy:.3f} dz={dz:.3f}",
                  flush=True)
        return bool(held)

    def _fruit_near_gripper(self, idx, arm, xy_tol=0.05, z_lo=-0.03, z_hi=0.08):
        """True if the fruit is still between / under the jaws (contact optional)."""
        xy, dz = self._fruit_gripper_dist(idx, arm)
        return xy <= float(xy_tol) and float(z_lo) <= dz <= float(z_hi)
    def _pinch_fruit_pair(self, *arm_idx_pairs):
        """Pinch still-moving belt apples together; hold by friction (no pause).

        Returns a list of bools aligned with ``arm_idx_pairs``.
        """
        pairs = [(arm, idx) for arm, idx in arm_idx_pairs if idx is not None]
        if not pairs:
            return []
        if len(pairs) == 1:
            arm, idx = pairs[0]
            return [self._pinch_fruit(idx, arm)]

        (arm_l, idx_l), (arm_r, idx_r) = pairs[0], pairs[1]
        self._close_grippers_tracking_pair(
            idx_l, arm_l, idx_r, arm_r, target_pos=0.0, n_steps=20
        )

        to_free = []
        for arm, idx in pairs:
            xy, dz = self._fruit_gripper_dist(idx, arm)
            near = (self._item_y[idx] is not None
                    and xy <= 0.04 and -0.02 <= dz <= self.ATTACH_Z_MAX + 0.02)
            contact = self._fruit_held_by_gripper(idx)
            if contact or near:
                to_free.append((arm, idx))
        for _arm, idx in to_free:
            self._free_fruit_for_physical_grasp(idx)
        if to_free:
            self._settle_grasp_contacts(15)
            for _arm, idx in to_free:
                self._enable_fruit_gravity(idx)
            self._settle_grasp_contacts(15)

        freed = {idx for _arm, idx in to_free}
        out = []
        for arm, idx in pairs:
            if idx not in freed:
                out.append(False)
                continue
            held = self._fruit_held_by_gripper(idx) or self._fruit_near_gripper(idx, arm)
            if held:
                self._mark_fruit_held(idx, arm)
            out.append(bool(held))
        return out

    def _reach_and_attach(self, idx, arm):
        """From the ready hover: close immediately while tracking the apple.

        The arm is already at the station. A second ``move_to_pose`` before the
        jaws move was the close delay (fruit slid ahead / TCP lagged). Lead is
        only the known close duration from belt speed; tracked close recenters XY.
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        side = self.item_sides[idx]
        for attempt in range(3):
            if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
                return False

            # Only micro-correct if the ready hover is badly off; otherwise close now.
            xy, dz = self._fruit_gripper_dist(idx, arm)
            if xy > 0.045 or dz > 0.10 or dz < -0.02:
                lead = self._intercept_lead_y(side, reach_steps=30, close_steps=16)
                grasp_pose = self._plan_final_grasp(
                    idx, arm, lead_y=lead * (1.0 + 0.2 * float(attempt))
                )
                if grasp_pose is None:
                    if dbg:
                        print("[pack_fruits]  no final grasp pose", flush=True)
                    return False
                self.plan_success = True
                self.move(self.move_to_pose(arm, grasp_pose))
                self.plan_success = True

            if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
                return False

            # Close from here — tracked close lowers Z and centers XY.
            if self._pinch_fruit(idx, arm, n_steps=16):
                return True
            if dbg:
                xy, dz = self._fruit_gripper_dist(idx, arm)
                print(f"[pack_fruits]  pinch missed — retry "
                      f"xy={xy:.3f} dz={dz:.3f} y={self._item_y[idx]}", flush=True)
            # Fast reopen (do not use Action close/open — 200 steps lets the
            # apple ride past the station before the next attempt).
            self._set_gripper_fast(arm, 0.35, n_steps=8)
            if self._item_y[idx] is None:
                self._reseat_on_belt(idx)
        if dbg:
            print("[pack_fruits]  reach finished but not close enough to pinch",
                  flush=True)
        return False

    def _reach_and_attach_pair(self, idx_l, idx_r, arm_l, arm_r):
        """Reach for both fruits; pinch whichever ends up close, then close together."""
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        got_l = got_r = False
        for attempt in range(3):
            need_l = (not got_l and self._item_y[idx_l] is not None
                      and self._item_y[idx_l] >= self.pick_y_end)
            need_r = (not got_r and self._item_y[idx_r] is not None
                      and self._item_y[idx_r] >= self.pick_y_end)
            if not need_l and not need_r:
                break
            pose_l = self._plan_final_grasp(idx_l, arm_l) if need_l else None
            pose_r = self._plan_final_grasp(idx_r, arm_r) if need_r else None
            acts = []
            if pose_l is not None:
                acts.append(self.move_to_pose(arm_l, pose_l))
            if pose_r is not None:
                acts.append(self.move_to_pose(arm_r, pose_r))
            if not acts:
                break
            self.plan_success = True
            self.move(*acts)
            self.plan_success = True
            if need_l and self._item_y[idx_l] is not None and self._tcp_near_fruit(idx_l, arm_l):
                got_l = True
            if need_r and self._item_y[idx_r] is not None and self._tcp_near_fruit(idx_r, arm_r):
                got_r = True

        pairs = []
        if got_l:
            pairs.append((arm_l, idx_l))
        if got_r:
            pairs.append((arm_r, idx_r))
        if not pairs:
            if dbg:
                print("[pack_fruits]  pair reach done gotL=False gotR=False", flush=True)
            return False, False
        held_flags = self._pinch_fruit_pair(*pairs)
        ok_l = ok_r = False
        for (arm, idx), held in zip(pairs, held_flags):
            if idx == idx_l:
                ok_l = bool(held)
            if idx == idx_r:
                ok_r = bool(held)
            if not held:
                # missed pinch — put back on the belt for a later solo attempt
                self._welded[idx] = False
                self._weld_arm[idx] = None
                if self._item_y[idx] is None:
                    self._reseat_on_belt(idx)
                self.plan_success = True
                self.move(self.open_gripper(arm))
                self.plan_success = True
        if dbg:
            print(f"[pack_fruits]  pair reach done gotL={ok_l} gotR={ok_r}", flush=True)
        return ok_l, ok_r

    def _settle_after_drop(self, idx, target_xy, resend_on_miss=True):
        """Mark packed if the fruit fell into the basket; otherwise resend.

        No teleport / "near-miss nudge" — the fruit must land under gravity.
        With ``resend_on_miss`` cleared the fruit stays where it landed, so a
        deliberate mis-sort is final instead of getting another lap.
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fruit = self.items[idx]
        side = self.item_sides[idx]
        if self._fruit_in_basket(idx):
            self._mark_packed(idx)
            if dbg:
                print(f"[pack_fruits]  dropped in basket "
                      f"p={np.round(fruit.get_pose().p, 3)}", flush=True)
            return
        if not resend_on_miss:
            # a deliberate wrong-basket drop still lands in a basket, so pin it;
            # anything that fell short stays dynamic and rests on the table
            landed = self._xy_inside_basket(
                np.array(fruit.get_pose().p, dtype=float)[:2],
                self._basket_for_target(target_xy),
            )
            self._mark_packed(idx, freeze=False)
            if dbg:
                print(f"[pack_fruits]  mis-packed {self.item_types[idx]}_{idx} "
                      f"p={np.round(fruit.get_pose().p, 3)} "
                      f"in_target_basket={bool(landed)}", flush=True)
            return
        p = np.array(fruit.get_pose().p, dtype=float)
        if dbg:
            print(f"[pack_fruits]  miss p={np.round(p, 3)} — resend", flush=True)
        rigid = self._item_comps[idx]
        if rigid is not None:
            rigid.set_kinematic(True)
            rigid.set_disable_gravity(True)
        self._over_basket[idx] = False
        self._item_y[idx] = float(self.BELT_Y_FAR)
        self._set_fruit_pose(
            idx, self.belt_cx[side], self.BELT_Y_FAR, self._fruit_ride_z
        )

    def _intercept_and_grasp(self, idx, arm, side):
        """Hover above the belt, wait for the fruit, then reach and pinch.

        The fruit never pauses on the belt. The arm hovers over the pick
        station, waits for the fruit to arrive, then reaches and closes while
        tracking the still-moving apple; only a confirmed pinch frees it for
        a friction hold (no teleport / EE weld).
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))

        self._set_gripper_fast(arm, 1.0, n_steps=16)

        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            return False

        pre_pose = self._plan_station_pre(idx, arm)
        if pre_pose is None:
            if dbg:
                print("[pack_fruits]  no pre-grasp pose at station", flush=True)
            return False

        self.plan_success = True
        ok = self.move(self.move_to_pose(arm, pre_pose))
        self.plan_success = True
        if ok is False:
            if dbg:
                print("[pack_fruits]  failed to reach station hover", flush=True)
            return False

        # Pre-shape tightly while waiting so the final pinch is a short snap.
        # Fast path — Action close is 200 steps and is only needed for demos.
        self._set_gripper_fast(arm, 0.35, n_steps=12)

        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            if dbg:
                print(f"[pack_fruits]  missed while approaching station "
                      f"y={self._item_y[idx]}", flush=True)
            return False

        if not self._wait_fruit_at_station(idx, side):
            return False

        return self._reach_and_attach(idx, arm)

    def _accumulate_basket_contacts(self):
        """PACKING_DEBUG-only: called every physics step so transient
        mid-trajectory contacts (not just the contact state at the instant
        a move finishes) are caught between ``_debug_report_basket_contact``
        checkpoints. Uses the full articulation link set (not just
        ``robot.gripper_name``, which is only the finger links) so
        wrist/forearm clipping against the rim shows up too, not just
        fingertip contact.
        """
        if not hasattr(self, "_robot_link_names"):
            self._robot_link_names = set(
                l.get_name() for l in
                (self.robot.left_entity.get_links() + self.robot.right_entity.get_links())
            )
        if not hasattr(self, "_contact_hits_pending"):
            self._contact_hits_pending = set()
        if not hasattr(self, "_contact_min_sep_pending"):
            self._contact_min_sep_pending = None
        if not hasattr(self, "_contact_step_range_pending"):
            self._contact_step_range_pending = None
        basket_names = {b.actor.get_name() for b in self.baskets.values()}
        hit_this_step = False
        for c in self.scene.get_contacts():
            n0 = c.bodies[0].entity.name
            n1 = c.bodies[1].entity.name
            hit_link = None
            if n0 in basket_names and n1 in self._robot_link_names:
                hit_link = n1
            elif n1 in basket_names and n0 in self._robot_link_names:
                hit_link = n0
            if hit_link is None:
                continue
            hit_this_step = True
            self._contact_hits_pending.add(hit_link)
            for pt in c.points:
                sep = float(pt.separation)
                if self._contact_min_sep_pending is None or sep < self._contact_min_sep_pending:
                    self._contact_min_sep_pending = sep
        if hit_this_step:
            s = int(self._step_ctr)
            if self._contact_step_range_pending is None:
                self._contact_step_range_pending = [s, s]
            else:
                self._contact_step_range_pending[0] = min(self._contact_step_range_pending[0], s)
                self._contact_step_range_pending[1] = max(self._contact_step_range_pending[1], s)

    def _debug_reset_basket_contact(self):
        self._contact_hits_pending = set()
        self._contact_min_sep_pending = None
        self._contact_step_range_pending = None
        self._contact_phase_start_step = int(getattr(self, "_step_ctr", 0))

    def _debug_report_basket_contact(self, tag):
        """Report + clear whatever robot-link/basket contacts have
        accumulated (via ``_accumulate_basket_contacts``) since the last
        reset/report. ``min_sep`` is the deepest (most negative = real
        penetration; small positive = within the collision margin but not
        actually overlapping) separation seen across all contact points.
        ``first_step``/``last_step`` (relative to the phase's start step)
        pin down WHEN within the phase's trajectory the contact occurred —
        near 0 means "already touching at the start pose", a value close to
        the phase's total step count means "only at the very end", and
        anything comfortably in between (with start/end themselves clear)
        means the collision happened mid-trajectory (e.g. the interpolated
        joint-space arc dipping the wrist through/near the rim), not at
        either planned endpoint.
        """
        if not bool(os.environ.get("PACKING_DEBUG")):
            return
        hits = getattr(self, "_contact_hits_pending", set())
        min_sep = getattr(self, "_contact_min_sep_pending", None)
        step_range = getattr(self, "_contact_step_range_pending", None)
        phase_start = getattr(self, "_contact_phase_start_step", None)
        phase_len = (int(self._step_ctr) - phase_start) if phase_start is not None else None
        if hits:
            sep_str = f"{min_sep:.4f}" if min_sep is not None else "?"
            if step_range is not None and phase_start is not None:
                rel = f"steps[{step_range[0]-phase_start}:{step_range[1]-phase_start}]/{phase_len}"
            else:
                rel = "steps=?"
            print(f"[pack_fruits]  BASKET-CONTACT [{tag}] links={sorted(hits)} min_sep={sep_str} {rel}", flush=True)
        self._contact_hits_pending = set()
        self._contact_min_sep_pending = None
        self._contact_step_range_pending = None

    def _held_target_ee_pose(self, idx, arm, xyz):
        """EE pose that places the held fruit at world ``xyz`` (planning only).

        Uses a snapshot fruit↔EE offset — the fruit is not glued; it must
        ride with jaw friction during the move.
        """
        arm_name = "left" if str(arm) == "left" else "right"
        ee_pose = self._ee_pose_full(arm_name)
        fruit_pose = self.items[idx].get_pose()
        offset = ee_pose.inv() * fruit_pose
        target_fruit = sapien.Pose(
            np.asarray(xyz, dtype=float).tolist(), list(fruit_pose.q)
        )
        return target_fruit * offset.inv()

    def _weld_target_ee_pose(self, idx, xyz):
        """Compat alias — prefer ``_held_target_ee_pose`` with an arm tag."""
        arm = self._weld_arm[idx] if self._weld_arm[idx] is not None else "left"
        return self._held_target_ee_pose(idx, arm, xyz)

    def _ik_arm_joints_for_ee(self, arm, ee_pose7):
        """IK joint solution for a planning-EE pose ``[x,y,z,qw,qx,qy,qz]``.

        Uses Curobo ``solve_ik`` (not trajopt). Returns a length-6 numpy
        array, or None if IK fails. Needed because ``move_to_pose``/
        ``move_by_displacement`` silently no-op pure +Z lifts from the
        post-grasp configuration.
        """
        import torch
        from curobo.types.math import Pose as CuroboPose

        arm_name = "left" if str(arm) == "left" else "right"
        trans_target = self.robot._trans_from_gripper_to_endlink(
            list(ee_pose7), arm_tag=arm_name,
        )
        planner = self.robot.left_planner if arm_name == "left" else self.robot.right_planner
        world_target = np.concatenate([np.array(trans_target.p), np.array(trans_target.q)])
        world_base = np.concatenate([
            np.array(planner.robot_origion_pose.p),
            np.array(planner.robot_origion_pose.q),
        ])
        tp_p, tp_q = planner._trans_from_world_to_base(world_base, world_target)
        tp_p = np.array(tp_p, dtype=float)
        tp_q = np.array(tp_q, dtype=float)
        if "aloha-agilex" not in str(getattr(planner, "yml_path", "")):
            tp_p = tp_p + np.array(planner.frame_bias, dtype=float)
        goal = CuroboPose.from_list(list(tp_p) + list(tp_q))
        ik = planner.motion_gen.solve_ik(goal, return_seeds=1)
        if not bool(ik.success.reshape(-1)[0].item()):
            return None
        return ik.solution.detach().cpu().numpy().reshape(-1).astype(float)

    def _drive_arm_joints(self, arm, q_goal, n_steps=50):
        """Smoothly drive arm joints from the current drive targets to ``q_goal``.

        Uses per-step ``set_qpos`` + drive targets so the motion actually
        reaches the IK solution (drive-only interpolation was observed to
        stall near the grasp configuration).
        """
        arm_name = "left" if str(arm) == "left" else "right"
        joints = (self.robot.left_arm_joints if arm_name == "left"
                  else self.robot.right_arm_joints)
        entity = (self.robot.left_entity if arm_name == "left"
                  else self.robot.right_entity)
        planner = (self.robot.left_planner if arm_name == "left"
                   else self.robot.right_planner)
        active = entity.get_active_joints()
        name_to_qpos_i = {j.get_name(): i for i, j in enumerate(active)}
        q_start = np.array([float(j.get_drive_target()[0]) for j in joints], dtype=float)
        q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
        n = max(1, int(n_steps))
        for i in range(1, n + 1):
            a = float(i) / float(n)
            q = (1.0 - a) * q_start + a * q_goal
            v = (q_goal - q_start) / float(n)
            qpos = np.array(entity.get_qpos(), dtype=float)
            for j, jn in enumerate(planner.active_joints_name):
                if j >= len(q):
                    break
                if jn in name_to_qpos_i:
                    qpos[name_to_qpos_i[jn]] = q[j]
            entity.set_qpos(qpos)
            self.robot.set_arm_joints(q, v, arm_name)
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _raise_along_z(self, idx, arm, lift_z=None):
        """Raise the gripper (and friction-held fruit) straight up by ``lift_z`` meters.

        Re-squeezes briefly, then uses many small continuous displacements so
        jaw contacts are not yanked apart (no physics retune).
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        if lift_z is None:
            lift_z = self.pick_lift
        arm_name = "left" if str(arm) == "left" else "right"
        # Firm the jaws before the lift (arm motion only).
        self._close_gripper_tracking_fruit(idx, arm, target_pos=0.0, n_steps=6)
        self._settle_grasp_contacts(8)
        ee0 = np.array(
            self.robot.get_left_ee_pose() if arm_name == "left"
            else self.robot.get_right_ee_pose(),
            dtype=float,
        )
        fp0 = np.array(self.items[idx].get_pose().p, dtype=float)
        n_chunks = 8
        chunk = float(lift_z) / float(n_chunks)
        for _ in range(n_chunks):
            self.plan_success = True
            self.move(self.move_by_displacement(arm, z=chunk, move_axis="world"))
            self.plan_success = True
            if not self._fruit_near_gripper(idx, arm):
                break
        if dbg:
            ee1 = np.array(
                self.robot.get_left_ee_pose() if arm_name == "left"
                else self.robot.get_right_ee_pose(),
                dtype=float,
            )
            fp1 = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[pack_fruits]  raise +Z asked={lift_z:.3f} "
                  f"ee_dz={ee1[2]-ee0[2]:.3f} fruit_dz={fp1[2]-fp0[2]:.3f} "
                  f"fruit_z={fp1[2]:.3f} contact={self._fruit_held_by_gripper(idx)} "
                  f"near={self._fruit_near_gripper(idx, arm)}",
                  flush=True)

    def _fruit_xy_gap(self, idx, target_xy):
        """Horizontal distance from the held fruit to a drop target."""
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        return float(np.hypot(float(target_xy[0]) - fp[0], float(target_xy[1]) - fp[1]))

    def _slide_hover_z(self, idx):
        """Carry height that clears every basket rim on the way in."""
        fruit_z = float(self.items[idx].get_pose().p[2])
        if not self.basket_top_z:
            return fruit_z
        rim = max(self.basket_top_z.values())
        return max(fruit_z, float(rim) + self.fruit_r + 0.02)

    def _slide_xy_to_target(self, idx, arm, target_xy, tries=3, tol=0.03):
        """Slide the held fruit to ``target_xy`` at a rim-clearing height.

        Displacements are measured from the fruit pose (pick_ripe_apple style)
        so a soft friction hold still converges without an EE weld.
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        hover_z = self._slide_hover_z(idx)
        for _try in range(tries):
            if not self._fruit_near_gripper(idx, arm):
                if dbg:
                    print(f"[pack_fruits]  slide abort — lost fruit_{idx}",
                          flush=True)
                return False
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            gap_xy = self._fruit_xy_gap(idx, target_xy)
            if dbg:
                print(f"[pack_fruits]  slide try={_try} fp={fp.round(4)} "
                      f"gap_xy={gap_xy:.4f} hover_z={hover_z:.3f}", flush=True)
            if gap_xy < tol:
                return True
            self.plan_success = True
            self.move(self.move_by_displacement(
                arm,
                x=float(target_xy[0]) - fp[0],
                y=float(target_xy[1]) - fp[1],
                z=float(hover_z) - fp[2],
                move_axis="world",
            ))
            self.plan_success = True
        return self._fruit_xy_gap(idx, target_xy) < tol

    def _slide_over_basket(self, idx, arm, target_xy, lift_z=None, tries=3, tol=0.03):
        """Raise ≥10 cm along Z in place, then slide horizontally over the basket."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        if lift_z is None:
            lift_z = self.pick_lift

        # 1) raise straight up (same XY)
        self._raise_along_z(idx, arm, lift_z=lift_z)

        # 2) horizontal slide at a height that clears the basket rim
        reached = self._slide_xy_to_target(idx, arm, target_xy, tries=tries, tol=tol)
        if dbg:
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[pack_fruits]  slide residual_xy="
                  f"{self._fruit_xy_gap(idx, target_xy):.4f} "
                  f"fruit_z={fp[2]:.3f} basket_top="
                  f"{self.basket_top_z[self.item_types[idx]]:.3f}", flush=True)
        return reached

    def _hangs_over_basket(self, idx, target_xy):
        """True when the held fruit is clear of the target basket's walls."""
        p = np.array(self.items[idx].get_pose().p, dtype=float)
        ftype = self._basket_for_target(target_xy)
        return self._xy_inside_basket(p[:2], ftype, margin=self.fruit_r)

    def _ensure_over_basket(self, idx, arm, target_xy, tries=2):
        """True once the held fruit hangs over ``target_xy``, retrying the slide.

        The gripper must never open short of the basket mouth — a stalled or
        unplannable slide used to drop the fruit onto the table.
        """
        if self._hangs_over_basket(idx, target_xy):
            return True
        self._slide_xy_to_target(idx, arm, target_xy, tries=tries, tol=0.015)
        return self._hangs_over_basket(idx, target_xy)

    def _abort_drop_to_belt(self, idx, arm):
        """Send an undroppable fruit back for another lap instead of the table."""
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[pack_fruits]  drop aborted for fruit_{idx} "
                  f"p={np.round(self.items[idx].get_pose().p, 3)} — back to belt",
                  flush=True)
        self._release_fruit(idx)
        self._over_basket[idx] = False
        self._item_y[idx] = float(self.BELT_Y_FAR)
        rigid = self._item_comps[idx]
        if rigid is not None:
            rigid.set_kinematic(True)
            rigid.set_disable_gravity(True)
        self._set_fruit_pose(
            idx, self.belt_cx[self.item_sides[idx]], self.BELT_Y_FAR,
            self._fruit_ride_z,
        )
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True

    def _carry_and_drop(self, idx, arm, target_xy, resend_on_miss=True):
        """Raise along Z, slide over the basket, open gripper, let fruit fall.

        Returns whether the fruit was actually released over the basket.
        """
        self._slide_over_basket(idx, arm, target_xy, lift_z=self.pick_lift)
        if not self._fruit_near_gripper(idx, arm):
            # Lost the apple mid-carry — put it back rather than drop on the table.
            self._abort_drop_to_belt(idx, arm)
            self.plan_success = True
            self.move(self.back_to_origin(arm))
            self.plan_success = True
            return False
        over_basket = self._ensure_over_basket(idx, arm, target_xy)
        self._over_basket[idx] = bool(over_basket)
        if not over_basket:
            self._abort_drop_to_belt(idx, arm)
            self.plan_success = True
            self.move(self.back_to_origin(arm))
            self.plan_success = True
            return False

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self._release_fruit(idx)
        self._belt_dwell(80)  # give gravity time to settle into the basket
        self._settle_after_drop(idx, target_xy, resend_on_miss=resend_on_miss)
        self.plan_success = True
        self.move(self.back_to_origin(arm))
        self.plan_success = True
        return True

    def _carry_and_drop_pair(
        self, idx_l, arm_l, target_l, idx_r, arm_r, target_r, resend_on_miss=True,
    ):
        """Raise, slide, and release both fruits over their baskets together.

        Returns ``(released_l, released_r)`` — True when that fruit was opened
        over its basket (vs aborted back onto the belt).
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))

        # 1) raise both grippers +Z (continuous displacement — keep jaw contacts)
        self._raise_along_z(idx_l, arm_l, lift_z=self.pick_lift)
        self._raise_along_z(idx_r, arm_r, lift_z=self.pick_lift)

        # 2) slide horizontally over each basket (fruit-relative displacement)
        for _try in range(3):
            fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
            fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
            hover_z_l = self._slide_hover_z(idx_l)
            hover_z_r = self._slide_hover_z(idx_r)
            gl = float(np.hypot(float(target_l[0]) - fp_l[0], float(target_l[1]) - fp_l[1]))
            gr = float(np.hypot(float(target_r[0]) - fp_r[0], float(target_r[1]) - fp_r[1]))
            if dbg:
                print(
                    f"[pack_fruits]  pair slide try={_try} gap_l={gl:.4f} gap_r={gr:.4f}",
                    flush=True,
                )
            if gl < 0.03 and gr < 0.03:
                break
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm_l,
                    x=float(target_l[0]) - fp_l[0],
                    y=float(target_l[1]) - fp_l[1],
                    z=float(hover_z_l) - fp_l[2],
                    move_axis="world",
                ),
                self.move_by_displacement(
                    arm_r,
                    x=float(target_r[0]) - fp_r[0],
                    y=float(target_r[1]) - fp_r[1],
                    z=float(hover_z_r) - fp_r[2],
                    move_axis="world",
                ),
            )
            self.plan_success = True

        if dbg:
            fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
            fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
            print(
                f"[pack_fruits]  pair slide residual_xy "
                f"l={np.hypot(target_l[0] - fp_l[0], target_l[1] - fp_l[1]):.4f} "
                f"r={np.hypot(target_r[0] - fp_r[0], target_r[1] - fp_r[1]):.4f} "
                f"fruit_z l={fp_l[2]:.3f} r={fp_r[2]:.3f}",
                flush=True,
            )

        over_l = self._ensure_over_basket(idx_l, arm_l, target_l)
        over_r = self._ensure_over_basket(idx_r, arm_r, target_r)
        self._over_basket[idx_l] = bool(over_l)
        self._over_basket[idx_r] = bool(over_r)

        if not over_l:
            self._abort_drop_to_belt(idx_l, arm_l)
        if not over_r:
            self._abort_drop_to_belt(idx_r, arm_r)

        drop = [
            (i, a, t)
            for i, a, t, ok in (
                (idx_l, arm_l, target_l, over_l),
                (idx_r, arm_r, target_r, over_r),
            )
            if ok
        ]
        if drop:
            self.plan_success = True
            self.move(*[self.open_gripper(a) for _i, a, _t in drop])
            for i, _a, _t in drop:
                self._release_fruit(i)
            self._belt_dwell(80)
            for i, _a, t in drop:
                self._settle_after_drop(i, t, resend_on_miss=resend_on_miss)

        self.plan_success = True
        self.move(self.back_to_origin(arm_l), self.back_to_origin(arm_r))
        self.plan_success = True
        return bool(over_l), bool(over_r)

    def _pack_item(self, idx):
        """Intercept one still-moving fruit → pinch → carry → release."""
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fruit = self.items[idx]
        ftype = self.item_types[idx]
        belt_side = self.item_sides[idx]
        # Color-matched arm (red→left, green→right). In two-color episodes the
        # fruit already rides that arm's belt; default/opt2 may still cross.
        arm_side = self.TYPE_SIDE.get(ftype, belt_side)
        arm = ArmTag(arm_side)
        target_xy = self._basket_target_xy(idx)

        self._begin_spawn_hold()
        self._grasping_idxs.add(idx)
        try:
            if self._item_y[idx] is None:
                return

            if dbg:
                print(f"[pack_fruits] pack {ftype}_{idx} belt={belt_side} arm={arm_side} "
                      f"stream_y={self._item_y[idx]:.3f} "
                      f"pose={np.round(fruit.get_pose().p, 3)}", flush=True)

            # grasp uses the fruit's belt for geometry; arm is color-matched
            if not self._intercept_and_grasp(idx, arm, belt_side):
                if dbg:
                    print("[pack_fruits]  grasp failed — fruit stays on belt", flush=True)
                self.plan_success = True
                try:
                    self.move(self.back_to_origin(arm))
                except Exception:
                    pass
                self.plan_success = True
                self._note_pack_failure(idx)
                return

            if dbg:
                print(f"[pack_fruits]  pinched; ee={np.round(self._ee_pos(arm_side), 3)} "
                      f"fruit={np.round(fruit.get_pose().p, 3)} "
                      f"contact={self._fruit_held_by_gripper(idx)}", flush=True)

            dropped = self._carry_and_drop(idx, arm, target_xy)
            if not dropped:
                self._note_pack_failure(idx)
        finally:
            self._grasping_idxs.discard(idx)
            self._end_spawn_hold()

    def _note_pack_failure(self, idx):
        """Cap retries so a stuck cross-reach cannot spin forever.

        On give-up, despawn/hide the fruit (same as riding off the belt end) so
        a failed apple does not linger on the conveyor while the next wave runs.
        """
        if idx < 0 or idx >= len(self._pack_fail_counts):
            return
        if self._packed[idx] or self._missed[idx]:
            return
        self._pack_fail_counts[idx] += 1
        if self._pack_fail_counts[idx] < int(self._pack_fail_limit):
            return
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[pack_fruits]  giving up on fruit_{idx} after "
                  f"{self._pack_fail_counts[idx]} failed packs — despawn",
                  flush=True)
        # Force a full despawn even if already marked mid-failure elsewhere.
        self._missed[idx] = False
        self._despawn_off_belt(idx)

    def _pack_pair(self, idx_l, idx_r):
        """Pick left+right fruits simultaneously, then carry to baskets together."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        left, right = ArmTag("left"), ArmTag("right")
        target_l = self._basket_target_xy(idx_l)
        target_r = self._basket_target_xy(idx_r)

        self._begin_spawn_hold()
        self._grasping_idxs.update({idx_l, idx_r})
        try:
            if dbg:
                print(f"[pack_fruits] pack_pair L={self.item_types[idx_l]}_{idx_l} "
                      f"R={self.item_types[idx_r]}_{idx_r}", flush=True)

            self.plan_success = True
            self.move(self.open_gripper(left), self.open_gripper(right))
            self.plan_success = True

            pre_l = self._plan_station_pre(idx_l, left)
            pre_r = self._plan_station_pre(idx_r, right)
            if pre_l is None or pre_r is None:
                if dbg:
                    print("[pack_fruits]  pair: missing pre-grasp — fallback single",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                self._pack_item(idx_l)
                self._pack_item(idx_r)
                return

            self.plan_success = True
            ok = self.move(
                self.move_to_pose(left, pre_l),
                self.move_to_pose(right, pre_r),
            )
            self.plan_success = True
            if ok is False:
                if dbg:
                    print("[pack_fruits]  pair: hover failed — fallback single", flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                self._pack_item(idx_l)
                self._pack_item(idx_r)
                return

            if not self._wait_pair_at_station(idx_l, idx_r):
                if dbg:
                    print("[pack_fruits]  pair: wait at station failed — fallback",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            ok_l, ok_r = self._reach_and_attach_pair(idx_l, idx_r, left, right)

            if not ok_l and not ok_r:
                if dbg:
                    print("[pack_fruits]  pair: neither close enough — fallback",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            if ok_l and not ok_r:
                if dbg:
                    print("[pack_fruits]  pair: only left attached", flush=True)
                # the other fruit is still time-critical on the belt — try it
                # first; the attached one is welded and safe to carry after
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                else:
                    self.move(self.back_to_origin(right))
                    self.plan_success = True
                self._carry_and_drop(idx_l, left, target_l)
                return
            if ok_r and not ok_l:
                if dbg:
                    print("[pack_fruits]  pair: only right attached", flush=True)
                self._grasping_idxs.discard(idx_l)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                else:
                    self.move(self.back_to_origin(left))
                    self.plan_success = True
                self._carry_and_drop(idx_r, right, target_r)
                return

            if dbg:
                print("[pack_fruits]  pair: both attached", flush=True)

            self._carry_and_drop_pair(
                idx_l, left, target_l, idx_r, right, target_r,
            )
        finally:
            self._grasping_idxs.discard(idx_l)
            self._grasping_idxs.discard(idx_r)
            self._end_spawn_hold()

    def _ready_by_side(self):
        """Return oldest ready fruit index per belt (or None)."""
        if self._grasping_idxs:
            return {"left": None, "right": None}
        latest_start = self.pick_station_y + 0.08
        ready = {"left": None, "right": None}
        for i in range(self.n_items):
            if (not self._spawned_mask[i] or self._packed[i]
                    or self._welded[i] or self._item_y[i] is None):
                continue
            y = self._item_y[i]
            if latest_start <= y <= self.pick_y:
                side = self.item_sides[i]
                if ready[side] is None:
                    ready[side] = i
        return ready

    def _dispatch_pack(self, ready):
        """Pack the ready colored fruit (always solo — one colored apple at a time)."""
        left_i, right_i = ready["left"], ready["right"]
        # Prefer the fruit that is farther along (smaller y) if somehow both
        # ready — should not happen with solo spawn, but be safe.
        candidates = [i for i in (left_i, right_i) if i is not None]
        if not candidates:
            return False
        if len(candidates) == 2:
            yl = self._item_y[left_i]
            yr = self._item_y[right_i]
            pick = left_i if (yl is not None and (yr is None or yl <= yr)) else right_i
        else:
            pick = candidates[0]
        self._pack_item(pick)
        return True

    # ------------------------------------------------------------- policy
    def play_once(self):
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))

        # belts run continuously for the whole episode (including during arm motion,
        # because take_dense_action calls _update_kinematic_tasks every step)
        self._belt_running = True
        min_speed = max(min(self.belt_speed.values()), 1e-6)
        max_steps = int(
            (self.BELT_Y_FAR - self.BELT_Y_NEAR + self.n_items * self.spawn_gap)
            / min_speed
        ) * self.advance_every + self.n_items * 1200 + 4000

        guard = 0
        def _wave_done():
            return all(
                self._packed[i] or self._missed[i] for i in range(self.n_items)
            ) and self._spawned >= self.n_items

        while guard < max_steps and not _wave_done():
            guard += 1
            ready = self._ready_by_side()
            if not self._dispatch_pack(ready):
                self._belt_dwell(max(1, self.advance_every))

        for _ in range(self.n_items):
            ready = self._ready_by_side()
            if ready["left"] is None and ready["right"] is None:
                break
            self._dispatch_pack(ready)

        self._belt_dwell(40)
        self._belt_running = False

        if self.check_success():
            self.plan_success = True

        if dbg:
            print(f"[pack_fruits] done plan={self.plan_success} mode={self.spawn_mode} "
                  f"types={self.item_types} sides={self.item_sides}", flush=True)
            for i in range(self.n_items):
                p = self.items[i].get_pose().p
                print(f"[pack_fruits]  {self.item_types[i]}_{i} p={np.round(p, 3)} "
                      f"in={self._fruit_in_basket(i)} packed={self._packed[i]} "
                      f"missed={self._missed[i]}",
                      flush=True)

        self.info["info"] = {
            "{A}": (
                f"{self.n_items} apples "
                f"({self.n_apple} red, {self.n_green} green)"
                if self.two_colors_enabled
                else f"{self.n_items} {'red' if self.active_colors[0] == 'apple' else 'green'} apples"
            ),
            "{B}": f"076_breadbasket/base{self.basket_id}",
            "{a}": (
                "both arms" if self.two_colors_enabled
                else ("left arm" if self.active_colors[0] == "apple" else "right arm")
            ),
        }
        return self.info

    # ------------------------------------------------------------- success
    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        n = int(getattr(self, "n_items", 0) or 0)
        self._metric_spawn_step = [None] * n
        self._metric_pack_step = [None] * n
        self._metric_release_offset = [None] * n   # offset at the last release, normalised
        self._metric_packed_offsets = []           # committed on a successful pack

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _latch_spawn_metric(self, idx):
        try:
            if self._metric_spawn_step[idx] is None:
                self._metric_spawn_step[idx] = self._metric_step()
        except Exception:
            pass

    def _latch_release_offset(self, idx):
        """Chebyshev XY offset from the fruit's own basket mouth centre, at let-go."""
        try:
            ftype = self.item_types[idx]
            c = np.asarray(self.basket_centers[ftype], dtype=np.float64)
            half_x, half_y = self.basket_half_xy.get(ftype, self.BASKET_HALF_XY)
            p = np.array(self.items[idx].get_pose().p, dtype=np.float64)
            self._metric_release_offset[idx] = float(max(
                abs(float(p[0]) - float(c[0])) / max(float(half_x), 1e-9),
                abs(float(p[1]) - float(c[1])) / max(float(half_y), 1e-9),
            ))
        except Exception:
            pass

    def _latch_pack_metric(self, idx):
        """Called from _mark_packed the first time a fruit resolves into a basket."""
        try:
            if self._metric_pack_step[idx] is not None:
                return
            self._metric_pack_step[idx] = self._metric_step()
            off = self._metric_release_offset[idx]
            if off is not None:
                self._metric_packed_offsets.append(float(off))
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `pack_latency_steps` — mean steps from a fruit appearing at the far
        end of the belt until it is resolved into a basket, averaged over the fruits
        actually packed. This is the belt-paced decision+transfer window.
        extra2 `drop_offset_norm` — mean Chebyshev distance from the basket mouth
        centre at the instant the jaws let go, in units of basket half-extent
        (1.0 = right on the rim). LOWER is better.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        lats = []
        try:
            for i in range(int(self.n_items)):
                a = self._metric_spawn_step[i]
                b = self._metric_pack_step[i]
                if a is not None and b is not None:
                    lats.append(max(int(b) - int(a), 0))
        except Exception:
            lats = []
        mean_lat = (sum(lats) / len(lats)) if lats else None
        out["pack_latency_steps"] = None if mean_lat is None else round(float(mean_lat), 3)
        out["pack_latency_s"] = None if mean_lat is None else round(float(mean_lat) * dt, 4)
        out["first_pack_latency_steps"] = min(lats) if lats else None
        out["packs_counted"] = len(lats)

        offs = list(getattr(self, "_metric_packed_offsets", []) or [])
        out["drop_offset_norm"] = round(sum(offs) / len(offs), 4) if offs else None
        out["worst_drop_offset_norm"] = round(max(offs), 4) if offs else None
        return out

    def check_success(self):
        if not all(self._fruit_in_basket(i) for i in range(self.n_items)):
            return False
        n_dist = int(getattr(self, "n_distractor_slots", 0) or 0)
        if any(self._distractor_in_any_basket(s) for s in range(n_dist)):
            return False
        return True

    def get_obs(self):
        obs = super().get_obs()
        in_ok = [bool(self._fruit_in_basket(i)) for i in range(self.n_items)]
        obs["pack_fruits"] = {
            "n_items": int(self.n_items),
            "n_apple": int(self.n_apple),
            "n_green": int(self.n_green),
            "n_orange": int(self.n_green),  # legacy alias
            "n_spawned": int(self._spawned),
            "n_packed": int(sum(1 for p in self._packed if p)),
            "n_missed": int(sum(1 for m in self._missed if m)),
            "n_correct": int(sum(in_ok)),
            "n_distractor_plan": int(getattr(self, "n_distractor_plan", 0)),
            "two_colors_enabled": bool(self.two_colors_enabled),
            "distractor_enabled": bool(self.distractor_enabled),
            "active_colors": list(self.active_colors),
            "spawn_mode": str(self.spawn_mode),
            "spawn_delay_s": float(self.spawn_delay_s),
            "item_types": list(self.item_types),
            "item_sides": list(self.item_sides),
            "item_y": [None if y is None else float(y) for y in self._item_y],
            "item_in_correct_basket": in_ok,
            "belt_speed_left": float(self.belt_speed["left"]),
            "belt_speed_right": float(self.belt_speed["right"]),
            "belt_gap": float(self.belt_gap),
            "belt_running": bool(self._belt_running),
            "fruit_scale": float(self.fruit_scale),
        }
        return obs
