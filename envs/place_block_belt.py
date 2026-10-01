from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
from scene_utils import print_c, get_quat_euler


class place_block_belt(Base_Task):
    """Set a tall, top-heavy block onto a moving conveyor belt so it drops into a
    bowl placed at the belt exit.

    Layout is randomly mirrored about x=0 each episode:
      - default: belt flows +x, block loads on the left, left arm places, bowl at +x exit
      - mirrored: belt flows -x, block loads on the right, right arm places, bowl at -x exit

    The block always spawns >= ``block_belt_clearance`` (default 5 cm) clear of the
    belt's start edge, with y jittered by +/- ``block_y_jitter`` (default 5 cm)
    about the belt centerline.

    Task options (independent; set in ``task_args.place_block_belt``):
      - Opt 1 — moving bowl:  ``bowl_move_enabled``
        (bowl traverses the FULL belt y-range even when Opt 2 is also on)
      - Opt 2 — belt blocker: ``blocker_enabled``
        (rod ≥15 cm past load edge; y-size 20–30% of belt width; x-thickness
        nominal ±50%; flush to a random y-edge so the expert must release in
        the unblocked lane)

    A red placement line sits ``place_line_offset`` (default 10 cm) past the load
    edge across the belt. The cube must first contact the belt *before* that
    line; contact past it fails the episode.

    Belt speed and (if Opt 1) bowl speed are sampled each episode as
    nominal ± ``speed_jitter_frac`` (default ±20%).

    The scripted expert:
      1) grasps the block on the start side,
      2) lifts it and carries it above the belt surface,
      3) MATCHES the gripper's horizontal (x) velocity to the belt speed by a
         short move_by_displacement along the belt direction just before release,
      4) opens the gripper; the belt then carries the upright block to the exit,
         where it falls into the bowl on the table.

    The belt motion is driven step-by-step in the overridden _update_kinematic_tasks:
    once the block is released and resting on the belt, the block is set kinematic
    and advanced along the belt direction by belt_speed * dt every physics step.

    Single-arm (arm on the load side). Metric: tilt_score =
    clamp(1 - max_tilt_angle/theta_max, 0, 1).
    """

    # ----------------------------------------------------------------- params (class defaults)
    BELT_SPEED_DEFAULT = 0.4          # m/s lateral belt surface speed (signed via belt_dir)
    BELT_X_START_DEFAULT = -0.2        # load-side x (default / non-mirrored layout)
    BELT_X_END_DEFAULT = 0.2           # exit-side x (default / non-mirrored layout)
    BELT_Y_DEFAULT = -0.02             # y of the belt centerline (mid zone)
    BELT_HALF_W_DEFAULT = 0.15         # belt half-width in y
    BELT_RIDE_STEPS_DEFAULT = 600      # max physics steps to let the block ride the belt
    BELT_RELEASE_DELAY_STEPS_DEFAULT = 12  # wait after opening so the block fully drops first
    BELT_BAND_SPACING_DEFAULT = 0.055  # spacing between visible tread bands
    BELT_BAND_HALF_X_DEFAULT = 0.012   # half-length of each moving tread band along x
    BELT_BAND_HALF_Z_DEFAULT = 0.0006  # thin render-only band height
    BELT_BAND_Z_OFFSET_DEFAULT = 0.0008  # lift above the top face to avoid z-fighting
    RANDOM_BELT_FLIP_DEFAULT = True    # randomly mirror layout about x=0
    BOWL_X_OFFSET_DEFAULT = 0.04       # bowl center offset beyond the physical belt exit
    BOWL_Y_OFFSET_DEFAULT = 0.0        # bowl shares the belt lane by default
    BOWL_Y_RANDOM_RANGE_DEFAULT = 0.08 # per-episode y randomization around bowl_y_offset
    # Opt 1 — moving bowl
    BOWL_MOVE_ENABLED_DEFAULT = True   # Opt 1 master switch: oscillate bowl along belt y
    BOWL_MOVE_SPEED_DEFAULT = 0.30     # Opt 1: nominal m/s bowl oscillation (then ± speed_jitter)
    BOWL_MOVE_RANGE_DEFAULT = 0.12     # Opt 1: half-range of the bowl motion around its start y
    BOWL_INNER_RADIUS_DEFAULT = 0.065  # geometric success radius around bowl center
    BOWL_SUCCESS_HEIGHT_DEFAULT = 0.10 # block center must settle below this height above bowl base
    BOWL_SETTLE_STEPS_DEFAULT = 140    # extra physics steps after the block leaves the belt

    # Per-episode speed jitter applied to belt_speed and (if Opt 1) bowl_move_speed.
    SPEED_JITTER_FRAC_DEFAULT = 0.20   # sample U(nom*(1-j), nom*(1+j)) — default ±20%

    BLOCK_HALF_W_DEFAULT = 0.02       # block half-width (x,y) -> narrow base (top-heavy)
    BLOCK_HALF_H_DEFAULT = 0.02      # block half-height (z) -> 12 cm tall
    BLOCK_MASS_DEFAULT = 0.05          # block mass (kg)
    BLOCK_COM_FRAC_DEFAULT = 0.55      # COM height as fraction of half-height above center (top-heavy)
    BLOCK_BELT_CLEARANCE_DEFAULT = 0.05  # min edge-to-edge gap from block to belt (m)
    BLOCK_Y_JITTER_DEFAULT = 0.05        # +/- y randomization about belt centerline (m)
    BLOCK_X_SPAN_DEFAULT = 0.06          # extra spawn depth beyond the clearance line (m)

    # Placement line: cube must first touch the belt before this offset from load edge.
    PLACE_LINE_OFFSET_DEFAULT = 0.10     # m; red line across belt (fail if contact past it)
    PLACE_LINE_HALF_X_DEFAULT = 0.003    # m; visual strip half-thickness along flow
    PLACE_LINE_HALF_Z_DEFAULT = 0.0015   # m; visual strip half-height above belt

    # Opt 2 — belt blocker rod
    BLOCKER_ENABLED_DEFAULT = False      # Opt 2 master switch: rod occluding part of the belt lane
    BLOCKER_START_CLEARANCE_DEFAULT = 0.15  # Opt 2: min near-face distance from load edge (m)
    BLOCKER_LENGTH_FRAC_MIN_DEFAULT = 0.20  # Opt 2: rod y-length as fraction of belt width
    BLOCKER_LENGTH_FRAC_MAX_DEFAULT = 0.30  # Opt 2: upper y-length frac
    BLOCKER_HALF_X_DEFAULT = 0.012       # Opt 2: nominal rod half-thickness along flow (m)
    BLOCKER_HALF_X_JITTER_FRAC_DEFAULT = 0.50  # Opt 2: sample half_x ∈ nom*(1±this)
    BLOCKER_HALF_Z_DEFAULT = 0.04        # rod half-height above belt surface (m)
    BLOCKER_END_MARGIN_DEFAULT = 0.04    # keep rod short of the exit edge (m)

    THETA_MAX_DEG_DEFAULT = 30.0       # tilt threshold for the metric / success

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's task_args block BEFORE init
        self._cfg = kwags.get("task_args", {}).get("place_block_belt", {})
        # reset per-episode belt/tilt state up front so a reused instance can't leak the
        # belt drive into the next episode's load_camera() (which steps the sim before
        # load_actors() rebinds self.block).
        self._belt_active = False
        self._released = False
        self._block_kinematic = False
        self.max_tilt_deg = 0.0
        self.tilt_score = 0.0
        self.reached_end = False
        self._reset_metric_state()
        self._block_dyn = None
        self._release_q = [1.0, 0.0, 0.0, 0.0]
        self._belt_q = [1.0, 0.0, 0.0, 0.0]
        self._belt_bands = []
        self._belt_band_offsets = np.zeros(0, dtype=np.float64)
        self._belt_band_phase = 0.0
        self._belt_band_start_x = 0.0
        self._belt_band_span = 0.0
        self._belt_band_z = 0.0
        self._belt_band_half_x = 0.0
        self._belt_band_min_x = 0.0
        self._belt_band_max_x = 0.0
        self._belt_drop_x = 0.0
        self._block_dropped = False
        self._drop_settle_steps = 0
        self.placed_on_belt = False
        self.dropped_at_start_left = False
        self.in_bowl = False
        self.bowl_id = None
        self._bowl_dyn = None
        self.bowl_start_z = 0.0
        self._bowl_base_x = 0.0
        self._bowl_base_y = 0.0
        self._bowl_move_dir = 1.0
        self._belt_start_zone_boundary = 0.0
        self._belt_contact_latched = False
        self._belt_contact_x = None
        self._belt_contact_y = None
        self.belt_mirrored = False
        self.belt_dir = 1.0
        self.blocker_enabled = False
        self.blocker = None
        self.blocker_x = 0.0
        self.blocker_y = 0.0
        self.blocker_half_y = 0.0
        self.blocker_half_x = 0.0
        self.clear_y_lo = 0.0
        self.clear_y_hi = 0.0
        self.hit_blocker = False
        self.avoided_blocker = True
        self.place_line = None
        self.place_line_x = 0.0
        super()._init_task_env_(**kwags)

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.belt_speed = float(cfg.get("belt_speed", self.BELT_SPEED_DEFAULT))
        self.belt_x_start = float(cfg.get("belt_x_start", self.BELT_X_START_DEFAULT))
        self.belt_x_end = float(cfg.get("belt_x_end", self.BELT_X_END_DEFAULT))
        self.belt_y = float(cfg.get("belt_y", self.BELT_Y_DEFAULT))
        self.belt_half_w = float(cfg.get("belt_half_w", self.BELT_HALF_W_DEFAULT))
        self.belt_ride_steps = int(cfg.get("belt_ride_steps", self.BELT_RIDE_STEPS_DEFAULT))
        self.belt_release_delay_steps = int(cfg.get("belt_release_delay_steps", self.BELT_RELEASE_DELAY_STEPS_DEFAULT))
        self.block_half_w = float(cfg.get("block_half_w", self.BLOCK_HALF_W_DEFAULT))
        self.block_half_h = float(cfg.get("block_half_h", self.BLOCK_HALF_H_DEFAULT))
        self.block_mass = float(cfg.get("block_mass", self.BLOCK_MASS_DEFAULT))
        self.block_com_frac = float(cfg.get("block_com_frac", self.BLOCK_COM_FRAC_DEFAULT))
        self.theta_max_deg = float(cfg.get("theta_max_deg", self.THETA_MAX_DEG_DEFAULT))
        self.block_belt_clearance = float(cfg.get("block_belt_clearance", self.BLOCK_BELT_CLEARANCE_DEFAULT))
        self.block_y_jitter = float(cfg.get("block_y_jitter", self.BLOCK_Y_JITTER_DEFAULT))
        self.block_x_span = float(cfg.get("block_x_span", self.BLOCK_X_SPAN_DEFAULT))
        random_belt_flip = bool(cfg.get("random_belt_flip", self.RANDOM_BELT_FLIP_DEFAULT))
        self.place_line_offset = float(
            cfg.get("place_line_offset", self.PLACE_LINE_OFFSET_DEFAULT)
        )
        self.blocker_enabled = bool(cfg.get("blocker_enabled", self.BLOCKER_ENABLED_DEFAULT))
        self.blocker_start_clearance = float(
            cfg.get("blocker_start_clearance", self.BLOCKER_START_CLEARANCE_DEFAULT)
        )
        self.blocker_length_frac_min = float(
            cfg.get("blocker_length_frac_min", self.BLOCKER_LENGTH_FRAC_MIN_DEFAULT)
        )
        self.blocker_length_frac_max = float(
            cfg.get("blocker_length_frac_max", self.BLOCKER_LENGTH_FRAC_MAX_DEFAULT)
        )
        self.blocker_half_x_nom = float(cfg.get("blocker_half_x", self.BLOCKER_HALF_X_DEFAULT))
        self.blocker_half_x_jitter_frac = float(
            cfg.get("blocker_half_x_jitter_frac", self.BLOCKER_HALF_X_JITTER_FRAC_DEFAULT)
        )
        self.blocker_half_x = float(self.blocker_half_x_nom)  # overwritten per-ep in _spawn_belt_blocker
        self.blocker_half_z = float(cfg.get("blocker_half_z", self.BLOCKER_HALF_Z_DEFAULT))
        self.blocker_end_margin = float(cfg.get("blocker_end_margin", self.BLOCKER_END_MARGIN_DEFAULT))

        self.bowl_x_offset = float(cfg.get("bowl_x_offset", self.BOWL_X_OFFSET_DEFAULT))
        self.bowl_y_offset = float(cfg.get("bowl_y_offset", self.BOWL_Y_OFFSET_DEFAULT))
        self.bowl_y_random_range = float(cfg.get("bowl_y_random_range", self.BOWL_Y_RANDOM_RANGE_DEFAULT))
        self.bowl_move_enabled = bool(cfg.get("bowl_move_enabled", self.BOWL_MOVE_ENABLED_DEFAULT))
        self.bowl_move_speed = float(cfg.get("bowl_move_speed", self.BOWL_MOVE_SPEED_DEFAULT))
        self.bowl_move_range = float(cfg.get("bowl_move_range", self.BOWL_MOVE_RANGE_DEFAULT))
        self.bowl_inner_radius = float(cfg.get("bowl_inner_radius", self.BOWL_INNER_RADIUS_DEFAULT))
        self.bowl_success_height = float(cfg.get("bowl_success_height", self.BOWL_SUCCESS_HEIGHT_DEFAULT))
        self.bowl_settle_steps = int(cfg.get("bowl_settle_steps", self.BOWL_SETTLE_STEPS_DEFAULT))

        # Per-episode speed sampling: nominal ± speed_jitter_frac (default ±20%).
        speed_jitter = float(cfg.get("speed_jitter_frac", self.SPEED_JITTER_FRAC_DEFAULT))
        speed_jitter = float(np.clip(speed_jitter, 0.0, 0.95))
        belt_nom = float(self.belt_speed)
        self.belt_speed = float(
            np.random.uniform(belt_nom * (1.0 - speed_jitter), belt_nom * (1.0 + speed_jitter))
        )
        if self.bowl_move_enabled:
            bowl_nom = float(self.bowl_move_speed)
            self.bowl_move_speed = float(
                np.random.uniform(bowl_nom * (1.0 - speed_jitter), bowl_nom * (1.0 + speed_jitter))
            )
        # Other light randomization
        self.block_half_h = float(np.random.uniform(self.block_half_h * 0.9, self.block_half_h * 1.1))
        self.block_mass = float(np.random.uniform(self.block_mass * 0.8, self.block_mass * 1.2))

        # Randomly mirror the whole layout about x=0 so either arm can be the placer.
        # Default: start=-0.2 → end=+0.2 (flow +x, left arm). Mirrored: start=+0.2 → end=-0.2.
        self.belt_mirrored = bool(random_belt_flip and (np.random.rand() < 0.5))
        if self.belt_mirrored:
            self.belt_x_start = -self.belt_x_start
            self.belt_x_end = -self.belt_x_end
        self.belt_dir = 1.0 if self.belt_x_end > self.belt_x_start else -1.0

        belt_step_dx = max(self.belt_speed * self.scene.get_timestep(), 1e-6)

        table_z = 0.74 + self.table_z_bias
        self.table_top_z = table_z

        # ------- support box to elevate the belt 10 cm above the table
        support_box_height = 0.10  # 10 cm elevation
        support_box_half_h = support_box_height * 0.5
        belt_cx = 0.5 * (self.belt_x_start + self.belt_x_end)
        belt_half_len_x = 0.5 * abs(self.belt_x_end - self.belt_x_start)
        belt_xy_padding = 0.02
        self._belt_cx = belt_cx
        self._belt_half_len_x = belt_half_len_x + belt_xy_padding
        self._belt_half_w = self.belt_half_w + belt_xy_padding
        # Red placement line: first belt contact must be before this x (else fail).
        self.place_line_x = float(self.belt_x_start + self.belt_dir * self.place_line_offset)
        self._belt_start_zone_boundary = float(self.place_line_x)

        support_box_pose = sapien.Pose(
            p=[belt_cx, self.belt_y, table_z + support_box_half_h],
            q=[1, 0, 0, 0],
        )
        self.support_box = create_box(
            scene=self,
            pose=support_box_pose,
            half_size=(self._belt_half_len_x, self._belt_half_w, support_box_half_h),
            color=(0.25, 0.25, 0.28),
            name="belt_support",
            is_static=True,
        )

        # ------- belt (match the holding/support box footprint)
        self.belt_thickness = 0.012
        belt_pose = sapien.Pose(
            p=[belt_cx, self.belt_y, table_z + support_box_height + self.belt_thickness * 0.5],
            q=[1, 0, 0, 0],
        )
        self.belt = create_box(
            scene=self,
            pose=belt_pose,
            half_size=(self._belt_half_len_x, self._belt_half_w, self.belt_thickness * 0.5),
            color=(0.10, 0.10, 0.12),
            name="conveyor_belt",
            is_static=True,
        )
        self.belt_surface_z = table_z + support_box_height + self.belt_thickness  # top face of the belt
        self._init_belt_visual_motion()
        self._spawn_place_line()
        self._spawn_belt_blocker()

        # ------- receiving bowl on the table at the exit side of the belt
        bowl_x_clearance = max(self.bowl_x_offset, self.bowl_inner_radius + 0.005)
        bowl_x = self._belt_cx + self.belt_dir * (self._belt_half_len_x + bowl_x_clearance)
        full_y_lo = float(self.belt_y - self.belt_half_w)
        full_y_hi = float(self.belt_y + self.belt_half_w)
        if self.bowl_move_enabled:
            # Opt 1: bowl always traverses the FULL belt y-range, even when Opt 2
            # blocker is on (expert still releases only in the clear lane).
            bowl_lane_low, bowl_lane_high = full_y_lo, full_y_hi
            bowl_y = bowl_lane_low
        elif self.blocker_enabled and self.blocker is not None and self.clear_y_hi > self.clear_y_lo:
            # Static bowl + blocker: seat the bowl in the clear lane.
            bowl_lane_low, bowl_lane_high = float(self.clear_y_lo), float(self.clear_y_hi)
            bowl_y = float(np.random.uniform(bowl_lane_low, bowl_lane_high))
        else:
            bowl_lane_low, bowl_lane_high = full_y_lo, full_y_hi
            bowl_y = float(np.random.uniform(bowl_lane_low, bowl_lane_high))
        self.bowl_id = int(np.random.choice([1, 2, 3, 4, 5, 6, 7]))
        # Convex decomposition keeps a usable cavity and works for both static and
        # kinematic Opt-1 bowls (triangle-mesh / nonconvex is static-only in PhysX).
        self.bowl = create_actor(
            self,
            pose=sapien.Pose([bowl_x, bowl_y, table_z], [0.5, 0.5, 0.5, 0.5]),
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=not self.bowl_move_enabled,
        )
        self._bowl_dyn = None
        for c in self.bowl.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._bowl_dyn = c
                if self.bowl_move_enabled:
                    try:
                        c.set_disable_gravity(True)
                        c.set_kinematic(True)
                    except Exception:
                        pass
            # Sticky bowl interior so the cube seats instead of skating through.
            if hasattr(c, "get_collision_shapes"):
                for sh in c.get_collision_shapes():
                    try:
                        m = sh.get_physical_material()
                        m.set_static_friction(1.8)
                        m.set_dynamic_friction(1.5)
                        m.set_restitution(0.0)
                    except Exception:
                        pass
        self.bowl_start_z = float(self.bowl.get_pose().p[2])
        self._bowl_base_x = float(bowl_x)
        self._bowl_base_y = float(bowl_y)
        self._bowl_move_dir = 1.0 if self.bowl_move_enabled else float(np.random.choice([-1.0, 1.0]))
        # Release only after the whole block has cleared the physical belt edge;
        # otherwise the dynamic block can remain supported and appear to stop there.
        self._belt_drop_x = self._belt_cx + self.belt_dir * (
            self._belt_half_len_x + self.block_half_w + 0.005
        )
        ride_dist = abs(self._belt_drop_x - self.belt_x_start) + 0.02
        min_ride_steps = int(np.ceil(ride_dist / belt_step_dx)) + self.bowl_settle_steps
        self.belt_ride_steps = max(self.belt_ride_steps, min_ride_steps)

        # ------- tall top-heavy block on the load/start side, >= clearance from the belt edge,
        # with y jittered +/- block_y_jitter about the belt centerline.
        start_edge_x = self._belt_cx - self.belt_dir * self._belt_half_len_x
        clear = max(0.0, self.block_belt_clearance)
        # Closest allowed block center (edge-to-edge gap == clear).
        closest_x = start_edge_x - self.belt_dir * (self.block_half_w + clear)
        farthest_x = closest_x - self.belt_dir * self.block_x_span
        block_x_lo = float(min(closest_x, farthest_x))
        block_x_hi = float(max(closest_x, farthest_x))
        # Keep the block on the same table half as the load edge (arm reachability).
        if self.belt_dir > 0:
            block_x_hi = min(block_x_hi, -0.02)
            block_x_lo = max(block_x_lo, -0.34)
        else:
            block_x_lo = max(block_x_lo, 0.02)
            block_x_hi = min(block_x_hi, 0.34)
        if block_x_hi < block_x_lo + 1e-3:
            # Degenerate after clamping: pin to the closest valid center.
            block_x = float(np.clip(closest_x, -0.34, 0.34))
            if abs(block_x) < 0.02:
                block_x = -0.05 if self.belt_dir > 0 else 0.05
        else:
            block_x = float(np.random.uniform(block_x_lo, block_x_hi))
        block_y = float(self.belt_y + np.random.uniform(-self.block_y_jitter, self.block_y_jitter))
        # Long-box contact frames are approach-asymmetric: left-arm loads use +90° yaw,
        # right-arm (mirrored) loads need the mirrored -90° frame or grasp IK fails.
        block_q = get_quat_euler([0, 90, 0]) if self.belt_dir > 0 else get_quat_euler([0, -90, 0])
        block_pose = rand_pose(
            xlim=[block_x, block_x],
            ylim=[block_y, block_y],
            zlim=[table_z + self.block_half_h],
            qpos=block_q,
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 18],
        )
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(self.block_half_w, self.block_half_w, self.block_half_h),
            color=(0.85, 0.35, 0.10),
            name="tall_block",
            boxtype="long",
            is_static=False,
        )
        self.block.set_mass(self.block_mass)

        # make the block TOP-HEAVY: raise its center of mass well above the geometric center
        self._block_dyn = None
        for c in self.block.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._block_dyn = c
                try:
                    cm = c.get_cmass_local_pose()
                    com_z = self.block_com_frac * self.block_half_h
                    c.set_cmass_local_pose(sapien.Pose(p=[cm.p[0], cm.p[1], com_z], q=cm.q))
                except Exception:
                    pass
                # Friction for belt ride + bowl catch; no bounce into/through the bowl.
                try:
                    for sh in c.get_collision_shapes():
                        m = sh.get_physical_material()
                        m.set_static_friction(1.6)
                        m.set_dynamic_friction(1.4)
                        m.set_restitution(0.0)
                except Exception:
                    pass

        self.add_prohibit_area(self.belt, padding=0.03)
        self.add_prohibit_area(self.bowl, padding=0.04)
        self.add_prohibit_area(self.block, padding=0.05)
        if self.blocker is not None:
            self.add_prohibit_area(self.blocker, padding=0.02)

        # ------- belt / tilt bookkeeping
        self._belt_active = False
        self._released = False
        self._block_kinematic = False
        self.max_tilt_deg = 0.0
        self.tilt_score = 0.0
        self.reached_end = False
        self._reset_metric_state()
        self._ride_steps_done = 0
        self._block_dropped = False
        self._drop_settle_steps = 0
        self.placed_on_belt = False
        self.dropped_at_start_left = False
        self.in_bowl = False
        self.hit_blocker = False
        self.avoided_blocker = True
        self._bowl_move_dir = float(np.random.choice([-1.0, 1.0]))
        self._belt_contact_latched = False
        self._belt_contact_x = None
        self._belt_contact_y = None

    def _spawn_place_line(self):
        """Red strip across the belt at ``place_line_offset`` past the load edge.

        Visual cue for the placement rule: first contact must be before this line.
        """
        half_x = float(self.PLACE_LINE_HALF_X_DEFAULT)
        half_z = float(self.PLACE_LINE_HALF_Z_DEFAULT)
        half_y = max(float(self._belt_half_w) - 0.002, 0.01)
        line_z = self.belt_surface_z + self.BELT_BAND_Z_OFFSET_DEFAULT + half_z + 0.0005
        self.place_line = create_visual_box(
            scene=self,
            pose=sapien.Pose(
                [self.place_line_x, self.belt_y, line_z],
                [1, 0, 0, 0],
            ),
            half_size=(half_x, half_y, half_z),
            color=(0.95, 0.08, 0.08),
            name="place_line",
        )

    # ----------------------------------------------------------------- optional belt blocker
    def _spawn_belt_blocker(self):
        """Place a static rod on the belt that occludes part of the y-lane.

        - x position: near face ≥ ``blocker_start_clearance`` (default 15 cm) past load edge
        - y size (width across belt): 20–30% of belt width
        - x size (thickness along flow): nominal ``blocker_half_x`` ± 50%
        Flush to a random y-edge so a contiguous clear lane remains for placement.
        """
        self.blocker = None
        self.hit_blocker = False
        self.avoided_blocker = True
        belt_width = 2.0 * float(self.belt_half_w)
        # Full-width clear lane when disabled (expert can use any y on the belt).
        self.clear_y_lo = float(self.belt_y - self.belt_half_w)
        self.clear_y_hi = float(self.belt_y + self.belt_half_w)
        self.blocker_x = 0.0
        self.blocker_y = 0.0
        self.blocker_half_y = 0.0
        if not self.blocker_enabled:
            return

        # y extent: 20–30% of belt width
        frac_lo = min(self.blocker_length_frac_min, self.blocker_length_frac_max)
        frac_hi = max(self.blocker_length_frac_min, self.blocker_length_frac_max)
        frac = float(np.random.uniform(frac_lo, frac_hi))
        rod_len_y = float(np.clip(frac * belt_width, 0.02, belt_width * 0.9))
        self.blocker_half_y = 0.5 * rod_len_y

        # x thickness: nominal half_x ± jitter (default ±50%)
        x_jit = float(np.clip(self.blocker_half_x_jitter_frac, 0.0, 0.95))
        nom_hx = max(float(self.blocker_half_x_nom), 1e-4)
        self.blocker_half_x = float(np.random.uniform(nom_hx * (1.0 - x_jit), nom_hx * (1.0 + x_jit)))

        end_edge_x = self._belt_cx + self.belt_dir * self._belt_half_len_x
        # Near face of the rod must be ≥ clearance past the logical load edge (belt_x_start).
        clear = max(float(self.blocker_start_clearance), 0.0)
        x_a = self.belt_x_start + self.belt_dir * (clear + self.blocker_half_x)
        x_b = end_edge_x - self.belt_dir * max(self.blocker_end_margin, self.blocker_half_x)
        x_lo, x_hi = (x_a, x_b) if x_a <= x_b else (x_b, x_a)
        if x_hi < x_lo + 1e-3:
            self.blocker_x = float(x_a)
        else:
            self.blocker_x = float(np.random.uniform(x_lo, x_hi))

        # Flush rod to a random belt y-edge so the complementary side stays open.
        flush_low = bool(np.random.rand() < 0.5)
        y_low = float(self.belt_y - self.belt_half_w)
        y_high = float(self.belt_y + self.belt_half_w)
        if flush_low:
            self.blocker_y = y_low + self.blocker_half_y
            self.clear_y_lo = self.blocker_y + self.blocker_half_y
            self.clear_y_hi = y_high
        else:
            self.blocker_y = y_high - self.blocker_half_y
            self.clear_y_lo = y_low
            self.clear_y_hi = self.blocker_y - self.blocker_half_y

        # Keep a usable clear lane for the cube (+ small margin).
        lane_margin = self.block_half_w + 0.01
        self.clear_y_lo = float(self.clear_y_lo + lane_margin)
        self.clear_y_hi = float(self.clear_y_hi - lane_margin)
        if self.clear_y_hi < self.clear_y_lo + 0.02:
            # Degenerate: shrink rod and reopen a minimal lane on the opposite side.
            self.blocker_half_y = 0.5 * 0.2 * belt_width
            if flush_low:
                self.blocker_y = y_low + self.blocker_half_y
                self.clear_y_lo = self.blocker_y + self.blocker_half_y + lane_margin
                self.clear_y_hi = y_high - lane_margin
            else:
                self.blocker_y = y_high - self.blocker_half_y
                self.clear_y_lo = y_low + lane_margin
                self.clear_y_hi = self.blocker_y - self.blocker_half_y - lane_margin

        rod_z = self.belt_surface_z + self.blocker_half_z
        self.blocker = create_box(
            scene=self,
            pose=sapien.Pose(
                p=[self.blocker_x, self.blocker_y, rod_z],
                q=[1, 0, 0, 0],
            ),
            half_size=(self.blocker_half_x, self.blocker_half_y, self.blocker_half_z),
            color=(0.75, 0.15, 0.12),
            name="belt_blocker",
            is_static=True,
        )

    def _y_overlaps_blocker(self, y: float) -> bool:
        if not self.blocker_enabled or self.blocker is None:
            return False
        return bool(abs(float(y) - self.blocker_y) <= (self.blocker_half_y + self.block_half_w + 0.005))

    def _y_in_clear_lane(self, y: float) -> bool:
        y = float(y)
        if not self.blocker_enabled or self.blocker is None:
            return True
        return bool(self.clear_y_lo <= y <= self.clear_y_hi)

    def _choose_release_lane_y(self, preferred_y: float | None = None) -> float:
        """Pick a release y on the belt; with a blocker, stay inside the clear lane."""
        if preferred_y is None:
            preferred_y = float(self.belt_y)
        preferred_y = float(preferred_y)
        if not self.blocker_enabled or self.blocker is None:
            return preferred_y
        if self.clear_y_lo <= preferred_y <= self.clear_y_hi:
            return preferred_y
        # Prefer the clear-lane point closest to the preferred y (usually bowl lane).
        return float(np.clip(preferred_y, self.clear_y_lo, self.clear_y_hi))

    # ----------------------------------------------------------------- tilt helper
    def _current_tilt_deg(self):
        # angle between the block's local +z (its long axis) and the world +z axis
        m = self.block.get_pose().to_transformation_matrix()
        up = m[:3, 2]
        up = up / (np.linalg.norm(up) + 1e-9)
        cos_t = float(np.clip(abs(up[2]), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))

    def _on_belt(self):
        p = self.block.get_pose().p
        near_belt = (
            abs(p[0] - self._belt_cx) <= (self._belt_half_len_x + 0.03)
            and abs(p[1] - self.belt_y) <= (self._belt_half_w + 0.03)
            and p[2] < (self.belt_surface_z + self.block_half_h + 0.03)
        )
        return bool(near_belt)

    def _block_in_bowl(self):
        if getattr(self, "bowl", None) is None or getattr(self, "block", None) is None:
            return False
        block_p = np.array(self.block.get_pose().p, dtype=np.float64)
        bowl_p = np.array(self.bowl.get_pose().p, dtype=np.float64)
        xy_close = float(np.linalg.norm(block_p[:2] - bowl_p[:2])) <= self.bowl_inner_radius
        below_rim = block_p[2] <= (self.bowl_start_z + self.bowl_success_height)
        above_table = block_p[2] >= (self.table_top_z - 0.02)
        return bool(xy_close and below_rim and above_table)

    def _is_start_drop_x(self, x: float) -> bool:
        """True if first contact is before the red placement line (not past it)."""
        x = float(x)
        boundary = float(
            getattr(self, "place_line_x", getattr(self, "_belt_start_zone_boundary", self.belt_x_start))
        )
        if getattr(self, "belt_dir", 1.0) > 0:
            return bool(x <= boundary)
        return bool(x >= boundary)

    def _latch_belt_contact(self):
        if getattr(self, "_belt_contact_latched", False) or getattr(self, "block", None) is None:
            return
        if not self._on_belt():
            return
        p = np.array(self.block.get_pose().p, dtype=np.float64)
        self._belt_contact_x = float(p[0])
        self._belt_contact_y = float(p[1])
        # Keep legacy key name; means "dropped at the belt start / load zone".
        self.dropped_at_start_left = self._is_start_drop_x(self._belt_contact_x)
        self._belt_contact_latched = True

    def _bowl_lane_bounds(self):
        """Y-range for Opt 1 bowl oscillation.

        Always the full belt width — even with Opt 2 blocker active — so the bowl
        still travels up and down the entire exit lane. Cube release stays constrained
        to the clear lane via ``_choose_release_lane_y``.
        """
        return float(self.belt_y - self.belt_half_w), float(self.belt_y + self.belt_half_w)

    def _advance_bowl_motion(self):
        if not getattr(self, "bowl_move_enabled", False) or getattr(self, "bowl", None) is None:
            return
        pose = self.bowl.get_pose()
        prev_y = float(pose.p[1])
        dt = float(self.scene.get_timestep())
        next_y = float(prev_y + self._bowl_move_dir * abs(self.bowl_move_speed) * dt)
        low, high = self._bowl_lane_bounds()
        if high <= low:
            return
        if next_y > high:
            next_y = high
            self._bowl_move_dir = -1.0
        elif next_y < low:
            next_y = low
            self._bowl_move_dir = 1.0
        dy = next_y - prev_y
        # Latch *before* moving the bowl. Use XY-over-bowl (not full in_bowl):
        # success height is only reached after ~0.03 m of fall, by which time a
        # set_pose bowl has already walked out from under the cube.
        ride = False
        if (
            abs(dy) > 1e-9
            and getattr(self, "block", None) is not None
            and not getattr(self, "_block_kinematic", False)
            and getattr(self, "_block_dropped", False)
        ):
            bp = np.array(self.block.get_pose().p, dtype=np.float64)
            xy = float(np.linalg.norm(bp[:2] - np.array([float(pose.p[0]), prev_y])))
            ride = bool(xy <= float(self.bowl_inner_radius))
        new_pose = sapien.Pose([self._bowl_base_x, next_y, pose.p[2]], pose.q)
        try:
            if self._bowl_dyn is not None:
                self._bowl_dyn.set_kinematic_target(new_pose)
            self.bowl.actor.set_pose(new_pose)
        except Exception:
            try:
                self.bowl.actor.set_pose(new_pose)
            except Exception:
                pass
        if ride:
            bp = self.block.get_pose()
            # Keep the cube on the bowl centerline in y while it falls in.
            # Incremental +=dy still drifts under PhysX contact with a set_pose bowl.
            carried = sapien.Pose(
                [float(bp.p[0]), next_y, float(bp.p[2])],
                list(bp.q),
            )
            try:
                self.block.actor.set_pose(carried)
                if self._block_dyn is not None:
                    v = np.array(self._block_dyn.get_linear_velocity(), dtype=np.float64)
                    self._block_dyn.set_linear_velocity(
                        np.array([float(v[0]), dy / max(dt, 1e-6), float(v[2])])
                    )
            except Exception:
                pass

    def _belt_travel_time(self, release_x: float) -> float:
        """Seconds for the cube to ride from ``release_x`` to the belt exit drop."""
        return abs(float(self._belt_drop_x) - float(release_x)) / max(self.belt_speed, 1e-8)

    def _predict_bowl_y(self, future_time: float) -> float:
        """Bowl y after ``future_time`` (same clamp/flip stepper as ``_advance_bowl_motion``)."""
        if getattr(self, "bowl", None) is None:
            return float(self.belt_y)
        pose = self.bowl.get_pose()
        y_now = float(pose.p[1])
        if (not getattr(self, "bowl_move_enabled", False)) or future_time <= 0.0:
            return y_now

        speed = abs(float(self.bowl_move_speed))
        if speed <= 1e-8:
            return y_now

        low, high = self._bowl_lane_bounds()
        if high <= low:
            return float(np.clip(y_now, low, high))

        dt = float(self.scene.get_timestep())
        n_steps = int(max(0, round(float(future_time) / max(dt, 1e-8))))
        y = float(np.clip(y_now, low, high))
        direction = 1.0 if self._bowl_move_dir >= 0 else -1.0
        for _ in range(n_steps):
            next_y = float(y + direction * speed * dt)
            if next_y > high:
                next_y = high
                direction = -1.0
            elif next_y < low:
                next_y = low
                direction = 1.0
            y = next_y
        return float(np.clip(y, low, high))

    def _estimate_drop_lane_y(self, release_x: float, release_time: float = 0.0) -> float:
        """Predicted bowl y when the cube reaches the belt exit.

        drop_time = release_time + belt_length_remaining / belt_speed
        """
        if getattr(self, "bowl", None) is None:
            return float(self.belt_y)
        if not getattr(self, "bowl_move_enabled", False):
            return float(self.bowl.get_pose().p[1])
        return self._predict_bowl_y(release_time + self._belt_travel_time(release_x))

    def _past_belt_drop(self, x: float) -> bool:
        x = float(x)
        if getattr(self, "belt_dir", 1.0) > 0:
            return bool(x >= self._belt_drop_x)
        return bool(x <= self._belt_drop_x)

    def _past_belt_end(self, x: float) -> bool:
        x = float(x)
        if getattr(self, "belt_dir", 1.0) > 0:
            return bool(x >= self.belt_x_end)
        return bool(x <= self.belt_x_end)

    def _move_block_to_belt_clearance(self, arm_tag: ArmTag, clearance: float):
        block_bottom_z = float(self.block.get_pose().p[2]) - self.block_half_h
        dz = (self.belt_surface_z + float(clearance)) - block_bottom_z
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz))

    def _dwell(self, n_steps: int):
        for i in range(max(0, int(n_steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _bowl_align_tol(self) -> float:
        # Tight: first-within-window was the *edge* of the bowl and still missed.
        return float(min(self.bowl_inner_radius * 0.12, 0.008))

    def _bowl_align_max_steps(self) -> int:
        """Cover a bit more than two bowl oscillation periods."""
        dt = float(self.scene.get_timestep())
        low, high = self._bowl_lane_bounds()
        span = max(float(high - low), 1e-3)
        speed = max(abs(float(self.bowl_move_speed)), 1e-6)
        period = 2.0 * span / speed
        return int(np.ceil(2.2 * period / max(dt, 1e-8))) + 50

    def _gripper_open_lead_time(self, arm_tag: ArmTag) -> float:
        """Sim time the upcoming open_gripper action will take (plan only, no motion)."""
        dt = float(self.scene.get_timestep())
        try:
            if arm_tag == "right":
                res = self.set_gripper(right_pos=1.0, set_tag="right")
            else:
                res = self.set_gripper(left_pos=1.0, set_tag="left")
            n = int(res.get("num_step", 50)) if res else 50
        except Exception:
            n = 50
        return max(0, n) * dt

    def _bowl_y_trajectory(self, n_steps: int) -> np.ndarray:
        """Bowl y after 0..n_steps of the same clamp/flip driver as ``_advance_bowl_motion``."""
        y = float(self.bowl.get_pose().p[1]) if getattr(self, "bowl", None) is not None else float(self.belt_y)
        n_steps = max(0, int(n_steps))
        if not getattr(self, "bowl_move_enabled", False) or n_steps == 0:
            return np.full(n_steps + 1, y, dtype=np.float64)
        speed = abs(float(self.bowl_move_speed))
        dt = float(self.scene.get_timestep())
        low, high = self._bowl_lane_bounds()
        y = float(np.clip(y, low, high))
        direction = 1.0 if self._bowl_move_dir >= 0 else -1.0
        out = np.empty(n_steps + 1, dtype=np.float64)
        for i in range(n_steps + 1):
            out[i] = y
            next_y = float(y + direction * speed * dt)
            if next_y > high:
                next_y = high
                direction = -1.0
            elif next_y < low:
                next_y = low
                direction = 1.0
            y = next_y
        return out

    def _steps_until_drop_alignment(
        self,
        release_x: float,
        release_y: float,
        lead_time: float = 0.0,
        max_steps: int | None = None,
        tol: float | None = None,
    ) -> int:
        if not getattr(self, "bowl_move_enabled", False):
            return 0

        dt = float(self.scene.get_timestep())
        tol = float(tol if tol is not None else self._bowl_align_tol())
        if max_steps is None:
            max_steps = self._bowl_align_max_steps()
        max_steps = max(0, int(max_steps))
        lead_steps = int(round(float(lead_time) / max(dt, 1e-8)))
        travel_steps = int(round(self._belt_travel_time(release_x) / max(dt, 1e-8)))
        land_offset = max(0, lead_steps + travel_steps)
        traj = self._bowl_y_trajectory(max_steps + land_offset)
        target = float(release_y)
        best_steps = 0
        best_err = float("inf")
        for step in range(max_steps + 1):
            err = abs(float(traj[step + land_offset]) - target)
            if err < best_err:
                best_err = err
                best_steps = step
            if err <= tol:
                return step
        return best_steps

    def _wait_for_bowl_alignment(
        self,
        release_x: float,
        release_y: float,
        lead_time: float = 0.0,
    ):
        """Hold still until the bowl will be in ``release_y`` when the cube lands.

        Closed-loop: coarse wait from the trajectory search, then step until the
        live bowl pose predicts an intercept (same pattern as ``load_train``).
        """
        if not getattr(self, "bowl_move_enabled", False):
            return
        tol = self._bowl_align_tol()
        max_steps = self._bowl_align_max_steps()
        wait = self._steps_until_drop_alignment(
            release_x=release_x,
            release_y=release_y,
            lead_time=lead_time,
            max_steps=max_steps,
            tol=tol,
        )
        self._dwell(wait)
        for _ in range(max_steps):
            err = abs(
                self._estimate_drop_lane_y(release_x=release_x, release_time=lead_time)
                - float(release_y)
            )
            if err <= tol:
                break
            self._dwell(1)

    # ----------------------------------------------------------------- belt visuals
    def _init_belt_visual_motion(self):
        # Render-only tread bands travel with the top surface so the conveyor visibly moves.
        self._belt_bands = []
        self._belt_band_phase = 0.0

        band_half_x = float(self.BELT_BAND_HALF_X_DEFAULT)
        self._belt_band_half_x = band_half_x
        band_half_y = 0.0
        band_half_z = float(self.BELT_BAND_HALF_Z_DEFAULT)
        edge_margin = 0.002
        band_half_y = max(self._belt_half_w - edge_margin, 0.001)

        self._belt_band_min_x = self._belt_cx - self._belt_half_len_x + band_half_x + edge_margin
        self._belt_band_max_x = self._belt_cx + self._belt_half_len_x - band_half_x - edge_margin
        # Bands originate at the load edge and travel toward the exit.
        self._belt_band_start_x = (
            self._belt_band_min_x if self.belt_dir > 0 else self._belt_band_max_x
        )
        self._belt_band_span = max(self._belt_band_max_x - self._belt_band_min_x, 1e-6)
        self._belt_band_z = self.belt_surface_z + self.BELT_BAND_Z_OFFSET_DEFAULT + band_half_z

        spacing = max(float(self.BELT_BAND_SPACING_DEFAULT), band_half_x * 2.5)
        band_count = max(4, int(np.ceil(max(self._belt_band_span, spacing) / spacing)))
        self._belt_band_offsets = np.linspace(0.0, self._belt_band_span, band_count, endpoint=False)

        band_colors = (
            (0.22, 0.22, 0.24),
            (0.32, 0.32, 0.35),
        )
        for i, offset in enumerate(self._belt_band_offsets):
            x = self._belt_band_start_x + self.belt_dir * float(offset)
            x = float(np.clip(x, self._belt_band_min_x, self._belt_band_max_x))
            band = create_visual_box(
                scene=self,
                pose=sapien.Pose([x, self.belt_y, self._belt_band_z], [1, 0, 0, 0]),
                half_size=(band_half_x, band_half_y, band_half_z),
                color=band_colors[i % len(band_colors)],
                name=f"belt_band_{i}",
            )
            self._belt_bands.append(band)

    def _advance_belt_visual_motion(self):
        if not getattr(self, "_belt_bands", None) or self._belt_band_span <= 0.0:
            return

        self._belt_band_phase = (
            self._belt_band_phase + self.belt_speed * self.scene.get_timestep()
        ) % self._belt_band_span
        for band, offset in zip(self._belt_bands, self._belt_band_offsets):
            x = self._belt_band_start_x + self.belt_dir * float(
                (offset + self._belt_band_phase) % self._belt_band_span
            )
            x = float(np.clip(x, self._belt_band_min_x, self._belt_band_max_x))
            band.set_pose(sapien.Pose([x, self.belt_y, self._belt_band_z], [1, 0, 0, 0]))

    # ----------------------------------------------------------------- per-step hook (belt drive)
    def _update_kinematic_tasks(self):
        # base hook first (drives DOMINO dynamic-object motion, if any)
        super()._update_kinematic_tasks()
        self._advance_belt_visual_motion()
        self._advance_bowl_motion()
        self._track_belt_metrics()

        if getattr(self, "_released", False) and getattr(self, "block", None) is not None:
            t = self._current_tilt_deg()
            self.max_tilt_deg = max(self.max_tilt_deg, t)
            if self._on_belt():
                self.placed_on_belt = True
                self._latch_belt_contact()
            self.in_bowl = self._block_in_bowl()

        # only drive the block while it is still being carried by the belt
        if getattr(self, "_belt_active", False) and getattr(self, "block", None) is not None:
            if self._block_dyn is not None and getattr(self, "_released", False) and self._on_belt():
                # The belt grabs the block. On the first captured step, hand the block over to the
                # belt's kinematic drive: snap it onto the belt surface in its (released, upright)
                # orientation so the moving belt carries it cleanly. A grossly mismatched release
                # is what produces a large pre-capture tilt (recorded above into max_tilt_deg).
                if not self._block_kinematic:
                    try:
                        self._block_dyn.set_linear_velocity(np.zeros(3))
                        self._block_dyn.set_angular_velocity(np.zeros(3))
                        self._block_dyn.set_kinematic(True)
                    except Exception:
                        pass
                    self._block_kinematic = True
                    self._belt_q = self._release_q  # ride upright, as released
                dt = self.scene.get_timestep()
                pose = self.block.get_pose()
                new_p = np.array(pose.p, dtype=np.float64)
                new_p[0] += self.belt_dir * self.belt_speed * dt
                # keep the block seated on the belt surface
                new_p[2] = self.belt_surface_z + self.block_half_h

                # If the cube is riding in the blocked y-lane, collide with the rod:
                # stop the kinematic conveyor hand-off and let physics take over.
                if (
                    self.blocker_enabled
                    and self.blocker is not None
                    and self._y_overlaps_blocker(new_p[1])
                    and abs(new_p[0] - self.blocker_x) <= (self.blocker_half_x + self.block_half_w + 0.01)
                ):
                    self.hit_blocker = True
                    self.avoided_blocker = False
                    self._belt_active = False
                    self.block.actor.set_pose(sapien.Pose(p=new_p.tolist(), q=self._belt_q))
                    if self._block_dyn is not None:
                        try:
                            self._block_dyn.set_kinematic(False)
                            self._block_dyn.set_linear_velocity(
                                np.array([self.belt_dir * self.belt_speed, 0.0, 0.0])
                            )
                            self._block_dyn.set_angular_velocity(np.zeros(3))
                        except Exception:
                            pass
                    self._block_kinematic = False
                    return

                self.block.actor.set_pose(sapien.Pose(p=new_p.tolist(), q=self._belt_q))
                if self._past_belt_drop(new_p[0]):
                    self.reached_end = True
                    self._belt_active = False
                    self._block_dropped = True
                    self._drop_settle_steps = 0
                    if self._block_dyn is not None:
                        try:
                            self._block_dyn.set_kinematic(False)
                            self._block_dyn.set_linear_velocity(
                                np.array([self.belt_dir * self.belt_speed, 0.0, 0.0])
                            )
                            self._block_dyn.set_angular_velocity(np.zeros(3))
                        except Exception:
                            pass
                    self._block_kinematic = False
                elif self._past_belt_end(new_p[0]):
                    self.reached_end = True

    def _ride_belt(self):
        # Dwell while the belt carries the block to the end, then keep recording through the
        # configured post-drop settle window so the final fall into the bowl appears in saved video.
        for i in range(self.belt_ride_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            self._ride_steps_done += 1
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._block_dropped:
                self._drop_settle_steps += 1
                if self._drop_settle_steps >= self.bowl_settle_steps:
                    for j in range(20):
                        self._update_kinematic_tasks()
                        self.scene.step()
                        if self.save_freq and (j % self.save_freq == 0):
                            self._take_picture()
                    self.in_bowl = self._block_in_bowl()
                    if self.save_freq:
                        self._take_picture()
                    break

    # ----------------------------------------------------------------- policy
    def play_once(self):
        # Arm on the load side: left for default (+x flow), right for mirrored (-x flow).
        arm_tag = ArmTag("right" if self.block.get_pose().p[0] > 0 else "left")
        moving_bowl = bool(getattr(self, "bowl_move_enabled", False))

        # 1) grasp the tall block, then lift clear of the table.
        self.move(self.grasp_actor(self.block, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="arm"))

        # Release at the load/start of the belt, always before the red placement line.
        match_dist = max(0.02, self.belt_speed * 0.5)
        max_match = max(
            0.01,
            abs(float(self.place_line_x) - float(self.belt_x_start)) - (self.block_half_w + 0.01),
        )
        match_dist = float(min(match_dist, max_match))
        hover_x = self.belt_x_start
        release_x = hover_x + self.belt_dir * match_dist

        # 2) Choose WHERE to drop (release y on the belt).
        #    Opt1: fixed lane (belt center / clear lane). Do not chase the bowl —
        #    wait until the bowl will be under that lane when the cube exits.
        #    Static bowl: place in the bowl's current y.
        if moving_bowl:
            preferred_y = float(self.belt_y)
        else:
            preferred_y = (
                float(self.bowl.get_pose().p[1])
                if getattr(self, "bowl", None) is not None
                else float(self.belt_y)
            )
        target_lane_y = self._choose_release_lane_y(preferred_y)

        # Move over the chosen lane and lower onto the belt.
        dx = hover_x - float(self.block.get_pose().p[0])
        dy = target_lane_y - float(self.block.get_pose().p[1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))
        self._move_block_to_belt_clearance(arm_tag=arm_tag, clearance=0.005)

        # 3) Match belt speed with a short +x stroke, then we are at the release pose.
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=self.belt_dir * match_dist))

        # 4) Opt1 timing: knowns are bowl_speed, belt_speed, and remaining belt length.
        #    drop_time = now + open_gripper + settle + (drop_x - release_x) / belt_speed
        #    Wait until predicted bowl_y(drop_time) ≈ release_y, then open.
        if moving_bowl:
            release_x = float(self.block.get_pose().p[0])
            release_y = float(self.block.get_pose().p[1])
            lead_time = (
                self._gripper_open_lead_time(arm_tag)
                + self.belt_release_delay_steps * self.scene.get_timestep()
            )
            self._wait_for_bowl_alignment(
                release_x=release_x,
                release_y=release_y,
                lead_time=lead_time,
            )

        # 5) Release onto the belt; belt drive carries the cube to the exit / bowl.
        self._release_q = [1.0, 0.0, 0.0, 0.0]
        self._released = True
        self.move(self.open_gripper(arm_tag))
        self._dwell(self.belt_release_delay_steps)
        self._belt_active = True

        self._ride_belt()

        self.tilt_score = float(np.clip(1.0 - self.max_tilt_deg / self.theta_max_deg, 0.0, 1.0))

        if self.blocker_enabled and self.blocker is not None and not self.hit_blocker:
            contact_y = self._belt_contact_y
            if contact_y is None:
                contact_y = float(self.block.get_pose().p[1])
            self.avoided_blocker = bool(self._y_in_clear_lane(contact_y))

        self.info["info"] = {
            "{A}": "tall_block",
            "{B}": f"002_bowl/base{self.bowl_id}",
            "{C}": "conveyor_belt",
            "{a}": str(arm_tag),
            "{flip}": "mirrored" if self.belt_mirrored else "default",
            "{opt1}": "on" if self.bowl_move_enabled else "off",
            "{opt2}": "on" if self.blocker_enabled else "off",
        }
        return self.info

    # ------------------------------------------------- experiment metrics
    def _reset_metric_state(self):
        """Clear every per-episode metric latch (called from each reset site)."""
        self._metric_lift_step = None       # block first clear of the table
        self._metric_contact_step = None    # block first seated on the belt
        self._metric_drop_step = None       # block left the belt at the exit
        self._metric_bowl_step = None       # block first inside the bowl
        self._metric_min_bowl_offset = None # closest horizontal approach after the drop

    def _metric_step(self) -> int:
        return int(getattr(self, "_exp_sim_steps", 0) or 0)

    def _track_belt_metrics(self):
        """Per-step latches: lift, belt contact, exit drop, closest bowl approach."""
        try:
            if getattr(self, "block", None) is None:
                return
            p = np.array(self.block.get_pose().p, dtype=np.float64)
            if (self._metric_lift_step is None
                    and float(p[2]) >= float(self.table_top_z) + float(self.block_half_h) + 0.06):
                self._metric_lift_step = self._metric_step()
            if self._metric_contact_step is None and getattr(self, "_belt_contact_latched", False):
                self._metric_contact_step = self._metric_step()
            if self._metric_drop_step is None and getattr(self, "_block_dropped", False):
                self._metric_drop_step = self._metric_step()
            if self._metric_drop_step is None or getattr(self, "bowl", None) is None:
                return
            # Only after the cube has left the belt — while riding it passes over
            # nothing and the bowl is still sweeping to meet it.
            bowl = np.array(self.bowl.get_pose().p, dtype=np.float64)
            horiz = float(np.linalg.norm(p[:2] - bowl[:2]))
            prev = self._metric_min_bowl_offset
            if prev is None or horiz < prev:
                self._metric_min_bowl_offset = horiz
            if self._metric_bowl_step is None and self._block_in_bowl():
                self._metric_bowl_step = self._metric_step()
        except Exception:
            pass

    def _compute_metrics(self):
        """Human-experiment extras.

        extra1 `place_latency_steps` — steps from lifting the block clear of the table
        until it is seated on the belt. Under Opt 1 the bowl oscillates, so this window
        is where the operator decides *when* to commit the cube to the belt.
        extra2 `bowl_offset_norm` — closest horizontal block-to-bowl-centre distance
        after the cube leaves the belt, divided by the bowl inner radius. LOWER is
        better; <= 1.0 means it fell inside the mouth.
        """
        out = {}
        dt = 0.0
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            pass

        a = getattr(self, "_metric_lift_step", None)
        b = getattr(self, "_metric_contact_step", None)
        lat = None if (a is None or b is None) else max(int(b) - int(a), 0)
        out["place_latency_steps"] = lat
        out["place_latency_s"] = None if lat is None else round(lat * dt, 4)

        c = getattr(self, "_metric_drop_step", None)
        d = getattr(self, "_metric_bowl_step", None)
        ride = None if (b is None or c is None) else max(int(c) - int(b), 0)
        out["ride_latency_steps"] = ride
        out["ride_latency_s"] = None if ride is None else round(ride * dt, 4)
        fall = None if (c is None or d is None) else max(int(d) - int(c), 0)
        out["fall_latency_steps"] = fall

        off = getattr(self, "_metric_min_bowl_offset", None)
        try:
            denom = max(float(self.bowl_inner_radius), 1e-9)
            out["bowl_offset_norm"] = None if off is None else round(float(off) / denom, 4)
        except Exception:
            out["bowl_offset_norm"] = None
        # Reuse the task's own upright-carry score (HIGHER is better, 1.0 = no tilt).
        # Saturates at 0 once max_tilt_deg exceeds theta_max_deg, so also report the
        # raw angle — it stays discriminating past that clip.
        try:
            out["tilt_score"] = round(float(self.tilt_score), 4)
            out["max_tilt_deg"] = round(float(self.max_tilt_deg), 3)
        except Exception:
            out["tilt_score"] = None
            out["max_tilt_deg"] = None
        return out

    # ----------------------------------------------------------------- success + metric
    def check_success(self):
        # success = first contact before red place line, cleared blocker lane, ends in bowl
        self._latch_belt_contact()
        self.placed_on_belt = bool(self.placed_on_belt or self._on_belt())
        self.in_bowl = bool(self._block_in_bowl())
        if self.blocker_enabled and self.blocker is not None:
            contact_y = self._belt_contact_y
            if contact_y is None and getattr(self, "block", None) is not None:
                contact_y = float(self.block.get_pose().p[1])
            self.avoided_blocker = bool(
                (not self.hit_blocker)
                and (contact_y is not None)
                and self._y_in_clear_lane(contact_y)
            )
        else:
            self.avoided_blocker = True
        success = bool(
            self.dropped_at_start_left
            and self.placed_on_belt
            and self.avoided_blocker
            and self.in_bowl
        )
        self.info["placed_on_belt"] = self.placed_on_belt
        self.info["dropped_at_start_left"] = self.dropped_at_start_left
        self.info["placed_before_line"] = bool(self.dropped_at_start_left)
        self.info["place_line_x"] = float(getattr(self, "place_line_x", 0.0))
        self.info["belt_mirrored"] = bool(getattr(self, "belt_mirrored", False))
        self.info["blocker_enabled"] = bool(getattr(self, "blocker_enabled", False))
        self.info["avoided_blocker"] = bool(self.avoided_blocker)
        self.info["hit_blocker"] = bool(self.hit_blocker)
        self.info["in_bowl"] = success
        self.info["tilt_score"] = float(self.tilt_score)
        return success

    # ----------------------------------------------------------------- record state per-frame
    def get_obs(self):
        obs = super().get_obs()
        cur_tilt = self._current_tilt_deg() if hasattr(self, "block") else 0.0
        obs["belt"] = {
            "tilt_deg": float(cur_tilt),
            "max_tilt_deg": float(getattr(self, "max_tilt_deg", 0.0)),
            "theta_max_deg": float(getattr(self, "theta_max_deg", self.THETA_MAX_DEG_DEFAULT)),
            "tilt_score": float(np.clip(1.0 - getattr(self, "max_tilt_deg", 0.0) /
                                        getattr(self, "theta_max_deg", self.THETA_MAX_DEG_DEFAULT), 0.0, 1.0)),
            "belt_speed": float(getattr(self, "belt_speed", self.BELT_SPEED_DEFAULT)),
            "belt_dir": float(getattr(self, "belt_dir", 1.0)),
            "belt_mirrored": bool(getattr(self, "belt_mirrored", False)),
            "blocker_enabled": bool(getattr(self, "blocker_enabled", False)),
            "blocker_x": float(getattr(self, "blocker_x", 0.0)),
            "blocker_y": float(getattr(self, "blocker_y", 0.0)),
            "blocker_half_y": float(getattr(self, "blocker_half_y", 0.0)),
            "clear_y_lo": float(getattr(self, "clear_y_lo", 0.0)),
            "clear_y_hi": float(getattr(self, "clear_y_hi", 0.0)),
            "avoided_blocker": bool(getattr(self, "avoided_blocker", True)),
            "hit_blocker": bool(getattr(self, "hit_blocker", False)),
            "place_line_x": float(getattr(self, "place_line_x", 0.0)),
            "place_line_offset": float(getattr(self, "place_line_offset", self.PLACE_LINE_OFFSET_DEFAULT)),
            "reached_end": bool(getattr(self, "reached_end", False)),
            "placed_on_belt": bool(getattr(self, "placed_on_belt", False)),
            "dropped_at_start_left": bool(getattr(self, "dropped_at_start_left", False)),
            "placed_before_line": bool(getattr(self, "dropped_at_start_left", False)),
            "in_bowl": bool(getattr(self, "in_bowl", False)),
            "belt_contact_x": float(getattr(self, "_belt_contact_x", 0.0) or 0.0),
            "bowl_center": self.bowl.get_pose().p.tolist() if getattr(self, "bowl", None) is not None else [0.0, 0.0, 0.0],
        }
        return obs
