from .catch_valley_ball import catch_valley_ball
from .utils import *
import numpy as np
import os
import sapien
import sapien.physx
import transforms3d as t3d


class stop_valley_ball(catch_valley_ball):
    """Hold a small ping-pong bat in the air to stop a red ball leaving the valley ramp.

    Same curved valley / launch setup as ``catch_valley_ball``, but the expert grasps a
    ping-pong bat (circular head + handle) from a facing-ramp holder and holds it
    mid-flight so the ball hits the red circular head. Success is ball-head contact;
    handle contact does not count, and the ball landing on the table before a head
    hit is a failure.

    Options mirror the parent (independent toggles):
      - Default - red ball travels straight toward the exit edge.
      - Option 1 - ``wall_bounce_enabled``: red ball rebounds from a side rail.
      - Option 2 - ``enable_distractor``: black distractor ball; hitting it with the
        bat fails. Red must be the one that contacts the bat head.
      - Mirror - ``random_mirror`` / ``mirrored``: exit toward +/-x; matching arm holds
        the bat.
    """

    # Wider track (+30% y) and faster ball than the valley loader defaults.
    RAMP_HALF_WIDTH_DEFAULT = 0.1625  # 0.125 * 1.3
    # Mesh catcher is spawned then hidden; do not inherit the procedural box model.
    CATCHER_MODEL_DEFAULT = "062_plasticbox"
    BOWL_ID_DEFAULT = 1
    START_FREEZE_S_DEFAULT = 0.0  # no start hold, the ball drops immediately
    INITIAL_FORWARD_SPEED_DEFAULT = 0.11
    LAUNCH_SPEED_DEFAULT = 0.40
    DROP_TIME_DEFAULT = 0.40
    # Small ping-pong bat head (radius); handle + holder sized relative to this.
    PANEL_RADIUS_DEFAULT = 0.055
    PANEL_HALF_THICK_DEFAULT = 0.006
    PANEL_MASS_DEFAULT = 0.12
    PANEL_COLOR_DEFAULT = [0.88, 0.20, 0.16]  # rubber-red head
    HANDLE_COLOR_DEFAULT = [0.62, 0.42, 0.22]
    HANDLE_HALF_LEN_DEFAULT = 0.050  # half-length along -Y (stick)
    HANDLE_HALF_W_DEFAULT = 0.010  # half-height (Z)
    HANDLE_HALF_T_DEFAULT = 0.007  # half-thickness (X, face axis)
    HOLDER_HEIGHT_DEFAULT = 0.022
    # Fraction of table-landing flight time used for the mid-air intercept.
    INTERCEPT_FLIGHT_FRAC_DEFAULT = 0.42
    INTERCEPT_MIN_CLEARANCE_DEFAULT = 0.06  # m above table for "in air"
    # Head y well below the ramp (ramp y~[-0.08, +0.24]) so the holder never overlaps.
    # Handle still extends further -Y toward the robot for an easy grasp.
    PANEL_SPAWN_Y_MIN_DEFAULT = -0.20
    PANEL_SPAWN_Y_MAX_DEFAULT = -0.14
    BAT_NAME = "pingpong_bat"

    def setup_demo(self, **kwags):
        # Uses stop_valley_ball task_args (not the catch parent block).
        self._cfg = kwags.get("task_args", {}).get("stop_valley_ball", {})
        self._loaded = False
        self._ball_phase = None
        self._distractor_phase = None
        self._ball_freeze_armed = False
        self._start_freeze_pending = 0
        self._freeze_i = 0
        self._expert_demo = False
        self._bowl_ready = False
        self._bowl_welded = False
        self._arm_ball_contact = False
        self._panel_hit = False
        self._distractor_panel_hit = False
        self._ball_table_before_hit = False
        self._metric_release_step = None
        self._metric_hit_step = None
        self._metric_hit_radial = None
        self.distractor = None
        self._distractor_rigid = None
        self.enable_distractor = False
        self.wall_bounce_enabled = False
        self.mirrored = False
        self.side = 1.0
        self.panel = None
        self.bat_holder_parts = []
        self.intercept = None
        # Bypass catch_valley_ball.setup_demo: this task reads its own cfg block
        # and does not use the push box.
        from ._base_task import Base_Task

        Base_Task._init_task_env_(self, **kwags)
        self._start_ball_motion(expert_demo=False)

    def _bat_head_center_z_on_holder(self):
        """World z of the bat head center when seated on the holder."""
        # Bottom of the circular head rests on the holder top.
        return float(self.table_top + self.holder_height + self.panel_radius)

    def _create_pingpong_bat(self, pose: sapien.Pose):
        """Graspable bat: circular head (axis X, face toward ramp) + handle (-Y).

        Origin is the head center so intercept / hit checks stay head-centric.
        Handle sticks toward -Y (robot side) so the bat silhouette is obvious
        from top-down and the grip is an easy top-down grasp.
        """
        scene, pose = preprocess(self, pose)
        head_r = float(self.panel_radius)
        head_ht = float(self.panel_half_thick)
        h_len = float(self.handle_half_len)
        h_w = float(self.handle_half_w)
        h_t = float(self.handle_half_t)
        # Handle attached to the -Y edge of the head, slightly below center so it
        # seats in the holder cradle.
        handle_local = sapien.Pose([0.0, -(head_r + h_len), -0.012])

        entity = sapien.Entity()
        entity.set_name(self.BAT_NAME)
        entity.set_pose(pose)

        mat = scene.default_physical_material
        rigid = sapien.physx.PhysxRigidDynamicComponent()
        head_col = sapien.physx.PhysxCollisionShapeCylinder(
            radius=head_r, half_length=head_ht, material=mat
        )
        head_col.set_local_pose(sapien.Pose([0.0, 0.0, 0.0]))
        rigid.attach(head_col)
        # Box half-sizes: X=thickness, Y=length along stick, Z=height.
        handle_col = sapien.physx.PhysxCollisionShapeBox(
            half_size=[h_t, h_len, h_w], material=mat
        )
        handle_col.set_local_pose(handle_local)
        rigid.attach(handle_col)

        head_mat = sapien.render.RenderMaterial(base_color=[*self.panel_color[:3], 1.0])
        handle_mat = sapien.render.RenderMaterial(base_color=[*self.handle_color[:3], 1.0])
        render = sapien.render.RenderBodyComponent()
        head_vis = sapien.render.RenderShapeCylinder(
            radius=head_r, half_length=head_ht, material=head_mat
        )
        head_vis.set_local_pose(sapien.Pose([0.0, 0.0, 0.0]))
        render.attach(head_vis)
        handle_vis = sapien.render.RenderShapeBox([h_t, h_len, h_w], handle_mat)
        handle_vis.set_local_pose(handle_local)
        render.attach(handle_vis)

        entity.add_component(rigid)
        entity.add_component(render)
        scene.add_entity(entity)

        # Top-down grasps on the handle mid-stick. Actor.get_point scales t by scale.
        # Put translation in Y (handle axis): t_y = -1 * scale[1].
        grasp_scale = [
            float(h_t),
            float(head_r + 0.55 * h_len),  # mid-handle Y
            float(h_w),
        ]
        contact_points_pose = [
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [2.0, 2.0, 2.0],
            "scale": grasp_scale,
            "target_pose": [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            "contact_points_pose": contact_points_pose,
            "transform_matrix": np.eye(4, dtype=np.float64).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": ["Center of the ping-pong bat head."],
        }
        actor = Actor(entity, data)
        actor.set_mass(self.panel_mass)
        if rigid is not None:
            phys_mat = sapien.physx.PhysxMaterial(
                static_friction=0.85,
                dynamic_friction=0.80,
                restitution=0.05,
            )
            for shape in rigid.get_collision_shapes():
                shape.set_physical_material(phys_mat)
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(pose)
        return actor

    def _create_bat_holder(self, x, y):
        """Static cradle under the bat: head face toward the ramp, handle toward -Y."""
        self.bat_holder_parts = []
        color = [0.40, 0.38, 0.36]
        h = float(self.holder_height)
        head_r = float(self.panel_radius)
        h_len = float(self.handle_half_len)
        # Long base covering head + handle footprint.
        base_y = y - 0.5 * (head_r + 2.0 * h_len) * 0.35
        base = create_box(
            self,
            pose=sapien.Pose(
                [x, base_y, self.table_top + 0.005],
                [1.0, 0.0, 0.0, 0.0],
            ),
            half_size=[0.034, 0.5 * (head_r + 2.0 * h_len) + 0.02, 0.005],
            color=color,
            is_static=True,
            name="bat_holder_base",
        )
        self.bat_holder_parts.append(base)
        # Two side rails (gap along X) cradle the thin bat face/handle.
        rail_half_x = 0.006
        rail_half_y = 0.5 * (head_r + 2.0 * h_len) * 0.55 + 0.01
        rail_half_z = 0.5 * h
        rail_z = self.table_top + rail_half_z
        gap = float(self.handle_half_t) + 0.004
        rail_y = y - 0.35 * head_r
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            rail = create_box(
                self,
                pose=sapien.Pose(
                    [x + sign * (gap + rail_half_x), rail_y, rail_z],
                    [1.0, 0.0, 0.0, 0.0],
                ),
                half_size=[rail_half_x, rail_half_y, rail_half_z],
                color=color,
                is_static=True,
                name=f"bat_holder_rail_{tag}",
            )
            self.bat_holder_parts.append(rail)
        # End stop at the handle tip (-Y) so the bat can't slide out.
        stop_y = y - (head_r + 2.0 * h_len) - 0.008
        stop = create_box(
            self,
            pose=sapien.Pose([x, stop_y, rail_z], [1.0, 0.0, 0.0, 0.0]),
            half_size=[gap + rail_half_x + 0.004, 0.006, rail_half_z],
            color=color,
            is_static=True,
            name="bat_holder_endstop",
        )
        self.bat_holder_parts.append(stop)
        for part in self.bat_holder_parts:
            self.add_prohibit_area(part, padding=0.015)

    def _hide_bowl(self):
        """Park the unused parent bowl far away so it cannot interfere."""
        if getattr(self, "bowl", None) is None:
            return
        park = sapien.Pose([80.0, 80.0, 80.0])
        try:
            self.bowl.set_pose(park)
        except Exception:
            try:
                self.bowl.actor.set_pose(park)
            except Exception:
                pass
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            try:
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic_target(park)
            except Exception:
                pass

    def _compute_intercept(self):
        """Ballistic mid-air point where the panel face should meet the red ball."""
        self._update_release_velocity()
        self._compute_landing()
        z0 = float(self.ball_exit[2])
        vz = float(self.release_velocity[2])
        t_apex = max(0.0, vz / self.GRAVITY)
        t = float(self.INTERCEPT_FLIGHT_FRAC_DEFAULT) * float(self.flight_time)
        # Prefer near the apex when it is still clearly airborne.
        if t_apex > 0.05:
            t = float(np.clip(0.85 * t_apex, 0.12, max(0.12, 0.65 * self.flight_time)))
        t = float(np.clip(t, 0.10, max(0.10, 0.70 * self.flight_time)))

        pos = (
            np.asarray(self.ball_exit, dtype=np.float64)
            + np.asarray(self.release_velocity, dtype=np.float64) * t
            + np.array([0.0, 0.0, -0.5 * self.GRAVITY * t * t])
        )
        # Seat the panel so the ball strikes the face (center behind contact by
        # thickness + radius along the travel direction).
        face_offset = self.panel_half_thick + self.ball_radius + 0.002
        pos[0] = float(pos[0] + self.side * face_offset)
        # Keep a comfortable mid-air band and within arm reach.
        min_z = self.table_top + self.INTERCEPT_MIN_CLEARANCE_DEFAULT + self.panel_radius * 0.35
        max_z = self.table_top + 0.28
        pos[2] = float(np.clip(pos[2], min_z, max_z))
        reach = 0.36
        if float(self.side) > 0.0:
            pos[0] = float(np.clip(pos[0], float(self.red_line_x) + 0.02, reach))
        else:
            pos[0] = float(np.clip(pos[0], -reach, float(self.red_line_x) - 0.02))
        self.intercept = pos
        self.intercept_time = t

    def load_actors(self):
        c = self._cfg
        self.panel_radius = float(c.get("panel_radius", self.PANEL_RADIUS_DEFAULT))
        self.panel_half_thick = float(c.get("panel_half_thick", self.PANEL_HALF_THICK_DEFAULT))
        self.panel_mass = float(c.get("panel_mass", self.PANEL_MASS_DEFAULT))
        self.panel_color = list(c.get("panel_color", self.PANEL_COLOR_DEFAULT))
        self.handle_color = list(c.get("handle_color", self.HANDLE_COLOR_DEFAULT))
        self.handle_half_len = float(c.get("handle_half_len", self.HANDLE_HALF_LEN_DEFAULT))
        self.handle_half_w = float(c.get("handle_half_w", self.HANDLE_HALF_W_DEFAULT))
        self.handle_half_t = float(c.get("handle_half_t", self.HANDLE_HALF_T_DEFAULT))
        self.holder_height = float(c.get("holder_height", self.HOLDER_HEIGHT_DEFAULT))
        self.intercept_flight_frac = float(
            c.get("intercept_flight_frac", self.INTERCEPT_FLIGHT_FRAC_DEFAULT)
        )
        self.panel_radius = float(np.clip(self.panel_radius, 0.035, 0.10))
        self.panel_half_thick = float(np.clip(self.panel_half_thick, 0.004, 0.015))
        self.handle_half_len = float(np.clip(self.handle_half_len, 0.025, 0.07))
        self.handle_half_w = float(np.clip(self.handle_half_w, 0.005, 0.015))
        self.handle_half_t = float(np.clip(self.handle_half_t, 0.004, 0.012))
        self.holder_height = float(np.clip(self.holder_height, 0.015, 0.05))
        self.panel_mass = max(float(self.panel_mass), 0.05)

        # Valley / ball / temp mesh catcher only - not the parent's box swap or
        # on-lane ball start (this task keeps the aerial drop).
        self._load_valley_actors()
        self._hide_bowl()

        y_lo = float(c.get("panel_spawn_y_min", self.PANEL_SPAWN_Y_MIN_DEFAULT))
        y_hi = float(c.get("panel_spawn_y_max", self.PANEL_SPAWN_Y_MAX_DEFAULT))
        bat_x = float(self.side * np.random.uniform(0.28, 0.34))
        if self.mirrored:
            bat_x += float(self.MIRROR_X_SHIFT)
        bat_y = float(np.random.uniform(min(y_lo, y_hi), max(y_lo, y_hi)))

        # Holder first, then seat the bat upright with its face toward the ramp.
        self._create_bat_holder(bat_x, bat_y)
        bat_z = self._bat_head_center_z_on_holder()
        bat_pose = sapien.Pose([bat_x, bat_y, bat_z], [1.0, 0.0, 0.0, 0.0])
        self.panel = self._create_pingpong_bat(bat_pose)
        # Alias so inherited weld / EE helpers keep working.
        self.bowl = self.panel
        self.bowl_outer_radius = self.panel_radius
        self.bowl_inner_radius = self.panel_radius * 0.85
        self.bowl_height = 2.0 * self.panel_radius + 2.0 * self.handle_half_len

        self._compute_intercept()
        self._panel_hit = False
        self._distractor_panel_hit = False
        self._ball_table_before_hit = False
        self._metric_release_step = None
        self._metric_hit_step = None
        self._metric_hit_radial = None
        self.add_prohibit_area(self.panel, padding=0.04)

    def _ball_actor_name(self, ball_actor):
        if ball_actor is None:
            return None
        if hasattr(ball_actor, "get_name"):
            try:
                return ball_actor.get_name()
            except Exception:
                pass
        try:
            return ball_actor.actor.get_name()
        except Exception:
            return None

    def _ball_near_bat_head(self, ball_actor, radial_slack=None, axial_slack=None):
        """True if ball center is at the circular (red) head, not the handle."""
        if ball_actor is None or getattr(self, "panel", None) is None:
            return False
        if radial_slack is None:
            radial_slack = self.ball_radius * 0.25
        if axial_slack is None:
            axial_slack = 0.006
        ball_p = np.asarray(ball_actor.get_pose().p, dtype=np.float64)
        head_p = np.asarray(self.panel.get_pose().p, dtype=np.float64)
        # Bat local: cylinder axis = X, circular face in YZ; handle along -Y.
        d = ball_p - head_p
        axial = abs(float(d[0]))
        radial = float(np.linalg.norm(d[1:3]))
        return bool(
            axial <= self.panel_half_thick + self.ball_radius + float(axial_slack)
            and radial <= self.panel_radius + float(radial_slack)
        )

    def _ball_hit_bat_head(self, ball_actor):
        """Success contact: ball touches the red circular head (handle does not count)."""
        ball_name = self._ball_actor_name(ball_actor)
        if ball_name is None or getattr(self, "panel", None) is None:
            return False
        # Require proximity to the head so a handle-only PhysX contact is rejected.
        if not self._ball_near_bat_head(ball_actor):
            return False
        bat_name = self.BAT_NAME
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if {name0, name1} == {ball_name, bat_name}:
                return True
        # Thin-disk tunneling fallback: ball center clearly inside the head disc.
        ball_p = np.asarray(ball_actor.get_pose().p, dtype=np.float64)
        head_p = np.asarray(self.panel.get_pose().p, dtype=np.float64)
        d = ball_p - head_p
        axial = abs(float(d[0]))
        radial = float(np.linalg.norm(d[1:3]))
        return bool(
            axial <= self.panel_half_thick + 0.003
            and radial <= self.panel_radius - 0.002
        )

    def _ball_on_table_past_ramp(self, ball_actor):
        """True once the ball has left the ramp and is resting on / touching the table."""
        if ball_actor is None:
            return False
        p = np.asarray(ball_actor.get_pose().p, dtype=np.float64)
        left_ramp = bool(
            float(self.side) * float(p[0])
            > float(self.side) * float(self.ramp_exit_x) + self.ball_radius
        )
        if not left_ramp:
            return False
        # Height check (primary): ball center at/near table.
        if float(p[2]) <= self.table_top + self.ball_radius + 0.008:
            return True
        # Contact check with the table entity when present.
        ball_name = self._ball_actor_name(ball_actor)
        if ball_name is None:
            return False
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            other = name1 if name0 == ball_name else name0 if name1 == ball_name else None
            if other is None:
                continue
            if other == "table" or other.endswith("_table") or "table" in other.lower():
                return True
        return False

    def _update_panel_hits(self):
        """Track head hits and table-before-hit failures after PhysX release."""
        if self._ball_phase != "released":
            return
        # Latch the release step once so hit latency is measured from the launch.
        if self._metric_release_step is None:
            self._metric_release_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        # Head hit first wins; table contact before any head hit is a permanent fail.
        if not self._panel_hit and self._ball_hit_bat_head(self.ball):
            self._panel_hit = True
            self._latch_hit_metrics()
        if (
            (not self._panel_hit)
            and (not self._ball_table_before_hit)
            and self._ball_on_table_past_ramp(self.ball)
        ):
            self._ball_table_before_hit = True
        if (
            self.enable_distractor
            and not self._distractor_panel_hit
            and self._ball_hit_bat_head(self.distractor)
        ):
            self._distractor_panel_hit = True

    def _reset_metric_state(self):
        """Also clear the bat-hit latches (parent only knows the catch ones)."""
        super()._reset_metric_state()
        self._metric_hit_step = None
        self._metric_hit_radial = None

    def _latch_hit_metrics(self):
        """Record when the ball met the bat head and how far off centre it was."""
        self._metric_hit_step = int(getattr(self, "_exp_sim_steps", 0) or 0)
        try:
            ball_p = np.asarray(self.ball.get_pose().p, dtype=np.float64)
            head_p = np.asarray(self.panel.get_pose().p, dtype=np.float64)
            # Bat local frame: cylinder axis = X, circular face spans YZ.
            radial = float(np.linalg.norm((ball_p - head_p)[1:3]))
            self._metric_hit_radial = radial / max(float(self.panel_radius), 1e-9)
        except Exception:
            self._metric_hit_radial = None

    def _compute_metrics(self):
        """extra1 = release->hit latency, extra2 = radial miss in bat-head radii."""
        metrics = {
            "hit_latency_steps": None,
            "hit_latency_s": None,
            "radial_offset_norm": self._metric_hit_radial,
        }
        start = self._metric_release_step
        hit = self._metric_hit_step
        if start is not None and hit is not None and hit >= start:
            steps = int(hit - start)
            metrics["hit_latency_steps"] = steps
            try:
                metrics["hit_latency_s"] = round(steps * float(self.scene.get_timestep()), 6)
            except Exception:
                pass
        return metrics

    # ------------------------------------------------------- expert-free gates
    # Deliberately the same numbers ``interactive/base/interactive_stop_valley_ball.py``
    # uses for teleop: the eval latch is that same latch, so it must not be easier or
    # harder than the one a human gets.
    EVAL_LATCH_PROXIMITY_M = 0.11     # teleop _LATCH_PROXIMITY_M
    EVAL_LATCH_SETTLE_STEPS = 12      # the teleop _dwell(12) before judging the close
    # The expert's grasp_actor closes to ``gripper_pos=0.0`` from a fully open hand, so
    # unlike pack_fruits there is no pre-shape level to place a threshold between;
    # teleop's own 0.5 separates the two commanded values cleanly.
    EVAL_CLOSE_WIDTH = 0.5
    EVAL_OPEN_WIDTH = 0.6             # hysteresis band avoids latch/unlatch chatter

    def _bat_handle_segment(self):
        """World endpoints of the graspable handle (head rim -> tip along -Y)."""
        head = np.asarray(self.panel.get_pose().p, dtype=float)
        head_r = float(getattr(self, "panel_radius", self.PANEL_RADIUS_DEFAULT))
        h_len = float(getattr(self, "handle_half_len", self.HANDLE_HALF_LEN_DEFAULT))
        start = head + np.array([0.0, -head_r, -0.012], dtype=float)
        end = head + np.array([0.0, -(head_r + 2.0 * h_len), -0.012], dtype=float)
        return start, end

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        point = np.asarray(point, dtype=float)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        span = end - start
        denom = float(np.dot(span, span))
        if denom < 1e-12:
            return float(np.linalg.norm(point - start))
        t = float(np.clip(np.dot(point - start, span) / denom, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + t * span)))

    def _eval_gripper_width(self, side: str) -> float:
        """Normalized gripper opening in ``[0, 1]`` (1 = open, 0 = closed)."""
        try:
            return float(self.robot.get_left_gripper_val() if side == "left"
                         else self.robot.get_right_gripper_val())
        except Exception:
            return 1.0

    def _eval_gripper_probe_points(self, side: str):
        """World points used for the nearness test (TCP preferred, then EE)."""
        points = []
        for getter in (f"get_{side}_tcp_pose", f"get_{side}_ee_pose"):
            fn = getattr(self.robot, getter, None)
            if callable(fn):
                try:
                    points.append(np.asarray(fn()[:3], dtype=float))
                except Exception:
                    pass
        if not points:
            try:
                points.append(np.asarray(self._end_effector_position(side), dtype=float))
            except Exception:
                pass
        return points

    def _eval_bat_contact(self, side: str) -> bool:
        """Fingers of *this* arm touching the bat.

        ``get_gripper_actor_contact_position`` is arm-agnostic (it matches any link in
        ``robot.gripper_name``), so the link set is rebuilt per side here; otherwise the
        left hand could latch a bat the right hand is holding.
        """
        art = getattr(self.robot, f"{side}_entity", None)
        if art is None:
            return False
        arm_links = {
            link.get_name() for link in art.get_links()
            if "finger" in link.get_name() or "gripper" in link.get_name()
        }
        if not arm_links:
            return False
        try:
            for contact in self.scene.get_contacts():
                name0 = contact.bodies[0].entity.name
                name1 = contact.bodies[1].entity.name
                if self.BAT_NAME not in (name0, name1):
                    continue
                if (name1 if name0 == self.BAT_NAME else name0) in arm_links:
                    return True
        except Exception:
            pass
        return False

    def _eval_can_latch_bat(self, side: str) -> bool:
        """Teleop's own latch test: fingers on the bat, or the hand on the handle."""
        if self._eval_bat_contact(side):
            return True
        start, end = self._bat_handle_segment()
        return any(
            self._point_segment_distance(p, start, end) <= self.EVAL_LATCH_PROXIMITY_M
            for p in self._eval_gripper_probe_points(side)
        )

    def _arm_contact_exempt(self) -> bool:
        """Whether brushing the ball right now is holding the bat rather than an arm catch."""
        if getattr(self, "_bowl_ready", False):
            return True
        # ``check_success`` hard-fails on ``_arm_ball_contact``, and holding the bat at the
        # intercept puts the wrist exactly where the ball flies, so without this exemption
        # the task is unsolvable.  ``play_once`` sets ``_bowl_ready`` for the expert; off the
        # expert path the exemption applies on the same terms, i.e. while the bat is held.
        return bool(self._runs_without_expert() and getattr(self, "_bowl_welded", False))

    def _sync_eval_gates(self):
        """Latch the bat to a gripper that closes on it, for the whole expert-free episode.

        ``_create_pingpong_bat`` builds the bat *kinematic* with gravity off, and the only
        thing that ever moves it is ``_update_welded_bowl`` following the EE, which needs
        ``_weld_bowl_to_end_effector``.  That is called only from ``play_once`` (right after
        the expert's grasp) and from the teleop latch in
        ``interactive/base/interactive_stop_valley_ball.py``.  Without this gate a policy could
        close its fingers on the bat indefinitely and the bat would not leave the holder,
        making the task unsolvable rather than merely hard.

        Reached from ``Base_Task._update_kinematic_tasks`` via ``_update_valley_kinematics``
        behind ``_runs_without_expert()``, so it is live for eval / interactive rollouts and
        never during ``play_once``.
        """
        # Diagnostics only: STOP_VALLEY_NO_EVAL_LATCH switches the latch off so the same
        # motion can be compared with and without it.
        if os.environ.get("STOP_VALLEY_NO_EVAL_LATCH"):
            return
        if not getattr(self, "_loaded", False) or getattr(self, "panel", None) is None:
            return
        g = self._eval_gate_state
        st = g.get("latch")
        if st is None:
            # Lives in _eval_gate_state so it is reset per episode along with the gates.
            st = g["latch"] = {"arm": None, "pending": None, "step": 0}
        st["step"] += 1

        # ---- already holding: release when the jaws open -------------------------
        if st["arm"] is not None:
            if self._eval_gripper_width(st["arm"]) > self.EVAL_OPEN_WIDTH:
                self._unweld_bowl()
                # Bat no longer held: brushing the ball with a free arm is a real arm
                # catch again.  (The expert never releases, so this cannot affect it.)
                self._bowl_ready = False
                st["arm"] = None
            return

        # ---- pending close: judge it after the same settle teleop uses -----------
        pending = st["pending"]
        if pending is not None:
            side, at = pending
            if st["step"] < at:
                return
            st["pending"] = None
            if (self._eval_gripper_width(side) <= self.EVAL_CLOSE_WIDTH
                    and self._eval_can_latch_bat(side)):
                self.bowl = self.panel
                self._weld_bowl_to_end_effector(side)
                st["arm"] = side
                self._bowl_ready = True
            return

        # ---- arm the latch ------------------------------------------------------
        # Level-triggered, not edge-triggered: a policy can arrive at the handle with the
        # jaws already shut, and an edge that fired earlier over empty table would never
        # come back.
        for side in ("left", "right"):
            if (self._eval_gripper_width(side) <= self.EVAL_CLOSE_WIDTH
                    and self._eval_can_latch_bat(side)):
                st["pending"] = (side, st["step"] + int(self.EVAL_LATCH_SETTLE_STEPS))
                return

    def _update_kinematic_tasks(self):
        # Bat is welded to the EE; do not use the parent's box (no-weld) update.
        self._update_valley_kinematics(weld_bowl=True)
        if not getattr(self, "_loaded", False):
            return
        self._update_panel_hits()

    def _bat_face_quat(self):
        """Identity quat: cylinder axis along +X, circular face toward the ramp."""
        return [1.0, 0.0, 0.0, 0.0]

    def _bat_min_center_z(self) -> float:
        """Lowest head-center Z that keeps bat geometry above the tabletop.

        The head is a cylinder whose axis is +X, so the disc extends
        ``panel_radius`` below the actor origin. The handle sits slightly lower.
        Kinematic welding teleports the bat, so we clamp instead of relying on
        PhysX table contacts.
        """
        head_below = float(getattr(self, "panel_radius", self.PANEL_RADIUS_DEFAULT))
        handle_below = 0.012 + float(
            getattr(self, "handle_half_w", self.HANDLE_HALF_W_DEFAULT)
        )
        return float(self.table_top) + max(head_below, handle_below) + 0.002

    def interactive_ee_z_floor(self, side, pose=None):
        """Keep the welded bat from tunneling through the table during teleop."""
        if not bool(getattr(self, "_bowl_welded", False)):
            return None
        arm = str(getattr(self, "_bowl_arm", "") or "")
        if arm and str(side) != arm:
            return None
        offset = getattr(self, "_bowl_ee_offset", None)
        off_z = float(offset[2]) if offset is not None else 0.0
        return float(self._bat_min_center_z() - off_z)

    def _weld_bowl_to_end_effector(self, arm_tag):
        """Weld like the parent, but keep the bat face locked toward the ramp."""
        # Snap orientation before measuring the EE offset so the weld carries the
        # facing-ramp pose (grasp_actor can tip the bat flat).
        p = np.asarray(self.panel.get_pose().p, dtype=np.float64)
        faced = sapien.Pose(p.tolist(), self._bat_face_quat())
        try:
            self.panel.set_pose(faced)
        except Exception:
            self.panel.actor.set_pose(faced)
        super()._weld_bowl_to_end_effector(arm_tag)

    def _update_welded_bowl(self):
        """Follow the EE in translation; force face-toward-ramp orientation."""
        if not self._bowl_welded:
            return
        rigid = self._get_rigid(self.bowl)
        target_position = (
            self._end_effector_position(self._bowl_arm) + self._bowl_ee_offset
        )
        # Clamp head center so the disc/handle cannot sink through the table.
        target_position[2] = max(float(target_position[2]), self._bat_min_center_z())
        target = sapien.Pose(target_position.tolist(), self._bat_face_quat())
        self.bowl.actor.set_pose(target)
        if rigid is not None:
            rigid.set_kinematic_target(target)

    def play_once(self):
        arm_tag = ArmTag("left" if self.mirrored else "right")

        self._start_ball_motion(expert_demo=True)
        self._panel_hit = False
        self._distractor_panel_hit = False
        self._ball_table_before_hit = False
        self._metric_release_step = None
        self._metric_hit_step = None
        self._metric_hit_radial = None

        self.move(self.grasp_actor(self.panel, arm_tag=arm_tag, pre_grasp_dis=0.10))
        self._weld_bowl_to_end_effector(arm_tag)

        # Lift clear of the holder, then translate to the mid-air intercept.
        panel_now = np.asarray(self.panel.get_pose().p, dtype=np.float64)
        lift_z = float(max(0.12, self.intercept[2] - panel_now[2]))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_z, move_axis="world"))

        for _ in range(3):
            panel_now = np.asarray(self.panel.get_pose().p, dtype=np.float64)
            displacement = np.asarray(self.intercept, dtype=np.float64) - panel_now
            if float(np.linalg.norm(displacement)) < 0.012:
                break
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(displacement[0]),
                y=float(displacement[1]),
                z=float(displacement[2]),
                move_axis="world",
            ))

        # Bat is ready in the air - keep welded / gripper closed and wait for impact.
        self._bowl_ready = True
        remaining_drop = max(0, self.drop_steps - self._drop_i)
        remaining_roll = max(0, self.roll_steps - self._roll_i)
        self._dwell(
            self._remaining_start_freeze()
            + remaining_drop
            + remaining_roll
            + self.flight_steps
            + self.settle_steps
        )
        self._check_arm_ball_contact()
        self._update_panel_hits()

        self.info["info"] = {
            "{A}": "valley ball",
            "{B}": "ping-pong bat",
            "{a}": str(arm_tag),
            "{opt}": self._option_label(),
            "{flip}": "mirrored" if self.mirrored else "default",
        }
        return self.info

    def check_success(self):
        if not getattr(self, "_loaded", False) or self._ball_phase != "released":
            self._last_fail_reason = "ball_not_released"
            return False
        self._check_arm_ball_contact()
        self._update_panel_hits()
        if self._arm_ball_contact:
            self._last_fail_reason = "arm_ball_contact"
            return False
        if self._distractor_panel_hit:
            self._last_fail_reason = "distractor_hit_panel"
            return False
        # Table contact before a valid head hit is always a miss.
        if self._ball_table_before_hit:
            self._last_fail_reason = "ball_table_before_hit"
            return False
        if not self._panel_hit:
            self._last_fail_reason = "ball_missed_panel"
            return False
        # Bat must still be held off the table (mid-air stop, not a table block).
        panel_z = float(self.panel.get_pose().p[2])
        min_air_z = self.table_top + self.INTERCEPT_MIN_CLEARANCE_DEFAULT
        if panel_z < min_air_z:
            self._last_fail_reason = "panel_not_in_air"
            return False
        self._last_fail_reason = ""
        return True

    def get_obs(self):
        obs = super().get_obs()
        try:
            panel_p = list(map(float, self.panel.get_pose().p)) if self.panel is not None else [0, 0, 0]
            intercept = (
                list(map(float, self.intercept))
                if self.intercept is not None
                else [0.0, 0.0, 0.0]
            )
            obs["valley_stop"] = {
                "panel_position": panel_p,
                "intercept": intercept,
                "intercept_time": float(getattr(self, "intercept_time", 0.0)),
                "panel_hit": float(self._panel_hit),
                "distractor_panel_hit": float(self._distractor_panel_hit),
                "ball_table_before_hit": float(self._ball_table_before_hit),
                "panel_radius": float(self.panel_radius),
                "panel_half_thick": float(self.panel_half_thick),
                "option_label": self._option_label(),
            }
        except Exception:
            pass
        return obs
