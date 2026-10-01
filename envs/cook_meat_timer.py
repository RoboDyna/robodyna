"""Cook-meat task with an on-table pie timer that tracks cooking progress.

Same manipulation / options as ``cook_meat``, plus a white circular disc that
fills clockwise section-by-section at the cooking rate:

  * green  — doneness 0 → lower success threshold (full pie = lower bound)
  * yellow — resets and fills until the upper success threshold
  * red    — stays full once overcooked; freezes when cooking stops

The timer advances with the same cook-key gate as cooking: latched ON
(default / Opt 2) or held down (Opt 1 / Opt 1+2). Keys are green when up and
red when depressed.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from .cook_meat import cook_meat
from .utils.create_actor import create_visual_box


class cook_meat_timer(cook_meat):
    """``cook_meat`` with a per-station pie timer visual."""

    # Keep the tighter pre-widening window: cook_meat relaxed its own success band
    # to 0.40–0.60, but the timer variant is meant to stay the harder read.
    # (Both yaml sections pin this explicitly; this is the no-config fallback.)
    TARGET_DONENESS_RANGE_DEFAULT: ClassVar[tuple[float, float]] = (0.45, 0.55)

    TIMER_RADIUS_DEFAULT: ClassVar[float] = 0.040
    TIMER_HALF_H: ClassVar[float] = 0.0025
    TIMER_N_SECTIONS: ClassVar[int] = 24
    TIMER_RIM_WIDTH: ClassVar[float] = 0.007
    # North (+y) of the cutting board — top of the table from the robot's view.
    TIMER_OFF_X: ClassVar[float] = 0.0
    TIMER_OFF_Y: ClassVar[float] = 0.14  # board center → timer center along +y
    TIMER_Z_LIFT: ClassVar[float] = 0.006  # above table top (table-frame)

    COLOR_DISC: ClassVar[list[float]] = [0.96, 0.96, 0.96]
    COLOR_GREEN: ClassVar[list[float]] = [0.20, 0.78, 0.28]
    COLOR_GREEN_RIM: ClassVar[list[float]] = [0.08, 0.48, 0.14]
    COLOR_YELLOW: ClassVar[list[float]] = [0.95, 0.82, 0.12]
    COLOR_YELLOW_RIM: ClassVar[list[float]] = [0.72, 0.52, 0.04]
    COLOR_RED: ClassVar[list[float]] = [0.90, 0.14, 0.12]
    COLOR_RED_RIM: ClassVar[list[float]] = [0.55, 0.05, 0.05]

    def setup_demo(self, **kwargs: Any) -> None:
        """Merge ``task_args.cook_meat_timer`` into the parent cook_meat config."""
        ta = dict(kwargs.get("task_args", {}) or {})
        merged = dict(ta.get("cook_meat", {}) or {})
        merged.update(ta.get("cook_meat_timer", {}) or {})
        ta = dict(ta)
        ta["cook_meat"] = merged
        ta["cook_meat_timer"] = merged
        kwargs = dict(kwargs)
        kwargs["task_args"] = ta
        super().setup_demo(**kwargs)

    def load_actors(self) -> None:
        """Spawn cook stations, then attach one pie timer per station."""
        super().load_actors()
        self.timer_radius = float(
            self._cook_cfg.get("timer_radius", self.TIMER_RADIUS_DEFAULT)
        )
        self.timer_n_sections = int(
            self._cook_cfg.get("timer_n_sections", self.TIMER_N_SECTIONS)
        )
        self.timer_n_sections = max(8, min(48, self.timer_n_sections))
        for st in self.stations:
            self._spawn_station_pie_timer(st)
            self._update_station_pie_timer(st)

    # ---------------------------------------------------------------- pie timer
    def _timer_phase_and_fill(self, doneness: float) -> tuple[str, float]:
        """Map doneness → (phase, fill fraction in [0, 1])."""
        low, high = map(float, self.target_doneness_range)
        d = float(np.clip(doneness, 0.0, 1.0))
        if d <= 0.0:
            return "empty", 0.0
        if d <= low:
            return "green", float(d / max(low, 1e-6))
        if d <= high:
            return "yellow", float((d - low) / max(high - low, 1e-6))
        return "red", 1.0

    def _phase_colors(self, phase: str) -> tuple[list[float], list[float]]:
        if phase == "green":
            return list(self.COLOR_GREEN), list(self.COLOR_GREEN_RIM)
        if phase == "yellow":
            return list(self.COLOR_YELLOW), list(self.COLOR_YELLOW_RIM)
        if phase == "red":
            return list(self.COLOR_RED), list(self.COLOR_RED_RIM)
        return list(self.COLOR_DISC), list(self.COLOR_DISC)

    def _hidden_pose(self) -> sapien.Pose:
        """Park unused wedge parts below the table (fire-ring pattern)."""
        return sapien.Pose([0.0, 0.0, -1.0], [1, 0, 0, 0])

    def _spawn_station_pie_timer(self, station: dict[str, Any]) -> None:
        """Build a white disc + N fill/rim wedges via visual-only boxes."""
        # create_visual_box(self, ...) adds table_z_bias — pass table-frame Z.
        table_z = 0.74
        board_xy = station.get("board_xy")
        if board_xy is not None:
            # Prefer the cutting board: same X, further +y (north / top of table).
            cx = float(board_xy[0]) + float(self.TIMER_OFF_X)
            cy = float(board_xy[1]) + float(self.TIMER_OFF_Y)
        else:
            # Fallback if a station has no board yet.
            side = float(station.get("side", 1.0))
            lat = 1.0 if side > 0 else -1.0
            pan_xy = np.asarray(station["skillet"].get_pose().p[:2], dtype=float)
            cx = float(pan_xy[0] + lat * 0.18)
            cy = float(pan_xy[1] + 0.28)
        cx = float(np.clip(cx, -0.40, 0.40))
        cy = float(np.clip(cy, -0.22, 0.32))

        radius = float(self.timer_radius)
        n = int(self.timer_n_sections)
        half_h = float(self.TIMER_HALF_H)
        rim_w = float(self.TIMER_RIM_WIDTH)
        disc_z = float(table_z + self.TIMER_Z_LIFT + half_h)
        fill_z = disc_z + half_h + 0.001
        rim_z = fill_z + 0.0005

        d_theta = 2.0 * np.pi / n
        # White base disc ≈ ring of radial boxes (avoids cylinder wash-out).
        base_shapes: list[Any] = []
        base_r_mid = 0.5 * radius
        base_tang = max(0.0025, base_r_mid * d_theta * 0.62)
        base_half = [base_r_mid, base_tang, half_h]
        for i in range(n):
            theta = 0.5 * np.pi - (i + 0.5) * d_theta
            c, s = float(np.cos(theta)), float(np.sin(theta))
            q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], theta).tolist()
            part = create_visual_box(
                self,
                sapien.Pose([cx + base_r_mid * c, cy + base_r_mid * s, disc_z], q),
                half_size=base_half,
                color=tuple(self.COLOR_DISC),
                name=f"pie_timer_base_{station['tag']}_{i}",
            )
            for comp in part.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    base_shapes.extend(list(comp.render_shapes))

        outer_fill = max(0.008, radius - rim_w)
        r_mid = 0.5 * outer_fill
        r_rim = radius - 0.5 * rim_w
        tang_fill = max(0.0025, r_mid * d_theta * 0.60)
        tang_rim = max(0.0025, r_rim * d_theta * 0.60)
        fill_half = [r_mid, tang_fill, half_h * 0.70]
        rim_half = [0.5 * rim_w, tang_rim, half_h * 0.85]

        fill_parts: list[Any] = []
        rim_parts: list[Any] = []
        fill_homes: list[sapien.Pose] = []
        rim_homes: list[sapien.Pose] = []
        fill_shapes: list[Any] = []
        rim_shapes: list[Any] = []

        for i in range(n):
            theta = 0.5 * np.pi - (i + 0.5) * d_theta
            c, s = float(np.cos(theta)), float(np.sin(theta))
            q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], theta).tolist()
            f_home = sapien.Pose(
                [cx + r_mid * c, cy + r_mid * s, fill_z], q
            )
            r_home = sapien.Pose(
                [cx + r_rim * c, cy + r_rim * s, rim_z], q
            )
            # Start hidden; _update lights the active wedge count.
            fill = create_visual_box(
                self,
                self._hidden_pose(),
                half_size=fill_half,
                color=tuple(self.COLOR_GREEN),
                name=f"pie_timer_fill_{station['tag']}_{i}",
            )
            rim = create_visual_box(
                self,
                self._hidden_pose(),
                half_size=rim_half,
                color=tuple(self.COLOR_GREEN_RIM),
                name=f"pie_timer_rim_{station['tag']}_{i}",
            )
            fill_parts.append(fill)
            rim_parts.append(rim)
            fill_homes.append(f_home)
            rim_homes.append(r_home)
            for comp in fill.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    fill_shapes.extend(list(comp.render_shapes))
            for comp in rim.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    rim_shapes.extend(list(comp.render_shapes))

        station["pie_timer"] = {
            "center": (cx, cy, disc_z + float(self.table_z_bias)),
            "base_shapes": base_shapes,
            "fill_parts": fill_parts,
            "rim_parts": rim_parts,
            "fill_homes": fill_homes,
            "rim_homes": rim_homes,
            "fill_shapes": fill_shapes,
            "rim_shapes": rim_shapes,
            "n": n,
            "phase": "empty",
            "fill": 0.0,
            "n_lit": 0,
        }

    @staticmethod
    def _set_shapes_color(shapes: list[Any], rgb: list[float]) -> None:
        color = list(rgb)[:3] + [1.0]
        for shape in shapes:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _update_station_pie_timer(self, station: dict[str, Any]) -> None:
        """Show / recolor pie sections from the station's current doneness."""
        timer = station.get("pie_timer")
        if not timer:
            return
        phase, fill = self._timer_phase_and_fill(float(station.get("doneness", 0.0)))
        n = int(timer["n"])
        n_lit = 0 if phase == "empty" else int(np.clip(int(np.round(fill * n)), 0, n))
        if timer.get("phase") == phase and int(timer.get("n_lit", -1)) == n_lit:
            timer["fill"] = float(fill)
            return

        fill_rgb, rim_rgb = self._phase_colors(phase)
        hidden = self._hidden_pose()
        for i in range(n):
            on = i < n_lit
            fill_parts = timer["fill_parts"]
            rim_parts = timer["rim_parts"]
            if on:
                fill_parts[i].set_pose(timer["fill_homes"][i])
                rim_parts[i].set_pose(timer["rim_homes"][i])
                self._set_shapes_color([timer["fill_shapes"][i]], fill_rgb)
                self._set_shapes_color([timer["rim_shapes"][i]], rim_rgb)
            else:
                fill_parts[i].set_pose(hidden)
                rim_parts[i].set_pose(hidden)
        timer["phase"] = phase
        timer["fill"] = float(fill)
        timer["n_lit"] = int(n_lit)

    def _advance_station_cook(self, station: dict[str, Any]) -> None:
        """Advance doneness (parent) and refresh the pie timer in lockstep."""
        super()._advance_station_cook(station)
        self._update_station_pie_timer(station)

    # -------------------------------------------------- interactive capture hooks
    def interactive_record_visual_state(self) -> dict:
        """Per-station doneness — the only input the pie timer needs."""
        return {
            "doneness": [
                float(st.get("doneness", 0.0))
                for st in (getattr(self, "stations", None) or [])
            ]
        }

    def interactive_restore_visual_state(self, state) -> None:
        """Re-light the pie wedges for a logged frame during offscreen capture.

        Wedges are visual-only boxes toggled by pose, so restoring physx actor
        poses alone leaves the dial stuck on its end-of-episode reading. The
        phase / n_lit latch is cleared so the refresh is not skipped.
        """
        if not state:
            return
        levels = state.get("doneness") or []
        for station, level in zip(getattr(self, "stations", None) or [], levels):
            station["doneness"] = float(level)
            timer = station.get("pie_timer")
            if timer:
                timer["phase"] = None
                timer["n_lit"] = -1
            self._update_station_pie_timer(station)

    def get_obs(self) -> dict[str, Any]:
        """Include per-station pie-timer phase / fill in the cooking obs."""
        obs = super().get_obs()
        cooking = obs.setdefault("cooking", {})
        stations = getattr(self, "stations", None) or []
        phases, fills, n_lits = [], [], []
        for st in stations:
            timer = st.get("pie_timer") or {}
            phase, fill = self._timer_phase_and_fill(float(st.get("doneness", 0.0)))
            phases.append(str(timer.get("phase", phase)))
            fills.append(float(timer.get("fill", fill)))
            n_lits.append(int(timer.get("n_lit", 0)))
        cooking["pie_timer_phase"] = phases
        cooking["pie_timer_fill"] = fills
        cooking["pie_timer_n_lit"] = n_lits
        return obs
