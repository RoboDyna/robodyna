"""Part 4 coach: suite-specific advanced actions.

Base: rolling ball → mallet.
Household: stove knob → multi-stage force key.
"""
from __future__ import annotations

from _interactive_common import print_instructions, run_viewer_loop
from _key_hud import TutorialKeyHud, build_part4_key_hud

_MOVE_KEYS = (
    ("q", "Q"),
    ("e", "E"),
    ("left", "left"),
    ("right", "right"),
    ("up", "up"),
    ("down", "down"),
)

_STAGE_BALL = (
    "ball",
    (
        ("space", "Space"),
        *_MOVE_KEYS,
    ),
    "Pick up the ball: move onto it, close (Space), lift (E). "
    "Touching it is fine — it only respawns if it falls off the table.",
)
_STAGE_STOVE = (
    "stove",
    (
        ("space", "Space"),
        ("r", "R"),
        ("t", "T"),
        *_MOVE_KEYS,
    ),
    "Turn on/off the stove: arrows + Q/E onto the knob, close (Space), yaw left (R) to light "
    "the fire, yaw back (T) to turn it off.",
)
_STAGE_MALLET = (
    "mallet",
    (("space", "Space"), ("e", "E")),
    "Pick up the mallet: close on the handle (Space), then lift (E).",
)
_STAGE_FORCE = (
    "force_key",
    (
        ("q", "Q"),
        ("e", "E"),
        ("left", "left"),
        ("right", "right"),
        ("up", "up"),
        ("down", "down"),
    ),
    "Press multistep button: press Q until the bar enters the yellow region (success). "
    "Yellow moves — release (E), then press again for the next band, "
    "through full force.",
)

# Base suite: pick dynamic + pick tool. Household: knob + multi-step press.
_STAGES_BY_SUITE: dict[str, tuple[tuple[str, tuple[tuple[str, str], ...], str], ...]] = {
    "base": (_STAGE_BALL, _STAGE_MALLET),
    "household": (_STAGE_STOVE, _STAGE_FORCE),
}


def stages_for_suite(suite: str):
    key = str(suite or "base").strip().lower()
    if key not in _STAGES_BY_SUITE:
        raise ValueError(f"Unknown part-4 suite: {suite!r} (expected base|household)")
    return _STAGES_BY_SUITE[key]


class Part4Coach:
    def __init__(self, env, hud: TutorialKeyHud, stages):
        self.env = env
        self.hud = hud
        self.stages = stages
        self.stage_index = 0
        self._prev: dict[str, bool] = {}
        self.done = False
        self._armed = False
        self._last_feedback = ""
        # After ball success, wait for release before spawning the next prop.
        self._pending_advance = False
        self._release_prompted = False
        for _stage, keys, _prompt in stages:
            for window_key, _label in keys:
                self._prev[window_key] = False

    def _edge(self, window, key: str) -> bool:
        down = bool(window.key_down(key))
        edge = down and not self._prev.get(key, False)
        self._prev[key] = down
        return edge

    def _ensure_left_arm(self) -> None:
        if self._armed:
            return
        controls = getattr(self.env, "_interactive_robot_controls", None)
        if controls is None:
            return
        try:
            if not controls.selected:
                controls.selected = ("left",)
                self.env._interactive_selected_arms = ("left",)
                controls._highlight_selected()
            self._armed = True
        except Exception:
            pass

    def start(self) -> None:
        stage, _keys, prompt = self.stages[0]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def _reset_arms(self) -> None:
        """Home both arms (open grippers) before the next stage props appear."""
        controls = getattr(self.env, "_interactive_robot_controls", None)
        if controls is None:
            return
        restored = []
        try:
            restored = list(
                controls.return_arms_to_origin(("left", "right"), open_grippers=True)
                or []
            )
        except Exception as exc:
            print(f"[tutorial] arm reset failed: {exc}")
            return
        try:
            controls._command.clear()
        except Exception:
            pass
        try:
            controls.selected = ("left",)
            self.env._interactive_selected_arms = ("left",)
            controls._highlight_selected()
        except Exception:
            pass
        if restored:
            print("Arms reset for next stage: " + " + ".join(restored))

    def _advance(self) -> None:
        finished = self.stages[self.stage_index][0]
        print_instructions(f"{finished} — done.")
        self.stage_index += 1
        self._last_feedback = ""
        self.hud.flash_kwargs = {}
        self._pending_advance = False
        self._release_prompted = False
        if self.stage_index >= len(self.stages):
            self.done = True
            self.env._tutorial_complete = True
            print_instructions("Advanced actions — part 4 complete.")
            return
        self._reset_arms()
        stage, _keys, prompt = self.stages[self.stage_index]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def _try_finish_pending(self) -> None:
        """Park the held ball, then spawn the next stage once contact is gone."""
        if not self._pending_advance:
            return
        released = True
        try:
            released = bool(self.env.tutorial_ball_released())
        except Exception:
            released = True
        if not released:
            try:
                self.env.tutorial_force_open_grippers()
            except Exception:
                pass
            if not self._release_prompted:
                self._release_prompted = True
                print_instructions(
                    "Ball caught — open the gripper (Space) so the next object can appear."
                )
            return
        try:
            self.env._park_ball_safe()
            self.env._defer_hide_ball = False
        except Exception:
            pass
        self._advance()

    def _sync_force_kwargs(self) -> None:
        if self.stages[self.stage_index][0] != "force_key":
            self.hud.flash_kwargs = {}
            return
        force = float(getattr(self.env, "_key_force_n", 0.0) or 0.0)
        peak = float(getattr(self.env, "_press_peak_force", 0.0) or 0.0)
        target = int(getattr(self.env, "_force_target_level", 1) or 1)
        cleared = int(getattr(self.env, "_force_cleared", 0) or 0)
        feedback = str(getattr(self.env, "_force_feedback", "") or "")
        thresholds = tuple(getattr(self.env, "FORCE_THRESHOLDS", (3.0, 6.0, 10.0, 14.0)))
        if feedback and feedback != self._last_feedback:
            self._last_feedback = feedback
            print_instructions(feedback)
        self.hud.flash_kwargs = {
            "force_n": force,
            "peak_n": peak,
            "target_level": target,
            "cleared": cleared,
            "thresholds": thresholds,
            "feedback": feedback,
        }
        self.hud._play_drawn = None
        self.hud._lesson_key = None

    def update_keys(self, window) -> None:
        """HUD flashes only — never spawn/despawn (runs during render)."""
        if self.done or window is None:
            return
        self._ensure_left_arm()
        _stage, keys, _prompt = self.stages[self.stage_index]
        held: set[str] = set()
        for window_key, label in keys:
            if window.key_down(window_key):
                held.add(label)
            if self._edge(window, window_key):
                self.hud.flash(label)
        self.hud.set_held(held)
        self._sync_force_kwargs()

    def update_stage(self, window, _step: int) -> None:
        """Advance props on the physics tick, not during viewer.render()."""
        if self.done:
            return
        self._ensure_left_arm()
        if self._pending_advance:
            self._try_finish_pending()
            return
        if not self.env.tutorial_stage_complete():
            return
        finished = self.stages[self.stage_index][0]
        if finished == "ball":
            self._pending_advance = True
            self._try_finish_pending()
            return
        self._advance()

    def is_done(self, _step: int):
        if self.done:
            return True, "advanced actions"
        return False


def run_part4(env, suite: str = "base") -> int:
    stages = stages_for_suite(suite)
    env._tutorial_complete = False
    hud = build_part4_key_hud(env.scene, stages=tuple(s[0] for s in stages))
    coach = Part4Coach(env, hud, stages)
    coach.start()
    hud.on_frame = coach.update_keys
    run_viewer_loop(
        env,
        on_step=coach.update_stage,
        is_done=coach.is_done,
        extra_plugins=[hud],
        report_result=False,
    )
    return 0
