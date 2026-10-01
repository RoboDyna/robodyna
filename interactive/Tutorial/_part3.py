"""Part 3 coach: grasp a cube, hold a button, toggle a switch, push a box."""
from __future__ import annotations

from _interactive_common import print_instructions, run_viewer_loop
from _key_hud import TutorialKeyHud, build_part3_key_hud

_MOVE_KEYS = (
    ("q", "Q"),
    ("e", "E"),
    ("left", "left"),
    ("right", "right"),
    ("up", "up"),
    ("down", "down"),
)
_PRESS_KEYS = (("space", "Space"), *_MOVE_KEYS)

# (stage, window keys to highlight, prompt)
_STAGES: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "grasp",
        _PRESS_KEYS,
        "Pick the cube: arrows to move, E/Q for height, Space to close, then lift.",
    ),
    (
        "hold",
        _PRESS_KEYS,
        "Press the button: arrows onto it, Space to close, Q to hold, E to lift off.",
    ),
    (
        "switch",
        _PRESS_KEYS,
        "Turn key on/off: arrows onto it, Space to close, Q to turn ON (red), Q again OFF.",
    ),
    (
        "push",
        _PRESS_KEYS,
        "Push the box over the line: close gripper, lower arm, and push.",
    ),
)


class Part3Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.stage_index = 0
        self._prev: dict[str, bool] = {}
        self.done = False
        self._armed = False
        for _stage, keys, _prompt in _STAGES:
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
        stage, _keys, prompt = _STAGES[0]
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
        finished = _STAGES[self.stage_index][0]
        print_instructions(f"{finished} — done.")
        self.stage_index += 1
        if self.stage_index >= len(_STAGES):
            self.done = True
            self.env._tutorial_complete = True
            print_instructions("Basic actions — part 3 complete.")
            return
        self._reset_arms()
        stage, _keys, prompt = _STAGES[self.stage_index]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def update_keys(self, window) -> None:
        """HUD flashes only — never spawn/despawn (runs during render)."""
        if self.done or window is None:
            return
        self._ensure_left_arm()
        _stage, keys, _prompt = _STAGES[self.stage_index]
        held: set[str] = set()
        for window_key, label in keys:
            if window.key_down(window_key):
                held.add(label)
            if self._edge(window, window_key):
                self.hud.flash(label)
        self.hud.set_held(held)

    def update_stage(self, window, _step: int) -> None:
        """Advance props on the physics tick, not during viewer.render()."""
        if self.done:
            return
        self._ensure_left_arm()
        if self.env.tutorial_stage_complete():
            self._advance()

    def is_done(self, _step: int):
        if self.done:
            return True, "basic actions"
        return False


def run_part3(env) -> int:
    env._tutorial_complete = False
    hud = build_part3_key_hud(env.scene)
    coach = Part3Coach(env, hud)
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
