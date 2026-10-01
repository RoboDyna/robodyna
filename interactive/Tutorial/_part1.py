"""Part 1 coach: test arm-select keys 1/2/3, then camera-view key V."""
from __future__ import annotations

import time

from _interactive_common import print_instructions, run_viewer_loop
from _key_hud import TutorialKeyHud, build_key_hud

_PLAY_KEYS = (
    ("1", "1"),
    ("2", "2"),
    ("3", "3"),
    ("v", "V"),
    ("escape", "Esc"),
)
# Pause so the tested green keycaps are visible before the overlay changes.
_TESTED_HOLD = 0.45


class Part1Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.pressed = {"1": False, "2": False, "3": False}
        self._prev = {"1": False, "2": False, "3": False, "v": False, "escape": False}
        self.stage = "arms"
        self._hold_until = 0.0
        self._after_hold = None

    def _edge(self, window, key: str) -> bool:
        down = bool(window.key_down(key))
        edge = down and not self._prev[key]
        self._prev[key] = down
        return edge

    def _enter_play(self) -> None:
        self.stage = "play"
        self.hud.flash_enabled = True
        self.hud.set_stage("play")
        print_instructions(
            "Keep experimenting with 1 / 2 / 3 and V. Keys flash when pressed. Esc quits."
        )

    def _finish_hold(self) -> None:
        nxt = self._after_hold
        self._after_hold = None
        self._hold_until = 0.0
        if nxt == "view":
            self.stage = "view"
            self.hud.set_stage("view")
            print_instructions("Now press V to switch between camera views.")
        elif nxt == "play":
            self._enter_play()

    def update(self, window) -> None:
        if window is None:
            return
        if self._after_hold:
            if time.perf_counter() >= self._hold_until:
                self._finish_hold()
            return
        if self.stage == "arms":
            for key in ("1", "2", "3"):
                if self._edge(window, key) and not self.pressed[key]:
                    self.pressed[key] = True
                    self.hud.mark_arm_pressed(key)
                    names = {"1": "left", "2": "right", "3": "both"}
                    print_instructions(
                        f"Arm key {key} ({names[key]}) — "
                        + ", ".join(
                            k if self.pressed[k] else f"{k}?"
                            for k in ("1", "2", "3")
                        )
                    )
            if all(self.pressed.values()):
                self._hold_until = time.perf_counter() + _TESTED_HOLD
                self._after_hold = "view"
        elif self.stage == "view":
            if self._edge(window, "v"):
                self.hud.mark_v_pressed()
                self._hold_until = time.perf_counter() + _TESTED_HOLD
                self._after_hold = "play"
        elif self.stage == "play":
            held: set[str] = set()
            for sapien_key, label in _PLAY_KEYS:
                if window.key_down(sapien_key):
                    held.add(label)
                if self._edge(window, sapien_key):
                    self.hud.flash(label)
            self.hud.set_held(held)


def run_part1(env) -> int:
    env._tutorial_complete = False
    hud = build_key_hud(env.scene)
    coach = Part1Coach(env, hud)
    hud.on_frame = coach.update
    run_viewer_loop(
        env,
        on_step=lambda window, _step: coach.update(window),
        is_done=None,
        extra_plugins=[hud],
        report_result=False,
    )
    return 0
