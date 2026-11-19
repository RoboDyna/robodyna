"""Thin HTTP client for the FastWAM inference server (``fwam_server.py``).

One FastWAM process needs about 28 GiB of GPU memory, so the model can run on a separate GPU or
node from the simulator. Arrays use the same base64 encoding as ``policy/xvla/xvla_client.py``.

FastWAM chunk inference is stateless: ``infer_action`` takes the current composite frame, the
proprio vector, the text context and a noise seed derived from ``(episode_count, step_count)``.
The server therefore holds only the weights, and all episode state lives in this client:

* the ``pending_actions`` queue and ``replan_steps`` (one request per replan, not per step)
* ``episode_count`` / ``step_count``, sent so the server reproduces the noise seed
* ``take_action(action, action_type="qpos")``
* the training-prompt remap (see ``deploy_policy.py``)

``get_action`` is the single inference entry point, so timing it measures the full round trip.
"""

from __future__ import annotations

import json
import os
import time
from base64 import b64encode, b64decode
from collections import deque
from pathlib import Path

import numpy as np
import requests
from numpy.lib.format import dtype_to_descr


def _encode(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"__numpy__": b64encode(a.data).decode(),
            "dtype": dtype_to_descr(a.dtype),
            "shape": list(a.shape)}


def _decode(d: dict) -> np.ndarray:
    return np.frombuffer(b64decode(d["__numpy__"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"])


class FastWAMClient:
    """Drop-in stand-in for ``WorldActionRobotWinPolicy`` that infers over HTTP."""

    def __init__(self, host="127.0.0.1", port=8020, replan_steps=24, seed=None,
                 timeout=600.0, connect_retries=60, retry_wait=10.0, instruction_hook=None):
        self.url = f"http://{host}:{port}"
        self.replan_steps = int(replan_steps)
        self.seed = seed
        # Applied inside get_action, before the string is sent. The server looks the prompt up
        # by exact hash, so the remap must happen on this side.
        self.instruction_hook = instruction_hook or (lambda s: s)
        self.timeout = float(timeout)
        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        # Diagnostic, off by default: when FASTWAM_PREDICT_DIR is set, every replan also asks the
        # server for the world model's predicted future frames and writes them there (see
        # _dump_prediction).
        self.predict_dir = os.environ.get("FASTWAM_PREDICT_DIR") or None
        self._predict_broken = False
        self._session = requests.Session()
        self._wait_ready(connect_retries, retry_wait)

    def _wait_ready(self, retries: int, wait: float) -> None:
        """Poll ``/health`` until the server is up; loading the model takes about 130 s."""
        last = None
        for i in range(int(retries)):
            try:
                r = self._session.get(f"{self.url}/health", timeout=10.0)
                if r.ok:
                    info = r.json()
                    print(f"\033[34m[fwam-client] server ready: {info}\033[0m", flush=True)
                    return
                last = f"HTTP {r.status_code}"
            except Exception as exc:
                last = repr(exc)
            if i == 0:
                print(f"\033[33m[fwam-client] waiting for {self.url} ...\033[0m", flush=True)
            time.sleep(wait)
        raise RuntimeError(f"FastWAM server at {self.url} never became ready ({last})")

    # -- inference ------------------------------------------------------------------------------
    def _payload(self, observation, instruction) -> dict:
        obs = observation["observation"]
        return {
            "head": _encode(np.asarray(obs["head_camera"]["rgb"], dtype=np.uint8)),
            "left": _encode(np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8)),
            "right": _encode(np.asarray(obs["right_camera"]["rgb"], dtype=np.uint8)),
            "joint": _encode(np.asarray(observation["joint_action"]["vector"], dtype=np.float32)),
            "instruction": str(self.instruction_hook(instruction)),
            "episode": int(self.episode_count),
            "step": int(self.step_count),
            "seed": None if self.seed is None else int(self.seed),
        }

    def get_action(self, observation, instruction) -> np.ndarray:
        """-> [T, 14] denormalised joint-position chunk, exactly what upstream returns."""
        r = self._session.post(f"{self.url}/act", json=self._payload(observation, instruction),
                               timeout=self.timeout)
        r.raise_for_status()
        return _decode(r.json()["action"])

    def _dump_prediction(self, observation, instruction, chunk) -> None:
        """Write the world model's predicted future for this replan, next to the rollout.

        A FastWAM rollout contains no prediction: the deployed entry point runs the video expert
        once at ``timestep_video=0`` only to prefill a KV cache, and denoises only the action
        latents. ``/predict`` runs the joint path on the same observation and noise seed, which
        does denoise video, and returns the actions from both branches; ``joint_vs_act`` below is
        the distance between the prediction-path action and the executed one. Frame 0 is the
        conditioning observation round-tripped through the VAE, so consecutive replans show the
        predicted future next to what happened.

        Never fatal: a failing diagnostic disables itself and the scored episode continues.
        """
        if self._predict_broken:
            return
        try:
            t0 = time.perf_counter()
            r = self._session.post(f"{self.url}/predict",
                                   json=self._payload(observation, instruction),
                                   timeout=self.timeout)
            r.raise_for_status()
            out = r.json()
            root = Path(self.predict_dir) / f"ep{self.episode_count:02d}"
            root.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(out["frames"]):
                (root / f"t{self.step_count:05d}_f{i:02d}.png").write_bytes(b64decode(b))
            joint = np.asarray(_decode(out["action"]), dtype=np.float32)
            act = np.asarray(chunk, dtype=np.float32)
            with (Path(self.predict_dir) / "meta.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "episode": int(self.episode_count),
                    "step": int(self.step_count),
                    "frames": len(out["frames"]),
                    "infer_s": out.get("infer_s"),
                    "round_trip_s": round(time.perf_counter() - t0, 3),
                    # Motion each branch commands over the chunk, and the distance between the
                    # two branches.
                    "act_span": round(float(np.abs(act[-1] - act[0]).max()), 5),
                    "joint_span": round(float(np.abs(joint[-1] - joint[0]).max()), 5),
                    "joint_vs_act": round(float(np.abs(joint - act).max()), 5),
                }) + "\n")
        except Exception as exc:
            self._predict_broken = True
            print(f"\033[33m[fwam-client] prediction dump disabled after {type(exc).__name__}: "
                  f"{exc}\033[0m", flush=True)

    # -- the bookkeeping upstream's step() does -------------------------------------------------
    def _fill_action_queue(self, observation, instruction) -> None:
        chunk = self.get_action(observation, instruction)
        # In addition to get_action: the episode is scored on the deployed path's action.
        if self.predict_dir:
            self._dump_prediction(observation, instruction, chunk)
        for i in range(min(self.replan_steps, chunk.shape[0])):
            self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when the action queue is empty.")
            self._fill_action_queue(observation, task_env.get_instruction())
        if not self.pending_actions:
            print("\033[33m[fwam-client] no action generated; skipping step\033[0m", flush=True)
            return
        task_env.take_action(self.pending_actions.popleft(), action_type="qpos")
        self.step_count += 1

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.step_count = 0

    # Per-episode timing is reported by the server, so these upstream hooks are no-ops here.
    def reset_timing_rollout(self) -> None:
        pass

    def report_timing(self, *a, **kw) -> None:
        pass
