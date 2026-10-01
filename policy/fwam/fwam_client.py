"""Thin HTTP client for the FastWAM inference server (``fwam_server.py``).

Why a server at all: one FastWAM process needs ~28 GiB and every local GPU is 16 GiB, so the
model and the simulator cannot share a card here the way they did on the H200 box. Splitting them
across nodes is what makes the policy runnable on trt at all -- and it removes SAPIEN's Vulkan
context from the model's budget, which is the last ~2 GiB that would not otherwise fit. Same shape
as the X-VLA split (``policy/xvla/xvla_client.py``), and deliberately the same wire format.

The split is safe here because FastWAM's chunk inference is a **pure function**. ``infer_action``
takes the current composite frame, the proprio vector, the text context and a noise seed derived
from ``(episode_count, step_count)``; there is no history, no KV carried between calls, no
call-order state (upstream's own docstring on ``_next_inference_seed`` says as much). So the server
holds weights and nothing else, and every piece of episode state lives here.

What stays on this side, mirroring ``WorldActionRobotWinPolicy.step`` exactly:

* the ``pending_actions`` queue and ``replan_steps`` -- one request per replan, not per step
* ``episode_count`` / ``step_count``, which the server needs only to reproduce the noise seed
* ``take_action(action, action_type="qpos")``
* the training-prompt remap, which is a benchmark-side concern (see ``deploy_policy.py``)

``get_action`` is the inference entry point on purpose: ``script/eval_policy.py`` rebinds it to
charge inference latency to simulation time in async mode. Under the split it now measures the
round trip, which is the honest cost of this deployment -- a client that hid the network behind a
background thread would quietly score async as though inference were free.
"""

from __future__ import annotations

import json
import time
from base64 import b64encode, b64decode
from collections import deque

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
        # Applied inside get_action, i.e. inside whatever the async protocol wraps around it, and
        # before the string reaches the wire -- the server looks the prompt up by exact hash, so a
        # remap applied any later would simply miss the cache.
        self.instruction_hook = instruction_hook or (lambda s: s)
        self.timeout = float(timeout)
        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._session = requests.Session()
        self._wait_ready(connect_retries, retry_wait)

    def _wait_ready(self, retries: int, wait: float) -> None:
        """Model load is ~130 s, and under SLURM the sim job can start before the server job does.

        Failing fast here would turn a scheduling race into a combination scored 0%, which is
        indistinguishable in the journals from a policy that simply cannot do the task.
        """
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
    def get_action(self, observation, instruction) -> np.ndarray:
        """-> [T, 14] denormalised joint-position chunk, exactly what upstream returns."""
        obs = observation["observation"]
        payload = {
            "head": _encode(np.asarray(obs["head_camera"]["rgb"], dtype=np.uint8)),
            "left": _encode(np.asarray(obs["left_camera"]["rgb"], dtype=np.uint8)),
            "right": _encode(np.asarray(obs["right_camera"]["rgb"], dtype=np.uint8)),
            "joint": _encode(np.asarray(observation["joint_action"]["vector"], dtype=np.float32)),
            "instruction": str(self.instruction_hook(instruction)),
            "episode": int(self.episode_count),
            "step": int(self.step_count),
            "seed": None if self.seed is None else int(self.seed),
        }
        r = self._session.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return _decode(r.json()["action"])

    # -- the bookkeeping upstream's step() does -------------------------------------------------
    def _fill_action_queue(self, observation, instruction) -> None:
        chunk = self.get_action(observation, instruction)
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

    # Upstream prints a per-episode timing table; the split moves that to the server side.
    def reset_timing_rollout(self) -> None:
        pass

    def report_timing(self, *a, **kw) -> None:
        pass
