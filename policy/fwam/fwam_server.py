"""FastWAM inference server: holds the weights, answers one chunk per request.

Run this in FastWAM's own environment, on a GPU that has nothing else on it; the simulators run
elsewhere and talk to it over HTTP (``policy/fwam/fwam_client.py``). Nothing about the inference
path changes -- the server calls upstream's ``_infer_action_chunk`` unmodified, so the composite
build, the z-score normalisation and the noise-seed derivation are all still upstream's code.

Two deliberate departures from upstream's in-process adapter, both needed to fit a 16 GiB card:

1. **The text encoder is not loaded.** ``WorldActionRobotWinPolicy.__init__`` hardcodes
   ``load_text_encoder = True``, which pulls umT5-XXL (10.24 GiB, 36% of the budget) onto the GPU
   to re-encode a fixed string every 24 actions. Training never did that: the dataloader fetches a
   **precomputed** embedding keyed by sha256 of the exact prompt, and the run's cache is the
   complete set of prompts this checkpoint ever saw. So the encoder is skipped and the cached
   ``context``/``context_mask`` are fed instead -- which ``infer_action`` accepts as a first-class
   alternative to ``prompt`` (``fastwam.py:1083``), and which upstream's own LIBERO server uses.
   This is not an approximation of training; it is what training actually did.
2. **The simulator is on another machine**, so SAPIEN's Vulkan context is not in this process.

A cache miss is fatal on purpose. Passing a prompt the checkpoint never saw would still produce
actions -- just conditioned on nothing meaningful -- and would show up only as an unexplained 0%.

    python policy/fwam/fwam_server.py --port 8020 [--step 13170] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import socket
import sys
import threading
import time
from base64 import b64encode, b64decode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from numpy.lib.format import dtype_to_descr

FASTWAM_ROOT = Path(os.environ.get("FASTWAM_ROOT", "./third_party/fastwam"))
DEFAULT_RUN = "runs/FastWAM_RoboDynaV2_gpu-h200-103_20260825_144659"
SIM_TASK = "robodyna_uncond_3cam_384_1e-4"
CONTEXT_LEN = 128            # data.train.processor.context_len in the run's config


def _encode(a):
    a = np.ascontiguousarray(a)
    return {"__numpy__": b64encode(a.data).decode(),
            "dtype": dtype_to_descr(a.dtype), "shape": list(a.shape)}


def _decode(d):
    return np.frombuffer(b64decode(d["__numpy__"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _load_upstream(root: Path):
    for entry in (str(root), str(root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    path = root / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py"
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM RoboTwin adapter not found at {path}")
    spec = importlib.util.spec_from_file_location("fastwam_robotwin_deploy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TextContextCache:
    """The training cache, read the way ``robot_video_dataset._get_cached_text_context`` reads it."""

    def __init__(self, cache_dir: Path, context_len: int = CONTEXT_LEN):
        self.dir = Path(cache_dir)
        self.context_len = int(context_len)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"text embedding cache not found: {self.dir}")
        self._mem = {}
        n = len(list(self.dir.glob(f"*.t5_len{self.context_len}.wan22ti2v5b.pt")))
        print(f"[fwam-server] text cache {self.dir} ({n} prompts)", flush=True)

    def get(self, prompt: str):
        if prompt in self._mem:
            return self._mem[prompt]
        import torch
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = self.dir / f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not path.is_file():
            raise KeyError(
                f"prompt not in the training text cache ({path.name}). This checkpoint was never "
                f"conditioned on this string, so serving it would measure nothing. Prompt: {prompt!r}"
            )
        payload = torch.load(path, map_location="cpu")
        context, mask = payload["context"], payload["mask"].bool()
        if context.shape[0] != self.context_len or mask.shape[0] != self.context_len:
            raise ValueError(f"cached context_len mismatch in {path}")
        self._mem[prompt] = (context, mask)
        return context, mask


def build_policy(root: Path, run_dir: str, step: int, replan_steps: int):
    import torch
    from omegaconf import OmegaConf

    upstream = _load_upstream(root)

    run = Path(run_dir)
    if not run.is_absolute():
        run = root / run
    checkpoint = run / "checkpoints" / "weights" / f"step_{step:06d}.pt"
    stats = run / "dataset_stats.json"
    for p in (checkpoint, stats):
        if not p.is_file():
            raise FileNotFoundError(p)

    args = {
        "ckpt_setting": str(checkpoint),          # upstream's name for "checkpoint path"
        "dataset_stats_path": str(stats),
        "sim_task": SIM_TASK,
        "sim_cfg_name": "sim_robotwin.yaml",
        "replan_steps": int(replan_steps),
    }

    # Force load_text_encoder=False without forking upstream's 60-line __init__: the flag is set on
    # the config right before it is handed to hydra, so intercepting `instantiate` in the upstream
    # module's own namespace is the smallest possible edit and it fails loudly if upstream ever
    # stops routing through it.
    orig_instantiate = upstream.instantiate
    seen = {"patched": False}

    @functools.wraps(orig_instantiate)
    def _no_text_encoder(cfg, *a, **kw):
        try:
            has = OmegaConf.is_config(cfg) and "load_text_encoder" in cfg
        except Exception:
            has = False
        if has:
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            cfg.load_text_encoder = False
            seen["patched"] = True
        return orig_instantiate(cfg, *a, **kw)

    upstream.instantiate = _no_text_encoder
    try:
        t0 = time.perf_counter()
        policy = upstream.get_model(args)
    finally:
        upstream.instantiate = orig_instantiate
    if not seen["patched"]:
        raise RuntimeError(
            "load_text_encoder was never intercepted -- upstream stopped building the model "
            "through `instantiate`. Refusing to serve, because the fallback silently loads a "
            "10 GiB text encoder that will not fit alongside the model on a 16 GiB card."
        )
    print(f"[fwam-server] model built in {time.perf_counter() - t0:.0f} s", flush=True)

    cache = TextContextCache(root / "data" / "text_embeds_cache" / "robodyna")

    # Swap prompt -> cached context at the single point where the prompt is consumed, leaving
    # `_infer_action_chunk` (composite build, normalisation, seed derivation) untouched.
    raw_infer = policy.model.infer_action

    @functools.wraps(raw_infer)                 # keeps inspect.signature working for the
    def infer(*a, **kw):                        # `num_video_frames` capability check upstream does
        prompt = kw.pop("prompt", None)
        if prompt is not None:
            context, mask = cache.get(prompt)
            kw["context"], kw["context_mask"] = context, mask
        kw.pop("negative_prompt", None)         # only meaningful with a text encoder + cfg > 1
        return raw_infer(*a, **kw)

    policy.model.infer_action = infer
    if getattr(policy.model, "text_encoder", "missing") not in (None, "missing"):
        raise RuntimeError("text encoder is resident despite load_text_encoder=False")

    if torch.cuda.is_available():
        print(f"[fwam-server] cuda alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB "
              f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
    return policy, checkpoint


def make_handler(policy, checkpoint, lock):
    import torch

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):              # one line per request would drown the sweep log
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            self._send(200, {
                "ok": True,
                "checkpoint": str(checkpoint),
                "host": socket.gethostname(),
                "cuda_alloc_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            })

        def do_POST(self):
            if self.path != "/act":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n))
                observation = {
                    "observation": {
                        "head_camera": {"rgb": _decode(req["head"])},
                        "left_camera": {"rgb": _decode(req["left"])},
                        "right_camera": {"rgb": _decode(req["right"])},
                    },
                    "joint_action": {"vector": _decode(req["joint"])},
                }
                # One GPU, one inference at a time. Queuing here is what lets many simulators
                # share a card without each of them needing to know about the others.
                with lock:
                    policy.episode_count = int(req.get("episode", 0))
                    policy.step_count = int(req.get("step", 0))
                    policy.seed = req.get("seed")
                    t0 = time.perf_counter()
                    chunk = policy._infer_action_chunk(observation, str(req["instruction"]))
                    dt = time.perf_counter() - t0
                self._send(200, {"action": _encode(np.asarray(chunk, dtype=np.float32)),
                                 "infer_s": round(dt, 3)})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--root", default=str(FASTWAM_ROOT))
    ap.add_argument("--run", default=os.environ.get("FASTWAM_RUN", DEFAULT_RUN))
    ap.add_argument("--step", type=int, default=int(os.environ.get("FASTWAM_STEP", 13170)))
    ap.add_argument("--replan", type=int, default=int(os.environ.get("FASTWAM_REPLAN", 24)))
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    # FastWAM resolves the Wan2.2 base as `checkpoints/<model_id>` relative to the process cwd.
    os.chdir(root)
    policy, checkpoint = build_policy(root, args.run, args.step, args.replan)

    lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy, checkpoint, lock))
    server.daemon_threads = True
    print(f"[fwam-server] listening on {socket.gethostname()}:{args.port} "
          f"(step {args.step})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
