"""FastWAM inference server: holds the weights and returns one action chunk per request.

Run this in FastWAM's own environment on a dedicated GPU; simulators connect over HTTP through
``policy/fwam/fwam_client.py``. The server calls upstream's ``_infer_action_chunk`` unmodified, so
the composite build, z-score normalisation and noise-seed derivation are upstream's code.

Two differences from upstream's in-process adapter keep the model within a 16 GiB card:

1. The text encoder is not loaded. ``WorldActionRobotWinPolicy.__init__`` sets
   ``load_text_encoder = True``, which puts umT5-XXL (10.24 GiB) on the GPU. Training used
   precomputed embeddings keyed by the sha256 of the exact prompt, so the server feeds the cached
   ``context``/``context_mask`` to ``infer_action`` instead of ``prompt``, as upstream's LIBERO
   server does.
2. The simulator runs in another process, so SAPIEN's Vulkan context does not share this GPU.

A prompt that is not in the training cache raises an error instead of producing actions
conditioned on an unseen string.

    python policy/fwam/fwam_server.py --port 8020 [--step 13170] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import io
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
# Placeholder: pass --run or set FASTWAM_RUN to the FastWAM training run (relative to --root).
DEFAULT_RUN = "runs/<fastwam_run>"
SIM_TASK = "robodyna_uncond_3cam_384_1e-4"
CONTEXT_LEN = 128            # data.train.processor.context_len in the run's config


def _encode(a):
    a = np.ascontiguousarray(a)
    return {"__numpy__": b64encode(a.data).decode(),
            "dtype": dtype_to_descr(a.dtype), "shape": list(a.shape)}


def _decode(d):
    return np.frombuffer(b64decode(d["__numpy__"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _png(image):
    """Encode a predicted frame for the wire as PNG. Nine 384x320 frames per replan are 3.3 MB
    raw and the client only writes them to disk, so compression saves about 10x of bandwidth."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode()


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
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"], payload["mask"]
        if context.shape[0] != self.context_len or mask.shape[0] != self.context_len:
            raise ValueError(f"cached context_len mismatch in {path}")
        # Zero the padded positions and pass an all-ones mask. Every upstream producer of a
        # context does the same: the training dataloader (`robot_video_dataset.py`), the online
        # encoder (`fastwam.encode_prompt`) and upstream's cached-context loader
        # (`experiments/libero/task_prompt_cache.py`). Passing the cached payload through unmodified
        # would leave about 100 padding embeddings in cross-attention behind a restrictive mask,
        # an input the model never saw in training, and lowers its score.
        mask = mask.bool()
        context = context.clone()
        context[~mask] = 0.0
        mask = torch.ones_like(mask)
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

    # Force load_text_encoder=False by intercepting `instantiate` in the upstream module's
    # namespace, where the flag is set on the config just before hydra builds the model. A
    # RuntimeError below fires if an upstream version does not build the model through it.
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

    # Replace the prompt with its cached context where the prompt is consumed, leaving
    # `_infer_action_chunk` (composite build, normalisation, seed derivation) unchanged.
    raw_infer = policy.model.infer_action

    @functools.wraps(raw_infer)                 # keeps inspect.signature working for the
    def infer(*a, **kw):                        # `num_video_frames` capability check upstream does
        if kw.get("prompt") is not None:
            kw["context"], kw["context_mask"] = cache.get(kw["prompt"])
        # `prompt` is a required argument that accepts None, so it must be supplied even when the
        # caller did not name it; otherwise TypeError is raised before the mutual-exclusion check.
        # `infer_joint` forwards `prompt=None` alongside its own context to cross-check the
        # deployed path.
        if not a:
            kw["prompt"] = None
        kw.pop("negative_prompt", None)         # only meaningful with a text encoder + cfg > 1
        return raw_infer(*a, **kw)

    policy.model.infer_action = infer
    # `/predict` needs the two things `_infer_action_chunk` takes from module scope: the prompt
    # template and the text cache that replaces the text encoder. They are stored on the policy,
    # not as globals, so the endpoint always uses the context built for this checkpoint.
    policy._fwam_prompt_template = upstream.DEFAULT_PROMPT
    policy._fwam_text_cache = cache
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

        def log_message(self, *a):              # suppress per-request access logging
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

        def _request(self):
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
            return req, observation

        def _bind(self, req):
            """Install this request's noise-seed inputs. Must be called holding ``lock``.

            These are process-wide fields on a policy shared by every simulator, and the noise seed
            is derived from them. Binding them outside the critical section would let another
            client overwrite them between the assignment and the inference.
            """
            policy.episode_count = int(req.get("episode", 0))
            policy.step_count = int(req.get("step", 0))
            policy.seed = req.get("seed")

        def do_POST(self):
            route = {"/act": self._act, "/predict": self._predict}.get(self.path)
            if route is None:
                return self._send(404, {"error": "not found"})
            try:
                route()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _act(self):
            req, observation = self._request()
            # One inference at a time per GPU, so several simulators can share one card.
            with lock:
                self._bind(req)
                t0 = time.perf_counter()
                chunk = policy._infer_action_chunk(observation, str(req["instruction"]))
                dt = time.perf_counter() - t0
            self._send(200, {"action": _encode(np.asarray(chunk, dtype=np.float32)),
                             "infer_s": round(dt, 3)})

        def _predict(self):
            """Also denoise the video branch, so the world model's prediction can be inspected.

            The deployed path does not produce a prediction. ``infer_action`` runs the video expert
            once, at ``timestep_video=0`` on the clean first-frame latent, only to prefill a KV
            cache for the action expert; it denoises only ``latents_action``.

            ``infer_joint`` denoises ``latents_video`` and returns the action from both branches.
            With ``test_action_with_infer_action`` it re-runs the deployed path under the same seed
            and warns when the two disagree, so one request returns the predicted future and checks
            that its action matches the executed one.

            This is a separate endpoint because it is much heavier than ``/act`` (a full video
            denoise plus a VAE decode of every predicted frame).
            """
            req, observation = self._request()
            with lock:
                self._bind(req)
                prompt = policy._fwam_prompt_template.format(task=str(req["instruction"]))
                context, context_mask = policy._fwam_text_cache.get(prompt)
                image = policy._build_robotwin_image_tensor(observation)
                proprio = policy._normalize_state(
                    np.asarray(observation["joint_action"]["vector"], dtype=np.float32))
                seed = policy._next_inference_seed()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = policy.model.infer_joint(
                        prompt=None,
                        input_image=image,
                        num_video_frames=int(policy._num_video_frames),
                        action_horizon=int(policy.action_horizon),
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        num_inference_steps=int(policy.num_inference_steps),
                        sigma_shift=policy.sigma_shift,
                        seed=seed,
                        rand_device=policy.rand_device,
                        tiled=policy.tiled,
                        # Needs a seed; unseeded runs simply lose the cross-check, not the video.
                        test_action_with_infer_action=seed is not None,
                    )
                dt = time.perf_counter() - t0
                joint_action = policy._denormalize_action(out["action"])[0]
                frames = [_png(f) for f in out["video"]]
            self._send(200, {
                "frames": frames,
                "action": _encode(np.asarray(joint_action, dtype=np.float32)),
                "infer_s": round(dt, 3),
            })

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
