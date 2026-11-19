"""RoboDyna policy adapter for FastWAM (Wan2.2-TI2V-5B world-action model).

FastWAM ships a RoboTwin deploy adapter with the interface ``script/eval_policy.py`` expects
(module-level ``encode_obs`` / ``get_model`` / ``eval`` / ``reset_model``, camera RGB plus
``joint_action["vector"]`` in, ``take_action(..., action_type="qpos")`` out). This module wraps
that adapter instead of re-implementing inference, and adds four things:

* ``ckpt_setting`` stays a short result label. The checkpoint path is resolved from
  ``(fastwam_run_dir, checkpoint_id)`` or ``fastwam_checkpoint``.
* The hydra task group is pinned to the one the checkpoint was trained with (``SIM_TASK``).
  The default group in ``sim_robotwin.yaml`` loads without error but uses RoboTwin's
  normalisation and frame geometry.
* The process cwd stays at the repo root, which ``./task_config/...`` needs. FastWAM looks up
  ``checkpoints/<model_id>`` relative to the cwd, so a ``checkpoints`` symlink in the repo root
  is required (see README.md).
* Chunk inference is exposed as ``get_action`` so it can be timed with
  ``TASK_ENV.inference_window()``. FastWAM itself infers inside ``step()``.

The package is named ``fwam`` because ``script/eval_policy.py`` puts ``./policy`` on
``sys.path`` and a local ``fastwam`` package would collide with the upstream library.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

FASTWAM_ROOT = Path(os.environ.get(
    "FASTWAM_ROOT", "./third_party/fastwam"))
# Placeholder: set fastwam_run_dir (or FASTWAM_RUN for the server) to the FastWAM training run.
DEFAULT_RUN = "runs/<fastwam_run>"
# The hydra task group the checkpoint was trained under (see the run's config.yaml).
SIM_TASK = "robodyna_uncond_3cam_384_1e-4"

_upstream = None

# --- training-time language conditioning -------------------------------------------------------
#
# FastWAM conditions on a cached T5 embedding of ``DEFAULT_PROMPT.format(task=<instruction>)``, so
# the exact instruction string is part of the model input.
#
# The released checkpoint was trained with a longer, scene-describing wording of each task
# instruction than the canonical lines in ``task_config/task_instructions.json``. All 35 training
# prompts hash to the files in the run's ``data/text_embeds_cache/robodyna``.
#
# ``train_prompts.json`` maps each canonical instruction to its training wording, and this map is
# applied by default. Set ``FASTWAM_TRAIN_PROMPT=0`` to pass the canonical instruction through
# unchanged, e.g. to measure prompt sensitivity.
_PROMPT_MAP_PATH = Path(__file__).with_name("train_prompts.json")
_prompt_map = None


def _training_instruction(instruction: str) -> str:
    global _prompt_map
    if os.environ.get("FASTWAM_TRAIN_PROMPT", "1") == "0":
        return instruction
    if _prompt_map is None:
        try:
            _prompt_map = json.loads(_PROMPT_MAP_PATH.read_text())["map"]
        except Exception as exc:            # missing/corrupt map must not silently change behaviour
            print(f"\033[33m[fwam] no training-prompt map ({exc}); using benchmark wording\033[0m",
                  flush=True)
            _prompt_map = {}
    entry = _prompt_map.get(instruction)
    if entry is None:
        # Report each unmapped instruction once; it usually means the map is out of date.
        if instruction not in _unmapped:
            _unmapped.add(instruction)
            print(f"\033[33m[fwam] instruction not in training-prompt map, passing through: "
                  f"{instruction!r}\033[0m", flush=True)
        return instruction
    trained = entry["train_instruction"]
    if instruction not in _remapped:
        _remapped.add(instruction)
        print(f"\033[34m[fwam] instruction -> training wording for {entry['task']}: "
              f"{trained!r}\033[0m", flush=True)
    return trained


_unmapped = set()
_remapped = set()


def _load_upstream(root: Path):
    """Import FastWAM's RoboTwin adapter by path, under a name that cannot collide with ours."""
    global _upstream
    if _upstream is not None:
        return _upstream
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
    _upstream = module
    return module


def _expose_get_action(policy):
    """Expose chunk inference as ``get_action`` and apply the training-prompt map.

    Upstream ``step()`` calls ``_infer_action_chunk``, which is routed through ``get_action``.
    ``get_action`` calls the base-class method directly, so the two do not recurse.
    """
    base = type(policy)

    class _TimedFastWAM(base):
        def get_action(self, observation, instruction):
            chunk = base._infer_action_chunk(self, observation, _training_instruction(instruction))
            if os.environ.get("FASTWAM_DEBUG_ACTIONS") == "1":
                _debug_chunk(self, observation, chunk)
            return chunk

        def _infer_action_chunk(self, observation, instruction):
            return self.get_action(observation, instruction)

    policy.__class__ = _TimedFastWAM
    return policy


def _debug_chunk(policy, observation, chunk):
    """Print per-chunk motion statistics (enabled with FASTWAM_DEBUG_ACTIONS=1)."""
    import numpy as np
    st = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    ch = np.asarray(chunk, dtype=np.float32)
    per_step = np.abs(np.diff(ch, axis=0)).max(axis=1)
    print(f"[fwam-dbg] t={getattr(policy, 'step_count', -1)} chunk={tuple(ch.shape)} "
          f"|a0-state|max={np.abs(ch[0] - st).max():.4f} "
          f"|aT-a0|max={np.abs(ch[-1] - ch[0]).max():.4f} "
          f"per_step_max={np.round(per_step, 4).tolist()}", flush=True)


def _resolve_checkpoint(usr_args, root: Path) -> Path:
    explicit = usr_args.get("fastwam_checkpoint")
    if explicit:
        path = Path(str(explicit)).expanduser()
    else:
        run_dir = Path(str(usr_args.get("fastwam_run_dir") or DEFAULT_RUN)).expanduser()
        if not run_dir.is_absolute():
            run_dir = root / run_dir
        step = int(usr_args.get("checkpoint_id") or 2000)
        path = run_dir / "checkpoints" / "weights" / f"step_{step:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {path}")
    return path


def _resolve_stats(usr_args, checkpoint: Path) -> Path:
    """Normalisation stats live beside the run, two levels up from ``checkpoints/weights``."""
    explicit = usr_args.get("dataset_stats_path")
    if explicit:
        path = Path(str(explicit)).expanduser()
    else:
        path = checkpoint.parents[2] / "dataset_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM dataset stats not found: {path}")
    return path


def encode_obs(observation):
    """Pass-through: FastWAM consumes the RoboDyna observation dict directly."""
    return observation


def _get_client(usr_args, endpoint: str):
    """Remote inference through ``fwam_server.py``, selected with ``FASTWAM_SERVER=host:port``."""
    from fwam.fwam_client import FastWAMClient

    host, _, port = endpoint.partition(":")
    client = FastWAMClient(
        host=host or "127.0.0.1",
        port=int(port or 8020),
        replan_steps=int(usr_args.get("replan_steps") or 24),
        seed=usr_args.get("seed"),
        instruction_hook=_training_instruction,
    )
    print(f"\033[34m[fwam] remote inference at {endpoint} | replan {client.replan_steps}\033[0m",
          flush=True)
    return client


def get_model(usr_args):
    endpoint = os.environ.get("FASTWAM_SERVER", "").strip()
    if endpoint:
        return _get_client(usr_args, endpoint)

    root = Path(str(usr_args.get("fastwam_root") or FASTWAM_ROOT)).expanduser()
    upstream = _load_upstream(root)

    checkpoint = _resolve_checkpoint(usr_args, root)
    args = dict(usr_args)
    args["ckpt_setting"] = str(checkpoint)   # upstream's name for "checkpoint path"
    args["dataset_stats_path"] = str(_resolve_stats(usr_args, checkpoint))
    args.setdefault("sim_task", SIM_TASK)
    args.setdefault("sim_cfg_name", "sim_robotwin.yaml")

    print(f"\033[34m[fwam] checkpoint {checkpoint}\033[0m", flush=True)
    print(f"\033[34m[fwam] sim_task {args['sim_task']} | stats {args['dataset_stats_path']}\033[0m",
          flush=True)
    return _expose_get_action(upstream.get_model(args))


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
