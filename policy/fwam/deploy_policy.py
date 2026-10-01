"""RoboDyna policy adapter for FastWAM (Wan2.2-TI2V-5B world-action model).

FastWAM already ships a RoboTwin deploy adapter with exactly the contract
``script/eval_policy.py`` drives -- module-level ``encode_obs`` / ``get_model`` / ``eval`` /
``reset_model``, observations read as ``observation["observation"][head|left|right_camera]["rgb"]``
plus ``observation["joint_action"]["vector"]``, and actions applied via
``take_action(chunk_step, action_type="qpos")``. So this file deliberately does **not**
re-implement inference. It does the four things upstream's adapter cannot know about:

* **Keeps ``ckpt_setting`` a tag.** Upstream reads the checkpoint *path* out of ``ckpt_setting``;
  our journals, result directories and LaTeX tables are all keyed on it as a short label
  (``fastwam_robodynav2``). The checkpoint is resolved here from ``(run_dir, checkpoint_id)``
  instead, matching how pi0.5 and X-VLA are addressed.
* **Pins the hydra task group** to the one the checkpoint was trained under. Composing
  ``sim_robotwin.yaml`` with its own default (``robotwin_uncond_3cam_384_1e-4``) would build a
  model whose shapes happen to load but whose normaliser and frame geometry are RoboTwin's, not
  RoboDyna's -- a silently wrong evaluation rather than a crash.
* **Leaves our cwd alone.** ``checkpoints/<model_id>`` is looked up relative to the process cwd,
  and we must stay in the repo root for ``./task_config/...``. A ``checkpoints`` symlink in the
  repo root serves the lookup (see README.md).
* **Makes inference latency visible to the async protocol.** ``script/eval_policy.py`` charges
  async mode by wrapping ``model.get_action`` in ``TASK_ENV.inference_window()``. FastWAM infers
  inside ``step()``, which the harness cannot see, so without the re-export below every async run
  would be scored as though inference were free -- i.e. identical to frozen, and quietly so.

Note the package is ``fwam``, not ``fastwam``: upstream's library *is* ``fastwam``, and
``script/eval_policy.py`` puts ``./policy`` on ``sys.path``, so a package of that name here would
shadow (or be shadowed by) the real one depending on path order.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

FASTWAM_ROOT = Path(os.environ.get(
    "FASTWAM_ROOT", "./third_party/fastwam"))
DEFAULT_RUN = "runs/FastWAM_RoboDynaV2_gpu-h200-103_20260825_144659"
# The hydra task group the checkpoint was trained under (see the run's config.yaml).
SIM_TASK = "robodyna_uncond_3cam_384_1e-4"

_upstream = None

# --- training-time language conditioning -------------------------------------------------------
#
# FastWAM conditions on a cached T5 embedding of ``DEFAULT_PROMPT.format(task=<instruction>)``, so
# the exact instruction string is part of the model's input distribution, not cosmetic metadata.
#
# The benchmark's wording moved after this checkpoint was trained: commit ``a73da892`` replaced the
# long scene-describing paragraphs with the short canonical lines now in
# ``task_config/task_instructions.json``, and the published dataset followed on 2026-08-25T17:19:46
# ("Align language prompts with ..."). This run started 2026-08-25 14:46:59 -- before that -- and
# the evidence is exact rather than circumstantial: hashing the *pre*-realignment prompts the way
# ``robot_video_dataset._get_cached_text_context`` does reproduces **all 35** filenames in the run's
# ``data/text_embeds_cache/robodyna``, while the current wording reproduces none of them.
#
# So feeding ``instruction_for(task)`` would condition the model on a string it never saw. The map
# below restores the training-time wording; set ``FASTWAM_TRAIN_PROMPT=0`` to evaluate on the
# current benchmark wording instead (the honest measurement of prompt sensitivity, not the
# measurement of the policy). Regenerate with ``tools/fwam_train_prompts.py``.
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
        # Unknown wording is reported once per string: it means the benchmark text moved again and
        # this map is stale, which would otherwise show up only as an unexplained drop in score.
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
    """Re-export chunk inference as ``get_action`` so the async protocol can charge it.

    ``_wrap_inference_latency`` rebinds ``model.get_action`` as an *instance* attribute, so the
    override below resolves to the timed wrapper once the harness has installed it and to the
    plain method before that. Calling the base class explicitly is what keeps it from recursing.
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
    """Is the policy actually commanding motion, or echoing the current pose?"""
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
    """Split deployment: the model lives on another GPU (often another node). See fwam_server.py.

    Selected by ``FASTWAM_SERVER=host:port`` rather than by config, because whether the model is
    co-located is a property of the machine the sweep lands on, not of the checkpoint.
    """
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
