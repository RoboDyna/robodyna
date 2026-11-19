import sys
import os
import subprocess
import threading
import faulthandler
import time

sys.path.append("./")
sys.path.append(f"./policy")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from task_config.task_instructions import instruction_for

import numpy as np
from pathlib import Path
from collections import deque
import traceback
import json

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from eval_metrics import (
    EvalMetricsTracker,
    AggregatedMetrics,
    EpisodeMetrics,
    PenaltyEvent,
    env_uses_dynamic,
)
from evaluation_protocol import configure_evaluation

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

EVAL_MODES = ("frozen", "async")

# Attempts a fixed seed gets to pass the scripted expert before the run gives up on it. The expert
# is stochastic on the dynamic scenarios (about 20% of attempts fail), so a single attempt per seed
# would let one unlucky roll abort the whole 10-episode run.
EXPERT_TRIES = int(os.environ.get("EXPERT_TRIES", "3"))
# Whether a fixed seed that the expert does not solve aborts the run. Every seed in
# task_config/eval_seeds.yml was validated with three consecutive expert solves, so a failure here
# is expert noise rather than evidence that the seed is unsolvable. With the gate off the seed is
# evaluated anyway.
#
# The expert pass still runs either way: on a dynamic task play_once() produces
# ``_saved_dynamic_motion_info`` (the moving object's start pose, trajectory and duration), which
# the policy episode replays through init_dynamic_motion_for_eval(), and that call raises if it is
# missing. With the gate off only the verdict is discarded.
#
# With the gate off the expert is not re-run to obtain a solve, since the verdict is not used. The
# EXPERT_TRIES budget then only covers a pass that crashed or recorded no trajectory, which leaves
# the policy episode nothing to run on.
#
# The gate affects which scene a flaky seed is evaluated on: gated, a seed that solves on the second
# attempt uses attempt 2's trajectory; ungated, it uses attempt 1's. Seeds that solve on the first
# attempt, which is nearly all of them, are unaffected.
EXPERT_GATE = os.environ.get("EVAL_EXPERT_GATE", "1").lower() not in ("0", "false", "no", "off")

# Seconds without progress before the run is declared stalled. A rollout can occasionally hang
# inside a native call (0% CPU, never returns), typically at an action-chunk boundary around the
# inference call and the observation readback. The watchdog dumps every thread's stack and exits
# so the run can be retried. The threshold is long because the first inference of a run includes
# JAX tracing and compilation, which takes minutes, and in frozen mode the world does not step
# meanwhile. Set EVAL_STALL_S=0 to disable the watchdog.
EVAL_STALL_S = float(os.environ.get("EVAL_STALL_S", "600"))
# Per-seed verdicts are journalled here so that an interrupted run resumes from the last graded
# episode instead of starting over.
EPISODE_CACHE_ROOT = os.environ.get("EVAL_EPISODE_CACHE", os.path.join("run_results", "eval_episodes"))


def _start_stall_watchdog(get_signal, stall_s=EVAL_STALL_S):
    """Exit the process once ``get_signal()`` has not changed for ``stall_s``, after dumping stacks.

    The signal has to advance in every phase of a run, so it is taken at the scene: ``sim_step_cnt``
    counts physics steps in the expert validation pass, the kinematic loops and the policy rollout.
    Frozen mode does not step the world while the policy thinks, hence the generous threshold; this
    is a stall detector, not a latency budget.

    ``get_signal`` returns ``started=False`` until a scene exists, so the minutes spent loading the
    checkpoint, during which no counter moves, are not mistaken for a stall.
    """
    if stall_s <= 0:
        return
    faulthandler.enable()

    def _watch():
        last, last_change = object(), time.monotonic()
        while True:
            time.sleep(15.0)
            try:
                started, signal = get_signal()
            except Exception:  # the env can be mid-teardown; that is not a stall
                started, signal = False, last
            now = time.monotonic()
            if signal != last:
                last, last_change = signal, now
                continue
            if not started:
                # Nothing is expected to move yet, so the clock has not started either.
                last_change = now
                continue
            if now - last_change < stall_s:
                continue
            sys.stderr.write(
                f"\n!! stall watchdog: no progress for {stall_s:.0f}s at {signal}; thread dump:\n"
            )
            sys.stderr.flush()
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            sys.stderr.flush()
            # os._exit, not sys.exit: the stalled thread is inside a native call and will not
            # honour an exception, and atexit handlers would touch the same stalled renderer.
            os._exit(97)

    threading.Thread(target=_watch, name="stall-watchdog", daemon=True).start()


def _episode_cache_path(task_name, scenario, mode, ckpt_setting, checkpoint_id):
    os.makedirs(EPISODE_CACHE_ROOT, exist_ok=True)
    scenario = (scenario or "default").replace("+", "_")
    tag = f"{task_name}__{scenario}__{mode}__{ckpt_setting}__{checkpoint_id}"
    return os.path.join(EPISODE_CACHE_ROOT, tag + ".jsonl")


def _episode_from_dict(record):
    """Rebuild an EpisodeMetrics from the dict AggregatedMetrics.get_all_episodes() emits."""
    return EpisodeMetrics(
        success=bool(record.get("success")),
        steps_taken=int(record.get("steps_taken") or 0),
        max_steps=int(record.get("max_steps") or 0),
        seed=record.get("seed"),
        route_completion=float(record.get("route_completion") or 0.0),
        manipulation_score=float(record.get("manipulation_score") or 0.0),
        partial_score=record.get("partial_score"),
        penalty_events=[
            PenaltyEvent(
                event_type=p.get("event_type", ""),
                timestep=int(p.get("timestep") or 0),
                penalty_factor=float(p.get("penalty_factor") or 1.0),
                details=p.get("details", ""),
            )
            for p in (record.get("penalty_events") or [])
        ],
        total_penalty_factor=float(record.get("total_penalty_factor") or 1.0),
        fail_reason=record.get("fail_reason"),
        metric_detail=record.get("metric_detail") or {},
    )


def _load_episode_cache(path, seeds):
    """Verdicts already journalled for seeds that are still in the protocol, newest wins.

    A seed absent from ``seeds`` is dropped rather than replayed: re-banking a stale seed changes
    which scene the protocol asks for, and a verdict for the seed it replaced is not a result for it.
    """
    cached = {}
    if not path or not os.path.exists(path):
        return cached
    wanted = {int(s) for s in seeds}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A SIGKILL mid-write leaves a torn final line; the completed ones above it are fine.
                continue
            seed = record.get("seed")
            if seed is None or int(seed) not in wanted:
                continue
            cached[int(seed)] = record
    return cached


def _append_episode_cache(path, record):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wrap_inference_latency(TASK_ENV, model):
    """Route the policy's inference call through the env's inference window.

    ``deploy_policy.eval()`` calls ``model.get_action()`` synchronously and nothing steps the
    scene until it returns, so in frozen mode every dynamic task gives the policy a free pause:
    the ball stops mid-flight, the belt stops, the timer stops. Async mode charges that pause to
    simulation time. Wrapping ``get_action`` here, instead of editing each policy's
    ``deploy_policy.py``, gives every policy in policy/ both modes. A deploy_policy must therefore
    not open its own inference window, or async mode would charge the latency twice.
    """
    raw = model.get_action

    def timed(*a, **kw):
        with TASK_ENV.inference_window():
            return raw(*a, **kw)

    model.get_action = timed


def _inference_protocol_line(TASK_ENV) -> str:
    """One self-describing line for _result.txt: how this run treated inference latency."""
    s = TASK_ENV.async_inference_stats("run")
    if s["mode"] == "frozen":
        return "Inference mode: frozen (world held still during inference)\n"
    return (f"Inference mode: async, {s['calls']} calls over {s['episodes']} episode(s), "
            f"{s['mean_charged_ms']:.0f} ms avg charged, {s['steps']} sim steps "
            f"({s['sim_seconds_advanced']:.2f} s) advanced\n")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = "canonical"
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    # demo_dynamic.yml's eval_step_limit is a runaway guard sized in saved frames for collection.
    # In eval the same number counts policy actions, about 40x the longest demonstration, so the
    # per-task cap from task_config/eval_step_limits.yml is passed on the command line instead.
    if usr_args.get("eval_step_limit"):
        args["eval_step_limit"] = int(usr_args["eval_step_limit"])
    # Evaluation mode. "frozen" (the default) is the synchronous protocol, which the async code
    # path does not touch; "async" charges the policy's inference latency to simulation time.
    mode = str(usr_args.get("eval_mode_name", "frozen")).strip().lower()
    if mode not in EVAL_MODES:
        raise SystemExit(f"unknown eval mode {mode!r} (expected one of {EVAL_MODES})")
    args["async_inference"] = mode == "async"
    args["async_cap_s"] = float(usr_args.get("async_cap_s", 1.0))
    args["async_fixed_s"] = (None if usr_args.get("async_fixed_s") in (None, "", "none")
                             else float(usr_args["async_fixed_s"]))
    suite, scenario, evaluation_seeds = configure_evaluation(
        task_name, args, usr_args.get("scenario")
    )
    # Diagnostic only: evaluate on seeds the checkpoint was trained on (data/<task>/<cfg>/seed.txt)
    # to separate a wrong inference config from a policy that does not generalise. Benchmark runs
    # always use task_config/eval_seeds.yml.
    if os.environ.get("EVAL_SEEDS_OVERRIDE"):
        evaluation_seeds = [int(s) for s in os.environ["EVAL_SEEDS_OVERRIDE"].split(",")]
        suite = f"{suite}[seed-override]"
        print(f"\033[93m[diagnostic] EVAL_SEEDS_OVERRIDE active: {evaluation_seeds}\033[0m")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # The run directory names the scenario and mode as well as the timestamp, so frozen and async
    # runs started in the same second do not share a directory and overwrite each other's
    # _metrics.json. ``or`` handles a scenario key that is present but None.
    _sc = usr_args.get("scenario") or "default"
    _md = usr_args.get("eval_mode_name") or "frozen"
    run_tag = f"{current_time}__{_sc}__{_md}"
    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{run_tag}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # EVAL_VIDEO_LOG=0 disables the eval-only render + ffmpeg video pipe. Scoring does not read
    # anything produced by it (the policy's observations come from get_obs), so this does not change
    # the metrics. It helps on machines where the extra render occasionally hangs a rollout.
    _video_env = os.environ.get("EVAL_VIDEO_LOG", "1").lower() not in ("0", "false", "no", "off")
    if args["eval_video_log"] and _video_env:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[96mUse Dynamic:\033[0m inferred from task env")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    # Armed before the model loads so a stall in the first episode is still caught. Until a scene
    # exists it reports not-started, and the checkpoint load is not counted as a stall.
    _start_stall_watchdog(lambda: (
        getattr(TASK_ENV, "sim_step_cnt", None) is not None,
        (
            getattr(TASK_ENV, "test_num", -1),
            getattr(TASK_ENV, "take_action_cnt", -1),
            getattr(TASK_ENV, "sim_step_cnt", -1),
        ),
    ))
    episode_cache = _episode_cache_path(
        task_name, _sc, _md, ckpt_setting, usr_args.get("checkpoint_id")
    )
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    st_seed = evaluation_seeds[0]
    suc_nums = []
    test_num = len(evaluation_seeds)
    topk = 1

    print(
        f"Fixed evaluation protocol: suite={suite}, scenario={scenario or 'default'}, "
        f"seeds={evaluation_seeds}"
    )

    # Snapshot before the run so the timing summary can say which episodes it measured.
    _replayed_seeds = set(_load_episode_cache(episode_cache, evaluation_seeds))
    if _replayed_seeds:
        print(f"Episode journal: {sorted(_replayed_seeds)} will be replayed, not re-simulated")

    model = get_model(usr_args)
    st_seed, suc_num, aggregated_metrics = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        evaluation_seeds=evaluation_seeds,
        episode_cache=episode_cache,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    # Save basic result
    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Evaluation suite: {suite}\n")
        file.write(f"Scenario: {scenario or 'default'}\n")
        file.write(f"Seeds: {','.join(map(str, evaluation_seeds))}\n")
        file.write(_inference_protocol_line(TASK_ENV) + "\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    # Save detailed metrics report
    metrics_report_path = os.path.join(save_dir, f"_metrics_report.txt")
    with open(metrics_report_path, "w") as file:
        file.write(aggregated_metrics.to_detailed_report())
    
    # Save metrics as JSON for programmatic access
    metrics_json_path = os.path.join(save_dir, f"_metrics.json")
    with open(metrics_json_path, "w") as file:
        summary = aggregated_metrics.get_summary()
        # Convert nan to null for JSON compatibility
        for k, v in summary.items():
            if isinstance(v, float) and np.isnan(v):
                summary[k] = None
        summary["evaluation_protocol"] = {
            "suite": suite,
            "scenario": scenario or "default",
            "seeds": evaluation_seeds,
            # Scores are only comparable between runs with the same inference mode, so it is
            # recorded next to the seeds.
            "inference_mode": "async" if args.get("async_inference") else "frozen",
            "inference_cap_s": args.get("async_cap_s", 1.0),
        }
        # Run-level totals: the counters live on the env and survive the per-episode re-arm.
        summary["inference_timing"] = TASK_ENV.async_inference_stats("run")
        # Success rates survive a resume because every verdict is journalled; these timing
        # counters only cover the episodes this process simulated, so both counts are recorded.
        summary["inference_timing"]["episodes_measured_this_process"] = len(
            [e for e in aggregated_metrics.get_all_episodes() if e["seed"] not in _replayed_seeds]
        )
        summary["inference_timing"]["episodes_replayed_from_cache"] = len(_replayed_seeds)
        json.dump(summary, file, indent=2)
    
    # Save per-episode detailed results
    episodes_json_path = os.path.join(save_dir, f"_episodes_detail.json")
    with open(episodes_json_path, "w") as file:
        episodes_data = aggregated_metrics.get_all_episodes()
        json.dump(episodes_data, file, indent=2)

    print(f"Data has been saved to {file_path}")
    print(f"Metrics report saved to {metrics_report_path}")
    print(f"Per-episode details saved to {episodes_json_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                evaluation_seeds=None,
                episode_cache=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    # Seeds the expert did not solve that were evaluated anyway because the gate is off. They are
    # reported separately from the validated seeds.
    ungated_seeds = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    fixed_seeds = [int(seed) for seed in (evaluation_seeds or [])]
    fixed_seed_index = 0
    if fixed_seeds:
        test_num = len(fixed_seeds)
    now_seed = fixed_seeds[0] if fixed_seeds else st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    
    aggregated_metrics = AggregatedMetrics()

    _wrap_inference_latency(TASK_ENV, model)

    cached_episodes = _load_episode_cache(episode_cache, fixed_seeds) if fixed_seeds else {}
    if cached_episodes:
        print(
            f"\033[96mResuming: {len(cached_episodes)}/{len(fixed_seeds)} seeds already graded in "
            f"{episode_cache}\033[0m"
        )

    while succ_seed < test_num:
        if fixed_seeds:
            if fixed_seed_index >= len(fixed_seeds):
                break
            now_seed = fixed_seeds[fixed_seed_index]
            fixed_seed_index += 1

        if now_seed in cached_episodes:
            # Replayed from the journal instead of re-simulated.
            replayed = _episode_from_dict(cached_episodes[now_seed])
            aggregated_metrics.add_episode(replayed)
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
            now_id += 1
            TASK_ENV.test_num += 1
            if replayed.success:
                TASK_ENV.suc += 1
            verdict = "\033[92mSuccess\033[0m" if replayed.success else "\033[91mFail\033[0m"
            print(f"seed {now_seed}: {verdict} (cached, skipped)")
            continue

        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            # Gated: the scripted expert is not deterministic on the dynamic scenarios, so each
            # seed gets a few attempts and is only rejected once all of them fail.
            # Ungated: stop at the first attempt that leaves a replayable scene. The verdict is
            # discarded, so the remaining attempts only cover a pass that crashed or recorded no
            # trajectory.
            attempts = EXPERT_TRIES if fixed_seeds else 1
            gated = EXPERT_GATE or not fixed_seeds
            expert_ok, last_error = False, None

            def scene_replayable() -> bool:
                """Whether the attempt left a scene the policy episode can run on.

                A dynamic task replays the moving object's trajectory recorded by the expert's
                play_once(), and init_dynamic_motion_for_eval() raises if it is missing, so an
                expert pass that did not crash is not necessarily usable.
                """
                return (not getattr(TASK_ENV, "use_dynamic", False)
                        or getattr(TASK_ENV, "_saved_dynamic_motion_info", None) is not None)

            # Whether the most recent attempt ran to completion. An expert that built the scene but
            # did not solve it can still be evaluated ungated; a crashed expert leaves no
            # trajectory for the policy episode to replay.
            scene_ready = False
            used = 0
            for attempt in range(1, attempts + 1):
                used = attempt
                scene_ready = False
                try:
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                    episode_info = TASK_ENV.play_once()
                    # Grade before tearing the scene down, as collect_data.py does. close_env()
                    # drops TASK_ENV.scene, and a check_success() that reads live physics (e.g.
                    # catch_cuboid's gripper contacts) would otherwise fail on a valid seed.
                    solved = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
                    TASK_ENV.close_env()
                    scene_ready = True
                    if solved:
                        expert_ok = True
                        break
                    last_error = None
                except UnStableError as e:
                    TASK_ENV.close_env()
                    last_error = e
                except Exception as e:  # planner/setup failure on this attempt
                    TASK_ENV.close_env()
                    last_error = e
                if not gated and scene_ready and scene_replayable():
                    break  # usable scene in hand; the verdict is not being read, so do not retry
                if attempt < attempts:
                    print(f"\033[93m  expert check failed for seed {now_seed} "
                          f"(attempt {attempt}/{attempts}), retrying\033[0m")

            have_motion = scene_replayable()

            if not expert_ok and not gated and scene_ready and have_motion:
                # The expert built the scene but did not solve it. Evaluate the policy on it anyway
                # and record the seed, so the summary lists the episodes without an expert solve.
                ungated_seeds.append(now_seed)
                print(f"\033[93m  seed {now_seed}: expert did not solve it; verdict ignored "
                      f"(EVAL_EXPERT_GATE=0), evaluating the policy on this seed anyway\033[0m")
                expert_ok = True

            if not expert_ok:
                if fixed_seeds:
                    why = (f"{type(last_error).__name__}: {last_error}" if last_error
                           else "expert ran but did not solve the scene")
                    hint = ("" if gated else
                            " The expert produced no replayable scene on its last attempt"
                            f" (completed={scene_ready}, dynamic_motion={have_motion}), so there is"
                            " nothing for the policy to run on; EVAL_EXPERT_GATE=0 cannot rescue"
                            " this one.")
                    raise RuntimeError(
                        f"Fixed evaluation seed {now_seed} failed the expert check "
                        f"{used}x ({why}).{hint}"
                    ) from last_error
                now_seed += 1
                args["render_freq"] = render_freq
                continue

        succ_seed += 1
        suc_test_seed_list.append(now_seed)

        # Save dynamic motion info from Expert phase
        if os.environ.get("ROLLOUT_TRACE"):
            # The expert validates the seed in its own setup_demo and the policy then gets a second
            # setup_demo on the same seed. Print the expert's scene descriptor so it can be compared
            # with the policy episode's descriptor in _episodes_detail.json.
            print(f"[diagnostic] seed {now_seed} EXPERT scene info: {episode_info['info']}")
        saved_dynamic_motion_info = getattr(TASK_ENV, '_saved_dynamic_motion_info', None)

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        
        # Restore dynamic motion info
        if saved_dynamic_motion_info is not None:
            TASK_ENV._saved_dynamic_motion_info = saved_dynamic_motion_info
        
        instruction = instruction_for(args["task_name"])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        episode_uses_dynamic = env_uses_dynamic(TASK_ENV, args)

        # Initialize dynamic object motion for policy evaluation
        if episode_uses_dynamic:
            dynamic_init_success = TASK_ENV.init_dynamic_motion_for_eval()
            if not dynamic_init_success:
                print(f"Error: Failed to initialize dynamic motion for seed {now_seed}, skipping...")
                TASK_ENV.close_env()
                TASK_ENV.release_episode_resources()
                if fixed_seeds:
                    raise RuntimeError(
                        f"Fixed evaluation seed {now_seed} failed dynamic-motion initialization."
                    )
                now_seed += 1
                succ_seed -= 1
                continue

        # Check object lifting and contact-stop configurations
        check_z_threshold = None
        target_actor_check = None
        start_z = None
        has_lifted = False
        stop_on_contact = True

        dynamic_config = getattr(TASK_ENV, 'get_dynamic_motion_config', lambda: None)()
        if dynamic_config and 'check_z_threshold' in dynamic_config:
            check_z_threshold = dynamic_config['check_z_threshold']
            if 'check_z_actor' in dynamic_config:
                target_actor_check = dynamic_config['check_z_actor']
            else:
                target_actor_check = dynamic_config['target_actor']
            start_z = target_actor_check.get_pose().p[2]
        if dynamic_config and 'stop_on_contact' in dynamic_config:
            stop_on_contact = dynamic_config['stop_on_contact']

        # Initialize metrics tracker for this episode
        metrics_tracker = EvalMetricsTracker(TASK_ENV, args)
        metrics_tracker.on_episode_start()
        TASK_ENV._metrics_tracker = metrics_tracker

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        fail_reason = None
        # Diagnostic trace (ROLLOUT_TRACE=<path>): record what the policy commands and what the
        # robot reaches, to tell a policy that predicts "hold" from a motion that take_action drops
        # (TOPP returns no waypoints, so set_arm_joints is skipped).
        trace_path = os.environ.get("ROLLOUT_TRACE")
        if trace_path:
            _trace = []
            _raw_take_action = TASK_ENV.take_action
            _raw_scene_step = TASK_ENV.scene.step
            _sim_steps = [0]

            def _counting_step(*a, _f=_raw_scene_step, **kw):
                _sim_steps[0] += 1
                return _f(*a, **kw)

            TASK_ENV.scene.step = _counting_step

            def _traced_take_action(action, *a, _f=_raw_take_action, **kw):
                before = np.array(TASK_ENV.robot.get_left_arm_jointState() +
                                  TASK_ENV.robot.get_right_arm_jointState())
                n0 = _sim_steps[0]
                out = _f(action, *a, **kw)
                after = np.array(TASK_ENV.robot.get_left_arm_jointState() +
                                 TASK_ENV.robot.get_right_arm_jointState())
                # Collection saved one training frame every save_freq=15 sim steps, so sim_steps is
                # directly comparable to 15: a larger number means the world advances further per
                # policy action than it did between training frames.
                _trace.append({"step": int(TASK_ENV.take_action_cnt),
                               "sim_steps": _sim_steps[0] - n0,
                               "cmd": np.asarray(action, dtype=float).tolist(),
                               "qpos_before": before.tolist(), "qpos_after": after.tolist()})
                return out

            TASK_ENV.take_action = _traced_take_action
        # Re-armed per episode so the printed line is that episode's cost; run-level totals
        # accumulate inside the env.
        TASK_ENV.configure_async_inference(
            enabled=bool(args.get("async_inference", False)),
            cap_s=float(args.get("async_cap_s", 1.0)),
            fixed_s=args.get("async_fixed_s"),
        )
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)

            if check_z_threshold is not None and not has_lifted:
                curr_z = target_actor_check.get_pose().p[2]
                if curr_z - start_z > check_z_threshold:
                    has_lifted = True

            if TASK_ENV.eval_success:
                if check_z_threshold is not None and not has_lifted:
                    succ = False
                    fail_reason = "object_not_lifted"
                    break
                succ = True
                break
            
            # Check if dynamic object is out of bounds (lost from view)
            out_of_bounds = False
            if episode_uses_dynamic:
                # Stop motion if gripper contacts dynamic object
                if TASK_ENV.check_gripper_contact_dynamic_object() and stop_on_contact:
                    TASK_ENV.stop_dynamic_object_motion()
                
                # Only check boundaries if object hasn't been grasped
                if not getattr(TASK_ENV, '_dynamic_object_stopped', False):
                    if not TASK_ENV.check_dynamic_object_boundary():
                        TASK_ENV.eval_fail = True
                        fail_reason = "out_of_bounds"
                        out_of_bounds = True
            
            if out_of_bounds:
                metrics_tracker.record_out_of_bounds()
                break
            if TASK_ENV.eval_fail:
                fail_reason = "eval_fail"
                break
        # task_total_reward += TASK_ENV.episode_score
        if trace_path:
            TASK_ENV.take_action = _raw_take_action
            TASK_ENV.scene.step = _raw_scene_step
            with open(f"{trace_path}.seed{now_seed}.json", "w") as _tf:
                json.dump(_trace, _tf)
            print(f"[diagnostic] wrote {len(_trace)} traced actions to {trace_path}.seed{now_seed}.json")
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        elif fail_reason == "out_of_bounds":
            print("\033[91mFail! (Object out of bounds)\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Collect episode metrics. The harness itself only names three causes
        # ("object_not_lifted", "out_of_bounds", "eval_fail"); for a plain timeout
        # get_fail_reason returns the task's latched reason or one derived from the
        # partial-score rubric ("did not grasp the cuboid").
        if not succ and not fail_reason:
            try:
                fail_reason = TASK_ENV.get_fail_reason() or ""
            except Exception:
                fail_reason = ""
        episode_metrics = metrics_tracker.get_episode_metrics(succ, fail_reason, seed=now_seed)
        aggregated_metrics.add_episode(episode_metrics)
        # Journal the verdict before the next episode starts. The report serialiser is reused so
        # the cached record and _episodes_detail.json stay identical.
        _append_episode_cache(episode_cache, aggregated_metrics.get_all_episodes()[-1])
        TASK_ENV._metrics_tracker = None
        
        # Print episode metrics summary
        partial_score_text = (
            "N/A" if episode_metrics.partial_score is None
            else f"{episode_metrics.partial_score:.2f}"
        )
        print(f"  MS: \033[96m{episode_metrics.manipulation_score:.1f}\033[0m | "
              f"RC: \033[96m{episode_metrics.route_completion:.1f}%\033[0m | "
              f"PS: \033[96m{partial_score_text}\033[0m")
        ep_stats = TASK_ENV.async_inference_stats("episode")
        if ep_stats["mode"] == "async":
            n, charged, steps = ep_stats["calls"], ep_stats["charged_s"], ep_stats["steps"]
            if n:
                print(f"  Inference: \033[96m{n}\033[0m calls, "
                      f"\033[96m{ep_stats['mean_charged_ms']:.0f}\033[0m ms avg charged, "
                      f"world advanced \033[96m{charged:.2f}\033[0m s "
                      f"(\033[96m{steps}\033[0m sim steps)")
        if episode_metrics.penalty_events:
            penalty_str = ", ".join([f"{p.event_type}({p.penalty_factor})" for p in episode_metrics.penalty_events])
            print(f"  Penalties: \033[93m{penalty_str}\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((TASK_ENV.test_num + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.release_episode_resources()

        TASK_ENV.test_num += 1

        # Print running statistics
        summary = aggregated_metrics.get_summary()
        average_partial_score = summary["partial_score_mean"]
        average_partial_score_text = (
            "N/A" if average_partial_score is None else f"{average_partial_score:.3f}"
        )
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m | "
            # Avg MS is not printed: manipulation_score = route_completion x total_penalty_factor,
            # and with use_dynamic false (every eval config) route_completion is 100*success, so it
            # equals 100*SR. It is still written to _episodes_detail.json.
            f"Avg PS: \033[95m{average_partial_score_text}\033[0m | "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        if not fixed_seeds:
            now_seed += 1

    if ungated_seeds:
        print(f"\033[93mExpert gate off: {len(ungated_seeds)} of {TASK_ENV.test_num} episode(s) ran "
              f"on seeds the expert did not solve this run: {ungated_seeds}\033[0m")
    return now_seed, TASK_ENV.suc, aggregated_metrics


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
