"""RoboDyna policy adapter for X-VLA (EE6D, absolute end-effector).

Interface mirrors policy/pi05/deploy_policy.py: encode_obs / get_model / eval /
reset_model, driven by script/eval_policy.py.

Differences from pi05 that matter:
  * pi05 predicts 14-D joint targets and drives take_action(..., 'qpos');
    X-VLA predicts a 20-D absolute EE6D chunk and drives take_action(..., 'ee').
  * domain_id is 19 -- the soft-prompt slot this checkpoint was trained under.
    (X-VLA's own RoboTwin client uses 6, which is the RoboTwin2 slot.)
  * Camera order is (head, left, right), matching how the model was trained
    (CAMERA_VIEW = head, left_wrist, right_wrist). pi05's encode_obs uses a
    different order for its own reasons; do not copy it here.

Quaternion / gripper conventions live in xvla_codec.py, which is round-trip
tested against real dataset frames. Do not reimplement them inline.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xvla_client import XVLAClient
from xvla_codec import endpose_to_ee6d, ee6d_to_sim_action


def encode_obs(observation):
    """-> (3 x HxWx3 uint8 in head/left/right order, 16-D RoboDyna endpose)."""
    obs = observation["observation"]
    images = [
        np.asarray(obs["head_camera"]["rgb"]),
        np.asarray(obs["left_camera"]["rgb"]),
        np.asarray(obs["right_camera"]["rgb"]),
    ]
    ep = observation["endpose"]
    endpose = np.concatenate([
        np.asarray(ep["left_endpose"], dtype=np.float64).reshape(7),
        np.asarray([ep["left_gripper"]], dtype=np.float64),
        np.asarray(ep["right_endpose"], dtype=np.float64).reshape(7),
        np.asarray([ep["right_gripper"]], dtype=np.float64),
    ])
    return images, endpose


def get_model(usr_args):
    return XVLAClient(
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 8010)),
        domain_id=int(usr_args.get("domain_id", 19)),
    )


def eval(TASK_ENV, model, observation):
    if model.instruction is None:
        model.set_language(TASK_ENV.get_instruction())

    images, endpose = encode_obs(observation)
    proprio = endpose_to_ee6d(endpose)

    # Mode 2 charges the wall-clock cost of this call to simulation time; in
    # mode 1 the context manager is a no-op and physics stays frozen.
    with TASK_ENV.inference_window():
        actions = model.act(images, proprio)

    for action in ee6d_to_sim_action(np.atleast_2d(actions)):
        TASK_ENV.take_action(action, action_type="ee")
        if TASK_ENV.eval_success:
            return


def reset_model(model):
    model.reset()
