"""RoboDyna policy adapter for pi05 served remotely over websocket.

Same model and observation encoding as policy/pi05, but the policy runs in a
separate process (policy/pi05_ws/serve_pi05.py), so the simulator environment
does not need openpi. Like X-VLA, every inference call includes a
serialisation round trip, which async mode charges to simulation time.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def encode_obs(observation):
    # Same order as policy/pi05: head, right, left.
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


class RemotePI0:
    def __init__(self, host, port, pi0_step):
        self.client = WebsocketClientPolicy(host=host, port=int(port))
        self.pi0_step = int(pi0_step)
        self.instruction = None
        self.observation_window = None

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        front, right, left = img_arr[0], img_arr[1], img_arr[2]
        self.observation_window = {
            "state": np.asarray(state),
            "images": {
                "cam_high": np.transpose(front, (2, 0, 1)),
                "cam_left_wrist": np.transpose(left, (2, 0, 1)),
                "cam_right_wrist": np.transpose(right, (2, 0, 1)),
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return np.asarray(self.client.infer(self.observation_window)["actions"])

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None


def get_model(usr_args):
    return RemotePI0(
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 8765),
        pi0_step=usr_args.get("pi0_step", 50),
    )


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # script/eval_policy.py times model.get_action for the async protocol.
    actions = model.get_action()[: model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        if TASK_ENV.eval_success:
            return
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
