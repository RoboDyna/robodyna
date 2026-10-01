"""Serve the RoboDyna pi05 checkpoint over openpi's websocket protocol.

Runs in policy/pi05/.venv (Python 3.11 + openpi + jax). The simulator stays in
the repo .venv, which has sapien 3.0.3 + curobo but no openpi -- and the eval
seeds were expert-validated under 3.0.3, so the sim must not move.

openpi's own scripts/serve_policy.py cannot be used directly: it calls
create_trained_policy() without robotwin_repo_id, and this config's asset_id is
only known from the checkpoint's assets dir (as policy/pi05/pi_model.py does),
so norm-stat loading would fail.
"""
import argparse
import os

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_robodyna")
    ap.add_argument("--dir", required=True, help="checkpoint dir (…/<step>)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    # Same asset-id discovery as pi_model.py: the checkpoint's assets dir holds
    # exactly the repo id whose norm stats were used in training.
    asset_id = os.listdir(os.path.join(args.dir, "assets"))[0]
    print(f"[serve_pi05] config={args.config} dir={args.dir} asset_id={asset_id}", flush=True)

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), args.dir, robotwin_repo_id=asset_id,
    )
    print("[serve_pi05] policy loaded, starting websocket server", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
