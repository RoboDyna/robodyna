"""Thin HTTP client for the X-VLA inference server (``deploy.py`` / ``model.run``).

The server exposes POST /act and expects, per X-VLA's FastAPI app:
    image0/1/2            json_numpy-encoded HxWx3 uint8 (or 1-D encoded bytes)
    proprio               json_numpy-encoded [20] float
    language_instruction  str
    domain_id             int
and returns {"action": [[...20], ...]} already run through the action space's
postprocess (so gripper channels are sigmoid'd, 1 = closed).

json_numpy is not installed in the simulator venv and that venv is shared with
other users, so its wire format ({"__numpy__": b64, "dtype": descr, "shape"})
is reproduced here rather than adding a dependency to a shared environment.
"""

from __future__ import annotations
import json
from base64 import b64encode

import numpy as np
import requests
from numpy.lib.format import dtype_to_descr


def _encode(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return json.dumps({
        "__numpy__": b64encode(a.data).decode(),
        "dtype": dtype_to_descr(a.dtype),
        "shape": a.shape,
    })


class XVLAClient:
    def __init__(self, host="127.0.0.1", port=8010, domain_id=19, timeout=120.0):
        self.url = f"http://{host}:{port}/act"
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self.instruction = None

    def set_language(self, instruction: str):
        self.instruction = instruction

    def reset(self):
        self.instruction = None

    def act(self, images, proprio) -> np.ndarray:
        """images: 3 x HxWx3 uint8 in (head, left, right) order. -> [T, 20]."""
        payload = {
            "domain_id": self.domain_id,
            "language_instruction": self.instruction or "",
            "proprio": _encode(np.asarray(proprio, dtype=np.float32)),
        }
        for i, im in enumerate(images[:3]):
            payload[f"image{i}"] = _encode(np.ascontiguousarray(im, dtype=np.uint8))
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if "action" not in body:
            raise RuntimeError(f"X-VLA server returned no action: {str(body)[:200]}")
        return np.asarray(body["action"], dtype=np.float64)
