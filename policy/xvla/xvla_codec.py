"""EE6D codec between RoboDyna's endpose convention and X-VLA's 20-D action slot.

RoboDyna / RoboTwin conventions, each verified against the data or the generator
rather than assumed -- all three are silent-failure modes if got wrong:

* Quaternions are **scalar-first (wxyz)**. The generator builds endpose with
  ``transforms3d.quaternions.mat2quat`` (wxyz), and the sim consumes ee actions
  through ``_trans_from_gripper_to_endlink`` -> ``t3d.quaternions.quat2mat``
  (wxyz) -> ``sapien.Pose`` (wxyz). Confirmed independently against the
  dataset's own recorded wrist-camera extrinsics: R_cam^T @ R_endpose is
  constant to 0.013 deg under wxyz vs 4.64 deg under xyzw.
  NOTE: X-VLA's official RoboTwin client uses scipy's scalar-last default, which
  does not match this simulator. Do not copy it.
* rot6d is the first two columns of the rotation matrix flattened row-major,
  i.e. **interleaved**: c1 = v[0::2], c2 = v[1::2].
* Gripper is **1 = open** in the sim (``is_gripper_open: val > 0.8``) while
  X-VLA's EE6D space uses 1 = closed, so it is inverted in both directions.

scipy on the sim box is 1.10.1, which has no ``scalar_first`` kwarg, so wxyz
<-> xyzw is done explicitly with fancy indexing. Do not "simplify" this to
``scalar_first=True``: it raises on the evaluation host.

Layouts
-------
RoboDyna endpose (16):  [l_xyz(3), l_quat wxyz(4), l_grip(1), r_xyz(3), r_quat(4), r_grip(1)]
X-VLA EE6D      (20):  [l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]
Sim ee action   (16):  same layout as endpose
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ["endpose_to_ee6d", "ee6d_to_sim_action", "quat_wxyz_to_rot6d", "rot6d_to_quat_wxyz"]

_WXYZ_TO_XYZW = [1, 2, 3, 0]
_XYZW_TO_WXYZ = [3, 0, 1, 2]


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion [...,4] -> X-VLA's interleaved rot6d [...,6]."""
    q = np.asarray(q, dtype=np.float64)
    m = R.from_quat(q[..., _WXYZ_TO_XYZW].reshape(-1, 4)).as_matrix()
    return m[:, :, :2].reshape(q.shape[:-1] + (6,))


def rot6d_to_quat_wxyz(v6: np.ndarray) -> np.ndarray:
    """Interleaved rot6d [...,6] -> wxyz quaternion [...,4], via Gram-Schmidt.

    v6 is [c1x, c2x, c1y, c2y, c1z, c2z]; de-interleave with [0::2] / [1::2].
    Slicing it as [0:3] / [3:6] silently yields a different rotation.
    """
    v6 = np.asarray(v6, dtype=np.float64)
    if v6.shape[-1] != 6:
        raise ValueError(f"expected last dim 6, got {v6.shape[-1]}")
    a1, a2 = v6[..., 0::2], v6[..., 1::2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    m = np.stack([b1, b2, b3], axis=-1).reshape(-1, 3, 3)
    q = R.from_matrix(m).as_quat()[:, _XYZW_TO_WXYZ]
    q = np.where(q[:, :1] < 0, -q, q)      # fix the double cover for readability
    return q.reshape(v6.shape[:-1] + (4,))


def endpose_to_ee6d(endpose: np.ndarray) -> np.ndarray:
    """RoboDyna endpose [...,16] -> X-VLA proprio [...,20]. Gripper -> 1=closed."""
    e = np.asarray(endpose, dtype=np.float64)
    arms = []
    for base in (0, 8):
        arms.append(np.concatenate([
            e[..., base:base + 3],
            quat_wxyz_to_rot6d(e[..., base + 3:base + 7]),
            1.0 - e[..., base + 7:base + 8],
        ], axis=-1))
    return np.concatenate(arms, axis=-1)


def ee6d_to_sim_action(action20: np.ndarray) -> np.ndarray:
    """X-VLA action [...,20] -> sim ee action [...,16].

    Gripper channels arrive already sigmoid'd by the model's postprocess
    (1 = closed); the sim wants the normalised value where 1 = open and
    interpolates it from get_gripper_val(), so invert and clip into [0, 1].
    """
    a = np.asarray(action20, dtype=np.float64)
    out = []
    for base in (0, 10):
        out.append(np.concatenate([
            a[..., base:base + 3],
            rot6d_to_quat_wxyz(a[..., base + 3:base + 9]),
            np.clip(1.0 - a[..., base + 9:base + 10], 0.0, 1.0),
        ], axis=-1))
    return np.concatenate(out, axis=-1)
