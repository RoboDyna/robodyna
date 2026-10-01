#!/usr/bin/env bash
# Create / refresh the RoboDyna conda env and install simulation + GUI deps.
# Usage (from repo root):
#   bash script/install_robodyna.sh
# Optional:
#   ROBODYNA_ENV=myenv bash script/install_robodyna.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ROBODYNA_ENV:-robodyna}"
PYTHON_VERSION="${ROBODYNA_PYTHON:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Miniforge and retry." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "== System packages (Vulkan + FFmpeg) =="
echo "If missing, run:"
echo "  sudo apt update && sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg"
echo

echo "== Conda env: ${ENV_NAME} (Python ${PYTHON_VERSION}) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env already exists; activating."
else
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi
conda activate "$ENV_NAME"

echo "== Pip packages =="
python -m pip install --upgrade pip setuptools wheel
# Base RoboTwin / DOMINO stack (script/requirements.txt still pins sapien 3.0.0b1).
python -m pip install -r script/requirements.txt
# RoboDyna standardizes on SAPIEN 3.0.3 for collection and evaluation.
python -m pip install "sapien==3.0.3"
# GUI / config helpers used by interactive/*.py
python -m pip install "Pillow" "PyYAML"

echo "== pytorch3d (optional for core collection; skip failures are OK) =="
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git" || \
  echo "WARNING: pytorch3d install failed; continuing without it."

echo "== Patch sapien URDF loader (utf-8) =="
SAPIEN_LOCATION="$(python -c 'import sapien, pathlib; print(pathlib.Path(sapien.__file__).resolve().parent)')"
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
if [[ -f "$URDF_LOADER" ]]; then
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$URDF_LOADER"
fi

echo "== Patch mplib planner (drop collide short-circuit) =="
if python -c 'import mplib' >/dev/null 2>&1; then
  MPLIB_LOCATION="$(python -c 'import mplib, pathlib; print(pathlib.Path(mplib.__file__).resolve().parent)')"
  PLANNER="${MPLIB_LOCATION}/planner.py"
  if [[ -f "$PLANNER" ]]; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
  fi
else
  echo "mplib not installed; skipping planner patch."
fi

echo "== CuRobo =="
if [[ ! -d envs/curobo/.git ]]; then
  git clone --depth 1 https://github.com/NVlabs/curobo.git envs/curobo
fi
(
  cd envs/curobo
  python -m pip install -e . --no-build-isolation
)

echo
echo "Installation complete in conda env '${ENV_NAME}'."
echo "Next:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. bash script/_download_assets.sh   # RoboDyna minimal runtime assets (~1.2 GiB)"
echo "  3. export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json && unset DISPLAY"
echo
echo "aarch64 / GB10: prefer scripts/build_domino_aarch64.sh instead of this script."
