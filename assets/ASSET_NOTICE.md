---
license: mit
---

# RoboDyna runtime assets

This package contains only the assets needed by the RoboDyna task suite.

## Upstream attribution

The selected stock object meshes and UR5–WSG embodiment meshes are derived
from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0),
which is distributed under the MIT License. Their original license and
attribution continue to apply to those files.

The `dyna_assets` and selected `dyna_textures` entries are RoboDyna additions.

## Scope

The package intentionally excludes unused RoboTwin objects, unused robot
embodiments, and random background textures. RoboDyna's checked-in task
configurations set `random_background: false`; enabling that option requires
supplying compatible background textures separately.
