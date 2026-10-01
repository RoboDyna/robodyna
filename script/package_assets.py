#!/usr/bin/env python3
"""Build the minimal RoboDyna runtime-asset archive for Hugging Face.

Run this only from a checkout that already contains the required source assets:

    python script/package_assets.py --output /tmp/robodyna-assets-v1.tar.gz

The source defaults to this checkout's ``assets`` directory. The manifest is
version-controlled, so the archive can be recreated and audited without
pulling the full RoboTwin asset release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MANIFEST_PATH = ASSETS / "asset_manifest.json"
NOTICE_PATH = ASSETS / "ASSET_NOTICE.md"


def _copy_entry(source_root: Path, stage_root: Path, relative: str) -> None:
    source = source_root / relative
    destination = stage_root / relative
    if not source.exists():
        raise FileNotFoundError(f"Required asset is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        shutil.copy2(source, destination, follow_symlinks=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(source_root: Path, output: Path) -> tuple[int, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="robodyna-assets-") as temp:
        stage = Path(temp) / "runtime"
        stage.mkdir()
        for object_dir in manifest["object_directories"]:
            _copy_entry(source_root, stage, f"objects/{object_dir}")
        for directory in manifest["custom_asset_directories"]:
            _copy_entry(source_root, stage, directory)
        for texture in manifest["texture_files"]:
            _copy_entry(source_root, stage, texture)
        for entry in manifest["embodiment_entries"]:
            _copy_entry(source_root, stage, entry)
        shutil.copy2(MANIFEST_PATH, stage / MANIFEST_PATH.name)
        shutil.copy2(NOTICE_PATH, stage / NOTICE_PATH.name)

        with tarfile.open(output, "w:gz", compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                archive.add(path, arcname=path.relative_to(stage), recursive=False)

    return output.stat().st_size, _sha256(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ASSETS,
        help="asset root to package (default: this checkout's assets directory)",
    )
    parser.add_argument("--output", type=Path, required=True, help="archive destination")
    args = parser.parse_args()

    size, digest = build_archive(args.source.resolve(), args.output)
    print(f"Created {args.output.resolve()} ({size / 2**20:.1f} MiB)")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
