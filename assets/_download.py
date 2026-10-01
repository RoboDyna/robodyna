#!/usr/bin/env python3
"""Download and safely extract RoboDyna's minimal runtime-asset package."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download


ASSET_REPOSITORY = "RoboDyna/RoboDyna-assets"
ARCHIVE_NAME = "robodyna-assets-v1.tar.gz"
CHECKSUM_NAME = "robodyna-assets-v1.sha256"
ASSETS_DIR = Path(__file__).resolve().parent
REQUIRED_ENTRIES = (
    "objects",
    "dyna_assets",
    "dyna_textures",
    "embodiments/ur5-wsg/meshes",
)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract only regular archive members rooted inside ``destination``."""
    for member in archive.getmembers():
        member_path = PurePosixPath(member.name)
        if (
            member_path.is_absolute()
            or ".." in member_path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise RuntimeError(f"Unsafe path in asset archive: {member.name!r}")
    archive.extractall(destination)


def is_installed(destination: Path) -> bool:
    return all((destination / entry).exists() for entry in REQUIRED_ENTRIES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract the archive even when all required assets are present",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=ASSETS_DIR,
        help="asset root to populate (default: this checkout's assets directory)",
    )
    args = parser.parse_args()
    destination = args.destination.resolve()

    if is_installed(destination) and not args.force:
        print("RoboDyna runtime assets are already installed. Use --force to refresh them.")
        return

    archive_path = hf_hub_download(
        repo_id=ASSET_REPOSITORY,
        repo_type="dataset",
        filename=ARCHIVE_NAME,
    )
    checksum_path = hf_hub_download(
        repo_id=ASSET_REPOSITORY,
        repo_type="dataset",
        filename=CHECKSUM_NAME,
    )
    expected = Path(checksum_path).read_text(encoding="utf-8").split()[0]
    actual = _sha256(Path(archive_path))
    if actual != expected:
        raise RuntimeError(
            f"Asset archive checksum mismatch: expected {expected}, got {actual}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {ARCHIVE_NAME} into {destination}")
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract(archive, destination)
    print("RoboDyna runtime assets installed.")


if __name__ == "__main__":
    main()
