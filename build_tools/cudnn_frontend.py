# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Build helpers for the vendored cuDNN frontend Python package."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Optional
from urllib.parse import unquote, urlparse


PACKAGE_NAME = "nvidia-cudnn-frontend"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cudnn_frontend_root(repo_root: Optional[Path] = None) -> Path:
    """Path to the vendored cuDNN frontend submodule."""
    root = _repo_root() if repo_root is None else Path(repo_root)
    return root / "3rdparty" / "cudnn-frontend"


def cudnn_frontend_version(repo_root: Optional[Path] = None) -> str:
    """Read the vendored cuDNN frontend Python package version without importing it."""
    init_py = cudnn_frontend_root(repo_root) / "python" / "cudnn" / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(
            f"Could not find vendored cuDNN frontend at {init_py}. "
            "Run `git submodule update --init --recursive`."
        )
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py.read_text(), re.M)
    if match is None:
        raise RuntimeError(f"Could not parse cuDNN frontend version from {init_py}.")
    return match.group(1)


def installed_cudnn_frontend_version() -> Optional[str]:
    """Return installed nvidia-cudnn-frontend version, if present."""
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None


def _installed_cudnn_frontend_direct_url() -> Optional[str]:
    """Return the installed package direct URL metadata, if present."""
    try:
        dist = metadata.distribution(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return None
    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        return None
    try:
        return json.loads(direct_url).get("url")
    except json.JSONDecodeError:
        return None


def _direct_url_matches_path(url: Optional[str], path: Path) -> bool:
    if url is None:
        return False
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return False
    return Path(unquote(parsed.path)).resolve() == path.resolve()


def _pip_install_command(vendored_root: Path, *, no_build_isolation: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
    ]
    if no_build_isolation:
        command.append("--no-build-isolation")
    command.append(str(vendored_root))
    return command


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    path_string = str(path)
    if key in env and env[key]:
        existing_paths = env[key].split(os.pathsep)
        if path_string not in existing_paths:
            env[key] = os.pathsep.join([path_string, env[key]])
    else:
        env[key] = path_string


def _pybind11_cmake_dir() -> Optional[Path]:
    try:
        import pybind11  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None

    try:
        cmake_dir = Path(pybind11.get_cmake_dir())
    except AttributeError:
        return None
    if (cmake_dir / "pybind11Config.cmake").is_file() or (
        cmake_dir / "pybind11-config.cmake"
    ).is_file():
        return cmake_dir
    return None


def _pip_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PIP_CACHE_DIR", str(Path(tempfile.gettempdir()) / "te-pip-cache"))
    pybind11_cmake_dir = _pybind11_cmake_dir()
    if pybind11_cmake_dir is not None:
        env.setdefault("pybind11_DIR", str(pybind11_cmake_dir))
        pybind11_prefix = pybind11_cmake_dir
        if (
            pybind11_cmake_dir.parent.name == "cmake"
            and pybind11_cmake_dir.parent.parent.name == "share"
        ):
            pybind11_prefix = pybind11_cmake_dir.parent.parent.parent
        _prepend_env_path(env, "CMAKE_PREFIX_PATH", pybind11_prefix)
    return env


def install_from_submodule(
    repo_root: Optional[Path] = None,
    *,
    force: bool = False,
    no_build_isolation: bool = True,
) -> None:
    """Install nvidia-cudnn-frontend from the vendored submodule.

    The default uses ``--no-build-isolation`` so CI can use the already-installed
    build dependencies and avoid accidentally fetching a different build stack.
    """
    vendored_root = cudnn_frontend_root(repo_root)
    if not (vendored_root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Could not find vendored cuDNN frontend at {vendored_root}. "
            "Run `git submodule update --init --recursive`."
        )

    vendored_version = cudnn_frontend_version(repo_root)
    installed_version = installed_cudnn_frontend_version()
    installed_from_vendored = _direct_url_matches_path(
        _installed_cudnn_frontend_direct_url(), vendored_root
    )
    if installed_version == vendored_version and installed_from_vendored and not force:
        print(
            f"{PACKAGE_NAME}=={installed_version} is already installed from the vendored submodule."
        )
        return

    if installed_version is None:
        print(f"Installing {PACKAGE_NAME}=={vendored_version} from {vendored_root}.")
    elif installed_version == vendored_version:
        print(
            f"Reinstalling {PACKAGE_NAME}=={installed_version} from {vendored_root} "
            "to use Transformer Engine's vendored submodule."
        )
    else:
        print(
            f"Replacing {PACKAGE_NAME}=={installed_version} with vendored "
            f"{PACKAGE_NAME}=={vendored_version} from {vendored_root}."
        )
    subprocess.check_call(
        _pip_install_command(vendored_root, no_build_isolation=no_build_isolation),
        env=_pip_env(),
    )


def build_wheel(
    wheel_dir: Path,
    repo_root: Optional[Path] = None,
    *,
    no_build_isolation: bool = True,
) -> None:
    """Build a wheel for the vendored nvidia-cudnn-frontend package."""
    vendored_root = cudnn_frontend_root(repo_root)
    if not (vendored_root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Could not find vendored cuDNN frontend at {vendored_root}. "
            "Run `git submodule update --init --recursive`."
        )
    wheel_dir = Path(wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wheel_dir)]
    if no_build_isolation:
        command.append("--no-build-isolation")
    command.append(str(vendored_root))
    subprocess.check_call(command, env=_pip_env())


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the vendored submodule package version")

    install_parser = subparsers.add_parser("install", help="Install from the vendored submodule")
    install_parser.add_argument(
        "--force", action="store_true", help="Reinstall even if versions match"
    )
    install_parser.add_argument(
        "--build-isolation",
        action="store_true",
        help="Allow pip build isolation for the cuDNN frontend build",
    )

    wheel_parser = subparsers.add_parser("wheel", help="Build a wheel from the vendored submodule")
    wheel_parser.add_argument("wheel_dir", type=Path)
    wheel_parser.add_argument(
        "--build-isolation",
        action="store_true",
        help="Allow pip build isolation for the cuDNN frontend build",
    )

    args = parser.parse_args()
    if args.command == "version":
        print(cudnn_frontend_version())
    elif args.command == "install":
        install_from_submodule(
            force=args.force,
            no_build_isolation=not args.build_isolation,
        )
    elif args.command == "wheel":
        build_wheel(args.wheel_dir, no_build_isolation=not args.build_isolation)


if __name__ == "__main__":
    _main()
