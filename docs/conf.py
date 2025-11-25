# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import datetime
import os
import pathlib
import subprocess
from builtins import str

# Basic project info
project = "Transformer Engine"
author = "NVIDIA CORPORATION & AFFILIATES"

# Copyright statement
release_year = 2022
current_year = datetime.date.today().year
if current_year == release_year:
    copyright_year = release_year
else:
    copyright_year = str(release_year) + "-" + str(current_year)
copyright = f"{copyright_year}, NVIDIA CORPORATION & AFFILIATES. All rights reserved."

# Transformer Engine root directory
root_path = pathlib.Path(__file__).resolve().parent.parent

# Git hash
git_sha = os.getenv("GIT_SHA")
if not git_sha:
    try:
        git_sha = (
            subprocess.check_output(["git", "log", "--pretty=format:'%h'", "-n1"])
            .decode("ascii")
            .replace("'", "")
            .strip()
        )
    except:
        git_sha = "0000000"
git_sha = git_sha[:7] if len(git_sha) > 7 else git_sha

# Version
with open(root_path / "build_tools" / "VERSION.txt", "r") as f:
    _raw_version = f.readline().strip()
if "dev" in _raw_version:
    version = str(_raw_version + "-" + git_sha)
else:
    version = str(_raw_version)
release = _raw_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "nbsphinx",
    "breathe",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"

master_doc = "index"

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sphinx = False

html_css_files = [
    "css/nvidia_font.css",
    "css/nvidia_footer.css",
]

html_theme_options = {
    "collapse_navigation": False,
    "logo_only": False,
    "version_selector": False,
    "language_selector": False,
}

napoleon_custom_sections = [
    ("Parallelism parameters", "params_style"),
    ("Optimization parameters", "params_style"),
    ("Values", "params_style"),
    ("Graphing parameters", "params_style"),
    ("FP8-related parameters", "params_style"),
]

breathe_projects = {"TransformerEngine": root_path / "docs" / "doxygen" / "xml"}
breathe_default_project = "TransformerEngine"

autoapi_generate_api_docs = False
autoapi_dirs = [root_path / "transformer_engine"]
autoapi_ignore = ["*/_[!_]*"]
