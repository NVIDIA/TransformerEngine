# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import sphinx_rtd_theme
from sphinx.ext.autodoc.mock import mock
from sphinx.ext.autodoc import between, ClassDocumenter, AttributeDocumenter
from sphinx.util import inspect
from builtins import str
from enum import Enum
import re
import subprocess
from pathlib import Path
from datetime import date

te_path = os.path.dirname(os.path.realpath(__file__))

with open(te_path + "/../build_tools/VERSION.txt", "r") as f:
    te_version = f.readline().strip()

release_year = 2022

current_year = date.today().year
if current_year == release_year:
    copyright_year = release_year
else:
    copyright_year = str(release_year) + "-" + str(current_year)

project = "Transformer Engine"
copyright = "{}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.".format(copyright_year)
author = "NVIDIA CORPORATION & AFFILIATES"

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

version = str(te_version + "-" + git_sha)
release = te_version

# hack: version is used for html creation, so put the version picker
# link here as well:
option_on = " selected"
option_off = ""
release_opt = option_on
option_nr = 0
version = (
    version
    + """<br/>
Version select: <select onChange="window.location.href = this.value" onFocus="this.selectedIndex = {0}">
    <option value="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html"{1}>Current release</option>
    <option value="https://docs.nvidia.com/deeplearning/transformer-engine/documentation-archive.html">Older releases</option>
</select>""".format(
        option_nr, release_opt
    )
)

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
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]
html_show_sphinx = False

html_css_files = [
    "css/nvidia_font.css",
    "css/nvidia_footer.css",
]

html_theme_options = {"display_version": True, "collapse_navigation": False, "logo_only": False}

napoleon_custom_sections = [
    ("Parallelism parameters", "params_style"),
    ("Optimization parameters", "params_style"),
    ("Values", "params_style"),
]

breathe_projects = {"TransformerEngine": os.path.abspath("doxygen/xml/")}
breathe_default_project = "TransformerEngine"

autoapi_generate_api_docs = False
autoapi_dirs = ["../transformer_engine"]
