# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

with open(te_path + "/../VERSION", "r") as f:
    te_version = f.readline()

release_year = 2022

current_year = date.today().year
if current_year == release_year:
    copyright_year = release_year
else:
    copyright_year = str(release_year) + "-" + str(current_year)

project = u'Transformer Engine'
copyright = u'{}, NVIDIA CORPORATION & AFFILIATES. All rights reserved.'.format(copyright_year)
author = u'NVIDIA CORPORATION & AFFILIATES'
version = te_version
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.mathjax',
        'sphinx.ext.napoleon',
        'nbsphinx',
        'breathe']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = '.rst'

master_doc = 'index'

pygments_style = 'sphinx'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

html_theme_options = {
        'display_version': True,
        'collapse_navigation': False,
        'logo_only': False
}

napoleon_custom_sections = [('Parallelism parameters', 'params_style'),
                            ('Optimization parameters', 'params_style'),
                            ('Values', 'params_style')]

breathe_projects = {"TransformerEngine": os.path.abspath("doxygen/xml/")}
breathe_default_project = "TransformerEngine"
