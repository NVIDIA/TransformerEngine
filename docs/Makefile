# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile sphinx_rtd_theme
	PYTHONPATH=sphinx_rtd_theme:$(PYTHONPATH) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Patch Sphinx RTD theme to add version selector in sidebar
sphinx_rtd_theme:
	git clone --depth=1 -b 3.0.1 --single-branch https://github.com/readthedocs/sphinx_rtd_theme.git
	bash -c "cd sphinx_rtd_theme; git apply ../version_select.patch"
