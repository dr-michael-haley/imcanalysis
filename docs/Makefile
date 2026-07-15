# Sphinx documentation helpers.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help generate check-generated html clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

generate:
	@$(PYTHON) tools/generate_docs.py

check-generated:
	@$(PYTHON) tools/generate_docs.py --check

html: generate
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile generate
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
