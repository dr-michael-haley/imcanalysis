SHELL := /bin/bash

PYTHON ?= python

.PHONY: install uninstall update envs docs-generate docs-check docs-html

install:
	@echo "🚀 Running installer..."
	@bash install/setup.sh

uninstall:
	@echo "🧹 Running uninstaller..."
	@bash install/uninstall.sh

update:
	@echo "🔄 Updating repository..."
	@git pull
	@bash install/setup.sh
	
envs:
	@echo "🐍 Setting up Conda environments..."
	@bash install/setup_envs.sh

docs-generate:
	@$(PYTHON) docs/tools/generate_docs.py

docs-check:
	@$(PYTHON) docs/tools/generate_docs.py --check

docs-html:
	@$(MAKE) -C docs html PYTHON=$(PYTHON)
