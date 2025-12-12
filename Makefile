SHELL := /bin/bash

.PHONY: install uninstall update envs

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