SHELL := /bin/bash

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