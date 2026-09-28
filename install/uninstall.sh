#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

echo "🧹 Uninstalling IMC Analysis..."

# Remove PATH entry
remove_matching "# SBT shell helpers" "$HOME/.profile"
remove_matching "# SBT shell helpers" "$HOME/.bashrc"
remove_matching "imcanalysis/Bash_scripts" "$HOME/.profile"

# Remove alias
remove_matching "alias cds=" "$HOME/.bashrc"

# Remove config sourcing
remove_matching ".imc_config" "$HOME/.profile"
remove_matching ".imc_config" "$HOME/.bashrc"

echo "✔ Cleaned PATH, aliases, and config sourcing."

# Optional: remove config file (ask user)
if [[ -f "$HOME/.imc_config" ]]; then
    read -p "Delete ~/.imc_config? [y/N]: " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        rm "$HOME/.imc_config"
        echo "✔ Deleted ~/.imc_config"
    else
        echo "✔ Preserved ~/.imc_config"
    fi
fi

echo "🎉 Uninstallation complete."
