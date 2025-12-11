#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

IMC_DIR="$HOME/imcanalysis"
BASH_DIR="$IMC_DIR/Bash_scripts"
SLURM_DIR="$IMC_DIR/SLURM_scripts"
CONFIG="$HOME/.imc_config"

echo "🔧 Installing IMC Analysis tools..."

###############################################
# 1. Validate repository
###############################################
if [[ ! -d "$IMC_DIR" ]]; then
    echo "❌ ERROR: $IMC_DIR does not exist."
    echo "Clone first:"
    echo "  git clone <repo> ~/imcanalysis"
    exit 1
fi

[[ -d "$BASH_DIR" ]] || { echo "❌ Missing $BASH_DIR"; exit 1; }
[[ -d "$SLURM_DIR" ]] || { echo "❌ Missing $SLURM_DIR"; exit 1; }

echo "✔ Repository validated."

###############################################
# 2. Ensure scripts executable
###############################################
chmod +x "$BASH_DIR"/* "$SLURM_DIR"/* 2>/dev/null
echo "✔ Script permissions updated."

###############################################
# 3. Add PATH entry to ~/.profile
###############################################
PATH_LINE="export PATH=\"$BASH_DIR:\$PATH\""
append_if_missing "$PATH_LINE" "$HOME/.profile"
echo "✔ Added Bash_scripts to PATH."

###############################################
# 4. Add cds alias to ~/.bashrc
###############################################
ALIAS_LINE='alias cds=". \"$HOME/imcanalysis/Bash_scripts/cds\""'
append_if_missing "$ALIAS_LINE" "$HOME/.bashrc"
echo "✔ Added alias 'cds'."

###############################################
# 5. Create ~/.imc_config if missing
###############################################
if [[ ! -f "$CONFIG" ]]; then
    echo "Creating configuration file: $CONFIG"
    read -p "Enter SLURM notification email: " email
    read -p "Enter OpenAI API key: " openai_key

    cat <<EOF > "$CONFIG"
export IMC_EMAIL="$email"
export OPENAI_API_KEY="$openai_key"
EOF

    chmod 600 "$CONFIG"
    echo "✔ Created ~/.imc_config (secure permissions)."
else
    echo "✔ Existing ~/.imc_config preserved."
fi

###############################################
# 6. Source ~/.imc_config in profile + bashrc
###############################################
SRC_LINE='[ -f "$HOME/.imc_config" ] && source "$HOME/.imc_config"'
append_if_missing "$SRC_LINE" "$HOME/.profile"
append_if_missing "$SRC_LINE" "$HOME/.bashrc"

echo "✔ Enabled automatic loading of ~/.imc_config."

###############################################
# Finish
###############################################
echo "🎉 IMC Analysis successfully installed!"
echo "➡️  Run: source ~/.profile && source ~/.bashrc"
