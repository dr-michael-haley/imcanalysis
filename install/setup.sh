#!/bin/bash
set -e
source "$(dirname "$0")/common.sh"

IMC_DIR="${SBT_TOOLKIT_ROOT:-$HOME/imcanalysis}"
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
    echo '  Clone to ~/imcanalysis or export SBT_TOOLKIT_ROOT=/path/to/imcanalysis'
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
printf -v ROOT_LINE 'export SBT_TOOLKIT_ROOT=%q # SBT shell helpers' "$IMC_DIR"
append_if_missing "$ROOT_LINE" "$HOME/.profile"
append_if_missing "$ROOT_LINE" "$HOME/.bashrc"
printf -v PATH_LINE 'export PATH=%q:"$PATH" # SBT shell helpers' "$BASH_DIR"
append_if_missing "$PATH_LINE" "$HOME/.profile"
echo "✔ Added Bash_scripts to PATH."

###############################################
# 4. Add cds alias to ~/.bashrc
###############################################
printf -v CDS_COMMAND '. %q' "$BASH_DIR/cds"
printf -v ALIAS_LINE 'alias cds=%q' "$CDS_COMMAND"
append_if_missing "$ALIAS_LINE" "$HOME/.bashrc"
echo "✔ Added alias 'cds'."

###############################################
# 5. Create ~/.imc_config if missing
###############################################
if [[ ! -f "$CONFIG" ]]; then
    echo "Creating configuration file: $CONFIG"
    read -p "Enter SLURM notification email: " email
    read -p "Enter OpenAI API key (or leave blank): " openai_key

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

#######################################
# 7. Ensure ~/.bash_profile sources profile + bashrc
#######################################
BASH_PROFILE="$HOME/.bash_profile"

BP_LINE_PROFILE='[ -f "$HOME/.profile" ] && source "$HOME/.profile"'
BP_LINE_BASHRC='[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"'

touch "$BASH_PROFILE"

if ! grep -Fxq "$BP_LINE_PROFILE" "$BASH_PROFILE"; then
    echo "$BP_LINE_PROFILE" >> "$BASH_PROFILE"
    echo "✔ Added sourcing of ~/.profile to ~/.bash_profile"
fi

if ! grep -Fxq "$BP_LINE_BASHRC" "$BASH_PROFILE"; then
    echo "$BP_LINE_BASHRC" >> "$BASH_PROFILE"
    echo "✔ Added sourcing of ~/.bashrc to ~/.bash_profile"
fi


###############################################
# Finish
###############################################
echo "🎉 IMC Analysis successfully installed!"
echo "➡️  Run: source ~/.profile && source ~/.bashrc, OR just log out and back in again."
