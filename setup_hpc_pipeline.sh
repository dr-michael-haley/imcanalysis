#!/bin/bash

IMC_DIR="$HOME/imcanalysis"
BASH_DIR="$IMC_DIR/Bash_scripts"
SLURM_DIR="$IMC_DIR/SLURM_scripts"

echo "Setting up IMC Analysis environment..."

#############################
# 0. Verify imcanalysis directory exists
#############################
if [[ ! -d "$IMC_DIR" ]]; then
    echo "❌ ERROR: $IMC_DIR does not exist."
    echo "Please clone the repository into your home directory:"
    echo "    git clone <repo-url> ~/imcanalysis"
    exit 1
fi

# Check subdirectories
if [[ ! -d "$BASH_DIR" ]]; then
    echo "❌ ERROR: Missing directory: $BASH_DIR"
    echo "The repository is incomplete or was not cloned correctly."
    exit 1
fi

if [[ ! -d "$SLURM_DIR" ]]; then
    echo "❌ ERROR: Missing directory: $SLURM_DIR"
    echo "The repository is incomplete or was not cloned correctly."
    exit 1
fi

echo "✔ Found imcanalysis directory and required subfolders."

#############################
# 1. Add Bash_scripts to PATH
#############################
if ! grep -q "$BASH_DIR" "$HOME/.profile" 2>/dev/null; then
    echo "export PATH=\"$BASH_DIR:\$PATH\"" >> "$HOME/.profile"
    echo "✔ Added $BASH_DIR to PATH in ~/.profile"
else
    echo "✔ PATH entry for Bash_scripts already present."
fi

#############################
# 2. Add cds alias to ~/.bashrc
#############################
ALIAS_LINE='alias cds='\''. "$HOME/imcanalysis/Bash_scripts/cds"'\'''

if ! grep -Fxq "$ALIAS_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$ALIAS_LINE" >> "$HOME/.bashrc"
    echo "✔ Added alias 'cds' to ~/.bashrc"
else
    echo "✔ Alias 'cds' already present."
fi

#############################
# 3. Make scripts executable
#############################
chmod +x "$BASH_DIR"/* 2>/dev/null
chmod +x "$SLURM_DIR"/* 2>/dev/null

echo "✔ Scripts marked executable."

##################################
# 4. Ensure ~/.imc_config exists
##################################
CONFIG_FILE="$HOME/.imc_config"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Creating config file at $CONFIG_FILE"

    read -p "Enter your email for SLURM notifications: " user_email
    read -p "Enter your OpenAI API key: " user_api_key

    cat <<EOF > "$CONFIG_FILE"
export IMC_EMAIL="$user_email"
export OPENAI_API_KEY="$user_api_key"
EOF

    chmod 600 "$CONFIG_FILE"
    echo "✔ Config file created and permissions set to 600 (not readable by other users)."
else
    echo "✔ Config file already exists: $CONFIG_FILE"
fi

#######################################################
# 5. Source ~/.imc_config from ~/.bashrc and ~/.profile
#######################################################
SRC_LINE='[ -f "$HOME/.imc_config" ] && source "$HOME/.imc_config"'

# Add to .bashrc (interactive shells)
if ! grep -Fxq "$SRC_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$SRC_LINE" >> "$HOME/.bashrc"
    echo "✔ Added config sourcing to ~/.bashrc"
else
    echo "✔ ~/.bashrc already sources ~/.imc_config"
fi

# Add to .profile (login shells)
if ! grep -Fxq "$SRC_LINE" "$HOME/.profile" 2>/dev/null; then
    echo "$SRC_LINE" >> "$HOME/.profile"
    echo "✔ Added config sourcing to ~/.profile"
else
    echo "✔ ~/.profile already sources ~/.imc_config"
fi

############
# Finished
############
echo "🎉 Setup complete!"
echo "➡️  Run:  source ~/.bashrc   and   source ~/.profile, or just log out and back in."
