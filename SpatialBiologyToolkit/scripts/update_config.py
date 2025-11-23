# update_config.py

"""
Script to update an existing config.yaml file with new default values.

This script will:
1. Load the existing config file
2. Merge in any missing parameters from the default dataclass definitions
3. Save the updated config file while preserving all existing user settings

Usage:
    python update_config.py [config_file_path]

If no config_file_path is provided, it defaults to 'config.yaml' in the current directory.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import config_and_utils
sys.path.insert(0, str(Path(__file__).parent))

from config_and_utils import load_config, generate_default_config_dict
import yaml
import logging

def update_config_with_defaults(config_file: str = 'config.yaml', backup: bool = False):
    """
    Update an existing config file with new default values without overwriting user settings.
    
    Parameters:
    -----------
    config_file : str
        Path to the config file to update
    backup : bool
        If True, creates a backup of the original config file before updating
    """
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    config_path = Path(config_file)
    
    # Check if config file exists
    if not config_path.exists():
        logging.error(f"Config file not found: {config_file}")
        logging.info("Creating new config file with default values...")
        load_config(config_file)
        logging.info(f"Created new config file: {config_file}")
        return
    
    # Create backup if requested
    if backup:
        backup_path = config_path.with_suffix('.yaml.backup')
        import shutil
        shutil.copy2(config_path, backup_path)
        logging.info(f"Created backup: {backup_path}")
    
    # Load existing config
    logging.info(f"Loading existing config: {config_file}")
    with open(config_file, 'r') as f:
        existing_config = yaml.safe_load(f) or {}
    
    # Get default config
    defaults = generate_default_config_dict()
    
    # Count changes
    changes_made = 0
    sections_added = []
    params_added = {}
    
    # Merge defaults into existing config
    for section, default_values in defaults.items():
        if section not in existing_config:
            existing_config[section] = default_values
            sections_added.append(section)
            changes_made += len(default_values)
        else:
            params_added[section] = []
            for key, value in default_values.items():
                if key not in existing_config[section]:
                    existing_config[section][key] = value
                    params_added[section].append(key)
                    changes_made += 1
    
    # Save updated config
    if changes_made > 0:
        with open(config_file, 'w') as f:
            yaml.safe_dump(existing_config, f, default_flow_style=False, sort_keys=False)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Config file updated successfully!")
        logging.info(f"{'='*60}")
        logging.info(f"Total parameters added: {changes_made}")
        
        if sections_added:
            logging.info(f"\nNew sections added: {', '.join(sections_added)}")
        
        for section, params in params_added.items():
            if params:
                logging.info(f"\n[{section}] - Added {len(params)} parameter(s):")
                for param in params:
                    logging.info(f"  - {param}")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Updated config saved to: {config_file}")
        if backup:
            logging.info(f"Original config backed up to: {backup_path}")
        logging.info(f"{'='*60}\n")
    else:
        logging.info("No changes needed - config file is already up to date!")

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config.yaml'
    
    # Check if --backup flag is provided
    backup = '--backup' in sys.argv
    
    print("\n" + "="*60)
    print("Config File Update Utility")
    print("="*60 + "\n")
    
    update_config_with_defaults(config_file, backup=backup)

if __name__ == "__main__":
    main()
