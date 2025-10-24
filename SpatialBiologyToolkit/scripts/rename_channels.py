#!/usr/bin/env python3
"""
Rename channel names in TIFF files based on a channel rename mapping.

This script renames the channel_name part of TIFF filenames based on mappings 
defined in a channel_rename.csv file. This is useful when the same marker 
was accidentally assigned to different metals and needs to be consolidated.

Usage:
    python rename_channels.py                                                     # Uses defaults
    python rename_channels.py --execute                                           # Execute with defaults
    python rename_channels.py --tiff_folder tiffs --rename_file metadata/channel_rename.csv --execute
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Import shared utilities
try:
    from .config_and_utils import cleanstring
except ImportError:
    from config_and_utils import cleanstring


class ChannelRenamer:
    """
    Class to rename channel names in TIFF filenames based on a mapping file.
    """
    
    def __init__(self, tiff_folder: str, dry_run: bool = True):
        """
        Initialize the ChannelRenamer.
        
        Parameters
        ----------
        tiff_folder : str
            Path to the folder containing TIFF files organized by ROI.
        dry_run : bool, optional
            If True, only show what would be renamed without actually renaming. Defaults to True.
        """
        self.tiff_folder = Path(tiff_folder)
        self.dry_run = dry_run
        self.renamed_count = 0
        self.error_count = 0
        
        if not self.tiff_folder.exists():
            raise FileNotFoundError(f"TIFF folder not found: {self.tiff_folder}")
    
    def parse_tiff_filename(self, filename: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parse a TIFF filename to extract components.
        
        Expected format: XX_YY_channelname_channellabel.tiff
        Where XX is channel index, YY is ROI index.
        
        Parameters
        ----------
        filename : str
            The TIFF filename to parse.
            
        Returns
        -------
        tuple or None
            (channel_index, roi_index, channel_name, channel_label) or None if parsing fails.
        """
        # Remove .tiff extension
        stem = filename.replace('.tiff', '').replace('.tif', '')
        
        # Expected pattern: XX_YY_channelname_channellabel
        # Split by underscore and expect at least 4 parts
        parts = stem.split('_')
        
        if len(parts) < 4:
            logging.warning(f"Filename {filename} doesn't match expected pattern XX_YY_channelname_channellabel")
            return None
        
        try:
            channel_index = parts[0]
            roi_index = parts[1]
            channel_name = parts[2]
            # Join the remaining parts as channel_label (in case it contains underscores)
            channel_label = '_'.join(parts[3:])
            
            return channel_index, roi_index, channel_name, channel_label
        except (ValueError, IndexError) as e:
            logging.warning(f"Error parsing filename {filename}: {e}")
            return None
    
    def create_new_filename(self, channel_index: str, roi_index: str, 
                          new_channel_name: str, channel_label: str) -> str:
        """
        Create a new filename with the renamed channel name.
        
        Parameters
        ----------
        channel_index : str
            The channel index (XX).
        roi_index : str
            The ROI index (YY).
        new_channel_name : str
            The new channel name.
        channel_label : str
            The channel label.
            
        Returns
        -------
        str
            The new filename.
        """
        # Clean the channel name to ensure it's filesystem-safe
        clean_channel_name = cleanstring(new_channel_name)
        clean_label = cleanstring(channel_label)
        return f"{channel_index}_{roi_index}_{clean_channel_name}_{clean_label}.tiff"
    
    def load_rename_file(self, rename_file: str) -> Dict[str, str]:
        """
        Load a channel rename file and prepare it for renaming.
        
        Parameters
        ----------
        rename_file : str
            Path to the channel rename CSV file.
            
        Returns
        -------
        dict
            Mapping from old channel_name to new channel_name.
        """
        if not Path(rename_file).exists():
            raise FileNotFoundError(f"Channel rename file not found: {rename_file}")
        
        rename_df = pd.read_csv(rename_file)
        
        # Try different possible column names for flexibility
        old_col = None
        new_col = None
        
        # Look for old channel name column
        for col in ['old_channel_name', 'from', 'old', 'source']:
            if col in rename_df.columns:
                old_col = col
                break
        
        # Look for new channel name column  
        for col in ['new_channel_name', 'to', 'new', 'target']:
            if col in rename_df.columns:
                new_col = col
                break
        
        # If standard names not found, use first two columns
        if old_col is None or new_col is None:
            if len(rename_df.columns) >= 2:
                old_col = rename_df.columns[0]
                new_col = rename_df.columns[1]
                logging.info(f"Using columns: '{old_col}' -> '{new_col}'")
            else:
                raise ValueError("Rename file must have at least 2 columns")
        
        # Create mapping dictionary
        rename_mapping = dict(zip(rename_df[old_col], rename_df[new_col]))
        
        # Remove any NaN mappings
        rename_mapping = {k: v for k, v in rename_mapping.items() 
                         if pd.notna(k) and pd.notna(v)}
        
        logging.info(f"Loaded {len(rename_mapping)} channel rename mappings:")
        for old, new in rename_mapping.items():
            logging.info(f"  {old} -> {new}")
        
        return rename_mapping
    
    def rename_roi_folder(self, roi_folder: Path, rename_mapping: Dict[str, str]) -> Tuple[int, int]:
        """
        Rename channel names in a single ROI folder.
        
        Parameters
        ----------
        roi_folder : Path
            Path to the ROI folder.
        rename_mapping : dict
            Mapping from old channel_name to new channel_name.
            
        Returns
        -------
        tuple
            (number_renamed, number_errors)
        """
        renamed = 0
        errors = 0
        
        # Get all TIFF files in the folder
        tiff_files = list(roi_folder.glob('*.tiff')) + list(roi_folder.glob('*.tif'))
        
        if not tiff_files:
            logging.info(f"No TIFF files found in {roi_folder}")
            return renamed, errors
        
        logging.info(f"Processing {len(tiff_files)} files in {roi_folder.name}")
        
        for tiff_file in tiff_files:
            try:
                # Parse the current filename
                parsed = self.parse_tiff_filename(tiff_file.name)
                if not parsed:
                    errors += 1
                    continue
                
                channel_index, roi_index, channel_name, channel_label = parsed
                
                # Check if this channel name should be renamed
                if channel_name in rename_mapping:
                    new_channel_name = rename_mapping[channel_name]
                    
                    # Only rename if the channel names are different
                    if channel_name != new_channel_name:
                        new_filename = self.create_new_filename(
                            channel_index, roi_index, new_channel_name, channel_label
                        )
                        new_path = roi_folder / new_filename
                        
                        if self.dry_run:
                            logging.info(f"Would rename: {tiff_file.name} -> {new_filename}")
                        else:
                            # Check if target file already exists
                            if new_path.exists():
                                logging.warning(f"Target file already exists: {new_filename}")
                                errors += 1
                                continue
                            
                            # Perform the rename
                            tiff_file.rename(new_path)
                            logging.info(f"Renamed: {tiff_file.name} -> {new_filename}")
                        
                        renamed += 1
                    else:
                        logging.debug(f"No change needed for {tiff_file.name} (same channel name)")
                else:
                    logging.debug(f"Channel {channel_name} not in rename mapping for {tiff_file.name}")
                    
            except Exception as e:
                logging.error(f"Error processing {tiff_file.name}: {e}")
                errors += 1
        
        return renamed, errors
    
    def rename_channels(self, rename_file: str) -> None:
        """
        Rename channel names in all TIFF files using a rename mapping file.
        
        Parameters
        ----------
        rename_file : str
            Path to the channel rename CSV file.
        """
        logging.info(f"Loading channel rename file: {rename_file}")
        rename_mapping = self.load_rename_file(rename_file)
        
        if not rename_mapping:
            logging.warning("No rename mappings found. Nothing to do.")
            return
        
        # Process all ROI folders
        roi_folders = [f for f in self.tiff_folder.iterdir() if f.is_dir()]
        
        if not roi_folders:
            logging.warning(f"No ROI folders found in {self.tiff_folder}")
            return
        
        logging.info(f"Found {len(roi_folders)} ROI folders")
        
        for roi_folder in tqdm(roi_folders, desc="Processing ROI folders"):
            renamed, errors = self.rename_roi_folder(roi_folder, rename_mapping)
            self.renamed_count += renamed
            self.error_count += errors
    
    def print_summary(self) -> None:
        """Print a summary of the renaming results."""
        action = "Would rename" if self.dry_run else "Renamed"
        logging.info(f"\\nChannel Renaming Summary:")
        logging.info(f"  {action}: {self.renamed_count} files")
        logging.info(f"  Errors: {self.error_count} files")
        
        if self.dry_run and self.renamed_count > 0:
            logging.info(f"\\nTo actually perform the renaming, run with --execute flag")


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Rename channel names in TIFF filenames based on a mapping file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with default settings (uses metadata/channel_rename.csv)
  python rename_channels.py
  
  # Execute with default settings
  python rename_channels.py --execute
  
  # Specify custom paths
  python rename_channels.py --tiff_folder tiffs --rename_file metadata/channel_rename.csv --execute
  
  # Use config file
  python rename_channels.py --config config.yaml --execute

Channel rename CSV format:
  The CSV file should have columns for old and new channel names.
  Supported column names:
    - old_channel_name, new_channel_name (preferred)
    - from, to
    - old, new  
    - source, target
  
  Example:
    old_channel_name,new_channel_name
    Yb170,Er170
    Nd142,Nd143
    In113,In115
        """
    )
    
    parser.add_argument('--tiff_folder', type=str, 
                       help='Path to folder containing TIFF files organized by ROI (default: from config or "tiffs")')
    parser.add_argument('--rename_file', type=str,
                       help='Path to channel rename CSV file (default: from config or "metadata/channel_rename.csv")')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually rename files (default: dry run only)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    return parser


def main():
    """Main function."""
    parser = setup_argument_parser()
    
    # Parse only the arguments we know about, ignore others (like --override from config system)
    args, unknown = parser.parse_known_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] [ChannelRenamer] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Determine parameters - either from command line or config file
    if args.rename_file or args.tiff_folder:
        # Use command line arguments
        tiff_folder = args.tiff_folder or 'tiffs'
        rename_file = args.rename_file or 'metadata/channel_rename.csv'
    else:
        # Try to load from config file manually (avoid process_config_with_overrides)
        try:
            import yaml
            config_file = args.config
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                
                general_config = config.get('general', {})
                tiff_folder = general_config.get('raw_images_folder', 'tiffs')
                metadata_folder = general_config.get('metadata_folder', 'metadata')
                rename_file = str(Path(metadata_folder) / 'channel_rename.csv')
            else:
                logging.info(f"Config file {config_file} not found, using defaults")
                tiff_folder = 'tiffs'
                rename_file = 'metadata/channel_rename.csv'
                
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            logging.info("Using default values: tiff_folder='tiffs', rename_file='metadata/channel_rename.csv'")
            tiff_folder = 'tiffs'
            rename_file = 'metadata/channel_rename.csv'
    
    # Validate rename file exists
    if not Path(rename_file).exists():
        logging.error(f"Channel rename file not found: {rename_file}")
        logging.info("Create a CSV file with old_channel_name and new_channel_name columns")
        logging.info("Example:")
        logging.info("  old_channel_name,new_channel_name")
        logging.info("  Yb170,Er170")
        logging.info("  Nd142,Nd143")
        return 1
    
    # Initialize renamer
    try:
        renamer = ChannelRenamer(tiff_folder, dry_run=not args.execute)
        
        if args.execute:
            logging.info("EXECUTING: Files will be renamed")
        else:
            logging.info("DRY RUN: No files will be renamed")
        
        # Perform renaming
        logging.info(f"Using rename file: {rename_file}")
        renamer.rename_channels(rename_file)
        
        # Print summary
        renamer.print_summary()
        
        return 0 if renamer.error_count == 0 else 1
        
    except Exception as e:
        logging.error(f"Error during channel renaming: {e}")
        return 1


if __name__ == "__main__":
    exit(main())