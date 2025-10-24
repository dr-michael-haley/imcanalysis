#!/usr/bin/env python3
"""
Harmonize TIFF file names using a panel file.

This script standardizes channel label names in TIFF files based on a reference panel file.
It finds files using channel_name and renames the channel_label part to match the 
standardized values from the panel file.

Usage:
    python harmonize_filenames.py                                                    # Uses defaults
    python harmonize_filenames.py --execute                                          # Execute with defaults
    python harmonize_filenames.py --tiff_folder tiffs --panel_file metadata/panel.csv --execute
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Import shared utilities
try:
    from .config_and_utils import cleanstring, setup_logging, process_config_with_overrides
except ImportError:
    from config_and_utils import cleanstring, setup_logging, process_config_with_overrides


class FilenameHarmonizer:
    """
    Class to harmonize TIFF filenames based on panel information.
    """
    
    def __init__(self, tiff_folder: str, dry_run: bool = True):
        """
        Initialize the FilenameHarmonizer.
        
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
        self.deleted_count = 0
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
                          channel_name: str, new_channel_label: str) -> str:
        """
        Create a new filename with the harmonized channel label.
        
        Parameters
        ----------
        channel_index : str
            The channel index (XX).
        roi_index : str
            The ROI index (YY).
        channel_name : str
            The channel name.
        new_channel_label : str
            The new standardized channel label.
            
        Returns
        -------
        str
            The new filename.
        """
        # Clean the new channel label to ensure it's filesystem-safe
        clean_label = cleanstring(new_channel_label)
        return f"{channel_index}_{roi_index}_{channel_name}_{clean_label}.tiff"
    
    def load_panel_file(self, panel_file: str) -> pd.DataFrame:
        """
        Load a panel file and prepare it for harmonization.
        
        Parameters
        ----------
        panel_file : str
            Path to the panel CSV file.
            
        Returns
        -------
        pd.DataFrame
            Panel data with cleaned channel labels.
        """
        panel_df = pd.read_csv(panel_file)
        
        # Ensure required columns exist
        required_cols = ['channel_name', 'channel_label']
        missing_cols = [col for col in required_cols if col not in panel_df.columns]
        if missing_cols:
            raise ValueError(f"Panel file missing required columns: {missing_cols}")
        
        # Clean channel labels for consistency
        panel_df['channel_label_clean'] = panel_df['channel_label'].apply(lambda x: cleanstring(str(x)) if pd.notna(x) else '')
        
        # Create mapping from channel_name to standardized channel_label
        channel_mapping = dict(zip(panel_df['channel_name'], panel_df['channel_label_clean']))
        
        # Create delete mapping if delete column exists
        delete_mapping = {}
        if 'delete' in panel_df.columns:
            # Convert delete column to boolean, treating various representations as True
            panel_df['delete_bool'] = panel_df['delete'].apply(
                lambda x: str(x).lower() in ['true', '1', 'yes', 'y'] if pd.notna(x) else False
            )
            delete_mapping = dict(zip(panel_df['channel_name'], panel_df['delete_bool']))
            
            # Log channels marked for deletion
            channels_to_delete = panel_df[panel_df['delete_bool']]['channel_name'].tolist()
            if channels_to_delete:
                logging.info(f"Channels marked for deletion: {channels_to_delete}")
        
        logging.info(f"Loaded panel with {len(channel_mapping)} channel mappings")
        return panel_df, channel_mapping, delete_mapping
    

    
    def harmonize_roi_folder(self, roi_folder: Path, channel_mapping: Dict[str, str], delete_mapping: Dict[str, bool] = None) -> Tuple[int, int, int]:
        """
        Harmonize filenames in a single ROI folder.
        
        Parameters
        ----------
        roi_folder : Path
            Path to the ROI folder.
        channel_mapping : dict
            Mapping from channel_name to standardized channel_label.
        delete_mapping : dict, optional
            Mapping from channel_name to delete flag.
            
        Returns
        -------
        tuple
            (number_renamed, number_deleted, number_errors)
        """
        renamed = 0
        deleted = 0
        errors = 0
        
        if delete_mapping is None:
            delete_mapping = {}
        
        # Get all TIFF files in the folder
        tiff_files = list(roi_folder.glob('*.tiff')) + list(roi_folder.glob('*.tif'))
        
        if not tiff_files:
            logging.info(f"No TIFF files found in {roi_folder}")
            return renamed, deleted, errors
        
        logging.info(f"Processing {len(tiff_files)} files in {roi_folder.name}")
        
        for tiff_file in tiff_files:
            try:
                # Parse the current filename
                parsed = self.parse_tiff_filename(tiff_file.name)
                if not parsed:
                    errors += 1
                    continue
                
                channel_index, roi_index, channel_name, current_label = parsed
                
                # Check if this channel should be deleted
                if delete_mapping.get(channel_name, False):
                    if self.dry_run:
                        logging.info(f"Would delete: {tiff_file.name}")
                    else:
                        tiff_file.unlink()
                        logging.info(f"Deleted: {tiff_file.name}")
                    deleted += 1
                    continue
                
                # Look up the standardized channel label
                if channel_name in channel_mapping:
                    new_label = channel_mapping[channel_name]
                    
                    # Only rename if the labels are different
                    if cleanstring(current_label) != new_label:
                        new_filename = self.create_new_filename(
                            channel_index, roi_index, channel_name, new_label
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
                        logging.debug(f"No change needed for {tiff_file.name}")
                else:
                    logging.warning(f"Channel {channel_name} not found in panel mapping for {tiff_file.name}")
                    errors += 1
                    
            except Exception as e:
                logging.error(f"Error processing {tiff_file.name}: {e}")
                errors += 1
        
        return renamed, deleted, errors
    
    def harmonize(self, panel_file: str) -> None:
        """
        Harmonize all TIFF files using a panel file.
        
        Parameters
        ----------
        panel_file : str
            Path to the panel CSV file.
        """
        logging.info(f"Loading panel file: {panel_file}")
        _, channel_mapping, delete_mapping = self.load_panel_file(panel_file)
        
        # Process all ROI folders
        roi_folders = [f for f in self.tiff_folder.iterdir() if f.is_dir()]
        
        if not roi_folders:
            logging.warning(f"No ROI folders found in {self.tiff_folder}")
            return
        
        logging.info(f"Found {len(roi_folders)} ROI folders")
        
        for roi_folder in tqdm(roi_folders, desc="Processing ROI folders"):
            renamed, deleted, errors = self.harmonize_roi_folder(roi_folder, channel_mapping, delete_mapping)
            self.renamed_count += renamed
            self.deleted_count += deleted
            self.error_count += errors
    
    def print_summary(self) -> None:
        """Print a summary of the harmonization results."""
        action = "Would rename" if self.dry_run else "Renamed"
        delete_action = "Would delete" if self.dry_run else "Deleted"
        logging.info(f"\nHarmonization Summary:")
        logging.info(f"  {action}: {self.renamed_count} files")
        logging.info(f"  {delete_action}: {self.deleted_count} files")
        logging.info(f"  Errors: {self.error_count} files")
        
        if self.dry_run and (self.renamed_count > 0 or self.deleted_count > 0):
            logging.info(f"\nTo actually perform the changes, run with --execute flag")


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Harmonize TIFF filenames using panel information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with default settings (uses metadata/panel.csv)
  python harmonize_filenames.py
  
  # Execute with default settings
  python harmonize_filenames.py --execute
  
  # Specify custom paths
  python harmonize_filenames.py --tiff_folder tiffs --panel_file metadata/panel.csv --execute
  
  # Use config file
  python harmonize_filenames.py --config config.yaml --execute
        """
    )
    
    parser.add_argument('--tiff_folder', type=str, 
                       help='Path to folder containing TIFF files organized by ROI (default: from config or "tiffs")')
    parser.add_argument('--panel_file', type=str,
                       help='Path to panel CSV file (default: from config or "metadata/panel.csv")')
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
        format='%(asctime)s [%(levelname)s] [FilenameHarmonizer] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Determine parameters - either from command line or config file
    if args.panel_file or args.tiff_folder:
        # Use command line arguments
        tiff_folder = args.tiff_folder or 'tiffs'
        panel_file = args.panel_file or 'metadata/panel.csv'
    else:
        # Try to load from config file
        try:
            config = process_config_with_overrides()
            general_config = config.get('general', {})
            
            tiff_folder = general_config.get('raw_images_folder', 'tiffs')
            metadata_folder = general_config.get('metadata_folder', 'metadata')
            panel_file = str(Path(metadata_folder) / 'panel.csv')
                
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            logging.info("Using default values: tiff_folder='tiffs', panel_file='metadata/panel.csv'")
            tiff_folder = 'tiffs'
            panel_file = 'metadata/panel.csv'
    
    # Validate panel file exists
    if not Path(panel_file).exists():
        logging.error(f"Panel file not found: {panel_file}")
        return 1
    
    # Initialize harmonizer
    try:
        harmonizer = FilenameHarmonizer(tiff_folder, dry_run=not args.execute)
        
        if args.execute:
            logging.info("EXECUTING: Files will be renamed")
        else:
            logging.info("DRY RUN: No files will be renamed")
        
        # Perform harmonization
        logging.info(f"Using panel file: {panel_file}")
        harmonizer.harmonize(panel_file)
        
        # Print summary
        harmonizer.print_summary()
        
        return 0 if harmonizer.error_count == 0 else 1
        
    except Exception as e:
        logging.error(f"Error during harmonization: {e}")
        return 1


if __name__ == "__main__":
    exit(main())