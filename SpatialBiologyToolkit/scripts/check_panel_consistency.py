#!/usr/bin/env python3
"""
Check consistency between panel file and denoised images.

This script compares the panel file with the actual TIFF files in each ROI folder 
within the denoised_images_folder to identify missing files and verify that the 
expected channels are present.

Usage:
    python check_panel_consistency.py                                              # Uses defaults
    python check_panel_consistency.py --images_folder processed --panel_file metadata/panel.csv
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Import shared utilities
try:
    from .config_and_utils import cleanstring
except ImportError:
    from config_and_utils import cleanstring


class PanelConsistencyChecker:
    """
    Class to check consistency between panel file and denoised images.
    """
    
    def __init__(self, images_folder: str, panel_file: str):
        """
        Initialize the PanelConsistencyChecker.
        
        Parameters
        ----------
        images_folder : str
            Path to the folder containing denoised images organized by ROI.
        panel_file : str
            Path to the panel CSV file.
        """
        self.images_folder = Path(images_folder)
        self.panel_file = Path(panel_file)
        
        # Statistics
        self.total_rois = 0
        self.total_missing_files = 0
        self.total_extra_files = 0
        self.rois_with_issues = 0
        
        # Detailed results
        self.roi_results = {}
        
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_folder}")
        
        if not self.panel_file.exists():
            raise FileNotFoundError(f"Panel file not found: {self.panel_file}")
    
    def parse_tiff_filename(self, filename: str) -> Tuple[str, str, str, str]:
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
        tuple
            (channel_index, roi_index, channel_name, channel_label) or None if parsing fails.
        """
        # Remove .tiff extension
        stem = filename.replace('.tiff', '').replace('.tif', '')
        
        # Expected pattern: XX_YY_channelname_channellabel
        # Split by underscore and expect at least 4 parts
        parts = stem.split('_')
        
        if len(parts) < 4:
            return None
        
        try:
            channel_index = parts[0]
            roi_index = parts[1]
            channel_name = parts[2]
            # Join the remaining parts as channel_label (in case it contains underscores)
            channel_label = '_'.join(parts[3:])
            
            return channel_index, roi_index, channel_name, channel_label
        except (ValueError, IndexError):
            return None
    
    def load_panel_file(self) -> pd.DataFrame:
        """
        Load the panel file and extract expected channels.
        
        Returns
        -------
        pd.DataFrame
            Panel data with channel information.
        """
        panel_df = pd.read_csv(self.panel_file)
        
        # Ensure required columns exist
        required_cols = ['channel_name', 'channel_label']
        missing_cols = [col for col in required_cols if col not in panel_df.columns]
        if missing_cols:
            raise ValueError(f"Panel file missing required columns: {missing_cols}")
        
        # Filter out channels marked for deletion if delete column exists
        if 'delete' in panel_df.columns:
            # Convert delete column to boolean
            panel_df['delete_bool'] = panel_df['delete'].apply(
                lambda x: str(x).lower() in ['true', '1', 'yes', 'y'] if pd.notna(x) else False
            )
            # Keep only channels not marked for deletion
            panel_df = panel_df[~panel_df['delete_bool']].copy()
            
            deleted_channels = panel_df[panel_df['delete_bool']]['channel_name'].tolist()
            if deleted_channels:
                logging.info(f"Excluding channels marked for deletion: {deleted_channels}")
        
        # Clean channel labels for consistency
        panel_df['channel_label_clean'] = panel_df['channel_label'].apply(
            lambda x: cleanstring(str(x)) if pd.notna(x) else ''
        )
        
        logging.info(f"Loaded panel with {len(panel_df)} expected channels")
        return panel_df
    
    def get_expected_filenames(self, panel_df: pd.DataFrame, roi_name: str, roi_index: str) -> Set[str]:
        """
        Generate expected filenames for a ROI based on the panel.
        
        Parameters
        ----------
        panel_df : pd.DataFrame
            Panel dataframe.
        roi_name : str
            Name of the ROI folder.
        roi_index : str
            ROI index to use in filenames.
            
        Returns
        -------
        set
            Set of expected filenames.
        """
        expected_files = set()
        
        for idx, row in panel_df.iterrows():
            channel_name = row['channel_name']
            channel_label = row['channel_label_clean']
            
            # Generate expected filename: XX_YY_channelname_channellabel.tiff
            # Use the panel index as channel index (padded to 2 digits)
            channel_index = f"{idx:02d}"
            filename = f"{channel_index}_{roi_index}_{channel_name}_{channel_label}.tiff"
            expected_files.add(filename)
        
        return expected_files
    
    def extract_roi_index_from_folder(self, roi_folder: Path) -> str:
        """
        Extract ROI index from the first TIFF file in the folder.
        
        Parameters
        ----------
        roi_folder : Path
            Path to the ROI folder.
            
        Returns
        -------
        str
            ROI index, or "00" if not found.
        """
        # Get first TIFF file to extract ROI index
        tiff_files = list(roi_folder.glob('*.tiff')) + list(roi_folder.glob('*.tif'))
        
        if tiff_files:
            parsed = self.parse_tiff_filename(tiff_files[0].name)
            if parsed:
                return parsed[1]  # roi_index
        
        # Default to "00" if we can't extract ROI index
        return "00"
    
    def check_roi_folder(self, roi_folder: Path, panel_df: pd.DataFrame) -> Dict:
        """
        Check consistency for a single ROI folder.
        
        Parameters
        ----------
        roi_folder : Path
            Path to the ROI folder.
        panel_df : pd.DataFrame
            Panel dataframe.
            
        Returns
        -------
        dict
            Results for this ROI.
        """
        roi_name = roi_folder.name
        
        # Get all TIFF files in the folder
        tiff_files = list(roi_folder.glob('*.tiff')) + list(roi_folder.glob('*.tif'))
        actual_files = {f.name for f in tiff_files}
        
        # Extract ROI index from existing files
        roi_index = self.extract_roi_index_from_folder(roi_folder)
        
        # Get expected filenames based on panel
        expected_files = self.get_expected_filenames(panel_df, roi_name, roi_index)
        
        # Find missing and extra files
        missing_files = expected_files - actual_files
        extra_files = actual_files - expected_files
        
        # Parse actual files to get channel information
        actual_channels = set()
        unparseable_files = []
        
        for filename in actual_files:
            parsed = self.parse_tiff_filename(filename)
            if parsed:
                channel_name = parsed[2]
                actual_channels.add(channel_name)
            else:
                unparseable_files.append(filename)
        
        # Get expected channels from panel
        expected_channels = set(panel_df['channel_name'])
        missing_channels = expected_channels - actual_channels
        extra_channels = actual_channels - expected_channels
        
        result = {
            'roi_name': roi_name,
            'roi_index': roi_index,
            'total_files': len(actual_files),
            'expected_files': len(expected_files),
            'missing_files': list(missing_files),
            'extra_files': list(extra_files),
            'missing_channels': list(missing_channels),
            'extra_channels': list(extra_channels),
            'unparseable_files': unparseable_files,
            'has_issues': len(missing_files) > 0 or len(extra_files) > 0 or len(unparseable_files) > 0
        }
        
        return result
    
    def check_all_rois(self) -> None:
        """
        Check consistency for all ROI folders.
        """
        logging.info(f"Loading panel file: {self.panel_file}")
        panel_df = self.load_panel_file()
        
        # Get all ROI folders
        roi_folders = [f for f in self.images_folder.iterdir() if f.is_dir()]
        
        if not roi_folders:
            logging.warning(f"No ROI folders found in {self.images_folder}")
            return
        
        self.total_rois = len(roi_folders)
        logging.info(f"Found {self.total_rois} ROI folders")
        
        # Check each ROI folder
        for roi_folder in tqdm(roi_folders, desc="Checking ROI folders"):
            result = self.check_roi_folder(roi_folder, panel_df)
            self.roi_results[result['roi_name']] = result
            
            # Update statistics
            self.total_missing_files += len(result['missing_files'])
            self.total_extra_files += len(result['extra_files'])
            
            if result['has_issues']:
                self.rois_with_issues += 1
    
    def print_summary(self) -> None:
        """Print a summary of the consistency check results."""
        logging.info(f"\n{'='*60}")
        logging.info("PANEL CONSISTENCY CHECK SUMMARY")
        logging.info(f"{'='*60}")
        
        logging.info(f"Total ROIs checked: {self.total_rois}")
        logging.info(f"ROIs with issues: {self.rois_with_issues}")
        logging.info(f"ROIs without issues: {self.total_rois - self.rois_with_issues}")
        logging.info(f"Total missing files: {self.total_missing_files}")
        logging.info(f"Total extra files: {self.total_extra_files}")
        
        # Overall status
        if self.rois_with_issues == 0:
            logging.info("\n✅ ALL ROIs are consistent with the panel file!")
        else:
            logging.warning(f"\n⚠️  {self.rois_with_issues} ROIs have inconsistencies")
        
        logging.info(f"\n{'='*60}")
    
    def print_detailed_results(self, show_all: bool = False) -> None:
        """
        Print detailed results for each ROI.
        
        Parameters
        ----------
        show_all : bool, optional
            If True, show results for all ROIs. If False, only show ROIs with issues.
        """
        logging.info("\nDETAILED RESULTS:")
        logging.info("-" * 40)
        
        for roi_name, result in sorted(self.roi_results.items()):
            if not show_all and not result['has_issues']:
                continue
            
            status = "❌ ISSUES" if result['has_issues'] else "✅ OK"
            logging.info(f"\nROI: {roi_name} [{status}]")
            logging.info(f"  Files: {result['total_files']} actual / {result['expected_files']} expected")
            
            if result['missing_files']:
                logging.info(f"  Missing files ({len(result['missing_files'])}):")
                for filename in sorted(result['missing_files'])[:5]:  # Show first 5
                    logging.info(f"    - {filename}")
                if len(result['missing_files']) > 5:
                    logging.info(f"    ... and {len(result['missing_files']) - 5} more")
            
            if result['extra_files']:
                logging.info(f"  Extra files ({len(result['extra_files'])}):")
                for filename in sorted(result['extra_files'])[:5]:  # Show first 5
                    logging.info(f"    + {filename}")
                if len(result['extra_files']) > 5:
                    logging.info(f"    ... and {len(result['extra_files']) - 5} more")
            
            if result['missing_channels']:
                logging.info(f"  Missing channels: {result['missing_channels']}")
            
            if result['extra_channels']:
                logging.info(f"  Extra channels: {result['extra_channels']}")
            
            if result['unparseable_files']:
                logging.info(f"  Unparseable files: {result['unparseable_files']}")
    
    def save_results_to_csv(self, output_file: str = "panel_consistency_report.csv") -> None:
        """
        Save detailed results to a CSV file.
        
        Parameters
        ----------
        output_file : str, optional
            Output CSV filename.
        """
        if not self.roi_results:
            logging.warning("No results to save")
            return
        
        # Prepare data for CSV
        csv_data = []
        for roi_name, result in self.roi_results.items():
            csv_data.append({
                'roi_name': result['roi_name'],
                'roi_index': result['roi_index'],
                'has_issues': result['has_issues'],
                'total_files': result['total_files'],
                'expected_files': result['expected_files'],
                'missing_files_count': len(result['missing_files']),
                'extra_files_count': len(result['extra_files']),
                'missing_channels_count': len(result['missing_channels']),
                'extra_channels_count': len(result['extra_channels']),
                'unparseable_files_count': len(result['unparseable_files']),
                'missing_files': '; '.join(result['missing_files']),
                'extra_files': '; '.join(result['extra_files']),
                'missing_channels': '; '.join(result['missing_channels']),
                'extra_channels': '; '.join(result['extra_channels']),
                'unparseable_files': '; '.join(result['unparseable_files'])
            })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        logging.info(f"Detailed results saved to: {output_file}")


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Check consistency between panel file and denoised images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check with default settings
  python check_panel_consistency.py
  
  # Specify custom paths
  python check_panel_consistency.py --images_folder processed --panel_file metadata/panel.csv
  
  # Show all ROIs (not just those with issues)
  python check_panel_consistency.py --show_all
  
  # Save detailed report to CSV
  python check_panel_consistency.py --save_csv consistency_report.csv
        """
    )
    
    parser.add_argument('--images_folder', type=str, 
                       help='Path to folder containing denoised images organized by ROI (default: from config or "processed")')
    parser.add_argument('--panel_file', type=str,
                       help='Path to panel CSV file (default: from config or "metadata/panel.csv")')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--show_all', action='store_true',
                       help='Show results for all ROIs, not just those with issues')
    parser.add_argument('--save_csv', type=str,
                       help='Save detailed results to CSV file')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    return parser


def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] [PanelChecker] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Determine parameters - either from command line or config file
    if args.images_folder or args.panel_file:
        # Use command line arguments
        images_folder = args.images_folder or 'processed'
        panel_file = args.panel_file or 'metadata/panel.csv'
    else:
        # Try to load from config file manually
        try:
            import yaml
            config_file = args.config
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                
                general_config = config.get('general', {})
                images_folder = general_config.get('denoised_images_folder', 'processed')
                metadata_folder = general_config.get('metadata_folder', 'metadata')
                panel_file = str(Path(metadata_folder) / 'panel.csv')
            else:
                logging.info(f"Config file {config_file} not found, using defaults")
                images_folder = 'processed'
                panel_file = 'metadata/panel.csv'
                
        except Exception as e:
            logging.warning(f"Could not load config: {e}")
            logging.info("Using default values: images_folder='processed', panel_file='metadata/panel.csv'")
            images_folder = 'processed'
            panel_file = 'metadata/panel.csv'
    
    # Initialize checker and run analysis
    try:
        checker = PanelConsistencyChecker(images_folder, panel_file)
        
        logging.info(f"Checking consistency between:")
        logging.info(f"  Panel file: {panel_file}")
        logging.info(f"  Images folder: {images_folder}")
        
        # Perform the check
        checker.check_all_rois()
        
        # Print results
        checker.print_summary()
        checker.print_detailed_results(show_all=args.show_all)
        
        # Save to CSV if requested
        if args.save_csv:
            checker.save_results_to_csv(args.save_csv)
        
        # Exit with appropriate code
        return 0 if checker.rois_with_issues == 0 else 1
        
    except Exception as e:
        logging.error(f"Error during consistency check: {e}")
        return 1


if __name__ == "__main__":
    exit(main())