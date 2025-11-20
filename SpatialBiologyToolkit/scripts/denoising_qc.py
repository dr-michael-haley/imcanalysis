#!/usr/bin/env python3
"""
Standalone script for generating denoising QC side-by-side comparisons.

This script creates side-by-side visualizations comparing raw and denoised images
for quality control purposes. It can be run independently without modifying the
main config flags, making it easy to regenerate QC images after adjusting parameters
like the number of ROIs to include.

Usage:
    python denoising_qc.py
    python denoising_qc.py --config custom_config.yaml
    python denoising_qc.py --override denoising.qc_num_rois=5
"""

from .config_and_utils import (
    GeneralConfig,
    DenoisingConfig,
    filter_config_for_dataclass,
    process_config_with_overrides,
    setup_logging,
)
from .denoising import qc_check_side_by_side


def main():
    """Main function to run denoising QC visualization."""
    pipeline_stage = 'DenoisingQC'
    
    # Load configuration
    config_data = process_config_with_overrides()
    
    # Setup logging
    setup_logging(config_data.get('logging', {}), pipeline_stage)
    
    # Get parameters from config
    general_config = GeneralConfig(**filter_config_for_dataclass(config_data.get('general', {}), GeneralConfig))
    denoise_config = DenoisingConfig(**filter_config_for_dataclass(config_data.get('denoising', {}), DenoisingConfig))
    
    # Run QC side-by-side comparison
    print(f"\nGenerating denoising QC visualizations...")
    print(f"Channels: {denoise_config.channels if denoise_config.channels else 'All channels from panel'}")
    print(f"Number of ROIs per channel: {denoise_config.qc_num_rois if denoise_config.qc_num_rois else 'All ROIs'}")
    print(f"Output directory: {general_config.qc_folder}/{denoise_config.qc_image_dir}\n")
    
    qc_check_side_by_side(general_config, denoise_config)
    
    print("\nDenoising QC generation complete!")


if __name__ == "__main__":
    main()
