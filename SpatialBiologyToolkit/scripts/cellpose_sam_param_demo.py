#!/usr/bin/env python3
"""
Demo script for CellPose-SAM parameter scanning.

This script demonstrates how to run parameter scanning with CellPose-SAM.
"""

from cellpose_sam import parameter_scan_cpsam, process_all_rois
from config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig
)

def run_parameter_scan_demo():
    """Run a demo parameter scan with CellPose-SAM."""
    
    # Load configuration
    config_data = process_config_with_overrides()
    
    # Setup logging
    setup_logging(config_data.get('logging', {}), 'CellposeSAM_ParamScan_Demo')
    
    # Get configuration objects
    general_config = GeneralConfig(**config_data.get('general', {}))
    mask_config = CreateMasksConfig(**config_data.get('createmasks', {}))
    
    # Enable parameter scanning
    mask_config.run_parameter_scan = True
    mask_config.param_a = 'cellprob_threshold'
    mask_config.param_a_values = [-2.0, -1.0, 0.0, 1.0]
    mask_config.param_b = 'flow_threshold'
    mask_config.param_b_values = [0.3, 0.4, 0.5]
    
    # Set the CellPose-SAM model (you can also scan across different models)
    mask_config.cell_pose_sam_model = 'cpsam'  # Default CellPose-SAM model
    # Other options might include: 'nuclei', 'cyto', 'cyto2', etc. (if compatible)
    
    # Limit to specific ROIs for demo (optional)
    # mask_config.specific_rois = ['ROI_001', 'ROI_002']
    
    print("Running CellPose-SAM parameter scan demo...")
    print(f"Parameter A: {mask_config.param_a} = {mask_config.param_a_values}")
    print(f"Parameter B: {mask_config.param_b} = {mask_config.param_b_values}")
    print(f"Total combinations: {len(mask_config.param_a_values) * len(mask_config.param_b_values)}")
    
    # Run parameter scan
    parameter_scan_cpsam(general_config, mask_config)
    
    print("Parameter scan completed!")

if __name__ == "__main__":
    run_parameter_scan_demo()