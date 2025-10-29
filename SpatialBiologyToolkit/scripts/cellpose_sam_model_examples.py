#!/usr/bin/env python3
"""
Example showing different CellPose-SAM model configurations.
"""

from cellpose_sam import process_all_rois, parameter_scan_cpsam
from config_and_utils import (
    GeneralConfig,
    CreateMasksConfig
)

def example_cpsam_model_usage():
    """Demonstrate different model configurations for CellPose-SAM."""
    
    # General configuration
    general_config = GeneralConfig()
    
    # Example 1: Standard CellPose-SAM model
    print("=== Example 1: Standard CellPose-SAM ===")
    config1 = CreateMasksConfig(
        cell_pose_sam_model='cpsam',  # Primary CellPose-SAM model
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        specific_rois=['ROI_001', 'ROI_002']  # Limit for demo
    )
    print(f"Model: {config1.cell_pose_sam_model}")
    
    # Example 2: Using traditional nuclei model with CellPose-SAM workflow
    print("\n=== Example 2: Traditional Nuclei Model ===")
    config2 = CreateMasksConfig(
        cell_pose_sam_model='nuclei',  # Traditional CellPose nuclei model
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        specific_rois=['ROI_001', 'ROI_002']
    )
    print(f"Model: {config2.cell_pose_sam_model}")
    
    # Example 3: Parameter scan comparing models
    print("\n=== Example 3: Model Comparison Parameter Scan ===")
    config3 = CreateMasksConfig(
        run_parameter_scan=True,
        param_a='cell_pose_sam_model',
        param_a_values=['cpsam', 'nuclei'],  # Compare CellPose-SAM vs traditional
        param_b='cellprob_threshold',
        param_b_values=[-1.0, 0.0, 1.0],
        specific_rois=['ROI_001', 'ROI_002']
    )
    print(f"Models to compare: {config3.param_a_values}")
    print(f"Thresholds to test: {config3.param_b_values}")
    print(f"Total combinations: {len(config3.param_a_values) * len(config3.param_b_values)}")
    
    # Example 4: Advanced parameter scan with fixed model
    print("\n=== Example 4: Advanced Parameter Scan (Fixed Model) ===")
    config4 = CreateMasksConfig(
        cell_pose_sam_model='cpsam',  # Fixed to CellPose-SAM
        run_parameter_scan=True,
        param_a='cellprob_threshold',
        param_a_values=[-2.0, -1.0, 0.0, 1.0, 2.0],
        param_b='flow_threshold',
        param_b_values=[0.3, 0.4, 0.5, 0.6],
        specific_rois=['ROI_001', 'ROI_002', 'ROI_003']
    )
    print(f"Fixed model: {config4.cell_pose_sam_model}")
    print(f"Total combinations: {len(config4.param_a_values) * len(config4.param_b_values)}")
    
    # Demonstrate usage (commented out to avoid actual execution)
    # process_all_rois(general_config, config1)
    # parameter_scan_cpsam(general_config, config3)
    
    return config1, config2, config3, config4

def create_example_config_yaml():
    """Create example YAML configuration files."""
    
    # Standard configuration
    standard_config = """
# Standard CellPose-SAM Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Model Configuration
  cell_pose_model: 'nuclei'           # For CellPose v3 script
  cell_pose_sam_model: 'cpsam'        # For CellPose-SAM script (DEFAULT)
  
  # Basic Parameters
  cellprob_threshold: 0.0
  flow_threshold: 0.4
  cellpose_cell_diameter: 10.0
  
  # Processing Options
  run_deblur: true
  run_upscale: true
  perform_qc: true
  
  # Size Filtering
  min_cell_area: 15
  max_size_fraction: 0.4
  expand_masks: 1
  
  # Parameter Scanning (disabled by default)
  run_parameter_scan: false
"""
    
    # Model comparison configuration
    model_comparison_config = """
# Model Comparison Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Parameter Scanning: Compare Models
  run_parameter_scan: true
  param_a: 'cell_pose_sam_model'
  param_a_values: ['cpsam', 'nuclei', 'cyto2']
  param_b: 'cellprob_threshold'
  param_b_values: [-2.0, -1.0, 0.0, 1.0]
  
  # Fixed Parameters
  flow_threshold: 0.4
  cellpose_cell_diameter: 10.0
  
  # Limit ROIs for testing
  specific_rois: ['ROI_001', 'ROI_002', 'ROI_003']
"""
    
    # Advanced parameter scan
    advanced_scan_config = """
# Advanced Parameter Scan Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Fixed Model
  cell_pose_sam_model: 'cpsam'
  
  # Parameter Scanning: Optimize Thresholds
  run_parameter_scan: true
  param_a: 'cellprob_threshold'
  param_a_values: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
  param_b: 'flow_threshold'
  param_b_values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  
  # Fixed Parameters
  cellpose_cell_diameter: 10.0
  run_deblur: true
  run_upscale: true
  
  # Process all ROIs
  # specific_rois: null  # Will process all available ROIs
"""
    
    return {
        'standard': standard_config,
        'model_comparison': model_comparison_config,
        'advanced_scan': advanced_scan_config
    }

if __name__ == "__main__":
    print("CellPose-SAM Model Configuration Examples")
    print("=" * 50)
    
    # Show configuration examples
    configs = example_cpsam_model_usage()
    
    # Show YAML examples
    print("\n=== Example YAML Configurations ===")
    yaml_configs = create_example_config_yaml()
    
    for name, config_text in yaml_configs.items():
        print(f"\n--- {name.replace('_', ' ').title()} Configuration ---")
        print(config_text[:200] + "..." if len(config_text) > 200 else config_text)
    
    print("\nSee CELLPOSE_MODEL_CONFIG.md for complete documentation.")