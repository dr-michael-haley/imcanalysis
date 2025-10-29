#!/usr/bin/env python3
"""
Demo: CellPose-SAM Model Parameter Scanning

This script demonstrates how to set up parameter scanning across different
CellPose models using the CellPose-SAM workflow.
"""

from cellpose_sam import parameter_scan_cpsam
from config_and_utils import (
    process_config_with_overrides,
    setup_logging,
    GeneralConfig,
    CreateMasksConfig
)

def demo_model_parameter_scan():
    """Demonstrate model parameter scanning configurations."""
    
    print("CellPose-SAM Model Parameter Scanning Demo")
    print("=" * 50)
    
    # Load base configuration
    config_data = process_config_with_overrides()
    setup_logging(config_data.get('logging', {}), 'CellposeSAM_ModelScan_Demo')
    
    general_config = GeneralConfig(**config_data.get('general', {}))
    
    # Example 1: Compare CellPose-SAM vs Traditional Models
    print("\n=== Example 1: Model Comparison ===")
    config1 = CreateMasksConfig(**config_data.get('createmasks', {}))
    config1.run_parameter_scan = True
    config1.param_a = 'cell_pose_sam_model'
    config1.param_a_values = ['cpsam', 'nuclei']  # Compare main models
    config1.param_b = 'cellprob_threshold'
    config1.param_b_values = [-1.0, 0.0, 1.0]
    config1.specific_rois = ['ROI_001', 'ROI_002']  # Limit for demo
    
    print(f"Models to compare: {config1.param_a_values}")
    print(f"Thresholds to test: {config1.param_b_values}")
    print(f"Total combinations: {len(config1.param_a_values) * len(config1.param_b_values)}")
    
    # Example 2: Comprehensive Model + Threshold Scan
    print("\n=== Example 2: Comprehensive Model Scan ===")
    config2 = CreateMasksConfig(**config_data.get('createmasks', {}))
    config2.run_parameter_scan = True
    config2.param_a = 'cell_pose_sam_model'
    config2.param_a_values = ['cpsam', 'nuclei', 'cyto2']  # Multiple models
    config2.param_b = 'cellprob_threshold'
    config2.param_b_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    config2.specific_rois = ['ROI_001', 'ROI_002', 'ROI_003']
    
    print(f"Models to compare: {config2.param_a_values}")
    print(f"Thresholds to test: {config2.param_b_values}")
    print(f"Total combinations: {len(config2.param_a_values) * len(config2.param_b_values)}")
    
    # Example 3: Model + Flow Threshold
    print("\n=== Example 3: Model + Flow Threshold ===")
    config3 = CreateMasksConfig(**config_data.get('createmasks', {}))
    config3.run_parameter_scan = True
    config3.param_a = 'cell_pose_sam_model'
    config3.param_a_values = ['cpsam', 'nuclei']
    config3.param_b = 'flow_threshold'
    config3.param_b_values = [0.3, 0.4, 0.5, 0.6]
    config3.cellprob_threshold = 0.0  # Fixed
    config3.specific_rois = ['ROI_001', 'ROI_002']
    
    print(f"Models to compare: {config3.param_a_values}")
    print(f"Flow thresholds to test: {config3.param_b_values}")
    print(f"Fixed cellprob_threshold: {config3.cellprob_threshold}")
    print(f"Total combinations: {len(config3.param_a_values) * len(config3.param_b_values)}")
    
    # Show expected outputs
    print("\n=== Expected Outputs ===")
    print("Folder structure will be created as:")
    print("QC/")
    print("├── CellposeSAM_ParameterScan_cell_pose_sam_model_cellprob_threshold/")
    print("│   ├── param_cell_pose_sam_model-cpsam_cellprob_threshold--1_0/")
    print("│   ├── param_cell_pose_sam_model-cpsam_cellprob_threshold-0_0/")
    print("│   ├── param_cell_pose_sam_model-nuclei_cellprob_threshold--1_0/")
    print("│   ├── param_cell_pose_sam_model-nuclei_cellprob_threshold-0_0/")
    print("│   ├── CellposeSAM_ParameterScan_All.csv")
    print("│   ├── ParameterScan_Objects_kept.png")
    print("│   ├── ParameterScan_Objects_per_mm2.png")
    print("│   └── ParameterScan_Objects_Kept_Heatmap.png")
    print("masks/")
    print("├── param_cell_pose_sam_model-cpsam_cellprob_threshold--1_0/")
    print("├── param_cell_pose_sam_model-cpsam_cellprob_threshold-0_0/")
    print("└── ...")
    
    # Demonstrate how to run (commented out to avoid actual execution)
    print("\n=== How to Run ===")
    print("# To run Example 1:")
    print("# parameter_scan_cpsam(general_config, config1)")
    
    print("\n# To run Example 2:")
    print("# parameter_scan_cpsam(general_config, config2)")
    
    print("\n# To run Example 3:")
    print("# parameter_scan_cpsam(general_config, config3)")
    
    return config1, config2, config3

def create_model_scan_config_examples():
    """Create example configuration files for model parameter scanning."""
    
    # Basic model comparison
    basic_config = """
# Basic Model Comparison Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Basic Model Parameter Scan
  run_parameter_scan: true
  param_a: 'cell_pose_sam_model'
  param_a_values: ['cpsam', 'nuclei']
  param_b: 'cellprob_threshold'
  param_b_values: [-1.0, 0.0, 1.0]
  
  # Fixed parameters
  flow_threshold: 0.4
  cellpose_cell_diameter: 10.0
  
  # Limit ROIs for testing
  specific_rois: ['ROI_001', 'ROI_002', 'ROI_003']
"""
    
    # Comprehensive model scan
    comprehensive_config = """
# Comprehensive Model Scanning Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Comprehensive Model + Threshold Scan
  run_parameter_scan: true
  param_a: 'cell_pose_sam_model'
  param_a_values: ['cpsam', 'nuclei', 'cyto2']
  param_b: 'cellprob_threshold'
  param_b_values: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
  
  # Fixed parameters
  flow_threshold: 0.4
  cellpose_cell_diameter: 10.0
  run_deblur: true
  run_upscale: true
  
  # Process all ROIs (comment out specific_rois to process all)
  # specific_rois: null
"""
    
    # Model + flow threshold scan
    flow_config = """
# Model + Flow Threshold Configuration
general:
  masks_folder: 'masks'
  qc_folder: 'QC'
  denoised_images_folder: 'processed'

createmasks:
  # Model + Flow Threshold Scan
  run_parameter_scan: true
  param_a: 'cell_pose_sam_model'
  param_a_values: ['cpsam', 'nuclei']
  param_b: 'flow_threshold'
  param_b_values: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  
  # Fixed parameters
  cellprob_threshold: 0.0
  cellpose_cell_diameter: 10.0
  
  # Limit ROIs for testing
  specific_rois: ['ROI_001', 'ROI_002']
"""
    
    return {
        'basic_model_comparison': basic_config,
        'comprehensive_model_scan': comprehensive_config,
        'model_flow_threshold': flow_config
    }

if __name__ == "__main__":
    # Run the demo
    configs = demo_model_parameter_scan()
    
    # Show configuration file examples
    print("\n" + "=" * 50)
    print("YAML Configuration Examples")
    print("=" * 50)
    
    yaml_configs = create_model_scan_config_examples()
    
    for name, config_text in yaml_configs.items():
        print(f"\n--- {name.replace('_', ' ').title()} ---")
        print(config_text)
    
    print("\nTo use any configuration:")
    print("1. Save the YAML content to a file (e.g., 'model_scan_config.yaml')")
    print("2. Run: python cellpose_sam.py --config model_scan_config.yaml")
    print("\nOr use command line overrides:")
    print("python cellpose_sam.py --override createmasks.param_a=cell_pose_sam_model \\")
    print("                       --override createmasks.param_a_values=cpsam,nuclei \\")
    print("                       --override createmasks.run_parameter_scan=true")