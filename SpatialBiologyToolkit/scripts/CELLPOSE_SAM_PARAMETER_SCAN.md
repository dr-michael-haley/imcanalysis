# CellPose-SAM Parameter Scan Implementation

## Overview

I've successfully added parameter scan functionality to the `cellpose_sam.py` script that's compatible with the original `createmasks_cellpose3.py` parameter scanning approach, but simplified for CellPose-SAM usage.

## Key Features Added

### 1. Parameter Scan Function (`parameter_scan_cpsam`)

**Purpose**: Run multiple parameter combinations on all ROIs and compare segmentation performance.

**Key Differences from Original**:
- **Simpler approach**: Processes ALL ROIs (no random sampling)
- **Organized output**: Creates separate mask and QC folders for each parameter combination
- **Enhanced plotting**: Creates both bar plots and heatmaps for performance comparison

### 2. Backward Compatibility

**Configuration**: Reuses existing parameter scan configuration entries:
- `run_parameter_scan`: Boolean to enable parameter scanning
- `param_a` / `param_a_values`: First parameter and its test values
- `param_b` / `param_b_values`: Second parameter and its test values
- `specific_rois`: Optional list to limit ROIs (if not set, uses all ROIs)

### 3. Enhanced Output Structure

**Folder Organization**:
```
QC/
└── CellposeSAM_ParameterScan_{param_a}_{param_b}/
    ├── param_{param_a}-{value1}_{param_b}-{value1}/
    │   ├── CellposeSAM_overlay/
    │   ├── CellposeSAM_raw_overlay/
    │   └── CellposeSAM_results_*.csv
    ├── param_{param_a}-{value1}_{param_b}-{value2}/
    │   └── ...
    ├── CellposeSAM_ParameterScan_All.csv  # Combined results
    ├── ParameterScan_Objects_kept.png      # Bar plots
    ├── ParameterScan_Objects_per_mm2.png
    ├── ParameterScan_Objects_excluded.png
    └── ParameterScan_Objects_Kept_Heatmap.png

masks/
├── param_{param_a}-{value1}_{param_b}-{value1}/
│   ├── ROI_001.tiff
│   ├── ROI_002.tiff
│   └── ...
├── param_{param_a}-{value1}_{param_b}-{value2}/
│   └── ...
```

## Usage Examples

### Basic Parameter Scan in Config

```yaml
createmasks:
  run_parameter_scan: true
  param_a: 'cellprob_threshold'
  param_a_values: [-2.0, -1.0, 0.0, 1.0]
  param_b: 'flow_threshold' 
  param_b_values: [0.3, 0.4, 0.5]
  specific_rois: ['ROI_001', 'ROI_002']  # Optional: limit ROIs for testing
```

### Command Line Usage

```bash
# Run parameter scan with config file
python cellpose_sam.py --config config.yaml

# Run parameter scan with overrides
python cellpose_sam.py --override createmasks.run_parameter_scan=true \
                       --override createmasks.param_a=cellprob_threshold \
                       --override createmasks.param_a_values=-2.0,-1.0,0.0,1.0
```

## Available Parameters for Scanning

**Most Useful Parameters**:
- `cellprob_threshold`: Cell probability threshold (-6.0 to 6.0)
- `flow_threshold`: Flow error threshold (0.0 to 3.0)
- `cellpose_cell_diameter`: Expected cell diameter in pixels
- `max_size_fraction`: Maximum cell size as fraction of image
- `min_cell_area`: Minimum cell area in pixels
- `expand_masks`: Mask expansion distance in pixels

## Output Analysis

### 1. Results CSV Files
- **Individual parameter sets**: `CellposeSAM_results_{param_string}.csv`
- **Combined results**: `CellposeSAM_ParameterScan_All.csv`

### 2. Summary Plots
- **Bar plots**: Show mean ± standard deviation for each metric
- **Heatmap**: 2D visualization of parameter combinations
- **Metrics tracked**: Objects kept, objects per mm², objects excluded

### 3. Performance Comparison
The combined CSV allows easy comparison of:
- Segmentation quality across parameter combinations
- ROI-to-ROI consistency
- Optimal parameter selection

## Integration with Existing Workflow

**Seamless Integration**: 
- Uses same configuration structure as original script
- Maintains all existing functionality when `run_parameter_scan=false`
- Compatible with existing preprocessing and analysis pipelines

**Decision Logic**:
```python
if (mask_config.run_parameter_scan and
    mask_config.param_a and mask_config.param_a_values and
    mask_config.param_b and mask_config.param_b_values):
    parameter_scan_cpsam(general_config, mask_config)
else:
    process_all_rois(general_config, mask_config)  # Normal mode
```

## Benefits

1. **Systematic optimization**: Test multiple parameter combinations systematically
2. **Comprehensive evaluation**: Processes all ROIs (no sampling bias)
3. **Visual comparison**: Automated generation of comparison plots
4. **Organized output**: Clear folder structure for each parameter combination
5. **Data-driven decisions**: Quantitative metrics for parameter selection
6. **Backward compatibility**: Works with existing configuration files

This implementation provides a robust parameter optimization framework for CellPose-SAM segmentation while maintaining the simplicity and effectiveness of the original approach.