# CellPose Model Configuration Update

## Overview

The configuration has been updated to support separate model specifications for CellPose v3 (original `createmasks_cellpose3.py`) and CellPose-SAM (`cellpose_sam.py`) workflows.

## New Configuration Fields

### CreateMasksConfig

Two separate model configuration fields are now available:

```python
@dataclass
class CreateMasksConfig:
    # ... other fields ...
    cell_pose_model: str = 'nuclei'          # For CellPose v3 (original createmasks)
    cell_pose_sam_model: str = 'cpsam'       # For CellPose-SAM (cellpose_sam script)
    # ... other fields ...
```

## Usage

### Configuration File (config.yaml)

```yaml
createmasks:
  # For original CellPose v3 script (createmasks_cellpose3.py)
  cell_pose_model: 'nuclei'          # Options: 'nuclei', 'cyto', 'cyto2', etc.
  
  # For CellPose-SAM script (cellpose_sam.py)
  cell_pose_sam_model: 'cpsam'       # Options: 'cpsam', 'nuclei', 'cyto', etc.
  
  # Other parameters...
  cellprob_threshold: 0.0
  flow_threshold: 0.4
```

### Command Line Overrides

```bash
# Set CellPose-SAM model
python cellpose_sam.py --override createmasks.cell_pose_sam_model=nuclei

# Set CellPose v3 model
python createmasks_cellpose3.py --override createmasks.cell_pose_model=cyto2
```

## Available Models

### CellPose-SAM Compatible Models

**Primary CellPose-SAM Model**:
- `cpsam` - The main CellPose-SAM model (default)

**CellPose v3 Models (compatible with CellPose v4)**:
- `nuclei` - Nuclear segmentation
- `cyto` - Cytoplasm segmentation  
- `cyto2` - Improved cytoplasm segmentation
- `cyto3` - Latest cytoplasm model (if available)

**Note**: CellPose v3 models can run on CellPose v4, but CellPose v4 models cannot run on CellPose v3.

### Model Selection Guidelines

**For Nuclear Segmentation**:
- Primary choice: `cpsam` (best performance for challenging nuclei)
- Alternative: `nuclei` (standard CellPose nuclear model)

**For Cytoplasm Segmentation**:
- Use: `cyto2` or `cyto3` (CellPose-SAM doesn't have dedicated cytoplasm models)

## Parameter Scanning with Models

You can now include model type as a parameter to scan:

### Example 1: Model Comparison

```yaml
createmasks:
  run_parameter_scan: true
  param_a: 'cell_pose_sam_model'
  param_a_values: ['cpsam', 'nuclei', 'cyto2']
  param_b: 'cellprob_threshold'
  param_b_values: [-2.0, -1.0, 0.0, 1.0]
```

### Example 2: Traditional Parameter Scan

```yaml
createmasks:
  cell_pose_sam_model: 'cpsam'       # Fixed model
  run_parameter_scan: true
  param_a: 'cellprob_threshold'
  param_a_values: [-2.0, -1.0, 0.0, 1.0]
  param_b: 'flow_threshold'
  param_b_values: [0.3, 0.4, 0.5]
```

## Benefits

1. **Flexibility**: Can use any CellPose v3 model with CellPose-SAM workflow
2. **Comparison**: Easy to compare CellPose-SAM vs traditional models
3. **Backward Compatibility**: Existing configurations continue to work
4. **Clear Separation**: Distinct model settings for different workflows
5. **Parameter Scanning**: Can optimize both parameters and model selection

## Migration Guide

**Existing Users**: No changes required - default values maintain current behavior.

**New Users**: 
- Use `cell_pose_sam_model: 'cpsam'` for best CellPose-SAM performance
- Consider `cell_pose_sam_model: 'nuclei'` for comparison with traditional methods

## Technical Details

**Model Initialization**: The CellPose-SAM script now uses:
```python
cp_sam_model = models.CellposeModel(
    pretrained_model=config.cell_pose_sam_model,
    gpu=use_gpu
)
```

**Parameter Scanning**: When `cell_pose_sam_model` is a scan parameter, the model is reinitialized for each parameter combination to ensure proper model loading.

**Result Tracking**: The `Model_type` field in results now shows the actual model used rather than hardcoded 'cpsam'.