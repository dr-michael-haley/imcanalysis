# CellPose Diameter Parameter Compatibility Guide

## Summary

The `diameter` parameter is **still supported** in CellPose v4.0.1+, but its internal behavior has changed. Our scripts maintain backward compatibility while accounting for these differences.

## Diameter Parameter Behavior

### CellPose v3 (Original)
- `diameter` directly affects model inference
- Used for flow computation and dynamics
- Scaling was more implicit

### CellPose v4+ (Current)
- `diameter` is used for **explicit image scaling**
- Formula: `scaling_factor = 30.0 / diameter`
- Image is rescaled before model inference
- Results are then rescaled back to original size

## Our Implementation

### Configuration (Unchanged)
```yaml
createmasks:
  cellpose_cell_diameter: 10.0  # Works in both v3 and v4+
```

### Internal Handling (Updated)
```python
# Calculate diameter for segmentation
diameter_for_segmentation = config.cellpose_cell_diameter
if config.run_upscale:
    # Use actual upscale target diameter (17.0 for nuclei)
    diameter_for_segmentation = config.upscale_target_diameter

# CellPose v4+ will internally scale image by: 30.0 / diameter_for_segmentation
cellpose_scaling = 30.0 / diameter_for_segmentation
```

## Practical Examples

### Example 1: No Upscaling
- `cellpose_cell_diameter: 10.0`
- `diameter_for_segmentation: 10.0`
- **CellPose internal scaling**: `30.0 / 10.0 = 3.0x`

### Example 2: With Upscaling (upsample_nuclei)
- `cellpose_cell_diameter: 10.0` (base)
- `upscale_target_diameter: 17.0` (from upsample_nuclei)
- `diameter_for_segmentation: 17.0`
- **CellPose internal scaling**: `30.0 / 17.0 = 1.76x`

## Compatibility Benefits

### Backward Compatibility
- Scripts work with both CellPose v3 and v4+
- Same configuration files work across versions
- Diameter values have consistent meaning to users

### Forward Compatibility  
- Takes advantage of v4+ improvements
- Accounts for new scaling behavior
- Provides detailed logging for debugging

## Results Tracking

Our updated results now include:
```csv
Diameter_used,Diameter_base,CellPose_scaling_factor,Upscale_ratio,Upscale_target_diameter
17.0,10.0,1.76,1.7,17.0
```

This allows you to:
1. **Verify scaling factors** match expectations
2. **Debug diameter issues** across CellPose versions
3. **Compare results** between v3 and v4+ workflows

## Migration Notes

### If upgrading from CellPose v3 to v4+:
1. **Keep same diameter values** in configuration
2. **Check results CSV** for new scaling factors
3. **Compare segmentation quality** - v4+ should be similar or better
4. **Monitor log messages** for any warnings

### If you see unexpected results:
1. **Check `CellPose_scaling_factor`** in results CSV
2. **Verify `Diameter_used`** matches expectations
3. **Compare with `Upscale_target_diameter`** if using upscaling
4. **Review log messages** for scaling information

## Best Practices

1. **Use consistent CellPose version** across entire pipeline
2. **Test on small subset** when changing versions
3. **Compare QC images** before and after version changes
4. **Document CellPose version** in your analysis notes

The diameter parameter continues to be a crucial tuning parameter for segmentation quality, regardless of CellPose version.