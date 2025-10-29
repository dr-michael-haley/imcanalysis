# CellPose v4.0.1+ Model Compatibility Changes

## Important Breaking Changes

CellPose v4.0.1+ has introduced significant changes to the model API that affect our segmentation scripts.

## What Changed

### 1. Deprecated Parameters
- `model_type` parameter is **completely ignored** in v4.0.1+
- `rescale` parameter is **deprecated** in v4.0.1+
- Only `pretrained_model` parameter is used for model loading

### 2. Changed Parameters  
- `diameter` parameter **still works** but behavior may differ
- In v4+: `diameter` is used for image scaling (scaling = 30.0 / diameter)
- In v3: `diameter` was used more directly for model inference

### 3. Available Models
**CellPose v3 (OLD):**
- `'nuclei'`, `'cyto'`, `'cyto2'`, `'livecell'` - built-in models
- `'cpsam'` - CellPose-SAM model

**CellPose v4.0.1+ (NEW):**
- `'cpsam'` - **ONLY** built-in model available
- User-trained models (via full file paths)

### 3. Traditional Models No Longer Available
The following models from CellPose v3 are **no longer available**:
- `'nuclei'`
- `'cyto'`
- `'cyto2'`
- `'livecell'`

## Impact on Our Scripts

### cellpose_sam.py ✅ FIXED
- Updated `load_cellpose_model()` to use only `pretrained_model` parameter
- Added fallback logic: if traditional model names are specified, warns and uses `'cpsam'`
- Compatible with CellPose v4.0.1+

### createmasks_cellpose3.py ⚠️ INCOMPATIBLE
- Still uses old `model_type` parameter
- Will work with CellPose v3 but NOT with v4.0.1+
- Consider this script **deprecated** for new installations

## Configuration Changes

### Before (CellPose v3):
```yaml
createmasks:
  cell_pose_model: 'nuclei'        # Works in v3
  cell_pose_sam_model: 'cpsam'     # Works in v3 and v4+
```

### After (CellPose v4.0.1+):
```yaml
createmasks:
  cell_pose_model: 'nuclei'        # IGNORED in v4+ (kept for compatibility)
  cell_pose_sam_model: 'cpsam'     # ONLY option in v4+
```

## Recommendations

### For New Projects
1. **Use CellPose v4.0.1+** with `cellpose_sam.py` script
2. **Use CellPose-SAM model** (`'cpsam'`) which provides better segmentation
3. **Avoid** `createmasks_cellpose3.py` script

### For Existing Projects
1. **Option A:** Upgrade to CellPose v4.0.1+ and use `cellpose_sam.py`
2. **Option B:** Stay with CellPose v3 and use `createmasks_cellpose3.py`
3. **Do NOT mix** - use consistent CellPose version across entire pipeline

### For Custom Models
If you have user-trained models:
1. **CellPose v3:** Use `model_type='/path/to/model'`
2. **CellPose v4+:** Use `pretrained_model='/path/to/model'`

## Version Detection

To check your CellPose version:
```python
import cellpose
print(cellpose.__version__)
```

## Migration Path

### If you're currently using:
```yaml
createmasks:
  cell_pose_model: 'nuclei'
```

### Change to:
```yaml
createmasks:
  cell_pose_sam_model: 'cpsam'
  run_parameter_scan: false  # Use cellpose_sam.py instead
```

And use the `cellpose_sam.py` script instead of `createmasks_cellpose3.py`.

## Performance Notes

CellPose-SAM (`'cpsam'`) typically provides:
- **Better accuracy** than traditional models
- **Similar speed** to CellPose v3 models
- **More robust** segmentation on challenging data
- **GPU acceleration** for faster processing

## Troubleshooting

### Error: "model_type argument is not used in v4.0.1+"
- **Cause:** Using old API with CellPose v4+
- **Fix:** Update scripts to use `pretrained_model` parameter only

### Warning: "pretrained model nuclei not found, using default model"
- **Cause:** Trying to load traditional model in CellPose v4+
- **Fix:** Use `'cpsam'` model or provide full path to user model

### Error: Traditional model not working
- **Cause:** CellPose v4+ doesn't include traditional models
- **Fix:** Either downgrade to v3 or switch to CellPose-SAM workflow

This change ensures our pipeline is future-compatible and takes advantage of the latest CellPose improvements.