# Debugging Differential Expression Analysis

## Issue Identified
The differential expression analysis in backgating wasn't running due to several logic issues:

1. **Override Logic**: The code was checking if ANY channel override was specified, including `specify_blue`, which is commonly set to 'DNA1'. This prevented differential expression from running.

2. **Missing Debug Output**: There was no visibility into whether DE was being triggered or not.

## Fixes Applied

### 1. Fixed Override Logic
- Changed from checking `specify_red is None and specify_green is None and specify_blue is None`
- To checking `specify_red is None and specify_green is None`
- **Reasoning**: Blue channel is typically structural (DNA1) and shouldn't prevent DE for functional markers

### 2. Added Comprehensive Debug Output
- Shows configuration being loaded in visualization module
- Shows population processing details in backgating assessment  
- Shows decision logic for when to run differential expression
- Shows detailed DE analysis results

### 3. Force Recalculation in 'full' Mode
- In 'full' mode, always recalculate markers unless Red/Green are explicitly overridden
- In 'save_markers' mode, only calculate if missing
- **Reasoning**: Users expect 'full' mode to recompute everything

## Testing the Fix

### Method 1: Check Debug Output
When running backgating, you should now see output like:
```
Determining top markers for each population using differential expression...
Settings DataFrame shape after initialization: (5, 8)
Population categories to process: ['Pop_1', 'Pop_2', 'Pop_3']
Processing population: Pop_1
  Current Red marker: None
  Is Red marker NaN? True
  Should calculate markers? True
  Starting differential expression analysis for Pop_1
  Input data: 1000 cells, 25 markers
  ...
```

### Method 2: Run Debug Script
Execute the debug script to test DE directly:
```bash
python debug_differential_expression.py
```

### Method 3: Check Settings File
After running backgating, check the `backgating_settings_*.csv` file:
- If DE worked: Markers should be functionally relevant (CD3, CD4, etc.)
- If DE failed: Markers might be alphabetical or based on mean expression only

## Configuration
Your current config shows:
- `backgating_use_differential_expression: true` ✓
- `backgating_mode: full` ✓  
- `backgating_specify_blue: DNA1` ✓ (This should now work correctly)
- `backgating_specify_red: null` ✓ (Allows DE to select)
- `backgating_specify_green: null` ✓ (Allows DE to select)

## Expected Behavior Now
1. **DE Analysis Runs**: Should see detailed output about marker selection
2. **Functional Markers**: Red/Green channels should get biologically relevant markers
3. **Blue Channel**: Should remain as DNA1 as configured
4. **Error Handling**: Falls back gracefully if DE fails

## Next Steps
1. Run backgating and check for DE debug output
2. If still not working, check that scanpy is installed and functioning
3. Verify that the populations in your data have sufficient cells for DE analysis