# Configuration Error Handling

## Problem Solved

When reorganizing configuration structure (moving AI/backgating settings from `general` to `visualization` section), existing config files caused errors:

```
TypeError: GeneralConfig.__init__() got an unexpected keyword argument 'enable_ai'
```

This happened because old config files still had `enable_ai`, `enable_backgating`, etc. under the `general` section, but the new `GeneralConfig` class no longer accepts these keys.

## Solution Implemented

### **Robust Config Object Creation**

Added `filter_config_for_dataclass()` function that:
- Filters config dictionaries to only include valid keys for each dataclass
- Logs clear warnings for unrecognized keys (including their values)
- Prevents TypeErrors when creating config objects
- Does NOT automatically move keys to avoid future compatibility issues

**Example Usage:**
```python
# Old (error-prone)
general_config = GeneralConfig(**config.get('general', {}))

# New (robust)
general_config = GeneralConfig(**filter_config_for_dataclass(config.get('general', {}), GeneralConfig))
```

**Example Warning Output:**
```
WARNING - Ignoring unrecognized config key 'enable_ai' = True in GeneralConfig configuration section. Please check if this key belongs in a different config section.
WARNING - Ignoring unrecognized config key 'enable_backgating' = False in GeneralConfig configuration section. Please check if this key belongs in a different config section.
```

## Error Handling Features

### **Graceful Degradation**
- Unknown keys are filtered out with clear warnings showing key names and values
- Missing sections get default values
- Process continues normally after filtering
- No automatic key migration (prevents future compatibility issues)

### **Clear User Feedback**
- Warnings include both key name and value for easy identification
- Messages suggest checking if keys belong in different sections
- Users can manually update their configs based on the warnings

### **Preservation of Valid Settings**
- All valid configuration values are preserved
- Only invalid/unrecognized keys are filtered out
- Default values only added for truly missing keys

## Benefits

- **Zero Crashes**: Invalid config keys won't cause TypeErrors
- **Clear Feedback**: Users see exactly which keys are problematic and their values
- **Simple Maintenance**: No complex key migration logic to maintain
- **Future-Proof**: Approach works for any configuration restructuring
- **User Control**: Users decide how to handle misplaced keys based on warnings

## User Action Required

When users see warnings like:
```
WARNING - Ignoring unrecognized config key 'enable_ai' = True in GeneralConfig configuration section. Please check if this key belongs in a different config section.
```

They should:
1. **Check the current documentation** for the correct config structure
2. **Move the key to the appropriate section** (e.g., `enable_ai` belongs in `visualization`)
3. **Remove the key from the wrong section**

Example fix:
```yaml
# Before (causes warnings)
general:
  enable_ai: true
  enable_backgating: false

# After (correct)
general:
  # ... only general settings ...

visualization:
  enable_ai: true
  enable_backgating: false
```

This approach ensures robustness while keeping the solution simple and maintainable for future configuration changes.