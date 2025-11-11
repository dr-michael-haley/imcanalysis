# Logging Configuration Guide

The logging system has been enhanced to provide better control over output formatting and prevent duplicate console messages.

## New Configuration Options

Add these options to your `config.yaml` under the `logging` section:

```yaml
logging:
  log_file: 'pipeline.log'          # File to write logs to
  level: 'INFO'                     # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  to_console: true                  # Whether to output to console/terminal
  console_only: false               # Only log to console, skip file logging
  prevent_duplicate_console: true   # Prevent duplicate console output (recommended)
  use_custom_format: true          # Use custom timestamp format vs basic format
```

## Usage Scenarios

### 1. Standard Development (Default)
```yaml
logging:
  log_file: 'pipeline.log'
  level: 'INFO'
  to_console: true
  console_only: false
  prevent_duplicate_console: true
```
- Logs to both file and console
- No duplicate messages
- Custom timestamp format

### 2. SLURM Jobs (Clean Output)
```yaml
logging:
  console_only: true
  to_console: true
  prevent_duplicate_console: true
  level: 'INFO'
```
- Only outputs to console (SLURM captures this)
- No file creation on compute nodes
- Single, clean log messages

### 3. File-Only Logging
```yaml
logging:
  log_file: 'detailed.log'
  to_console: false
  console_only: false
  prevent_duplicate_console: true
  level: 'DEBUG'
```
- Only writes to file
- No console output
- Useful for batch processing

### 4. Debug Mode
```yaml
logging:
  log_file: 'debug.log'
  level: 'DEBUG'
  to_console: true
  console_only: false
  prevent_duplicate_console: true
```
- Maximum verbosity
- Both file and console output
- Detailed debugging information

## Command Line Overrides

You can override logging settings from the command line:

```bash
# SLURM-friendly logging
python scripts/basic_visualizations.py --override logging.console_only=true

# Debug mode
python scripts/basic_visualizations.py --override logging.level=DEBUG

# No console output
python scripts/basic_visualizations.py --override logging.to_console=false
```

## Migration from Old Config

If you have an existing config file, the system will automatically add the new options with sensible defaults:

- `console_only: false` - Maintains file logging
- `prevent_duplicate_console: true` - Fixes double output issue
- `use_custom_format: true` - Keeps current timestamp format

## Troubleshooting

### Still Seeing Double Output?
Set `prevent_duplicate_console: true` (this is the default for new configs).

### No Console Output in SLURM?
Try setting `console_only: true` to avoid file system issues on compute nodes.

### Too Verbose?
Change `level` from `'INFO'` to `'WARNING'` or `'ERROR'`.

### Want Timestamped vs Simple Output?
Toggle `use_custom_format` between `true` (timestamps) and `false` (simple).

## Technical Details

The fix addresses the double logging issue by:
1. Clearing existing handlers before setup
2. Not using `logging.basicConfig(filename=...)` which auto-creates console handlers
3. Manually creating file and console handlers as needed
4. Setting `propagate=False` to prevent message bubbling

This gives you full control over logging behavior while maintaining backward compatibility.