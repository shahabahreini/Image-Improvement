# Feature Implementation: Filter Selection for Tile Processing

## Architecture Overview

```
LargeImage_Tiler.py (split command)
    ↓
    ├─→ parse arguments (new: --filter, --filter-args, --create-script, --custom-command)
    ↓
split_image()
    ↓
create_processing_script(filter_type, filter_args)
    ├─→ filter_type == "nlm"
    │   └─→ NLM_Filter .py command
    ├─→ filter_type == "bm3d"
    │   └─→ Basic_BM3D.py command
    ├─→ filter_type == "enhanced_bm3d"
    │   └─→ Enhanced_BM3D.py command
    └─→ filter_type == "custom"
        └─→ custom_command
    ↓
Generate process_tiles.sh with proper commands
    ↓
User runs: ./process_tiles.sh
    ↓
    Each tile gets processed with selected filter
    ↓
Merge processed tiles: merge command
    ↓
Final output
```

## New Parameters Added to create_processing_script()

```python
def create_processing_script(
    self,
    tiles_dir: str,
    denoising_command: str,
    filter_type: str = "enhanced_bm3d",        # NEW: filter selection
    filter_args: Optional[str] = None,          # NEW: filter arguments
) -> str:
```

## New Command-Line Arguments

```
Split Command:
  --filter {nlm,bm3d,enhanced_bm3d,custom}
      Which denoising filter to use
      Default: enhanced_bm3d

  --filter-args ARGS
      Additional arguments for the filter
      Example: "--workers 2 --tile-size 2048"

  --create-script
      Auto-generate processing script
      Creates: process_tiles.sh

  --custom-command CMD
      Custom command (required if --filter custom)
      Placeholder: {input}, {output}
```

## Generated Script Structure

```bash
#!/bin/bash
# Auto-generated tile processing script
# Filter type: enhanced_bm3d
# Found 48 tiles to process

set -e  # Exit on any error

echo "Processing tile 1/48: tile_0000_0000.tif"
python3 "Enhanced_BM3D.py" "tile_0000_0000.tif" "denoised_tile_0000_0000.tif" --profile refilter
if [ $? -ne 0 ]; then
    echo "Failed to process tile_0000_0000.tif"
    exit 1
fi

echo "Processing tile 2/48: tile_0000_0001.tif"
python3 "Enhanced_BM3D.py" "tile_0000_0001.tif" "denoised_tile_0000_0001.tif" --profile refilter
if [ $? -ne 0 ]; then
    echo "Failed to process tile_0000_0001.tif"
    exit 1
fi

... (more tiles)

echo "All tiles processed successfully!"
```

## Filter Command Templates

### NLM Filter

```
Input:  tile_0000_0000.tif
Output: denoised_tile_0000_0000.tif
Command: python3 "NLM_Filter .py" "tile_0000_0000.tif" "denoised_tile_0000_0000.tif" nlm_ultrafast
```

### BM3D Filter

```
Input:  tile_0000_0000.tif
Output: denoised_tile_0000_0000.tif
Command: python3 "Basic_BM3D.py" "tile_0000_0000.tif" "denoised_tile_0000_0000.tif" --profile refilter
```

### Enhanced BM3D Filter (Default)

```
Input:  tile_0000_0000.tif
Output: denoised_tile_0000_0000.tif
Command: python3 "Enhanced_BM3D.py" "tile_0000_0000.tif" "denoised_tile_0000_0000.tif" --profile refilter
```

### Custom Filter

```
Input:  tile_0000_0000.tif
Output: denoised_tile_0000_0000.tif
Command: {your custom command with {input} and {output} replaced}
```

## Data Flow Examples

### Example 1: Default Enhanced BM3D

```
split image.tif tiles/ --create-script
    ↓
Generates: process_tiles.sh with Enhanced_BM3D.py commands
    ↓
cd tiles/ && ./process_tiles.sh
    ↓
Creates: denoised_tile_*.tif files
```

### Example 2: NLM Filter with Arguments

```
split image.tif tiles/ --filter nlm --filter-args "--fast" --create-script
    ↓
Generates: process_tiles.sh with NLM_Filter .py commands
    ↓
Each command includes: nlm_ultrafast --fast
```

### Example 3: Custom Filter

```
split image.tif tiles/ --filter custom --custom-command "my_tool {input} {output}" --create-script
    ↓
Generates: process_tiles.sh with custom command
    ↓
Each command: my_tool tile_*.tif denoised_tile_*.tif
```

## Class Method Signature Changes

### Before

```python
def create_processing_script(self, tiles_dir: str, denoising_command: str) -> str:
```

### After

```python
def create_processing_script(
    self,
    tiles_dir: str,
    denoising_command: str,
    filter_type: str = "enhanced_bm3d",
    filter_args: Optional[str] = None,
) -> str:
```

## Backward Compatibility

- If `--filter` is not specified, defaults to `enhanced_bm3d`
- If `--create-script` is not used, no processing script is generated
- Existing code still works as-is
- All parameters are optional with sensible defaults

## Files Structure After Implementation

```
Image Improvement/
├── LargeImage_Tiler.py          (modified)
├── NLM_Filter .py                (unchanged)
├── Basic_BM3D.py                 (unchanged)
├── Enhanced_BM3D.py              (unchanged)
├── FILTER_SELECTION_GUIDE.md     (new)
├── QUICK_REFERENCE.md            (new)
├── IMPLEMENTATION_SUMMARY.md     (new)
└── example_filter_usage.sh       (new)
```

## Test Scenarios

1. ✓ Default behavior (no filter args)
2. ✓ NLM filter selection
3. ✓ BM3D filter selection
4. ✓ Enhanced BM3D selection (default)
5. ✓ Custom filter with command
6. ✓ Filter arguments passing
7. ✓ Script generation
8. ✓ Merge operation (unchanged)
