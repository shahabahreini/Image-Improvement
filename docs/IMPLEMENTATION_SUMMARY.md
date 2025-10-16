# Filter Selection Implementation Summary

## What Was Added

Enhanced `LargeImage_Tiler.py` with filter selection capability for tile processing.

## Key Changes

### 1. New Command-Line Arguments (Split Command)

```
--filter {nlm,bm3d,enhanced_bm3d,custom}
    Choose which denoising filter to use (default: enhanced_bm3d)

--filter-args ARGS
    Additional arguments to pass to the selected filter

--create-script
    Create an automated processing script for all tiles

--custom-command CMD
    Custom command to use when --filter is set to custom
```

### 2. Updated `create_processing_script()` Method

Now accepts:

- `filter_type`: Type of filter (nlm, bm3d, enhanced_bm3d, or custom)
- `filter_args`: Optional additional arguments for the filter

Automatically generates correct command syntax for each filter:

- **NLM**: `python3 "NLM_Filter .py" "{input}" "{output}" nlm_ultrafast`
- **BM3D**: `python3 "Basic_BM3D.py" "{input}" "{output}" --profile refilter`
- **Enhanced BM3D**: `python3 "Enhanced_BM3D.py" "{input}" "{output}" --profile refilter`
- **Custom**: Uses provided command with `{input}` and `{output}` placeholders

### 3. Enhanced Processing Script Generation

Generated `process_tiles.sh` now includes:

- Filter type in header comments
- Proper command formatting for each filter type
- Error handling and progress reporting
- Executable permissions set automatically

## Usage Examples

### Quick Start (Using Default Enhanced BM3D)

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --create-script
cd tiles/
./process_tiles.sh
cd ..
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif
```

### Using NLM Filter (Fastest)

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter nlm \
  --create-script
```

### Using BM3D Filter

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter bm3d \
  --create-script
```

### Using Custom Filter

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter custom \
  --custom-command "python3 my_denoise.py {input} {output}" \
  --create-script
```

### With Filter Arguments

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter enhanced_bm3d \
  --filter-args "--workers 2 --tile-size 2048" \
  --create-script
```

## Files Modified

- `LargeImage_Tiler.py`: Added filter selection logic and command generation

## Files Created

- `FILTER_SELECTION_GUIDE.md`: Comprehensive guide with examples and best practices
- `example_filter_usage.sh`: Quick reference examples
- `IMPLEMENTATION_SUMMARY.md`: This file

## Benefits

1. **Flexibility**: Choose the right filter for your use case
2. **Ease of Use**: Automatic script generation with proper formatting
3. **Compatibility**: Seamless integration with existing filters
4. **Extensibility**: Support for custom filters via `--filter custom`
5. **Quality Control**: Different filters offer different speed/quality tradeoffs

## Filter Comparison Quick Reference

| Filter        | Speed    | Quality   | Best For                     |
| ------------- | -------- | --------- | ---------------------------- |
| NLM           | Fastest  | Good      | Quick processing, previews   |
| BM3D          | Medium   | Excellent | Balanced results             |
| Enhanced BM3D | Medium   | Excellent | Production use, large images |
| Custom        | Variable | Variable  | Specific needs               |

## How It Works

1. User specifies filter type with `--filter` argument
2. Script generation creates `process_tiles.sh` with:
   - Correct Python command for selected filter
   - File input/output handling
   - Progress tracking
3. User runs the generated script
4. All tiles are processed with the selected filter
5. Results are merged back using existing merge functionality

## Backward Compatibility

Fully backward compatible. If `--filter` is not specified, defaults to `enhanced_bm3d`.
