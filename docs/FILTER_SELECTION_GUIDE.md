# Filter Selection Guide for LargeImage_Tiler

## Overview

The `LargeImage_Tiler.py` script now supports selecting different denoising filters to process tiles after splitting. You can choose between three built-in filters or provide a custom command.

## Available Filters

### 1. NLM (Non-Local Means) Filter

- **Script**: `NLM_Filter .py`
- **Best for**: Quick processing with good quality
- **Strengths**: Fast, preserves colors and edges
- **Usage**:
  ```bash
  python3 LargeImage_Tiler.py split input.tif tiles/ --filter nlm --create-script
  ```

### 2. BM3D (Basic)

- **Script**: `Basic_BM3D.py`
- **Best for**: Standard denoising with texture preservation
- **Strengths**: Good balance between speed and quality
- **Usage**:
  ```bash
  python3 LargeImage_Tiler.py split input.tif tiles/ --filter bm3d --create-script
  ```

### 3. Enhanced BM3D (Recommended)

- **Script**: `Enhanced_BM3D.py`
- **Best for**: High-quality denoising with advanced features
- **Strengths**:
  - VST (Variance Stabilizing Transform) for dark region handling
  - YUV colorspace processing
  - Advanced texture preservation
  - Memory-efficient tiling for large images
- **Usage**:
  ```bash
  python3 LargeImage_Tiler.py split input.tif tiles/ --filter enhanced_bm3d --create-script
  ```

### 4. Custom Filter

- **For**: Using a custom denoising command or script
- **Usage**:
  ```bash
  python3 LargeImage_Tiler.py split input.tif tiles/ --filter custom --custom-command "my_denoise.py {input} {output}" --create-script
  ```

## Complete Examples

### Basic Usage with Enhanced BM3D (Default)

```bash
# 1. Split image into tiles with processing script
python3 LargeImage_Tiler.py split large_image.tif output_tiles/ --tile-size 1024 --create-script

# 2. Process tiles
cd output_tiles/
./process_tiles.sh

# 3. Merge tiles back together
cd ..
python3 LargeImage_Tiler.py merge output_tiles/tile_metadata.json output_tiles/ final_output.tif
```

### Using NLM Filter with Custom Arguments

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter nlm \
  --filter-args "--extra-param value" \
  --create-script
```

### Using Custom Filter

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter custom \
  --custom-command "python3 my_custom_filter.py {input} {output}" \
  --create-script
```

## Command Options

### Split Command Options

- `--tile-size SIZE`: Size of each tile in pixels (default: 1024)
- `--prefix NAME`: Prefix for tile filenames (default: tile)
- `--filter {nlm,bm3d,enhanced_bm3d,custom}`: Choose denoising filter (default: enhanced_bm3d)
- `--filter-args ARGS`: Additional arguments to pass to the filter
- `--create-script`: Automatically generate a processing script
- `--custom-command CMD`: Command to use with `--filter custom`

### Generated Processing Script

When you use `--create-script`, a `process_tiles.sh` script is created with:

- Proper error handling (exits on first error)
- Progress reporting for each tile
- Comments indicating which filter is being used

## Filter Comparison

| Feature              | NLM       | BM3D      | Enhanced BM3D     |
| -------------------- | --------- | --------- | ----------------- |
| Speed                | Fast      | Medium    | Medium            |
| Quality              | Good      | Excellent | Excellent         |
| Memory Usage         | Low       | Medium    | Low (with tiling) |
| Dark Region Handling | Basic     | Basic     | Advanced (VST)    |
| Color Preservation   | Excellent | Excellent | Excellent         |
| Large Image Support  | Good      | Good      | Excellent (tiled) |

## Workflow Example

```bash
# Step 1: Activate virtual environment
source .venv/bin/activate

# Step 2: Split image and create processing script (using Enhanced BM3D)
python3 LargeImage_Tiler.py split aerial_dem.tif dem_tiles/ \
  --tile-size 2048 \
  --prefix dem_tile \
  --filter enhanced_bm3d \
  --create-script

# Step 3: Process all tiles
cd dem_tiles/
./process_tiles.sh

# Step 4: Merge back
cd ..
python3 LargeImage_Tiler.py merge dem_tiles/dem_tile_metadata.json dem_tiles/ denoised_dem.tif

# Done!
echo "Denoising complete! Output: denoised_dem.tif"
```

## Tips and Best Practices

1. **Choose the right tile size**:

   - Smaller tiles (512-1024): Lower memory usage, slower processing
   - Larger tiles (2048-4096): Higher memory usage, faster processing

2. **Filter selection**:

   - Use NLM for quick previews
   - Use BM3D for balanced results
   - Use Enhanced BM3D for production quality

3. **Monitor processing**:

   - The script prints progress (Tile X/Y)
   - Check that output filenames are generated correctly (denoised*tile*\*.tif)

4. **Custom filters**:
   - Use `{input}` and `{output}` placeholders in custom commands
   - Ensure your custom filter can be called from the command line
   - Test with a single tile first

## Troubleshooting

### Script not executing

```bash
chmod +x output_tiles/process_tiles.sh
```

### Filter not found

- Verify filter script exists (NLM_Filter .py, Basic_BM3D.py, Enhanced_BM3D.py)
- Check file names match exactly (note the space in "NLM_Filter .py")

### Out of memory errors

- Reduce `--tile-size`
- Use Enhanced BM3D which handles large tiles better
- Process fewer tiles in parallel

### Denoised tiles not created

- Check that the filter command is correct
- Run a single tile manually to debug the command
- Check `./process_tiles.sh` contents to verify command formatting
