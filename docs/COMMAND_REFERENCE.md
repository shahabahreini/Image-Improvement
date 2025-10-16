# Complete Command Reference

## Split Command Syntax

```bash
python3 LargeImage_Tiler.py split INPUT OUTPUT_DIR [OPTIONS]
```

## All Available Options

### Positional Arguments

```
INPUT               Input image file (required)
OUTPUT_DIR          Directory to save tiles (required)
```

### Optional Arguments

#### Tiling Options

```
--tile-size SIZE    Size of each tile in pixels
                    Default: 1024
                    Example: --tile-size 2048

--prefix NAME       Prefix for tile filenames
                    Default: tile
                    Example: --prefix dem_tile
                    Result: dem_tile_0000_0000.tif
```

#### Filter Selection

```
--filter TYPE       Which filter to use for processing
                    Options: nlm, bm3d, enhanced_bm3d, custom
                    Default: enhanced_bm3d
                    Example: --filter nlm
```

#### Filter Arguments

```
--filter-args ARGS  Additional arguments for the selected filter
                    Applied when processing tiles
                    Example: --filter-args "--workers 2"
```

#### Script Generation

```
--create-script     Generate automated processing script
                    Creates: process_tiles.sh in tiles directory
                    Flag (no value needed)
                    Example: --create-script
```

#### Custom Filter

```
--custom-command CMD    Custom processing command
                        Required if --filter is set to custom
                        Use {input} and {output} as placeholders
                        Example: --custom-command "denoise {input} -> {output}"
```

## Merge Command Syntax

```bash
python3 LargeImage_Tiler.py merge METADATA TILES_DIR OUTPUT [OPTIONS]
```

### Merge Arguments

```
METADATA            Path to tile_metadata.json
TILES_DIR           Directory containing denoised tiles
OUTPUT              Output file path
--tile-prefix NAME  Prefix of denoised tile filenames
                    Default: denoised_tile
```

## Complete Examples

### 1. Simple - Using Default Enhanced BM3D

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ --create-script
```

- Tile size: 1024
- Prefix: tile
- Filter: enhanced_bm3d
- Generates: process_tiles.sh

### 2. With Custom Tile Size

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --tile-size 2048 \
  --create-script
```

### 3. Using NLM Filter (Fast)

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --filter nlm \
  --create-script
```

### 4. Using BM3D Filter

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --filter bm3d \
  --create-script
```

### 5. With Custom Prefix

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --prefix dem_tile \
  --filter enhanced_bm3d \
  --create-script
```

### 6. With Filter Arguments

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --filter enhanced_bm3d \
  --filter-args "--workers 4 --disable-yuv" \
  --create-script
```

### 7. Using Custom Filter

```bash
python3 LargeImage_Tiler.py split aerial.tif tiles/ \
  --filter custom \
  --custom-command "python3 my_denoise.py {input} {output}" \
  --create-script
```

### 8. Full Pipeline

```bash
# Step 1: Split with NLM for preview
python3 LargeImage_Tiler.py split image.tif tiles_preview/ \
  --tile-size 1024 \
  --prefix preview_tile \
  --filter nlm \
  --create-script

# Step 2: Process preview
cd tiles_preview/
./process_tiles.sh
cd ..

# Step 3: Check results
python3 LargeImage_Tiler.py merge tiles_preview/preview_tile_metadata.json tiles_preview/ preview_output.tif

# Step 4: If happy, do full quality with Enhanced BM3D
python3 LargeImage_Tiler.py split image.tif tiles_final/ \
  --tile-size 2048 \
  --prefix final_tile \
  --filter enhanced_bm3d \
  --create-script

# Step 5: Process final
cd tiles_final/
./process_tiles.sh
cd ..

# Step 6: Create final output
python3 LargeImage_Tiler.py merge tiles_final/final_tile_metadata.json tiles_final/ final_output.tif
```

### 9. Multiple Formats Combined

```bash
python3 LargeImage_Tiler.py split dem.tif tile_output/ \
  --tile-size 2048 \
  --prefix dem \
  --filter enhanced_bm3d \
  --filter-args "--tile-size 2048 --overlap 256" \
  --create-script
```

### 10. Custom Filter for Special Processing

```bash
python3 LargeImage_Tiler.py split noisy.tif tiles/ \
  --filter custom \
  --custom-command "python3 special_denoise.py {input} -o {output} --quality high" \
  --create-script
```

## Merge Command Examples

### Basic Merge

```bash
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif
```

### Custom Tile Prefix (if renamed)

```bash
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif \
  --tile-prefix my_processed_tile
```

## Quick Decision Tree

```
Do you want to split an image?
│
└─→ YES
    │
    └─→ Choose filter:
        │
        ├─→ FASTEST (NLM)
        │   Command: split IMAGE.tif TILES/ --filter nlm --create-script
        │
        ├─→ BALANCED (BM3D)
        │   Command: split IMAGE.tif TILES/ --filter bm3d --create-script
        │
        ├─→ BEST QUALITY (Enhanced BM3D) [DEFAULT]
        │   Command: split IMAGE.tif TILES/ --create-script
        │
        └─→ CUSTOM
            Command: split IMAGE.tif TILES/ --filter custom \
                     --custom-command "YOUR_COMMAND {input} {output}" --create-script

Do you want to merge tiles?
│
└─→ YES
    Command: merge TILES/tile_metadata.json TILES/ OUTPUT.tif
```

## Error Messages and Fixes

| Error                                                | Solution                                                  |
| ---------------------------------------------------- | --------------------------------------------------------- |
| `ModuleNotFoundError: cv2`                           | Activate virtual environment: `source .venv/bin/activate` |
| `large_image required`                               | Install: `pip install large-image[sources]`               |
| `--custom-command required when using filter custom` | Add: `--custom-command "YOUR_COMMAND"`                    |
| `No tile files found`                                | Check that tiles were created in split phase              |
| `Process failed`                                     | Check process_tiles.sh for error details                  |

## Useful Combinations

### For Large Aerial Images

```bash
python3 LargeImage_Tiler.py split dem.tif tiles/ \
  --tile-size 2048 \
  --filter enhanced_bm3d \
  --filter-args "--workers 4 --max-memory 8.0" \
  --create-script
```

### For Quick Testing

```bash
python3 LargeImage_Tiler.py split image.tif test_tiles/ \
  --tile-size 512 \
  --prefix test \
  --filter nlm \
  --create-script
```

### For Maximum Quality

```bash
python3 LargeImage_Tiler.py split image.tif hq_tiles/ \
  --tile-size 4096 \
  --filter enhanced_bm3d \
  --filter-args "--profile refilter --workers 2" \
  --create-script
```

### For Memory-Constrained Systems

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --tile-size 512 \
  --filter enhanced_bm3d \
  --filter-args "--tile-size 512 --max-memory 2.0" \
  --create-script
```
