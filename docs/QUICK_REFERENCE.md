# Quick Reference: Filter Selection

## All Options in One Command

```bash
python3 LargeImage_Tiler.py split INPUT OUTPUT \
  --tile-size 1024 \
  --prefix tile \
  --filter {nlm|bm3d|enhanced_bm3d|custom} \
  --filter-args "ADDITIONAL_ARGS" \
  --create-script \
  --custom-command "COMMAND_FOR_CUSTOM_FILTER"
```

## One-Liners by Filter Type

### Enhanced BM3D (Recommended)

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter enhanced_bm3d --create-script
```

### BM3D

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter bm3d --create-script
```

### NLM

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script
```

### Custom

```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter custom --custom-command "python3 my_filter.py {input} {output}" --create-script
```

## What Each Filter Does

| Filter            | Command                                                              | Speed  | Quality   | Use Case              |
| ----------------- | -------------------------------------------------------------------- | ------ | --------- | --------------------- |
| **nlm**           | `python3 "NLM_Filter .py" "{input}" "{output}" nlm_ultrafast`        | Fast   | Good      | Previews, quick tests |
| **bm3d**          | `python3 "Basic_BM3D.py" "{input}" "{output}" --profile refilter`    | Medium | Excellent | Standard denoising    |
| **enhanced_bm3d** | `python3 "Enhanced_BM3D.py" "{input}" "{output}" --profile refilter` | Medium | Excellent | Production (default)  |
| **custom**        | Your custom command                                                  | Varies | Varies    | Special processing    |

## Default Behavior

If you don't specify `--filter`:

- Defaults to `enhanced_bm3d` (highest quality)
- Requires `--create-script` to generate processing script
- Other options work with their defaults

## Example Workflow

```bash
# 1. Split with NLM (fast preview)
python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script

# 2. Process tiles
cd tiles && ./process_tiles.sh && cd ..

# 3. Merge result
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ preview.tif

# 4. If happy, split again with Enhanced BM3D for final quality
python3 LargeImage_Tiler.py split image.tif tiles_hq/ --filter enhanced_bm3d --create-script
cd tiles_hq && ./process_tiles.sh && cd ..
python3 LargeImage_Tiler.py merge tiles_hq/tile_metadata.json tiles_hq/ final.tif
```

## With Arguments

```bash
# Pass filter-specific arguments
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter enhanced_bm3d \
  --filter-args "--workers 4 --tile-size 2048" \
  --create-script
```

## Custom Filter Template

Your custom filter must:

1. Accept input file as first argument
2. Accept output file as second argument
3. Support being called from command line

Example custom filter command:

```bash
--filter custom --custom-command "python3 my_denoise.py {input} {output}"
```

The script will replace `{input}` and `{output}` with actual file names.
