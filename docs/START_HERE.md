# Start Here: Filter Selection Feature

## What's New?

You can now **choose which denoising filter to use** when processing image tiles!

## The 4 Filters Available

| Filter | Command | Speed | Quality | When to Use |
|--------|---------|-------|---------|-------------|
| **NLM** | `--filter nlm` | Fastest | Good | Quick tests, previews |
| **BM3D** | `--filter bm3d` | Medium | Excellent | Standard denoising |
| **Enhanced BM3D** | `--filter enhanced_bm3d` | Medium | Excellent | Production, large images |
| **Custom** | `--filter custom --custom-command "..."` | Varies | Varies | Your own tool |

## Fastest Way to Get Started

### 1. Default (Recommended)
```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --create-script
cd tiles/
./process_tiles.sh
cd ..
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif
```

### 2. Fast (NLM)
```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script
cd tiles/
./process_tiles.sh
cd ..
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif
```

### 3. Standard (BM3D)
```bash
python3 LargeImage_Tiler.py split image.tif tiles/ --filter bm3d --create-script
cd tiles/
./process_tiles.sh
cd ..
python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif
```

## Command Options

```
--filter {nlm|bm3d|enhanced_bm3d|custom}
    Choose filter type (default: enhanced_bm3d)

--filter-args "ARGUMENTS"
    Pass extra arguments to the filter

--create-script
    Generate process_tiles.sh automatically

--custom-command "COMMAND"
    Use custom command (with --filter custom only)
```

## Examples

### With arguments
```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter enhanced_bm3d \
  --filter-args "--workers 2 --tile-size 2048" \
  --create-script
```

### Custom filter
```bash
python3 LargeImage_Tiler.py split image.tif tiles/ \
  --filter custom \
  --custom-command "python3 my_denoise.py {input} {output}" \
  --create-script
```

## Documentation Map

| Need | Document | Time |
|------|----------|------|
| Quick reference | `QUICK_REFERENCE.md` | 5 min |
| All commands | `COMMAND_REFERENCE.md` | 10 min |
| Complete guide | `FILTER_SELECTION_GUIDE.md` | 15 min |
| Copy-paste examples | `example_filter_usage.sh` | 2 min |
| Technical details | `ARCHITECTURE.md` | 20 min |
| What changed | `IMPLEMENTATION_SUMMARY.md` | 10 min |
| Documentation index | `README_FILTER_SELECTION.md` | 5 min |

## Key Features

✅ **Choose your filter** - 4 options available
✅ **Automatic scripts** - Generates process_tiles.sh with correct commands
✅ **Pass arguments** - Send extra parameters to any filter
✅ **Custom tools** - Use your own denoising software
✅ **Backward compatible** - Old code still works
✅ **Well documented** - 8 comprehensive guides

## Getting Help

```bash
# See all options
python3 LargeImage_Tiler.py split --help

# See merge options
python3 LargeImage_Tiler.py merge --help

# Read quick reference
cat QUICK_REFERENCE.md

# See command examples
cat example_filter_usage.sh
```

## Filter Selection Guide

**Choose NLM if:**
- You want fastest processing
- You just want to preview
- You have limited compute

**Choose BM3D if:**
- You want standard quality
- You have medium processing power
- You need good texture preservation

**Choose Enhanced BM3D if:** (Default)
- You want best quality
- This is production work
- You want auto-handling of large images
- You have good computing resources

**Choose Custom if:**
- You have a special denoising tool
- You need specific processing
- Your tool has unique capabilities

## Typical Workflow

```
1. Choose filter based on your needs
   ↓
2. Split image with chosen filter
   python3 LargeImage_Tiler.py split INPUT TILES/ --filter CHOICE --create-script
   ↓
3. Process tiles
   cd TILES/ && ./process_tiles.sh
   ↓
4. Merge results
   python3 LargeImage_Tiler.py merge TILES/tile_metadata.json TILES/ OUTPUT.tif
```

## Common Questions

**Q: Which filter should I use?**
A: Enhanced BM3D (the default). It has the best balance of quality and performance.

**Q: Can I change filters?**
A: Yes, just specify a different `--filter` option.

**Q: What if I don't specify a filter?**
A: It defaults to Enhanced BM3D (highest quality).

**Q: Can I use my own denoising tool?**
A: Yes, use `--filter custom --custom-command "your_command {input} {output}"`

**Q: Do old commands still work?**
A: Yes, 100% backward compatible.

**Q: Where's the generated script?**
A: In the tiles directory as `process_tiles.sh`

## Next Steps

1. **Quick start**: Run one of the examples above
2. **Learn more**: Read `QUICK_REFERENCE.md`
3. **Explore options**: Check `COMMAND_REFERENCE.md`
4. **Get advanced**: See `FILTER_SELECTION_GUIDE.md`

## Success Indicators

You'll know it's working when:
- ✓ Tiles are created in your output directory
- ✓ `process_tiles.sh` is generated and executable
- ✓ Script runs without errors
- ✓ Denoised tiles are created (denoised_tile_*.tif)
- ✓ Merge creates final output image

Ready to go? Pick a filter and run the command above!
