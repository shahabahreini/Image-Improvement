#!/bin/bash
# Example Usage: Filter Selection with LargeImage_Tiler

echo "Filter Selection Examples for LargeImage_Tiler"
echo "=============================================="
echo ""

echo "1. ENHANCED BM3D (Recommended - Default)"
echo "   python3 LargeImage_Tiler.py split image.tif tiles/ --create-script"
echo ""

echo "2. BASIC BM3D"
echo "   python3 LargeImage_Tiler.py split image.tif tiles/ --filter bm3d --create-script"
echo ""

echo "3. NLM FILTER (Fastest)"
echo "   python3 LargeImage_Tiler.py split image.tif tiles/ --filter nlm --create-script"
echo ""

echo "4. CUSTOM FILTER"
echo "   python3 LargeImage_Tiler.py split image.tif tiles/ \\"
echo "     --filter custom \\"
echo "     --custom-command 'python3 my_filter.py {input} {output}' \\"
echo "     --create-script"
echo ""

echo "5. WITH ADDITIONAL FILTER ARGUMENTS"
echo "   python3 LargeImage_Tiler.py split image.tif tiles/ \\"
echo "     --filter enhanced_bm3d \\"
echo "     --filter-args '--workers 4 --disable-yuv' \\"
echo "     --create-script"
echo ""

echo "After splitting, process tiles and merge:"
echo "   cd tiles/"
echo "   ./process_tiles.sh"
echo "   cd .."
echo "   python3 LargeImage_Tiler.py merge tiles/tile_metadata.json tiles/ output.tif"
