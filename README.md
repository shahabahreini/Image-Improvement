# Image Improvement

A comprehensive suite of Python tools for image denoising and processing, optimized for aerial imagery and DEM (Digital Elevation Model) enhancement.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Quick Start](#quick-start)
- [Tools Overview](#tools-overview)
- [Usage Guide](#usage-guide)
  - [Basic BM3D Denoising](#1-basic-bm3d-denoising)
  - [Enhanced BM3D Denoising](#2-enhanced-bm3d-denoising)
  - [NLM Filtering](#3-nlm-filtering-for-geotiff)
  - [Large Image Tiling](#4-large-image-tiling)
- [Workflow Examples](#workflow-examples)
- [Output Files](#output-files)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **BM3D Denoising**: State-of-the-art block-matching 3D denoising algorithm
- **Non-Local Means (NLM) Filtering**: Advanced spatial filtering for GeoTIFF files
- **Large Image Tiling**: Split and process massive images without memory constraints
- **Batch Processing**: Multi-threaded processing of image collections
- **Quality Metrics**: Comprehensive evaluation of denoising results
- **GeoTIFF Support**: Preserves spatial metadata and geographic information
- **Multi-threaded Processing**: Parallel processing for faster batch operations

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)
- Minimum 4GB RAM (8GB+ recommended for batch processing)

### Setup

1. **Check for existing virtual environment:**

   ```bash
   ls -la | grep venv
   ```

2. **Create virtual environment (if needed):**

   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment:**

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Single Image Denoising (30 seconds)

```bash
source venv/bin/activate
python3 Basic_BM3D.py input_image.jpg output_clean.jpg
```

### Batch Processing Directory

```bash
source venv/bin/activate
python3 Enhanced_BM3D.py ./input_folder ./output_folder --batch
```

### Large Image Processing

```bash
source venv/bin/activate
python3 LargeImage_Tiler.py split large_image.tif ./tiles
python3 LargeImage_Tiler.py merge ./tiles/tile_metadata.json ./tiles result.tif
```

---

## Tools Overview

| Tool                    | Purpose                           | Best For                                | Speed  |
| ----------------------- | --------------------------------- | --------------------------------------- | ------ |
| **Basic_BM3D.py**       | Simple BM3D denoising             | Single images, quick processing         | Fast   |
| **Enhanced_BM3D.py**    | Advanced BM3D with YUV processing | Batch operations, high quality, GeoTIFF | Medium |
| **NLM_Filter.py**       | Non-Local Means filtering         | GeoTIFF files, spatial data             | Varies |
| **LargeImage_Tiler.py** | Split/merge large images          | Massive images (>10000x10000px)         | N/A    |

---

## Usage Guide

### 1. Basic BM3D Denoising

**Simple, fast denoising for individual images.**

```bash
python3 Basic_BM3D.py input_image.jpg output_denoised.jpg
```

#### Command Options

| Option         | Values           | Description                                               |
| -------------- | ---------------- | --------------------------------------------------------- |
| `--sigma`      | FLOAT            | Noise level (auto-estimated if not provided)              |
| `--profile`    | `np`, `refilter` | `np`: normal (faster), `refilter`: high quality (2-stage) |
| `--batch`      | -                | Process entire directory                                  |
| `--pattern`    | "\*.jpg"         | File pattern for batch mode                               |
| `--comparison` | -                | Save before/after comparison image                        |

#### Examples

```bash
# Auto noise estimation
python3 Basic_BM3D.py noisy.jpg denoised.jpg

# Specific noise level
python3 Basic_BM3D.py noisy.jpg denoised.jpg --sigma 0.05

# High-quality refilter mode
python3 Basic_BM3D.py noisy.jpg denoised.jpg --profile refilter

# Batch process with comparison
python3 Basic_BM3D.py ./input_dir ./output_dir --batch --comparison
```

---

### 2. Enhanced BM3D Denoising

**Advanced denoising with YUV color space optimization and large image support.**

```bash
python3 Enhanced_BM3D.py input_image.jpg output_denoised.jpg
```

#### Command Options

| Option          | Values           | Description                                |
| --------------- | ---------------- | ------------------------------------------ |
| `--sigma`       | FLOAT            | Noise standard deviation                   |
| `--profile`     | `np`, `refilter` | Processing profile                         |
| `--batch`       | -                | Batch processing mode                      |
| `--pattern`     | "\*.jpg"         | File pattern for batch                     |
| `--comparison`  | -                | Save comparison image                      |
| `--workers`     | INT              | Number of parallel workers (auto-detected) |
| `--disable-yuv` | -                | Disable YUV color optimization             |
| `--tile-size`   | INT              | Tile size for large images (default: 2048) |
| `--overlap`     | INT              | Tile overlap for blending (default: 128)   |
| `--max-memory`  | FLOAT            | Max memory per worker in GB (default: 8.0) |

#### Examples

```bash
# Enhanced processing of single image
python3 Enhanced_BM3D.py aerial_dem.jpg denoised_dem.jpg

# Batch with 4 workers
python3 Enhanced_BM3D.py ./input_dir ./output_dir --batch --workers 4

# Large image with custom tile size
python3 Enhanced_BM3D.py large_image.tif output.tif --tile-size 1024 --overlap 256

# GeoTIFF processing (preserves metadata)
python3 Enhanced_BM3D.py dem_data.tif dem_denoised.tif --profile refilter

# High-quality batch with comparison
python3 Enhanced_BM3D.py ./noisy ./clean --batch --profile refilter --comparison
```

---

### 3. NLM Filtering for GeoTIFF

**Non-Local Means filtering optimized for geospatial data.**

```bash
python3 NLM_Filter.py input_geotiff.tif output_geotiff.tif filter_type
```

#### Available Filter Types

| Filter Type                  | Speed  | Quality   | Best For              |
| ---------------------------- | ------ | --------- | --------------------- |
| `nlm`                        | Slow   | Excellent | High-quality DEM data |
| `nlm_fast`                   | Medium | Good      | Balanced results      |
| `nlm_ultrafast`              | Fast   | Good      | Large images          |
| `nlm_ultrafast_sharp`        | Fast   | Very Good | Sharp results needed  |
| `nlm_ultrafast_sharp_strong` | Fast   | Excellent | Maximum sharpening    |

#### Examples

```bash
# Standard NLM (best quality)
python3 NLM_Filter.py input.tif output.tif nlm

# Fast variant for quick processing
python3 NLM_Filter.py input.tif output.tif nlm_fast

# Large image with sharpening
python3 NLM_Filter.py large_dem.tif result.tif nlm_ultrafast_sharp

# Strong sharpening for detailed enhancement
python3 NLM_Filter.py input.tif output.tif nlm_ultrafast_sharp_strong
```

---

### 4. Large Image Tiling

**Split massive images into tiles, process them, and merge results.**

#### Split Image into Tiles

```bash
python3 LargeImage_Tiler.py split input_image.tif output_directory
```

**Options:**

- `--tile-size INT`: Size of each tile in pixels (default: 1024)
- `--prefix STRING`: Prefix for tile filenames (default: tile)
- `--create-script COMMAND`: Generate processing script

#### Merge Processed Tiles

```bash
python3 LargeImage_Tiler.py merge metadata.json tiles_directory output_merged.tif
```

**Options:**

- `--tile-prefix STRING`: Prefix of processed tiles (default: denoised_tile)

#### Complete Workflow

```bash
# Step 1: Split large image
python3 LargeImage_Tiler.py split large_dem.tif ./tiles --tile-size 2048 --prefix dem_tile

# Step 2a: Option A - Auto processing (if script was created)
cd ./tiles
./process_tiles.sh

# Step 2b: Option B - Manual processing
python3 ../Enhanced_BM3D.py dem_tile_0000_0000.tif denoised_dem_tile_0000_0000.tif

# Step 3: Merge back together
cd ..
python3 LargeImage_Tiler.py merge ./tiles/dem_tile_metadata.json ./tiles result_merged.tif
```

---

## Workflow Examples

### Scenario 1: Quick Single Image Denoising

**Time: ~2-5 minutes depending on image size**

```bash
source venv/bin/activate
python3 Basic_BM3D.py aerial_photo.jpg clean_photo.jpg --comparison
```

Result: `clean_photo.jpg` and `clean_photo_comparison.jpg`

---

### Scenario 2: Batch Processing Directory with Quality Metrics

**Time: Varies by number of images and image size**

```bash
source venv/bin/activate
python3 Enhanced_BM3D.py ./noisy_images ./clean_images --batch --profile refilter --workers 4
```

Results: Denoised images in `./clean_images/` with quality report

---

### Scenario 3: Processing Very Large GeoTIFF (50000x50000px)

**Time: 30+ minutes depending on system**

```bash
source venv/bin/activate

# Split into manageable tiles
python3 LargeImage_Tiler.py split dem_50000x50000.tif ./dem_tiles --tile-size 2048

# Process tiles
cd dem_tiles
for tile in dem_tile_*.tif; do
    echo "Processing $tile..."
    python3 ../Enhanced_BM3D.py "$tile" "denoised_$tile" --profile np
done
cd ..

# Merge results
python3 LargeImage_Tiler.py merge ./dem_tiles/dem_tile_metadata.json ./dem_tiles dem_final.tif
```

---

### Scenario 4: NLM Filtering GeoTIFF DEM Data

**Time: 5-20 minutes depending on method and size**

```bash
source venv/bin/activate

# For large images (fastest)
python3 NLM_Filter.py dem_original.tif dem_filtered.tif nlm_ultrafast_sharp

# For medium images
python3 NLM_Filter.py dem_original.tif dem_filtered.tif nlm_fast

# For maximum quality (slowest)
python3 NLM_Filter.py dem_original.tif dem_filtered.tif nlm
```

---

## Output Files

### Basic/Enhanced BM3D Outputs

- `denoised_*.jpg/png/tif`: Main denoised image
- `denoised_*_comparison.*`: Before/after comparison (if `--comparison` used)
- `processing_report.txt`: Batch processing summary

### NLM Filter Outputs

- Output GeoTIFF with preserved metadata and spatial reference

### Large Image Tiler Outputs

- `*_metadata.json`: Tile information and reconstruction data
- `denoised_tile_*.tif`: Processed individual tiles
- Final merged result: Combined output image

---

## Performance Tips

| Goal                    | Recommendation                                              |
| ----------------------- | ----------------------------------------------------------- |
| **Avoid Memory Issues** | Use `LargeImage_Tiler.py` for images >8000x8000px           |
| **Maximize Speed**      | Use `--profile np`, reduce `--workers`, use `nlm_ultrafast` |
| **Best Quality**        | Use `--profile refilter`, `nlm_ultrafast_sharp_strong`      |
| **Batch Processing**    | Adjust `--workers` based on: available_RAM / 2GB per worker |
| **GeoTIFF Data**        | Always use `Enhanced_BM3D.py` to preserve spatial metadata  |
| **Monitor Progress**    | Check system resources during first run to tune settings    |

---

## Troubleshooting

### ImportError: BM3D not available

```bash
pip install bm3d
```

### ImportError: large_image not available

```bash
pip install large-image[common]
```

### Out of Memory errors

- Reduce `--tile-size` parameter
- Decrease number of `--workers`
- Use smaller `--max-memory` value
- Split image into more tiles

### GeoTIFF metadata not preserved

**Solution:** Use `Enhanced_BM3D.py` instead of `Basic_BM3D.py` for GeoTIFF files

### Slow processing

- Use faster filter variants (`nlm_ultrafast` instead of `nlm`)
- Switch from `--profile refilter` to `--profile np`
- Increase `--workers` if RAM available
- Reduce image resolution or tile size

### Processing hangs or freezes

- Check available disk space for temporary files
- Verify input file is not corrupted
- Try processing smaller tile size
- Monitor system resources (CPU, RAM, I/O)

---

## License

MIT License - See LICENSE.md for details
