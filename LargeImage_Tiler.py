#!/usr/bin/env python3
"""
Large Image Tiler - Split and merge large images using the large_image library
Robust handling of various image formats with proper tile management.
"""

import numpy as np
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import cv2

try:
    import large_image

    LARGE_IMAGE_AVAILABLE = True
except ImportError:
    LARGE_IMAGE_AVAILABLE = False
    raise ImportError(
        "large_image required. Install with: pip install large-image[sources]"
    )

try:
    import rasterio
    from rasterio.transform import from_bounds

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print(
        "Warning: rasterio not available. GeoTIFF metadata preservation will be limited."
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LargeImageTiler:
    """Split and merge large images using the large_image library."""

    def __init__(self, tile_size: int = 1024):
        """
        Initialize large image tiler.

        Args:
            tile_size: Size of each tile (width and height)
        """
        if not LARGE_IMAGE_AVAILABLE:
            raise ImportError("large_image library is required")

        self.tile_size = tile_size

    def split_image(
        self, input_path: str, output_dir: str, prefix: str = "tile"
    ) -> Dict:
        """
        Split large image into tiles using large_image library.

        Args:
            input_path: Path to input image
            output_dir: Directory to save tiles
            prefix: Prefix for tile filenames

        Returns:
            Metadata dictionary for reconstruction
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Opening {input_path} with large_image")

        # Open source with large_image
        source = large_image.open(str(input_path))

        # Get image metadata
        metadata_info = source.getMetadata()
        width = metadata_info["sizeX"]
        height = metadata_info["sizeY"]
        bands = metadata_info.get("bands", 3)

        logger.info(f"Image size: {width}x{height}, Bands: {bands}")
        logger.info(f"Tile size: {self.tile_size}x{self.tile_size}")

        # Calculate expected number of tiles
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        expected_tiles = tiles_x * tiles_y

        logger.info(f"Expected tiles: {tiles_x}x{tiles_y} = {expected_tiles}")

        # Store metadata for reconstruction
        metadata = {
            "input_file": str(input_path.name),
            "original_width": width,
            "original_height": height,
            "bands": bands,
            "tile_size": self.tile_size,
            "tiles_x": tiles_x,
            "tiles_y": tiles_y,
            "tiles": [],
        }

        # Try to preserve geospatial metadata if available
        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(input_path) as src:
                    if src.crs:
                        metadata["crs"] = src.crs.to_string()
                        metadata["transform"] = list(src.transform)
                        logger.info("Preserved geospatial metadata")
            except Exception as e:
                logger.warning(f"Could not read geospatial metadata: {e}")

        tile_count = 0

        # Iterate through tiles using large_image
        for tile_info in source.tileIterator(
            tile_size={"width": self.tile_size, "height": self.tile_size},
            format=large_image.constants.TILE_FORMAT_NUMPY,
        ):
            tile_count += 1

            # Extract tile information
            x = tile_info["x"]
            y = tile_info["y"]
            tile_data = tile_info["tile"]

            # Calculate tile position in grid
            tile_x = x // self.tile_size
            tile_y = y // self.tile_size

            # Generate tile filename
            tile_filename = f"{prefix}_{tile_y:04d}_{tile_x:04d}.tif"
            tile_path = output_dir / tile_filename

            # Convert from RGB to BGR for OpenCV compatibility
            if len(tile_data.shape) == 3 and tile_data.shape[2] >= 3:
                # large_image returns RGB, convert to BGR for consistency with OpenCV
                tile_bgr = cv2.cvtColor(tile_data, cv2.COLOR_RGB2BGR)
            else:
                tile_bgr = tile_data

            # Save tile
            cv2.imwrite(str(tile_path), tile_bgr)

            # Store tile metadata
            tile_info_dict = {
                "filename": tile_filename,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "x": x,
                "y": y,
                "width": tile_data.shape[1],
                "height": tile_data.shape[0],
                "channels": tile_data.shape[2] if len(tile_data.shape) == 3 else 1,
            }
            metadata["tiles"].append(tile_info_dict)

            if tile_count % 10 == 0:
                logger.info(f"Created {tile_count} tiles...")

        logger.info(f"Created {tile_count} tiles total")

        # Save metadata
        metadata_path = output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Tiling complete! Metadata saved to {metadata_path}")

        return metadata

    def merge_image(
        self,
        metadata_path: str,
        tiles_dir: str,
        output_path: str,
        tile_prefix: str = "denoised_tile",
    ) -> None:
        """
        Merge processed tiles back into a single image.

        Args:
            metadata_path: Path to metadata JSON file
            tiles_dir: Directory containing processed tiles
            output_path: Path for output merged image
            tile_prefix: Prefix of processed tile filenames
        """
        metadata_path = Path(metadata_path)
        tiles_dir = Path(tiles_dir)
        output_path = Path(output_path)

        logger.info(f"Merging tiles from {tiles_dir} to {output_path}")

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        width = metadata["original_width"]
        height = metadata["original_height"]
        bands = metadata["bands"]

        logger.info(f"Reconstructing {width}x{height} image with {bands} channels")

        # Create output image
        if bands == 1:
            output_image = np.zeros((height, width), dtype=np.uint8)
        else:
            output_image = np.zeros((height, width, bands), dtype=np.uint8)

        tiles_processed = 0

        for tile_info in metadata["tiles"]:
            # Find corresponding processed tile
            original_name = tile_info["filename"]
            processed_name = original_name.replace("tile_", f"{tile_prefix}_")
            processed_path = tiles_dir / processed_name

            if not processed_path.exists():
                logger.warning(f"Processed tile not found: {processed_path}")
                continue

            # Load processed tile
            if bands == 1:
                tile_data = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
            else:
                tile_data = cv2.imread(str(processed_path), cv2.IMREAD_COLOR)

            if tile_data is None:
                logger.warning(f"Could not load tile: {processed_path}")
                continue

            # Get tile position
            x = tile_info["x"]
            y = tile_info["y"]
            tile_height, tile_width = tile_data.shape[:2]

            # Place tile in output image
            y_end = min(y + tile_height, height)
            x_end = min(x + tile_width, width)

            if bands == 1:
                output_image[y:y_end, x:x_end] = tile_data[: y_end - y, : x_end - x]
            else:
                output_image[y:y_end, x:x_end] = tile_data[: y_end - y, : x_end - x]

            tiles_processed += 1
            if tiles_processed % 10 == 0:
                logger.info(f"Merged {tiles_processed}/{len(metadata['tiles'])} tiles")

        # Save merged image
        if (
            output_path.suffix.lower() in [".tif", ".tiff"]
            and RASTERIO_AVAILABLE
            and "crs" in metadata
        ):
            # Save as GeoTIFF with spatial reference
            logger.info("Saving as GeoTIFF with spatial metadata")

            # Prepare data for rasterio (channels first)
            if len(output_image.shape) == 3:
                output_data = np.transpose(output_image, (2, 0, 1))
            else:
                output_data = output_image[np.newaxis, :, :]

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": bands,
                "dtype": output_data.dtype,
                "crs": metadata["crs"],
                "transform": rasterio.Affine(*metadata["transform"]),
                "compress": "lzw",
            }

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(output_data)
        else:
            # Save as regular image
            cv2.imwrite(str(output_path), output_image)

        logger.info(f"Merge complete! Output saved to {output_path}")
        logger.info(f"Successfully merged {tiles_processed} tiles")

    def create_processing_script(
        self,
        tiles_dir: str,
        denoising_command: str,
        filter_type: str = "enhanced_bm3d",
        filter_args: Optional[str] = None,
    ) -> str:
        """
        Generate a batch processing script for all tiles.

        Args:
            tiles_dir: Directory containing tiles
            denoising_command: Base command or script name for processing
            filter_type: Type of filter to use ('nlm', 'bm3d', 'enhanced_bm3d')
            filter_args: Additional arguments to pass to the filter

        Returns:
            Path to generated script
        """
        tiles_dir = Path(tiles_dir)

        # Find all tile files
        tile_files = list(tiles_dir.glob("tile_*.tif"))
        if not tile_files:
            tile_files = list(tiles_dir.glob("tile_*.jpg"))
        if not tile_files:
            tile_files = list(tiles_dir.glob("tile_*.png"))

        tile_files.sort()

        if not tile_files:
            logger.warning(f"No tile files found in {tiles_dir}")
            return ""

        # Generate processing script
        script_path = tiles_dir / "process_tiles.sh"

        # Determine filter command based on filter_type
        if filter_type.lower() == "nlm":
            filter_script = "NLM_Filter .py"
            filter_cmd_template = (
                f'python3 "{filter_script}" "{{input}}" "{{output}}" nlm_ultrafast'
            )
        elif filter_type.lower() == "bm3d":
            filter_script = "Basic_BM3D.py"
            filter_cmd_template = (
                f'python3 "{filter_script}" "{{input}}" "{{output}}" --profile refilter'
            )
        elif filter_type.lower() == "enhanced_bm3d":
            filter_script = "Enhanced_BM3D.py"
            filter_cmd_template = (
                f'python3 "{filter_script}" "{{input}}" "{{output}}" --profile refilter'
            )
        else:
            # Use custom command if provided
            filter_cmd_template = denoising_command

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Auto-generated tile processing script\n")
            f.write(f"# Filter type: {filter_type}\n")
            f.write(f"# Found {len(tile_files)} tiles to process\n\n")
            f.write("set -e  # Exit on any error\n\n")

            for i, tile_file in enumerate(tile_files, 1):
                input_name = tile_file.name
                output_name = f"denoised_{input_name}"

                f.write(f'echo "Processing tile {i}/{len(tile_files)}: {input_name}"\n')

                # Build the filter command
                if filter_type.lower() in ["nlm", "bm3d", "enhanced_bm3d"]:
                    filter_cmd = filter_cmd_template.format(
                        input=input_name, output=output_name
                    )
                else:
                    filter_cmd = filter_cmd_template.replace(
                        "{input}", input_name
                    ).replace("{output}", output_name)

                if filter_args:
                    filter_cmd += f" {filter_args}"

                f.write(f"{filter_cmd}\n")
                f.write(f"if [ $? -ne 0 ]; then\n")
                f.write(f'    echo "Failed to process {input_name}"\n')
                f.write(f"    exit 1\n")
                f.write(f"fi\n\n")

            f.write('echo "All tiles processed successfully!"\n')

        # Make script executable
        script_path.chmod(0o755)

        logger.info(f"Processing script created: {script_path}")
        logger.info(f"Using filter: {filter_type}")

        return str(script_path)


def main():
    """CLI interface for large image tiling operations."""
    parser = argparse.ArgumentParser(
        description="Split and merge large images using large_image library",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split image into tiles")
    split_parser.add_argument("input", help="Input image file")
    split_parser.add_argument("output_dir", help="Output directory for tiles")
    split_parser.add_argument(
        "--tile-size", type=int, default=1024, help="Size of each tile (pixels)"
    )
    split_parser.add_argument(
        "--prefix", default="tile", help="Prefix for tile filenames"
    )
    split_parser.add_argument(
        "--filter",
        choices=["nlm", "bm3d", "enhanced_bm3d", "custom"],
        default="enhanced_bm3d",
        help="Filter to apply after tiling (nlm, bm3d, enhanced_bm3d, or custom command)",
    )
    split_parser.add_argument(
        "--filter-args",
        help="Additional arguments to pass to the filter script",
    )
    split_parser.add_argument(
        "--create-script",
        action="store_true",
        help="Create processing script for the selected filter",
    )
    split_parser.add_argument(
        "--custom-command",
        help="Custom command to use if --filter is set to custom",
    )

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge processed tiles")
    merge_parser.add_argument(
        "metadata", help="Metadata JSON file from split operation"
    )
    merge_parser.add_argument("tiles_dir", help="Directory containing processed tiles")
    merge_parser.add_argument("output", help="Output merged image file")
    merge_parser.add_argument(
        "--tile-prefix",
        default="denoised_tile",
        help="Prefix of processed tile filenames",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "split":
            # Initialize tiler
            tiler = LargeImageTiler(tile_size=args.tile_size)

            # Split image
            metadata = tiler.split_image(args.input, args.output_dir, args.prefix)

            print(f"\nSplitting complete!")
            print(f"Created {len(metadata['tiles'])} tiles in {args.output_dir}")
            print(f"Metadata saved as: {args.prefix}_metadata.json")

            # Create processing script if requested
            if args.create_script:
                if args.filter == "custom" and not args.custom_command:
                    print("Error: --custom-command required when using --filter custom")
                    return 1

                custom_cmd = args.custom_command if args.filter == "custom" else None

                script_path = tiler.create_processing_script(
                    args.output_dir,
                    custom_cmd or "",
                    filter_type=args.filter,
                    filter_args=args.filter_args,
                )
                print(f"Processing script created: {script_path}")
                print(f"\nTo process all tiles:")
                print(f"  cd {args.output_dir}")
                print(f"  ./process_tiles.sh")

            print(f"\nTo merge processed tiles:")
            print(
                f"  python3 {Path(__file__).name} merge {args.output_dir}/{args.prefix}_metadata.json {args.output_dir} output_merged.tif"
            )

        elif args.command == "merge":
            # Initialize tiler
            tiler = LargeImageTiler()

            # Merge tiles
            tiler.merge_image(
                args.metadata, args.tiles_dir, args.output, args.tile_prefix
            )

            print(f"\nMerging complete!")
            print(f"Output saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
