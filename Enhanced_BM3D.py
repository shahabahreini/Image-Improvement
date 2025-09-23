#!/usr/bin/env python3
"""
Enhanced BM3D Denoising for Aerial Images - DEM Processing Enhancement
Advanced implementation with quality optimizations for photogrammetry.
"""

import numpy as np
import cv2
import argparse
import time
import math
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

try:
    import bm3d

    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    raise ImportError("BM3D not available. Install with: pip install bm3d")

try:
    from skimage.restoration import estimate_sigma

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print(
        "Warning: scikit-image not available. Noise estimation will use fallback method."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBM3DDenoiser:
    """Enhanced BM3D denoising optimized for aerial imagery DEM processing."""

    def __init__(
        self,
        enable_yuv_processing: bool = True,
        tile_size: int = 2048,
        overlap: int = 128,
    ):
        """
        Initialize enhanced BM3D denoiser.

        Args:
            enable_yuv_processing: Use YUV colorspace for better perceptual quality
            tile_size: Size of tiles for processing large images
            overlap: Overlap between tiles for seamless blending
        """
        if not BM3D_AVAILABLE:
            raise ImportError("BM3D library is required")
        self.enable_yuv = enable_yuv_processing
        self.tile_size = tile_size
        self.overlap = overlap

    def estimate_noise_sigma(self, image: np.ndarray) -> float:
        """
        Enhanced noise estimation using multiple methods.

        Args:
            image: Input image (uint8)

        Returns:
            Estimated noise sigma (robust estimation)
        """
        img_float = image.astype(np.float32) / 255.0

        if SKIMAGE_AVAILABLE:
            if len(image.shape) == 3:
                # Use luminance channel for more accurate noise estimation
                gray = (
                    cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    / 255.0
                )
                sigma = estimate_sigma(gray, channel_axis=None)
            else:
                sigma = estimate_sigma(img_float, channel_axis=None)
        else:
            # Robust noise estimation using median absolute deviation
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3
                else image
            )
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # Median absolute deviation method - more robust than variance
            mad = np.median(np.abs(laplacian - np.median(laplacian)))
            sigma = mad / 0.6745 / 255.0

        # Clamp sigma to reasonable range for aerial imagery
        sigma_clamped = float(np.clip(sigma, 0.001, 0.2))

        if sigma != sigma_clamped:
            logger.debug(f"Sigma clamped from {sigma:.4f} to {sigma_clamped:.4f}")

        return sigma_clamped

    def get_optimal_block_size(self, image_shape: Tuple[int, int], sigma: float) -> int:
        """
        Determine optimal block size based on image dimensions and noise level.

        Args:
            image_shape: Image (height, width)
            sigma: Noise level

        Returns:
            Optimal block size for BM3D
        """
        min_dim = min(image_shape[:2])

        # Adaptive block size based on image size and noise level
        if min_dim > 2048:
            block_size = 8 if sigma > 0.05 else 16
        elif min_dim > 1024:
            block_size = 8
        else:
            block_size = 4 if sigma < 0.02 else 8

        return block_size

    def process_yuv_colorspace(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Process image in YUV colorspace for better perceptual quality.
        Applies different denoising strengths to luminance and chrominance.

        Args:
            image: Input BGR image (uint8)
            sigma: Noise standard deviation

        Returns:
            Denoised image in BGR format
        """
        # Convert to YUV (better separation of luminance and chrominance)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32) / 255.0

        # Denoise Y channel (luminance) with full strength
        yuv[:, :, 0] = bm3d.bm3d(yuv[:, :, 0], sigma)

        # Denoise U,V channels (chrominance) with reduced strength
        # Human visual system is less sensitive to chrominance noise
        chroma_sigma = sigma * 0.5
        yuv[:, :, 1] = bm3d.bm3d(yuv[:, :, 1], chroma_sigma)
        yuv[:, :, 2] = bm3d.bm3d(yuv[:, :, 2], chroma_sigma)

        # Convert back to BGR
        yuv_uint8 = np.clip(yuv * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(yuv_uint8, cv2.COLOR_YUV2BGR)

    def denoise_image_advanced(
        self, image: np.ndarray, sigma: Optional[float] = None, profile: str = "np"
    ) -> Tuple[np.ndarray, dict]:
        """
        Advanced BM3D denoising with optimized parameters.

        Args:
            image: Input image (uint8)
            sigma: Noise standard deviation (auto-estimated if None)
            profile: Processing profile ('np' or 'refilter')

        Returns:
            Tuple of (denoised_image, metadata)
        """
        if not BM3D_AVAILABLE:
            raise RuntimeError("BM3D not available")

        # Convert to float [0,1]
        img_float = image.astype(np.float32) / 255.0

        # Estimate noise if not provided
        if sigma is None:
            sigma = self.estimate_noise_sigma(image)
            logger.info(f"Estimated noise sigma: {sigma:.4f}")

        # Determine processing strategy
        use_yuv = (
            self.enable_yuv
            and len(image.shape) == 3
            and min(image.shape[:2]) > 1024
            and sigma < 0.08
        )

        metadata = {
            "sigma_used": sigma,
            "profile": profile,
            "input_shape": image.shape,
            "input_dtype": str(image.dtype),
            "processing_method": "YUV" if use_yuv else "RGB",
            "block_size": self.get_optimal_block_size(image.shape, sigma),
        }

        if use_yuv:
            # Use YUV processing for large, low-noise images
            logger.info("Using YUV colorspace processing")
            denoised = self.process_yuv_colorspace(image, sigma)
        else:
            # Standard RGB processing
            if len(image.shape) == 3:
                denoised = np.zeros_like(img_float)
                for c in range(3):
                    if profile == "refilter":
                        # Two-stage BM3D for maximum quality
                        logger.info(f"Applying two-stage BM3D to channel {c}")
                        basic_estimate = bm3d.bm3d(
                            img_float[:, :, c],
                            sigma,
                            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING,
                        )
                        denoised[:, :, c] = bm3d.bm3d(
                            img_float[:, :, c],
                            sigma,
                            basic_estimate,
                            stage_arg=bm3d.BM3DStages.ALL_STAGES,
                        )
                    else:
                        # Single-stage BM3D with adaptive parameters
                        denoised[:, :, c] = bm3d.bm3d(img_float[:, :, c], sigma)

                # Convert back to uint8
                denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
            else:
                # Grayscale processing
                if profile == "refilter":
                    basic_estimate = bm3d.bm3d(
                        img_float, sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
                    )
                    denoised = bm3d.bm3d(
                        img_float,
                        sigma,
                        basic_estimate,
                        stage_arg=bm3d.BM3DStages.ALL_STAGES,
                    )
                else:
                    denoised = bm3d.bm3d(img_float, sigma)

                denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)

        return denoised, metadata

    def process_image_tiled(
        self, image: np.ndarray, sigma: Optional[float] = None
    ) -> np.ndarray:
        """Process large images using overlapping tiles to manage memory."""
        if min(image.shape[:2]) <= self.tile_size:
            return self.denoise_image_advanced(image, sigma)[0]

        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        # Calculate tile positions with overlap
        tiles_y = math.ceil((h - self.overlap) / (self.tile_size - self.overlap))
        tiles_x = math.ceil((w - self.overlap) / (self.tile_size - self.overlap))

        result = np.zeros_like(image)
        weight_map = np.zeros(image.shape[:2], dtype=np.float32)

        logger.info(f"Processing {tiles_y}x{tiles_x} tiles for {h}x{w} image")

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                # Calculate tile bounds
                y_start = ty * (self.tile_size - self.overlap)
                x_start = tx * (self.tile_size - self.overlap)
                y_end = min(y_start + self.tile_size, h)
                x_end = min(x_start + self.tile_size, w)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # Process tile
                denoised_tile = self.denoise_image_advanced(tile, sigma)[0]

                # Create weight mask for blending (higher in center)
                tile_h, tile_w = denoised_tile.shape[:2]
                weight = self._create_weight_mask(tile_h, tile_w)

                # Accumulate results with weighted blending
                if len(denoised_tile.shape) == 3:
                    result[y_start:y_end, x_start:x_end] += (
                        denoised_tile * weight[..., np.newaxis]
                    )
                else:
                    result[y_start:y_end, x_start:x_end] += denoised_tile * weight
                weight_map[y_start:y_end, x_start:x_end] += weight

        # Normalize by accumulated weights
        if len(result.shape) == 3:
            result = result / weight_map[..., np.newaxis]
        else:
            result = result / weight_map
        return result.astype(image.dtype)

    def _create_weight_mask(self, h: int, w: int) -> np.ndarray:
        """Create smooth weight mask for tile blending."""
        mask = np.ones((h, w), dtype=np.float32)
        fade_size = self.overlap // 2

        if fade_size > 0:
            # Apply fade to edges
            if h > fade_size:
                mask[:fade_size, :] *= np.linspace(0, 1, fade_size)[:, np.newaxis]
                mask[-fade_size:, :] *= np.linspace(1, 0, fade_size)[:, np.newaxis]
            if w > fade_size:
                mask[:, :fade_size] *= np.linspace(0, 1, fade_size)
                mask[:, -fade_size:] *= np.linspace(1, 0, fade_size)

        return mask

    def process_geotiff_file(
        self, input_path: str, output_path: str, sigma: Optional[float] = None
    ) -> dict:
        """Process GeoTIFF files directly with spatial metadata preservation."""
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            raise ImportError(
                "rasterio required for GeoTIFF processing. Install with: pip install rasterio"
            )

        start_time = time.time()

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()

            # Read image data
            if src.count >= 3:
                image = src.read([1, 2, 3])  # RGB channels
                image = np.transpose(image, (1, 2, 0))  # HWC format
            else:
                raise ValueError(f"Need at least 3 bands, found {src.count}")

            logger.info(f"Processing GeoTIFF {input_path} ({image.shape})")

            # Normalize based on dtype
            if image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            elif image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype in [np.float32, np.float64]:
                image = image.astype(np.float32)
                if image.max() > 1:
                    image = image / image.max()

            # Process with tiled approach
            denoised = self.process_image_tiled(image, sigma)

            # Convert back to original dtype
            if profile["dtype"] == "uint16":
                denoised = (np.clip(denoised, 0, 1) * 65535).astype(np.uint16)
            elif profile["dtype"] == "uint8":
                denoised = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
            else:
                denoised = denoised.astype(profile["dtype"])

            # Update profile for efficient output
            profile.update(
                count=3, compress="lzw", tiled=True, blockxsize=512, blockysize=512
            )

            # Write output
            denoised = np.transpose(denoised, (2, 0, 1))  # CHW format
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(denoised)

        processing_time = time.time() - start_time
        logger.info(f"GeoTIFF processing completed in {processing_time:.2f}s")

        return {
            "input_path": input_path,
            "output_path": output_path,
            "processing_time": processing_time,
            "image_shape": image.shape,
        }

    def compute_enhanced_metrics(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> dict:
        """
        Compute comprehensive image quality metrics.

        Args:
            original: Original noisy image
            denoised: Denoised image

        Returns:
            Dictionary of quality metrics
        """
        orig_float = original.astype(np.float32)
        den_float = denoised.astype(np.float32)

        # MSE and PSNR
        mse = np.mean((orig_float - den_float) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")

        # Convert to grayscale for structural metrics
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            den_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            den_gray = denoised

        # Enhanced SSIM approximation
        def compute_ssim_components(x, y):
            mu_x = np.mean(x)
            mu_y = np.mean(y)
            sigma_x = np.var(x)
            sigma_y = np.var(y)
            sigma_xy = np.mean((x - mu_x) * (y - mu_y))

            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
                (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            )
            return ssim

        ssim_approx = compute_ssim_components(orig_gray.flatten(), den_gray.flatten())

        # Noise metrics
        orig_std = np.std(orig_float)
        den_std = np.std(den_float)
        noise_reduction = (orig_std - den_std) / orig_std * 100

        # Edge preservation metrics
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        den_edges = cv2.Canny(den_gray, 50, 150)
        edge_correlation = np.corrcoef(orig_edges.flatten(), den_edges.flatten())[0, 1]

        # Texture preservation (local variance preservation)
        kernel = np.ones((5, 5), np.float32) / 25
        orig_local_var = (
            cv2.filter2D(orig_gray.astype(np.float32) ** 2, -1, kernel)
            - cv2.filter2D(orig_gray.astype(np.float32), -1, kernel) ** 2
        )
        den_local_var = (
            cv2.filter2D(den_gray.astype(np.float32) ** 2, -1, kernel)
            - cv2.filter2D(den_gray.astype(np.float32), -1, kernel) ** 2
        )
        texture_preservation = np.corrcoef(
            orig_local_var.flatten(), den_local_var.flatten()
        )[0, 1]

        return {
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim_enhanced": float(ssim_approx),
            "noise_reduction_percent": float(noise_reduction),
            "edge_preservation": float(edge_correlation),
            "texture_preservation": float(texture_preservation),
            "original_std": float(orig_std),
            "denoised_std": float(den_std),
        }

    def process_single_image(
        self,
        input_path: str,
        output_path: str,
        sigma: Optional[float] = None,
        profile: str = "np",
        save_comparison: bool = False,
    ) -> dict:
        """
        Process single image with enhanced quality metrics.

        Args:
            input_path: Input image path
            output_path: Output image path
            sigma: Noise level (auto-estimated if None)
            profile: BM3D profile ('np' or 'refilter')
            save_comparison: Save side-by-side comparison

        Returns:
            Comprehensive processing results and metrics
        """
        start_time = time.time()

        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        logger.info(f"Processing {input_path} ({image.shape})")

        # Denoise with advanced method
        denoised, metadata = self.denoise_image_advanced(image, sigma, profile)

        # Compute enhanced quality metrics
        metrics = self.compute_enhanced_metrics(image, denoised)

        # Add processing time
        processing_time = time.time() - start_time
        metadata["processing_time"] = processing_time

        # Save denoised image
        cv2.imwrite(output_path, denoised)
        logger.info(f"Saved denoised image to {output_path} ({processing_time:.2f}s)")

        # Save comparison if requested
        if save_comparison:
            comparison_path = output_path.replace(".", "_comparison.")
            # Resize if images are too large for comparison
            if image.shape[1] > 2000:
                scale = 2000 / image.shape[1]
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                orig_resized = cv2.resize(image, (new_width, new_height))
                den_resized = cv2.resize(denoised, (new_width, new_height))
                comparison = np.hstack([orig_resized, den_resized])
            else:
                comparison = np.hstack([image, denoised])
            cv2.imwrite(comparison_path, comparison)
            logger.info(f"Saved comparison to {comparison_path}")

        return {
            "input_path": input_path,
            "output_path": output_path,
            "processing_metadata": metadata,
            "quality_metrics": metrics,
        }

    def process_batch_parallel(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.jpg",
        sigma: Optional[float] = None,
        profile: str = "np",
        max_workers: Optional[int] = None,
        max_memory_gb: float = 8.0,
    ) -> List[dict]:
        """
        Multi-threaded batch processing with memory management.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
            sigma: Fixed noise level (auto-estimated per image if None)
            profile: BM3D profile
            max_workers: Maximum number of threads
            max_memory_gb: Maximum memory per worker (GB)

        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find images - support both regular images and GeoTIFFs
        files = []
        for ext in [pattern, "*.tif", "*.tiff", "*.png", "*.bmp"]:
            files.extend(list(input_path.glob(ext)))

        # Remove duplicates while preserving order
        seen = set()
        files = [f for f in files if not (f in seen or seen.add(f))]

        logger.info(f"Found {len(files)} files to process")

        if not files:
            logger.warning(f"No files found matching patterns")
            return []

        # Adaptive worker count based on memory
        if max_workers is None:
            try:
                available_memory = psutil.virtual_memory().available / (1024**3)  # GB
                max_workers = max(
                    1, min(4, int(available_memory / max_memory_gb), len(files))
                )
            except:
                max_workers = min(2, mp.cpu_count(), len(files))

        logger.info(
            f"Using {max_workers} workers (available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB)"
        )

        # Process in batches to prevent memory exhaustion
        batch_size = max_workers * 2
        results = []

        for i in range(0, len(files), batch_size):
            batch_files = files[i : i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{math.ceil(len(files)/batch_size)}"
            )
            batch_results = self._process_batch_chunk(
                batch_files, output_path, max_workers, sigma, profile
            )
            results.extend(batch_results)

            # Force garbage collection between batches
            gc.collect()

        # Print summary
        if results:
            # Calculate metrics for results that have them
            results_with_metrics = [
                r
                for r in results
                if "quality_metrics" in r and "processing_metadata" in r
            ]

            if results_with_metrics:
                avg_psnr = np.mean(
                    [r["quality_metrics"]["psnr"] for r in results_with_metrics]
                )
                avg_texture = np.mean(
                    [
                        r["quality_metrics"]["texture_preservation"]
                        for r in results_with_metrics
                    ]
                )
                avg_time = np.mean(
                    [
                        r["processing_metadata"]["processing_time"]
                        for r in results_with_metrics
                    ]
                )

                logger.info(f"\nBatch processing summary:")
                logger.info(f"Successfully processed: {len(results)}/{len(files)}")
                logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
                logger.info(f"Average texture preservation: {avg_texture:.3f}")
                logger.info(f"Average processing time: {avg_time:.2f}s per image")
            else:
                logger.info(f"Successfully processed: {len(results)}/{len(files)}")

        return results

    def _process_batch_chunk(
        self,
        files: List[Path],
        output_dir: Path,
        max_workers: int,
        sigma: Optional[float],
        profile: str,
    ) -> List[dict]:
        """Process a chunk of files with proper resource management."""
        results = []
        failed_files = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for file_path in files:
                output_file = output_dir / f"denoised_{file_path.name}"

                # Check if it's a GeoTIFF
                if file_path.suffix.lower() in [".tif", ".tiff"]:
                    future = executor.submit(
                        self.process_geotiff_file,
                        str(file_path),
                        str(output_file),
                        sigma,
                    )
                else:
                    future = executor.submit(
                        self.process_single_image,
                        str(file_path),
                        str(output_file),
                        sigma,
                        profile,
                    )
                futures[future] = file_path

            for future in futures:
                file_path = futures[future]
                try:
                    result = future.result(timeout=600)  # 10-minute timeout per image
                    results.append(result)

                    # Log metrics if available
                    if "quality_metrics" in result:
                        metrics = result["quality_metrics"]
                        logger.info(
                            f"✓ {file_path.name}: PSNR={metrics['psnr']:.2f}dB, "
                            f"Noise reduction={metrics['noise_reduction_percent']:.1f}%"
                        )
                    else:
                        logger.info(f"✓ {file_path.name}: Processed successfully")

                except Exception as e:
                    logger.error(f"✗ Failed to process {file_path.name}: {e}")
                    failed_files.append(str(file_path))

        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files in this batch")

        return results


def main():
    """Enhanced CLI interface for BM3D denoising."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced BM3D Denoising for Aerial Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("output", help="Output image or directory")
    parser.add_argument(
        "--sigma",
        type=float,
        help="Noise standard deviation (auto-estimated if not provided)",
    )
    parser.add_argument(
        "--profile",
        choices=["np", "refilter"],
        default="np",
        help="BM3D profile (np=normal, refilter=higher quality)",
    )
    parser.add_argument("--batch", action="store_true", help="Process directory")
    parser.add_argument(
        "--pattern", default="*.jpg", help="File pattern for batch processing"
    )
    parser.add_argument(
        "--comparison", action="store_true", help="Save before/after comparison"
    )
    parser.add_argument(
        "--workers", type=int, help="Number of parallel workers for batch processing"
    )
    parser.add_argument(
        "--disable-yuv",
        action="store_true",
        help="Disable YUV colorspace processing optimization",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile size for large image processing",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between tiles for seamless blending",
    )
    parser.add_argument(
        "--max-memory", type=float, default=8.0, help="Maximum memory per worker (GB)"
    )

    args = parser.parse_args()

    # Initialize enhanced denoiser
    denoiser = EnhancedBM3DDenoiser(
        enable_yuv_processing=not args.disable_yuv,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    if args.batch:
        # Batch processing
        results = denoiser.process_batch_parallel(
            args.input,
            args.output,
            pattern=args.pattern,
            sigma=args.sigma,
            profile=args.profile,
            max_workers=args.workers,
            max_memory_gb=args.max_memory,
        )

        # Save detailed report
        if results:
            report_path = Path(args.output) / "processing_report.txt"
            with open(report_path, "w") as f:
                f.write("Enhanced BM3D Processing Report\n")
                f.write("=" * 50 + "\n\n")

                for result in results:
                    f.write(f"File: {Path(result['input_path']).name}\n")
                    f.write(
                        f"Processing method: {result['processing_metadata']['processing_method']}\n"
                    )
                    f.write(
                        f"Sigma used: {result['processing_metadata']['sigma_used']:.4f}\n"
                    )
                    f.write(f"PSNR: {result['quality_metrics']['psnr']:.2f} dB\n")
                    f.write(
                        f"Texture preservation: {result['quality_metrics']['texture_preservation']:.3f}\n"
                    )
                    f.write(
                        f"Processing time: {result['processing_metadata']['processing_time']:.2f}s\n"
                    )
                    f.write("-" * 30 + "\n")

            print(f"Detailed report saved to {report_path}")
    else:
        # Single image processing
        result = denoiser.process_single_image(
            args.input,
            args.output,
            sigma=args.sigma,
            profile=args.profile,
            save_comparison=args.comparison,
        )

        # Print results
        metadata = result["processing_metadata"]
        metrics = result["quality_metrics"]

        print(f"Processing complete!")
        print(f"Method: {metadata['processing_method']}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"Noise reduction: {metrics['noise_reduction_percent']:.1f}%")
        print(f"Edge preservation: {metrics['edge_preservation']:.3f}")
        print(f"Texture preservation: {metrics['texture_preservation']:.3f}")
        print(f"Processing time: {metadata['processing_time']:.2f}s")


if __name__ == "__main__":
    main()
