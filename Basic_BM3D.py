#!/usr/bin/env python3
"""
BM3D Denoising for Aerial Images - DEM Processing Enhancement
Focused implementation for texture-preserving noise reduction.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging

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
    print("Warning: scikit-image not available. Noise estimation will be approximate.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM3DDenoiser:
    """BM3D denoising optimized for aerial imagery DEM processing."""

    def __init__(self):
        """Initialize BM3D denoiser with optimal parameters for aerial imagery."""
        if not BM3D_AVAILABLE:
            raise ImportError("BM3D library is required")

    def estimate_noise_sigma(self, image: np.ndarray) -> float:
        """
        Estimate noise standard deviation.

        Args:
            image: Input image (uint8)

        Returns:
            Estimated noise sigma
        """
        # Convert to float [0,1] range
        img_float = image.astype(np.float32) / 255.0

        if SKIMAGE_AVAILABLE:
            if len(image.shape) == 3:
                sigma = estimate_sigma(img_float, channel_axis=-1, average_sigmas=True)
            else:
                sigma = estimate_sigma(img_float, channel_axis=None)
        else:
            # Simple noise estimation using Laplacian variance
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sigma = np.sqrt(laplacian_var) / 255.0

        return float(sigma)

    def denoise_image(
        self, image: np.ndarray, sigma: Optional[float] = None, profile: str = "np"
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply BM3D denoising to aerial image.

        Args:
            image: Input image (uint8)
            sigma: Noise standard deviation (auto-estimated if None)
            profile: BM3D profile ('np' for normal, 'refilter' for better quality)

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

        metadata = {
            "sigma_used": sigma,
            "profile": profile,
            "input_shape": image.shape,
            "input_dtype": str(image.dtype),
        }

        # Apply BM3D denoising
        if len(image.shape) == 3:
            # Color image - process each channel
            denoised = np.zeros_like(img_float)
            for c in range(3):
                # Apply BM3D denoising - simplified approach
                denoised[:, :, c] = bm3d.bm3d(img_float[:, :, c], sigma)
        else:
            # Grayscale image - simplified approach
            denoised = bm3d.bm3d(img_float, sigma)

        # Convert back to uint8
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)

        return denoised_uint8, metadata

    def compute_quality_metrics(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> dict:
        """
        Compute image quality metrics for comparison.

        Args:
            original: Original noisy image
            denoised: Denoised image

        Returns:
            Dictionary of quality metrics
        """
        # Convert to float for calculations
        orig_float = original.astype(np.float32)
        den_float = denoised.astype(np.float32)

        # MSE and PSNR
        mse = np.mean((orig_float - den_float) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # SSIM approximation using correlation coefficient
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            den_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            den_gray = denoised

        # Structural similarity approximation
        corr_coeff = np.corrcoef(orig_gray.flatten(), den_gray.flatten())[0, 1]
        ssim_approx = max(0, corr_coeff)

        # Noise reduction (difference in standard deviation)
        orig_std = np.std(orig_float)
        den_std = np.std(den_float)
        noise_reduction = (orig_std - den_std) / orig_std * 100

        # Edge preservation (compare gradients)
        orig_grad = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)
        den_grad = cv2.Sobel(den_gray, cv2.CV_64F, 1, 1, ksize=3)
        edge_preservation = np.corrcoef(
            np.array(orig_grad.flat), np.array(den_grad.flat)
        )[0, 1]

        return {
            "mse": float(mse),
            "psnr": float(psnr),
            "ssim_approx": float(ssim_approx),
            "noise_reduction_percent": float(noise_reduction),
            "edge_preservation": float(edge_preservation),
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
        Process single image file.

        Args:
            input_path: Input image path
            output_path: Output image path
            sigma: Noise level (auto-estimated if None)
            profile: BM3D profile
            save_comparison: Save side-by-side comparison

        Returns:
            Processing results and metrics
        """
        # Load image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not load image: {input_path}")

        logger.info(f"Processing {input_path} ({image.shape})")

        # Denoise
        denoised, metadata = self.denoise_image(image, sigma, profile)

        # Compute quality metrics
        metrics = self.compute_quality_metrics(image, denoised)

        # Save denoised image
        cv2.imwrite(output_path, denoised)
        logger.info(f"Saved denoised image to {output_path}")

        # Save comparison if requested
        if save_comparison:
            comparison_path = output_path.replace(".", "_comparison.")
            comparison = np.hstack([image, denoised])
            cv2.imwrite(comparison_path, comparison)

        # Combine results
        results = {
            "input_path": input_path,
            "output_path": output_path,
            "processing_metadata": metadata,
            "quality_metrics": metrics,
        }

        return results

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.jpg",
        sigma: Optional[float] = None,
        profile: str = "np",
    ) -> list:
        """
        Process batch of images.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pattern: File pattern to match
            sigma: Fixed noise level (auto-estimated per image if None)
            profile: BM3D profile

        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find images
        files = list(input_path.glob(pattern))
        logger.info(f"Found {len(files)} files to process")

        results = []

        for file_path in files:
            try:
                output_file = output_path / f"denoised_{file_path.name}"

                result = self.process_single_image(
                    str(file_path), str(output_file), sigma=sigma, profile=profile
                )

                results.append(result)

                # Log metrics
                metrics = result["quality_metrics"]
                logger.info(
                    f"{file_path.name}: PSNR={metrics['psnr']:.2f}dB, "
                    f"Noise reduction={metrics['noise_reduction_percent']:.1f}%"
                )

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return results


def main():
    """CLI interface for BM3D denoising."""
    import argparse

    parser = argparse.ArgumentParser(description="BM3D Denoising for Aerial Images")
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

    args = parser.parse_args()

    denoiser = BM3DDenoiser()

    if args.batch:
        results = denoiser.process_batch(
            args.input,
            args.output,
            pattern=args.pattern,
            sigma=args.sigma,
            profile=args.profile,
        )

        # Print summary
        if results:
            avg_psnr = np.mean([r["quality_metrics"]["psnr"] for r in results])
            avg_noise_reduction = np.mean(
                [r["quality_metrics"]["noise_reduction_percent"] for r in results]
            )
            print(f"\nBatch processing complete!")
            print(f"Average PSNR: {avg_psnr:.2f} dB")
            print(f"Average noise reduction: {avg_noise_reduction:.1f}%")
    else:
        result = denoiser.process_single_image(
            args.input,
            args.output,
            sigma=args.sigma,
            profile=args.profile,
            save_comparison=args.comparison,
        )

        metrics = result["quality_metrics"]
        print(f"Processing complete!")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"Noise reduction: {metrics['noise_reduction_percent']:.1f}%")
        print(f"Edge preservation: {metrics['edge_preservation']:.3f}")


if __name__ == "__main__":
    main()
