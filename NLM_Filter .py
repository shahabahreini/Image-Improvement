import os
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import gaussian
import argparse
import time
from tqdm import tqdm


def load_rgb_geotiff(image_path):
    print("Loading GeoTIFF image...")
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs

        print(f"✓ Original image shape: {image.shape}")
        print(f"✓ Original channels: {src.count}")
        print(f"✓ Original dtype: {src.dtypes[0]}")

    print("Converting image format...")
    image = reshape_as_image(image)

    # Ensure we only take RGB channels and normalize to 0-1 range
    if image.shape[2] >= 3:
        image = image[:, :, :3]  # Take only first 3 channels (RGB)
    else:
        raise ValueError(
            f"Image only has {image.shape[2]} channels, need at least 3 for RGB"
        )

    print("Normalizing image data...")
    # Normalize to 0-1 range based on data type
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float64) / 65535.0
    elif image.dtype == np.float32 or image.dtype == np.float64:
        # Assume already in 0-1 range, but clip just in case
        image = np.clip(image.astype(np.float64), 0, 1)
    else:
        # For other types, normalize by max value
        image = image.astype(np.float64)
        if image.max() > 1:
            image = image / image.max()

    print("✓ Image loaded and preprocessed successfully")
    return image, profile, transform, crs


def apply_filter_per_channel(image, filter_func):
    print("Applying filter to individual channels...")
    channels = []
    for c in tqdm(range(3), desc="Processing channels"):
        channels.append(filter_func(image[:, :, c]))
    return np.stack(channels, axis=2)


def apply_nlm_rgb(image):
    """
    Apply Non-Local Means denoising to RGB image.
    Process all channels together for better performance and color preservation.
    """
    print("Starting Non-Local Means denoising...")
    print("Estimating noise parameters...")
    # Estimate noise standard deviation across all channels
    sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))
    print(f"✓ Estimated noise sigma: {sigma_est:.4f}")

    # Process all channels together - this is much faster and preserves color relationships
    patch_kw = dict(
        patch_size=5,  # Size of patches used for comparison
        patch_distance=6,  # Maximum distance to search for patches
        channel_axis=-1,  # Specify that channels are in the last dimension
    )

    print("Applying Non-Local Means filter (this may take a while)...")
    start_time = time.time()
    # Apply NLM denoising to all channels simultaneously
    denoised = denoise_nl_means(
        image,
        h=1.1 * sigma_est,  # Filtering strength
        sigma=sigma_est,  # Noise standard deviation
        fast_mode=True,  # Use faster approximation
        **patch_kw,
    )
    elapsed = time.time() - start_time
    print(f"✓ Non-Local Means completed in {elapsed:.1f} seconds")

    return denoised


def apply_nlm_rgb_fast(image):
    """
    Faster version of NLM with reduced patch parameters.
    """
    sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))

    patch_kw = dict(
        patch_size=3,  # Smaller patch size for speed
        patch_distance=4,  # Smaller search distance for speed
        channel_axis=-1,
    )

    denoised = denoise_nl_means(
        image,
        h=1.1 * sigma_est,  # Slightly reduced filtering strength
        sigma=sigma_est,
        fast_mode=True,
        **patch_kw,
    )

    return denoised


def apply_nlm_rgb_ultrafast(image):
    """
    Ultra-fast version that processes at reduced resolution for large images.
    """
    from skimage.transform import rescale, resize

    h, w = image.shape[:2]
    print(f"Image dimensions: {h} x {w}")

    # If image is large, process at half resolution
    if h > 2000 or w > 2000:
        print("Large image detected - processing at reduced resolution for speed...")
        # Downsample with less anti-aliasing to preserve more detail
        scale = 0.5
        print(f"Downsampling to {scale}x resolution...")
        small_image = rescale(image, scale, anti_aliasing=False, channel_axis=-1)

        # Process smaller image with reduced smoothing
        print("Estimating noise parameters...")
        sigma_est = np.mean(estimate_sigma(small_image, channel_axis=-1))
        print(f"✓ Estimated noise sigma: {sigma_est:.4f}")

        print("Applying Non-Local Means to downsampled image...")
        start_time = time.time()
        denoised_small = denoise_nl_means(
            small_image,
            h=0.4 * sigma_est,  # Reduced from 0.6 for less smoothing
            sigma=sigma_est,
            patch_size=3,
            patch_distance=4,
            fast_mode=True,
            channel_axis=-1,
        )
        elapsed = time.time() - start_time
        print(f"✓ Filtering completed in {elapsed:.1f} seconds")

        # Upsample back with less anti-aliasing to preserve edges
        print("Upsampling back to original resolution...")
        denoised = rescale(
            denoised_small, 1 / scale, anti_aliasing=False, channel_axis=-1
        )

        # Ensure output has same shape as input
        if denoised.shape[:2] != image.shape[:2]:
            print("Adjusting output dimensions...")
            denoised = resize(denoised, image.shape[:2], anti_aliasing=False)

        print("✓ Ultra-fast processing completed")
        return denoised
    else:
        # Process normally for smaller images
        print("Processing at full resolution...")
        return apply_nlm_rgb_fast(image)


def apply_nlm_rgb_ultrafast_sharp(image):
    """
    Ultra-fast NLM with post-processing sharpening.
    """
    from skimage.filters import gaussian

    print("Applying ultra-fast NLM + sharpening...")
    # Apply the ultra-fast NLM
    denoised = apply_nlm_rgb_ultrafast(image)

    print("Applying post-processing sharpening...")

    # Apply moderate sharpening directly here
    def moderate_unsharp_channel(channel):
        # Create blurred version
        blurred = gaussian(channel, sigma=1.2)
        # Create sharpened version with moderate parameters
        sharpened = channel + 0.6 * (channel - blurred)
        return np.clip(sharpened, 0, 1)

    sharpened = apply_filter_per_channel(denoised, moderate_unsharp_channel)
    print("✓ Sharpening completed")

    return sharpened


def apply_nlm_rgb_ultrafast_sharp_strong(image):
    """
    Ultra-fast NLM with strong post-processing sharpening.
    """
    from skimage.filters import gaussian

    print("Applying ultra-fast NLM + strong sharpening...")
    # Apply the ultra-fast NLM
    denoised = apply_nlm_rgb_ultrafast(image)

    print("Applying strong post-processing sharpening...")

    # Apply strong sharpening with radius=2.0 and amount=0.75
    def strong_unsharp_channel(channel):
        # Create blurred version with radius=2.0
        blurred = gaussian(channel, sigma=2.0)
        # Create sharpened version with amount=0.75
        sharpened = channel + 0.75 * (channel - blurred)
        return np.clip(sharpened, 0, 1)

    sharpened = apply_filter_per_channel(denoised, strong_unsharp_channel)
    print("✓ Strong sharpening completed")

    return sharpened


def save_to_geotiff(filtered_rgb, profile, transform, crs, output_path):
    print("Preparing image for saving...")
    print(f"✓ Filtered image shape: {filtered_rgb.shape}")
    print(f"✓ Filtered image dtype: {filtered_rgb.dtype}")
    print(
        f"✓ Filtered image range: {filtered_rgb.min():.3f} to {filtered_rgb.max():.3f}"
    )

    print("Converting to 16-bit format...")
    # Ensure the image is in the correct range [0, 1]
    filtered_rgb = np.clip(filtered_rgb, 0, 1)

    # Convert to 16-bit unsigned integer
    filtered_rgb = (filtered_rgb * 65535).astype(np.uint16)

    # Ensure we have exactly 3 channels
    if filtered_rgb.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {filtered_rgb.shape[2]}")

    print("Reorganizing array dimensions...")
    # Convert from HWC to CHW format for rasterio
    filtered_rgb = np.transpose(filtered_rgb, (2, 0, 1))

    print(f"✓ Final array shape for saving: {filtered_rgb.shape}")

    print("Creating output profile...")
    # Create a clean profile for RGB output
    output_profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": None,
        "width": filtered_rgb.shape[2],
        "height": filtered_rgb.shape[1],
        "count": 3,  # Exactly 3 channels for RGB
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": False,
        "interleave": "pixel",  # Change to pixel interleave for better RGB support
        "photometric": "rgb",  # Explicitly specify RGB photometric interpretation
    }

    print(f"Writing to file: {output_path}")
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(filtered_rgb)
        # Set color interpretation explicitly
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        ]

    print("✓ File saved successfully")


def get_filter_function(filter_name):
    filters = {
        "nlm": apply_nlm_rgb,
        "nlm_fast": apply_nlm_rgb_fast,
        "nlm_ultrafast": apply_nlm_rgb_ultrafast,
        "nlm_ultrafast_sharp": apply_nlm_rgb_ultrafast_sharp,
        "nlm_ultrafast_sharp_strong": apply_nlm_rgb_ultrafast_sharp_strong,
    }
    return filters.get(filter_name.lower())


def main(input_path, output_path, filter_name):
    print("=" * 50)
    print(f"Starting image filtering process")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Filter: {filter_name}")
    print("=" * 50)

    # Load image
    image, profile, transform, crs = load_rgb_geotiff(input_path)
    print(f"✓ Image loaded - Shape: {image.shape}")

    # Get and apply filter
    filter_func = get_filter_function(filter_name)
    if filter_func is None:
        raise ValueError(f"Unknown filter: {filter_name}")

    print(f"Applying '{filter_name}' filter...")
    overall_start = time.time()
    filtered = filter_func(image)
    overall_elapsed = time.time() - overall_start

    print(f"✓ Filtering completed in {overall_elapsed:.1f} seconds total")

    # Save result
    save_to_geotiff(filtered, profile, transform, crs, output_path)

    print("=" * 50)
    print("✓ PROCESSING COMPLETE!")
    print(f"Total processing time: {overall_elapsed:.1f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply RGB filter to GeoTIFF")
    parser.add_argument("input_path", type=str, help="Path to input GeoTIFF")
    parser.add_argument("output_path", type=str, help="Path to save filtered GeoTIFF")
    parser.add_argument(
        "filter_name",
        type=str,
        help="Filter to apply: nlm, nlm_fast, nlm_ultrafast, nlm_ultrafast_sharp, nlm_ultrafast_sharp_strong",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.filter_name)
