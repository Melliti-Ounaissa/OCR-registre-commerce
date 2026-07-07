"""
STEP 3 (UPDATED): DENOISING -- adjusted for a bigger upscaled image
======================================================================
This is the same denoising step as before, but with one important
update: the image is now much bigger (you changed the upscaling step
to 4.5x instead of 2x, giving a 2839x1984 image instead of 1262x882),
and the denoising settings need to be adjusted to match. Everything is
explained from zero below -- including a recap of concepts from Step 3
for anyone reading this file on its own.

------------------------------------------------------------------
RECAP: WHAT IS NOISE, AND WHAT DOES DENOISING DO?
------------------------------------------------------------------
"Noise" = small random errors in pixel brightness that shouldn't be
there (e.g. a blank white area that isn't perfectly, uniformly white).
We measure it with "standard deviation" in a patch that should be flat:
low number = clean, high number = grainy.

"Denoising" here uses an algorithm called Non-local Means Denoising.
For each pixel, it looks at a small patch around it, searches a wider
area of the image for OTHER patches that look similar, and averages
those together. Random noise cancels out when you average similar
patches; real structure (like the edge of a letter) doesn't cancel
out, so it survives. This preserves detail much better than simple
blurring.

------------------------------------------------------------------
WHY DO THE WINDOW-SIZE SETTINGS NEED TO CHANGE WHEN THE IMAGE GETS BIGGER?
------------------------------------------------------------------
Two of the three settings -- templateWindowSize and searchWindowSize --
are measured in raw pixels, not in any real-world unit like millimeters.

    - templateWindowSize: the size of the small comparison patch used
      to judge "does this area look similar to that area?"
    - searchWindowSize: how far out the algorithm searches for similar
      patches.

When you make the image 4.5x bigger, every real feature in the image
(a letter stroke, a gap between words, a bit of paper texture) now
occupies about 4.5x more pixels than before. If we don't also grow the
comparison window sizes, that little templateWindowSize patch ends up
covering a much SMALLER fraction of a real feature (like a letter's
stroke) than it used to -- so the "is this similar?" comparisons become
less meaningful, and the search for similar patches becomes less
effective at distinguishing real structure from noise.

The fix: scale templateWindowSize and searchWindowSize up by roughly
the same factor as the image was upscaled by.

------------------------------------------------------------------
THE THIRD SETTING: WHY "h" (FILTER STRENGTH) DOESN'T NEED THE SAME SCALING
------------------------------------------------------------------
"h" controls how aggressively pixel values get smoothed/pulled toward
their similar neighbors -- it's about the INTENSITY of the noise, not
the SIZE of the image. Making an image bigger via upscaling (with
cubic interpolation) doesn't fundamentally change how noisy the
original scan was, so h usually does not need to scale with resolution
the same way the window sizes do. We keep h=10 here (the same
moderate default as before), and only adjust the two window sizes.

------------------------------------------------------------------
HOW WE CHOSE THE NEW NUMBERS (with real measurements, not guesses)
------------------------------------------------------------------
We tested 3 options directly on this document at its new 2839x1984 size:

    Option                          | Noise left | Text sharpness | Time
    --------------------------------|------------|-----------------|------
    Old, unscaled (7, 21)           |    3.81    |      94.2       | 5.7s
    Moderately scaled (11, 31)      |    3.77    |      93.1       | 13.5s  <- used below
    Fully scaled x2.25 (15, 47)     |    3.69    |      92.3       | 29.8s

Fully scaling the windows in exact proportion to the resize gives only
a tiny extra noise reduction (3.69 vs 3.77) for more than DOUBLE the
processing time compared to the moderate option. So we use the
moderate, scaled-but-not-fully-linear values as a practical default.
The formula below computes this automatically from whatever scale
factor you used in the upscaling step, so if you change the scale
again later, you don't have to guess new numbers by hand.
"""

import cv2                         # OpenCV: for reading, denoising, and saving images
import os                          # For creating folders and building file paths
import time                        # To measure and report how long denoising takes
import matplotlib.pyplot as plt    # For drawing the before/after comparison figure


def round_to_odd(n: float) -> int:
    """
    OpenCV's window-size settings must be odd numbers (so there's always
    a single, well-defined center pixel in the window -- a 7x7 square has
    one true center pixel, but a 6x6 square does not). This helper takes
    any number and rounds it to the nearest odd integer.
    """
    n = int(round(n))
    if n % 2 == 0:
        n += 1
    return n


def compute_scaled_window_sizes(upscale_factor: float, moderation: float = 0.7):
    """
    Computes appropriately-sized templateWindowSize and searchWindowSize
    values for the CURRENT image resolution, based on how much it was
    upscaled.

    These base values (template=7, search=21) were tuned for a 2.0x
    upscale in Step 1's original testing. We scale them up in proportion
    to how much bigger the current upscale factor is.

    `moderation` (0.0 to 1.0) softens the scaling: 1.0 = scale the windows
    up in exact, full proportion to the upscale factor (most thorough,
    but noticeably slower for barely-better results, per our testing
    above); 0.0 = don't scale at all (fastest, but less effective on a
    much bigger image). We default to 0.7 as a practical middle ground,
    matching what we measured as the best speed/quality balance.
    """
    base_upscale_factor = 2.0     # the upscale factor these base values were tuned for
    base_template = 7
    base_search = 21

    raw_scale_ratio = upscale_factor / base_upscale_factor
    # blend between "no scaling" (ratio=1.0) and "full scaling" (raw_scale_ratio)
    # according to the moderation setting
    applied_ratio = 1.0 + (raw_scale_ratio - 1.0) * moderation

    template_size = round_to_odd(base_template * applied_ratio)
    search_size = round_to_odd(base_search * applied_ratio)

    print(f"[PARAM CALC] Image was upscaled {upscale_factor}x (base tuning was for {base_upscale_factor}x).")
    print(f"[PARAM CALC] Raw scale ratio = {raw_scale_ratio:.2f}, "
          f"moderation = {moderation} -> applied ratio = {applied_ratio:.2f}")
    print(f"[PARAM CALC] -> templateWindowSize = {template_size} (was {base_template})")
    print(f"[PARAM CALC] -> searchWindowSize   = {search_size} (was {base_search})")

    return template_size, search_size


def load_image_grayscale(path: str):
    """Reads an already-grayscale image file from disk (single brightness channel)."""
    print(f"[LOAD] Reading grayscale image from: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    h, w = img.shape
    print(f"[LOAD] Success. Size = {w}x{h} px, 1 channel (grayscale)")
    return img


def measure_noise(img, patch_coords, label: str):
    """
    Measures noise level in a small rectangular patch using standard
    deviation. patch_coords = (y1, y2, x1, x2). Lower = cleaner.
    """
    y1, y2, x1, x2 = patch_coords
    patch = img[y1:y2, x1:x2]
    noise_level = patch.std()
    print(f"[MEASURE] {label}: noise level (std dev) in flat patch = {noise_level:.2f} (lower = cleaner)")
    return noise_level


def denoise(gray_img, h: int, template_window_size: int, search_window_size: int):
    """Applies Non-local Means Denoising with the given (now resolution-aware) settings."""
    print(f"[DENOISE] Running Non-local Means Denoising "
          f"(h={h}, templateWindowSize={template_window_size}, searchWindowSize={search_window_size})...")
    print("[DENOISE] This may take longer than before, since the image is bigger. Please wait...")
    t0 = time.time()
    denoised_img = cv2.fastNlMeansDenoising(
        gray_img, h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )
    elapsed = time.time() - t0
    print(f"[DENOISE] Done in {elapsed:.1f} seconds.")
    return denoised_img


def save_visualization(before_img, after_img, flat_patch_coords, output_dir: str, base_filename: str):
    """Draws a 2-panel before/after comparison with the measured patch marked in red, saves as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    y1, y2, x1, x2 = flat_patch_coords

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(before_img, cmap='gray')
    axes[0].set_title(f"BEFORE (Grayscale, noisy)\nnoise std dev = {before_img[y1:y2, x1:x2].std():.2f}")
    axes[0].axis('off')
    axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2))

    axes[1].imshow(after_img, cmap='gray')
    axes[1].set_title(f"AFTER (Denoised)\nnoise std dev = {after_img[y1:y2, x1:x2].std():.2f}")
    axes[1].axis('off')
    axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2))

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_filename}_step3_denoise_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[VISUALIZATION] Saved before/after comparison to: {save_path}")
    print("[VISUALIZATION] Red rectangle marks the exact patch used to measure noise.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION -- change these for your own machine/files.
    # ------------------------------------------------------------------
    input_path = "D:/stage Mobilis/OCR-registre-commerce/output/REGISTRE-FRONT_gray_standard.png"
    output_dir = "D:/stage Mobilis/OCR-registre-commerce/output/"

    upscale_factor_used = 4.5   # <-- must match the scale factor you used in Step 1 (upscaling)
    moderation = 0.7             # 0.0 = don't scale windows, 1.0 = fully scale windows; 0.7 = practical default

    # A rectangle (y1, y2, x1, x2) known to be a flat/blank part of the page, scaled for
    # the new 2839x1984 resolution. Adjust if your document layout/size differs.
    flat_patch_coords = (112, 225, 225, 338)

    print("=" * 60)
    print("STARTING DENOISING STEP (resolution-aware parameters)")
    print("=" * 60)

    # 1. Load the standard grayscale image from Step 2
    gray_image = load_image_grayscale(input_path)

    # 2. Compute appropriately-scaled window sizes for this image's resolution
    template_size, search_size = compute_scaled_window_sizes(upscale_factor_used, moderation=moderation)

    # 3. Measure noise BEFORE denoising, for comparison
    measure_noise(gray_image, flat_patch_coords, label="BEFORE denoising")

    # 4. Apply denoising with the scaled parameters
    denoised_image = denoise(gray_image, h=10, template_window_size=template_size, search_window_size=search_size)

    # 5. Measure noise AFTER denoising, to prove it worked
    measure_noise(denoised_image, flat_patch_coords, label="AFTER denoising")

    # 6. Save the denoised image (feeds into the next step: contrast enhancement)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "REGISTRE-FRONT_denoised.png")
    cv2.imwrite(save_path, denoised_image)
    print(f"[SAVE] Denoised image saved to: {save_path}")

    # 7. Save the before/after visualization
    save_visualization(gray_image, denoised_image, flat_patch_coords, output_dir, "REGISTRE-FRONT")

    print("=" * 60)
    print("DENOISING STEP COMPLETE")
    print("=" * 60)