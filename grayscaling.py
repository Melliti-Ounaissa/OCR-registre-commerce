"""
STEP 2: GRAYSCALE CONVERSION (and reducing the "SPECIMEN" watermark)
======================================================================

Before reading this code, let's build up the concepts from zero.

------------------------------------------------------------------
WHAT IS A DIGITAL IMAGE, REALLY?
------------------------------------------------------------------
A digital image is just a big grid (table) of numbers. Each little
square in that grid is called a "pixel" (short for "picture element").

For a COLOR image, each pixel isn't just ONE number -- it's actually
THREE numbers stacked together:
    - How much RED light is at that spot        (0 = none, 255 = max)
    - How much GREEN light is at that spot       (0 = none, 255 = max)
    - How much BLUE light is at that spot        (0 = none, 255 = max)

Mixing different amounts of Red, Green and Blue light is how screens
(and printers) create every color you see -- this is called the "RGB"
color model. So a color image is really THREE grids of numbers stacked
on top of each other. In code, each of these grids is called a
"channel". A color image = 3 channels. This is why, in code, you'll
see img.shape print something like (441, 631, 3):
    441 = height in pixels, 631 = width in pixels, 3 = the 3 channels.

IMPORTANT GOTCHA: OpenCV (cv2) stores channels in the order
Blue, Green, Red ("BGR") instead of the more common "RGB" order. This
is just a historical quirk of the library -- so whenever you split a
color image with OpenCV, remember the order is B, G, R, not R, G, B.

------------------------------------------------------------------
WHAT IS "GRAYSCALE"?
------------------------------------------------------------------
"Grayscale" means an image with only ONE number per pixel instead of
three. That single number represents brightness only: 0 = pure black,
255 = pure white, and everything in between is a shade of gray. There's
no color information left at all -- just "how light or dark is this
spot".

------------------------------------------------------------------
WHY CONVERT TO GRAYSCALE BEFORE OCR?
------------------------------------------------------------------
OCR (the technology that reads text from images) doesn't care what
color the ink or paper is -- it only cares about CONTRAST: is this
spot dark ink, or light paper? Color is irrelevant, extra information
that just makes the computer's job harder and slower. So we simplify:
    - 3 channels (Red, Green, Blue) -> 1 channel (just brightness)
    - This is 3x less data to process
    - Many OCR and image-processing functions (like the thresholding
      and denoising steps that come later) *require* a single-channel
      (grayscale) image as input -- they don't work directly on color.

------------------------------------------------------------------
TWO WAYS TO MAKE A GRAYSCALE IMAGE (and why we need a special trick here)
------------------------------------------------------------------
METHOD A -- "Standard" grayscale conversion (cv2.cvtColor):
    This is the normal, textbook way. Instead of just averaging the
    3 numbers equally, it uses a weighted formula that mimics how
    human eyes perceive brightness (we're most sensitive to green
    light, least to blue):
        gray = 0.299*Red + 0.587*Green + 0.114*Blue
    This works great for normal photos.

METHOD B -- Picking a single color channel on its own:
    Your document has a large, faint green "SPECIMEN" watermark
    stamped diagonally across it. That watermark has a slight GREEN
    color bias, while the real black printed text has NO color bias
    (true black/dark gray ink looks equally dark in every channel).

    We tested this on your actual image. At a spot where the green
    watermark stroke is, the raw pixel values were approximately:
        Blue = 224,  Green = 241,  Red = 220
    ...notice Green is the brightest of the three there -- that's the
    watermark's color signature. Meanwhile, at a spot with real black
    printed text, the values were approximately:
        Blue = 57,   Green = 59,   Red = 56
    ...all three channels are low and nearly equal -- there's no color
    bias in real black ink.

    Because of this, if we throw away the Green and Blue channels and
    keep ONLY the Red channel, the green-tinted watermark pixels
    (which have a relatively high Red value, close to the white paper)
    get pushed closer to white and become fainter/less visible, while
    the black text (which is low in every channel, including Red)
    stays just as dark and readable. This is a simple, cheap trick to
    suppress a colored watermark without needing fancier tools.

    In code, this is done with:
        b, g, r = cv2.split(img)   # split the image into 3 separate grids
        gray = r                  # keep only the Red channel, discard the rest
"""

import cv2                         # OpenCV: for reading/splitting/converting/saving images
import os                          # For creating folders and building file paths
import numpy as np                 # For numeric operations on the pixel grids
import matplotlib.pyplot as plt    # For drawing the before/after comparison figure


def load_image(path: str):
    """Reads an image file from disk into memory."""
    print(f"[LOAD] Reading image from: {path}")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    h, w, channels = img.shape
    print(f"[LOAD] Success. Size = {w}x{h} px, Channels = {channels} (Blue, Green, Red)")
    return img


def standard_grayscale(img):
    """
    METHOD A: The normal/textbook way to convert a color image to grayscale.
    Uses the weighted formula: 0.299*Red + 0.587*Green + 0.114*Blue
    """
    print("[GRAYSCALE - Standard] Converting using cv2.cvtColor (weighted RGB average)...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[GRAYSCALE - Standard] Done. Average brightness = {gray.mean():.1f} / 255")
    return gray


def red_channel_grayscale(img):
    """
    METHOD B: Instead of averaging all 3 channels, keep ONLY the Red channel.
    This suppresses the green-tinted "SPECIMEN" watermark more effectively
    than a standard grayscale average would, while keeping true black text dark.
    """
    print("[GRAYSCALE - Red channel] Splitting image into Blue, Green, Red channels...")
    b, g, r = cv2.split(img)   # remember: OpenCV order is Blue, Green, Red
    print(f"[GRAYSCALE - Red channel]   Blue channel avg brightness  = {b.mean():.1f} / 255")
    print(f"[GRAYSCALE - Red channel]   Green channel avg brightness = {g.mean():.1f} / 255")
    print(f"[GRAYSCALE - Red channel]   Red channel avg brightness   = {r.mean():.1f} / 255  <- keeping this one")
    print("[GRAYSCALE - Red channel] Done. Discarding Blue and Green channels.")
    return r


def save_visualization(original_bgr, gray_standard, gray_red_channel, output_dir: str, base_filename: str):
    """
    Draws a 3-panel comparison: original color, standard grayscale,
    and the red-channel version (watermark suppressed). Saves as a PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Convert BGR -> RGB just for correct display in matplotlib (doesn't affect saved data files)
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("BEFORE\nOriginal color image")
    axes[0].axis('off')

    axes[1].imshow(gray_standard, cmap='gray')
    axes[1].set_title("Method A\nStandard grayscale\n(watermark still visible)")
    axes[1].axis('off')

    axes[2].imshow(gray_red_channel, cmap='gray')
    axes[2].set_title("Method B\nRed channel only\n(watermark suppressed)")
    axes[2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{base_filename}_step2_grayscale_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[VISUALIZATION] Saved before/after comparison to: {save_path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION -- change these for your own machine/files.
    # We use the UPSCALED image produced by the previous step (Step 1)
    # as the input here, so the pipeline chains together.
    # ------------------------------------------------------------------
    input_path = "D:/stage Mobilis/OCR-registre-commerce/output/REGISTRE-FRONT_upscaled.png"
    output_dir = "D:/stage Mobilis/OCR-registre-commerce/output/"

    print("=" * 60)
    print("STARTING GRAYSCALE CONVERSION STEP")
    print("=" * 60)

    # 1. Load the (already upscaled) image
    color_image = load_image(input_path)

    # 2. Convert using Method A (standard grayscale) -- for comparison purposes
    gray_standard = standard_grayscale(color_image)

    # 3. Convert using Method B (red channel only) -- this is the one we'll
    #    actually use going forward in the pipeline, since it suppresses the watermark
    gray_final = red_channel_grayscale(color_image)

    # 4. Save both grayscale results to disk
    os.makedirs(output_dir, exist_ok=True)
    path_a = os.path.join(output_dir, "REGISTRE-FRONT_gray_standard.png")
    path_b = os.path.join(output_dir, "REGISTRE-FRONT_gray_red_channel.png")
    cv2.imwrite(path_a, gray_standard)
    cv2.imwrite(path_b, gray_final)
    print(f"[SAVE] Standard grayscale saved to: {path_a}")
    print(f"[SAVE] Red-channel (watermark-suppressed) grayscale saved to: {path_b}")

    # 5. Save the 3-panel visual comparison
    save_visualization(color_image, gray_standard, gray_final, output_dir, "REGISTRE-FRONT")

    print("=" * 60)
    print("GRAYSCALE CONVERSION STEP COMPLETE")
    print("=" * 60)