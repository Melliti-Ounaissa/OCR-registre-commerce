import cv2      
import os       
import matplotlib.pyplot as plt   


def load_image(path: str):
    print(f"[LOAD] Reading image from: {path}")
    img = cv2.imread(path)  
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    height, width = img.shape[:2]
    print(f"[LOAD] Success. Original size = {width}px (width) x {height}px (height)")
    return img


def upscale(img, scale: float = 3):
    h, w = img.shape[:2]              
    new_width = int(w * scale)
    new_height = int(h * scale)
    new_size = (new_width, new_height)

    print(f"[UPSCALE] Resizing from {w}x{h} to {new_width}x{new_height} "
          f"(scale factor = {scale}x) using CUBIC interpolation...")

    upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

    print("[UPSCALE] Done.")
    return upscaled_img


def save_visualization(original, upscaled, output_dir: str, base_filename: str):
    """
    Creates a side-by-side "before vs after" picture so a human can visually
    confirm the upscaling worked, and saves it as a PNG file in output_dir.
    This does NOT modify the actual image data -- it's just for our own inspection.
    """
    os.makedirs(output_dir, exist_ok=True)  # create the output folder if it doesn't exist yet

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # OpenCV loads colors as BGR (Blue-Green-Red) order, but matplotlib expects
    # RGB (Red-Green-Blue) order, so we convert before displaying, otherwise
    # colors would look swapped/wrong on screen.
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"BEFORE (Original)\n{original.shape[1]}x{original.shape[0]} px")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"AFTER (Upscaled)\n{upscaled.shape[1]}x{upscaled.shape[0]} px")
    axes[1].axis('off')

    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{base_filename}_step1_upscale_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # close the figure to free up memory

    print(f"[VISUALIZATION] Saved before/after comparison to: {save_path}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION -- change these two lines for your own machine/files
    # ------------------------------------------------------------------
    input_path = "D:/stage Mobilis/OCR-registre-commerce/input/REGISTRE-FRONT.png"
    output_dir = "D:/stage Mobilis/OCR-registre-commerce/output/"
    scale_factor = 4.5   # try 2.0 or 3.0 -- higher = bigger image, but diminishing returns

    print("=" * 60)
    print("STARTING UPSCALING STEP")
    print("=" * 60)

    # 1. Load the original image from disk
    original_image = load_image(input_path)

    # 2. Upscale it
    upscaled_image = upscale(original_image, scale=scale_factor)

    # 3. Save the actual upscaled image file (so later steps / OCR can use it)
    upscaled_save_path = os.path.join(output_dir, "REGISTRE-FRONT_upscaled.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(upscaled_save_path, upscaled_image)
    print(f"[SAVE] Upscaled image saved to: {upscaled_save_path}")

    # 4. Save a before/after visualization figure so we can visually inspect the result
    save_visualization(original_image, upscaled_image, output_dir, "REGISTRE-FRONT")

    print("=" * 60)
    print("UPSCALING STEP COMPLETE")
    print("=" * 60)