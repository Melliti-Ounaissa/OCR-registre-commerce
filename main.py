import cv2
import os
import json
import re
import pytesseract

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Ensure local directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# If Tesseract is not in your system PATH, uncomment and set the line below:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Step 1: Preprocessing with Watermark Removal ---
def preprocess_image(image_path, output_path):
    """
    Splits color channels to eliminate red/orange watermarks,
    then applies upscaling and clean binarization.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Split channels (OpenCV uses BGR order)
    b, g, r = cv2.split(img_color)

    # Use the Red channel as grayscale to eliminate the red SPECIMEN stamp
    img_gray = r

    # Upscale to make small, thick fonts readable
    img_upscaled = cv2.resize(img_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Denoise to smooth out edges of small fonts
    img_denoised = cv2.fastNlMeansDenoising(img_upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Binarize cleanly using adaptive thresholding
    img_binary = cv2.adaptiveThreshold(
        img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )

    cv2.imwrite(output_path, img_binary)
    return output_path

# --- Step 2: Local Classical OCR ---
def extract_text_with_tesseract(image_path):
    """
    Runs Tesseract locally with combined Arabic and French language data.
    --psm 3 enables fully automatic page segmentation layout analysis.
    """
    custom_config = r'--psm 3 -l ara+fra'
    raw_text = pytesseract.image_to_string(image_path, config=custom_config)
    return raw_text

# --- Step 3: Field Extraction & Validation ---
def validate_date(date_str):
    date_str = date_str.strip()
    patterns = [r"^\d{4}/\d{2}/\d{2}$", r"^\d{4}-\d{2}-\d{2}$"]
    return any(re.match(pattern, date_str) for pattern in patterns)

def autocorrect_date(date_str):
    """Attempts to fix common OCR misreads in dates before validation."""
    # Replace common misreads (L, l, |, \, _) with a slash
    corrected = re.sub(r'[Ll|\\_]', '/', date_str)
    
    # If the OCR missed the separators entirely (e.g., 20001204), try to inject them
    if re.match(r"^\d{8}$", corrected):
        corrected = f"{corrected[:4]}/{corrected[4:6]}/{corrected[6:]}"
        
    return corrected

def autocorrect_name(name_str):
    """Removes digits and special characters from fields that should strictly be text."""
    # Substitute any digit with an empty string
    return re.sub(r'\d+', '', name_str).strip()

def extract_and_validate_fields(raw_text):
    """Parses structural values, applies autocorrection, and validates."""
    extracted_data = {}
    validation_report = {"valid": True, "errors": []}
    
    lines = raw_text.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip()
            
            if not key or not value:
                continue

            # --- Autocorrection & Validation Rules ---
            
            # 1. Date Fields
            if "تاريخ" in key or "date" in key.lower():
                value = autocorrect_date(value)
                # Validate the newly corrected date
                if not re.match(r"^\d{4}[/-]\d{2}[/-]\d{2}$", value):
                    validation_report["errors"].append(f"Unfixable date format for '{key}': {value}")
                    validation_report["valid"] = False

            # 2. Name Fields (Assuming 'الاسم' and 'اللقب' are text-only)
            elif any(name_keyword in key for name_keyword in ["الاسم", "اللقب", "nom", "prenom"]):
                original_value = value
                value = autocorrect_name(value)
                if not value:
                    validation_report["errors"].append(f"Name field '{key}' became empty after stripping invalid characters from: {original_value}")
                    validation_report["valid"] = False

            extracted_data[key] = value

    return extracted_data, validation_report

# --- Core Execution Framework ---
def process_documents():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        print(f"\n--- Processing: {filename} ---")
        base_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)
        
        # 1. Image Adjustments
        preprocessed_path = os.path.join(OUTPUT_DIR, f"{base_name}_preprocessed.jpg")
        try:
            preprocess_image(input_path, preprocessed_path)
            print("Preprocessing complete (Red-channel filter applied).")
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            continue

        # 2. Local Token-Free OCR Execution
        try:
            raw_text = extract_text_with_tesseract(preprocessed_path)
            txt_path = os.path.join(OUTPUT_DIR, f"{base_name}_ocr.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print("Local bilingual OCR extraction complete.")
        except Exception as e:
            print(f"OCR execution failed: {e}")
            continue

        # 3. Data Extraction & Serialization
        fields, validation = extract_and_validate_fields(raw_text)
        
        json_output = {
            "file": filename,
            "validation_status": validation,
            "extracted_fields": fields
        }
        
        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_fields.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=4)
        
        print(f"Data successfully structured and saved to {json_path}")

if __name__ == "__main__":
    process_documents()