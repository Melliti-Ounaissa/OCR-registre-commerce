import cv2
import os
import json
import re
import base64
import requests

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Set this in your environment: export OPENROUTER_API_KEY="sk-or-..."
# Get a key at https://openrouter.ai/keys (pay-as-you-go, no subscription needed)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VLM_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# Set to True if you want to keep the old red-channel/binarization step.
# Recommended: False. VLMs handle watermarks and color noise natively,
# and binarization can strip context the model would otherwise use.
USE_PREPROCESSING = False

# Ensure local directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: (Optional) Preprocessing with Watermark Removal ---
def preprocess_image(image_path, output_path):
    """
    Splits color channels to eliminate red/orange watermarks,
    then applies upscaling and clean binarization.
    Kept for reference / classical-OCR fallback; not used by default
    when calling the VLM.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read image: {image_path}")

    b, g, r = cv2.split(img_color)
    img_gray = r
    img_upscaled = cv2.resize(img_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img_denoised = cv2.fastNlMeansDenoising(img_upscaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
    img_binary = cv2.adaptiveThreshold(
        img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )

    cv2.imwrite(output_path, img_binary)
    return output_path

# --- Step 2: OCR via Qwen3-VL-235B-A22B-Instruct (hosted, through OpenRouter) ---
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_text_with_qwen(image_path):
    """
    Sends the image to Qwen3-VL-235B-A22B-Instruct via OpenRouter's
    OpenAI-compatible /chat/completions endpoint and returns the raw
    extracted text. No local GPU/VRAM required.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a key at https://openrouter.ai/keys"
        )

    b64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    prompt = (
        "You are an OCR engine for Algerian government administrative documents "
        "(registre de commerce), written in Arabic with some French. "
        "Extract every visible text field from this document, preserving the "
        "original key/value structure exactly as printed, one field per line, "
        "formatted strictly as 'key: value'. Do not translate anything, do not "
        "summarize, and do not add commentary — output only the extracted fields."
    )

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

# --- Step 3: Field Extraction & Validation ---
def validate_date(date_str):
    date_str = date_str.strip()
    patterns = [r"^\d{4}/\d{2}/\d{2}$", r"^\d{4}-\d{2}-\d{2}$"]
    return any(re.match(pattern, date_str) for pattern in patterns)

def autocorrect_date(date_str):
    """Attempts to fix common OCR misreads in dates before validation."""
    corrected = re.sub(r'[Ll|\\_]', '/', date_str)
    if re.match(r"^\d{8}$", corrected):
        corrected = f"{corrected[:4]}/{corrected[4:6]}/{corrected[6:]}"
    return corrected

def autocorrect_name(name_str):
    """Removes digits and special characters from fields that should strictly be text."""
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

            if "تاريخ" in key or "date" in key.lower():
                value = autocorrect_date(value)
                if not re.match(r"^\d{4}[/-]\d{2}[/-]\d{2}$", value):
                    validation_report["errors"].append(f"Unfixable date format for '{key}': {value}")
                    validation_report["valid"] = False

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

        # 1. (Optional) Image Adjustments
        ocr_input_path = input_path
        if USE_PREPROCESSING:
            preprocessed_path = os.path.join(OUTPUT_DIR, f"{base_name}_preprocessed.jpg")
            try:
                preprocess_image(input_path, preprocessed_path)
                ocr_input_path = preprocessed_path
                print("Preprocessing complete (Red-channel filter applied).")
            except Exception as e:
                print(f"Preprocessing failed: {e}")
                continue

        # 2. OCR via Qwen3-VL-235B-A22B-Instruct
        try:
            raw_text = extract_text_with_qwen(ocr_input_path)
            txt_path = os.path.join(OUTPUT_DIR, f"{base_name}_ocr.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print("Qwen3-VL OCR extraction complete.")
        except Exception as e:
            print(f"OCR execution failed: {e}")
            continue

        # 3. Data Extraction & Serialization
        fields, validation = extract_and_validate_fields(raw_text)

        json_output = {
            "file": filename,
            "validation_status": validation,
            "extracted_fields": fields,
        }

        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_fields.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=4)

        print(f"Data successfully structured and saved to {json_path}")

if __name__ == "__main__":
    process_documents()