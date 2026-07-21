import cv2
import os
import json
import re
import base64
import requests
import time

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VLM_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

USE_PREPROCESSING = True

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Preprocessing (Watermark Removal + Grayscale Enhancement) ---
def preprocess_image(image_path, output_path):
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Extract Red channel (r) to naturally eliminate red watermarks
    b, g, r = cv2.split(img_color)
    img_gray = r

    # Upscale resolution for higher letter precision
    img_upscaled = cv2.resize(img_gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Use CLAHE to boost contrast WITHOUT destroying faint strokes or handwritten text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_upscaled)

    cv2.imwrite(output_path, img_enhanced)
    return output_path

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Step 2: Pure Plain Text OCR Transcription ---
def extract_raw_plain_text(image_path):
    """Transcribes all text on the document line-by-line as clean plain text."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    b64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    prompt = (
        "You are an expert OCR engine. Transcribe all visible text from this document image "
        "into clean plain text in standard top-to-bottom reading order.\n\n"
        "RULES:\n"
        "1. Do NOT output JSON, key-value pairs, or markdown formatting.\n"
        "2. Preserve the exact words, numbers, and layout line-by-line.\n"
        "3. Include all field labels, filled values, numbers, and boilerplate text.\n"
        "4. Do not add commentary or explanations."
    )

    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Step 3: Structuring Plain Text into JSON Fields (WITH SMART TABLE HANDLING) ---
def extract_fields_to_json(image_path, raw_ocr_text):
    """Structures form fields and tables into clean JSON using advanced spatial rules."""
    b64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    prompt = f"""You are a highly precise document structuring AI. Analyze the document image and its plain text transcription to extract structured fields into a valid JSON object.

RAW TRANSCRIPTION:
{raw_ocr_text}

SMART TABLE & GRID HANDLING RULES (CRITICAL):
1. RIGHT-TO-LEFT ALIGNMENT: The document is in Arabic. You MUST evaluate spatial relationships from Right to Left.
2. HORIZONTAL KEY-VALUE GRIDS: Some layout grids are just form fields side-by-side. If a label is on the right (e.g., "قطاع النشاط") and a text value is visually in the cell to its immediate left (e.g., "خدمات"), extract it as a simple key-value pair: {{"قطاع النشاط": "خدمات"}}. Do NOT skip the left-side value.
3. VERTICAL DATA TABLES: For multi-column tables with headers (e.g., "الممثل أو الممثلون الشرعيون"):
   - Identify the column headers strictly from Right to Left.
   - Map row values purely by their vertical physical alignment under each specific header.
   - PREVENT COLUMN SHIFTING: If a cell under a header is visually empty (e.g., the "الإسم واللقب" column has no text below it), you MUST output "". DO NOT pull text from the adjacent left column to fill an empty right column. 
   - Ensure stacked multi-line values stay in their proper column (e.g., "1971" and "سوق أهراس" both fall vertically under the Date and Place of Birth header. "الجزائر" falls vertically under the Address header).

GENERAL EXTRACTION GUIDELINES:
4. IGNORE BOILERPLATE: Do NOT extract legal warnings, instructions, or long paragraphs (e.g., "المعلومات التي يتعرض لها...", "طبقا لأحكام...") as form fields. Ignore them completely.
5. EXACT VALUES: Look closely at dotted lines for handwritten/faint entries. If a legitimate field exists but is blank, output "".
6. ARABIC SPELLING: Transcribe field names exactly as printed.

Output ONLY a valid JSON object."""

    payload = {
        "model": VLM_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}},
                ],
            }
        ],
        "temperature": 0.1,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Step 4: Recursive Validation ---
def autocorrect_date(date_str):
    corrected = re.sub(r'[Ll|\\_]', '/', date_str)
    if re.match(r"^\d{8}$", corrected):
        corrected = f"{corrected[:4]}/{corrected[4:6]}/{corrected[6:]}"
    return corrected

def autocorrect_name(name_str):
    return re.sub(r'\d+', '', name_str).strip()

def validate_and_clean_json(raw_json_str):
    validation_report = {"valid": True, "errors": []}

    try:
        clean_json = raw_json_str.strip()
        if clean_json.startswith("```json"): clean_json = clean_json[7:]
        if clean_json.startswith("```"): clean_json = clean_json[3:]
        if clean_json.endswith("```"): clean_json = clean_json[:-3]

        extracted_data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        validation_report["valid"] = False
        validation_report["errors"].append(f"VLM Output was not valid JSON: {e}")
        return {}, validation_report

    def process_node(key, value):
        if isinstance(value, str):
            if not value.strip():
                return value

            k_lower = key.lower()

            if any(word in k_lower for word in ["تاريخ", "date"]):
                has_letters = bool(re.search(r'[a-zA-Z\u0600-\u06FF]', value))
                if not has_letters:
                    value = autocorrect_date(value)
                    if not re.match(r"^\d{4}[/-]\d{2}[/-]\d{2}$", value):
                        validation_report["errors"].append(f"Unfixable date format for '{key}': {value}")
                        validation_report["valid"] = False

            elif any(word in k_lower for word in ["الاسم", "اللقب", "nom", "prenom", "اسم"]):
                original_v = value
                value = autocorrect_name(value)
                if not value and original_v.strip():
                    validation_report["errors"].append(f"Name field '{key}' became empty after stripping digits.")
                    validation_report["valid"] = False
            return value

        elif isinstance(value, dict):
            return {k: process_node(k, v) for k, v in value.items()}

        elif isinstance(value, list):
            return [process_node(key, item) if not isinstance(item, dict) else {k: process_node(k, v) for k, v in item.items()} for item in value]

        return value

    validated_data = process_node("", extracted_data) if isinstance(extracted_data, dict) else extracted_data
    return validated_data, validation_report

# --- Core Processing Pipeline ---
def process_documents():
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        start_time = time.time()
        print(f"\n--- Processing: {filename} ---")
        base_name = os.path.splitext(filename)[0]
        input_path = os.path.join(INPUT_DIR, filename)

        ocr_input_path = input_path
        if USE_PREPROCESSING:
            preprocessed_path = os.path.join(OUTPUT_DIR, f"{base_name}_preprocessed.jpg")
            try:
                preprocess_image(input_path, preprocessed_path)
                ocr_input_path = preprocessed_path
                print("Preprocessing complete (CLAHE Grayscale).")
            except Exception as e:
                print(f"Preprocessing failed: {e}")
                continue

        # 1. Plain Text Transcription
        try:
            raw_ocr_text = extract_raw_plain_text(ocr_input_path)

            txt_path = os.path.join(OUTPUT_DIR, f"{base_name}_ocr.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_ocr_text)
            print(f"Plain text OCR saved to {txt_path}")
        except Exception as e:
            print(f"Plain text OCR failed: {e}")
            continue

        # 2. JSON Field Extraction
        try:
            raw_json_str = extract_fields_to_json(ocr_input_path, raw_ocr_text)
        except Exception as e:
            print(f"JSON field extraction failed: {e}")
            continue

        # 3. Clean and Validate
        fields, validation = validate_and_clean_json(raw_json_str)
        elapsed_time = round(time.time() - start_time, 2)

        # 4. Save Structured JSON Output
        json_output = {
            "file": filename,
            "processing_time_seconds": elapsed_time,
            "validation_status": validation,
            "extracted_fields": fields,
        }

        json_path = os.path.join(OUTPUT_DIR, f"{base_name}_fields.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=4)

        print(f"Structured JSON saved to {json_path}")
        print(f"Pipeline finished in {elapsed_time}s")

if __name__ == "__main__":
    process_documents()