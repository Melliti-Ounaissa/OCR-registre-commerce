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

# --- Step 1: Preprocessing ---
def preprocess_image(image_path, output_path):
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Could not read image: {image_path}")

    b, g, r = cv2.split(img_color)
    img_gray = r
    img_upscaled = cv2.resize(img_gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_upscaled)

    cv2.imwrite(output_path, img_enhanced)
    return output_path

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_openrouter_with_retry(payload, headers, max_retries=4, base_delay=3):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            print(f"  Attempt {attempt}/{max_retries} failed ({last_error}).")
        else:
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            try:
                error_body = response.json()
            except ValueError:
                error_body = response.text
            last_error = f"HTTP {response.status_code}: {error_body}"
            if response.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"API Error Details:\n{last_error}")
            print(f"  Attempt {attempt}/{max_retries} failed ({last_error}).")
        if attempt < max_retries:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError(f"API Error Details (after {max_retries} attempts):\n{last_error}")

# --- Step 2: Pure Plain Text OCR Transcription ---
def extract_raw_plain_text(image_path):
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    b64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    prompt = (
        "You are a highly precise OCR engine for Arabic administrative documents. "
        "Extract all visible text exactly as printed. "
        "CRITICAL RULES:\n"
        "1. Read top-to-bottom, strictly Right-to-Left.\n"
        "2. DOTTED LINES: Pay extremely close attention to faint text written on dotted lines (e.g., 'خدمات'). "
        "Ensure this text is transcribed.\n"
        "3. ISOLATED WORDS: Do not skip solitary words floating in table cells or headers.\n"
        "4. ALIGNMENT: If two distinct words are far apart on the same horizontal line (e.g., a label on the right and a value on the left), "
        "transcribe both, separated by a space, keeping them on the same line.\n"
        "5. MULTI-LINE CELLS: If a single table cell contains stacked text (e.g., a year above a city), transcribe both on consecutive lines.\n"
        "Do not format as JSON. Output only plain text."
    )

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}}]}],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    return call_openrouter_with_retry(payload, headers)

# --- Step 3: Structuring Plain Text into JSON Fields ---
def extract_fields_to_json(image_path, raw_ocr_text):
    b64_image = encode_image_to_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    prompt = f"""You are an advanced document structuring AI. Analyze the image and transcription to output a strict JSON object.

RAW TRANSCRIPTION:
{raw_ocr_text}

CRITICAL RULES FOR ARABIC DOCUMENTS (MUST FOLLOW STRICTLY):

1. STRICT COLUMN ALIGNMENT (PREVENT SHIFTING): 
   - Table columns MUST be read from RIGHT to LEFT.
   - You must project an imaginary vertical box down from every column header. Only text strictly within that box belongs to that header.
   - If a cell under a header is physically empty, its JSON value MUST be an empty string `""`.
   - NEVER shift data from a neighboring column to fill an empty cell. (e.g., Do not move a location into an empty 'Name' column).

2. MULTIPLE LINES IN ONE CELL:
   - If a single cell contains vertically stacked text (e.g., '1971' on top and 'سوق أهراس' below it), join them into ONE string (e.g., '1971 سوق أهراس') for that specific column. Do not let the bottom text spill into an adjacent empty column.

3. NO HALLUCINATION OF KEYS:
   - Use ONLY the exact words printed in the document as your JSON keys.
   - Do NOT invent keys based on context. If the document prints "الصفة" (Capacity), use "الصفة". Do NOT invent "المهنة" (Profession) just because the value is "مسير".

4. KEY-VALUE EXTRACTION ON DOTTED LINES:
   - Look carefully at dotted lines extending to the left of labels (e.g., 'قطاع النشاط' or 'النشاط أو الأنشطة الممارسة'). The text written on these lines (even if far to the left, like 'خدمات') is the value for that label.
   - If no text is written on the line for a label, return `""` for that label. Do not steal values from the line below it.

Output ONLY a valid JSON object. No markdown, no explanations."""

    payload = {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}}]}],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    return call_openrouter_with_retry(payload, headers)

# --- Step 4: Recursive Validation ---
def autocorrect_date(date_str):
    corrected = re.sub(r'[Ll|\\_]', '/', date_str)
    if re.match(r"^\d{8}$", corrected):
        corrected = f"{corrected[:4]}/{corrected[4:6]}/{corrected[6:]}"
    return corrected

def autocorrect_name(name_str):
    return re.sub(r'\d+', '', name_str).strip()

# Accepts amounts like: "50,000,000.00", "5,000,000.00 دج", "1,500,000.00 د.ج",
# "دج 1,500,000.00", "1500000 DZD", "1,500,000.00 dz" (currency optional, before or after)
AMOUNT_CURRENCY = r'(?:دج|د\.ج|DZD|DA|dz)'
AMOUNT_NUMBER = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
AMOUNT_PATTERN = rf'^{AMOUNT_CURRENCY}?\s*(?:{AMOUNT_NUMBER})\s*{AMOUNT_CURRENCY}?$'

def autocorrect_amount(amount_str):
    # Normalize common OCR artifacts: Arabic thousand/decimal separators, stray spaces around punctuation
    corrected = amount_str.strip()
    corrected = corrected.replace('٬', ',').replace('،', ',')
    corrected = re.sub(r'\s*\.\s*', '.', corrected)
    corrected = re.sub(r'\s*,\s*', ',', corrected)
    corrected = re.sub(r'\s+', ' ', corrected)
    return corrected

def is_valid_amount(value):
    return bool(re.match(AMOUNT_PATTERN, value.strip(), re.IGNORECASE))

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
            elif any(word in k_lower for word in ["مبلغ", "رأس مال", "رأسمال", "montant", "capital"]):
                value = autocorrect_amount(value)
                if not is_valid_amount(value):
                    validation_report["errors"].append(f"Invalid amount format for '{key}': {value}")
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

        try:
            raw_ocr_text = extract_raw_plain_text(ocr_input_path)
            txt_path = os.path.join(OUTPUT_DIR, f"{base_name}_ocr.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_ocr_text)
            print(f"Plain text OCR saved to {txt_path}")
        except Exception as e:
            print(f"Plain text OCR failed: {e}")
            continue

        try:
            raw_json_str = extract_fields_to_json(ocr_input_path, raw_ocr_text)
        except Exception as e:
            print(f"JSON field extraction failed: {e}")
            continue

        fields, validation = validate_and_clean_json(raw_json_str)
        elapsed_time = round(time.time() - start_time, 2)

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