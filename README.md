# OCR-registre-commerce

End-to-end pipeline that extracts structured data from scanned Algerian "registre de commerce" documents (Arabic/French, right-to-left tables).

## Pipeline overview

1. **Preprocessing** (`preprocess_image`)
   Reads each image with OpenCV, keeps the red channel as a grayscale proxy (better contrast on these scans than a standard grayscale conversion), upscales 1.5x with cubic interpolation, then applies CLAHE contrast enhancement. The result is saved to `output/<name>_preprocessed.jpg`.

2. **Raw OCR transcription** (`extract_raw_plain_text`)
   Sends the preprocessed image to a vision-language model (Qwen3-VL, via the OpenRouter API) with a prompt tuned for these documents: strict right-to-left reading order, careful reading of faint text on dotted lines, isolated table-cell words, and multi-line cells. Output is plain text, saved to `output/<name>_ocr.txt`.

3. **Structured field extraction** (`extract_fields_to_json`)
   Sends the same image plus the raw OCR text back to the VLM with a second prompt that enforces strict column alignment, forbids inventing JSON keys (must match the exact Arabic labels printed on the document), and handles stacked/multi-line table cells and dotted-line key-value pairs. Output is a JSON object.

4. **Validation & cleanup** (`validate_and_clean_json`)
   Parses the VLM's JSON (stripping markdown fences if present) and walks every field:
   - **Date fields** (keys containing `تاريخ`/`date`): normalizes separators and reformats 8-digit strings to `YYYY/MM/DD`; flags anything that still doesn't match.
   - **Name fields** (keys containing `الاسم`, `اللقب`, `nom`, `prenom`, `اسم`): strips stray digits; flags if the field becomes empty.
   - **Amount fields** (keys containing `مبلغ`, `رأس مال`, `رأسمال`, `montant`, `capital`): normalizes Arabic thousand/decimal separators and spacing, then validates against the expected format — a number with optional thousands separators and decimals, with an optional currency marker (`دج`, `د.ج`, `DZD`, `DA`, `dz`) either before or after the number (e.g. `1,500,000.00`, `5,000,000.00 دج`, `دج 1,500,000.00`). Flags anything that doesn't match.

   Each field is checked recursively through nested objects and lists (e.g. multiple legal representatives, multiple activity codes). Errors are collected into a validation report rather than raising, so one bad field doesn't stop the run.

5. **Pipeline orchestration** (`process_documents`)
   Loops over every image in `input/`, runs steps 1-4, times the run, and writes `output/<name>_fields.json` containing the source filename, processing time, validation report, and extracted fields.

## Requirements

- Python packages listed in `requirements.txt`
- An `OPENROUTER_API_KEY` environment variable set to a valid OpenRouter API key

## Usage

```bash
export OPENROUTER_API_KEY=your_key_here
python main.py
```

Place input images in `input/`; results (preprocessed images, raw OCR text, and structured JSON) are written to `output/`.

## Possible next steps

- Add more validation rules (e.g. mandatory-field checks, nationality/place-of-birth consistency)
- Compare others VLM OCR 