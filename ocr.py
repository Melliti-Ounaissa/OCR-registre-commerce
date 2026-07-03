import pytesseract
# IMPORTANT: Import the pipeline function from your other file
from image_preprocessing import preprocess_pipeline

# Point to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(preprocessed_image):
    custom_config = r'--oem 3 --psm 6'
    print("Starting OCR extraction (this might take a few seconds)...")
    
    text = pytesseract.image_to_string(
        preprocessed_image, 
        lang='ara+fra', 
        config=custom_config
    )
    return text

# Main execution for the OCR step
if __name__ == "__main__":
    # 1. Define the image you want to process
    input_path = "D:/stage Mobilis/OCR-registre-commerce/input/REGISTRE-FRONT.png"
    
    # 2. Run the preprocessing pipeline imported from your other file
    print("Running preprocessing pipeline...")
    results = preprocess_pipeline(input_path, scale=2.0)
    
    # 3. Extract the final binarized image from the dictionary
    final_image = results["6_binarized"]
    
    # 4. Pass that image to Tesseract
    extracted_text = extract_text_from_image(final_image)

    print("\n--- EXTRACTED TEXT ---")
    print(extracted_text)