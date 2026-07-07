import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_sauvola

def advanced_ocr_pipeline(image_path, output_dir="output"):
    print(f"[START] Initializing End-to-End Pipeline for: {image_path}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    vis_steps = {}

    # -------------------------------------------------------------------------
    # STEP 1: LOAD IMAGE
    # -------------------------------------------------------------------------
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Critical: Could not open or read target file: {image_path}")
        return
    print(f"[STEP 1] Image loaded. Input Array Shape: {img.shape}")
    vis_steps['01_Original'] = img.copy()

    # -------------------------------------------------------------------------
    # STEP 2: UPSCALING
    # -------------------------------------------------------------------------
    target_width = 2200
    h, w, _ = img.shape
    if w < target_width:
        scale_factor = target_width / w
        img_scaled = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
        print(f"[STEP 2] Upscaled image by {scale_factor:.2f}x.")
    else:
        img_scaled = img.copy()
        print("[STEP 2] Rescaling skipped; width meets standard thresholds.")
    vis_steps['02_Upscaled'] = img_scaled

    # -------------------------------------------------------------------------
    # STEP 3: GRAYSCALE & CHROMINANCE SEPARATION
    # -------------------------------------------------------------------------
    _, gray, _ = cv2.split(img_scaled)
    print("[STEP 3] Extracted isolated green channel matrix.")
    vis_steps['03_Green_Channel_Gray'] = gray

    # -------------------------------------------------------------------------
    # STEP 4: DESKEWING
    # -------------------------------------------------------------------------
    thresh_skew = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh_skew > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    elif angle > 45: angle = 90 - angle
    
    if abs(angle) > 0.1:
        h_s, w_s = gray.shape
        M = cv2.getRotationMatrix2D((w_s // 2, h_s // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w_s, h_s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        print(f"[STEP 4] Rotated document by {angle:.2f}°.")
    else:
        print("[STEP 4] Alignment verified.")
    vis_steps['04_Deskewed_Gray'] = gray

    # -------------------------------------------------------------------------
    # STEP 5: ADAPTIVE CONTRAST ENHANCEMENT (CLAHE)
    # -------------------------------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    print("[STEP 5] Executed local adaptive histogram equalization (CLAHE).")
    vis_steps['05_Enhanced_Contrast'] = contrast_img

    # -------------------------------------------------------------------------
    # STEP 6: BILATERAL EDGE-PRESERVING DENOISING
    # -------------------------------------------------------------------------
    denoised_img = cv2.bilateralFilter(contrast_img, d=9, sigmaColor=75, sigmaSpace=75)
    print("[STEP 6] Suppressed sensory noise using Bilateral filtering.")
    vis_steps['06_Denoised_Pixels'] = denoised_img

    # -------------------------------------------------------------------------
    # STEP 7: SAUVOLA BINARIZATION
    # -------------------------------------------------------------------------
    # Sauvola binarization works exceptionally well for document backgrounds
    window_size = 25
    thresh_sauvola = threshold_sauvola(denoised_img, window_size=window_size)
    
    # Create the binary image (Pixels above threshold are white 255, below are black 0)
    binary_img = (denoised_img > thresh_sauvola).astype(np.uint8) * 255
    print("[STEP 7] Applied Sauvola Binarization for optimal document thresholding.")
    vis_steps['07_Sauvola_Binary'] = binary_img

    # -------------------------------------------------------------------------
    # STEP 8: POST-BINARIZATION DENOISING & TEXT HEALING (CLOSING)
    # -------------------------------------------------------------------------
    # In OpenCV, morphological operations act on WHITE pixels.
    # Since our text is currently BLACK on a WHITE background, we must invert it first.
    inverted_for_morph = cv2.bitwise_not(binary_img)
    
    # Define a small kernel to bridge fractures in the Arabic cursive strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    # Apply MORPH_CLOSE (Dilate then Erode) to fill holes and bridge gaps in the white text
    closed_inverted = cv2.morphologyEx(inverted_for_morph, cv2.MORPH_CLOSE, kernel_close)
    
    # Invert back so text is BLACK again
    final_binary = cv2.bitwise_not(closed_inverted)
    
    print("[STEP 8] Executed Morphological Closing to heal fractured text strokes.")
    vis_steps['08_Healed_Binary'] = final_binary

    # -------------------------------------------------------------------------
    # STEP 9: GEOMETRIC HORIZONTAL WORD SEGMENTATION
    # -------------------------------------------------------------------------
    inverted_final = cv2.bitwise_not(final_binary)
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))
    word_mask = cv2.dilate(inverted_final, word_kernel, iterations=1)
    
    opencv_segmentation_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    geo_word_count = 0
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if w_box > 8 and h_box > 8:
            cv2.rectangle(opencv_segmentation_vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            geo_word_count += 1
            
    print(f"[STEP 9] OpenCV Isolated {geo_word_count} word structural blobs.")
    vis_steps['09_OpenCV_Word_Boxes'] = opencv_segmentation_vis

    # -------------------------------------------------------------------------
    # STEP 10: OCR EXECUTION (Tesseract)
    # -------------------------------------------------------------------------
    # CHANGED TO PSM 11: "Sparse text. Find as much text as possible in no particular order."
    # This prevents Tesseract from ignoring form fields trapped inside table borders.
    print("[STEP 10] Querying Tesseract OCR engine (ara+fra) with PSM 11...")
    custom_config = r'--psm 11'
    ocr_data = pytesseract.image_to_data(final_binary, lang='ara+fra', config=custom_config, output_type=Output.DICT)
    
    tesseract_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    n_boxes = len(ocr_data['text'])
    extracted_words_list = []
    
    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 10: # Lowered confidence threshold slightly for scattered fields
            word_text = ocr_data['text'][i].strip()
            if word_text:
                x, y, w_w, h_w = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                cv2.rectangle(tesseract_vis, (x, y), (x + w_w, y + h_w), (255, 0, 0), 2)
                extracted_words_list.append((i, word_text, ocr_data['conf'][i]))
                
    print(f"[STEP 10] Tesseract isolated {len(extracted_words_list)} target terms.")
    vis_steps['10_Tesseract_Words'] = tesseract_vis

    # -------------------------------------------------------------------------
    # STEP 11: TEXT POST-PROCESSING & SAVING
    # -------------------------------------------------------------------------
    cleaned_document_tokens = []
    for token_id, raw_text, confidence in extracted_words_list:
        clean_text = "".join([char for char in raw_text if char.isalnum() or char in ".:-/,_ "]).strip()
        if len(clean_text) > 0:
            cleaned_document_tokens.append(clean_text)
            
    reconstructed_text_payload = " ".join(cleaned_document_tokens)
    
    # Save the text to the output folder
    text_output_path = os.path.join(output_dir, f"{base_filename}_extracted_text.txt")
    with open(text_output_path, "w", encoding="utf-8") as text_file:
        text_file.write(reconstructed_text_payload)
        
    print(f"\n[EXPORTING] Saved full extracted text to: {text_output_path}")

    # -------------------------------------------------------------------------
    # IMAGE EXPORT PROTOCOL
    # -------------------------------------------------------------------------
    print("[EXPORTING] Saving high-resolution step visualizations to the output directory...")
    for step_name, img_matrix in vis_steps.items():
        file_name = f"{base_filename}_{step_name}.jpg"
        file_path = os.path.join(output_dir, file_name)
        success = cv2.imwrite(file_path, img_matrix)
        if success:
            print(f"  -> Saved: {file_path}")
        else:
            print(f"  -> [ERROR] Failed to save: {file_path}")

    print("\n[PIPELINE COMPLETE] All steps successfully processed and exported.")

if __name__ == "__main__":
    input_file = os.path.join("input", "D:/stage mobilis/OCR-registre-commerce/input/REGISTRE-FRONT.png")
    
    if os.path.exists(input_file):
        advanced_ocr_pipeline(input_file)
    else:
        print(f"Please check your path. Could not locate: {input_file}")