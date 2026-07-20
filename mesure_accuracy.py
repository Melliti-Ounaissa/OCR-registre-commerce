import difflib

def calculate_ocr_metrics(ground_truth_text, ocr_extracted_text):
    """
    Computes baseline sequence similarity matching ratios 
    to visually cross-reference processing performance.
    """
    matcher = difflib.SequenceMatcher(None, ground_truth_text, ocr_extracted_text)
    accuracy_ratio = matcher.ratio()
    
    print(f"Estimated Sequence Accuracy Match: {accuracy_ratio * 100:.2f}%")
    
    # Generate a command-line visualization of differences
    diff = difflib.ndiff(ground_truth_text.splitlines(), ocr_extracted_text.splitlines())
    print("\n--- Visual Text Structural Diff Analysis ---")
    print("\n".join(diff))

# Example evaluation call:
# calculate_ocr_metrics("الاسم : سمية", "الاسم  سمية")