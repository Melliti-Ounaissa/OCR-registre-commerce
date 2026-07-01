# import cv2
# import numpy as np
import PIL.Image



# load image from path
im_file = "D:/stage Mobilis/OCR-registre-commerce/input/REGISTRE-FRONT.png"
im = PIL.Image.open(im_file)



# resizing
# def upscale(img, scale: float = 2.0):
#     h, w = img.shape[:2]
#     new_size = (int(w * scale), int(h * scale))
#     return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)

# # convert to grayscale
# def to_grayscale(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # denoising
# def denoise(gray):
#     return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

# # contrast enhancement
# def enhance_contrast(gray):
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     return clahe.apply(gray)

# # deskewing
# def deskew(gray):
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     if len(coords) == 0:
#         return gray

#     angle = cv2.minAreaRect(coords)[-1]
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     if abs(angle) < 0.5:      # already straight enough, skip
#         return gray

#     (h, w) = gray.shape[:2]
#     M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#     return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# # binarization
# def binarize(gray):
#     return cv2.adaptiveThreshold(
#         gray, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         blockSize=25,
#         C=15
#     )
    
    
    
# # preprocesing pipeline
# def preprocess_pipeline(path: str, scale: float = 2.0) -> dict:
#     original  = load_image(path)
#     upscaled  = upscale(original, scale=scale)
#     gray      = to_grayscale(upscaled)
#     denoised  = denoise(gray)
#     contrasted = enhance_contrast(denoised)
#     deskewed  = deskew(contrasted)
#     binarized = binarize(deskewed)

#     return {
#         "original": original, "upscaled": upscaled, "gray": gray,
#         "denoised": denoised, "contrasted": contrasted,
#         "deskewed": deskewed, "binarized": binarized,
#     }



im.save("D:/stage Mobilis/OCR-registre-commerce/output/REGISTRE-FRONT.png", "PNG")