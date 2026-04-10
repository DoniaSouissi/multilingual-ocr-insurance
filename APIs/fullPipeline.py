from fastapi import FastAPI, UploadFile, File, APIRouter
from paddleocr import PaddleOCR
from langdetect import detect
from PIL import Image
import io
from pipeline_french import detect_typo_french
from pipeline_arabic import detect_typo_arabic
import numpy as np 
from pipeline_arabic import detect_typo_arabic_from_pil
from pipeline_french import detect_typo_french_from_pil



router = APIRouter()



# Lightweight PaddleOCR instances
ocr_ar = PaddleOCR(lang='arabic', use_angle_cls=False, use_gpu=False)
ocr_fr = PaddleOCR(lang='latin', use_angle_cls=False, use_gpu=False)

def get_sample_text(paddle_ocr, image):
    np_image = np.array(image)  # Convert PIL.Image to numpy.ndarray
    result = paddle_ocr.ocr(np_image, cls=False)
    if not result or not result[0]:
        return ""
    return " ".join([line[1][0] for line in result[0]])




@router.post("/pipeline_ocr")
async def pipeline_ocr(file: UploadFile = File(...)):
    # Load the image
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 1: Run both PaddleOCR (quick scan only)
    ar_text = get_sample_text(ocr_ar, image_pil)
    fr_text = get_sample_text(ocr_fr, image_pil)

    # Decide language based on longer OCR output
    lang = "unknown"
    if len(ar_text.strip()) > len(fr_text.strip()):
        lang = "ar"
        final_text = detect_typo_arabic_from_pil(image_pil)
    elif len(fr_text.strip()) > 0:
        lang = "fr"
        final_text = detect_typo_french_from_pil(image_pil)
    else:
        return {"error": "No text detected for language identification."}

    return {
        "detected_language": lang,
        "extracted_text": final_text
    }