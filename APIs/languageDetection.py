from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from typing import List, Dict
from paddleocr import PaddleOCR
from bidi.algorithm import get_display
import arabic_reshaper
from typing import List
from PIL import Image
import io
from ArabicOcr import arabicocr  
from io import BytesIO
from frenchExtraction import extract_french_text
from arabicExtraction import extract_arabic_text
from langdetect import detect

router = APIRouter()
#-------------Language Detection----------------
# Initialize OCR models for French and Arabic
ocr_fr = PaddleOCR(use_angle_cls=True, lang='fr')
ocr_ar = PaddleOCR(use_angle_cls=True, lang='ar')

def is_arabic(text: str) -> bool:
    """Check if the text contains Arabic characters."""
    for ch in text:
        if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F':
            return True
    return False

def format_arabic_rtl(text: str) -> str:
    """Format Arabic text for proper display."""
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

async def detect_language_from_np(image_np: np.ndarray):
    
    #Detect language from a numpy image array.
    #Returns the detected language(s) with confidence scores.
    
    # image_np is a numpy array; perform OCR
    result_fr = ocr_fr.ocr(image_np, cls=True)
    result_ar = ocr_ar.ocr(image_np, cls=True)

    combined_results = result_fr[0] + result_ar[0]
    seen_texts = set()
    texts = []

    for item in combined_results:
        text = item[1][0]
        confidence = item[1][1]
        if text.strip() and text not in seen_texts:
            seen_texts.add(text)
            if is_arabic(text):
                formatted_text = format_arabic_rtl(text)
                texts.append({
                    "text": formatted_text,
                    "lang": "ar",
                    "confidence": confidence
                })
            else:
                texts.append({
                    "text": text,
                    "lang": "fr",
                    "confidence": confidence
                })

    return texts

@router.post("/detect-language")
async def detect_language(file: UploadFile = File(...)):
    """
    Detect the language of text in the uploaded image.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Perform OCR for both French and Arabic
    result_fr = ocr_fr.ocr(image_np, cls=True)
    result_ar = ocr_ar.ocr(image_np, cls=True)

    combined_results = result_fr[0] + result_ar[0]
    seen_texts = set()
    texts: List[Dict[str, str]] = []

    for item in combined_results:
        text = item[1][0]
        confidence = item[1][1]
        if text.strip() and text not in seen_texts:
            seen_texts.add(text)
            if is_arabic(text):
                formatted_text = format_arabic_rtl(text)
                texts.append({
                    "text": formatted_text,
                    "lang": "ar",
                    "confidence": confidence
                })
            else:
                texts.append({
                    "text": text,
                    "lang": "fr",
                    "confidence": confidence
                })

    return JSONResponse(content={"texts": texts})
