from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
from ArabicOcr import arabicocr

router = APIRouter()

@router.post("/extract-arabic-text")
async def extract_arabic_text(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        filename = file.filename
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            results.append({
                "filename": filename,
                "predicted": "Unsupported file type"
            })
            continue

        try:
            image_bytes = await file.read()
            temp_path = f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(image_bytes)

            output_path = f"out_{filename}"

            # Perform OCR
            ocr_results = arabicocr.arabic_ocr(temp_path, output_path)
            words_conf = [(item[1].strip(), round(float(item[2]), 2)) for item in ocr_results if item[1].strip()]
            predicted = words_conf[0][0] if words_conf else ""

            results.append({
                "filename": filename,
                "predicted": predicted
            })

            # Clean up
            os.remove(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            results.append({
                "filename": filename,
                "predicted": f"Error: {str(e)}"
            })

    return results



