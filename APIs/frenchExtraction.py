from fastapi import APIRouter, UploadFile, File
from typing import List
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import io

router = APIRouter()

# Load TrOCR processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_extraction = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
model_extraction.eval()

@router.post("/extract-french-text")
async def extract_french_text(files: List[UploadFile] = File(...)):
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
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Preprocess and predict
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model_extraction.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            results.append({
                "filename": filename,
                "predicted": generated_text
            })

        except Exception as e:
            results.append({
                "filename": filename,
                "predicted": f"Error: {str(e)}"
            })

    return results
