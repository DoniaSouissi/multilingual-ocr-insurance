import streamlit as st
import requests
from PIL import Image
import io

# FastAPI endpoint URL (change to your actual URL if hosted elsewhere)
API_URL = "http://localhost:8000/run"  # Adjust if deployed remotely

st.set_page_config(page_title="Text Extraction Pipeline", layout="wide")
st.title("End-to-End Text Extraction")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded Image", use_column_width=True)
    st.image(image, use_container_width=True)

    if st.button("Run Pipeline"):
        with st.spinner("Processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Processing complete!")

            # Show evaluation metrics
            eval_metrics = result.get("evaluation", {})
            st.subheader("📊 Evaluation Metrics")
            st.metric("Precision", eval_metrics.get("precision"))
            st.metric("Recall", eval_metrics.get("recall"))
            st.metric("F1 Score", eval_metrics.get("f1_score"))

            # Display extracted segments
            st.subheader("📝 Extracted Text Segments")
            for seg in result.get("segments", []):
                st.markdown(f"""
                **Segment ID:** {seg["segment_id"]}  
                **Text Type:** {seg["text_type"]}  
                **Language:** {seg["language"]}  
                **Confidence:** {seg["confidence"]}  
                **Extracted Text:** `{seg["extracted_text"]}`  
                **Corrected Text:** `{seg["corrected_text"]}`
                ---
                """)
        else:
            st.error(f"❌ Error: {response.status_code} - {response.text}")














"""from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import supervision as sv
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import uuid
import os 
from fastapi.responses import FileResponse
import tempfile
import zipfile
from typing import List, Dict
from paddleocr import PaddleOCR
from bidi.algorithm import get_display
import arabic_reshaper
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import io, os
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File
from typing import List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io, os
from difflib import SequenceMatcher
from ArabicOcr import arabicocr  
from io import BytesIO


app = FastAPI()
#----------------SEGMENTATION----------------
model = YOLO("/home/donia/deployment1/APIs/best.pt")

segmented_storage = {}  # Store images temporarily in memory
@app.post("/segment")
async def segment_image_upload(file: UploadFile = File(...)):
    contents = await file.read()
    input_img = Image.open(io.BytesIO(contents)).convert("RGB")

    image = np.array(input_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(source=image, conf=0.5, save=False, imgsz=640, verbose=False)
    names = model.names

    masks = results[0].masks
    if masks is None:
        return JSONResponse(content={"error": "No segmentation detected."})

    mask_array = masks.data.cpu().numpy()
    H, W, _ = image.shape
    ids = []

    for i, mask in enumerate(mask_array):
        class_id = int(results[0].boxes.cls[i].item())
        class_name = names[class_id]

        resized_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = resized_mask > 0.5

        segmented = np.zeros_like(image)
        segmented[mask_bool] = image[mask_bool]

        segment_id = str(uuid.uuid4())
        segmented_storage[segment_id] = segmented
        ids.append({"id": segment_id, "class": class_name})

    return {"segments": ids}


@app.get("/segment/{segment_id}")
def get_segment_image(segment_id: str):
    if segment_id not in segmented_storage:
        return JSONResponse(status_code=404, content={"error": "Segment not found"})

    segmented = segmented_storage[segment_id]
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(segmented_rgb)

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")


#---------------- OBJECT DETECTION----------------

model_detection = YOLO("/home/donia/deployment1/APIs/best_detection.pt")

@app.post("/detect", response_class=StreamingResponse)
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO OBB detection
    results = model_detection.predict(source=image_bgr, conf=0.5, save=False, imgsz=640, verbose=False)

    boxes = results[0].obb

    if boxes is None or boxes.xyxyxyxy is None or len(boxes.xyxyxyxy) == 0:
        print("No OBB detections.")
        return StreamingResponse(io.BytesIO(contents), media_type="image/png")

    obb_coords = boxes.xyxyxyxy.cpu().numpy()       # shape: (N, 8)
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    names = model_detection.names

    for i in range(len(obb_coords)):
        polygon = obb_coords[i].astype(np.int32).reshape((-1, 1, 2))
        class_id = int(classes[i])
        conf = float(scores[i])
        class_name = names[class_id]
        label = f"{class_name} ({conf:.2f})"

        # Set color and thickness
        if class_name.lower() == "typo":
            color = (0, 0, 255)     # Red
            thickness = 2
            font_thickness = 2
        elif class_name.lower() == "handwritten":
            color = (255, 0, 0)     # Blue
            thickness = 2
            font_thickness = 2
        else:
            color = (0, 255, 0)     # Green (fallback)
            thickness = 1
            font_thickness = 1

        # Draw OBB
        cv2.polylines(image_bgr, [polygon], isClosed=True, color=color, thickness=thickness)

        # Draw label (bold by increasing thickness)
        x, y = polygon[0][0]
        cv2.putText(image_bgr, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, font_thickness, cv2.LINE_AA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output_pil = Image.fromarray(image_rgb)

    img_byte_arr = io.BytesIO()
    output_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")


#------------------EAST----------------
# Load the pre-trained EAST model
east_model_path = '/home/donia/deployment1/APIs/frozen_east_text_detection.pb'
net = cv2.dnn.readNet(east_model_path)

def detect_text_east(image, conf_threshold=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Set the new width and height to multiples of 32
    newW, newH = (W // 32) * 32, (H // 32) * 32
    rW, rH = W / float(newW), H / float(newH)

    # Resize the image and grab the new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # Define the two output layer names for the EAST detector model
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # Construct a blob from the image and then perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Decode the predictions
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < conf_threshold:
                continue

            # Compute the offset factor
            offsetX = x * 4.0
            offsetY = y * 4.0

            # Extract the rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Compute the height and width of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute the starting and ending coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = cv2.dnn.NMSBoxesRotated(
        [cv2.RotatedRect(((startX + endX) / 2, (startY + endY) / 2), (endX - startX, endY - startY), 0)
         for (startX, startY, endX, endY) in rects],
        confidences,
        conf_threshold,
        0.4
    )

    final_boxes = []
    if len(boxes) > 0:
        for i in boxes.flatten():
            startX, startY, endX, endY = rects[i]
            # Scale the coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            final_boxes.append((startX, startY, endX, endY))

    return final_boxes


def order_boxes_by_reading_order(boxes, y_threshold=10):
    
    #Orders boxes in natural reading order: top-to-bottom, left-to-right.
    #Expects boxes in format: (startX, startY, endX, endY)
    
    # Calculate the center point of each box
    box_centers = []
    for box in boxes:
        startX, startY, endX, endY = box
        centerY = (startY + endY) / 2
        centerX = (startX + endX) / 2
        box_centers.append((centerY, centerX, box))

    # Sort boxes by centerY (top to bottom)
    box_centers.sort(key=lambda x: x[0])

    # Group boxes into lines
    lines = []
    current_line = []
    prev_centerY = None
    for centerY, centerX, box in box_centers:
        if prev_centerY is None:
            current_line.append((centerX, box))
            prev_centerY = centerY
        else:
            if abs(centerY - prev_centerY) < y_threshold:
                current_line.append((centerX, box))
                prev_centerY = (prev_centerY + centerY) / 2  # Update average centerY
            else:
                # Sort current line by centerX (left to right)
                current_line.sort(key=lambda x: x[0])
                lines.append([b for _, b in current_line])
                current_line = [(centerX, box)]
                prev_centerY = centerY
    if current_line:
        current_line.sort(key=lambda x: x[0])
        lines.append([b for _, b in current_line])

    # Flatten the list of lines into a single list of boxes
    ordered_boxes = [box for line in lines for box in line]
    return ordered_boxes


@app.post("/crop_ordered")
async def crop_and_order(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect text regions
    boxes = detect_text_east(image_bgr)

    if not boxes:
        return {"message": "No text regions detected."}

    # Order boxes in reading order
    ordered_boxes = order_boxes_by_reading_order(boxes)

    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, (startX, startY, endX, endY) in enumerate(ordered_boxes):
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(image_bgr.shape[1], endX)
            endY = min(image_bgr.shape[0], endY)

            cropped = image_bgr[startY:endY, startX:endX]
            _, buffer = cv2.imencode('.png', cropped)
            zipf.writestr(f"region_{idx+1}.png", buffer.tobytes())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": "attachment; filename=cropped_regions.zip"
    })

#-------------Language Detection----------------
# Initialize OCR models for French and Arabic
ocr_fr = PaddleOCR(use_angle_cls=True, lang='fr')
ocr_ar = PaddleOCR(use_angle_cls=True, lang='ar')

def is_arabic(text: str) -> bool:
    #Check if the text contains Arabic characters.
    for ch in text:
        if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F':
            return True
    return False

def format_arabic_rtl(text: str) -> str:
    #Format Arabic text for proper display.
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

@app.post("/detect-language")
async def detect_language(file: UploadFile = File(...)):
    
    #Detect the language of text in the uploaded image.
    
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

#--------------------French Extraction----------------

# Load the processor and model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model_extraction = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Set the model to evaluation mode
model_extraction.eval()

@app.post("/extract-french-text/")
async def extract_french_text(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Generate text
        with torch.no_grad():
            generated_ids = model_extraction.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return JSONResponse(content={"extracted_text": generated_text.strip()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




#--------------------Arabic Extraction----------------
def char_accuracy(pred: str, target: str) -> float:
    return SequenceMatcher(None, pred, target).ratio()

def char_accuracy_arabic(pred: str, target: str) -> float:
    min_len = min(len(pred), len(target))
    correct = sum(p == t for p, t in zip(pred, target))
    return correct / max(len(target), 1)
@app.post("/extract-arabic/")
async def extract_arabic(files: List[UploadFile] = File(...)):
    results = []
    total = 0
    exact_matches = 0
    char_acc_sum = 0.0

    for file in files:
        filename = file.filename
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            image_bytes = await file.read()
            temp_path = f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(image_bytes)

            output_path = f"out_{filename}"

            ocr_results = arabicocr.arabic_ocr(temp_path, output_path)
            words_conf = [(item[1].strip(), round(float(item[2]), 2)) for item in ocr_results if item[1].strip()]
            predicted = words_conf[0][0] if words_conf else ""
            ground_truth = os.path.splitext(filename)[0].strip()

            acc = char_accuracy_arabic(predicted, ground_truth)
            char_acc_sum += acc
            total += 1
            if predicted == ground_truth:
                exact_matches += 1

            results.append({
                "filename": filename,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "char_accuracy": round(acc, 2)
            })

            # Clean temp file
            os.remove(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            results.append({
                "filename": filename,
                "error": f"Failed to process image: {str(e)}"
            })

    summary = {
        "exact_match_accuracy": round((exact_matches / total) * 100, 2) if total else 0,
        "average_char_accuracy": round((char_acc_sum / total) * 100, 2) if total else 0,
        "total_samples": total
    }

    return {"results": results, "summary": summary}"""




