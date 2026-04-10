"""
from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException
import io
from PIL import Image
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import zipfile
from typing import List, Dict
import io, os
from difflib import SequenceMatcher
from fastapi import FastAPI, UploadFile, File
from typing import List
from PIL import Image
import torch
import io, os
from difflib import SequenceMatcher 
from io import BytesIO
from fastapi.responses import JSONResponse
from objectDetection import model_detect
from cropping import detect_text_east
from fastapi.responses import JSONResponse


router = APIRouter()

@router.post("/compare_counts")
async def compare_detection_and_cropping(file: UploadFile = File(...)):
    contents = await file.read()

    # Call both internal functions directly
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # OBB Detection
    results = model_detect.predict(source=image_bgr, conf=0.5, save=False, imgsz=640, verbose=False)
    obb_boxes = results[0].obb
    num_detect = 0
    if obb_boxes and obb_boxes.xyxyxyxy is not None:
        num_detect = len(obb_boxes.xyxyxyxy)

    # EAST Cropping
    crop_boxes = detect_text_east(image_bgr)
    num_crop = len(crop_boxes)

    return {
        "obb_detected_boxes": num_detect,
        "east_detected_boxes": num_crop,
        "match": num_detect == num_crop
    }
"""
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
import cv2

router = APIRouter()

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0

    iou = interArea / unionArea
    return iou

def match_boxes(detection_boxes, cropping_boxes, iou_threshold=0.5):
    matches = []
    for det_box in detection_boxes:
        best_iou = 0
        best_crop_box = None
        for crop_box in cropping_boxes:
            boxA = (det_box['startX'], det_box['startY'], det_box['endX'], det_box['endY'])
            boxB = (crop_box['startX'], crop_box['startY'], crop_box['endX'], crop_box['endY'])
            iou = compute_iou(boxA, boxB)
            if iou > best_iou:
                best_iou = iou
                best_crop_box = crop_box
        if best_iou >= iou_threshold:
            matches.append({
                "detection_box": det_box,
                "cropping_box": best_crop_box,
                "iou": best_iou
            })
    return matches

@router.post("/compare_boxes")
def compare_boxes(file: UploadFile = File(...)):
    contents =  file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Import inside the function to avoid circular imports
    from objectDetection import model_detect
    from cropping import detect_text_east, order_boxes_by_reading_order

    # Object Detection
    results = model_detect.predict(source=image_bgr, conf=0.5, save=False, imgsz=640, verbose=False)
    boxes = results[0].obb

    detection_boxes = []
    if boxes is not None and boxes.xyxyxyxy is not None and len(boxes.xyxyxyxy) > 0:
        obb_coords = boxes.xyxyxyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        names = model_detect.names

        for i in range(len(obb_coords)):
            polygon = obb_coords[i].reshape(4, 2)
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            startX = int(np.min(x_coords))
            startY = int(np.min(y_coords))
            endX = int(np.max(x_coords))
            endY = int(np.max(y_coords))
            class_name = names[int(classes[i])]
            detection_boxes.append({
                "startX": startX,
                "startY": startY,
                "endX": endX,
                "endY": endY,
                "class": class_name,
                "score": float(scores[i])
            })

    # Cropping
    crop_boxes = detect_text_east(image_bgr)
    ordered_boxes = order_boxes_by_reading_order(crop_boxes)
    cropping_boxes = [
        {"startX": x1, "startY": y1, "endX": x2, "endY": y2}
        for (x1, y1, x2, y2) in ordered_boxes
    ]

    # Match boxes
    matches = match_boxes(detection_boxes, cropping_boxes, iou_threshold=0.5)

    return{
        "matches": matches,
        "total_detections": len(detection_boxes),
        "total_croppings": len(cropping_boxes),
        "matched_pairs": len(matches)
    }
