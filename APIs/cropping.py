"""from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException
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

router = APIRouter()
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


@router.post("/crop_ordered")
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

router = APIRouter()

#------------------EAST----------------
# Load the pre-trained EAST model
east_model_path = './frozen_east_text_detection.pb'
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

@router.post("/crop_coords")
async def crop_coords(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = detect_text_east(image_bgr)

    if not boxes:
        return {"boxes": []}

    ordered_boxes = order_boxes_by_reading_order(boxes)
    coords = [
        {"startX": x1, "startY": y1, "endX": x2, "endY": y2}
        for (x1, y1, x2, y2) in ordered_boxes
    ]

    return {"boxes": coords}
