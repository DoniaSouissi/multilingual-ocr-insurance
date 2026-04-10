from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

router = APIRouter()

model_detect = YOLO("./best_detection.pt")

@router.post("/detect_coords")
async def detect_object_coords(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model_detect.predict(source=image_bgr, conf=0.5, save=False, imgsz=640, verbose=False)
    boxes = results[0].obb

    if boxes is None or boxes.xyxyxyxy is None or len(boxes.xyxyxyxy) == 0:
        return {"boxes": []}

    obb_coords = boxes.xyxyxyxy.cpu().numpy()  # (N, 8)
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    names = model_detect.names

    output = []
    for i in range(len(obb_coords)):
        polygon = obb_coords[i].reshape(4, 2)
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        startX = int(np.min(x_coords))
        startY = int(np.min(y_coords))
        endX = int(np.max(x_coords))
        endY = int(np.max(y_coords))
        class_name = names[int(classes[i])]
        output.append({
            "startX": startX,
            "startY": startY,
            "endX": endX,
            "endY": endY,
            "class": class_name,
            "score": float(scores[i])
        })

    return {"boxes": output}
