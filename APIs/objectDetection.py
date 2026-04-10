from fastapi import FastAPI, File, UploadFile, APIRouter
from PIL import Image
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io


router = APIRouter()

#---------------- OBJECT DETECTION----------------

model_detect = YOLO("./best_detection.pt")

@router.post("/detect", response_class=StreamingResponse)
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO OBB detection
    results = model_detect.predict(source=image_bgr, conf=0.5, save=False, imgsz=640, verbose=False)

    boxes = results[0].obb

    if boxes is None or boxes.xyxyxyxy is None or len(boxes.xyxyxyxy) == 0:
        print("No OBB detections.")
        return StreamingResponse(io.BytesIO(contents), media_type="image/png")

    obb_coords = boxes.xyxyxyxy.cpu().numpy()       # shape: (N, 8)
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    names = model_detect.names

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

