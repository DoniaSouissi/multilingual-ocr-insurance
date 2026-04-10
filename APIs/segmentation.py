from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import JSONResponse
import io
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import uuid
from PIL import Image



router = APIRouter()
#----------------SEGMENTATION----------------
model_segment = YOLO("./best.pt")

segmented_storage = {}  # Store images temporarily in memory
@router.post("/segment")
async def segment_image_upload(file: UploadFile = File(...)):
    contents = await file.read()
    input_img = Image.open(io.BytesIO(contents)).convert("RGB")

    image = np.array(input_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model_segment.predict(source=image, conf=0.5, save=False, imgsz=640, verbose=False)
    names = model_segment.names

    masks = results[0].masks
    if masks is None:
        return JSONResponse(content={"error": "No segmentation detected."})

    mask_array = masks.data.cpu().numpy()
    H, W, _ = image.shape
    ids = []

    for i, mask in enumerate(mask_array):
        if results[0].boxes is not None and len(results[0].boxes.cls) > i:
            class_id = int(results[0].boxes.cls[i].item())
            class_name = names[class_id]
        else:
            class_name = "unknown"


        resized_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = resized_mask > 0.5

        segmented = np.zeros_like(image)
        segmented[mask_bool] = image[mask_bool]

        segment_id = str(uuid.uuid4())
        segmented_storage[segment_id] = segmented
        ids.append({"id": segment_id, "class": class_name})

    return {"segments": ids}


@router.get("/segment/{segment_id}")
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
