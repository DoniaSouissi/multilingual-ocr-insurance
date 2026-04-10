import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

# 1. Load the model 
model = YOLO("/home/donia/deployment1/segmentation_api/best.pt") 

# 2. Segmentation function
def segment_image(input_img):
    image = np.array(input_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model.predict(source=image, conf=0.5, save=False, imgsz=640, verbose=False)
    names = model.names

    masks = results[0].masks
    if masks is None:
        return ["No segmentation detected."]

    mask_array = masks.data.cpu().numpy()
    H, W, _ = image.shape
    segmented_outputs = []

    for i, mask in enumerate(mask_array):
        class_id = int(results[0].boxes.cls[i].item())
        class_name = names[class_id]

        # Resize mask to match image shape
        resized_mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_bool = resized_mask > 0.5

        segmented = np.zeros_like(image)
        segmented[mask_bool] = image[mask_bool]

        segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
        segmented_outputs.append((segmented_rgb, class_name))

    return segmented_outputs

# 3. Launch Gradio interface
interface = gr.Interface(
    fn=segment_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Gallery(label="Segmented Outputs"),
    title="YOLO Segmentation Model"
)

interface.launch()
