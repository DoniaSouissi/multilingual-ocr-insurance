import uuid
import os
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from ArabicOcr import arabicocr
from objectDetection import model_detect
import subprocess
import shutil

router = APIRouter()

def correct_arabic_fields_from_text(extracted_text: str, model: str = "mistral") -> str:
    arabic_fields = [
        "تاريخ الحادث", "الساعة", "المكان", "الجرحى", "نعم", "لا", "الاضرار", "المادية", "الشهود",
        "شركة", "تأمين", "مؤمنة", "من", "طرف", "رخصة", "صالحة", "إلى",
        "هوية", "السائق", "الاسم", "اللقب", "العنوان", "قيادة", "مسلمة", "بتاريخ",
        "المؤمن", "الهاتف", "السيارة", "النوع", "رقم", "الاتجاه", "آت", "ذاهب",
        "نقطة", "الصدم", "البارزة", "الأضرار", "الظاهرية", "ملاحظات", "السائقين"
    ]
    reference_words = "، ".join(arabic_fields)

    prompt =   f"""أنت مساعد ذكي مختص في تصحيح نتائج OCR للوثائق الخاصة بحوادث السيارات.

النص التالي يحتوي على كلمات تم استخراجها من وثيقة ممسوحة ضوئيًا، لكنها تحتوي على أخطاء في التعرف الضوئي للحروف (OCR).

إليك الكلمات المرجعية الصحيحة والمتوقعة في هذا النوع من الوثائق:
{reference_words}

🔒 تعليمات صارمة:
1. صحّح فقط الكلمات التي تحتوي على أخطاء OCR استنادًا إلى الكلمات المرجعية أعلاه.
2. لا تغيّر الكلمات الصحيحة، اتركها كما هي.
3.صحِّح الكلمات المتشابهة بصريًا (على سبيل المثال "كة" ← "شركة"، "مق"←"مؤمنة"، "آمنة"←"مؤمنة"، "تكافا"←"تكافل"، "النيسابة"←"النيابة"، "هسوية"←"هوية"، ، "رخمة"←"رخصة"، "السباقة"←"السياقة"، "التا"←"التأمين"، "نىة "←"سنية"،     
 "نقدلآ"←"نقطة"، "الاصلبة"←"الاصلية"  ).
4. حافظ على نفس ترتيب الكلمات والمسافات كما هي في النص الأصلي.
5. لا تضف أي كلمات جديدة ولا تعليقات ولا رموز خاصة.لا تعد صياغة النص
6. إذا كانت الكلمة غير واضحة، حاول تصحيحها حسب النطق أو الشكل القريب.
7. الجواب يكون فقط بكلمات مصححة، مفصولة بمسافات.
8. استخدم اللغة العربية فقط، ولا تدرج أي كلمات أو أحرف بلغة أخرى.
9.لا تستخدم سطر جديد أو \\n إطلاقًا
10. Don't add any *Notes*.
🔍 النص المراد تصحيحه:
{extracted_text}:"""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1500
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode()}")

        corrected = result.stdout.decode().strip()
        
        if corrected.startswith("📝 الكلمات المصححة:"):
            corrected = corrected.replace("📝 الكلمات المصححة:", "").strip()

        return corrected

    except subprocess.TimeoutExpired:
        raise RuntimeError("Le processus Ollama a dépassé le temps limite.")
    except FileNotFoundError:
        raise RuntimeError("Ollama n’est pas installé. Merci de l’installer d’abord.")
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l’appel à Ollama : {str(e)}")

def order_boxes_by_arabic(detections, y_threshold=15):
    centers = []
    for det in detections:
        x1, y1, x2, y2 = det["bounding_box"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_y, center_x, det))
    centers.sort(key=lambda x: x[0])
    lines, current_line = [], []
    prev_y = None
    for cy, cx, det in centers:
        if prev_y is None or abs(cy - prev_y) < y_threshold:
            current_line.append((cx, det))
            prev_y = cy if prev_y is None else (prev_y + cy) / 2
        else:
            current_line.sort(key=lambda x: x[0], reverse=True)
            lines.append([d for _, d in current_line])
            current_line = [(cx, det)]
            prev_y = cy
    if current_line:
        current_line.sort(key=lambda x: x[0], reverse=True)
        lines.append([d for _, d in current_line])
    return [det for line in lines for det in line]



def detect_typo_arabic_from_pil(image_pil: Image.Image):

    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


    results = model_detect.predict(
        source=image_bgr,
        conf=0.5, 
        save=False, 
        imgsz=640, 
        verbose=False)
    boxes = results[0].obb

    if not boxes or boxes.xyxyxyxy is None or len(boxes.xyxyxyxy) == 0:
        return {"message": "لم يتم اكتشاف أي أخطاء إملائية", "typos": [], "combined_text": "", "corrected_text": ""}

    obb_coords = boxes.xyxyxyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    names = model_detect.names

    typo_detections = [
        {
            "polygon": obb_coords[i],
            "score": scores[i],
            "bounding_box": cv2.boundingRect(obb_coords[i].astype(np.int32).reshape((-1, 1, 2))),
            "original_index": i
        }
        for i in range(len(obb_coords)) if names[int(classes[i])].lower() == "typo"
    ]

    if not typo_detections:
        return {"message": "لم يتم اكتشاف أي أخطاء إملائية", "typos": [], "combined_text": "", "corrected_text": ""}

    ordered_detections = order_boxes_by_arabic(typo_detections)
    typo_texts, typo_details = [], []

    TEMP_DIR = "./temp_images"
    os.makedirs(TEMP_DIR, exist_ok=True)

    for i, det in enumerate(ordered_detections):
        x, y, w, h = det["bounding_box"]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(image_bgr.shape[1], x + w), min(image_bgr.shape[0], y + h)

        cropped = image_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        temp_input = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.jpg")
        temp_output = os.path.join(TEMP_DIR, f"output_{uuid.uuid4().hex}.jpg")

        if not cv2.imwrite(temp_input, cropped):
            continue

        try:
            results = arabicocr.arabic_ocr(temp_input, temp_output)
            words = [res[1] for res in results if len(res) > 1]
            recognized_text = " ".join(words).strip()
        except Exception:
            recognized_text = ""

        for path in [temp_input, temp_output]:
            if os.path.exists(path):
                os.remove(path)

        if recognized_text:
            typo_texts.append(recognized_text)
            typo_details.append({
                "typo_id": i + 1,
                "text": recognized_text,
                "detection_score": float(det["score"]),
                "reading_order": i + 1,
                "position": {
                    "x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)
                }
            })

    combined_text = " ".join(typo_texts)
    
    try:
        corrected_text = correct_arabic_fields_from_text(combined_text)
    except Exception as e:
        corrected_text = combined_text + " (non corrigé)"
        print(f"[Correction Error]: {e}")

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    return {
        "message": "Traitement réussi",
        "typos": typo_details,
        "combined_text": combined_text,
        "corrected_text": corrected_text
    }


@router.post("/Pipeline Arabic")
async def detect_typo_arabic(image: UploadFile = File(...)):
    contents = await image.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    boxes = detect_typo_arabic_from_pil(image_pil)
    return {"boxes": boxes}


