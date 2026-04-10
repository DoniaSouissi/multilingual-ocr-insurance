from fastapi import File, UploadFile, APIRouter
import io
from PIL import Image
import numpy as np
import cv2
from paddleocr import PaddleOCR
import torch
import subprocess
from objectDetection import model_detect
from frenchExtraction import processor, model_extraction
import re


def correct_fields_from_text(extracted_text: str, model: str = "mistral") -> str:
    champs_typiques = [
        "date de l'accident", "heure", "lieu", "blessés", "même légers",
        "dégâts matériels", "autre", "qu'aux véhicules A et B", "oui", "non",
        "témoins", "noms,", "addresses", "et", "(à souligner", "s'il", "s'agit",
        "d'un passager", "de", "A", "ou", "B)", "Société", "d’Assurances",
        "Véhicule", "assuré", "par", "Police", "d’Assurance", "N°", "Agence",
        "Attestation", "valable", "du", "au", "Identité", "du", "Conducteur",
        "Nom", "Prénom", "Adresse", "Permis", "de", "conduire", "Délivré", "le",
        "Assuré", "(voir", "attest.", "d’assur.)", "Tél.","Identité ", "du",
        "Véhicule", "Marque", "Type",
        "N°", "d’immatriculation", "Sens", "suivi", "Venant", "de", "Allant",
        "à", "Indiquer", "par", "une flèche", "le point", 
        "de", "choc", "initial", "dégâts", "apparents",
        "observations", "conducteurs"
    ]

    # Normalize and deduplicate the list
    champs_typiques = sorted(set(word.lower() for word in champs_typiques))
    reference_words = ", ".join(champs_typiques)

    # Clean extracted text
    extracted_text = extracted_text.lower()
    extracted_text = re.sub(r"[-_:]", " ", extracted_text)   # Remove noise
    extracted_text = re.sub(r"\s+", " ", extracted_text).strip()

    prompt = f"""
Tu es un assistant OCR intelligent.

Le texte suivant contient des mots extraits d’un document scanné avec des erreurs.

Ta tâche :
- **Ne change jamais la langue**
- Corrige chaque mot extrait en utilisant uniquement la **liste de référence** ci-dessous.
- Corrige les fautes d’orthographe.
- Corrige les mots visuellement proches (ex: "no" → "n°").
- Supprime les caractères parasites : tirets (-), underscores (_), deux-points (:).
- Ne change pas l’ordre des mots.
- N’utilise que les mots de la liste.
- Retourne uniquement les mots corrigés, séparés par un espace.
- Ne donne **aucune explication**.
- Le texte corrigé doit utiliser de vrais sauts de ligne (retours à la ligne visibles), un pour chaque champ ou groupe de mots logique.
- La seule langue acceptée est le Français.



Liste de référence :
{reference_words}

Texte extrait :
{extracted_text}

Corrige uniquement. Ne retourne que les mots corrigés.
"""

    print("📝 Prompt sent to Mistral:\n", prompt)

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=800
        )

        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.decode()}")
        return result.stdout.decode().strip()

    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama process timed out")

ocr_model = PaddleOCR(use_angle_cls=False, lang='fr', det=False, rec=True)


router = APIRouter()


def detect_typo_french_from_pil(image_pil: Image.Image):
    """
    Endpoint pour détecter uniquement les fautes de frappe (typos) dans une image
    et retourner le texte combiné dans l'ordre de lecture avec correction Mistral
    """
    # 1. Lire et préparer l'image
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 2. Effectuer la détection avec le modèle
    results = model_detect.predict(
        source=image_bgr, 
        conf=0.5, 
        save=False, 
        imgsz=640, 
        verbose=False
    )
    boxes = results[0].obb
    
    # 3. Vérifier s'il y a des détections
    if boxes is None or boxes.xyxyxyxy is None or len(boxes.xyxyxyxy) == 0:
        print("Aucune détection trouvée.")
        return {"message": "Aucune faute de frappe détectée", "typos": [], "combined_text": ""}
    
    # 4. Récupérer les données des détections
    obb_coords = boxes.xyxyxyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    names = model_detect.names
    
    # 5. Filtrer uniquement les détections de type "typo" avec leurs boîtes englobantes
    typo_detections_with_boxes = []
    for i in range(len(obb_coords)):
        class_id = int(classes[i])
        class_name = names[class_id]
        
        # Ne traiter que si c'est une faute de frappe
        if class_name.lower() == "typo":
            polygon = obb_coords[i].astype(np.int32).reshape((-1, 1, 2))
            rect = cv2.boundingRect(polygon)
            x, y, w, h = rect
            
            # Créer une boîte englobante au format (startX, startY, endX, endY)
            bounding_box = (x, y, x + w, y + h)
            
            typo_detections_with_boxes.append({
                'polygon': obb_coords[i],
                'score': scores[i],
                'bounding_box': bounding_box,
                'original_index': i
            })
    
    # 6. Si aucune faute de frappe détectée, retourner message
    if not typo_detections_with_boxes:
        print("Aucune faute de frappe détectée.")
        return {"message": "Aucune faute de frappe détectée", "typos": [], "combined_text": ""}
    
    # 7. Ordonner les détections selon l'ordre de lecture
    def order_boxes_by_reading_order(detections, y_threshold=10):
        """
        Ordonne les boîtes dans l'ordre de lecture naturel : haut-bas, gauche-droite
        """
        # Calculer le centre de chaque boîte
        box_centers = []
        for detection in detections:
            startX, startY, endX, endY = detection['bounding_box']
            centerY = (startY + endY) / 2
            centerX = (startX + endX) / 2
            box_centers.append((centerY, centerX, detection))
        
        # Trier par centerY (haut vers bas)
        box_centers.sort(key=lambda x: x[0])
        
        # Grouper en lignes
        lines = []
        current_line = []
        prev_centerY = None
        
        for centerY, centerX, detection in box_centers:
            if prev_centerY is None:
                current_line.append((centerX, detection))
                prev_centerY = centerY
            else:
                if abs(centerY - prev_centerY) < y_threshold:
                    current_line.append((centerX, detection))
                    prev_centerY = (prev_centerY + centerY) / 2
                else:
                    # Trier la ligne actuelle par centerX (gauche vers droite)
                    current_line.sort(key=lambda x: x[0])
                    lines.append([d for _, d in current_line])
                    current_line = [(centerX, detection)]
                    prev_centerY = centerY
        
        if current_line:
            current_line.sort(key=lambda x: x[0])
            lines.append([d for _, d in current_line])
        
        # Aplatir la liste des lignes en une seule liste ordonnée
        ordered_detections = [detection for line in lines for detection in line]
        return ordered_detections
    
    # 8. Appliquer l'ordre de lecture
    ordered_typo_detections = order_boxes_by_reading_order(typo_detections_with_boxes)
    
    # 9. Traiter chaque détection de faute de frappe dans l'ordre et collecter le texte
    typo_texts = []
    typo_details = []
    
    for i, detection in enumerate(ordered_typo_detections):
        polygon = detection['polygon'].astype(np.int32).reshape((-1, 1, 2))
        
        # 9a. Découper la région du polygone
        rect = cv2.boundingRect(polygon)
        x, y, w, h = rect
        cropped = image_bgr[y:y+h, x:x+w].copy()
        
        # 9b. Créer un masque pour extraire uniquement la zone du polygone
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        polygon_relative = polygon - [x, y]
        cv2.fillPoly(mask, [polygon_relative], 255)
        cropped_masked = cv2.bitwise_and(cropped, cropped, mask=mask)
        
        # 9c. Reconnaissance OCR de la faute de frappe
        """crop_pil = Image.fromarray(cv2.cvtColor(cropped_masked, cv2.COLOR_BGR2RGB))
        ocr_result = ocr_model.ocr(np.array(crop_pil), cls=False)
        
        if ocr_result and ocr_result[0]:
            recognized_text = ocr_result[0][0][1][0]
            confidence = ocr_result[0][0][1][1]
            print(f"[FAUTE DE FRAPPE {i+1}] OCR: {recognized_text}")
        else:
            recognized_text = ""
            confidence = 0.0"""
        try:
            # Convert cropped image to RGB and PIL format
            cropped_rgb = cv2.cvtColor(cropped_masked, cv2.COLOR_BGR2RGB)
            cropped_pil = Image.fromarray(cropped_rgb)

            # Preprocess image for TrOCR
            pixel_values = processor(images=cropped_pil, return_tensors="pt").pixel_values
            pixel_values = pixel_values.unsqueeze(0) if pixel_values.dim() == 3 else pixel_values

            # Generate prediction
            with torch.no_grad():
                generated_ids = model_extraction.generate(pixel_values)

            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            confidence = 0.99  # TrOCR doesn't provide confidence scores natively
            print(f"TrOCR Output: {recognized_text}")

        except Exception as e:
            recognized_text = ""
            confidence = 0.0
            print(f"[OCR ERROR - TrOCR] {e}")

        
        # Collecter le texte et les détails dans l'ordre de lecture
        if recognized_text.strip():
            typo_texts.append(recognized_text.strip())
            typo_details.append({
                "typo_id": i + 1,
                "text": recognized_text.strip(),
                "confidence": confidence,
                "detection_score": float(detection['score']),
                "reading_order": i + 1,
                "position": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })
    
    # 10. Combiner tout le texte des fautes de frappe dans l'ordre de lecture
    combined_text = " ".join(typo_texts)
    
    # 11. Correction du texte brut avec Mistral
    corrected_text = ""
    correction_error = None
    individual_corrections = []
    
    if combined_text.strip():
        try:
            print(f"[CORRECTION] Texte brut: {combined_text}")
            corrected_text = correct_fields_from_text(combined_text)
            print(f"[CORRECTION] Texte corrigé: {corrected_text}")
            
            # 12. Optionnel: Correction individuelle de chaque typo
            corrected_words = corrected_text.split() if corrected_text else []
            for i, (original_word, corrected_word) in enumerate(zip(typo_texts, corrected_words)):
                if i < len(typo_details):
                    typo_details[i]["corrected_text"] = corrected_word
                    individual_corrections.append({
                        "original": original_word,
                        "corrected": corrected_word,
                        "position": typo_details[i]["position"]
                    })
            
        except Exception as e:
            correction_error = str(e)
            corrected_text = combined_text  # Garder le texte original en cas d'erreur
            print(f"[ERREUR CORRECTION] {correction_error}")
    
    return {
        "typos": typo_details,
        "raw_text": combined_text,
        "corrected_text": corrected_text,
        "correction_error": correction_error,
        "total_typos": len(typo_details),
        "correction_applied": correction_error is None and bool(corrected_text)
    }

@router.post("/Pipeline French")
async def detect_typo_french(image: UploadFile = File(...)):
    contents = await image.read()
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    boxes = detect_typo_french_from_pil(image_pil)
    return {"boxes": boxes}
