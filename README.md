# Intelligent System for the Automatic Extraction of Data from Multi-language Accident Reports 
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-red)
![AI-Models](https://img.shields.io/badge/AI-YOLOv11%20%7C%20TrOCR%20%7C%20Mistral-green)

## 📌 Project Overview
This system automates data extraction from bilingual (Arabic and French) car accident reports. By combining advanced computer vision and deep learning, the solution identifies document structures, distinguishes between handwritten and printed text, and performs high-accuracy OCR to streamline insurance claim management.

### Key Features
* **Bilingual Support:** Specialized extraction pipelines for Arabic and French.
* **Intelligent Segmentation:** Uses **YOLOv11** to detect and crop major report sections (Part A, Part B, Header, etc.).
* **Writing Style Classification:** Employs **YOLOv11L-OBB** to distinguish between handwritten and printed text zones.
* **High-Precision OCR:** Integrates **TrOCR** for French and **ArabicOCR** for Arabic text recognition.
* **Word-Level Processing:** Utilizes the **EAST (Efficient and Accurate Scene Text)** detector to crop individual words for improved recognition accuracy.
* **Language Detection:** Automatically routes text regions to the correct engine using **Langdetect**.

---

## 🏗 System Architecture & Pipeline
The project follows a modular pipeline designed for production-ready deployment:

1. **Segmentation:** The report is split into functional zones using YOLOv11.
2. **Classification:** Zones are labeled as "handwritten" or "printed".
3. **Language Detection:** Identifies if a zone is Arabic or French to select the OCR model.
4. **Text Extraction:** * **French:** Processed via TrOCR.
   * **Arabic:** Processed via ArabicOCR/PaddleOCR.
5. **Deployment:** Served via a **FastAPI** backend and an interactive **Streamlit** frontend, all containerized with **Docker**.
---

## 📂 Project Structure
```text
├── backend/            # FastAPI application logic
│   ├── models/         # YOLOv11 weights & OCR configurations
│   ├── modules/        # Segmentation, Detection, and OCR engines
│   ├── main.py         # API entry point
│   └── Dockerfile      # Backend containerization
├── frontend/           # Streamlit web interface
│   ├── app.py          # Interface logic
│   └── Dockerfile      # Frontend containerization
├── notebooks/          # Research and training scripts
├── data/               # Dataset samples (Reports)
├── docker-compose.yml  # Orchestration for full-stack deployment
└── requirements.txt    # Project dependencies
