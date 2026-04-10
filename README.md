# Accident-Extract AI: Intelligent Multi-language Report Processing

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20FastAPI-green)

## 📌 Project Overview
[cite_start]Developed as a **National Bachelor’s Degree (PFE)** project at the **Horizon School of Digital Technologies**, this system automates data extraction from bilingual (Arabic and French) car accident reports[cite: 1, 2]. [cite_start]By combining advanced computer vision and deep learning, the solution identifies document structures, distinguishes between handwritten and printed text, and performs high-accuracy OCR to streamline insurance claim management[cite: 3, 6].

### Key Features
* [cite_start]**Bilingual Support:** Specialized extraction pipelines for Arabic and French[cite: 7].
* [cite_start]**Intelligent Segmentation:** Uses **YOLOv11** to detect and crop major report sections (Part A, Part B, Header, etc.)[cite: 205, 327].
* [cite_start]**Writing Style Classification:** Employs **YOLOv11L-OBB** to distinguish between handwritten and printed text zones[cite: 229, 332].
* [cite_start]**High-Precision OCR:** Integrates **TrOCR** for French and **ArabicOCR** for Arabic text recognition[cite: 5, 302, 305].
* [cite_start]**Word-Level Processing:** Utilizes the **EAST (Efficient and Accurate Scene Text)** detector to crop individual words for improved recognition accuracy[cite: 339, 341].
* [cite_start]**Language Detection:** Automatically routes text regions to the correct engine using **Langdetect**[cite: 245, 335].

---

## 🏗 System Architecture & Pipeline
The project follows a modular pipeline designed for production-ready deployment:

1. [cite_start]**Segmentation:** The report is split into functional zones using YOLOv11[cite: 327].
2. [cite_start]**Classification:** Zones are labeled as "handwritten" or "printed"[cite: 333].
3. [cite_start]**Language Detection:** Identifies if a zone is Arabic or French to select the OCR model[cite: 238, 245].
4. [cite_start]**Text Extraction:** * **French:** Processed via TrOCR[cite: 342].
   * [cite_start]**Arabic:** Processed via ArabicOCR/PaddleOCR[cite: 5, 305].
5. [cite_start]**Deployment:** Served via a **FastAPI** backend and an interactive **Streamlit** frontend, all containerized with **Docker**[cite: 311, 312].

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
