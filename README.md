# Lab Report Digitization API

Convert PDF/image lab reports into structured JSON with patient details, test results, and confidence scores.  
The system combines preprocessing + OCR, rule-based extraction, post-processing cleanup, and human-in-the-loop (HITL) corrections that incrementally train a lightweight model to boost confidence for recurring fields.

---

## 🚀 Highlights
- PDF/image uploads; per-page preprocessing (deskew, denoise, binarize).
- OCR with word-level boxes and line reconstruction.
- Post-processing that filters headers/footers/ranges, merges wrapped lines, and canonicalizes test names.
- Rule-based extraction for patient fields and assays with confidence scores and `"needs_review"`.
- HITL corrections; after ≥5, auto-train a memory model; blend with rule confidence.
- Simple demo UI and Swagger docs.

---

## 📂 Directory Layout
```bash
project/
├─ run.py
├─ main.py
├─ modules/
│ ├─ preprocessing.py
│ ├─ ocr_processor.py
│ ├─ rule_based_extractor.py
│ ├─ ml_extractor.py
│ └─ postprocess.py # optional
├─ data/
│ ├─ input/ # uploaded files (temp)
│ ├─ processed/ # page_XX.png, tokens_page_XX.json
│ └─ corrections/ # saved correction JSONs
├─ outputs/ # result_*.json
├─ models/ # field_classifiers.pkl (after training)
├─ static/ # UI assets (optional)
├─ README.md
└─ requirements.txt
```

---

## 🛠 Prerequisites
- Python **3.10+**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed  
  - Windows default: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- [Poppler for pdf2image](https://github.com/oschwartz10612/poppler-windows)  
  - Windows example: `C:\poppler-25.07.0\Library\bin`

---

## ⚙️ Installation
Clone and enter the project directory:

```bash
git clone <repo-url>
cd project
```

---

## 🔧 Configuration

Configure external binary paths (Windows examples) in main.py:
```bash
preprocessor = FilePreprocessor(poppler_path=r"C:\poppler-25.07.0\Library\bin")
ocr_processor = OCRProcessor(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
```
Ensure folder anchoring is enabled:

```bash
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
```
Use P("...") helper for all project-relative paths.

---

## 📄 Usage
Single Report

- Upload a PDF/JPG/PNG in the demo UI.

- Output JSON appears on the page and is written to outputs/result_*.json.

- Patient details and test table include confidence scores and "needs_review" flags.

### Submitting Corrections (HITL)

Prepare a correction payload:
```bash
{
  "original": { "patient": {...}, "tests": [...] },
  "corrected": { "patient": {...}, "tests": [...] }
}
```
Submit via Swagger:

POST /correct → saves correction to `data/corrections/`.

```Auto-training:``` After ≥5 corrections, `next /correct triggers` training.
Model saved to `models/field_classifiers.pkl`.

---

## 📡 API Endpoints

- `GET /` → Demo upload page

- `POST /upload` → Process uploaded file → returns structured JSON

- `POST /correct` → Save corrections; triggers training after ≥5

- `GET /health` → Component status (preprocessor, OCR, rule extractor, ML)

- `GET /stats` → Totals for processed reports, corrections, and trained fields

---

## License
```bash
MIT License
Copyright (c) 2025 Advait Singh

```

---


