"""
Module 2: OCR & Tokenization
Goal: Read text and get each word with its position
"""
import pytesseract
from PIL import Image
import json
import os
import logging

logger = logging.getLogger("modules.ocr_processor")

class OCRProcessor:
    def __init__(self, tesseract_path: str = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
        # Validate availability
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR initialized successfully")

    def extract_text_with_positions(self, image_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        image = Image.open(image_path)
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"
        )
        tokens = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            try:
                conf = int(float(data["conf"][i]))
            except Exception:
                conf = -1
            if txt and conf > 30:
                left = int(data["left"][i]); top = int(data["top"][i])
                width = int(data["width"][i]); height = int(data["height"][i])
                tokens.append({
                    "text": txt,
                    "left": left, "top": top,
                    "width": width, "height": height,
                    "right": left + width, "bottom": top + height,
                    "confidence": conf
                })
        tokens.sort(key=lambda x: (x["top"], x["left"]))
        page_name = os.path.splitext(os.path.basename(image_path))[0]
        out = {"image_path": image_path, "tokens": tokens, "total_tokens": len(tokens)}
        with open(os.path.join(output_dir, f"tokens_{page_name}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return out

    def extract_lines(self, tokens, line_threshold: int = 10):
        if not tokens:
            return []
        lines = []
        current = [tokens[0]]
        current_y = tokens[0]["top"]
        for t in tokens[1:]:
            if abs(t["top"] - current_y) <= line_threshold:
                current.append(t)
            else:
                current.sort(key=lambda x: x["left"])
                lines.append(current)
                current = [t]
                current_y = t["top"]
        if current:
            current.sort(key=lambda x: x["left"])
            lines.append(current)
        return lines

    def get_line_text(self, line_tokens):
        return " ".join(t["text"] for t in line_tokens)
