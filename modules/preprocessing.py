"""
Module 1: File Input & Preprocessing
Goal: Get a clean image from any PDF/photo so OCR works well
"""
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger = logging.getLogger("modules.preprocessing")

class FilePreprocessor:
    def __init__(self, dpi=300, poppler_path=None):
        self.dpi = dpi
        self.poppler_path = poppler_path

    def process_file(self, file_path: str, output_dir: str):
        ext = file_path.lower().split(".")[-1]
        os.makedirs(output_dir, exist_ok=True)
        if ext == "pdf":
            return self._process_pdf(file_path, output_dir)
        if ext in ["jpg", "jpeg", "png"]:
            return [self._process_image(file_path, output_dir)]
        raise ValueError(f"Unsupported file format: {ext}")

    def _process_pdf(self, pdf_path, output_dir):
        kwargs = {"dpi": self.dpi}
        if self.poppler_path:
            kwargs["poppler_path"] = self.poppler_path
        images = convert_from_path(pdf_path, **kwargs)
        out = []
        for i, im in enumerate(images):
            tmp = os.path.join(output_dir, f"temp_{i+1}.png")
            cleaned = os.path.join(output_dir, f"page_{i+1:02d}.png")
            try:
                im.save(tmp)
                self._clean_image(tmp, cleaned)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except PermissionError:
                        pass
            out.append(cleaned)
            logger.info("Saved cleaned image: %s", cleaned)
        logger.info("Processed %d page(s)", len(out))
        return out

    def _process_image(self, image_path: str, output_dir: str):
        name = os.path.splitext(os.path.basename(image_path))[0]
        cleaned = os.path.join(output_dir, f"{name}_cleaned.png")
        self._clean_image(image_path, cleaned)
        logger.info("Saved cleaned image: %s", cleaned)
        return cleaned

    def _clean_image(self, input_path: str, output_path: str):
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self._deskew(gray)
        den = cv2.fastNlMeansDenoising(gray)
        thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(output_path, cleaned)

    def _deskew(self, image):
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is None or len(lines) == 0:
            return image
        angles = []
        for l in lines[:10]:
            if l is None or len(l) == 0:
                continue
            rho_theta = l[0]
            if rho_theta is None or len(rho_theta) < 2:
                continue
            theta = float(rho_theta[1])
            angle = np.degrees(theta) - 90.0
            if abs(angle) < 45:
                angles.append(angle)
        if not angles:
            return image
        median = float(np.median(angles))
        if abs(median) <= 0.5:
            return image
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), median, 1.0)
        corrected = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return corrected
