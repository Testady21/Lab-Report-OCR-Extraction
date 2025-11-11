"""
Module 3: Rule-based extraction
Simple regex/keyword heuristics for patient fields and tests
"""
import re

class RuleBasedExtractor:
    def __init__(self):
        # Example patterns; extend to your lab formats
        self.name_keys = [r"patient\s*name", r"pt\.?\s*name", r"name"]
        self.age_keys = [r"\bage\b", r"age[:\s]"]
        self.gender_keys = [r"\bsex\b", r"gender"]
        self.id_keys = [r"(patient|reg|uhid|id)\s*[:#]?\s*([A-Za-z0-9\-]+)"]
        self.date_keys = [r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"]

        # Test rows: naive pattern "Name value unit"
        self.value_pat = re.compile(r"([-+]?\d+(\.\d+)?)")
        self.unit_pat = re.compile(r"\b(g/dL|mg/dL|mmol/L|/µL|/uL|IU/L|%)\b", re.I)

    def extract(self, text: str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        patient = {}
        tests = []

        # Patient fields (very simple; improve as needed)
        for l in lines[:40]:
            low = l.lower()
            if "name" in low and "patient" in low and "name" not in patient:
                patient["name"] = l.split(":")[-1].strip()
            if any(k in low for k in ["age", "yrs", "years"]) and "age" not in patient:
                m = re.search(r"(\d{1,3})", l)
                if m: patient["age"] = int(m.group(1))
            if any(k in low for k in ["gender", "sex"]) and "gender" not in patient:
                if "male" in low: patient["gender"] = "Male"
                elif "female" in low: patient["gender"] = "Female"
            if "uhid" in low or "reg" in low or "patient id" in low:
                m = re.search(r"[:#]\s*([A-Za-z0-9\-]+)", l)
                if m: patient["patient_id"] = m.group(1)
            if "date" in low and "date" not in patient:
                m = re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", l)
                if m: patient["date"] = m.group(0)

        # Tests: very naive parse; refine per lab layout
        for l in lines:
            val = self.value_pat.search(l)
            if not val: continue
            unit = self.unit_pat.search(l)
            name = l
            tests.append({
                "name": name if len(name) < 60 else name[:60],
                "value": float(val.group(1)),
                "unit": unit.group(0) if unit else None,
                "confidence": 0.6  # baseline; blended later with ML
            })

        return {"patient": patient, "tests": tests}
