"""
Module 5: Simple ML + HITL + enhanced extraction
Trains simple text classifiers from corrections and blends confidences
"""
import os
import json
import pickle
from collections import defaultdict

class SimpleMLExtractor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.field_classifiers = {}  # stub: field -> set of seen strings
        self.is_trained = False

    def load_models(self):
        path = os.path.join(self.model_dir, "field_classifiers.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.field_classifiers = obj.get("field_classifiers", {})
            self.is_trained = bool(self.field_classifiers)
        else:
            self.is_trained = False

    def save_models(self):
        path = os.path.join(self.model_dir, "field_classifiers.pkl")
        with open(path, "wb") as f:
            pickle.dump({"field_classifiers": self.field_classifiers}, f)

    def train(self, corrections):
        field_memory = defaultdict(set)
        for c in corrections:
            corr = c.get("corrected", {})
            patient = corr.get("patient", {})
            for k, v in patient.items():
                if isinstance(v, str) and v.strip():
                    field_memory[k].add(v.strip().lower())
        self.field_classifiers = {k: set(vs) for k, vs in field_memory.items()}
        self.is_trained = True
        self.save_models()
        # Return trivial "scores"
        return {k: len(v) for k, v in self.field_classifiers.items()}

    def score_patient_field(self, field_name, value):
        if not self.is_trained:
            return 0.0
        if not isinstance(value, str):
            return 0.0
        mem = self.field_classifiers.get(field_name, set())
        return 1.0 if value.strip().lower() in mem else 0.5 if mem else 0.0

class HITLManager:
    def __init__(self, corrections_dir="data/corrections"):
        self.corrections_dir = corrections_dir
        os.makedirs(self.corrections_dir, exist_ok=True)

    def save_correction(self, original, corrected, report_id=None):
        rid = report_id or f"corr_{len(os.listdir(self.corrections_dir))+1:04d}"
        path = os.path.join(self.corrections_dir, f"{rid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"original": original, "corrected": corrected}, f, indent=2, ensure_ascii=False)
        return rid

    def get_training_data(self):
        data = []
        for fn in os.listdir(self.corrections_dir):
            if not fn.endswith(".json"): continue
            with open(os.path.join(self.corrections_dir, fn), "r", encoding="utf-8") as f:
                data.append(json.load(f))
        return data

class EnhancedExtractor:
    def __init__(self, rule_extractor, ml_extractor, hitl_manager):
        self.rule = rule_extractor
        self.ml = ml_extractor
        self.hitl = hitl_manager

    def extract_with_ml_enhancement(self, text: str):
        base = self.rule.extract(text)
        patient = base.get("patient", {})
        tests = base.get("tests", [])
        conf_scores = {"patient": {}}

        # Blend confidences for patient fields
        for k, v in list(patient.items()):
            ml_score = self.ml.score_patient_field(k, v) if isinstance(v, str) else 0.0
            # Combine: 60% rule baseline, 40% ML
            rule_score = 0.6
            combined = 0.6 * rule_score + 0.4 * ml_score
            conf_scores["patient"][k] = round(combined, 3)

        needs_review = []
        for k, c in conf_scores["patient"].items():
            if c < 0.7:
                needs_review.append(f"patient.{k}")

        # Keep rule confidences for tests; could add ML later
        out = {
            "patient": patient,
            "tests": tests,
            "confidence_scores": conf_scores,
            "needs_review": needs_review
        }
        return out
