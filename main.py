"""
Main FastAPI Application - Lab Report Digitization System
Combines all modules into a REST API with demo interface
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

# Local modules
from modules.preprocessing import FilePreprocessor
from modules.ocr_processor import OCRProcessor
from modules.rule_based_extractor import RuleBasedExtractor
from modules.ml_extractor import SimpleMLExtractor, HITLManager, EnhancedExtractor

# ---------- Project root + path helper ----------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT) 
def P(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Project root: %s", PROJECT_ROOT)

app = FastAPI(
    title="Lab Report Digitization API",
    description="Automatically extract and digitize patient details and test results from lab reports",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Ensure project-local directories exist ----------
for d in ["data/input", "data/processed", "data/corrections", "outputs", "static", "models"]:
    os.makedirs(P(d), exist_ok=True)

# ---------- Initialize components (Windows tool paths configurable) ----------
# Adjust these paths if installed elsewhere
preprocessor = FilePreprocessor(poppler_path=r"C:\poppler-25.07.0\Library\bin")
ocr_processor = OCRProcessor(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
rule_extractor = RuleBasedExtractor()
ml_extractor = SimpleMLExtractor(model_dir=P("models"))
hitl_manager = HITLManager(corrections_dir=P("data/corrections"))

# Load ML models if present
ml_extractor.load_models()

# Combined extractor (rule-based + ML)
enhanced_extractor = EnhancedExtractor(rule_extractor, ml_extractor, hitl_manager)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def root():
    html = """<!DOCTYPE html><html><head><title>Lab Report Digitization Demo</title>
    <style>
    body{font-family:Arial;margin:40px;background:#f5f5f5}
    .container{max-width:860px;margin:0 auto;background:#fff;padding:30px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}
    .upload{border:2px dashed #c8c8c8;padding:30px;text-align:center;border-radius:8px}
    button{background:#2563eb;color:#fff;padding:12px 20px;border:0;border-radius:6px;cursor:pointer}
    button:hover{background:#1e40af}
    .result{margin-top:24px;padding:16px;background:#fafafa;border-radius:6px}
    .ok{color:#166534;background:#dcfce7;padding:8px 12px;border-radius:4px;display:inline-block}
    .err{color:#991b1b;background:#fee2e2;padding:8px 12px;border-radius:4px;display:inline-block}
    table{width:100%;border-collapse:collapse;margin-top:10px}
    th,td{border:1px solid #ddd;padding:8px}
    th{background:#f0f0f0}
    .confidence-high{color:#15803d}
    .confidence-medium{color:#a16207}
    .confidence-low{color:#b91c1c}
    pre{background:#f7f7f7;padding:10px;border-radius:4px;overflow:auto}
    </style></head><body><div class="container">
    <h2>🏥 Lab Report Digitization</h2>
    <form id="form" enctype="multipart/form-data"><div class="upload">
    <input type="file" id="file" name="file" accept=".pdf,.jpg,.jpeg,.png" required /><br/><br/>
    <button type="submit">Process Report</button></div></form>
    <div id="loading" style="display:none;">🔄 Processing...</div><div id="result"></div>
    </div>
    <script>
    const form=document.getElementById('form');
    form.addEventListener('submit',async(e)=>{
      e.preventDefault();
      const file=document.getElementById('file').files[0];
      if(!file) return alert('Please select a file');
      const result=document.getElementById('result');
      const loading=document.getElementById('loading');
      loading.style.display='block'; result.innerHTML='';
      const fd=new FormData(); fd.append('file',file);
      try{
        const r=await fetch('/upload',{method:'POST',body:fd});
        const data=await r.json(); if(!r.ok) throw new Error(data.detail||'Error');
        let html='<div class="result"><div class="ok">✅ Processed</div>';
        if(data.patient){
          html+='<h4>👤 Patient</h4>';
          for(const [k,v] of Object.entries(data.patient)){
            const c=(data.confidence_scores?.patient?.[k] ?? 0);
            const cls=c>0.8?'confidence-high':(c>0.5?'confidence-medium':'confidence-low');
            html+=`<div>${k}: ${v} <span class="${cls}">(${(c*100).toFixed(0)}%)</span></div>`;
          }
        }
        if(data.tests?.length){
          html+='<h4>🧪 Tests</h4><table><tr><th>Name</th><th>Value</th><th>Unit</th><th>Confidence</th></tr>';
          for(const t of data.tests){
            const cls=t.confidence>0.8?'confidence-high':(t.confidence>0.5?'confidence-medium':'confidence-low');
            html+=`<tr><td>${t.name}</td><td>${t.value}</td><td>${t.unit||'-'}</td><td class="${cls}">${(t.confidence*100).toFixed(0)}%</td></tr>`;
          }
          html+='</table>';
        }
        if(data.needs_review?.length){
          html+='<h4>⚠️ Needs Review</h4><ul>'+data.needs_review.map(x=>`<li>${x}</li>`).join('')+'</ul>';
        }
        html+='<h4>📋 JSON</h4><pre>'+JSON.stringify(data,null,2)+'</pre></div>';
        result.innerHTML=html;
      }catch(err){
        result.innerHTML=`<div class="err">Error: ${err.message}</div>`;
      }finally{
        loading.style.display='none';
      }
    });
    </script></body></html>"""
    return html

@app.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    allowed = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    with tempfile.NamedTemporaryFile(delete=False, dir=P("data/input"), suffix=f"_{file.filename}") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        result = await process_lab_report(path)
        fname = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(P("outputs", fname), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["output_file"] = fname
        return result
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError:
                pass

async def process_lab_report(file_path: str) -> Dict:
    processed_images = preprocessor.process_file(file_path, P("data/processed"))
    all_tokens = []
    all_text = ""

    for img in processed_images:
        o = ocr_processor.extract_text_with_positions(img, P("data/processed"))
        all_tokens.extend(o["tokens"])
        lines = ocr_processor.extract_lines(o["tokens"])
        for line in lines:
            all_text += ocr_processor.get_line_text(line) + "\n"

    extraction = enhanced_extractor.extract_with_ml_enhancement(all_text)
    extraction["metadata"] = {
        "processed_images": len(processed_images),
        "total_tokens": len(all_tokens),
        "processing_timestamp": datetime.now().isoformat(),
        "original_filename": os.path.basename(file_path)
    }
    return extraction

@app.post("/correct")
async def submit_correction(
    correction_json: str = Form(...),
    report_id: Optional[str] = Form(None)
):
    try:
        data = json.loads(correction_json)
        cid = hitl_manager.save_correction(
            data.get("original", {}),
            data.get("corrected", {}),
            report_id
        )
        corrections = hitl_manager.get_training_data()
        if len(corrections) >= 5:
            scores = ml_extractor.train(corrections)
            return {"message": "Correction saved and model retrained", "correction_id": cid, "training_scores": scores}
        return {"message": "Correction saved", "correction_id": cid, "note": f"Need {5-len(corrections)} more corrections to retrain"}
    except Exception as e:
        logger.exception("Save correction error")
        raise HTTPException(status_code=500, detail=f"Could not save correction: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "preprocessor": "ready",
            "ocr": "ready",
            "rule_extractor": "ready",
            "ml_extractor": "trained" if ml_extractor.is_trained else "not_trained"
        }
    }

@app.get("/stats")
async def get_stats():
    corrections = hitl_manager.get_training_data()
    outputs = len([f for f in os.listdir(P("outputs")) if f.endswith(".json")])
    return {
        "total_corrections": len(corrections),
        "total_processed_reports": outputs,
        "ml_model_trained": ml_extractor.is_trained,
        "available_field_classifiers": list(ml_extractor.field_classifiers.keys()) if ml_extractor.is_trained else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
