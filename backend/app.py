"""
SmartSnap FastAPI Backend Application.
Provides REST API for:
- Photograph upload and OpenCV feature extraction
- Mamdani Fuzzy Inference Engine execution
- Interactive fuzzy simulation playground
- Dashboard statistics and history tracking
- Static serving of frontend build when available
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

if __package__:
    from .services.analysis_service import AnalysisService, ModelUnavailableError, UPLOAD_DIR
else:
    from services.analysis_service import AnalysisService, ModelUnavailableError, UPLOAD_DIR

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="SmartSnap Hybrid Image Quality Assessment API",
    description="Computer Vision, Random Forest MOS regression, and Mamdani fuzzy recommendations",
    version="2.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploaded image storage for preview
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize analysis service
service = AnalysisService()


class SimulationRequest(BaseModel):
    category: str = Field(default="General Photography")
    inputs: Dict[str, float] = Field(
        default={
            "blur": 25.0,
            "brightness": 50.0,
            "contrast": 60.0,
            "noise": 15.0,
            "face_visibility": 75.0,
            "composition": 70.0,
        }
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": "SmartSnap",
        "fuzzy_ready": True,
        "random_forest_ready": service.predictor is not None,
        "model_version": service.predictor.model_version if service.predictor else None,
        "model_error": service.model_error,
    }


@app.get("/api/fuzzy/metadata")
def get_fuzzy_metadata():
    """
    Returns full variable membership curve specifications and rule base for frontend visualization.
    """
    return service.fuzzy_engine.get_system_metadata()


@app.post("/api/analyze")
async def analyze_photo(
    file: UploadFile = File(...),
    category: str = Form("General Photography")
):
    """
    Uploads a photograph, predicts KonIQ MOS with the Random Forest, then applies
    Mamdani fuzzy inference and returns both model and final recommendations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    valid_exts = [".jpg", ".jpeg", ".png"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Use JPG, JPEG, or PNG.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        record = service.analyze_image_file(content, file.filename, category=category)
        return record
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        LOGGER.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=f"Image processing error: {error}")


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    """
    Runs fuzzy inference simulation on arbitrary user input sliders.
    Uses the exact same Mamdani engine as real photograph processing.
    """
    try:
        res = service.simulate_fuzzy(req.inputs, category=req.category)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/api/dashboard")
def get_dashboard():
    return service.get_dashboard_stats()


@app.get("/api/history")
def get_history():
    return service.get_history()


@app.get("/api/history/{record_id}")
def get_record(record_id: str):
    record = service.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.delete("/api/history")
def clear_history():
    service.clear_history()
    return {"status": "success", "message": "History cleared"}


# Mount static production build of frontend if it exists
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    app_module = "backend.app:app" if __package__ else "app:app"
    uvicorn.run(app_module, host="127.0.0.1", port=8000, reload=True)
