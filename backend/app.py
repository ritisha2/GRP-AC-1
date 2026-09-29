"""
ADVaIT FastAPI Application
Production REST API exposing:
- POST /predict: Batch quality inference & Golden Batch score
- POST /train: Retrain 11 ML regression models
- GET /health: API & model readiness health check
- GET /feature-report: Dataset intelligence overview & links
- GET /golden-profile: Historical Golden Profile statistical distributions
- POST /export/pdf, /export/excel, /export/csv: Downloadable report endpoints
- GET /sample-batches: Pre-loaded industrial batch presets
"""

import os
import sys
import json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.schemas import BatchProductionInput, BatchPredictionResponse
from backend.predict import predict_batch, load_target_models
from backend.golden_batch import load_golden_profile
from backend.dataset_intelligence import generate_intelligence_data, REPORTS_DIR
from backend.train import train_all_models, MODELS_DIR
from backend.reports_generator import (
    generate_pdf_batch_report,
    generate_excel_batch_report,
    generate_csv_batch_report
)

app = FastAPI(
    title="ADVaIT — BT Rubber Quality Prediction API",
    description="Industrial AI Backend for BT Rubber batch validation, process feature engineering, quality parameter prediction, and Golden Batch evaluation.",
    version="1.0.0"
)

# Enable CORS for Streamlit and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("Pre-loading ML models into memory...")
    load_target_models()
    load_golden_profile()

@app.get("/health", tags=["System"])
def health_check():
    models = load_target_models()
    profile = load_golden_profile()
    return {
        "status": "healthy",
        "service": "ADVaIT Industrial AI Backend",
        "models_loaded": len(models),
        "total_targets": 11,
        "golden_profile_ready": "features" in profile,
        "engineered_features_profiled": len(profile.get("features", {}))
    }

@app.post("/predict", tags=["Inference"])
def predict_endpoint(batch: BatchProductionInput):
    try:
        res = predict_batch(batch.dict())
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {str(e)}")

@app.post("/train", tags=["Model Lifecycle"])
def trigger_training(background_tasks: BackgroundTasks):
    try:
        # Run training in background so API remains responsive
        background_tasks.add_task(train_all_models)
        return {
            "status": "Training initiated",
            "message": "11-Target cross-validation benchmark and model training started in background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate training: {str(e)}")

@app.get("/feature-report", tags=["Dataset Intelligence"])
def get_feature_report():
    try:
        intel = generate_intelligence_data()
        html_path = os.path.join(REPORTS_DIR, "dataset_intelligence_report.html")
        pdf_path = os.path.join(REPORTS_DIR, "dataset_intelligence_report.pdf")
        
        return {
            "total_rows": intel["total_rows"],
            "total_columns": intel["total_cols_prod"],
            "engineered_features": intel["total_features"],
            "duplicate_batches": intel["duplicate_batches"],
            "predictability_scores": intel["predictability_scores"],
            "html_report_available": os.path.exists(html_path),
            "pdf_report_available": os.path.exists(pdf_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating intelligence report: {str(e)}")

@app.get("/feature-report/download/{format}", tags=["Dataset Intelligence"])
def download_feature_report(format: str):
    if format.lower() == "pdf":
        path = os.path.join(REPORTS_DIR, "dataset_intelligence_report.pdf")
        media_type = "application/pdf"
    elif format.lower() == "html":
        path = os.path.join(REPORTS_DIR, "dataset_intelligence_report.html")
        media_type = "text/html"
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Supported: pdf, html")
        
    if not os.path.exists(path):
        from backend.dataset_intelligence import run_dataset_intelligence
        run_dataset_intelligence()
        
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))

@app.get("/golden-profile", tags=["Golden Batch"])
def get_golden_profile():
    prof = load_golden_profile()
    return prof

@app.get("/prediction-history", tags=["Prediction History"])
def get_prediction_history():
    history_path = os.path.join(CURRENT_DIR, "data", "prediction_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

@app.post("/export/pdf", tags=["Reporting"])
def export_pdf(batch: BatchProductionInput):
    pred = predict_batch(batch.dict())
    path = generate_pdf_batch_report(pred)
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))

@app.post("/export/excel", tags=["Reporting"])
def export_excel(batch: BatchProductionInput):
    pred = predict_batch(batch.dict())
    path = generate_excel_batch_report(pred)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(path)
    )

@app.post("/export/csv", tags=["Reporting"])
def export_csv(batch: BatchProductionInput):
    pred = predict_batch(batch.dict())
    path = generate_csv_batch_report(pred)
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))

@app.get("/sample-batches", tags=["Testing Presets"])
def get_sample_batches():
    res = {
        "golden_sample": {
            "batch_id": "BT-040624-0",
            "batch_start_time": "11:30",
            "steam_valve_open_temp": 105.0,
            "steam_valve_open_pressure": 0.0,
            "steam_valve_open_time": "13:20",
            "steam_valve_close_temp": 172.0,
            "steam_valve_close_pressure": 8.5,
            "steam_valve_close_time": "16:40",
            "heat_valve_close_temp": 242.0,
            "heat_valve_close_pressure": 14.2,
            "heat_valve_close_time": "16:45",
            "low_tf_temp_duration": 340.0,
            "cooking_start_time": "13:20",
            "cooking_stop_time": "17:25",
            "cooking_duration": 245.0,
            "cooling_valve_open_temp": 232.0,
            "cooling_valve_open_pressure": 9.5,
            "cooling_valve_open_time": "17:25",
            "bottom_door_open_temp": 120.0,
            "bottom_door_open_pressure": 0.0,
            "bottom_door_open_time": "18:05",
            "standard_batch_duration": 400.0,
            "actual_batch_duration": 395.0,
            "batch_end_time": "18:05"
        },
        "warning_sample": {
            "batch_id": "BT-040580-WARN",
            "batch_start_time": "08:00",
            "steam_valve_open_temp": 95.0,
            "steam_valve_open_pressure": 0.2,
            "steam_valve_open_time": "08:45",
            "steam_valve_close_temp": 160.0,
            "steam_valve_close_pressure": 7.0,
            "steam_valve_close_time": "12:10",
            "heat_valve_close_temp": 230.0,
            "heat_valve_close_pressure": 12.0,
            "heat_valve_close_time": "12:15",
            "low_tf_temp_duration": 280.0,
            "cooking_start_time": "08:50",
            "cooking_stop_time": "12:50",
            "cooking_duration": 240.0,
            "cooling_valve_open_temp": 215.0,
            "cooling_valve_open_pressure": 8.0,
            "cooling_valve_open_time": "12:55",
            "bottom_door_open_temp": 135.0,
            "bottom_door_open_pressure": 0.3,
            "bottom_door_open_time": "13:50",
            "standard_batch_duration": 400.0,
            "actual_batch_duration": 440.0,
            "batch_end_time": "13:55"
        },
        "non_golden_sample": {
            "batch_id": "BT-040310-FAIL",
            "batch_start_time": "02:00",
            "steam_valve_open_temp": 75.0,
            "steam_valve_open_pressure": 0.0,
            "steam_valve_open_time": "03:15",
            "steam_valve_close_temp": 140.0,
            "steam_valve_close_pressure": 5.0,
            "steam_valve_close_time": "05:30",
            "heat_valve_close_temp": 205.0,
            "heat_valve_close_pressure": 9.0,
            "heat_valve_close_time": "05:40",
            "low_tf_temp_duration": 150.0,
            "cooking_start_time": "03:20",
            "cooking_stop_time": "06:10",
            "cooking_duration": 170.0,
            "cooling_valve_open_temp": 185.0,
            "cooling_valve_open_pressure": 5.5,
            "cooling_valve_open_time": "06:15",
            "bottom_door_open_temp": 160.0,
            "bottom_door_open_pressure": 1.2,
            "bottom_door_open_time": "06:45",
            "standard_batch_duration": 400.0,
            "actual_batch_duration": 310.0,
            "batch_end_time": "06:50"
        }
    }
    
    # Merge holdout test batches if available
    holdout_path = os.path.join(CURRENT_DIR, "data", "holdout_test_batches.json")
    if os.path.exists(holdout_path):
        try:
            with open(holdout_path, "r", encoding="utf-8") as f:
                res["holdout_test_batches"] = json.load(f)
        except Exception:
            pass
            
    return res



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
