"""
ADVaIT Prediction Engine
Orchestrates the complete batch inference pipeline:
Input Validation -> Time Normalization -> Feature Engineering (~45 features) ->
11 ML Model Inferences -> Golden Batch Scoring (0-100%) -> SHAP Explainability -> Timeline
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.constants import (
    TARGET_QUALITY_PARAMETERS, QUALITY_SPECIFICATIONS,
    STATUS_GOLDEN, STATUS_WARNING, STATUS_NON_GOLDEN
)
from backend.preprocessing import (
    validate_production_input_fastapi,
    parse_time_to_minutes,
    calculate_time_delta_minutes
)
from backend.feature_engineering import engineer_batch_features
from backend.golden_batch import evaluate_batch
from backend.explainability import explain_batch_prediction
from backend.train import target_to_slug, MODELS_DIR

_MODELS_CACHE = {}
_FEATURE_NAMES = None
HISTORY_FILE = os.path.join(CURRENT_DIR, "data", "prediction_history.json")

def save_prediction_to_history(result: Dict[str, Any], batch_input: Dict[str, Any]):
    """Appends every prediction run automatically to prediction_history.json."""
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        history = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
        record = {
            "timestamp": datetime.now().isoformat(),
            "batch_id": result.get("batch_id"),
            "golden_score": result.get("golden_score"),
            "status": result.get("status"),
            "predictions": {k: v.get("predicted_value") for k, v in result.get("predictions", {}).items()},
            "input_summary": batch_input
        }
        history.append(record)
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[Warning] Could not save prediction history: {e}")

def get_feature_names() -> List[str]:
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        names_path = os.path.join(MODELS_DIR, "feature_names.json")
        if os.path.exists(names_path):
            with open(names_path, "r", encoding="utf-8") as f:
                _FEATURE_NAMES = json.load(f)
        else:
            _FEATURE_NAMES = []
    return _FEATURE_NAMES

def load_target_models() -> Dict[str, Any]:
    global _MODELS_CACHE
    if len(_MODELS_CACHE) >= len(TARGET_QUALITY_PARAMETERS):
        return _MODELS_CACHE
        
    for target in TARGET_QUALITY_PARAMETERS:
        slug = target_to_slug(target)
        model_file = os.path.join(MODELS_DIR, f"{slug}.joblib")
        if os.path.exists(model_file):
            try:
                _MODELS_CACHE[target] = joblib.load(model_file)
            except Exception as e:
                print(f"[Warning] Could not load model for {target}: {e}")
                
    return _MODELS_CACHE

def construct_timeline(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constructs process milestone events with elapsed minutes from Batch Start.
    """
    t_start = parse_time_to_minutes(data.get("batch_start_time", "13:00"))
    
    events_raw = [
        ("Start", data.get("batch_start_time", "13:00"), "#58a6ff"),
        ("Steam Open", data.get("steam_valve_open_time", "13:20"), "#d29922"),
        ("Steam Close", data.get("steam_valve_close_time", "16:40"), "#f0883e"),
        ("Cooking", data.get("cooking_start_time", "13:20"), "#a371f7"),
        ("Cooling", data.get("cooling_valve_open_time", "17:25"), "#388bfd"),
        ("Bottom Door", data.get("bottom_door_open_time", "18:05"), "#3fb950"),
        ("End", data.get("batch_end_time", "18:05"), "#2ea043"),
    ]
    
    timeline = []
    for label, raw_time, color in events_raw:
        m = parse_time_to_minutes(raw_time)
        elapsed = calculate_time_delta_minutes(t_start, m)
        if pd.isna(elapsed):
            elapsed = 0.0
        timeline.append({
            "stage": label,
            "clock_time": str(raw_time),
            "elapsed_minutes": round(elapsed, 1),
            "color": color
        })
        
    # Sort events by elapsed minutes
    timeline.sort(key=lambda x: x["elapsed_minutes"])
    return timeline

def predict_batch(batch_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete prediction pipeline for an industrial BT Rubber batch.
    """
    # 1. Validation
    validate_production_input_fastapi(batch_input)
    
    # 2. Feature Engineering (~45 industrial features)
    engineered_features = engineer_batch_features(batch_input)
    
    # 3. Model Inference across 11 Quality Parameters
    models = load_target_models()
    feature_names = get_feature_names()
    if not feature_names:
        feature_names = list(engineered_features.keys())
        
    row_df = pd.DataFrame([{fn: engineered_features.get(fn, 0.0) for fn in feature_names}])
    
    predictions = {}
    predicted_values_dict = {}
    
    for target in TARGET_QUALITY_PARAMETERS:
        model = models.get(target)
        spec = QUALITY_SPECIFICATIONS.get(target, {"lsl": 0, "target": 50, "usl": 100, "unit": ""})
        lsl, targ, usl = spec["lsl"], spec["target"], spec["usl"]
        
        if model is not None:
            try:
                pred_val = float(model.predict(row_df)[0])
            except Exception:
                try:
                    pred_val = float(model.predict(row_df.values)[0])
                except Exception:
                    pred_val = float(targ)
        else:
            # Fallback if model training incomplete
            pred_val = float(targ)

            
        pred_val = round(pred_val, 2)
        predicted_values_dict[target] = pred_val
        
        # Determine status
        if lsl <= pred_val <= usl:
            status = "Within range"
            score = 100.0
        elif pred_val < lsl:
            status = "Below specification"
            score = max(20.0, 100.0 - ((lsl - pred_val) / max(usl - lsl, 1e-4)) * 100.0)
        else:
            status = "Above specification"
            score = max(20.0, 100.0 - ((pred_val - usl) / max(usl - lsl, 1e-4)) * 100.0)
            
        predictions[target] = {
            "parameter": target,
            "predicted_value": pred_val,
            "lsl": lsl,
            "target": targ,
            "usl": usl,
            "unit": spec.get("unit", ""),
            "status": status,
            "score": round(score, 1)
        }
        
    # 4. Golden Batch Scoring Engine (0-100%)
    golden_eval = evaluate_batch(engineered_features, predicted_values_dict)
    
    # 5. Timeline construction
    timeline = construct_timeline(batch_input)
    
    # 6. SHAP Explainability for all 11 targets
    explainability = {}
    for target in TARGET_QUALITY_PARAMETERS:
        model = models.get(target)
        exp = explain_batch_prediction(target, model, engineered_features)
        explainability[target] = exp
        
    res = {
        "batch_id": str(batch_input.get("batch_id", "BT-UNKNOWN")),
        "golden_score": golden_eval["golden_score"],
        "status": golden_eval["status"],
        "golden_evaluation": golden_eval,
        "predictions": predictions,
        "engineered_features": engineered_features,
        "timeline_events": timeline,
        "explainability": explainability
    }
    
    # Save to persistent history log
    save_prediction_to_history(res, batch_input)
    
    return res
