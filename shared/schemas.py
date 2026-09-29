"""
Data Schemas
Pydantic schemas for batch inputs, predictions, Golden Batch scoring, and explainability.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class BatchProductionInput(BaseModel):
    batch_id: str = Field(..., description="Batch Identifier, e.g. BT-040624-0", examples=["BT-040624-0"])
    batch_start_time: str = Field(..., description="Start time (HH:MM or HH:MM:SS)", examples=["11:30"])
    
    # Steam Stage
    steam_valve_open_temp: float = Field(..., description="Steam Valve Open Temp in deg C", ge=-50.0, le=350.0, examples=[100.0])
    steam_valve_open_pressure: float = Field(..., description="Steam Valve Open Pressure in kg/cm2", ge=0.0, le=50.0, examples=[0.0])
    steam_valve_open_time: str = Field(..., description="Steam Valve Open Time (HH:MM)", examples=["13:20"])
    steam_valve_close_temp: float = Field(..., description="Steam Valve Close Temp in deg C", ge=-50.0, le=350.0, examples=[170.0])
    steam_valve_close_pressure: float = Field(..., description="Steam Valve Close Pressure in kg/cm2", ge=0.0, le=50.0, examples=[8.5])
    steam_valve_close_time: str = Field(..., description="Steam Valve Close Time (HH:MM)", examples=["16:40"])
    
    # Heat Stage
    heat_valve_close_temp: float = Field(..., description="Heat Valve Close Temp in deg C", ge=-50.0, le=350.0, examples=[240.0])
    heat_valve_close_pressure: float = Field(..., description="Heat Valve Close Pressure in kg/cm2", ge=0.0, le=50.0, examples=[14.0])
    heat_valve_close_time: str = Field(..., description="Heat Valve Close Time (HH:MM)", examples=["16:45"])
    
    # Cooking Stage
    low_tf_temp_duration: float = Field(..., description="Low TF Temp Duration in minutes (below 260 deg C)", ge=0.0, le=1440.0, examples=[340.0])
    cooking_start_time: str = Field(..., description="Cooking Start Time (HH:MM)", examples=["13:20"])
    cooking_stop_time: str = Field(..., description="Cooking Stop Time (HH:MM)", examples=["17:25"])
    cooking_duration: float = Field(..., description="Cooking Duration in minutes or hours", ge=0.0, le=1440.0, examples=[245.0])
    
    # Cooling Stage
    cooling_valve_open_temp: float = Field(..., description="Cooling Valve Open Temp in deg C", ge=-50.0, le=350.0, examples=[230.0])
    cooling_valve_open_pressure: float = Field(..., description="Cooling Valve Open Pressure in kg/cm2", ge=0.0, le=50.0, examples=[9.5])
    cooling_valve_open_time: str = Field(..., description="Cooling Valve Open Time (HH:MM)", examples=["17:25"])
    
    # Bottom Door
    bottom_door_open_temp: float = Field(..., description="Bottom Door Open Temp in deg C", ge=-50.0, le=350.0, examples=[120.0])
    bottom_door_open_pressure: float = Field(..., description="Bottom Door Open Pressure in kg/cm2", ge=0.0, le=50.0, examples=[0.0])
    bottom_door_open_time: str = Field(..., description="Bottom Door Open Time (HH:MM)", examples=["18:05"])
    
    # Batch Durations
    standard_batch_duration: float = Field(..., description="Standard Batch Duration in hours (e.g., 5.0 Hrs)", ge=0.0, le=2000.0, examples=[5.0])
    actual_batch_duration: float = Field(..., description="Actual Batch Duration in minutes", ge=0.0, le=2000.0, examples=[395.0])
    batch_end_time: str = Field(..., description="Batch End Time (HH:MM)", examples=["18:05"])

class QualityPredictionItem(BaseModel):
    parameter: str
    predicted_value: float
    lsl: float
    target: float
    usl: float
    unit: str
    status: str  # Within range, Warning High, Warning Low
    score: float

class GoldenBatchScoreResponse(BaseModel):
    golden_score: float
    status: str  # Golden, Warning, Non-Golden
    process_score: float
    quality_score: float
    penalties: List[Dict[str, Any]]
    radar_comparison: Dict[str, Dict[str, float]]

class SHAPFeatureImpact(BaseModel):
    feature: str
    impact: float
    impact_pct: float
    feature_value: float

class TargetExplainability(BaseModel):
    target: str
    base_value: float
    prediction: float
    top_drivers: List[SHAPFeatureImpact]

class BatchPredictionResponse(BaseModel):
    batch_id: str
    golden_score: float
    status: str
    golden_evaluation: GoldenBatchScoreResponse
    predictions: Dict[str, QualityPredictionItem]
    engineered_features: Dict[str, float]
    timeline_events: List[Dict[str, Any]]
    explainability: Dict[str, TargetExplainability]

class DatasetReportSummary(BaseModel):
    total_rows: int
    total_columns: int
    missing_value_pct: float
    zero_value_pct: float
    duplicate_batches: int
    feature_importances: Dict[str, Dict[str, float]]
    predictability_scores: Dict[str, float]
    html_report_path: Optional[str] = None
    pdf_report_path: Optional[str] = None
