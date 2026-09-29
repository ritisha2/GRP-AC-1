"""
ADVaIT Feature Engineering Engine
Calculates ~45 industrial process features categorized into:
- Category A: Time Features (12)
- Category B: Temperature Features (7)
- Category C: Pressure Features (6)
- Category D: Rate Features (4)
- Category E: Process Interaction Features (4)
- Missingness Indicators & Industrial Gradient Indices (12)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import os
import sys

# Ensure ADVaIT root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from backend.preprocessing import parse_time_to_minutes, calculate_time_delta_minutes, parse_duration_to_minutes
except ImportError:
    from preprocessing import parse_time_to_minutes, calculate_time_delta_minutes, parse_duration_to_minutes


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    """Safe division handling zero or nan denominators."""
    if pd.isna(num) or pd.isna(den) or abs(den) < 1e-7:
        return default
    return float(num / den)

def parse_num(val, default: float = 0.0) -> float:
    """Safely parse float numbers."""
    if pd.isna(val) or val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

def engineer_batch_features(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Given a dictionary containing the 23 production inputs,
    computes approximately 45 industrial process features.
    """
    # Parse base timestamps into minutes since midnight
    t_batch_start = parse_time_to_minutes(data.get("batch_start_time"))
    t_steam_open = parse_time_to_minutes(data.get("steam_valve_open_time"))
    t_steam_close = parse_time_to_minutes(data.get("steam_valve_close_time"))
    t_heat_close = parse_time_to_minutes(data.get("heat_valve_close_time"))
    t_cooking_start = parse_time_to_minutes(data.get("cooking_start_time"))
    t_cooking_stop = parse_time_to_minutes(data.get("cooking_stop_time"))
    t_cooling_open = parse_time_to_minutes(data.get("cooling_valve_open_time"))
    t_bottom_door = parse_time_to_minutes(data.get("bottom_door_open_time"))
    t_batch_end = parse_time_to_minutes(data.get("batch_end_time"))

    # Temperatures
    t_steam_open_temp = parse_num(data.get("steam_valve_open_temp"))
    t_steam_close_temp = parse_num(data.get("steam_valve_close_temp"))
    t_heat_close_temp = parse_num(data.get("heat_valve_close_temp"))
    t_cooling_open_temp = parse_num(data.get("cooling_valve_open_temp"))
    t_bottom_door_temp = parse_num(data.get("bottom_door_open_temp"))

    # Pressures
    p_steam_open = parse_num(data.get("steam_valve_open_pressure"))
    p_steam_close = parse_num(data.get("steam_valve_close_pressure"))
    p_heat_close = parse_num(data.get("heat_valve_close_pressure"))
    p_cooling_open = parse_num(data.get("cooling_valve_open_pressure"))
    p_bottom_door = parse_num(data.get("bottom_door_open_pressure"))

    # Durations from inputs
    raw_cooking_dur = parse_duration_to_minutes(data.get("cooking_duration"))
    raw_low_tf = parse_duration_to_minutes(data.get("low_tf_temp_duration"))
    raw_std_dur = parse_duration_to_minutes(data.get("standard_batch_duration"))
    raw_actual_dur = parse_duration_to_minutes(data.get("actual_batch_duration"))


    features = {}

    # ==========================================
    # Category A — Time Features (12 features)
    # ==========================================
    # 1. Steam Open Delay: Steam Open − Batch Start
    features["steam_open_delay"] = calculate_time_delta_minutes(t_batch_start, t_steam_open)
    # 2. Steam Duration: Steam Close − Steam Open
    features["steam_duration"] = calculate_time_delta_minutes(t_steam_open, t_steam_close)
    # 3. Cooking Start Delay: Cooking Start − Batch Start
    features["cooking_start_delay"] = calculate_time_delta_minutes(t_batch_start, t_cooking_start)
    # 4. Cooking Duration Verified: Cooking Stop − Cooking Start
    features["cooking_duration_verified"] = calculate_time_delta_minutes(t_cooking_start, t_cooking_stop)
    # Fallback to reported duration if verified is 0/nan
    if pd.isna(features["cooking_duration_verified"]) or features["cooking_duration_verified"] <= 0:
        features["cooking_duration_verified"] = raw_cooking_dur

    # 5. Cooling Delay: Cooling Open − Cooking Stop
    features["cooling_delay"] = calculate_time_delta_minutes(t_cooking_stop, t_cooling_open)
    # 6. Unloading Delay: Bottom Door Open − Cooling Open
    features["unloading_delay"] = calculate_time_delta_minutes(t_cooling_open, t_bottom_door)
    # 7. Total Batch Duration: Batch End − Batch Start
    total_calc_dur = calculate_time_delta_minutes(t_batch_start, t_batch_end)
    features["total_batch_duration"] = total_calc_dur if not pd.isna(total_calc_dur) and total_calc_dur > 0 else raw_actual_dur
    # 8. Batch Duration Deviation: Actual − Standard
    features["batch_duration_deviation"] = raw_actual_dur - raw_std_dur

    tot_dur = features["total_batch_duration"] if features["total_batch_duration"] > 0 else 360.0
    # 9. Steam Ratio: Steam Duration / Total Batch
    features["steam_ratio"] = safe_div(features["steam_duration"], tot_dur)
    # 10. Cooking Ratio: Cooking Duration / Total Batch
    features["cooking_ratio"] = safe_div(features["cooking_duration_verified"], tot_dur)
    # 11. Cooling Ratio: Cooling Duration / Total Batch (proxied by cooling delay)
    features["cooling_ratio"] = safe_div(features["cooling_delay"], tot_dur)
    # 12. Unloading Ratio: Unloading Delay / Total Batch
    features["unloading_ratio"] = safe_div(features["unloading_delay"], tot_dur)

    # ==========================================
    # Category B — Temperature Features (7 features)
    # ==========================================
    # 13. Steam Temperature Rise: Steam Close Temp − Steam Open Temp
    features["steam_temp_rise"] = t_steam_close_temp - t_steam_open_temp
    # 14. Steam-to-Heat Difference: Steam Close − Heat Close
    features["steam_to_heat_diff"] = t_steam_close_temp - t_heat_close_temp
    # 15. Heat-to-Cooling Drop: Heat Close − Cooling
    features["heat_to_cooling_drop"] = t_heat_close_temp - t_cooling_open_temp
    # 16. Cooling-to-Bottom Drop: Cooling − Bottom Door
    features["cooling_to_bottom_drop"] = t_cooling_open_temp - t_bottom_door_temp
    # 17. Total Temperature Drop: Steam Close − Bottom Door
    features["total_temp_drop"] = t_steam_close_temp - t_bottom_door_temp
    # 18. Cooling Efficiency: (Heat Close − Cooling) / Cooling Delay
    features["cooling_efficiency"] = safe_div(features["heat_to_cooling_drop"], max(features["cooling_delay"], 1.0))
    # 19. Normalized Temperature Ratio: Cooling / Steam Close
    features["normalized_temp_ratio"] = safe_div(t_cooling_open_temp, max(t_steam_close_temp, 1.0))

    # ==========================================
    # Category C — Pressure Features (6 features)
    # ==========================================
    # 20. Steam Pressure Rise: Steam Close − Steam Open
    features["steam_pressure_rise"] = p_steam_close - p_steam_open
    # 21. Steam-to-Heat Pressure Change: Steam Close − Heat Close
    features["steam_to_heat_press_change"] = p_steam_close - p_heat_close
    # 22. Heat-to-Cooling Pressure Drop: Heat Close − Cooling
    features["heat_to_cooling_press_drop"] = p_heat_close - p_cooling_open
    # 23. Cooling-to-Bottom Pressure Drop: Cooling − Bottom Door
    features["cooling_to_bottom_press_drop"] = p_cooling_open - p_bottom_door
    # 24. Overall Pressure Reduction: Steam Close − Bottom Door
    features["overall_press_reduction"] = p_steam_close - p_bottom_door
    # 25. Pressure Stability: Derived from variance of stage pressure transitions
    press_steps = [features["steam_pressure_rise"], features["steam_to_heat_press_change"], features["heat_to_cooling_press_drop"]]
    features["pressure_stability"] = float(np.std(press_steps)) if len(press_steps) > 0 else 0.0

    # ==========================================
    # Category D — Rate Features (4 features)
    # ==========================================
    # 26. Heating Rate: Temp Rise / Steam Duration
    features["heating_rate"] = safe_div(features["steam_temp_rise"], max(features["steam_duration"], 1.0))
    # 27. Pressure Increase Rate: Pressure Rise / Steam Duration
    features["pressure_increase_rate"] = safe_div(features["steam_pressure_rise"], max(features["steam_duration"], 1.0))
    # 28. Cooling Rate: Temp Drop / Cooling Delay
    features["cooling_rate"] = safe_div(features["heat_to_cooling_drop"], max(features["cooling_delay"], 1.0))
    # 29. Depressurization Rate: Pressure Drop / Cooling Delay
    features["depressurization_rate"] = safe_div(features["heat_to_cooling_press_drop"], max(features["cooling_delay"], 1.0))

    # ==========================================
    # Category E — Process Interaction Features (4 features)
    # ==========================================
    # 30. Low TF Ratio: Low TF Duration / Cooking Duration
    features["low_tf_ratio"] = safe_div(raw_low_tf, max(features["cooking_duration_verified"], 1.0))
    # 31. Steam Efficiency: Temp Rise / Steam Duration (matches heating rate)
    features["steam_efficiency"] = safe_div(features["steam_temp_rise"], max(features["steam_duration"], 1.0))
    # 32. Thermal Exposure Score: Steam Close Temp × Cooking Duration
    features["thermal_exposure_score"] = float(t_steam_close_temp * features["cooking_duration_verified"])
    # 33. Process Sync Delay: Cooking Start − Steam Close
    features["process_sync_delay"] = calculate_time_delta_minutes(t_steam_close, t_cooking_start)

    # ==========================================
    # Missingness Indicators & Auxiliary Industrial Indices (12 features)
    # ==========================================
    features["steam_open_temp_missing"] = 1.0 if data.get("steam_valve_open_temp") is None or pd.isna(data.get("steam_valve_open_temp")) else 0.0
    features["steam_open_press_missing"] = 1.0 if data.get("steam_valve_open_pressure") is None or pd.isna(data.get("steam_valve_open_pressure")) else 0.0
    features["steam_close_temp_missing"] = 1.0 if data.get("steam_valve_close_temp") is None or pd.isna(data.get("steam_valve_close_temp")) else 0.0
    features["heat_close_temp_missing"] = 1.0 if data.get("heat_valve_close_temp") is None or pd.isna(data.get("heat_valve_close_temp")) else 0.0
    features["cooling_open_temp_missing"] = 1.0 if data.get("cooling_valve_open_temp") is None or pd.isna(data.get("cooling_valve_open_temp")) else 0.0
    features["bottom_door_temp_missing"] = 1.0 if data.get("bottom_door_open_temp") is None or pd.isna(data.get("bottom_door_open_temp")) else 0.0
    features["low_tf_duration_missing"] = 1.0 if data.get("low_tf_temp_duration") is None or pd.isna(data.get("low_tf_temp_duration")) else 0.0

    # Interaction & Gradient indices
    features["thermal_inertia_index"] = safe_div(features["steam_temp_rise"], abs(features["steam_pressure_rise"]) + 0.1)
    features["cooking_cooling_temp_ratio"] = safe_div(t_heat_close_temp, max(t_cooling_open_temp, 1.0))
    features["overall_thermal_gradient"] = safe_div(features["total_temp_drop"], max(tot_dur, 1.0))
    features["pressure_gradient_ratio"] = safe_div(features["overall_press_reduction"], max(tot_dur, 1.0))
    features["effective_curing_index"] = features["cooking_duration_verified"] * safe_div(t_heat_close_temp, 200.0)

    # Clean up any potential NaNs or Infs
    for k, v in features.items():
        if pd.isna(v) or np.isinf(v):
            features[k] = 0.0
        else:
            features[k] = round(float(v), 4)

    return features

def engineer_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized/batched feature engineering for a historical production DataFrame.
    Maps raw column names from production_data.csv to canonical fields.
    """
    records = []
    
    # Mapping dict for typical raw column headers in production_data.csv
    for _, row in df.iterrows():
        row_dict = {}
        for col, val in row.items():
            col_str = str(col).lower()
            if "start time" in col_str and "batch" in col_str:
                row_dict["batch_start_time"] = val
            elif "steam valve open temp" in col_str or ("steam" in col_str and "open" in col_str and "temp" in col_str):
                row_dict["steam_valve_open_temp"] = val
            elif "steam valve open pressure" in col_str or ("steam" in col_str and "open" in col_str and "pressure" in col_str):
                row_dict["steam_valve_open_pressure"] = val
            elif "steam valve open time" in col_str or ("steam" in col_str and "open" in col_str and "time" in col_str):
                row_dict["steam_valve_open_time"] = val
            elif "steam valve close temp" in col_str or ("steam" in col_str and "close" in col_str and "temp" in col_str):
                row_dict["steam_valve_close_temp"] = val
            elif "steam valve close pressure" in col_str or ("steam" in col_str and "close" in col_str and "pressure" in col_str):
                row_dict["steam_valve_close_pressure"] = val
            elif "steam valve close time" in col_str or ("steam" in col_str and "close" in col_str and "time" in col_str):
                row_dict["steam_valve_close_time"] = val
            elif "heat valve close temp" in col_str or ("heat" in col_str and "close" in col_str and "temp" in col_str):
                row_dict["heat_valve_close_temp"] = val
            elif "heat valve close pressure" in col_str or ("heat" in col_str and "close" in col_str and "pressure" in col_str):
                row_dict["heat_valve_close_pressure"] = val
            elif "heat valve close time" in col_str or ("heat" in col_str and "close" in col_str and "time" in col_str):
                row_dict["heat_valve_close_time"] = val
            elif "low tf" in col_str:
                row_dict["low_tf_temp_duration"] = val
            elif "cooking start time" in col_str:
                row_dict["cooking_start_time"] = val
            elif "cooking stop time" in col_str:
                row_dict["cooking_stop_time"] = val
            elif "cooking duration" in col_str:
                row_dict["cooking_duration"] = val
            elif "cooling valve open temp" in col_str:
                row_dict["cooling_valve_open_temp"] = val
            elif "cooling valve open pressure" in col_str:
                row_dict["cooling_valve_open_pressure"] = val
            elif "cooling valve open time" in col_str:
                row_dict["cooling_valve_open_time"] = val
            elif "bottom door open temp" in col_str:
                row_dict["bottom_door_open_temp"] = val
            elif "bottom door open pressure" in col_str:
                row_dict["bottom_door_open_pressure"] = val
            elif "bottom door open time" in col_str:
                row_dict["bottom_door_open_time"] = val
            elif "standard batch duration" in col_str:
                row_dict["standard_batch_duration"] = val
            elif "actual batch duration" in col_str:
                row_dict["actual_batch_duration"] = val
            elif "batch end time" in col_str:
                row_dict["batch_end_time"] = val
            elif "batch id" in col_str:
                row_dict["batch_id"] = val

        feats = engineer_batch_features(row_dict)
        feats["batch_id"] = str(row_dict.get("batch_id", ""))
        records.append(feats)
        
    feat_df = pd.DataFrame(records)
    return feat_df

if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "data", "production_data.csv")
    if os.path.exists(data_path):
        df_prod = pd.read_csv(data_path)
        feats = engineer_features_dataframe(df_prod)
        print(f"Engineered {feats.shape[1] - 1} features across {len(feats)} batches.")
        print("Feature columns:", [c for c in feats.columns if c != 'batch_id'])
