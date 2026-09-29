"""
ADVaIT Preprocessing Engine
Handles timestamp normalization (with midnight crossing), missingness tracking,
and physical industrial batch validation.
"""

from datetime import datetime, time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from fastapi import HTTPException

def parse_time_to_minutes(t_val) -> float:
    """
    Parses a time representation (datetime, time, HH:MM string, or float/int)
    into minutes since midnight (0 to 1439.99).
    """
    if pd.isna(t_val) or t_val is None or t_val == "" or str(t_val).strip().lower() in ["nan", "none", "null", "not recorded", "not applicable", "na"]:
        return np.nan
        
    if isinstance(t_val, (datetime, pd.Timestamp)):
        return t_val.hour * 60.0 + t_val.minute + t_val.second / 60.0
    if isinstance(t_val, time):
        return t_val.hour * 60.0 + t_val.minute + t_val.second / 60.0
        
    t_str = str(t_val).strip()
    # If string contains full timestamp e.g. "2026-03-07 18:40:55"
    if " " in t_str:
        t_str = t_str.split(" ")[-1]
        
    parts = t_str.split(":")
    try:
        if len(parts) >= 2:
            hours = float(parts[0])
            mins = float(parts[1])
            secs = float(parts[2]) if len(parts) > 2 else 0.0
            return (hours * 60.0 + mins + secs / 60.0) % 1440.0
        else:
            val = float(t_str)
            # If it looks like hours (e.g. 13.5)
            if 0 <= val <= 24:
                return (val * 60.0) % 1440.0
            return val % 1440.0
    except Exception:
        return np.nan

def parse_duration_to_minutes(d_val) -> float:
    """
    Parses a duration that could be numeric minutes (e.g. 235.0)
    or time string (e.g. '03:55:00', '04:00', '03:55') into total minutes as a float.
    """
    if pd.isna(d_val) or d_val is None or str(d_val).strip().lower() in ["", "nan", "none", "null"]:
        return 0.0
    if isinstance(d_val, (int, float)):
        return float(d_val)
    if isinstance(d_val, time):
        return float(d_val.hour * 60 + d_val.minute + d_val.second / 60.0)
    d_str = str(d_val).strip()
    if ":" in d_str:
        parts = d_str.split(":")
        try:
            hrs = float(parts[0])
            mins = float(parts[1]) if len(parts) > 1 else 0.0
            secs = float(parts[2]) if len(parts) > 2 else 0.0
            return hrs * 60.0 + mins + secs / 60.0
        except Exception:
            return 0.0
    try:
        return float(d_str)
    except Exception:
        return 0.0

def calculate_time_delta_minutes(start_mins: float, end_mins: float) -> float:

    """
    Calculates duration in minutes between start and end times,
    robustly handling midnight crossing (e.g. 23:30 to 01:15 is +105 mins).
    """
    if pd.isna(start_mins) or pd.isna(end_mins):
        return np.nan
        
    delta = end_mins - start_mins
    if delta < 0:
        # Midnight crossing occurred
        delta += 1440.0
    return delta

def validate_production_dict(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates physical constraints on batch inputs:
    - Pressures must be non-negative
    - Durations must be non-negative
    - Chronological consistency: steam close >= steam open, cooking stop >= cooking start
    """
    # Check pressures
    pressure_fields = [
        "steam_valve_open_pressure",
        "steam_valve_close_pressure",
        "heat_valve_close_pressure",
        "cooling_valve_open_pressure",
        "bottom_door_open_pressure"
    ]
    for p_field in pressure_fields:
        val = data.get(p_field)
        if val is not None and not pd.isna(val):
            try:
                fval = float(val)
                if fval < -0.01:
                    return False, f"Invalid pressure on '{p_field}': {fval} kg/cm2. Pressure cannot be negative."
            except ValueError:
                return False, f"Invalid numeric format for '{p_field}': {val}"
                
    # Check durations
    duration_fields = [
        "low_tf_temp_duration",
        "cooking_duration",
        "standard_batch_duration",
        "actual_batch_duration"
    ]
    for d_field in duration_fields:
        val = data.get(d_field)
        if val is not None and not pd.isna(val):
            try:
                fval = float(val)
                if fval < -0.01:
                    return False, f"Invalid duration on '{d_field}': {fval} mins. Duration cannot be negative."
            except ValueError:
                return False, f"Invalid numeric format for '{d_field}': {val}"
                
    # Check chronological sanity
    steam_open_mins = parse_time_to_minutes(data.get("steam_valve_open_time"))
    steam_close_mins = parse_time_to_minutes(data.get("steam_valve_close_time"))
    if not pd.isna(steam_open_mins) and not pd.isna(steam_close_mins):
        delta = calculate_time_delta_minutes(steam_open_mins, steam_close_mins)
        if delta < 0 or delta > 1200:
            return False, f"Steam Close Time ({data.get('steam_valve_close_time')}) cannot precede Steam Open Time ({data.get('steam_valve_open_time')})."

    cooking_start_mins = parse_time_to_minutes(data.get("cooking_start_time"))
    cooking_stop_mins = parse_time_to_minutes(data.get("cooking_stop_time"))
    if not pd.isna(cooking_start_mins) and not pd.isna(cooking_stop_mins):
        delta = calculate_time_delta_minutes(cooking_start_mins, cooking_stop_mins)
        if delta < 0 or delta > 1200:
            return False, f"Cooking Stop Time ({data.get('cooking_stop_time')}) cannot precede Cooking Start Time ({data.get('cooking_start_time')})."

    return True, ""

def validate_production_input_fastapi(data: Dict[str, Any]):
    """FastAPI helper that raises HTTPException(422) if validation fails."""
    is_valid, err_msg = validate_production_dict(data)
    if not is_valid:
        raise HTTPException(status_code=422, detail={"error": "Industrial Process Validation Error", "message": err_msg})
