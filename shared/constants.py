"""
Project Constants
Process definitions, quality target limits, 23 production inputs,
and Golden Batch classification rules.
"""

PRODUCTION_INPUT_FIELDS = [
    "batch_id",
    "batch_start_time",
    "steam_valve_open_temp",
    "steam_valve_open_pressure",
    "steam_valve_open_time",
    "steam_valve_close_temp",
    "steam_valve_close_pressure",
    "steam_valve_close_time",
    "heat_valve_close_temp",
    "heat_valve_close_pressure",
    "heat_valve_close_time",
    "low_tf_temp_duration",
    "cooking_start_time",
    "cooking_stop_time",
    "cooking_duration",
    "cooling_valve_open_temp",
    "cooling_valve_open_pressure",
    "cooling_valve_open_time",
    "bottom_door_open_temp",
    "bottom_door_open_pressure",
    "bottom_door_open_time",
    "standard_batch_duration",
    "actual_batch_duration",
    "batch_end_time",
]

TARGET_QUALITY_PARAMETERS = [
    "AC Mooney",
    "Ash%",
    "CB%",
    "AE%",
    "VM%",
    "RHC",
    "Specific Gravity",
    "Mv",
    "TS",
    "EB",
    "Hardness",
]

# Mapping between internal target names and raw dataset column variants
QUALITY_COL_MAPPING = {
    "Hardness": ["Hardness"],
    "EB": ["EB"],
    "TS": ["TS"],
    "Mv": ["Mv"],
    "Specific Gravity": ["Sp.Gravity", "Sp Gravity", "Specific Gravity"],
    "RHC": ["RHC"],
    "VM%": ["V.M.%", "VM%", "V.M. %"],
    "AE%": ["A.E.%", "AE%", "A.E. %"],
    "CB%": ["C.B.%", "CB%", "C.B. %"],
    "Ash%": ["Ash%", "Ash %"],
    "AC Mooney": ["AC Mooney", "AC_Mooney"],
}

# Standard Industrial Specification Ranges for BT Rubber (from plant quality standard sheet)
QUALITY_SPECIFICATIONS = {
    "AC Mooney": {"lsl": 50.0, "target": 55.0, "usl": 60.0, "unit": "MU", "spec_str": "50 - 60"},
    "Ash%": {"lsl": 3.00, "target": 5.00, "usl": 7.00, "unit": "%", "spec_str": "3.00 - 7.00"},
    "CB%": {"lsl": 28.00, "target": 32.00, "usl": 36.00, "unit": "%", "spec_str": "28.00 - 36.00"},
    "AE%": {"lsl": 6.00, "target": 9.00, "usl": 12.00, "unit": "%", "spec_str": "6.00 - 12.00"},
    "VM%": {"lsl": 0.00, "target": 0.20, "usl": 1.00, "unit": "%", "spec_str": "1 Max."},
    "RHC": {"lsl": 50.0, "target": 56.0, "usl": 100.0, "unit": "%", "spec_str": "50 Min"},
    "Specific Gravity": {"lsl": 1.12, "target": 1.14, "usl": 1.16, "unit": "g/cc", "spec_str": "1.14 ± 0.02"},
    "Mv": {"lsl": 30.0, "target": 37.5, "usl": 45.0, "unit": "MU", "spec_str": "30 - 45"},
    "TS": {"lsl": 75.0, "target": 88.0, "usl": 150.0, "unit": "kg/cm2", "spec_str": "75 Min."},
    "EB": {"lsl": 480.0, "target": 520.0, "usl": 750.0, "unit": "%", "spec_str": "480 Min."},
    "Hardness": {"lsl": 48.0, "target": 51.0, "usl": 54.0, "unit": "Shore A", "spec_str": "51 ± 3"},
}

# Golden Batch Classification Score Thresholds
GOLDEN_THRESHOLD = 90.0
WARNING_THRESHOLD = 75.0

STATUS_GOLDEN = "Golden"
STATUS_WARNING = "Warning"
STATUS_NON_GOLDEN = "Non-Golden"

# Key features used for radar and high-level live process intelligence
RADAR_FEATURES = [
    "steam_duration",
    "heating_rate",
    "steam_temp_rise",
    "steam_pressure_rise",
    "cooking_ratio",
    "cooling_rate",
]
