"""
End-to-End Pipeline Automated Test Suite
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from shared.constants import TARGET_QUALITY_PARAMETERS, PRODUCTION_INPUT_FIELDS
from backend.preprocessing import validate_production_dict, parse_time_to_minutes, calculate_time_delta_minutes
from backend.feature_engineering import engineer_batch_features
from backend.predict import predict_batch
from backend.golden_batch import load_golden_profile
from backend.reports_generator import generate_pdf_batch_report, generate_excel_batch_report, generate_csv_batch_report

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_batch = {
            "batch_id": "TEST-001",
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
        }

    def test_time_normalization_and_midnight_crossing(self):
        m1 = parse_time_to_minutes("23:45")
        m2 = parse_time_to_minutes("00:30")
        self.assertEqual(m1, 23 * 60 + 45)
        self.assertEqual(m2, 30.0)
        delta = calculate_time_delta_minutes(m1, m2)
        self.assertEqual(delta, 45.0)

    def test_validation_rules_reject_invalid(self):
        # Negative pressure
        bad_batch = dict(self.sample_batch)
        bad_batch["steam_valve_open_pressure"] = -2.5
        valid, msg = validate_production_dict(bad_batch)
        self.assertFalse(valid)
        self.assertIn("Pressure cannot be negative", msg)

        # Inverted timestamps
        bad_batch2 = dict(self.sample_batch)
        bad_batch2["steam_valve_open_time"] = "15:00"
        bad_batch2["steam_valve_close_time"] = "13:00"
        valid2, msg2 = validate_production_dict(bad_batch2)
        self.assertFalse(valid2)

    def test_feature_engineering_count(self):
        feats = engineer_batch_features(self.sample_batch)
        self.assertGreaterEqual(len(feats), 45)
        self.assertIn("steam_duration", feats)
        self.assertIn("heating_rate", feats)
        self.assertIn("cooling_efficiency", feats)
        self.assertIn("pressure_stability", feats)
        self.assertIn("thermal_exposure_score", feats)

    def test_end_to_end_predict_batch(self):
        res = predict_batch(self.sample_batch)
        self.assertEqual(res["batch_id"], "TEST-001")
        self.assertGreaterEqual(res["golden_score"], 0.0)
        self.assertLessEqual(res["golden_score"], 100.0)
        self.assertIn(res["status"], ["Golden", "Warning", "Non-Golden"])

        # Check all 11 predictions
        self.assertEqual(len(res["predictions"]), 11)
        for target in TARGET_QUALITY_PARAMETERS:
            self.assertIn(target, res["predictions"])
            item = res["predictions"][target]
            self.assertIn("predicted_value", item)
            self.assertIn("status", item)

        # Check SHAP explainability
        self.assertEqual(len(res["explainability"]), 11)

    def test_reports_generation(self):
        res = predict_batch(self.sample_batch)
        pdf_path = generate_pdf_batch_report(res, "test_report.pdf")
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

        excel_path = generate_excel_batch_report(res, "test_report.xlsx")
        self.assertTrue(os.path.exists(excel_path))
        self.assertGreater(os.path.getsize(excel_path), 1000)

        csv_path = generate_csv_batch_report(res, "test_report.csv")
        self.assertTrue(os.path.exists(csv_path))
        self.assertGreater(os.path.getsize(csv_path), 50)

if __name__ == "__main__":
    unittest.main()
