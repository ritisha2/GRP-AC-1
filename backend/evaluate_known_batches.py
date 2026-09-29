"""
Evaluate Model Performance on Known Holdout Batches and Training Samples
Compares predicted quality parameters against ground truth laboratory measurements.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from backend.predict import predict_batch
from shared.constants import TARGET_QUALITY_PARAMETERS, QUALITY_SPECIFICATIONS

def evaluate():
    holdout_path = os.path.join(CURRENT_DIR, "data", "holdout_test_batches.json")
    metrics_path = os.path.join(CURRENT_DIR, "models", "metrics.json")
    
    print("=" * 105)
    print("                ADVaIT MODEL PERFORMANCE EVALUATION & BENCHMARK REPORT")
    print("=" * 105)

    # 1. Print Cross-Validation Summary Metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        print("\n[1] 10-FOLD CROSS-VALIDATION MODEL BENCHMARKS (Across 9 Machine Learning Algorithms)")
        print("-" * 105)
        print(f"{'Target Quality Parameter':<22} | {'Best Algorithm':<18} | {'CV Accuracy (%)':<16} | {'CV MAE':<10} | {'CV RMSE':<10}")
        print("-" * 105)
        for param in TARGET_QUALITY_PARAMETERS:
            m = metrics_data.get(param, {})
            algo = m.get("best_algorithm", "N/A")
            acc = m.get("cv_accuracy_pct", 0.0)
            mae = m.get("cv_mae", 0.0)
            rmse = m.get("cv_rmse", 0.0)
            print(f"{param:<22} | {algo:<18} | {acc:<16.2f}% | {mae:<10.4f} | {rmse:<10.4f}")
        print("-" * 105)

    # 2. Evaluate Holdout Test Batches (Unseen Data)
    if os.path.exists(holdout_path):
        with open(holdout_path, "r", encoding="utf-8") as f:
            holdout_data = json.load(f)

        print("\n[2] EVALUATION ON HELD-OUT UNSEEN FACTORY TEST BATCHES")

        gt_key_map = {
            "Hardness": "Hardness",
            "EB": "EB",
            "TS": "TS",
            "Mv": "Mv",
            "Specific Gravity": "Sp.Gravity",
            "RHC": "RHC",
            "VM%": "V.M.%",
            "AE%": "A.E.%",
            "CB%": "C.B.%",
            "Ash%": "Ash%",
            "AC Mooney": "AC Mooney"
        }

        for item in holdout_data:
            batch_id = item["batch_id"]
            prod_row = item["production_row"]
            gt_quality = item["ground_truth_quality"]

            result = predict_batch(prod_row)
            predictions = result["predictions"]
            golden_score = result["golden_score"]
            status = result["status"]

            print(f"\n---------------------------------------------------------------------------------------------------------")
            print(f" BATCH ID: {batch_id}  |  Golden Score: {golden_score:.1f}% ({status})")
            print(f"---------------------------------------------------------------------------------------------------------")
            print(f"{'Parameter':<18} | {'Actual (Lab)':<12} | {'Predicted':<10} | {'Abs Error':<10} | {'% Error':<9} | {'Spec Range':<16} | {'Status'}")
            print("-" * 105)

            total_abs_error = 0
            total_pct_error = 0
            count = 0

            for param in TARGET_QUALITY_PARAMETERS:
                gt_key = gt_key_map.get(param, param)
                actual = gt_quality.get(gt_key, np.nan)
                pred_info = predictions.get(param, {})
                predicted = pred_info.get("predicted_value", np.nan)
                spec_lsl = pred_info.get("lsl", 0)
                spec_usl = pred_info.get("usl", 0)
                unit = pred_info.get("unit", "")
                pred_status = pred_info.get("status", "")

                if not pd.isna(actual) and not pd.isna(predicted):
                    abs_err = abs(predicted - actual)
                    pct_err = (abs_err / abs(actual)) * 100.0 if actual != 0 else 0.0
                    total_abs_error += abs_err
                    total_pct_error += pct_err
                    count += 1

                    spec_str = f"[{spec_lsl} - {spec_usl}] {unit}"
                    print(f"{param:<18} | {actual:<12.3f} | {predicted:<10.2f} | {abs_err:<10.3f} | {pct_err:<8.2f}% | {spec_str:<16} | {pred_status}")

            avg_pct_error = total_pct_error / count if count > 0 else 0.0
            overall_acc = max(0.0, 100.0 - avg_pct_error)
            print("-" * 105)
            print(f" AVERAGE PARAMETER PREDICTION ACCURACY FOR {batch_id}: {overall_acc:.2f}% (Mean Relative Error: {avg_pct_error:.2f}%)")

if __name__ == "__main__":
    evaluate()
