"""
Evaluate Model Performance on Training Set Batches
Compares predictions against ground truth laboratory values for training set rows.
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
from shared.constants import TARGET_QUALITY_PARAMETERS

def evaluate_train_samples():
    prod_path = os.path.join(CURRENT_DIR, "data", "production_data.csv")
    qual_path = os.path.join(CURRENT_DIR, "data", "quality_data.csv")

    if not os.path.exists(prod_path) or not os.path.exists(qual_path):
        print("Training files missing.")
        return

    prod_df = pd.read_csv(prod_path)
    qual_df = pd.read_csv(qual_path)

    # Merge on Batch ID / Batch No.
    merged = pd.concat([prod_df.reset_index(drop=True), qual_df.reset_index(drop=True)], axis=1)

    print("=" * 105)
    print(f"       ADVaIT MODEL PERFORMANCE EVALUATION ON TRAINING DATASET ({len(merged)} BATCHES)")
    print("=" * 105)

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

    # Evaluate 3 sample training batches
    sample_rows = merged.head(3)

    for idx, row in sample_rows.iterrows():
        batch_id = row.get("Batch ID", f"Batch-{idx}")
        input_dict = row.to_dict()

        result = predict_batch(input_dict)
        predictions = result["predictions"]
        golden_score = result["golden_score"]

        print(f"\n---------------------------------------------------------------------------------------------------------")
        print(f" TRAINING SAMPLE BATCH: {batch_id}  |  Golden Score: {golden_score:.1f}%")
        print(f"---------------------------------------------------------------------------------------------------------")
        print(f"{'Parameter':<18} | {'Actual (Lab)':<12} | {'Predicted':<10} | {'Abs Error':<10} | {'% Error':<9} | {'Spec Status'}")
        print("-" * 105)

        total_pct_err = 0
        cnt = 0
        for param in TARGET_QUALITY_PARAMETERS:
            gt_key = gt_key_map.get(param, param)
            actual = float(row.get(gt_key, np.nan))
            pred_info = predictions.get(param, {})
            predicted = pred_info.get("predicted_value", np.nan)
            pred_status = pred_info.get("status", "")

            if not pd.isna(actual) and not pd.isna(predicted):
                abs_err = abs(predicted - actual)
                pct_err = (abs_err / abs(actual)) * 100.0 if actual != 0 else 0.0
                total_pct_err += pct_err
                cnt += 1
                print(f"{param:<18} | {actual:<12.3f} | {predicted:<10.2f} | {abs_err:<10.3f} | {pct_err:<8.2f}% | {pred_status}")

        avg_acc = max(0.0, 100.0 - (total_pct_err / cnt)) if cnt > 0 else 0.0
        print("-" * 105)
        print(f" AVERAGE PARAMETER PREDICTION ACCURACY: {avg_acc:.2f}%")

if __name__ == "__main__":
    evaluate_train_samples()
