"""
ADVaIT Data Loader
Fetches data from BT.xlsx, converts it into separate production and quality CSVs,
removes 2 non-essential columns from each as instructed, and formats clean datasets.
"""

import os
import re
import json
import pandas as pd
import numpy as np


DEFAULT_SOURCE_PATH = r"C:\Users\Admin\Downloads\BT.xlsx"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def extract_and_convert_data(source_excel_path=DEFAULT_SOURCE_PATH, target_dir=DATA_DIR):
    os.makedirs(target_dir, exist_ok=True)
    
    if not os.path.exists(source_excel_path):
        raise FileNotFoundError(f"Source file not found at: {source_excel_path}")
        
    print(f"Reading {source_excel_path}...")
    xl = pd.ExcelFile(source_excel_path)
    sheet_name = 'BT-AC10 Production Insights' if 'BT-AC10 Production Insights' in xl.sheet_names else xl.sheet_names[0]
    raw_df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    
    # Headers are in row index 8 (9th row)
    raw_headers = raw_df.iloc[8].tolist()
    
    # Clean header names
    clean_headers = []
    for i, h in enumerate(raw_headers):
        if pd.isna(h):
            clean_headers.append(f"Col_{i}")
        else:
            h_str = str(h).strip().replace('\n', ' ')
            # Clean non-ascii characters like degree symbol encoding artifacts
            h_str = re.sub(r'[^\x00-\x7F]+', 'deg', h_str)
            clean_headers.append(h_str)
            
    # Rows 11 onwards are batch records
    data_df = raw_df.iloc[11:].copy().reset_index(drop=True)
    data_df.columns = clean_headers
    
    # Filter out empty rows if any
    data_df = data_df[data_df.iloc[:, 5].notna()].reset_index(drop=True)
    print(f"Extracted {len(data_df)} batch records.")
    
    # Separate Production Data (columns 0 to 70) and Quality Data (columns 72 onwards)
    prod_cols = clean_headers[:71]
    qual_cols = clean_headers[72:]
    
    prod_df = data_df[prod_cols].copy()
    qual_df = data_df[qual_cols].copy()
    
    # Save initial converted CSVs
    raw_prod_csv = os.path.join(target_dir, "raw_production_data.csv")
    raw_qual_csv = os.path.join(target_dir, "raw_quality_data.csv")
    prod_df.to_csv(raw_prod_csv, index=False)
    qual_df.to_csv(raw_qual_csv, index=False)
    print(f"Saved initial CSVs:\n  {raw_prod_csv}\n  {raw_qual_csv}")
    
    # Instruction: "also after converting the data remove any 2 columns from both the csv's.
    # after doing this with the remaning data implement the model work and other things."
    
    # From Production CSV: Remove 'Comment' and 'Customer' (columns with unstructured metadata / empty)
    # This leaves all 23 production inputs + Batch ID perfectly intact!
    prod_cols_to_drop = []
    for candidate in ['Comment', 'Customer', 'Batch kWh', 'S/N']:
        if candidate in prod_df.columns and len(prod_cols_to_drop) < 2:
            prod_cols_to_drop.append(candidate)
            
    # From Quality CSV:
    # First 12 columns are: Batch No. and the 11 lab measurements.
    # The columns after include 'Col_84' (empty) and 'Success % off all parameters average'.
    # We drop exactly 2 columns: 'Col_84' and 'Success % off all parameters average' (or the duplicate set)
    qual_cols_to_drop = []
    for candidate in ['Col_84', 'Success % off all parameters average']:
        if candidate in qual_df.columns and len(qual_cols_to_drop) < 2:
            qual_cols_to_drop.append(candidate)
            
    # If not found by exact string, drop the last 2 auxiliary columns
    if len(qual_cols_to_drop) < 2:
        qual_cols_to_drop = [c for c in qual_df.columns if 'Col_84' in c or 'Success' in c][:2]
        
    print(f"Dropping 2 columns from Production Data: {prod_cols_to_drop}")
    cleaned_prod_df = prod_df.drop(columns=prod_cols_to_drop)
    
    print(f"Dropping 2 columns from Quality Data: {qual_cols_to_drop}")
    cleaned_qual_df = qual_df.drop(columns=qual_cols_to_drop)
    
    # If quality dataframe contains duplicated columns from repeated testing columns, keep first occurrence
    _, unique_indices = np.unique(cleaned_qual_df.columns, return_index=True)
    cleaned_qual_df = cleaned_qual_df.iloc[:, sorted(unique_indices)]
    
    # HOLD OUT 2 RANDOM BATCH ROWS FOR USER TESTING:
    # Selected batches: BT-07585-0 and BT-040152-0 (indices 35 and 102)
    holdout_indices = [35, 102]
    holdout_batches_info = []
    
    for h_idx in holdout_indices:
        p_row = cleaned_prod_df.iloc[h_idx].to_dict()
        q_row = cleaned_qual_df.iloc[h_idx].to_dict()
        b_id = str(p_row.get("Batch ID"))
        holdout_batches_info.append({
            "batch_id": b_id,
            "production_row": p_row,
            "ground_truth_quality": q_row
        })
        
    holdout_json_path = os.path.join(target_dir, "holdout_test_batches.json")
    with open(holdout_json_path, "w", encoding="utf-8") as f:
        # Convert non-serializable objects
        clean_holdout = []
        for item in holdout_batches_info:
            clean_item = {
                "batch_id": item["batch_id"],
                "production_row": {str(k): (str(v) if pd.notna(v) else "") for k, v in item["production_row"].items()},
                "ground_truth_quality": {str(k): (float(v) if pd.notna(v) and str(v).replace('.','',1).isdigit() else str(v)) for k, v in item["ground_truth_quality"].items()}
            }
            clean_holdout.append(clean_item)
        json.dump(clean_holdout, f, indent=2)
        
    print(f"Held out 2 random batches for user testing: {[h['batch_id'] for h in holdout_batches_info]}")
    print(f"Saved holdout test batches -> {holdout_json_path}")
    
    # Drop the 2 rows from training data
    cleaned_prod_df = cleaned_prod_df.drop(index=holdout_indices).reset_index(drop=True)
    cleaned_qual_df = cleaned_qual_df.drop(index=holdout_indices).reset_index(drop=True)
    
    # Save final cleaned production and quality CSVs (144 batches)
    final_prod_csv = os.path.join(target_dir, "production_data.csv")
    final_qual_csv = os.path.join(target_dir, "quality_data.csv")
    cleaned_prod_df.to_csv(final_prod_csv, index=False)
    cleaned_qual_df.to_csv(final_qual_csv, index=False)
    
    print(f"Saved cleaned datasets (after removing 2 random test batch rows):")
    print(f"  Production: {final_prod_csv} (Shape: {cleaned_prod_df.shape})")
    print(f"  Quality:    {final_qual_csv} (Shape: {cleaned_qual_df.shape})")
    
    return cleaned_prod_df, cleaned_qual_df


if __name__ == "__main__":
    extract_and_convert_data()
