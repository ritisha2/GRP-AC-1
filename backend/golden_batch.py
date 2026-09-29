"""
ADVaIT Golden Batch Engine
Calculates historical Golden Profiles (P10, P25, Median, P75, P90) for process features
and computes the 0–100% Golden Batch Score with granular deviation attribution.
"""

import os
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.constants import (
    GOLDEN_THRESHOLD, WARNING_THRESHOLD,
    STATUS_GOLDEN, STATUS_WARNING, STATUS_NON_GOLDEN,
    QUALITY_SPECIFICATIONS, RADAR_FEATURES
)

PROFILE_PATH = os.path.join(CURRENT_DIR, "models", "golden_profile.json")

def build_golden_profile(df_features: pd.DataFrame, df_quality: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Computes historical percentiles (P10, P25, Median, P75, P90) for all engineered process features
    and saves the Golden Profile.
    """
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    numeric_cols = [c for c in df_features.columns if c != "batch_id" and pd.api.types.is_numeric_dtype(df_features[c])]
    
    profile = {
        "features": {},
        "radar_medians": {},
        "quality_specs": QUALITY_SPECIFICATIONS
    }
    
    for col in numeric_cols:
        s = df_features[col].dropna()
        if len(s) == 0:
            continue
        p10 = float(np.percentile(s, 10))
        p25 = float(np.percentile(s, 25))
        median = float(np.percentile(s, 50))
        p75 = float(np.percentile(s, 75))
        p90 = float(np.percentile(s, 90))
        
        profile["features"][col] = {
            "p10": round(p10, 4),
            "p25": round(p25, 4),
            "median": round(median, 4),
            "p75": round(p75, 4),
            "p90": round(p90, 4),
            "iqr": round(p75 - p25, 4),
        }
        
    for rf in RADAR_FEATURES:
        if rf in profile["features"]:
            profile["radar_medians"][rf] = profile["features"][rf]["median"]
        else:
            profile["radar_medians"][rf] = 1.0

    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        
    print(f"Built Golden Profile across {len(profile['features'])} features -> {PROFILE_PATH}")
    return profile

def load_golden_profile() -> Dict[str, Any]:
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Fallback to building if data exists
    from backend.dataset_intelligence import load_merged_data
    _, _, df_feats, df_qual = load_merged_data()
    return build_golden_profile(df_feats, df_qual)

def score_feature_value(val: float, prof: Dict[str, float]) -> Tuple[float, str]:
    """
    Scores one feature value:
    - P25 to P75: 100
    - P10 to P90: 70
    - Outside: Penalty (down to 10-40)
    """
    p10, p25, med, p75, p90 = prof["p10"], prof["p25"], prof["median"], prof["p75"], prof["p90"]
    iqr = max(p75 - p25, 1e-4)
    
    if p25 <= val <= p75:
        return 100.0, "Optimal (P25-P75)"
    elif p10 <= val <= p90:
        return 70.0, "Acceptable (P10-P90)"
    else:
        # Distance penalty
        dist = min(abs(val - p10), abs(val - p90))
        penalty = min(50.0, (dist / iqr) * 20.0)
        score = max(10.0, 70.0 - penalty)
        direction = "Above Golden Range" if val > p90 else "Below Golden Range"
        return round(score, 1), direction

def evaluate_batch(
    engineered_features: Dict[str, float],
    predicted_qualities: Dict[str, float]
) -> Dict[str, Any]:
    """
    Evaluates an entire batch:
    1. Scores process adherence against historical Golden Profile.
    2. Scores predicted quality parameters against industrial specifications.
    3. Blends scores into Final Golden Batch Score (0-100%).
    4. Categorizes into Golden, Warning, or Non-Golden.
    5. Returns radar comparison data and root cause penalties.
    """
    profile = load_golden_profile()
    feature_profiles = profile.get("features", {})
    
    process_scores = []
    penalties = []
    
    # Process features scoring
    for feat_name, feat_val in engineered_features.items():
        if feat_name in feature_profiles and "missing" not in feat_name:
            f_prof = feature_profiles[feat_name]
            score, status = score_feature_value(feat_val, f_prof)
            process_scores.append(score)
            if score < 75.0:
                penalties.append({
                    "type": "Process Deviation",
                    "feature": feat_name,
                    "value": feat_val,
                    "median": f_prof["median"],
                    "range": f"[{f_prof['p25']} - {f_prof['p75']}]",
                    "status": status,
                    "score": score
                })

    avg_process_score = float(np.mean(process_scores)) if process_scores else 80.0
    
    # Quality specifications scoring
    quality_scores = []
    for q_name, q_val in predicted_qualities.items():
        spec = QUALITY_SPECIFICATIONS.get(q_name)
        if spec:
            lsl, target, usl = spec["lsl"], spec["target"], spec["usl"]
            span = max(usl - lsl, 1e-4)
            if lsl <= q_val <= usl:
                q_score = 100.0
            else:
                dist = min(abs(q_val - lsl), abs(q_val - usl))
                penalty = min(60.0, (dist / span) * 100.0)
                q_score = max(15.0, 80.0 - penalty)
                penalties.append({
                    "type": "Quality Deviation",
                    "feature": q_name,
                    "value": round(q_val, 2),
                    "median": target,
                    "range": f"[{lsl} - {usl}]",
                    "status": "Out of Specification",
                    "score": round(q_score, 1)
                })
            quality_scores.append(q_score)
            
    # Quality specifications in-spec count
    in_spec_count = sum(
        1 for q_name, q_val in predicted_qualities.items()
        if q_name in QUALITY_SPECIFICATIONS and
        QUALITY_SPECIFICATIONS[q_name]["lsl"] <= q_val <= QUALITY_SPECIFICATIONS[q_name]["usl"]
    )
    total_q = max(len(predicted_qualities), 1)
    
    # Golden Batch Quality Compliance Rate (e.g., 10 / 11 = 90.91%)
    quality_compliance_pct = round((in_spec_count / total_q) * 100.0, 2)
    
    # Primary Golden Batch Score directly reflects the Quality Spec Compliance
    golden_score = quality_compliance_pct
    
    if golden_score >= 90.0:
        status = STATUS_GOLDEN
    elif golden_score >= 75.0:
        status = STATUS_WARNING
    else:
        status = STATUS_NON_GOLDEN
        
    # Process Radar comparison
    radar_comparison = {}
    for rf in RADAR_FEATURES:
        cur_val = engineered_features.get(rf, 0.0)
        golden_med = profile.get("radar_medians", {}).get(rf, 1.0)
        radar_comparison[rf] = {
            "current": round(float(cur_val), 3),
            "golden_median": round(float(golden_med), 3)
        }

    return {
        "golden_score": golden_score,
        "status": status,
        "process_score": round(avg_process_score, 1),
        "quality_score": round(quality_compliance_pct, 1),
        "in_spec_count": in_spec_count,
        "total_targets": total_q,
        "penalties": penalties,
        "radar_comparison": radar_comparison
    }

if __name__ == "__main__":
    prof = load_golden_profile()
    print("Golden Profile loaded. Available features:", len(prof.get("features", {})))
