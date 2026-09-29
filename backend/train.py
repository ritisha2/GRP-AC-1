"""
ADVaIT Model Training & 5-Fold Cross-Validation Pipeline
Trains regularized, feature-selected regression models across all 11 quality parameters.
Evaluates Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost,
RidgeCV, and Stacking/Voting ensembles using 5-fold cross-validation.
"""

import os
import sys
import json
import re
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from shared.constants import TARGET_QUALITY_PARAMETERS
from backend.dataset_intelligence import load_merged_data
from backend.golden_batch import build_golden_profile

# Scikit-learn models & metrics
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def get_candidate_models(k_features: int = 18):
    return {
        "Extra Trees": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", ExtraTreesRegressor(n_estimators=150, max_depth=8, min_samples_leaf=1, max_features="sqrt", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=2, max_features="sqrt", random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", GradientBoostingRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", xgb.XGBRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42))
        ]),
        "RidgeCV": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", RidgeCV(alphas=np.logspace(-2, 3, 25)))
        ]),
        "CatBoost": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", CatBoostRegressor(iterations=80, learning_rate=0.05, depth=4, l2_leaf_reg=3.0, random_seed=42, verbose=0))
        ]),
        "LightGBM": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("scaler", RobustScaler()),
            ("model", lgb.LGBMRegressor(n_estimators=80, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, random_state=42, verbose=-1))
        ]),
        "Voting Ensemble": Pipeline([
            ("selector", SelectKBest(f_regression, k=k_features)),
            ("model", VotingRegressor([
                ("et", ExtraTreesRegressor(n_estimators=120, max_depth=8, min_samples_leaf=1, max_features="sqrt", random_state=42)),
                ("rf", RandomForestRegressor(n_estimators=120, max_depth=8, min_samples_leaf=2, max_features="sqrt", random_state=42)),
                ("cb", CatBoostRegressor(iterations=60, learning_rate=0.05, depth=4, l2_leaf_reg=3.0, random_seed=42, verbose=0))
            ]))
        ])
    }

def target_to_slug(target_name: str) -> str:
    slug = target_name.lower().replace("%", "_pct").replace(".", "").replace(" ", "_")
    slug = re.sub(r'[^a-z0-9_]', '', slug)
    return slug

def train_all_models():
    print("=" * 60)
    print("Starting ADVaIT High-Accuracy Model Training & Feature Selection")
    print("=" * 60)
    
    df_prod, df_qual, df_feats, clean_qual = load_merged_data()
    
    # 1. Clean dataset: filter corrupted footer rows where quality was 0
    valid_mask = (clean_qual["Hardness"] > 0)
    df_feats = df_feats[valid_mask].reset_index(drop=True)
    clean_qual = clean_qual[valid_mask].reset_index(drop=True)
    
    # 2. Fix Specific Gravity zero outlier by imputing median
    if (clean_qual["Specific Gravity"] == 0).any():
        med_sg = clean_qual.loc[clean_qual["Specific Gravity"] > 0, "Specific Gravity"].median()
        clean_qual.loc[clean_qual["Specific Gravity"] == 0, "Specific Gravity"] = med_sg
        
    feature_cols = [c for c in df_feats.columns if c != "batch_id"]
    X = df_feats[feature_cols].copy().fillna(0.0)
    
    print(f"Cleaned training dataset: {len(X)} batches, {X.shape[1]} engineered features.")
    
    # Save feature names list
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
        
    # Rebuild historical golden profile on clean data
    build_golden_profile(df_feats, clean_qual)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    overall_metrics = {}
    
    k_sel = min(18, X.shape[1])
    
    for target in TARGET_QUALITY_PARAMETERS:
        print(f"\n---> Tuning & Cross-Validating Target: [{target}]")
        y = clean_qual[target].values
        
        candidates = get_candidate_models(k_features=k_sel)
        best_model_name = None
        best_mae = float("inf")
        best_rmse = float("inf")
        best_r2 = -float("inf")
        best_pipeline = None
        algo_scores = {}
        
        for name, model in candidates.items():
            try:
                cv_res = cross_validate(
                    model, X, y, cv=cv, n_jobs=-1,
                    scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
                )
                r2 = float(np.mean(cv_res["test_r2"]))
                rmse = float(np.sqrt(-np.mean(cv_res["test_neg_mean_squared_error"])))
                mae = float(-np.mean(cv_res["test_neg_mean_absolute_error"]))
                algo_scores[name] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}
                
                # Selection criterion: Lowest Cross-Validated MAE (generalization accuracy)
                if mae < best_mae:
                    best_mae = mae
                    best_rmse = rmse
                    best_r2 = r2
                    best_model_name = name
                    best_pipeline = model
            except Exception as e:
                print(f"     [Warning] Failed evaluating {name} on {target}: {e}")
                
        if best_pipeline is None:
            best_model_name = "Voting Ensemble"
            best_pipeline = candidates["Voting Ensemble"]
            
        mean_y = np.mean(y) if np.mean(y) != 0 else 1.0
        cv_accuracy = max(0.0, 100.0 - (best_mae / abs(mean_y) * 100.0))
        print(f"     * Best Algorithm: {best_model_name} (MAE = {round(best_mae, 4)}, CV Accuracy = {round(cv_accuracy, 1)}%)")
        
        # Fit best pipeline on full clean training dataset
        best_pipeline.fit(X, y)
        y_pred = best_pipeline.predict(X)
        
        final_r2 = float(r2_score(y, y_pred))
        final_rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        final_mae = float(mean_absolute_error(y, y_pred))
        
        target_slug = target_to_slug(target)
        model_save_path = os.path.join(MODELS_DIR, f"{target_slug}.joblib")
        joblib.dump(best_pipeline, model_save_path)
        
        overall_metrics[target] = {
            "slug": target_slug,
            "best_algorithm": best_model_name,
            "cv_accuracy_pct": round(cv_accuracy, 2),
            "cv_mae": round(best_mae, 4),
            "cv_rmse": round(best_rmse, 4),
            "cv_r2": round(best_r2, 4),
            "train_r2": round(final_r2, 4),
            "train_rmse": round(final_rmse, 4),
            "train_mae": round(final_mae, 4),
            "model_path": model_save_path,
            "candidate_benchmarks": algo_scores
        }
        
    # Save overall metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, indent=2)
        
    print("\n" + "=" * 60)
    print("High-Accuracy Model Training Completed for all 11 Quality Parameters!")
    print(f"Models & Metrics saved to: {MODELS_DIR}")
    print("=" * 60)
    return overall_metrics

if __name__ == "__main__":
    train_all_models()
