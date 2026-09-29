"""
Streamlit Dashboard
Pure White / Light Industrial SCADA Interface for Rubber Quality Prediction,
Process Feature Engineering, Golden Batch Evaluation, and SHAP Explainability.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import time, datetime
import streamlit as st

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from frontend.theme import apply_industrial_theme
from frontend.components import (
    create_quality_gauge,
    create_radar_chart,
    create_timeline_chart,
    create_shap_bar_chart,
    create_input_output_correlation_heatmap,
    create_input_output_scatter_plot,
    create_feature_distribution_plot
)
from shared.constants import (
    TARGET_QUALITY_PARAMETERS, QUALITY_SPECIFICATIONS,
    STATUS_GOLDEN, STATUS_WARNING, STATUS_NON_GOLDEN
)

# Optional local backend import for fallback
try:
    from backend.predict import predict_batch as local_predict_batch
    from backend.reports_generator import (
        generate_pdf_batch_report,
        generate_excel_batch_report,
        generate_csv_batch_report
    )
    LOCAL_BACKEND_AVAILABLE = True
except Exception:
    LOCAL_BACKEND_AVAILABLE = False

API_BASE_URL = os.environ.get("API_URL", os.environ.get("ADVAIT_API_URL", "http://127.0.0.1:8000"))

st.set_page_config(
    page_title="Rubber Batch Quality Prediction System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply White Theme
st.markdown(apply_industrial_theme(), unsafe_allow_html=True)

def parse_time_str(t_str):
    if not t_str or ":" not in str(t_str):
        return time(12, 0)
    parts = str(t_str).strip().split(":")
    try:
        h = int(float(parts[0])) % 24
        m = int(float(parts[1])) % 60
        return time(h, m)
    except Exception:
        return time(12, 0)

# Load holdout batches dynamically from JSON
holdout_path = os.path.join(PARENT_DIR, "backend", "data", "holdout_test_batches.json")
holdout_batches_data = []
if os.path.exists(holdout_path):
    try:
        with open(holdout_path, "r", encoding="utf-8") as f:
            holdout_batches_data = json.load(f)
    except Exception:
        pass

def get_holdout_inputs(batch_id: str):
    for h in holdout_batches_data:
        if h.get("batch_id") == batch_id:
            p = h["production_row"]
            raw_c_dur = p.get("Cooking Duration (HH:MM)", "04:00")
            dur_m = 240.0
            if ":" in str(raw_c_dur):
                parts = str(raw_c_dur).split(":")
                dur_m = float(parts[0]) * 60.0 + float(parts[1])
            return {
                "batch_id": batch_id,
                "batch_start_time": parse_time_str(p.get("Batch Start Time (HH:MM)")),
                "steam_valve_open_temp": float(p.get("Steam Valve Open Temp (degC)", 138.0) or 138.0),
                "steam_valve_open_pressure": float(p.get("Steam Valve Open Pressure (kg/cm2)", 0.4) or 0.4),
                "steam_valve_open_time": parse_time_str(p.get("Steam Valve Open Time (HH:MM)")),
                "steam_valve_close_temp": float(p.get("Steam Valve Close Temp (degC)", 150.0) or 150.0),
                "steam_valve_close_pressure": float(p.get("Steam Valve Close Pressure (kg/cm2)", 7.05) or 7.05),
                "steam_valve_close_time": parse_time_str(p.get("Steam Valve Close Time (HH:MM)")),
                "heat_valve_close_temp": float(p.get("Heat Valve Close Temp (degC)", 217.0) or 217.0),
                "heat_valve_close_pressure": float(p.get("Heat Valve Close Pressure (kg/cm2)", 13.45) or 13.45),
                "heat_valve_close_time": parse_time_str(p.get("Heat Valve Close Time (HH:MM)")),
                "low_tf_temp_duration": float(p.get("Low TF Temp Duration(Mins) (Below 260degC)", 363.0) or 363.0),
                "cooking_start_time": parse_time_str(p.get("Cooking Start Time (HH:MM)")),
                "cooking_stop_time": parse_time_str(p.get("Cooking Stop Time (HH:MM)")),
                "cooking_duration": dur_m,
                "cooling_valve_open_temp": float(p.get("Cooling Valve Open Temp(degC)", 204.0) or 204.0),
                "cooling_valve_open_pressure": float(p.get("Cooling Valve Open Pressure (kg/cm2)", 0.17) or 0.17),
                "cooling_valve_open_time": parse_time_str(p.get("Cooling Valve Open Time (HH:MM)")),
                "bottom_door_open_temp": float(p.get("Bottom Door Open Temp(degC)", 134.0) or 134.0),
                "bottom_door_open_pressure": float(p.get("Bottom Door Open Pressure (kg/cm2)", 0.08) or 0.08),
                "bottom_door_open_time": parse_time_str(p.get("Bottom Door Open Time (HH:MM)")),
                "standard_batch_duration": 5.0,
                "actual_batch_duration": float(p.get("Actual Batch Duration (Mins)", 657.0) or 657.0),
                "batch_end_time": parse_time_str(p.get("Batch End Time (HH:MM)"))
            }
    return None

@st.cache_data
def get_historical_correlation_data():
    try:
        from backend.dataset_intelligence import load_merged_data
        _, _, df_feats, df_qual = load_merged_data()
        valid_mask = (df_qual["Hardness"] > 0)
        df_feats = df_feats[valid_mask].reset_index(drop=True)
        df_qual = df_qual[valid_mask].reset_index(drop=True)
        
        input_mapping = {
            "cooking_duration_verified": "Cooking Duration (Mins)",
            "steam_duration": "Steam Duration (Mins)",
            "heating_rate": "Heating Rate (°C/min)",
            "cooling_rate": "Cooling Rate (°C/min)",
            "steam_temp_rise": "Steam Temp Rise (°C)",
            "total_batch_duration": "Total Batch Duration (Mins)",
            "batch_duration_deviation": "Duration Deviation (Mins)",
            "heat_to_cooling_drop": "Heat-to-Cooling Drop (°C)",
            "steam_pressure_rise": "Steam Pressure Rise (kg/cm²)",
            "thermal_exposure_score": "Thermal Exposure Score",
            "effective_curing_index": "Effective Curing Index",
            "cooling_delay": "Cooling Delay (Mins)",
            "low_tf_ratio": "Low TF Duration Ratio",
            "pressure_stability": "Pressure Stability Index"
        }
        
        active_inputs = [col for col in input_mapping.keys() if col in df_feats.columns]
        active_labels = [input_mapping[col] for col in active_inputs]
        
        corr_matrix = pd.DataFrame(index=active_labels, columns=TARGET_QUALITY_PARAMETERS, dtype=float)
        for col, label in zip(active_inputs, active_labels):
            for out in TARGET_QUALITY_PARAMETERS:
                if out in df_qual.columns:
                    val = float(df_feats[col].corr(df_qual[out]))
                    corr_matrix.loc[label, out] = round(val, 2)
                else:
                    corr_matrix.loc[label, out] = 0.0
                    
        return df_feats, df_qual, corr_matrix, input_mapping
    except Exception as e:
        return None, None, None, {}

# Session state initialization: pre-filled with factory sheet batch BT-040152-0
if "batch_input" not in st.session_state:
    holdout_default = get_holdout_inputs("BT-040152-0")
    if holdout_default:
        st.session_state.batch_input = holdout_default
    else:
        st.session_state.batch_input = {
            "batch_id": "BT-040152-0",
            "batch_start_time": time(22, 1),
            "standard_batch_duration": 5.0,
            "actual_batch_duration": 356.0,
            "steam_valve_open_temp": 100.0,
            "steam_valve_open_pressure": 0.32,
            "steam_valve_open_time": time(23, 6),
            "steam_valve_close_temp": 131.0,
            "steam_valve_close_pressure": 5.03,
            "steam_valve_close_time": time(23, 16),
            "heat_valve_close_temp": 226.0,
            "heat_valve_close_pressure": 15.55,
            "heat_valve_close_time": time(2, 15),
            "low_tf_temp_duration": 131.0,
            "cooking_start_time": time(22, 38),
            "cooking_stop_time": time(2, 14),
            "cooking_duration": 216.0,
            "cooling_valve_open_temp": 219.0,
            "cooling_valve_open_pressure": 1.29,
            "cooling_valve_open_time": time(2, 18),
            "bottom_door_open_temp": 100.0,
            "bottom_door_open_pressure": 0.19,
            "bottom_door_open_time": time(3, 20),
            "batch_end_time": time(3, 58)
        }

if "ground_truth_quality" not in st.session_state:
    st.session_state.ground_truth_quality = None
    if holdout_batches_data:
        for hb in holdout_batches_data:
            if hb.get("batch_id") == "BT-040152-0":
                st.session_state.ground_truth_quality = hb.get("ground_truth_quality")
                break

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ==========================================
# Main Header Bar
# ==========================================
st.markdown("""
<div class='header-container'>
    <div>
        <span class='logo-text'>Rubber Batch <span class='logo-accent'>Quality System</span></span>
        <span style='font-size:15px; color:#64748b; margin-left:10px; font-weight:500;'>Industrial Batch Intelligence & Quality Prediction</span>
    </div>
    <div>
        <span class='sub-badge'>Production SCADA PoC</span>
    </div>
</div>
""", unsafe_allow_html=True)

# API status check
api_online = False
try:
    r = requests.get(f"{API_BASE_URL}/health", timeout=1.2)
    if r.status_code == 200:
        api_online = True
except Exception:
    api_online = False

# ==========================================
# Operational Toolbar (Main Page)
# ==========================================
tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([4, 3, 2.5, 2.5])

with tb_col1:
    st.markdown("<p style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:6px;'>⚡ Load Batch Presets</p>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("🧪 BT-040152-0", use_container_width=True, help="Load Sheet Batch BT-040152-0"):
            h_input = get_holdout_inputs("BT-040152-0")
            if h_input:
                st.session_state.batch_input = h_input
                if holdout_batches_data:
                    for hb in holdout_batches_data:
                        if hb.get("batch_id") == "BT-040152-0":
                            st.session_state.ground_truth_quality = hb.get("ground_truth_quality")
            st.rerun()
    with col_p2:
        if st.button("🟢 Golden", use_container_width=True, help="Load an optimal Golden Batch observation"):
            st.session_state.ground_truth_quality = None
            st.session_state.batch_input.update({
                "batch_id": "BT-040624-0",
                "batch_start_time": time(11, 30),
                "steam_valve_open_temp": 105.0,
                "steam_valve_open_pressure": 0.0,
                "steam_valve_open_time": time(13, 20),
                "steam_valve_close_temp": 172.0,
                "steam_valve_close_pressure": 8.5,
                "steam_valve_close_time": time(16, 40),
                "heat_valve_close_temp": 242.0,
                "heat_valve_close_pressure": 14.2,
                "heat_valve_close_time": time(16, 45),
                "low_tf_temp_duration": 340.0,
                "cooking_start_time": time(13, 20),
                "cooking_stop_time": time(17, 25),
                "cooking_duration": 245.0,
                "cooling_valve_open_temp": 232.0,
                "cooling_valve_open_pressure": 9.5,
                "cooling_valve_open_time": time(17, 25),
                "bottom_door_open_temp": 120.0,
                "bottom_door_open_pressure": 0.0,
                "bottom_door_open_time": time(18, 5),
                "standard_batch_duration": 400.0,
                "actual_batch_duration": 395.0,
                "batch_end_time": time(18, 5)
            })
            st.rerun()

    with col_p3:
        if st.button("🔴 Off-Spec", use_container_width=True, help="Load an off-specification batch"):
            st.session_state.ground_truth_quality = None
            st.session_state.batch_input.update({
                "batch_id": "BT-040310-FAIL",
                "batch_start_time": time(2, 0),
                "steam_valve_open_temp": 75.0,
                "steam_valve_open_pressure": 0.0,
                "steam_valve_open_time": time(3, 15),
                "steam_valve_close_temp": 140.0,
                "steam_valve_close_pressure": 5.0,
                "steam_valve_close_time": time(5, 30),
                "heat_valve_close_temp": 205.0,
                "heat_valve_close_pressure": 9.0,
                "heat_valve_close_time": time(5, 40),
                "low_tf_temp_duration": 150.0,
                "cooking_start_time": time(3, 20),
                "cooking_stop_time": time(6, 10),
                "cooking_duration": 170.0,
                "cooling_valve_open_temp": 185.0,
                "cooling_valve_open_pressure": 5.5,
                "cooling_valve_open_time": time(6, 15),
                "bottom_door_open_temp": 160.0,
                "bottom_door_open_pressure": 1.2,
                "bottom_door_open_time": time(6, 45),
                "standard_batch_duration": 5.0,
                "actual_batch_duration": 310.0,
                "batch_end_time": time(6, 50)
            })
            st.rerun()

with tb_col2:
    st.markdown("<p style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:6px;'>🧪 Held-Out Test Batches</p>", unsafe_allow_html=True)
    if holdout_batches_data:
        st.markdown("<div style='font-size:12px; font-weight:600; color:#2563eb;'>• BT-07585-0<br>• BT-040152-0</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px; color:#64748b;'>No holdout batches loaded</div>", unsafe_allow_html=True)

with tb_col3:
    st.markdown("<p style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:6px;'>⚙️ Backend Engine Status</p>", unsafe_allow_html=True)
    if api_online:
        st.success("🟢 FastAPI Active (8000)")
    else:
        st.info("⚡ Embedded Engine Active")

with tb_col4:
    st.markdown("<p style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase; margin-bottom:6px;'>📊 Dataset Intelligence</p>", unsafe_allow_html=True)
    rep_html_path = os.path.join(PARENT_DIR, "backend", "reports", "dataset_intelligence_report.html")
    if os.path.exists(rep_html_path):
        with open(rep_html_path, "r", encoding="utf-8") as f:
            st.download_button(
                "📥 Report (HTML)",
                data=f.read(),
                file_name="dataset_intelligence_report.html",
                mime="text/html",
                use_container_width=True
            )

st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

# ==========================================
# Production Entry Panel (Proper Alignment)
# ==========================================
st.markdown("<div class='section-header'>📋 Production Observation Entry (23 Industrial Inputs)</div>", unsafe_allow_html=True)

with st.expander("⚙️ View and Edit Batch Production Values", expanded=True):
    col_a, col_b, col_c = st.columns(3)

    # Column 1: Batch Info & Steam Stage
    with col_a:
        st.markdown("<div class='card-title'>🔹 Batch Identification</div>", unsafe_allow_html=True)
        st.session_state.batch_input["batch_id"] = st.text_input("Batch ID", value=st.session_state.batch_input["batch_id"])
        st.session_state.batch_input["batch_start_time"] = st.time_input("Batch Start Time (HH:MM)", value=st.session_state.batch_input["batch_start_time"])
        st.session_state.batch_input["standard_batch_duration"] = st.number_input("Standard Batch Duration (Hrs)", value=float(st.session_state.batch_input["standard_batch_duration"]), step=0.5, help="Standard recipe target duration in Hours (e.g., 5.0 Hrs as recorded in BT production data)")
        st.session_state.batch_input["actual_batch_duration"] = st.number_input("Actual Batch Duration (Mins)", value=float(st.session_state.batch_input["actual_batch_duration"]), step=5.0, help="Actual duration in minutes as recorded in BT production data")

        st.markdown("<div class='card-title' style='margin-top:16px;'>♨️ Steam Stage</div>", unsafe_allow_html=True)
        st.session_state.batch_input["steam_valve_open_temp"] = st.number_input("Steam Valve Open Temp (°C)", value=float(st.session_state.batch_input["steam_valve_open_temp"]), step=1.0)
        st.session_state.batch_input["steam_valve_open_pressure"] = st.number_input("Steam Valve Open Pressure (kg/cm²)", value=float(st.session_state.batch_input["steam_valve_open_pressure"]), step=0.5)
        st.session_state.batch_input["steam_valve_open_time"] = st.time_input("Steam Valve Open Time (HH:MM)", value=st.session_state.batch_input["steam_valve_open_time"])
        st.session_state.batch_input["steam_valve_close_temp"] = st.number_input("Steam Valve Close Temp (°C)", value=float(st.session_state.batch_input["steam_valve_close_temp"]), step=1.0)
        st.session_state.batch_input["steam_valve_close_pressure"] = st.number_input("Steam Valve Close Pressure (kg/cm²)", value=float(st.session_state.batch_input["steam_valve_close_pressure"]), step=0.5)
        st.session_state.batch_input["steam_valve_close_time"] = st.time_input("Steam Valve Close Time (HH:MM)", value=st.session_state.batch_input["steam_valve_close_time"])

    # Column 2: Heat Stage & Cooking Stage
    with col_b:
        st.markdown("<div class='card-title'>🔥 Heat Stage</div>", unsafe_allow_html=True)
        st.session_state.batch_input["heat_valve_close_temp"] = st.number_input("Heat Valve Close Temp (°C)", value=float(st.session_state.batch_input["heat_valve_close_temp"]), step=1.0)
        st.session_state.batch_input["heat_valve_close_pressure"] = st.number_input("Heat Valve Close Pressure (kg/cm²)", value=float(st.session_state.batch_input["heat_valve_close_pressure"]), step=0.5)
        st.session_state.batch_input["heat_valve_close_time"] = st.time_input("Heat Valve Close Time (HH:MM)", value=st.session_state.batch_input["heat_valve_close_time"])

        st.markdown("<div class='card-title' style='margin-top:16px;'>🍲 Cooking Stage</div>", unsafe_allow_html=True)
        st.session_state.batch_input["low_tf_temp_duration"] = st.number_input("Low TF Temp Duration (Mins)", value=float(st.session_state.batch_input["low_tf_temp_duration"]), step=5.0, help="Low TF Temp Duration in minutes (Below 260°C) as recorded in BT production data")
        st.session_state.batch_input["cooking_start_time"] = st.time_input("Cooking Start Time (HH:MM)", value=st.session_state.batch_input["cooking_start_time"])
        st.session_state.batch_input["cooking_stop_time"] = st.time_input("Cooking Stop Time (HH:MM)", value=st.session_state.batch_input["cooking_stop_time"])
        st.session_state.batch_input["cooking_duration"] = st.number_input("Cooking Duration (Mins)", value=float(st.session_state.batch_input["cooking_duration"]), step=5.0, help="Cooking Duration in minutes (e.g., 441 min for 07:21)")

    # Column 3: Cooling Stage & Bottom Door Unload
    with col_c:
        st.markdown("<div class='card-title'>❄️ Cooling Stage</div>", unsafe_allow_html=True)
        st.session_state.batch_input["cooling_valve_open_temp"] = st.number_input("Cooling Valve Open Temp (°C)", value=float(st.session_state.batch_input["cooling_valve_open_temp"]), step=1.0)
        st.session_state.batch_input["cooling_valve_open_pressure"] = st.number_input("Cooling Valve Open Pressure (kg/cm²)", value=float(st.session_state.batch_input["cooling_valve_open_pressure"]), step=0.5)
        st.session_state.batch_input["cooling_valve_open_time"] = st.time_input("Cooling Valve Open Time", value=st.session_state.batch_input["cooling_valve_open_time"])

        st.markdown("<div class='card-title' style='margin-top:16px;'>🚪 Bottom Door & Unload</div>", unsafe_allow_html=True)
        st.session_state.batch_input["bottom_door_open_temp"] = st.number_input("Bottom Door Open Temp (°C)", value=float(st.session_state.batch_input["bottom_door_open_temp"]), step=1.0)
        st.session_state.batch_input["bottom_door_open_pressure"] = st.number_input("Bottom Door Open Pressure (kg/cm²)", value=float(st.session_state.batch_input["bottom_door_open_pressure"]), step=0.5)
        st.session_state.batch_input["bottom_door_open_time"] = st.time_input("Bottom Door Open Time", value=st.session_state.batch_input["bottom_door_open_time"])
        st.session_state.batch_input["batch_end_time"] = st.time_input("Batch End Time", value=st.session_state.batch_input["batch_end_time"])

# ==========================================
# Predict Action Button
# ==========================================
st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
predict_clicked = st.button("🚀 PREDICT BATCH QUALITY & EVALUATE GOLDEN STATUS", type="primary", use_container_width=True)

if predict_clicked:
    payload = {}
    for k, v in st.session_state.batch_input.items():
        if isinstance(v, time):
            payload[k] = v.strftime("%H:%M")
        else:
            payload[k] = v

    # Check if manually entered batch_id matches any held-out test batch for ground truth comparison
    curr_id = str(payload.get("batch_id", "")).strip().lower()
    st.session_state.ground_truth_quality = None
    if holdout_batches_data:
        for hb in holdout_batches_data:
            if hb.get("batch_id", "").strip().lower() == curr_id:
                st.session_state.ground_truth_quality = hb.get("ground_truth_quality")
                break

    with st.spinner("Engineering 45 process features & evaluating 11 ML models..."):
        success = False
        if api_online:
            try:
                resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=8.0)
                if resp.status_code == 200:
                    st.session_state.prediction_result = resp.json()
                    success = True
                elif resp.status_code == 422:
                    # Input validation failure from FastAPI
                    try:
                        err_json = resp.json()
                        detail = err_json.get("detail", {})
                        msg = detail.get("message") if isinstance(detail, dict) else str(detail)
                    except Exception:
                        msg = resp.text
                    st.error(f"⚠️ **Process Input Notice**: {msg}")
                    # Mark as handled to avoid redundant fallback error
                    success = True
                else:
                    st.error(f"FastAPI Server Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.warning(f"FastAPI call error ({e}), switching to local inference.")

        if not success and LOCAL_BACKEND_AVAILABLE:
            try:
                res = local_predict_batch(payload)
                st.session_state.prediction_result = res
                success = True
            except Exception as e:
                err_msg = str(e)
                if hasattr(e, 'detail') and isinstance(e.detail, dict):
                    err_msg = e.detail.get('message', str(e.detail))
                st.error(f"⚠️ **Process Input Notice**: {err_msg}")

# ==========================================
# Prediction Results Section
# ==========================================
if st.session_state.prediction_result is not None:
    res = st.session_state.prediction_result
    st.divider()

    # Hero Golden Batch Card
    golden_score = res.get("golden_score", 0.0)
    status = res.get("status", "Non-Golden")
    status_class = "status-golden" if status == "Golden" else ("status-warning" if status == "Warning" else "status-nongolden")
    status_icon = "🟢" if status == "Golden" else ("🟡" if status == "Warning" else "🔴")
    score_color = "#15803d" if status == "Golden" else ("#a16207" if status == "Warning" else "#b91c1c")

    st.markdown(f"""
    <div class='golden-hero-card'>
        <div style='font-size: 13px; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 1px;'>
            Batch Quality Adherence Assessment
        </div>
        <div class='golden-hero-score' style='color: {score_color};'>
            {golden_score:.2f}%
        </div>
        <div class='status-pill {status_class}'>
            {status_icon} {status.upper()} BATCH
        </div>
        <div class='hero-submetrics'>
            <div>Quality In-Spec: <strong>{res.get('golden_evaluation', {}).get('in_spec_count', 10)} / {res.get('golden_evaluation', {}).get('total_targets', 11)} ({golden_score:.2f}%)</strong></div>
            <div>Process Adherence: <strong>{res.get('golden_evaluation', {}).get('process_score', 0):.1f}%</strong></div>
            <div>Batch ID: <strong>{res.get('batch_id')}</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 11 Laboratory Quality Parameters Prediction & Accuracy Table
    st.markdown("<div class='section-header' style='margin-top:24px;'>🧪 Predicted Laboratory Quality Parameters & Tolerance Table (11 Targets)</div>", unsafe_allow_html=True)
    
    predictions = res.get("predictions", {})
    gt_dict = st.session_state.ground_truth_quality or {}
    
    table_rows = []
    total_acc = []

    for target in TARGET_QUALITY_PARAMETERS:
        pred_item = predictions.get(target, {})
        p_val = pred_item.get("predicted_value", 0.0)
        unit = pred_item.get("unit", "")
        spec_info = QUALITY_SPECIFICATIONS.get(target, {})
        lsl = spec_info.get("lsl", pred_item.get("lsl", 0.0))
        usl = spec_info.get("usl", pred_item.get("usl", 0.0))
        target_val = spec_info.get("target", (lsl + usl) / 2.0)
        spec_text = spec_info.get("spec_str", f"{lsl:g} - {usl:g}")

        # Find matching ground truth key if available
        gt_val = None
        for k, v in gt_dict.items():
            k_clean = k.replace(".", "").replace("%", "").replace(" ", "").lower()
            t_clean = target.replace(".", "").replace("%", "").replace(" ", "").lower()
            if k_clean == t_clean or k_clean in t_clean or t_clean in k_clean:
                try:
                    gt_val = float(str(v).replace("%", "").strip())
                except Exception:
                    gt_val = v
                break

        row = {
            "Quality Parameter": target,
            "Plant Spec": spec_text,
            "MIN": f"{lsl:g}",
            "TARGET": f"{target_val:g}",
            "MAX": f"{usl:g}",
            "ML Predicted Value": f"{p_val:.2f} {unit}",
        }

        if gt_val is not None and isinstance(gt_val, (int, float)):
            delta = p_val - gt_val
            pct_err = abs(delta) / max(abs(gt_val), 1e-4) * 100.0
            acc = max(0.0, 100.0 - pct_err)
            total_acc.append(acc)
            row["Actual Lab Ground Truth"] = f"{gt_val:.2f} {unit}"
            row["Delta (Pred - Actual)"] = f"{delta:+.2f} {unit}"
            row["Prediction Accuracy (%)"] = f"{acc:.1f}%"

        is_ok = (p_val >= lsl) and (p_val <= usl)
        row["Tolerance Status"] = "✅ Within Spec" if is_ok else ("⚠️ Above Max" if p_val > usl else "⚠️ Below Min")
        table_rows.append(row)

    if table_rows:
        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        if total_acc:
            mean_accuracy = sum(total_acc) / len(total_acc)
            st.markdown(f"""
            <div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:10px 16px; margin-top:8px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center;'>
                <span style='font-size:13px; font-weight:700; color:#166534;'>🎯 Mean Multi-Target Prediction Accuracy: <strong style='font-size:16px; color:#15803d;'>{mean_accuracy:.1f}%</strong> across all 11 Laboratory Quality Parameters</span>
                <span style='font-size:12px; color:#4b5563;'>Calculated against held-out laboratory ground truth</span>
            </div>
            """, unsafe_allow_html=True)

    # 11 Laboratory Quality Predictions in Neat 4-Column Grid
    st.markdown("<div class='section-header' style='margin-top:20px;'>🧪 Predicted Laboratory Quality Parameters (11 Targets)</div>", unsafe_allow_html=True)

    predictions = res.get("predictions", {})

    cols = st.columns(4)
    for idx, (p_name, item) in enumerate(predictions.items()):
        with cols[idx % 4]:
            stat = item["status"]
            val = item["predicted_value"]
            unit = item["unit"]
            lsl, usl = item["lsl"], item["usl"]
            badge_class = "badge-ok" if "Within" in stat else "badge-out"
            
            st.markdown(f"""
            <div class='param-card'>
                <div>
                    <div class='param-name'>{p_name}</div>
                    <div class='param-val'>{val} <span style='font-size:13px; color:#64748b; font-weight:600;'>{unit}</span></div>
                </div>
                <div class='param-spec'>
                    <span>[{lsl} - {usl}]</span>
                    <span class='param-status-badge {badge_class}'>{stat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # Visual Analytics Tabs
    tab_corr, tab_shap, tab_features = st.tabs([
        "📊 Input-Output Correlation",
        "🧠 Explainable AI (SHAP)",
        "⚙️ All 45 Process Features"
    ])

    with tab_corr:
        st.markdown(f"##### 📊 Process Input (Y-Axis) vs. Quality Output (X-Axis) Correlation Analysis — Batch: `{res.get('batch_id')}`")
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 14px; margin-bottom:12px; font-size:12px; color:#475569;'>
            <b>Industrial Process Correlation Matrix:</b> Shows how machine settings and process inputs (<span style='color:#1e293b; font-weight:700;'>Y-Axis</span>) statistically drive the 11 predicted quality parameters (<span style='color:#1e293b; font-weight:700;'>X-Axis</span>) for the predicted batch. Hover over any cell to see the current batch's input value, predicted quality value, and directional sensitivity.
        </div>
        """, unsafe_allow_html=True)

        hist_feats, hist_qual, corr_matrix, in_map = get_historical_correlation_data()
        
        if corr_matrix is not None and not corr_matrix.empty:
            curr_feats = res.get("engineered_features", {})
            mapped_batch_inputs = {}
            for col_k, label_v in in_map.items():
                mapped_batch_inputs[label_v] = curr_feats.get(col_k, 0.0)
                
            curr_batch_preds = {k: v.get("predicted_value", 0.0) for k, v in predictions.items()}
            
            fig_matrix = create_input_output_correlation_heatmap(
                corr_matrix, mapped_batch_inputs, curr_batch_preds, res.get("batch_id", "Current Batch")
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 🔍 Interactive Operating Curve Drilldown for Current Batch")
            st.caption("Select any Model Input (Y) and Model Output (X) to visualize where this batch operates on the factory correlation trendline.")
            
            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                sel_input_label = st.selectbox(
                    "Select Model Input Variable (Y-Axis):",
                    options=list(in_map.values()),
                    index=0,
                    key="corr_sel_inp"
                )
            with c_sel2:
                sel_output_label = st.selectbox(
                    "Select Model Output Parameter (X-Axis):",
                    options=TARGET_QUALITY_PARAMETERS,
                    index=0,
                    key="corr_sel_out"
                )
                
            rev_map = {v: k for k, v in in_map.items()}
            sel_input_col = rev_map.get(sel_input_label)
            
            if hist_feats is not None and hist_qual is not None and sel_input_col in hist_feats.columns and sel_output_label in hist_qual.columns:
                batch_in_val = float(curr_feats.get(sel_input_col, 0.0))
                batch_out_val = float(curr_batch_preds.get(sel_output_label, 0.0))
                spec_param = QUALITY_SPECIFICATIONS.get(sel_output_label)
                
                fig_scatter, r_val = create_input_output_scatter_plot(
                    hist_qual[sel_output_label],
                    hist_feats[sel_input_col],
                    sel_input_label,
                    sel_output_label,
                    batch_in_val,
                    batch_out_val,
                    res.get("batch_id", "Current Batch"),
                    spec_param
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    r_desc = "Strong Positive" if r_val >= 0.5 else ("Moderate Positive" if r_val >= 0.2 else ("Strong Negative" if r_val <= -0.5 else ("Moderate Negative" if r_val <= -0.2 else "Weak/Neutral")))
                    st.metric("Correlation Coefficient (r)", f"{r_val:+.2f}", r_desc)
                with sc2:
                    st.metric("Current Batch Input (Y)", f"{batch_in_val:g}", sel_input_label)
                with sc3:
                    stat_txt = predictions.get(sel_output_label, {}).get("status", "Within Spec")
                    st.metric("Predicted Quality Output (X)", f"{batch_out_val:g}", stat_txt)
        else:
            st.info("Correlation dataset is loading...")

    with tab_shap:
        st.markdown("##### Explainable AI: Feature Attribution (SHAP)")
        explain_map = res.get("explainability", {})
        selected_target = st.selectbox("Select Laboratory Quality Target to Explain:", list(explain_map.keys()), index=0)
        
        if selected_target in explain_map:
            exp_item = explain_map[selected_target]
            st.markdown(f"**Target:** `{selected_target}` | **Predicted:** `{exp_item.get('prediction')}` | **Base Value:** `{exp_item.get('base_value')}`")
            
            drivers = exp_item.get("top_drivers", [])
            if drivers:
                top_3 = ", ".join([f"**{d['feature']}** ({d['impact']:+.2f})" for d in drivers[:3]])
                st.info(f"💡 **{selected_target}** prediction was primarily driven by: {top_3}")
                
            fig_shap = create_shap_bar_chart(exp_item)
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True)

    with tab_features:
        st.markdown(f"##### 🔬 EDA & Process Feature Intelligence — Batch: `{res.get('batch_id')}`")
        st.caption("Visualizing the relationship between engineered process features and historical factory distributions.")
        
        hist_feats, _, _, in_map = get_historical_correlation_data()
        feats_dict = res.get("engineered_features", {})
        
        if hist_feats is not None:
            available_feats = [c for c in feats_dict.keys() if c in hist_feats.columns]
            friendly_options = {c: in_map.get(c, c.replace('_', ' ').title()) for c in available_feats}
            
            sel_feat_col = st.selectbox(
                "Select Engineered Feature to Inspect Factory Distribution & Current Batch Value:",
                options=available_feats,
                format_func=lambda x: friendly_options.get(x, x),
                index=0
            )
            
            if sel_feat_col:
                curr_val = float(feats_dict.get(sel_feat_col, 0.0))
                fig_dist, stats = create_feature_distribution_plot(
                    hist_feats[sel_feat_col],
                    curr_val,
                    friendly_options.get(sel_feat_col, sel_feat_col),
                    res.get("batch_id", "Current Batch")
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.metric("Current Batch Value", f"{curr_val:g}")
                with d2:
                    st.metric("Factory Median", f"{stats['median']:g}")
                with d3:
                    st.metric("Optimal Range (P25 - P75)", f"[{stats['p25']:g} - {stats['p75']:g}]")
                with d4:
                    status_feat = "Optimal (P25-P75)" if stats['p25'] <= curr_val <= stats['p75'] else ("Acceptable (P10-P90)" if stats['p10'] <= curr_val <= stats['p90'] else "Outlier Deviation")
                    st.metric("Batch Adherence", status_feat)
                    
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 📋 All 45 Process Engineered Feature Values")
        df_f = pd.DataFrame([{"Feature Variable": k, "Batch Value": v} for k, v in feats_dict.items()])
        st.dataframe(df_f, use_container_width=True, height=280)

    # Downloadable Reports Center
    st.divider()
    st.markdown("<div class='section-header'>📥 Export Industrial Reports</div>", unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)
    if LOCAL_BACKEND_AVAILABLE:
        try:
            pdf_path = generate_pdf_batch_report(res)
            with open(pdf_path, "rb") as f:
                with col_r1:
                    st.download_button("📄 Download PDF Batch Report", data=f.read(), file_name=os.path.basename(pdf_path), mime="application/pdf", use_container_width=True)
                    
            excel_path = generate_excel_batch_report(res)
            with open(excel_path, "rb") as f:
                with col_r2:
                    st.download_button("📊 Download Excel Workbook", data=f.read(), file_name=os.path.basename(excel_path), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                    
            csv_path = generate_csv_batch_report(res)
            with open(csv_path, "rb") as f:
                with col_r3:
                    st.download_button("📑 Download CSV Batch Record", data=f.read(), file_name=os.path.basename(csv_path), mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Error preparing export reports: {e}")

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 20px; color: #94a3b8; font-size: 12px; border-top: 1px solid #e2e8f0;'>
    Rubber Quality Prediction PoC | Industrial SCADA Intelligence Architecture
</div>
""", unsafe_allow_html=True)
