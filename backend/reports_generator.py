"""
ADVaIT Reports Generator
Generates comprehensive industrial batch reports:
- PDF Batch Report (ReportLab)
- Excel Multi-tab Report (openpyxl)
- CSV Batch Report
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_radar_image(radar_data: Dict[str, Dict[str, float]], output_path: str):
    """Generates a static radar chart image for embedding into PDF."""
    categories = [k.replace('_', ' ').title() for k in radar_data.keys()]
    current_vals = [v["current"] for v in radar_data.values()]
    golden_vals = [v["golden_median"] for v in radar_data.values()]
    
    # Normalize for radar visual comparison
    norm_current = []
    norm_golden = []
    for c, g in zip(current_vals, golden_vals):
        base = max(abs(g), 1.0)
        norm_current.append(min(2.0, max(0.0, c / base)))
        norm_golden.append(1.0)
        
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    norm_current += norm_current[:1]
    norm_golden += norm_golden[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 3.8), subplot_kw=dict(polar=True), facecolor='#ffffff')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=8, color='#334155')
    ax.set_rlabel_position(0)
    plt.yticks([0.5, 1.0, 1.5], ["50%", "Golden (100%)", "150%"], color="#94a3b8", size=7)
    plt.ylim(0, 2.0)
    
    # Golden median reference line
    ax.plot(angles, norm_golden, linewidth=1.5, linestyle='dashed', color='#2563eb', label='Golden Median')
    ax.fill(angles, norm_golden, '#3b82f6', alpha=0.15)
    
    # Current batch
    ax.plot(angles, norm_current, linewidth=2, linestyle='solid', color='#16a34a', label='Current Batch')
    ax.fill(angles, norm_current, '#22c55e', alpha=0.25)
    
    plt.title("Process Radar vs Golden Batch Median", size=10, weight='bold', color='#0f172a', pad=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close()

def generate_pdf_batch_report(prediction_result: Dict[str, Any], output_filename: str = None) -> str:
    batch_id = prediction_result.get("batch_id", "BT-BATCH")
    safe_id = batch_id.replace("/", "_").replace("\\", "_")
    if not output_filename:
        output_filename = f"report_{safe_id}.pdf"

        
    pdf_path = os.path.join(REPORTS_DIR, output_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=32, rightMargin=32, topMargin=32, bottomMargin=32)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A')
    )
    sub_style = ParagraphStyle(
        'SubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569')
    )
    sec_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=4
    )

    story.append(Paragraph("ADVaIT — Industrial BT Rubber Quality Batch Report", title_style))
    story.append(Paragraph(f"Batch ID: <b>{batch_id}</b> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Process: AC-10 Autoclave", sub_style))
    story.append(Spacer(1, 10))

    # Header Summary Cards Table
    golden_score = prediction_result.get("golden_score", 0.0)
    status = prediction_result.get("status", "Non-Golden")
    status_bg = colors.HexColor('#DCFCE7') if status == 'Golden' else (colors.HexColor('#FEF9C3') if status == 'Warning' else colors.HexColor('#FEE2E2'))
    status_fg = colors.HexColor('#166534') if status == 'Golden' else (colors.HexColor('#854D0E') if status == 'Warning' else colors.HexColor('#991B1B'))

    summary_data = [
        ["Batch Evaluation", "Golden Batch Score", "Process Adherence", "Quality Score"],
        [status.upper(), f"{golden_score:.1f}%", f"{prediction_result.get('golden_evaluation', {}).get('process_score', 0):.1f}%", f"{prediction_result.get('golden_evaluation', {}).get('quality_score', 0):.1f}%"]
    ]
    t_summary = Table(summary_data, colWidths=[130, 130, 140, 140])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (0,1), (0,1), status_bg),
        ('TEXTCOLOR', (0,1), (0,1), status_fg),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))

    # 11 Predicted Quality Values Table
    story.append(Paragraph("11 Predicted Laboratory Quality Parameters", sec_style))
    qual_rows = [["Parameter", "Predicted Value", "Specification (LSL - USL)", "Target", "Status"]]
    predictions = prediction_result.get("predictions", {})
    for p_name, item in predictions.items():
        val_str = f"{item['predicted_value']:.2f} {item['unit']}"
        range_str = f"[{item['lsl']} - {item['usl']}]"
        targ_str = f"{item['target']}"
        status_str = item['status']
        qual_rows.append([p_name, val_str, range_str, targ_str, status_str])

    t_qual = Table(qual_rows, colWidths=[120, 110, 140, 70, 100])
    t_qual.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_qual)
    story.append(Spacer(1, 10))

    # Radar Chart & Timeline side-by-side or stacked
    radar_img_path = os.path.join(REPORTS_DIR, f"temp_radar_{safe_id}.png")
    radar_data = prediction_result.get("golden_evaluation", {}).get("radar_comparison", {})
    if radar_data:
        generate_radar_image(radar_data, radar_img_path)
        story.append(Paragraph("Process Adherence Radar Comparison", sec_style))
        story.append(Image(radar_img_path, width=280, height=210))
        story.append(Spacer(1, 10))

    # Process Timeline Table
    story.append(Paragraph("Batch Process Timeline Milestones", sec_style))
    time_events = prediction_result.get("timeline_events", [])
    tl_data = [["Milestone Stage", "Clock Time", "Elapsed Minutes from Start"]]
    for ev in time_events:
        tl_data.append([ev["stage"], ev["clock_time"], f"+{ev['elapsed_minutes']} min"])
    t_tl = Table(tl_data, colWidths=[180, 160, 200])
    t_tl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_tl)
    story.append(Spacer(1, 10))

    # Key Engineered Features Table
    story.append(Paragraph("Key Industrial Engineered Process Features", sec_style))
    feats = prediction_result.get("engineered_features", {})
    key_feat_names = [
        "steam_duration", "heating_rate", "steam_temp_rise", "steam_pressure_rise",
        "cooking_duration_verified", "cooking_ratio", "cooling_delay", "cooling_rate",
        "thermal_exposure_score", "low_tf_ratio", "batch_duration_deviation", "total_batch_duration"
    ]
    kf_rows = [["Feature Name", "Calculated Value", "Feature Name", "Calculated Value"]]
    for i in range(0, len(key_feat_names), 2):
        f1 = key_feat_names[i]
        v1 = feats.get(f1, "N/A")
        f2 = key_feat_names[i+1] if i+1 < len(key_feat_names) else ""
        v2 = feats.get(f2, "") if f2 else ""
        kf_rows.append([f1.replace('_', ' ').title(), str(v1), f2.replace('_', ' ').title(), str(v2)])

    t_kf = Table(kf_rows, colWidths=[150, 120, 150, 120])
    t_kf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_kf)

    doc.build(story)
    
    # Clean up temp radar image
    if os.path.exists(radar_img_path):
        try:
            os.remove(radar_img_path)
        except Exception:
            pass
            
    print(f"Generated PDF Batch Report: {pdf_path}")
    return pdf_path

def generate_excel_batch_report(prediction_result: Dict[str, Any], output_filename: str = None) -> str:
    batch_id = prediction_result.get("batch_id", "BT-BATCH")
    safe_id = batch_id.replace("/", "_").replace("\\", "_")
    if not output_filename:
        output_filename = f"report_{safe_id}.xlsx"
        
    excel_path = os.path.join(REPORTS_DIR, output_filename)
    
    # Sheet 1: Batch Summary & Quality Predictions
    qual_list = []
    for p_name, item in prediction_result.get("predictions", {}).items():
        qual_list.append({
            "Target Quality Parameter": p_name,
            "Predicted Value": item["predicted_value"],
            "Unit": item["unit"],
            "LSL": item["lsl"],
            "Target": item["target"],
            "USL": item["usl"],
            "Status": item["status"],
            "Score": item["score"]
        })
    df_qual = pd.DataFrame(qual_list)
    
    # Sheet 2: All 45 Engineered Process Features
    feats_list = [{"Feature": k, "Engineered Value": v} for k, v in prediction_result.get("engineered_features", {}).items()]
    df_feats = pd.DataFrame(feats_list)
    
    # Sheet 3: Timeline Events
    df_timeline = pd.DataFrame(prediction_result.get("timeline_events", []))
    
    # Sheet 4: Golden Batch & Root Cause Penalties
    penalties = prediction_result.get("golden_evaluation", {}).get("penalties", [])
    df_penalties = pd.DataFrame(penalties) if penalties else pd.DataFrame([{"Notice": "No significant deviations observed"}])
    
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_qual.to_excel(writer, sheet_name="Quality Predictions", index=False)
        df_feats.to_excel(writer, sheet_name="Engineered Features", index=False)
        df_timeline.to_excel(writer, sheet_name="Timeline Milestones", index=False)
        df_penalties.to_excel(writer, sheet_name="Golden Batch Evaluation", index=False)
        
    print(f"Generated Excel Batch Report: {excel_path}")
    return excel_path

def generate_csv_batch_report(prediction_result: Dict[str, Any], output_filename: str = None) -> str:
    batch_id = prediction_result.get("batch_id", "BT-BATCH")
    safe_id = batch_id.replace("/", "_").replace("\\", "_")
    if not output_filename:
        output_filename = f"report_{safe_id}.csv"
        
    csv_path = os.path.join(REPORTS_DIR, output_filename)
    
    flat_data = {
        "batch_id": batch_id,
        "golden_score": prediction_result.get("golden_score"),
        "status": prediction_result.get("status")
    }
    # Add predictions
    for p_name, item in prediction_result.get("predictions", {}).items():
        flat_data[f"pred_{p_name}"] = item["predicted_value"]
        flat_data[f"status_{p_name}"] = item["status"]
    # Add features
    for k, v in prediction_result.get("engineered_features", {}).items():
        flat_data[f"feat_{k}"] = v
        
    df_flat = pd.DataFrame([flat_data])
    df_flat.to_csv(csv_path, index=False)
    print(f"Generated CSV Batch Report: {csv_path}")
    return csv_path
