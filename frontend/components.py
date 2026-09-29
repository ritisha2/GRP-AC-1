"""
UI Components (Light / White Industrial SCADA Theme)
Interactive Plotly visualizers:
- Industrial Quality Gauges with Lower/Target/Upper specification zones
- Process Radar Chart (Current Batch vs. Golden Batch Median)
- Interactive Process Timeline (Milestones & Stage Delays)
- SHAP Impact attribution bar charts
- Real-time Live Process Intelligence Bar
"""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def calculate_live_telemetry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates instantaneous process telemetry for the Live Intelligence Bar."""
    t_steam_open = float(data.get("steam_valve_open_temp", 0.0) or 0.0)
    t_steam_close = float(data.get("steam_valve_close_temp", 0.0) or 0.0)
    temp_rise = max(0.0, t_steam_close - t_steam_open)

    def parse_m(t_str):
        if not t_str or ":" not in str(t_str):
            return 0.0
        parts = str(t_str).split(":")
        try:
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return 0.0

    m_steam_open = parse_m(data.get("steam_valve_open_time"))
    m_steam_close = parse_m(data.get("steam_valve_close_time"))

    steam_dur = m_steam_close - m_steam_open
    if steam_dur < 0:
        steam_dur += 1440.0

    heating_rate = temp_rise / max(steam_dur, 1.0)
    
    std_dur = float(data.get("standard_batch_duration", 400.0) or 400.0)
    act_dur = float(data.get("actual_batch_duration", 395.0) or 395.0)
    deviation = act_dur - std_dur

    # Status indicators
    steam_status = "🟢" if 160 <= steam_dur <= 240 else ("🟡" if 130 <= steam_dur <= 280 else "🔴")
    temp_status = "🟢" if 55 <= temp_rise <= 100 else ("🟡" if 40 <= temp_rise <= 115 else "🔴")
    rate_status = "🟢" if 0.25 <= heating_rate <= 0.65 else ("🟡" if 0.15 <= heating_rate <= 0.85 else "🔴")
    dev_status = "🟢" if abs(deviation) <= 15 else ("🟡" if abs(deviation) <= 40 else "🔴")

    return {
        "steam_duration": round(steam_dur, 1),
        "steam_status": steam_status,
        "temp_rise": round(temp_rise, 1),
        "temp_status": temp_status,
        "heating_rate": round(heating_rate, 3),
        "rate_status": rate_status,
        "batch_deviation": round(deviation, 1),
        "dev_status": dev_status
    }

def create_quality_gauge(param_name: str, pred_val: float, lsl: float, target: float, usl: float, unit: str):
    """
    Renders an industrial Plotly gauge with LSL, Target, USL tolerance bands on a crisp white card.
    """
    min_axis = min(lsl * 0.85, pred_val * 0.85)
    max_axis = max(usl * 1.15, pred_val * 1.15)
    
    # Status color
    if lsl <= pred_val <= usl:
        bar_color = "#16a34a" # Emerald Green
    elif pred_val < lsl:
        bar_color = "#dc2626" # Red
    else:
        bar_color = "#d97706" # Amber

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_val,
        delta={'reference': target, 'increasing': {'color': "#2563eb"}, 'decreasing': {'color': "#7c3aed"}},
        title={
            'text': f"<b>{param_name}</b><br><span style='font-size:0.75em;color:#64748b;font-weight:500;'>Target: {target} {unit}</span>",
            'font': {'size': 13, 'color': '#0f172a', 'family': 'Inter'}
        },
        number={'suffix': f" {unit}", 'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#0f172a'}},
        gauge={
            'axis': {
                'range': [min_axis, max_axis],
                'tickwidth': 1,
                'tickcolor': "#94a3b8",
                'tickfont': {'size': 9, 'color': '#64748b', 'family': 'JetBrains Mono'}
            },
            'bar': {'color': bar_color, 'thickness': 0.26},
            'bgcolor': "#f1f5f9",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [min_axis, lsl], 'color': 'rgba(239, 68, 68, 0.12)'},
                {'range': [lsl, usl], 'color': 'rgba(34, 197, 94, 0.14)'},
                {'range': [usl, max_axis], 'color': 'rgba(239, 68, 68, 0.12)'}
            ],
            'threshold': {
                'line': {'color': "#2563eb", 'width': 3},
                'thickness': 0.8,
                'value': target
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font={'color': "#0f172a", 'family': 'Inter'},
        margin=dict(l=15, r=15, t=40, b=15),
        height=185
    )
    return fig

def create_radar_chart(radar_data: Dict[str, Dict[str, float]]):
    """
    Creates an interactive Process Radar comparing Current Batch vs Golden Batch Median on a clean white background.
    """
    categories = [k.replace('_', ' ').title() for k in radar_data.keys()]
    current_vals = [v["current"] for v in radar_data.values()]
    golden_vals = [v["golden_median"] for v in radar_data.values()]

    # Normalize relative to Golden Median for uniform radar scale
    norm_cur = []
    norm_gold = []
    for c, g in zip(current_vals, golden_vals):
        base = max(abs(g), 1e-4)
        norm_cur.append(round(min(2.0, max(0.0, c / base)), 3))
        norm_gold.append(1.0)

    # Close the loop
    cat_loop = categories + [categories[0]]
    cur_loop = norm_cur + [norm_cur[0]]
    gold_loop = norm_gold + [norm_gold[0]]

    fig = go.Figure()

    # Golden Median reference
    fig.add_trace(go.Scatterpolar(
        r=gold_loop,
        theta=cat_loop,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.08)',
        line=dict(color='#2563eb', width=2, dash='dash'),
        name='Golden Batch Median (100%)'
    ))

    # Current Batch
    fig.add_trace(go.Scatterpolar(
        r=cur_loop,
        theta=cat_loop,
        fill='toself',
        fillcolor='rgba(22, 163, 74, 0.20)',
        line=dict(color='#16a34a', width=2.5),
        name='Current Batch'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#ffffff',
            radialaxis=dict(
                visible=True,
                range=[0, 1.8],
                tickvals=[0.5, 1.0, 1.5],
                ticktext=["50%", "Golden (100%)", "150%"],
                tickfont=dict(size=9, color="#64748b", family='Inter'),
                gridcolor="#e2e8f0"
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#0f172a", family='Inter', weight='bold'),
                gridcolor="#e2e8f0"
            )
        ),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color="#334155", size=11, family='Inter')
        ),
        margin=dict(l=40, r=40, t=30, b=40),
        height=340
    )
    return fig

def create_timeline_chart(timeline_events: List[Dict[str, Any]]):
    """
    Renders an industrial milestone timeline chart on a clean white background.
    """
    stages = [e["stage"] for e in timeline_events]
    elapsed = [e["elapsed_minutes"] for e in timeline_events]
    clocks = [e["clock_time"] for e in timeline_events]
    
    # Custom light theme palette for milestone nodes
    stage_colors = {
        "Start": "#2563eb",
        "Steam Open": "#d97706",
        "Steam Close": "#ea580c",
        "Cooking": "#7c3aed",
        "Cooling": "#0284c7",
        "Bottom Door": "#059669",
        "End": "#16a34a"
    }
    colors = [stage_colors.get(s, "#2563eb") for s in stages]

    fig = go.Figure()

    # Timeline connector line
    fig.add_trace(go.Scatter(
        x=elapsed,
        y=[0] * len(elapsed),
        mode='lines',
        line=dict(color='#cbd5e1', width=5),
        hoverinfo='skip',
        showlegend=False
    ))

    # Stage milestone nodes
    fig.add_trace(go.Scatter(
        x=elapsed,
        y=[0] * len(elapsed),
        mode='markers+text',
        marker=dict(
            size=22,
            color=colors,
            line=dict(color='#ffffff', width=3)
        ),
        text=[f"<b>{s}</b><br><span style='color:#64748b;font-size:0.9em'>{c}</span><br><b style='color:#0f172a'>+{e}m</b>" for s, c, e in zip(stages, clocks, elapsed)],
        textposition="top center",
        textfont=dict(size=11, color='#0f172a', family='Inter'),
        hoverinfo='text',
        hovertext=[f"Stage: {s}<br>Time: {c}<br>Elapsed: {e} mins" for s, c, e in zip(stages, clocks, elapsed)],
        showlegend=False
    ))

    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        xaxis=dict(
            title=dict(text="Elapsed Minutes from Batch Start", font=dict(size=11, color="#64748b", family='Inter')),
            tickfont=dict(size=10, color="#64748b", family='JetBrains Mono'),
            gridcolor="#f1f5f9",
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            range=[-0.4, 0.9]
        ),
        margin=dict(l=20, r=20, t=60, b=30),
        height=200
    )
    return fig

def create_shap_bar_chart(explain_item: Dict[str, Any]):
    """
    Creates a Plotly horizontal impact chart for SHAP feature drivers on a clean white background.
    """
    drivers = explain_item.get("top_drivers", [])
    if not drivers:
        return None

    features = [d["feature"].replace("_", " ").title() for d in drivers[::-1]]
    impacts = [d["impact"] for d in drivers[::-1]]
    pcts = [d["impact_pct"] for d in drivers[::-1]]
    bar_colors = ['#16a34a' if imp >= 0 else '#dc2626' for imp in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{imp:+.2f} ({pct}%)" for imp, pct in zip(impacts, pcts)],
        textposition="outside",
        textfont=dict(size=10, color="#0f172a", family='JetBrains Mono')
    ))

    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(color="#334155", family='Inter'),
        xaxis=dict(
            title=dict(text="SHAP Feature Contribution to Predicted Value", font=dict(size=11, color="#64748b")),
            gridcolor="#f1f5f9",
            zerolinecolor="#cbd5e1"
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#0f172a", weight='bold')
        ),
        margin=dict(l=20, r=50, t=10, b=30),
        height=230
    )
    return fig

def create_input_output_correlation_heatmap(
    corr_df: pd.DataFrame,
    batch_inputs_dict: Dict[str, float],
    batch_preds_dict: Dict[str, float],
    batch_id: str = "Current Batch"
):
    """
    Renders an industrial Correlation Heatmap:
    - Y-axis: Model Inputs (Process variables)
    - X-axis: Model Outputs (11 Quality Parameters)
    - Cell values: Pearson r correlation coefficients (-1.0 to +1.0)
    - Custom hover template showing the current batch's input value, predicted output value, and correlation impact.
    """
    x_labels = list(corr_df.columns)
    y_labels = list(corr_df.index)
    z_values = corr_df.values

    custom_hover = []
    text_annotations = []
    for i, y_name in enumerate(y_labels):
        row_hover = []
        row_text = []
        curr_in_val = batch_inputs_dict.get(y_name, 0.0)
        for j, x_name in enumerate(x_labels):
            r_val = float(z_values[i][j]) if not pd.isna(z_values[i][j]) else 0.0
            curr_pred_val = batch_preds_dict.get(x_name, 0.0)
            
            if abs(r_val) >= 0.5:
                strength = "Strong"
            elif abs(r_val) >= 0.2:
                strength = "Moderate"
            else:
                strength = "Weak"
            direction = "Positive" if r_val >= 0 else "Negative"
            
            hover_str = (
                f"<b>Batch: {batch_id}</b><br><br>"
                f"<b>Model Input (Y):</b> {y_name}<br>"
                f"<b>Current Input Value:</b> {curr_in_val}<br><br>"
                f"<b>Model Output (X):</b> {x_name}<br>"
                f"<b>Predicted Quality Value:</b> {curr_pred_val}<br><br>"
                f"<b>Correlation (r):</b> {r_val:+.2f} ({strength} {direction})<br>"
                f"<i>{'Increasing this input tends to increase ' if r_val >= 0 else 'Increasing this input tends to decrease '}{x_name}</i>"
            )
            row_hover.append(hover_str)
            row_text.append(f"{r_val:+.2f}")
        custom_hover.append(row_hover)
        text_annotations.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono", color="#0f172a"),
        hoverinfo="text",
        hovertext=custom_hover,
        colorscale="RdBu_r",
        zmid=0.0,
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(
            title=dict(text="<b>Correlation (r)</b>", font=dict(size=11, color="#475569")),
            tickfont=dict(size=10, family="JetBrains Mono", color="#475569"),
            thickness=14,
            len=0.9
        )
    ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(
            text=f"<b>Model Input (Y-Axis) vs. Model Output (X-Axis) Correlation Heatmap — Batch: {batch_id}</b>",
            font=dict(size=14, color="#0f172a", family="Inter")
        ),
        xaxis=dict(
            title=dict(text="<b>Model Outputs (11 Quality Parameters)</b>", font=dict(size=12, color="#1e293b", family="Inter")),
            tickfont=dict(size=11, color="#1e293b", family="Inter", weight="bold"),
            side="top",
            gridcolor="#f1f5f9"
        ),
        yaxis=dict(
            title=dict(text="<b>Model Inputs (Process & Recipe Variables)</b>", font=dict(size=12, color="#1e293b", family="Inter")),
            tickfont=dict(size=11, color="#1e293b", family="Inter"),
            autorange="reversed",
            gridcolor="#f1f5f9"
        ),
        margin=dict(l=200, r=40, t=80, b=40),
        height=max(480, len(y_labels) * 32)
    )
    return fig

def create_input_output_scatter_plot(
    df_x_history: pd.Series,
    df_y_history: pd.Series,
    input_name: str,
    output_name: str,
    batch_input_val: float,
    batch_pred_val: float,
    batch_id: str,
    spec: Dict[str, Any] = None
):
    """
    Renders an interactive scatter plot:
    - X-axis: Model Output (Quality target scale)
    - Y-axis: Model Input (Process feature scale)
    - Background points: Historical factory batches
    - Trendline: Correlation regression curve
    - Highlighted operating point: Currently predicted batch ID
    """
    fig = go.Figure()

    if spec:
        lsl, usl = spec.get("lsl", 0.0), spec.get("usl", 100.0)
        fig.add_vrect(
            x0=lsl, x1=usl,
            fillcolor="rgba(34, 197, 94, 0.08)",
            line=dict(color="rgba(34, 197, 94, 0.4)", width=1, dash="dot"),
            annotation_text="In-Spec Range",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#15803d")
        )

    fig.add_trace(go.Scatter(
        x=df_x_history,
        y=df_y_history,
        mode="markers",
        name="Historical Factory Batches",
        marker=dict(size=7, color="#94a3b8", opacity=0.6, line=dict(width=1, color="#64748b")),
        hovertemplate="<b>Historical Batch</b><br>" + f"{output_name} (Output): " + "%{x:.2f}<br>" + f"{input_name} (Input): " + "%{y:.2f}<extra></extra>"
    ))

    valid = ~(df_x_history.isna() | df_y_history.isna() | np.isinf(df_x_history) | np.isinf(df_y_history))
    x_clean = df_x_history[valid]
    y_clean = df_y_history[valid]
    if len(x_clean) > 5 and np.std(x_clean) > 1e-4:
        poly = np.polyfit(x_clean, y_clean, 1)
        x_line = np.linspace(float(x_clean.min()), float(x_clean.max()), 50)
        y_line = np.polyval(poly, x_line)
        corr_coef = float(np.corrcoef(x_clean, y_clean)[0, 1])
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Correlation Trend (r = {corr_coef:+.2f})",
            line=dict(color="#2563eb", width=2, dash="dash")
        ))
    else:
        corr_coef = 0.0

    fig.add_trace(go.Scatter(
        x=[batch_pred_val],
        y=[batch_input_val],
        mode="markers+text",
        name=f"Current Batch: {batch_id}",
        text=[f"🎯 <b>{batch_id}</b>"],
        textposition="top center",
        textfont=dict(size=12, color="#b91c1c", family="Inter"),
        marker=dict(size=16, color="#ef4444", symbol="diamond", line=dict(width=2, color="#ffffff")),
        hovertemplate=(
            f"<b>Current Batch: {batch_id}</b><br><br>"
            f"<b>Model Output (X):</b> {output_name} = {batch_pred_val:.2f}<br>"
            f"<b>Model Input (Y):</b> {input_name} = {batch_input_val:.2f}<extra></extra>"
        )
    ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(
            text=f"<b>Correlation Relationship: {input_name} (Y) vs. {output_name} (X) — Batch: {batch_id}</b>",
            font=dict(size=13, color="#0f172a", family="Inter")
        ),
        xaxis=dict(
            title=dict(text=f"<b>Model Output: {output_name}</b>", font=dict(size=11, color="#1e293b", family="Inter")),
            gridcolor="#f1f5f9",
            tickfont=dict(size=10, color="#64748b", family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(text=f"<b>Model Input: {input_name}</b>", font=dict(size=11, color="#1e293b", family="Inter")),
            gridcolor="#f1f5f9",
            tickfont=dict(size=10, color="#64748b", family="JetBrains Mono")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig, corr_coef

def create_feature_distribution_plot(
    series: pd.Series,
    current_val: float,
    feature_name: str,
    batch_id: str
):
    """
    Renders an EDA Distribution Plot (Histogram + Percentile Bands) for any engineered process feature,
    marking the currently predicted batch's operating value.
    """
    clean_s = series.dropna()
    clean_s = clean_s[~np.isinf(clean_s)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=clean_s,
        name="Historical Distribution",
        nbinsx=25,
        marker=dict(color="rgba(37, 99, 235, 0.4)", line=dict(color="#2563eb", width=1)),
        hovertemplate="Value: %{x}<br>Batch Count: %{y}<extra></extra>"
    ))
    
    p10 = float(np.percentile(clean_s, 10))
    p25 = float(np.percentile(clean_s, 25))
    median = float(np.median(clean_s))
    p75 = float(np.percentile(clean_s, 75))
    p90 = float(np.percentile(clean_s, 90))
    
    fig.add_vrect(
        x0=p25, x1=p75,
        fillcolor="rgba(34, 197, 94, 0.12)",
        line=dict(color="rgba(34, 197, 94, 0.5)", width=1, dash="dot"),
        annotation_text="Optimal P25-P75",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#15803d")
    )
    
    fig.add_vline(
        x=current_val,
        line=dict(color="#ef4444", width=3, dash="dash"),
        annotation_text=f"Current: {current_val:g}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#b91c1c", family="Inter")
    )
    
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(
            text=f"<b>Historical Distribution: {feature_name} — Batch: {batch_id}</b>",
            font=dict(size=13, color="#0f172a", family="Inter")
        ),
        xaxis=dict(
            title=dict(text=f"<b>{feature_name}</b>", font=dict(size=11, color="#1e293b")),
            gridcolor="#f1f5f9",
            tickfont=dict(size=10, color="#64748b", family="JetBrains Mono")
        ),
        yaxis=dict(
            title=dict(text="<b>Frequency (Batch Count)</b>", font=dict(size=11, color="#1e293b")),
            gridcolor="#f1f5f9",
            tickfont=dict(size=10, color="#64748b", family="JetBrains Mono")
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        showlegend=False
    )
    return fig, {"p10": p10, "p25": p25, "median": median, "p75": p75, "p90": p90}
