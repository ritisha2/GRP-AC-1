"""
UI Theme & Custom Styling
Pure White / Light Industrial SCADA Theme with high-contrast typography,
clean borders, aligned cards, and crisp metrics.
"""

def apply_industrial_theme():
    return """
    <style>
    /* Typography & Font Imports */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    code, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Hide Sidebar Completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    button[data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Top Header Bar */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    
    .logo-text {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
    }
    .logo-accent {
        color: #2563eb;
    }
    
    .sub-badge {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Live Process Telemetry Strip */
    .telemetry-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03);
    }
    
    .telem-item {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 6px 12px;
        border-left: 3px solid #e2e8f0;
    }
    .telem-item:first-child {
        border-left: 3px solid #2563eb;
    }
    
    .telem-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .telem-value {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Section Headings and Group Titles */
    .section-header {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .card-title {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #1e293b;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Form Inputs Overrides for Perfect White SCADA Look */
    div[data-testid="stForm"], div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    
    .stTextInput input, .stNumberInput input, .stTimeInput input {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTimeInput input:focus {
        border-color: #2563eb !important;
        background-color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    
    label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 2px !important;
    }

    /* Hero Golden Score Card */
    .golden-hero-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .golden-hero-score {
        font-size: 58px;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
        margin: 6px 0 12px 0;
        line-height: 1;
    }
    
    .status-pill {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .status-golden {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    
    .status-warning {
        background: #fef9c3;
        color: #a16207;
        border: 1px solid #fde047;
    }
    
    .status-nongolden {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }

    /* Sub-metrics inside hero card */
    .hero-submetrics {
        margin-top: 16px;
        display: flex;
        justify-content: center;
        gap: 32px;
        font-size: 13px;
        color: #475569;
        font-weight: 500;
    }
    .hero-submetrics strong {
        color: #0f172a;
        font-size: 14px;
    }

    /* 11 Quality Parameter Cards (Consistent Grid) */
    .param-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 12px;
        transition: all 0.15s ease-in-out;
    }
    .param-card:hover {
        border-color: #94a3b8;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    }
    
    .param-name {
        font-size: 12px;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .param-val {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        margin: 4px 0;
    }
    
    .param-spec {
        font-size: 11px;
        color: #64748b;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px dashed #e2e8f0;
        padding-top: 6px;
        margin-top: 4px;
    }
    
    .param-status-badge {
        font-size: 10px;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
    }
    .badge-ok {
        background: #dcfce7;
        color: #15803d;
    }
    .badge-out {
        background: #fee2e2;
        color: #b91c1c;
    }

    /* Primary Action Button */
    div.stButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }

    /* Download Buttons */
    div.stDownloadButton > button {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }
    div.stDownloadButton > button:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #64748b !important;
        border-radius: 6px 6px 0 0 !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        background: #ffffff !important;
        border-top: 3px solid #2563eb !important;
        font-weight: 700 !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* Card Containers */
    .content-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
    }

    /* Clean Dividers */
    hr {
        border: none !important;
        height: 1px !important;
        background-color: #e2e8f0 !important;
        margin: 24px 0 !important;
    }
    </style>
    """
