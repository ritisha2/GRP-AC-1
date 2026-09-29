# Rubber Batch Quality Prediction System

An end-to-end intelligent quality monitoring and prediction system for industrial rubber batch processing. The system analyzes real-time autoclave process data (temperatures, pressures, and timing), predicts laboratory quality metrics, evaluates process adherence against golden batch standards, and provides real-time dashboards and reports.

---

## Overview

In industrial rubber manufacturing, batch quality parameters are traditionally verified through lab tests after the curing process. This system predicts key laboratory quality parameters during or immediately after the batch run, enabling operators to:
- Predict **11 key quality metrics** before lab results arrive.
- Monitor process parameters across critical heating, cooking, and cooling stages.
- Benchmark batch performance using a **Golden Batch Score (0–100%)**.
- Export batch quality reports in **PDF**, **Excel**, and **CSV** formats.

---

## Key Features

- **FastAPI Backend**: Fast REST API for making predictions, downloading reports, and inspecting statistical profiles.
- **Interactive Streamlit Dashboard**: User-friendly UI for loading sample batches, entering custom process values, and viewing real-time quality scores.
- **Automated Feature Engineering**: Automatically derives process rates, thermal exposure indices, pressure stability, and timing deviations from raw process inputs.
- **Golden Batch Scoring**: Calculates process adherence and quality compliance to classify batches into **Golden**, **Warning**, or **Non-Golden** categories.
- **Multi-Format Export**: Generates industrial PDF reports, detailed multi-sheet Excel files, and CSV summaries.

---

## Technical Architecture

```
                       ┌────────────────────────────┐
                       │   Process Data (CSV/Excel) │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │    Data Loader & Preproc   │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │    Feature Engineering     │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                     ┌───────────────┴───────────────┐
                     │                               │
                     ▼                               ▼
       ┌───────────────────────────┐   ┌───────────────────────────┐
       │   FastAPI Backend Server  │   │ ML Quality Predictors     │
       │   (REST Endpoints & Export)│◄──┤ (11 Quality Parameters)   │
       └─────────────┬─────────────┘   └───────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Streamlit Web Dashboard   │
       └───────────────────────────┘
```

---

## Project Structure

```
├── backend/
│   ├── app.py                      # FastAPI REST application
│   ├── predict.py                  # Batch prediction engine
│   ├── feature_engineering.py      # Feature engineering module
│   ├── preprocessing.py            # Data cleaning & validation
│   ├── golden_batch.py             # Golden Batch statistical scoring
│   ├── reports_generator.py        # PDF, Excel, and CSV exporter
│   ├── train.py                    # Model training pipeline
│   ├── data_loader.py              # Raw data extraction utility
│   ├── dataset_intelligence.py     # Dataset profiling utility
│   ├── data/                       # CSV datasets and test batch files
│   ├── models/                     # Trained ML model artifacts
│   └── reports/                    # Exported batch reports
├── frontend/
│   ├── dashboard.py                # Streamlit dashboard application
│   ├── components.py               # Reusable UI components & metrics
│   └── theme.py                    # Custom UI styles & formatting
├── shared/
│   ├── constants.py                # Parameter limits & target definitions
│   └── schemas.py                  # Pydantic data models
├── docs/                           # Extended documentation
├── tests/                          # Automated unit test suite
├── requirements.txt                # Python package dependencies
└── README.md                       # Project overview & instructions
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ritisha2/GRP-AC-1.git
   cd GRP-AC-1
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### 1. Start the Backend API

Launch the FastAPI server:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Once running, interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the Frontend Dashboard

In a separate terminal, launch the Streamlit app:
```bash
python -m streamlit run frontend/dashboard.py --server.port 8501
```

Open your browser and navigate to `http://localhost:8501`.

---

## API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check and model status |
| `POST` | `/predict` | Evaluates process parameters and returns 11 predicted quality metrics |
| `GET` | `/golden-profile` | Returns statistical baseline ranges (P10, P25, Median, P75, P90) |
| `GET` | `/sample-batches` | Provides pre-loaded sample batch data for testing |
| `POST` | `/export/pdf` | Downloads an industrial PDF batch report |
| `POST` | `/export/excel` | Downloads a multi-sheet Excel report |
| `POST` | `/export/csv` | Downloads a CSV export of the batch results |

---

## Running Tests

Run the test suite to verify system functionality:
```bash
python -m unittest discover -s tests
```

---

## License

Distributed under standard project terms.
