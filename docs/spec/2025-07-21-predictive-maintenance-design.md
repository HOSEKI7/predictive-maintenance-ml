# Design Spec: Predictive Maintenance / Anomaly Detection System

## Problem
CS student needs a portfolio project relevant to Industrial IoT / AI Engineering role within 3 days.

## Solution
Predictive Maintenance system using AI4I 2020 dataset with two detection layers:
- XGBoost classifier for known failure types
- Isolation Forest for unsupervised anomaly detection

Combined into 3-level risk indicator (green/yellow/red) on a Streamlit dashboard with real-time simulation from historical data.

## Dataset
- **Source:** AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- **Rows:** 10,000 | **Features:** Air temp, Process temp, Rotational speed, Torque, Tool wear, Type (product quality)
- **Target:** Machine failure (binary, ~3.4% failure rate) + 5 failure subtypes
- **Limitation:** No raw vibration data; Torque used as proxy for electrical load

## Architecture

### Components
1. **`src/data_loader.py`** — Load CSV, basic cleaning, export
2. **`src/preprocessing.py`** — Encode `Type`, feature matrix, stratified train/test split (80/20)
3. **`src/train_classifier.py`** — XGBoost with `scale_pos_weight` for class imbalance
4. **`src/train_anomaly.py`** — Isolation Forest on normal data only (unsupervised)
5. **`src/evaluate.py`** — Classification metrics (precision/recall/F1 per class), confusion matrix, anomaly score analysis
6. **`app.py`** — Streamlit dashboard with real-time simulation

### Risk Logic
| Color | Condition |
|-------|-----------|
| **Red** | XGBoost predicts failure = True |
| **Yellow** | XGBoost predicts normal, anomaly score > threshold (top 5%) |
| **Green** | Both models agree normal |

### Dashboard Features
- Line charts (Plotly) for each sensor — historical + real-time update
- Current risk status card with color indicator
- Alert history log with timestamps
- Auto-play simulation from test set

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Streamlit

## Deployment
GitHub + Streamlit Community Cloud (free, deploy from repo)

## Evaluation
- **Primary metrics:** Recall, F1-score per class (not accuracy — class imbalance)
- **Confusion matrix** — saved as image for README
- **Train/test split:** 80/20 with `stratify=y`

## Out of Scope
- Real IoT sensor connection
- Deep learning (LSTM, Autoencoder)
- Online learning / model retraining
- Vibration signal processing

## Timeline
- **Day 1:** Data preprocessing, create `src/` package
- **Day 2:** Model training & evaluation
- **Day 3:** Dashboard, deployment, documentation
