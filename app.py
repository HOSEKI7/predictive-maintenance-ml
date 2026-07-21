import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data_loader import load_processed
from src.train_classifier import load_model as load_clf
from src.train_anomaly import load_model as load_if

st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="📊", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Geist', sans-serif; }
.metric-value { font-variant-numeric: tabular-nums; }
</style>""", unsafe_allow_html=True)

SENSOR_METADATA = {
    "air_temperature_k": {"label": "Air Temperature", "unit": "K"},
    "process_temperature_k": {"label": "Process Temperature", "unit": "K"},
    "rotational_speed_rpm": {"label": "Rotational Speed", "unit": "rpm"},
    "torque_nm": {"label": "Torque", "unit": "Nm"},
    "tool_wear_min": {"label": "Tool Wear", "unit": "min"},
}

COLORS = {"green": "#409a6b", "yellow": "#c4902a", "red": "#b84646"}

SENSOR_COLORS = {
    "air_temperature_k": "#4a80b8",
    "process_temperature_k": "#b85a5a",
    "rotational_speed_rpm": "#4a8f6a",
    "torque_nm": "#7a6bbf",
    "tool_wear_min": "#b88a3a",
}

FEATURE_COLS = list(SENSOR_METADATA.keys()) + ["Type"]


@st.cache_resource
def load_models():
    clf = load_clf("xgb_classifier.pkl")
    if_model = load_if("isolation_forest.pkl")
    return clf, if_model


@st.cache_resource
def load_test_data():
    return load_processed("test_data")


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.idx = 0
    st.session_state.history = []
    st.session_state.alerts = []
    st.session_state.running = False


def compute_risk(y_clf_pred, y_anomaly_pred):
    if y_clf_pred == 1:
        return "red"
    if y_anomaly_pred == 1:
        return "yellow"
    return "green"


def get_anomaly_threshold(model, X_ref, percentile=5):
    scores = model.decision_function(X_ref)
    return np.percentile(scores, percentile)


def next_row():
    test_data = st.session_state.test_data
    idx = st.session_state.idx

    if idx >= len(test_data):
        st.session_state.running = False
        return None

    row = test_data.iloc[idx]
    st.session_state.idx = idx + 1

    clf, if_model = st.session_state.clf, st.session_state.if_model
    X = row[FEATURE_COLS].to_frame().T

    y_clf = clf.predict(X)[0]
    anomaly_score = if_model.decision_function(X)[0]
    y_anomaly = 1 if anomaly_score <= st.session_state.threshold else 0

    risk = compute_risk(y_clf, y_anomaly)

    entry = {
        "idx": idx,
        "row": row,
        "risk": risk,
        "anomaly_score": anomaly_score,
        "y_clf": y_clf,
        "y_anomaly": y_anomaly,
        "actual_failure": row["Machine failure"],
    }
    st.session_state.history.append(entry)

    if risk in ("red", "yellow"):
        alert = {
            "time": f"Row {idx + 1}",
            "risk": risk,
            "message": (
                "Failure predicted by classifier"
                if risk == "red"
                else "Anomaly detected (high anomaly score)"
            ),
            "actual": "FAILURE" if row["Machine failure"] else "normal",
        }
        st.session_state.alerts.append(alert)

    return entry


def build_sensor_charts(history):
    if not history:
        return None

    df = pd.DataFrame([e["row"] for e in history])
    df["risk"] = [e["risk"] for e in history]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[m["label"] for m in SENSOR_METADATA.values()],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    plots = [
        ("air_temperature_k", 1, 1),
        ("process_temperature_k", 1, 2),
        ("rotational_speed_rpm", 2, 1),
        ("torque_nm", 2, 2),
        ("tool_wear_min", 3, 1),
    ]

    for sensor, row, col in plots:
        color = SENSOR_COLORS[sensor]
        y_vals = df[sensor].values
        x_vals = list(range(len(y_vals)))

        risk_colors = [COLORS[r] for r in df["risk"]]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=SENSOR_METADATA[sensor]["label"],
                line=dict(color=color, width=2),
                marker=dict(color=risk_colors, size=6),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=600,
        title_text="Sensor Readings (color = risk level)",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_xaxes(title_text="Time Step", row=3, col=2)

    return fig


def main():
    st.title("Predictive Maintenance Dashboard")
    st.markdown(
        "Real-time sensor monitoring with failure prediction (XGBoost) "
        "and anomaly detection (Isolation Forest)."
    )

    clf, if_model = load_models()
    test_data = load_test_data()

    st.session_state.clf = clf
    st.session_state.if_model = if_model
    st.session_state.test_data = test_data

    if "threshold" not in st.session_state:
        normal_mask = test_data["Machine failure"] == 0
        X_normal = test_data[normal_mask][FEATURE_COLS]
        st.session_state.threshold = get_anomaly_threshold(if_model, X_normal)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        speed = st.select_slider(
            "Simulation Speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal",
        )
        speed_map = {"Slow": 0.5, "Normal": 0.2, "Fast": 0.05}

    with col2:
        if st.button("▶ Play" if not st.session_state.running else "⏸ Pause"):
            st.session_state.running = not st.session_state.running

    with col3:
        if st.button("↺ Reset"):
            st.session_state.idx = 0
            st.session_state.history = []
            st.session_state.alerts = []
            st.session_state.running = False
            st.rerun()

    status_col, charts_col = st.columns([1, 2])

    with status_col:
        st.subheader("Current Status")

        if st.session_state.history:
            latest = st.session_state.history[-1]
            risk = latest["risk"]
            row = latest["row"]

            st.markdown(
                f"<div style='padding:20px; border-radius:10px; "
                f"background-color:{COLORS[risk]}; text-align:center; "
                f"color:white; font-size:24px; font-weight:bold;'>"
                f"{risk.upper()}</div>",
                unsafe_allow_html=True,
            )

            st.metric("Air Temperature", f"{row['air_temperature_k']:.1f} K")
            st.metric("Process Temperature", f"{row['process_temperature_k']:.1f} K")
            st.metric("Rotational Speed", f"{row['rotational_speed_rpm']:.0f} rpm")
            st.metric("Torque", f"{row['torque_nm']:.1f} Nm")
            st.metric("Tool Wear", f"{row['tool_wear_min']:.0f} min")
            st.metric("Product Type", ["L", "M", "H"][int(row["Type"])])
            st.metric("Anomaly Score", f"{latest['anomaly_score']:.4f}")

            progress = min(st.session_state.idx / len(test_data), 1.0)
            st.progress(progress, text=f"Row {st.session_state.idx} / {len(test_data)}")
        else:
            st.info("Press Play to start simulation")

    with charts_col:
        st.subheader("Sensor Trends")
        fig = build_sensor_charts(st.session_state.history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Alert Log")
    alert_container = st.container()
    with alert_container:
        if st.session_state.alerts:
            alert_df = pd.DataFrame(st.session_state.alerts[::-1])
            for _, a in alert_df.iterrows():
                color = COLORS[a["risk"]]
                st.markdown(
                    f"<div style='padding:8px; margin:4px 0; border-left:4px solid {color}; "
                    f"background-color:#f8f9fa; border-radius:4px;'>"
                    f"<strong>{a['time']}</strong> — "
                    f"<span style='color:{color};'>{a['risk'].upper()}</span>: "
                    f"{a['message']} "
                    f"<em>(actual: {a['actual']})</em>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No alerts yet")

    if st.session_state.running:
        entry = next_row()
        if entry is not None:
            time.sleep(speed_map[speed])
            st.rerun()


if __name__ == "__main__":
    main()
