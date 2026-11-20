import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import export_graphviz, export_text

from src.data_prep import generate_dataset
from src.train_models import train_all

# --- Page Configuration ---
st.set_page_config(
    page_title="Student ML Predictor",
    page_icon="🎓",
    layout="wide"
)

# --- Custom CSS for Cleaner Spacing & Card Borders ---
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Auto-Setup & Model Caching ---
@st.cache_resource
def load_pipeline():
    required_files = [
        "models/scaler.pkl",
        "models/linear_reg.pkl",
        "models/logistic_reg.pkl",
        "models/decision_tree.pkl",
        "data/student_data.csv"
    ]
    if not all(os.path.exists(f) for f in required_files):
        generate_dataset()
        train_all()
        
    scaler = joblib.load("models/scaler.pkl")
    lin_reg = joblib.load("models/linear_reg.pkl")
    log_reg = joblib.load("models/logistic_reg.pkl")
    dt_clf = joblib.load("models/decision_tree.pkl")
    df = pd.read_csv("data/student_data.csv")
    return scaler, lin_reg, log_reg, dt_clf, df

scaler, lin_reg, log_reg, dt_clf, df = load_pipeline()

# --- Header ---
st.title("🎓 Student Performance & Advising Predictor")
st.caption("Powered by Linear Regression, Logistic Regression, and Decision Tree Classifier on the UCI Student Dataset")

# --- Sidebar Inputs ---
st.sidebar.header("🎛️ Student Profile Controls")
st.sidebar.markdown("Adjust the sliders below to simulate a student's academic profile:")

study_hours = st.sidebar.slider("Weekly Study Hours", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
attendance = st.sidebar.slider("Attendance (%)", min_value=40.0, max_value=100.0, value=88.0, step=1.0)
prev_grade = st.sidebar.slider("Previous Grade (G1/G2 Avg)", min_value=30.0, max_value=100.0, value=70.0, step=1.0)
assignments = st.sidebar.slider("Engagement / Assignments (0–10)", min_value=0, max_value=10, value=8, step=1)

# --- Prepare Input & Run Inference ---
features = ["study_hours", "attendance", "prev_grade", "assignments_done"]
raw_input = pd.DataFrame([[study_hours, attendance, prev_grade, assignments]], columns=features)
scaled_input = scaler.transform(raw_input)

pred_score = float(np.clip(lin_reg.predict(scaled_input)[0], 0, 100))
pass_prob = float(log_reg.predict_proba(scaled_input)[0][1] * 100)
pred_tier = dt_clf.predict(raw_input)[0]

# --- Top Row: Clean Bordered KPI Cards ---
st.markdown("### 📌 Live Model Predictions")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.metric(
        label="Linear Regression • Final Score",
        value=f"{pred_score:.1f} / 100",
        delta=f"{pred_score - prev_grade:+.1f} vs Previous Grade"
    )

with col2:
    status_label = "Pass Likely (≥ 60)" if pass_prob >= 50 else "Fail Risk (< 60)"
    st.metric(
        label="Logistic Regression • Pass Probability",
        value=f"{pass_prob:.1f}%",
        delta=status_label,
        delta_color="normal" if pass_prob >= 50 else "inverse"
    )

with col3:
    st.metric(
        label="Decision Tree • Advising Category",
        value=pred_tier,
        delta="Gini Rule Classification",
        delta_color="off"
    )

st.write("")  # Clean vertical breathing room

# --- Clean Tabbed Workspace (Prevents Cramped Charts) ---
tab1, tab2, tab3 = st.tabs([
    "📊 Cohort Comparison", 
    "⚖️ Feature Importance", 
    "🌳 Decision Tree Rules"
])

# === TAB 1: Interactive Plotly Histogram ===
with tab1:
    st.subheader("Where Does This Student Stand in the Cohort?")
    st.markdown("Hover over any bar to inspect student counts. The dashed red line tracks your current slider prediction.")
    
    fig_hist = px.histogram(
        df,
        x="final_score",
        nbins=25,
        color="tier",
        color_discrete_map={"At-Risk": "#E74C3C", "Average": "#3498DB", "Honors": "#2ECC71"},
        labels={"final_score": "Final Exam Score (0–100)", "tier": "Academic Tier"},
        opacity=0.8
    )
    
    # Add vertical markers for Current Prediction and Pass Cutoff
    fig_hist.add_vline(
        x=pred_score,
        line_width=3,
        line_dash="dash",
        line_color="#FF4B4B",
        annotation_text=f" Current Student: {pred_score:.1f}",
        annotation_position="top right"
    )
    fig_hist.add_vline(
        x=60,
        line_width=2,
        line_dash="dot",
        line_color="gray",
        annotation_text=" Pass Cutoff (60)",
        annotation_position="top left"
    )
    
    fig_hist.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.08
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# === TAB 2: Clean Grouped Feature Weights ===
with tab2:
    st.subheader("How Much Does Each Feature Matter?")
    st.markdown("Compares normalized **Linear Regression** weights against **Decision Tree** Gini importance.")
    
    lin_weights = np.abs(lin_reg.coef_) / np.sum(np.abs(lin_reg.coef_))
    dt_weights = dt_clf.feature_importances_
    readable_names = ["Weekly Study Hours", "Attendance %", "Previous Grade", "Assignments / Engagement"]
    
    fig_imp = go.Figure(data=[
        go.Bar(
            name="Linear Regression (Normalized |Weight|)",
            x=readable_names,
            y=np.round(lin_weights, 3),
            marker_color="#3B82F6",
            text=np.round(lin_weights, 2),
            textposition="auto"
        ),
        go.Bar(
            name="Decision Tree (Gini Importance)",
            x=readable_names,
            y=np.round(dt_weights, 3),
            marker_color="#F59E0B",
            text=np.round(dt_weights, 2),
            textposition="auto"
        )
    ])
    
    fig_imp.update_layout(
        barmode="group",
        height=450,
        yaxis_title="Relative Importance (0.0 to 1.0)",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# === TAB 3: Spacious Graphviz Decision Tree + Text Rules ===
with tab3:
    st.subheader("Decision Tree Splitting Logic")
    st.markdown("Zoom or pan across the tree diagram below, or expand the text rules for quick reading.")
    
    # Generate crisp vector Graphviz DOT string
    dot_data = export_graphviz(
        dt_clf,
        out_file=None,
        feature_names=["Study Hours", "Attendance %", "Prev Grade", "Assignments"],
        class_names=dt_clf.classes_,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False
    )
    # Inject clean typography and spacing into the DOT graph
    dot_data = dot_data.replace(
        "digraph Tree {",
        'digraph Tree {\nranksep=0.6;\nnodesep=0.4;\nnode [fontname="Arial", fontsize=11, margin="0.15,0.1"];'
    )
    st.graphviz_chart(dot_data, use_container_width=True)
    
    with st.expander("📄 View Plain-Text If-Else Decision Rules"):
        rules_text = export_text(
            dt_clf,
            feature_names=["Study Hours", "Attendance %", "Prev Grade", "Assignments"]
        )
        st.code(rules_text, language="text")