# ============================================================
# STREAMLIT WEB APP: HEART DISEASE PREDICTION
# Author: Lokesh
# Run: streamlit run app.py
#
# PREREQUISITES:
#   1. Run heart_disease_dissertation.py FIRST to generate:
#      - best_model.pkl
#      - scaler.pkl
#      - feature_names.pkl
#   2. pip install streamlit joblib scikit-learn xgboost pandas numpy
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIGURATION — must be the FIRST streamlit command
# ============================================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS  —  dark navy theme for full text visibility
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global font ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Page background ── */
.stApp {
    background-color: #0F172A !important;
}
section[data-testid="stSidebar"] {
    background-color: #1E293B !important;
}
.block-container {
    background-color: #0F172A !important;
    padding-top: 1rem !important;
}

/* ── ALL text forced visible on dark bg ── */
p, span, div, label, h1, h2, h3, h4, h5, h6,
.stMarkdown, .stText,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span {
    color: #E2E8F0 !important;
}

/* ── Input / select / radio labels ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stTextInput"] label {
    color: #F1F5F9 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* ── Radio option text ── */
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #CBD5E1 !important;
    font-size: 0.88rem !important;
}

/* ── Number input box ── */
div[data-testid="stNumberInput"] input {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.25) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background-color: #1E293B !important;
    color: #F1F5F9 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] svg {
    fill: #94A3B8 !important;
}

/* ── Selectbox dropdown ── */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #E2E8F0 !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #334155 !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] td,
div[data-testid="stExpander"] th {
    color: #CBD5E1 !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.8rem 1rem !important;
}
div[data-testid="stMetricLabel"] p {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
}
div[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Progress bar ── */
div[data-testid="stProgress"] > div {
    background-color: #334155 !important;
    border-radius: 6px !important;
}
div[data-testid="stProgress"] > div > div {
    background-color: #3B82F6 !important;
    border-radius: 6px !important;
}

/* ── Info / error / warning ── */
div[data-testid="stAlert"] {
    background-color: #1E293B !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}
div[data-testid="stAlert"] p {
    color: #E2E8F0 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    padding: 0.85rem 1.5rem !important;
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.4) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #334155 !important;
    margin: 1.5rem 0 !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero-icon { font-size: 3rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #F8FAFC !important;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 0.88rem;
    color: #94A3B8 !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #60A5FA !important;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1E3A5F;
}

/* ── Field hint ── */
.field-hint {
    font-size: 0.76rem;
    color: #64748B !important;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
    line-height: 1.45;
}

/* ── BMI badge ── */
.bmi-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.3rem;
    margin-bottom: 0.5rem;
}
.bmi-normal { background: #064E3B; color: #6EE7B7 !important; }
.bmi-under  { background: #1E3A5F; color: #93C5FD !important; }
.bmi-over   { background: #451A03; color: #FCD34D !important; }
.bmi-obese  { background: #450A0A; color: #FCA5A5 !important; }

/* ── Result cards ── */
.result-high {
    background: #1C0A0A;
    border: 1px solid #7F1D1D;
    border-left: 5px solid #EF4444;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
}
.result-low {
    background: #052E16;
    border: 1px solid #14532D;
    border-left: 5px solid #22C55E;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.5rem;
}
.result-tag {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-high .result-tag { color: #FCA5A5 !important; }
.result-low  .result-tag { color: #86EFAC !important; }
.result-pct {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.result-high .result-pct { color: #F87171 !important; }
.result-low  .result-pct { color: #4ADE80 !important; }
.result-desc { font-size: 0.88rem; color: #94A3B8 !important; }

/* ── Risk / OK items ── */
.risk-item {
    background: #1C1109;
    border-left: 3px solid #F97316;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    font-size: 0.85rem;
    color: #FED7AA !important;
    margin-bottom: 7px;
}
.ok-item {
    background: #052E16;
    border-left: 3px solid #22C55E;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    font-size: 0.85rem;
    color: #86EFAC !important;
    margin-bottom: 7px;
}

/* ── Summary table ── */
.summary-table {
    width: 100%;
    font-size: 0.85rem;
    border-collapse: collapse;
    margin-top: 0.5rem;
    background: #1E293B;
    border-radius: 10px;
    overflow: hidden;
}
.summary-table th {
    text-align: left;
    color: #64748B !important;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 8px 14px;
    background: #0F172A;
    border-bottom: 1px solid #334155;
}
.summary-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #0F172A;
    color: #E2E8F0 !important;
}
.summary-table td:first-child {
    color: #94A3B8 !important;
    font-weight: 500;
    width: 45%;
}
.summary-table tr:last-child td { border-bottom: none; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #475569 !important;
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model    = joblib.load('best_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    return model, scaler, features

if not os.path.exists('best_model.pkl'):
    st.error("""
    ⚠️ **Model files not found!**

    Please run `heart_disease_dissertation.py` first to generate:
    - `best_model.pkl`
    - `scaler.pkl`
    - `feature_names.pkl`

    Then restart this Streamlit app.
    """)
    st.stop()

model, scaler, feature_names = load_artifacts()

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-icon">❤️</div>
    <div class="hero-title">Heart Disease Predictor</div>
    <div class="hero-sub">MSc Dissertation &mdash; Cardiovascular Risk Assessment using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About this application"):
    st.markdown("""
    This application uses a machine learning model trained on the **Cardiovascular Disease Dataset**
    (70,000 patient records). It was built as part of an MSc dissertation comparing
    **5 algorithms**: Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost.

    **Best model deployed**: XGBoost (highest ROC-AUC after GridSearchCV tuning)

    | Input | Output |
    |---|---|
    | Patient clinical & lifestyle data | Probability of cardiovascular disease + binary prediction |

    > ⚠️ *This tool is for educational and research purposes only — not a substitute for professional medical advice.*
    """)

# ============================================================
# SECTION 1 — PATIENT DEMOGRAPHICS
# ============================================================

st.markdown('<div class="section-header">👤 Patient demographics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=45, step=1)
    st.markdown('<p class="field-hint">Cardiovascular risk increases significantly after age 50.</p>', unsafe_allow_html=True)

    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, step=1)
    st.markdown('<p class="field-hint">Used with weight to calculate Body Mass Index (BMI).</p>', unsafe_allow_html=True)

with col2:
    gender = st.selectbox(
        "Gender",
        options=[(1, "Female"), (2, "Male")],
        format_func=lambda x: x[1]
    )
    st.markdown('<p class="field-hint">Encoded as 1 = Female, 2 = Male (dataset convention).</p>', unsafe_allow_html=True)

    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
    st.markdown('<p class="field-hint">Used with height to calculate Body Mass Index (BMI).</p>', unsafe_allow_html=True)

# Live BMI badge
bmi = weight / ((height / 100) ** 2)
if bmi < 18.5:
    bmi_class, bmi_label = "bmi-under", "Underweight"
elif bmi < 25:
    bmi_class, bmi_label = "bmi-normal", "Normal weight"
elif bmi < 30:
    bmi_class, bmi_label = "bmi-over", "Overweight"
else:
    bmi_class, bmi_label = "bmi-obese", "Obese"

st.markdown(
    f'<span class="bmi-badge {bmi_class}">⚖️ BMI: {bmi:.1f} kg/m² — {bmi_label}</span>',
    unsafe_allow_html=True
)

# ============================================================
# SECTION 2 — BLOOD PRESSURE & LABS
# ============================================================

st.markdown('<div class="section-header">🩸 Blood pressure &amp; laboratory results</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    ap_hi = st.number_input("Systolic blood pressure (mmHg)", min_value=50, max_value=250, value=120, step=1)
    st.markdown('<p class="field-hint">Top number in a reading. Normal &lt;120. Stage 1: 130–139. Stage 2 hypertension: ≥140.</p>', unsafe_allow_html=True)

    cholesterol = st.selectbox(
        "Cholesterol level",
        options=[(1, "Normal"), (2, "Above normal"), (3, "Well above normal")],
        format_func=lambda x: x[1]
    )
    st.markdown('<p class="field-hint">Elevated cholesterol is a key independent cardiovascular risk factor.</p>', unsafe_allow_html=True)

with col4:
    ap_lo = st.number_input("Diastolic blood pressure (mmHg)", min_value=30, max_value=150, value=80, step=1)
    st.markdown('<p class="field-hint">Bottom number in a reading. Normal &lt;80 mmHg. Elevated: ≥90 mmHg.</p>', unsafe_allow_html=True)

    gluc = st.selectbox(
        "Glucose level",
        options=[(1, "Normal"), (2, "Above normal"), (3, "Well above normal")],
        format_func=lambda x: x[1]
    )
    st.markdown('<p class="field-hint">Elevated glucose may indicate pre-diabetes or diabetes — both raise heart risk.</p>', unsafe_allow_html=True)

# ============================================================
# SECTION 3 — LIFESTYLE
# ============================================================

st.markdown('<div class="section-header">🏃 Lifestyle factors</div>', unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)

with col5:
    smoke = st.radio("Smoking", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], horizontal=True)
    st.markdown('<p class="field-hint">Smoking is one of the strongest independent cardiovascular risk factors.</p>', unsafe_allow_html=True)

with col6:
    alco = st.radio("Alcohol intake", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], horizontal=True)
    st.markdown('<p class="field-hint">Regular consumption is linked to raised blood pressure.</p>', unsafe_allow_html=True)

with col7:
    active = st.radio("Physically active", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1], horizontal=True)
    st.markdown('<p class="field-hint">Regular exercise significantly lowers cardiovascular disease risk.</p>', unsafe_allow_html=True)

# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict heart disease risk")

if predict_clicked:

    if ap_hi <= ap_lo:
        st.error("⚠️ Systolic BP must be greater than Diastolic BP. Please check your inputs.")
        st.stop()

    gender_val      = gender[0]
    cholesterol_val = cholesterol[0]
    gluc_val        = gluc[0]
    smoke_val       = smoke[0]
    alco_val        = alco[0]
    active_val      = active[0]

    input_dict = {
        'age':         [age],
        'gender':      [gender_val],
        'height':      [height],
        'weight':      [weight],
        'ap_hi':       [ap_hi],
        'ap_lo':       [ap_lo],
        'cholesterol': [cholesterol_val],
        'gluc':        [gluc_val],
        'smoke':       [smoke_val],
        'alco':        [alco_val],
        'active':      [active_val],
        'bmi':         [round(bmi, 2)]
    }

    input_df     = pd.DataFrame(input_dict)[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # ── Result card ──────────────────────────────────────────

    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-tag">⚠️ High risk — cardiovascular disease likely</div>
            <div class="result-pct">{probability*100:.1f}%</div>
            <div class="result-desc">Predicted probability of cardiovascular disease based on your inputs.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-tag">✅ Low risk — no disease detected</div>
            <div class="result-pct">{probability*100:.1f}%</div>
            <div class="result-desc">Predicted probability of cardiovascular disease based on your inputs.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 📊 Risk probability breakdown")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("❤️ Disease probability", f"{probability*100:.1f}%")
        st.progress(float(probability))
    with m2:
        st.metric("💚 Healthy probability", f"{(1-probability)*100:.1f}%")
        st.progress(float(1 - probability))

    # ── Input summary ─────────────────────────────────────────

    st.markdown("#### 📋 Your input summary")

    chol_map = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
    gluc_map = {1: "Normal", 2: "Above normal", 3: "Well above normal"}

    summary_rows = [
        ("Age",               f"{age} years"),
        ("Gender",            "Female" if gender_val == 1 else "Male"),
        ("Height",            f"{height} cm"),
        ("Weight",            f"{weight} kg"),
        ("BMI",               f"{bmi:.1f} kg/m² — {bmi_label}"),
        ("Systolic BP",       f"{ap_hi} mmHg"),
        ("Diastolic BP",      f"{ap_lo} mmHg"),
        ("Cholesterol",       chol_map[cholesterol_val]),
        ("Glucose",           gluc_map[gluc_val]),
        ("Smoking",           "Yes" if smoke_val else "No"),
        ("Alcohol",           "Yes" if alco_val else "No"),
        ("Physically active", "Yes" if active_val else "No"),
    ]

    rows_html = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in summary_rows
    )
    st.markdown(f"""
    <table class="summary-table">
        <thead><tr><th>Feature</th><th>Value</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ── Clinical risk factors ─────────────────────────────────

    st.markdown("#### 🩺 Clinical risk factors identified")

    risks = []
    if age >= 60:
        risks.append(f"Age {age} — very high cardiovascular risk (60+)")
    elif age >= 50:
        risks.append(f"Age {age} — elevated cardiovascular risk (50+)")
    if ap_hi >= 140:
        risks.append(f"Systolic BP {ap_hi} mmHg — Stage 2 hypertension (≥140 mmHg)")
    elif ap_hi >= 130:
        risks.append(f"Systolic BP {ap_hi} mmHg — Stage 1 hypertension (130–139 mmHg)")
    if ap_lo >= 90:
        risks.append(f"Diastolic BP {ap_lo} mmHg — elevated (≥90 mmHg)")
    if cholesterol_val == 3:
        risks.append("Cholesterol well above normal — significant independent risk factor")
    elif cholesterol_val == 2:
        risks.append("Cholesterol above normal — monitor and consider dietary changes")
    if gluc_val == 3:
        risks.append("Glucose well above normal — possible diabetes, raises cardiovascular risk")
    elif gluc_val == 2:
        risks.append("Glucose above normal — possible pre-diabetes")
    if bmi >= 30:
        risks.append(f"BMI {bmi:.1f} kg/m² — Obese (≥30), increases cardiac workload")
    elif bmi >= 25:
        risks.append(f"BMI {bmi:.1f} kg/m² — Overweight (25–29.9)")
    if smoke_val:
        risks.append("Smoking — one of the strongest independent cardiovascular risk factors")
    if alco_val:
        risks.append("Alcohol intake — linked to raised blood pressure and arrhythmia")
    if not active_val:
        risks.append("Physical inactivity — significantly raises cardiovascular disease risk")

    if risks:
        for r in risks:
            st.markdown(f'<div class="risk-item">⚠️ {r}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-item">✅ No major clinical risk factors identified from your inputs.</div>', unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────

    st.info("""
    ⚠️ **Disclaimer:** This prediction is generated by a machine learning model trained on
    population-level data. It is intended for **educational and research purposes only**.
    Please consult a qualified healthcare professional for medical advice.
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Built with ❤️ by Lokesh &nbsp;|&nbsp; MSc Dissertation &nbsp;|&nbsp; Machine Learning for Healthcare<br>
    Model: XGBoost (Tuned with GridSearchCV) &nbsp;|&nbsp; Dataset: Cardiovascular Disease Dataset (70K records)
</div>
""", unsafe_allow_html=True)