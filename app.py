"""
using the UCI Credit Card Default dataset.
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Page config 
st.set_page_config(
    page_title="XplainCredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    /* Title */
    .main-title  { font-size:2.4rem; font-weight:700; color:#4361ee; margin-bottom:0; }
    .sub-title   { font-size:1rem; color:#888; margin-top:0; margin-bottom:2rem; }

    /* Risk cards */
    .risk-card   { padding:1.5rem; border-radius:12px; text-align:center; margin-bottom:1rem; }
    .risk-low    { background:#d4edda; border:2px solid #28a745; }
    .risk-medium { background:#fff3cd; border:2px solid #ffc107; }
    .risk-high   { background:#ffe5d0; border:2px solid #fd7e14; }
    .risk-vhigh  { background:#f8d7da; border:2px solid #dc3545; }
    .risk-label  { font-size:1.8rem; font-weight:700; margin:0; }
    .risk-score  { font-size:1.1rem; margin-top:.3rem; }

    /* Metric boxes */
    .metric-box  { background:	#131122; border-radius:8px; padding:1rem;
                   text-align:center; border:1px solid #dee2e6; }

    /* Section headers */
    .sec-hdr     { font-size:1.15rem; font-weight:600; color:#444;
                   border-left:4px solid #4361ee; padding-left:.8rem;
                   margin:1.5rem 0 .8rem 0; }

    /* Tip boxes */
    .tip-box     { background:#f0f4ff; border-left:4px solid #4361ee;
                   border-radius:0 8px 8px 0; padding:.75rem 1rem;
                   margin:.5rem 0; font-size:.9rem; color:#222; }

    /* Footer */
    .footer      { text-align:center; color:#aaa; font-size:.8rem;
                   margin-top:3rem; padding-top:1rem; border-top:1px solid #eee; }

    /* Sidebar tweaks */
    .stSlider > label { font-size:.9rem !important; }
            
    div[data-testid="stAlert"] {
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Load artefacts 
@st.cache_resource
def load_artifacts():
    model      = joblib.load("model/xplaincredit_model.pkl")
    features   = joblib.load("model/feature_names.pkl")
    bg_sample  = joblib.load("model/shap_background.pkl")
    train_stats= joblib.load("model/train_stats.pkl")
    return model, features, bg_sample, train_stats

try:
    model, FEATURES, shap_bg, stats = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Run `python train_model.py` first, then restart the app.")
    st.stop()

# Helpers 
RISK_TIERS = {
    (0.00, 0.25): ("Low Risk",       "🟢", "risk-low",   "#28a745"),
    (0.25, 0.50): ("Medium Risk",    "🟡", "risk-medium","#ffc107"),
    (0.50, 0.75): ("High Risk",      "🟠", "risk-high",  "#fd7e14"),
    (0.75, 1.01): ("Very High Risk", "🔴", "risk-vhigh", "#dc3545"),
}

def get_risk_tier(p):
    for (lo, hi), info in RISK_TIERS.items():
        if lo <= p < hi:
            return info
    return ("Very High Risk", "🔴", "risk-vhigh", "#dc3545")

# labels for the SHAP waterfall chart
FEATURE_LABELS = {
    "LIMIT_BAL"        : "Credit Limit (NT$)",
    "SEX"              : "Gender",
    "EDUCATION"        : "Education Level",
    "MARRIAGE"         : "Marital Status",
    "AGE"              : "Age",
    "PAY_0"            : "Repay Status Sep",
    "PAY_2"            : "Repay Status Aug",
    "PAY_3"            : "Repay Status Jul",
    "PAY_4"            : "Repay Status Jun",
    "PAY_5"            : "Repay Status May",
    "PAY_6"            : "Repay Status Apr",
    "BILL_AMT1"        : "Bill Statement Sep (NT$)",
    "BILL_AMT2"        : "Bill Statement Aug (NT$)",
    "BILL_AMT3"        : "Bill Statement Jul (NT$)",
    "BILL_AMT4"        : "Bill Statement Jun (NT$)",
    "BILL_AMT5"        : "Bill Statement May (NT$)",
    "BILL_AMT6"        : "Bill Statement Apr (NT$)",
    "PAY_AMT1"         : "Payment Sep (NT$)",
    "PAY_AMT2"         : "Payment Aug (NT$)",
    "PAY_AMT3"         : "Payment Jul (NT$)",
    "PAY_AMT4"         : "Payment Jun (NT$)",
    "PAY_AMT5"         : "Payment May (NT$)",
    "PAY_AMT6"         : "Payment Apr (NT$)",
    "MAX_DELAY"        : "Max Delay (months)",
    "AVG_BILL_AMT"     : "Avg Bill Amount (NT$)",
    "AVG_PAY_AMT"      : "Avg Payment Amount (NT$)",
    "TOTAL_BILL"       : "Total 6-Month Bill (NT$)",
    "TOTAL_PAY"        : "Total 6-Month Payment (NT$)",
    "PAY_RATIO"        : "Payment-to-Bill Ratio",
    "UTIL_RATIO"       : "Credit Utilisation",
    "NUM_LATE_PAYMENTS": "# Months w/ Late Payment",
    "NUM_ON_TIME"      : "# Months Paid On Time",
}

PAY_STATUS_OPTIONS = {
    -2: "-2  (No consumption)",
    -1: "-1  (Paid in full)",
     0: " 0  (Revolving credit used)",
     1: " 1  (Payment 1 month late)",
     2: " 2  (Payment 2 months late)",
     3: " 3  (Payment 3 months late)",
     4: " 4  (Payment 4 months late)",
     5: " 5  (Payment 5 months late)",
     6: " 6  (Payment 6+ months late)",
}

def build_input(limit_bal, sex, education, marriage, age,
                pay0, pay2, pay3, pay4, pay5, pay6,
                bill1, bill2, bill3, bill4, bill5, bill6,
                pamt1, pamt2, pamt3, pamt4, pamt5, pamt6):
    """Construct one-row DataFrame from sidebar inputs."""
    total_bill = bill1+bill2+bill3+bill4+bill5+bill6
    total_pay  = pamt1+pamt2+pamt3+pamt4+pamt5+pamt6
    pay_cols   = [pay0, pay2, pay3, pay4, pay5, pay6]

    row = {
        "LIMIT_BAL"        : limit_bal,
        "SEX"              : sex,
        "EDUCATION"        : education,
        "MARRIAGE"         : marriage,
        "AGE"              : age,
        "PAY_0"            : pay0,
        "PAY_2"            : pay2,
        "PAY_3"            : pay3,
        "PAY_4"            : pay4,
        "PAY_5"            : pay5,
        "PAY_6"            : pay6,
        "BILL_AMT1"        : bill1,
        "BILL_AMT2"        : bill2,
        "BILL_AMT3"        : bill3,
        "BILL_AMT4"        : bill4,
        "BILL_AMT5"        : bill5,
        "BILL_AMT6"        : bill6,
        "PAY_AMT1"         : pamt1,
        "PAY_AMT2"         : pamt2,
        "PAY_AMT3"         : pamt3,
        "PAY_AMT4"         : pamt4,
        "PAY_AMT5"         : pamt5,
        "PAY_AMT6"         : pamt6,
        "MAX_DELAY"        : max(pay_cols),
        "AVG_BILL_AMT"     : (bill1+bill2+bill3+bill4+bill5+bill6)/6,
        "AVG_PAY_AMT"      : (pamt1+pamt2+pamt3+pamt4+pamt5+pamt6)/6,
        "TOTAL_BILL"       : total_bill,
        "TOTAL_PAY"        : total_pay,
        "PAY_RATIO"        : total_pay/(total_bill+1) if total_bill >= 0 else 0,
        "UTIL_RATIO"       : bill1/(limit_bal+1),
        "NUM_LATE_PAYMENTS": sum(p > 0 for p in pay_cols),
        "NUM_ON_TIME"      : sum(p <= 0 for p in pay_cols),
    }
    return pd.DataFrame([row])[FEATURES]


def tips(prob, pay0, util, num_late, limit_bal, pay_ratio):
    out = []
    if pay0 >= 2:
        out.append(f"September repayment status shows {pay0} month(s) late. Paying the minimum due "
                   f"immediately will reduce future delay flags.")
    if num_late >= 3:
        out.append(f"{num_late} of the last 6 months had late payments. Consistent on-time payments "
                   f"are the single biggest predictor of lower default risk.")
    if util > 0.8:
        out.append(f"Credit utilisation is very high ({util:.0%}). Reducing outstanding balance or "
                   f"requesting a limit increase can improve the risk profile.")
    if pay_ratio < 0.1 and prob > 0.3:
        out.append("Total payments are very low relative to total bills. Even small extra payments "
                   "each month reduce compounding balances and risk score.")
    if limit_bal < 30000:
        out.append("A low credit limit combined with high bills is a red flag. A clean repayment "
                   "history over 6 months typically enables a limit review.")
    if not out:
        out.append("Profile looks strong. Keeping all payments on time and maintaining low "
                   "utilisation will preserve this low-risk status.")
    return out


# SIDEBAR

with st.sidebar:
    st.markdown("## 📋 Applicant Profile")
    st.caption("Based on Taiwan credit card data (UCI dataset)")
    st.markdown("---")

    # Demographic
    st.markdown("**Demographics**")
    age       = st.slider("Age", 21, 79, 35)
    sex       = st.radio("Gender", options=[1, 2],
                         format_func=lambda x: "Male" if x == 1 else "Female",
                         horizontal=True)
    education = st.selectbox("Education",
                              options=[1, 2, 3, 4],
                              format_func={1:"Graduate School",2:"University",
                                           3:"High School",4:"Others"}.get)
    marriage  = st.selectbox("Marital Status",
                              options=[1, 2, 3],
                              format_func={1:"Married",2:"Single",3:"Others"}.get)

    st.markdown("---")
    # Credit
    st.markdown("**Credit Profile**")
    limit_bal = st.number_input("Credit Limit (NT$)", 10_000, 1_000_000, 200_000, step=10_000)

    st.markdown("---")
    st.markdown("**Repayment Status** *(last 6 months)*")
    st.caption("-2=no use  -1=paid in full  0=revolving  1–8=months late")
    pay_keys = {
        "PAY_0": ("September (most recent)", 0),
        "PAY_2": ("August",  0),
        "PAY_3": ("July",    0),
        "PAY_4": ("June",    0),
        "PAY_5": ("May",     0),
        "PAY_6": ("April",   0),
    }
    pay_vals = {}
    for key, (label, default) in pay_keys.items():
        pay_vals[key] = st.slider(label, -2, 8, default, key=key)

    st.markdown("---")
    st.markdown("**Bill Statements (NT$)**")
    bill1 = st.number_input("September", value=20000, step=1000, key="b1")
    bill2 = st.number_input("August",    value=18000, step=1000, key="b2")
    bill3 = st.number_input("July",      value=15000, step=1000, key="b3")
    bill4 = st.number_input("June",      value=14000, step=1000, key="b4")
    bill5 = st.number_input("May",       value=13000, step=1000, key="b5")
    bill6 = st.number_input("April",     value=12000, step=1000, key="b6")

    st.markdown("---")
    st.markdown("**Payments Made (NT$)**")
    pamt1 = st.number_input("September", value=2000, step=500, key="p1")
    pamt2 = st.number_input("August",    value=1500, step=500, key="p2")
    pamt3 = st.number_input("July",      value=1500, step=500, key="p3")
    pamt4 = st.number_input("June",      value=1000, step=500, key="p4")
    pamt5 = st.number_input("May",       value=1000, step=500, key="p5")
    pamt6 = st.number_input("April",     value=1000, step=500, key="p6")

    st.markdown("---")
    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")


# MAIN

st.markdown('<p class="main-title">💳 XplainCredit</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Explainable AI for Credit Card Default Risk Assessment '
    '&nbsp;|&nbsp; UCI Taiwan Credit Dataset</p>',
    unsafe_allow_html=True,
)

# Landing / model stats
if not predict_btn:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-box"><h3>{stats['roc_auc']:.3f}</h3>
    <b>ROC-AUC</b><br><small>Test set performance</small></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-box"><h3>{stats['cv_auc_mean']:.3f} ± {stats['cv_auc_std']:.3f}</h3>
    <b>5-Fold CV AUC</b><br><small>Robust generalisation</small></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-box"><h3>{stats['n_train']+stats['n_test']:,}</h3>
    <b>Training Records</b><br><small>Real credit card data</small></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-box"><h3>{stats['default_rate']:.1%}</h3>
    <b>Default Rate</b><br><small>In original dataset</small></div>""", unsafe_allow_html=True)

    st.info("👈 Fill in the applicant details in the sidebar and click **Analyze Risk** to get started.")

    st.markdown('<p class="sec-hdr">What this tool does</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class="metric-box"><h2>🎯</h2>
        <b>Risk Tier Prediction</b><br>
        <small>Low / Medium / High / Very High</small></div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="metric-box"><h2>🔎</h2>
        <b>SHAP Explainability</b><br>
        <small>Why was this decision made?</small></div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class="metric-box"><h2>🧪</h2>
        <b>What-If Simulator</b><br>
        <small>How can the applicant improve?</small></div>""", unsafe_allow_html=True)

else:
    # Build input & predict
    input_df = build_input(
        limit_bal, sex, education, marriage, age,
        pay_vals["PAY_0"], pay_vals["PAY_2"], pay_vals["PAY_3"],
        pay_vals["PAY_4"], pay_vals["PAY_5"], pay_vals["PAY_6"],
        bill1, bill2, bill3, bill4, bill5, bill6,
        pamt1, pamt2, pamt3, pamt4, pamt5, pamt6,
    )

    prob                         = model.predict_proba(input_df)[0][1]
    risk_label, risk_icon, risk_class, risk_color = get_risk_tier(prob)

    util_ratio  = bill1 / (limit_bal + 1)
    pay_ratio   = (pamt1+pamt2+pamt3+pamt4+pamt5+pamt6) / (bill1+bill2+bill3+bill4+bill5+bill6+1)
    num_late    = sum(pay_vals[k] > 0 for k in pay_vals)

    # ROW 1: Risk card + key metrics 
    col_risk, col_metrics = st.columns([1, 2])

    with col_risk:
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <p style="font-size:3rem;margin:0">{risk_icon}</p>
            <p class="risk-label" style="color:{risk_color}">{risk_label}</p>
            <p class="risk-score">Default Probability: <b>{prob:.1%}</b></p>
        </div>""", unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<p class="sec-hdr">Key Derived Metrics</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Credit Utilisation", f"{util_ratio:.1%}",
                  delta="High" if util_ratio > 0.8 else "OK",
                  delta_color="inverse" if util_ratio > 0.8 else "normal")
        m2.metric("Payment Ratio",  f"{pay_ratio:.1%}",
                  delta="Low" if pay_ratio < 0.1 else "OK",
                  delta_color="inverse" if pay_ratio < 0.1 else "normal")
        m3.metric("Late Month(s)", num_late,
                  delta="Risky" if num_late >= 3 else "Fine",
                  delta_color="inverse" if num_late >= 3 else "normal")
        m4.metric("Sep Delay", pay_vals["PAY_0"],
                  delta="Overdue" if pay_vals["PAY_0"] > 0 else "OK",
                  delta_color="inverse" if pay_vals["PAY_0"] > 0 else "normal")

    st.markdown("---")

    # ROW 2: SHAP + Tips
    col_shap, col_tips = st.columns([3, 2])

    with col_shap:
        st.markdown('<p class="sec-hdr">🔎 Why this decision? (SHAP explanation)</p>',
                    unsafe_allow_html=True)
        with st.spinner("Calculating SHAP values …"):
            try:
                explainer   = shap.TreeExplainer(model, shap_bg)
                shap_values = explainer.shap_values(input_df)
                sv          = shap_values[0] if shap_values.ndim == 2 else shap_values[1][0]

                labels = [FEATURE_LABELS.get(f, f) for f in FEATURES]
                sorted_idx = np.argsort(np.abs(sv))[::-1][:12]
                vals       = sv[sorted_idx]
                bar_labels = [labels[i] for i in sorted_idx]
                bar_colors = ["#dc3545" if v > 0 else "#28a745" for v in vals]

                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.barh(range(len(vals)), vals[::-1],
                        color=bar_colors[::-1], height=0.6)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(bar_labels[::-1], fontsize=8)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (impact on default probability)", fontsize=8)
                ax.set_title("Feature Impact — Red = increases default risk, Green = reduces it",
                             fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption("The longer the bar, the more that feature influenced the prediction.")
            except Exception as e:
                st.error(f"SHAP error: {e}")

    with col_tips:
        st.markdown('<p class="sec-hdr">💡 How to Reduce Default Risk</p>',
                    unsafe_allow_html=True)
        tip_list = tips(prob, pay_vals["PAY_0"], util_ratio, num_late, limit_bal, pay_ratio)
        for t in tip_list:
            st.markdown(f'<div class="tip-box">• {t}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ROW 3: What-If Simulator 
    st.markdown('<p class="sec-hdr">🧪 What-If Simulator</p>', unsafe_allow_html=True)
    st.caption("Adjust to simulate how changes affect the risk score.")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_pay0  = st.slider("Simulate Sep Repayment Status", -2, 8, pay_vals["PAY_0"], key="sp0")
    with s2:
        sim_limit = st.slider("Simulate Credit Limit (NT$)", 10_000, 1_000_000, limit_bal,
                               step=10_000, key="slim")
    with s3:
        sim_pamt1 = st.slider("Simulate Sep Payment (NT$)", 0, 200_000, pamt1, step=1000, key="sp1")

    sim_df = build_input(
        sim_limit, sex, education, marriage, age,
        sim_pay0, pay_vals["PAY_2"], pay_vals["PAY_3"],
        pay_vals["PAY_4"], pay_vals["PAY_5"], pay_vals["PAY_6"],
        bill1, bill2, bill3, bill4, bill5, bill6,
        sim_pamt1, pamt2, pamt3, pamt4, pamt5, pamt6,
    )
    sim_prob = model.predict_proba(sim_df)[0][1]
    sim_label, sim_icon, sim_class, sim_color = get_risk_tier(sim_prob)
    delta_prob = sim_prob - prob

    sa, sb, sc = st.columns(3)
    sa.metric("Original Risk",  f"{prob:.1%}")
    sb.metric("Simulated Risk", f"{sim_prob:.1%}",
              delta=f"{delta_prob:+.1%}", delta_color="inverse")
    sc.markdown(f"""
    <div class="risk-card {sim_class}" style="padding:.8rem">
        <p style="margin:0;font-size:1.4rem">
            {sim_icon} <b style="color:{sim_color}">{sim_label}</b>
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ROW 4: Full report 
    with st.expander("📄 View Full Applicant Report"):
        edu_map = {1:"Graduate School",2:"University",3:"High School",4:"Others"}
        mar_map = {1:"Married",2:"Single",3:"Others"}
        total_bill_val = bill1+bill2+bill3+bill4+bill5+bill6
        total_pay_val  = pamt1+pamt2+pamt3+pamt4+pamt5+pamt6
        st.markdown(f"""
| Field | Value |
|---|---|
| Age | {age} |
| Gender | {"Male" if sex==1 else "Female"} |
| Education | {edu_map.get(education, "Others")} |
| Marital Status | {mar_map.get(marriage, "Others")} |
| Credit Limit | NT$ {limit_bal:,} |
| Sep Repayment Status | {pay_vals["PAY_0"]} |
| Months with Late Payment | {num_late} / 6 |
| Total 6-Month Bill | NT$ {total_bill_val:,} |
| Total 6-Month Payment | NT$ {total_pay_val:,} |
| Payment-to-Bill Ratio | {pay_ratio:.1%} |
| Credit Utilisation (Sep) | {util_ratio:.1%} |
| **Default Probability** | **{prob:.1%}** |
| **Risk Tier** | **{risk_icon} {risk_label}** |
        """)

# Footer
st.markdown(
    '<p class="footer">XplainCredit • Built with Python, XGBoost & SHAP '
    '• UCI Credit Card Default Dataset</p>',
    unsafe_allow_html=True,
)
