"""
FarmScore Ghana demo for farmer records and lending support.
Decision support for lenders, not a lending decision.
Based on practical farm data and simple recordkeeping.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FarmScore Ghana | Decision Support Demo",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

PALETTE = {
    "primary": "#0F6E56",
    "accent": "#1D9E75",
    "info": "#3B8BD4",
    "warning": "#B87817",
    "danger": "#D94F3A",
    "soft": "#E1F5EE",
    "surface": "#F7FBF9",
    "text": "#14332B",
    "muted": "#64716D",
    "border": "#DDEBE5",
}
FONT_FAMILY = "Inter, Segoe UI, Arial, sans-serif"

st.markdown(f"""
<style>
    :root {{
        --primary: {PALETTE['primary']};
        --accent: {PALETTE['accent']};
        --info: {PALETTE['info']};
        --warning: {PALETTE['warning']};
        --danger: {PALETTE['danger']};
        --soft: {PALETTE['soft']};
        --surface: {PALETTE['surface']};
        --text: {PALETTE['text']};
        --muted: {PALETTE['muted']};
        --border: {PALETTE['border']};
    }}
    .stApp {{ background: linear-gradient(180deg, #f7fcfa 0%, #f2f7f4 100%); }}
    .main-title {{
        font-size: 2.45rem;
        font-weight: 800;
        color: var(--primary);
        letter-spacing: -0.02em;
        margin-bottom: .2rem;
    }}
    .subtitle {{
        font-size: 1.02rem;
        color: var(--muted);
        line-height: 1.6;
        margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin: 0 0 .35rem 0;
    }}
    .hero-card {{
        background: linear-gradient(135deg, rgba(15,110,86,0.97), rgba(29,158,117,0.95));
        color: white;
        padding: 1.25rem 1.35rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(15,110,86,0.16);
    }}
    .hero-eyebrow {{
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        opacity: .86;
    }}
    .hero-title {{
        font-size: 1.45rem;
        font-weight: 700;
        margin: .3rem 0 .4rem;
        line-height: 1.25;
    }}
    .hero-copy {{
        font-size: .95rem;
        opacity: .95;
        line-height: 1.6;
    }}
    .hero-pill {{
        display: inline-block;
        padding: .35rem .6rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        color: white;
        font-size: .82rem;
        font-weight: 600;
        margin-right: .45rem;
        margin-top: .5rem;
    }}
    .score-box {{
        background: linear-gradient(135deg, #fff, #f6fbf8);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 1rem 1rem 1.1rem;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(20,51,43,0.06);
        margin-bottom: .8rem;
    }}
    .score-number {{
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        color: var(--primary);
    }}
    .score-label {{
        font-size: 1rem;
        font-weight: 700;
        margin-top: .35rem;
        color: var(--text);
        letter-spacing: .01em;
    }}
    .metric-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .9rem 1rem;
        margin: .2rem 0 .7rem;
        box-shadow: 0 8px 18px rgba(20,51,43,0.04);
    }}
    .metric-label {{
        font-size: .76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--muted);
        margin-bottom: .25rem;
    }}
    .metric-value {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }}
    .metric-caption {{
        font-size: .8rem;
        color: var(--muted);
        margin-top: .25rem;
    }}
    .stAlert {{ border-radius: 12px; }}
    .context-box {{
        background: var(--soft);
        border-radius: 14px;
        padding: .9rem 1rem;
        margin: .6rem 0 1rem;
        border-left: 4px solid var(--primary);
        color: var(--text);
        line-height: 1.5;
    }}
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: .7rem;
        margin-top: .65rem;
    }}
    .stat-tile {{
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: .8rem .9rem;
        box-shadow: 0 6px 16px rgba(20,51,43,0.04);
    }}
    .stat-tile .label {{
        font-size: .74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--muted);
        margin-bottom: .25rem;
    }}
    .stat-tile .value {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
    }}
    .sidebar-section {{
        background: linear-gradient(180deg, rgba(225,245,238,0.72), rgba(255,255,255,0.95));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: .8rem .8rem .9rem;
        margin-bottom: .7rem;
    }}
    .sidebar-section-title {{
        font-size: .82rem;
        font-weight: 800;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: .12em;
        margin-bottom: .35rem;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND MODEL — cached so it only runs once
# ============================================================

REGIONS = {
    'Ashanti':       {'drought_risk': 0.15, 'market_score': 0.75},
    'Brong-Ahafo':   {'drought_risk': 0.22, 'market_score': 0.55},
    'Central':       {'drought_risk': 0.10, 'market_score': 0.65},
    'Eastern':       {'drought_risk': 0.15, 'market_score': 0.60},
    'Greater Accra': {'drought_risk': 0.05, 'market_score': 0.90},
    'Northern':      {'drought_risk': 0.45, 'market_score': 0.35},
    'Upper East':    {'drought_risk': 0.55, 'market_score': 0.25},
    'Upper West':    {'drought_risk': 0.50, 'market_score': 0.28},
    'Volta':         {'drought_risk': 0.22, 'market_score': 0.48},
    'Western':       {'drought_risk': 0.10, 'market_score': 0.58},
}

CROPS = {
    'Cocoa':    {'yield_mean': 420,   'yield_std': 65,   'price_usd': 2.80, 'risk': 0.18},
    'Maize':    {'yield_mean': 1850,  'yield_std': 320,  'price_usd': 0.25, 'risk': 0.32},
    'Cassava':  {'yield_mean': 17500, 'yield_std': 2200, 'price_usd': 0.05, 'risk': 0.22},
    'Yam':      {'yield_mean': 14000, 'yield_std': 1800, 'price_usd': 0.12, 'risk': 0.28},
    'Rice':     {'yield_mean': 3000,  'yield_std': 480,  'price_usd': 0.45, 'risk': 0.28},
    'Plantain': {'yield_mean': 10500, 'yield_std': 1400, 'price_usd': 0.08, 'risk': 0.20},
}

REGION_CROPS = {
    'Western':      ['Cocoa', 'Plantain', 'Cassava'],
    'Ashanti':      ['Cocoa', 'Plantain', 'Cassava', 'Maize'],
    'Brong-Ahafo':  ['Maize', 'Yam', 'Cassava', 'Rice'],
    'Eastern':      ['Cocoa', 'Cassava', 'Maize', 'Plantain'],
    'Central':      ['Cassava', 'Maize', 'Plantain'],
    'Greater Accra':['Cassava', 'Maize'],
    'Volta':        ['Cassava', 'Yam', 'Maize'],
    'Northern':     ['Maize', 'Rice', 'Yam'],
    'Upper East':   ['Rice', 'Maize', 'Yam'],
    'Upper West':   ['Maize', 'Yam', 'Rice'],
}

ALL_FEATURES = [
    'farm_size_ha', 'years_experience', 'household_size',
    'mobile_money', 'market_access', 'irrigation',
    'education_level', 'prev_loan', 'has_secondary_crop',
    'yield_stability', 'income_to_loan_ratio', 'combined_risk',
    'credit_signals', 'income_per_member',
    'region_enc', 'crop_enc'
]


@st.cache_resource(show_spinner="Training FarmScore model on 5,000 Ghana farmers…")
def train_model():
    """Generate data and train the Random Forest model."""
    np.random.seed(42)
    N = 5000

    region_list = list(REGIONS.keys())
    region_weights = [0.12, 0.16, 0.08, 0.10, 0.04, 0.14, 0.10, 0.08, 0.10, 0.08]
    farmer_regions = np.random.choice(region_list, size=N, p=region_weights)
    farmer_crops   = [np.random.choice(REGION_CROPS[r]) for r in farmer_regions]

    farm_sizes      = np.clip(np.random.lognormal(0.6, 0.7, N), 0.3, 25.0)
    experience      = np.random.randint(2, 42, N)
    household_size  = np.random.choice(range(2, 14), N,
                        p=[0.05,0.10,0.14,0.17,0.16,0.13,0.10,0.07,0.04,0.02,0.01,0.01])
    mobile_money    = np.random.binomial(1, 0.62, N)
    market_access   = np.array([int(np.random.random() < REGIONS[r]['market_score'])
                                 for r in farmer_regions])
    irrigation      = np.random.binomial(1, 0.09, N)
    education       = np.random.choice([0,1,2,3], N, p=[0.18,0.30,0.38,0.14])
    prev_loan       = np.random.binomial(1, 0.28, N)
    has_secondary   = np.random.binomial(1, 0.55, N)

    region_stability = np.array([1 - REGIONS[r]['drought_risk'] for r in farmer_regions])
    crop_stability   = np.array([1 - CROPS[c]['risk'] for c in farmer_crops])
    yield_stability  = (region_stability*0.5 + crop_stability*0.5
                        + np.random.normal(0, 0.08, N)).clip(0.1, 1.0)

    estimated_income = np.array([
        farm_sizes[i] * CROPS[farmer_crops[i]]['yield_mean'] *
        CROPS[farmer_crops[i]]['price_usd'] * 0.60 * (1 + np.random.normal(0, 0.2))
        for i in range(N)
    ]).clip(50, 20000)

    loan_amount = (estimated_income * np.random.uniform(0.3, 1.5, N)).clip(100, 10000)

    # Engineered features
    income_to_loan  = (estimated_income / loan_amount.clip(1)).clip(0, 20)
    drought_risk    = np.array([REGIONS[r]['drought_risk'] for r in farmer_regions])
    crop_risk       = np.array([CROPS[c]['risk'] for c in farmer_crops])
    combined_risk   = drought_risk * 0.5 + crop_risk * 0.5
    credit_signals  = (mobile_money + market_access + irrigation + prev_loan +
                       has_secondary + (education >= 2).astype(int))
    income_per_member = estimated_income / household_size

    # Encoders
    le_region = LabelEncoder().fit(region_list)
    le_crop   = LabelEncoder().fit(list(CROPS.keys()))
    region_enc = le_region.transform(farmer_regions)
    crop_enc   = le_crop.transform(farmer_crops)

    # Target variable
    def norm(a): return (a - a.min()) / (a.max() - a.min() + 1e-9)
    log_odds = (
        1.2 * yield_stability + 0.8 * norm(farm_sizes) + 0.7 * norm(experience)
        + 0.6 * market_access + 0.5 * mobile_money + 0.4 * irrigation
        + 0.3 * norm(education) + 0.3 * has_secondary + 0.4 * prev_loan
        - 0.3 * norm(household_size) - 0.5 * drought_risk
        - 1.5 + np.random.normal(0, 0.6, N)
    )
    loan_repaid = (np.random.random(N) < 1 / (1 + np.exp(-log_odds))).astype(int)

    X = pd.DataFrame({
        'farm_size_ha': farm_sizes, 'years_experience': experience,
        'household_size': household_size, 'mobile_money': mobile_money,
        'market_access': market_access, 'irrigation': irrigation,
        'education_level': education, 'prev_loan': prev_loan,
        'has_secondary_crop': has_secondary, 'yield_stability': yield_stability,
        'income_to_loan_ratio': income_to_loan, 'combined_risk': combined_risk,
        'credit_signals': credit_signals, 'income_per_member': income_per_member,
        'region_enc': region_enc, 'crop_enc': crop_enc,
    })
    y = pd.Series(loan_repaid)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    return model, explainer, le_region, le_crop, X


def compute_features(farm_size, experience, household_size, mobile_money,
                     market_access, irrigation, education, prev_loan,
                     has_secondary, region, crop, loan_amount):
    """Turn raw farmer inputs into model-ready feature vector."""
    yield_stab = (
        (1 - REGIONS[region]['drought_risk']) * 0.5 +
        (1 - CROPS[crop]['risk']) * 0.5 +
        np.random.normal(0, 0.03)
    )
    est_income = (farm_size * CROPS[crop]['yield_mean'] * CROPS[crop]['price_usd'] * 0.60)
    income_to_loan = min(est_income / max(loan_amount, 1), 20.0)
    combined_risk = REGIONS[region]['drought_risk'] * 0.5 + CROPS[crop]['risk'] * 0.5
    credit_sigs = (int(mobile_money) + int(market_access) + int(irrigation) +
                   int(prev_loan) + int(has_secondary) + int(education >= 2))
    income_per_member = est_income / household_size

    return {
        'farm_size_ha': farm_size,
        'years_experience': experience,
        'household_size': household_size,
        'mobile_money': int(mobile_money),
        'market_access': int(market_access),
        'irrigation': int(irrigation),
        'education_level': education,
        'prev_loan': int(prev_loan),
        'has_secondary_crop': int(has_secondary),
        'yield_stability': round(yield_stab, 3),
        'income_to_loan_ratio': round(income_to_loan, 3),
        'combined_risk': round(combined_risk, 3),
        'credit_signals': credit_sigs,
        'income_per_member': round(income_per_member, 1),
        'est_income': round(est_income, 0),
    }


def score_to_label(score):
    """
    NOTE: FarmScore does not approve or decline anyone. It produces a
    structured score that a licensed lender can use in its own review
    process.
    """
    if score >= 780: return "Strong Evidence", PALETTE["accent"], "High confidence record"
    if score >= 700: return "Good Evidence",   PALETTE["info"], "Reliable record"
    if score >= 620: return "Developing",      PALETTE["warning"], "Needs more history"
    if score >= 550: return "Limited Evidence", PALETTE["danger"], "Thin record"
    return "Insufficient Data", "#8B0000", "Verification needed"


def render_metric_card(title, value, caption, icon=""):
    icon_text = f"{icon} " if icon else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon_text}{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model, explainer, le_region, le_crop, X_train_df = train_model()


# ============================================================
# SIDEBAR — FARMER PROFILE INPUTS
# ============================================================

with st.sidebar:
    st.markdown("## Farmer Profile")
    st.markdown("*Enter the farmer's details to generate a score.*")
    st.divider()

    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Farm Details</div>", unsafe_allow_html=True)
    region = st.selectbox("Region", list(REGIONS.keys()), index=0)

    valid_crops = REGION_CROPS[region]
    crop = st.selectbox("Primary crop", valid_crops)

    farm_size = st.slider("Farm size (hectares)", 0.3, 20.0, 2.5, 0.1)

    has_secondary = st.checkbox("Grows a secondary crop", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Farmer Background</div>", unsafe_allow_html=True)
    experience = st.slider("Years of farming experience", 1, 40, 12)

    education = st.select_slider(
        "Education level",
        options=[0, 1, 2, 3],
        value=2,
        format_func=lambda x: ['None', 'Primary', 'JHS/Middle', 'SHS/Higher'][x]
    )

    household_size = st.number_input("Household size", 2, 14, 5)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Financial Access</div>", unsafe_allow_html=True)
    mobile_money = st.checkbox("Has mobile money account", value=True)
    market_access = st.checkbox("Has reliable market access", value=True)
    irrigation    = st.checkbox("Has irrigation access", value=False)
    prev_loan     = st.checkbox("Previous loan history (formal/informal)", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><div class='sidebar-section-title'>Loan Request</div>", unsafe_allow_html=True)
    loan_amount = st.number_input("Loan amount requested (USD)", 100, 10000, 1200, 100)
    st.markdown("</div>", unsafe_allow_html=True)

    score_btn = st.button("Calculate FarmScore", type="primary", use_container_width=True)


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown('<p class="main-title">FarmScore Ghana</p>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Turning farm activity into a clear, explainable evidence score for lenders.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
  <div class="hero-eyebrow">Decision-support demo</div>
  <div class="hero-title">From farmer activity to a transparent evidence score.</div>
  <div class="hero-copy">FarmScore translates field signals, financial behavior, and risk context into a readable record that institutions can review responsibly.</div>
  <div>
    <span class="hero-pill">Explainable</span>
    <span class="hero-pill">Institution-ready</span>
    <span class="hero-pill">No automated approval</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="context-box">FarmScore is decision-support infrastructure, not a lender. The score is designed for institutional review, documentation, and oversight.</div>', unsafe_allow_html=True)

with st.expander("What FarmScore is and is not"):
    st.markdown("""
    - **Is:** a data layer that converts farm activity such as yields, region, mobile money use,
      and market access into a structured score.
    - **Is:** auditable, with a SHAP breakdown so an institution can see the factors behind the result.
    - **Isn't:** a bank, lender, or deposit-taking institution. FarmScore does not approve or decline loans.
      Lending decisions remain with the licensed financial institution.
    - **Data handling:** in a production system, farmer records would be encrypted in transit and at rest,
      and processed in line with GDPR.
    """)

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric_card("Smallholder farmers", "3.4M", "in Ghana")
with metric_cols[1]:
    render_metric_card("Formal credit access", "< 5%", "still remains limited")
with metric_cols[2]:
    render_metric_card("Agriculture share", "~20%", "of national GDP")
with metric_cols[3]:
    render_metric_card("Model strength", "0.84 ROC-AUC", "measured on the demo data")

st.divider()

# ---- Compute features for current sidebar state ----
with st.spinner("Preparing your FarmScore profile..."):
    feats = compute_features(
        farm_size, experience, household_size, mobile_money,
        market_access, irrigation, education, prev_loan,
        has_secondary, region, crop, loan_amount
    )

    region_enc_val = le_region.transform([region])[0]
    crop_enc_val   = le_crop.transform([crop])[0]

    feature_row = pd.DataFrame([{
        **{k: v for k, v in feats.items() if k != 'est_income'},
        'region_enc': region_enc_val,
        'crop_enc':   crop_enc_val,
    }])[ALL_FEATURES]

    proba = model.predict_proba(feature_row)[0][1]
    raw_score = int(300 + proba * 550)
    rating, rating_color, decision = score_to_label(raw_score)

# ---- Two main columns ----
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="section-title">Trust & Reliability Signal</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="score-box">
      <div class="score-number">{raw_score}</div>
      <div class="score-label">{rating}</div>
    </div>
    """, unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=raw_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{rating}", 'font': {'size': 20, 'color': rating_color}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': '#8DA59D', 'tickvals': [300, 400, 500, 600, 700, 800, 850]},
            'bar': {'color': rating_color},
            'bgcolor': 'white',
            'borderwidth': 1,
            'bordercolor': '#BFD6CD',
            'steps': [
                {'range': [300, 550], 'color': '#FBE8E8'},
                {'range': [550, 620], 'color': '#FFF2CF'},
                {'range': [620, 700], 'color': '#E4F5EC'},
                {'range': [700, 850], 'color': '#D5EDE3'},
            ],
            'threshold': {
                'line': {'color': PALETTE['primary'], 'width': 3},
                'thickness': 0.85,
                'value': raw_score
            }
        }
    ))
    fig_gauge.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT_FAMILY, 'size': 13, 'color': PALETTE['text']}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if "High confidence" in decision:
        st.success(f"{decision}  —  Estimated score: **{proba:.0%}**")
    elif "Needs more history" in decision:
        st.warning(f"{decision}  —  Estimated score: **{proba:.0%}**")
    else:
        st.error(f"{decision}  —  Estimated score: **{proba:.0%}**")
    st.caption("This is a decision-support signal for institutional review and not a final approval.")

    st.markdown("---")
    st.markdown('<div class="section-title">Farmer Record Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-tile"><div class="label">Region</div><div class="value">{region}</div></div>
      <div class="stat-tile"><div class="label">Primary crop</div><div class="value">{crop}</div></div>
      <div class="stat-tile"><div class="label">Farm size</div><div class="value">{farm_size} ha</div></div>
      <div class="stat-tile"><div class="label">Est. income</div><div class="value">${feats['est_income']:,.0f} USD</div></div>
      <div class="stat-tile"><div class="label">Income / loan</div><div class="value">{feats['income_to_loan_ratio']:.1f}×</div></div>
      <div class="stat-tile"><div class="label">Supporting signals</div><div class="value">{feats['credit_signals']} / 6</div></div>
      <div class="stat-tile"><div class="label">Yield stability</div><div class="value">{feats['yield_stability']:.2f} / 1.00</div></div>
      <div class="stat-tile"><div class="label">Drought risk</div><div class="value">{REGIONS[region]['drought_risk']:.0%}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Structured, institution-ready record export (JSON)
    import json
    record_export = {
        "farmer_record": {
            "region": region, "primary_crop": crop, "farm_size_ha": farm_size,
            "years_experience": experience, "household_size": household_size,
            "estimated_annual_income_usd": feats['est_income'],
            "verified_trust_signals": feats['credit_signals'],
            "yield_stability_index": feats['yield_stability'],
            "regional_drought_risk": REGIONS[region]['drought_risk'],
        },
        "evidence_score": {
            "value": raw_score, "scale": "300-850",
            "tier": rating, "reliability_signal": round(float(proba), 4),
        },
        "disclaimer": "Decision-support evidence only. Not a credit approval or denial. "
                       "Final lending decisions rest with the licensed financial institution.",
    }
    st.download_button(
        "⬇️ Export structured record (JSON)",
        data=json.dumps(record_export, indent=2),
        file_name=f"farmscore_record_{region.lower().replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
    )


with right_col:
    st.markdown('<div class="section-title">Factors Behind the Result</div>', unsafe_allow_html=True)

    # SHAP waterfall for this farmer
    single_row = feature_row.iloc[[0]]
    shap_vals_row = explainer.shap_values(single_row)

    # Robust extraction of Class 1 (positive prediction) SHAP values for the 16 features
    if isinstance(shap_vals_row, list):
        # Handle list format: elements per class -> each is shape (1, 16)
        sv = shap_vals_row[1][0] if len(shap_vals_row) > 1 else shap_vals_row[0][0]
    else:
        shap_array = np.asarray(shap_vals_row)
        if shap_array.ndim == 3:
            # Handle 3D array format: (samples, features, classes) -> select sample 0 and class 1
            sv = shap_array[0, :, 1] if shap_array.shape[2] > 1 else shap_array[0, :, 0]
        elif shap_array.ndim == 2:
            # Handle 2D array format: (samples, features) -> select sample 0
            sv = shap_array[0]
        else:
            sv = shap_array.flatten()

    feature_labels = [
        'Farm size (ha)', 'Years experience', 'Household size',
        'Mobile money', 'Market access', 'Irrigation',
        'Education level', 'Previous loan', 'Secondary crop',
        'Yield stability', 'Income/loan ratio', 'Combined risk',
        'Credit signals count', 'Income per member',
        'Region', 'Crop type'
    ]

    
    shap_df = pd.DataFrame({
        'feature': feature_labels,
        'shap_value': sv,
        'abs_shap': np.abs(sv)
    }).sort_values('abs_shap', ascending=True).tail(10)

    shap_df['color'] = shap_df['shap_value'].apply(
        lambda x: '#1D9E75' if x > 0 else '#D94F3A'
    )
    shap_df['label'] = shap_df['shap_value'].apply(
        lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"
    )

    fig_shap = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=shap_df['color'],
        text=shap_df['label'],
        textposition='outside',
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))
    fig_shap.add_vline(x=0, line_width=1.5, line_color='gray', line_dash='dot')
    fig_shap.update_layout(
        height=380,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis_title="SHAP value (impact on credit score)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT_FAMILY, 'size': 12, 'color': PALETTE['text']},
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.caption("Green bars **increase** the score  ·  Red bars **decrease** the score  ·  "
               "Bar length shows how much impact that factor had")

    # Peer comparison
    st.markdown("---")
    st.markdown('<div class="section-title">How does this farmer compare to peers?</div>', unsafe_allow_html=True)

    # Compare against farmers with same region
    peer_mask = X_train_df['region_enc'] == region_enc_val
    peer_scores = (300 + model.predict_proba(X_train_df[peer_mask])[:, 1] * 550).astype(int)

    percentile = (peer_scores < raw_score).mean() * 100

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=peer_scores,
        nbinsx=30,
        marker_color='#B5D4F4',
        name='Peer farmers',
        opacity=0.8
    ))
    fig_dist.add_vline(
        x=raw_score,
        line_width=2.5, line_color='#0F6E56',
        annotation_text=f" This farmer: {raw_score}",
        annotation_font_color='#0F6E56',
        annotation_font_size=13
    )
    fig_dist.update_layout(
        height=200,
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis_title="FarmScore",
        yaxis_title="# Farmers",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT_FAMILY, 'size': 11, 'color': PALETTE['text']},
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(f"This farmer scores better than **{percentile:.0f}%** of {region} farmers in our dataset.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:0.8rem; padding: 8px 0'>
    FarmScore Ghana • Portfolio Project • Decision support, not a lender<br>
    Built with Python, scikit-learn, SHAP, and Streamlit • Data based on <b>FAOSTAT</b>, <b>World Bank</b><br>
    Designed for review and documentation in lending workflows
    </div>
    """, 
    unsafe_allow_html=True
)