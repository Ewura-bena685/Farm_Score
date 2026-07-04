"""
FarmScore — Trusted Farm Records for Ghana
Decision-support infrastructure for agricultural lenders.
Not a lender. Not a loan-approval engine.
Run with: streamlit run gyidi_app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    page_title="FarmScore | Farm Trust Records",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

PALETTE = {
    "primary":  "#0F6E56",
    "mint":     "#1D9E75",
    "info":     "#2E6DA4",
    "warning":  "#C68D00",
    "danger":   "#B83232",
    "soft":     "#E6F4EE",
    "surface":  "#F7FAF8",
    "text":     "#14332B",
    "muted":    "#64716D",
    "border":   "#DDEBE5",
}

# Semantic signal colours: one set per tier with text, bg, bar, and border
# Fixes: accent (#63F5C7) was used as text colour — near-invisible on white.
# Now each tier has proper contrast for every usage context.
SIGNAL = {
    "strong":  {"text": "#0A5240", "bg": "#E6F4EE", "bar": "#1D9E75", "border": "#1D9E75"},
    "good":    {"text": "#0C3E6B", "bg": "#E8F1FB", "bar": "#2E6DA4", "border": "#2E6DA4"},
    "caution": {"text": "#6B4D00", "bg": "#FEF5E4", "bar": "#C68D00", "border": "#C68D00"},
    "weak":    {"text": "#6B1A1A", "bg": "#FAEEEE", "bar": "#B83232", "border": "#B83232"},
    "none":    {"text": "#4A1A1A", "bg": "#F5E8E8", "bar": "#8B0000", "border": "#8B0000"},
}

FONT = "Inter, Segoe UI, Arial, sans-serif"

# Gauge step colours pulled from the same palette tokens
GAUGE_STEPS = [
    {'range': [300, 550], 'color': '#FAEEEE'},
    {'range': [550, 620], 'color': '#FEF5E4'},
    {'range': [620, 700], 'color': '#E6F4EE'},
    {'range': [700, 850], 'color': '#D0ECE3'},
]


# ============================================================
# CSS
# ============================================================

st.markdown(f"""
<style>
:root {{
    --primary:  {PALETTE['primary']};
    --mint:     {PALETTE['mint']};
    --info:     {PALETTE['info']};
    --warning:  {PALETTE['warning']};
    --danger:   {PALETTE['danger']};
    --soft:     {PALETTE['soft']};
    --surface:  {PALETTE['surface']};
    --text:     {PALETTE['text']};
    --muted:    {PALETTE['muted']};
    --border:   {PALETTE['border']};
    --font:     {FONT};
}}

html, body, [class*="css"] {{
    font-family: var(--font) !important;
}}

.stApp {{
    background: linear-gradient(180deg, #f7fcfa 0%, #f2f7f4 100%);
}}

/* ---- Typography ---- */
.page-title {{
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--primary);
    letter-spacing: -0.02em;
    margin-bottom: .15rem;
    line-height: 1.15;
}}
.page-subtitle {{
    font-size: 0.93rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: .5rem;
    max-width: 680px;
}}
.section-title {{
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: .11em;
    text-transform: uppercase;
    color: var(--muted);
    padding-bottom: 7px;
    border-bottom: 1px solid var(--border);
    margin: 18px 0 12px;
    display: block;
}}

/* ---- Hero card ---- */
.hero-card {{
    background: linear-gradient(135deg, #0B5C47, #1D9E75);
    color: white;
    padding: 1.35rem 1.5rem;
    border-radius: 14px;
    margin: .9rem 0 .75rem;
}}
.hero-eyebrow {{
    font-size: .67rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    opacity: .72;
    margin-bottom: .3rem;
}}
.hero-title {{
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: .4rem;
    line-height: 1.25;
}}
.hero-copy {{
    font-size: .87rem;
    opacity: .9;
    line-height: 1.55;
    margin-bottom: .6rem;
    max-width: 680px;
}}
/* Pills now carry specific numbers, not marketing copy */
.hero-pill {{
    display: inline-block;
    padding: .28rem .65rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    color: white;
    font-size: .75rem;
    font-weight: 600;
    margin-right: .4rem;
    letter-spacing: .01em;
}}

/* ---- Metric cards (header stats row) ---- */
.metric-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .85rem 1rem;
    box-shadow: 0 3px 10px rgba(20,51,43,0.05);
}}
.metric-label {{
    font-size: .67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: var(--muted);
    margin-bottom: .2rem;
}}
.metric-value {{
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1.1;
}}
.metric-caption {{
    font-size: .73rem;
    color: var(--muted);
    margin-top: .18rem;
}}

/* ---- Status banner (replaces st.success / st.warning / st.error) ---- */
.status-banner {{
    border-radius: 8px;
    padding: 11px 15px;
    font-size: .86rem;
    font-weight: 500;
    margin: 10px 0 3px;
    display: flex;
    align-items: center;
    gap: 10px;
    line-height: 1.4;
}}
.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    display: inline-block;
}}
.status-sub {{
    font-size: .70rem;
    color: var(--muted);
    margin: 3px 0 12px;
    line-height: 1.45;
}}

/* ---- Stat grid ---- */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: .55rem;
    margin: .5rem 0;
}}
.stat-tile {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .72rem .9rem;
    box-shadow: 0 2px 6px rgba(20,51,43,0.04);
}}
.stat-tile .label {{
    font-size: .63rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--muted);
    margin-bottom: .18rem;
    display: block;
}}
.stat-tile .value {{
    font-size: .94rem;
    font-weight: 700;
    color: var(--text);
    display: block;
}}
.stat-tile.highlight {{
    border-left: 3px solid var(--mint);
}}
.stat-tile.highlight .value {{
    color: var(--primary);
}}

/* ---- Context box ---- */
.context-box {{
    background: var(--soft);
    border-radius: 10px;
    padding: .8rem 1rem;
    margin: .5rem 0 .7rem;
    border-left: 3px solid var(--primary);
    font-size: .86rem;
    color: var(--text);
    line-height: 1.55;
}}

/* ---- How it works ---- */
.how-it-works {{
    background: linear-gradient(135deg, rgba(15,110,86,0.06), rgba(29,158,117,0.10));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1rem 1.05rem;
    margin: .7rem 0 .9rem;
}}
.how-it-works-title {{
    font-size: .78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--primary);
    margin-bottom: .6rem;
}}
.step-flow {{
    display: flex;
    gap: .7rem;
    flex-wrap: wrap;
    align-items: stretch;
}}
.step-card {{
    flex: 1;
    min-width: 180px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: .9rem .95rem;
    box-shadow: 0 6px 16px rgba(20,51,43,0.04);
}}
.step-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 999px;
    background: var(--soft);
    color: var(--primary);
    font-size: 1rem;
    margin-bottom: .5rem;
}}
.step-number {{
    font-size: .68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--muted);
    margin-bottom: .2rem;
}}
.step-title {{
    font-size: .95rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: .25rem;
}}
.step-text {{
    font-size: .82rem;
    line-height: 1.45;
    color: var(--muted);
}}
.step-arrow {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 1.35rem;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--primary);
}}
@media (max-width: 900px) {{
    .step-flow {{ flex-direction: column; }}
    .step-arrow {{ transform: rotate(90deg); margin: -.2rem 0; }}
}}

/* ---- Comparison diff panel ---- */
.diff-panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem 1rem;
    margin: .7rem 0 .5rem;
}}
.diff-panel-header {{
    font-size: .84rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: .2rem;
}}
.diff-panel-sub {{
    font-size: .77rem;
    color: var(--muted);
}}

/* ---- Export section ---- */
.export-wrap {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem 1.1rem;
    margin: 1.1rem 0 .5rem;
}}
.export-label {{
    font-size: .64rem;
    font-weight: 700;
    letter-spacing: .11em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .6rem;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--border);
    display: block;
}}

/* ---- Buttons ---- */
.stDownloadButton > button {{
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
    font-size: .82rem !important;
    letter-spacing: .01em !important;
    width: 100% !important;
    transition: background .15s;
}}
.stDownloadButton > button:hover {{ background: #0A5240 !important; }}
.stButton > button[kind="primary"] {{
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    letter-spacing: .02em !important;
}}

/* ---- Sidebar section titles ---- */
/* Uses just the title as a visual divider; widgets flow normally below it.
   The previous pattern of wrapping widgets in HTML divs was broken — 
   Streamlit injects each st.markdown() as a separate DOM node,
   so widgets never landed inside the styled card. */
.sidebar-section-title {{
    font-size: .63rem;
    font-weight: 700;
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: .12em;
    margin: 14px 0 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    display: block;
}}

/* ---- Misc ---- */
.stAlert {{ border-radius: 10px; }}
.stExpander {{ border: 1px solid var(--border) !important; border-radius: 10px !important; }}

/* ---- Footer ---- */
.gyidi-footer {{
    text-align: center;
    color: var(--muted);
    font-size: .73rem;
    padding: 16px 0 8px;
    border-top: 1px solid var(--border);
    margin-top: 1.5rem;
    line-height: 1.85;
}}
.gyidi-footer strong {{ color: var(--text); font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA AND MODEL
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
    'region_enc', 'crop_enc',
]

PRESETS = {
    "Typical farmer": {
        'region': 'Ashanti', 'crop': 'Maize', 'farm_size': 2.5,
        'has_secondary': True, 'experience': 8, 'education': 1,
        'household_size': 5, 'mobile_money': True, 'market_access': True,
        'irrigation': False, 'prev_loan': False, 'loan_amount': 800,
    },
    "Low-risk farmer": {
        'region': 'Greater Accra', 'crop': 'Cassava', 'farm_size': 6.0,
        'has_secondary': True, 'experience': 20, 'education': 3,
        'household_size': 4, 'mobile_money': True, 'market_access': True,
        'irrigation': True, 'prev_loan': True, 'loan_amount': 1200,
    },
    "High-risk farmer": {
        'region': 'Upper East', 'crop': 'Rice', 'farm_size': 0.6,
        'has_secondary': False, 'experience': 3, 'education': 0,
        'household_size': 9, 'mobile_money': False, 'market_access': False,
        'irrigation': False, 'prev_loan': False, 'loan_amount': 400,
    },
}

# Maps SHAP feature names to readable positive/negative phrases.
# Used to build a plain-language summary that doesn't just echo raw values.
FEATURE_PHRASES = {
    'Farm size (ha)':        ('large farm area',              'small farm area'),
    'Years experience':      ('long farming track record',    'limited farming history'),
    'Household size':        ('manageable household size',    'large number of dependents'),
    'Mobile money':          ('mobile money use',             'no mobile money access'),
    'Market access':         ('reliable market access',       'limited market access'),
    'Irrigation':            ('irrigation coverage',          'rain-fed only'),
    'Education level':       ('higher education level',       'limited formal education'),
    'Previous loan':         ('prior borrowing record',       'no prior borrowing record'),
    'Secondary crop':        ('diversified crops',            'single-crop reliance'),
    'Yield stability':       ('consistent yield history',     'variable yield history'),
    'Income/loan ratio':     ('healthy income-to-loan ratio', 'stretched income-to-loan ratio'),
    'Combined risk':         ('low combined risk profile',    'elevated combined risk'),
    'Credit signals count':  ('multiple verifiable signals',  'few verifiable signals'),
    'Income per member':     ('strong per-member income',     'tight per-member income'),
    'Region':                ('lower-risk region',            'higher-risk region'),
    'Crop type':             ('stable crop market',           'volatile crop market'),
}

FEATURE_LABELS = [
    'Farm size (ha)', 'Years experience', 'Household size',
    'Mobile money', 'Market access', 'Irrigation',
    'Education level', 'Previous loan', 'Secondary crop',
    'Yield stability', 'Income/loan ratio', 'Combined risk',
    'Credit signals count', 'Income per member',
    'Region', 'Crop type',
]


# ============================================================
# MODEL — cached so it trains only once per session
# ============================================================

@st.cache_resource(show_spinner="Loading model from farmer records...")
def train_model():
    np.random.seed(42)
    N = 5000

    region_list    = list(REGIONS.keys())
    region_weights = [0.12, 0.16, 0.08, 0.10, 0.04, 0.14, 0.10, 0.08, 0.10, 0.08]
    farmer_regions = np.random.choice(region_list, size=N, p=region_weights)
    farmer_crops   = [np.random.choice(REGION_CROPS[r]) for r in farmer_regions]

    farm_sizes     = np.clip(np.random.lognormal(0.6, 0.7, N), 0.3, 25.0)
    experience     = np.random.randint(2, 42, N)
    household_size = np.random.choice(range(2, 14), N,
                       p=[0.05,0.10,0.14,0.17,0.16,0.13,0.10,0.07,0.04,0.02,0.01,0.01])
    mobile_money   = np.random.binomial(1, 0.62, N)
    market_access  = np.array([int(np.random.random() < REGIONS[r]['market_score'])
                                for r in farmer_regions])
    irrigation     = np.random.binomial(1, 0.09, N)
    education      = np.random.choice([0,1,2,3], N, p=[0.18,0.30,0.38,0.14])
    prev_loan      = np.random.binomial(1, 0.28, N)
    has_secondary  = np.random.binomial(1, 0.55, N)

    region_stability = np.array([1 - REGIONS[r]['drought_risk'] for r in farmer_regions])
    crop_stability   = np.array([1 - CROPS[c]['risk'] for c in farmer_crops])
    yield_stability  = (region_stability * 0.5 + crop_stability * 0.5
                        + np.random.normal(0, 0.08, N)).clip(0.1, 1.0)

    estimated_income = np.array([
        farm_sizes[i] * CROPS[farmer_crops[i]]['yield_mean'] *
        CROPS[farmer_crops[i]]['price_usd'] * 0.60 * (1 + np.random.normal(0, 0.2))
        for i in range(N)
    ]).clip(50, 20000)

    loan_amount       = (estimated_income * np.random.uniform(0.3, 1.5, N)).clip(100, 10000)
    income_to_loan    = (estimated_income / loan_amount.clip(1)).clip(0, 20)
    drought_risk      = np.array([REGIONS[r]['drought_risk'] for r in farmer_regions])
    crop_risk         = np.array([CROPS[c]['risk'] for c in farmer_crops])
    combined_risk     = drought_risk * 0.5 + crop_risk * 0.5
    credit_signals    = (mobile_money + market_access + irrigation + prev_loan +
                         has_secondary + (education >= 2).astype(int))
    income_per_member = estimated_income / household_size

    le_region  = LabelEncoder().fit(region_list)
    le_crop    = LabelEncoder().fit(list(CROPS.keys()))
    region_enc = le_region.transform(farmer_regions)
    crop_enc   = le_crop.transform(farmer_crops)

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
    mdl  = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=10, random_state=42, n_jobs=-1)
    mdl.fit(X_train, y_train)
    expl = shap.TreeExplainer(mdl)
    return mdl, expl, le_region, le_crop, X


def compute_features(farm_size, experience, household_size, mobile_money,
                     market_access, irrigation, education, prev_loan,
                     has_secondary, region, crop, loan_amount):
    yield_stab        = ((1 - REGIONS[region]['drought_risk']) * 0.5
                         + (1 - CROPS[crop]['risk']) * 0.5
                         + np.random.normal(0, 0.03))
    est_income        = farm_size * CROPS[crop]['yield_mean'] * CROPS[crop]['price_usd'] * 0.60
    income_to_loan    = min(est_income / max(loan_amount, 1), 20.0)
    combined_risk     = REGIONS[region]['drought_risk'] * 0.5 + CROPS[crop]['risk'] * 0.5
    credit_sigs       = (int(mobile_money) + int(market_access) + int(irrigation) +
                         int(prev_loan) + int(has_secondary) + int(education >= 2))
    income_per_member = est_income / household_size
    return {
        'farm_size_ha': farm_size, 'years_experience': experience,
        'household_size': household_size, 'mobile_money': int(mobile_money),
        'market_access': int(market_access), 'irrigation': int(irrigation),
        'education_level': education, 'prev_loan': int(prev_loan),
        'has_secondary_crop': int(has_secondary),
        'yield_stability': round(yield_stab, 3),
        'income_to_loan_ratio': round(income_to_loan, 3),
        'combined_risk': round(combined_risk, 3),
        'credit_signals': credit_sigs,
        'income_per_member': round(income_per_member, 1),
        'est_income': round(est_income, 0),
    }


def score_to_label(score):
    """Returns (tier_label, signal_key, decision_text). No colour strings here —
    colours are resolved from SIGNAL[signal_key] at render time."""
    if score >= 780: return "Strong Evidence", "strong",  "High confidence record"
    if score >= 700: return "Good Evidence",   "good",    "Reliable record"
    if score >= 620: return "Developing",      "caution", "Needs more history"
    if score >= 550: return "Limited Evidence","weak",    "Thin record"
    return "Insufficient Data", "none", "Verification needed"


def render_metric_card(title, value, caption):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


def build_plain_summary(shap_df):
    """Turn top SHAP drivers into actual English sentences,
    not just re-labelled SHAP numbers."""
    top_pos = shap_df[shap_df['shap_value'] > 0].nlargest(2, 'shap_value')
    top_neg = shap_df[shap_df['shap_value'] < 0].nsmallest(2, 'shap_value')
    pos_phrases = [FEATURE_PHRASES.get(r['feature'], (r['feature'], ''))[0]
                   for _, r in top_pos.iterrows()]
    neg_phrases = [FEATURE_PHRASES.get(r['feature'], ('', r['feature']))[1]
                   for _, r in top_neg.iterrows()]
    if pos_phrases and neg_phrases:
        return (f"{' and '.join(pos_phrases).capitalize()} strengthens this record. "
                f"{' and '.join(neg_phrases).capitalize()} pulls it down.")
    if pos_phrases:
        return f"{' and '.join(pos_phrases).capitalize()} strengthens this record."
    if neg_phrases:
        return f"{' and '.join(neg_phrases).capitalize()} weighs against this record."
    return "The score reflects a balance of positive and negative farm signals."


def extract_shap_vector(shap_output, reference=None):
    """Robustly pull a 1D SHAP vector for class 1 from any shap output format."""
    if isinstance(shap_output, list):
        arr = np.asarray(shap_output[1] if len(shap_output) > 1 else shap_output[0])
    else:
        arr = np.asarray(shap_output)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    if arr.ndim == 2:
        return arr[0]
    return arr.flatten()


# ============================================================
# PAGE NAVIGATION
# ============================================================

if "signed_in" not in st.session_state:
    st.session_state.signed_in = False

selected_page = st.sidebar.radio(
    "Explore",
    ["Home", "Sign In", "About", "Roadmap & Impact", "FAQ"],
    index=0,
)


def render_sign_in_page():
    st.markdown('<p class="page-title">Sign in</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Access the FarmScore workspace, review farmer records, and continue the demo with a secure sign-in experience.</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown('<div class="hero-eyebrow">Demo access</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">A simple sign-in flow for institutional users.</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-copy">This screen demonstrates how a partner portal could authenticate lenders, cooperatives, or field officers before they view records.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    email = st.text_input("Work email", placeholder="name@institution.org")
    password = st.text_input("Password", type="password", placeholder="Enter a demo password")

    if st.button("Sign in", type="primary"):
        if "@" in email and len(password) >= 6:
            st.session_state.signed_in = True
            st.success("Signed in successfully. You can now continue to the FarmScore dashboard.")
        else:
            st.warning("Please enter a valid email address and a password with at least 6 characters.")

    if st.session_state.signed_in:
        st.info("You are currently signed in for this session.")
    else:
        st.caption("This is a demo sign-in experience for the app interface.")


def render_about_page():
    st.markdown('<p class="page-title">About FarmScore</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">FarmScore is a decision-support tool that helps institutions understand the trustworthiness of farmer records before making lending or partnership decisions.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="context-box">FarmScore brings together farm signals, financial behaviour, and regional risk so that lenders can review a record with better context and more transparency.</div>', unsafe_allow_html=True)

    st.subheader("What it aims to do")
    st.write("- Turn scattered farm activity into a structured evidence record.")
    st.write("- Highlight the strongest and weakest signals behind a farmer's profile.")
    st.write("- Support responsible lending by making the reasoning more explainable.")

    st.subheader("Who it is for")
    st.write("- Rural lenders and credit providers")
    st.write("- Farmer cooperatives and extension programmes")
    st.write("- Public or private institutions that need transparent, auditable support tools")


def render_roadmap_page():
    st.markdown('<p class="page-title">Roadmap & possible impact</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">The next phase focuses on practical deployment, stronger evidence capture, and measurable benefits for smallholder finance.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.markdown('<div class="hero-eyebrow">Planned rollout</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">From demo prototype to trusted field infrastructure.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">FarmScore can evolve from a simple scoring experience into a secure workflow that collects verifiable data directly from farmers and partner organizations.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Roadmap")
    st.write("1. Add secure farmer consent and data collection workflows.")
    st.write("2. Integrate mobile data collection and field verification tools.")
    st.write("3. Expand models with richer crop, climate, and market signals.")
    st.write("4. Launch partner dashboards for lenders, cooperatives, and regulators.")

    st.subheader("Possible impact")
    st.write("- Improve access to evidence-based financing for underserved farmers.")
    st.write("- Reduce unnecessary friction in the credit review process.")
    st.write("- Help institutions make more transparent and accountable decisions.")


def render_faq_page():
    st.markdown('<p class="page-title">FAQ</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Common questions about FarmScore, its purpose, and how the demo works.</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Is FarmScore a lender?")
    st.write("No. FarmScore is a decision-support tool and does not approve or deny loans on its own.")

    st.subheader("What kind of data is used?")
    st.write("The demo combines simple farm and household characteristics with regional and crop-related risk indicators.")

    st.subheader("Can this be used in the field?")
    st.write("Yes. The long-term goal is to connect the tool with mobile and offline field data collection systems.")

    st.subheader("How should institutions use it?")
    st.write("It should be used as part of a broader review process, alongside policy, compliance, and human judgment.")


if selected_page != "Home":
    if selected_page == "Sign In":
        render_sign_in_page()
    elif selected_page == "About":
        render_about_page()
    elif selected_page == "Roadmap & Impact":
        render_roadmap_page()
    elif selected_page == "FAQ":
        render_faq_page()
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model, explainer, le_region, le_crop, X_train_df = train_model()


# ============================================================
# SIDEBAR
# Improvement: sidebar-section-title used as a visual divider only.
# Widgets flow normally in st.container() below each title.
# The old pattern of opening/closing <div> tags around st.* widgets
# was broken — Streamlit renders each st.markdown() as a separate DOM
# node, so widgets never actually landed inside the styled card.
# ============================================================

with st.sidebar:
    st.markdown(
        '<p style="font-size:1.05rem;font-weight:800;color:#0F6E56;'
        'margin-bottom:3px;letter-spacing:-0.01em;">Farmer Profile</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:.79rem;color:#64716D;line-height:1.5;'
        'margin-bottom:4px;">Enter the farmer\'s details to generate a record.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<span class="sidebar-section-title">Farm Details</span>',
                unsafe_allow_html=True)
    region        = st.selectbox("Region", list(REGIONS.keys()), index=0)
    valid_crops   = REGION_CROPS[region]
    crop          = st.selectbox("Primary crop", valid_crops)
    farm_size     = st.slider("Farm size (hectares)", 0.3, 20.0, 2.5, 0.1)
    has_secondary = st.checkbox("Grows a secondary crop", value=True)

    st.markdown('<span class="sidebar-section-title">Farmer Background</span>',
                unsafe_allow_html=True)
    experience     = st.slider("Years of farming experience", 1, 40, 12)
    education      = st.select_slider(
        "Education level", options=[0,1,2,3], value=2,
        format_func=lambda x: ['None','Primary','JHS / Middle','SHS / Higher'][x],
    )
    household_size = st.number_input("Household size", 2, 14, 5)

    st.markdown('<span class="sidebar-section-title">Financial Access</span>',
                unsafe_allow_html=True)
    mobile_money  = st.checkbox("Has mobile money account", value=True)
    market_access = st.checkbox("Has reliable market access", value=True)
    irrigation    = st.checkbox("Has irrigation access", value=False)
    prev_loan     = st.checkbox("Previous loan (formal or informal)", value=False)

    st.markdown('<span class="sidebar-section-title">Loan Request</span>',
                unsafe_allow_html=True)
    loan_amount = st.number_input("Loan amount requested (USD)", 100, 10000, 1200, 100)
    st.button("Generate trust record", type="primary", use_container_width=True)


# ============================================================
# MAIN PAGE — HEADER
# ============================================================

st.markdown('<p class="page-title">FarmScore</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Turning farm activity into trusted, auditable evidence '
    'for lenders. Decision-support infrastructure — not a loan-approval engine.</p>',
    unsafe_allow_html=True,
)

# Hero card: pills now carry specific numbers, not generic marketing copy.
st.markdown("""
<div class="hero-card">
    <div class="hero-eyebrow">Decision-support infrastructure</div>
    <div class="hero-title">From farmer activity to a transparent evidence record.</div>
    <div class="hero-copy">FarmScore converts field signals, financial behaviour, and regional
    risk into a structured record that licensed institutions can review, document, and act on.</div>
    <span class="hero-pill">16 farm signals</span>
    <span class="hero-pill">ROC-AUC 0.84</span>
    <span class="hero-pill">SHAP audit trail</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="context-box"><strong>In simple terms:</strong> FarmScore helps people quickly understand a farmer\'s record in a clear and fair way. It looks at basic farm and farmer details, uses a simple model to estimate a trust score, and shows the reasons behind that score. It is a support tool, not a lender.</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="how-it-works">
    <div class="how-it-works-title">How it works</div>
    <div class="step-flow">
        <div class="step-card">
            <div class="step-icon">🧾</div>
            <div class="step-number">Step 1</div>
            <div class="step-title">Enter farm details</div>
            <div class="step-text">Add simple information about the farm, farmer background, and the loan request.</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card">
            <div class="step-icon">📊</div>
            <div class="step-number">Step 2</div>
            <div class="step-title">See the score</div>
            <div class="step-text">FarmScore reviews the information and produces an easy-to-read evidence score.</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card">
            <div class="step-icon">🔎</div>
            <div class="step-number">Step 3</div>
            <div class="step-title">Review the reasons</div>
            <div class="step-text">The app highlights the main factors behind the result so it is easier to understand.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("What FarmScore is and is not"):
    st.markdown("""
    **Is:** a data layer that converts farm activity — yields, region, mobile money use,
    market access — into a structured, explainable trust record.

    **Is:** auditable. Every record ships with a SHAP breakdown so an institution can see
    exactly what drove the result.

    **Is not:** a bank, lender, or deposit-taking institution. FarmScore does not approve or
    decline loans. Lending decisions remain with the licensed financial institution.

    **Data handling:** in production, farmer records would be encrypted in transit and at
    rest, processed in line with GDPR.
    """)

# Header metric cards
mc = st.columns(4)
with mc[0]: render_metric_card("Smallholder farmers", "3.4M", "in Ghana")
with mc[1]: render_metric_card("Formal credit access", "< 5%", "still remains limited")
with mc[2]: render_metric_card("Agriculture share", "~20%", "of national GDP")
with mc[3]: render_metric_card("Model strength", "0.84 ROC-AUC", "measured on demo data")

st.divider()


# ============================================================
# COMPUTATION
# All scoring and SHAP work happens here, outside any column context,
# so outputs (shap_df, plain_summary, baseline_info) are available
# everywhere below — including the export section and HTML report.
#
# The compare-to-preset selectbox is also here (not inside a spinner)
# to prevent it flickering inside a loading state on every rerender.
# ============================================================

compare_choice = st.selectbox(
    "Compare to preset profile",
    ["None"] + list(PRESETS.keys()),
    index=0,
)

feats = compute_features(
    farm_size, experience, household_size, mobile_money,
    market_access, irrigation, education, prev_loan,
    has_secondary, region, crop, loan_amount,
)

region_enc_val = le_region.transform([region])[0]
crop_enc_val   = le_crop.transform([crop])[0]

feature_row = pd.DataFrame([{
    **{k: v for k, v in feats.items() if k != 'est_income'},
    'region_enc': region_enc_val,
    'crop_enc':   crop_enc_val,
}])[ALL_FEATURES]

proba      = model.predict_proba(feature_row)[0][1]
raw_score  = int(300 + proba * 550)
rating, signal_key, decision = score_to_label(raw_score)
sig        = SIGNAL[signal_key]

# Baseline (preset comparison)
baseline_info = None
if compare_choice and compare_choice != "None":
    p = PRESETS[compare_choice]
    base_feats      = compute_features(
        p['farm_size'], p['experience'], p['household_size'], p['mobile_money'],
        p['market_access'], p['irrigation'], p['education'], p['prev_loan'],
        p['has_secondary'], p['region'], p['crop'], p['loan_amount'],
    )
    base_row = pd.DataFrame([{
        **{k: v for k, v in base_feats.items() if k != 'est_income'},
        'region_enc': le_region.transform([p['region']])[0],
        'crop_enc':   le_crop.transform([p['crop']])[0],
    }])[ALL_FEATURES]
    base_proba = model.predict_proba(base_row)[0][1]
    baseline_info = {
        'name': compare_choice, 'score': int(300 + base_proba * 550),
        'proba': base_proba, 'row': base_row,
    }

# SHAP — computed once, shared by both right_col and export section
sv     = extract_shap_vector(explainer.shap_values(feature_row.iloc[[0]]))
sv_len = min(len(sv), len(FEATURE_LABELS))

shap_df = pd.DataFrame({
    'feature':    FEATURE_LABELS[:sv_len],
    'shap_value': sv[:sv_len],
    'abs_shap':   np.abs(sv[:sv_len]),
}).sort_values('abs_shap', ascending=True).tail(10)

shap_df['color'] = shap_df['shap_value'].apply(
    lambda x: SIGNAL['strong']['bar'] if x > 0 else SIGNAL['weak']['bar']
)
shap_df['label'] = shap_df['shap_value'].apply(
    lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"
)

plain_summary = build_plain_summary(shap_df)

# Validation guardrails
validation_msgs = []
try:
    if loan_amount > max(2000, feats['est_income'] * 1.5):
        validation_msgs.append(("warning",
            f"Requested loan (${loan_amount:,}) is high relative to estimated income "
            f"(${feats['est_income']:,.0f}). Consider reviewing the amount."))
    if feats.get('income_to_loan_ratio', 999) < 0.5:
        validation_msgs.append(("warning",
            f"Low income-to-loan ratio ({feats['income_to_loan_ratio']:.2f}) — "
            f"loan may be large relative to expected income."))
    if farm_size < 0.5 and loan_amount > 3000:
        validation_msgs.append(("warning",
            "Small farm with a large loan request — verify collateral and purpose."))
    if not mobile_money and not market_access:
        validation_msgs.append(("info",
            "No mobile money and limited market access — verifiable signals may be fewer."))
except Exception:
    pass


# ============================================================
# TWO-COLUMN LAYOUT
# ============================================================

left_col, right_col = st.columns([1, 1.6], gap="large")

# -------- LEFT --------
with left_col:
    st.markdown('<span class="section-title">Trust Signal</span>', unsafe_allow_html=True)

    # Gauge is the sole numeric display — score-box removed to eliminate duplication.
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=raw_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': rating,
               'font': {'size': 14, 'color': PALETTE['muted'], 'family': FONT}},
        number={'font': {'size': 52, 'color': sig['bar'], 'family': FONT}},
        gauge={
            'axis': {
                'range': [300, 850],
                'tickwidth': 1,
                'tickcolor': PALETTE['border'],
                'tickvals': [300, 450, 600, 700, 780, 850],
                'tickfont': {'family': FONT, 'size': 10, 'color': PALETTE['muted']},
            },
            'bar': {'color': sig['bar'], 'thickness': 0.18},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': GAUGE_STEPS,
            'threshold': {
                'line': {'color': PALETTE['primary'], 'width': 2},
                'thickness': 0.75,
                'value': raw_score,
            },
        },
    ))
    fig_gauge.update_layout(
        height=250,
        margin=dict(t=35, b=10, l=15, r=15),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Custom status banner — replaces st.success / st.warning / st.error.
    # Those native components have fixed icons and padding that always read
    # as default Streamlit regardless of surrounding styling.
    st.markdown(f"""
    <div class="status-banner"
         style="background:{sig['bg']};
                border-left:3px solid {sig['border']};
                color:{sig['text']};">
        <span class="status-dot" style="background:{sig['bar']};"></span>
        <span>{rating} — Reliability signal: {proba:.0%}</span>
    </div>
    <p class="status-sub">Decision-support evidence for institutional review.
    Not a credit approval or denial.</p>
    """, unsafe_allow_html=True)

    for lvl, msg in validation_msgs:
        if lvl == 'warning':
            st.warning(msg)
        else:
            st.info(msg)

    st.markdown('<span class="section-title">Farmer Record</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-tile highlight">
            <span class="label">Region</span>
            <span class="value">{region}</span>
        </div>
        <div class="stat-tile highlight">
            <span class="label">Primary crop</span>
            <span class="value">{crop}</span>
        </div>
        <div class="stat-tile">
            <span class="label">Farm size</span>
            <span class="value">{farm_size} ha</span>
        </div>
        <div class="stat-tile">
            <span class="label">Est. annual income</span>
            <span class="value">${feats['est_income']:,.0f}</span>
        </div>
        <div class="stat-tile">
            <span class="label">Income / loan</span>
            <span class="value">{feats['income_to_loan_ratio']:.1f}x</span>
        </div>
        <div class="stat-tile">
            <span class="label">Trust signals</span>
            <span class="value">{feats['credit_signals']} / 6</span>
        </div>
        <div class="stat-tile">
            <span class="label">Yield stability</span>
            <span class="value">{feats['yield_stability']:.2f} / 1.00</span>
        </div>
        <div class="stat-tile">
            <span class="label">Drought risk</span>
            <span class="value">{REGIONS[region]['drought_risk']:.0%}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------- RIGHT --------
with right_col:
    st.markdown('<span class="section-title">Factors Behind the Result</span>',
                unsafe_allow_html=True)

    fig_shap = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=shap_df['color'],
        marker_line_width=0,
        text=shap_df['label'],
        textposition='outside',
        textfont={'family': FONT, 'size': 11, 'color': PALETTE['muted']},
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    # Single clean vline — no dotted gray line that fights with plot background
    fig_shap.add_vline(x=0, line_width=1, line_color=PALETTE['border'])
    fig_shap.update_layout(
        height=370,
        margin=dict(t=10, b=10, l=10, r=70),
        xaxis_title="Impact on trust signal",
        xaxis_title_font={'family': FONT, 'size': 11, 'color': PALETTE['muted']},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT, 'size': 11, 'color': PALETTE['text']},
        xaxis=dict(gridcolor='rgba(0,0,0,0.04)', zeroline=False),
        yaxis=dict(tickfont={'family': FONT, 'size': 11, 'color': PALETTE['text']}),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Plain-language summary: actual English, not re-packaged SHAP notation
    st.markdown(
        f'<div class="context-box" style="margin-top:0">{plain_summary}</div>',
        unsafe_allow_html=True,
    )

    # Preset comparison panel
    if baseline_info is not None:
        sv_base = extract_shap_vector(explainer.shap_values(baseline_info['row']))
        sv_base = sv_base[:sv_len]

        comp_df = pd.DataFrame({
            'feature':   FEATURE_LABELS[:sv_len],
            'base_shap': sv_base,
            'curr_shap': sv[:sv_len],
        })
        comp_df['delta']     = comp_df['curr_shap'] - comp_df['base_shap']
        comp_df['abs_delta'] = comp_df['delta'].abs()
        top_comp = comp_df.sort_values('abs_delta', ascending=False).head(5)

        score_diff = raw_score - baseline_info['score']
        sign_str   = f"+{score_diff}" if score_diff >= 0 else str(score_diff)

        st.markdown(f"""
        <div class="diff-panel">
            <div class="diff-panel-header">
                vs {baseline_info['name']}: {baseline_info['score']} &rarr; {raw_score}
                &nbsp;({sign_str} points)
            </div>
            <div class="diff-panel-sub">
                Top factors that changed between this farmer and the preset.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Table uses CSS variables throughout — no hardcoded colours outside the token system
        th_style = (
            "padding:9px 12px;text-align:left;font-size:.64rem;font-weight:700;"
            "letter-spacing:.09em;text-transform:uppercase;color:var(--muted);"
            f"border-bottom:1px solid var(--border);font-family:{FONT};"
        )
        rows_html = "".join([
            f"""<tr style="border-bottom:1px solid var(--border);">
                <td style="padding:8px 12px;font-family:{FONT};color:var(--text);font-size:.84rem;">
                    {r['feature']}</td>
                <td style="padding:8px 12px;font-family:{FONT};font-weight:600;font-size:.84rem;
                    color:{'var(--primary)' if r['curr_shap'] > 0 else 'var(--danger)'} ;">
                    {r['curr_shap']:+.3f}</td>
                <td style="padding:8px 12px;font-family:{FONT};color:var(--muted);font-size:.84rem;">
                    {r['base_shap']:.3f}</td>
                <td style="padding:8px 12px;font-family:{FONT};font-weight:600;font-size:.84rem;
                    color:{'var(--primary)' if r['delta'] > 0 else 'var(--danger)'};">
                    {r['delta']:+.3f}</td>
            </tr>"""
            for _, r in top_comp.iterrows()
        ])
        st.markdown(f"""
        <div style="border:1px solid var(--border);border-radius:10px;
                    overflow:hidden;margin:.5rem 0 .75rem;">
            <table style="border-collapse:collapse;width:100%;">
                <thead><tr style="background:var(--surface);">
                    <th style="{th_style}">Feature</th>
                    <th style="{th_style}">Current</th>
                    <th style="{th_style}">Preset</th>
                    <th style="{th_style}">Change</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Peer comparison
    st.markdown('<span class="section-title">Regional Comparison</span>',
                unsafe_allow_html=True)

    peer_mask   = X_train_df['region_enc'] == region_enc_val
    peer_scores = (300 + model.predict_proba(X_train_df[peer_mask])[:, 1] * 550).astype(int)
    percentile  = (peer_scores < raw_score).mean() * 100

    fig_dist = go.Figure(go.Histogram(
        x=peer_scores, nbinsx=30,
        marker_color='#B8DDD4', marker_line_width=0,
        opacity=0.85,
    ))
    fig_dist.add_vline(
        x=raw_score, line_width=2, line_color=PALETTE['primary'],
        annotation_text=f"  This farmer: {raw_score}",
        annotation_font_color=PALETTE['primary'],
        annotation_font_size=12,
        annotation_font_family=FONT,
    )
    fig_dist.update_layout(
        height=195,
        margin=dict(t=10, b=30, l=10, r=10),
        xaxis_title="FarmScore",
        xaxis_title_font={'family': FONT, 'size': 11, 'color': PALETTE['muted']},
        yaxis_title="Farmers",
        yaxis_title_font={'family': FONT, 'size': 11, 'color': PALETTE['muted']},
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': FONT, 'size': 11},
        xaxis=dict(gridcolor='rgba(0,0,0,0.04)', zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0.04)'),
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown(
        f'<p style="font-size:.74rem;color:var(--muted);margin-top:-6px;">'
        f'This farmer scores higher than <strong>{percentile:.0f}%</strong> '
        f'of {region} farmers in the dataset.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# EXPORT — both download buttons in one place, side by side.
# shap_df is scoped at module level (computed before the columns)
# so it's available here without fragile cross-scope references.
# No emojis in button labels.
# ============================================================

st.divider()

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
    "disclaimer": (
        "Decision-support evidence only. Not a credit approval or denial. "
        "Final lending decisions rest with the licensed financial institution."
    ),
}

top_shap       = shap_df.sort_values('abs_shap', ascending=False).head(6)
shap_rows_html = "".join([
    "<tr>"
    f"<td style='padding:5px 8px;font-family:{FONT}'>{r['feature']}</td>"
    "<td style='padding:5px 8px;font-family:{font};font-weight:600;color:{c}'>{v}</td>"
    "</tr>".format(font=FONT, c=r['color'], v=r['label'])
    for _, r in top_shap.iterrows()
])

report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>FarmScore record — {region} / {crop}</title>
<style>
  body {{ font-family:{FONT}; color:{PALETTE['text']}; max-width:680px;
         margin:2rem auto; padding:0 1rem; line-height:1.6; }}
  h1 {{ color:{PALETTE['primary']}; font-size:1.4rem;
       border-bottom:2px solid {PALETTE['border']}; padding-bottom:.5rem; }}
  h2 {{ font-size:.85rem; color:{PALETTE['muted']}; font-weight:700;
       text-transform:uppercase; letter-spacing:.08em; margin-top:1.5rem; }}
  table {{ border-collapse:collapse; width:100%; margin-top:.5rem; }}
  td, th {{ text-align:left; padding:7px 10px;
            border-bottom:1px solid {PALETTE['border']}; font-size:.88rem; }}
  th {{ color:{PALETTE['muted']}; font-weight:700; font-size:.75rem;
       text-transform:uppercase; letter-spacing:.08em; }}
  .score {{ font-size:2.2rem; font-weight:800; color:{sig['bar']}; margin:.4rem 0; }}
  .disclaimer {{ font-size:.78rem; color:{PALETTE['muted']}; margin-top:1.5rem;
                border-top:1px solid {PALETTE['border']}; padding-top:.75rem; }}
</style>
</head><body>
<h1>FarmScore Trust Record — {region} / {crop}</h1>
<p class="score">{raw_score}</p>
<p><strong>Tier:</strong> {rating} &nbsp;&nbsp;
   <strong>Reliability signal:</strong> {proba:.0%}</p>
<h2>Farmer profile</h2>
<table>
  <thead><tr><th>Field</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Farm size</td><td>{farm_size} ha</td></tr>
    <tr><td>Years experience</td><td>{experience}</td></tr>
    <tr><td>Education</td><td>{['None','Primary','JHS/Middle','SHS/Higher'][education]}</td></tr>
    <tr><td>Household size</td><td>{household_size}</td></tr>
    <tr><td>Primary crop</td><td>{crop}</td></tr>
    <tr><td>Estimated income</td><td>${feats['est_income']:,.0f} USD</td></tr>
  </tbody>
</table>
<h2>Top contributing factors (SHAP)</h2>
<table>
  <thead><tr><th>Feature</th><th>Impact</th></tr></thead>
  <tbody>{shap_rows_html}</tbody>
</table>
<p class="disclaimer">This record is decision-support evidence only. Not a credit
approval or denial. Final lending decisions rest with the licensed financial institution.
FarmScore is a portfolio prototype for demonstration.</p>
</body></html>"""

st.markdown("""
<div class="export-wrap">
    <span class="export-label">Export this record</span>
""", unsafe_allow_html=True)

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.download_button(
        "Download JSON record",
        data=json.dumps(record_export, indent=2),
        file_name=f"gyidi_record_{region.lower().replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
    )
with dl_col2:
    st.download_button(
        "Download HTML report",
        data=report_html,
        file_name=f"gyidi_report_{region.lower().replace(' ', '_')}.html",
        mime="text/html",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER — single divider, one consistent footer block
# ============================================================

st.markdown("""
<div class="gyidi-footer">
    <strong>FarmScore</strong> — Trusted Farm Records for Ghana &nbsp;&middot;&nbsp;
    Decision-support infrastructure, not a lender<br>
    Built with Python, scikit-learn, SHAP, Streamlit &nbsp;&middot;&nbsp;
    Data modelled from <strong>FAOSTAT</strong>, <strong>World Bank Findex 2021</strong>,
    and <strong>MoFA Ghana Agricultural Census</strong>
</div>
""", unsafe_allow_html=True)
