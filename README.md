# FarmScore Ghana
### Turning farm activity into trusted records for finance

> Built to support compliance-minded agricultural finance by turning farm activity into clear, documented evidence.

## The Problem

Over 3.4 million smallholder farmers in Ghana produce about 20 percent of national GDP. Fewer than 5 percent have access to formal credit. The issue is not that they are unreliable. The issue is that the financial system cannot see them clearly. Banks cannot easily tell which farmers are productive and dependable because there is no shared, trusted record of farm activity.

The gap is visibility.

## What This Project Does

FarmScore Ghana is a decision-support data layer. It is not a lender and it does not approve or decline loans. It converts agricultural activity into a structured score that a licensed bank or cooperative can use in its own underwriting process.

| Input signal | What it tells an institution |
|---|---|
| 5-year yield stability | Income predictability |
| Farm size and primary crop | Production capacity and collateral context |
| Regional drought risk | Exposure to systemic shocks |
| Market access and mobile money use | Formal-economy participation |
| Years of farming experience | Operational reliability |
| Education and secondary crop | Resilience and diversification |

The output is a 300 to 850 score with a record of the factors behind it. Every number can be traced back to the data that produced it. This matters because a score that cannot be explained is a liability for a lender, while a clear one is an asset.

### Why this is infrastructure, not a lender

This project is scoped the way credible agri-finance infrastructure platforms operate. It produces evidence. The lending decision stays with the licensed financial institution. That is not a limitation. It is the design choice that allows this kind of technology to scale responsibly across different regulatory environments.

## Project Structure

```
farmscore-ghana/
├── requirements.txt
├── data/
│   ├── raw/                       # Source data from FAOSTAT and generated records
│   └── processed/                 # Cleaned feature matrix and charts
├── notebooks/
│   └── farmscore_analysis.ipynb   # Full end-to-end analysis
└── app/
    ├── generate_notebook.py       # Script that builds the notebook
    └── streamlit_app.py           # Interactive demo for the score and the factors behind it
```

## Data Sources

| Source | Data Used | Access |
|---|---|---|
| [FAOSTAT](https://www.fao.org/faostat) | Ghana crop yields from 2000 to 2022 | Free, no login |
| [World Bank Findex](https://globalfindex.worldbank.org/) | Financial inclusion rates | Free, no login |
| MoFA Ghana (2010 Agricultural Census) | Farm size distributions and crop area by region | Public |

> Note: Farmer profiles are synthetic, generated to match real statistical distributions from the sources above. Crop yield data closely mirrors actual FAOSTAT figures. In a production deployment, individual farmer records would be encrypted in transit and at rest and processed in line with GDPR, consistent with how compliance-minded agri-finance infrastructure operates.

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | about 0.84 |
| Precision for reliable records | about 0.79 |
| Recall for reliable records | about 0.82 |
| Training dataset size | 5,000 farmer records |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full analysis notebook
jupyter notebook notebooks/farmscore_analysis.ipynb

# 3. Launch the interactive demo
streamlit run app/streamlit_app.py
```

The Streamlit app lets you build a farmer profile, see the score and the factors behind it, and export a structured JSON record for use in an underwriting workflow.

## Why This Matters for Ghana and for Europe

For Ghana, unlocking agricultural finance for smallholder farmers is associated with productivity gains and lower rural poverty. This directly supports Ghana's Planting for Food and Jobs programme and UN SDGs 1, 2, 8, and 10.

For Europe, transparent credit infrastructure is one of the fastest-growing areas in responsible fintech, and one of the few areas where strong regulatory expectations and emerging-market financial inclusion can reinforce each other. Operators connecting Ghana and the Netherlands sit at that intersection.

## Author's Note

I built this because I wanted a portfolio project that was technical and useful, not generic. It uses random forest, SHAP, and proper train and test evaluation, but it also addresses a problem I care about: farmers being locked out of formal finance not because they are unreliable, but because there is no clear record of their activity. If you are working on agricultural finance in Ghana, the Netherlands, or beyond, I would be glad to talk.

Skills demonstrated: Python, pandas, scikit-learn, SHAP, Streamlit, feature engineering, model evaluation, transparent modeling, data storytelling, and regulatory-aware system design.
