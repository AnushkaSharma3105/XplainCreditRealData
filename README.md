# 💳 XplainCredit — Credit Card Default Risk Assessment

Explainable AI dashboard for predicting credit card default risk using the
**UCI Taiwan Credit Card Default Dataset** (30,000 real records).

---

## 🗂️ Project Structure

```
xplaincredit/
├── app.py                  ← Streamlit dashboard
├── train_model.py          ← Model training script (run once)
├── requirements.txt        ← Python dependencies
├── data/
│   └── credit_data.csv     ← UCI dataset (you supply this)
└── model/                  ← Auto-created by train_model.py
    ├── xplaincredit_model.pkl
    ├── feature_names.pkl
    ├── shap_background.pkl
    ├── train_stats.pkl
    └── eval_plots.png
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- `pip` package manager

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare the data
Place the UCI dataset in the `data/` folder:
```
data/credit_data.csv
```

The raw file from UCI is an `.xls` file named  
`default of credit card clients.xls`.

**Convert it once with Python:**
```python
import pandas as pd

df = pd.read_excel(
    "default of credit card clients.xls",
    engine="xlrd",     # requires: pip install xlrd==2.0.1
    header=1           # row 0 is a secondary header — skip it
)
df.to_csv("data/credit_data.csv", index=False)
```

Or on macOS/Linux with LibreOffice:
```bash
soffice --headless --convert-to csv \
    "default of credit card clients.xls" \
    --outdir data/
# then rename the output to credit_data.csv
```

### 5. Train the model (run once)
```bash
python train_model.py
```
This will:
- Engineer features from the raw data
- Train an XGBoost classifier (~2 minutes)
- Print evaluation metrics (ROC-AUC, classification report, 5-fold CV)
- Save all model files to `model/`

Expected output:
```
ROC-AUC : ~0.784
5-fold CV AUC: ~0.779 ± 0.005
```

### 6. Launch the dashboard
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 📊 Dataset Details

| Property      | Value                              |
|---------------|------------------------------------|
| Source        | UCI Machine Learning Repository    |
| Records       | 30,000 credit card holders         |
| Time period   | April–September 2005               |
| Location      | Taiwan                             |
| Target        | Default payment next month (0/1)   |
| Default rate  | ~22.1%                             |
| Features      | 23 original + 9 engineered         |

### Key Features

| Feature    | Description                                              |
|------------|----------------------------------------------------------|
| LIMIT_BAL  | Credit limit (NT dollars)                                |
| SEX        | 1=Male, 2=Female                                         |
| EDUCATION  | 1=Grad school, 2=University, 3=High school, 4=Other      |
| MARRIAGE   | 1=Married, 2=Single, 3=Other                             |
| AGE        | Age in years                                             |
| PAY_0…6    | Repayment status (-2 to 8; negative=paid, positive=late) |
| BILL_AMT1–6| Monthly bill statement amount (NT$)                      |
| PAY_AMT1–6 | Monthly payment amount (NT$)                             |

### Engineered Features (added by train_model.py)

| Feature           | Formula                             |
|-------------------|-------------------------------------|
| MAX_DELAY         | max(PAY_0…PAY_6)                    |
| AVG_BILL_AMT      | mean(BILL_AMT1…6)                   |
| AVG_PAY_AMT       | mean(PAY_AMT1…6)                    |
| TOTAL_BILL        | sum(BILL_AMT1…6)                    |
| TOTAL_PAY         | sum(PAY_AMT1…6)                     |
| PAY_RATIO         | TOTAL_PAY / (TOTAL_BILL + 1)        |
| UTIL_RATIO        | BILL_AMT1 / (LIMIT_BAL + 1)         |
| NUM_LATE_PAYMENTS | count of months with PAY > 0        |
| NUM_ON_TIME       | count of months with PAY ≤ 0        |

---

## 🔬 Model Details

| Property         | Value                        |
|------------------|------------------------------|
| Algorithm        | XGBoost (GBM)                |
| Imbalance handling | scale_pos_weight (~3.5x)   |
| n_estimators     | Up to 500, early stopping 30 |
| max_depth        | 6                            |
| learning_rate    | 0.05                         |
| Explainability   | SHAP TreeExplainer           |

---

## 🖥️ App Features

1. **Risk Tier Prediction** — Low / Medium / High / Very High with probability
2. **Key Metrics Dashboard** — Credit utilisation, payment ratio, delay counts
3. **SHAP Waterfall Chart** — Top 12 features driving the prediction
4. **Improvement Tips** — Personalised advice based on the applicant's profile
5. **What-If Simulator** — Adjust repayment status, limit, or payment and see impact
6. **Full Applicant Report** — Downloadable summary table

---

## 📦 Troubleshooting

**`FileNotFoundError: model/xplaincredit_model.pkl`**  
→ Run `python train_model.py` first.

**`ModuleNotFoundError: No module named 'xgboost'`**  
→ Run `pip install -r requirements.txt` (inside your venv).

**`xlrd.biffh.XLRDError`**  
→ Make sure you installed `xlrd==2.0.1` (not 1.x).

**SHAP calculation is slow**  
→ Normal for the first prediction; subsequent ones use cached background.

---

*Built for Final Year Project — XplainCredit | Python + XGBoost + SHAP + Streamlit*
