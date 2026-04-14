"""
Trains an XGBoost classifier on the UCI Credit Card Default dataset
(default_of_credit_card_clients.xls / credit_data.csv) and saves all
artefacts needed by the Streamlit app.

Run once before starting the app:
    python train_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# Paths 
DATA_PATH  = os.path.join("data", "credit_data.csv")
MODEL_DIR  = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
print("📂 Loading data …")
df = pd.read_csv(DATA_PATH, header=1)          # row-0 is the English header
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# target column- DEFAULT_PAYMENT_NEXT_MONTH
TARGET = "DEFAULT_PAYMENT_NEXT_MONTH"
if TARGET not in df.columns:
    # Fallback – common name variants
    for cand in ["DEFAULT PAYMENT NEXT MONTH", "default payment next month"]:
        if cand.upper().replace(" ", "_") in df.columns:
            TARGET = cand.upper().replace(" ", "_")
            break

print(f"   Rows: {len(df):,}  |  Target: {TARGET}")
print(f"   Class balance:\n{df[TARGET].value_counts()}\n")

# Feature engineering
# Drop ID – not predictive
df.drop(columns=["ID"], errors="ignore", inplace=True)

# Clamp rare EDUCATION / MARRIAGE codes to 'Other'
# EDUCATION: 1=grad school, 2=university, 3=high school, 4=others; 0,5,6=unknown→4
df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3, 4] else 4)
# MARRIAGE: 1=married, 2=single, 3=others; 0=unknown→3
df["MARRIAGE"]  = df["MARRIAGE"].apply(lambda x: x if x in [1, 2, 3] else 3)

# Repayment status: PAY_0 … PAY_6 are already numeric (-2 to 8)
# Derived features
pay_cols  = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

df["MAX_DELAY"]           = df[pay_cols].max(axis=1)
df["AVG_BILL_AMT"]        = df[bill_cols].mean(axis=1)
df["AVG_PAY_AMT"]         = df[pay_amt_cols].mean(axis=1)
df["TOTAL_BILL"]          = df[bill_cols].sum(axis=1)
df["TOTAL_PAY"]           = df[pay_amt_cols].sum(axis=1)
df["PAY_RATIO"]           = np.where(
    df["TOTAL_BILL"] > 0, df["TOTAL_PAY"] / (df["TOTAL_BILL"] + 1), 0
)
df["UTIL_RATIO"]          = np.where(
    df["LIMIT_BAL"] > 0, df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1), 0
)
df["NUM_LATE_PAYMENTS"]   = (df[pay_cols] > 0).sum(axis=1)
df["NUM_ON_TIME"]         = (df[pay_cols] <= 0).sum(axis=1)

FEATURES = [
    # Original
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
    # Engineered
    "MAX_DELAY","AVG_BILL_AMT","AVG_PAY_AMT",
    "TOTAL_BILL","TOTAL_PAY","PAY_RATIO","UTIL_RATIO",
    "NUM_LATE_PAYMENTS","NUM_ON_TIME",
]

X = df[FEATURES]
y = df[TARGET]

# Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Class-imbalance weight 
neg, pos     = (y_train == 0).sum(), (y_train == 1).sum()
scale_weight = neg / pos
print(f"scale_pos_weight = {scale_weight:.2f}  (neg={neg}, pos={pos})\n")

# Train XGBoost
print("🚀 Training XGBoost …")
model = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_weight,
    use_label_encoder = False,
    eval_metric       = "auc",
    early_stopping_rounds = 30,
    random_state      = 42,
    n_jobs            = -1,
)
model.fit(
    X_train, y_train,
    eval_set          = [(X_test, y_test)],
    verbose           = 50,
)

# Evaluate
print("\n📊 Evaluation on test set:")
y_pred       = model.predict(X_test)
y_prob       = model.predict_proba(X_test)[:, 1]
auc          = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC : {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["No Default","Default"]))

# Cross-validation AUC
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores   = cross_val_score(
    xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        use_label_encoder=False, eval_metric="auc",
        random_state=42, n_jobs=-1,
    ),
    X, y, cv=cv, scoring="roc_auc"
)
print(f"\n5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save artefacts 
print("\n💾 Saving model artefacts …")
joblib.dump(model,    os.path.join(MODEL_DIR, "xplaincredit_model.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_names.pkl"))

# Save a small background sample for SHAP 
bg_sample = X_train.sample(min(500, len(X_train)), random_state=42)
joblib.dump(bg_sample, os.path.join(MODEL_DIR, "shap_background.pkl"))

# Save training stats for the dashboard
train_stats = {
    "n_train"        : int(len(X_train)),
    "n_test"         : int(len(X_test)),
    "roc_auc"        : float(round(auc, 4)),
    "cv_auc_mean"    : float(round(cv_scores.mean(), 4)),
    "cv_auc_std"     : float(round(cv_scores.std(), 4)),
    "default_rate"   : float(round(y.mean(), 4)),
    "features"       : FEATURES,
}
joblib.dump(train_stats, os.path.join(MODEL_DIR, "train_stats.pkl"))

# Quick diagnostic plots (saved to model/ for reference)
# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cm  = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["No Default","Default"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix")
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name="XGBoost")
axes[1].set_title(f"ROC Curve  (AUC = {auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "eval_plots.png"), dpi=120)
plt.close()

print("\n✅ Done!  All files saved to ./model/")
print("   Run the app with:  streamlit run app.py")
