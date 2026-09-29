"""
Financial Statement Fraud Risk-Screening Model
Data: Bao, Ke, Li, Yu & Zhang (2020, Journal of Accounting Research) —
      SEC AAER-labeled accounting fraud dataset, Compustat financials 1991-2014.
      https://github.com/JarFraud/FraudDetection

Goal: build an interpretable risk-scoring model that ranks firm-years by
fraud risk, so a limited internal-audit/control team can prioritize review
of the highest-risk filings instead of sampling at random.

Run from the repo root: python src/02_train_model.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/data_FraudDetection_JAR2020.csv")

# --- Data-quality note (methodology decision, documented) -----------------
# AAERs are filed with a multi-year average lag after the misstated fiscal
# year (SEC enforcement takes time to investigate). Firm-years after ~2010
# are therefore under-labeled -- frauds that will eventually be charged
# haven't been charged yet in this snapshot. We exclude fyear > 2010 from
# evaluation to avoid crediting/penalizing the model on mislabeled data.
df = df[df.fyear <= 2010].copy()

FEATURES = ["dch_wc", "ch_rsst", "dch_rec", "dch_inv", "soft_assets", "ch_cs",
            "ch_cm", "ch_roa", "issue", "bm", "dpi", "reoa", "EBIT", "ch_fcf"]
LABEL = "misstate"

data = df.dropna(subset=[LABEL]).copy()
data = data.dropna(subset=FEATURES, thresh=len(FEATURES) - 3)  # allow a few missing per row

# --- Temporal (out-of-time) split: train on the past, test on the future --
# This mirrors how a risk-monitoring model would actually be deployed: fit
# on history, then score filings you haven't seen yet. A random split would
# leak future information and overstate performance.
train = data[data.fyear <= 2001]
test = data[(data.fyear > 2001) & (data.fyear <= 2010)]

X_train, y_train = train[FEATURES], train[LABEL]
X_test, y_test = test[FEATURES], test[LABEL]

print(f"Train: {len(train):,} firm-years, {y_train.sum():.0f} fraud ({y_train.mean():.3%})  [fyear <= 2001]")
print(f"Test:  {len(test):,} firm-years, {y_test.sum():.0f} fraud ({y_test.mean():.3%})  [2002-2010]")

# --- Model 1: Logistic Regression (interpretable, coefficients = risk drivers)
logit = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
])
logit.fit(X_train, y_train)
logit_scores = logit.predict_proba(X_test)[:, 1]

# --- Model 2: Random Forest (robustness check for non-linear risk patterns)
rf = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
])
rf.fit(X_train, y_train)
rf_scores = rf.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, scores):
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    print(f"\n{name}:")
    print(f"  ROC-AUC:      {auc:.3f}")
    print(f"  PR-AUC (AP):  {ap:.3f}  (baseline/random = {y_true.mean():.3%})")
    order = np.argsort(-scores)
    y_sorted = y_true.values[order]
    for pct in [1, 5, 10, 20]:
        k = max(1, int(len(y_sorted) * pct / 100))
        capture = y_sorted[:k].sum() / y_sorted.sum()
        print(f"  Top {pct:>2}% flagged captures {capture:.1%} of actual frauds "
              f"({y_sorted[:k].sum():.0f}/{y_sorted.sum():.0f}), lift = {capture/(pct/100):.1f}x random")
    return auc, ap

evaluate("Logistic Regression", y_test, logit_scores)
evaluate("Random Forest", y_test, rf_scores)

# --- Risk drivers (logistic coefficients, standardized) --------------------
coefs = pd.Series(logit.named_steps["clf"].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
print("\nTop risk drivers (standardized logistic coefficients):")
print(coefs)

# --- Save artifacts for reporting/charting script ---------------------------
test_out = test[["fyear", "gvkey", LABEL]].copy()
test_out["logit_score"] = logit_scores
test_out["rf_score"] = rf_scores
test_out.to_csv("outputs/test_scores.csv", index=False)
coefs.to_csv("outputs/risk_drivers.csv", header=["coef"])
train[["fyear"]].assign(y=y_train.values).to_csv("outputs/train_summary.csv", index=False)

joblib.dump(logit, "outputs/logit_model.joblib")
joblib.dump(rf, "outputs/rf_model.joblib")
print("\nSaved to outputs/: test_scores.csv, risk_drivers.csv, logit_model.joblib, rf_model.joblib")
